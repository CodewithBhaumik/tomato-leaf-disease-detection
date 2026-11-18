"""Loads the trained models listed in experiments/deployment.json once, and runs inference.

Both predictor types receive the same canonical 256x256 RGB array produced by tomatoleaf.preprocessing,
then apply exactly the transformation used when the model was evaluated offline.
"""
import base64
import logging
import threading
import time
from io import BytesIO

import joblib
import numpy as np
import torch
from PIL import Image

from tomatoleaf.config import CLASS_KEYS, CLASSES, EXPERIMENTS_DIR, MODELS_DIR
from tomatoleaf.features import extract_features, feature_vector
from tomatoleaf.gradcam import GradCAM, overlay
from tomatoleaf.models import gradcam_layer, load_checkpoint
from tomatoleaf.preprocessing import dihedral_views, preprocess_for_cnn
from tomatoleaf.utils import load_json

log = logging.getLogger("tomato.predictor")


class Predictor:
    def __init__(self, entry: dict):
        self.key, self.name, self.family = entry["key"], entry["name"], entry["family"]
        self.kind, self.feature_set = entry["kind"], entry["feature_set"]
        self.params, self.test_metrics = entry.get("params"), entry.get("test_metrics")
        self.class_keys = list(entry.get("class_keys") or CLASS_KEYS)
        self.threshold = entry.get("confidence_threshold")  # per-model override chosen on validation (else global)
        # selective-prediction operating point measured offline (threshold, coverage, accuracy when answering), if any
        self.operating_point = entry.get("operating_point")

    def predict_proba(self, rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def info(self) -> dict:
        return {"key": self.key, "name": self.name, "family": self.family}


class TorchPredictor(Predictor):
    score_type = "calibrated_softmax"

    def __init__(self, entry: dict, device: torch.device):
        super().__init__(entry)
        self.model, ckpt = load_checkpoint(MODELS_DIR / entry["path"], device)
        self.class_keys = list(ckpt["class_keys"])
        self.device = device
        self.arch = ckpt["arch"]
        self.size = int(ckpt["input_size"])
        self.tta_views = int(entry.get("tta_views") or 1)
        self.temperature = float(entry.get("temperature") or 1.0)
        self.input = (f"{self.size}x{self.size} RGB, ImageNet mean/std"
                      + (f", {self.tta_views}-view test-time augmentation" if self.tta_views > 1 else ""))
        self._cam_lock = threading.Lock()

    @torch.no_grad()
    def predict_proba(self, rgb: np.ndarray) -> np.ndarray:
        x = preprocess_for_cnn(rgb, self.size).to(self.device)
        if self.tta_views > 1:
            x = dihedral_views(x, self.tta_views)
        logits = self.model(x).float().mean(dim=0).cpu().numpy()
        z = logits / self.temperature
        e = np.exp(z - z.max())
        return e / e.sum()

    def explain(self, rgb: np.ndarray, class_index: int) -> dict:
        """Grad-CAM heat map for `class_index`, blended over the 256x256 input, as a PNG data URL.

        The hook is attached only for this call (and serialised with a lock) so normal predictions are unaffected.
        """
        with self._cam_lock:
            cam = GradCAM(self.model, gradcam_layer(self.model, self.arch))
            try:
                heat, _, _ = cam(preprocess_for_cnn(rgb, self.size).to(self.device), class_index=class_index)
            finally:
                cam.close()
                self.model.zero_grad(set_to_none=True)
        buf = BytesIO()
        Image.fromarray(overlay(rgb, heat)).save(buf, format="PNG")
        return {"method": "Grad-CAM", "class_index": class_index,
                "overlay": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")}

    @property
    def calibration(self):
        return {"method": "temperature_scaling", "temperature": self.temperature}


class EnsemblePredictor(Predictor):
    """Average of calibrated member probabilities (each member: its own checkpoint, temperature and TTA setting)."""
    score_type = "calibrated_softmax"

    def __init__(self, entry: dict, device: torch.device):
        super().__init__(entry)
        self.members = [TorchPredictor({**m, "key": m["key"], "name": m["key"], "family": entry["family"],
                                        "kind": "transfer", "feature_set": entry["feature_set"]}, device)
                        for m in entry["members"]]
        self.input = " + ".join(f"{m.arch} ({m.size}px{', TTA' if m.tta_views > 1 else ''})" for m in self.members)
        self.class_keys = self.members[0].class_keys

    def predict_proba(self, rgb: np.ndarray) -> np.ndarray:
        return np.mean([m.predict_proba(rgb) for m in self.members], axis=0)

    def explain(self, rgb: np.ndarray, class_index: int) -> dict:
        out = self.members[0].explain(rgb, class_index)
        out["note"] = f"Grad-CAM of the first ensemble member ({self.members[0].arch})."
        return out

    @property
    def calibration(self):
        return {"method": "temperature_scaling (per member)", "temperature": None,
                "member_temperatures": [m.temperature for m in self.members]}


class SklearnPredictor(Predictor):
    score_type = "predict_proba"
    calibration = None

    def __init__(self, entry: dict):
        super().__init__(entry)
        bundle = joblib.load(MODELS_DIR / entry["path"])
        self.pipeline = bundle["pipeline"]
        self.class_keys = list(bundle.get("class_keys") or CLASS_KEYS)
        if "clf__n_jobs" in self.pipeline.get_params():
            self.pipeline.set_params(clf__n_jobs=1)
        self.input = f"256x256 RGB -> handcrafted features ({self.feature_set})"

    def predict_proba(self, rgb: np.ndarray) -> np.ndarray:
        x = feature_vector(extract_features(rgb), self.feature_set)[None, :]
        return self.pipeline.predict_proba(x)[0]


class ModelRegistry:
    """The models listed in deployment.json. The API loads them all at startup (lazy=False); the Streamlit demo
    passes lazy=True so each model is loaded the first time it is used (less memory on a small free server)."""

    def __init__(self, deployment_path=EXPERIMENTS_DIR / "deployment.json", device: str = "cpu", lazy: bool = False):
        self.device = torch.device(device)
        self.models: dict[str, Predictor] = {}
        self.errors: dict[str, str] = {}
        self.default_key, self.threshold, self.entries = None, 0.6, []
        self._lock = threading.Lock()
        if not deployment_path.exists():
            self.errors["deployment"] = f"{deployment_path.name} not found - run the training pipeline first"
            return
        deployment = load_json(deployment_path)
        self.default_key = deployment["default_model"]
        self.threshold = float(deployment["confidence_threshold"])
        self.entries = deployment["models"]
        if not lazy:
            for entry in self.entries:
                self.load(entry["key"])

    def load(self, key: str) -> Predictor | None:
        """Load one model once. A broken model file is recorded in `errors` (and returns None), never raised:
        it must not take the whole app down."""
        with self._lock:
            if key not in self.models and key not in self.errors:
                entry = next((e for e in self.entries if e["key"] == key), None)
                if entry is None:
                    raise KeyError(f"unknown model {key!r}")
                builders = {"torch": lambda e: TorchPredictor(e, self.device), "sklearn": SklearnPredictor,
                            "ensemble": lambda e: EnsemblePredictor(e, self.device)}
                try:
                    self.models[key] = builders[entry["type"]](entry)
                except Exception as exc:
                    log.exception("Failed to load %s", key)
                    self.errors[key] = str(exc)
            return self.models.get(key)

    @property
    def ready(self) -> bool:
        return self.default_key in self.models

    def threshold_for(self, predictor: Predictor) -> float:
        """The model's own validation-chosen threshold if it has one, else the global one."""
        return float(predictor.threshold) if predictor.threshold is not None else self.threshold

    def predict(self, rgb: np.ndarray, key: str | None = None, explain: bool = False) -> dict:
        predictor = self.models.get(key or self.default_key) or self.load(key or self.default_key)
        if predictor is None:
            raise KeyError(f"model {key or self.default_key!r} is not available")
        t0 = time.perf_counter()
        probs = predictor.predict_proba(rgb)
        elapsed = 1000 * (time.perf_counter() - t0)
        top = int(np.argmax(probs))  # index into this model's own class list
        top_cls = CLASSES[CLASS_KEYS.index(predictor.class_keys[top])]
        explanation = None
        if explain:
            if isinstance(predictor, (TorchPredictor, EnsemblePredictor)):
                explanation = predictor.explain(rgb, top)
                explanation["class_index"] = top_cls["index"]
            else:
                explanation = {"method": None, "overlay": None,
                               "note": "Grad-CAM needs a convolutional network; this classical model uses handcrafted "
                                       "colour/HOG features, whose importance is shown on the Models page."}
        confidence = float(probs[top])
        threshold = self.threshold_for(predictor)
        uncertain = confidence < threshold  # below the threshold the app gives no diagnosis ("Uncertain")
        is_healthy = top_cls["key"] == "healthy"
        status = "Uncertain" if uncertain else ("Healthy" if is_healthy else "Diseased")
        return {
            "model": predictor.info,
            "prediction": {"index": top_cls["index"], "key": top_cls["key"], "name": top_cls["name"],
                           "confidence": confidence, "is_healthy": is_healthy, "status": status},
            "uncertain": uncertain,
            "confidence_threshold": threshold,
            "operating_point": predictor.operating_point,
            "classes": predictor.class_keys,
            "probabilities": [{"index": CLASS_KEYS.index(k), "key": k, "name": CLASSES[CLASS_KEYS.index(k)]["name"],
                               "probability": float(probs[i])} for i, k in enumerate(predictor.class_keys)],
            "score_type": predictor.score_type,
            "calibration": predictor.calibration,
            "inference_ms": elapsed,
            "explanation": explanation,
        }

    def compare(self, rgb: np.ndarray) -> list:
        """Run every available model on the same image (deployment order), without explanations."""
        return [self.predict(rgb, entry["key"]) for entry in self.entries if self.load(entry["key"]) is not None]
