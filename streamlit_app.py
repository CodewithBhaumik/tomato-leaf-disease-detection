"""Tomato Leaf Disease Detection - online demo (Streamlit Community Cloud).

Runs the same trained models, preprocessing, calibration and decision rule as the project's FastAPI backend:
backend/services/predictor.py and src/tomatoleaf/ are copied unchanged from the main project by
scripts/15_build_streamlit_app.py. Model weights are downloaded on first use from the GitHub release named in
models_manifest.json and checked against their SHA-256 hashes before they are loaded.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from backend.services import predictor as predictor_module  # noqa: E402
from tomatoleaf.config import CLASSES, MODELS_DIR  # noqa: E402
from tomatoleaf.preprocessing import canonicalize, open_image  # noqa: E402

torch.set_num_threads(int(os.getenv("TOMATO_TORCH_THREADS", "2")))  # the free server has about 2 CPU cores

st.set_page_config(page_title="Tomato Leaf Disease Detection", page_icon="🍅", layout="wide",
                   menu_items={"About": "Tomato leaf disease classification: a BTech Machine Learning Lab project. "
                                        "Not a substitute for an agricultural expert."})

PRIMARY, MUTED, THRESHOLD_COLOR = "#2E7D4F", "#BFCBC1", "#9A6B12"
NAME = {c["key"]: c["name"] for c in CLASSES}
SHORT = {"early_blight": "Early Blight", "healthy": "Healthy", "late_blight": "Late Blight", "leaf_miner": "Leaf Miner",
         "magnesium_deficiency": "Magnesium", "nitrogen_deficiency": "Nitrogen", "potassium_deficiency": "Potassium",
         "spotted_wilt_virus": "Spotted Wilt"}
CITATION = ("Tomato-Village dataset (Gehlot, Saxena & Gandhi, *Multimedia Systems*, 2023, "
            "[doi:10.1007/s00530-023-01158-y](https://doi.org/10.1007/s00530-023-01158-y))")
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "BMP"}  # same rules as the FastAPI upload endpoint
Image.MAX_IMAGE_PIXELS = 60_000_000
_DOWNLOAD_LOCK = threading.Lock()


# ----------------------------------------------------------------------------------------------- data and models

@st.cache_data(show_spinner=False)
def read_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


ANALYTICS = read_json("experiments/analytics.json")
DEPLOYMENT = read_json("experiments/deployment.json")
MANIFEST = read_json("models_manifest.json")
DISEASE_INFO = read_json("backend/data/disease_info.json")
DISEASES = {c["key"]: c for c in DISEASE_INFO["classes"]}
SAMPLES = read_json("assets/test-samples/manifest.json")["samples"]
ENTRIES = {e["key"]: e for e in DEPLOYMENT["models"]}
DEFAULT = DEPLOYMENT["default_model"]


@st.cache_resource(show_spinner=False)
def model_store() -> Path:
    """Where the weights live: TOMATO_MODELS_DIR if set, else experiments/models if writable, else a temp folder."""
    candidates = [Path(os.environ["TOMATO_MODELS_DIR"])] if os.getenv("TOMATO_MODELS_DIR") else []
    candidates += [MODELS_DIR, Path(tempfile.gettempdir()) / "tomato-leaf-models"]
    for folder in candidates:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".write-test"
            probe.write_bytes(b"")
            probe.unlink()
        except OSError:
            continue
        predictor_module.MODELS_DIR = folder  # the predictors load their checkpoints from here
        return folder
    raise RuntimeError("No writable folder for the model files")


@st.cache_resource(show_spinner=False)
def registry():
    model_store()
    return predictor_module.ModelRegistry(device="cpu", lazy=True)  # each model is loaded on first use


def model_files(key: str) -> list[str]:
    entry = ENTRIES[key]
    return [m["path"] for m in entry["members"]] if entry.get("members") else [entry["path"]]


def present(rel: str) -> bool:
    f = model_store() / rel
    return f.exists() and f.stat().st_size == MANIFEST["files"][rel]["bytes"]


def download(rel: str, on_progress) -> None:
    info = MANIFEST["files"][rel]
    base = os.getenv("TOMATO_MODELS_URL") or f"https://github.com/{MANIFEST['repo']}/releases/download/{MANIFEST['tag']}"
    target = model_store() / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest, done = hashlib.sha256(), 0
    with urllib.request.urlopen(f"{base}/{info['asset']}", timeout=60) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            on_progress(done)
    if digest.hexdigest() != info["sha256"]:  # never load a file that is not exactly the one that was evaluated
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{info['asset']} failed its SHA-256 check (corrupted download); please try again")
    partial.replace(target)


def ensure_models(keys) -> None:
    """Download the weight files these models need (only the first time), with a progress bar."""
    todo = [p for p in dict.fromkeys(p for k in keys for p in model_files(k)) if not present(p)]
    if not todo:
        return
    total = sum(MANIFEST["files"][p]["bytes"] for p in todo)
    with st.status(f"Downloading model weights ({total / 1e6:.0f} MB, first use only)…", expanded=True) as status:
        bar, finished = st.progress(0.0), 0
        with _DOWNLOAD_LOCK:  # one download at a time across visitors; re-check each file inside the lock
            for rel in todo:
                if not present(rel):
                    download(rel, lambda n: bar.progress(min(1.0, (finished + n) / total)))
                finished += MANIFEST["files"][rel]["bytes"]
                bar.progress(min(1.0, finished / total))
        status.update(label="Model weights ready", state="complete", expanded=False)


@st.cache_data(show_spinner=False, max_entries=64)
def canonical(data: bytes) -> np.ndarray:
    """Validate an image like the API does; return the 256x256 RGB array every model receives (or raise ValueError)."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
    except Image.DecompressionBombError:
        raise ValueError("The image resolution is too large (maximum 60 megapixels).")
    except Exception:
        raise ValueError("This file is not a readable image. Please use a JPEG, PNG, WEBP or BMP photo.")
    if fmt not in ALLOWED_FORMATS:
        raise ValueError(f"Unsupported image format ({fmt}). Please use a JPEG, PNG, WEBP or BMP photo.")
    try:
        img = open_image(data)
    except Image.DecompressionBombError:
        raise ValueError("The image resolution is too large (maximum 60 megapixels).")
    except Exception:
        raise ValueError("The image file is corrupt or truncated and could not be decoded.")
    return np.asarray(canonicalize(img), dtype=np.uint8)


@st.cache_data(show_spinner=False, max_entries=256)
def analyse(data: bytes, key: str, explain: bool) -> dict:
    """The API's prediction for this photo and model (cached: Streamlit reruns the script on every click)."""
    return registry().predict(canonical(data), key, explain)


# --------------------------------------------------------------------------------------------------- helpers

def pct(x, digits: int = 1) -> str:
    return "—" if x is None else f"{100 * float(x):.{digits}f}%"


def percent_column():
    """Table column for values stored as 0-100, shown with one decimal."""
    return st.column_config.NumberColumn(format="%.1f%%")


def sample_file(key: str) -> Path:
    return ROOT / "assets" / "test-samples" / f"{key}.jpg"


def model_label(key: str) -> str:
    return "Focused 5-class model (recommended)" if key == DEFAULT else ENTRIES[key]["name"]


def operating_point(key: str):
    """Selective-prediction numbers: from the check of this deployed code if available, else the offline evaluation."""
    op = ENTRIES[key].get("operating_point")
    if not op:
        return None
    check = ANALYTICS.get("deployment_check") or {}
    if check.get("model_key") == key and check.get("test"):
        s = check["test"]["in_scope"]
        return {"threshold": op["threshold"], "photos": s["photos"], "answered": s["answered"], "coverage": s["coverage"],
                "accuracy": s["accuracy_answered"], "out_of_scope": check["test"]["out_of_scope"]}
    n = op.get("test_photos")
    return {"threshold": op["threshold"], "photos": n, "answered": round(op["test"]["coverage"] * n) if n else None,
            "coverage": op["test"]["coverage"], "accuracy": op["test"]["accuracy_kept"], "out_of_scope": None}


def selected_8class():
    return next(m for m in ANALYTICS["models"] if m["selected"])


def best_classical():  # chosen like everything else: by validation macro-F1
    return max((m for m in ANALYTICS["models"] if m["family"] == "Machine Learning"), key=lambda m: m["val"]["f1_macro"])


def five_selected():
    five = ANALYTICS["five_class"]
    return next(m for m in five["models"] if m["key"] == five["selected"])


def ci(pair, digits: int = 1) -> str:
    return f"{pct(pair[0], digits)}–{pct(pair[1], digits)}" if pair else "—"


def disease_details(info: dict) -> None:
    if info.get("cause"):
        st.caption("Cause: " + info["cause"])
    left, right = st.columns(2, gap="medium")
    left.markdown("**Symptoms**\n\n" + "\n".join(f"- {s}" for s in info.get("symptoms", [])))
    right.markdown(("**Good practice**" if info.get("is_healthy") else "**Management**") + "\n\n"
                   + "\n".join(f"- {s}" for s in info.get("management", [])))
    refs = info.get("references") or []
    if refs:
        st.markdown("**Sources:** " + " · ".join(f"[{r['title']}]({r['url']}) ({r['publisher']})" for r in refs))


def footer() -> None:
    st.divider()
    st.caption(f"Photos: {CITATION}, shown with citation for academic demonstration. "
               + DISEASE_INFO.get("disclaimer", "This is an experimental tool, not a substitute for an agricultural expert."))


def page_style() -> None:
    """Less empty space above the page title (the default leaves room for a sidebar header we do not use)."""
    st.html("<style>[data-testid='stMainBlockContainer'], .block-container { padding-top: 3.5rem; }</style>")


def hide_from_search_engines() -> None:
    """Ask crawlers not to index this demo (it shows photos from a dataset that has no licence)."""
    st.html("""<script>
      (function () {
        var d = document;
        try { if (window.parent && window.parent.document) { d = window.parent.document; } } catch (e) {}
        if (!d.querySelector('meta[name="robots"]')) {
          var m = d.createElement('meta'); m.name = 'robots'; m.content = 'noindex, nofollow'; d.head.appendChild(m);
        }
      })();
    </script>""", unsafe_allow_javascript=True)


def probability_chart(probs: list, top: str, threshold: float) -> alt.LayerChart:
    df = pd.DataFrame({"Class": [p["name"] for p in probs], "Probability": [p["probability"] for p in probs],
                       "Color": [PRIMARY if p["key"] == top else MUTED for p in probs]})
    base = alt.Chart(df).encode(
        y=alt.Y("Class:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200, labelFontSize=13)),
        x=alt.X("Probability:Q", title=None, scale=alt.Scale(domain=[0, 1.14]),
                axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], format="%")),
        tooltip=[alt.Tooltip("Class:N"), alt.Tooltip("Probability:Q", format=".1%")])
    bars = base.mark_bar(cornerRadiusEnd=4, size=18).encode(color=alt.Color("Color:N", scale=None, legend=None))
    labels = base.mark_text(align="left", dx=5, fontSize=12).encode(text=alt.Text("Probability:Q", format=".1%"))
    rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(strokeDash=[5, 4], color=THRESHOLD_COLOR).encode(x="x:Q")
    return (bars + labels + rule).properties(height=34 * len(df) + 24)


# --------------------------------------------------------------------------------------------- page: detect

def choose_sample(key: str) -> None:
    st.session_state["sample"] = key


def pick_model() -> str:
    st.subheader("1 · Model", anchor=False)
    key = st.selectbox("Model", list(ENTRIES), format_func=model_label, key="model", label_visibility="collapsed")
    entry, scope = ENTRIES[key], ENTRIES[key]["class_keys"]
    if len(scope) < len(CLASSES):
        st.markdown(f"**Recognises {len(scope)} conditions:** " + ", ".join(NAME[k] for k in scope))
        st.caption("Not covered: " + ", ".join(NAME[k] for k in NAME if k not in scope)
                   + f". A photo of one of these still gets one of the {len(scope)} labels above; choose an "
                   f"{len(CLASSES)}-class model to include them.")
    else:
        st.markdown(f"**Recognises all {len(CLASSES)} conditions.**")
    op = operating_point(key)
    if op:
        st.info(f"Gives a diagnosis only when it is at least **{pct(op['threshold'], 0)}** confident. On "
                f"{op['photos']} unseen test photos of its {len(scope)} conditions it answered {op['answered']} "
                f"({pct(op['coverage'], 0)}), and **{pct(op['accuracy'])}** of those answers were correct. "
                "The rest are reported as “Uncertain”.", icon=":material/verified:")
    else:
        m = entry.get("test_metrics") or {}
        st.caption(f"Test accuracy {pct(m.get('accuracy'))}, macro-F1 {pct(m.get('f1_macro'))} on "
                   f"{ANALYTICS['dataset']['clean_split']['test']} unseen photos of all {len(CLASSES)} classes. Below "
                   f"{pct(DEPLOYMENT['confidence_threshold'], 0)} confidence a result is reported as “Uncertain”.")
    return key


def pick_photo(scope: list):
    st.subheader("2 · Leaf photo", anchor=False)
    source = st.segmented_control("Photo source", ["Test photos", "Upload", "Camera"], default="Test photos",
                                  required=True, key="source", label_visibility="collapsed")
    if source == "Upload":
        f = st.file_uploader("Upload a photo of one tomato leaf (JPEG, PNG, WEBP or BMP, up to 10 MB)",
                             type=["jpg", "jpeg", "png", "webp", "bmp"], key="upload")
        return (f.getvalue(), None, f.name) if f else (None, None, None)
    if source == "Camera":
        f = st.camera_input("Photograph one leaf on a plain, light background", key="camera")
        return (f.getvalue(), None, "camera photo") if f else (None, None, None)
    st.caption("One held-out test photo per condition, never used for training or model selection. "
               "The button shows the true label.")
    chosen = st.session_state.get("sample")
    for start in range(0, len(SAMPLES), 4):  # one row of columns per 4 photos keeps the rows aligned
        for col, s in zip(st.columns(4, gap="small"), SAMPLES[start:start + 4]):
            with col:
                st.image(str(sample_file(s["key"])), width="stretch")
                st.button(SHORT[s["key"]], key=f"pick_{s['key']}", width="stretch", on_click=choose_sample,
                          args=(s["key"],), type="primary" if chosen == s["key"] else "secondary",
                          help=f"True label: {s['name']}" + ("" if s["key"] in scope else " (outside this model's scope)"))
                if s["key"] not in scope:
                    st.caption("not in this model")
    if chosen is None:
        return None, None, None
    return sample_file(chosen).read_bytes(), chosen, f"held-out test photo, true label {NAME[chosen]}"


def compare_all(data: bytes, truth) -> None:
    ensure_models(list(ENTRIES))
    rows = []
    with st.spinner("Running every model on this photo…"):
        for key in ENTRIES:
            r = analyse(data, key, False)
            p = r["prediction"]
            row = {"Model": model_label(key), "Classes": len(r["classes"]), "Prediction": p["name"],
                   "Confidence": 100 * p["confidence"], "Result": "Uncertain" if r["uncertain"] else "Diagnosis"}
            if truth:
                row["Correct?"] = "not in scope" if truth not in r["classes"] else ("yes" if p["key"] == truth else "no")
            rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Confidence": st.column_config.ProgressColumn("Confidence", format="%.1f%%",
                                                                              min_value=0, max_value=100)})
    st.caption("“Correct?” checks the top class even when the result is “Uncertain”. Each model is scored only on the "
               "classes it was trained on, so 5-class and 8-class results are not directly comparable. Logistic "
               "Regression, SVM and Random Forest use handcrafted colour features; the others are convolutional networks.")


def show_result(key: str, data, truth, source_label) -> None:
    st.subheader("3 · Result", anchor=False)
    if data is None:
        with st.container(border=True):
            st.markdown("Choose a test photo, upload a photo or use your camera. The result appears here.")
            st.caption("The very first analysis can take about a minute while the server downloads and loads the model.")
        return
    try:
        rgb = canonical(data)
    except ValueError as exc:
        st.error(str(exc), icon=":material/error:")
        return
    try:
        ensure_models([key])
        with st.spinner("Analysing the leaf…"):
            r = analyse(data, key, True)
    except Exception as exc:  # e.g. the model files could not be downloaded
        st.error(f"The model could not run: {exc}", icon=":material/error:")
        return

    pred, threshold = r["prediction"], r["confidence_threshold"]
    confidence, scope = pred["confidence"], r["classes"]
    with st.container(border=True):
        if r["uncertain"]:
            st.markdown(":orange-badge[:material/help: Uncertain: no diagnosis]")
            st.markdown("### Not confident enough to diagnose", anchors=False)
            st.markdown(f"Top guess: **{pred['name']}** at {pct(confidence)} confidence, below the "
                        f"{pct(threshold, 0)} threshold. Retake a sharp photo of one leaf on a plain, light background, "
                        "or ask an agricultural expert.")
        else:
            badge = (":green-badge[:material/eco: Healthy]" if pred["is_healthy"]
                     else ":red-badge[:material/coronavirus: Disease or deficiency detected]")
            st.markdown(badge)
            st.markdown(f"### {pred['name']}", anchors=False)
            st.markdown(f"Confidence **{pct(confidence)}**: above the {pct(threshold, 0)} threshold, so the app "
                        "gives a diagnosis.")
        st.progress(min(1.0, confidence), text=f"Confidence {pct(confidence)} (threshold {pct(threshold, 0)})")
        if truth:
            right = pred["key"] == truth
            if truth not in scope:
                verdict = ":gray-badge[outside this model's scope]"
                note = " This model was not trained on it, so it cannot get it right."
            elif r["uncertain"]:
                verdict = ":orange-badge[no diagnosis given]"
                note = f" (the top guess was {'right' if right else 'wrong'})."
            else:
                verdict = ":green-badge[:material/check: correct]" if right else ":red-badge[:material/close: wrong]"
                note = ""
            st.markdown(f"True label: **{NAME[truth]}** {verdict}{note}")
        st.caption(f"{r['model']['name']} · {source_label} · {r['inference_ms']:.0f} ms on the server's CPU")

    left, right = st.columns(2, gap="medium")
    left.image(rgb, caption="What the model sees: the photo centre-cropped to 256 × 256", width="stretch")
    explanation = r.get("explanation") or {}
    if explanation.get("overlay"):
        right.image(base64.b64decode(explanation["overlay"].split(",", 1)[1]), width="stretch",
                    caption=f"Grad-CAM: the regions that most raised the score for {pred['name']}")
        if explanation.get("note"):
            right.caption(explanation["note"])
    else:
        right.info(explanation.get("note") or "No heat map for this model.", icon=":material/info:")

    st.markdown(f"**Scores for the {len(scope)} classes this model knows**")
    st.altair_chart(probability_chart(r["probabilities"], pred["key"], threshold), width="stretch")
    st.caption("Calibrated probabilities (temperature scaling); the dashed line is the confidence threshold. They show "
               "how strongly the model prefers each class it was trained on. They are not a guarantee, and a photo of a "
               "condition outside these classes is still given one of them.")

    info = DISEASES.get(pred["key"])
    if info:
        with st.expander(f"About {info['name']}" + (" (the top guess)" if r["uncertain"] else "")):
            st.write(info.get("summary", ""))
            disease_details(info)

    if st.toggle(f"Compare all {len(ENTRIES)} models on this photo", key="compare_all"):
        compare_all(data, truth)


def page_detect() -> None:
    st.title("Tomato leaf disease detection", anchor=False)
    st.markdown("Upload a photo of one tomato leaf, take one with your camera, or try a **held-out test photo**: one "
                "the models never saw while they were trained or selected.")
    left, right = st.columns([5, 7], gap="large")
    with left:
        key = pick_model()
        data, truth, source_label = pick_photo(ENTRIES[key]["class_keys"])
    with right:
        show_result(key, data, truth, source_label)
    footer()


# -------------------------------------------------------------------------------------------- page: results

def page_results() -> None:
    five, sel5, sel8, ml = ANALYTICS["five_class"], five_selected(), selected_8class(), best_classical()
    op, check = operating_point(DEFAULT), ANALYTICS.get("deployment_check")
    st.title("Results", anchor=False)
    st.markdown("Every number on this page comes from the project's saved results. Models were chosen on validation "
                "data only, and the held-out test photos were scored once. The split keeps every leaf (and its "
                "augmented copies) in a single part, so no test leaf was seen in training.")

    cols = st.columns(4)
    cols[0].metric("Focused 5-class model", pct(sel5["test"]["accuracy"]), border=True,
                   help="Test accuracy of the web app's default model on held-out photos of its 5 classes.")
    cols[0].caption(f"test accuracy · 95% CI {ci(sel5['test'].get('accuracy_ci'))} · {five['split']['test']} photos")
    if op:
        cols[1].metric("5-class model, when confident", pct(op["accuracy"]), border=True,
                       help="Accuracy on the photos where the default model is confident enough to give a diagnosis.")
        cols[1].caption(f"answers {pct(op['coverage'], 0)} of photos (confidence ≥ {pct(op['threshold'], 0)})")
    cols[2].metric("Best 8-class model", pct(sel8["test"]["accuracy"]), border=True,
                   help="Selected by validation macro-F1 among all 8-class candidates.")
    cols[2].caption(f"{sel8['name']} · {ANALYTICS['dataset']['clean_split']['test']} photos")
    cols[3].metric("Best classical ML (8-class)", pct(ml["test"]["accuracy"]), border=True,
                   help="Best classical model by validation macro-F1.")
    cols[3].caption(f"{ml['name']} on colour features")

    st.subheader("How the deployed model decides", anchor=False)
    sel = five.get("selective") or {}
    st.markdown(f"The default model gives a diagnosis only when its calibrated confidence is at least "
                f"**{pct(sel.get('threshold'), 0)}**. That threshold was fixed on the validation photos: the lowest one "
                f"at which accuracy on the answered photos reached {pct(sel.get('target_accuracy'), 0)} while still "
                "answering at least half of them. It was then applied once to the test photos. Lower-confidence photos "
                "are reported as “Uncertain” instead of being guessed.")
    if sel:
        rows = [{"": "Photos answered", "Validation": pct(sel["val"]["coverage"]), "Test": pct(sel["test"]["coverage"])},
                {"": "Accuracy when answering", "Validation": pct(sel["val"]["accuracy_kept"]),
                 "Test": pct(sel["test"]["accuracy_kept"])},
                {"": "Accuracy on the “Uncertain” photos", "Validation": pct(sel["val"]["accuracy_rejected"]),
                 "Test": pct(sel["test"]["accuracy_rejected"])}]
        if check and check.get("test"):
            s = check["test"]["in_scope"]
            for row, value in zip(rows, (s["coverage"], s["accuracy_answered"], s["accuracy_referred"])):
                row["Test, deployed app"] = pct(value)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if check and check.get("test"):
        s, o = check["test"]["in_scope"], check["test"]["out_of_scope"]
        st.markdown(
            f"**Checked through the deployed code.** All {s['photos']} in-scope test photos were sent through the "
            f"app's prediction code (CPU, float32): {s['answered']} answered, {s['answered_correct']} of them correct, "
            f"and overall accuracy {pct(s['accuracy_all'])}, the same as the offline evaluation. The offline run used "
            "the GPU in mixed precision, which moves confidences by up to about 0.004, so a photo sitting right at the "
            "threshold can land on the other side; that is why the coverage figures can differ by one photo.")
        st.warning(
            f"**Limit: conditions outside the scope.** Of the {o['photos']} test photos of the three dropped conditions "
            f"(Early Blight, Late Blight, Potassium Deficiency), only {o['referred']} were held back as “Uncertain”; "
            f"the other {o['answered']} received one of the five in-scope labels, which is necessarily wrong. The "
            "confidence threshold does not detect unknown conditions. Use an 8-class model for them.",
            icon=":material/warning:")

    st.subheader("Why not the 97.6% you get on the official split?", anchor=False)
    lk, ds = ANALYTICS["dataset"]["leakage"], ANALYTICS["dataset"]
    st.markdown(f"The dataset ships {ds['total_images']:,} images but only {ds['unique_photos']:,} unique photographs: "
                f"{ds['augmented_copies']:,} files are pre-augmented copies, spread across the official train, "
                f"validation and test folders. {lk['test_images_with_train_sibling']} of the {lk['test_images']} official "
                f"test images ({pct(lk['test_images_with_train_sibling'] / lk['test_images'])}) have a copy of the same "
                "photo in the training folder, so a model can score highly by recognising photos it has already seen. "
                "This project re-split the original photographs by leaf, and reports only those leakage-free results.")
    bench = pd.DataFrame(ANALYTICS["leakage_benchmark"])
    if not bench.empty:
        table = bench.pivot_table(index="model", columns="protocol", values="accuracy", sort=False).reset_index()
        table = table.rename(columns={"model": "Model"})
        protocols = [c for c in table.columns if c != "Model"]
        table[protocols] = table[protocols] * 100
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c + " accuracy", format="%.1f%%") for c in protocols})

    st.subheader(f"All {len(CLASSES)}-class models", anchor=False)
    rows = [{"Model": ("★ " if m["selected"] else "") + m["name"], "Family": m["family"],
             "Val macro-F1": 100 * m["val"]["f1_macro"], "Test accuracy": 100 * m["test"]["accuracy"],
             "Test macro-F1": 100 * m["test"]["f1_macro"], "Macro-F1 95% CI": ci(m["test"].get("f1_macro_ci")),
             "CPU ms / photo": m.get("inference_ms_per_image")} for m in ANALYTICS["models"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Val macro-F1": percent_column(), "Test accuracy": percent_column(), "Test macro-F1": percent_column(),
        "CPU ms / photo": st.column_config.NumberColumn(format="%.0f")})
    st.caption("★ = selected by validation macro-F1 (macro-F1 counts every class equally, which matters because the "
               "classes are very unbalanced).")

    st.subheader(f"Focused {len(five['classes'])}-class model", anchor=False)
    st.markdown(f"Scope rule: {five['rule']}. Dropped: " + ", ".join(NAME[k] for k in five["dropped"]) + ".")
    eight = five.get("eight_class_model_on_same_photos")
    if eight:
        st.caption(f"For a fair comparison: on the same {eight['photos']} photos the 8-class model scores "
                   f"{pct(eight['accuracy'])}, so most of the gain comes from the narrower task.")
    rows = [{"Model": ("★ " if m["selected"] else "") + m["name"], "Family": m["family"],
             "Val macro-F1": 100 * m["val"]["f1_macro"], "Test accuracy": 100 * m["test"]["accuracy"],
             "Accuracy 95% CI": ci(m["test"].get("accuracy_ci")), "Test macro-F1": 100 * m["test"]["f1_macro"]}
            for m in five["models"]]
    rows.sort(key=lambda r: -r["Val macro-F1"])
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Val macro-F1": percent_column(), "Test accuracy": percent_column(),
                                "Test macro-F1": percent_column()})

    cm = sel5.get("confusion_matrix")
    if cm:
        names = [SHORT[k] for k in five["classes"]]
        cells = pd.DataFrame([{"True": names[i], "Predicted": names[j], "Photos": int(cm[i][j])}
                              for i in range(len(names)) for j in range(len(names))])
        base = alt.Chart(cells).encode(x=alt.X("Predicted:N", sort=names, axis=alt.Axis(labelAngle=0, orient="top")),
                                       y=alt.Y("True:N", sort=names))
        heat = base.mark_rect(cornerRadius=3, stroke="#FBFBF8", strokeWidth=2).encode(  # empty cells stay near-white
            color=alt.Color("Photos:Q", scale=alt.Scale(type="symlog", range=["#F3F6F2", "#1F5C38"]), legend=None),
            tooltip=["True:N", "Predicted:N", "Photos:Q"])
        text = base.mark_text(fontSize=13).encode(
            text="Photos:Q", color=alt.condition("datum.Photos > 40", alt.value("white"), alt.value("#17251B")))
        st.markdown(f"**Confusion matrix of the selected model** ({five['split']['test']} test photos; rows = true class)")
        st.altair_chart((heat + text).properties(height=320), width="stretch")
    footer()


# -------------------------------------------------------------------------------------------- page: library

def page_library() -> None:
    st.title("Disease library", anchor=False)
    st.markdown("The eight leaf conditions in the dataset, with symptoms, management guidance and sources. "
                f"Photos: {CITATION}.")
    scope = ENTRIES[DEFAULT]["class_keys"]
    for key, info in DISEASES.items():
        with st.container(border=True):
            photos, text = st.columns([2, 3], gap="medium")
            files = sorted((ROOT / "assets" / "samples").glob(f"{key}-*.jpg"))
            for col, f in zip(photos.columns(max(1, len(files)), gap="small"), files):
                col.image(str(f), width="stretch")
            with text:
                st.subheader(info["name"], anchor=False)
                st.markdown(f":gray-badge[{info.get('category', '')}] "
                            + (":green-badge[in the default model]" if key in scope else ":gray-badge[8-class models only]"))
                st.write(info.get("summary", ""))
                with st.expander("Symptoms, management and sources"):
                    disease_details(info)
    footer()


# ---------------------------------------------------------------------------------------------- page: about

def page_about() -> None:
    sel5, sel8, ml, op = five_selected(), selected_8class(), best_classical(), operating_point(DEFAULT)
    st.title("About this project", anchor=False)
    st.markdown(f"""
A machine-learning lab project that classifies photos of tomato leaves into eight conditions (healthy, two fungal
diseases, a viral disease, an insect pest and three nutrient deficiencies) and compares **classical machine learning**
on handcrafted features with **deep learning**.

**How it was built**
- **Data:** {CITATION}. Its official split leaks: augmented copies of the same photo sit in both train and test.
  The project re-split the {ANALYTICS['dataset']['unique_photos']:,} original photographs by leaf
  ({' / '.join(str(v) for v in ANALYTICS['dataset']['clean_split'].values())} train / validation / test).
- **Classical ML:** Logistic Regression, SVM and Random Forest on colour and HOG features, tuned with grouped cross-validation.
- **Deep learning:** a custom CNN and fine-tuned MobileNetV2, EfficientNet-B0 and ConvNeXt-Tiny, with test-time
  augmentation and ensembles. Everything is selected on validation macro-F1; the test set is scored once.
- **Reliability:** temperature-scaled probabilities, bootstrap confidence intervals over leaves, Grad-CAM heat maps,
  and an “Uncertain” answer when the model is not confident.

**Results on unseen leaves**
- Best classical model ({ml['name']}): {pct(ml['test']['accuracy'])} accuracy (8 classes).
- Best 8-class deep model ({sel8['name']}): {pct(sel8['test']['accuracy'])} accuracy.
- Focused 5-class model (this app's default): {pct(sel5['test']['accuracy'])} accuracy;
  {pct(op['accuracy']) if op else '—'} on the {pct(op['coverage'], 0) if op else '—'} of photos where it is confident.

**Limits.** The photos show detached leaves on white paper, so field photos with busy backgrounds may fail. Some classes
have very few photos (for example nitrogen and potassium deficiency), and the 5-class model cannot recognise the three
conditions it was not trained on.

This online version runs the project's own prediction code on Streamlit Community Cloud. The full project, with the
React + FastAPI web app, the report and every experiment, was submitted for the BTech Machine Learning Lab.
""")
    st.link_button("Source code of this demo on GitHub", f"https://github.com/{MANIFEST['repo']}",
                   icon=":material/code:")
    footer()


# ---------------------------------------------------------------------------------------------------- router

navigation = st.navigation([
    st.Page(page_detect, title="Detect", icon=":material/eco:", url_path="detect", default=True),
    st.Page(page_results, title="Results", icon=":material/monitoring:", url_path="results"),
    st.Page(page_library, title="Disease library", icon=":material/local_library:", url_path="library"),
    st.Page(page_about, title="About", icon=":material/info:", url_path="about"),
], position="top")
page_style()
hide_from_search_engines()
navigation.run()
