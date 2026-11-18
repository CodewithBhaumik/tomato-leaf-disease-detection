"""Image preprocessing shared by training scripts and the FastAPI backend.

Training and inference both go through `open_image` -> `canonicalize` -> model-specific step,
so an uploaded photo is transformed exactly like a dataset image.
"""
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF

from .config import CANONICAL_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODEL_INPUT_SIZE


def open_image(source) -> Image.Image:
    """Open a path, bytes or file object; apply EXIF orientation; return an RGB image."""
    if isinstance(source, (bytes, bytearray)):
        source = BytesIO(source)
    img = Image.open(source)
    img.load()
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)
    return img.convert("RGB")


def canonicalize(img: Image.Image, size: int = CANONICAL_SIZE) -> Image.Image:
    """Centre-crop to a square and resize to 256x256 (the dataset geometry). No-op for dataset images."""
    w, h = img.size
    if w != h:
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def load_canonical_array(source) -> np.ndarray:
    """Path/bytes -> canonical 256x256x3 uint8 RGB array."""
    return np.asarray(canonicalize(open_image(source)), dtype=np.uint8)


def to_image_tensor(rgb: np.ndarray) -> tv_tensors.Image:
    """HxWx3 uint8 array -> 3xHxW uint8 tensor (copies read-only arrays, e.g. from np.asarray(PIL image))."""
    rgb = np.ascontiguousarray(rgb)
    if not rgb.flags.writeable:
        rgb = rgb.copy()
    return tv_tensors.Image(torch.from_numpy(rgb).permute(2, 0, 1))


class RandomRotateWithPaperFill(torch.nn.Module):
    """Rotate by a random angle and fill the exposed corners with the image's own border (paper) colour.

    The authors' pre-augmented copies have black rotation corners in only 5 classes; filling with the
    paper colour avoids teaching the model that artifact.
    """

    def __init__(self, degrees: float = 30.0, p: float = 0.5):
        super().__init__()
        self.degrees, self.p = degrees, p

    def forward(self, img):
        if torch.rand(1).item() >= self.p:
            return img
        border = torch.cat([img[:, :4, :].reshape(3, -1), img[:, -4:, :].reshape(3, -1),
                            img[:, :, :4].reshape(3, -1), img[:, :, -4:].reshape(3, -1)], dim=1)
        fill = border.float().median(dim=1).values.round().int().tolist()
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees).item())
        return TF.rotate(img, angle, interpolation=T.InterpolationMode.BILINEAR, fill=fill)


def train_transform(size: int = MODEL_INPUT_SIZE) -> T.Compose:
    """Training-only augmentation. No hue jitter: colour is the diagnostic signal for deficiencies."""
    return T.Compose([
        RandomRotateWithPaperFill(degrees=30, p=0.5),
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.85, 1.18), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int = MODEL_INPUT_SIZE) -> T.Compose:
    """Deterministic transform for validation, test and live inference."""
    return T.Compose([
        T.Resize(size, antialias=True),
        T.CenterCrop(size),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


_EVAL = {}


def preprocess_for_cnn(rgb: np.ndarray, size: int = MODEL_INPUT_SIZE) -> torch.Tensor:
    """Canonical 256x256x3 uint8 array -> normalised 1x3xSxS float tensor (inference path; S = model input size)."""
    if size not in _EVAL:
        _EVAL[size] = eval_transform(size)
    return _EVAL[size](to_image_tensor(rgb)).unsqueeze(0).as_subclass(torch.Tensor)


def dihedral_views(x: torch.Tensor, n_views: int = 8) -> torch.Tensor:
    """Test-time augmentation: the 8 flips/90-degree rotations of a batch (a detached leaf has no 'up')."""
    views = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(-2, -1))
        views += [r, torch.flip(r, dims=(-1,))]
    return torch.cat(views[:n_views], dim=0)
