"""Augmentations orientées robustesse forensique.

Principes :

* **Crops natifs** (jamais de redimensionnement systématique) : le réseau doit
  apprendre les artefacts à l'échelle du pixel.
* **Dégradations réalistes** appliquées aléatoirement pour éviter le
  surapprentissage à un format/pipeline : recompression JPEG, rééchantillonnage,
  flou léger. Elles sont appliquées aux deux classes, empêchant le modèle
  d'exploiter des raccourcis (ex. « PNG ⇒ IA », « JPEG ⇒ réel »).
* Pas de jitter de couleur agressif : il modifierait les statistiques de bruit.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class AugmentConfig:
    jpeg_p: float = 0.5
    jpeg_quality: tuple[int, int] = (55, 100)
    rescale_p: float = 0.3
    rescale_range: tuple[float, float] = (0.5, 1.5)
    blur_p: float = 0.1
    blur_sigma: tuple[float, float] = (0.3, 1.2)
    hflip_p: float = 0.5
    grayscale_p: float = 0.0


class RandomJPEG:
    def __init__(self, p: float, quality: tuple[int, int]):
        self.p, self.quality = p, quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        q = random.randint(*self.quality)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, subsampling=random.choice([0, 2]))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomRescale:
    _RESAMPLERS = (Image.BILINEAR, Image.BICUBIC, Image.LANCZOS)

    def __init__(self, p: float, scale: tuple[float, float], min_side: int = 64):
        self.p, self.scale, self.min_side = p, scale, min_side

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        s = random.uniform(*self.scale)
        w, h = img.size
        nw, nh = max(self.min_side, int(round(w * s))), max(self.min_side, int(round(h * s)))
        return img.resize((nw, nh), random.choice(self._RESAMPLERS))


class RandomGaussianBlur:
    def __init__(self, p: float, sigma: tuple[float, float]):
        self.p, self.sigma = p, sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.sigma)))


class RandomHorizontalFlip:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < self.p else img


def _pad_to_min(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    ph, pw = max(0, size - h), max(0, size - w)
    if ph == 0 and pw == 0:
        return arr
    return np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)), mode="symmetric")


class RandomCropNative:
    """Crop aléatoire ``size × size`` à résolution native (rembourrage symétrique si besoin)."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> np.ndarray:
        arr = _pad_to_min(np.asarray(img, dtype=np.uint8), self.size)
        h, w = arr.shape[:2]
        top = random.randint(0, h - self.size)
        left = random.randint(0, w - self.size)
        return arr[top:top + self.size, left:left + self.size]


class CenterCropNative:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> np.ndarray:
        arr = _pad_to_min(np.asarray(img, dtype=np.uint8), self.size)
        h, w = arr.shape[:2]
        top, left = (h - self.size) // 2, (w - self.size) // 2
        return arr[top:top + self.size, left:left + self.size]


class ToNormalizedTensor:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, arr: np.ndarray):
        import torch

        x = (arr.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class Compose:
    def __init__(self, ops):
        self.ops = list(ops)

    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        return x


def build_train_transform(size: int, cfg: AugmentConfig | None = None, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> Compose:
    cfg = cfg or AugmentConfig()
    return Compose([
        RandomRescale(cfg.rescale_p, cfg.rescale_range, min_side=size // 2),
        RandomHorizontalFlip(cfg.hflip_p),
        RandomGaussianBlur(cfg.blur_p, cfg.blur_sigma),
        RandomJPEG(cfg.jpeg_p, cfg.jpeg_quality),
        RandomCropNative(size),
        ToNormalizedTensor(mean, std),
    ])


def build_eval_transform(size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> Compose:
    return Compose([CenterCropNative(size), ToNormalizedTensor(mean, std)])


def build_degradation(kind: str):
    """Dégradations déterministes pour le protocole de robustesse (evaluate.py)."""
    if kind == "none":
        return lambda img: img
    if kind.startswith("jpeg"):
        q = int(kind[4:])
        def _jpeg(img: Image.Image) -> Image.Image:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        return _jpeg
    if kind.startswith("resize"):
        s = float(kind[6:])
        def _resize(img: Image.Image) -> Image.Image:
            w, h = img.size
            return img.resize((max(64, int(w * s)), max(64, int(h * s))), Image.BICUBIC)
        return _resize
    if kind.startswith("blur"):
        sigma = float(kind[4:])
        return lambda img: img.filter(ImageFilter.GaussianBlur(radius=sigma))
    raise ValueError(f"dégradation inconnue : {kind}")
