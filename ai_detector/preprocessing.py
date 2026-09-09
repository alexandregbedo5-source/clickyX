"""Prétraitement : chargement robuste, orientation EXIF, conversion, fenêtrage.

Principe directeur : **ne jamais redimensionner** l'image avant analyse. Les
indices forensiques (pics spectraux, traces de dématriçage, périodicité du
décodeur latent) vivent à la résolution native et disparaissent au moindre
rééchantillonnage. On travaille donc sur des fenêtres/crops natifs.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF", "HEIF", "AVIF"}

# Coefficients de luminance ITU-R BT.601 (identiques à PIL "L").
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class ImageLoadError(ValueError):
    """Image illisible ou format non supporté."""


@dataclass
class LoadedImage:
    """Image décodée en float32 [0, 1], canaux RGB, plus métadonnées utiles."""

    rgb: np.ndarray  # H x W x 3, float32 in [0, 1]
    width: int
    height: int
    format: str | None
    mode: str
    has_exif: bool
    source: str
    warnings: list[str] = field(default_factory=list)

    @property
    def min_side(self) -> int:
        return min(self.width, self.height)


def _pil_to_loaded(img: Image.Image, source: str) -> LoadedImage:
    fmt = img.format
    mode = img.mode
    has_exif = bool(getattr(img, "getexif", None) and len(img.getexif()) > 0)
    # Orientation EXIF : indispensable pour que les crops correspondent au rendu.
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB",):
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    warnings: list[str] = []
    if fmt is not None and fmt.upper() not in SUPPORTED_FORMATS:
        warnings.append(f"format_inhabituel:{fmt}")
    return LoadedImage(
        rgb=arr,
        width=arr.shape[1],
        height=arr.shape[0],
        format=fmt,
        mode=mode,
        has_exif=has_exif,
        source=source,
        warnings=warnings,
    )


def load_image_from_path(path: str | Path) -> LoadedImage:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    try:
        with Image.open(p) as img:
            img.load()
            return _pil_to_loaded(img, source=str(p))
    except UnidentifiedImageError as exc:
        raise ImageLoadError(f"Fichier non reconnu comme image : {p}") from exc
    except OSError as exc:
        raise ImageLoadError(f"Impossible de décoder l'image : {p} ({exc})") from exc


def load_image_from_bytes(data: bytes, source: str = "<bytes>") -> LoadedImage:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return _pil_to_loaded(img, source=source)
    except UnidentifiedImageError as exc:
        raise ImageLoadError("Données binaires non reconnues comme image") from exc
    except OSError as exc:
        raise ImageLoadError(f"Impossible de décoder l'image ({exc})") from exc


def load_image_from_base64(payload: str) -> LoadedImage:
    """Accepte le base64 brut ou une data-URL ``data:image/png;base64,...``."""
    if "," in payload and payload.lstrip().lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload, validate=False)
    except (ValueError, TypeError) as exc:
        raise ImageLoadError("Base64 invalide") from exc
    if not raw:
        raise ImageLoadError("Base64 vide")
    return load_image_from_bytes(raw, source="<base64>")


def load_image_from_array(rgb: np.ndarray, source: str = "<array>") -> LoadedImage:
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ImageLoadError("Tableau attendu de forme HxWx3 (ou HxWx4)")
    arr = arr[..., :3]
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    return LoadedImage(
        rgb=np.ascontiguousarray(arr),
        width=arr.shape[1],
        height=arr.shape[0],
        format=None,
        mode="RGB",
        has_exif=False,
        source=source,
    )


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Luminance float32 HxW dans [0, 1]."""
    return np.tensordot(rgb, _LUMA, axes=([2], [0])).astype(np.float32)


def reflect_pad_to_min(arr: np.ndarray, min_size: int) -> np.ndarray:
    """Rembourre par réflexion pour atteindre ``min_size`` sur chaque axe spatial."""
    h, w = arr.shape[:2]
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    pads: list[tuple[int, int]] = [(pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)]
    if arr.ndim == 3:
        pads.append((0, 0))
    # 'symmetric' évite de dupliquer la ligne de bord et supporte des marges plus
    # grandes que l'image (réflexions itérées) pour les très petites vignettes.
    return np.pad(arr, pads, mode="symmetric")


def center_window(arr: np.ndarray, size: int) -> np.ndarray:
    """Fenêtre carrée centrée de côté ``min(size, H, W)`` à résolution native."""
    h, w = arr.shape[:2]
    side = min(size, h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return arr[top:top + side, left:left + side]


def grid_crops(arr: np.ndarray, size: int, max_crops: int = 5) -> list[np.ndarray]:
    """Crops natifs ``size × size`` : centre puis coins, sans chevauchement excessif.

    Si l'image est plus petite que ``size`` sur un axe, elle est rembourrée par
    réflexion (jamais agrandie par interpolation).
    """
    arr = reflect_pad_to_min(arr, size)
    h, w = arr.shape[:2]
    positions: list[tuple[int, int]] = [((h - size) // 2, (w - size) // 2)]
    if max_crops > 1 and (h > size or w > size):
        corners = [(0, 0), (0, w - size), (h - size, 0), (h - size, w - size)]
        for c in corners:
            if c not in positions:
                positions.append(c)
    positions = positions[:max_crops]
    return [np.ascontiguousarray(arr[t:t + size, l:l + size]) for t, l in positions]


def iter_tiles(arr: np.ndarray, size: int, max_tiles: int) -> Iterable[np.ndarray]:
    """Tuiles régulières non chevauchantes (pour analyses locales)."""
    arr = reflect_pad_to_min(arr, size)
    h, w = arr.shape[:2]
    ny, nx = h // size, w // size
    count = 0
    for j in range(ny):
        for i in range(nx):
            if count >= max_tiles:
                return
            yield arr[j * size:(j + 1) * size, i * size:(i + 1) * size]
            count += 1
