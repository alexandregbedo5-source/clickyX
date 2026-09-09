"""Fixtures : images synthétiques de contrôle (aucune donnée externe)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import zoom

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_photo_like(n: int = 512, seed: int = 0) -> np.ndarray:
    """Champ 1/f (statistique d'image naturelle) + zones plates + bruit de grenaille."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    fr = np.sqrt(fx**2 + fy**2)
    fr[0, 0] = 1.0
    phase = np.exp(2j * np.pi * rng.random((n, n)))
    img = np.fft.ifft2(phase / fr**1.1).real
    img = (img - img.min()) / (img.max() - img.min())
    # Un « ciel » plat en dégradé (0,15 → 0,85) : blocs plats à diverses intensités,
    # nécessaires pour mesurer la dépendance bruit/intensité (bruit de grenaille).
    sky = np.linspace(0.15, 0.85, n)[:, None]
    img[: n // 3] = sky[: n // 3]
    rgb = np.stack([img, img * 0.95 + 0.02, img * 0.9 + 0.05], axis=-1)
    shot = rng.normal(0.0, 1.0, rgb.shape) * np.sqrt(rgb * 0.0006 + 0.00002)
    return np.clip(rgb + shot, 0.0, 1.0).astype(np.float32)


def make_upsampled_like(n: int = 512, seed: int = 1, factor: int = 8, checker_amplitude: float = 0.005) -> np.ndarray:
    """Contenu basse résolution agrandi ×factor + damier périodique de période ``factor``.

    Simule les artefacts d'un décodeur génératif (convolutions transposées /
    décodeur latent ×8) : structure périodique faible, invisible à l'œil, mais
    cohérente sur toute l'image.
    """
    rng = np.random.default_rng(seed)
    low = rng.random((n // factor, n // factor, 3))
    up = zoom(low, (factor, factor, 1), order=1)
    checker = np.tile(rng.normal(0.0, 1.0, (factor, factor, 1)), (n // factor, n // factor, 1))
    return np.clip(up + checker_amplitude * checker, 0.0, 1.0).astype(np.float32)


@pytest.fixture(scope="session")
def photo_like() -> np.ndarray:
    return make_photo_like()


@pytest.fixture(scope="session")
def upsampled_like() -> np.ndarray:
    return make_upsampled_like()


@pytest.fixture(scope="session")
def image_files(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("imgs")
    out: dict[str, Path] = {}
    for name, arr in (("photo.png", make_photo_like(320)), ("upsampled.png", make_upsampled_like(320))):
        p = d / name
        Image.fromarray((arr * 255).astype(np.uint8)).save(p)
        out[name] = p
    jpg = d / "photo.jpg"
    Image.fromarray((make_photo_like(320, seed=3) * 255).astype(np.uint8)).save(jpg, quality=85)
    out["photo.jpg"] = jpg
    small = d / "tiny.png"
    Image.fromarray((make_photo_like(40, seed=4) * 255).astype(np.uint8)).save(small)
    out["tiny.png"] = small
    return out


@pytest.fixture(scope="session")
def placeholder_onnx(tmp_path_factory) -> Path:
    """Placeholder ONNX généré à la volée (indépendant du fichier versionné)."""
    pytest.importorskip("torch")
    from training.export_onnx import main as export_main

    out = tmp_path_factory.mktemp("model") / "detector.onnx"
    export_main(["--bootstrap", "--arch", "tiny", "--out", str(out), "--version", "test-placeholder"])
    return out
