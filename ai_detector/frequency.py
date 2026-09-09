"""Analyse fréquentielle (FFT) — module 1 du pipeline.

Hypothèse physique : une photographie suit approximativement une loi de
puissance ``P(f) ∝ 1/f^α`` (α ≈ 2–3) dans le spectre de puissance, sans
structure périodique privilégiée. Les générateurs (upsampling transposé des GAN,
décodeur latent ×8 des modèles de diffusion, super-résolution) introduisent :

* des **pics spectraux** à des fréquences rationnelles (1/2, 1/4, 1/8 cycle/pixel) ;
* un **écart à la loi de puissance** en hautes fréquences (excès ou déficit) ;
* une **rugosité** anormale du profil radial.

Aucun de ces indices n'est universel ; ils sont convertis en probabilité par une
régression logistique calibrée (voir :mod:`ai_detector.calibration`).

Implémentation : spectres de puissance moyennés sur des tuiles natives 256×256
(fenêtre de Hann). La moyenne renforce les périodicités cohérentes de l'image et
atténue le contenu, cf. Corvi et al., « Intriguing properties of synthetic images », 2023.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ai_detector.calibration import FeatureMapping
from ai_detector.preprocessing import iter_tiles, reflect_pad_to_min

TILE = 256
MAX_TILES = 16
EPS = 1e-12

# Fréquences (cycles/pixel) des périodicités typiques d'upsampling ×2/×4/×8.
GRID_FREQS = (0.5, 0.25, 0.125)


@dataclass
class SpectrumFeatures:
    slope: float
    hf_residual: float
    hf_residual_abs: float
    hf_energy_ratio: float
    peak_max_z: float
    peak_density: float
    grid_peak_z: float
    anisotropy: float
    profile_roughness: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


@dataclass
class FrequencyResult:
    fft_score: float
    features: dict[str, float]
    contributions: dict[str, float] = field(default_factory=dict)
    tiles_used: int = 0


def _hann2d(n: int) -> np.ndarray:
    w = np.hanning(n).astype(np.float32)
    return np.outer(w, w)


def _freq_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.fft.fftshift(np.fft.fftfreq(n))  # cycles/pixel in [-0.5, 0.5)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    fr = np.sqrt(fx**2 + fy**2)
    return fx, fy, fr


def average_power_spectrum(gray: np.ndarray, tile: int = TILE, max_tiles: int = MAX_TILES) -> tuple[np.ndarray, int]:
    """Spectre de puissance moyen (centré, tile × tile) sur des tuiles natives."""
    gray = reflect_pad_to_min(gray, tile)
    win = _hann2d(tile)
    acc = np.zeros((tile, tile), dtype=np.float64)
    count = 0
    for t in iter_tiles(gray, tile, max_tiles):
        t = t.astype(np.float32)
        t = (t - t.mean()) * win
        spec = np.fft.fftshift(np.fft.fft2(t))
        acc += np.abs(spec) ** 2
        count += 1
    if count == 0:  # pragma: no cover — iter_tiles garantit ≥ 1 tuile après padding
        raise ValueError("Image trop petite pour l'analyse spectrale")
    return acc / count, count


def radial_profile(power: np.ndarray, n_bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Profil radial moyen : fréquences centrales des anneaux et log-puissance."""
    n = power.shape[0]
    _, _, fr = _freq_grid(n)
    edges = np.linspace(0.0, 0.5, n_bins + 1)
    idx = np.clip(np.digitize(fr.ravel(), edges) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=power.ravel(), minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    prof = sums / np.maximum(counts, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, np.log(prof + EPS)


def spectral_peaks(power: np.ndarray, min_freq: float = 0.08, bg_size: int = 7, z_thresh: float = 4.0) -> dict[str, float]:
    """Détection de pics isolés dans le log-spectre après soustraction d'un fond médian.

    Retourne ``peak_max_z`` (proéminence max en z robuste), ``peak_density``
    (fraction de pixels dépassant ``z_thresh``) et ``grid_peak_z`` (proéminence
    max aux fréquences rationnelles d'upsampling).
    """
    n = power.shape[0]
    fx, fy, fr = _freq_grid(n)
    logp = np.log(power + EPS)
    background = ndimage.median_filter(logp, size=bg_size, mode="reflect")
    resid = logp - background
    mask = fr > min_freq
    vals = resid[mask]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826 + EPS
    z = (resid - med) / mad
    z_masked = np.where(mask, z, -np.inf)

    peak_max_z = float(np.max(z_masked))
    peak_density = float(np.mean(z[mask] > z_thresh))

    grid_z = -np.inf
    # Tolérance ±1 bin autour de chaque fréquence de grille (les deux axes et diagonales).
    df = 1.0 / n
    for g in GRID_FREQS:
        for gx, gy in ((g, 0.0), (0.0, g), (g, g), (-g, 0.0), (0.0, -g), (-g, g), (g, -g), (-g, -g)):
            sel = (np.abs(fx - gx) <= df * 1.01) & (np.abs(fy - gy) <= df * 1.01)
            if np.any(sel):
                grid_z = max(grid_z, float(np.max(z[sel])))
    if not np.isfinite(grid_z):
        grid_z = 0.0
    return {"peak_max_z": peak_max_z, "peak_density": peak_density, "grid_peak_z": grid_z}


def compute_spectrum_features(gray: np.ndarray) -> tuple[SpectrumFeatures, int]:
    power, tiles = average_power_spectrum(gray)
    n = power.shape[0]
    fx, fy, fr = _freq_grid(n)

    # --- Loi de puissance et résidu haute fréquence ---------------------------
    centers, logprof = radial_profile(power)
    fit_band = (centers >= 0.02) & (centers <= 0.25)
    hf_band = (centers >= 0.30) & (centers <= 0.49)
    x = np.log(centers[fit_band])
    y = logprof[fit_band]
    slope, intercept = np.polyfit(x, y, 1)
    pred_hf = slope * np.log(centers[hf_band]) + intercept
    hf_residual = float(np.mean(logprof[hf_band] - pred_hf))

    # Rugosité : dérivée seconde du profil (en log) au-dessus de la bande d'ajustement.
    rough_band = centers >= 0.10
    second_diff = np.diff(logprof[rough_band], n=2)
    profile_roughness = float(np.std(second_diff)) if second_diff.size > 2 else 0.0

    # --- Répartition d'énergie --------------------------------------------------
    total = float(np.sum(power[fr > 0.02])) + EPS
    hf_energy_ratio = float(np.sum(power[fr > 0.25])) / total

    band = (fr >= 0.10) & (fr <= 0.5)
    angle = np.abs(np.arctan2(fy, fx))
    near_axis = band & ((angle < np.deg2rad(10)) | (np.abs(angle - np.pi / 2) < np.deg2rad(10)) | (angle > np.deg2rad(170)))
    near_diag = band & ((np.abs(angle - np.pi / 4) < np.deg2rad(10)) | (np.abs(angle - 3 * np.pi / 4) < np.deg2rad(10)))
    e_axis = float(np.mean(power[near_axis])) + EPS
    e_diag = float(np.mean(power[near_diag])) + EPS
    anisotropy = float(np.log(e_axis / e_diag))

    # --- Pics spectraux -----------------------------------------------------------
    peaks = spectral_peaks(power)

    feats = SpectrumFeatures(
        slope=float(slope),
        hf_residual=hf_residual,
        hf_residual_abs=abs(hf_residual),
        hf_energy_ratio=hf_energy_ratio,
        peak_max_z=peaks["peak_max_z"],
        peak_density=peaks["peak_density"],
        grid_peak_z=peaks["grid_peak_z"],
        anisotropy=anisotropy,
        profile_roughness=profile_roughness,
    )
    return feats, tiles


class FrequencyAnalyzer:
    """Convertit une image en niveaux de gris en ``fft_score`` ∈ [0, 1]."""

    def __init__(self, mapping: FeatureMapping):
        self.mapping = mapping

    def analyze(self, gray: np.ndarray) -> FrequencyResult:
        feats, tiles = compute_spectrum_features(gray)
        values = feats.as_dict()
        score = self.mapping.score(values)
        return FrequencyResult(
            fft_score=float(score),
            features=values,
            contributions=self.mapping.contributions(values),
            tiles_used=tiles,
        )
