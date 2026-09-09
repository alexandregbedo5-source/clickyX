"""Analyse du bruit résiduel — module 2 du pipeline.

Hypothèse physique : le bruit d'une photographie provient d'un capteur réel.
Il possède des propriétés mesurables qu'un décodeur génératif ne reproduit pas
naturellement :

* **bruit de grenaille (shot noise)** : la variance du bruit croît avec
  l'intensité locale (statistique de Poisson des photons) ;
* **traces de dématriçage (CFA/Bayer)** : l'interpolation du canal vert crée une
  périodicité de période 2 dans la dérivée seconde (Gallagher & Chen, 2008) ;
* **absence de périodicité artificielle** : le résidu d'une image synthétique
  présente souvent des pics d'autocorrélation aux pas d'upsampling du décodeur
  (×8 pour les VAE de diffusion latente, cf. Corvi et al., 2023) et des pics
  dans son spectre.

Le résidu est ``x − médiane3×3(x)`` (débruiteur simple, rapide, sans dépendance).
Les statistiques de bruit sont mesurées sur des **blocs plats** (faible variance
du contenu débruité) pour ne pas confondre bruit et texture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, stats

from ai_detector.calibration import FeatureMapping
from ai_detector.frequency import average_power_spectrum, spectral_peaks
from ai_detector.preprocessing import reflect_pad_to_min, to_grayscale

BLOCK = 16
FLAT_STD_THRESHOLD = 0.02
MIN_FLAT_BLOCKS = 20
AC_LAGS = (2, 4, 8, 16)
EPS = 1e-12


@dataclass
class NoiseFeatures:
    flat_noise_std: float
    shot_noise_corr: float
    flat_kurtosis: float
    residual_ac_peak: float
    residual_spec_peak_z: float
    residual_grid_peak_z: float
    cfa_strength: float
    jpeg_grid_strength: float
    channel_residual_corr: float
    flat_block_fraction: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


@dataclass
class NoiseResult:
    noise_score: float
    features: dict[str, float]
    contributions: dict[str, float] = field(default_factory=dict)
    flat_blocks: int = 0


def median_residual(channel: np.ndarray, size: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (résidu, image débruitée)."""
    den = ndimage.median_filter(channel, size=size, mode="reflect")
    return channel - den, den


def _block_view(arr: np.ndarray, block: int) -> np.ndarray:
    h, w = arr.shape
    hb, wb = (h // block) * block, (w // block) * block
    a = arr[:hb, :wb]
    return a.reshape(hb // block, block, wb // block, block).swapaxes(1, 2)  # (nY, nX, b, b)


def flat_block_statistics(residual: np.ndarray, denoised: np.ndarray, block: int = BLOCK) -> dict[str, float]:
    """Statistiques du bruit sur les blocs de contenu plat."""
    rb = _block_view(residual, block)
    db = _block_view(denoised, block)
    content_std = db.std(axis=(2, 3))
    mean_int = db.mean(axis=(2, 3))
    noise_var = rb.var(axis=(2, 3))
    flat = (content_std < FLAT_STD_THRESHOLD) & (mean_int > 0.05) & (mean_int < 0.95)
    n_flat = int(flat.sum())
    n_total = int(flat.size)
    out = {
        "flat_noise_std": float("nan"),
        "shot_noise_corr": float("nan"),
        "flat_kurtosis": float("nan"),
        "flat_block_fraction": n_flat / max(n_total, 1),
        "n_flat": n_flat,
    }
    if n_flat < MIN_FLAT_BLOCKS:
        return out
    out["flat_noise_std"] = float(np.sqrt(np.median(noise_var[flat])))
    # Corrélation de rang entre intensité et variance du bruit (bruit de grenaille).
    if np.std(mean_int[flat]) > 1e-6 and np.std(noise_var[flat]) > 1e-12:
        rho = stats.spearmanr(mean_int[flat], noise_var[flat]).statistic
        out["shot_noise_corr"] = float(rho) if np.isfinite(rho) else float("nan")
    samples = rb[flat].ravel()
    if samples.size > 100 and np.std(samples) > 1e-9:
        # log1p : le kurtosis explose sur des résidus quasi nuls (quantification 8 bits).
        k = float(stats.kurtosis(samples, fisher=True, bias=False))
        out["flat_kurtosis"] = float(np.log1p(max(k, 0.0)))
    return out


def residual_autocorrelation_peak(residual: np.ndarray, lags: tuple[int, ...] = AC_LAGS) -> float:
    """Proéminence maximale de l'autocorrélation normalisée aux pas d'upsampling.

    Une valeur nettement positive signale une périodicité (grille de décodeur).
    """
    r = residual - residual.mean()
    spec = np.fft.fft2(r)
    ac = np.fft.ifft2(np.abs(spec) ** 2).real
    ac /= (ac[0, 0] + EPS)
    best = -np.inf
    for lag in lags:
        if lag + 1 >= min(ac.shape):
            continue
        for a, b, c in ((ac[0, lag], ac[0, lag - 1], ac[0, lag + 1]), (ac[lag, 0], ac[lag - 1, 0], ac[lag + 1, 0])):
            best = max(best, float(a - 0.5 * (b + c)))
    return best if np.isfinite(best) else 0.0


def cfa_interpolation_strength(green: np.ndarray) -> float:
    """Force de la périodicité de période 2 du dématriçage (Gallagher & Chen, 2008).

    On calcule |∇²G|, on somme le long des diagonales, puis on mesure la
    proéminence (log-ratio à la médiane) du pic de Nyquist du spectre 1D.
    Élevée pour une image issue directement d'un capteur Bayer ; proche de 0 après
    rééchantillonnage, forte compression ou synthèse.
    """
    lap = np.abs(ndimage.laplace(green.astype(np.float64), mode="reflect"))
    h, w = lap.shape
    # Somme sur les diagonales i + j = k  →  séquence de longueur h + w − 1
    idx = (np.arange(h)[:, None] + np.arange(w)[None, :]).ravel()
    diag = np.bincount(idx, weights=lap.ravel(), minlength=h + w - 1)
    counts = np.bincount(idx, minlength=h + w - 1).astype(np.float64)
    diag = diag / np.maximum(counts, 1.0)
    diag = diag[8:-8]  # bords (peu d'échantillons)
    diag = diag - diag.mean()
    n = diag.size
    if n < 32:
        return 0.0
    if n % 2:  # longueur paire → le dernier bin de rfft est exactement Nyquist
        diag = diag[:-1]
        n -= 1
    mag = np.abs(np.fft.rfft(diag))
    nyq = mag[-1]
    ref = np.median(mag[n // 8:-1]) + EPS
    return float(np.log((nyq + EPS) / ref))


def jpeg_grid_strength(gray: np.ndarray) -> float:
    """Blocking 8×8 : log-ratio des gradients aux frontières de blocs vs ailleurs."""
    g = gray.astype(np.float64)
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    cols = np.arange(dx.shape[1])
    rows = np.arange(dy.shape[0])
    bx = dx[:, (cols % 8) == 7].mean() if np.any((cols % 8) == 7) else 0.0
    ix = dx[:, (cols % 8) != 7].mean() + EPS
    by = dy[(rows % 8) == 7, :].mean() if np.any((rows % 8) == 7) else 0.0
    iy = dy[(rows % 8) != 7, :].mean() + EPS
    return float(0.5 * (np.log((bx + EPS) / ix) + np.log((by + EPS) / iy)))


def channel_residual_correlation(res_rgb: list[np.ndarray]) -> float:
    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a = a.ravel() - a.mean()
        b = b.ravel() - b.mean()
        d = np.sqrt(np.dot(a, a) * np.dot(b, b)) + EPS
        return float(np.dot(a, b) / d)
    return 0.5 * (corr(res_rgb[0], res_rgb[1]) + corr(res_rgb[1], res_rgb[2]))


def compute_noise_features(rgb: np.ndarray) -> tuple[NoiseFeatures, int]:
    rgb = reflect_pad_to_min(rgb, 64)
    residuals: list[np.ndarray] = []
    denoised: list[np.ndarray] = []
    for c in range(3):
        r, d = median_residual(rgb[..., c])
        residuals.append(r)
        denoised.append(d)
    gray = to_grayscale(rgb)
    res_gray = to_grayscale(np.stack(residuals, axis=-1))
    den_gray = to_grayscale(np.stack(denoised, axis=-1))

    flat = flat_block_statistics(res_gray, den_gray)
    ac_peak = residual_autocorrelation_peak(res_gray)
    power, _ = average_power_spectrum(res_gray)
    peaks = spectral_peaks(power, min_freq=0.08)
    cfa = cfa_interpolation_strength(rgb[..., 1])
    jpeg = jpeg_grid_strength(gray)
    chan = channel_residual_correlation(residuals)

    feats = NoiseFeatures(
        flat_noise_std=flat["flat_noise_std"],
        shot_noise_corr=flat["shot_noise_corr"],
        flat_kurtosis=flat["flat_kurtosis"],
        residual_ac_peak=ac_peak,
        residual_spec_peak_z=peaks["peak_max_z"],
        residual_grid_peak_z=peaks["grid_peak_z"],
        cfa_strength=cfa,
        jpeg_grid_strength=jpeg,
        channel_residual_corr=chan,
        flat_block_fraction=flat["flat_block_fraction"],
    )
    return feats, int(flat["n_flat"])


class NoiseAnalyzer:
    """Convertit une image RGB en ``noise_score`` ∈ [0, 1]."""

    def __init__(self, mapping: FeatureMapping):
        self.mapping = mapping

    def analyze(self, rgb: np.ndarray) -> NoiseResult:
        feats, n_flat = compute_noise_features(rgb)
        values = feats.as_dict()
        score = self.mapping.score(values)
        return NoiseResult(
            noise_score=float(score),
            features=values,
            contributions=self.mapping.contributions(values),
            flat_blocks=n_flat,
        )
