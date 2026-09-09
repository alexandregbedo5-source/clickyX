import numpy as np

from ai_detector.calibration import Calibration
from ai_detector.noise import (
    NoiseAnalyzer,
    cfa_interpolation_strength,
    compute_noise_features,
    jpeg_grid_strength,
    residual_autocorrelation_peak,
)


def test_features_finite_or_nan_only_when_expected(photo_like):
    feats, n_flat = compute_noise_features(photo_like)
    d = feats.as_dict()
    assert n_flat >= 20
    for k in ("flat_noise_std", "shot_noise_corr", "residual_ac_peak", "residual_grid_peak_z", "cfa_strength", "jpeg_grid_strength"):
        assert np.isfinite(d[k]), k


def test_shot_noise_correlation_positive_for_photo_like(photo_like):
    feats, _ = compute_noise_features(photo_like)
    # Variance du bruit ∝ intensité par construction ⇒ corrélation de rang positive.
    assert feats.shot_noise_corr > 0.2


def test_upsampled_content_has_stronger_residual_grid(photo_like, upsampled_like):
    natural, _ = compute_noise_features(photo_like)
    synthetic, _ = compute_noise_features(upsampled_like)
    assert synthetic.residual_grid_peak_z > natural.residual_grid_peak_z


def test_autocorrelation_peak_detects_period_8():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (256, 256))
    periodic = base + 0.8 * np.tile(rng.normal(0, 1, (8, 8)), (32, 32))
    assert residual_autocorrelation_peak(periodic) > residual_autocorrelation_peak(base) + 0.05


def test_jpeg_grid_strength_increases_with_compression(image_files):
    from ai_detector.preprocessing import load_image_from_path, to_grayscale

    png = to_grayscale(load_image_from_path(image_files["photo.png"]).rgb)
    jpg = to_grayscale(load_image_from_path(image_files["photo.jpg"]).rgb)
    assert jpeg_grid_strength(jpg) > jpeg_grid_strength(png)


def test_cfa_strength_detects_period2_pattern():
    rng = np.random.default_rng(0)
    smooth = rng.normal(0.5, 0.05, (256, 256))
    pattern = smooth.copy()
    pattern[::2, ::2] += 0.05  # motif Bayer-like de période 2
    assert cfa_interpolation_strength(pattern) > cfa_interpolation_strength(smooth)


def test_no_flat_blocks_gives_nan_and_neutral_score():
    textured = np.random.default_rng(0).random((256, 256, 3)).astype(np.float32)
    feats, n_flat = compute_noise_features(textured)
    assert n_flat < 20
    assert np.isnan(feats.flat_noise_std)
    score = NoiseAnalyzer(Calibration.defaults().noise).analyze(textured).noise_score
    assert 0.0 <= score <= 1.0
