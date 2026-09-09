import numpy as np

from ai_detector.calibration import Calibration
from ai_detector.frequency import FrequencyAnalyzer, average_power_spectrum, compute_spectrum_features, spectral_peaks
from ai_detector.preprocessing import to_grayscale


def test_average_power_spectrum_shape_and_tiles(photo_like):
    power, tiles = average_power_spectrum(to_grayscale(photo_like))
    assert power.shape == (256, 256)
    assert tiles == 4
    assert np.all(np.isfinite(power)) and np.all(power >= 0)


def test_features_are_finite(photo_like, upsampled_like):
    for img in (photo_like, upsampled_like):
        feats, _ = compute_spectrum_features(to_grayscale(img))
        for k, v in feats.as_dict().items():
            assert np.isfinite(v), k


def test_photo_like_follows_power_law(photo_like):
    feats, _ = compute_spectrum_features(to_grayscale(photo_like))
    # Champ 1/f^1.1 en amplitude → pente de puissance ≈ −2,2 (fenêtrage compris).
    assert -3.5 < feats.slope < -1.0


def test_upsampling_grid_creates_grid_peaks(photo_like, upsampled_like):
    natural, _ = compute_spectrum_features(to_grayscale(photo_like))
    synthetic, _ = compute_spectrum_features(to_grayscale(upsampled_like))
    assert synthetic.grid_peak_z > natural.grid_peak_z
    assert synthetic.profile_roughness > natural.profile_roughness


def test_spectral_peaks_detects_injected_peak():
    rng = np.random.default_rng(0)
    n = 256
    power = np.exp(rng.normal(0, 0.1, (n, n)))
    power[n // 2 + n // 4, n // 2] *= 1e4  # pic à f = 0.25 sur l'axe vertical
    peaks = spectral_peaks(power)
    assert peaks["grid_peak_z"] > 20
    assert peaks["peak_max_z"] >= peaks["grid_peak_z"]


def test_analyzer_score_in_unit_interval(photo_like, upsampled_like):
    analyzer = FrequencyAnalyzer(Calibration.defaults().frequency)
    r_photo = analyzer.analyze(to_grayscale(photo_like))
    r_synth = analyzer.analyze(to_grayscale(upsampled_like))
    for r in (r_photo, r_synth):
        assert 0.0 <= r.fft_score <= 1.0
        assert set(r.contributions) == set(r.features)
    assert r_synth.fft_score > r_photo.fft_score


def test_small_image_is_padded_not_rejected():
    tiny = np.random.default_rng(0).random((48, 70)).astype(np.float32)
    feats, tiles = compute_spectrum_features(tiny)
    assert tiles == 1
    assert np.isfinite(feats.grid_peak_z)
