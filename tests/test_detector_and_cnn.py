import base64
from pathlib import Path

import numpy as np
import pytest

from ai_detector.cnn import CnnDetector
from ai_detector.config import DetectorConfig
from ai_detector.detector import AiImageDetector
from ai_detector.preprocessing import ImageLoadError, grid_crops, load_image_from_base64, reflect_pad_to_min

CONTRACT_KEYS = {"is_ai_generated", "confidence", "fft_score", "noise_score", "cnn_score"}


def _detector(model_path: Path | str = "/nonexistent/detector.onnx") -> AiImageDetector:
    return AiImageDetector(DetectorConfig(model_path=Path(model_path), calibration_path=Path("/nonexistent/calibration.json")))


def test_contract_v1_keys_and_ranges(image_files):
    result = _detector().detect_path(image_files["photo.png"])
    payload = result.contract_v1()
    assert set(payload) == CONTRACT_KEYS
    assert isinstance(payload["is_ai_generated"], bool)
    for k in ("confidence", "fft_score", "noise_score", "cnn_score"):
        assert 0.0 <= payload[k] <= 1.0
    assert payload["is_ai_generated"] == (payload["confidence"] >= 0.5)


def test_without_model_falls_back_to_handcrafted(image_files):
    det = _detector()
    assert det.fusion_mode == "handcrafted"
    r = det.detect_path(image_files["photo.png"])
    assert r.cnn_score == 0.5
    assert r.model_version is None
    assert any(w.startswith("cnn_indisponible") for w in r.warnings)
    assert r.details is not None and r.details.fusion.mode == "handcrafted"


def test_upsampled_scores_higher_than_photo_like(image_files):
    det = _detector()
    photo = det.detect_path(image_files["photo.png"])
    synth = det.detect_path(image_files["upsampled.png"])
    assert synth.confidence > photo.confidence


def test_tiny_image_warns_but_succeeds(image_files):
    r = _detector().detect_path(image_files["tiny.png"])
    assert any(w.startswith("image_tres_petite") for w in r.warnings)
    assert 0.0 <= r.confidence <= 1.0


def test_missing_file_and_bad_payloads(tmp_path):
    det = _detector()
    with pytest.raises(FileNotFoundError):
        det.detect_path(tmp_path / "nope.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ImageLoadError):
        det.detect_path(bad)
    with pytest.raises(ImageLoadError):
        load_image_from_base64("!!!")


def test_base64_and_data_url_inputs(image_files):
    det = _detector()
    raw = image_files["photo.jpg"].read_bytes()
    b64 = base64.b64encode(raw).decode()
    r1 = det.detect_base64(b64)
    r2 = det.detect_base64("data:image/jpeg;base64," + b64)
    r3 = det.detect_path(image_files["photo.jpg"])
    assert r1.confidence == pytest.approx(r2.confidence) == pytest.approx(r3.confidence)
    assert r1.details.image.format == "JPEG"


def test_include_details_false_omits_details(image_files):
    r = _detector().detect_path(image_files["photo.png"], include_details=False)
    assert r.details is None
    dumped = r.model_dump(exclude_none=True)
    assert "details" not in dumped and CONTRACT_KEYS <= set(dumped)


def test_grid_crops_native_resolution_and_padding():
    img = np.zeros((300, 500, 3), dtype=np.float32)
    crops = grid_crops(img, 224, max_crops=5)
    assert len(crops) == 5 and all(c.shape == (224, 224, 3) for c in crops)
    small = np.zeros((100, 120, 3), dtype=np.float32)
    crops = grid_crops(small, 224, max_crops=5)
    assert len(crops) == 1 and crops[0].shape == (224, 224, 3)
    assert reflect_pad_to_min(np.zeros((10, 10)), 64).shape == (64, 64)


def test_cnn_detector_missing_model_is_neutral():
    cnn = CnnDetector("/nonexistent.onnx")
    assert not cnn.available
    r = cnn.predict(np.zeros((256, 256, 3), dtype=np.float32))
    assert r.cnn_score == 0.5 and not r.available and r.warning


def test_placeholder_onnx_loads_and_is_excluded_from_fusion(placeholder_onnx, image_files):
    cnn = CnnDetector(placeholder_onnx, max_crops=3)
    assert cnn.available and not cnn.trained
    assert cnn.input_size == 224 and cnn.arch == "tiny"
    r = cnn.predict(np.random.default_rng(0).random((300, 300, 3)).astype(np.float32))
    assert r.crops == 3 and r.cnn_score == pytest.approx(0.5, abs=1e-6)  # tête à zéro ⇒ logit 0

    det = _detector(placeholder_onnx)
    assert det.fusion_mode == "handcrafted"
    res = det.detect_path(image_files["photo.png"])
    assert res.model_version == "test-placeholder"
    assert res.details.cnn.available and not res.details.cnn.trained
    assert any("cnn_non_entraine" in w for w in res.warnings)


def test_versioned_placeholder_in_repo_is_valid():
    repo_model = Path(__file__).resolve().parents[1] / "model" / "detector.onnx"
    if not repo_model.is_file():
        pytest.skip("model/detector.onnx absent")
    cnn = CnnDetector(repo_model)
    assert cnn.available
    assert cnn.metadata.get("output") == "logit"
    assert "input_size" in cnn.metadata
