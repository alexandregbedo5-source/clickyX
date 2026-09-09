import base64
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_detector.cli import main as cli_main
from ai_detector.config import DetectorConfig, ServerConfig
from ai_detector.detector import AiImageDetector
from ai_detector.server import create_app

CONTRACT_KEYS = {"is_ai_generated", "confidence", "fft_score", "noise_score", "cnn_score"}


@pytest.fixture(scope="module")
def client():
    det = AiImageDetector(DetectorConfig(model_path=Path("/nonexistent.onnx"), calibration_path=Path("/nonexistent.json")))
    app = create_app(ServerConfig(token=None), detector=det)
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"  # pas de CNN entraîné
    assert body["fusion_mode"] == "handcrafted"
    assert body["model"]["loaded"] is False


def test_detect_by_path_returns_contract(client, image_files):
    r = client.post("/detect-ai-image", json={"image_path": str(image_files["photo.png"])})
    assert r.status_code == 200
    body = r.json()
    assert CONTRACT_KEYS <= set(body)
    assert body["is_ai_generated"] == (body["confidence"] >= 0.5)
    assert "details" in body and body["details"]["fusion"]["mode"] == "handcrafted"


def test_detect_by_base64(client, image_files):
    b64 = base64.b64encode(image_files["photo.jpg"].read_bytes()).decode()
    r = client.post("/detect-ai-image", json={"image_base64": b64, "include_details": False})
    assert r.status_code == 200
    assert "details" not in r.json()


def test_validation_errors(client):
    assert client.post("/detect-ai-image", json={}).status_code == 422
    assert client.post("/detect-ai-image", json={"image_path": "a", "image_base64": "b"}).status_code == 422


def test_not_found_and_unsupported(client, tmp_path):
    r = client.post("/detect-ai-image", json={"image_path": str(tmp_path / "missing.png")})
    assert r.status_code == 404 and r.json()["error"]["code"] == "image_not_found"
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"xx")
    r = client.post("/detect-ai-image", json={"image_path": str(bad)})
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_image"


def test_contract_endpoint(client):
    body = client.get("/detect-ai-image/contract").json()
    assert body["contract_version"] == "1.0"
    assert CONTRACT_KEYS <= set(body["response"]["properties"])
    assert "image_path" in body["request"]["properties"]


def test_token_protection(image_files):
    det = AiImageDetector(DetectorConfig(model_path=Path("/nonexistent.onnx"), calibration_path=Path("/nonexistent.json")))
    app = create_app(ServerConfig(token="secret"), detector=det)
    c = TestClient(app)
    payload = {"image_path": str(image_files["photo.png"])}
    assert c.post("/detect-ai-image", json=payload).status_code == 401
    assert c.post("/detect-ai-image", json=payload, headers={"x-ai-detector-token": "secret"}).status_code == 200


def test_cli_json_is_strict_contract(image_files, capsys, monkeypatch):
    monkeypatch.setenv("AI_DETECTOR_MODEL_PATH", "/nonexistent.onnx")
    monkeypatch.setenv("AI_DETECTOR_CALIBRATION_PATH", "/nonexistent.json")
    code = cli_main(["detect", str(image_files["photo.png"]), "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == CONTRACT_KEYS


def test_cli_missing_file_exit_code(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AI_DETECTOR_MODEL_PATH", "/nonexistent.onnx")
    code = cli_main(["detect", str(tmp_path / "nope.png"), "--json"])
    assert code == 2
