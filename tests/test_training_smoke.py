"""Smoke test du pipeline d'entraînement complet sur CPU (jeu minuscule synthétique).

train.py → export_onnx.py → evaluate.py (--onnx) → calibrate_fusion.py.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from tests.conftest import make_photo_like, make_upsampled_like  # noqa: E402
from training.data import balanced_sampler_weights, group_split, load_samples, summarize  # noqa: E402
from training.metrics import auc_score, compute_metrics  # noqa: E402


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data")
    for split, n in (("train", 6), ("val", 4)):
        for label_dir, maker in (("real", make_photo_like), ("ai", make_upsampled_like)):
            d = root / split / label_dir
            d.mkdir(parents=True)
            for i in range(n):
                arr = maker(96, seed=hash((split, label_dir, i)) % 10_000)
                Image.fromarray((arr * 255).astype(np.uint8)).save(d / f"{i}.png")
    return root


def test_data_scanning_and_balancing(tiny_dataset):
    samples = load_samples(tiny_dataset, "folder", "train")
    s = summarize(samples)
    assert s["n"] == 12 and s["n_real"] == 6 and s["n_ai"] == 6
    w = balanced_sampler_weights(samples)
    assert sum(w) == pytest.approx(1.0)
    split = group_split(samples, 0.5, seed=0, group_by_generator=False)
    assert set(split.values()) == {"train", "val"}


def test_metrics_basic():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    m = compute_metrics(y, s)
    assert m["auc"] == pytest.approx(1.0) and m["accuracy"] == 1.0 and m["eer"] == 0.0
    assert auc_score(np.array([0, 1]), np.array([0.9, 0.1])) == pytest.approx(0.0)


def test_train_export_evaluate_calibrate(tiny_dataset, tmp_path):
    from training.calibrate_fusion import main as calibrate_main
    from training.evaluate import main as evaluate_main
    from training.export_onnx import main as export_main
    from training.train import main as train_main

    run_dir = tmp_path / "run"
    assert train_main([
        "--data-root", str(tiny_dataset), "--layout", "folder", "--arch", "tiny", "--no-pretrained",
        "--img-size", "64", "--epochs", "2", "--batch-size", "4", "--workers", "0", "--ema", "0",
        "--early-stopping-patience", "0", "--out-dir", str(run_dir), "--device", "cpu", "--log-interval", "1",
    ]) == 0
    assert (run_dir / "best.pt").is_file() and (run_dir / "last.pt").is_file()
    history = json.loads((run_dir / "history.json").read_text())
    assert len(history) == 2 and "val" in history[-1]

    onnx_path = tmp_path / "detector.onnx"
    assert export_main(["--checkpoint", str(run_dir / "best.pt"), "--out", str(onnx_path), "--version", "smoke"]) == 0
    card = json.loads((tmp_path / "model_card.json").read_text())
    assert card["trained"] is True and card["input"]["shape"][-1] == 64

    from ai_detector.cnn import CnnDetector

    cnn = CnnDetector(onnx_path)
    assert cnn.available and cnn.trained and cnn.input_size == 64 and cnn.model_version == "smoke"

    eval_dir = tmp_path / "eval"
    assert evaluate_main(["--onnx", str(onnx_path), "--data-root", str(tiny_dataset), "--layout", "folder",
                          "--split", "val", "--robustness", "jpeg75", "--out-dir", str(eval_dir)]) == 0
    report = json.loads((eval_dir / "report.json").read_text())
    assert set(report["conditions"]) == {"none", "jpeg75"}
    assert (eval_dir / "scores.csv").is_file() and (eval_dir / "report.md").is_file()

    calib_out = tmp_path / "calibration.json"
    assert calibrate_main(["--data-root", str(tiny_dataset), "--layout", "folder", "--split", "val",
                           "--onnx", str(onnx_path), "--out", str(calib_out), "--folds", "2"]) == 0
    calib = json.loads(calib_out.read_text())
    assert set(calib["fusion"]["full"]["weights"]) == {"fft", "noise", "cnn"}
    assert calib["metadata"]["report"]["fusion_full_oof"] is not None

    # Le moteur charge la calibration produite et passe en mode "full".
    from ai_detector.config import DetectorConfig
    from ai_detector.detector import AiImageDetector

    det = AiImageDetector(DetectorConfig(model_path=onnx_path, calibration_path=calib_out))
    assert det.fusion_mode == "full"
    r = det.detect_array(make_photo_like(96, seed=123))
    assert r.details.fusion.mode == "full" and r.model_version == "smoke"
