import json

import numpy as np
import pytest

from ai_detector.calibration import Calibration, FeatureMapping, logit, sigmoid
from ai_detector.calibration_defaults import DEFAULT_CALIBRATION
from ai_detector.fusion import FusionEngine


def test_sigmoid_logit_roundtrip():
    for p in (0.01, 0.3, 0.5, 0.9, 0.999):
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-6)


def test_feature_mapping_missing_feature_is_neutral():
    m = FeatureMapping(features=["a", "b"], mean=[0, 0], std=[1, 1], weights=[2.0, 2.0], bias=0.0)
    assert m.score({"a": 0.0, "b": 0.0}) == pytest.approx(0.5)
    assert m.score({"a": float("nan"), "b": None}) == pytest.approx(0.5)
    assert m.score({"a": 10.0}) == pytest.approx(sigmoid(2.0 * 4.0))  # clip à ±4


def test_defaults_are_consistent():
    c = Calibration.defaults()
    assert len(c.frequency.features) == len(c.frequency.weights)
    assert len(c.noise.features) == len(c.noise.weights)
    assert c.threshold == 0.5
    assert set(c.fusion_full.weights) == {"fft", "noise", "cnn"}
    assert set(c.fusion_handcrafted.weights) == {"fft", "noise"}


def test_calibration_json_roundtrip(tmp_path):
    c = Calibration.defaults()
    p = tmp_path / "calibration.json"
    c.save(p)
    loaded = Calibration.load(p)
    assert loaded.to_dict()["frequency"] == c.to_dict()["frequency"]
    assert Calibration.load(tmp_path / "missing.json").version == c.version


def test_calibration_rejects_unknown_schema(tmp_path):
    p = tmp_path / "calibration.json"
    d = dict(DEFAULT_CALIBRATION)
    d["schema_version"] = 99
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        Calibration.load(p)


def test_fusion_modes_and_monotonicity():
    engine = FusionEngine(Calibration.defaults())
    full = engine.fuse(0.8, 0.7, 0.95, cnn_usable=True)
    hand = engine.fuse(0.8, 0.7, 0.95, cnn_usable=False)
    assert full.mode == "full" and "cnn" in full.weights
    assert hand.mode == "handcrafted" and "cnn" not in hand.weights
    assert full.is_ai_generated and hand.is_ai_generated
    low = engine.fuse(0.2, 0.3, 0.05, cnn_usable=True)
    assert not low.is_ai_generated
    assert low.confidence < hand.confidence < full.confidence
    assert low.verdict_confidence == pytest.approx(1 - low.confidence)


def test_neutral_inputs_give_bias_only():
    engine = FusionEngine(Calibration.defaults())
    r = engine.fuse(0.5, 0.5, 0.5, cnn_usable=True)
    assert r.confidence == pytest.approx(sigmoid(engine.calibration.fusion_full.bias))
    assert np.isclose(sum(r.logit_contributions.values()), 0.0, atol=1e-9)
