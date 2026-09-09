"""Fusion des scores modulaires — module 4 du pipeline.

Fusion tardive en espace logit :

    z = b + Σ_i w_i · logit(s_i)        p = σ(z)

Deux jeux de paramètres coexistent (``model/calibration.json``) :

* ``full`` — fft + noise + cnn, lorsque le CNN est chargé **et** entraîné ;
* ``handcrafted`` — fft + noise, sinon.

``confidence`` est la probabilité estimée que l'image soit synthétique ;
``is_ai_generated`` est vrai si ``confidence ≥ threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_detector.calibration import Calibration, logit, sigmoid


@dataclass
class FusionResult:
    confidence: float
    is_ai_generated: bool
    threshold: float
    mode: str  # "full" | "handcrafted"
    weights: dict[str, float]
    logit_contributions: dict[str, float] = field(default_factory=dict)

    @property
    def verdict_confidence(self) -> float:
        """Confiance dans le verdict binaire (symétrique) : max(p, 1 − p)."""
        return max(self.confidence, 1.0 - self.confidence)


class FusionEngine:
    def __init__(self, calibration: Calibration):
        self.calibration = calibration

    def fuse(self, fft_score: float, noise_score: float, cnn_score: float, *, cnn_usable: bool) -> FusionResult:
        params = self.calibration.fusion_full if cnn_usable else self.calibration.fusion_handcrafted
        mode = "full" if cnn_usable else "handcrafted"
        scores = {"fft": fft_score, "noise": noise_score}
        if cnn_usable:
            scores["cnn"] = cnn_score
        contributions = {k: params.weights.get(k, 0.0) * logit(v) for k, v in scores.items()}
        z = params.bias + sum(contributions.values())
        p = sigmoid(z)
        return FusionResult(
            confidence=float(p),
            is_ai_generated=bool(p >= self.calibration.threshold),
            threshold=self.calibration.threshold,
            mode=mode,
            weights={k: params.weights.get(k, 0.0) for k in scores},
            logit_contributions={k: float(v) for k, v in contributions.items()},
        )
