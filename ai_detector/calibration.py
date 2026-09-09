"""Calibration : transformation des caractéristiques brutes en probabilités.

Chaque module hand-crafted (fréquentiel, bruit) produit un vecteur de
caractéristiques physiques interprétables. Une régression logistique — dont les
paramètres proviennent de ``model/calibration.json`` (ajusté par
``training/calibrate_fusion.py``) ou, à défaut, de valeurs par défaut
conservatrices — les convertit en score dans [0, 1].

Aucune « signature universelle » n'est supposée : les poids sont appris sur des
données et peuvent être recalibrés à mesure que les générateurs évoluent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ai_detector.calibration_defaults import DEFAULT_CALIBRATION

CALIBRATION_SCHEMA_VERSION = 1
_Z_CLIP = 4.0


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


@dataclass
class FeatureMapping:
    """Régression logistique sur caractéristiques standardisées.

    ``score = sigmoid(bias + Σ w_i · clip((x_i − mean_i) / std_i, ±4))``.
    Une caractéristique manquante (NaN) est imputée à sa moyenne, donc neutre.
    """

    features: list[str]
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float = 0.0

    def __post_init__(self) -> None:
        n = len(self.features)
        if not (len(self.mean) == len(self.std) == len(self.weights) == n):
            raise ValueError("FeatureMapping : longueurs incohérentes")

    def standardize(self, values: Mapping[str, float]) -> np.ndarray:
        z = np.zeros(len(self.features), dtype=np.float64)
        for i, name in enumerate(self.features):
            x = values.get(name)
            if x is None or not np.isfinite(x):
                continue
            s = self.std[i] if self.std[i] > 1e-12 else 1.0
            z[i] = float(np.clip((float(x) - self.mean[i]) / s, -_Z_CLIP, _Z_CLIP))
        return z

    def score(self, values: Mapping[str, float]) -> float:
        z = self.standardize(values)
        return sigmoid(self.bias + float(np.dot(np.asarray(self.weights), z)))

    def contributions(self, values: Mapping[str, float]) -> dict[str, float]:
        """Contribution (en logit) de chaque caractéristique — pour l'explicabilité."""
        z = self.standardize(values)
        return {name: float(w * zi) for name, w, zi in zip(self.features, self.weights, z)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "weights": [float(v) for v in self.weights],
            "bias": float(self.bias),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FeatureMapping":
        return cls(
            features=list(d["features"]),
            mean=[float(v) for v in d["mean"]],
            std=[float(v) for v in d["std"]],
            weights=[float(v) for v in d["weights"]],
            bias=float(d.get("bias", 0.0)),
        )


@dataclass
class FusionParams:
    """Fusion en espace logit des trois scores modulaires."""

    weights: dict[str, float]
    bias: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"weights": dict(self.weights), "bias": float(self.bias)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FusionParams":
        return cls(weights={k: float(v) for k, v in d["weights"].items()}, bias=float(d.get("bias", 0.0)))


@dataclass
class Calibration:
    frequency: FeatureMapping
    noise: FeatureMapping
    fusion_full: FusionParams
    fusion_handcrafted: FusionParams
    threshold: float = 0.5
    source: str = "defaults"
    version: str = "defaults"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def defaults(cls) -> "Calibration":
        return cls.from_dict(DEFAULT_CALIBRATION)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Calibration":
        fusion = d["fusion"]
        return cls(
            frequency=FeatureMapping.from_dict(d["frequency"]),
            noise=FeatureMapping.from_dict(d["noise"]),
            fusion_full=FusionParams.from_dict(fusion["full"]),
            fusion_handcrafted=FusionParams.from_dict(fusion["handcrafted"]),
            threshold=float(fusion.get("threshold", 0.5)),
            source=str(d.get("source", "unknown")),
            version=str(d.get("version", "unknown")),
            metadata=dict(d.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "version": self.version,
            "source": self.source,
            "frequency": self.frequency.to_dict(),
            "noise": self.noise.to_dict(),
            "fusion": {
                "threshold": self.threshold,
                "full": self.fusion_full.to_dict(),
                "handcrafted": self.fusion_handcrafted.to_dict(),
            },
            "metadata": self.metadata,
        }

    @classmethod
    def load(cls, path: str | Path | None) -> "Calibration":
        """Charge ``calibration.json`` si présent et valide, sinon les défauts."""
        if path is None:
            return cls.defaults()
        p = Path(path)
        if not p.is_file():
            return cls.defaults()
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if int(data.get("schema_version", 1)) != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(f"calibration.json : schema_version non supporté ({data.get('schema_version')})")
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
