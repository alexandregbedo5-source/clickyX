"""Détecteur Deep Learning (ONNX Runtime) — module 3 du pipeline.

Le réseau (EfficientNet / ConvNeXt, entraîné via ``training/train.py`` et exporté
par ``training/export_onnx.py``) consomme des **crops à résolution native** :
aucun redimensionnement n'est appliqué aux images plus grandes que l'entrée
(les artefacts de synthèse sont à l'échelle du pixel). Plusieurs crops (centre +
coins) sont évalués et leurs logits moyennés.

Le contrat du fichier ONNX est porté par ses ``metadata_props`` :

* ``input_size`` (ex. « 224 »), ``mean``/``std`` (normalisation, « r,g,b »),
* ``output`` : « logit » (sortie [N,1]) ou « prob » ou « logits2 » ([N,2] softmax),
* ``trained`` : « true »/« false » — un modèle non entraîné (placeholder de
  signature) est chargé mais **exclu de la fusion**.

Sans modèle disponible, le module renvoie un score neutre (0.5) et le signale ;
le pipeline continue avec les indices physiques.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ai_detector.calibration import logit, sigmoid
from ai_detector.preprocessing import grid_crops

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NEUTRAL_SCORE = 0.5


@dataclass
class CnnResult:
    cnn_score: float
    available: bool
    trained: bool
    crops: int = 0
    crop_scores: list[float] = field(default_factory=list)
    padded: bool = False
    arch: str | None = None
    model_version: str | None = None
    warning: str | None = None


def _parse_triplet(value: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not value:
        return default
    parts = [float(v) for v in value.replace(";", ",").split(",") if v.strip()]
    if len(parts) == 1:
        return (parts[0],) * 3  # type: ignore[return-value]
    if len(parts) != 3:
        return default
    return parts[0], parts[1], parts[2]


class CnnDetector:
    """Encapsule une session ONNX Runtime et la stratégie de crops."""

    def __init__(self, model_path: str | Path | None, max_crops: int = 5, threads: int = 0):
        self.model_path = Path(model_path) if model_path else None
        self.max_crops = max(1, int(max_crops))
        self.threads = int(threads)
        self.available = False
        self.trained = False
        self.metadata: dict[str, str] = {}
        self.input_size = 224
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD
        self.output_kind = "logit"
        self.arch: str | None = None
        self.model_version: str | None = None
        self.load_error: str | None = None
        self.providers: list[str] = []
        self._session = None
        self._input_name = ""
        self._load()

    # ------------------------------------------------------------------ chargement
    def _load(self) -> None:
        if self.model_path is None or not self.model_path.is_file():
            self.load_error = f"modèle ONNX introuvable : {self.model_path}"
            LOGGER.warning(self.load_error)
            return
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.log_severity_level = 3
            if self.threads > 0:
                options.intra_op_num_threads = self.threads
            session = ort.InferenceSession(str(self.model_path), options, providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001 — tout échec de chargement doit dégrader proprement
            self.load_error = f"échec du chargement ONNX : {exc}"
            LOGGER.warning(self.load_error)
            return

        meta = session.get_modelmeta().custom_metadata_map or {}
        self.metadata = dict(meta)
        self.input_size = int(meta.get("input_size", self.input_size))
        self.mean = _parse_triplet(meta.get("mean"), IMAGENET_MEAN)
        self.std = _parse_triplet(meta.get("std"), IMAGENET_STD)
        self.output_kind = meta.get("output", "logit").lower()
        self.trained = str(meta.get("trained", "true")).lower() in ("1", "true", "yes")
        self.arch = meta.get("arch")
        self.model_version = meta.get("version")
        self._input_name = session.get_inputs()[0].name
        self.providers = list(session.get_providers())
        self._session = session
        self.available = True
        if not self.trained:
            LOGGER.warning("Modèle ONNX marqué trained=false : exclu de la fusion (placeholder de signature).")

    # ------------------------------------------------------------------ inférence
    def _to_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        batch = np.stack([(c.astype(np.float32) - mean) / std for c in crops], axis=0)
        return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))

    def _outputs_to_logits(self, out: np.ndarray) -> np.ndarray:
        out = np.asarray(out, dtype=np.float64)
        if out.ndim == 2 and out.shape[1] == 2:
            # softmax 2 classes : classe 1 = synthétique
            m = out.max(axis=1, keepdims=True)
            e = np.exp(out - m)
            p = e[:, 1] / e.sum(axis=1)
            return np.array([logit(float(v)) for v in p])
        flat = out.reshape(out.shape[0], -1)[:, 0]
        if self.output_kind == "prob":
            return np.array([logit(float(v)) for v in flat])
        return flat

    def predict(self, rgb: np.ndarray) -> CnnResult:
        if not self.available or self._session is None:
            return CnnResult(
                cnn_score=NEUTRAL_SCORE, available=False, trained=False,
                warning=self.load_error or "modèle indisponible",
            )
        h, w = rgb.shape[:2]
        padded = min(h, w) < self.input_size
        crops = grid_crops(rgb, self.input_size, self.max_crops)
        batch = self._to_batch(crops)
        out = self._session.run(None, {self._input_name: batch})[0]
        logits = self._outputs_to_logits(out)
        mean_logit = float(np.mean(logits))
        result = CnnResult(
            cnn_score=sigmoid(mean_logit),
            available=True,
            trained=self.trained,
            crops=len(crops),
            crop_scores=[sigmoid(float(v)) for v in logits],
            padded=padded,
            arch=self.arch,
            model_version=self.model_version,
        )
        if not self.trained:
            result.warning = "cnn_non_entraine:score_neutre_exclu_de_la_fusion"
        elif padded:
            result.warning = "image_plus_petite_que_l_entree_cnn:rembourrage_par_reflexion"
        return result

    def info(self) -> dict:
        return {
            "path": str(self.model_path) if self.model_path else None,
            "loaded": self.available,
            "trained": self.trained,
            "arch": self.arch,
            "version": self.model_version,
            "input_size": self.input_size if self.available else None,
            "providers": self.providers,
            "error": self.load_error,
        }
