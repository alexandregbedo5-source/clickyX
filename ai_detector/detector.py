"""Orchestrateur du pipeline forensic.

    Image → Prétraitement → FFT → Bruit résiduel → CNN → Fusion → Score final
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from ai_detector.calibration import Calibration
from ai_detector.cnn import CnnDetector
from ai_detector.config import DetectorConfig
from ai_detector.frequency import FrequencyAnalyzer
from ai_detector.fusion import FusionEngine
from ai_detector.noise import NoiseAnalyzer
from ai_detector.preprocessing import (
    LoadedImage,
    center_window,
    load_image_from_array,
    load_image_from_base64,
    load_image_from_bytes,
    load_image_from_path,
    to_grayscale,
)
from ai_detector.schemas import (
    CnnDetails,
    DetectAiImageResponse,
    DetectionDetails,
    FusionDetails,
    ImageInfo,
    ModuleDetails,
)

LOGGER = logging.getLogger(__name__)


def _clean_features(values: dict[str, float]) -> dict[str, float | None]:
    return {k: (float(v) if v is not None and np.isfinite(v) else None) for k, v in values.items()}


class AiImageDetector:
    """Détecteur complet, réutilisable (charge le modèle une seule fois)."""

    def __init__(self, config: DetectorConfig | None = None, calibration: Calibration | None = None):
        self.config = config or DetectorConfig.from_env()
        self.calibration = calibration or Calibration.load(self.config.calibration_path)
        self.frequency = FrequencyAnalyzer(self.calibration.frequency)
        self.noise = NoiseAnalyzer(self.calibration.noise)
        self.cnn = CnnDetector(self.config.model_path, max_crops=self.config.cnn_max_crops, threads=self.config.onnx_threads)
        self.fusion = FusionEngine(self.calibration)

    # ------------------------------------------------------------------ état
    @property
    def cnn_usable(self) -> bool:
        return self.cnn.available and self.cnn.trained

    @property
    def fusion_mode(self) -> str:
        return "full" if self.cnn_usable else "handcrafted"

    # ------------------------------------------------------------------ entrées
    def detect_path(self, path: str | Path, include_details: bool = True) -> DetectAiImageResponse:
        return self.detect_loaded(load_image_from_path(path), include_details)

    def detect_bytes(self, data: bytes, include_details: bool = True) -> DetectAiImageResponse:
        return self.detect_loaded(load_image_from_bytes(data), include_details)

    def detect_base64(self, payload: str, include_details: bool = True) -> DetectAiImageResponse:
        return self.detect_loaded(load_image_from_base64(payload), include_details)

    def detect_array(self, rgb: np.ndarray, include_details: bool = True) -> DetectAiImageResponse:
        return self.detect_loaded(load_image_from_array(rgb), include_details)

    # ------------------------------------------------------------------ pipeline
    def detect_loaded(self, image: LoadedImage, include_details: bool = True) -> DetectAiImageResponse:
        timings: dict[str, float] = {}
        warnings = list(image.warnings)
        t0 = time.perf_counter()

        if image.min_side < self.config.min_image_side:
            warnings.append(f"image_tres_petite:{image.width}x{image.height}:fiabilite_reduite")

        window = center_window(image.rgb, self.config.analysis_window)
        gray = to_grayscale(window)
        timings["preprocessing"] = (time.perf_counter() - t0) * 1000

        t = time.perf_counter()
        freq = self.frequency.analyze(gray)
        timings["frequency"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        noise = self.noise.analyze(window)
        timings["noise"] = (time.perf_counter() - t) * 1000
        if noise.flat_blocks < 20:
            warnings.append("peu_de_zones_plates:statistiques_de_bruit_partielles")

        t = time.perf_counter()
        cnn = self.cnn.predict(image.rgb)
        timings["cnn"] = (time.perf_counter() - t) * 1000
        if cnn.warning:
            warnings.append(cnn.warning if cnn.available else "cnn_indisponible:fusion_indices_physiques_seuls")

        t = time.perf_counter()
        fused = self.fusion.fuse(freq.fft_score, noise.noise_score, cnn.cnn_score, cnn_usable=self.cnn_usable)
        timings["fusion"] = (time.perf_counter() - t) * 1000
        timings["total"] = (time.perf_counter() - t0) * 1000

        details = None
        if include_details:
            details = DetectionDetails(
                image=ImageInfo(width=image.width, height=image.height, format=image.format,
                                has_exif=image.has_exif, source=image.source),
                frequency=ModuleDetails(features=_clean_features(freq.features), contributions=freq.contributions),
                noise=ModuleDetails(features=_clean_features(noise.features), contributions=noise.contributions),
                cnn=CnnDetails(available=cnn.available, trained=cnn.trained, arch=cnn.arch,
                               model_version=cnn.model_version, crops=cnn.crops,
                               crop_scores=[round(s, 6) for s in cnn.crop_scores], padded=cnn.padded),
                fusion=FusionDetails(mode=fused.mode, threshold=fused.threshold, weights=fused.weights,  # type: ignore[arg-type]
                                     logit_contributions=fused.logit_contributions,
                                     verdict_confidence=fused.verdict_confidence),
                timings_ms={k: round(v, 2) for k, v in timings.items()},
                calibration_version=self.calibration.version,
            )

        return DetectAiImageResponse(
            is_ai_generated=fused.is_ai_generated,
            confidence=round(fused.confidence, 6),
            fft_score=round(freq.fft_score, 6),
            noise_score=round(noise.noise_score, 6),
            cnn_score=round(cnn.cnn_score, 6),
            model_version=cnn.model_version if cnn.available else None,
            warnings=warnings,
            details=details,
        )
