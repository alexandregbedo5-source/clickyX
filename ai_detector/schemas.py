"""Contrat d'API officiel — ``POST /detect-ai-image``.

Ce module est la **source de vérité** du contrat partagé avec l'interface
(Honorat) et l'intégration locale (Tybiane). Les champs du contrat de base sont
figés ; toute évolution est **additive** (champs optionnels supplémentaires) et
documentée dans ``docs/ai_detector.md``.

Contrat de base (v1) :

    Entrée  : {"image_path": "..."}
    Sortie  : {"is_ai_generated": bool, "confidence": float,
               "fft_score": float, "noise_score": float, "cnn_score": float}
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "1.0"

ScoreField = Field(ge=0.0, le=1.0)


class DetectAiImageRequest(BaseModel):
    """Entrée : un chemin local **ou** (extension additive) une image en base64."""

    model_config = ConfigDict(extra="ignore")

    image_path: str | None = Field(
        default=None,
        description="Chemin absolu (ou relatif au serveur) vers l'image à analyser.",
        examples=["C:/Users/alex/Pictures/photo.jpg", "/home/alex/img.png"],
    )
    image_base64: str | None = Field(
        default=None,
        description="Extension v1.1 : image encodée en base64 (data-URL acceptée). Alternative à image_path.",
    )
    include_details: bool = Field(
        default=True,
        description="Extension v1.1 : inclure le bloc `details` (caractéristiques, contributions, temps).",
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "DetectAiImageRequest":
        if bool(self.image_path) == bool(self.image_base64):
            raise ValueError("Fournir exactement un champ parmi `image_path` et `image_base64`.")
        return self


class ImageInfo(BaseModel):
    width: int
    height: int
    format: str | None = None
    has_exif: bool = False
    source: str


class ModuleDetails(BaseModel):
    """Caractéristiques brutes et contributions (logit) d'un module hand-crafted."""

    features: dict[str, float | None]
    contributions: dict[str, float]


class CnnDetails(BaseModel):
    available: bool
    trained: bool
    arch: str | None = None
    model_version: str | None = None
    crops: int = 0
    crop_scores: list[float] = Field(default_factory=list)
    padded: bool = False


class FusionDetails(BaseModel):
    mode: Literal["full", "handcrafted"]
    threshold: float
    weights: dict[str, float]
    logit_contributions: dict[str, float]
    verdict_confidence: float = ScoreField


class DetectionDetails(BaseModel):
    image: ImageInfo
    frequency: ModuleDetails
    noise: ModuleDetails
    cnn: CnnDetails
    fusion: FusionDetails
    timings_ms: dict[str, float]
    calibration_version: str


class DetectAiImageResponse(BaseModel):
    """Sortie. Les cinq premiers champs constituent le contrat v1 figé."""

    model_config = ConfigDict(extra="forbid")

    is_ai_generated: bool = Field(description="Verdict binaire : confidence ≥ seuil (0.5 par défaut).")
    confidence: float = Field(ge=0.0, le=1.0, description="Probabilité estimée que l'image soit générée par IA.")
    fft_score: float = Field(ge=0.0, le=1.0, description="Score du module fréquentiel (0 = photo, 1 = synthèse).")
    noise_score: float = Field(ge=0.0, le=1.0, description="Score du module bruit résiduel.")
    cnn_score: float = Field(ge=0.0, le=1.0, description="Score du réseau de neurones (0.5 neutre si indisponible).")

    # --- Extensions additives (v1.1) : optionnelles, ignorables par les clients v1.
    contract_version: str = Field(default=CONTRACT_VERSION, description="Version du contrat de sortie.")
    model_version: str | None = Field(default=None, description="Version du modèle ONNX utilisé, si chargé.")
    warnings: list[str] = Field(default_factory=list, description="Avertissements de fiabilité (codes stables).")
    details: DetectionDetails | None = Field(default=None, description="Explicabilité : caractéristiques et contributions.")

    def contract_v1(self) -> dict[str, Any]:
        """Projection stricte sur les cinq champs du contrat v1."""
        return {
            "is_ai_generated": self.is_ai_generated,
            "confidence": self.confidence,
            "fft_score": self.fft_score,
            "noise_score": self.noise_score,
            "cnn_score": self.cnn_score,
        }


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class ModelStatus(BaseModel):
    loaded: bool
    trained: bool
    path: str | None = None
    arch: str | None = None
    version: str | None = None
    input_size: int | None = None
    providers: list[str] = Field(default_factory=list)
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str = "clickyx-ai-detector"
    version: str
    contract_version: str = CONTRACT_VERSION
    model: ModelStatus
    calibration_version: str
    fusion_mode: Literal["full", "handcrafted"]
