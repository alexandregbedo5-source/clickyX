"""Configuration du moteur (chemins, réseau, seuils).

Toutes les valeurs sont surchargeables par variables d'environnement afin que
l'intégration locale (Tybiane) puisse relocaliser les modèles sans toucher au code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ENV_MODEL_DIR = "AI_DETECTOR_MODEL_DIR"
ENV_MODEL_PATH = "AI_DETECTOR_MODEL_PATH"
ENV_CALIBRATION_PATH = "AI_DETECTOR_CALIBRATION_PATH"
ENV_HOST = "AI_DETECTOR_HOST"
ENV_PORT = "AI_DETECTOR_PORT"
ENV_TOKEN = "AI_DETECTOR_TOKEN"
ENV_CORS_ORIGINS = "AI_DETECTOR_CORS_ORIGINS"
ENV_THREADS = "AI_DETECTOR_THREADS"

DEFAULT_HOST = "127.0.0.1"
# Port volontairement éloigné du bridge ClickyX (32123) et des ports usuels.
DEFAULT_PORT = 32188


def default_model_dir() -> Path:
    env = os.environ.get(ENV_MODEL_DIR)
    if env:
        return Path(env).expanduser()
    candidate = REPO_ROOT / "model"
    if candidate.is_dir():
        return candidate
    return Path.cwd() / "model"


@dataclass
class DetectorConfig:
    """Paramètres d'exécution du détecteur."""

    model_path: Path = field(default_factory=lambda: Path(
        os.environ.get(ENV_MODEL_PATH) or default_model_dir() / "detector.onnx"
    ))
    calibration_path: Path = field(default_factory=lambda: Path(
        os.environ.get(ENV_CALIBRATION_PATH) or default_model_dir() / "calibration.json"
    ))
    # Fenêtre carrée maximale (pixels, résolution native) analysée par les modules
    # fréquentiel et bruit. Pas de redimensionnement : il détruirait les artefacts.
    analysis_window: int = 1024
    # Nombre maximal de crops natifs envoyés au CNN (moyenne des logits).
    cnn_max_crops: int = 5
    # Taille minimale acceptée ; en dessous, l'image est rembourrée par réflexion
    # et un avertissement de fiabilité est émis.
    min_image_side: int = 64
    onnx_threads: int = int(os.environ.get(ENV_THREADS, "0") or 0)

    @classmethod
    def from_env(cls) -> "DetectorConfig":
        return cls()


@dataclass
class ServerConfig:
    host: str = field(default_factory=lambda: os.environ.get(ENV_HOST, DEFAULT_HOST))
    port: int = field(default_factory=lambda: int(os.environ.get(ENV_PORT, str(DEFAULT_PORT))))
    token: str | None = field(default_factory=lambda: os.environ.get(ENV_TOKEN) or None)
    cors_origins: list[str] = field(default_factory=lambda: [
        o.strip() for o in os.environ.get(ENV_CORS_ORIGINS, "*").split(",") if o.strip()
    ])
