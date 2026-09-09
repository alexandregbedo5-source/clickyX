"""Contrats d'API partagés (JSON Schema, TypeScript).

Régénérer le JSON Schema après toute évolution de :mod:`ai_detector.schemas` :

    python -m ai_detector.contracts
"""

from __future__ import annotations

import json
from pathlib import Path

from ai_detector.schemas import (
    CONTRACT_VERSION,
    DetectAiImageRequest,
    DetectAiImageResponse,
    ErrorResponse,
    HealthResponse,
)

CONTRACTS_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = CONTRACTS_DIR / "detect_ai_image.schema.json"


def build_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://clickyx.local/contracts/detect-ai-image.schema.json",
        "title": "ClickyX AI Image Detector — contrat POST /detect-ai-image",
        "contract_version": CONTRACT_VERSION,
        "endpoint": {"method": "POST", "path": "/detect-ai-image", "default_base_url": "http://127.0.0.1:32188"},
        "$defs": {
            "DetectAiImageRequest": DetectAiImageRequest.model_json_schema(ref_template="#/$defs/{model}"),
            "DetectAiImageResponse": DetectAiImageResponse.model_json_schema(ref_template="#/$defs/{model}"),
            "ErrorResponse": ErrorResponse.model_json_schema(ref_template="#/$defs/{model}"),
            "HealthResponse": HealthResponse.model_json_schema(ref_template="#/$defs/{model}"),
        },
    }


def _flatten_defs(schema: dict) -> dict:
    """Remonte les sous-définitions Pydantic (``$defs`` imbriqués) au niveau racine."""
    root_defs = schema["$defs"]
    for name in list(root_defs):
        nested = root_defs[name].pop("$defs", {})
        for k, v in nested.items():
            root_defs.setdefault(k, v)
    return schema


def write_schema(path: Path = SCHEMA_PATH) -> Path:
    schema = _flatten_defs(build_schema())
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_schema())
