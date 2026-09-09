"""Serveur HTTP local (FastAPI) exposant le contrat ``POST /detect-ai-image``.

* Écoute par défaut sur ``127.0.0.1:32188`` (loopback uniquement, hors ligne).
* Authentification optionnelle par en-tête ``x-ai-detector-token`` (variable
  ``AI_DETECTOR_TOKEN``), sur le modèle du bridge ClickyX.
* ``GET /health`` — état du service et du modèle.
* ``GET /detect-ai-image/contract`` — JSON Schema de la requête/réponse.

Lancement : ``python -m ai_detector serve`` ou ``uvicorn ai_detector.server:app``.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_detector import __version__
from ai_detector.config import ServerConfig
from ai_detector.detector import AiImageDetector
from ai_detector.preprocessing import ImageLoadError
from ai_detector.schemas import (
    CONTRACT_VERSION,
    DetectAiImageRequest,
    DetectAiImageResponse,
    ErrorResponse,
    HealthResponse,
    ModelStatus,
)

LOGGER = logging.getLogger(__name__)

TOKEN_HEADER = "x-ai-detector-token"


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@lru_cache(maxsize=1)
def get_detector() -> AiImageDetector:
    return AiImageDetector()


def create_app(server_config: ServerConfig | None = None, detector: AiImageDetector | None = None) -> FastAPI:
    cfg = server_config or ServerConfig()

    app = FastAPI(
        title="ClickyX — AI Image Detector",
        version=__version__,
        description=(
            "Moteur de forensic numérique local. Estime la probabilité qu'une image ait été "
            "générée par IA à partir d'indices fréquentiels, statistiques (bruit) et neuronaux (ONNX)."
        ),
        contact={"name": "Équipe ClickyX — IA Forensics"},
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.state.detector = detector

    def resolve_detector() -> AiImageDetector:
        if app.state.detector is None:
            app.state.detector = get_detector()
        return app.state.detector

    async def require_token(x_ai_detector_token: str | None = Header(default=None, alias=TOKEN_HEADER)) -> None:
        if cfg.token and x_ai_detector_token != cfg.token:
            raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Jeton invalide ou absent."})

    @app.exception_handler(HTTPException)
    async def _http_exc(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            return _error(exc.status_code, str(detail["code"]), str(detail.get("message", "")))
        return _error(exc.status_code, "http_error", str(detail))

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    async def health() -> HealthResponse:
        det = resolve_detector()
        info = det.cnn.info()
        return HealthResponse(
            status="ok" if det.cnn_usable else "degraded",
            version=__version__,
            model=ModelStatus(**info),
            calibration_version=det.calibration.version,
            fusion_mode=det.fusion_mode,  # type: ignore[arg-type]
        )

    @app.get("/detect-ai-image/contract", tags=["contract"])
    async def contract() -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "request": DetectAiImageRequest.model_json_schema(),
            "response": DetectAiImageResponse.model_json_schema(),
            "error": ErrorResponse.model_json_schema(),
        }

    @app.post(
        "/detect-ai-image",
        response_model=DetectAiImageResponse,
        response_model_exclude_none=True,
        responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse},
                   404: {"model": ErrorResponse}, 415: {"model": ErrorResponse}},
        tags=["detection"],
        dependencies=[Depends(require_token)],
    )
    async def detect_ai_image(payload: DetectAiImageRequest) -> DetectAiImageResponse:
        det = resolve_detector()
        try:
            if payload.image_path:
                return det.detect_path(payload.image_path, include_details=payload.include_details)
            return det.detect_base64(payload.image_base64 or "", include_details=payload.include_details)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail={"code": "image_not_found", "message": f"Fichier introuvable : {exc}"})
        except ImageLoadError as exc:
            raise HTTPException(status_code=415, detail={"code": "unsupported_image", "message": str(exc)})
        except PermissionError as exc:
            raise HTTPException(status_code=400, detail={"code": "permission_denied", "message": str(exc)})

    return app


# Application par défaut pour ``uvicorn ai_detector.server:app``.
app = create_app()


def run(host: str | None = None, port: int | None = None, log_level: str = "info") -> None:
    import uvicorn

    cfg = ServerConfig()
    uvicorn.run(app, host=host or cfg.host, port=port or cfg.port, log_level=log_level)


if __name__ == "__main__":
    run()
