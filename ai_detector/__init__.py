"""ClickyX — moteur de forensic numérique pour images générées par IA.

Pipeline : image → prétraitement → analyse fréquentielle → bruit résiduel
→ CNN (ONNX Runtime) → fusion → score final.

Point d'entrée principal : :class:`ai_detector.detector.AiImageDetector`.
"""

from ai_detector.detector import AiImageDetector
from ai_detector.schemas import DetectAiImageRequest, DetectAiImageResponse

__version__ = "0.1.0"
__all__ = ["AiImageDetector", "DetectAiImageRequest", "DetectAiImageResponse", "__version__"]
