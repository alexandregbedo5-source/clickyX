"""Interface en ligne de commande.

    python -m ai_detector detect photo.jpg            # rapport lisible
    python -m ai_detector detect photo.jpg --json     # contrat v1 strict (5 champs)
    python -m ai_detector detect photo.jpg --json --details
    python -m ai_detector serve --port 32188
    python -m ai_detector info

Le mode ``--json`` imprime exactement le contrat sur stdout : il permet une
intégration « sidecar » sans HTTP (lecture de stdout par le backend Tauri).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ai_detector import __version__
from ai_detector.config import DEFAULT_PORT, DetectorConfig
from ai_detector.preprocessing import ImageLoadError


def _cmd_detect(args: argparse.Namespace) -> int:
    from ai_detector.detector import AiImageDetector

    detector = AiImageDetector(DetectorConfig.from_env())
    exit_code = 0
    outputs = []
    for path in args.images:
        try:
            result = detector.detect_path(path, include_details=args.details or not args.json)
        except FileNotFoundError:
            print(json.dumps({"error": {"code": "image_not_found", "message": path}}), file=sys.stderr)
            exit_code = 2
            continue
        except ImageLoadError as exc:
            print(json.dumps({"error": {"code": "unsupported_image", "message": str(exc)}}), file=sys.stderr)
            exit_code = 2
            continue
        if args.json:
            payload = result.model_dump(exclude_none=True) if args.details else result.contract_v1()
            outputs.append(payload if len(args.images) > 1 else payload)
        else:
            _print_human(path, result)
    if args.json:
        print(json.dumps(outputs[0] if len(outputs) == 1 else outputs, ensure_ascii=False, indent=None if args.compact else 2))
    return exit_code


def _print_human(path: str, result) -> None:  # noqa: ANN001
    verdict = "SYNTHÉTIQUE (IA)" if result.is_ai_generated else "PHOTOGRAPHIE"
    print(f"{Path(path).name}")
    print(f"  verdict      : {verdict}  (confidence={result.confidence:.3f})")
    print(f"  fft_score    : {result.fft_score:.3f}")
    print(f"  noise_score  : {result.noise_score:.3f}")
    print(f"  cnn_score    : {result.cnn_score:.3f}")
    if result.details:
        print(f"  fusion       : {result.details.fusion.mode}  (poids {result.details.fusion.weights})")
        print(f"  temps total  : {result.details.timings_ms.get('total', 0):.0f} ms")
    for w in result.warnings:
        print(f"  ! {w}")


def _cmd_serve(args: argparse.Namespace) -> int:
    from ai_detector.server import run

    run(host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    from ai_detector.detector import AiImageDetector

    det = AiImageDetector(DetectorConfig.from_env())
    print(json.dumps({
        "version": __version__,
        "model": det.cnn.info(),
        "calibration": {"version": det.calibration.version, "source": det.calibration.source,
                        "path": str(det.config.calibration_path)},
        "fusion_mode": det.fusion_mode,
    }, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-detector", description="ClickyX — détection d'images générées par IA (local).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_detect = sub.add_parser("detect", help="Analyser une ou plusieurs images.")
    p_detect.add_argument("images", nargs="+")
    p_detect.add_argument("--json", action="store_true", help="Sortie JSON (contrat v1 strict).")
    p_detect.add_argument("--details", action="store_true", help="Avec --json : inclure warnings et details.")
    p_detect.add_argument("--compact", action="store_true", help="JSON sur une ligne.")
    p_detect.set_defaults(func=_cmd_detect)

    p_serve = sub.add_parser("serve", help="Démarrer le serveur HTTP local.")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None, help=f"défaut : {DEFAULT_PORT}")
    p_serve.add_argument("--log-level", default="info")
    p_serve.set_defaults(func=_cmd_serve)

    p_info = sub.add_parser("info", help="État du modèle et de la calibration.")
    p_info.set_defaults(func=_cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
