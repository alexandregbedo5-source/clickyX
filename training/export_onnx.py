"""Export ONNX du détecteur + vérification ONNX Runtime + model card.

    # Export d'un checkpoint entraîné
    python training/export_onnx.py --checkpoint runs/effb0/best.pt --out model/detector.onnx --version 1.0.0

    # Placeholder de signature (non entraîné) : garantit le contrat d'E/S
    python training/export_onnx.py --bootstrap --arch tiny --out model/detector.onnx

Contrat ONNX :
    entrée  ``image``  float32 [N, 3, S, S]  (normalisation ImageNet, S = input_size)
    sortie  ``logit``  float32 [N, 1]        (σ(logit) = probabilité « générée par IA »)
    metadata_props : arch, input_size, mean, std, output=logit, trained, version, …
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from training.common import REPO_ROOT, git_commit, now_iso, save_json  # noqa: E402
from training.model import ARCH_PRESETS, IMAGENET_MEAN, IMAGENET_STD, build_model, count_parameters, load_checkpoint  # noqa: E402

INPUT_NAME = "image"
OUTPUT_NAME = "logit"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ONNX du détecteur.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Checkpoint .pt produit par train.py")
    src.add_argument("--bootstrap", action="store_true", help="Exporter un placeholder non entraîné (signature).")
    p.add_argument("--arch", default="tiny", choices=sorted(ARCH_PRESETS), help="(bootstrap) architecture.")
    p.add_argument("--out", default=str(REPO_ROOT / "model" / "detector.onnx"))
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--version", default=None, help="Version sémantique du modèle (défaut : dérivée de la date).")
    p.add_argument("--training-data", default=None, help="Description libre des données d'entraînement.")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--model-card", default=None, help="Chemin du model_card.json (défaut : à côté du .onnx).")
    return p.parse_args(argv)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_metadata(onnx_path: Path, props: dict[str, str]) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    existing = {p.key for p in model.metadata_props}
    for key, value in props.items():
        if key in existing:
            for p in model.metadata_props:
                if p.key == key:
                    p.value = value
        else:
            entry = model.metadata_props.add()
            entry.key, entry.value = key, value
    model.producer_name = "clickyx-ai-detector"
    model.doc_string = "ClickyX AI image detector — σ(logit) = P(image générée par IA)"
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))


def export(model: torch.nn.Module, input_size: int, out: Path, opset: int) -> None:
    model = model.eval().cpu()
    dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exportateur TorchScript volontairement : plus stable que l'export dynamo pour
    # les backbones timm (EfficientNet/ConvNeXt) et suffisant pour l'opset 17.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            model, dummy, str(out),
            input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )


def verify(model: torch.nn.Module, onnx_path: Path, input_size: int, n: int = 3, tol: float = 2e-3) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal((n, 3, input_size, input_size)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).cpu().numpy()
    got = sess.run(None, {INPUT_NAME: x})[0]
    if got.shape != (n, 1):
        raise RuntimeError(f"Sortie ONNX de forme {got.shape}, attendu {(n, 1)}")
    diff = float(np.max(np.abs(ref - got)))
    if diff > tol:
        raise RuntimeError(f"Écart torch/onnxruntime trop élevé : {diff:.3e} > {tol}")
    return diff


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    ckpt: dict = {}
    if args.bootstrap:
        model = build_model(args.arch, pretrained=False, drop_rate=0.0)
        # Tête mise à zéro : logit = 0 ⇒ score 0,5 exactement (neutre, sans fausse assurance).
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
        trained = False
        arch = args.arch
        version = args.version or "0.0.0-placeholder"
    else:
        model, ckpt = load_checkpoint(args.checkpoint)
        trained = True
        arch = model.arch
        version = args.version or f"1.0.0+{now_iso()[:10].replace('-', '')}"
    input_size = model.input_size

    export(model, input_size, out, args.opset)
    diff = None if args.no_verify else verify(model, out, input_size)

    metrics = ckpt.get("metrics") if ckpt else None
    props = {
        "arch": arch,
        "input_size": str(input_size),
        "mean": ",".join(f"{v:.4f}" for v in IMAGENET_MEAN),
        "std": ",".join(f"{v:.4f}" for v in IMAGENET_STD),
        "output": "logit",
        "trained": "true" if trained else "false",
        "version": version,
        "created_at": now_iso(),
        "framework": f"torch {torch.__version__}",
        "opset": str(args.opset),
        "git_commit": git_commit() or "",
        "training_data": args.training_data or ("" if trained else "none (placeholder)"),
        "val_auc": f"{metrics['auc']:.4f}" if metrics and metrics.get("auc") is not None else "",
        "task": "ai-generated-image-detection",
        "contract": "sigmoid(logit) = P(ai_generated); crops natifs input_size×input_size, normalisation ImageNet",
    }
    add_metadata(out, props)

    card = {
        "name": "ClickyX AI image detector",
        "file": out.name,
        "version": version,
        "trained": trained,
        "arch": arch,
        "parameters": count_parameters(model),
        "input": {"name": INPUT_NAME, "shape": ["batch", 3, input_size, input_size], "dtype": "float32",
                  "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
                  "policy": "crops à résolution native (centre + coins), moyenne des logits"},
        "output": {"name": OUTPUT_NAME, "shape": ["batch", 1], "dtype": "float32", "semantics": "sigmoid(logit) = P(ai_generated)"},
        "opset": args.opset,
        "size_bytes": out.stat().st_size,
        "sha256": _sha256(out),
        "created_at": props["created_at"],
        "git_commit": props["git_commit"] or None,
        "training": {
            "data": props["training_data"] or None,
            "epoch": ckpt.get("epoch") if ckpt else None,
            "val_metrics": metrics,
            "config": ckpt.get("train_config") if ckpt else None,
        },
        "verification": {"onnxruntime_max_abs_diff": diff},
        "notes": (
            "PLACEHOLDER NON ENTRAÎNÉ : garantit uniquement le contrat d'entrée/sortie. "
            "Le moteur l'exclut de la fusion (trained=false). Remplacer par un export de training/train.py."
            if not trained else
            "Modèle entraîné. Le moteur l'intègre à la fusion (mode full)."
        ),
    }
    card_path = Path(args.model_card) if args.model_card else out.with_name("model_card.json")
    save_json(card, card_path)
    print(json.dumps({"onnx": str(out), "size_bytes": card["size_bytes"], "trained": trained, "arch": arch,
                      "input_size": input_size, "max_abs_diff": diff, "model_card": str(card_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
