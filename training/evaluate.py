"""Évaluation du détecteur : métriques globales, par générateur, robustesse.

    # Checkpoint PyTorch
    python training/evaluate.py --checkpoint runs/effb0/best.pt --data-root DATA --layout genimage --split val

    # Modèle ONNX via le chemin d'inférence de production (ai_detector.cnn)
    python training/evaluate.py --onnx model/detector.onnx --data-root DATA --layout folder --split test \
        --robustness jpeg75 jpeg50 resize0.5 blur1.0

    # Pipeline complet (fft + noise + cnn + fusion) tel que servi par l'API
    python training/evaluate.py --full-pipeline --data-root DATA --layout folder --split test

Sorties dans ``--out-dir`` : ``report.json``, ``report.md``, ``scores.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from training.common import REPO_ROOT, now_iso, save_json, setup_logging  # noqa: E402
from training.data import Sample, load_samples, summarize  # noqa: E402
from training.metrics import compute_metrics, metrics_table_markdown  # noqa: E402
from training.transforms import build_degradation  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Évaluation du détecteur.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint")
    src.add_argument("--onnx")
    src.add_argument("--full-pipeline", action="store_true", help="Évaluer ai_detector.AiImageDetector complet.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--layout", default="folder", choices=["folder", "genimage", "manifest"])
    p.add_argument("--split", default="val")
    p.add_argument("--generators", nargs="*", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--robustness", nargs="*", default=[], help="ex. jpeg75 jpeg50 resize0.5 blur1.0")
    p.add_argument("--max-crops", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--model-path", default=None, help="(full-pipeline) ONNX à utiliser.")
    p.add_argument("--calibration-path", default=None, help="(full-pipeline) calibration.json à utiliser.")
    p.add_argument("--out-dir", default="runs/eval")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _load_pil(sample: Sample) -> Image.Image:
    with Image.open(sample.path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


class TorchScorer:
    def __init__(self, checkpoint: str, device: str, batch_size: int, workers: int, max_crops: int):
        import torch

        from training.common import get_device
        from training.model import load_checkpoint

        self.torch = torch
        self.device = get_device(device)
        self.model, self.ckpt = load_checkpoint(checkpoint, map_location=self.device)
        self.model.to(self.device).eval()
        self.input_size = self.model.input_size
        self.max_crops = max_crops
        self.name = f"torch:{Path(checkpoint).name}"

    def score(self, img: Image.Image) -> float:
        from ai_detector.preprocessing import grid_crops
        from training.transforms import IMAGENET_MEAN, IMAGENET_STD

        arr = np.asarray(img, dtype=np.float32) / 255.0
        crops = grid_crops(arr, self.input_size, self.max_crops)
        mean, std = np.asarray(IMAGENET_MEAN, np.float32), np.asarray(IMAGENET_STD, np.float32)
        batch = np.stack([((c - mean) / std).transpose(2, 0, 1) for c in crops]).astype(np.float32)
        with self.torch.no_grad():
            logits = self.model(self.torch.from_numpy(batch).to(self.device)).float().cpu().numpy().reshape(-1)
        return float(1.0 / (1.0 + np.exp(-logits.mean())))


class OnnxScorer:
    def __init__(self, onnx_path: str, max_crops: int):
        from ai_detector.cnn import CnnDetector

        self.det = CnnDetector(onnx_path, max_crops=max_crops)
        if not self.det.available:
            raise SystemExit(f"Impossible de charger {onnx_path} : {self.det.load_error}")
        self.name = f"onnx:{Path(onnx_path).name}"

    def score(self, img: Image.Image) -> float:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return self.det.predict(arr).cnn_score


class FullPipelineScorer:
    def __init__(self, model_path: str | None, calibration_path: str | None):
        from ai_detector.config import DetectorConfig
        from ai_detector.detector import AiImageDetector

        cfg = DetectorConfig()
        if model_path:
            cfg.model_path = Path(model_path)
        if calibration_path:
            cfg.calibration_path = Path(calibration_path)
        self.det = AiImageDetector(cfg)
        self.name = f"full-pipeline[{self.det.fusion_mode}]"
        self.last: dict[str, float] = {}

    def score(self, img: Image.Image) -> float:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        r = self.det.detect_array(arr, include_details=False)
        self.last = {"fft_score": r.fft_score, "noise_score": r.noise_score, "cnn_score": r.cnn_score}
        return r.confidence


def run_condition(scorer, samples: list[Sample], degradation: str, log) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    degrade = build_degradation(degradation)
    scores, labels, rows = [], [], []
    for i, s in enumerate(samples):
        try:
            img = degrade(_load_pil(s))
            sc = scorer.score(img)
        except Exception as exc:  # noqa: BLE001
            log.warning("échec %s : %s", s.path, exc)
            continue
        scores.append(sc)
        labels.append(s.label)
        row = {"path": s.path, "label": s.label, "generator": s.generator, "degradation": degradation, "score": sc}
        row.update(getattr(scorer, "last", {}))
        rows.append(row)
        if (i + 1) % 200 == 0:
            log.info("[%s] %d/%d", degradation, i + 1, len(samples))
    return np.asarray(labels), np.asarray(scores), rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    log = setup_logging(out_dir)
    samples = load_samples(args.data_root, args.layout, args.split, args.generators, args.max_samples, args.seed)
    log.info("évaluation sur %s", summarize(samples))

    if args.checkpoint:
        scorer = TorchScorer(args.checkpoint, args.device, args.batch_size, args.workers, args.max_crops)
    elif args.onnx:
        scorer = OnnxScorer(args.onnx, args.max_crops)
    else:
        scorer = FullPipelineScorer(args.model_path, args.calibration_path)

    conditions = ["none"] + [c for c in args.robustness if c != "none"]
    report: dict = {"created_at": now_iso(), "scorer": scorer.name, "split": args.split, "layout": args.layout,
                    "data_root": args.data_root, "threshold": args.threshold, "dataset": summarize(samples),
                    "conditions": {}, "per_generator": {}}
    all_rows: list[dict] = []
    for cond in conditions:
        labels, scores, rows = run_condition(scorer, samples, cond, log)
        all_rows.extend(rows)
        m = compute_metrics(labels, scores, args.threshold)
        report["conditions"][cond] = m
        log.info("[%s] auc=%.4f acc=%.4f bal_acc=%.4f eer=%.4f", cond, m["auc"], m["accuracy"], m["balanced_accuracy"], m["eer"])
        if cond == "none":
            by_gen: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for r in rows:
                by_gen[r["generator"]].append((r["label"], r["score"]))
            reals = [(l, s) for l, s in zip(labels, scores) if l == 0]
            for gen, pairs in sorted(by_gen.items()):
                if pairs and pairs[0][0] == 1:
                    # AUC générateur-vs-réelles : chaque générateur confronté à toutes les réelles.
                    ll = np.array([p[0] for p in pairs] + [p[0] for p in reals])
                    ss = np.array([p[1] for p in pairs] + [p[1] for p in reals])
                    report["per_generator"][gen] = compute_metrics(ll, ss, args.threshold)
                else:
                    ll = np.array([p[0] for p in pairs]); ss = np.array([p[1] for p in pairs])
                    report["per_generator"][gen] = {"n": int(ll.size), "accuracy": float(np.mean((ss >= args.threshold) == ll)),
                                                    "mean_score": float(ss.mean())}

    save_json(report, out_dir / "report.json")
    with (out_dir / "scores.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = sorted({k for r in all_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    md = [f"# Rapport d'évaluation — {scorer.name}", "", f"Généré le {report['created_at']} · split `{args.split}` · "
          f"{report['dataset']['n']} images ({report['dataset']['n_real']} réelles / {report['dataset']['n_ai']} IA) · seuil {args.threshold}", "",
          "## Conditions (robustesse)", "", metrics_table_markdown(report["conditions"]), "", "## Par générateur (vs toutes les réelles)", "",
          metrics_table_markdown({k: v for k, v in report["per_generator"].items() if "auc" in v},
                                 columns=("n", "accuracy", "auc", "tpr_recall_ai", "eer"))]
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("rapport écrit dans %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
