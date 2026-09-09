"""Entraînement du détecteur CNN (PyTorch).

Exemples :

    # GenImage complet (disposition officielle), EfficientNet-B0, GPU Colab
    python training/train.py --data-root /content/GenImage --layout genimage \
        --arch efficientnet_b0 --epochs 8 --batch-size 64 --amp --out-dir runs/effb0

    # Mélange hétérogène via manifest (AIGenImages2026 + réelles), ConvNeXt-T
    python training/train.py --data-root data/manifest.csv --layout manifest --arch convnext_tiny

Protocole anti-surapprentissage (voir docs/ai_detector.md) : échantillonnage
équilibré classes × générateurs, augmentations de dégradation, weight decay,
early stopping sur l'AUC de validation, EMA des poids, label smoothing léger.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from training.common import get_device, git_commit, now_iso, save_json, seed_everything, setup_logging  # noqa: E402
from training.data import ForensicDataset, load_samples, make_dataloader, summarize  # noqa: E402
from training.metrics import compute_metrics  # noqa: E402
from training.model import ARCH_PRESETS, build_model, count_parameters, save_checkpoint  # noqa: E402
from training.transforms import AugmentConfig, build_eval_transform, build_train_transform  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraînement du détecteur d'images IA.")
    p.add_argument("--data-root", required=True, help="Racine du dataset ou chemin du manifest CSV.")
    p.add_argument("--layout", default="folder", choices=["folder", "genimage", "manifest"])
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--generators", nargs="*", default=None, help="(genimage) sous-ensemble de générateurs.")
    p.add_argument("--arch", default="efficientnet_b0", choices=sorted(ARCH_PRESETS))
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--img-size", type=int, default=None, help="Taille de crop (défaut : préréglage de l'architecture).")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--head-lr-mult", type=float, default=10.0, help="Multiplicateur de LR pour la tête.")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=0.5)
    p.add_argument("--freeze-backbone-epochs", type=int, default=0, help="Époques initiales avec backbone gelé.")
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--drop-rate", type=float, default=0.2)
    p.add_argument("--ema", type=float, default=0.999, help="Décroissance EMA des poids (0 = désactivé).")
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--no-balanced-sampling", action="store_true")
    p.add_argument("--jpeg-p", type=float, default=0.5)
    p.add_argument("--rescale-p", type=float, default=0.3)
    p.add_argument("--blur-p", type=float, default=0.1)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="Précision mixte (GPU).")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="runs/detector")
    p.add_argument("--resume", default=None, help="Checkpoint last.pt à reprendre.")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--log-interval", type=int, default=50)
    return p.parse_args(argv)


class ModelEma:
    def __init__(self, model: nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src.detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(src)


def cosine_with_warmup(step: int, total: int, warmup: int, base_lr: float, min_lr_ratio: float = 0.02) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, amp: bool) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x).float().squeeze(1)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores) if scores else np.zeros(0)
    l = np.concatenate(labels) if labels else np.zeros(0)
    return compute_metrics(l, s), l, s


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    log = setup_logging(out_dir)
    seed_everything(args.seed)
    device = get_device(args.device)
    log.info("device=%s torch=%s", device, torch.__version__)

    img_size = args.img_size or ARCH_PRESETS[args.arch].input_size
    aug = AugmentConfig(jpeg_p=args.jpeg_p, rescale_p=args.rescale_p, blur_p=args.blur_p)

    train_samples = load_samples(args.data_root, args.layout, args.train_split, args.generators, args.max_train_samples, args.seed)
    val_samples = load_samples(args.data_root, args.layout, args.val_split, args.generators, args.max_val_samples, args.seed)
    log.info("train: %s", summarize(train_samples))
    log.info("val:   %s", summarize(val_samples))

    train_ds = ForensicDataset(train_samples, build_train_transform(img_size, aug))
    val_ds = ForensicDataset(val_samples, build_eval_transform(img_size))
    train_loader = make_dataloader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                                  balanced=not args.no_balanced_sampling, seed=args.seed)
    val_loader = make_dataloader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(args.arch, pretrained=not args.no_pretrained, drop_rate=args.drop_rate, input_size=img_size).to(device)
    log.info("arch=%s params=%.2fM img_size=%d", args.arch, count_parameters(model) / 1e6, img_size)

    head_params = list(model.head.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()
    ema = ModelEma(model, args.ema) if args.ema > 0 else None

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    start_epoch, global_step, best_auc, bad_epochs = 0, 0, -1.0, 0
    history: list[dict] = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ema is not None and "ema_state_dict" in ckpt:
            ema.module.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = start_epoch * steps_per_epoch
        best_auc = float(ckpt.get("best_auc", -1.0))
        history = list(ckpt.get("history", []))
        log.info("reprise depuis %s (époque %d)", args.resume, start_epoch)

    save_json({**vars(args), "img_size": img_size, "git_commit": git_commit(), "started_at": now_iso(),
               "train_summary": summarize(train_samples), "val_summary": summarize(val_samples)}, out_dir / "train_config.json")

    for epoch in range(start_epoch, args.epochs):
        frozen = epoch < args.freeze_backbone_epochs
        model.freeze_backbone(frozen)
        model.train()
        t_epoch = time.time()
        running, n_seen = 0.0, 0
        for it, (x, y, _) in enumerate(train_loader):
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = cosine_with_warmup(global_step, total_steps, warmup_steps, base)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if args.label_smoothing > 0:
                y = y * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                logits = model(x).squeeze(1)
                loss = criterion(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            running += loss.item() * x.size(0)
            n_seen += x.size(0)
            global_step += 1
            if (it + 1) % args.log_interval == 0:
                log.info("ep %d it %d/%d loss %.4f lr %.2e", epoch, it + 1, steps_per_epoch, running / n_seen, optimizer.param_groups[0]["lr"])

        record = {"epoch": epoch, "train_loss": running / max(n_seen, 1), "epoch_seconds": time.time() - t_epoch, "frozen_backbone": frozen}
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            eval_model = ema.module if ema is not None else model
            metrics, _, _ = evaluate(eval_model, val_loader, device, args.amp)
            record["val"] = metrics
            log.info("ep %d  val auc %.4f  acc %.4f  bal_acc %.4f  eer %.4f", epoch, metrics["auc"], metrics["accuracy"],
                     metrics["balanced_accuracy"], metrics["eer"])
            auc = metrics["auc"] if np.isfinite(metrics["auc"]) else metrics["accuracy"]
            if auc > best_auc:
                best_auc, bad_epochs = auc, 0
                save_checkpoint(eval_model, out_dir / "best.pt", epoch=epoch, metrics=metrics, best_auc=best_auc,
                                train_config=vars(args), git_commit=git_commit(), saved_at=now_iso())
                log.info("nouveau meilleur modèle (auc=%.4f) → %s", best_auc, out_dir / "best.pt")
            else:
                bad_epochs += 1
        history.append(record)
        save_json(history, out_dir / "history.json")
        save_checkpoint(model, out_dir / "last.pt", epoch=epoch, optimizer=optimizer.state_dict(), best_auc=best_auc,
                        history=history, ema_state_dict=(ema.module.state_dict() if ema is not None else None),
                        train_config=vars(args))
        if args.early_stopping_patience and bad_epochs >= args.early_stopping_patience:
            log.info("early stopping (patience %d)", args.early_stopping_patience)
            break

    log.info("terminé. meilleur auc=%.4f. checkpoints dans %s", best_auc, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
