"""Métriques de classification binaire (numpy uniquement).

Convention : label 1 = image générée par IA, score = probabilité estimée d'IA.
"""

from __future__ import annotations

import numpy as np


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[idx]
    fps = (1 + idx) - tps
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int(y.size - y.sum()), 1)
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thr = np.r_[np.inf, s[idx]]
    return fpr, tpr, thr


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.min() == labels.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    return float(np.sum(precision * y) / y.sum())


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(np.asarray(labels).astype(int), np.asarray(scores, dtype=np.float64))
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if np.any(ok) else 0.0


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thr = roc_curve(np.asarray(labels).astype(int), np.asarray(scores, dtype=np.float64))
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float(0.5 * (fpr[i] + fnr[i])), float(thr[i])


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    bins = np.clip((scores * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if not np.any(m):
            continue
        ece += m.mean() * abs(scores[m].mean() - labels[m].mean())
    return float(ece)


def compute_metrics(labels, scores, threshold: float = 0.5) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    n_pos, n_neg = tp + fn, tn + fp
    tpr = tp / n_pos if n_pos else float("nan")
    tnr = tn / n_neg if n_neg else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * precision * tpr / (precision + tpr) if (tp + fp) and n_pos and (precision + tpr) > 0 else float("nan")
    eer, eer_thr = equal_error_rate(labels, scores) if n_pos and n_neg else (float("nan"), float("nan"))
    return {
        "n": int(labels.size),
        "n_real": int(n_neg),
        "n_ai": int(n_pos),
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(labels.size, 1)),
        "balanced_accuracy": float(np.nanmean([tpr, tnr])),
        "tpr_recall_ai": float(tpr),
        "tnr_specificity_real": float(tnr),
        "precision_ai": float(precision),
        "f1_ai": float(f1),
        "auc": auc_score(labels, scores) if n_pos and n_neg else float("nan"),
        "average_precision": average_precision(labels, scores) if n_pos else float("nan"),
        "tpr_at_fpr_1pct": tpr_at_fpr(labels, scores, 0.01) if n_pos and n_neg else float("nan"),
        "tpr_at_fpr_5pct": tpr_at_fpr(labels, scores, 0.05) if n_pos and n_neg else float("nan"),
        "eer": float(eer),
        "eer_threshold": float(eer_thr),
        "ece": expected_calibration_error(labels, scores),
        "brier": float(np.mean((scores - labels) ** 2)),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},  # type: ignore[dict-item]
    }


def metrics_table_markdown(rows: dict[str, dict[str, float]], columns: tuple[str, ...] = ("n", "accuracy", "balanced_accuracy", "auc", "average_precision", "tpr_recall_ai", "tnr_specificity_real", "eer")) -> str:
    header = "| Sous-ensemble | " + " | ".join(columns) + " |"
    sep = "|" + "---|" * (len(columns) + 1)
    lines = [header, sep]
    for name, m in rows.items():
        cells = []
        for c in columns:
            v = m.get(c, float("nan"))
            cells.append(f"{v}" if isinstance(v, int) else (f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else "—"))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
