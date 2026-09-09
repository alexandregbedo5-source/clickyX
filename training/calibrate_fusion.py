"""Calibration des modules hand-crafted et de la fusion sur un jeu étiqueté.

Ajuste, par régression logistique régularisée :

1. la transformation caractéristiques fréquentielles → ``fft_score`` ;
2. la transformation caractéristiques de bruit → ``noise_score`` ;
3. la fusion ``(fft, noise[, cnn]) → confidence`` — sur des scores **hors-pli**
   (validation croisée K-fold) pour ne pas sur-évaluer la fusion.

Écrit ``model/calibration.json`` (lu par le moteur au démarrage) et un CSV de
caractéristiques pour analyse.

    python training/calibrate_fusion.py --data-root DATA --layout folder --split val \
        --onnx model/detector.onnx --out model/calibration.json --source-name "GenImage val (SD1.5+MJ+ADM) 2026-09"
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from ai_detector.calibration import Calibration, FeatureMapping, FusionParams, logit  # noqa: E402
from ai_detector.calibration_defaults import DEFAULT_CALIBRATION  # noqa: E402
from ai_detector.cnn import CnnDetector  # noqa: E402
from ai_detector.frequency import compute_spectrum_features  # noqa: E402
from ai_detector.noise import compute_noise_features  # noqa: E402
from ai_detector.preprocessing import center_window, load_image_from_path, to_grayscale  # noqa: E402
from training.common import git_commit, now_iso, save_json, setup_logging  # noqa: E402
from training.data import Sample, load_samples, summarize  # noqa: E402
from training.metrics import auc_score, compute_metrics  # noqa: E402

FREQ_FEATURES = DEFAULT_CALIBRATION["frequency"]["features"]
NOISE_FEATURES = DEFAULT_CALIBRATION["noise"]["features"]
Z_CLIP = 4.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration des scores hand-crafted et de la fusion.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--layout", default="folder", choices=["folder", "genimage", "manifest"])
    p.add_argument("--split", default="val")
    p.add_argument("--generators", nargs="*", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--onnx", default=None, help="Modèle ONNX (si entraîné, la fusion 'full' est ajustée).")
    p.add_argument("--analysis-window", type=int, default=1024)
    p.add_argument("--l2", type=float, default=1.0, help="Régularisation L2 (plus grand = poids plus petits).")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", default="model/calibration.json")
    p.add_argument("--features-csv", default=None)
    p.add_argument("--source-name", default=None, help="Description du jeu de calibration (traçabilité).")
    p.add_argument("--version", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- régression logistique
def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    """Régression logistique (biais non régularisé), L-BFGS-B, classes pondérées à l'équilibre."""
    n, d = X.shape
    w_pos = 0.5 / max(y.sum(), 1)
    w_neg = 0.5 / max((1 - y).sum(), 1)
    sw = np.where(y == 1, w_pos, w_neg) * n

    def f(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = theta[:-1], theta[-1]
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        loss = -np.sum(sw * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))) / n + 0.5 * l2 * np.dot(w, w) / n
        g = sw * (p - y)
        grad_w = X.T @ g / n + l2 * w / n
        grad_b = np.sum(g) / n
        return float(loss), np.r_[grad_w, grad_b]

    res = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B")
    return res.x[:-1], float(res.x[-1])


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # colonnes entièrement NaN
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Z = (X - mean) / std
    Z = np.where(np.isfinite(Z), Z, 0.0)  # NaN → moyenne (neutre)
    return np.clip(Z, -Z_CLIP, Z_CLIP)


def fit_mapping(X: np.ndarray, y: np.ndarray, names: list[str], l2: float) -> FeatureMapping:
    mean, std = standardize_fit(X)
    Z = standardize_apply(X, mean, std)
    w, b = fit_logistic(Z, y, l2)
    return FeatureMapping(features=list(names), mean=mean.tolist(), std=std.tolist(), weights=w.tolist(), bias=b)


def oof_scores(X: np.ndarray, y: np.ndarray, names: list[str], l2: float, folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    out = np.zeros(len(y))
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        m = fit_mapping(X[train], y[train], names, l2)
        Z = standardize_apply(X[test], np.asarray(m.mean), np.asarray(m.std))
        out[test] = 1.0 / (1.0 + np.exp(-(Z @ np.asarray(m.weights) + m.bias)))
    return out


# ----------------------------------------------------------------------------- extraction
def extract(samples: list[Sample], cnn: CnnDetector | None, window: int, log) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    Xf, Xn, cnn_scores, labels, rows = [], [], [], [], []
    for i, s in enumerate(samples):
        try:
            img = load_image_from_path(s.path)
        except Exception as exc:  # noqa: BLE001
            log.warning("ignorée %s : %s", s.path, exc)
            continue
        win = center_window(img.rgb, window)
        gray = to_grayscale(win)
        ff, _ = compute_spectrum_features(gray)
        nf, _ = compute_noise_features(win)
        fd, nd = ff.as_dict(), nf.as_dict()
        c = cnn.predict(img.rgb).cnn_score if cnn is not None else float("nan")
        Xf.append([fd[k] for k in FREQ_FEATURES])
        Xn.append([nd[k] for k in NOISE_FEATURES])
        cnn_scores.append(c)
        labels.append(s.label)
        rows.append({"path": s.path, "label": s.label, "generator": s.generator, "cnn_score": c, **fd, **nd})
        if (i + 1) % 100 == 0:
            log.info("%d/%d", i + 1, len(samples))
    return np.asarray(Xf, float), np.asarray(Xn, float), np.asarray(cnn_scores, float), np.asarray(labels, int), rows


def univariate_aucs(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for j, n in enumerate(names):
        col = X[:, j]
        ok = np.isfinite(col)
        out[n] = auc_score(y[ok], col[ok]) if ok.sum() > 10 and len(set(y[ok])) == 2 else None
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    log = setup_logging(None)
    samples = load_samples(args.data_root, args.layout, args.split, args.generators, args.max_samples, args.seed)
    log.info("calibration sur %s", summarize(samples))

    cnn = None
    if args.onnx:
        cnn = CnnDetector(args.onnx)
        if not cnn.available:
            log.warning("ONNX non chargé (%s) : fusion 'full' non ajustée", cnn.load_error)
            cnn = None
        elif not cnn.trained:
            log.warning("ONNX marqué trained=false : ignoré pour la fusion 'full'")
            cnn = None

    Xf, Xn, cs, y, rows = extract(samples, cnn, args.analysis_window, log)
    if len(set(y.tolist())) < 2:
        raise SystemExit("Il faut des exemples des deux classes.")

    if args.features_csv:
        Path(args.features_csv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.features_csv).open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # 1–2. Modules hand-crafted -------------------------------------------------
    freq_map = fit_mapping(Xf, y, FREQ_FEATURES, args.l2)
    noise_map = fit_mapping(Xn, y, NOISE_FEATURES, args.l2)
    fft_oof = oof_scores(Xf, y, FREQ_FEATURES, args.l2, args.folds, args.seed)
    noise_oof = oof_scores(Xn, y, NOISE_FEATURES, args.l2, args.folds, args.seed)

    # 3. Fusion sur scores hors-pli ---------------------------------------------
    lf = np.array([logit(v) for v in fft_oof])
    ln = np.array([logit(v) for v in noise_oof])
    w_hc, b_hc = fit_logistic(np.c_[lf, ln], y, args.l2 * 0.1)
    hand = FusionParams(weights={"fft": float(w_hc[0]), "noise": float(w_hc[1])}, bias=b_hc)
    hc_conf = 1.0 / (1.0 + np.exp(-(np.c_[lf, ln] @ w_hc + b_hc)))

    report = {
        "n": int(len(y)), "n_real": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
        "generators": summarize(samples)["generators"],
        "univariate_auc": {"frequency": univariate_aucs(Xf, y, FREQ_FEATURES), "noise": univariate_aucs(Xn, y, NOISE_FEATURES)},
        "module_auc_oof": {"fft_score": auc_score(y, fft_oof), "noise_score": auc_score(y, noise_oof)},
        "fusion_handcrafted_oof": compute_metrics(y, hc_conf),
    }

    if cnn is not None and np.all(np.isfinite(cs)):
        lc = np.array([logit(v) for v in cs])
        w_full, b_full = fit_logistic(np.c_[lf, ln, lc], y, args.l2 * 0.1)
        full = FusionParams(weights={"fft": float(w_full[0]), "noise": float(w_full[1]), "cnn": float(w_full[2])}, bias=b_full)
        full_conf = 1.0 / (1.0 + np.exp(-(np.c_[lf, ln, lc] @ w_full + b_full)))
        report["cnn_auc"] = auc_score(y, cs)
        report["fusion_full_oof"] = compute_metrics(y, full_conf)
    else:
        full = FusionParams.from_dict(DEFAULT_CALIBRATION["fusion"]["full"])
        report["fusion_full_oof"] = None

    version = args.version or f"calib-{now_iso()[:10]}"
    calib = Calibration(
        frequency=freq_map, noise=noise_map, fusion_full=full, fusion_handcrafted=hand, threshold=0.5,
        source=args.source_name or f"{args.layout}:{args.data_root}:{args.split}", version=version,
        metadata={"created_at": now_iso(), "git_commit": git_commit(), "l2": args.l2, "folds": args.folds,
                  "cnn_model": (cnn.model_version if cnn else None), "report": report},
    )
    calib.save(out)
    log.info("calibration écrite : %s", out)
    log.info("AUC hors-pli — fft: %.3f  noise: %.3f  fusion(hc): %.3f%s", report["module_auc_oof"]["fft_score"],
             report["module_auc_oof"]["noise_score"], report["fusion_handcrafted_oof"]["auc"],
             f"  fusion(full): {report['fusion_full_oof']['auc']:.3f}" if report.get("fusion_full_oof") else "")
    save_json(report, out.with_name(out.stem + "_report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
