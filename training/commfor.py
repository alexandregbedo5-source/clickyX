"""Matérialisation d'un sous-ensemble de *Community Forensics-Small* (Hugging Face) en dossiers.

Pourquoi un script dédié plutôt que ``datasets`` en streaming :

* chaque fichier parquet du dataset est **un seul row group** de ~3 000 images (35 Mo à 4,1 Go) et la
  colonne ``image_data`` est encodée par dictionnaire (page de dictionnaire de plusieurs centaines de Mo) :
  lire une seule ligne oblige à charger quasi tout le fichier en mémoire. Le streaming ``datasets`` +
  tampon de mélange dépasse les 12 Go d'un runtime Colab et fait redémarrer la session ;
* les fichiers sont **triés par classe** (0–92 : IA, 94–185 : réelles) et par source : un tirage aléatoire
  de fichiers donne des milliers de réelles avant la première image IA.

Ici : liste fixe de fichiers couvrant les deux classes et des sources variées, téléchargement sur disque
d'un fichier à la fois, lecture par petits lots, **un sous-processus par fichier** (isolation mémoire),
images écrites **telles quelles** (pas de recompression), marqueurs de reprise par fichier.

Usage (notebook Colab) :

    python -m training.commfor --out DATA/commfor_small --max-per-class 6000

Disposition produite : ``out/real/<source>/…`` et ``out/ai/<générateur>/…`` (format pivot de
``training/data.py build-manifest``).
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO_ID = "OwensLab/CommunityForensics-Small"
COLUMNS = ["image_name", "image_data", "model_name", "label"]


@dataclass(frozen=True)
class Shard:
    num: int
    label: int  # 1 = IA, 0 = réelle
    size_gb: float  # taille approximative du parquet (téléchargement et pic mémoire)
    content: str

    @property
    def filename(self) -> str:
        return f"data/HFCF_small_{self.num}.parquet"


# Plan par défaut, établi à partir des métadonnées parquet du dataset (septembre 2026).
# IA : diffusion latente « Systematic » (~75 générateurs HF par fichier), GAN, diffusion pixel, autres.
# Réelles : COCO (640×480 JPEG), VISION (photos smartphone JPEG), Landscapes HQ (PNG), FFHQ (visages 1024² PNG).
DEFAULT_PLAN: tuple[Shard, ...] = (
    Shard(3, 1, 1.22, "LatDiff Systematic (~77 générateurs)"),
    Shard(41, 1, 1.22, "LatDiff Systematic (~76 générateurs)"),
    Shard(66, 1, 1.06, "LatDiff + GAN (21 générateurs)"),
    Shard(70, 1, 0.43, "GAN + Other"),
    Shard(77, 1, 0.42, "GAN + LatDiff"),
    Shard(83, 1, 2.00, "PixDiff + GAN"),
    Shard(117, 0, 0.15, "COCO"),
    Shard(116, 0, 1.13, "COCO + VISION"),
    Shard(115, 0, 2.63, "VISION"),
    Shard(156, 0, 1.80, "LandscapesHQ"),
    Shard(94, 0, 4.10, "FFHQ (visages 1024²)"),
)


def plan_quotas(plan: Sequence[Shard], max_per_class: int) -> dict[int, int]:
    """Nombre d'images à extraire par fichier pour atteindre ``max_per_class`` dans chaque classe."""
    quotas: dict[int, int] = {}
    for label in (0, 1):
        shards = [s for s in plan if s.label == label]
        if shards:
            q = math.ceil(max_per_class / len(shards))
            quotas.update({s.num: q for s in shards})
    return quotas


def dest_dir(out: Path, label: int, model_name: str | None) -> Path:
    name = (model_name or ("unknown" if label else "real")).replace("/", "__")
    return out / ("ai" if label == 1 else "real") / (name if label == 1 else name.lower())


def extract_parquet(path: Path, out: Path, quota: int, shard_num: int, batch_size: int = 8) -> Counter:
    """Lit ``path`` par petits lots et écrit jusqu'à ``quota`` images (octets d'origine)."""
    import pyarrow.parquet as pq
    from PIL import Image

    pf = pq.ParquetFile(str(path), pre_buffer=False, buffer_size=4 << 20)
    counts: Counter = Counter()
    written = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=COLUMNS, use_threads=False):
        for row in batch.to_pylist():
            data = row["image_data"]
            try:
                img = Image.open(io.BytesIO(data))
                fmt = (img.format or "PNG").upper()
                img.verify()
            except Exception:
                counts["corrompues"] += 1
                continue
            label = int(row["label"])
            ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif"}.get(fmt, "." + fmt.lower())
            sub = dest_dir(out, label, row["model_name"])
            sub.mkdir(parents=True, exist_ok=True)
            (sub / f"{shard_num}_{Path(row['image_name']).stem}{ext}").write_bytes(data)
            counts[label] += 1
            written += 1
            if written >= quota:
                break
        if written >= quota:
            break
    del pf
    gc.collect()
    return counts


def run_shard(shard: Shard, out: Path, quota: int, tmp: Path) -> Counter:
    """Télécharge un fichier parquet, l'extrait puis le supprime (exécuté dans un sous-processus)."""
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(REPO_ID, shard.filename, repo_type="dataset", local_dir=str(tmp)))
    try:
        counts = extract_parquet(path, out, quota, shard.num)
    finally:
        path.unlink(missing_ok=True)
        shutil.rmtree(tmp / ".cache", ignore_errors=True)
    return counts


def _marker(out: Path, shard: Shard) -> Path:
    return out / "_shards" / f"{shard.num}.json"


def _summary(out: Path) -> dict:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif"}
    res: dict = {}
    for cls in ("real", "ai"):
        root = out / cls
        if not root.is_dir():
            res[cls] = {"total": 0, "groups": {}}
            continue
        groups = {d.name: sum(1 for p in d.rglob("*") if p.suffix.lower() in exts) for d in sorted(root.iterdir()) if d.is_dir()}
        flat = sum(1 for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)  # ancienne disposition à plat
        if flat:
            groups["(racine)"] = flat
        res[cls] = {"total": sum(groups.values()), "groups": groups}
    return res


def materialize(out: Path, max_per_class: int, plan: Sequence[Shard] = DEFAULT_PLAN, max_shard_gb: float = 4.5,
                tmp: Path | None = None) -> dict:
    """Orchestrateur : un sous-processus par fichier, reprise via ``out/_shards/<n>.json``."""
    out.mkdir(parents=True, exist_ok=True)
    tmp = tmp or out / "_tmp"
    if (out / "_done").exists():
        print(f"{out} déjà matérialisé (_done présent) : rien à faire.")
        return _summary(out)
    quotas = plan_quotas(plan, max_per_class)
    selected = [s for s in plan if s.size_gb <= max_shard_gb]
    skipped = [s for s in plan if s.size_gb > max_shard_gb]
    for s in skipped:
        print(f"[ignoré] fichier {s.num} ({s.content}, {s.size_gb:.1f} Go) > --max-shard-gb {max_shard_gb}")
    # petits fichiers d'abord : progrès visible rapidement, le plus gros en dernier
    for shard in sorted(selected, key=lambda s: s.size_gb):
        marker = _marker(out, shard)
        if marker.exists():
            print(f"[ok, déjà fait] fichier {shard.num} ({shard.content})")
            continue
        quota = quotas[shard.num]
        t0 = time.time()
        print(f"[{'IA' if shard.label else 'réel'}] fichier {shard.num} — {shard.content} — {shard.size_gb:.1f} Go, "
              f"{quota} images à extraire …", flush=True)
        cmd = [sys.executable, "-m", "training.commfor", "--shard", str(shard.num), "--out", str(out),
               "--quota", str(quota), "--tmp", str(tmp)]
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent), capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            hint = " (processus tué : mémoire insuffisante ? relancer, ou baisser --max-shard-gb)" if proc.returncode < 0 or proc.returncode == 137 else ""
            print(f"  ÉCHEC fichier {shard.num} (code {proc.returncode}){hint}\n  " + "\n  ".join(tail), flush=True)
            continue
        counts = json.loads(proc.stdout.strip().splitlines()[-1])
        marker.parent.mkdir(exist_ok=True)
        marker.write_text(json.dumps({"shard": shard.num, "counts": counts, "seconds": round(time.time() - t0)}))
        print(f"  → {counts} en {time.time() - t0:.0f} s", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    summary = _summary(out)
    done_all = all(_marker(out, s).exists() for s in selected)
    if done_all:
        (out / "_done").touch()
    print(f"\nRésumé : {summary['real']['total']} réelles, {summary['ai']['total']} IA "
          f"({len(summary['ai']['groups'])} générateurs){'' if done_all else ' — INCOMPLET : relancer la commande pour reprendre'}")
    for cls in ("real", "ai"):
        top = sorted(summary[cls]["groups"].items(), key=lambda kv: -kv[1])[:8]
        print(f"  {cls} : " + ", ".join(f"{k}={v}" for k, v in top) + (" …" if len(summary[cls]["groups"]) > 8 else ""))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Matérialiser un sous-ensemble de Community Forensics-Small en dossiers.")
    p.add_argument("--out", required=True, help="Dossier de sortie (real/<source>/, ai/<générateur>/).")
    p.add_argument("--max-per-class", type=int, default=6000)
    p.add_argument("--max-shard-gb", type=float, default=4.5,
                   help="Ignorer les fichiers plus gros (pic mémoire ≈ taille du fichier). 4.5 inclut FFHQ (4,1 Go).")
    p.add_argument("--tmp", default=None, help="Dossier temporaire de téléchargement (défaut : <out>/_tmp).")
    p.add_argument("--shard", type=int, default=None, help="(interne) traiter un seul fichier dans ce processus.")
    p.add_argument("--quota", type=int, default=None, help="(interne) nombre d'images pour --shard.")
    args = p.parse_args(argv)
    out = Path(args.out)
    if args.shard is not None:
        shard = next((s for s in DEFAULT_PLAN if s.num == args.shard), None) or Shard(args.shard, -1, 0.0, "hors plan")
        counts = run_shard(shard, out, args.quota or args.max_per_class, Path(args.tmp or out / "_tmp"))
        print(json.dumps({str(k): v for k, v in counts.items()}))
        return 0
    materialize(out, args.max_per_class, max_shard_gb=args.max_shard_gb, tmp=Path(args.tmp) if args.tmp else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
