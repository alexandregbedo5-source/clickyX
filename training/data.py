"""Chargement des jeux de données forensiques.

Dispositions supportées (``--layout``) :

* ``folder``   : ``root/{train,val,test}/{real|nature}/…`` et ``root/{split}/{ai|fake}/…``
* ``genimage`` : disposition officielle GenImage —
                 ``root/<Générateur>/[dossier intermédiaire]/{train,val}/{ai,nature}/…``
* ``manifest`` : CSV ``path,label,generator[,split]`` (label 1 = IA). C'est le
                 format pivot pour mélanger des sources hétérogènes
                 (AIGenImages2026 + photos réelles, Community Forensics, etc.).

Le module fournit aussi un échantillonneur **équilibré** (labels *et*
générateurs), condition nécessaire pour ne pas apprendre le style d'un seul
générateur, et une CLI de construction de manifest :

    python -m training.data build-manifest --ai-dir DATA/aigen2026 --real-dir DATA/real \
        --out DATA/manifest.csv --val-fraction 0.1 --group-by-generator
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REAL_DIR_NAMES = {"real", "nature", "0", "genuine", "photo", "photos", "authentic"}
AI_DIR_NAMES = {"ai", "fake", "1", "generated", "synthetic", "fakes"}
SPLIT_ALIASES = {"train": ("train", "training"), "val": ("val", "valid", "validation", "test_val"), "test": ("test", "eval")}


@dataclass(frozen=True)
class Sample:
    path: str
    label: int  # 1 = IA, 0 = réelle
    generator: str  # nom du générateur ou source réelle ("real:<source>")
    source: str  # dataset d'origine


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def _find_split_dir(base: Path, split: str) -> Path | None:
    for alias in SPLIT_ALIASES.get(split, (split,)):
        cand = base / alias
        if cand.is_dir():
            return cand
    return None


def scan_folder_layout(root: Path, split: str, source: str = "folder") -> list[Sample]:
    split_dir = _find_split_dir(root, split)
    if split_dir is None:
        raise FileNotFoundError(f"Split '{split}' introuvable sous {root}")
    samples: list[Sample] = []
    for child in sorted(split_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name.lower()
        if name in REAL_DIR_NAMES:
            label, gen = 0, f"real:{root.name}"
        elif name in AI_DIR_NAMES:
            label, gen = 1, root.name
        else:
            continue
        samples.extend(Sample(str(p), label, gen, source) for p in list_images(child))
    return samples


def scan_genimage(root: Path, split: str, generators: Sequence[str] | None = None) -> list[Sample]:
    """Disposition GenImage : un dossier par générateur, éventuellement un dossier intermédiaire."""
    samples: list[Sample] = []
    for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        gen_name = gen_dir.name
        if generators and gen_name not in generators:
            continue
        split_dir = _find_split_dir(gen_dir, split)
        if split_dir is None:
            inter = [p for p in gen_dir.iterdir() if p.is_dir()]
            for cand in inter:
                split_dir = _find_split_dir(cand, split)
                if split_dir is not None:
                    break
        if split_dir is None:
            continue
        for sub in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            lname = sub.name.lower()
            if lname in AI_DIR_NAMES:
                samples.extend(Sample(str(p), 1, gen_name, "genimage") for p in list_images(sub))
            elif lname in REAL_DIR_NAMES:
                samples.extend(Sample(str(p), 0, f"real:imagenet", "genimage") for p in list_images(sub))
    return samples


def scan_manifest(path: Path, split: str | None = None) -> list[Sample]:
    samples: list[Sample] = []
    base = path.parent
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if split and row.get("split") and row["split"] != split:
                continue
            p = Path(row["path"])
            if not p.is_absolute():
                p = base / p
            samples.append(Sample(str(p), int(row["label"]), row.get("generator") or ("ai" if int(row["label"]) else "real"),
                                  row.get("source") or "manifest"))
    return samples


def write_manifest(samples: Iterable[Sample], path: Path, splits: dict[str, str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "label", "generator", "source", "split"])
        for s in samples:
            writer.writerow([s.path, s.label, s.generator, s.source, (splits or {}).get(s.path, "")])


def load_samples(root: str | Path, layout: str, split: str, generators: Sequence[str] | None = None,
                 max_samples: int | None = None, seed: int = 0) -> list[Sample]:
    root = Path(root)
    if layout == "folder":
        samples = scan_folder_layout(root, split)
    elif layout == "genimage":
        samples = scan_genimage(root, split, generators)
    elif layout == "manifest":
        samples = scan_manifest(root, split)
    else:
        raise ValueError(f"layout inconnu : {layout}")
    if not samples:
        raise RuntimeError(f"Aucune image trouvée ({layout}, split={split}) sous {root}")
    if max_samples and len(samples) > max_samples:
        rng = random.Random(seed)
        samples = rng.sample(samples, max_samples)
    return samples


def summarize(samples: Sequence[Sample]) -> dict:
    by_label = Counter(s.label for s in samples)
    by_gen = Counter(s.generator for s in samples)
    return {"n": len(samples), "n_real": by_label.get(0, 0), "n_ai": by_label.get(1, 0),
            "generators": dict(sorted(by_gen.items()))}


def balanced_sampler_weights(samples: Sequence[Sample]) -> list[float]:
    """Poids tels que chaque classe pèse 1/2 et, dans une classe, chaque générateur pèse autant."""
    per_label_gens: dict[int, Counter] = defaultdict(Counter)
    for s in samples:
        per_label_gens[s.label][s.generator] += 1
    weights = []
    for s in samples:
        n_gens = len(per_label_gens[s.label])
        n_in_gen = per_label_gens[s.label][s.generator]
        weights.append(1.0 / (2.0 * n_gens * n_in_gen))
    return weights


def group_split(samples: Sequence[Sample], val_fraction: float, seed: int, group_by_generator: bool) -> dict[str, str]:
    """Split train/val. Si ``group_by_generator``, des générateurs entiers vont en val
    (mesure la généralisation à des générateurs jamais vus)."""
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    if group_by_generator:
        ai_gens = sorted({s.generator for s in samples if s.label == 1})
        rng.shuffle(ai_gens)
        n_val = max(1, int(round(len(ai_gens) * val_fraction))) if len(ai_gens) > 1 else 0
        val_gens = set(ai_gens[:n_val])
        for s in samples:
            if s.label == 1:
                assignment[s.path] = "val" if s.generator in val_gens else "train"
        reals = [s for s in samples if s.label == 0]
        rng.shuffle(reals)
        n_val_real = int(round(len(reals) * val_fraction))
        for i, s in enumerate(reals):
            assignment[s.path] = "val" if i < n_val_real else "train"
    else:
        by_gen: dict[str, list[Sample]] = defaultdict(list)
        for s in samples:
            by_gen[s.generator].append(s)
        for gen_samples in by_gen.values():
            rng.shuffle(gen_samples)
            n_val = int(round(len(gen_samples) * val_fraction))
            for i, s in enumerate(gen_samples):
                assignment[s.path] = "val" if i < n_val else "train"
    return assignment


# --------------------------------------------------------------------------- torch
class ForensicDataset:
    """Dataset PyTorch : renvoie ``(tensor, label float32, index)``.

    Défini sans héritage direct de ``torch.utils.data.Dataset`` pour que ce module
    reste importable sans torch (construction de manifests) ; la compatibilité
    avec ``DataLoader`` n'exige que ``__len__``/``__getitem__``.
    """

    def __init__(self, samples: Sequence[Sample], transform, on_error: str = "skip"):
        self.samples = list(samples)
        self.transform = transform
        self.on_error = on_error

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image, ImageOps

        s = self.samples[idx]
        try:
            with Image.open(s.path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                x = self.transform(img)
        except Exception:  # noqa: BLE001 — fichier corrompu : on remplace par un voisin
            if self.on_error != "skip":
                raise
            return self[(idx + 1) % len(self.samples)]
        import torch

        return x, torch.tensor(float(s.label), dtype=torch.float32), idx


def make_dataloader(dataset: ForensicDataset, batch_size: int, shuffle: bool, num_workers: int,
                    balanced: bool = False, seed: int = 0):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    sampler = None
    if balanced and shuffle:
        weights = balanced_sampler_weights(dataset.samples)
        gen = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=gen)
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False, persistent_workers=num_workers > 0)


# --------------------------------------------------------------------------- CLI
def _cmd_build_manifest(args: argparse.Namespace) -> int:
    samples: list[Sample] = []
    for d in args.ai_dir or []:
        root = Path(d)
        subdirs = [p for p in root.iterdir() if p.is_dir()] if args.generator_from_subdir else []
        if subdirs:
            for sub in sorted(subdirs):
                samples.extend(Sample(str(p), 1, sub.name, root.name) for p in list_images(sub))
        else:
            samples.extend(Sample(str(p), 1, root.name, root.name) for p in list_images(root))
    for d in args.real_dir or []:
        root = Path(d)
        samples.extend(Sample(str(p), 0, f"real:{root.name}", root.name) for p in list_images(root))
    if not samples:
        raise SystemExit("Aucune image trouvée.")
    assignment = group_split(samples, args.val_fraction, args.seed, args.group_by_generator)
    write_manifest(samples, Path(args.out), assignment)
    summary = summarize(samples)
    n_val = sum(1 for v in assignment.values() if v == "val")
    print(f"manifest écrit : {args.out}  ({summary['n']} images, {summary['n_real']} réelles, {summary['n_ai']} IA, "
          f"{len(summary['generators'])} générateurs/sources, {n_val} en val)")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    import json

    samples = load_samples(args.root, args.layout, args.split)
    print(json.dumps(summarize(samples), indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Outils de données pour l'entraînement du détecteur.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build-manifest", help="Construire un manifest CSV à partir de dossiers.")
    p.add_argument("--ai-dir", action="append", help="Dossier d'images IA (répétable).")
    p.add_argument("--real-dir", action="append", help="Dossier d'images réelles (répétable).")
    p.add_argument("--generator-from-subdir", action="store_true", default=True,
                   help="Le premier niveau de sous-dossier d'un --ai-dir est le nom du générateur (défaut).")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--group-by-generator", action="store_true", help="Générateurs entiers réservés à la validation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_build_manifest)
    s = sub.add_parser("summary", help="Résumé d'un jeu de données.")
    s.add_argument("--root", required=True)
    s.add_argument("--layout", default="folder", choices=["folder", "genimage", "manifest"])
    s.add_argument("--split", default="train")
    s.set_defaults(func=_cmd_summary)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
