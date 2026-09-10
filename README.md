# ClickyX — AI Image Detector (module IA Forensics)

Moteur de **forensic numérique local** qui estime la probabilité qu'une image ait été
générée par une IA. Il ne s'agit pas d'un simple classificateur : trois analyses
indépendantes — **spectre fréquentiel (FFT)**, **bruit résiduel du capteur** et
**réseau de neurones (ONNX)** — sont fusionnées en une estimation probabiliste
explicable. Tout s'exécute **hors ligne**, sur CPU, sous Windows, Linux et macOS.

Ce module est développé sur la branche `feature/ai-image-detector` du projet
[ClickyX](https://github.com/alexandregbedo5-source/clickyX) (hackathon, équipe de 3).
Il ne touche ni aux composants React, ni aux providers offline : il expose un
**contrat d'API** (`POST /detect-ai-image`) consommé par l'interface et par
l'intégration locale. Documentation complète : [`docs/ai_detector.md`](docs/ai_detector.md).

```
Image → Prétraitement → Analyse fréquentielle → Bruit résiduel → CNN (ONNX) → Fusion → Score final
```

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt                          # inférence (numpy, scipy, pillow, onnxruntime, fastapi)
pip install -e .                                         # rend `ai_detector` importable + commande `ai-detector`

python -m ai_detector detect photo.jpg                   # rapport lisible
python -m ai_detector detect photo.jpg --json            # contrat v1 strict (5 champs)
python -m ai_detector serve                              # API sur http://127.0.0.1:32188 (docs : /docs)
```

```bash
curl -X POST http://127.0.0.1:32188/detect-ai-image \
     -H "Content-Type: application/json" \
     -d '{"image_path": "/chemin/vers/image.png"}'
```

```json
{
  "is_ai_generated": true,
  "confidence": 0.92,
  "fft_score": 0.81,
  "noise_score": 0.73,
  "cnn_score": 0.95
}
```

## Contenu du dépôt

| Chemin | Rôle |
|---|---|
| `ai_detector/` | Moteur d'inférence : `preprocessing`, `frequency`, `noise`, `cnn`, `fusion`, `detector`, `server` (FastAPI), `cli` |
| `ai_detector/contracts/` | Contrat partagé : JSON Schema + types TypeScript (`ai_detector.d.ts`) |
| `training/` | `train.py`, `evaluate.py`, `export_onnx.py`, `calibrate_fusion.py`, `data.py`, `transforms.py`, `model.py` |
| `training/colab/ai_detector_colab.ipynb` | Notebook Google Colab complet (données → entraînement → export → calibration) |
| `model/detector.onnx` | Modèle ONNX **entraîné v1.0.0** (`efficientnet_b0`, 16 Mo, `trained=true`) + `model_card.json` + `calibration.json` |
| `docs/ai_detector.md` | Architecture détaillée, datasets, protocole d'entraînement, rapport de performances, contrat d'API |
| `tests/` | 48 tests pytest (modules, API, CLI, matérialisation des données, pipeline d'entraînement de bout en bout sur CPU) |

## État du modèle CNN

Le fichier versionné `model/detector.onnx` est le **modèle entraîné v1.0.0**
(`efficientnet_b0`, 4,01 M paramètres, 16 Mo, `trained=true`). Le moteur le charge et
**inclut le CNN dans la fusion** : `GET /health` renvoie `status: ok` et `python -m
ai_detector info` affiche `fusion_mode: full`. Le contrat ONNX est
`image[N,3,224,224] → logit[N,1]` ; métriques et empreinte SHA-256 dans `model_card.json`.

> **Performances (modèle v1.0.0, entraîné le 9 septembre 2026).** **AUC 0,978** en validation
> (générateurs jamais vus) et **AUC 0,954** sur AIGenImages2026 (19 modèles 2024–2025 jamais
> vus), robustesse AUC ≥ 0,94 sous JPEG / redimensionnement / flou. **Fusion calibrée** livrée
> (`model/calibration.json`, source `calib-2026-09-10`) : AUC hors-pli `fft` 0,920 · `noise`
> 0,928 · `cnn` 1,000 · **fusion full 1,000** (validation, 360 images). Détails dans
> `docs/ai_detector.md` § 9.3.1 et § 7. Fichiers binaires commités via Git normal (16 Mo, LFS
> non requis).
>
> Historique : tant qu'aucun modèle entraîné n'est présent, le dépôt embarque un **placeholder
> de signature** (`trained=false`) que le moteur exclut de la fusion (mode `handcrafted`,
> `status: degraded`) ; remplacer le fichier par l'export entraîné active le mode `full` sans
> modification de code.

## Entraînement (Colab ou poste GPU)

```bash
pip install -r requirements-training.txt
python -m training.data build-manifest --ai-dir DATA/ai --real-dir DATA/real --group-by-generator --out DATA/manifest.csv
python training/train.py --data-root DATA/manifest.csv --layout manifest --arch efficientnet_b0 --epochs 8 --amp
python training/evaluate.py --checkpoint runs/detector/best.pt --data-root DATA/manifest.csv --layout manifest --split val --robustness jpeg75 resize0.5
python training/export_onnx.py --checkpoint runs/detector/best.pt --out model/detector.onnx --version 1.0.0
python training/calibrate_fusion.py --data-root DATA/manifest.csv --layout manifest --split val --onnx model/detector.onnx --out model/calibration.json
```

## Tests

```bash
pip install -e ".[dev,training]"
pytest
```

## Configuration (variables d'environnement)

| Variable | Défaut | Rôle |
|---|---|---|
| `AI_DETECTOR_MODEL_PATH` | `model/detector.onnx` | Modèle ONNX |
| `AI_DETECTOR_CALIBRATION_PATH` | `model/calibration.json` | Paramètres de calibration (défauts intégrés si absent) |
| `AI_DETECTOR_HOST` / `AI_DETECTOR_PORT` | `127.0.0.1` / `32188` | Adresse d'écoute du serveur |
| `AI_DETECTOR_TOKEN` | — | Si défini, exige l'en-tête `x-ai-detector-token` |
| `AI_DETECTOR_THREADS` | `0` (auto) | Threads ONNX Runtime |

## Intégration dans ClickyX

Tous les fichiers de ce module vivent dans des chemins nouveaux (`ai_detector/`, `training/`,
`model/`, `docs/ai_detector.md`, `tests/*.py`, `pyproject.toml`, `requirements*.txt`) afin
d'être fusionnés sans conflit avec `feature/offline-engine` et `feature/local-ai-ui`.
Le seul fichier commun est `.gitignore`, dont les entrées Python sont ajoutées sans doublon.

Le script `scripts/integrate_into_clickyx.py` fait tout en une commande (clone si besoin,
branche `feature/ai-image-detector` créée depuis `master`, copie des chemins ci-dessus,
fusion du `.gitignore`, vérification que rien n'est ignoré par git, commit, push) :

```bash
# depuis la racine de ce dépôt, avec l'environnement Python activé
python scripts/integrate_into_clickyx.py --clone --push          # clone ClickyX dans ./clickyX
python scripts/integrate_into_clickyx.py --target ../clickyX --push   # clone déjà présent
python scripts/integrate_into_clickyx.py --target ../clickyX --dry-run  # voir sans modifier
```

Options utiles : `--run-tests` (lance pytest dans le dépôt cible avant de commiter),
`--no-commit`, `--allow-dirty`. Le script est idempotent : relancé après une mise à jour du
module, il ne commite que les différences.

Le contenu de ce README est repris dans `docs/ai_detector.md`, qui reste la référence.
