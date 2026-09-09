# ClickyX — Détecteur d'images générées par IA

*Module IA Forensics · branche `feature/ai-image-detector` · version 0.1.0 · contrat d'API v1.0*

---

## 1. Objectif

Estimer, **localement et hors ligne**, la probabilité qu'une image ait été produite par un
modèle génératif (diffusion latente, GAN, modèles propriétaires récents) plutôt que par un
appareil photographique. Le système doit :

* analyser des caractéristiques **invisibles à l'œil** (spectre, statistiques du bruit) ;
* produire une **estimation probabiliste** calibrée, jamais un verdict absolu ;
* rester **explicable** (chaque module expose ses indices et leur contribution) ;
* fonctionner sur CPU sous Windows, Linux et macOS via **ONNX Runtime**, sans service externe.

## 2. Hypothèse de recherche

| Photographie | Image générée |
|---|---|
| Photons → capteur (matrice de Bayer) → dématriçage → traitement → compression | Bruit latent → réseau (attention/convolutions) → décodeur (upsampling ×8 typiquement) → post-traitement |
| Bruit de grenaille (Poisson) dont la variance croît avec l'intensité | Pas de modèle physique de bruit ; résidu façonné par le décodeur |
| Traces de dématriçage de période 2 (canal vert) | Périodicités aux pas d'upsampling (2, 4, 8 px) |
| Spectre ≈ loi de puissance `P(f) ∝ 1/f^α` sans pics privilégiés | Pics spectraux aux fréquences rationnelles ; écart à la loi de puissance |

Deux images peuvent être visuellement identiques et différer par ces statistiques. **Aucune
signature n'est universelle** : les modèles évoluent, les post-traitements (redimensionnement,
JPEG, réseaux sociaux) effacent une partie des traces. D'où :

1. plusieurs familles d'indices **indépendantes** (physique + apprentissage) ;
2. une **fusion probabiliste** dont les poids sont **appris** sur données (recalibrables) ;
3. des **avertissements de fiabilité** explicites (image minuscule, CNN absent, peu de zones plates…).

## 3. Architecture

```
                 ┌──────────────────────────────────────────────────────────────┐
  image_path /   │  Prétraitement  (ai_detector/preprocessing.py)               │
  image_base64 ─►│  décodage · orientation EXIF · RGB float32 · fenêtre native  │
                 └───────┬──────────────────────┬──────────────────────┬─────────┘
                         ▼                      ▼                      ▼
             ┌───────────────────┐  ┌───────────────────┐  ┌────────────────────────┐
             │ 1. Fréquentiel    │  │ 2. Bruit résiduel  │  │ 3. CNN (ONNX Runtime)   │
             │ frequency.py      │  │ noise.py           │  │ cnn.py                  │
             │ FFT tuiles 256²   │  │ x − médiane 3×3    │  │ crops natifs 224² ×5    │
             │ loi de puissance  │  │ bruit de grenaille │  │ EfficientNet/ConvNeXt   │
             │ pics de grille    │  │ CFA · autocorr.    │  │ moyenne des logits      │
             │ → fft_score       │  │ → noise_score      │  │ → cnn_score             │
             └─────────┬─────────┘  └─────────┬─────────┘  └───────────┬────────────┘
                       └──────────────────────┼────────────────────────┘
                                              ▼
                              ┌──────────────────────────────┐
                              │ 4. Fusion (fusion.py)         │
                              │ z = b + Σ wᵢ·logit(sᵢ)        │
                              │ confidence = σ(z)             │
                              │ is_ai_generated = p ≥ 0,5     │
                              └──────────────────────────────┘
```

| Fichier | Rôle |
|---|---|
| `ai_detector/preprocessing.py` | Chargement robuste (chemin, bytes, base64, tableau), EXIF, fenêtrage/crops natifs, rembourrage symétrique |
| `ai_detector/frequency.py` | Spectres de puissance moyennés (Hann, tuiles 256×256), profil radial, détection de pics, `fft_score` |
| `ai_detector/noise.py` | Résidu médian, statistiques sur blocs plats, autocorrélation, CFA, blocking JPEG, `noise_score` |
| `ai_detector/cnn.py` | Session ONNX Runtime, lecture des `metadata_props`, stratégie de crops, `cnn_score` |
| `ai_detector/calibration.py` | Régression logistique caractéristiques → score ; chargement de `model/calibration.json` |
| `ai_detector/fusion.py` | Fusion en espace logit, modes `full` / `handcrafted` |
| `ai_detector/detector.py` | Orchestrateur `AiImageDetector` (charge le modèle une fois, chronomètre chaque étape) |
| `ai_detector/schemas.py` | **Source de vérité du contrat** (Pydantic) |
| `ai_detector/server.py`, `cli.py` | Serveur FastAPI (`/detect-ai-image`, `/health`, `/detect-ai-image/contract`) et CLI |
| `training/*.py` | Données, augmentations, modèles, entraînement, évaluation, export ONNX, calibration |
| `model/` | `detector.onnx`, `model_card.json`, `calibration.json` (optionnel) |

**Principe transversal : aucun redimensionnement avant analyse.** Les artefacts de synthèse
vivent à l'échelle du pixel ; un rééchantillonnage les détruit (voir § 9.3). Les modules
travaillent sur une fenêtre centrale ≤ 1024 px et des crops à résolution native.

## 4. Module 1 — Analyse fréquentielle (`fft_score`)

**Calcul.** L'image en luminance est découpée en jusqu'à 16 tuiles natives 256×256 ; chaque
tuile centrée est pondérée par une fenêtre de Hann 2D, puis `P = |FFT2|²`. Les spectres sont
**moyennés** : le contenu (variable d'une tuile à l'autre) s'atténue tandis que les
périodicités cohérentes de l'image s'additionnent (Corvi et al., 2023).

**Caractéristiques** (toutes exportées dans `details.frequency.features`) :

| Nom | Définition | Sens attendu |
|---|---|---|
| `slope` | Pente de `log P(f)` vs `log f` sur `f ∈ [0,02 ; 0,25]` cycle/px (loi de puissance) | Informatif après calibration |
| `hf_residual`, `hf_residual_abs` | Écart moyen (log) entre spectre observé et loi de puissance extrapolée sur `f ∈ [0,30 ; 0,49]` | Excès (GAN, netteté artificielle) ou déficit (décodeurs lisses) |
| `hf_energy_ratio` | Énergie `f > 0,25` / énergie `f > 0,02` | — |
| `peak_max_z` | Proéminence max d'un pic isolé : `(log P − médiane 7×7)` en z robuste (MAD), `f > 0,08` | ↑ synthèse |
| `peak_density` | Fraction de pixels du spectre avec `z > 4` | ↑ synthèse |
| `grid_peak_z` | Proéminence max aux fréquences `{½, ¼, ⅛}` cycle/px (axes + diagonales) — périodes 2/4/8 px des upsamplings | **↑ synthèse (indice le plus fort mesuré)** |
| `anisotropy` | `log(E_axes / E_diagonales)` sur `f ∈ [0,1 ; 0,5]` | Faible |
| `profile_roughness` | Écart-type de la dérivée seconde du profil radial (log) pour `f ≥ 0,1` | ↑ synthèse |

**Score.** `fft_score = σ(b + Σ wᵢ · clip((xᵢ − μᵢ)/σᵢ, ±4))`. Les `(μ, σ, w, b)` viennent de
`model/calibration.json` (ajusté par `training/calibrate_fusion.py`) ou des défauts de
`ai_detector/calibration_defaults.py`.

Références : Durall et al. 2020 (*Watch your Up-Convolution*), Zhang et al. 2019
(*Detecting and Simulating Artifacts in GAN Fake Images*), Corvi et al. 2023 (*Intriguing
properties of synthetic images: from GANs to diffusion models*).

## 5. Module 2 — Bruit résiduel (`noise_score`)

**Résidu.** `r_c = x_c − médiane₃ₓ₃(x_c)` par canal (débruiteur rapide, sans dépendance).
Les statistiques de bruit sont mesurées sur les **blocs plats** 16×16 (écart-type du contenu
débruité < 0,02, intensité moyenne dans ]0,05 ; 0,95[) pour ne pas confondre bruit et texture.

| Nom | Définition | Sens attendu |
|---|---|---|
| `flat_noise_std` | Médiane de l'écart-type du résidu sur blocs plats | Images trop « propres » ↑ synthèse (faible) |
| `shot_noise_corr` | Corrélation de Spearman (intensité moyenne, variance du résidu) sur blocs plats | Positive pour un capteur (bruit de grenaille) → ↓ synthèse |
| `flat_kurtosis` | `log1p(kurtosis)` du résidu sur blocs plats | Informatif après calibration |
| `residual_ac_peak` | Proéminence de l'autocorrélation normalisée aux lags 2/4/8/16 : `ac[l] − ½(ac[l−1] + ac[l+1])` | ↑ synthèse (grille de décodeur) |
| `residual_spec_peak_z`, `residual_grid_peak_z` | Mêmes détecteurs de pics que § 4, appliqués au **spectre du résidu** | **↑ synthèse (fort)** |
| `cfa_strength` | Périodicité de période 2 de `|∇²G|` sommé sur les diagonales (Gallagher & Chen, 2008) : trace de dématriçage | ↓ synthèse si présent |
| `jpeg_grid_strength` | Log-ratio des gradients aux frontières de blocs 8×8 vs ailleurs | Contexte (explique des pics de grille) |
| `channel_residual_corr` | Corrélation des résidus R–G et G–B | Informatif après calibration |
| `flat_block_fraction` | Fraction de blocs plats | Contexte / fiabilité |

Si moins de 20 blocs plats sont disponibles, les statistiques correspondantes valent `null`,
sont **neutres** dans la régression et l'avertissement `peu_de_zones_plates:…` est émis.

Références : Popescu & Farid 2005 (CFA), Gallagher & Chen 2008, Lukáš/Fridrich/Goljan 2006
(PRNU), Corvi et al. 2023 (autocorrélation du résidu des modèles de diffusion).

## 6. Module 3 — Détecteur Deep Learning (`cnn_score`)

### 6.1 Architectures étudiées

| Architecture | Paramètres | ONNX fp32 (mesuré) | Latence CPU, 5 crops 224² (mesurée, Xeon 4 vCPU AVX-512) | Commentaire |
|---|---|---|---|---|
| **EfficientNet-B0** (défaut) | 5,3 M | 16 Mo | ≈ 20 ms (50–100 ms attendus sur portable) | Meilleur compromis rapidité/robustesse pour un usage desktop |
| EfficientNetV2-S | 21 M | ≈ 80 Mo | ≈ 3× B0 | Plus précis, coût supérieur |
| ConvNeXt-Tiny | 28 M | 111 Mo | ≈ 100 ms | Meilleure généralisation inter-générateurs dans la littérature (GenImage, Community Forensics) |
| ResNet-50 | 25 M | ≈ 100 Mo | ≈ 4× B0 | Référence historique (CNNDetection) |

Tous sont **entièrement convolutifs** (global pooling) : le choix de la taille de crop est
libre à l'entraînement et conservé à l'inférence via les métadonnées ONNX. Les architectures
à base de CLIP (UnivFD, RINE) généralisent très bien mais pèsent > 300 Mo et exigent un
redimensionnement à 224 px qui détruit les artefacts de bas niveau ; elles sont écartées
pour un moteur local léger, mais `training/model.py` reste extensible (`ARCH_PRESETS`).

### 6.2 Stratégie d'inférence

* Crops **224×224 à résolution native** : centre + 4 coins (≤ 5), aucun redimensionnement ;
  une image plus petite que 224 px est rembourrée par réflexion (avertissement).
* Normalisation ImageNet, batch unique, **moyenne des logits** puis sigmoïde.
* Fournisseur `CPUExecutionProvider` (déterministe, portable) ; `AI_DETECTOR_THREADS` limite
  le parallélisme.

### 6.3 Contrat du fichier ONNX

```
entrée   image : float32 [batch, 3, S, S]        (S = metadata input_size, défaut 224)
sortie   logit : float32 [batch, 1]              σ(logit) = P(générée par IA)
metadata_props : arch, input_size, mean, std, output=logit, trained (true/false),
                 version, created_at, framework, opset, git_commit, training_data, val_auc
```

`training/export_onnx.py` écrit ces métadonnées, vérifie la parité PyTorch ↔ ONNX Runtime
(`max |Δ| < 2·10⁻³`) et génère `model/model_card.json` (SHA-256, métriques, configuration).

### 6.4 Placeholder versionné

`model/detector.onnx` est actuellement un **placeholder de signature** (`arch=tiny`,
`trained=false`, tête à zéro ⇒ logit 0 ⇒ `cnn_score = 0,5`). Le moteur le charge (ce qui
valide le chemin ONNX Runtime de bout en bout) mais **l'exclut de la fusion** ; `GET /health`
renvoie `status: degraded`, `fusion_mode: handcrafted`. Remplacer le fichier par un export
entraîné suffit à activer le mode `full`, sans modification de code.

## 7. Module 4 — Fusion et calibration

Fusion tardive en espace logit :

\[
z = b + w_{\text{fft}}\,\mathrm{logit}(s_{\text{fft}}) + w_{\text{noise}}\,\mathrm{logit}(s_{\text{noise}}) + w_{\text{cnn}}\,\mathrm{logit}(s_{\text{cnn}}), \qquad
\texttt{confidence} = \sigma(z)
\]

* **Mode `full`** (CNN chargé **et** `trained=true`) : trois termes.
* **Mode `handcrafted`** (sinon) : jeu de poids distinct sur `fft` et `noise` uniquement.
* `is_ai_generated = confidence ≥ threshold` (0,5 par défaut) ; `details.fusion.verdict_confidence = max(p, 1−p)`.

`model/calibration.json` (schéma v1) regroupe les trois régressions (`frequency`, `noise`,
`fusion.full`, `fusion.handcrafted`) et leur traçabilité (`source`, `version`, AUC hors-pli,
date, commit). `training/calibrate_fusion.py` :

1. extrait les caractéristiques des deux modules et le `cnn_score` sur un jeu de validation étiqueté ;
2. ajuste chaque module par régression logistique L2 (classes rééquilibrées, L-BFGS) ;
3. ajuste la fusion sur des **scores hors-pli** (validation croisée K=5) afin de ne pas surestimer sa performance ;
4. écrit `calibration.json` + `calibration_report.json` (AUC univariées, AUC par module, métriques de fusion).

Sans fichier, les défauts de `calibration_defaults.py` s'appliquent : ils encodent le sens
physique des indices avec des poids modérés et des statistiques de normalisation observées
sur le jeu de contrôle du § 9.

## 8. Datasets

### 8.1 Étude comparative

| Dataset | Année | Volume | Générateurs | Réelles | Points forts | Limites |
|---|---|---|---|---|---|---|
| **GenImage** (Zhu et al., NeurIPS 2023) | 2023 | 1,35 M IA + 1,33 M réelles | SD 1.4/1.5, Midjourney v5, ADM, GLIDE, Wukong, VQDM, BigGAN | ImageNet (JPEG) | Volume, protocole *cross-generator* et *degraded* standard | Générateurs 2023 ; réelles = ImageNet (basse résolution, JPEG) ⇒ raccourci « JPEG = réel » |
| **Community Forensics** (Park & Owens, CVPR 2025) | 2024–25 | 2,7 M IA de **4 803** modèles (`Small` : 278 K + 278 K réelles) | Milliers de LDM open-source + modèles commerciaux | LAION, COCO, FFHQ, VISION, Landscapes HQ… | Diversité de générateurs inégalée ⇒ généralisation démontrée | 1,1 To (complet) ; `Small` sur-représente les dérivés Stable Diffusion |
| **AIGenImages2026** (WildFC, mever-team, 2026) | 2025 | 5 439 IA (4 880 train / 559 test) | **19 modèles 2024–2025** : SDXL, FLUX.1 pro, SD 3.5, Reve, HiDream, GPT-Image 1/1.5, Ideogram 3, Midjourney v7, Imagen 4, Gemini 2.5/3 Pro, Firefly 5, FLUX.2 (dev/pro/max), Z-Image Turbo, Seedream 4.5 | aucune (à apparier) | Générateurs les plus récents, prompts réalistes, métadonnées chronologiques | Petit ; IA seulement ; ≤ 305 images/modèle |
| WildFC (même papier) | 2025 | 2 884 IA *in-the-wild* | Inconnus (fact-checking) | — | Distribution réelle des réseaux sociaux (73 % JPEG, WEBP) | Faible volume, étiquetage faible |
| Chameleon (Yan et al., 2024) | 2024 | 26 K | Images IA « indiscernables » validées par humains | Oui | Test de difficulté maximale | Test uniquement |
| Synthbuster (Bammey, 2023) | 2023 | 9 K | DALL·E 2/3, MJ v5, SD 1/2/XL, Firefly, GLIDE | RAISE (RAW) | Réelles non compressées de qualité | Petit |

### 8.2 Choix retenu

* **Entraînement principal** : *Community Forensics-Small* (réelles + IA de licences
  redistribuables, milliers de générateurs) **complété** par un sous-ensemble équilibré de
  *GenImage* (couverture GAN + diffusion pixel-space : ADM, GLIDE, BigGAN, VQDM) si l'espace
  Colab le permet. La diversité de générateurs est le premier facteur de généralisation
  (Park & Owens, 2025 : la performance croît avec le nombre de modèles sources).
* **Validation (sélection de modèle)** : générateurs **entiers jamais vus** (`--group-by-generator`).
* **Test hors distribution** : *AIGenImages2026* (modèles 2025) appariées à des réelles
  réservées ; puis *Chameleon*/*WildFC* si disponibles. Les 4 880 images d'entraînement
  d'AIGenImages2026 servent, en dernier lieu, à un *fine-tuning* court avec tampon de rejeu
  (5 % des données antérieures, cf. WildFC) pour ne pas oublier les générateurs anciens.

### 8.3 Équilibrage

* **Classes** : `WeightedRandomSampler` → 50 % réelles / 50 % IA à chaque époque.
* **Générateurs** : à l'intérieur de la classe IA, chaque générateur pèse autant
  (`training/data.py::balanced_sampler_weights`) : un modèle sur-représenté (SD 1.5) ne
  domine pas l'apprentissage.
* **Sources réelles** : idem par source (COCO, FFHQ, VISION…), pour ne pas apprendre
  « visage = réel ».
* **Formats** : les augmentations JPEG/rééchantillonnage s'appliquent aux **deux** classes
  afin de neutraliser les raccourcis de conteneur (PNG ⇒ IA, JPEG ⇒ réel), biais bien
  documenté sur GenImage.

### 8.4 Contre le surapprentissage

| Levier | Implémentation |
|---|---|
| Pré-entraînement ImageNet, backbone gelé 1 époque puis LR différenciés (tête ×10) | `train.py --freeze-backbone-epochs 1 --head-lr-mult 10` |
| Weight decay 0,05, dropout 0,2, label smoothing 0,02 | `--weight-decay --drop-rate --label-smoothing` |
| Augmentations de dégradation : JPEG q 55–100 (p 0,5), rééchantillonnage ×0,5–1,5 (p 0,3), flou σ 0,3–1,2 (p 0,1), flip | `training/transforms.py::AugmentConfig` |
| Crops aléatoires natifs (pas de *resize* global) | `RandomCropNative` |
| EMA des poids (0,999), cosinus + warmup | `ModelEma`, `cosine_with_warmup` |
| Sélection sur AUC de **générateurs jamais vus** + early stopping (patience 3) | `--group-by-generator`, `--early-stopping-patience` |
| Évaluation systématique de robustesse et hors distribution | `evaluate.py --robustness`, manifest AIGenImages2026 |

## 9. Protocole d'entraînement et d'évaluation

### 9.1 Étapes (notebook `training/colab/ai_detector_colab.ipynb`)

1. **Données** : matérialisation Community Forensics-Small par `training/commfor.py` (liste fixe de fichiers parquet couvrant diffusion latente/GAN/diffusion pixel et COCO/VISION/Landscapes HQ/FFHQ, un fichier à la fois dans un sous-processus, images écrites telles quelles → dossiers `real/<source>/`, `ai/<modèle>/`), téléchargement AIGenImages2026, construction des manifests (`training/data.py build-manifest`). Le streaming `datasets` est à proscrire : chaque parquet est un row group unique de ~3 000 images (jusqu'à 4 Go) et les fichiers sont triés par classe.
2. **Entraînement** : `train.py --arch efficientnet_b0 --epochs 8 --batch-size 64 --lr 3e-4 --amp --freeze-backbone-epochs 1 --ema 0.999` (≈ 25–40 min sur T4 pour 12 K images ; adapter `--max-train-samples`).
3. **Évaluation** : `evaluate.py --checkpoint … --robustness jpeg75 jpeg50 resize0.5 blur1.0` sur la validation, puis sur AIGenImages2026.
4. **Export** : `export_onnx.py --checkpoint best.pt --out model/detector.onnx --version 1.0.0`.
5. **Calibration** : `calibrate_fusion.py --onnx model/detector.onnx --out model/calibration.json`.
6. **Pipeline complet** : `evaluate.py --full-pipeline` (chiffres finaux tels que servis par l'API).
7. **Vérification** : `python -m ai_detector info` ⇒ `fusion_mode: full`.

### 9.2 Métriques (`training/metrics.py`)

Accuracy et *balanced accuracy* au seuil 0,5, **AUC**, *average precision*, TPR à FPR 1 % et
5 %, **EER**, ECE (calibration), Brier, matrice de confusion ; ventilation **par générateur**
(chaque générateur vs toutes les réelles) et **par condition de dégradation**.

### 9.3 Rapport de performances

> **État au 9 septembre 2026.** Le CNN versionné est un placeholder : aucun chiffre de CNN
> entraîné n'est encore disponible. Les résultats ci-dessous concernent les **indices
> physiques seuls** (mode `handcrafted`) et servent de **référence basse**. Ils seront
> remplacés par la sortie du notebook Colab (§ 9.1, étapes 3 et 6).

**Jeu de contrôle.** 170 images de *Community Forensics-Small* : 55 photographies FFHQ
(1024², recadrées à 512² natif) vs 115 images de 7 modèles de diffusion latente (512², PNG).
Jeu volontairement petit et **non représentatif** (visages alignés vs SD) : il valide le sens
et la robustesse des indices, pas la performance finale.

*Pipeline complet, paramètres par défaut (aucun apprentissage sur ce jeu) :*

| Condition | n | Accuracy | Bal. acc. | AUC | AP | TPR (IA) | TNR (réel) | EER |
|---|---|---|---|---|---|---|---|---|
| Images natives | 170 | 0,829 | 0,846 | **0,923** | 0,964 | 0,800 | 0,891 | 0,147 |
| JPEG q75 | 170 | 0,624 | 0,689 | 0,797 | 0,892 | 0,504 | 0,873 | 0,289 |
| Redimensionnement ×0,75 | 170 | 0,441 | 0,587 | 0,778 | 0,893 | 0,174 | 1,000 | 0,329 |

*Calibration ajustée sur ce même jeu (AUC hors-pli, 5 plis) :* `fft_score` 0,919 ·
`noise_score` 0,957 · fusion `handcrafted` **0,976** (accuracy 0,924).

*AUC univariée des indices (1 = plus élevé chez les IA) :*

| Indice | AUC | Indice | AUC |
|---|---|---|---|
| `grid_peak_z` (spectre image) | **0,906** | `residual_grid_peak_z` (spectre du résidu) | **0,892** |
| `flat_kurtosis` | 0,776 | `cfa_strength` | 0,725 |
| `profile_roughness` | 0,695 | `residual_spec_peak_z` | 0,699 |
| `peak_max_z` | 0,672 | `anisotropy` | 0,648 |
| `shot_noise_corr` | 0,420 (plus élevé chez les réelles) | `residual_ac_peak` | 0,323 (FFHQ rééchantillonné) |

*Robustesse de l'indice principal (`grid_peak_z`, moyenne ± é.-t.) :*

| Traitement | Photos FFHQ | Images de diffusion latente |
|---|---|---|
| Natif (PNG) | 2,23 ± 0,43 | 3,79 ± 1,31 (68 % > 3) |
| JPEG q95 / q90 | 2,32 ± 0,46 / — | — / 3,33 ± 1,29 (46 % > 3) |
| JPEG q75 | 2,48 ± 0,62 | 2,93 ± 1,14 (32 % > 3) |
| JPEG q60 | 2,59 ± 0,66 | — |
| Redimensionnement ×0,75 | — | 2,09 |

Lecture : la compression JPEG d'une photo n'imite que faiblement la grille (2,2 → 2,6 à
q60), mais elle **efface progressivement** la grille des images synthétiques, et un simple
rééchantillonnage la supprime totalement. C'est la justification quantitative du module CNN
(entraîné avec ces dégradations) et de la fusion probabiliste : les indices physiques sont
précis quand ils sont présents, mais pas exhaustifs.

**Latence mesurée** (Xeon 4 vCPU, sans GPU) : ≈ 190 ms par image 512² et 230–340 ms par
image 768² pour les modules physiques ; EfficientNet-B0 ONNX ajoute ≈ 20 ms (5 crops),
ConvNeXt-Tiny ≈ 100 ms.

**Références de la littérature** (ordres de grandeur attendus après entraînement) :
ResNet-50 entraîné sur SD 1.4 (GenImage) : 99 % sur le même générateur, 60–80 % en
*cross-generator* ; entraînement sur Community Forensics : AUC moyenne ≈ 0,95–0,98 sur
générateurs inconnus ; détecteurs statiques évalués sur AIGenImages2026/in-the-wild
(WildFC 2026) : 50–70 % d'accuracy, 80–92 % après adaptation continue. Objectif hackathon :
**AUC > 0,95 en validation « générateurs jamais vus »** et **> 0,85 sur AIGenImages2026**.

### 9.4 Rapport à compléter après entraînement Colab

```
Modèle : efficientnet_b0 · version 1.0.0 · données : …
Validation (générateurs jamais vus) : AUC … · bal. acc … · EER …
AIGenImages2026 (test) : AUC … · accuracy … · pire générateur : … (AUC …)
Robustesse : JPEG75 AUC … · JPEG50 … · resize0.5 … · blur1.0 …
Pipeline complet (fusion full) : AUC … · accuracy … · ECE …
```

## 10. Contrat d'API officiel

### 10.1 Endpoint

```
POST http://127.0.0.1:32188/detect-ai-image
Content-Type: application/json
[x-ai-detector-token: <jeton>]        (seulement si AI_DETECTOR_TOKEN est défini)
```

**Entrée (v1)** :

```json
{ "image_path": "C:/Users/alex/Pictures/photo.jpg" }
```

**Sortie (v1 — champs figés)** :

```json
{
  "is_ai_generated": true,
  "confidence": 0.92,
  "fft_score": 0.81,
  "noise_score": 0.73,
  "cnn_score": 0.95
}
```

| Champ | Type | Sémantique |
|---|---|---|
| `is_ai_generated` | bool | `confidence ≥ 0,5` |
| `confidence` | float [0,1] | **Probabilité que l'image soit générée par IA** (0,08 = très probablement une photo). Pour afficher une confiance dans le verdict : `max(confidence, 1 − confidence)` (fourni dans `details.fusion.verdict_confidence`) |
| `fft_score` | float [0,1] | Score du module fréquentiel |
| `noise_score` | float [0,1] | Score du module bruit résiduel |
| `cnn_score` | float [0,1] | Score du CNN ; **0,5 (neutre)** si le modèle est absent ou non entraîné |

### 10.2 Extensions additives (v1.1, optionnelles — ignorables par un client v1)

Entrée : `image_base64` (alternative à `image_path`, data-URL acceptée), `include_details`
(bool, défaut `true`). Sortie : `contract_version`, `model_version`, `warnings[]` (codes
stables : `cnn_indisponible:…`, `cnn_non_entraine:…`, `image_tres_petite:…`,
`peu_de_zones_plates:…`, `image_plus_petite_que_l_entree_cnn:…`), `details`
(caractéristiques et contributions par module, informations image, mode et poids de fusion,
`verdict_confidence`, temps par étape, version de calibration).

### 10.3 Erreurs

| HTTP | `error.code` | Cas |
|---|---|---|
| 404 | `image_not_found` | Chemin inexistant |
| 415 | `unsupported_image` | Fichier illisible / format non supporté / base64 invalide |
| 400 | `permission_denied` | Lecture refusée |
| 401 | `unauthorized` | Jeton manquant ou invalide |
| 422 | (validation FastAPI) | Ni `image_path` ni `image_base64`, ou les deux |

Format : `{"error": {"code": "...", "message": "..."}}`.

### 10.4 Endpoints annexes

* `GET /health` → `status` (`ok` si CNN entraîné chargé, sinon `degraded`), `version`, `model{loaded, trained, arch, version, input_size, providers}`, `calibration_version`, `fusion_mode`.
* `GET /detect-ai-image/contract` → JSON Schema (requête, réponse, erreur).
* `GET /docs` → Swagger UI (développement).

### 10.5 Politique d'évolution

1. Les cinq champs v1 ne changent **jamais** de nom, de type ni de sémantique.
2. Toute évolution est **additive** (nouveaux champs optionnels), documentée ici et dans
   `ai_detector/schemas.py` (`CONTRACT_VERSION`), et régénérée dans
   `ai_detector/contracts/detect_ai_image.schema.json` (`python -m ai_detector.contracts`)
   et `ai_detector/contracts/ai_detector.d.ts`.
3. Un changement incompatible imposerait un nouveau chemin (`/v2/detect-ai-image`) en
   conservant `/detect-ai-image`.

### 10.6 Intégration

**Interface (Honorat).** Types prêts à l'emploi : `ai_detector/contracts/ai_detector.d.ts`.
Afficher `confidence` (jauge 0–100 %), les trois sous-scores (barres), les `warnings`
(fiabilité) et, en mode détaillé, `details.frequency/noise.contributions` (explication).
Historique : conserver `contract_version`, `model_version`, `details.calibration_version`.

**Intégration locale (Tybiane).** Deux options, sans HTTP ou avec :

* *Sidecar HTTP* : lancer `python -m ai_detector serve --port 32188` (ou l'exécutable
  PyInstaller) au démarrage, sonder `GET /health`, appeler `POST /detect-ai-image`.
* *Processus à la demande* : `python -m ai_detector detect <chemin> --json` imprime
  exactement le contrat v1 sur stdout (code 2 si image introuvable/illisible).

Variables d'environnement : voir README (`AI_DETECTOR_MODEL_PATH`, `AI_DETECTOR_PORT`,
`AI_DETECTOR_TOKEN`…). Le serveur n'écoute que sur la boucle locale.

## 11. Exécution locale hors ligne

* Dépendances d'inférence uniquement : `numpy`, `scipy`, `pillow`, `onnxruntime`, `fastapi`,
  `uvicorn`, `pydantic` (aucun PyTorch au runtime). ONNX Runtime fournit des roues CPU pour
  Windows x64, Linux x64/arm64, macOS x64/arm64.
* Aucun accès réseau : modèle et calibration sont lus sur disque ; pas de télémétrie.
* Empaquetage : `pyinstaller --onefile -n clickyx-ai-detector ai_detector/__main__.py
  --add-data model:model` produit un binaire par OS (à intégrer aux scripts de release
  ClickyX, hors périmètre de cette branche).
* Tests : `pytest` (44 tests : modules, API, CLI, entraînement de bout en bout sur CPU).

## 12. Limites connues et pistes

* **Post-traitements** : redimensionnement, recompression forte et captures d'écran
  effacent les indices physiques ; la robustesse repose alors sur le CNN entraîné avec ces
  dégradations. Signaler à l'utilisateur les images de faible résolution (`warnings`).
* **Dérive des générateurs** : recalibrer et ré-entraîner périodiquement (AIGenImages2026,
  WildFC) avec tampon de rejeu ; les scripts et le notebook sont conçus pour cette boucle.
* **Attaques adverses** : hors périmètre ; un attaquant informé peut masquer les pics
  spectraux (bruit additif, rééchantillonnage).
* **Images retouchées / hybrides** (inpainting local) : le score est global ; une
  localisation par tuiles (`details` par tuile, carte de chaleur) est une extension
  naturelle du pipeline actuel.
* **Faux positifs** possibles sur : rendus 3D, illustrations numériques, scans, images
  fortement agrandies. Ce sont des images « synthétiques » au sens physique, non des photos.

## 13. Références

* Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*, NeurIPS 2023.
* Park & Owens, *Community Forensics: Using Thousands of Generators to Train Fake Image Detectors*, CVPR 2025.
* mever-team, *Automated In-the-Wild Data Collection for Continual AI Generated Image Detection* (WildFC / AIGenImages2026), arXiv 2605.02567, 2026.
* Corvi, Cozzolino, Zingarini, Poggi, Nagano, Verdoliva, *On the detection of synthetic images generated by diffusion models* (ICASSP 2023) et *Intriguing properties of synthetic images: from GANs to diffusion models* (CVPRW 2023).
* Durall, Keuper, Keuper, *Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions*, CVPR 2020.
* Zhang, Karaman, Chang, *Detecting and Simulating Artifacts in GAN Fake Images*, WIFS 2019.
* Wang et al., *CNN-generated images are surprisingly easy to spot… for now*, CVPR 2020.
* Gallagher & Chen, *Image authentication by detecting traces of demosaicing*, CVPRW 2008 ; Popescu & Farid, *Exposing digital forgeries in color filter array interpolated images*, IEEE TSP 2005.
* Lukáš, Fridrich, Goljan, *Digital camera identification from sensor pattern noise*, IEEE TIFS 2006.
* Ojha, Li, Lee, *Towards Universal Fake Image Detectors that Generalize Across Generative Models*, CVPR 2023 ; Koutlis & Papadopoulos, *RINE*, ECCV 2024.
* Yan et al., *A Sanity Check for AI-generated Image Detection* (Chameleon), 2024 ; Bammey, *Synthbuster*, IEEE OJSP 2023.
