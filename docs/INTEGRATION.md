# ClickyX — Intégration des trois briques

Ce document décrit comment les trois contributions du hackathon s'assemblent en une seule
application ClickyX fonctionnelle. Il s'adresse à l'équipe (intégration, démo, dépannage) et
complète les documentations spécialisées de chaque brique.

## 1. Vue d'ensemble

ClickyX combine trois modules développés en parallèle, puis fusionnés sur une branche
d'intégration :

| Brique | Auteur·e | Rôle | Documentation |
|---|---|---|---|
| **Interface locale (UI)** | Honorat | Écrans React « Detect / Results / History / Settings / Help », indicateur hors ligne, client d'API | [`src/ui/docs/guide-utilisateur.md`](../src/ui/docs/guide-utilisateur.md) |
| **Moteur hors ligne** | Tybiane | Bridge Rust (Ollama / Whisper / TTS système), statut de connectivité, gestion des modèles locaux | [`docs/BRIDGE_API.md`](BRIDGE_API.md) |
| **Détecteur IA Forensics** | Alexandre | Moteur Python (FFT + bruit + CNN ONNX), API `POST /detect-ai-image`, modèle entraîné | [`docs/ai_detector.md`](ai_detector.md) |

Les trois s'exécutent **localement et hors ligne**, sur CPU, sous Windows / Linux / macOS.

## 2. Schéma d'assemblage

L'UI parle à **deux services locaux distincts** :

- le **bridge** (`127.0.0.1:32123`) via l'IPC Tauri (`invoke`) pour l'état hors ligne et les modèles ;
- le **détecteur** (`127.0.0.1:32188`) via HTTP (`POST /detect-ai-image`) pour l'analyse d'images.

```
React (src/) ── onglet "AI Detector" ─▶ LocalAiView (src/ui/local-ai/, Honorat)
   │
   ├── invoke() Tauri ─▶ Moteur hors ligne (Tybiane)  src-tauri/src/offline/  bridge 127.0.0.1:32123
   │
   └── fetch() HTTP ───▶ Détecteur IA (Alexandre)      127.0.0.1:32188  POST /detect-ai-image
```

## 3. Contrats et points de raccordement

Les contrats étaient déjà alignés entre les branches ; l'intégration n'a demandé aucune
réécriture de code métier.

### 3.1 Détecteur IA <-> UI

| Élément | Détecteur (Python) | UI (TypeScript) | État |
|---|---|---|---|
| Base URL | `127.0.0.1:32188` (`ai_detector/config.py`) | `AI_DETECTOR_DEFAULT_BASE_URL` (`constants.ts`) | identiques |
| En-tête d'auth | `x-ai-detector-token` (`server.py`) | `AI_DETECTOR_TOKEN_HEADER` | identiques |
| Réponse détection | `DetectAiImageResponse` (Pydantic, `schemas.py`) | `DetectAiImageResponse` (`types.ts`) | champ pour champ |
| Santé | `GET /health` | `getDetectorHealth()` | status / fusion_mode / model |
| Erreurs | `{"error": {"code", "message"}}` | `parseDetectorError()` | identiques |

Le contrat officiel (5 champs : `is_ai_generated`, `confidence`, `fft_score`, `noise_score`,
`cnn_score`) est la source de vérité côté détecteur (`ai_detector/schemas.py`) et est répliqué
en JSON Schema + `.d.ts` dans `ai_detector/contracts/`.

### 3.2 Moteur hors ligne <-> UI

L'UI appelle d'abord les commandes Tauri (`offline_status`, `offline_refresh`,
`get_offline_config`, `list_local_providers`, `list_local_models`) ; en repli, elle interroge
le bridge HTTP `http://127.0.0.1:32123/offline/status`, puis un dernier repli navigateur.
Voir [`docs/BRIDGE_API.md`](BRIDGE_API.md).

## 4. Câblage effectué à l'intégration

La seule modification de code apportée par l'intégration relie l'écran de Honorat à la
navigation principale, **sans toucher au code métier** des briques :

- `src/context/AppContext.tsx` : ajout de l'onglet `"detect"` au type `Tab`.
- `src/App.tsx` : entrée « AI Detector » dans `TABS`, import paresseux de `LocalAiView`,
  et rendu dans le `switch(activeTab)`.

`LocalAiView` (`src/views/LocalAiView.tsx`) encapsule déjà les sous-écrans Detect / Results /
History / Settings / Help fournis par Honorat.

## 5. Lancer l'application intégrée en local

> **Chaque commande se lance séparément, sur sa propre ligne.** « puis » / « ensuite » dans ce
> guide sont des mots de liaison, pas des commandes — ne les tapez pas.

### 5.0 Prérequis et pièges courants

- **Se placer sur la branche d'intégration** avant tout :
  `git fetch origin`, puis `git checkout integration`, puis `git pull`.
- **Node.js >= 18** et **npm** pour l'interface.
- **Rust (stable) + toolchain Tauri** uniquement pour `npm run tauri dev` (application de bureau).
  Vérifier avec `rustc --version`. Si Rust n'est pas installé, utiliser `npm run dev` (interface
  web seule) — c'est le plus rapide pour une démo.
- **Python 3.10+** pour le détecteur.
- **La première compilation Tauri est longue** (plusieurs minutes) : c'est normal, elle compile
  le code Rust. Les lancements suivants sont rapides.
- **Deux terminaux distincts** : un pour le détecteur (§ 5.1), un pour l'application (§ 5.2).
  Laisser les deux ouverts pendant l'utilisation.

Erreurs fréquentes :

| Message | Cause | Correctif |
|---|---|---|
| `npm error 404 ... GET .../puis` | « puis » tapé comme une commande | Lancer les commandes une par une |
| `Cannot reach the local service` dans l'UI | détecteur non démarré | Lancer `python -m ai_detector serve` (§ 5.1) |
| `.venv\Scripts\activate` introuvable | venv pas encore créé | `python -m venv .venv` d'abord |
| `rustc: command not found` / erreur Tauri | Rust absent | Installer Rust, ou utiliser `npm run dev` |

### 5.1 Détecteur IA (terminal 1)

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate   |   Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m ai_detector serve                 # http://127.0.0.1:32188
```

Vérifier : `python -m ai_detector info` doit afficher `trained: true` et `fusion_mode: full`.

### 5.2 Application ClickyX (terminal 2)

```bash
npm install
# attendre la fin de l'installation, PUIS lancer l'une des deux :
npm run tauri dev        # application de bureau complète (bridge inclus) — nécessite Rust
npm run dev              # interface web seule (rapide, sans bridge natif) — pas besoin de Rust
```

Sur Windows (PowerShell), taper chaque commande sur sa propre ligne, l'une après l'autre.

Dans l'application : onglet **AI Detector**, déposer une image, lancer l'analyse. Le verdict
et les trois scores (FFT / bruit / CNN) s'affichent via les composants `VerdictBadge` / `ScoreBar`.

### 5.3 Configuration

- URL du détecteur et jeton : Settings -> **AI Detection** (par défaut `127.0.0.1:32188`, jeton vide).
- Mode hors ligne, URLs Ollama / Whisper : Settings -> **Offline**.

## 6. Vérifications automatisées

| Suite | Commande | Périmètre |
|---|---|---|
| Frontend (Vitest) | `npm run test` | UI locale + services (dont client détecteur) |
| Type-check | `npx tsc --noEmit` | cohérence TypeScript de bout en bout |
| Build | `npm run build` | bundle de production (inclut `LocalAiView`) |
| Détecteur (pytest) | `python -m pytest` | modules, API, CLI, contrat, entraînement |

## 7. Notes de compatibilité

- Le détecteur et le bridge écoutent sur **loopback uniquement** ; aucun port n'est exposé au réseau.
- Le modèle entraîné `model/detector.onnx` (16 Mo) est versionné via Git normal (LFS non requis).
- Si le détecteur est absent ou arrêté, l'UI affiche un état « injoignable » sans planter
  (`getDetectorHealth()` renvoie `status: "unreachable"`).
- Si le CNN entraîné est absent, le détecteur bascule en mode `handcrafted` (FFT + bruit) et
  `GET /health` renvoie `status: degraded` — l'UI reste fonctionnelle.
