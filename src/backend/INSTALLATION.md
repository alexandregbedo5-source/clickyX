# ClickyX — Installation hors ligne

Vue d’ensemble de ce qui a été ajouté : [MOTEUR_HORS_LIGNE.md](./MOTEUR_HORS_LIGNE.md).

Ce guide installe ClickyX pour qu’il **démarre et fonctionne sans Internet**
(Wi-Fi coupé, Ethernet débranché, pare-feu bloquant toute connexion sortante).

Les téléchargements ci-dessous se font **une seule fois, en ligne**.
Ensuite, l’application n’a plus besoin du WAN.

## 1. Prérequis (une fois, en ligne)

- Node.js 22+
- Rust stable (1.77+)
- npm
- [Ollama](https://ollama.com) (LLM local)
- Optionnel : [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (STT local)

### Linux (Ubuntu 22.04+)

```sh
sudo apt-get update
sudo apt-get install -y \
  libwebkit2gtk-4.1-dev \
  libappindicator3-dev \
  librsvg2-dev \
  patchelf \
  libxdo-dev \
  libasound2-dev \
  libpulse-dev \
  libspeechd-dev
```

### macOS

```sh
xcode-select --install
```

### Windows

- Visual Studio C++ Build Tools
- WebView2 (Windows 10 1803+ / Windows 11)

## 2. Dépendances du projet (une fois, en ligne)

```sh
git clone https://github.com/alexandregbedo5-source/clickyX.git
cd clickyX
git checkout feature/offline-engine
npm ci
```

Après `npm ci`, le dossier `node_modules/` est local. Plus besoin du registre npm.

## 3. Modèle LLM local — Ollama

```sh
# Installer Ollama depuis https://ollama.com puis :
ollama serve
ollama pull llama3.2        # recommandé (~2 Go)
# alternatives plus légères
ollama pull llama3.2:1b
ollama pull qwen2.5:3b
ollama pull mistral
```

Vérification hors ligne :

```sh
# Couper le réseau, puis :
ollama list
curl http://127.0.0.1:11434/api/version
```

ClickyX parle à `http://127.0.0.1:11434` uniquement. Aucun appel cloud n’est
nécessaire une fois le modèle présent sur le disque (`~/.ollama/models`).

## 4. Modèle STT local — Whisper

Deux backends sont acceptés. Un seul suffit.

### Option A — serveur HTTP whisper.cpp

```sh
# Depuis le dépôt whisper.cpp, après compilation :
./build/bin/whisper-server -m models/ggml-base.bin --host 127.0.0.1 --port 8090
```

### Option B — binaire CLI

Placez le modèle ici (créé automatiquement au premier lancement) :

| Plateforme | Dossier |
|---|---|
| Linux | `~/.local/share/clickyx/models/whisper/` |
| macOS | `~/Library/Application Support/clickyx/models/whisper/` |
| Windows | `%APPDATA%/clickyx/models/whisper/` |

Fichiers reconnus : `ggml-tiny.bin`, `ggml-base.bin`, `ggml-small.bin`.

Téléchargement **en ligne**, une fois :

```sh
# Linux / macOS
mkdir -p "$HOME/.local/share/clickyx/models/whisper"
curl -L -o "$HOME/.local/share/clickyx/models/whisper/ggml-base.bin" \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
```

Puis, hors ligne, ClickyX utilise `whisper-cli` / `whisper-cpp` / `whisper`
s’il est dans le `PATH`, ou `CLICKYX_WHISPER_CLI=/chemin/vers/whisper-cli`.

## 5. TTS hors ligne

Aucun modèle à télécharger. ClickyX bascule automatiquement sur la voix
système (SAPI, AVFoundation, Speech Dispatcher).

Sur Linux : `sudo apt-get install speech-dispatcher`.

## 6. Lancer ClickyX sans Internet

```sh
# Forcer le mode hors ligne (Wi-Fi déjà coupé ou non)
export CLICKYX_OFFLINE=1
export CLICKYX_OLLAMA_URL=http://127.0.0.1:11434
export CLICKYX_WHISPER_URL=http://127.0.0.1:8090
export CLICKYX_OLLAMA_MODEL=llama3.2

npm run tauri dev
# ou le binaire déjà compilé :
# ./src-tauri/target/release/clickyx
```

Variables d’environnement reconnues :

| Variable | Rôle |
|---|---|
| `CLICKYX_OFFLINE` / `CLICKYX_FORCE_OFFLINE` | Interdit tout accès WAN |
| `CLICKYX_OLLAMA_URL` | URL du daemon Ollama |
| `CLICKYX_OLLAMA_MODEL` | Tag Ollama par défaut |
| `CLICKYX_WHISPER_URL` | Serveur whisper.cpp |
| `CLICKYX_WHISPER_CLI` | Chemin du binaire whisper.cpp |
| `CLICKYX_WHISPER_MODEL` | `tiny` / `base` / `small` |

## 7. Vérifications

```sh
# Santé locale (bridge ClickyX)
curl http://127.0.0.1:32123/health
curl http://127.0.0.1:32123/offline/status

# Commandes Tauri (depuis le runtime)
# offline_status
# list_local_providers
# list_local_models
# ollama_health
```

`offline/status` doit indiquer `blocks_wan: true` et, si Ollama tourne,
`ollama_reachable: true`.

## 8. Stockage local

```
<platform data dir>/clickyx/
  models/whisper/     ggml-*.bin
  models/llm/         réservé
  cache/              inventaire / santé
  offline/            state.json, inventory.json
```

Aucun cache n’est lu depuis le réseau au démarrage.

## 9. Ce qui reste volontairement indisponible hors ligne

- Mises à jour (`releases.clickyx.app`)
- Génération 3D Tripo3D
- Google Workspace / OAuth
- Providers cloud : Anthropic, OpenAI, Deepgram, ElevenLabs, Cartesia

Ces chemins échouent **immédiatement** (pas de timeout long).
Le chat, la voix et le bridge basculent sur Ollama / Whisper / TTS système.
