# Audit des dépendances Internet — ClickyX

Branche : `feature/offline-engine`
Périmètre : `src/core`, `src/services`, `src/backend`, `src-tauri/src`
Le module IA (`src-tauri/src/ai/`) n’est pas modifié.

## Dépendances WAN et alternatives locales

| Dépendance | Fichier d’origine | Alternative hors ligne |
|---|---|---|
| Anthropic Messages API | `ai/anthropic.rs`, `bridge.rs` | Ollama via `offline::ollama` + `route_chat` |
| OpenAI Chat Completions | `ai/openai.rs`, `bridge.rs` | Ollama local (`127.0.0.1:11434`) |
| Google Generative Language | `ai/google.rs` | Ollama |
| Catalogue modèles OpenAI | `ai/catalog.rs` | `ModelManager` (inventaire disque + `ollama list`) |
| Deepgram STT | `audio/stt.rs` | Whisper local (`whisper-local`) |
| OpenAI Whisper API | `audio/stt.rs` | whisper.cpp HTTP / CLI |
| AssemblyAI | `audio/stt.rs` | Whisper local |
| ElevenLabs / Cartesia / Deepgram Aura / Edge TTS | `audio/tts.rs` | TTS système |
| Updater `releases.clickyx.app` | `updater.rs` | Ignoré si WAN bloqué |
| GitHub releases | `updater.rs` | Ignoré si WAN bloqué |
| Tripo3D | `gen3d.rs` | Refusé par `guard_wan` |
| HuggingFace ggml (Whisper) | `offline/models.rs` | Téléchargement **uniquement** si WAN autorisé ; sinon import fichier |
| Registre Ollama (`ollama pull`) | `offline/ollama.rs` | Refusé hors ligne ; modèles déjà pullés restent utilisables |
| Vite / npm registry | `package.json` | Dev uniquement — `npm ci` une fois, puis cache local |

## Comportement au démarrage (sans Internet)

1. `OfflineManager::bootstrap` sonde le WAN en ≤ 400 ms (`1.1.1.1:443` / `8.8.8.8:443`).
2. Échec = `blocks_wan = true`. Aucun retry bloquant.
3. Check updater **non lancé**.
4. Pipeline voix : STT `LocalWhisper`, TTS `System`.
5. Chat / bridge : routage Ollama. Si Ollama n’est pas encore lancé, erreur locale immédiate (pas d’appel cloud).
6. Loopback (`127.0.0.1`, `localhost`, `::1`) reste autorisé.

## Commandes Tauri ajoutées

- `offline_status` / `offline_refresh`
- `get_offline_config` / `update_offline_config`
- `list_local_providers`
- `list_local_models`
- `download_local_model` (WAN requis)
- `import_whisper_model` (fichier local)
- `ollama_health`
- `clear_offline_cache`

## Bridge

- `GET /offline/status`
- `GET /models` enrichi avec l’inventaire local
- `POST /v1/messages` et `POST /v1/responses` basculent sur Ollama hors ligne
