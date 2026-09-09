# Moteur hors ligne ClickyX

Branche : `feature/offline-engine`  
Auteur : infrastructure / exécution locale  
Périmètre : coulisses uniquement (pas d’écran, pas de module IA)

## En une phrase

ClickyX peut maintenant **s’ouvrir et servir** même si le Wi-Fi est coupé, le câble Ethernet débranché, ou un pare-feu bloque tout Internet.

## Le problème

Sans réseau, l’app essayait encore d’appeler des services en ligne :

- le chat allait vers Claude / GPT / Google
- la dictée allait vers Deepgram ou Whisper cloud
- la voix allait vers ElevenLabs et cie
- au démarrage, l’app vérifiait les mises à jour sur Internet

Résultat : timeouts, erreurs, ou app qui a l’air plantée.

## Ce qui a été ajouté

Trois pièces, plus le branchement dans le backend existant.

### OfflineManager

Le régisseur. Au lancement il regarde **très vite** (moins d’une demi-seconde) s’il y a Internet.

- S’il n’y en a pas → il coupe les appels cloud.
- Le **localhost** reste autorisé (Ollama, Whisper sur la machine).
- On peut forcer le mode hors ligne : `CLICKYX_OFFLINE=1`

### ModelManager

L’inventaire des modèles **déjà sur le disque**.

- Liste ce qu’Ollama a déjà téléchargé
- Liste les fichiers Whisper locaux (`ggml-base.bin`, etc.)
- Refuse de télécharger quoi que ce soit si on est hors ligne
- Accepte d’importer un fichier déjà présent sur l’ordinateur

### ProviderRegistry

L’annuaire : qui peut travailler sans Internet ?

| Rôle | En ligne (avant) | Hors ligne (maintenant) |
|---|---|---|
| Discussion | Anthropic / OpenAI / Google | **Ollama** sur `127.0.0.1:11434` |
| Dictée | Deepgram / Whisper cloud | **Whisper local** (serveur ou binaire) |
| Voix | ElevenLabs / Cartesia / etc. | **Voix du système** (déjà sur l’OS) |

## Ce qui n’a pas été touché

- Aucun écran, aucun composant React
- Aucune modification du module IA (`src-tauri/src/ai/`)
- Les clés cloud et les réglages existants restent valables dès que le réseau revient

## Où est le code

```
src-tauri/src/offline/     moteur Rust (celui qui compte au runtime)
src/core/                  OfflineManager, ModelManager, ProviderRegistry (TypeScript)
src/services/              cache, Ollama, Whisper, stockage
src/backend/               ce dossier : docs + point d’entrée
```

Commandes Tauri ajoutées (pas d’UI branchée dessus) :

- `offline_status` / `offline_refresh`
- `list_local_providers` / `list_local_models`
- `download_local_model` (uniquement si Internet)
- `import_whisper_model` (fichier local)
- `ollama_health`

Le bridge local répond aussi : `GET http://127.0.0.1:32123/offline/status`

## Faut-il installer des trucs à la main ?

**Oui, une fois, quand tu as encore Internet.** Ensuite plus besoin.

| À installer | Obligatoire ? | Pourquoi |
|---|---|---|
| Ollama + un modèle (`ollama pull llama3.2`) | **Oui** pour le chat hors ligne | Sinon personne ne répond aux questions |
| Whisper local (whisper.cpp ou fichier `ggml-*.bin`) | Seulement si tu veux dicter hors ligne | Sinon la dictée reste cloud |
| Voix système | Non | Déjà là. Sur Linux : `speech-dispatcher` si ça ne parle pas |
| `npm ci` / Rust | Comme d’habitude pour compiler l’app | Une fois, en ligne |

Après ça : `export CLICKYX_OFFLINE=1`, `ollama serve`, puis lancer ClickyX.

Le détail pas à pas est dans [INSTALLATION.md](./INSTALLATION.md).  
La liste des appels Internet et de leurs remplacements est dans [AUDIT.md](./AUDIT.md).

## Ce qui reste volontairement cassé hors ligne

Ces fonctions ont besoin d’Internet. Elles s’arrêtent tout de suite, sans faire attendre :

- mises à jour de l’application
- génération 3D (Tripo3D)
- Google Workspace / OAuth
- n’importe quel provider cloud si Ollama / Whisper / voix système n’est pas là

## Comment vérifier

```sh
# Ollama répond en local
curl http://127.0.0.1:11434/api/version

# ClickyX dit s’il est hors ligne
curl http://127.0.0.1:32123/offline/status
```

`blocks_wan` à `true` = pas d’Internet, l’app ne doit plus essayer le cloud.  
`ollama_reachable` à `true` = le chat local est prêt.
