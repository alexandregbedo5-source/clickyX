# ClickyX — Guide utilisateur : hors ligne & détection d’images IA

Branche : `feature/local-ai-ui`  
Public : utilisatrices et utilisateurs de l’application (pas les moteurs)

Ce guide décrit uniquement l’interface. Les moteurs (offline / ML) sont consommés tels quels via leurs APIs.

---

## 1. Où trouver les écrans

| Besoin | Chemin |
|---|---|
| Analyser une image | Home → **Detect AI** (à côté de Back) |
| Même expérience | Settings → **AI Detection** |
| Forcer le hors ligne, URLs locales | Settings → **Offline** |
| Indicateur réseau | Barre de statut en bas (pastille Online / Local / Offline) |
| Raccourcis | `Ctrl+K` puis « Detect AI image » ou « Offline settings » |

L’historique, les résultats et l’aide sont des onglets internes de cette expérience (Detect / Results / History / Settings / Help).

---

## 2. Mode hors ligne

### Ce que montre l’indicateur

- **Online** — Internet joignable. Les providers cloud restent possibles.
- **Local only** — pas d’Internet (ou mode forcé), mais un service local (Ollama / Whisper) a été vu. Le chat doit rester sur la machine.
- **Offline** — ni Internet ni service local détecté.

Cliquez l’indicateur pour ouvrir Settings → Offline.

### Réglages utiles

- **Force offline** — bloque les appels cloud même si le Wi‑Fi est là.
- **Fall back to local models** — si le cloud échoue, bascule vers Ollama / Whisper / voix système.
- **Ollama URL** — défaut `http://127.0.0.1:11434`
- **Whisper URL** — défaut `http://127.0.0.1:8090`

Ces boutons appellent `offline_status`, `get_offline_config` et `update_offline_config` (ou `GET /offline/status` sur le bridge). Si le moteur offline n’est pas encore branché, l’écran se rabat sur l’état navigateur (`navigator.onLine`) et n’invente pas de données.

### Pour que le chat marche sans réseau

1. Installer [Ollama](https://ollama.com) **une fois, en ligne**.
2. `ollama serve` puis `ollama pull llama3.2`
3. Couper le réseau et vérifier que l’indicateur passe à Local / Offline.
4. Parler à ClickyX : la requête doit rester en local.

La voix système n’a pas besoin de clé. Whisper local n’est nécessaire que pour dicter hors ligne.

---

## 3. Détection d’images IA

### Démarrer le service (une fois)

Le moteur écoute **uniquement en local** :

```text
POST http://127.0.0.1:32188/detect-ai-image
```

Exemple de lancement (branche `feature/ai-image-detector`) :

```sh
python -m ai_detector serve
```

L’interface n’entraîne rien et n’importe aucun modèle. Elle envoie l’image et affiche la réponse.

### Analyser une image

1. Ouvrir **Detect**.
2. Glisser-déposer un PNG, JPEG, WebP, BMP ou GIF (12 Mo max), ou cliquer pour choisir.
3. Vérifier l’aperçu.
4. **Analyze image**.

Pendant l’appel, une barre de progression suit les étapes locales (lecture → FFT → bruit → CNN → fusion). L’API ne pousse pas d’événements : la barre avance pendant l’attente, puis passe à 100 % à la réponse.

### Lire un résultat

Réponse minimale garantie :

```json
{
  "is_ai_generated": true,
  "confidence": 0.92
}
```

- **Likely AI-generated** — `is_ai_generated: true` (confiance ≥ seuil du moteur, 0,5 par défaut).
- **Likely a photograph** — `false`.
- Une confiance entre 40 % et 60 % est affichée comme **Uncertain**.

Les scores optionnels `fft_score`, `noise_score` et `cnn_score` apparaissent si le moteur les envoie. Un CNN à 50 % signifie souvent « modèle absent / neutre ».

Ce n’est **pas une preuve juridique**. C’est une estimation forensique locale.

### Historique

Chaque analyse peut être enregistrée dans le stockage local du profil (`localStorage`).  
Rien n’est envoyé dans le cloud. Vous pouvez supprimer une entrée ou tout effacer.

Réglages : taille de l’historique, enregistrement automatique, jeton optionnel `x-ai-detector-token`, URL du détecteur.

---

## 4. Dépannage

| Symptôme | Que faire |
|---|---|
| « Detector not running » | Lancer `python -m ai_detector serve`, puis Refresh |
| Timeout / unreachable | Vérifier l’URL dans Settings (défaut `:32188`) et qu’aucun pare-feu local ne bloque loopback |
| 401 unauthorized | Renseigner le même jeton que `AI_DETECTOR_TOKEN` |
| Indicateur toujours Online alors que le Wi‑Fi est coupé | Le moteur offline n’est pas chargé : l’UI utilise alors seulement `navigator.onLine` |
| Chat muet hors ligne | `ollama serve` + modèle déjà téléchargé |
| Historique vide après redémarrage | Vérifier que « Save each result locally » est coché |

---

## 5. Vie privée

- Images : transmises uniquement à `127.0.0.1` (détecteur).
- Historique : machine locale uniquement.
- Aucune télémétrie ajoutée par cette interface.
