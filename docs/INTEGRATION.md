# ClickyX - Integration des trois briques

ClickyX assemble trois briques en une seule application locale et hors ligne.

## 1. Les trois briques

| Brique | Auteur | Role |
|---|---|---|
| Interface locale (UI) | Honorat | Ecrans React Detect / Results / History / Settings / Help, indicateur hors ligne, client d'API (src/ui/local-ai/) |
| Moteur hors ligne | Tybiane | Bridge Rust (Ollama / Whisper / TTS systeme), statut de connectivite (src-tauri/src/offline/) |
| Detecteur IA Forensics | Alexandre | Moteur Python (FFT + bruit + CNN ONNX), API POST /detect-ai-image, modele entraine |

## 2. Schema d'assemblage

L'UI parle a deux services locaux distincts :

- le bridge (127.0.0.1:32123) via l'IPC Tauri (invoke) pour l'etat hors ligne et les modeles ;
- le detecteur (127.0.0.1:32188) via HTTP (POST /detect-ai-image) pour l'analyse d'images.

## 3. Contrats

Les contrats etaient deja alignes : meme port (32188), meme en-tete (x-ai-detector-token),
memes champs de reponse. Le contrat officiel (is_ai_generated, confidence, fft_score,
noise_score, cnn_score) est defini dans ai_detector/schemas.py et replique cote UI dans
src/ui/local-ai/types.ts.

## 4. Cablage effectue

L'onglet 'AI Detector' a ete ajoute a la navigation (src/App.tsx et
src/context/AppContext.tsx), sans modifier le code metier des briques. LocalAiView
encapsule les sous-ecrans de Honorat.

## 5. Lancer en local

Terminal 1 (detecteur) : python -m ai_detector serve  (http://127.0.0.1:32188)
Terminal 2 (app) : npm install puis npm run tauri dev  (ou npm run dev pour le web).
Puis onglet AI Detector, deposer une image.

Verifier : python -m ai_detector info  =>  trained: true, fusion_mode: full.

## 6. Verifications

- Frontend : npm run test
- Type-check : npx tsc --noEmit
- Build : npm run build
- Detecteur : python -m pytest

## 7. Notes

- Detecteur et bridge ecoutent sur loopback uniquement ; aucun port expose au reseau.
- Le modele model/detector.onnx (16 Mo) est versionne via Git normal (LFS non requis).
- Si le detecteur est arrete, l'UI affiche 'injoignable' sans planter.
- Si le CNN entraine est absent, le detecteur passe en mode handcrafted (FFT + bruit).
