# Mise à jour ClickyX

## Option 1 : Mise à Jour Manuelle (Recommandée)

### Télécharger

1. Allez sur : https://github.com/alexandregbedo5-source/clickyX/releases
2. Téléchargez le dernier **ClickyX_0.1.3_x64-setup.exe**
3. Double-cliquez pour installer (remplacera l'ancienne version)

### Nouvelles Fonctionnalités v0.1.3

✨ **Ollama Support** (LLM Local)
- Intégration provider Ollama
- Support llama2, mistral, neural-chat, etc.
- Aucune API key nécessaire
- Lancer Ollama localement : `ollama serve`
- Configurer dans Settings → AI Providers → Ollama

✨ **Google LLM** (PaLM/Vertex)
- Support Google Generative AI API
- Modèle text-bison-001 supporté
- Configurez votre clé API dans Settings

✨ **Améliorations E2E**
- Tests visuels Playwright mis à jour
- Alignement DOM pour tests
- 90 tests unitaires passant

## Option 2 : Mise à Jour Automatique (À configurer)

[À implémenter] Tauri updater sera configuré pour vérifier les nouvelles versions automatiquement.

## Installation depuis Zéro

```bash
# Télécharger
curl -O https://github.com/alexandregbedo5-source/clickyX/releases/download/v0.1.3/ClickyX_0.1.3_x64-setup.exe

# Ou depuis PowerShell
Invoke-WebRequest -Uri "https://github.com/alexandregbedo5-source/clickyX/releases/download/v0.1.3/ClickyX_0.1.3_x64-setup.exe" -OutFile "ClickyX_installer.exe"

# Exécuter l'installateur
.\ClickyX_installer.exe
```

## Configuration Après Mise à Jour

### Google LLM
1. Settings → AI Providers → Google (PaLM / Vertex)
2. Entrez votre clé API Google
3. Modèle par défaut : `text-bison-001`
4. Sélectionnez "google" comme Default Provider

### Ollama Local
1. Installez Ollama : https://ollama.com
2. Lancez : `ollama serve`
3. Téléchargez un modèle : `ollama pull llama2`
4. Settings → AI Providers → Ollama (Local)
5. Model : `llama2`, Base URL : `http://localhost:11434`
6. Sélectionnez "ollama" comme Default Provider

## Dépannage

**Après mise à jour, l'app refuse de démarrer**
- Supprimez le répertoire config : `%APPDATA%\com.clickyx.app`
- Relancez l'app (recréera la config par défaut)

**Ollama ne se connecte pas**
- Vérifiez que `ollama serve` est lancé
- Testez : `curl http://localhost:11434/api/tags`

**Erreur API Google**
- Vérifiez votre clé API (commence par `AIza...`)
- Vérifiez que Google Generative AI API est activée dans GCP

## Signaler des Bugs

Ouvrez une issue sur GitHub : https://github.com/alexandregbedo5-source/clickyX/issues

Incluez :
- Version ClickyX (Settings → About)
- OS (Windows 10/11)
- Provider utilisé (Google, Ollama, etc.)
- Logs complets (`%APPDATA%\com.clickyx.app\logs`)
