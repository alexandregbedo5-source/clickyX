# 📦 ClickyX v0.1.4 — Mise à Jour Complète

Félicitations ! Vous avez maintenant une version à jour de ClickyX avec support Ollama et Google LLM.

## 🎯 Ce Qui Est Nouveau

### 1️⃣ Ollama (LLM Local, Gratuit)
- Utilisez des modèles open-source **totalement localement**
- Aucune API key, aucune limite d'utilisation
- Modèles supportés : llama2, mistral, neural-chat, orca, dolphin
- **Installation** :
  1. Télécharger : https://ollama.com
  2. Lancer : `ollama serve`
  3. Télécharger un modèle : `ollama pull mistral`
  4. Settings → AI Providers → Ollama → Configure

### 2️⃣ Google PaLM / Vertex AI
- Modèle text-bison-001 via Google Cloud
- Gratuit avec crédits Google Cloud
- **Configuration** :
  1. Obtenir clé API : Google Cloud Console
  2. Settings → AI Providers → Google → Entrez clé
  3. Sélectionner comme provider

### 3️⃣ Tests E2E & Qualité
- 90 tests unitaires ✅
- E2E Playwright visual tests ✅
- Build Rust + Frontend validés ✅

## 🚀 Installation

### Windows (Recommandé)

**Nouvelle installation** :
1. Téléchargez : https://github.com/alexandregbedo5-source/clickyX/releases/download/v0.1.4/ClickyX_0.1.3_x64-setup.exe
2. Double-cliquez l'installateur
3. L'app se lance automatiquement

**Mise à jour** (depuis v0.1.3) :
1. Téléchargez le nouvel installateur
2. Exécutez (remplace l'ancienne version)
3. Config est sauvegardée automatiquement

### Build depuis Source

```bash
git clone https://github.com/alexandregbedo5-source/clickyX
cd clickyX

# Installation dépendances
npm install
cargo build --release

# Dev avec hot-reload
npm run dev

# Build production
npm run build
npx tauri build
```

## ⚙️ Configuration Post-Installation

### Première Utilisation

1. **Ouvrir Settings** (⚙️ en bas)
2. **AI Providers** section :
   - **Anthropic / OpenAI / Google** : Entrez votre clé API
   - **Ollama** : Entrez le modèle et l'URL (par défaut OK)
3. **Default Provider** : Choisissez votre provider préféré
4. **Cliquez Save Settings**

### Utiliser Ollama

```bash
# Terminal 1: Lancer Ollama
ollama serve

# Terminal 2 (optionnel): Télécharger un modèle
ollama pull mistral
ollama pull llama2
ollama pull neural-chat

# ClickyX trouvera Ollama automatiquement via localhost:11434
```

### Utiliser Google

1. Allez sur https://console.cloud.google.com
2. Créez un projet
3. Activez "Google Generative Language API"
4. Créez une clé API
5. Copiez la clé dans ClickyX Settings

## 📚 Documentation Complète

| Document | Contenu |
|----------|---------|
| **OLLAMA.md** | Setup Ollama, models, troubleshooting |
| **UPDATE.md** | Installation, upgrade, new features |
| **CHANGELOG.md** | All version history & release notes |
| **README.md** | Overview & quick start |

## 🔧 Dépannage

### "Ollama connection refused"
```bash
# Vérifier qu'Ollama tourne
ollama list

# Lancer Ollama si absent
ollama serve
```

### "No provider configured"
- Settings → AI Providers → Choisissez un provider
- Entrez votre clé (sauf Ollama qui ne nécessite pas de clé)
- Cliquez Save Settings
- Relancez l'app (Tauri apps cache parfois la config)

### "Google API error"
- Vérifiez que votre clé commence par `AIza`
- Vérifiez que Google Generative Language API est activée dans GCP
- Vérifiez que votre compte n'a pas dépassé les quotas gratuits

### "ClickyX won't start"
```bash
# Supprimer la config et relancer (recréera config par défaut)
rmdir %APPDATA%\com.clickyx.app
# Relancer l'app
```

## 📊 Performance

| Provider | Vitesse | Coût | Privacy | Setup |
|----------|---------|------|---------|-------|
| **Ollama** | Lent (local) | Gratuit | ✅ Local | 5 min |
| **Google** | Rapide | Gratuit+credits | ⚠️ Cloud | 5 min |
| **OpenAI** | Rapide | Payant | ⚠️ Cloud | 5 min |
| **Anthropic** | Rapide | Payant | ⚠️ Cloud | 5 min |

**Recommandation** : Commencez par Ollama pour tester (gratuit, local, aucune config). Ajoutez Google/OpenAI pour plus de performance.

## 🐛 Signaler un Bug

Ouvrez une issue : https://github.com/alexandregbedo5-source/clickyX/issues

Incluez :
- Version ClickyX (Settings → About)
- OS (Windows 10/11)
- Steps to reproduce
- Logs (`%APPDATA%\com.clickyx.app\logs`)

## 🔐 Sécurité & Confidentialité

- **Clés API** : Stockées localement, jamais envoyées à tiers
- **Ollama** : Fonctionne entièrement offline
- **Google/OpenAI** : Votre input est envoyé à leurs serveurs (lire leurs privacy policies)

## 🎓 Prochaines Étapes

### À court terme (beta)
- [ ] Ollama streaming support
- [ ] Google vision (image understanding)
- [ ] Auto-updater avec vérification signature

### À moyen terme
- [ ] macOS build
- [ ] Linux build
- [ ] Web service deployment
- [ ] Docker container

## 📞 Support

- **GitHub Issues** : https://github.com/alexandregbedo5-source/clickyX/issues
- **GitHub Discussions** : https://github.com/alexandregbedo5-source/clickyX/discussions
- **Email** : [Contact dev]

---

**Merci d'utiliser ClickyX ! 🎉**

Bon développement et exploration de l'IA locale ! 🚀
