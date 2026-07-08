# Script pour créer une Release GitHub

## Prérequis
- GitHub CLI (`gh`) installé et authentifié
- Ou : Token GitHub personnel avec permissions `repo`

## Option 1 : Avec GitHub CLI (Recommandé)

```bash
cd C:\Users\ADMIN\clickyX

# Créer le tag
git tag v0.1.4 -m "Release v0.1.4: Ollama + Google LLM providers"

# Pousser le tag
git push origin v0.1.4

# Créer la Release avec description
gh release create v0.1.4 \
  --title "ClickyX v0.1.4" \
  --notes-file CHANGELOG.md \
  src-tauri/target/x86_64-pc-windows-msvc/release/bundle/nsis/ClickyX_0.1.3_x64-setup.exe \
  src-tauri/target/x86_64-pc-windows-msvc/release/bundle/msi/ClickyX_0.1.3_x64_en-US.msi
```

## Option 2 : Via GitHub Web UI (Manuel)

1. Allez sur : https://github.com/alexandregbedo5-source/clickyX/releases
2. Cliquez **"Create a new release"**
3. **Tag version**: `v0.1.4`
4. **Release title**: `ClickyX v0.1.4`
5. **Description**: Copiez le contenu de CHANGELOG.md section [0.1.4]
6. **Attachments**: Uploadez les 2 fichiers :
   - `ClickyX_0.1.3_x64-setup.exe` (4.69 MB)
   - `ClickyX_0.1.3_x64_en-US.msi` (6.69 MB)
7. Cochez **"This is a pre-release"** si souhaité
8. Cliquez **"Publish release"**

## Option 3 : Via cURL (Pour automation)

```powershell
$token = "ghp_YOUR_TOKEN_HERE"
$repo = "alexandregbedo5-source/clickyX"
$releaseNotes = @"
## Ollama & Google LLM Providers

### Added
- **Ollama**: Offline LLM support (llama2, mistral, neural-chat, etc.)
- **Google**: PaLM/Vertex API integration
- E2E visual test suites (90 tests passing)
- Complete documentation (OLLAMA.md + UPDATE.md)

### Download
- Windows NSIS Installer: ClickyX_0.1.3_x64-setup.exe
- Windows MSI Installer: ClickyX_0.1.3_x64_en-US.msi

### Install Instructions
1. Download the .exe installer
2. Run the installer (will upgrade previous version)
3. Settings → AI Providers → Configure Ollama or Google

See UPDATE.md for detailed upgrade guide.
"@

# Créer la Release
$body = @{
    tag_name = "v0.1.4"
    name = "ClickyX v0.1.4"
    body = $releaseNotes
    draft = $false
    prerelease = $false
} | ConvertTo-Json

$headers = @{
    Authorization = "Bearer $token"
    "X-GitHub-Api-Version" = "2022-11-28"
    Accept = "application/vnd.github+json"
}

$response = Invoke-RestMethod `
  -Uri "https://api.github.com/repos/$repo/releases" `
  -Method Post `
  -Headers $headers `
  -Body $body

Write-Host "Release created: $($response.html_url)"
$uploadUrl = $response.upload_url -replace '\{.*'

# Uploader les fichiers
$files = @(
    "src-tauri/target/x86_64-pc-windows-msvc/release/bundle/nsis/ClickyX_0.1.3_x64-setup.exe",
    "src-tauri/target/x86_64-pc-windows-msvc/release/bundle/msi/ClickyX_0.1.3_x64_en-US.msi"
)

foreach ($file in $files) {
    $fileName = [System.IO.Path]::GetFileName($file)
    $fileContent = [System.IO.File]::ReadAllBytes($file)
    
    Invoke-RestMethod `
      -Uri "$uploadUrl?name=$fileName" `
      -Method Post `
      -Headers $headers `
      -ContentType "application/octet-stream" `
      -Body $fileContent
    
    Write-Host "Uploaded: $fileName"
}
```

## Après Publication

- Vérifier la Release : https://github.com/alexandregbedo5-source/clickyX/releases/tag/v0.1.4
- Utilisateurs peuvent télécharger et installer
- update.json reste disponible pour Tauri updater

## Rollback (au besoin)

```bash
git tag -d v0.1.4
git push origin :refs/tags/v0.1.4
# Puis supprimer manuellement la Release depuis GitHub UI
```

---

**Note**: Version dans package.json reste 0.1.3 car c'est la version de l'app Tauri.
Le tag v0.1.4 représente la version de la Release (code + installers).
À la prochaine majeure release, synchroniser package.json et tags.
