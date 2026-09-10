//! ModelManager — local inventory, catalog, and optional downloads.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::offline::cache::FileCache;
use crate::offline::manager;
use crate::offline::ollama;
use crate::offline::storage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Llm,
    Whisper,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModel {
    pub id: String,
    pub display_name: String,
    pub kind: ModelKind,
    pub provider: String,
    pub installed: bool,
    pub path: Option<String>,
    pub size_bytes: u64,
    pub download_url: Option<String>,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInventory {
    pub models: Vec<LocalModel>,
    pub data_dir: String,
}

#[derive(Debug, Clone)]
pub struct ModelManager {
    cache: FileCache,
}

impl ModelManager {
    pub fn new() -> Self {
        let _ = storage::ensure_layout();
        Self {
            cache: FileCache::new(),
        }
    }

    pub fn catalog() -> Vec<LocalModel> {
        vec![
            LocalModel {
                id: "llama3.2".into(),
                display_name: "Llama 3.2 (Ollama)".into(),
                kind: ModelKind::Llm,
                provider: "ollama".into(),
                installed: false,
                path: None,
                size_bytes: 0,
                download_url: None,
                notes: "ollama pull llama3.2 — recommended default local LLM".into(),
            },
            LocalModel {
                id: "llama3.2:1b".into(),
                display_name: "Llama 3.2 1B (Ollama)".into(),
                kind: ModelKind::Llm,
                provider: "ollama".into(),
                installed: false,
                path: None,
                size_bytes: 0,
                download_url: None,
                notes: "Smallest useful chat model for low-RAM machines".into(),
            },
            LocalModel {
                id: "mistral".into(),
                display_name: "Mistral 7B (Ollama)".into(),
                kind: ModelKind::Llm,
                provider: "ollama".into(),
                installed: false,
                path: None,
                size_bytes: 0,
                download_url: None,
                notes: "ollama pull mistral".into(),
            },
            LocalModel {
                id: "qwen2.5:3b".into(),
                display_name: "Qwen 2.5 3B (Ollama)".into(),
                kind: ModelKind::Llm,
                provider: "ollama".into(),
                installed: false,
                path: None,
                size_bytes: 0,
                download_url: None,
                notes: "Good multilingual local default".into(),
            },
            whisper_catalog_entry("tiny", "Whisper tiny (ggml)", "ggml-tiny.bin"),
            whisper_catalog_entry("base", "Whisper base (ggml)", "ggml-base.bin"),
            whisper_catalog_entry("small", "Whisper small (ggml)", "ggml-small.bin"),
        ]
    }

    pub async fn inventory(&self) -> ModelInventory {
        if let Some(cached) = self.cache.get::<ModelInventory>("inventory") {
            return cached;
        }
        let inventory = self.scan().await;
        let _ = self.cache.set("inventory", &inventory, 30);
        inventory
    }

    pub async fn scan(&self) -> ModelInventory {
        let mut models = Self::catalog();

        let ollama_tags = ollama::list_tags().await.unwrap_or_default();
        for model in models.iter_mut() {
            if model.kind == ModelKind::Llm {
                if let Some(tag) = ollama_tags.iter().find(|t| {
                    t.name == model.id || t.name.starts_with(&format!("{}:", model.id))
                }) {
                    model.installed = true;
                    model.size_bytes = tag.size;
                }
            }
        }
        for tag in ollama_tags {
            if !models.iter().any(|m| m.id == tag.name) {
                models.push(LocalModel {
                    id: tag.name.clone(),
                    display_name: tag.name,
                    kind: ModelKind::Llm,
                    provider: "ollama".into(),
                    installed: true,
                    path: None,
                    size_bytes: tag.size,
                    download_url: None,
                    notes: "Installed via Ollama".into(),
                });
            }
        }

        for model in models.iter_mut() {
            if model.kind == ModelKind::Whisper {
                if let Some(path) = whisper_file_for(&model.id) {
                    model.installed = true;
                    model.path = Some(path.display().to_string());
                    model.size_bytes = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                }
            }
        }

        let inventory = ModelInventory {
            models,
            data_dir: storage::data_dir().display().to_string(),
        };
        let _ = persist_inventory(&inventory);
        inventory
    }

    pub async fn list_installed(&self) -> Vec<LocalModel> {
        self.inventory()
            .await
            .models
            .into_iter()
            .filter(|m| m.installed)
            .collect()
    }

    pub fn whisper_model_path(&self, id: &str) -> Option<PathBuf> {
        whisper_file_for(id)
    }

    pub fn register_whisper_file(&self, id: &str, path: &str) -> Result<LocalModel, String> {
        let src = PathBuf::from(path);
        if !src.is_file() {
            return Err(format!("Whisper model file not found: {path}"));
        }
        storage::ensure_layout()?;
        let dest = storage::whisper_models_dir().join(format!("ggml-{id}.bin"));
        if src != dest {
            fs::copy(&src, &dest).map_err(|e| format!("failed to copy model: {e}"))?;
        }
        let _ = self.cache.remove("inventory");
        Ok(LocalModel {
            id: id.to_string(),
            display_name: format!("Whisper {id}"),
            kind: ModelKind::Whisper,
            provider: "whisper-local".into(),
            installed: true,
            path: Some(dest.display().to_string()),
            size_bytes: fs::metadata(&dest).map(|m| m.len()).unwrap_or(0),
            download_url: None,
            notes: "Imported locally".into(),
        })
    }

    /// Download a Whisper ggml model. Refuses to run when WAN is blocked.
    pub async fn download(&self, id: &str) -> Result<LocalModel, String> {
        let catalog = Self::catalog();
        let spec = catalog
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| format!("Unknown model '{id}'"))?
            .clone();

        match spec.kind {
            ModelKind::Llm => {
                manager::guard_wan(&format!("ollama pull {id}"))?;
                ollama::pull_model(id).await?;
                let _ = self.cache.remove("inventory");
                let inventory = self.scan().await;
                inventory
                    .models
                    .into_iter()
                    .find(|m| m.id == id && m.installed)
                    .ok_or_else(|| format!("Ollama pull finished but '{id}' is not listed"))
            }
            ModelKind::Whisper => {
                let url = spec
                    .download_url
                    .clone()
                    .ok_or_else(|| "No download URL for this Whisper model".to_string())?;
                if !manager::allows_url(&url) {
                    return Err(format!(
                        "Cannot download {id} while offline. Copy ggml-{id}.bin into {}",
                        storage::whisper_models_dir().display()
                    ));
                }
                storage::ensure_layout()?;
                let dest = storage::whisper_models_dir().join(format!("ggml-{id}.bin"));
                download_file(&url, &dest).await?;
                let _ = self.cache.remove("inventory");
                Ok(LocalModel {
                    id: spec.id,
                    display_name: spec.display_name,
                    kind: ModelKind::Whisper,
                    provider: "whisper-local".into(),
                    installed: true,
                    path: Some(dest.display().to_string()),
                    size_bytes: fs::metadata(&dest).map(|m| m.len()).unwrap_or(0),
                    download_url: spec.download_url,
                    notes: "Downloaded locally".into(),
                })
            }
        }
    }

    pub fn invalidate(&self) {
        let _ = self.cache.remove("inventory");
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

fn whisper_catalog_entry(id: &str, name: &str, filename: &str) -> LocalModel {
    LocalModel {
        id: id.into(),
        display_name: name.into(),
        kind: ModelKind::Whisper,
        provider: "whisper-local".into(),
        installed: false,
        path: None,
        size_bytes: 0,
        download_url: Some(format!(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{filename}"
        )),
        notes: format!(
            "Place {filename} in {} or download while online",
            storage::whisper_models_dir().display()
        ),
    }
}

fn whisper_file_for(id: &str) -> Option<PathBuf> {
    let dir = storage::whisper_models_dir();
    let candidates = [
        dir.join(format!("ggml-{id}.bin")),
        dir.join(format!("{id}.bin")),
        dir.join(format!("ggml-{id}.en.bin")),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

fn persist_inventory(inventory: &ModelInventory) -> Result<(), String> {
    storage::ensure_layout()?;
    let json = serde_json::to_string_pretty(inventory).map_err(|e| e.to_string())?;
    fs::write(storage::inventory_path(), json).map_err(|e| e.to_string())
}

async fn download_file(url: &str, dest: &PathBuf) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("download failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("download HTTP {}", resp.status()));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("download read failed: {e}"))?;
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(dest, bytes).map_err(|e| format!("write failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_covers_llm_and_whisper() {
        let catalog = ModelManager::catalog();
        assert!(catalog.iter().any(|m| m.kind == ModelKind::Llm));
        assert!(catalog.iter().any(|m| m.kind == ModelKind::Whisper));
        assert!(catalog.iter().any(|m| m.id == "llama3.2"));
        assert!(catalog.iter().any(|m| m.id == "base"));
    }

    #[test]
    fn whisper_missing_file_is_none() {
        assert!(whisper_file_for("definitely-missing-model-id").is_none());
    }
}
