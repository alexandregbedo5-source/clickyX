//! Tauri commands for the offline engine.

use tauri::AppHandle;

use crate::config;
use crate::offline::manager::{self, OfflineStatus};
use crate::offline::models::{LocalModel, ModelInventory, ModelManager};
use crate::offline::ollama::{self, OllamaHealth};
use crate::offline::registry::{ProviderDescriptor, ProviderRegistry};
use crate::offline::storage::OfflineConfig;

#[tauri::command]
pub fn offline_status() -> OfflineStatus {
    manager::status()
}

#[tauri::command]
pub fn offline_refresh() -> OfflineStatus {
    let _ = manager::refresh();
    manager::status()
}

#[tauri::command]
pub fn get_offline_config() -> OfflineConfig {
    manager::config()
}

#[tauri::command]
pub fn update_offline_config(
    app: AppHandle,
    partial: serde_json::Value,
) -> Result<OfflineConfig, String> {
    let mut app_config = config::load_config(&app)?;
    if let Some(obj) = partial.as_object() {
        if let Some(v) = obj.get("force_offline").and_then(|v| v.as_bool()) {
            app_config.offline.force_offline = v;
        }
        if let Some(v) = obj.get("auto_fallback").and_then(|v| v.as_bool()) {
            app_config.offline.auto_fallback = v;
        }
        if let Some(v) = obj.get("ollama_base_url").and_then(|v| v.as_str()) {
            app_config.offline.ollama_base_url = v.to_string();
        }
        if let Some(v) = obj.get("whisper_base_url").and_then(|v| v.as_str()) {
            app_config.offline.whisper_base_url = v.to_string();
        }
        if let Some(v) = obj.get("whisper_cli").and_then(|v| v.as_str()) {
            app_config.offline.whisper_cli = Some(v.to_string());
        }
        if let Some(v) = obj.get("whisper_model").and_then(|v| v.as_str()) {
            app_config.offline.whisper_model = v.to_string();
        }
        if let Some(v) = obj.get("default_llm").and_then(|v| v.as_str()) {
            app_config.offline.default_llm = v.to_string();
        }
        if let Some(v) = obj.get("probe_timeout_ms").and_then(|v| v.as_u64()) {
            app_config.offline.probe_timeout_ms = v;
        }
    }
    config::save_config(&app, &app_config)?;
    manager::init(app_config.offline.clone());
    Ok(app_config.offline)
}

#[tauri::command]
pub fn list_local_providers() -> Vec<ProviderDescriptor> {
    ProviderRegistry::from_global().all()
}

#[tauri::command]
pub async fn list_local_models() -> ModelInventory {
    ModelManager::new().scan().await
}

#[tauri::command]
pub async fn download_local_model(id: String) -> Result<LocalModel, String> {
    ModelManager::new().download(&id).await
}

#[tauri::command]
pub fn import_whisper_model(id: String, path: String) -> Result<LocalModel, String> {
    ModelManager::new().register_whisper_file(&id, &path)
}

#[tauri::command]
pub async fn ollama_health() -> OllamaHealth {
    ollama::health().await
}

#[tauri::command]
pub fn clear_offline_cache() -> Result<(), String> {
    crate::offline::cache::FileCache::new().clear()
}
