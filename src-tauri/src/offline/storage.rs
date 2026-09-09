//! Local filesystem layout for models, cache, and offline state.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Persisted offline-engine settings. Independent from Alexandre's AiConfig.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OfflineConfig {
    /// Force WAN-off regardless of probes (`CLICKYX_OFFLINE=1` also sets this).
    pub force_offline: bool,
    /// When a cloud provider is unreachable, route chat/STT/TTS to local providers.
    pub auto_fallback: bool,
    pub ollama_base_url: String,
    pub whisper_base_url: String,
    /// Optional path or binary name for whisper.cpp CLI (`whisper-cli`, `whisper-cpp`).
    pub whisper_cli: Option<String>,
    pub whisper_model: String,
    /// Preferred local LLM id (Ollama tag), used when cloud models cannot run.
    pub default_llm: String,
    pub probe_timeout_ms: u64,
}

impl Default for OfflineConfig {
    fn default() -> Self {
        Self {
            force_offline: env_flag("CLICKYX_OFFLINE") || env_flag("CLICKYX_FORCE_OFFLINE"),
            auto_fallback: true,
            ollama_base_url: std::env::var("CLICKYX_OLLAMA_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".into()),
            whisper_base_url: std::env::var("CLICKYX_WHISPER_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8090".into()),
            whisper_cli: std::env::var("CLICKYX_WHISPER_CLI").ok(),
            whisper_model: std::env::var("CLICKYX_WHISPER_MODEL")
                .unwrap_or_else(|_| "base".into()),
            default_llm: std::env::var("CLICKYX_OLLAMA_MODEL")
                .unwrap_or_else(|_| "llama3.2".into()),
            probe_timeout_ms: 400,
        }
    }
}

pub fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

pub fn data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("clickyx")
}

pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

pub fn whisper_models_dir() -> PathBuf {
    models_dir().join("whisper")
}

pub fn llm_models_dir() -> PathBuf {
    models_dir().join("llm")
}

pub fn cache_dir() -> PathBuf {
    data_dir().join("cache")
}

pub fn offline_state_dir() -> PathBuf {
    data_dir().join("offline")
}

pub fn inventory_path() -> PathBuf {
    offline_state_dir().join("inventory.json")
}

pub fn state_path() -> PathBuf {
    offline_state_dir().join("state.json")
}

pub fn ensure_layout() -> Result<(), String> {
    for dir in [
        data_dir(),
        models_dir(),
        whisper_models_dir(),
        llm_models_dir(),
        cache_dir(),
        offline_state_dir(),
    ] {
        fs::create_dir_all(&dir).map_err(|e| format!("failed to create {}: {e}", dir.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_urls_are_loopback() {
        let cfg = OfflineConfig {
            force_offline: false,
            auto_fallback: true,
            ollama_base_url: "http://127.0.0.1:11434".into(),
            whisper_base_url: "http://127.0.0.1:8090".into(),
            whisper_cli: None,
            whisper_model: "base".into(),
            default_llm: "llama3.2".into(),
            probe_timeout_ms: 400,
        };
        assert!(cfg.ollama_base_url.contains("127.0.0.1"));
        assert!(cfg.whisper_base_url.contains("127.0.0.1"));
        assert!(cfg.auto_fallback);
    }

    #[test]
    fn env_flag_parses_truthy_values() {
        std::env::set_var("CLICKYX_TEST_FLAG", "true");
        assert!(env_flag("CLICKYX_TEST_FLAG"));
        std::env::set_var("CLICKYX_TEST_FLAG", "0");
        assert!(!env_flag("CLICKYX_TEST_FLAG"));
        std::env::remove_var("CLICKYX_TEST_FLAG");
    }

    #[test]
    fn ensure_layout_creates_dirs() {
        ensure_layout().expect("layout");
        assert!(whisper_models_dir().exists());
        assert!(cache_dir().exists());
        assert!(offline_state_dir().exists());
    }
}
