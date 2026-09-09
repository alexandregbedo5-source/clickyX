//! ClickyX offline engine.
//!
//! Local-first routing for LLM (Ollama), STT (Whisper), TTS (system),
//! model inventory, cache, and WAN gating. Does not modify the AI module.

pub mod cache;
pub mod commands;
pub mod manager;
pub mod models;
pub mod network;
pub mod ollama;
pub mod registry;
pub mod storage;
pub mod whisper;

use crate::ai;
use crate::offline::manager::OfflineManager;
use crate::offline::models::ModelManager;
use crate::offline::ollama::LocalChatMessage;
use crate::offline::registry::ProviderRegistry;
use crate::offline::storage::OfflineConfig;

pub use manager::{blocks_wan, config, guard_wan, is_offline, status};

/// Start the offline engine during app bootstrap. Never fails the process.
pub fn bootstrap(config: OfflineConfig) {
    if let Err(e) = storage::ensure_layout() {
        log::warn!("offline storage layout: {e}");
    }
    manager::init(config);
    let status = manager::status();
    log::info!(
        "OfflineManager ready: {:?} wan_blocked={} ollama={} whisper={}",
        status.connectivity,
        status.blocks_wan,
        status.ollama_reachable,
        status.whisper_reachable
    );
}

pub fn should_use_local_llm(default_provider: &str, openai_base_url: &str) -> bool {
    if crate::offline::network::is_loopback_url(openai_base_url) && default_provider == "openai" {
        return false;
    }
    if default_provider == "ollama" {
        return true;
    }
    let mgr_offline = manager::is_offline();
    let fallback = manager::with_manager(|m| m.auto_fallback() && m.ollama_reachable());
    mgr_offline || fallback
}

pub fn local_llm_model(requested: &str, offline_cfg: &OfflineConfig) -> String {
    if requested.contains("llama")
        || requested.contains("mistral")
        || requested.contains("qwen")
        || requested.contains("phi")
        || requested.contains("gemma")
        || requested.contains("neural-chat")
        || requested.contains("orca")
        || requested.contains("dolphin")
    {
        return requested.to_string();
    }
    offline_cfg.default_llm.clone()
}

pub async fn chat_local(messages: &[ai::ChatMessage], model: &str) -> Result<String, String> {
    let cfg = manager::config();
    let resolved = local_llm_model(model, &cfg);
    let mapped: Vec<LocalChatMessage> = messages
        .iter()
        .map(|m| LocalChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();
    ollama::chat(&mapped, &resolved).await
}

pub async fn chat_local_stream<F>(
    messages: &[ai::ChatMessage],
    model: &str,
    on_delta: F,
) -> Result<String, String>
where
    F: FnMut(&str),
{
    let cfg = manager::config();
    let resolved = local_llm_model(model, &cfg);
    let mapped: Vec<LocalChatMessage> = messages
        .iter()
        .map(|m| LocalChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();
    ollama::chat_stream(&mapped, &resolved, on_delta).await
}

/// Route chat: local Ollama when offline / fallback, otherwise caller uses AI module.
pub fn prefer_local_chat(default_provider: &str, openai_base_url: &str) -> bool {
    should_use_local_llm(default_provider, openai_base_url)
}

pub fn registry() -> ProviderRegistry {
    ProviderRegistry::from_global()
}

pub fn models() -> ModelManager {
    ModelManager::new()
}

pub fn manager_instance_status() -> manager::OfflineStatus {
    OfflineManager::new(manager::config()).status()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_llm_model_keeps_ollama_tags() {
        let cfg = OfflineConfig {
            default_llm: "llama3.2".into(),
            ..OfflineConfig::default()
        };
        assert_eq!(local_llm_model("mistral", &cfg), "mistral");
        assert_eq!(local_llm_model("gpt-4o", &cfg), "llama3.2");
        assert_eq!(local_llm_model("claude-sonnet-4-20250514", &cfg), "llama3.2");
    }

    #[test]
    fn ollama_provider_always_uses_local() {
        assert!(should_use_local_llm("ollama", "https://api.openai.com"));
    }
}
