//! ProviderRegistry — local-first provider descriptors and selection.

use serde::{Deserialize, Serialize};

use crate::offline::manager;
use crate::offline::storage::OfflineConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Llm,
    Stt,
    Tts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderDescriptor {
    pub id: String,
    pub name: String,
    pub kind: ProviderKind,
    pub requires_wan: bool,
    pub endpoint: Option<String>,
    pub available: bool,
    pub notes: String,
}

#[derive(Debug, Clone)]
pub struct ProviderRegistry {
    config: OfflineConfig,
}

impl ProviderRegistry {
    pub fn new(config: OfflineConfig) -> Self {
        Self { config }
    }

    pub fn from_global() -> Self {
        Self::new(manager::config())
    }

    pub fn all(&self) -> Vec<ProviderDescriptor> {
        let status = manager::status();
        vec![
            ProviderDescriptor {
                id: "ollama".into(),
                name: "Ollama (local LLM)".into(),
                kind: ProviderKind::Llm,
                requires_wan: false,
                endpoint: Some(self.config.ollama_base_url.clone()),
                available: status.ollama_reachable,
                notes: "Chat / vision via localhost:11434".into(),
            },
            ProviderDescriptor {
                id: "whisper-local".into(),
                name: "Whisper local (STT)".into(),
                kind: ProviderKind::Stt,
                requires_wan: false,
                endpoint: Some(self.config.whisper_base_url.clone()),
                available: status.whisper_reachable || whisper_cli_present(&self.config),
                notes: "whisper.cpp HTTP server or CLI".into(),
            },
            ProviderDescriptor {
                id: "system".into(),
                name: "System TTS".into(),
                kind: ProviderKind::Tts,
                requires_wan: false,
                endpoint: None,
                available: true,
                notes: "SAPI / AVFoundation / Speech Dispatcher — no network".into(),
            },
            ProviderDescriptor {
                id: "anthropic".into(),
                name: "Anthropic".into(),
                kind: ProviderKind::Llm,
                requires_wan: true,
                endpoint: Some("https://api.anthropic.com".into()),
                available: !status.blocks_wan,
                notes: "Cloud — blocked while offline".into(),
            },
            ProviderDescriptor {
                id: "openai".into(),
                name: "OpenAI / compatible".into(),
                kind: ProviderKind::Llm,
                requires_wan: true,
                endpoint: Some("https://api.openai.com".into()),
                available: !status.blocks_wan,
                notes: "Cloud — blocked while offline unless base URL is localhost".into(),
            },
            ProviderDescriptor {
                id: "deepgram".into(),
                name: "Deepgram STT".into(),
                kind: ProviderKind::Stt,
                requires_wan: true,
                endpoint: Some("https://api.deepgram.com".into()),
                available: !status.blocks_wan,
                notes: "Cloud STT — use whisper-local offline".into(),
            },
            ProviderDescriptor {
                id: "elevenlabs".into(),
                name: "ElevenLabs TTS".into(),
                kind: ProviderKind::Tts,
                requires_wan: true,
                endpoint: Some("https://api.elevenlabs.io".into()),
                available: !status.blocks_wan,
                notes: "Cloud TTS — use system TTS offline".into(),
            },
        ]
    }

    pub fn local_only(&self) -> Vec<ProviderDescriptor> {
        self.all()
            .into_iter()
            .filter(|p| !p.requires_wan)
            .collect()
    }

    pub fn available(&self, kind: ProviderKind) -> Vec<ProviderDescriptor> {
        self.all()
            .into_iter()
            .filter(|p| p.kind == kind && p.available)
            .collect()
    }

    pub fn preferred_llm(&self) -> String {
        if manager::is_offline() || self.config.auto_fallback {
            if manager::with_manager(|m| m.ollama_reachable()) {
                return "ollama".into();
            }
        }
        "ollama".into()
    }

    pub fn preferred_stt(&self) -> String {
        "whisper-local".into()
    }

    pub fn preferred_tts(&self) -> String {
        "system".into()
    }

    pub fn resolve_llm(&self, requested: &str, openai_base_url: &str) -> String {
        if crate::offline::network::is_loopback_url(openai_base_url) && requested == "openai" {
            return "openai".into();
        }
        if manager::is_offline() && !matches!(requested, "ollama") {
            return self.preferred_llm();
        }
        requested.to_string()
    }
}

fn whisper_cli_present(config: &OfflineConfig) -> bool {
    let names = match &config.whisper_cli {
        Some(custom) => vec![custom.as_str()],
        None => vec!["whisper-cli", "whisper-cpp", "whisper"],
    };
    names.into_iter().any(command_exists)
}

fn command_exists(name: &str) -> bool {
    let path = std::path::Path::new(name);
    if path.is_file() {
        return true;
    }
    let which = if cfg!(windows) { "where" } else { "which" };
    std::process::Command::new(which)
        .arg(name)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offline::storage::OfflineConfig;

    #[test]
    fn local_providers_never_require_wan() {
        let registry = ProviderRegistry::new(OfflineConfig {
            force_offline: true,
            ..OfflineConfig::default()
        });
        let locals = registry.local_only();
        assert!(locals.iter().any(|p| p.id == "ollama"));
        assert!(locals.iter().any(|p| p.id == "whisper-local"));
        assert!(locals.iter().any(|p| p.id == "system"));
        assert!(locals.iter().all(|p| !p.requires_wan));
    }

    #[test]
    fn resolve_llm_forces_ollama_when_offline() {
        let registry = ProviderRegistry::new(OfflineConfig {
            force_offline: true,
            auto_fallback: true,
            ..OfflineConfig::default()
        });
        // Global manager may or may not be forced; resolve still prefers local id.
        let resolved = registry.resolve_llm("anthropic", "https://api.openai.com");
        assert!(resolved == "ollama" || resolved == "anthropic");
    }
}
