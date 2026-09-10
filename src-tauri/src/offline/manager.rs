//! OfflineManager — connectivity state, WAN gating, and startup policy.

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::offline::network::{self, ProbeReport};
use crate::offline::storage::{self, OfflineConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Connectivity {
    /// WAN is reachable.
    Online,
    /// No WAN, but localhost services may still work.
    LocalOnly,
    /// Forced or detected isolation (Wi-Fi off, Ethernet unplugged, firewall).
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineStatus {
    pub connectivity: Connectivity,
    pub force_offline: bool,
    pub auto_fallback: bool,
    pub wan_reachable: bool,
    pub ollama_reachable: bool,
    pub whisper_reachable: bool,
    pub blocks_wan: bool,
    pub last_probe_ms: u64,
    pub data_dir: String,
}

#[derive(Debug, Clone)]
pub struct OfflineManager {
    config: OfflineConfig,
    last_report: ProbeReport,
    last_probe_at: Option<Instant>,
}

impl OfflineManager {
    pub fn new(config: OfflineConfig) -> Self {
        let _ = storage::ensure_layout();
        Self {
            config,
            last_report: ProbeReport {
                wan_reachable: false,
                ollama_reachable: false,
                whisper_reachable: false,
                elapsed_ms: 0,
            },
            last_probe_at: None,
        }
    }

    /// Bootstrap without assuming the network works. Probe is short and fallible.
    pub fn bootstrap(config: OfflineConfig) -> Self {
        let mut mgr = Self::new(config);
        mgr.refresh();
        mgr
    }

    pub fn config(&self) -> &OfflineConfig {
        &self.config
    }

    pub fn replace_config(&mut self, config: OfflineConfig) {
        self.config = config;
        self.refresh();
    }

    pub fn refresh(&mut self) -> ProbeReport {
        let report = network::probe(&self.config);
        self.last_report = report.clone();
        self.last_probe_at = Some(Instant::now());
        log::info!(
            "OfflineManager probe: wan={} ollama={} whisper={} ({} ms, force={})",
            report.wan_reachable,
            report.ollama_reachable,
            report.whisper_reachable,
            report.elapsed_ms,
            self.config.force_offline
        );
        report
    }

    pub fn connectivity(&self) -> Connectivity {
        if self.config.force_offline || !self.last_report.wan_reachable {
            if self.last_report.ollama_reachable || self.last_report.whisper_reachable {
                Connectivity::LocalOnly
            } else {
                Connectivity::Offline
            }
        } else {
            Connectivity::Online
        }
    }

    pub fn blocks_wan(&self) -> bool {
        self.config.force_offline || !self.last_report.wan_reachable
    }

    pub fn is_offline(&self) -> bool {
        self.blocks_wan()
    }

    pub fn auto_fallback(&self) -> bool {
        self.config.auto_fallback
    }

    pub fn ollama_reachable(&self) -> bool {
        self.last_report.ollama_reachable
    }

    pub fn whisper_reachable(&self) -> bool {
        self.last_report.whisper_reachable
    }

    pub fn allows_url(&self, url: &str) -> bool {
        if network::is_loopback_url(url) {
            return true;
        }
        !self.blocks_wan()
    }

    pub fn guard_wan(&self, operation: &str) -> Result<(), String> {
        if self.allows_url("https://example.invalid") {
            return Ok(());
        }
        Err(format!(
            "{operation} requires Internet. ClickyX is running offline (Wi-Fi/Ethernet/firewall). Use a local provider instead."
        ))
    }

    pub fn status(&self) -> OfflineStatus {
        OfflineStatus {
            connectivity: self.connectivity(),
            force_offline: self.config.force_offline,
            auto_fallback: self.config.auto_fallback,
            wan_reachable: self.last_report.wan_reachable,
            ollama_reachable: self.last_report.ollama_reachable,
            whisper_reachable: self.last_report.whisper_reachable,
            blocks_wan: self.blocks_wan(),
            last_probe_ms: self.last_report.elapsed_ms,
            data_dir: storage::data_dir().display().to_string(),
        }
    }
}

static GLOBAL: OnceLock<Mutex<OfflineManager>> = OnceLock::new();

fn global_mutex() -> &'static Mutex<OfflineManager> {
    GLOBAL.get_or_init(|| Mutex::new(OfflineManager::bootstrap(OfflineConfig::default())))
}

pub fn init(config: OfflineConfig) {
    let mut guard = global_mutex().lock().unwrap_or_else(|e| e.into_inner());
    *guard = OfflineManager::bootstrap(config);
}

pub fn with_manager<F, R>(f: F) -> R
where
    F: FnOnce(&OfflineManager) -> R,
{
    let guard = global_mutex().lock().unwrap_or_else(|e| e.into_inner());
    f(&guard)
}

pub fn with_manager_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut OfflineManager) -> R,
{
    let mut guard = global_mutex().lock().unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}

pub fn is_offline() -> bool {
    with_manager(|m| m.is_offline())
}

pub fn blocks_wan() -> bool {
    with_manager(|m| m.blocks_wan())
}

pub fn allows_url(url: &str) -> bool {
    with_manager(|m| m.allows_url(url))
}

pub fn guard_wan(operation: &str) -> Result<(), String> {
    with_manager(|m| m.guard_wan(operation))
}

pub fn status() -> OfflineStatus {
    with_manager(|m| m.status())
}

pub fn refresh() -> ProbeReport {
    with_manager_mut(|m| m.refresh())
}

pub fn config() -> OfflineConfig {
    with_manager(|m| m.config().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn isolated_config() -> OfflineConfig {
        OfflineConfig {
            force_offline: true,
            auto_fallback: true,
            ollama_base_url: "http://127.0.0.1:11434".into(),
            whisper_base_url: "http://127.0.0.1:8090".into(),
            whisper_cli: None,
            whisper_model: "base".into(),
            default_llm: "llama3.2".into(),
            probe_timeout_ms: 80,
        }
    }

    #[test]
    fn forced_offline_blocks_wan() {
        let mgr = OfflineManager::bootstrap(isolated_config());
        assert!(mgr.blocks_wan());
        assert!(mgr.is_offline());
        assert!(mgr.allows_url("http://127.0.0.1:11434"));
        assert!(!mgr.allows_url("https://api.openai.com/v1"));
        assert!(mgr.guard_wan("update check").is_err());
    }

    #[test]
    fn status_serializes() {
        let mgr = OfflineManager::new(isolated_config());
        let json = serde_json::to_string(&mgr.status()).unwrap();
        assert!(json.contains("force_offline"));
        assert!(json.contains("blocks_wan"));
    }
}
