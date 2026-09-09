//! Connectivity probes that never hang startup.
//!
//! WAN checks use a short TCP timeout. Loopback services (Ollama, Whisper)
//! are probed separately so a cut WAN / firewall still leaves localhost usable.

use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use crate::offline::storage::OfflineConfig;

/// Result of a connectivity sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeReport {
    pub wan_reachable: bool,
    pub ollama_reachable: bool,
    pub whisper_reachable: bool,
    pub elapsed_ms: u64,
}

impl ProbeReport {
    pub fn offline_wan(&self) -> bool {
        !self.wan_reachable
    }

    pub fn has_local_llm(&self) -> bool {
        self.ollama_reachable
    }

    pub fn has_local_stt(&self) -> bool {
        self.whisper_reachable
    }
}

pub fn is_loopback_host(host: &str) -> bool {
    let h = host.trim().trim_start_matches('[').trim_end_matches(']').to_ascii_lowercase();
    matches!(h.as_str(), "localhost" | "127.0.0.1" | "::1" | "0.0.0.0")
}

pub fn is_loopback_url(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    lower.contains("://127.0.0.1")
        || lower.contains("://localhost")
        || lower.contains("://[::1]")
        || lower.contains("://0.0.0.0")
        || lower.starts_with("127.0.0.1")
        || lower.starts_with("localhost")
}

fn parse_host_port(url: &str, default_port: u16) -> Option<(String, u16)> {
    let trimmed = url.trim();
    let without_scheme = trimmed
        .split("://")
        .nth(1)
        .unwrap_or(trimmed)
        .split('/')
        .next()
        .unwrap_or("");
    if without_scheme.is_empty() {
        return None;
    }
    if let Some(rest) = without_scheme.strip_prefix('[') {
        let (host, port_part) = rest.split_once(']')?;
        let port = port_part
            .trim_start_matches(':')
            .parse::<u16>()
            .unwrap_or(default_port);
        return Some((host.to_string(), port));
    }
    if let Some((host, port)) = without_scheme.rsplit_once(':') {
        if !host.contains(':') {
            let parsed = port.parse::<u16>().unwrap_or(default_port);
            return Some((host.to_string(), parsed));
        }
    }
    Some((without_scheme.to_string(), default_port))
}

pub fn probe_tcp(host: &str, port: u16, timeout: Duration) -> bool {
    let target = format!("{host}:{port}");
    let addrs = match target.to_socket_addrs() {
        Ok(addrs) => addrs.collect::<Vec<SocketAddr>>(),
        Err(_) => return false,
    };
    for addr in addrs {
        if TcpStream::connect_timeout(&addr, timeout).is_ok() {
            return true;
        }
    }
    false
}

pub fn probe_url(url: &str, default_port: u16, timeout: Duration) -> bool {
    match parse_host_port(url, default_port) {
        Some((host, port)) => probe_tcp(&host, port, timeout),
        None => false,
    }
}

/// Fast WAN + localhost sweep. Safe to call on the startup path.
pub fn probe(config: &OfflineConfig) -> ProbeReport {
    let started = Instant::now();
    let timeout = Duration::from_millis(config.probe_timeout_ms.max(50));

    let wan_reachable = if config.force_offline {
        false
    } else {
        probe_tcp("1.1.1.1", 443, timeout) || probe_tcp("8.8.8.8", 443, timeout)
    };

    let ollama_reachable = probe_url(&config.ollama_base_url, 11434, timeout);
    let whisper_reachable = probe_url(&config.whisper_base_url, 8090, timeout);

    ProbeReport {
        wan_reachable,
        ollama_reachable,
        whisper_reachable,
        elapsed_ms: started.elapsed().as_millis() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_url_detection() {
        assert!(is_loopback_url("http://127.0.0.1:11434"));
        assert!(is_loopback_url("http://localhost:8090/inference"));
        assert!(is_loopback_url("http://[::1]:9000"));
        assert!(!is_loopback_url("https://api.openai.com/v1"));
        assert!(!is_loopback_url("https://api.anthropic.com"));
    }

    #[test]
    fn loopback_host_detection() {
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("::1"));
        assert!(!is_loopback_host("api.openai.com"));
    }

    #[test]
    fn parse_host_port_variants() {
        assert_eq!(
            parse_host_port("http://127.0.0.1:11434", 80),
            Some(("127.0.0.1".into(), 11434))
        );
        assert_eq!(
            parse_host_port("http://localhost/api", 11434),
            Some(("localhost".into(), 11434))
        );
        assert_eq!(
            parse_host_port("http://[::1]:8090", 80),
            Some(("::1".into(), 8090))
        );
    }

    #[test]
    fn force_offline_probe_never_marks_wan() {
        let cfg = OfflineConfig {
            force_offline: true,
            ..OfflineConfig::default()
        };
        let report = probe(&cfg);
        assert!(!report.wan_reachable);
        assert!(report.offline_wan());
    }
}
