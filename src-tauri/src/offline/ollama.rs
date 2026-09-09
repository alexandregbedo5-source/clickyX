//! Local Ollama client. Talks only to a loopback (or configured) Ollama daemon.

use serde::{Deserialize, Serialize};

use crate::offline::manager;
use crate::offline::network;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaTag {
    pub name: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaHealth {
    pub reachable: bool,
    pub version: Option<String>,
    pub base_url: String,
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Option<Vec<TagModel>>,
}

#[derive(Debug, Deserialize)]
struct TagModel {
    name: String,
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct VersionResponse {
    version: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    message: Option<ChatMessageBody>,
}

#[derive(Debug, Deserialize)]
struct ChatMessageBody {
    content: Option<String>,
}

fn base_url() -> String {
    manager::config().ollama_base_url.trim_end_matches('/').to_string()
}

fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| format!("ollama client: {e}"))
}

pub async fn health() -> OllamaHealth {
    let url = format!("{}/api/version", base_url());
    if !network::is_loopback_url(&url) && manager::blocks_wan() {
        return OllamaHealth {
            reachable: false,
            version: None,
            base_url: base_url(),
        };
    }
    match client() {
        Ok(client) => match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                let version = resp
                    .json::<VersionResponse>()
                    .await
                    .ok()
                    .and_then(|v| v.version);
                OllamaHealth {
                    reachable: true,
                    version,
                    base_url: base_url(),
                }
            }
            _ => OllamaHealth {
                reachable: false,
                version: None,
                base_url: base_url(),
            },
        },
        Err(_) => OllamaHealth {
            reachable: false,
            version: None,
            base_url: base_url(),
        },
    }
}

pub async fn list_tags() -> Result<Vec<OllamaTag>, String> {
    let url = format!("{}/api/tags", base_url());
    if !network::is_loopback_url(&url) && manager::blocks_wan() {
        return Ok(vec![]);
    }
    let resp = client()?
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Ollama /api/tags failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Ollama /api/tags HTTP {}", resp.status()));
    }
    let parsed: TagsResponse = resp.json().await.map_err(|e| e.to_string())?;
    Ok(parsed
        .models
        .unwrap_or_default()
        .into_iter()
        .map(|m| OllamaTag {
            name: m.name,
            size: m.size.unwrap_or(0),
        })
        .collect())
}

pub async fn pull_model(name: &str) -> Result<(), String> {
    let url = format!("{}/api/pull", base_url());
    // The daemon itself needs WAN to reach registry.ollama.com.
    manager::guard_wan(&format!("download Ollama model {name}"))?;
    let body = serde_json::json!({ "name": name, "stream": false });
    let resp = client()?
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Ollama pull failed: {e}"))?;
    if !resp.status().is_success() {
        let txt = resp.text().await.unwrap_or_default();
        return Err(format!("Ollama pull error: {txt}"));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalChatMessage {
    pub role: String,
    pub content: String,
}

pub async fn chat(messages: &[LocalChatMessage], model: &str) -> Result<String, String> {
    let url = format!("{}/api/chat", base_url());
    if !network::is_loopback_url(&url) && manager::blocks_wan() {
        return Err("Ollama base URL is not local and WAN is blocked".into());
    }
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": false
    });
    let resp = client()?
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Ollama chat failed: {e}"))?;
    if !resp.status().is_success() {
        let txt = resp.text().await.unwrap_or_default();
        return Err(format!("Ollama chat error: {txt}"));
    }
    let parsed: ChatResponse = resp.json().await.map_err(|e| e.to_string())?;
    parsed
        .message
        .and_then(|m| m.content)
        .ok_or_else(|| "Unexpected Ollama chat response".into())
}

pub async fn chat_stream<F>(
    messages: &[LocalChatMessage],
    model: &str,
    mut on_delta: F,
) -> Result<String, String>
where
    F: FnMut(&str),
{
    let url = format!("{}/api/chat", base_url());
    if !network::is_loopback_url(&url) && manager::blocks_wan() {
        return Err("Ollama base URL is not local and WAN is blocked".into());
    }
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true
    });
    let resp = client()?
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Ollama stream failed: {e}"))?;
    if !resp.status().is_success() {
        let txt = resp.text().await.unwrap_or_default();
        return Err(format!("Ollama stream error: {txt}"));
    }
    let text = resp.text().await.map_err(|e| e.to_string())?;
    let mut assembled = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(chunk) = value
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
            {
                if !chunk.is_empty() {
                    assembled.push_str(chunk);
                    on_delta(chunk);
                }
            }
        }
    }
    if assembled.is_empty() {
        Err("Ollama stream produced no content".into())
    } else {
        Ok(assembled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_serializes() {
        let msg = LocalChatMessage {
            role: "user".into(),
            content: "hello".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("hello"));
    }
}
