use crate::ai::{AiError, AiProvider, ChatMessage, ImageInput};
use reqwest::Client;

pub struct OllamaProvider {
    system_prompt: String,
    base_url: String,
}

impl OllamaProvider {
    pub fn new(system_prompt: String, base_url: String) -> Self {
        Self { system_prompt, base_url }
    }
}

#[async_trait::async_trait]
impl AiProvider for OllamaProvider {
    async fn chat(&self, messages: &[ChatMessage], model: &str) -> Result<String, AiError> {
        let mut ollama_messages = vec![ChatMessage {
            role: "system".to_string(),
            content: self.system_prompt.clone(),
        }];
        ollama_messages.extend_from_slice(messages);

        let client = Client::new();
        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": model,
            "messages": ollama_messages,
            "stream": false
        });

        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AiError::Network(format!("Ollama connection failed: {}", e)))?;

        if !resp.status().is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(AiError::Api(format!("Ollama API error: {}", txt)));
        }

        let json: serde_json::Value = resp.json().await.map_err(|e| AiError::Decode(e.to_string()))?;
        if let Some(content) = json.get("message").and_then(|m| m.get("content")).and_then(|c| c.as_str()) {
            Ok(content.to_string())
        } else {
            Err(AiError::Decode("Unexpected Ollama response format".into()))
        }
    }

    async fn chat_stream(&self, _messages: &[ChatMessage], _model: &str) -> Result<crate::ai::streaming::StreamReceiver, AiError> {
        Err(AiError::Config("Ollama streaming not yet implemented".into()))
    }

    async fn chat_with_vision(&self, _messages: &[ChatMessage], _model: &str, _images: &[ImageInput]) -> Result<String, AiError> {
        Err(AiError::Config("Ollama vision not yet implemented".into()))
    }
}
