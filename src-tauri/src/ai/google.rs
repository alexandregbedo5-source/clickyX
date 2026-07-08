use crate::ai::{AiError, AiProvider, ChatMessage, ImageInput};
use reqwest::Client;

pub struct GoogleProvider {
    api_key: String,
    system_prompt: String,
    base_url: String,
}

impl GoogleProvider {
    pub fn new(api_key: String, system_prompt: String, base_url: String) -> Self {
        Self { api_key, system_prompt, base_url }
    }
}

#[async_trait::async_trait]
impl AiProvider for GoogleProvider {
    async fn chat(&self, messages: &[ChatMessage], model: &str) -> Result<String, AiError> {
        // Concatenate messages into a single prompt: system + user/assistant turns
        let mut prompt = self.system_prompt.clone();
        for m in messages {
            prompt.push_str("\n");
            prompt.push_str(&format!("{}: {}", m.role, m.content));
        }

        let client = Client::new();
        let url = format!("{}/v1beta2/models/{}:generateText?key={}", self.base_url.trim_end_matches('/'), model, self.api_key);
        let body = serde_json::json!({
            "prompt": { "text": prompt },
            "maxOutputTokens": 1024
        });

        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AiError::Network(e.to_string()))?;

        if !resp.status().is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(AiError::Api(format!("Google API error: {}", txt)));
        }

        let json: serde_json::Value = resp.json().await.map_err(|e| AiError::Decode(e.to_string()))?;
        // Expecting candidates[0].output
        if let Some(output) = json.get("candidates").and_then(|c| c.get(0)).and_then(|c0| c0.get("output")).and_then(|o| o.as_str()) {
            Ok(output.to_string())
        } else if let Some(output) = json.get("candidates").and_then(|c| c.get(0)).and_then(|c0| c0.get("content")).and_then(|o| o.as_str()) {
            Ok(output.to_string())
        } else {
            Err(AiError::Decode("Unexpected Google response format".into()))
        }
    }

    async fn chat_stream(&self, _messages: &[ChatMessage], _model: &str) -> Result<crate::ai::streaming::StreamReceiver, AiError> {
        Err(AiError::Config("Google streaming not implemented".into()))
    }

    async fn chat_with_vision(&self, _messages: &[ChatMessage], _model: &str, _images: &[ImageInput]) -> Result<String, AiError> {
        Err(AiError::Config("Google vision chat not implemented".into()))
    }
}
