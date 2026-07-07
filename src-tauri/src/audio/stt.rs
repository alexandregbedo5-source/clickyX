use base64::Engine;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SttProvider {
    Deepgram,
    OpenAIWhisper,
    AssemblyAI,
}

impl SttProvider {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "deepgram" => Some(Self::Deepgram),
            "whisper" | "openai" => Some(Self::OpenAIWhisper),
            "assemblyai" => Some(Self::AssemblyAI),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Deepgram => "deepgram",
            Self::OpenAIWhisper => "openai",
            Self::AssemblyAI => "assemblyai",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SttConfig {
    pub provider: SttProvider,
    pub api_key: String,
    pub base_url: Option<String>,
    pub language: String,
    pub timeout_secs: u64,
    pub max_retries: u32,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            provider: SttProvider::Deepgram,
            api_key: String::new(),
            base_url: None,
            language: "en".into(),
            timeout_secs: 30,
            max_retries: 3,
        }
    }
}

fn pcm_to_wav(pcm_data: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::new(&mut cursor, spec).map_err(|e| format!("WAV writer error: {e}"))?;
    for &sample in pcm_data {
        let clamped = sample.clamp(-1.0, 1.0);
        let sample_i16 = (clamped * i16::MAX as f32) as i16;
        writer
            .write_sample(sample_i16)
            .map_err(|e| format!("WAV write error: {e}"))?;
    }
    writer
        .finalize()
        .map_err(|e| format!("WAV finalize error: {e}"))?;
    Ok(cursor.into_inner())
}

pub async fn transcribe(
    audio_data: &[f32],
    config: &SttConfig,
    sample_rate: u32,
) -> Result<String, String> {
    if config.api_key.is_empty() && !config.base_url.as_ref().map_or(false, |u| u.contains("localhost") || u.contains("127.0.0.1")) {
        return Err(format!("No API key for provider {}", config.provider.name()));
    }

    let wav_bytes = pcm_to_wav(audio_data, sample_rate)?;

    match config.provider {
        SttProvider::Deepgram => transcribe_deepgram(&wav_bytes, config).await,
        SttProvider::OpenAIWhisper => transcribe_whisper(&wav_bytes, config).await,
        SttProvider::AssemblyAI => transcribe_assemblyai(&wav_bytes, config).await,
    }
}

async fn transcribe_deepgram(wav_data: &[u8], config: &SttConfig) -> Result<String, String> {
    let client = reqwest::Client::new();
    let url = format!(
        "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&language={}",
        config.language
    );

    let mut last_error = String::new();
    for attempt in 0..config.max_retries {
        let result = client
            .post(&url)
            .header("Authorization", format!("Token {}", config.api_key))
            .header("Content-Type", "audio/wav")
            .body(wav_data.to_vec())
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .send()
            .await;

        match result {
            Ok(resp) => {
                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    last_error = format!("Deepgram HTTP {}: {}", status, body);
                    continue;
                }
                let json: serde_json::Value = resp.json().await.unwrap();
                return Ok(json["results"]["channels"][0]["alternatives"][0]["transcript"].as_str().unwrap().to_string());
            }
            Err(e) => last_error = e.to_string(),
        }
    }
    Err(last_error)
}

async fn transcribe_whisper(wav_data: &[u8], config: &SttConfig) -> Result<String, String> {
    let client = reqwest::Client::new();
    let base_url = config.base_url.as_deref().unwrap_or("https://api.openai.com").trim_end_matches('/');
    let url = format!("{}/v1/audio/transcriptions", base_url);

    let mut last_error = String::new();
    for attempt in 0..config.max_retries {
        let form = reqwest::multipart::Form::new()
            .part("file", reqwest::multipart::Part::bytes(wav_data.to_vec()).file_name("audio.wav").mime_str("audio/wav").unwrap())
            .text("model", "whisper-1")
            .text("language", config.language.clone());

        let result = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", config.api_key))
            .multipart(form)
            .send()
            .await;

        match result {
            Ok(resp) => {
                if !resp.status().is_success() {
                    last_error = resp.text().await.unwrap_or_default();
                    continue;
                }
                let json: serde_json::Value = resp.json().await.unwrap();
                return Ok(json["text"].as_str().unwrap_or("").to_string());
            }
            Err(e) => last_error = e.to_string(),
        }
    }
    Err(last_error)
}

async fn transcribe_assemblyai(_wav_data: &[u8], _config: &SttConfig) -> Result<String, String> {
    Err("AssemblyAI local non supporté".into())
}