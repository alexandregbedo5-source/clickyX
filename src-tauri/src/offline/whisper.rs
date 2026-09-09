//! Local Whisper integration — HTTP (whisper.cpp / OpenAI-compatible) or CLI.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use crate::offline::manager;
use crate::offline::models::ModelManager;
use crate::offline::network;
use crate::offline::storage;

#[derive(Debug, Clone)]
pub struct WhisperResult {
    pub text: String,
    pub backend: String,
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

pub async fn transcribe_pcm(
    pcm_data: &[f32],
    sample_rate: u32,
    language: &str,
) -> Result<String, String> {
    let wav = pcm_to_wav(pcm_data, sample_rate)?;
    transcribe_wav(&wav, language).await.map(|r| r.text)
}

pub async fn transcribe_wav(wav: &[u8], language: &str) -> Result<WhisperResult, String> {
    let cfg = manager::config();

    if network::probe_url(&cfg.whisper_base_url, 8090, std::time::Duration::from_millis(250)) {
        if let Ok(text) = transcribe_http(wav, language, &cfg.whisper_base_url).await {
            return Ok(WhisperResult {
                text,
                backend: "whisper-http".into(),
            });
        }
    }

    if let Ok(text) = transcribe_cli(wav, language) {
        return Ok(WhisperResult {
            text,
            backend: "whisper-cli".into(),
        });
    }

    Err(format!(
        "Local Whisper is unavailable. Start whisper.cpp on {} or install whisper-cli, and place ggml-{}.bin in {}.",
        cfg.whisper_base_url,
        cfg.whisper_model,
        storage::whisper_models_dir().display()
    ))
}

async fn transcribe_http(wav: &[u8], language: &str, base_url: &str) -> Result<String, String> {
    let base = base_url.trim_end_matches('/');
    if !network::is_loopback_url(base) && manager::blocks_wan() {
        return Err("Whisper URL is not local and WAN is blocked".into());
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| e.to_string())?;

    // whisper.cpp server: POST /inference
    let inference = format!("{base}/inference");
    let form = reqwest::multipart::Form::new()
        .part(
            "file",
            reqwest::multipart::Part::bytes(wav.to_vec())
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .map_err(|e| e.to_string())?,
        )
        .text("language", language.to_string())
        .text("response_format", "json".to_string());

    if let Ok(resp) = client.post(&inference).multipart(form).send().await {
        if resp.status().is_success() {
            let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
            if let Some(text) = json
                .get("text")
                .and_then(|v| v.as_str())
                .or_else(|| json.get("transcription").and_then(|v| v.as_str()))
            {
                return Ok(text.trim().to_string());
            }
        }
    }

    // OpenAI-compatible local endpoint
    let openai_url = format!("{base}/v1/audio/transcriptions");
    let form = reqwest::multipart::Form::new()
        .part(
            "file",
            reqwest::multipart::Part::bytes(wav.to_vec())
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .map_err(|e| e.to_string())?,
        )
        .text("model", "whisper-1".to_string())
        .text("language", language.to_string());

    let resp = client
        .post(&openai_url)
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("Whisper HTTP failed: {e}"))?;
    if !resp.status().is_success() {
        let txt = resp.text().await.unwrap_or_default();
        return Err(format!("Whisper HTTP error: {txt}"));
    }
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(json
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string())
}

fn transcribe_cli(wav: &[u8], language: &str) -> Result<String, String> {
    let cfg = manager::config();
    let bin = resolve_cli(&cfg.whisper_cli)?;
    let model_path = ModelManager::new()
        .whisper_model_path(&cfg.whisper_model)
        .ok_or_else(|| {
            format!(
                "No local Whisper model '{}'. Copy ggml-{}.bin into {}",
                cfg.whisper_model,
                cfg.whisper_model,
                storage::whisper_models_dir().display()
            )
        })?;

    let tmp = write_temp_wav(wav)?;
    let output = Command::new(&bin)
        .arg("-m")
        .arg(&model_path)
        .arg("-f")
        .arg(&tmp)
        .arg("-l")
        .arg(language)
        .arg("-nt")
        .output()
        .map_err(|e| format!("failed to spawn {bin}: {e}"))?;
    let _ = std::fs::remove_file(&tmp);
    if !output.status.success() {
        return Err(format!(
            "whisper CLI failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        Err("whisper CLI produced empty transcript".into())
    } else {
        Ok(stdout)
    }
}

fn resolve_cli(configured: &Option<String>) -> Result<String, String> {
    let mut candidates = Vec::new();
    if let Some(custom) = configured {
        candidates.push(custom.clone());
    }
    candidates.extend(
        ["whisper-cli", "whisper-cpp", "whisper"]
            .into_iter()
            .map(|s| s.to_string()),
    );
    for name in candidates {
        let path = PathBuf::from(&name);
        if path.is_file() {
            return Ok(name);
        }
        let which = if cfg!(windows) { "where" } else { "which" };
        if Command::new(which)
            .arg(&name)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return Ok(name);
        }
    }
    Err("No whisper CLI found (whisper-cli / whisper-cpp / whisper)".into())
}

fn write_temp_wav(wav: &[u8]) -> Result<PathBuf, String> {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("clickyx-whisper-{}.wav", std::process::id()));
    let mut file = std::fs::File::create(&path).map_err(|e| e.to_string())?;
    file.write_all(wav).map_err(|e| e.to_string())?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_to_wav_writes_header() {
        let silence = vec![0.0_f32; 160];
        let wav = pcm_to_wav(&silence, 16000).unwrap();
        assert!(wav.len() > 44);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
    }
}
