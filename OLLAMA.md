# Ollama Integration with ClickyX

ClickyX now supports **Ollama**, a local LLM runtime, as an AI provider. This allows you to run open-source models (Llama, Mistral, Neural-Chat, etc.) entirely on your machine without cloud API calls.

## Setup

### 1. Install & Run Ollama

Download Ollama from [ollama.com](https://ollama.com) and start the service:

```bash
ollama serve
```

Ollama runs on `http://localhost:11434` by default.

### 2. Pull a Model

```bash
ollama pull llama2      # ~3.8 GB
ollama pull mistral     # ~4.1 GB  
ollama pull neural-chat # ~4.1 GB
```

To see available models: https://ollama.ai/library

### 3. Configure ClickyX

1. Open **Settings** → **AI Providers**
2. Scroll to **Ollama (Local)** section
3. Set:
   - **Model**: `llama2` (or your chosen model)
   - **Base URL**: `http://localhost:11434` (or your Ollama server address)
4. Select **Default Provider** → `Ollama (Local)`
5. Click **Save Settings**

### 4. Test

Try asking a question in ClickyX. Ollama will handle the request locally.

## Supported Models

Any model available on Ollama's library works with ClickyX:

- **llama2**: Meta's Llama 2 (7B-70B variants)
- **mistral**: Mistral 7B (highly recommended for fast inference)
- **neural-chat**: Intel's fine-tuned model
- **orca**: Open-source reasoning model
- **dolphin**: Uncensored models

Query available models with:
```bash
ollama list
```

## Performance Notes

- **System Requirements**: Minimum 4 GB RAM; 8+ GB recommended
- **Speed**: Depends on model size and GPU availability
- **GPU Acceleration**: Enable if your GPU supports it (NVIDIA/AMD)
- **Response Time**: Local inference is slower than cloud APIs but fully private

## Architecture

```
ClickyX Frontend (React/Tauri)
    ↓
OllamaProvider (Rust backend)
    ↓
HTTP POST: /api/chat
    ↓
Ollama Server (localhost:11434)
    ↓
Local LLM Model
```

The OllamaProvider sends chat messages to Ollama's `/api/chat` endpoint and parses responses.

## Troubleshooting

**"Connection refused"**
- Ensure Ollama is running: `ollama serve`
- Verify base URL matches your setup (default: `http://localhost:11434`)

**"Model not found"**
- Pull the model: `ollama pull <model_name>`
- Check with `ollama list`

**Slow responses**
- Try smaller models (mistral 7B is faster than llama2 70B)
- Enable GPU acceleration if available
- Check system resource usage

## More Resources

- Ollama Documentation: https://github.com/ollama/ollama
- Model Library: https://ollama.ai/library
- API Reference: https://github.com/ollama/ollama/blob/main/docs/api.md
