import { getOfflineManager } from "../offline/OfflineManager";
import type { OfflineConfig, ProviderDescriptor, ProviderKind } from "../offline/types";

export class ProviderRegistry {
  constructor(private readonly config: OfflineConfig) {}

  all(): ProviderDescriptor[] {
    const offline = getOfflineManager();
    const blocksWan = offline.blocksWan();
    return [
      {
        id: "ollama",
        name: "Ollama (local LLM)",
        kind: "llm",
        requires_wan: false,
        endpoint: this.config.ollama_base_url,
        available: offline.status().ollama_reachable,
        notes: "Chat via localhost:11434",
      },
      {
        id: "whisper-local",
        name: "Whisper local (STT)",
        kind: "stt",
        requires_wan: false,
        endpoint: this.config.whisper_base_url,
        available: offline.status().whisper_reachable,
        notes: "whisper.cpp HTTP or CLI",
      },
      {
        id: "system",
        name: "System TTS",
        kind: "tts",
        requires_wan: false,
        endpoint: null,
        available: true,
        notes: "OS voice — no network",
      },
      {
        id: "anthropic",
        name: "Anthropic",
        kind: "llm",
        requires_wan: true,
        endpoint: "https://api.anthropic.com",
        available: !blocksWan,
        notes: "Cloud — blocked while offline",
      },
      {
        id: "openai",
        name: "OpenAI / compatible",
        kind: "llm",
        requires_wan: true,
        endpoint: "https://api.openai.com",
        available: !blocksWan,
        notes: "Cloud — blocked while offline unless base URL is localhost",
      },
    ];
  }

  localOnly(): ProviderDescriptor[] {
    return this.all().filter((provider) => !provider.requires_wan);
  }

  ofKind(kind: ProviderKind): ProviderDescriptor[] {
    return this.all().filter((provider) => provider.kind === kind);
  }

  preferred(kind: ProviderKind): string {
    if (kind === "llm") return "ollama";
    if (kind === "stt") return "whisper-local";
    return "system";
  }
}

export function createProviderRegistry(config: OfflineConfig): ProviderRegistry {
  return new ProviderRegistry(config);
}
