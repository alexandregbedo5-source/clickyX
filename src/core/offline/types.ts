export type Connectivity = "online" | "local_only" | "offline";

export type ProviderKind = "llm" | "stt" | "tts";

export type ModelKind = "llm" | "whisper";

export interface OfflineStatus {
  connectivity: Connectivity;
  force_offline: boolean;
  auto_fallback: boolean;
  wan_reachable: boolean;
  ollama_reachable: boolean;
  whisper_reachable: boolean;
  blocks_wan: boolean;
  last_probe_ms: number;
  data_dir: string;
}

export interface OfflineConfig {
  force_offline: boolean;
  auto_fallback: boolean;
  ollama_base_url: string;
  whisper_base_url: string;
  whisper_cli: string | null;
  whisper_model: string;
  default_llm: string;
  probe_timeout_ms: number;
}

export interface ProviderDescriptor {
  id: string;
  name: string;
  kind: ProviderKind;
  requires_wan: boolean;
  endpoint: string | null;
  available: boolean;
  notes: string;
}

export interface LocalModel {
  id: string;
  display_name: string;
  kind: ModelKind;
  provider: string;
  installed: boolean;
  path: string | null;
  size_bytes: number;
  download_url: string | null;
  notes: string;
}

export interface ModelInventory {
  models: LocalModel[];
  data_dir: string;
}

export function defaultOfflineConfig(): OfflineConfig {
  return {
    force_offline: readEnvFlag("CLICKYX_OFFLINE") || readEnvFlag("CLICKYX_FORCE_OFFLINE"),
    auto_fallback: true,
    ollama_base_url: readEnv("CLICKYX_OLLAMA_URL") ?? "http://127.0.0.1:11434",
    whisper_base_url: readEnv("CLICKYX_WHISPER_URL") ?? "http://127.0.0.1:8090",
    whisper_cli: readEnv("CLICKYX_WHISPER_CLI"),
    whisper_model: readEnv("CLICKYX_WHISPER_MODEL") ?? "base",
    default_llm: readEnv("CLICKYX_OLLAMA_MODEL") ?? "llama3.2",
    probe_timeout_ms: 400,
  };
}

export function isLoopbackUrl(url: string): boolean {
  const lower = url.toLowerCase();
  return (
    lower.includes("://127.0.0.1") ||
    lower.includes("://localhost") ||
    lower.includes("://[::1]") ||
    lower.startsWith("127.0.0.1") ||
    lower.startsWith("localhost")
  );
}

export function readEnvFlag(name: string): boolean {
  const value = readEnv(name);
  if (!value) return false;
  return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
}

function readEnv(name: string): string | null {
  try {
    const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process
      ?.env;
    const value = env?.[name];
    return value && value.length > 0 ? value : null;
  } catch {
    return null;
  }
}
