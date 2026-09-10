import { DEFAULT_OFFLINE_CONFIG } from "./constants";
import { loadOfflineOverlay } from "./settings";

export interface OllamaChatMessage {
  role: string;
  content: string;
}

export interface StreamOllamaOptions {
  model?: string;
  baseUrl?: string;
  signal?: AbortSignal;
  onDelta: (text: string) => void;
}

function resolveBaseUrl(explicit?: string): string {
  return (explicit || loadOfflineOverlay().ollama_base_url || DEFAULT_OFFLINE_CONFIG.ollama_base_url).replace(
    /\/+$/,
    "",
  );
}

export function resolveOllamaModel(requested?: string | null): string {
  const overlay = loadOfflineOverlay();
  const fallback = overlay.default_llm || DEFAULT_OFFLINE_CONFIG.default_llm;
  if (!requested) return fallback;
  if (/^(claude|gpt-|o1|o3|gemini|text-bison)/i.test(requested)) return fallback;
  return requested;
}

export async function probeOllama(baseUrl?: string, timeoutMs = 1500): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${resolveBaseUrl(baseUrl)}/api/version`, { signal: controller.signal });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

export async function streamOllamaChat(
  messages: OllamaChatMessage[],
  options: StreamOllamaOptions,
): Promise<string> {
  const model = resolveOllamaModel(options.model);
  const url = `${resolveBaseUrl(options.baseUrl)}/api/chat`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: options.signal,
    body: JSON.stringify({ model, messages, stream: true }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      response.status === 404
        ? `Ollama model "${model}" is not installed. Run: ollama pull ${model}`
        : `Ollama error (${response.status}): ${detail || response.statusText}`,
    );
  }

  if (!response.body) {
    throw new Error("Ollama returned an empty stream.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let leftover = "";
  let full = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    leftover += decoder.decode(value, { stream: true });
    const lines = leftover.split("\n");
    leftover = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      let parsed: { message?: { content?: string }; error?: string; done?: boolean };
      try {
        parsed = JSON.parse(trimmed) as { message?: { content?: string }; error?: string; done?: boolean };
      } catch {
        continue;
      }
      if (parsed.error) throw new Error(parsed.error);
      const piece = parsed.message?.content ?? "";
      if (piece) {
        full += piece;
        options.onDelta(full);
      }
    }
  }

  return full;
}

export function preferLocalChat(): boolean {
  const overlay = loadOfflineOverlay();
  if (overlay.force_offline) return true;
  if (overlay.auto_fallback !== false && typeof navigator !== "undefined" && !navigator.onLine) {
    return true;
  }
  return false;
}
