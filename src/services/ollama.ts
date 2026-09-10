import { isLoopbackUrl } from "../core/offline/types";

export interface OllamaTag {
  name: string;
  size: number;
}

export interface OllamaHealth {
  reachable: boolean;
  version: string | null;
  base_url: string;
}

export function ollamaEndpoints(baseUrl: string) {
  const base = baseUrl.replace(/\/$/, "");
  return {
    version: `${base}/api/version`,
    tags: `${base}/api/tags`,
    chat: `${base}/api/chat`,
    pull: `${base}/api/pull`,
  };
}

export function assertLocalOllama(baseUrl: string, wanBlocked: boolean): void {
  if (wanBlocked && !isLoopbackUrl(baseUrl)) {
    throw new Error("Ollama URL is not local and WAN is blocked");
  }
}

export function parseOllamaTags(payload: { models?: Array<{ name: string; size?: number }> }): OllamaTag[] {
  return (payload.models ?? []).map((model) => ({
    name: model.name,
    size: model.size ?? 0,
  }));
}
