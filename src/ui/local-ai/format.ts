import type { Connectivity, DetectAiImageResponse } from "./types";

export function formatPercent(score: number | null | undefined): string {
  if (typeof score !== "number" || !Number.isFinite(score)) return "—";
  return `${Math.round(clamp01(score) * 100)}%`;
}

export function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function formatRelativeTime(timestamp: number, now = Date.now()): string {
  const delta = Math.max(0, now - timestamp);
  const minutes = Math.floor(delta / 60_000);
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(timestamp).toLocaleDateString();
}

export function verdictLabel(isAi: boolean): string {
  return isAi ? "Likely AI-generated" : "Likely a photograph";
}

export function verdictTone(result: Pick<DetectAiImageResponse, "is_ai_generated" | "confidence">): "ai" | "photo" | "uncertain" {
  if (result.confidence >= 0.4 && result.confidence <= 0.6) return "uncertain";
  return result.is_ai_generated ? "ai" : "photo";
}

export function connectivityLabel(connectivity: Connectivity): string {
  switch (connectivity) {
    case "online":
      return "Online";
    case "local_only":
      return "Local only";
    case "offline":
      return "Offline";
  }
}

export function connectivityHint(connectivity: Connectivity, ollama: boolean): string {
  if (connectivity === "online") return "Cloud providers are available.";
  if (connectivity === "local_only") {
    return ollama
      ? "No Internet — chat stays on local models."
      : "No Internet — install Ollama for local chat.";
  }
  return "No Internet and no local services detected.";
}

export function humanizeWarning(code: string): string {
  const known: Record<string, string> = {
    "cnn_indisponible:fusion_indices_physiques_seuls":
      "Neural model unavailable — verdict uses frequency and noise only.",
  };
  return known[code] ?? code.replace(/[_:]+/g, " ");
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 && unit > 0 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}
