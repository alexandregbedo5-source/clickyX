import { HISTORY_KEY } from "./constants";
import { loadUiSettings } from "./settings";
import type { DetectAiImageResponse, DetectionHistoryEntry } from "./types";

function canUseStorage(): boolean {
  return typeof localStorage !== "undefined";
}

export function loadHistory(): DetectionHistoryEntry[] {
  if (!canUseStorage()) return [];
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as DetectionHistoryEntry[];
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item) => item && typeof item.id === "string");
  } catch {
    return [];
  }
}

export function saveHistory(entries: DetectionHistoryEntry[]): DetectionHistoryEntry[] {
  const limit = loadUiSettings().historyLimit;
  const trimmed = entries.slice(0, limit);
  if (canUseStorage()) {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
  }
  return trimmed;
}

export function addHistoryEntry(
  input: Omit<DetectionHistoryEntry, "id" | "createdAt">,
): DetectionHistoryEntry[] {
  const entry: DetectionHistoryEntry = {
    ...input,
    id: createHistoryId(),
    createdAt: Date.now(),
  };
  return saveHistory([entry, ...loadHistory()]);
}

export function removeHistoryEntry(id: string): DetectionHistoryEntry[] {
  return saveHistory(loadHistory().filter((item) => item.id !== id));
}

export function clearHistory(): DetectionHistoryEntry[] {
  if (canUseStorage()) {
    localStorage.removeItem(HISTORY_KEY);
  }
  return [];
}

export function historyFromResult(
  result: DetectAiImageResponse,
  fileName: string,
  thumbnail: string | null,
): Omit<DetectionHistoryEntry, "id" | "createdAt"> {
  return {
    fileName,
    thumbnail,
    is_ai_generated: result.is_ai_generated,
    confidence: result.confidence,
    fft_score: scoreOrNull(result.fft_score),
    noise_score: scoreOrNull(result.noise_score),
    cnn_score: scoreOrNull(result.cnn_score),
    warnings: result.warnings ?? [],
    result,
  };
}

function scoreOrNull(value: number | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function createHistoryId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `det_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}
