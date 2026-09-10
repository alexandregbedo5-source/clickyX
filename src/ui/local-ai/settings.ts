import { DEFAULT_OFFLINE_CONFIG, DEFAULT_UI_SETTINGS, OFFLINE_OVERLAY_KEY, UI_SETTINGS_KEY } from "./constants";
import type { LocalAiUiSettings, OfflineConfig } from "./types";

function canUseStorage(): boolean {
  return typeof localStorage !== "undefined";
}

export function loadUiSettings(): LocalAiUiSettings {
  if (!canUseStorage()) return { ...DEFAULT_UI_SETTINGS };
  try {
    const raw = localStorage.getItem(UI_SETTINGS_KEY);
    if (!raw) return { ...DEFAULT_UI_SETTINGS };
    const parsed = JSON.parse(raw) as Partial<LocalAiUiSettings>;
    return {
      ...DEFAULT_UI_SETTINGS,
      ...parsed,
      detectorBaseUrl: (parsed.detectorBaseUrl || DEFAULT_UI_SETTINGS.detectorBaseUrl).replace(/\/+$/, ""),
      historyLimit: clampLimit(parsed.historyLimit ?? DEFAULT_UI_SETTINGS.historyLimit),
    };
  } catch {
    return { ...DEFAULT_UI_SETTINGS };
  }
}

export function saveUiSettings(partial: Partial<LocalAiUiSettings>): LocalAiUiSettings {
  const next = {
    ...loadUiSettings(),
    ...partial,
  };
  next.detectorBaseUrl = next.detectorBaseUrl.replace(/\/+$/, "");
  next.historyLimit = clampLimit(next.historyLimit);
  if (canUseStorage()) {
    localStorage.setItem(UI_SETTINGS_KEY, JSON.stringify(next));
  }
  return next;
}

export function resetUiSettings(): LocalAiUiSettings {
  if (canUseStorage()) {
    localStorage.removeItem(UI_SETTINGS_KEY);
  }
  return { ...DEFAULT_UI_SETTINGS };
}

function clampLimit(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_UI_SETTINGS.historyLimit;
  return Math.min(200, Math.max(5, Math.round(value)));
}

export function loadOfflineOverlay(): Partial<OfflineConfig> {
  if (!canUseStorage()) return {};
  try {
    const raw = localStorage.getItem(OFFLINE_OVERLAY_KEY);
    return raw ? (JSON.parse(raw) as Partial<OfflineConfig>) : {};
  } catch {
    return {};
  }
}

export function saveOfflineOverlay(partial: Partial<OfflineConfig>): OfflineConfig {
  const next = { ...DEFAULT_OFFLINE_CONFIG, ...loadOfflineOverlay(), ...partial };
  if (canUseStorage()) {
    localStorage.setItem(OFFLINE_OVERLAY_KEY, JSON.stringify(next));
  }
  return next;
}
