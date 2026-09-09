import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { detectAiImage, getDetectorHealth, getOfflineStatus } from "./api";
import { ANALYSIS_STAGES } from "./constants";
import {
  addHistoryEntry,
  clearHistory as clearStoredHistory,
  historyFromResult,
  loadHistory,
  removeHistoryEntry,
} from "./history";
import { makeThumbnail, readFileAsDataUrl, validateImageFile } from "./image";
import { loadUiSettings, saveUiSettings } from "./settings";
import type {
  DetectAiImageResponse,
  DetectionHistoryEntry,
  DetectorHealth,
  LocalAiScreen,
  LocalAiUiSettings,
  OfflineStatus,
} from "./types";

interface LocalAiCtx {
  screen: LocalAiScreen;
  setScreen: (screen: LocalAiScreen) => void;
  file: File | null;
  previewUrl: string | null;
  pickFile: (file: File) => Promise<string | null>;
  clearFile: () => void;
  analyze: () => Promise<void>;
  analyzing: boolean;
  progress: number;
  progressLabel: string;
  result: DetectAiImageResponse | null;
  error: string | null;
  history: DetectionHistoryEntry[];
  reloadHistory: () => void;
  removeHistory: (id: string) => void;
  clearHistory: () => void;
  openHistoryEntry: (entry: DetectionHistoryEntry) => void;
  uiSettings: LocalAiUiSettings;
  updateUiSettings: (partial: Partial<LocalAiUiSettings>) => LocalAiUiSettings;
  health: DetectorHealth | null;
  refreshHealth: () => Promise<void>;
  offline: OfflineStatus | null;
  refreshOffline: () => Promise<void>;
}

const LocalAiContext = createContext<LocalAiCtx | null>(null);

export function useLocalAi(): LocalAiCtx {
  const ctx = useContext(LocalAiContext);
  if (!ctx) throw new Error("useLocalAi must be used inside LocalAiProvider");
  return ctx;
}

interface ProviderProps {
  children: ReactNode;
  initialScreen?: LocalAiScreen;
}

export function LocalAiProvider({ children, initialScreen = "detect" }: ProviderProps) {
  const [screen, setScreen] = useState<LocalAiScreen>(initialScreen);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState("Waiting…");
  const [result, setResult] = useState<DetectAiImageResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<DetectionHistoryEntry[]>(() => loadHistory());
  const [uiSettings, setUiSettings] = useState<LocalAiUiSettings>(() => loadUiSettings());
  const [health, setHealth] = useState<DetectorHealth | null>(null);
  const [offline, setOffline] = useState<OfflineStatus | null>(null);
  const progressTimer = useRef<number | null>(null);

  const stopProgress = useCallback(() => {
    if (progressTimer.current != null) {
      window.clearInterval(progressTimer.current);
      progressTimer.current = null;
    }
  }, []);

  const startProgress = useCallback(() => {
    stopProgress();
    setProgress(0.04);
    setProgressLabel(ANALYSIS_STAGES[0].label);
    const started = Date.now();
    progressTimer.current = window.setInterval(() => {
      const elapsed = Date.now() - started;
      const fake = Math.min(0.97, 1 - Math.exp(-elapsed / 8_000));
      const stage = [...ANALYSIS_STAGES].reverse().find((item) => fake >= item.until - 0.2) ?? ANALYSIS_STAGES[0];
      setProgress(fake);
      setProgressLabel(stage.label);
    }, 120);
  }, [stopProgress]);

  const refreshHealth = useCallback(async () => {
    setHealth(await getDetectorHealth());
  }, []);

  const refreshOffline = useCallback(async () => {
    setOffline(await getOfflineStatus());
  }, []);

  useEffect(() => {
    void refreshHealth();
    void refreshOffline();
    return () => stopProgress();
  }, [refreshHealth, refreshOffline, stopProgress]);

  const pickFile = useCallback(async (next: File) => {
    const invalid = validateImageFile(next);
    if (invalid) {
      setError(invalid);
      return invalid;
    }
    const dataUrl = await readFileAsDataUrl(next);
    setFile(next);
    setPreviewUrl(dataUrl);
    setResult(null);
    setError(null);
    setScreen("detect");
    return null;
  }, []);

  const clearFile = useCallback(() => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setProgress(0);
  }, []);

  const analyze = useCallback(async () => {
    if (!file || !previewUrl || analyzing) return;
    setAnalyzing(true);
    setError(null);
    startProgress();
    try {
      const detection = await detectAiImage({
        image_base64: previewUrl,
        include_details: uiSettings.includeDetails,
      });
      stopProgress();
      setProgress(1);
      setProgressLabel("Done");
      setResult(detection);
      if (uiSettings.autoSaveHistory) {
        const thumbnail = await makeThumbnail(previewUrl);
        setHistory(addHistoryEntry(historyFromResult(detection, file.name, thumbnail)));
      }
      setScreen("results");
    } catch (err) {
      stopProgress();
      setProgress(0);
      setError(err instanceof Error ? err.message : "Detection failed.");
    } finally {
      setAnalyzing(false);
    }
  }, [analyzing, file, previewUrl, startProgress, stopProgress, uiSettings.autoSaveHistory, uiSettings.includeDetails]);

  const reloadHistory = useCallback(() => {
    setHistory(loadHistory());
  }, []);

  const removeHistory = useCallback((id: string) => {
    setHistory(removeHistoryEntry(id));
  }, []);

  const clearHistory = useCallback(() => {
    setHistory(clearStoredHistory());
  }, []);

  const openHistoryEntry = useCallback((entry: DetectionHistoryEntry) => {
    setResult(entry.result);
    setPreviewUrl(entry.thumbnail);
    setFile(null);
    setError(null);
    setScreen("results");
  }, []);

  const updateUiSettings = useCallback((partial: Partial<LocalAiUiSettings>) => {
    const next = saveUiSettings(partial);
    setUiSettings(next);
    return next;
  }, []);

  const value = useMemo<LocalAiCtx>(
    () => ({
      screen,
      setScreen,
      file,
      previewUrl,
      pickFile,
      clearFile,
      analyze,
      analyzing,
      progress,
      progressLabel,
      result,
      error,
      history,
      reloadHistory,
      removeHistory,
      clearHistory,
      openHistoryEntry,
      uiSettings,
      updateUiSettings,
      health,
      refreshHealth,
      offline,
      refreshOffline,
    }),
    [
      analyze,
      analyzing,
      clearFile,
      clearHistory,
      error,
      file,
      health,
      history,
      offline,
      openHistoryEntry,
      pickFile,
      previewUrl,
      progress,
      progressLabel,
      refreshHealth,
      refreshOffline,
      reloadHistory,
      removeHistory,
      result,
      screen,
      uiSettings,
      updateUiSettings,
    ],
  );

  return <LocalAiContext.Provider value={value}>{children}</LocalAiContext.Provider>;
}
