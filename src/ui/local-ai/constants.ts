import type { LocalAiUiSettings, OfflineConfig } from "./types";

export const AI_DETECTOR_DEFAULT_BASE_URL = "http://127.0.0.1:32188";
export const AI_DETECTOR_TOKEN_HEADER = "x-ai-detector-token";
export const BRIDGE_DEFAULT_BASE_URL = "http://127.0.0.1:32123";

export const UI_SETTINGS_KEY = "clickyx.local-ai.ui-settings";
export const OFFLINE_OVERLAY_KEY = "clickyx.local-ai.offline-config";
export const HISTORY_KEY = "clickyx.local-ai.history";

export const ACCEPTED_IMAGE_TYPES = [
  "image/png",
  "image/jpeg",
  "image/jpg",
  "image/webp",
  "image/bmp",
  "image/gif",
];

export const ACCEPTED_IMAGE_EXT = ".png,.jpg,.jpeg,.webp,.bmp,.gif";
export const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
export const DETECT_TIMEOUT_MS = 90_000;
export const THUMBNAIL_SIZE = 72;

export const DEFAULT_UI_SETTINGS: LocalAiUiSettings = {
  detectorBaseUrl: AI_DETECTOR_DEFAULT_BASE_URL,
  detectorToken: "",
  includeDetails: true,
  autoSaveHistory: true,
  historyLimit: 40,
  showScoreBreakdown: true,
};

export const DEFAULT_OFFLINE_CONFIG: OfflineConfig = {
  force_offline: false,
  auto_fallback: true,
  ollama_base_url: "http://127.0.0.1:11434",
  whisper_base_url: "http://127.0.0.1:8090",
  whisper_cli: null,
  whisper_model: "base",
  default_llm: "llama3.2",
  probe_timeout_ms: 400,
};

export const ANALYSIS_STAGES = [
  { id: "read", label: "Reading image…", until: 0.12 },
  { id: "preprocess", label: "Preprocessing…", until: 0.28 },
  { id: "fft", label: "Frequency analysis…", until: 0.48 },
  { id: "noise", label: "Residual noise…", until: 0.68 },
  { id: "cnn", label: "Neural network…", until: 0.88 },
  { id: "fusion", label: "Fusing scores…", until: 0.97 },
] as const;
