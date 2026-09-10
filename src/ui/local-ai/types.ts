/**
 * Local AI UI contracts.
 * Mirrors POST /detect-ai-image (ai_detector) and GET /offline/status
 * without importing engine or ML code.
 */

export type UnitScore = number;

export type DetectAiImageRequest =
  | {
      image_path: string;
      image_base64?: never;
      include_details?: boolean;
    }
  | {
      image_path?: never;
      image_base64: string;
      include_details?: boolean;
    };

export interface DetectAiImageResultV1 {
  is_ai_generated: boolean;
  confidence: UnitScore;
  fft_score: UnitScore;
  noise_score: UnitScore;
  cnn_score: UnitScore;
}

export interface DetectAiImageResponse extends DetectAiImageResultV1 {
  contract_version?: string;
  model_version?: string | null;
  warnings?: string[];
  details?: DetectionDetails | null;
}

export interface DetectionDetails {
  image: { width: number; height: number; format: string | null; has_exif: boolean; source: string };
  frequency: ModuleDetails;
  noise: ModuleDetails;
  cnn: {
    available: boolean;
    trained: boolean;
    arch: string | null;
    model_version: string | null;
    crops: number;
    crop_scores: number[];
    padded: boolean;
  };
  fusion: {
    mode: "full" | "handcrafted";
    threshold: number;
    weights: Record<string, number>;
    logit_contributions: Record<string, number>;
    verdict_confidence: UnitScore;
  };
  timings_ms: Record<string, number>;
  calibration_version: string;
}

export interface ModuleDetails {
  features: Record<string, number | null>;
  contributions: Record<string, number>;
}

export interface DetectorErrorBody {
  error: {
    code: string;
    message: string;
  };
}

export interface DetectorHealth {
  status: "ok" | "degraded" | "unreachable";
  service?: string;
  version?: string;
  contract_version?: string;
  model?: {
    loaded: boolean;
    trained: boolean;
    path: string | null;
    arch: string | null;
    version: string | null;
    error: string | null;
  };
  fusion_mode?: "full" | "handcrafted";
  message?: string;
}

export type Connectivity = "online" | "local_only" | "offline";

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
  /** Where the snapshot came from — engine, bridge, or browser fallback. */
  source?: "engine" | "bridge" | "browser";
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
  kind: "llm" | "stt" | "tts";
  requires_wan: boolean;
  endpoint: string | null;
  available: boolean;
  notes: string;
}

export interface LocalModel {
  id: string;
  display_name: string;
  kind: "llm" | "whisper";
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

export interface LocalAiUiSettings {
  detectorBaseUrl: string;
  detectorToken: string;
  includeDetails: boolean;
  autoSaveHistory: boolean;
  historyLimit: number;
  showScoreBreakdown: boolean;
}

export interface DetectionHistoryEntry {
  id: string;
  createdAt: number;
  fileName: string;
  thumbnail: string | null;
  is_ai_generated: boolean;
  confidence: number;
  fft_score: number | null;
  noise_score: number | null;
  cnn_score: number | null;
  warnings: string[];
  result: DetectAiImageResponse;
}

export type LocalAiScreen = "detect" | "results" | "history" | "settings" | "help";
