/**
 * ClickyX — AI Image Detector : contrat TypeScript (miroir de ai_detector/schemas.py).
 *
 * Fichier autonome, à copier/importer côté interface (Honorat) ou bridge (Tybiane).
 * Il n'est référencé par aucun code React de ce dépôt : c'est une définition de contrat.
 *
 * Endpoint : POST http://127.0.0.1:32188/detect-ai-image
 * Version du contrat : 1.0 (champs de base figés) — extensions 1.1 additives et optionnelles.
 */

/** Probabilité ou score normalisé dans [0, 1]. */
export type UnitScore = number;

/** Entrée : exactement un des deux champs `image_path` / `image_base64`. */
export type DetectAiImageRequest =
  | {
      /** Chemin local absolu vers l'image (contrat v1). */
      image_path: string;
      image_base64?: never;
      /** v1.1 — inclure le bloc `details` (défaut : true). */
      include_details?: boolean;
    }
  | {
      image_path?: never;
      /** v1.1 — image encodée en base64 (data-URL acceptée). */
      image_base64: string;
      include_details?: boolean;
    };

/** Sortie minimale garantie (contrat v1). */
export interface DetectAiImageResultV1 {
  /** Verdict binaire : `confidence >= seuil` (0.5 par défaut). */
  is_ai_generated: boolean;
  /** Probabilité estimée que l'image soit générée par IA. */
  confidence: UnitScore;
  /** Module fréquentiel (FFT) : 0 = photo, 1 = synthèse. */
  fft_score: UnitScore;
  /** Module bruit résiduel. */
  noise_score: UnitScore;
  /** Réseau de neurones (0.5 = neutre si indisponible ou non entraîné). */
  cnn_score: UnitScore;
}

/** Sortie complète (v1 + extensions additives v1.1). */
export interface DetectAiImageResponse extends DetectAiImageResultV1 {
  contract_version?: string; // "1.0"
  model_version?: string | null;
  /** Codes stables, ex. "cnn_indisponible:fusion_indices_physiques_seuls". */
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
    weights: Record<"fft" | "noise" | "cnn", number> | Record<string, number>;
    logit_contributions: Record<string, number>;
    /** Confiance dans le verdict binaire : max(p, 1 − p). */
    verdict_confidence: UnitScore;
  };
  timings_ms: Record<"preprocessing" | "frequency" | "noise" | "cnn" | "fusion" | "total", number>;
  calibration_version: string;
}

export interface ModuleDetails {
  /** Caractéristiques physiques brutes (null = non mesurable sur cette image). */
  features: Record<string, number | null>;
  /** Contribution (logit) de chaque caractéristique au score du module. */
  contributions: Record<string, number>;
}

export interface ErrorResponse {
  error: {
    code:
      | "image_not_found"
      | "unsupported_image"
      | "permission_denied"
      | "unauthorized"
      | "http_error"
      | string;
    message: string;
  };
}

export interface HealthResponse {
  status: "ok" | "degraded";
  service: "clickyx-ai-detector";
  version: string;
  contract_version: string;
  model: {
    loaded: boolean;
    trained: boolean;
    path: string | null;
    arch: string | null;
    version: string | null;
    input_size: number | null;
    providers: string[];
    error: string | null;
  };
  calibration_version: string;
  fusion_mode: "full" | "handcrafted";
}

/** En-tête d'authentification optionnel (si AI_DETECTOR_TOKEN est défini côté serveur). */
export const AI_DETECTOR_TOKEN_HEADER = "x-ai-detector-token";
export const AI_DETECTOR_DEFAULT_BASE_URL = "http://127.0.0.1:32188";
