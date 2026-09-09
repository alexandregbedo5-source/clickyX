import { invoke } from "../../bindings";
import {
  AI_DETECTOR_TOKEN_HEADER,
  BRIDGE_DEFAULT_BASE_URL,
  DEFAULT_OFFLINE_CONFIG,
  DETECT_TIMEOUT_MS,
} from "./constants";
import { loadOfflineOverlay, loadUiSettings, saveOfflineOverlay } from "./settings";
import type {
  DetectAiImageRequest,
  DetectAiImageResponse,
  DetectorHealth,
  LocalModel,
  ModelInventory,
  OfflineConfig,
  OfflineStatus,
  ProviderDescriptor,
} from "./types";

export class LocalAiApiError extends Error {
  code: string;
  status: number;

  constructor(message: string, code = "http_error", status = 0) {
    super(message);
    this.name = "LocalAiApiError";
    this.code = code;
    this.status = status;
  }
}

function detectorHeaders(): HeadersInit {
  const settings = loadUiSettings();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (settings.detectorToken.trim()) {
    headers[AI_DETECTOR_TOKEN_HEADER] = settings.detectorToken.trim();
  }
  return headers;
}

function detectorUrl(path: string): string {
  const base = loadUiSettings().detectorBaseUrl.replace(/\/+$/, "");
  return `${base}${path}`;
}

async function fetchJson<T>(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<{ ok: boolean; status: number; body: T | null; text: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    const text = await response.text();
    let body: T | null = null;
    if (text) {
      try {
        body = JSON.parse(text) as T;
      } catch {
        body = null;
      }
    }
    return { ok: response.ok, status: response.status, body, text };
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new LocalAiApiError("The request timed out.", "timeout", 0);
    }
    throw new LocalAiApiError(
      "Cannot reach the local service. Is it running on this machine?",
      "unreachable",
      0,
    );
  } finally {
    clearTimeout(timer);
  }
}

function parseDetectorError(body: unknown, fallback: string, status: number): LocalAiApiError {
  const record = body as { error?: { code?: string; message?: string }; detail?: unknown };
  const code = record?.error?.code ?? "http_error";
  const message = record?.error?.message ?? fallback;
  return new LocalAiApiError(message, code, status);
}

export async function detectAiImage(request: DetectAiImageRequest): Promise<DetectAiImageResponse> {
  const { body, ok, status } = await fetchJson<DetectAiImageResponse | { error?: { code?: string; message?: string } }>(
    detectorUrl("/detect-ai-image"),
    {
      method: "POST",
      headers: detectorHeaders(),
      body: JSON.stringify(request),
    },
    DETECT_TIMEOUT_MS,
  );

  if (!ok || !body || !("is_ai_generated" in body)) {
    throw parseDetectorError(body, "Detection failed.", status);
  }

  return normalizeDetection(body);
}

export async function getDetectorHealth(): Promise<DetectorHealth> {
  try {
    const { body, ok } = await fetchJson<DetectorHealth>(
      detectorUrl("/health"),
      { method: "GET", headers: detectorHeaders() },
      4_000,
    );
    if (!ok || !body) {
      return { status: "unreachable", message: "Detector did not respond." };
    }
    return body;
  } catch (error) {
    return {
      status: "unreachable",
      message: error instanceof Error ? error.message : "Detector unreachable.",
    };
  }
}

function normalizeDetection(raw: DetectAiImageResponse): DetectAiImageResponse {
  return {
    is_ai_generated: Boolean(raw.is_ai_generated),
    confidence: Number(raw.confidence ?? 0),
    fft_score: Number(raw.fft_score ?? 0),
    noise_score: Number(raw.noise_score ?? 0),
    cnn_score: Number(raw.cnn_score ?? 0.5),
    contract_version: raw.contract_version,
    model_version: raw.model_version,
    warnings: raw.warnings ?? [],
    details: raw.details ?? null,
  };
}

async function invokeSafe<T>(cmd: string, args?: Record<string, unknown>): Promise<T | null> {
  try {
    return await invoke<T>(cmd, args);
  } catch {
    return null;
  }
}

export async function getOfflineStatus(): Promise<OfflineStatus> {
  const fromEngine = await invokeSafe<OfflineStatus>("offline_status");
  if (fromEngine && fromEngine.connectivity) {
    return { ...fromEngine, source: "engine" };
  }

  try {
    const { body, ok } = await fetchJson<OfflineStatus>(
      `${BRIDGE_DEFAULT_BASE_URL}/offline/status`,
      { method: "GET" },
      2_500,
    );
    if (ok && body?.connectivity) {
      return { ...body, source: "bridge" };
    }
  } catch {
    /* browser fallback below */
  }

  return browserOfflineStatus();
}

export async function refreshOfflineStatus(): Promise<OfflineStatus> {
  const refreshed = await invokeSafe<OfflineStatus>("offline_refresh");
  if (refreshed?.connectivity) {
    return { ...refreshed, source: "engine" };
  }
  return getOfflineStatus();
}

export async function getOfflineConfig(): Promise<OfflineConfig> {
  const fromEngine = await invokeSafe<OfflineConfig>("get_offline_config");
  if (fromEngine) return fromEngine;
  return { ...DEFAULT_OFFLINE_CONFIG, ...loadOfflineOverlay() };
}

export async function updateOfflineConfig(partial: Partial<OfflineConfig>): Promise<OfflineConfig> {
  const updated = await invokeSafe<OfflineConfig>("update_offline_config", { partial });
  if (updated) return updated;
  return saveOfflineOverlay(partial);
}

export async function listLocalProviders(): Promise<ProviderDescriptor[]> {
  return (await invokeSafe<ProviderDescriptor[]>("list_local_providers")) ?? [];
}

export async function listLocalModels(): Promise<ModelInventory> {
  return (
    (await invokeSafe<ModelInventory>("list_local_models")) ?? {
      models: [] as LocalModel[],
      data_dir: "",
    }
  );
}

function browserOfflineStatus(): OfflineStatus {
  const overlay = loadOfflineOverlay();
  const wan = typeof navigator === "undefined" ? true : navigator.onLine;
  const force = overlay.force_offline === true;
  const blocks = force || !wan;
  return {
    connectivity: blocks ? "offline" : "online",
    force_offline: force,
    auto_fallback: overlay.auto_fallback ?? true,
    wan_reachable: wan,
    ollama_reachable: false,
    whisper_reachable: false,
    blocks_wan: blocks,
    last_probe_ms: 0,
    data_dir: "",
    source: "browser",
  };
}
