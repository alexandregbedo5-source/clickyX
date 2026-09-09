import {
  defaultOfflineConfig,
  isLoopbackUrl,
  type Connectivity,
  type OfflineConfig,
  type OfflineStatus,
} from "./types";

export interface OfflineManagerOptions {
  forceOffline?: boolean;
  autoFallback?: boolean;
  config?: Partial<OfflineConfig>;
}

/**
 * Frontend mirror of the Rust OfflineManager.
 * Uses local heuristics when Tauri is not injected (unit tests / SSR).
 */
export class OfflineManager {
  private config: OfflineConfig;
  private wanReachable = false;
  private ollamaReachable = false;
  private whisperReachable = false;

  constructor(options: OfflineManagerOptions = {}) {
    this.config = {
      ...defaultOfflineConfig(),
      ...options.config,
      force_offline: options.forceOffline ?? options.config?.force_offline ?? defaultOfflineConfig().force_offline,
      auto_fallback: options.autoFallback ?? options.config?.auto_fallback ?? true,
    };
  }

  getConfig(): OfflineConfig {
    return { ...this.config };
  }

  applyConfig(partial: Partial<OfflineConfig>): OfflineConfig {
    this.config = { ...this.config, ...partial };
    return this.getConfig();
  }

  markProbe(report: {
    wanReachable?: boolean;
    ollamaReachable?: boolean;
    whisperReachable?: boolean;
  }): void {
    if (typeof report.wanReachable === "boolean") this.wanReachable = report.wanReachable;
    if (typeof report.ollamaReachable === "boolean") this.ollamaReachable = report.ollamaReachable;
    if (typeof report.whisperReachable === "boolean") this.whisperReachable = report.whisperReachable;
  }

  connectivity(): Connectivity {
    if (this.config.force_offline || !this.wanReachable) {
      if (this.ollamaReachable || this.whisperReachable) return "local_only";
      return "offline";
    }
    return "online";
  }

  blocksWan(): boolean {
    return this.config.force_offline || !this.wanReachable;
  }

  isOffline(): boolean {
    return this.blocksWan();
  }

  allowsUrl(url: string): boolean {
    if (isLoopbackUrl(url)) return true;
    return !this.blocksWan();
  }

  guardWan(operation: string): void {
    if (this.blocksWan()) {
      throw new Error(
        `${operation} requires Internet. ClickyX is offline (Wi-Fi / Ethernet / firewall). Use a local provider.`,
      );
    }
  }

  status(): OfflineStatus {
    return {
      connectivity: this.connectivity(),
      force_offline: this.config.force_offline,
      auto_fallback: this.config.auto_fallback,
      wan_reachable: this.wanReachable,
      ollama_reachable: this.ollamaReachable,
      whisper_reachable: this.whisperReachable,
      blocks_wan: this.blocksWan(),
      last_probe_ms: 0,
      data_dir: "",
    };
  }

  preferLocalLlm(defaultProvider: string, openaiBaseUrl: string): boolean {
    if (isLoopbackUrl(openaiBaseUrl) && defaultProvider === "openai") return false;
    if (defaultProvider === "ollama") return true;
    return this.isOffline() || (this.config.auto_fallback && this.ollamaReachable);
  }
}

let singleton: OfflineManager | null = null;

export function getOfflineManager(): OfflineManager {
  if (!singleton) singleton = new OfflineManager();
  return singleton;
}

export function resetOfflineManagerForTests(): void {
  singleton = null;
}
