import { OfflineManager, getOfflineManager } from "../core/offline/OfflineManager";
import { ModelManager, getModelManager } from "../core/models/ModelManager";
import { ProviderRegistry } from "../core/providers/ProviderRegistry";
import { defaultOfflineConfig } from "../core/offline/types";
import { appCache } from "../services/cache";
import { describeLocalLayout } from "../services/storage";

export interface OfflineEngineSnapshot {
  status: ReturnType<OfflineManager["status"]>;
  providers: ReturnType<ProviderRegistry["all"]>;
  models: ReturnType<ModelManager["inventory"]>;
  layout: string[];
}

/**
 * Composition root for the TypeScript offline engine.
 * The Rust engine in src-tauri/src/offline is authoritative at runtime.
 */
export class OfflineEngine {
  readonly manager: OfflineManager;
  readonly models: ModelManager;
  readonly registry: ProviderRegistry;

  constructor() {
    this.manager = getOfflineManager();
    this.models = getModelManager();
    this.registry = new ProviderRegistry(this.manager.getConfig());
  }

  snapshot(): OfflineEngineSnapshot {
    const status = this.manager.status();
    return {
      status,
      providers: this.registry.localOnly(),
      models: this.models.inventory(),
      layout: describeLocalLayout(status.data_dir || "~/.local/share/clickyx"),
    };
  }

  clearCaches(): void {
    appCache.clear();
  }
}

export function createOfflineEngine(): OfflineEngine {
  return new OfflineEngine();
}

export const ENGINE_DEFAULTS = defaultOfflineConfig();
