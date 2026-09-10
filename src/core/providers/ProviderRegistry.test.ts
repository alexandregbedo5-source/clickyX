import { describe, expect, it, beforeEach } from "vitest";
import { ProviderRegistry } from "./ProviderRegistry";
import { defaultOfflineConfig } from "../offline/types";
import { OfflineManager, resetOfflineManagerForTests, getOfflineManager } from "../offline/OfflineManager";

describe("ProviderRegistry", () => {
  beforeEach(() => {
    resetOfflineManagerForTests();
    getOfflineManager().applyConfig({ force_offline: true });
  });

  it("exposes local Ollama, Whisper and system TTS", () => {
    const registry = new ProviderRegistry(defaultOfflineConfig());
    const local = registry.localOnly();
    expect(local.map((provider) => provider.id).sort()).toEqual([
      "ollama",
      "system",
      "whisper-local",
    ]);
    expect(local.every((provider) => !provider.requires_wan)).toBe(true);
    expect(registry.preferred("llm")).toBe("ollama");
    expect(registry.preferred("stt")).toBe("whisper-local");
    expect(registry.preferred("tts")).toBe("system");
  });

  it("marks cloud providers unavailable when WAN is blocked", () => {
    const manager = new OfflineManager({ forceOffline: true });
    expect(manager.blocksWan()).toBe(true);
    const registry = new ProviderRegistry(manager.getConfig());
    const anthropic = registry.all().find((provider) => provider.id === "anthropic");
    expect(anthropic?.available).toBe(false);
  });
});
