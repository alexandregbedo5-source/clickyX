import { describe, expect, it, beforeEach } from "vitest";
import { OfflineManager, resetOfflineManagerForTests } from "./OfflineManager";
import { isLoopbackUrl } from "./types";

describe("OfflineManager", () => {
  beforeEach(() => {
    resetOfflineManagerForTests();
  });

  it("blocks WAN when forced offline but keeps localhost", () => {
    const manager = new OfflineManager({ forceOffline: true });
    expect(manager.isOffline()).toBe(true);
    expect(manager.blocksWan()).toBe(true);
    expect(manager.allowsUrl("http://127.0.0.1:11434")).toBe(true);
    expect(manager.allowsUrl("https://api.openai.com/v1")).toBe(false);
    expect(() => manager.guardWan("update check")).toThrow(/offline/i);
  });

  it("reports local_only when a loopback provider is up", () => {
    const manager = new OfflineManager({ forceOffline: true });
    manager.markProbe({ ollamaReachable: true });
    expect(manager.connectivity()).toBe("local_only");
    expect(manager.preferLocalLlm("anthropic", "https://api.openai.com")).toBe(true);
  });

  it("does not hijack an OpenAI-compatible localhost endpoint", () => {
    const manager = new OfflineManager({ forceOffline: true });
    expect(manager.preferLocalLlm("openai", "http://127.0.0.1:11434/v1")).toBe(false);
  });

  it("detects loopback URLs", () => {
    expect(isLoopbackUrl("http://localhost:8090")).toBe(true);
    expect(isLoopbackUrl("https://api.anthropic.com")).toBe(false);
  });
});
