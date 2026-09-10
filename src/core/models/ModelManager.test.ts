import { describe, expect, it, beforeEach } from "vitest";
import { ModelManager, resetModelManagerForTests } from "./ModelManager";

describe("ModelManager", () => {
  beforeEach(() => {
    resetModelManagerForTests();
  });

  it("lists local LLM and Whisper catalog entries", () => {
    const manager = new ModelManager();
    const catalog = manager.catalog();
    expect(catalog.some((model) => model.id === "llama3.2")).toBe(true);
    expect(catalog.some((model) => model.kind === "whisper")).toBe(true);
  });

  it("refuses downloads while offline", () => {
    const manager = new ModelManager();
    const result = manager.canDownload("base", false);
    expect(result.ok).toBe(false);
    expect(result.reason).toMatch(/offline/i);
  });

  it("tracks imported models", () => {
    const manager = new ModelManager();
    manager.markInstalled({
      id: "base",
      display_name: "Whisper base",
      kind: "whisper",
      provider: "whisper-local",
      installed: true,
      path: "/tmp/ggml-base.bin",
      size_bytes: 10,
      download_url: null,
      notes: "imported",
    });
    expect(manager.listInstalled().map((model) => model.id)).toContain("base");
    expect(manager.canDownload("base", false).ok).toBe(true);
  });
});
