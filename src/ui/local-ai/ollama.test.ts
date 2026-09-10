import { describe, expect, it } from "vitest";
import { resolveOllamaModel } from "./ollama";

describe("resolveOllamaModel", () => {
  it("replaces cloud model ids with the local default", () => {
    expect(resolveOllamaModel("claude-sonnet-4-20250514")).toBe("llama3.2:1b");
    expect(resolveOllamaModel("gpt-4o")).toBe("llama3.2:1b");
  });

  it("keeps an explicit local tag", () => {
    expect(resolveOllamaModel("llama3.2:1b")).toBe("llama3.2:1b");
  });
});
