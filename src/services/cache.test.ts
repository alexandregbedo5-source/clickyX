import { describe, expect, it } from "vitest";
import { MemoryCache } from "./cache";
import { parseOllamaTags, ollamaEndpoints } from "./ollama";
import { whisperFileName } from "./storage";

describe("offline services", () => {
  it("caches values and expires them", () => {
    const cache = new MemoryCache();
    cache.set("k", 1, 1);
    expect(cache.get<number>("k")).toBe(1);
    cache.clear();
    expect(cache.get<number>("k")).toBeUndefined();
  });

  it("parses Ollama tags and builds loopback endpoints", () => {
    const tags = parseOllamaTags({
      models: [{ name: "llama3.2", size: 10 }, { name: "mistral" }],
    });
    expect(tags[0]?.name).toBe("llama3.2");
    expect(ollamaEndpoints("http://127.0.0.1:11434/").chat).toBe(
      "http://127.0.0.1:11434/api/chat",
    );
    expect(whisperFileName("base")).toBe("ggml-base.bin");
  });
});
