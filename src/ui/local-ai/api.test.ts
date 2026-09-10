import { afterEach, describe, expect, it, vi } from "vitest";
import { detectAiImage, getDetectorHealth, getOfflineStatus, LocalAiApiError } from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("detectAiImage", () => {
  it("posts to /detect-ai-image and returns the contract fields", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () =>
        JSON.stringify({
          is_ai_generated: true,
          confidence: 0.92,
          fft_score: 0.8,
          noise_score: 0.7,
          cnn_score: 0.85,
        }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await detectAiImage({ image_base64: "abc", include_details: false });
    expect(result.is_ai_generated).toBe(true);
    expect(result.confidence).toBe(0.92);
    expect(fetchMock).toHaveBeenCalled();
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/detect-ai-image");
    expect(init.method).toBe("POST");
  });

  it("surfaces API error codes", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 415,
        text: async () => JSON.stringify({ error: { code: "unsupported_image", message: "bad" } }),
      }),
    );

    await expect(detectAiImage({ image_base64: "x" })).rejects.toMatchObject({
      name: "LocalAiApiError",
      code: "unsupported_image",
    } satisfies Partial<LocalAiApiError>);
  });
});

describe("getDetectorHealth", () => {
  it("returns unreachable when the service is down", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("failed")));
    const health = await getDetectorHealth();
    expect(health.status).toBe("unreachable");
  });
});

describe("getOfflineStatus", () => {
  it("falls back to browser connectivity", async () => {
    const status = await getOfflineStatus();
    expect(["online", "offline", "local_only"]).toContain(status.connectivity);
    expect(status.source).toBeDefined();
  });
});
