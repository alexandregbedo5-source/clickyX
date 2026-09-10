import { afterEach, describe, expect, it } from "vitest";
import { HISTORY_KEY, UI_SETTINGS_KEY } from "./constants";
import {
  addHistoryEntry,
  clearHistory,
  historyFromResult,
  loadHistory,
  removeHistoryEntry,
} from "./history";
import type { DetectAiImageResponse } from "./types";

const sample: DetectAiImageResponse = {
  is_ai_generated: true,
  confidence: 0.92,
  fft_score: 0.8,
  noise_score: 0.7,
  cnn_score: 0.9,
};

afterEach(() => {
  localStorage.removeItem(HISTORY_KEY);
  localStorage.removeItem(UI_SETTINGS_KEY);
});

describe("detection history", () => {
  it("starts empty", () => {
    expect(loadHistory()).toEqual([]);
  });

  it("adds and removes entries locally", () => {
    addHistoryEntry(historyFromResult(sample, "cat.png", null));
    expect(loadHistory()).toHaveLength(1);
    expect(loadHistory()[0].fileName).toBe("cat.png");
    expect(loadHistory()[0].is_ai_generated).toBe(true);

    removeHistoryEntry(loadHistory()[0].id);
    expect(loadHistory()).toHaveLength(0);
  });

  it("clears all entries", () => {
    addHistoryEntry(historyFromResult(sample, "a.png", null));
    addHistoryEntry(historyFromResult({ ...sample, is_ai_generated: false, confidence: 0.1 }, "b.png", null));
    expect(loadHistory().length).toBeGreaterThan(0);
    clearHistory();
    expect(loadHistory()).toEqual([]);
  });
});
