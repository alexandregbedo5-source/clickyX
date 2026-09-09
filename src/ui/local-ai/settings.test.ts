import { afterEach, describe, expect, it } from "vitest";
import { DEFAULT_UI_SETTINGS, OFFLINE_OVERLAY_KEY, UI_SETTINGS_KEY } from "./constants";
import { loadOfflineOverlay, loadUiSettings, resetUiSettings, saveOfflineOverlay, saveUiSettings } from "./settings";

afterEach(() => {
  localStorage.removeItem(UI_SETTINGS_KEY);
  localStorage.removeItem(OFFLINE_OVERLAY_KEY);
});

describe("local AI UI settings", () => {
  it("returns defaults when empty", () => {
    expect(loadUiSettings()).toEqual(DEFAULT_UI_SETTINGS);
  });

  it("persists detector URL without trailing slash", () => {
    saveUiSettings({ detectorBaseUrl: "http://127.0.0.1:32188/" });
    expect(loadUiSettings().detectorBaseUrl).toBe("http://127.0.0.1:32188");
  });

  it("clamps history limit", () => {
    saveUiSettings({ historyLimit: 1 });
    expect(loadUiSettings().historyLimit).toBe(5);
    saveUiSettings({ historyLimit: 999 });
    expect(loadUiSettings().historyLimit).toBe(200);
  });

  it("resets to defaults", () => {
    saveUiSettings({ detectorToken: "secret" });
    expect(resetUiSettings().detectorToken).toBe("");
  });

  it("persists an offline overlay when the engine is absent", () => {
    saveOfflineOverlay({ force_offline: true, default_llm: "qwen2.5:3b" });
    expect(loadOfflineOverlay().force_offline).toBe(true);
    expect(loadOfflineOverlay().default_llm).toBe("qwen2.5:3b");
  });
});
