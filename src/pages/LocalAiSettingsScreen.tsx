import { useEffect, useState } from "react";
import { useAppContext } from "../context/AppContext";
import {
  getOfflineConfig,
  listLocalModels,
  listLocalProviders,
  updateOfflineConfig,
} from "../ui/local-ai/api";
import { formatBytes } from "../ui/local-ai/format";
import { AI_DETECTOR_DEFAULT_BASE_URL } from "../ui/local-ai/constants";
import { OfflineIndicator, useLocalAi } from "../ui/local-ai";
import type { LocalModel, OfflineConfig, ProviderDescriptor } from "../ui/local-ai/types";

export function LocalAiSettingsScreen() {
  const { uiSettings, updateUiSettings, refreshHealth, offline, refreshOffline } = useLocalAi();
  const { showToast } = useAppContext();
  const [offlineCfg, setOfflineCfg] = useState<OfflineConfig | null>(null);
  const [providers, setProviders] = useState<ProviderDescriptor[]>([]);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    void getOfflineConfig().then(setOfflineCfg);
    void listLocalProviders().then(setProviders);
    void listLocalModels().then((inv) => setModels(inv.models));
    void refreshOffline();
  }, [refreshOffline]);

  const saveDetector = () => {
    updateUiSettings(uiSettings);
    void refreshHealth();
    showToast("Detector settings saved", "success");
  };

  const saveOffline = async () => {
    if (!offlineCfg) return;
    setSaving(true);
    try {
      const next = await updateOfflineConfig(offlineCfg);
      setOfflineCfg(next);
      await refreshOffline();
      showToast("Offline settings saved", "success");
    } catch {
      showToast("Could not persist offline settings to the engine", "error");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="lai-view">
      <section className="lai-card">
        <div className="lai-row-spread">
          <h3 className="lai-title">Offline mode</h3>
          <OfflineIndicator compact={false} />
        </div>
        <p className="lai-copy">
          When the network is gone, ClickyX should stay on Ollama, local Whisper,
          and system voices. This screen only talks to the existing offline APIs.
        </p>

        {offline && (
          <div className="lai-status-grid">
            <div className="lai-status-pill">
              <strong>{offline.wan_reachable ? "WAN up" : "WAN down"}</strong>
              Internet probe
            </div>
            <div className="lai-status-pill">
              <strong>{offline.ollama_reachable ? "Ready" : "Not found"}</strong>
              Ollama
            </div>
            <div className="lai-status-pill">
              <strong>{offline.whisper_reachable ? "Ready" : "Not found"}</strong>
              Whisper
            </div>
            <div className="lai-status-pill">
              <strong>{offline.source ?? "unknown"}</strong>
              Status source
            </div>
          </div>
        )}

        {offlineCfg && (
          <>
            <label className="lai-toggle">
              <input
                type="checkbox"
                checked={offlineCfg.force_offline}
                onChange={(e) => setOfflineCfg({ ...offlineCfg, force_offline: e.target.checked })}
              />
              Force offline (block cloud calls)
            </label>
            <label className="lai-toggle">
              <input
                type="checkbox"
                checked={offlineCfg.auto_fallback}
                onChange={(e) => setOfflineCfg({ ...offlineCfg, auto_fallback: e.target.checked })}
              />
              Fall back to local models when cloud fails
            </label>
            <label className="lai-field">
              Default local LLM
              <input
                type="text"
                value={offlineCfg.default_llm}
                onChange={(e) => setOfflineCfg({ ...offlineCfg, default_llm: e.target.value })}
              />
            </label>
            <label className="lai-field">
              Ollama URL
              <input
                type="text"
                value={offlineCfg.ollama_base_url}
                onChange={(e) => setOfflineCfg({ ...offlineCfg, ollama_base_url: e.target.value })}
              />
            </label>
            <label className="lai-field">
              Whisper URL
              <input
                type="text"
                value={offlineCfg.whisper_base_url}
                onChange={(e) => setOfflineCfg({ ...offlineCfg, whisper_base_url: e.target.value })}
              />
            </label>
            <button type="button" className="lai-btn lai-btn-primary" disabled={saving} onClick={() => void saveOffline()}>
              {saving ? "Saving…" : "Save offline settings"}
            </button>
          </>
        )}

        {providers.length > 0 && (
          <div>
            <h4 className="lai-title">Local providers</h4>
            {providers.map((p) => (
              <div key={p.id} className="lai-row-spread lai-copy">
                <span>{p.name}</span>
                <span>{p.available ? "available" : "unavailable"}</span>
              </div>
            ))}
          </div>
        )}

        {models.length > 0 && (
          <div>
            <h4 className="lai-title">Local models</h4>
            {models.map((m) => (
              <div key={m.id} className="lai-row-spread lai-copy">
                <span>{m.display_name}</span>
                <span>{m.installed ? formatBytes(m.size_bytes) : "not installed"}</span>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="lai-card">
        <h3 className="lai-title">AI detector</h3>
        <label className="lai-field">
          Detector URL
          <input
            type="text"
            value={uiSettings.detectorBaseUrl}
            onChange={(e) => updateUiSettings({ detectorBaseUrl: e.target.value })}
            placeholder={AI_DETECTOR_DEFAULT_BASE_URL}
          />
        </label>
        <label className="lai-field">
          Optional token
          <input
            type="password"
            value={uiSettings.detectorToken}
            onChange={(e) => updateUiSettings({ detectorToken: e.target.value })}
            placeholder="x-ai-detector-token"
          />
        </label>
        <label className="lai-toggle">
          <input
            type="checkbox"
            checked={uiSettings.includeDetails}
            onChange={(e) => updateUiSettings({ includeDetails: e.target.checked })}
          />
          Request technical details
        </label>
        <label className="lai-toggle">
          <input
            type="checkbox"
            checked={uiSettings.autoSaveHistory}
            onChange={(e) => updateUiSettings({ autoSaveHistory: e.target.checked })}
          />
          Save each result locally
        </label>
        <label className="lai-toggle">
          <input
            type="checkbox"
            checked={uiSettings.showScoreBreakdown}
            onChange={(e) => updateUiSettings({ showScoreBreakdown: e.target.checked })}
          />
          Show FFT / noise / CNN scores
        </label>
        <label className="lai-field">
          History size
          <input
            type="number"
            min={5}
            max={200}
            value={uiSettings.historyLimit}
            onChange={(e) => updateUiSettings({ historyLimit: Number(e.target.value) })}
          />
        </label>
        <button type="button" className="lai-btn lai-btn-primary" onClick={saveDetector}>
          Save detector settings
        </button>
      </section>
    </div>
  );
}

export default LocalAiSettingsScreen;
