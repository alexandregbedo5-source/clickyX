import { useAppContext } from "../context/AppContext";
import { ScoreBar, VerdictBadge, useLocalAi } from "../ui/local-ai";
import { formatPercent, humanizeWarning } from "../ui/local-ai/format";

export function DetectResultsScreen() {
  const { result, previewUrl, file, setScreen, uiSettings, error } = useLocalAi();
  const { showToast } = useAppContext();

  if (!result) {
    return (
      <section className="lai-card">
        <h3 className="lai-title">Results</h3>
        <p className="lai-copy">No analysis yet. Drop an image on the Detect screen first.</p>
        <button type="button" className="lai-btn lai-btn-primary" onClick={() => setScreen("detect")}>
          Go to Detect
        </button>
      </section>
    );
  }

  const copyVerdict = async () => {
    const text = [
      result.is_ai_generated ? "Likely AI-generated" : "Likely a photograph",
      `confidence=${formatPercent(result.confidence)}`,
      `fft=${formatPercent(result.fft_score)}`,
      `noise=${formatPercent(result.noise_score)}`,
      `cnn=${formatPercent(result.cnn_score)}`,
    ].join(" · ");
    try {
      await navigator.clipboard.writeText(text);
      showToast("Verdict copied", "success");
    } catch {
      showToast("Could not copy", "error");
    }
  };

  return (
    <section className="lai-card" aria-labelledby="lai-results-title">
      <h3 id="lai-results-title" className="lai-title">Detection result</h3>
      {previewUrl && <img className="lai-preview" src={previewUrl} alt={file?.name ?? "Analyzed image"} />}
      <VerdictBadge result={result} />

      {uiSettings.showScoreBreakdown && (
        <div>
          <ScoreBar label="Frequency (FFT)" value={result.fft_score} hint="Periodic patterns typical of generators" />
          <ScoreBar label="Residual noise" value={result.noise_score} hint="Sensor noise vs synthetic smoothness" />
          <ScoreBar label="Neural net (CNN)" value={result.cnn_score} hint="0.5 means the model is unused or neutral" />
        </div>
      )}

      {result.warnings && result.warnings.length > 0 && (
        <ul className="lai-warn-list">
          {result.warnings.map((code) => (
            <li key={code}>{humanizeWarning(code)}</li>
          ))}
        </ul>
      )}

      {result.details && (
        <details className="lai-details">
          <summary>Technical details</summary>
          <p>
            {result.details.image.width}×{result.details.image.height}
            {result.details.image.format ? ` · ${result.details.image.format}` : ""}
            {result.details.fusion ? ` · ${result.details.fusion.mode} fusion` : ""}
          </p>
          {typeof result.details.timings_ms?.total === "number" && (
            <p>Analyzed in {Math.round(result.details.timings_ms.total)} ms</p>
          )}
        </details>
      )}

      {error && <div className="lai-error" role="alert">{error}</div>}

      <div className="lai-actions">
        <button type="button" className="lai-btn lai-btn-primary" onClick={() => setScreen("detect")}>
          Analyze another
        </button>
        <button type="button" className="lai-btn" onClick={() => void copyVerdict()}>
          Copy verdict
        </button>
        <button type="button" className="lai-btn" onClick={() => setScreen("history")}>
          History
        </button>
      </div>
    </section>
  );
}

export default DetectResultsScreen;
