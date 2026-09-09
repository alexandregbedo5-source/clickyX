import { useRef, useState } from "react";
import { Icon } from "../components/Icon";
import { OfflineIndicator, ProgressBar, useLocalAi } from "../ui/local-ai";

export function DetectScreen() {
  const {
    file,
    previewUrl,
    pickFile,
    clearFile,
    analyze,
    analyzing,
    progress,
    progressLabel,
    error,
    health,
    refreshHealth,
  } = useLocalAi();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const healthState = health?.status ?? "unreachable";

  const onFiles = async (list: FileList | null) => {
    const next = list?.[0];
    if (next) await pickFile(next);
  };

  return (
    <section className="lai-card" aria-labelledby="lai-detect-title">
      <div className="lai-row-spread">
        <h3 id="lai-detect-title" className="lai-title">AI image detection</h3>
        <OfflineIndicator compact />
      </div>
      <p className="lai-copy">
        Drop a picture. Analysis stays on this computer via the local detector
        at <code>127.0.0.1:32188</code>.
      </p>

      <div className={`lai-health lai-health-${healthState}`}>
        <span className="lai-health-dot" />
        <span>
          {healthState === "ok" && "Detector ready"}
          {healthState === "degraded" && "Detector running (neural model limited)"}
          {healthState === "unreachable" && "Detector not running — start the local service"}
        </span>
        <button type="button" className="lai-btn" onClick={() => void refreshHealth()} aria-label="Refresh detector status">
          <Icon name="refresh" size={12} />
        </button>
      </div>

      <label
        className={`lai-drop${dragOver ? " lai-drop-active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          void onFiles(e.dataTransfer.files);
        }}
      >
        <Icon name="camera" size={22} />
        <span>{file ? file.name : "Drop an image or click to choose"}</span>
        <span className="lai-copy">PNG, JPEG, WebP, BMP, GIF — max 12 MB</span>
        <input
          ref={inputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.webp,.bmp,.gif,image/*"
          onChange={(e) => void onFiles(e.target.files)}
        />
      </label>

      {previewUrl && (
        <img className="lai-preview" src={previewUrl} alt={file?.name ?? "Selected image preview"} />
      )}

      {analyzing && <ProgressBar value={progress} label={progressLabel} />}

      {error && <div className="lai-error" role="alert">{error}</div>}

      <div className="lai-actions">
        <button
          type="button"
          className="lai-btn lai-btn-primary"
          disabled={!file || analyzing}
          onClick={() => void analyze()}
        >
          {analyzing ? "Analyzing…" : "Analyze image"}
        </button>
        <button type="button" className="lai-btn" disabled={!file && !previewUrl} onClick={clearFile}>
          Clear
        </button>
      </div>
    </section>
  );
}

export default DetectScreen;
