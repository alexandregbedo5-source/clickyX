import { formatPercent, formatRelativeTime } from "../ui/local-ai/format";
import { useLocalAi } from "../ui/local-ai";

export function DetectHistoryScreen() {
  const { history, openHistoryEntry, removeHistory, clearHistory } = useLocalAi();

  return (
    <section className="lai-card" aria-labelledby="lai-history-title">
      <div className="lai-row-spread">
        <h3 id="lai-history-title" className="lai-title">Local history</h3>
        {history.length > 0 && (
          <button type="button" className="lai-btn lai-btn-danger" onClick={clearHistory}>
            Clear all
          </button>
        )}
      </div>
      <p className="lai-copy">
        Results stay in this browser profile only. Nothing is uploaded or synced.
      </p>

      {history.length === 0 ? (
        <p className="lai-copy">No detections yet.</p>
      ) : (
        <div className="lai-history" role="list">
          {history.map((entry) => (
            <div key={entry.id} className="lai-row" role="listitem">
              <button type="button" className="lai-history-item" onClick={() => openHistoryEntry(entry)}>
                {entry.thumbnail ? (
                  <img className="lai-thumb" src={entry.thumbnail} alt="" />
                ) : (
                  <div className="lai-thumb lai-thumb-empty">img</div>
                )}
                <div className="lai-history-meta">
                  <div className="lai-history-name">{entry.fileName}</div>
                  <div className="lai-history-sub">
                    {entry.is_ai_generated ? "AI" : "Photo"} · {formatPercent(entry.confidence)} · {formatRelativeTime(entry.createdAt)}
                  </div>
                </div>
              </button>
              <button
                type="button"
                className="lai-btn"
                aria-label={`Delete ${entry.fileName}`}
                onClick={() => removeHistory(entry.id)}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export default DetectHistoryScreen;
