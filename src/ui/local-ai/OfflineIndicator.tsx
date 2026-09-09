import { useEffect, useState } from "react";
import { getOfflineStatus, refreshOfflineStatus } from "./api";
import { connectivityHint, connectivityLabel } from "./format";
import type { OfflineStatus } from "./types";
import "./local-ai.css";

interface Props {
  compact?: boolean;
  onOpenSettings?: () => void;
}

export function OfflineIndicator({ compact = true, onOpenSettings }: Props) {
  const [status, setStatus] = useState<OfflineStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      const next = await getOfflineStatus();
      if (!cancelled) setStatus(next);
    };
    void load();
    const id = setInterval(() => {
      void load();
    }, 8_000);
    const onOnline = () => {
      void refreshOfflineStatus().then((next) => {
        if (!cancelled) setStatus(next);
      });
    };
    window.addEventListener("online", onOnline);
    window.addEventListener("offline", onOnline);
    return () => {
      cancelled = true;
      clearInterval(id);
      window.removeEventListener("online", onOnline);
      window.removeEventListener("offline", onOnline);
    };
  }, []);

  if (!status) {
    return (
      <button type="button" className="lai-offline-chip lai-offline-unknown" disabled={compact && !onOpenSettings} title="Checking connectivity">
        <span className="lai-offline-dot" />
        {compact ? "Net" : "Checking network…"}
      </button>
    );
  }

  const label = compact
    ? status.connectivity === "online"
      ? "Online"
      : status.connectivity === "local_only"
        ? "Local"
        : "Offline"
    : connectivityLabel(status.connectivity);

  return (
    <button
      type="button"
      className={`lai-offline-chip lai-offline-${status.connectivity}`}
      title={connectivityHint(status.connectivity, status.ollama_reachable)}
      onClick={onOpenSettings}
    >
      <span className="lai-offline-dot" />
      <span>{label}</span>
      {!compact && status.force_offline && <span className="lai-offline-forced">forced</span>}
    </button>
  );
}
