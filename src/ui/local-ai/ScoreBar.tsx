import { formatPercent } from "./format";

interface Props {
  label: string;
  value: number | null | undefined;
  hint?: string;
}

export function ScoreBar({ label, value, hint }: Props) {
  const pct = typeof value === "number" ? Math.round(Math.min(1, Math.max(0, value)) * 100) : 0;
  return (
    <div className="lai-score" title={hint}>
      <div className="lai-score-head">
        <span>{label}</span>
        <span className="lai-score-value">{formatPercent(value)}</span>
      </div>
      <div className="lai-score-track" aria-hidden="true">
        <div className="lai-score-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
