interface Props {
  value: number;
  label?: string;
  indeterminate?: boolean;
}

export function ProgressBar({ value, label, indeterminate }: Props) {
  const pct = Math.min(100, Math.max(0, Math.round(value * 100)));
  return (
    <div className="lai-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={indeterminate ? undefined : pct} aria-label={label ?? "Analysis progress"}>
      {label && <div className="lai-progress-label">{label}</div>}
      <div className="lai-progress-track">
        <div
          className={`lai-progress-fill${indeterminate ? " lai-progress-fill-indeterminate" : ""}`}
          style={indeterminate ? undefined : { width: `${pct}%` }}
        />
      </div>
      {!indeterminate && <div className="lai-progress-pct">{pct}%</div>}
    </div>
  );
}
