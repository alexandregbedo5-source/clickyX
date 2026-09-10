import { formatPercent, verdictLabel, verdictTone } from "./format";
import type { DetectAiImageResponse } from "./types";

interface Props {
  result: Pick<DetectAiImageResponse, "is_ai_generated" | "confidence">;
  compact?: boolean;
}

export function VerdictBadge({ result, compact }: Props) {
  const tone = verdictTone(result);
  return (
    <div className={`lai-verdict lai-verdict-${tone}${compact ? " lai-verdict-compact" : ""}`}>
      <span className="lai-verdict-kicker">
        {tone === "uncertain" ? "Uncertain" : result.is_ai_generated ? "Synthetic" : "Photograph"}
      </span>
      {!compact && <strong className="lai-verdict-title">{verdictLabel(result.is_ai_generated)}</strong>}
      <span className="lai-verdict-conf">{formatPercent(result.confidence)} confidence</span>
    </div>
  );
}
