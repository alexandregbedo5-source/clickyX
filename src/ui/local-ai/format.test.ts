import { describe, expect, it } from "vitest";
import { connectivityLabel, formatPercent, verdictLabel, verdictTone } from "./format";

describe("format helpers", () => {
  it("formats scores as percents", () => {
    expect(formatPercent(0.92)).toBe("92%");
    expect(formatPercent(null)).toBe("—");
  });

  it("labels verdicts", () => {
    expect(verdictLabel(true)).toContain("AI");
    expect(verdictLabel(false)).toContain("photograph");
  });

  it("marks mid-range confidence as uncertain", () => {
    expect(verdictTone({ is_ai_generated: true, confidence: 0.51 })).toBe("uncertain");
    expect(verdictTone({ is_ai_generated: true, confidence: 0.92 })).toBe("ai");
    expect(verdictTone({ is_ai_generated: false, confidence: 0.1 })).toBe("photo");
  });

  it("labels connectivity", () => {
    expect(connectivityLabel("local_only")).toBe("Local only");
    expect(connectivityLabel("offline")).toBe("Offline");
  });
});
