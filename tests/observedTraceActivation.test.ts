import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { meanObservedTraceActivation } from "../src/renderer/src/observedTraceActivation";

describe("observed activation presentation", () => {
  it("does not manufacture observations when a trace has only drive scores", () => {
    expect(meanObservedTraceActivation(undefined)).toBeNull();
    expect(meanObservedTraceActivation([])).toBeNull();
  });

  it("reports the existing normalized mean only from measured values", () => {
    expect(meanObservedTraceActivation([{ activation: 0.2 }, { activation: 0.8 }])).toBe(0.5);
    expect(meanObservedTraceActivation([{ activation: -0.2 }])).toBe(0);
    expect(meanObservedTraceActivation([{ activation: 2 }])).toBe(1);
    expect(meanObservedTraceActivation([{ activation: NaN }])).toBeNull();
  });

  it("keeps the decorative orb separate from whole-brain firing claims", () => {
    const app = readFileSync(new URL("../src/renderer/src/App.tsx", import.meta.url), "utf8");
    const activity = app.slice(app.indexOf("const measuredActivity ="), app.indexOf("const voicePondering ="));
    expect(activity).toContain("meanObservedTraceActivation(");
    expect(activity).not.toContain("driveScores");
    expect(app).toContain("not the fraction of the whole brain firing");
  });
});
