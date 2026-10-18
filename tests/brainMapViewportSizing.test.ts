import { describe, expect, it } from "vitest";
import { graphLabelOffset, graphViewportMetrics } from "../src/renderer/src/brainMapGraphPresentation";

describe("new measured Brain Map viewport sizing contract", () => {
  it("keeps selected labels12px and hit targets44px at the observed320px viewport", () => {
    const metrics = graphViewportMetrics(320, 440);
    expect(metrics.scale).toBe(0.32);
    expect(metrics.narrow).toBe(true);
    expect(metrics.labelFont * metrics.scale).toBe(12);
    expect(metrics.minimumHitRadius * 2 * metrics.scale).toBe(44);
    expect(metrics.minimumNodeRadius * metrics.scale).toBe(5);
    const x = 920, length = 20;
    const offset = graphLabelOffset(x, length, metrics);
    expect((x + offset + length * metrics.labelFont * 0.3) * metrics.scale).toBeLessThanOrEqual(312);
  });
  it("uses actual meet-scale for tall/wide desktop SVGs rather than width alone", () => {
    const metrics = graphViewportMetrics(960, 440);
    expect(metrics.narrow).toBe(false);
    expect(metrics.scale).toBe(440 / 650);
    expect(metrics.labelFont * metrics.scale).toBeCloseTo(12);
    expect(metrics.minimumHitRadius * 2 * metrics.scale).toBeCloseTo(44);
  });
});
