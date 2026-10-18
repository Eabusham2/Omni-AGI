import { describe, expect, it } from "vitest";
import { buildProgressTimingPresentation, storageOperationTimingPresentation } from "../src/renderer/src/buildProgressPresentation";

describe("truthful build and storage timing presentation", () => {
  it("shows an estimated total only from measured elapsed plus authoritative remaining time", () => {
    expect(buildProgressTimingPresentation({ phase: "packing", progress: 0.72, elapsedMs: 4_000, etaMs: 12_000, sinceUpdateMs: 3_000 }))
      .toMatchObject({ totalMs: 16_000, totalText: "16s" });
    expect(buildProgressTimingPresentation({ phase: "packing", progress: 0.99, elapsedMs: 4_000 }).totalText).toBeUndefined();
    expect(storageOperationTimingPresentation({ state: "complete", phase: "complete", elapsedMs: 19_000 }))
      .toMatchObject({ totalMs: 19_000, totalText: "19s" });
    expect(storageOperationTimingPresentation({ state: "failed", phase: "failed", elapsedMs: 19_000 }).totalText).toBeUndefined();
  });
  it("advances monotonic elapsed time and counts down only an authoritative ETA sample", () => {
    expect(buildProgressTimingPresentation({ phase: "packing", progress: 0.72, elapsedMs: 4_000, etaMs: 12_000, sinceUpdateMs: 3_000 }))
      .toMatchObject({ percent: 72, elapsedMs: 7_000, remainingMs: 9_000, elapsedText: "7s elapsed", remainingText: "9s remaining", terminal: false });
  });

  it("keeps a 99 percent nonterminal phase estimating instead of claiming completion", () => {
    expect(buildProgressTimingPresentation({ phase: "readiness", progress: 0.999, elapsedMs: 65_000 }))
      .toMatchObject({ percent: 99, elapsedText: "1m 05s elapsed", remainingText: "Estimating remaining time · Readiness", terminal: false });
    expect(buildProgressTimingPresentation({ phase: "readiness", progress: 1, elapsedMs: 66_000, etaMs: 0 }))
      .toMatchObject({ percent: 99, remainingText: "Estimating remaining time · Readiness", terminal: false });
  });

  it("shows 100 percent only for the durable complete phase", () => {
    expect(buildProgressTimingPresentation({ phase: "complete", progress: 1, elapsedMs: 70_000, terminal: true }))
      .toMatchObject({ percent: 100, remainingText: "Durably complete", terminal: true });
  });

  it("uses storage elapsed evidence and truthful unknown, paused, and terminal states", () => {
    expect(storageOperationTimingPresentation({ state: "running", phase: "copying blobs", elapsedMs: 9_000, sinceUpdateMs: 2_000 }))
      .toMatchObject({ elapsedText: "11s elapsed", remainingText: "Estimating remaining time · Copying blobs", terminal: false });
    expect(storageOperationTimingPresentation({ state: "paused", phase: "copying", elapsedMs: 12_000, sinceUpdateMs: 8_000, etaMs: 5_000 }))
      .toMatchObject({ elapsedMs: 12_000, remainingText: "Paused · remaining time not advancing", terminal: false });
    expect(storageOperationTimingPresentation({ state: "failed", phase: "failed", elapsedMs: 13_000 }))
      .toMatchObject({ remainingText: "Ended with error", terminal: true });
  });
});
