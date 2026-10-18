import type { ActionReviewerClock } from "../src/main/actionReviewer";
import type { ActionReviewerConfiguration } from "../src/shared/actionReviewer";

export function reviewTestClock(): ActionReviewerClock {
  return { now: () => Date.now(), delay: (ms, signal) => new Promise<void>((resolve, reject) => {
    const stop = (): void => { clearTimeout(timer); signal.removeEventListener("abort", stop); reject(signal.reason ?? new Error("stopped")); };
    const timer = setTimeout(() => { signal.removeEventListener("abort", stop); resolve(); }, ms);
    signal.addEventListener("abort", stop, { once: true }); if (signal.aborted) stop();
  }) };
}
export function reviewTestConfiguration(): ActionReviewerConfiguration {
  return { enabled: true, revision: "settings-1", ownerRevision: "device-owner-1", reviewerId: "fixture-adviser",
    budget: { timeoutMs: 5_000, maxRequestBytes: 8_192, maxContextBytes: 4_096, maxResponseBytes: 2_048 } };
}
export const observedAllow = { verdict: "allow", summary: "The observed proposal can continue under its current permissions.",
  evidence: [{ sourceId: "action.proposed", quote: "fixture.tool" }] };
