import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ActionReviewer, type ActionReviewerProposal } from "../src/main/actionReviewer";
import type { ActionReviewerRequest } from "../src/shared/actionReviewer";
import { reviewTestClock, reviewTestConfiguration, observedAllow } from "./actionReviewerFixtures";
const proposal: ActionReviewerProposal = { invocation: { brainId: "brain-one", toolId: "fixture.tool", action: "inspect", arguments: { value: "observed" } },
  permissionRevision: "permission-1", permissionLevels: [{ toolId: "fixture.tool", level: "full" }], requestId: "request-one", chatTurnId: "turn-one" };

describe("optional observed-action review", () => {
  beforeEach(() => { vi.useFakeTimers({ now: 10_000 }); });
  afterEach(() => { vi.useRealTimers(); });

  it("is off by default without a model call or a delay", async () => {
    const review = vi.fn(), reviewer = new ActionReviewer({ review, clock: reviewTestClock() }), signal = new AbortController().signal;
    const saved = await reviewer.snapshot("brain-one", signal);
    expect(saved.configuration.enabled).toBe(false); expect(await reviewer.inspect(proposal, saved, signal)).toBeUndefined();
    expect(review).not.toHaveBeenCalled(); expect(vi.getTimerCount()).toBe(0);
  });

  it("waits a measured full second even when the strict review finishes immediately", async () => {
    const review = vi.fn(async (_request: ActionReviewerRequest) => JSON.stringify(observedAllow)), reviewer = new ActionReviewer({ configuration: () => reviewTestConfiguration(), review, clock: reviewTestClock() });
    const signal = new AbortController().signal, saved = await reviewer.snapshot("brain-one", signal);
    let settled = false; const pending = reviewer.inspect(proposal, saved, signal).finally(() => { settled = true; });
    await vi.advanceTimersByTimeAsync(999); expect(settled).toBe(false);
    await vi.advanceTimersByTimeAsync(1); const result = await pending;
    expect(result).toMatchObject({ verdict: "allow", disposition: "reviewed", elapsedMs: 1_000, proposedInvocation: proposal.invocation });
    expect(review.mock.calls[0]?.[0].prompt).toContain("hidden intentions"); expect(vi.getTimerCount()).toBe(0);
  });

  it("also waits for actual review completion after the one-second floor", async () => {
    let finish!: (value: unknown) => void;
    const reviewer = new ActionReviewer({ configuration: () => reviewTestConfiguration(), review: () => new Promise(resolve => { finish = resolve; }), clock: reviewTestClock() });
    const signal = new AbortController().signal, saved = await reviewer.snapshot("brain-one", signal);
    let settled = false; const pending = reviewer.inspect(proposal, saved, signal).finally(() => { settled = true; });
    await vi.advanceTimersByTimeAsync(1_600); expect(settled).toBe(false); finish(observedAllow);
    expect(await pending).toMatchObject({ verdict: "allow", elapsedMs: 1_600 }); expect(vi.getTimerCount()).toBe(0);
  });

  it.each([
    "```json\n{\"verdict\":\"allow\",\"summary\":\"ok\",\"evidence\":[]}\n```",
    { ...observedAllow, grant: "full" },
    { ...observedAllow, verdict: "safe" },
    { ...observedAllow, evidence: [] },
    { ...observedAllow, evidence: [{ sourceId: "action.proposed", quote: "invented hidden intent" }] },
    { ...observedAllow, evidence: [{ sourceId: "unseen", quote: "fixture.tool" }] }
  ])("asks for an invalid/ungrounded whole verdict, without JSON repair %#", async raw => {
    const reviewer = new ActionReviewer({ configuration: () => reviewTestConfiguration(), review: async () => raw, clock: reviewTestClock() });
    const signal = new AbortController().signal, saved = await reviewer.snapshot("brain-one", signal), pending = reviewer.inspect(proposal, saved, signal);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(await pending).toMatchObject({ verdict: "ask", disposition: "unparseable", evidence: [] });
  });

  it("times out to Ask, aborts a late provider and still honors the full hold", async () => {
    const configuration = reviewTestConfiguration(); configuration.budget.timeoutMs = 20;
    let providerSignal: AbortSignal | undefined, finish!: (value: unknown) => void;
    const reviewer = new ActionReviewer({ configuration: () => configuration, review: async (_request, signal) => {
      providerSignal = signal; return new Promise(resolve => { finish = resolve; });
    }, clock: reviewTestClock() });
    const signal = new AbortController().signal, saved = await reviewer.snapshot("brain-one", signal);
    let settled = false; const pending = reviewer.inspect(proposal, saved, signal).finally(() => { settled = true; });
    await vi.advanceTimersByTimeAsync(999); expect(providerSignal?.aborted).toBe(true); expect(settled).toBe(false);
    await vi.advanceTimersByTimeAsync(1); expect(await pending).toMatchObject({ verdict: "ask", disposition: "timeout", elapsedMs: 1_000 });
    finish(observedAllow); await Promise.resolve(); expect(vi.getTimerCount()).toBe(0);
  });

  it("fails closed for a throwing or missing provider and for unavailable settings", async () => {
    for (const hooks of [{ review: async () => { throw new Error("offline"); } }, {}, { configuration: () => { throw new Error("settings unavailable"); } }]) {
      const reviewer = new ActionReviewer({ configuration: () => reviewTestConfiguration(), ...hooks, clock: reviewTestClock() });
      const signal = new AbortController().signal, saved = await reviewer.snapshot("brain-one", signal), pending = reviewer.inspect(proposal, saved, signal);
      await vi.advanceTimersByTimeAsync(1_000); expect(await pending).toMatchObject({ verdict: "ask", disposition: "unavailable" });
      if (!saved.available) await expect(reviewer.assertCurrent("brain-one", saved, signal)).rejects.toThrow(/unavailable/);
    }
  });

  it("refuses oversized exact context or action input rather than truncating it for a model", async () => {
    const configuration = reviewTestConfiguration(); configuration.budget.maxContextBytes = 40;
    const review = vi.fn(async () => observedAllow), reviewer = new ActionReviewer({ configuration: () => configuration, review,
      selectContext: async () => [{ id: "task-one", kind: "task", text: "x".repeat(100) }], clock: reviewTestClock() });
    const signal = new AbortController().signal, saved = await reviewer.snapshot("brain-one", signal), pending = reviewer.inspect(proposal, saved, signal);
    await vi.advanceTimersByTimeAsync(1_000); expect(await pending).toMatchObject({ verdict: "ask", disposition: "budget-refused" }); expect(review).not.toHaveBeenCalled();
    const exact = { ...proposal, invocation: { ...proposal.invocation, arguments: { text: "x".repeat(9_000) } } };
    const actionOnly = new ActionReviewer({ configuration: () => reviewTestConfiguration(), review, clock: reviewTestClock() });
    const action = actionOnly.inspect(exact, await actionOnly.snapshot("brain-one", signal), signal);
    await vi.advanceTimersByTimeAsync(1_000); const result = await action;
    expect(result).toMatchObject({ verdict: "ask", disposition: "budget-refused" }); expect(result?.proposedInvocation.arguments).toEqual(exact.invocation.arguments); expect(review).not.toHaveBeenCalled();
  });

  it("permits only evidence from explicitly selected task/work and keeps that context in history", async () => {
    const reviewer = new ActionReviewer({ configuration: () => reviewTestConfiguration(),
      selectContext: async () => [{ id: "task-one", kind: "task", text: "Inspect, do not modify." }],
      review: async () => ({ verdict: "ask", summary: "The selected task requires inspection only.", evidence: [{ sourceId: "task-one", quote: "do not modify" }] }), clock: reviewTestClock() });
    const signal = new AbortController().signal, pending = reviewer.inspect(proposal, await reviewer.snapshot("brain-one", signal), signal);
    await vi.advanceTimersByTimeAsync(1_000); expect(await pending).toMatchObject({ verdict: "ask", disposition: "reviewed",
      observedContext: [{ id: "task-one", kind: "task", text: "Inspect, do not modify." }] });
  });

  it("bounds Stop/configuration change even if the provider ignores cancellation", async () => {
    const reviewer = new ActionReviewer({ configuration: () => reviewTestConfiguration(), review: async () => new Promise(() => undefined), clock: reviewTestClock() });
    const controller = new AbortController(), saved = await reviewer.snapshot("brain-one", controller.signal), pending = reviewer.inspect(proposal, saved, controller.signal);
    const rejected = expect(pending).rejects.toThrow(/configuration changed/);
    await vi.advanceTimersByTimeAsync(100); reviewer.configurationChanged("brain-one"); await rejected;
    expect(vi.getTimerCount()).toBe(0);
  });

  it("cleans synchronous Stop from a callback before default timer setup", async () => {
    const controller = new AbortController(), reviewer = new ActionReviewer({ configuration: () => reviewTestConfiguration(),
      review: async () => { controller.abort(new Error("Stopped inside callback")); return observedAllow; } });
    const saved = await reviewer.snapshot("brain-one", controller.signal);
    await expect(reviewer.inspect(proposal, saved, controller.signal)).rejects.toThrow("Stopped inside callback");
    expect((reviewer as unknown as { active: Map<string, unknown> }).active.size).toBe(0); expect(vi.getTimerCount()).toBe(0);
  });
});
