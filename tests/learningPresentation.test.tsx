import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { LearningTask } from "../src/shared/learning";
import { LearningWorkspace } from "../src/renderer/src/LearningWorkspace";
import { drainLearningTaskEvents, learningConstraints, learningDomainList, learningSourcePercent, learningTaskStatus, mergeLearningTasks } from "../src/renderer/src/learningPresentation";
import { workspaceViewWaitsForForegroundTurn, EMPTY_CHAT_WORKSPACE_ACTIVITY } from "../src/renderer/src/chatWorkspaceActivity";

const task = (overrides: Partial<LearningTask> = {}): LearningTask => ({
  format: "omni-learning-task", version: 1, id: "task", brainId: "brain", requestId: "request", requestSha256: "test-only",
  kind: "specific-learn", state: "running", createdAt: "2026-10-11T00:00:00Z", updatedAt: "2026-10-11T00:00:00Z", revision: 1,
  policy: { revision: 1, sha256: "test-only", constraints: {}, instructions: "", changedAt: "2026-10-11T00:00:00Z", changedBy: "user" },
  sources: [], progress: .5, phase: "learning", label: "Learning sources", epochs: 1, pendingQuestions: [], traversalComplete: false,
  ...overrides
});

describe("learning workspace presentation", () => {
  it("keeps the newest durable task revision when a scan and live events race", () => {
    const fresh = task({ revision: 3, label: "latest" });
    const next = mergeLearningTasks([fresh], [task({ revision: 1 }), task({ id: "other", createdAt: "2026-10-12T00:00:00Z" })]);
    expect(next).toHaveLength(2); expect(next.find(value => value.id === "task")).toBe(fresh);
    expect(next[0]!.id).toBe("other");
  });
  it("retains an event batch when a deferred React updater runs after the pending map is cleared", () => {
    const fresh = task({ revision: 2, currentSource: "source", progress: .995 });
    const pending = new Map([[fresh.id, fresh]]);
    const received = drainLearningTaskEvents(pending);
    pending.set("other", task({ id: "other", revision: 4 }));
    expect(mergeLearningTasks([task()], received)[0]!.progress).toBe(.995);
    expect(received).toEqual([fresh]); expect(pending.size).toBe(1);
  });
  it("normalizes explicit domain filters without broadening paths or leaking credentials", () => {
    expect(learningDomainList("Docs.Example.com, https://docs.example.com/\nother.example.com")).toEqual([
      { domain: "docs.example.com", includeSubdomains: true }, { domain: "other.example.com", includeSubdomains: true }]);
    for (const value of ["example.com/private", "https://name:secret@example.com", "https://example.com?section=one", "https://example.com:8443"]) expect(() => learningDomainList(value)).toThrow();
    expect(learningDomainList("  ")).toBeUndefined();
  });
  it("retains per-domain date overrides and rejects ambiguous invalid scope text", () => {
    const scoped = learningConstraints({ allowed: "example.com", excluded: "ads.example.com", from: "2025-01-01", through: "2026-01-01",
      domainRules: '[{"domain":"docs.example.com","dates":null}]' });
    expect(scoped.domainDates).toEqual([{ domain: "docs.example.com", dates: null }]);
    expect(scoped.dates).toEqual({ from: "2025-01-01", through: "2026-01-01" });
    expect(() => learningConstraints({ allowed: "", excluded: "", from: "2026-01-01", through: "2025-01-01", domainRules: "" })).toThrow(/earliest/);
    expect(() => learningConstraints({ allowed: "", excluded: "", from: "", through: "", domainRules: "{}" })).toThrow(/list/);
  });
  it("does not turn a current-source percentage or premature completion label into proof of learning", () => {
    expect(learningSourcePercent(task())).toBeUndefined();
    expect(learningSourcePercent(task({ currentSource: "source", progress: 1 }))).toBe("99.9%");
    expect(learningTaskStatus(task({ state: "complete" }))).toBe("Awaiting final coverage receipt");
    expect(learningSourcePercent(task({ state: "complete", traversalComplete: true, progress: 1 }))).toBe("100%");
  });
  it("keeps the queue-aware learning and Fact Check surfaces available during a real foreground turn", () => {
    const active = { ...EMPTY_CHAT_WORKSPACE_ACTIVITY, turnActive: true, phase: "responding" as const };
    expect(workspaceViewWaitsForForegroundTurn("learn", active)).toBe(false);
    expect(workspaceViewWaitsForForegroundTurn("fact-check", active)).toBe(false);
  });
  it("shows actual source controls and a truthful disconnected preview, without fabricated jobs", () => {
    const html = renderToStaticMarkup(<LearningWorkspace brainId="brain" mode="specific-learn" />);
    expect(html).toContain("Add files"); expect(html).toContain("Add folder"); expect(html).toContain("Web sources");
    expect(html).toContain("Web scope and dates"); expect(html).toContain("No tasks yet");
    expect(html).toContain("available in the desktop app"); expect(html).not.toContain("100%");
    const facts = renderToStaticMarkup(<LearningWorkspace brainId="brain" mode="fact-check" />);
    expect(facts).toContain("Claims · one per line"); expect(facts).toContain("Start checking"); expect(facts).not.toContain("Add folder");
    expect(facts).not.toContain("all knowledge verified");
  });
});
