import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ApiTeacherPanel, teacherCurrentJob, teacherDurableReport, teacherNeedsKey, teacherSubmitIdentity } from "../src/renderer/src/ApiTeacherPanel";
import type { ApiTeacherTrainingReport, OmniApi, RuntimeJob } from "../src/shared/types";

const job = (patch: Partial<RuntimeJob> = {}): RuntimeJob => ({ id: "job-a", brainId: "brain-a", kind: "teacher", state: "running",
  progress: 0.5, label: "Learning", createdAt: "2026-10-15T00:00:00Z", updatedAt: "2026-10-15T00:01:00Z", ...patch });
describe("teacher panel optional provider and durable owner contracts", () => {
  it("offers both additions without inventing a default hosted model or reading credentials on render", () => {
    const api = new Proxy({}, { get() { throw new Error("No operation is permitted during rendering"); } }) as OmniApi["teacher"];
    const html = renderToStaticMarkup(<ApiTeacherPanel brainId="brain-a" api={api} onNotice={() => undefined} />);
    expect(html).toContain('value="openrouter"'); expect(html).toContain('value="ollama"'); expect(html).toContain("OmniCortex remains your brain");
    expect(html).toContain('type="password"'); expect(html).not.toContain("gpt-5"); expect(html).not.toContain("claude-sonnet");
  });
  it("local Ollama renders no key controls and keeps installed-model selection explicit", () => {
    const html = renderToStaticMarkup(<ApiTeacherPanel brainId="brain-a" initialProvider="ollama" onNotice={() => undefined} />);
    expect(html).toContain("127.0.0.1:11434"); expect(html).toContain("no API key"); expect(html).not.toContain('type="password"');
    expect(html).not.toContain("Store key"); expect(html).not.toContain("Remove key");
    expect(teacherNeedsKey("ollama")).toBe(false); expect(teacherNeedsKey("openrouter")).toBe(true);
  });
  it("keeps one submission identity across lost-ACK retries but not changed owner/provider/content", () => {
    const input = { brainId: "brain-a", provider: "ollama" as const, model: "local-model", prompts: ["question"] };
    const first = teacherSubmitIdentity(undefined, input, () => "request-a");
    expect(teacherSubmitIdentity(first, input, () => { throw new Error("Retry must not create another ID"); })).toBe(first);
    for (const changed of [{ ...input, brainId: "brain-b" }, { ...input, provider: "openrouter" as const }, { ...input, prompts: ["other question"] }])
      expect(teacherSubmitIdentity(first, changed, () => "request-b").requestId).toBe("request-b");
  });
  it("rejects cross-instance events and stale late start acknowledgements", () => {
    const completed = job({ state: "complete", progress: 1 });
    expect(teacherCurrentJob(completed, job({ brainId: "brain-b" }), "brain-a")).toBe(completed);
    expect(teacherCurrentJob(completed, job({ state: "queued" }), "brain-a")).toBe(completed);
    expect(teacherCurrentJob(completed, job({ updatedAt: "2026-10-15T00:00:59Z" }), "brain-a")).toBe(completed);
    expect(teacherCurrentJob(completed, job({ id: "older", createdAt: "2026-10-14T00:00:00Z" }), "brain-a")).toBe(completed);
  });
  it("uses typed durable state instead of treating a generic cancelled job as a paused/billed request", () => {
    expect(teacherDurableReport(job({ state: "cancelled" }))).toBeUndefined();
    const report = { durableState: "review-required", nextIndex: 0, uncertainItemIndexes: [0], resumable: false,
      learned: 0, requested: 1 } as ApiTeacherTrainingReport;
    expect(teacherDurableReport(job({ output: report }))).toBe(report);
    expect(teacherDurableReport(job({ output: { ...report, uncertainItemIndexes: [-1] } }))).toBeUndefined();
  });
});
