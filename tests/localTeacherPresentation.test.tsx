import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LocalTeacherPanel } from "../src/renderer/src/LocalTeacherPanel";
import { mergeTeacherJobs, splitTeacherLessons, teacherGenerationDraft, teacherGenerationFromDraft, teacherJobActive } from "../src/renderer/src/localTeacherPresentation";
import type { LocalTeacherApi, LocalTeacherGeneration, LocalTeacherJob } from "../src/shared/localTeacher";
const generation: LocalTeacherGeneration = { contextTokens: 4096, maxOutputTokens: 512, threads: 3, temperature: .7, seed: 0, timeoutMs: 0, maxOutputBytes: 65536 };
const job = (brainId: string, revision: number) => ({ format: "omni-local-teacher-job", version: 1, id: "job-a", brainId, originalBrainId: brainId, originalJobId: "job-a", requestId: "request-a", requestSha256: "a".repeat(64), teacherId: "teacher-a", generation, epochs: 1, state: "paused", revision, createdAt: "2026-10-13T00:00:00Z", updatedAt: "2026-10-13T00:00:00Z", promptCount: 2, generatedCount: 1, learnedCount: 0, phase: "safe-boundary", proof: "attributed-generation-and-native-source-receipts" } satisfies LocalTeacherJob);
describe("local teacher presentation without foreign identity substitution", () => {
  it("preserves multiline lesson bodies rather than manufacturing question-answer slots", () => {
    expect(splitTeacherLessons("Explain the example.\nKeep its code:\n```\nx = 1\n```\n---\nCheck an independent source.\n\nKeep this paragraph.")).toEqual(["Explain the example.\nKeep its code:\n```\nx = 1\n```", "Check an independent source.\n\nKeep this paragraph."]);
  });
  it("round-trips actual plan settings including no wall-clock deadline without guessing a fit", () => {
    expect(teacherGenerationFromDraft(teacherGenerationDraft(generation))).toEqual(generation);
    for (const patch of [{ contextTokens: "" }, { threads: "1.5" }, { timeoutMs: "-1" }, { seed: "4294967296" }, { temperature: "NaN" }, { maxOutputTokens: "4096" }]) expect(() => teacherGenerationFromDraft({ ...teacherGenerationDraft(generation), ...patch })).toThrow();
  });
  it("keeps newer source receipts and excludes another identity, without treating dormant copies as running", () => {
    const current = job("brain-a", 3);
    expect(mergeTeacherJobs([current], [job("brain-a", 1), job("brain-b", 9)], "brain-a")).toEqual([current]);
    expect(teacherJobActive({ ...current, state: "dormant" })).toBe(false); expect(teacherJobActive({ ...current, state: "waiting-load" })).toBe(true);
  });
  it("does not select files, plan, generate, train, call a provider or claim learned knowledge on initial markup", () => {
    const api = new Proxy({}, { get: () => { throw new Error("Unexpected teacher operation"); } }) as LocalTeacherApi;
    const html = renderToStaticMarkup(<LocalTeacherPanel brainId="brain-a" api={api} />);
    expect(html).toContain("Learn from a local GGUF teacher"); expect(html).toContain("Choose runtime &amp; GGUF files");
    expect(html).not.toContain("<details open"); expect(html).toContain("Start local lessons"); expect(html).toContain("disabled=");
    expect(html).not.toContain("API key"); expect(html).not.toContain("Estimated fit"); expect(html).not.toContain("loaded");
  });
});
