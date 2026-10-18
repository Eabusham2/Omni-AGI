import { describe, expect, it } from "vitest";
import type { UserQuestionRecord } from "../src/shared/userQuestions";
import {
  mergeUserQuestionRecords,
  userQuestionAnswerText,
  userQuestionCountdown,
  userQuestionDraftError,
  userQuestionDraftResponse,
  userQuestionInPanel,
  userQuestionResolutionLabel,
  userQuestionTimeoutSeconds
} from "../src/renderer/src/userQuestionPresentation";

export function fixtureQuestion(overrides: Partial<UserQuestionRecord> = {}): UserQuestionRecord {
  return { format: "omni-user-question", version: 1, id: "fixture", brainId: "one", requestId: "request",
    requestSha256: "fixture-only", kind: "task-choice", prompt: "Which sections?", mode: "multiple", minSelections: 2, maxSelections: 2,
    options: [{ id: "intro", label: "Introduction", recommended: true, rank: 1 }, { id: "examples", label: "Examples", rank: 2 }, { id: "appendix", label: "Appendix", rank: 3 }],
    allowOther: true, createdAt: "2026-10-02T12:00:00.000Z", updatedAt: "2026-10-02T12:00:00.000Z", status: "pending",
    timeout: { enabled: false, seconds: 1200, policyRevision: "default" }, ...overrides };
}

describe("question panel choices and state reconciliation", () => {
  it("keeps general conversation questions with their instance and leaves Learn/Fact Check questions task-scoped", () => {
    const native = fixtureQuestion({ source: { kind: "brain", turnId: "turn-native", actionId: "action-native" } });
    expect(userQuestionInPanel(native, "one", undefined, "conversation")).toBe(true);
    expect(userQuestionInPanel(fixtureQuestion({ source: { kind: "host" } }), "one", undefined, "conversation")).toBe(true);
    expect(userQuestionInPanel(fixtureQuestion(), "one", undefined, "conversation")).toBe(true);
    expect(userQuestionInPanel(native, "another", undefined, "conversation")).toBe(false);
    for (const kind of ["learn", "fact-check"] as const) {
      const task = fixtureQuestion({ source: { kind } });
      expect(userQuestionInPanel(task, "one", undefined, "conversation")).toBe(false);
      expect(userQuestionInPanel(task, "one", [task.id])).toBe(true);
      expect(userQuestionInPanel(task, "one", ["another-task-question"])).toBe(false);
    }
  });

  it("requires the selected minimum and submits stable option identities in ranked order", () => {
    const question = fixtureQuestion();
    expect(userQuestionDraftError(question, { optionIds: ["intro"], text: "", other: false })).toBe("Choose at least 2 options.");
    expect(userQuestionDraftError(question, { optionIds: ["intro", "examples", "appendix"], text: "", other: false })).toBe("Choose no more than 2 options.");
    expect(userQuestionDraftError(question, { optionIds: ["missing", "intro"], text: "", other: false })).toBe("Choose from the listed options.");
    expect(userQuestionDraftResponse(question, { optionIds: ["appendix", "intro"], text: "", other: false })).toEqual({ optionIds: ["intro", "appendix"] });
    expect(() => userQuestionDraftResponse(question, { optionIds: [], text: "", other: false })).toThrow("at least 2");
  });

  it("preserves written answers and requires explicit Other capability", () => {
    const draft = { optionIds: [], text: "  A custom ending.\nKeep the last scene.  ", other: true };
    expect(userQuestionDraftResponse(fixtureQuestion(), draft)).toEqual({ text: draft.text });
    expect(userQuestionDraftError(fixtureQuestion({ allowOther: false }), draft)).toBe("Choose one of the listed options.");
    expect(userQuestionDraftError(fixtureQuestion({ mode: "text" }), { ...draft, text: " \n" })).toBe("Enter your answer.");
    expect(userQuestionDraftError(fixtureQuestion({ status: "answered" }), draft)).toContain("already");
    expect(userQuestionDraftError(fixtureQuestion({ suspended: { reason: "copied-instance", fromBrainId: "parent", copiedAt: "2026-10-02T12:00:00.000Z" } }), draft)).toContain("Resume");
  });

  it("merges events arriving during disk reads without duplicate cards or losing their answers", () => {
    const first = fixtureQuestion({ id: "a" });
    const second = fixtureQuestion({ id: "b", createdAt: "2026-10-02T12:01:00.000Z" });
    const answered = fixtureQuestion({ id: "a", status: "answered", updatedAt: "2026-10-02T12:02:00.000Z" });
    expect(mergeUserQuestionRecords([first, second], [answered])).toEqual([answered, second]);
    expect(mergeUserQuestionRecords([first], [second, answered, answered])).toEqual([answered, second]);
    expect(mergeUserQuestionRecords([answered], [first])).toEqual([answered]);
    const resolution = { id: "answer", source: "user" as const, resolvedAt: answered.updatedAt, response: { optionIds: ["intro", "examples"] }, authority: "task-choice-only" as const, permissionsGranted: false as const };
    const sameTimeAnswer = { ...first, status: "answered" as const, resolution };
    expect(mergeUserQuestionRecords([sameTimeAnswer], [first])).toEqual([sameTimeAnswer]);
    const delivered = { ...sameTimeAnswer, deliveredAt: "2026-10-02T12:02:00.000Z" };
    expect(mergeUserQuestionRecords([delivered], [sameTimeAnswer])).toEqual([delivered]);
    expect(first.status).toBe("pending");
  });

  it("labels assumptions distinctly and never marks a countdown expiry as an answered question", () => {
    const question = fixtureQuestion({ timeout: { enabled: true, seconds: 1200, policyRevision: "one", deadlineAt: "2026-10-02T12:20:00.000Z" } });
    expect(userQuestionCountdown(question, Date.parse("2026-10-02T12:00:00.010Z"))).toBe("Auto choice in 20:00");
    expect(userQuestionCountdown(question, Date.parse("2026-10-02T12:19:01.000Z"))).toBe("Auto choice in 0:59");
    expect(userQuestionCountdown(question, Date.parse("2026-10-02T12:22:00.000Z"))).toBe("Assumption pending");
    expect(userQuestionResolutionLabel(question)).toBe("Waiting for you");
    const assumed = fixtureQuestion({ status: "assumed", resolution: { id: "resolution", source: "timeout", resolvedAt: "2026-10-02T12:20:00.000Z", response: { optionIds: ["intro", "examples"] }, authority: "task-choice-only", permissionsGranted: false } });
    expect(userQuestionResolutionLabel(assumed)).toBe("Assumed after timeout");
    expect(userQuestionAnswerText(assumed)).toBe("Introduction · Examples");
    expect(userQuestionCountdown(assumed, Date.now())).toBeUndefined();
  });

  it("validates user-entered minutes and seconds without rounding or changing them silently", () => {
    expect(userQuestionTimeoutSeconds("20", "0")).toBe(1200);
    expect(userQuestionTimeoutSeconds("0", "15")).toBe(15);
    expect(userQuestionTimeoutSeconds("45", "59")).toBe(2759);
    for (const [minutes, seconds] of [["", "0"], ["-1", "20"], ["0", "0"], ["1", "60"], ["1.5", "0"], ["0", "0.5"]]) {
      expect(() => userQuestionTimeoutSeconds(minutes!, seconds!)).toThrow();
    }
  });
});
