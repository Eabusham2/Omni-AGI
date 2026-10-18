import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { UserQuestionCard, UserQuestionHistory, UserQuestionSettings } from "../src/renderer/src/UserQuestionPanel";
import type { UserQuestionRecord } from "../src/shared/userQuestions";

const question = (overrides: Partial<UserQuestionRecord> = {}): UserQuestionRecord => ({
  format: "omni-user-question", version: 1, id: "fixture", brainId: "one", requestId: "request", requestSha256: "fixture-only",
  kind: "task-choice", prompt: "Which sections?", detail: "Full details remain available.\nDo not lose the long explanation.",
  mode: "multiple", minSelections: 2, maxSelections: 2, allowOther: true,
  options: [{ id: "intro", label: "Introduction", recommended: true, rank: 1 }, { id: "examples", label: "Examples", rank: 2 }],
  createdAt: "2026-10-02T12:00:00.000Z", updatedAt: "2026-10-02T12:00:00.000Z", status: "pending",
  timeout: { enabled: false, seconds: 1200, policyRevision: "default" }, ...overrides
});
const actions = { onRespond: async () => {}, onSkip: async () => {}, onResume: async () => {} };

describe("accessible question panel markup", () => {
  it("associates the question, checkbox choices, bounds and expandable detail with an answer form", () => {
    const markup = renderToStaticMarkup(<UserQuestionCard question={question()} now={0} {...actions} />);
    expect(markup).toContain("<fieldset");
    expect(markup).toContain("<legend");
    expect(markup).toContain("aria-labelledby=");
    expect(markup).toContain("aria-describedby=");
    expect(markup.match(/type="checkbox"/g)).toHaveLength(3);
    expect(markup).toContain("Choose 2 options.");
    expect(markup).toContain("Recommended");
    expect(markup).toContain("Other answer");
    expect(markup).toContain("<summary>Details</summary>");
    expect(markup).toContain("Do not lose the long explanation.");
    expect(markup).toMatch(/type="submit"[^>]*disabled/);
  });

  it("uses radio controls for single choice and a labelled textarea for free responses", () => {
    const radio = renderToStaticMarkup(<UserQuestionCard question={question({ mode: "single", minSelections: 1, maxSelections: 1 })} now={0} {...actions} />);
    expect(radio.match(/type="radio"/g)).toHaveLength(3);
    const written = renderToStaticMarkup(<UserQuestionCard question={question({ mode: "text", options: [], minSelections: 0, maxSelections: 0, allowOther: false })} now={0} {...actions} />);
    expect(written).toContain("Your answer");
    expect(written).toContain("<textarea");
    expect(written).not.toContain('type="radio"');
    expect(written).not.toContain('type="checkbox"');
  });

  it("keeps historical assumptions collapsed and accurately labelled", () => {
    const markup = renderToStaticMarkup(<UserQuestionHistory question={question({ status: "assumed", resolution: {
      id: "resolution", source: "timeout", resolvedAt: "2026-10-02T12:20:00.000Z", response: { optionIds: ["intro"] }, authority: "task-choice-only", permissionsGranted: false
    } })} />);
    expect(markup).toContain("Assumed after timeout");
    expect(markup).toContain("Introduction");
    expect(markup).not.toContain("Answered by you");
    expect(markup).not.toMatch(/<details[^>]* open/);
  });

  it("presents paused copies without active response controls and preserves the default disabled timeout", () => {
    const paused = renderToStaticMarkup(<UserQuestionCard question={question({ suspended: { reason: "copied-instance", fromBrainId: "parent", copiedAt: "2026-10-02T12:00:00.000Z" } })} now={0} {...actions} />);
    expect(paused).toMatch(/<fieldset[^>]*disabled/);
    expect(paused).toContain("Resume question");
    expect(paused).not.toContain("Send answer");
    const settings = renderToStaticMarkup(<UserQuestionSettings preferences={{ timeoutEnabled: false, timeoutSeconds: 1200 }} onSave={async (value) => value} />);
    expect(settings).toContain("Minutes");
    expect(settings).toContain("Seconds");
    expect(settings).toContain('value="20"');
    expect(settings).toContain('value="0"');
    expect(settings).not.toMatch(/type="checkbox"[^>]*checked/);
    expect(settings).toContain("Permissions always use their own approval.");
  });
});
