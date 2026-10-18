import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { AutomationRun, AutomationStepReceipt } from "../src/shared/automation";
import type { ScheduleRecord } from "../src/shared/clock";
import { AutomationWorkspace } from "../src/renderer/src/AutomationWorkspace";
import { actionFileInputs, actionScheduleDraft, actionScheduleEditRequest, actionScheduleRequest, pendingActionStep } from "../src/renderer/src/automationPresentation";

describe("action-file controls", () => {
  const ref = { definitionId: "file", revisionSha256: "a".repeat(64), inputs: { userText: "$ref is ordinary input" } };
  it("uses typed automation delivery without task prose or model Load", () => {
    const draft = actionScheduleDraft();
    const request = actionScheduleRequest(draft, ref);
    expect(request).toMatchObject({ delivery: "automation", automation: ref, loadWhenDue: false, delayMs: 600000 });
    expect(request).not.toHaveProperty("task");
  });
  it("retains exact edit revision and removes create-only fields", () => {
    const row = { id: "schedule", revision: 4, kind: "event" } as ScheduleRecord;
    const request = actionScheduleEditRequest(row, actionScheduleRequest({ ...actionScheduleDraft(), kind: "event", eventKey: "source.changed" }, ref));
    expect(request).toMatchObject({ scheduleId: "schedule", expectedRevision: 4, repeatMs: 0, task: "", automation: ref });
    expect(request).not.toHaveProperty("kind"); expect(request).not.toHaveProperty("eventKey");
  });
  it("rejects non-object inputs and invalid/overflowing durations", () => {
    for (const value of ["null", "[]", "4", "\"text\""]) expect(() => actionFileInputs(value)).toThrow("JSON object");
    for (const value of ["0", "-2", "Infinity", "999999999999999999999"]) expect(() => actionScheduleRequest({ ...actionScheduleDraft(), minutes: value }, ref)).toThrow("positive duration");
    expect(actionFileInputs('{"message":"$ref","values":[1,true]}')).toEqual({ message: "$ref", values: [1, true] });
  });
  it("finds only the exact run frontier's latest attempt", () => {
    const run = { id: "run", nextStep: 7 } as AutomationRun;
    const rows = [{ runId: "other", index: 7, attempt: 99 }, { runId: "run", index: 6, attempt: 99 },
      { runId: "run", index: 7, attempt: 1 }, { runId: "run", index: 7, attempt: 2 }] as AutomationStepReceipt[];
    expect(pendingActionStep(run, rows)).toBe(rows[3]);
    expect(pendingActionStep({ ...run, nextStep: 8 }, rows)).toBeUndefined();
  });
  it("does not create a run or grant authority while rendering the inspection surface", () => {
    const html = renderToStaticMarkup(<AutomationWorkspace brainId="read-only" />);
    expect(html).toContain("Action files"); expect(html).toContain("Create action file");
    expect(html).toContain("Run activity"); expect(html).not.toContain("Approve this request");
  });
});
