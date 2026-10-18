import type { AutomationJson, AutomationProgram, AutomationRun, AutomationStepReceipt } from "../../shared/automation";
import type { ScheduleCreateRequest, ScheduleRecord } from "../../shared/clock";

export const NEW_ACTION_FILE: AutomationProgram = {
  format: "omni-host-action-file", version: 1, name: "My action file",
  steps: [{ id: "read-time", kind: "tool", toolId: "system.time", action: "now", arguments: {} }]
};

export function actionFileInputs(value: string): Record<string, AutomationJson> {
  const parsed: unknown = JSON.parse(value);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Inputs must be a JSON object.");
  return parsed as Record<string, AutomationJson>;
}

export function pendingActionStep(run: AutomationRun, steps: readonly AutomationStepReceipt[]): AutomationStepReceipt | undefined {
  return steps.filter(step => step.runId === run.id && step.index === run.nextStep)
    .reduce<AutomationStepReceipt | undefined>((latest, step) => !latest || step.attempt > latest.attempt ? step : latest, undefined);
}

export interface ActionScheduleDraft {
  label: string; kind: "timer" | "at" | "event"; minutes: string; at: string; eventKey: string;
  repeat: boolean; repeatMinutes: string; missedPolicy: "coalesce" | "skip";
}
function durationMinutes(value: string): number {
  const milliseconds = Math.round(Number(value) * 60000);
  if (!Number.isSafeInteger(milliseconds) || milliseconds <= 0) throw new Error("Enter a positive duration that this device can represent.");
  return milliseconds;
}
export function actionScheduleRequest(draft: ActionScheduleDraft, ref: NonNullable<ScheduleRecord["automation"]>): ScheduleCreateRequest {
  if (!draft.label.trim()) throw new Error("Give the schedule a name.");
  let trigger: Partial<ScheduleCreateRequest>;
  if (draft.kind === "timer") trigger = { delayMs: durationMinutes(draft.minutes) };
  else if (draft.kind === "event") {
    if (!draft.eventKey.trim()) throw new Error("Enter the tracker event key.");
    trigger = { eventKey: draft.eventKey.trim() };
  } else {
    const date = new Date(draft.at);
    if (!draft.at || !Number.isFinite(date.getTime())) throw new Error("Enter a valid date and time.");
    trigger = { at: date.toISOString() };
  }
  return { label: draft.label.trim(), kind: draft.kind, delivery: "automation", automation: ref,
    loadWhenDue: false, timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    missedPolicy: draft.missedPolicy, ...trigger,
    ...(draft.repeat && draft.kind !== "event" ? { repeatMs: durationMinutes(draft.repeatMinutes) } : {}) };
}

export function actionScheduleDraft(row?: ScheduleRecord): ActionScheduleDraft {
  let at = "";
  if (row?.dueAt) {
    const date = new Date(row.dueAt);
    if (Number.isFinite(date.getTime())) at = new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 23);
  }
  return { label: row?.label ?? "Run action file", kind: row?.kind ?? "timer",
    minutes: String((row?.remainingMs ?? 600000) / 60000), at,
    eventKey: row?.eventKey ?? "source.changed", repeat: !!row?.repeatMs,
    repeatMinutes: String((row?.repeatMs ?? 3600000) / 60000), missedPolicy: row?.missedPolicy ?? "coalesce" };
}

/** The edit endpoint rejects create-only kind/eventKey fields. */
export function actionScheduleEditRequest(row: ScheduleRecord, request: ScheduleCreateRequest): Record<string, unknown> {
  const { kind: _kind, eventKey: _eventKey, ...editable } = request;
  return { ...editable, scheduleId: row.id, expectedRevision: row.revision, task: "", repeatMs: request.repeatMs ?? 0 };
}
