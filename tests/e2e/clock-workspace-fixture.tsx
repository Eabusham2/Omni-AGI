import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { ClockWorkspacePanel } from "../../src/renderer/src/ClockWorkspacePanel";
import type { ClockWorkspaceApi, ClockWorkspaceEvent, ScheduleRecord, StopwatchRecord } from "../../src/shared/clock";
import type { ToolInvocation, ToolExecutionResult } from "../../src/shared/types";
import "../../src/renderer/src/styles.css";

// Presentation fixture only. No ClockService, native worker, persistent schedule,
// notification, permission grant, model Load or external action is created.
const brainId = "clock-ui-fixture", now = () => new Date().toISOString();
const listeners = new Set<(event: ClockWorkspaceEvent) => void>();
const receipts: unknown[] = [];
const schedules: ScheduleRecord[] = [{ id: "saved", brainId, label: "Review source evidence before drafting the summary", revision: 1,
  kind: "event", state: "paused", delivery: "wake", task: "Read only the selected public sources, compare the evidence and report disagreements.",
  timeZone: "America/Chicago", eventKey: "source.changed", missedPolicy: "coalesce", loadWhenDue: false,
  nextIndex: 0, authorizationRevision: "synthetic-only", createdAt: now(), updatedAt: now(), pauseReason: "copied-instance" }];
const watches: StopwatchRecord[] = [];
const watchAnchors = new Map<string, number>();
const watchLaps = new Map<string, { number: number; elapsedMs: number; lapMs: number; recordedAt: string }[]>();
const elapsed = (row: StopwatchRecord) => row.elapsedMs + (row.state === "running" ? performance.now() - (watchAnchors.get(row.id) ?? performance.now()) : 0);
let renderReceipts = () => {};
const changed = () => listeners.forEach(listener => listener({ brainId, type: "changed", at: now() }));
const clock: ClockWorkspaceApi = {
  now: async () => ({ epochMilliseconds: Date.now(), utc: now(), local: now(), timeZone: "America/Chicago", utcOffsetMinutes: -300, monotonicMilliseconds: performance.now(), sessionId: "synthetic-ui" }),
  snapshot: async () => ({ brainId, clock: await clock.now(), schedulerRunning: true, counts: { schedules: schedules.length, stopwatches: watches.length, pending: 1, reviewRequired: 1 }, processResidentRequired: true }),
  schedules: async () => ({ entries: schedules.map(row => ({ ...structuredClone(row), ...(row.kind === "timer" && row.dueAt ? { remainingMs: Math.max(0, Date.parse(row.dueAt) - Date.now()) } : {}) })), total: schedules.length }),
  stopwatches: async () => ({ entries: watches.map(row => ({ ...row, elapsedMs: elapsed(row) })), total: watches.length }),
  laps: async (_id, watchId) => ({ entries: structuredClone(watchLaps.get(watchId) ?? []) }),
  occurrences: async () => ({ total: 1, entries: [{ id: "pending", brainId, scheduleId: "saved", index: 0, turnId: "fixture-turn", input: "Review the selected public evidence.", inputSha256: "a".repeat(64), delivery: "wake", label: "Copied task awaiting current access review", authorizationRevision: "synthetic-only", scheduleRevision: 1, loadWhenDue: false, dueAt: now(), createdAt: now(), updatedAt: now(), missedIntervals: 2, state: "review-required", waitingFor: "permission", detail: "Fixture only: previous permission is not carried into a copied instance." }] }),
  rates: async () => ({ entries: [], total: 0 }),
  onEvent: listener => { listeners.add(listener); return () => { listeners.delete(listener); }; }
};
async function execute(invocation: ToolInvocation): Promise<ToolExecutionResult> {
  const result: ToolExecutionResult = { id: crypto.randomUUID(), toolId: invocation.toolId, action: invocation.action, startedAt: now(), state: "complete", finishedAt: now() };
  if (!invocation.approvalToken) return { ...result, state: "approval-required", approvalToken: "synthetic-ui-approval", approvalExpiresAt: new Date(Date.now() + 60000).toISOString() };
  const args = invocation.arguments;
  receipts.push({ toolId: invocation.toolId, action: invocation.action, args, externalEffects: false });
  if (invocation.toolId === "studio.schedule") {
    const row = schedules.find(item => item.id === args.scheduleId);
    if (invocation.action === "create") schedules.push({ id: crypto.randomUUID(), brainId, label: String(args.label), revision: 1, kind: args.kind as ScheduleRecord["kind"], state: "active", delivery: args.delivery as ScheduleRecord["delivery"], task: String(args.task ?? ""), timeZone: String(args.timeZone), loadWhenDue: args.loadWhenDue === true, ...(args.kind === "timer" ? { remainingMs: Number(args.delayMs), dueAt: new Date(Date.now() + Number(args.delayMs)).toISOString() } : args.kind === "at" ? { dueAt: String(args.at) } : { eventKey: String(args.eventKey) }), ...(args.repeatMs ? { repeatMs: Number(args.repeatMs) } : {}), missedPolicy: args.missedPolicy as ScheduleRecord["missedPolicy"], nextIndex: 0, authorizationRevision: "synthetic-only", createdAt: now(), updatedAt: now() });
    else if (row && invocation.action === "edit") Object.assign(row, { label: String(args.label), task: String(args.task), delivery: args.delivery, loadWhenDue: args.loadWhenDue === true, revision: row.revision + 1, updatedAt: now() });
    else if (row) Object.assign(row, { state: invocation.action === "cancel" ? "cancelled" : invocation.action === "pause" ? "paused" : "active", revision: row.revision + 1 });
  } else if (invocation.toolId === "studio.stopwatch") {
    const row = watches.find(item => item.id === args.stopwatchId);
    if (invocation.action === "start") { const id = crypto.randomUUID(); watches.push({ id, brainId, label: String(args.label), revision: 1, state: "running", elapsedMs: 0, lapCount: 0, recordedAt: now(), sessionId: "synthetic-ui" }); watchAnchors.set(id, performance.now()); }
    else if (row && invocation.action === "lap") { const laps = watchLaps.get(row.id) ?? [], total = elapsed(row); laps.push({ number: laps.length + 1, elapsedMs: total, lapMs: total - (laps.at(-1)?.elapsedMs ?? 0), recordedAt: now() }); watchLaps.set(row.id, laps); row.lapCount = laps.length; }
    else if (row && invocation.action === "pause") Object.assign(row, { elapsedMs: elapsed(row), state: "paused", pauseReason: "user" });
    else if (row && invocation.action === "resume") { row.state = "running"; watchAnchors.set(row.id, performance.now()); }
    else if (row && invocation.action === "reset") { row.elapsedMs = 0; row.lapCount = 0; watchLaps.set(row.id, []); watchAnchors.set(row.id, performance.now()); }
  }
  changed(); renderReceipts(); return result;
}
window.omni = { clock, tool: { execute } } as unknown as typeof window.omni;
function Fixture() {
  const [narrow, setNarrow] = useState(false), [notice, setNotice] = useState("");
  const [, update] = useState(0); renderReceipts = () => update(value => value + 1);
  return <main style={{ maxWidth: 1100, margin: "0 auto", padding: 16, height: "100vh", overflowY: "auto", boxSizing: "border-box" }}>
    <h1>Time & schedules control check</h1><p>Synthetic interface only. No model, real timer, notification or external action runs.</p>
    <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBlock: 16 }}><button className="button" onClick={() => setNarrow(value => !value)}>{narrow ? "Full surface" : "320px surface"}</button><button className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark"; }}>Switch appearance</button></div>
    <p role="status">{notice}</p><div style={{ width: narrow ? 320 : "100%", maxWidth: "100%" }}><ClockWorkspacePanel brainId={brainId} onToast={setNotice} onActiveMode={() => setNotice("Fixture navigation: open the instance’s Active mode settings.")} /></div>
    <details><summary>Synthetic action receipts</summary><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{JSON.stringify(receipts, null, 2)}</pre></details>
  </main>;
}
document.documentElement.dataset.colorScheme = "dark";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
