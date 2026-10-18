import { createHash } from "node:crypto";
import { lstat } from "node:fs/promises";
import { join } from "node:path";
import { ClockStore, CLOCK_RELATIVE_PATH, clockHash, validateOccurrence, validateScheduleRecord, validateStopwatchRecord } from "./clockStore";
import { validTimeZone } from "./hostClock";
import type { NativeInferenceRate, ScheduleRecord, ScheduledOccurrence, StopwatchRecord } from "../shared/clock";
import type { AutomationMergeAliases } from "./automationPortability";

async function present(root: string): Promise<boolean> { return !!await lstat(join(root, CLOCK_RELATIVE_PATH)).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; }); }
export async function validateClockWorkspace(root: string, brainId: string): Promise<void> {
  if (!await present(root)) return;
  const store = await ClockStore.open(root, brainId, true);
  try {
    if (store.db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Saved clock database failed integrity verification.");
    for (const row of store.records<ScheduleRecord>("schedule")) { validateScheduleRecord(row, brainId); validTimeZone(row.timeZone); }
    for (const row of store.records<StopwatchRecord>("stopwatch")) {
      validateStopwatchRecord(row, brainId);
      if (Number(store.db.prepare("SELECT count(*) AS n FROM laps WHERE watch_id=?").get(row.id)?.n ?? 0) !== row.lapCount) throw new Error("Saved lap coverage differs from its stopwatch.");
      let elapsed = 0, number = 0;
      for (const lap of store.db.prepare("SELECT number,elapsed_ms,lap_ms FROM laps WHERE watch_id=? ORDER BY number").iterate(row.id)) {
        if (Number(lap.number) !== ++number || Number(lap.elapsed_ms) < elapsed || Number(lap.elapsed_ms) > row.elapsedMs || Math.abs(Number(lap.lap_ms) - (Number(lap.elapsed_ms) - elapsed)) > .001) throw new Error("Saved monotonic lap sequence is inconsistent.");
        elapsed = Number(lap.elapsed_ms);
      }
    }
    for (const row of store.records<ScheduledOccurrence>("occurrence")) { validateOccurrence(row, brainId); if (!store.get("schedule", row.scheduleId)) throw new Error("Saved wakeup lost its schedule owner."); }
    for (const row of store.db.prepare("SELECT payload FROM rates").iterate()) {
      const rate = JSON.parse(String(row.payload)) as NativeInferenceRate;
      if (!Number.isSafeInteger(rate.outputTokens) || rate.outputTokens < 0 || !Number.isFinite(rate.outputElapsedMs) || rate.outputElapsedMs < 0 || rate.source !== "native-output-stream" || rate.scope !== "one-turn-cumulative" || !Number.isFinite(Date.parse(rate.at)) ||
          (rate.tokensPerSecond === null ? rate.outputElapsedMs !== 0 : !Number.isFinite(rate.tokensPerSecond) || Math.abs(rate.tokensPerSecond - rate.outputTokens * 1000 / rate.outputElapsedMs) > 1e-6)) throw new Error("Saved speed differs from its count/duration metadata.");
    }
  } finally { store.close(); }
}

/** Only unpublished isolated copies: pause automation, retain exact input/effect receipts. */
export async function rekeyClockWorkspace(root: string, sourceBrainId: string, targetBrainId: string): Promise<void> {
  if (!await present(root)) return; await validateClockWorkspace(root, sourceBrainId);
  const store = await ClockStore.open(root, sourceBrainId);
  try { store.transaction(() => {
    store.db.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetBrainId);
    const rewrite = <T extends { state: string }>(kind: string, id: string, row: T): void => { store.db.prepare("UPDATE records SET state=?,payload=? WHERE kind=? AND id=?").run(row.state, JSON.stringify(row), kind, id); };
    for (const row of store.records<ScheduleRecord>("schedule")) rewrite("schedule", row.id, { ...row, brainId: targetBrainId,
      state: row.state === "active" ? "paused" : row.state, pendingPaused: true, pauseReason: "copied-instance", lineage: [...(row.lineage ?? []), { brainId: sourceBrainId, scheduleId: row.id }] });
    for (const row of store.records<StopwatchRecord>("stopwatch")) rewrite("stopwatch", row.id, { ...row, brainId: targetBrainId, state: "paused", pauseReason: "copied-instance", excludedUnmeasuredGap: row.excludedUnmeasuredGap || row.state === "running", sessionId: "copied-session" });
    for (const row of store.records<ScheduledOccurrence>("occurrence")) rewrite("occurrence", row.id, { ...row, brainId: targetBrainId, originBrainId: row.originBrainId ?? sourceBrainId,
      copiedUnstarted: row.copiedUnstarted || row.state === "pending" && !row.dispatchClaimedAt,
      ...(["committed", "notified", "delegated", "cancelled"].includes(row.state) ? {} : { state: "review-required", waitingFor: "review", detail: "Copied task and original receipt retained. Review its exact native continuation; no task or external effect was replayed." }) });
    // Preserve byte-exact historical commands without making their keys new-owner execution authority.
    store.db.exec("CREATE TABLE IF NOT EXISTS inherited_commands(parent_id TEXT,id TEXT,sha256 TEXT,result TEXT,PRIMARY KEY(parent_id,id));");
    if (sourceBrainId !== targetBrainId) {
      store.db.prepare("INSERT OR IGNORE INTO inherited_commands SELECT ?,id,sha256,result FROM commands").run(sourceBrainId);
      store.db.exec("DELETE FROM commands");
    }
  }); } finally { store.close(); }
  await validateClockWorkspace(root, targetBrainId);
}

/** Every row is retained; identical proven-ID/content variants deduplicate, conflicts stay separate. */
export async function mergeClockWorkspace(sourceRoot: string, targetRoot: string, owner: string, parentId: string, turnAliases?: ReadonlyMap<string, string>, automationAliases?: AutomationMergeAliases): Promise<void> {
  if (!await present(sourceRoot)) return; await validateClockWorkspace(sourceRoot, owner);
  const source = await ClockStore.open(sourceRoot, owner, true), target = await ClockStore.open(targetRoot, owner);
  const alias = (kind: string, id: string, body: unknown): string => clockHash(JSON.stringify({ kind, id, body }));
  const normalized = (value: unknown): unknown => { const row = { ...(value as Record<string, unknown>) }; delete row.brainId; delete row.lineage; delete row.authorizationRevision; return row; };
  try { target.transaction(() => {
    target.db.exec("CREATE TABLE IF NOT EXISTS clock_merge_aliases(parent_id TEXT,kind TEXT,source_id TEXT,target_id TEXT,PRIMARY KEY(parent_id,kind,source_id)); CREATE TABLE IF NOT EXISTS inherited_commands(parent_id TEXT,id TEXT,sha256 TEXT,result TEXT,PRIMARY KEY(parent_id,id)); CREATE TABLE IF NOT EXISTS inherited_events(parent_id TEXT,event_key TEXT,event_id TEXT,sha256 TEXT,PRIMARY KEY(parent_id,event_key,event_id));");
    for (const row of source.records<ScheduleRecord>("schedule")) {
      const id = alias("schedule", row.id, normalized(row));
      const previous = target.db.prepare("SELECT target_id FROM clock_merge_aliases WHERE parent_id=? AND kind='schedule' AND source_id=?").get(parentId, row.id);
      if (previous && previous.target_id !== id) throw new Error("The sealed parent clock changed during merge.");
      target.db.prepare("INSERT OR IGNORE INTO clock_merge_aliases VALUES(?,?,?,?)").run(parentId, "schedule", row.id, id);
      const definitionId = row.automation && automationAliases?.definitions.get(row.automation.definitionId);
      if (row.automation && !definitionId) throw new Error("Merged action schedule lost its exact action-file mapping.");
      if (!target.get("schedule", id)) target.put({ ...row, id, brainId: owner, ...(row.automation ? { automation: { ...row.automation, definitionId: definitionId! } } : {}), state: row.state === "active" ? "paused" : row.state, pendingPaused: true, pauseReason: "copied-instance", lineage: [...(row.lineage ?? []), { brainId: parentId, scheduleId: row.id }] }, "schedule");
    }
    for (const row of source.records<StopwatchRecord>("stopwatch")) {
      const digest = createHash("sha256"); for (const lap of source.db.prepare("SELECT * FROM laps WHERE watch_id=? ORDER BY number").iterate(row.id)) digest.update(JSON.stringify(lap));
      const id = alias("stopwatch", row.id, { row: normalized(row), laps: digest.digest("hex") }); if (target.get("stopwatch", id)) continue;
      target.put({ ...row, id, brainId: owner, state: "paused", pauseReason: "copied-instance" }, "stopwatch");
      for (const lap of source.db.prepare("SELECT * FROM laps WHERE watch_id=? ORDER BY number").iterate(row.id)) target.db.prepare("INSERT INTO laps VALUES(?,?,?,?,?)").run(id, Number(lap.number), Number(lap.elapsed_ms), Number(lap.lap_ms), String(lap.created_at));
    }
    for (const row of source.records<ScheduledOccurrence>("occurrence")) {
      const id = alias("occurrence", row.id, normalized(row));
      const scheduleId = target.db.prepare("SELECT target_id FROM clock_merge_aliases WHERE parent_id=? AND kind='schedule' AND source_id=?").get(parentId, row.scheduleId)?.target_id;
      if (typeof scheduleId !== "string") throw new Error("Merged wakeup lost its parent schedule.");
      const definitionId = row.automation && automationAliases?.definitions.get(row.automation.definitionId), delegatedRunId = row.delegatedRunId && automationAliases?.runs.get(row.delegatedRunId);
      if (row.automation && !definitionId || row.delegatedRunId && !delegatedRunId) throw new Error("Merged action occurrence lost its exact file/run mapping.");
      if (!target.get("occurrence", id)) target.put({ ...row, id, brainId: owner, scheduleId, turnId: turnAliases?.get(row.turnId) ?? row.turnId, originBrainId: row.originBrainId ?? parentId,
        ...(row.automation ? { automation: { ...row.automation, definitionId: definitionId! } } : {}), ...(delegatedRunId ? { delegatedRunId } : {}),
        ...(["committed", "notified", "delegated", "cancelled"].includes(row.state) ? {} : { state: "review-required" as const, waitingFor: "review" as const }) }, "occurrence");
    }
    for (const row of source.db.prepare("SELECT * FROM commands").iterate()) target.db.prepare("INSERT OR IGNORE INTO inherited_commands VALUES(?,?,?,?)").run(parentId, String(row.id), String(row.sha256), String(row.result));
    if (source.db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='inherited_commands'").get()) for (const row of source.db.prepare("SELECT * FROM inherited_commands").iterate()) target.db.prepare("INSERT OR IGNORE INTO inherited_commands VALUES(?,?,?,?)").run(String(row.parent_id), String(row.id), String(row.sha256), String(row.result));
    for (const row of source.db.prepare("SELECT * FROM rates").iterate()) target.db.prepare("INSERT OR IGNORE INTO rates VALUES(?,?)").run(alias("rate", String(row.turn_id), String(row.payload)), String(row.payload));
    for (const row of source.db.prepare("SELECT * FROM events").iterate()) {
      const inserted = target.db.prepare("INSERT OR IGNORE INTO inherited_events VALUES(?,?,?,?)").run(parentId, String(row.event_key), String(row.event_id), String(row.sha256));
      if (!Number(inserted.changes)) continue;
      const digest = createHash("sha256"); let count = 0, only = "";
      for (const item of target.db.prepare("SELECT DISTINCT sha256 FROM inherited_events WHERE event_key=? AND event_id=? ORDER BY sha256").iterate(String(row.event_key), String(row.event_id))) { count++; only = String(item.sha256); digest.update(only); }
      const combined = count === 1 ? only : digest.digest("hex");
      target.db.prepare("INSERT INTO events VALUES(?,?,?) ON CONFLICT(event_key,event_id) DO UPDATE SET sha256=excluded.sha256").run(String(row.event_key), String(row.event_id), combined);
    }
  }); } finally { source.close(); target.close(); }
  await validateClockWorkspace(targetRoot, owner);
}
