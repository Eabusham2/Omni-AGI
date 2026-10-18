import { createHash } from "node:crypto";
import { lstat, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { validateScheduledAutomationReference } from "./automationFormat";
import type { ClockPage, ClockPageQuery, ScheduleRecord, ScheduledOccurrence, StopwatchLap, StopwatchRecord } from "../shared/clock";

export const CLOCK_RELATIVE_PATH = "clock/workspace.sqlite3";
export const clockHash = (value: string): string => createHash("sha256").update(value).digest("hex");
export function clockId(value: unknown): string {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value)) throw new Error("Invalid clock owner or record identity.");
  return value;
}
export function clockObject(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid clock operation.");
  return value as Record<string, unknown>;
}
export function clockText(value: unknown, label: string, empty = false): string {
  if (typeof value !== "string" || value.includes("\0") || /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(value) || !empty && !value.trim()) throw new Error(`Enter ${label}.`);
  return value;
}
export function clockMilliseconds(value: unknown, minimum = 0): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < minimum || value > 8_640_000_000_000_000) throw new Error("Duration is outside supported clock precision.");
  return value;
}
export function clockInstant(value: unknown): string {
  if (typeof value !== "string" || !/(?:Z|[+-]\d\d:\d\d)$/.test(value) || !Number.isFinite(Date.parse(value))) throw new Error("A calendar instant needs an explicit UTC offset or Z.");
  return new Date(value).toISOString();
}
export function validateScheduleRecord(value: unknown, brainId: string): ScheduleRecord {
  const row = clockObject(value);
  clockId(row.id); if (row.brainId !== brainId) throw new Error("Saved schedule belongs to another instance.");
  clockText(row.label, "a schedule name"); clockText(row.task, "task text", true); clockMilliseconds(row.revision, 1); clockMilliseconds(row.nextIndex);
  if (typeof row.authorizationRevision !== "string" || !/^[a-f0-9]{64}$/.test(row.authorizationRevision)) throw new Error("Saved schedule has no permission-bound creation receipt.");
  if (!["timer", "at", "event"].includes(String(row.kind)) || !["active", "paused", "cancelled", "finished"].includes(String(row.state)) ||
      !["notify", "wake", "automation"].includes(String(row.delivery)) || !["coalesce", "skip"].includes(String(row.missedPolicy))) throw new Error("Invalid saved schedule state.");
  if (row.delivery === "automation") { validateScheduledAutomationReference(row.automation); if (row.task !== "") throw new Error("An automation schedule uses its typed file reference, not hidden task text."); }
  else if (row.automation !== undefined) throw new Error("Only automation delivery accepts an action-file reference.");
  if (row.delivery === "wake" && !String(row.task).trim()) throw new Error("A wakeup needs its saved task text.");
  if (typeof row.loadWhenDue !== "boolean" || row.loadWhenDue && row.delivery !== "wake") throw new Error("Scheduled Load is an explicit wakeup-only choice.");
  clockInstant(row.createdAt); clockInstant(row.updatedAt);
  if (row.dueAt !== undefined) clockInstant(row.dueAt);
  if (row.remainingMs !== undefined) clockMilliseconds(row.remainingMs);
  if (row.repeatMs !== undefined) clockMilliseconds(row.repeatMs, 1);
  if (row.kind === "event") clockText(row.eventKey, "an event key");
  else if (row.dueAt === undefined) throw new Error("A timed schedule has no deadline.");
  return row as unknown as ScheduleRecord;
}
export function validateStopwatchRecord(value: unknown, brainId: string): StopwatchRecord {
  const row = clockObject(value); clockId(row.id); clockId(row.sessionId);
  if (row.brainId !== brainId || !["running", "paused"].includes(String(row.state))) throw new Error("Invalid saved stopwatch owner/state.");
  clockText(row.label, "a stopwatch name"); clockMilliseconds(row.revision, 1); clockMilliseconds(row.lapCount); clockInstant(row.recordedAt);
  if (typeof row.elapsedMs !== "number" || !Number.isFinite(row.elapsedMs) || row.elapsedMs < 0) throw new Error("Invalid saved stopwatch duration.");
  return row as unknown as StopwatchRecord;
}
export function validateOccurrence(value: unknown, brainId: string): ScheduledOccurrence {
  const row = clockObject(value); for (const name of ["id", "scheduleId", "turnId"]) clockId(row[name]);
  if (row.brainId !== brainId || !["pending", "dispatching", "running", "committed", "notified", "delegated", "cancelled", "review-required"].includes(String(row.state)) ||
      !["notify", "wake", "automation"].includes(String(row.delivery))) throw new Error("Invalid wakeup owner/state.");
  if (row.delivery === "automation") { validateScheduledAutomationReference(row.automation); if (row.input !== "") throw new Error("An action occurrence cannot disguise a chat input."); }
  else if (row.automation !== undefined || row.state === "delegated") throw new Error("Only action delivery may retain a delegated run.");
  if (row.state === "delegated") clockId(row.delegatedRunId);
  clockMilliseconds(row.index); clockMilliseconds(row.missedIntervals); clockText(row.input, "task text", true);
  clockMilliseconds(row.scheduleRevision, 1);
  if (typeof row.authorizationRevision !== "string" || !/^[a-f0-9]{64}$/.test(row.authorizationRevision) || typeof row.loadWhenDue !== "boolean" || row.loadWhenDue && row.delivery !== "wake") throw new Error("Saved wakeup lost its exact authorization scope.");
  if (clockHash(String(row.input)) !== row.inputSha256) throw new Error("Saved wakeup task differs from its receipt.");
  for (const name of ["dueAt", "createdAt", "updatedAt"]) clockInstant(row[name]);
  return row as unknown as ScheduledOccurrence;
}

/** Short transactions, SQLite-owned cursors; never holds a neural lock or source data. */
export class ClockStore {
  private constructor(readonly db: DatabaseSync, readonly brainId: string) {}
  static async open(root: string, brainId: string, readOnly = false): Promise<ClockStore> {
    clockId(brainId); const directory = join(root, "clock"), path = join(root, CLOCK_RELATIVE_PATH);
    if (!readOnly) await mkdir(directory, { recursive: true, mode: 0o700 });
    const parent = await lstat(directory);
    if (!parent.isDirectory() || parent.isSymbolicLink()) throw new Error("Clock workspace must be an owned directory.");
    const file = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (file && (!file.isFile() || file.isSymbolicLink())) throw new Error("Clock database must be a regular owned file.");
    const db = new DatabaseSync(path, { readOnly });
    try {
      if (!readOnly) {
        // These short-lived metadata connections must not recreate WAL/SHM
        // files on every read-only poll. Writes still use atomic FULL-sync transactions.
        db.exec(`PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA busy_timeout=2500; PRAGMA cache_size=-256; PRAGMA temp_store=FILE;
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS records(kind TEXT NOT NULL,id TEXT NOT NULL,state TEXT NOT NULL,due_ms REAL,payload TEXT NOT NULL,PRIMARY KEY(kind,id));
          CREATE INDEX IF NOT EXISTS due_records ON records(kind,state,due_ms);
          CREATE TABLE IF NOT EXISTS laps(watch_id TEXT NOT NULL,number INTEGER NOT NULL,elapsed_ms REAL NOT NULL,lap_ms REAL NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(watch_id,number));
          CREATE TABLE IF NOT EXISTS commands(id TEXT PRIMARY KEY,sha256 TEXT NOT NULL,result TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS rates(turn_id TEXT PRIMARY KEY,payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events(event_key TEXT NOT NULL,event_id TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(event_key,event_id));`);
        db.prepare("INSERT OR IGNORE INTO meta VALUES('format','omni-clock-workspace-1')").run();
        db.prepare("INSERT OR IGNORE INTO meta VALUES('brainId',?)").run(brainId);
      }
      if (db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== "omni-clock-workspace-1" || db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) throw new Error("Clock workspace format or owner is incompatible.");
      return new ClockStore(db, brainId);
    } catch (error) { db.close(); throw error; }
  }
  close(): void { this.db.close(); }
  transaction<T>(operation: () => T): T {
    this.db.exec("BEGIN IMMEDIATE");
    try { const result = operation(); this.db.exec("COMMIT"); return result; }
    catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  get<T>(kind: "schedule" | "stopwatch" | "occurrence", id: string): T | undefined {
    const row = this.db.prepare("SELECT payload FROM records WHERE kind=? AND id=?").get(kind, clockId(id));
    return row ? JSON.parse(String(row.payload)) as T : undefined;
  }
  put(row: ScheduleRecord | StopwatchRecord | ScheduledOccurrence, kind: "schedule" | "stopwatch" | "occurrence"): void {
    if (kind === "schedule") validateScheduleRecord(row, this.brainId);
    else if (kind === "stopwatch") validateStopwatchRecord(row, this.brainId);
    else validateOccurrence(row, this.brainId);
    this.db.prepare("INSERT INTO records VALUES(?,?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET state=excluded.state,due_ms=excluded.due_ms,payload=excluded.payload")
      .run(kind, row.id, row.state, "dueAt" in row && row.dueAt !== undefined ? Date.parse(row.dueAt) : null, JSON.stringify(row));
  }
  page<T>(kind: "schedule" | "stopwatch" | "occurrence", query: ClockPageQuery = {}): ClockPage<T> {
    const after = query.after === undefined ? "" : clockId(query.after), limit = clockMilliseconds(query.limit ?? 50, 1);
    // Pagination bounds one IPC/render batch, never the stored record count.
    if (limit > 200) throw new Error("Use a page size from 1 to 200 and follow nextAfter.");
    const rows = this.db.prepare("SELECT id,payload FROM records WHERE kind=? AND id>? ORDER BY id LIMIT ?").all(kind, after, limit + 1);
    return { entries: rows.slice(0, limit).map(row => JSON.parse(String(row.payload)) as T),
      total: Number(this.db.prepare("SELECT count(*) AS n FROM records WHERE kind=?").get(kind)?.n ?? 0),
      ...(rows.length > limit ? { nextAfter: String(rows[limit - 1]!.id) } : {}) };
  }
  *records<T>(kind: "schedule" | "stopwatch" | "occurrence", states?: readonly string[]): Generator<T> {
    const query = `SELECT id,payload FROM records WHERE kind=? AND id>?${states?.length ? ` AND state IN (${states.map(() => "?").join(",")})` : ""} ORDER BY id LIMIT 50`;
    let after = "";
    while (true) {
      const rows = this.db.prepare(query).all(kind, after, ...(states ?? [])); if (!rows.length) return;
      for (const row of rows) { after = String(row.id); yield JSON.parse(String(row.payload)) as T; }
    }
  }
  laps(id: string, after = 0): { entries: StopwatchLap[]; nextAfter?: number } {
    clockId(id); clockMilliseconds(after);
    const rows = this.db.prepare("SELECT * FROM laps WHERE watch_id=? AND number>? ORDER BY number LIMIT 101").all(id, after);
    return { entries: rows.slice(0, 100).map(row => ({ number: Number(row.number), elapsedMs: Number(row.elapsed_ms), lapMs: Number(row.lap_ms), recordedAt: String(row.created_at) })),
      ...(rows.length > 100 ? { nextAfter: Number(rows[99]!.number) } : {}) };
  }
  command<T>(id: string, sha256: string, operation: () => T): T {
    return this.transaction(() => {
      const prior = this.db.prepare("SELECT sha256,result FROM commands WHERE id=?").get(clockId(id));
      if (prior) {
        if (prior.sha256 !== sha256) throw new Error("Clock request identity was reused with different arguments.");
        return JSON.parse(String(prior.result)) as T;
      }
      const result = operation(); this.db.prepare("INSERT INTO commands VALUES(?,?,?)").run(id, sha256, JSON.stringify(result)); return result;
    });
  }
}
