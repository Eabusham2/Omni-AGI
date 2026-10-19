import { createHash, randomUUID } from "node:crypto";
import { mkdir, lstat } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import type { ImaginationCaptureSettings, ImaginationCapturedFragment, ImaginationRecording } from "../shared/imaginationCapture";
import { sqliteFilename } from "./sqliteFilename";

export const IMAGINATION_WORKSPACE = "imagination/workspace.sqlite3";
export const captureHash = (value: unknown) => createHash("sha256").update(JSON.stringify(value)).digest("hex");
const safeId = (value: string) => { if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/.test(value)) throw new Error("Invalid imagination capture identity."); return value; };
const admitRead = (bytes: number) => { if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes * 8 > Math.min(freemem(), getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed)) throw new Error("Imagination metadata awaits transfer memory; nothing was truncated."); };
export function captureSeconds(value: number): number { if (!Number.isFinite(value) || value <= 0 || value > 300 || Math.round(value * 1000) !== value * 1000) throw new Error("Capture duration must be greater than zero and at most 300 seconds, to millisecond precision."); return value; }
type StoredFragment = Omit<ImaginationCapturedFragment, "mediaUrl" | "available">;
export interface CaptureWindowFragment {
  id: string; observedAt: string; ageMs: number; remainingMs: number; sha256: string; mimeType: string; bytes: number;
  inheritedId?: string;
  source: Omit<import("./imaginationCapture").DeliveredImaginationFragment, "resolve">;
}
export class ImaginationCaptureStore {
  private constructor(readonly db: DatabaseSync, readonly brainId: string) {}
  static async open(root: string, brainId: string, create = false): Promise<ImaginationCaptureStore | undefined> {
    safeId(brainId); const path = join(root, IMAGINATION_WORKSPACE), present = await lstat(path).catch(error => { if (error.code === "ENOENT") return undefined; throw error; });
    if (!present && !create) return undefined;
    if (create) await mkdir(join(root, "imagination"), { recursive: true, mode: 0o700 });
    for (const directory of [root, join(root, "imagination")]) { const info = await lstat(directory); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Capture workspace must be an owned real directory."); }
    if (present && (!present.isFile() || present.isSymbolicLink())) throw new Error("Unsafe imagination database.");
    const db = new DatabaseSync(sqliteFilename(path), { readOnly: !create });
    try {
      db.exec("PRAGMA cache_size=-128; PRAGMA mmap_size=0; PRAGMA busy_timeout=5000;");
      if (!present) {
        db.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE recordings(sequence INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,request_id TEXT UNIQUE NOT NULL,body TEXT NOT NULL,sha TEXT NOT NULL); CREATE TABLE fragments(recording_id TEXT NOT NULL,fragment_index INTEGER NOT NULL,id TEXT NOT NULL,blob_sha TEXT NOT NULL,body TEXT NOT NULL,sha TEXT NOT NULL,PRIMARY KEY(recording_id,fragment_index),UNIQUE(recording_id,id)); CREATE TABLE events(sequence INTEGER PRIMARY KEY,id TEXT NOT NULL,body TEXT NOT NULL,sha TEXT NOT NULL,previous_sha TEXT NOT NULL,row_sha TEXT NOT NULL); CREATE TABLE live_window(id TEXT PRIMARY KEY,body TEXT NOT NULL,sha TEXT NOT NULL);");
        db.prepare("INSERT INTO meta VALUES('format','omni-imagination-capture-1')").run(); db.prepare("INSERT INTO meta VALUES('brainId',?)").run(brainId);
        db.prepare("INSERT INTO meta VALUES('settings',?)").run(JSON.stringify({ version: 1, revision: 1, bufferSeconds: 300, alwaysSave: false, dormant: false }));
        db.prepare("INSERT INTO meta VALUES('eventHead',?)").run(JSON.stringify({ sequence: 0, sha: "0".repeat(64) }));
        db.prepare("INSERT INTO meta VALUES('windowHead',?)").run(captureHash([]));
      }
      const objects = db.prepare("SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").all();
      if (objects.length !== 5 || objects.some(row => row.type !== "table" || !["meta", "recordings", "fragments", "events", "live_window"].includes(String(row.name)))) throw new Error("Capture database contains an unsupported schema or trigger.");
      if (db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== "omni-imagination-capture-1" || db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) throw new Error("Capture workspace has another owner/format.");
      return new ImaginationCaptureStore(db, brainId);
    } catch (error) { db.close(); throw error; }
  }
  close(): void { this.db.close(); }
  private transaction<T>(work: () => T): T { this.db.exec("BEGIN IMMEDIATE"); try { const result = work(); this.db.exec("COMMIT"); return result; } catch (error) { this.db.exec("ROLLBACK"); throw error; } }
  settings(): ImaginationCaptureSettings {
    const result = JSON.parse(String(this.db.prepare("SELECT value FROM meta WHERE key='settings'").get()?.value));
    if (result.version !== 1 || !Number.isSafeInteger(result.revision) || result.revision < 1 || typeof result.alwaysSave !== "boolean" || typeof result.dormant !== "boolean") throw new Error("Invalid imagination settings."); captureSeconds(result.bufferSeconds); return result;
  }
  setSettings(expectedRevision: number, bufferSeconds: number, alwaysSave: boolean): ImaginationCaptureSettings {
    captureSeconds(bufferSeconds); if (typeof alwaysSave !== "boolean") throw new Error("Invalid always-save setting.");
    return this.transaction(() => { const old = this.settings(); if (old.revision !== expectedRevision) throw new Error("Capture settings changed; refresh before applying.");
      const next = { ...old, revision: old.revision + 1, bufferSeconds, alwaysSave, dormant: false };
      this.db.prepare("UPDATE meta SET value=? WHERE key='settings'").run(JSON.stringify(next)); return next; });
  }
  private event(id: string, value: unknown): void {
    const prior = this.db.prepare("SELECT sequence,row_sha FROM events ORDER BY sequence DESC LIMIT 1").get();
    const sequence = Number(prior?.sequence ?? 0) + 1, previous = String(prior?.row_sha ?? "0".repeat(64)), body = JSON.stringify(value), sha = captureHash(value);
    const rowSha = captureHash({ sequence, id, sha, previous });
    this.db.prepare("INSERT INTO events VALUES(?,?,?,?,?,?)").run(sequence, id, body, sha, previous, rowSha);
    this.db.prepare("UPDATE meta SET value=? WHERE key='eventHead'").run(JSON.stringify({ sequence, sha: rowSha }));
  }
  get(id: string): ImaginationRecording | undefined {
    admitRead(Number(this.db.prepare("SELECT length(CAST(body AS BLOB)) AS bytes FROM recordings WHERE id=?").get(safeId(id))?.bytes ?? 0));
    const row = this.db.prepare("SELECT * FROM recordings WHERE id=?").get(safeId(id)); if (!row) return undefined;
    const value = JSON.parse(String(row.body));
    if (captureHash(value) !== row.sha || value.format !== "omni-imagination-recording" || value.version !== 1 || value.brainId !== this.brainId || value.id !== id || value.sequence !== row.sequence || value.requestId !== row.request_id ||
        !["recording", "complete", "cancelled", "resource-paused", "dormant"].includes(value.state) || value.representation !== "verified-delivered-fragment-timeline") throw new Error("Capture recording changed or has another owner.");
    return value;
  }
  request(requestId: string): ImaginationRecording | undefined { const row = this.db.prepare("SELECT id FROM recordings WHERE request_id=?").get(safeId(requestId)); return row ? this.get(String(row.id)) : undefined; }
  create(requestId: string, mode: ImaginationRecording["mode"], requestedLookbackSeconds: number, availableLookbackSeconds: number, at: string): ImaginationRecording {
    const prior = this.request(requestId); if (prior) return prior;
    return this.transaction(() => {
      const id = randomUUID(), sequence = Number(this.db.prepare("SELECT MAX(sequence) AS n FROM recordings").get()?.n ?? 0) + 1;
      const value: ImaginationRecording = { format: "omni-imagination-recording", version: 1, id, brainId: this.brainId, sequence, requestId, mode, state: "recording", revision: 1, controlRevision: 1,
        startedAt: at, requestedLookbackSeconds, availableLookbackSeconds, fragmentCount: 0, bytes: 0, gapCount: 0,
        representation: "verified-delivered-fragment-timeline", label: "Delivered fragments — not a completed video" };
      this.db.prepare("INSERT INTO recordings VALUES(?,?,?,?,?)").run(sequence, id, requestId, JSON.stringify(value), captureHash(value)); this.event(id, value); return value;
    });
  }
  update(id: string, expectedRevision: number, work: (recording: ImaginationRecording) => void): ImaginationRecording {
    return this.transaction(() => { const value = this.get(id); if (!value || value.revision !== expectedRevision) throw new Error("Recording changed; refresh before applying.");
      work(value); value.revision++; value.controlRevision = value.revision; this.db.prepare("UPDATE recordings SET body=?,sha=? WHERE id=?").run(JSON.stringify(value), captureHash(value), id); this.event(id, value); return value; });
  }
  append(id: string, fragment: Omit<StoredFragment, "recordingId" | "index">): ImaginationRecording {
    return this.transaction(() => { const current = this.get(id); if (!current || current.state !== "recording") throw new Error("Recording is no longer active.");
      if (this.db.prepare("SELECT id FROM fragments WHERE recording_id=? AND id=?").get(id, fragment.id)) return current;
      const value = { ...fragment, recordingId: id, index: current.fragmentCount }, body = JSON.stringify(value);
      this.db.prepare("INSERT INTO fragments VALUES(?,?,?,?,?,?)").run(id, value.index, value.id, value.sha256, body, captureHash(value));
      current.fragmentCount++; current.bytes += fragment.bytes; current.revision++;
      this.db.prepare("UPDATE recordings SET body=?,sha=? WHERE id=?").run(JSON.stringify(current), captureHash(current), id); this.event(id, { recording: current, fragment: value }); return current;
    });
  }
  addGaps(id: string, count: number): void { if (!Number.isSafeInteger(count) || count < 1) return;
    this.transaction(() => { const value = this.get(id); if (!value || value.state !== "recording") return; value.gapCount += count; value.revision++;
      this.db.prepare("UPDATE recordings SET body=?,sha=? WHERE id=?").run(JSON.stringify(value), captureHash(value), id); this.event(id, value); }); }
  list(beforeSequence = Number.MAX_SAFE_INTEGER, limit = 20) {
    if (!Number.isSafeInteger(beforeSequence) || beforeSequence < 1 || !Number.isSafeInteger(limit) || limit < 1 || limit > 100) throw new Error("Invalid capture page bounds.");
    const entries = this.db.prepare("SELECT id FROM recordings WHERE sequence<? ORDER BY sequence DESC LIMIT ?").all(beforeSequence, limit).map(row => this.get(String(row.id))!);
    return { entries, total: Number(this.db.prepare("SELECT COUNT(*) AS n FROM recordings").get()?.n ?? 0), ...(entries.length === limit ? { nextBeforeSequence: entries.at(-1)!.sequence } : {}) };
  }
  fragments(id: string, afterIndex = -1, limit = 20): { entries: StoredFragment[]; total: number; nextAfterIndex?: number } {
    if (!Number.isSafeInteger(afterIndex) || afterIndex < -1 || !Number.isSafeInteger(limit) || limit < 1 || limit > 100) throw new Error("Invalid fragment page bounds.");
    const current = this.get(id); if (!current) throw new Error("Recording is unavailable.");
    admitRead(Number(this.db.prepare("SELECT SUM(bytes) AS total FROM (SELECT length(CAST(body AS BLOB)) AS bytes FROM fragments WHERE recording_id=? AND fragment_index>? ORDER BY fragment_index LIMIT ?)").get(id, afterIndex, limit)?.total ?? 0));
    const entries = this.db.prepare("SELECT * FROM fragments WHERE recording_id=? AND fragment_index>? ORDER BY fragment_index LIMIT ?").all(id, afterIndex, limit).map(row => {
      const value = JSON.parse(String(row.body)) as StoredFragment;
      if (captureHash(value) !== row.sha || value.recordingId !== id || value.index !== row.fragment_index || value.id !== row.id || value.sha256 !== row.blob_sha || !/^[a-f0-9]{64}$/.test(value.sha256) || value.origin.kind !== "spontaneous") throw new Error("Stored capture fragment changed."); return value;
    }); return { entries, total: current.fragmentCount, ...(entries.length === limit ? { nextAfterIndex: entries.at(-1)!.index } : {}) };
  }
  validate(): void {
    if (this.db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Capture database integrity failed."); this.settings();
    for (const row of this.db.prepare("SELECT id FROM recordings").iterate()) { const id = String(row.id); this.get(id); let after = -1;
      do { const page = this.fragments(id, after, 100); if (page.nextAfterIndex === undefined) break; after = page.nextAfterIndex; } while (true); }
    let previous = "0".repeat(64), sequence = 0;
    for (const row of this.db.prepare("SELECT * FROM events ORDER BY sequence").iterate()) {
      const sha = captureHash(JSON.parse(String(row.body)));
      if (row.sequence !== ++sequence || row.sha !== sha || row.previous_sha !== previous || row.row_sha !== captureHash({ sequence, id: row.id, sha, previous })) throw new Error("Capture event history changed."); previous = String(row.row_sha);
    }
    if (JSON.stringify({ sequence, sha: previous }) !== this.db.prepare("SELECT value FROM meta WHERE key='eventHead'").get()?.value) throw new Error("Capture history lost its committed head.");
    this.window();
  }
  window(): CaptureWindowFragment[] { admitRead(Number(this.db.prepare("SELECT SUM(length(CAST(body AS BLOB))) AS bytes FROM live_window").get()?.bytes ?? 0));
    const entries = this.db.prepare("SELECT * FROM live_window ORDER BY id").all().map(row => {
    const value = JSON.parse(String(row.body)) as CaptureWindowFragment;
    if (value.id !== row.id || captureHash(value) !== row.sha || !/^[a-f0-9]{64}$/.test(value.sha256) || !Number.isSafeInteger(value.bytes) || value.bytes < 1 ||
        !Number.isFinite(value.ageMs) || value.ageMs < 0 || !Number.isFinite(value.remainingMs) || value.remainingMs < 0 || value.remainingMs > 300000 || value.source.origin.kind !== "spontaneous") throw new Error("Live imagination window failed custody validation."); return value;
  }); if (captureHash(entries) !== this.db.prepare("SELECT value FROM meta WHERE key='windowHead'").get()?.value) throw new Error("Live imagination window lost its complete custody head."); return entries; }
  windowClock(): { paused: boolean; at: string } { const row = this.db.prepare("SELECT value FROM meta WHERE key='windowClock'").get();
    if (!row) return { paused: true, at: new Date().toISOString() }; const value = JSON.parse(String(row.value)); if (typeof value.paused !== "boolean" || !Number.isFinite(Date.parse(value.at))) throw new Error("Invalid rolling window clock."); return value; }
  setWindow(fragments: CaptureWindowFragment[], paused: boolean, at: string): void { this.transaction(() => {
    this.db.prepare("DELETE FROM live_window").run(); for (const value of fragments) this.db.prepare("INSERT INTO live_window VALUES(?,?,?)").run(value.id, JSON.stringify(value), captureHash(value));
    this.db.prepare("UPDATE meta SET value=? WHERE key='windowHead'").run(captureHash([...fragments].sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0)));
    this.db.prepare("INSERT OR REPLACE INTO meta VALUES('windowClock',?)").run(JSON.stringify({ paused, at })); }); }
  rekey(targetId: string): void {
    safeId(targetId); this.transaction(() => {
      for (const row of this.db.prepare("SELECT id FROM recordings ORDER BY sequence").iterate()) {
        const value = this.get(String(row.id))!; value.inheritedFrom ??= { brainId: this.brainId, recordingId: value.id }; value.brainId = targetId;
        if (value.state === "recording") value.state = "dormant"; value.revision++; value.controlRevision = value.revision;
        this.db.prepare("UPDATE recordings SET body=?,sha=? WHERE id=?").run(JSON.stringify(value), captureHash(value), value.id); this.event(value.id, value);
      }
      const settings = { ...this.settings(), dormant: true }; this.db.prepare("UPDATE meta SET value=? WHERE key='settings'").run(JSON.stringify(settings));
      const clock = this.windowClock(); this.db.prepare("INSERT OR REPLACE INTO meta VALUES('windowClock',?)").run(JSON.stringify({ ...clock, paused: true }));
      this.db.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetId);
    });
  }
}
