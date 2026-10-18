import { createHash } from "node:crypto";
import { lstat, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { PluginInteractivePanelLaunch, PluginInteractiveToolReceipt } from "../shared/pluginInteractivePanels";
import type { ToolInvocation } from "../shared/types";
import { runtimeHash, runtimeId, runtimeJson } from "./pluginRuntimeValidation";
import { configurePluginCustodyJournal } from "./pluginCustodyJournal";

export const PANEL_RECEIPT_PATH = "plugin-panels/workspace.sqlite3";
export const PANEL_RECEIPT_FORMAT = "omni-plugin-panel-receipts-1";
export interface PanelReceiptHooks { signal?: AbortSignal; checkDisk(bytes: number): Promise<void> }
export interface RetainedPanelRequest { hash: string; invocation: ToolInvocation; hostRequestId: string; createdAt: string; state: string; receipt?: PluginInteractiveToolReceipt; reviewId?: string; decision?: "approve" | "deny" }
const schema = [
  "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
  "CREATE TABLE views(id TEXT PRIMARY KEY,state TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL)",
  "CREATE TABLE requests(view_id TEXT NOT NULL,id TEXT NOT NULL,request_hash TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL,state TEXT NOT NULL,receipt TEXT,receipt_sha256 TEXT,review_id TEXT,decision TEXT,PRIMARY KEY(view_id,id))",
  "CREATE TABLE events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,view_id TEXT NOT NULL,request_id TEXT,kind TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL)",
  "CREATE TABLE archives(sha256 TEXT PRIMARY KEY,owner TEXT NOT NULL,bytes INTEGER NOT NULL)"
];
const normalized = (value: unknown) => String(value).replace(/[\s;]/g, "").toLowerCase();
function checked<T>(payload: unknown, sha: unknown): T { if (typeof payload !== "string" || runtimeHash(payload) !== sha) throw new Error("Panel receipt checksum failed."); const parsed: unknown = JSON.parse(payload); if (runtimeJson(parsed) !== payload) throw new Error("Panel receipt is not canonical."); return parsed as T; }
function regularSchema(db: DatabaseSync, owner: string): void { const allowed = new Set(schema.map(normalized)); let tables = 0;
  for (const row of db.prepare("SELECT type,name,sql FROM sqlite_master").iterate()) { if (row.name === "sqlite_sequence" || row.type === "index" && row.sql === null && String(row.name).startsWith("sqlite_autoindex_")) continue;
    if (row.type !== "table" || !allowed.has(normalized(row.sql))) throw new Error("Unsupported panel receipt schema."); tables++; }
  if (tables !== schema.length || db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== PANEL_RECEIPT_FORMAT || db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== owner || Number(db.prepare("SELECT COUNT(*) AS n FROM meta").get()?.n) !== 2) throw new Error("Panel receipt owner/format mismatch."); }

/** Complete operational replay custody; no channel nonce or approval grant. */
export class PluginPanelReceiptStore {
  private constructor(readonly db: DatabaseSync, public owner: string, readonly readOnly: boolean, private hooks?: PanelReceiptHooks) {}
  static async open(root: string, owner: string, hooks?: PanelReceiptHooks, create = false): Promise<PluginPanelReceiptStore | undefined> {
    runtimeId(owner); const directory = join(root, "plugin-panels"), path = join(root, PANEL_RECEIPT_PATH), rootInfo = await lstat(root); if (!rootInfo.isDirectory() || rootInfo.isSymbolicLink()) throw new Error("Panel custody needs a real brain directory.");
    let info = await lstat(directory).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; }); if (!info && !create) return undefined;
    if (!info) { if (!hooks) throw new Error("Panel custody needs real disk admission."); await hooks.checkDisk(65536); hooks.signal?.throwIfAborted(); await mkdir(directory, { mode: 0o700 }); info = await lstat(directory); }
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Unsafe panel receipt directory.");
    const file = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; }); if (!file && !create) throw new Error("Panel receipt database is missing.");
    if (file && (!file.isFile() || file.isSymbolicLink() || hooks && file.nlink !== 1)) throw new Error("Panel receipt file must be independently mutable.");
    const db = new DatabaseSync(path, { readOnly: !hooks, allowExtension: false });
    try { db.exec("PRAGMA trusted_schema=OFF; PRAGMA busy_timeout=2500; PRAGMA cache_size=-256; PRAGMA temp_store=FILE;");
      if (hooks) configurePluginCustodyJournal(db);
      if (!file && create) { for (const sql of schema) db.exec(sql); db.prepare("INSERT INTO meta VALUES('format',?)").run(PANEL_RECEIPT_FORMAT); db.prepare("INSERT INTO meta VALUES('brainId',?)").run(owner); }
      regularSchema(db, owner); return new PluginPanelReceiptStore(db, owner, !hooks, hooks);
    } catch (error) { db.close(); throw error; }
  }
  static async captured(path: string, owner?: string): Promise<PluginPanelReceiptStore> { const info = await lstat(path); if (!info.isFile() || info.isSymbolicLink()) throw new Error("Unsafe captured panel receipts."); const db = new DatabaseSync(path, { readOnly: true, allowExtension: false });
    try { db.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-256; PRAGMA temp_store=FILE;"); const actual = runtimeId(db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value); regularSchema(db, owner ?? actual); return new PluginPanelReceiptStore(db, actual, true); } catch (error) { db.close(); throw error; } }
  close(): void { this.db.close(); }
  private async admit(bytes: number, signal?: AbortSignal): Promise<void> { if (!this.hooks || this.readOnly) throw new Error("Panel receipt source is read-only."); this.hooks.signal?.throwIfAborted(); signal?.throwIfAborted();
    const check = this.hooks.checkDisk(bytes * 4 + 65536); if (signal) await new Promise<void>((resolve, reject) => { const abort = () => { reject(signal.reason ?? new Error("Panel receipt preparation cancelled.")); }; signal.addEventListener("abort", abort, { once: true });
      check.then(() => { signal.removeEventListener("abort", abort); resolve(); }, error => { signal.removeEventListener("abort", abort); reject(error); }); if (signal.aborted) abort(); }); else await check;
    this.hooks.signal?.throwIfAborted(); signal?.throwIfAborted(); }
  private event(view: string, id: string | null, kind: string, value: unknown): void { const payload = runtimeJson(value); this.db.prepare("INSERT INTO events(view_id,request_id,kind,payload,sha256) VALUES(?,?,?,?,?)").run(view, id, kind, payload, runtimeHash(payload)); }
  async view(launch: PluginInteractivePanelLaunch): Promise<void> { const { channelNonce: _neverPortable, ...publicView } = launch, payload = runtimeJson(publicView); await this.admit(Buffer.byteLength(payload));
    this.db.exec("BEGIN IMMEDIATE"); try { this.db.prepare("INSERT INTO views VALUES(?,'open',?,?)").run(launch.viewId, payload, runtimeHash(payload)); this.event(launch.viewId, null, "opened", publicView); this.db.exec("COMMIT"); } catch (error) { this.db.exec("ROLLBACK"); throw error; } }
  get(view: string, id: string): RetainedPanelRequest | undefined { const row = this.db.prepare("SELECT * FROM requests WHERE view_id=? AND id=?").get(view, id); if (!row) return undefined;
    const value = checked<{ hash: string; invocation: ToolInvocation; hostRequestId: string; createdAt: string }>(row.payload, row.sha256); if (value.hash !== row.request_hash) throw new Error("Panel actual operand hash changed.");
    return { ...value, state: String(row.state), ...(row.receipt === null ? {} : { receipt: checked<PluginInteractiveToolReceipt>(row.receipt, row.receipt_sha256) }), ...(row.review_id === null ? {} : { reviewId: String(row.review_id) }), ...(row.decision === null ? {} : { decision: String(row.decision) as "approve" | "deny" }) }; }
  async prepare(view: string, id: string, value: Pick<RetainedPanelRequest, "hash" | "invocation" | "hostRequestId">, signal?: AbortSignal): Promise<RetainedPanelRequest | undefined> { const intent = { ...value, createdAt: new Date().toISOString() }, payload = runtimeJson(intent); await this.admit(Buffer.byteLength(payload), signal);
    this.db.exec("BEGIN IMMEDIATE"); try { const prior = this.get(view, id); if (prior && prior.hash !== value.hash) throw new Error("Panel request ID changed actual operands.");
      if (!prior) { this.db.prepare("INSERT INTO requests VALUES(?,?,?,?,?,'prepared',NULL,NULL,NULL,NULL)").run(view, id, value.hash, payload, runtimeHash(payload)); this.event(view, id, "prepared", intent); } this.db.exec("COMMIT"); return prior;
    } catch (error) { this.db.exec("ROLLBACK"); throw error; } }
  async status(view: string, id: string, state: string, receipt?: PluginInteractiveToolReceipt, reviewId?: string, decision?: "approve" | "deny", signal?: AbortSignal): Promise<void> {
    const body = receipt === undefined ? undefined : runtimeJson(JSON.parse(JSON.stringify(receipt))); await this.admit(body === undefined ? 256 : Buffer.byteLength(body), signal);
    this.db.exec("BEGIN IMMEDIATE"); try { if (!this.db.prepare("SELECT 1 FROM requests WHERE view_id=? AND id=?").get(view, id)) throw new Error("Panel request has no durable preparation.");
      this.db.prepare("UPDATE requests SET state=?,receipt=?,receipt_sha256=?,review_id=?,decision=? WHERE view_id=? AND id=?").run(state, body ?? null, body === undefined ? null : runtimeHash(body), reviewId ?? null, decision ?? null, view, id);
      this.event(view, id, state, { state, ...(receipt ? { receipt: JSON.parse(body!) } : {}), ...(reviewId ? { reviewId } : {}), ...(decision ? { decision } : {}) }); this.db.exec("COMMIT");
    } catch (error) { this.db.exec("ROLLBACK"); throw error; } }
  async retire(view: string, reason: string): Promise<void> { await this.admit(Buffer.byteLength(reason)); this.db.exec("BEGIN IMMEDIATE"); try { this.db.prepare("UPDATE views SET state='closed' WHERE id=?").run(view); this.event(view, null, "closed", { reason }); this.db.exec("COMMIT"); } catch (error) { this.db.exec("ROLLBACK"); throw error; } }
  async record(view: string, id: string, kind: string, value: unknown): Promise<void> { const payload = runtimeJson(JSON.parse(JSON.stringify(value))); await this.admit(Buffer.byteLength(payload)); this.event(view, id, kind, JSON.parse(payload)); }
  private hashBlob(table: "views" | "requests" | "events", column: "payload" | "receipt", where: string, args: Array<string | number>, sha: string): number { const bytes = Number(this.db.prepare(`SELECT length(CAST(${column} AS BLOB)) AS n FROM ${table} WHERE ${where}`).get(...args)?.n), hash = createHash("sha256");
    if (!Number.isSafeInteger(bytes) || bytes < 0 || !/^[a-f0-9]{64}$/.test(sha)) throw new Error("Invalid panel replay blob."); for (let offset = 0; offset < bytes; offset += 65536) { this.hooks?.signal?.throwIfAborted(); hash.update(this.db.prepare(`SELECT substr(CAST(${column} AS BLOB),?,65536) AS body FROM ${table} WHERE ${where}`).get(offset + 1, ...args)?.body as Uint8Array); } if (hash.digest("hex") !== sha) throw new Error("Panel replay blob checksum failed."); return bytes; }
  history(after = 0, maximum = 262144): { records: Array<{ sequence: number; viewId: string; requestId?: string; kind: string; sha256: string; value?: unknown; bytes?: number }>; total: number; nextSequence?: number } {
    if (!Number.isSafeInteger(after) || after < 0 || !Number.isSafeInteger(maximum) || maximum < 4096 || maximum > 16777216) throw new Error("Invalid bounded panel history page."); const records: ReturnType<PluginPanelReceiptStore["history"]>["records"] = []; let used = 0, last = after;
    for (const row of this.db.prepare("SELECT sequence,view_id,request_id,kind,sha256,length(CAST(payload AS BLOB)) AS n FROM events WHERE sequence>? ORDER BY sequence").iterate(after)) { const n = Number(row.n), size = n + 256 > maximum ? 256 : n + 256; if (records.length && used + size > maximum) break;
      const sequence = Number(row.sequence); this.hashBlob("events", "payload", "sequence=?", [sequence], String(row.sha256)); records.push({ sequence, viewId: String(row.view_id), ...(row.request_id === null ? {} : { requestId: String(row.request_id) }), kind: String(row.kind), sha256: String(row.sha256), ...(n + 256 > maximum ? { bytes: n } : { value: checked(this.db.prepare("SELECT payload FROM events WHERE sequence=?").get(sequence)?.payload, row.sha256) }) }); used += size; last = sequence; }
    return { records, total: Number(this.db.prepare("SELECT COUNT(*) AS n FROM events").get()?.n), ...(this.db.prepare("SELECT 1 FROM events WHERE sequence>? LIMIT 1").get(last) ? { nextSequence: last } : {}) }; }
  bytes(sequence: number, offset: number, length: number): { text: string; bytes: number; sha256: string; nextOffset?: number } { if (!Number.isSafeInteger(sequence) || sequence < 1 || !Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(length) || length < 1 || length > 1048576) throw new Error("Invalid panel history byte page.");
    const row = this.db.prepare("SELECT sha256 FROM events WHERE sequence=?").get(sequence); if (!row) throw new Error("Unknown panel history record."); const sha256 = String(row.sha256), bytes = this.hashBlob("events", "payload", "sequence=?", [sequence], sha256), block = Buffer.from(this.db.prepare("SELECT substr(CAST(payload AS BLOB),?,?) AS body FROM events WHERE sequence=?").get(offset + 1, length, sequence)?.body as Uint8Array);
    return { text: block.toString("base64"), bytes, sha256, ...(offset + block.length < bytes ? { nextOffset: offset + block.length } : {}) }; }
  validate(): void { if (this.db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Panel receipt SQLite integrity failed."); for (const row of this.db.prepare("SELECT id,sha256 FROM views").iterate()) this.hashBlob("views", "payload", "id=?", [String(row.id)], String(row.sha256));
    for (const row of this.db.prepare("SELECT view_id,id,sha256,receipt_sha256 FROM requests").iterate()) { this.hashBlob("requests", "payload", "view_id=? AND id=?", [String(row.view_id), String(row.id)], String(row.sha256)); if (row.receipt_sha256 !== null) this.hashBlob("requests", "receipt", "view_id=? AND id=?", [String(row.view_id), String(row.id)], String(row.receipt_sha256)); }
    for (const row of this.db.prepare("SELECT sequence,sha256 FROM events").iterate()) this.hashBlob("events", "payload", "sequence=?", [Number(row.sequence)], String(row.sha256)); }
  async rekey(owner: string): Promise<void> { runtimeId(owner); await this.admit(256); this.db.exec("BEGIN IMMEDIATE"); try { this.event("custody", null, "copied-dormant", { sourceOwner: this.owner, targetOwner: owner }); this.db.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(owner); this.db.prepare("UPDATE views SET state='dormant'").run(); this.db.exec("COMMIT"); this.owner = owner; } catch (error) { this.db.exec("ROLLBACK"); throw error; } }
}
