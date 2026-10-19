import { createHash } from "node:crypto";
import { lstat, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { PluginRuntimeRpcReply, PluginRuntimeSession, PluginRuntimeSchemaLearning, PluginRuntimeSessionPage, PluginRuntimeHistoryPage, PluginRuntimeStateSourcePage, PluginRuntimeStateConflictPage } from "../shared/pluginRuntime";
import { RUNTIME_SHA, runtimeHash, runtimeId, runtimeJson } from "./pluginRuntimeValidation";
import { configurePluginCustodyJournal } from "./pluginCustodyJournal";
import { sqliteFilename } from "./sqliteFilename";

export const PLUGIN_RUNTIME_WORKSPACE = "plugin-runtime/workspace.sqlite3";
export const PLUGIN_RUNTIME_STORE_FORMAT = "omni-plugin-runtime-workspace-1";
export interface RuntimeStoreHooks { signal?: AbortSignal; checkDisk?(bytes: number): Promise<void> }
export interface RuntimeRpcRecord { state: string; requestHash: string; operation: string; arguments: Record<string, unknown>; reply?: PluginRuntimeRpcReply }
const schema = [
  "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
  "CREATE TABLE sessions(id TEXT PRIMARY KEY,state TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL)",
  "CREATE TABLE rpc(session_id TEXT NOT NULL,id TEXT NOT NULL,state TEXT NOT NULL,request_hash TEXT NOT NULL,request TEXT NOT NULL,reply TEXT,reply_sha256 TEXT,PRIMARY KEY(session_id,id))",
  "CREATE TABLE values_current(plugin_hash TEXT NOT NULL,module_id TEXT NOT NULL,key TEXT NOT NULL,revision INTEGER NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(plugin_hash,module_id,key))",
  "CREATE TABLE events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,kind TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL)",
  "CREATE TABLE schema_jobs(source_sha256 TEXT PRIMARY KEY,state TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL,receipt TEXT,receipt_sha256 TEXT)",
  "CREATE TABLE archives(sha256 TEXT PRIMARY KEY,owner TEXT NOT NULL,bytes INTEGER NOT NULL)",
  "CREATE TABLE state_conflicts(plugin_hash TEXT NOT NULL,module_id TEXT NOT NULL,key TEXT NOT NULL,PRIMARY KEY(plugin_hash,module_id,key))"
];
const normalizedSql = (value: unknown) => String(value).replace(/IF\s+NOT\s+EXISTS\s+/gi, "").replace(/[\s;]/g, "").toLowerCase();
const checked = <T>(payload: unknown, digest: unknown): T => {
  if (typeof payload !== "string" || runtimeHash(payload) !== digest) throw new Error("Runtime workspace row checksum failed.");
  const value: unknown = JSON.parse(payload); if (runtimeJson(value) !== payload) throw new Error("Runtime workspace row is not canonical."); return value as T;
};
function validateSchema(db: DatabaseSync, owner: string): void {
  const allowed = new Set(schema.map(normalizedSql));
  for (const row of db.prepare("SELECT type,name,sql FROM sqlite_master").iterate()) {
    if (row.type === "index" && row.sql === null && String(row.name).startsWith("sqlite_autoindex_") || row.name === "sqlite_sequence") continue;
    if (row.type !== "table" || !allowed.has(normalizedSql(row.sql))) throw new Error("Runtime workspace contains unsupported executable schema.");
  }
  const tables = Number(db.prepare("SELECT COUNT(*) AS n FROM sqlite_master WHERE type='table' AND name!='sqlite_sequence'").get()?.n);
  if (tables !== schema.length || db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== PLUGIN_RUNTIME_STORE_FORMAT ||
      db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== owner || Number(db.prepare("SELECT COUNT(*) AS n FROM meta").get()?.n) !== 2)
    throw new Error("Runtime workspace format/owner/schema mismatch.");
}
function pageBudget(value = 262144): number { if (!Number.isSafeInteger(value) || value < 4096 || value > 16777216) throw new Error("Invalid bounded runtime page budget."); return value; }
const STATES = new Set(["starting", "active", "stopping", "stopped", "failed", "dormant"]);

/** Operational history, never neural answers or reusable activation authority. */
export class PluginRuntimeStore {
  private readGuard?: () => Promise<void>;
  private constructor(readonly db: DatabaseSync, readonly root: string, public owner: string, readonly readOnly: boolean, private hooks: RuntimeStoreHooks) {}
  static async open(root: string, owner: string, options: RuntimeStoreHooks & { create?: boolean; readOnly?: boolean } = {}): Promise<PluginRuntimeStore | undefined> {
    runtimeId(owner); options.signal?.throwIfAborted();
    const rootInfo = await lstat(root); if (!rootInfo.isDirectory() || rootInfo.isSymbolicLink()) throw new Error("Runtime workspace needs an owned brain directory.");
    const directory = join(root, "plugin-runtime"), path = join(root, PLUGIN_RUNTIME_WORKSPACE);
    let directoryInfo = await lstat(directory).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (!directoryInfo && !options.create) return undefined;
    if (!directoryInfo) { await options.checkDisk?.(65536); await mkdir(directory, { mode: 0o700 }); directoryInfo = await lstat(directory); }
    if (!directoryInfo.isDirectory() || directoryInfo.isSymbolicLink()) throw new Error("Unsafe runtime workspace directory.");
    const info = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (!info && !options.create) throw new Error("Runtime workspace is missing its database.");
    if (info && (!info.isFile() || info.isSymbolicLink() || !options.readOnly && info.nlink !== 1)) throw new Error("Runtime writable state must not share a parent inode.");
    const db = new DatabaseSync(sqliteFilename(path), { readOnly: options.readOnly ?? false, allowExtension: false });
    try {
      db.exec("PRAGMA trusted_schema=OFF; PRAGMA busy_timeout=2500; PRAGMA cache_size=-256; PRAGMA temp_store=FILE;");
      const allowed = new Set(schema.map(normalizedSql));
      for (const row of db.prepare("SELECT type,name,sql FROM sqlite_master").iterate()) {
        if (row.type === "index" && row.sql === null && String(row.name).startsWith("sqlite_autoindex_") || row.name === "sqlite_sequence") continue;
        if (row.type !== "table" || !allowed.has(normalizedSql(row.sql))) throw new Error("Runtime workspace contains unsupported executable schema.");
      }
      if (!options.readOnly) configurePluginCustodyJournal(db);
      if (!options.readOnly && options.create) {
        for (const sql of schema) db.exec(sql.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS"));
        db.prepare("INSERT OR IGNORE INTO meta VALUES('format',?)").run(PLUGIN_RUNTIME_STORE_FORMAT);
        db.prepare("INSERT OR IGNORE INTO meta VALUES('brainId',?)").run(owner);
      }
      validateSchema(db, owner);
      return new PluginRuntimeStore(db, root, owner, options.readOnly ?? false, options);
    } catch (error) { db.close(); throw error; }
  }
  static async openFile(path: string, owner: string): Promise<PluginRuntimeStore> {
    runtimeId(owner); const info = await lstat(path); if (!info.isFile() || info.isSymbolicLink()) throw new Error("Unsafe captured runtime database.");
    const db = new DatabaseSync(sqliteFilename(path), { readOnly: true, allowExtension: false });
    try { db.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-256; PRAGMA temp_store=FILE;"); validateSchema(db, owner); return new PluginRuntimeStore(db, path, owner, true, {}); }
    catch (error) { db.close(); throw error; }
  }
  close(): void { this.db.close(); }
  bindReadGuard(check: () => Promise<void>): void { if (!this.readOnly) throw new Error("Source guard requires read-only custody."); this.readGuard = check; }
  async verifyRead(): Promise<void> { await this.readGuard?.(); }
  private async admit(bytes: number): Promise<void> { if (this.readOnly) throw new Error("Runtime source is read-only."); this.hooks.signal?.throwIfAborted(); await this.hooks.checkDisk?.(bytes + 65536); this.hooks.signal?.throwIfAborted(); }
  private event(kind: string, value: unknown): void { const payload = runtimeJson(value); this.db.prepare("INSERT INTO events(kind,payload,sha256) VALUES(?,?,?)").run(kind, payload, runtimeHash(payload)); }
  async session(value: PluginRuntimeSession): Promise<void> {
    const { processId: _diagnosticOnly, ...portable } = value; const payload = runtimeJson(portable); await this.admit(Buffer.byteLength(payload));
    this.db.exec("BEGIN IMMEDIATE"); try { this.db.prepare("INSERT INTO sessions VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state,payload=excluded.payload,sha256=excluded.sha256")
      .run(value.sessionId, value.state, payload, runtimeHash(payload)); this.event("session", portable); this.db.exec("COMMIT"); }
    catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  *sessions(): Generator<PluginRuntimeSession> { for (const row of this.db.prepare("SELECT id,state,payload,sha256 FROM sessions ORDER BY rowid").iterate()) {
    const value = checked<PluginRuntimeSession>(row.payload, row.sha256); if (value.sessionId !== row.id || !STATES.has(String(row.state)) || value.format !== "omni-plugin-runtime-session" || value.version !== 1 || "processId" in value) throw new Error("Runtime session identity failed.");
    yield { ...value, state: String(row.state) as PluginRuntimeSession["state"] }; } }
  sessionsPage(after = Number.MAX_SAFE_INTEGER, maximum?: number): PluginRuntimeSessionPage {
    if (!Number.isSafeInteger(after) || after < 0) throw new Error("Invalid runtime session cursor."); const budget = pageBudget(maximum), sessions: PluginRuntimeSession[] = []; let bytes = 0, last = after;
    for (const descriptor of this.db.prepare("SELECT rowid AS cursor,length(CAST(payload AS BLOB)) AS bytes FROM sessions WHERE rowid<? ORDER BY rowid DESC").iterate(after)) {
      const need = Number(descriptor.bytes); if (need > budget) throw new Error("Runtime session exceeds the selected transfer page budget."); if (bytes + need > budget) break;
      const row = this.db.prepare("SELECT id,state,payload,sha256 FROM sessions WHERE rowid=?").get(Number(descriptor.cursor))!;
      const value = checked<PluginRuntimeSession>(row.payload, row.sha256); if (value.sessionId !== row.id || !STATES.has(String(row.state)) || "processId" in value) throw new Error("Runtime session identity failed.");
      sessions.push({ ...value, state: String(row.state) as PluginRuntimeSession["state"] }); bytes += need; last = Number(descriptor.cursor);
    }
    return { sessions, total: Number(this.db.prepare("SELECT COUNT(*) AS n FROM sessions").get()?.n), ...(this.db.prepare("SELECT 1 FROM sessions WHERE rowid<? LIMIT 1").get(last) ? { nextRowid: last } : {}) };
  }
  rpc(session: string, id: string, maximumReplyBytes = 262144): RuntimeRpcRecord | undefined {
    const row = this.db.prepare("SELECT state,request_hash,request,reply_sha256,length(CAST(reply AS BLOB)) AS reply_bytes FROM rpc WHERE session_id=? AND id=?").get(runtimeId(session), runtimeId(id)); if (!row) return undefined;
    const request = checked<{ operation: string; arguments: Record<string, unknown> }>(row.request, row.request_hash);
    let retained: PluginRuntimeRpcReply | undefined;
    if (row.reply_sha256 !== null) {
      if (Number(row.reply_bytes) > maximumReplyBytes) { this.hashBlob("rpc", "reply", "session_id=? AND id=?", [session, id], String(row.reply_sha256));
        retained = { format: "omni-plugin-runtime-reply", version: 1, id, state: "uncertain", receipt: { id, bytes: Number(row.reply_bytes), sha256: String(row.reply_sha256) }, error: "Read this exact retained response in verified byte pages before interpreting its result; no effect is repeated." }; }
      else retained = checked<PluginRuntimeRpcReply>(this.db.prepare("SELECT reply FROM rpc WHERE session_id=? AND id=?").get(session, id)?.reply, row.reply_sha256);
    }
    return { state: String(row.state), requestHash: String(row.request_hash), ...request,
      ...(retained ? { reply: retained } : {}) };
  }
  async beginRpc(session: string, id: string, operation: string, args: Record<string, unknown>, maximumReplyBytes?: number): Promise<RuntimeRpcRecord | undefined> {
    const body = runtimeJson({ operation, arguments: args }), digest = runtimeHash(body); await this.admit(Buffer.byteLength(body));
    this.db.exec("BEGIN IMMEDIATE"); try { const prior = this.rpc(session, id, maximumReplyBytes);
      if (prior && prior.requestHash !== digest) throw new Error("Runtime request ID was reused with different actual operands.");
      if (!prior) { this.db.prepare("INSERT INTO rpc VALUES(?,?,?,?,?,NULL,NULL)").run(session, id, "prepared", digest, body); this.event("rpc-prepared", { session, id, digest }); }
      this.db.exec("COMMIT"); return prior;
    } catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  async markRpc(session: string, id: string, state: string, reply?: PluginRuntimeRpcReply): Promise<void> {
    const body = reply === undefined ? undefined : runtimeJson(reply); await this.admit(body === undefined ? 0 : Buffer.byteLength(body));
    if (!this.db.prepare("SELECT 1 FROM rpc WHERE session_id=? AND id=?").get(session, id)) throw new Error("Runtime request has no committed preparation.");
    this.db.exec("BEGIN IMMEDIATE"); try { this.db.prepare("UPDATE rpc SET state=?,reply=?,reply_sha256=? WHERE session_id=? AND id=?")
      .run(state, body ?? null, body === undefined ? null : runtimeHash(body), session, id); this.event("rpc-status", { session, id, state, ...(reply ? { reply } : {}) }); this.db.exec("COMMIT"); }
    catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  receipt(session: string, id: string, offset: number, length: number): { text: string; bytes: number; sha256: string; nextOffset?: number } {
    if (!Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(length) || length < 1) throw new Error("Invalid receipt page.");
    const info = this.db.prepare("SELECT length(CAST(reply AS BLOB)) AS n,reply_sha256 FROM rpc WHERE session_id=? AND id=? AND reply IS NOT NULL").get(session, id);
    if (!info) throw new Error("Runtime receipt is unavailable.");
    this.hashBlob("rpc", "reply", "session_id=? AND id=?", [session, id], String(info.reply_sha256));
    const row = this.db.prepare("SELECT substr(CAST(reply AS BLOB),?,?) AS body FROM rpc WHERE session_id=? AND id=?").get(offset + 1, length, session, id);
    const bytes = Number(info.n), block = Buffer.from(row?.body as Uint8Array); return { text: block.toString("base64"), bytes,
      sha256: String(info.reply_sha256), ...(offset + block.length < bytes ? { nextOffset: offset + block.length } : {}) };
  }
  valueInfo(plugin: string, module: string, key: string): { revision: number; bytes: number; sha256?: string } {
    if (this.db.prepare("SELECT 1 FROM state_conflicts WHERE plugin_hash=? AND module_id=? AND key=?").get(plugin, module, key)) throw new Error("Inherited module state conflicts; select a reviewed exact parent source before activation.");
    const row = this.db.prepare("SELECT revision,length(CAST(payload AS BLOB)) AS bytes,sha256 FROM values_current WHERE plugin_hash=? AND module_id=? AND key=?").get(plugin, module, key);
    return row ? { revision: Number(row.revision), bytes: Number(row.bytes), sha256: String(row.sha256) } : { revision: 0, bytes: 0 };
  }
  value(plugin: string, module: string, key: string, maximumBytes = 262144): { revision: number; value: unknown } {
    const info = this.valueInfo(plugin, module, key); if (!Number.isSafeInteger(maximumBytes) || maximumBytes < 1 || info.bytes > maximumBytes) throw new Error("Saved module value needs admitted RAM or verified state byte pages.");
    const row = this.db.prepare("SELECT revision,payload,sha256 FROM values_current WHERE plugin_hash=? AND module_id=? AND key=?").get(plugin, module, key);
    return row ? { revision: Number(row.revision), value: checked(row.payload, row.sha256) } : { revision: 0, value: null };
  }
  valuePage(plugin: string, module: string, key: string, maximumBytes: number): import("../shared/pluginRuntime").PluginRuntimeStateValue {
    const info = this.valueInfo(plugin, module, key); if (!Number.isSafeInteger(maximumBytes) || maximumBytes < 1) throw new Error("Invalid state transfer budget.");
    if (info.bytes > maximumBytes) { this.hashBlob("values_current", "payload", "plugin_hash=? AND module_id=? AND key=?", [plugin, module, key], info.sha256!); return { revision: info.revision, receipt: { bytes: info.bytes, sha256: info.sha256! } }; }
    return this.value(plugin, module, key, maximumBytes);
  }
  valueBytes(plugin: string, module: string, key: string, offset: number, length: number): { text: string; bytes: number; sha256: string; nextOffset?: number } {
    if (!Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(length) || length < 1 || length > 1048576) throw new Error("Invalid saved state byte page."); const info = this.valueInfo(plugin, module, key); if (!info.sha256) throw new Error("Saved module value is missing.");
    this.hashBlob("values_current", "payload", "plugin_hash=? AND module_id=? AND key=?", [plugin, module, key], info.sha256);
    const row = this.db.prepare("SELECT substr(CAST(payload AS BLOB),?,?) AS body FROM values_current WHERE plugin_hash=? AND module_id=? AND key=?").get(offset + 1, length, plugin, module, key), block = Buffer.from(row?.body as Uint8Array);
    return { text: block.toString("base64"), bytes: info.bytes, sha256: info.sha256, ...(offset + block.length < info.bytes ? { nextOffset: offset + block.length } : {}) };
  }
  async put(plugin: string, module: string, key: string, value: unknown, expected: number): Promise<number> {
    if (!Number.isSafeInteger(expected) || expected < 0) throw new Error("Invalid runtime state revision.");
    const payload = runtimeJson(value); await this.admit(Buffer.byteLength(payload)); this.db.exec("BEGIN IMMEDIATE");
    try { if (this.valueInfo(plugin, module, key).revision !== expected) throw new Error("Runtime state changed; review its actual revision.");
      const revision = expected + 1; this.db.prepare("INSERT INTO values_current VALUES(?,?,?,?,?,?) ON CONFLICT(plugin_hash,module_id,key) DO UPDATE SET revision=excluded.revision,payload=excluded.payload,sha256=excluded.sha256")
        .run(plugin, module, key, revision, payload, runtimeHash(payload)); this.event("state", { plugin, module, key, revision, value }); this.db.exec("COMMIT"); return revision;
    } catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  async record(kind: string, value: unknown): Promise<void> { const payload = runtimeJson(value); await this.admit(Buffer.byteLength(payload)); this.event(kind, value); }
  async queueSchema(job: PluginRuntimeSchemaLearning): Promise<void> {
    const payload = runtimeJson(job); await this.admit(Buffer.byteLength(payload));
    const prior = this.db.prepare("SELECT payload FROM schema_jobs WHERE source_sha256=?").get(job.sourceSha256);
    if (prior && prior.payload !== payload) throw new Error("Runtime schema source identity collided.");
    this.db.prepare("INSERT OR IGNORE INTO schema_jobs VALUES(?,'pending',?,?,NULL,NULL)").run(job.sourceSha256, payload, runtimeHash(payload));
  }
  *pendingSchemas(): Generator<PluginRuntimeSchemaLearning> { for (const row of this.db.prepare("SELECT payload,sha256 FROM schema_jobs WHERE state='pending' ORDER BY rowid").iterate()) yield checked(row.payload, row.sha256); }
  async completeSchema(sha: string, receipt: unknown): Promise<void> { const body = runtimeJson(receipt); await this.admit(Buffer.byteLength(body)); this.db.prepare("UPDATE schema_jobs SET state='complete',receipt=?,receipt_sha256=? WHERE source_sha256=?").run(body, runtimeHash(body), sha); }
  private hashBlob(table: string, column: string, where: string, args: Array<string | number>, digest: string): number {
    const bytes = Number(this.db.prepare(`SELECT length(CAST(${column} AS BLOB)) AS n FROM ${table} WHERE ${where}`).get(...args)?.n), sha = createHash("sha256");
    if (!Number.isSafeInteger(bytes) || bytes < 0 || !RUNTIME_SHA.test(digest)) throw new Error("Invalid runtime blob descriptor.");
    for (let offset = 0; offset < bytes; offset += 65536) { this.hooks.signal?.throwIfAborted(); const row = this.db.prepare(`SELECT substr(CAST(${column} AS BLOB),?,65536) AS body FROM ${table} WHERE ${where}`).get(offset + 1, ...args); sha.update(row?.body as Uint8Array); }
    if (sha.digest("hex") !== digest) throw new Error("Runtime workspace blob checksum failed."); return bytes;
  }
  history(after = 0, maximum?: number): PluginRuntimeHistoryPage {
    if (!Number.isSafeInteger(after) || after < 0) throw new Error("Invalid runtime event cursor."); const budget = pageBudget(maximum), events: PluginRuntimeHistoryPage["events"] = []; let used = 0, last = after;
    for (const row of this.db.prepare("SELECT sequence,kind,sha256,length(CAST(payload AS BLOB)) AS bytes FROM events WHERE sequence>? ORDER BY sequence").iterate(after)) {
      const size = Number(row.bytes), sequence = Number(row.sequence), need = Math.min(size, budget) + 192; if (events.length && used + need > budget) break;
      this.hashBlob("events", "payload", "sequence=?", [sequence], String(row.sha256));
      const item = size + 192 > budget ? { receipt: { bytes: size } } : { value: checked(this.db.prepare("SELECT payload FROM events WHERE sequence=?").get(sequence)?.payload, row.sha256) };
      events.push({ sequence, kind: String(row.kind), sha256: String(row.sha256), ...item }); used += size + 192 > budget ? 192 : need; last = sequence;
    }
    return { events, total: Number(this.db.prepare("SELECT COUNT(*) AS n FROM events").get()?.n), ...(this.db.prepare("SELECT 1 FROM events WHERE sequence>? LIMIT 1").get(last) ? { nextSequence: last } : {}) };
  }
  historyBytes(sequence: number, offset: number, length: number): { text: string; bytes: number; sha256: string; nextOffset?: number } {
    if (!Number.isSafeInteger(sequence) || sequence < 1 || !Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(length) || length < 1 || length > 1048576) throw new Error("Invalid runtime history byte page.");
    const row = this.db.prepare("SELECT sha256 FROM events WHERE sequence=?").get(sequence); if (!row) throw new Error("Runtime history record is missing.");
    const sha256 = String(row.sha256), bytes = this.hashBlob("events", "payload", "sequence=?", [sequence], sha256), block = Buffer.from(this.db.prepare("SELECT substr(CAST(payload AS BLOB),?,?) AS body FROM events WHERE sequence=?").get(offset + 1, length, sequence)?.body as Uint8Array);
    return { text: block.toString("base64"), bytes, sha256, ...(offset + block.length < bytes ? { nextOffset: offset + block.length } : {}) };
  }
  sources(after = "", limit = 64): PluginRuntimeStateSourcePage {
    if (after !== "" && !RUNTIME_SHA.test(after) || !Number.isSafeInteger(limit) || limit < 1 || limit > 1024) throw new Error("Invalid source page.");
    const sources = this.db.prepare("SELECT sha256,owner,bytes FROM archives WHERE sha256>? ORDER BY sha256 LIMIT ?").all(after, limit).map(row => ({ sha256: String(row.sha256), owner: runtimeId(row.owner), bytes: Number(row.bytes) }));
    for (const row of sources) if (!RUNTIME_SHA.test(row.sha256) || !Number.isSafeInteger(row.bytes) || row.bytes < 1) throw new Error("Invalid inherited runtime descriptor.");
    const last = sources.at(-1)?.sha256 ?? after; return { sources, total: Number(this.db.prepare("SELECT COUNT(*) AS n FROM archives").get()?.n), ...(this.db.prepare("SELECT 1 FROM archives WHERE sha256>? LIMIT 1").get(last) ? { nextSha256: last } : {}) };
  }
  conflicts(after = "", limit = 64): PluginRuntimeStateConflictPage {
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 1024) throw new Error("Invalid conflict page.");
    const rows = this.db.prepare("SELECT plugin_hash,module_id,key FROM state_conflicts WHERE plugin_hash||'/'||module_id||'/'||key>? ORDER BY plugin_hash,module_id,key LIMIT ?").all(after, limit),
      conflicts = rows.map(row => ({ pluginHash: String(row.plugin_hash), moduleId: String(row.module_id), key: String(row.key) })), last = conflicts.at(-1), cursor = last ? `${last.pluginHash}/${last.moduleId}/${last.key}` : after;
    return { conflicts, total: Number(this.db.prepare("SELECT COUNT(*) AS n FROM state_conflicts").get()?.n), ...(this.db.prepare("SELECT 1 FROM state_conflicts WHERE plugin_hash||'/'||module_id||'/'||key>? LIMIT 1").get(cursor) ? { nextKey: cursor } : {}) };
  }
  conflictCount(plugin: string, module: string): number { return Number(this.db.prepare("SELECT COUNT(*) AS n FROM state_conflicts WHERE plugin_hash=? AND module_id=?").get(plugin, module)?.n); }
  async addArchive(sha256: string, owner: string, bytes: number): Promise<void> {
    if (!RUNTIME_SHA.test(sha256) || !Number.isSafeInteger(bytes) || bytes < 1) throw new Error("Invalid exact inherited runtime source."); runtimeId(owner); await this.admit(512);
    const prior = this.db.prepare("SELECT owner,bytes FROM archives WHERE sha256=?").get(sha256);
    if (prior && (prior.owner !== owner || Number(prior.bytes) !== bytes)) throw new Error("Inherited runtime SHA descriptor collided.");
    this.db.prepare("INSERT OR IGNORE INTO archives VALUES(?,?,?)").run(sha256, owner, bytes);
  }
  async mergeValue(plugin: string, module: string, key: string, revision: number, payload: string, sha256: string, sourceSha256: string): Promise<void> {
    if (!RUNTIME_SHA.test(plugin) || !RUNTIME_SHA.test(sourceSha256) || !Number.isSafeInteger(revision) || revision < 1 || runtimeHash(payload) !== sha256) throw new Error("Invalid inherited state identity."); runtimeId(module); runtimeId(key); await this.admit(Buffer.byteLength(payload));
    this.db.exec("BEGIN IMMEDIATE"); try {
      const conflict = this.db.prepare("SELECT 1 FROM state_conflicts WHERE plugin_hash=? AND module_id=? AND key=?").get(plugin, module, key), prior = this.db.prepare("SELECT revision,sha256 FROM values_current WHERE plugin_hash=? AND module_id=? AND key=?").get(plugin, module, key);
      if (!conflict && !prior) this.db.prepare("INSERT INTO values_current VALUES(?,?,?,?,?,?)").run(plugin, module, key, revision, payload, sha256);
      else if (!conflict && (Number(prior?.revision) !== revision || prior?.sha256 !== sha256)) {
        this.db.prepare("DELETE FROM values_current WHERE plugin_hash=? AND module_id=? AND key=?").run(plugin, module, key);
        this.db.prepare("INSERT OR IGNORE INTO state_conflicts VALUES(?,?,?)").run(plugin, module, key);
        this.event("inherited-state-conflict", { plugin, module, key, sourceSha256 });
      }
      this.db.exec("COMMIT");
    } catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  async inheritConflict(plugin: string, module: string, key: string): Promise<void> {
    if (!RUNTIME_SHA.test(plugin)) throw new Error("Invalid inherited conflict identity."); runtimeId(module); runtimeId(key); await this.admit(512);
    this.db.exec("BEGIN IMMEDIATE"); try { this.db.prepare("DELETE FROM values_current WHERE plugin_hash=? AND module_id=? AND key=?").run(plugin, module, key);
      this.db.prepare("INSERT OR IGNORE INTO state_conflicts VALUES(?,?,?)").run(plugin, module, key); this.db.exec("COMMIT"); }
    catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  async selectValue(plugin: string, module: string, key: string, value: unknown, expected: number, sourceSha256: string): Promise<void> {
    if (!RUNTIME_SHA.test(plugin) || !RUNTIME_SHA.test(sourceSha256) || !Number.isSafeInteger(expected) || expected < 0) throw new Error("Invalid reviewed state selection."); runtimeId(module); runtimeId(key); const payload = runtimeJson(value); await this.admit(Buffer.byteLength(payload));
    this.db.exec("BEGIN IMMEDIATE"); try {
      const current = Number(this.db.prepare("SELECT revision FROM values_current WHERE plugin_hash=? AND module_id=? AND key=?").get(plugin, module, key)?.revision ?? 0);
      if (current !== expected) throw new Error("Runtime state selection revision changed.");
      this.db.prepare("DELETE FROM state_conflicts WHERE plugin_hash=? AND module_id=? AND key=?").run(plugin, module, key);
      this.db.prepare("INSERT INTO values_current VALUES(?,?,?,?,?,?) ON CONFLICT(plugin_hash,module_id,key) DO UPDATE SET revision=excluded.revision,payload=excluded.payload,sha256=excluded.sha256").run(plugin, module, key, expected + 1, payload, runtimeHash(payload));
      this.event("reviewed-state-selection", { plugin, module, key, sourceSha256, expectedRevision: expected, value }); this.db.exec("COMMIT");
    } catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  validate(): void {
    if (this.db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Runtime workspace SQLite integrity failed.");
    for (const row of this.db.prepare("SELECT id,state,sha256 FROM sessions").iterate()) { runtimeId(row.id); if (!STATES.has(String(row.state))) throw new Error("Invalid runtime session state."); this.hashBlob("sessions", "payload", "id=?", [String(row.id)], String(row.sha256)); }
    for (const row of this.db.prepare("SELECT session_id,id,request_hash,reply_sha256 FROM rpc").iterate()) { runtimeId(row.session_id); runtimeId(row.id); this.hashBlob("rpc", "request", "session_id=? AND id=?", [String(row.session_id), String(row.id)], String(row.request_hash)); if (row.reply_sha256 !== null) this.hashBlob("rpc", "reply", "session_id=? AND id=?", [String(row.session_id), String(row.id)], String(row.reply_sha256)); }
    for (const row of this.db.prepare("SELECT plugin_hash,module_id,key,revision,sha256 FROM values_current").iterate()) { if (!RUNTIME_SHA.test(String(row.plugin_hash)) || !Number.isSafeInteger(Number(row.revision)) || Number(row.revision) < 1) throw new Error("Invalid runtime state identity."); runtimeId(row.module_id); runtimeId(row.key); this.hashBlob("values_current", "payload", "plugin_hash=? AND module_id=? AND key=?", [String(row.plugin_hash), String(row.module_id), String(row.key)], String(row.sha256)); }
    for (const row of this.db.prepare("SELECT source_sha256,state,sha256 FROM schema_jobs").iterate()) { if (!RUNTIME_SHA.test(String(row.source_sha256)) || !["pending", "complete", "dormant"].includes(String(row.state))) throw new Error("Invalid schema custody state."); this.hashBlob("schema_jobs", "payload", "source_sha256=?", [String(row.source_sha256)], String(row.sha256)); }
    for (const row of this.db.prepare("SELECT sequence,sha256 FROM events").iterate()) this.hashBlob("events", "payload", "sequence=?", [Number(row.sequence)], String(row.sha256));
    for (const row of this.db.prepare("SELECT source_sha256,receipt_sha256 FROM schema_jobs WHERE receipt IS NOT NULL").iterate()) this.hashBlob("schema_jobs", "receipt", "source_sha256=?", [String(row.source_sha256)], String(row.receipt_sha256));
    for (const row of this.db.prepare("SELECT sha256,owner,bytes FROM archives").iterate()) if (!RUNTIME_SHA.test(String(row.sha256)) || !Number.isSafeInteger(Number(row.bytes)) || Number(row.bytes) < 1) throw new Error("Invalid runtime source descriptor."); else runtimeId(row.owner);
    for (const row of this.db.prepare("SELECT plugin_hash,module_id,key FROM state_conflicts").iterate()) { if (!RUNTIME_SHA.test(String(row.plugin_hash))) throw new Error("Invalid runtime conflict identity."); runtimeId(row.module_id); runtimeId(row.key); }
  }
  async rekey(owner: string): Promise<void> { runtimeId(owner); await this.admit(65536); this.db.exec("BEGIN IMMEDIATE");
    try { this.event("copied-dormant", { sourceOwner: this.owner, targetOwner: owner }); this.db.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(owner);
      this.db.prepare("UPDATE sessions SET state='dormant'").run(); this.db.prepare("UPDATE schema_jobs SET state='dormant' WHERE state='pending'").run(); this.db.exec("COMMIT"); this.owner = owner;
    } catch (error) { this.db.exec("ROLLBACK"); throw error; } }
}
