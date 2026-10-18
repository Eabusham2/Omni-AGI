import { randomUUID } from "node:crypto";
import { lstat, mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import type { AutomationDefinition, AutomationPage, AutomationPageQuery, AutomationProgram, AutomationRun, AutomationStepReceipt } from "../shared/automation";
import { automationCanonical, automationHash, automationId, automationObject, automationSha, validateAutomationProgram } from "./automationFormat";
import { writeMutableFileIsolated } from "./mutableFileIsolation";

export const AUTOMATION_RELATIVE_PATH = "automation/workspace.sqlite3";
export const AUTOMATION_FORMAT = "omni-host-automation-workspace-1";
type Kind = "definition" | "run";
const immutableSteps = new Set(["complete", "failed", "skipped", "reviewed-skip"]);
const jsonValue = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
const schema = new Map(Object.entries({
  meta: "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
  records: "CREATE TABLE records(kind TEXT NOT NULL,id TEXT NOT NULL,state TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(kind,id))",
  programs: "CREATE TABLE programs(sha256 TEXT PRIMARY KEY,bytes INTEGER NOT NULL)",
  requests: "CREATE TABLE requests(id TEXT PRIMARY KEY,arguments_sha256 TEXT NOT NULL,result TEXT NOT NULL)",
  steps: "CREATE TABLE steps(run_id TEXT NOT NULL,step_index INTEGER NOT NULL,attempt INTEGER NOT NULL,state TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(run_id,step_index,attempt))",
  events: "CREATE TABLE events(run_id TEXT NOT NULL,sequence INTEGER NOT NULL,kind TEXT NOT NULL,payload TEXT NOT NULL,previous_sha256 TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(run_id,sequence))",
  inherited_requests: "CREATE TABLE inherited_requests(parent_id TEXT,id TEXT,arguments_sha256 TEXT,result TEXT,PRIMARY KEY(parent_id,id))",
  automation_merge_aliases: "CREATE TABLE automation_merge_aliases(parent_id TEXT,kind TEXT,source_id TEXT,target_id TEXT,PRIMARY KEY(parent_id,kind,source_id))",
  inherited_event_chains: "CREATE TABLE inherited_event_chains(parent_id TEXT,run_id TEXT,sequence INTEGER,kind TEXT,payload TEXT,previous_sha256 TEXT,sha256 TEXT,PRIMARY KEY(parent_id,run_id,sequence))",
  record_state: "CREATE INDEX record_state ON records(kind,state,id)"
}));
const schemaSql = (value: unknown): string => String(value).replace(/IF\s+NOT\s+EXISTS\s+/gi, "").replace(/\s+|;$/g, "").toUpperCase();
function requireTextHeadroom(bytes: number): void {
  const need = bytes * 12 + 65536;
  if (!Number.isSafeInteger(bytes) || bytes < 0 || !Number.isSafeInteger(need) || need > freemem() / 2 || need > (getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 2) throw new Error("Action metadata awaits physical parse headroom; nothing was skipped or dispatched.");
}

export function validateAutomationDefinition(value: unknown, brainId: string): AutomationDefinition {
  const row = automationObject(value); automationId(row.id); automationSha(row.revisionSha256);
  if (row.brainId !== brainId || !Number.isSafeInteger(row.revision) || Number(row.revision) < 1 || typeof row.name !== "string") throw new Error("Invalid action-file owner/revision.");
  for (const key of ["createdAt", "updatedAt"]) if (typeof row[key] !== "string" || !Number.isFinite(Date.parse(String(row[key])))) throw new Error("Invalid action-file timestamp.");
  if (row.activation !== undefined) {
    const activation = automationObject(row.activation);
    if (typeof activation.enabled !== "boolean" || typeof activation.allowTriggers !== "boolean" ||
        activation.ownerBrainId !== brainId || typeof activation.ownerEpoch !== "string" ||
        !Number.isSafeInteger(activation.catalogRevision) || Number(activation.catalogRevision) < 1 ||
        typeof activation.policyRevision !== "string" || activation.revisionSha256 !== row.revisionSha256) throw new Error("Invalid reviewed action-file activation.");
  }
  return row as unknown as AutomationDefinition;
}
export function validateAutomationRun(value: unknown, brainId: string): AutomationRun {
  const row = automationObject(value); for (const key of ["id", "definitionId"]) automationId(row[key]); automationSha(row.revisionSha256); automationSha(row.inputSha256);
  if (row.brainId !== brainId || !["queued", "running", "waiting", "approval-required", "paused", "review-required", "stopping", "stopped", "complete", "failed"].includes(String(row.state)) ||
      !Number.isSafeInteger(row.nextStep) || Number(row.nextStep) < 0 || automationHash(automationCanonical(row.inputs)) !== row.inputSha256) throw new Error("Invalid action run/input ownership.");
  automationObject(row.inputs); automationObject(row.activation);
  for (const key of ["createdAt", "updatedAt"]) if (typeof row[key] !== "string" || !Number.isFinite(Date.parse(String(row[key])))) throw new Error("Invalid action-run timestamp.");
  return row as unknown as AutomationRun;
}
export function validateAutomationStep(value: unknown, runId: string): AutomationStepReceipt {
  const row = automationObject(value); automationId(row.stepId); automationId(row.requestId);
  if (row.runId !== runId || !Number.isSafeInteger(row.index) || Number(row.index) < 0 || !Number.isSafeInteger(row.attempt) || Number(row.attempt) < 1 ||
      !["intent", "waiting", "approval-required", "complete", "failed", "skipped", "unknown", "reviewed-skip"].includes(String(row.state))) throw new Error("Invalid action-step receipt.");
  if (row.invocation !== undefined && automationHash(automationCanonical(row.invocation)) !== row.invocationSha256) throw new Error("Action-step operands differ from their pre-effect intent.");
  if (row.outcome !== undefined) {
    const outcome = automationObject(row.outcome);
    if (outcome.state === "complete" && outcome.dispatchStarted !== true || row.state === "complete" && outcome.state !== "complete" ||
        row.state === "approval-required" && outcome.state !== "approval-required") throw new Error("Action-step outcome is not an actual dispatch disposition.");
  }
  return row as unknown as AutomationStepReceipt;
}

/** Owned immutable action files plus FULL-sync transactional operational receipts. */
export class AutomationStore {
  private constructor(readonly root: string, readonly brainId: string, readonly db: DatabaseSync) {}
  static async open(root: string, brainId: string, readOnly = false): Promise<AutomationStore> {
    automationId(brainId);
    const directory = join(root, "automation"), path = join(root, AUTOMATION_RELATIVE_PATH);
    if (!readOnly) await mkdir(join(directory, "programs"), { recursive: true, mode: 0o700 });
    for (const item of [root, directory, join(directory, "programs")]) {
      const info = await lstat(item); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Action workspace must be an owned directory.");
    }
    const file = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (file && (!file.isFile() || file.isSymbolicLink())) throw new Error("Action workspace database must be a regular private file.");
    const db = new DatabaseSync(path, { readOnly, allowExtension: false });
    try {
      db.exec("PRAGMA trusted_schema=OFF;");
      for (const row of db.prepare("SELECT type,name,sql FROM sqlite_master").iterate()) {
        if (row.type === "index" && /^sqlite_autoindex_[A-Za-z0-9_]+_\d+$/.test(String(row.name)) && row.sql === null) continue;
        const declaration = schema.get(String(row.name));
        if (!declaration || row.type !== (row.name === "record_state" ? "index" : "table") || schemaSql(row.sql) !== schemaSql(declaration)) throw new Error("Action database contains unsupported executable/schema objects.");
      }
      if (!readOnly) {
        db.exec(`PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA busy_timeout=2500; PRAGMA cache_size=-256; PRAGMA temp_store=FILE;
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS records(kind TEXT NOT NULL,id TEXT NOT NULL,state TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(kind,id));
          CREATE INDEX IF NOT EXISTS record_state ON records(kind,state,id);
          CREATE TABLE IF NOT EXISTS programs(sha256 TEXT PRIMARY KEY,bytes INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,arguments_sha256 TEXT NOT NULL,result TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS steps(run_id TEXT NOT NULL,step_index INTEGER NOT NULL,attempt INTEGER NOT NULL,state TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(run_id,step_index,attempt));
          CREATE TABLE IF NOT EXISTS events(run_id TEXT NOT NULL,sequence INTEGER NOT NULL,kind TEXT NOT NULL,payload TEXT NOT NULL,previous_sha256 TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(run_id,sequence));`);
        db.prepare("INSERT OR IGNORE INTO meta VALUES('format',?)").run(AUTOMATION_FORMAT);
        db.prepare("INSERT OR IGNORE INTO meta VALUES('brainId',?)").run(brainId);
        db.prepare("INSERT OR IGNORE INTO meta VALUES('catalogRevision','0')").run();
      }
      if (db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== AUTOMATION_FORMAT || db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) throw new Error("Action workspace has another format or owner.");
      return new AutomationStore(root, brainId, db);
    } catch (error) { db.close(); throw error; }
  }
  close(): void { this.db.close(); }
  transaction<T>(operation: () => T): T {
    this.db.exec("BEGIN IMMEDIATE"); try { const result = operation(); this.db.exec("COMMIT"); return result; }
    catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  private decoded<T>(payload: unknown, sha: unknown): T {
    if (typeof payload !== "string" || automationHash(payload) !== sha) throw new Error("Action workspace receipt checksum failed.");
    return JSON.parse(payload) as T;
  }
  get<T extends AutomationDefinition | AutomationRun>(kind: Kind, id: string): T | undefined {
    const size = this.db.prepare("SELECT length(CAST(payload AS BLOB)) AS bytes FROM records WHERE kind=? AND id=?").get(kind, automationId(id)); if (size) requireTextHeadroom(Number(size.bytes));
    const row = this.db.prepare("SELECT payload,sha256 FROM records WHERE kind=? AND id=?").get(kind, automationId(id));
    if (!row) return undefined;
    const value = this.decoded<T>(row.payload, row.sha256);
    return (kind === "definition" ? validateAutomationDefinition(value, this.brainId) : validateAutomationRun(value, this.brainId)) as T;
  }
  put(value: AutomationDefinition | AutomationRun, kind: Kind): void {
    const row = jsonValue(value); if (kind === "definition") validateAutomationDefinition(row, this.brainId); else validateAutomationRun(row, this.brainId);
    const payload = automationCanonical(row), state = kind === "definition" ? ((row as AutomationDefinition).activation?.enabled ? "active" : "dormant") : (row as AutomationRun).state;
    this.db.prepare("INSERT INTO records VALUES(?,?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET state=excluded.state,payload=excluded.payload,sha256=excluded.sha256")
      .run(kind, row.id, state, payload, automationHash(payload));
  }
  catalogRevision(): number { return Number(this.db.prepare("SELECT value FROM meta WHERE key='catalogRevision'").get()?.value); }
  nextCatalogRevision(expected: number): number {
    if (!Number.isSafeInteger(expected) || this.catalogRevision() !== expected) throw new Error("The action catalog changed; review its current revision.");
    this.db.prepare("UPDATE meta SET value=? WHERE key='catalogRevision'").run(String(expected + 1)); return expected + 1;
  }
  async saveProgram(value: unknown, checkWrite?: (bytes: number) => Promise<void>): Promise<{ program: AutomationProgram; sha256: string }> {
    const program = validateAutomationProgram(value), body = automationCanonical(program), sha256 = automationHash(body), path = join(this.root, "automation/programs", sha256 + ".omni-action.json");
    const file = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (file) {
      if (!file.isFile() || file.isSymbolicLink() || automationHash(await readFile(path, "utf8")) !== sha256) throw new Error("Immutable action-file bytes changed.");
    } else {
      await checkWrite?.(Buffer.byteLength(body) + 65536);
      await writeMutableFileIsolated(path, Buffer.from(body));
    }
    this.db.prepare("INSERT OR IGNORE INTO programs VALUES(?,?)").run(sha256, Buffer.byteLength(body));
    return { program, sha256 };
  }
  async program(sha: string, checkRead?: (bytes: number) => Promise<void>): Promise<AutomationProgram> {
    automationSha(sha); const row = this.db.prepare("SELECT bytes FROM programs WHERE sha256=?").get(sha);
    if (!row || !Number.isSafeInteger(row.bytes)) throw new Error("Saved action-file revision is unavailable.");
    const path = join(this.root, "automation/programs", sha + ".omni-action.json"), file = await lstat(path);
    if (!file.isFile() || file.isSymbolicLink() || file.size !== row.bytes) throw new Error("Action-file inventory changed.");
    const parseBytes = file.size * 12 + 65536;
    if (!Number.isSafeInteger(parseBytes) || parseBytes > freemem() / 2 || parseBytes > (getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 2) throw new Error("Action-file parsing awaits physical transfer headroom.");
    await checkRead?.(file.size * 12 + 65536); const body = await readFile(path, "utf8");
    if (automationHash(body) !== sha) throw new Error("Immutable action-file checksum failed.");
    return validateAutomationProgram(JSON.parse(body));
  }
  page<T extends AutomationDefinition | AutomationRun>(kind: Kind, query: AutomationPageQuery = {}): AutomationPage<T> {
    const after = query.after === undefined ? "" : automationId(query.after), limit = query.limit ?? 50;
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 200) throw new Error("Use action pages from 1 to 200, then follow the cursor.");
    const size = this.db.prepare("SELECT COALESCE(SUM(bytes),0) AS bytes FROM (SELECT length(CAST(payload AS BLOB)) AS bytes FROM records WHERE kind=? AND id>? ORDER BY id LIMIT ?)").get(kind, after, limit + 1); requireTextHeadroom(Number(size?.bytes ?? 0));
    const rows = this.db.prepare("SELECT id,payload,sha256 FROM records WHERE kind=? AND id>? ORDER BY id LIMIT ?").all(kind, after, limit + 1);
    return { entries: rows.slice(0, limit).map(row => this.decoded<T>(row.payload, row.sha256)), total: Number(this.db.prepare("SELECT COUNT(*) AS n FROM records WHERE kind=?").get(kind)?.n),
      ...(rows.length > limit ? { nextAfter: String(rows[limit - 1]!.id) } : {}) };
  }
  *records<T extends AutomationDefinition | AutomationRun>(kind: Kind, states?: readonly string[]): Generator<T> {
    let after = "";
    while (true) {
      const size = this.db.prepare(`SELECT COALESCE(SUM(bytes),0) AS bytes FROM (SELECT length(CAST(payload AS BLOB)) AS bytes FROM records WHERE kind=? AND id>?${states?.length ? ` AND state IN (${states.map(() => "?").join(",")})` : ""} ORDER BY id LIMIT 50)`).get(kind, after, ...(states ?? [])); requireTextHeadroom(Number(size?.bytes ?? 0));
      const rows = this.db.prepare(`SELECT id,payload,sha256 FROM records WHERE kind=? AND id>?${states?.length ? ` AND state IN (${states.map(() => "?").join(",")})` : ""} ORDER BY id LIMIT 50`)
        .all(kind, after, ...(states ?? [])); if (!rows.length) return;
      for (const row of rows) { after = String(row.id); yield this.decoded<T>(row.payload, row.sha256); }
    }
  }
  latestStep(runId: string, index: number): AutomationStepReceipt | undefined {
    const size = this.db.prepare("SELECT length(CAST(payload AS BLOB)) AS bytes FROM steps WHERE run_id=? AND step_index=? ORDER BY attempt DESC LIMIT 1").get(automationId(runId), index); if (size) requireTextHeadroom(Number(size.bytes));
    const row = this.db.prepare("SELECT payload,sha256 FROM steps WHERE run_id=? AND step_index=? ORDER BY attempt DESC LIMIT 1").get(automationId(runId), index);
    return row ? validateAutomationStep(this.decoded(row.payload, row.sha256), runId) : undefined;
  }
  putStep(value: AutomationStepReceipt): void {
    const row = jsonValue(value); validateAutomationStep(row, row.runId); const body = automationCanonical(row);
    const prior = this.db.prepare("SELECT payload,sha256,state FROM steps WHERE run_id=? AND step_index=? AND attempt=?").get(row.runId, row.index, row.attempt);
    if (prior && immutableSteps.has(String(prior.state)) && prior.payload !== body) throw new Error("A terminal external-effect receipt is immutable.");
    this.db.prepare("INSERT INTO steps VALUES(?,?,?,?,?,?) ON CONFLICT(run_id,step_index,attempt) DO UPDATE SET state=excluded.state,payload=excluded.payload,sha256=excluded.sha256")
      .run(row.runId, row.index, row.attempt, row.state, body, automationHash(body));
  }
  stepPage(runId: string, query: AutomationPageQuery = {}): AutomationPage<AutomationStepReceipt> {
    automationId(runId); const limit = query.limit ?? 50, after = query.after ?? "0000000000000000-0000000000000000";
    if (!/^\d{16}-\d{16}$/.test(after) || !Number.isSafeInteger(limit) || limit < 1 || limit > 200) throw new Error("Invalid action-step page cursor.");
    const [index, attempt] = after.split("-").map(Number);
    const size = this.db.prepare("SELECT COALESCE(SUM(bytes),0) AS bytes FROM (SELECT length(CAST(payload AS BLOB)) AS bytes FROM steps WHERE run_id=? AND (step_index>? OR (step_index=? AND attempt>?)) ORDER BY step_index,attempt LIMIT ?)").get(runId, index!, index!, attempt!, limit + 1); requireTextHeadroom(Number(size?.bytes ?? 0));
    const rows = this.db.prepare("SELECT step_index,attempt,payload,sha256 FROM steps WHERE run_id=? AND (step_index>? OR (step_index=? AND attempt>?)) ORDER BY step_index,attempt LIMIT ?").all(runId, index!, index!, attempt!, limit + 1);
    const key = (row: Record<string, unknown>) => `${String(row.step_index).padStart(16, "0")}-${String(row.attempt).padStart(16, "0")}`;
    return { entries: rows.slice(0, limit).map(row => validateAutomationStep(this.decoded(row.payload, row.sha256), runId)), total: Number(this.db.prepare("SELECT COUNT(*) AS n FROM steps WHERE run_id=?").get(runId)?.n),
      ...(rows.length > limit ? { nextAfter: key(rows[limit - 1]!) } : {}) };
  }
  command<T>(requestId: string, argsSha: string, operation: () => T): T {
    return this.transaction(() => {
      const old = this.commandResult<T>(requestId, argsSha); if (old !== undefined) return old;
      const result = operation(); this.db.prepare("INSERT INTO requests VALUES(?,?,?)").run(requestId, argsSha, automationCanonical(jsonValue(result))); return result;
    });
  }
  commandResult<T>(requestId: string, argsSha: string): T | undefined {
    const size = this.db.prepare("SELECT length(CAST(result AS BLOB)) AS bytes FROM requests WHERE id=?").get(automationId(requestId)); if (size) requireTextHeadroom(Number(size.bytes));
    const old = this.db.prepare("SELECT arguments_sha256,result FROM requests WHERE id=?").get(automationId(requestId));
    if (!old) return undefined;
    if (old.arguments_sha256 !== argsSha) throw new Error("Action request ID was reused with different operands.");
    return JSON.parse(String(old.result)) as T;
  }
  event(runId: string, kind: string, value: unknown): void {
    automationId(runId); automationId(kind);
    const previous = this.db.prepare("SELECT sequence,sha256 FROM events WHERE run_id=? ORDER BY sequence DESC LIMIT 1").get(runId);
    const sequence = Number(previous?.sequence ?? 0) + 1, previousSha = String(previous?.sha256 ?? "0".repeat(64)), body = automationCanonical(jsonValue(value));
    const sha = automationHash(automationCanonical({ runId, sequence, kind, body, previousSha }));
    this.db.prepare("INSERT INTO events VALUES(?,?,?,?,?,?)").run(runId, sequence, kind, body, previousSha, sha);
  }
  static id(): string { return randomUUID(); }
}
