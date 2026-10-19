import { chmod, lstat, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { DatabaseSync } from "node:sqlite";
import type {
  EnvironmentOperationReceipt,
  EnvironmentPage,
  EnvironmentToolResult,
  ManagedEnvironmentRecord
} from "../shared/environments";
import {
  ENVIRONMENT_OPERATION_FORMAT,
  ENVIRONMENT_RECORD_FORMAT
} from "../shared/environments";
import {
  environmentCanonical,
  environmentHash,
  environmentId,
  environmentRequestId,
  environmentSha
} from "./environmentFormat";
import { sqliteFilename } from "./sqliteFilename";

export const ENVIRONMENT_WORKSPACE_PATH = "environments/workspace.sqlite3";
export const ENVIRONMENT_WORKSPACE_FORMAT = "omni-managed-environments-1";

const schema = new Map(Object.entries({
  meta: "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
  environments: "CREATE TABLE environments(id TEXT PRIMARY KEY,kind TEXT NOT NULL,state TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL)",
  operations: "CREATE TABLE operations(id TEXT PRIMARY KEY,environment_id TEXT NOT NULL,request_id TEXT NOT NULL,state TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL)",
  requests: "CREATE TABLE requests(id TEXT PRIMARY KEY,arguments_sha256 TEXT NOT NULL,operation_id TEXT NOT NULL,result TEXT NOT NULL)",
  environment_page: "CREATE INDEX environment_page ON environments(id)",
  operation_environment: "CREATE INDEX operation_environment ON operations(environment_id,id)"
}));
const schemaSql = (value: unknown): string => String(value).replace(/IF\s+NOT\s+EXISTS\s+/gi, "").replace(/\s+|;$/g, "").toUpperCase();
const states = new Set(["creating", "ready", "loading", "loaded", "busy", "stopping", "stopped", "dormant", "review-required", "error"]);
const operationStates = new Set(["intent", "running", "complete", "cancelled", "failed", "interrupted"]);
const actions = new Set(["create", "load", "run", "stop"]);
const adapters = new Set(["darwin-sandbox-exec", "linux-bwrap", "windows-sandbox-cli", "lima"]);

function requireParseHeadroom(bytes: number): void {
  const need = bytes * 12 + 65536;
  if (!Number.isSafeInteger(bytes) || bytes < 0 || !Number.isSafeInteger(need) ||
      need > freemem() / 2 || need > (getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 2) {
    throw new Error("Environment metadata awaits physical parse headroom.");
  }
}

function requireTimestamp(value: unknown, label: string): string {
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value))) throw new Error(`Invalid ${label}.`);
  return value;
}

export function validateEnvironmentRecord(value: unknown, brainId: string): ManagedEnvironmentRecord {
  const row = value as Partial<ManagedEnvironmentRecord>;
  if (!row || typeof row !== "object" || Array.isArray(row) || row.format !== ENVIRONMENT_RECORD_FORMAT || row.version !== 1 ||
      row.brainId !== brainId || !["sandbox", "vm"].includes(String(row.kind)) || !adapters.has(String(row.adapter)) ||
      !states.has(String(row.state)) || !Number.isSafeInteger(row.revision) || Number(row.revision) < 1 ||
      typeof row.name !== "string" || !row.name.trim() || row.name.length > 96 || /[\0-\x1f\x7f]/.test(row.name) ||
      !["none", "guest-outbound"].includes(String(row.network))) throw new Error("Invalid managed environment identity/state.");
  environmentId(row.id); requireTimestamp(row.createdAt, "environment created timestamp"); requireTimestamp(row.updatedAt, "environment updated timestamp");
  if (row.externalId !== undefined && (typeof row.externalId !== "string" ||
      (row.adapter === "windows-sandbox-cli" ? !/^[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i.test(row.externalId)
        : !/^omni-[a-f0-9-]{8,80}$/.test(row.externalId)))) throw new Error("Invalid adapter-owned environment identity.");
  if (row.imageSha256 !== undefined) environmentSha(row.imageSha256, "environment image checksum");
  if (row.guestArchitecture !== undefined && !["aarch64", "x86_64"].includes(row.guestArchitecture)) throw new Error("Invalid guest architecture.");
  if (row.resourcePlan !== undefined && (!Number.isSafeInteger(row.resourcePlan.cpus) || row.resourcePlan.cpus < 1 ||
      !Number.isSafeInteger(row.resourcePlan.memoryBytes) || row.resourcePlan.memoryBytes < 1 ||
      !Number.isSafeInteger(row.resourcePlan.diskBytes) || row.resourcePlan.diskBytes < 1)) throw new Error("Invalid environment resource plan.");
  if (row.orphanedProcessRisk !== undefined && typeof row.orphanedProcessRisk !== "boolean") throw new Error("Invalid environment orphan-process marker.");
  if (row.kind === "sandbox") {
    const windows = row.adapter === "windows-sandbox-cli";
    if (!["darwin-sandbox-exec", "linux-bwrap", "windows-sandbox-cli"].includes(String(row.adapter)) || row.network !== "none" ||
        (!windows && row.externalId !== undefined) || (windows && ["creating", "ready", "stopped", "dormant"].includes(String(row.state)) && row.externalId !== undefined) ||
        (windows && ["loading", "loaded", "busy", "stopping", "review-required"].includes(String(row.state)) && row.externalId === undefined) ||
        row.imageSha256 !== undefined || row.guestArchitecture !== undefined || row.resourcePlan !== undefined ||
        row.orphanedProcessRisk !== undefined && (row.orphanedProcessRisk !== true ||
          (windows ? !["review-required", "stopping"].includes(String(row.state)) : row.state !== "review-required"))) throw new Error("Sandbox record contains VM authority or an inconsistent adapter/network state.");
  } else {
    if (row.adapter !== "lima" || row.network !== "guest-outbound" || !row.imageSha256 || !row.guestArchitecture || !row.resourcePlan ||
        row.orphanedProcessRisk !== undefined || row.state !== "dormant" && !row.externalId) throw new Error("VM record is missing its exact Lima/image/resource identity.");
  }
  if (row.lastOperationId !== undefined) environmentId(row.lastOperationId, "environment operation id");
  if (row.warning !== undefined && (typeof row.warning !== "string" || row.warning.length > 2000 || row.warning.includes("\0"))) throw new Error("Invalid environment warning.");
  if (row.inheritedFrom !== undefined) {
    if (!row.inheritedFrom || typeof row.inheritedFrom !== "object" || typeof row.inheritedFrom.brainId !== "string") throw new Error("Invalid inherited environment provenance.");
    environmentId(row.inheritedFrom.environmentId);
  }
  return row as ManagedEnvironmentRecord;
}

export function validateEnvironmentOperation(value: unknown, brainId: string): EnvironmentOperationReceipt {
  const row = value as Partial<EnvironmentOperationReceipt>;
  if (!row || typeof row !== "object" || Array.isArray(row) || row.format !== ENVIRONMENT_OPERATION_FORMAT || row.version !== 1 ||
      row.brainId !== brainId || !actions.has(String(row.action)) || !operationStates.has(String(row.state)) ||
      typeof row.dispatchStarted !== "boolean") throw new Error("Invalid environment operation receipt.");
  environmentId(row.id, "environment operation id"); environmentId(row.environmentId); environmentRequestId(row.requestId);
  environmentSha(row.argumentsSha256, "environment operation arguments hash");
  requireTimestamp(row.createdAt, "operation created timestamp"); requireTimestamp(row.updatedAt, "operation updated timestamp");
  if (row.error !== undefined && (typeof row.error !== "string" || row.error.length > 8000 || row.error.includes("\0"))) throw new Error("Invalid environment operation error.");
  if (row.process !== undefined && (!Number.isSafeInteger(row.process.exitCode) || typeof row.process.stdout !== "string" ||
      typeof row.process.stderr !== "string" || typeof row.process.truncated !== "boolean" ||
      row.process.guestOutput !== undefined && row.process.guestOutput !== "guest-output-unavailable" ||
      row.process.sandboxDisposition !== undefined && !["already-absent", "stopped", "unconfirmed"].includes(row.process.sandboxDisposition))) throw new Error("Invalid environment process receipt.");
  if (row.process?.guestOutput !== undefined && row.action !== "run") throw new Error("Only an environment Run receipt may describe guest output availability.");
  return row as EnvironmentOperationReceipt;
}

function requireOperationOutputInvariant(operation: EnvironmentOperationReceipt, environment: ManagedEnvironmentRecord): void {
  const unavailable = operation.process?.guestOutput === "guest-output-unavailable";
  const required = environment.adapter === "windows-sandbox-cli" && operation.action === "run" && operation.process !== undefined;
  if (unavailable !== required) throw new Error("Environment operation guest-output evidence differs from its exact adapter/action owner.");
  const disposition = operation.process?.sandboxDisposition;
  const dispositionRequired = environment.adapter === "windows-sandbox-cli" && operation.action === "stop" && operation.process !== undefined;
  if ((disposition !== undefined) !== dispositionRequired) throw new Error("Environment Stop disposition differs from its exact adapter/action owner.");
  if (disposition === "unconfirmed" && (operation.process!.exitCode === 0 || operation.state === "complete") ||
      (disposition === "stopped" || disposition === "already-absent") && (operation.process!.exitCode !== 0 || operation.state !== "complete")) {
    throw new Error("Environment Stop disposition contradicts its retained exit/state evidence.");
  }
}

async function requireDirectory(path: string, label: string, privateDirectory = false): Promise<void> {
  let info = await lstat(path);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error(`${label} must be a private real directory.`);
  if (privateDirectory && process.platform !== "win32") {
    const uid = typeof process.getuid === "function" ? process.getuid() : info.uid;
    if (info.uid !== uid) throw new Error(`${label} has another operating-system owner.`);
    if ((info.mode & 0o077) !== 0) { await chmod(path, 0o700); info = await lstat(path); }
    if (!info.isDirectory() || info.isSymbolicLink() || info.uid !== uid || (info.mode & 0o077) !== 0) throw new Error(`${label} is not private to the current operating-system owner.`);
  }
}

export class EnvironmentStore {
  private constructor(readonly root: string, readonly brainId: string, readonly db: DatabaseSync) {}

  static async open(root: string, brainId: string, readOnly = false): Promise<EnvironmentStore> {
    const base = join(root, "environments"), instances = join(base, "instances"), path = join(root, ENVIRONMENT_WORKSPACE_PATH);
    if (!readOnly) await mkdir(instances, { recursive: true, mode: 0o700 });
    await requireDirectory(root, "Brain root"); await requireDirectory(base, "Environment workspace", true); await requireDirectory(instances, "Environment instance directory", true);
    const file = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (file && (!file.isFile() || file.isSymbolicLink())) throw new Error("Environment database must be a private regular file.");
    const db = new DatabaseSync(sqliteFilename(path), { readOnly, allowExtension: false });
    try {
      db.exec("PRAGMA trusted_schema=OFF;");
      for (const row of db.prepare("SELECT type,name,sql FROM sqlite_master").iterate()) {
        if (row.type === "index" && /^sqlite_autoindex_[A-Za-z0-9_]+_\d+$/.test(String(row.name)) && row.sql === null) continue;
        const declaration = schema.get(String(row.name));
        if (!declaration || row.type !== (String(row.name).includes("_page") || String(row.name).startsWith("operation_") ? "index" : "table") || schemaSql(row.sql) !== schemaSql(declaration)) throw new Error("Environment database contains unsupported executable/schema objects.");
      }
      if (!readOnly) {
        db.exec(`PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA busy_timeout=2500; PRAGMA cache_size=-256; PRAGMA temp_store=FILE;
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS environments(id TEXT PRIMARY KEY,kind TEXT NOT NULL,state TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS environment_page ON environments(id);
          CREATE TABLE IF NOT EXISTS operations(id TEXT PRIMARY KEY,environment_id TEXT NOT NULL,request_id TEXT NOT NULL,state TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS operation_environment ON operations(environment_id,id);
          CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,arguments_sha256 TEXT NOT NULL,operation_id TEXT NOT NULL,result TEXT NOT NULL);`);
        db.prepare("INSERT OR IGNORE INTO meta VALUES('format',?)").run(ENVIRONMENT_WORKSPACE_FORMAT);
        db.prepare("INSERT OR IGNORE INTO meta VALUES('brainId',?)").run(brainId);
      }
      if (db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== ENVIRONMENT_WORKSPACE_FORMAT ||
          db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) throw new Error("Environment workspace has another format or owner.");
      return new EnvironmentStore(root, brainId, db);
    } catch (error) { db.close(); throw error; }
  }

  close(): void { this.db.close(); }
  transaction<T>(operation: () => T): T {
    this.db.exec("BEGIN IMMEDIATE");
    try { const result = operation(); this.db.exec("COMMIT"); return result; }
    catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  instanceDirectory(id: string): string { return join(this.root, "environments", "instances", environmentId(id)); }

  private decode<T>(payload: unknown, sha256: unknown): T {
    if (typeof payload !== "string") throw new Error("Environment metadata payload is invalid.");
    requireParseHeadroom(Buffer.byteLength(payload));
    if (environmentHash(payload) !== sha256) throw new Error("Environment metadata checksum failed.");
    return JSON.parse(payload) as T;
  }
  private environmentRow(row: Record<string, unknown>): ManagedEnvironmentRecord {
    const value = validateEnvironmentRecord(this.decode(row.payload, row.sha256), this.brainId);
    if (row.id !== value.id || row.kind !== value.kind || row.state !== value.state) throw new Error("Environment SQL index differs from its checksummed payload.");
    return value;
  }
  private operationRow(row: Record<string, unknown>): EnvironmentOperationReceipt {
    const value = validateEnvironmentOperation(this.decode(row.payload, row.sha256), this.brainId);
    if (row.id !== value.id || row.environment_id !== value.environmentId || row.request_id !== value.requestId || row.state !== value.state) throw new Error("Environment operation SQL index differs from its checksummed payload.");
    const environment = this.getEnvironment(value.environmentId); if (!environment) throw new Error("Environment operation lost its exact environment owner.");
    requireOperationOutputInvariant(value, environment);
    return value;
  }
  private toolResult(value: unknown, expectedOperationId?: string): EnvironmentToolResult {
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Environment request result is invalid.");
    const result = value as Partial<EnvironmentToolResult>, environment = validateEnvironmentRecord(result.environment, this.brainId), operation = validateEnvironmentOperation(result.operation, this.brainId);
    if (operation.environmentId !== environment.id || expectedOperationId && operation.id !== expectedOperationId) throw new Error("Environment request result differs from its exact operation owner.");
    requireOperationOutputInvariant(operation, environment);
    return { environment, operation };
  }

  getEnvironment(id: string): ManagedEnvironmentRecord | undefined {
    const row = this.db.prepare("SELECT id,kind,state,payload,sha256 FROM environments WHERE id=?").get(environmentId(id));
    return row ? this.environmentRow(row) : undefined;
  }
  putEnvironment(value: ManagedEnvironmentRecord): void {
    const row = validateEnvironmentRecord(structuredClone(value), this.brainId), payload = environmentCanonical(row);
    this.db.prepare("INSERT INTO environments VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET kind=excluded.kind,state=excluded.state,payload=excluded.payload,sha256=excluded.sha256")
      .run(row.id, row.kind, row.state, payload, environmentHash(payload));
  }
  page(after = "", limit = 50): EnvironmentPage {
    if (after) environmentId(after); if (!Number.isSafeInteger(limit) || limit < 1 || limit > 200) throw new Error("Use environment pages from 1 to 200.");
    const rows = this.db.prepare("SELECT id,kind,state,payload,sha256 FROM environments WHERE id>? ORDER BY id LIMIT ?").all(after, limit + 1);
    const entries = rows.slice(0, limit).map(row => this.environmentRow(row));
    return { entries, total: Number(this.db.prepare("SELECT COUNT(*) AS n FROM environments").get()?.n ?? 0),
      ...(rows.length > limit ? { nextAfter: entries.at(-1)!.id } : {}) };
  }
  *environments(): Generator<ManagedEnvironmentRecord> {
    for (const row of this.db.prepare("SELECT id,kind,state,payload,sha256 FROM environments ORDER BY id").iterate()) yield this.environmentRow(row);
  }

  getOperation(id: string): EnvironmentOperationReceipt | undefined {
    const row = this.db.prepare("SELECT id,environment_id,request_id,state,payload,sha256 FROM operations WHERE id=?").get(environmentId(id, "environment operation id"));
    return row ? this.operationRow(row) : undefined;
  }
  putOperation(value: EnvironmentOperationReceipt): void {
    const row = validateEnvironmentOperation(structuredClone(value), this.brainId), payload = environmentCanonical(row);
    const environment = this.getEnvironment(row.environmentId); if (!environment) throw new Error("Environment operation lost its exact environment owner.");
    requireOperationOutputInvariant(row, environment);
    this.db.prepare("INSERT INTO operations VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state,payload=excluded.payload,sha256=excluded.sha256")
      .run(row.id, row.environmentId, row.requestId, row.state, payload, environmentHash(payload));
  }
  operations(environmentIdValue: string): EnvironmentOperationReceipt[] {
    const id = environmentId(environmentIdValue), rows = this.db.prepare("SELECT id,environment_id,request_id,state,payload,sha256 FROM operations WHERE environment_id=? ORDER BY id").all(id);
    return rows.map(row => this.operationRow(row));
  }

  request(requestIdValue: string, argumentsSha256: string): { operationId: string; result?: EnvironmentToolResult } | undefined {
    const requestId = environmentRequestId(requestIdValue); environmentSha(argumentsSha256);
    const row = this.db.prepare("SELECT arguments_sha256,operation_id,result FROM requests WHERE id=?").get(requestId);
    if (!row) return undefined;
    if (row.arguments_sha256 !== argumentsSha256) throw new Error("Environment request ID was reused with different operands.");
    const operationId = environmentId(row.operation_id, "environment operation id");
    const operation = this.getOperation(operationId);
    if (!operation || operation.requestId !== requestId || operation.argumentsSha256 !== argumentsSha256) throw new Error("Environment request SQL binding differs from its operation payload.");
    if (!row.result) return { operationId };
    if (typeof row.result !== "string") throw new Error("Environment request result is invalid.");
    requireParseHeadroom(Buffer.byteLength(row.result));
    const result = this.toolResult(JSON.parse(row.result), operationId);
    if (environmentCanonical(result.operation) !== environmentCanonical(operation)) throw new Error("Environment request result operation differs from the retained ledger receipt.");
    return { operationId, result };
  }
  beginRequest(requestIdValue: string, argumentsSha256: string, operationIdValue: string): void {
    const requestId = environmentRequestId(requestIdValue), operationId = environmentId(operationIdValue, "environment operation id"); environmentSha(argumentsSha256);
    this.db.prepare("INSERT INTO requests VALUES(?,?,?,?)").run(requestId, argumentsSha256, operationId, "");
  }
  finishRequest(requestIdValue: string, result: EnvironmentToolResult): void {
    const requestId = environmentRequestId(requestIdValue), current = this.db.prepare("SELECT operation_id FROM requests WHERE id=?").get(requestId);
    if (!current || current.operation_id !== result.operation.id) throw new Error("Environment request lost its exact operation owner.");
    const checked = this.toolResult(structuredClone(result), String(current.operation_id));
    const operation = this.getOperation(String(current.operation_id));
    if (!operation || environmentCanonical(checked.operation) !== environmentCanonical(operation)) throw new Error("Environment request result does not equal its retained ledger operation.");
    const environment = this.getEnvironment(checked.environment.id);
    if (!environment || environmentCanonical(checked.environment) !== environmentCanonical(environment)) throw new Error("Environment request result does not equal its retained environment state.");
    this.db.prepare("UPDATE requests SET result=? WHERE id=?").run(environmentCanonical(checked), requestId);
  }

  recoverInterrupted(): number {
    let count = 0; const now = new Date().toISOString();
    this.transaction(() => {
      for (const operation of this.db.prepare("SELECT id,environment_id,request_id,state,payload,sha256 FROM operations WHERE state IN ('intent','running')").all()
        .map(row => this.operationRow(row))) {
        this.putOperation({ ...operation, state: "interrupted", updatedAt: now,
          error: "The previous host ended before this operation reached a committed terminal receipt; it was not replayed." }); count++;
      }
      for (const environment of this.db.prepare("SELECT id,kind,state,payload,sha256 FROM environments WHERE state IN ('creating','loading','busy','stopping')").all()
        .map(row => this.environmentRow(row))) {
        this.putEnvironment({ ...environment, state: environment.adapter === "windows-sandbox-cli" && !environment.externalId ? "error" : "review-required",
          revision: environment.revision + 1, updatedAt: now,
          ...(environment.kind === "sandbox" && ["busy", "stopping"].includes(environment.state) ? { orphanedProcessRisk: true } : {}),
          warning: "The host ended during an operation. Inspect the actual adapter state before loading or issuing a new effect." });
      }
    });
    return count;
  }
}
