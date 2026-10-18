import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, mkdir, open, opendir, readFile, rename, rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import { DatabaseSync, type SQLInputValue } from "node:sqlite";
import type { McpAppArtifactRecord, McpAppArtifactSummary, McpAppBlob, McpAppCapture, McpAppObservationArtifact, McpAppResourceArtifact, McpAppSource, McpAppViewProjection } from "../shared/mcpAppArtifacts";
import type { McpAppObservationReceipt, McpAppTaskData } from "../shared/mcpAppTaskData";
import type { McpCompletedAppCall } from "./mcpApps";
import type { McpAppCsp } from "../shared/mcpApps";
import { MCP_APP_MIME } from "../shared/mcpApps";
import { mcpAppCsp, mcpToolUi } from "./mcpAppMetadata";
import { copyMutableFileIsolated } from "./mutableFileIsolation";
import { mcpAppTaskData } from "./mcpAppTaskData";
import type { NativeAppObservationDisposition, NativeAppObservationResult, NativeAppObservationHead } from "../shared/nativeAppObservation";
import { checkedAppObservationResult } from "./mcpAppObservationReceipt";
import { checkedChatInputProvenance } from "../shared/chatInputProvenance";
import type { ChatMessage } from "../shared/types";
import { checkedAppObservationContent, verifyAppObservationAudit } from "./mcpAppObservationAudit";

const FORMAT = "omni-mcp-app-artifacts-1", SHA = /^[a-f0-9]{64}$/, ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const UUID = /^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i;
const leases = new Map<string, Promise<void>>();
const digest = (value: string | Uint8Array): string => createHash("sha256").update(value).digest("hex");
const object = (value: unknown): Record<string, unknown> => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const flatMetadata = (value: object): string => JSON.stringify(Object.entries(value).sort(([left], [right]) => left.localeCompare(right)));
function sql(value: unknown): SQLInputValue {
  if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "bigint" || value instanceof Uint8Array) return value;
  throw new Error("App artifact ledger omitted a required field.");
}
export interface McpAppArtifactHooks { signal?: AbortSignal; admit?(bytes: number): Promise<void>; checkDisk?(bytes: number): Promise<void>; }
export interface McpAppObservationClaim {
  record: McpAppObservationArtifact;
  recordJson: string;
  recordSha256: string;
  dataJson: string;
  disposition: NativeAppObservationDisposition;
  expectedFrontierSha256?: string;
}
export interface McpAppFollowup {
  claim: McpAppObservationClaim;
  capture: McpAppCapture;
  captureJson: string;
  native: NativeAppObservationResult;
  intent?: { turnId: string; inputSha256: string; state: string; error?: string };
}

/** The lock is independent of the neural writer: capture can happen inside it. */
export async function withMcpAppArtifactsLease<T>(brainRoot: string, action: () => Promise<T>): Promise<T> {
  const key = resolve(brainRoot), previous = leases.get(key) ?? Promise.resolve();
  let release!: () => void; const next = new Promise<void>(yes => { release = yes; }); leases.set(key, next);
  await previous;
  try { return await action(); } finally { release(); if (leases.get(key) === next) leases.delete(key); }
}
async function exists(path: string): Promise<boolean> {
  try { await lstat(path); return true; } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; }
}
async function fileHash(path: string, signal?: AbortSignal): Promise<string> {
  const before = await lstat(path); if (!before.isFile() || before.isSymbolicLink()) throw new Error("An App artifact blob is not an owned regular file.");
  const hash = createHash("sha256"); for await (const chunk of createReadStream(path, { highWaterMark: 64 * 1024, signal })) hash.update(chunk);
  const after = await lstat(path);
  if (after.ino !== before.ino || after.dev !== before.dev || after.size !== before.size || after.mtimeMs !== before.mtimeMs) throw new Error("An App artifact blob changed while being verified.");
  return hash.digest("hex");
}
async function assertTree(brainRoot: string): Promise<void> {
  for (const path of [brainRoot, join(brainRoot, "mcp-apps"), join(brainRoot, "mcp-apps", "blobs")]) {
    const info = await lstat(path); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("App artifact path is not a real owned directory.");
  }
  const index = await lstat(join(brainRoot, "mcp-apps", "index.sqlite3"));
  if (!index.isFile() || index.isSymbolicLink()) throw new Error("App artifact index is not a real owned file.");
}
function source(value: unknown): McpAppSource {
  const row = object(value);
  if (Object.keys(row).sort().join(",") !== "brainId,endpoint,serverId,serverLabel" || !ID.test(String(row.brainId)) || !ID.test(String(row.serverId)) ||
    typeof row.endpoint !== "string" || typeof row.serverLabel !== "string" || /[\0\r\n]/.test(row.endpoint + row.serverLabel)) throw new Error("An App artifact has invalid original source provenance.");
  return row as unknown as McpAppSource;
}
function blob(value: unknown): McpAppBlob {
  const row = object(value);
  if (Object.keys(row).sort().join(",") !== "bytes,mimeType,sha256" || !SHA.test(String(row.sha256)) || !Number.isSafeInteger(row.bytes) || Number(row.bytes) < 0 ||
    typeof row.mimeType !== "string" || row.mimeType.includes("\0")) throw new Error("An App artifact has invalid blob metadata.");
  return row as unknown as McpAppBlob;
}
export function validateMcpAppArtifact(value: unknown): McpAppArtifactRecord {
  const row = object(value);
  if (row.version !== 1 || row.untrusted !== true || !UUID.test(String(row.artifactId)) || typeof row.observedAt !== "string" || !Number.isFinite(Date.parse(row.observedAt))) throw new Error("Invalid inert MCP App artifact.");
  source(row.source);
  if (row.format === "omni-mcp-app-capture") {
    if (Object.keys(row).sort().join(",") !== "artifactId,format,input,observedAt,resourceUri,result,source,tool,untrusted,version") throw new Error("App capture contains session or unsupported state.");
    const tool = object(row.tool); const ui = mcpToolUi(tool);
    if (typeof tool.id !== "string" || typeof tool.remoteName !== "string" || tool.remoteName.includes("\0") || tool.clientOperation !== undefined ||
      ui.resourceUri !== row.resourceUri || !tool.inputSchema || typeof tool.inputSchema !== "object") throw new Error("App capture lost its original tool/resource identity.");
    if (blob(row.input).mimeType !== "application/json" || blob(row.result).mimeType !== "application/json") throw new Error("App call data must remain inert JSON.");
  } else if (row.format === "omni-mcp-app-resource") {
    if (Object.keys(row).sort().join(",") !== "artifactId,captureSha256,content,declaredCsp,format,metadata,observedAt,source,untrusted,uri,version" || !SHA.test(String(row.captureSha256)) ||
      typeof row.uri !== "string" || new URL(row.uri).protocol !== "ui:" || blob(row.content).mimeType !== MCP_APP_MIME) throw new Error("Invalid App resource artifact.");
    if (JSON.stringify(mcpAppCsp(row.metadata)) !== JSON.stringify(row.declaredCsp)) throw new Error("App resource declaration changed.");
  } else if (row.format === "omni-mcp-app-observation") {
    if (Object.keys(row).sort().join(",") !== "artifactId,captureSha256,content,format,id,method,observedAt,ordinal,previousSha256,reviewId,source,streamId,untrusted,version" ||
      !SHA.test(String(row.captureSha256)) || !UUID.test(String(row.id)) || !UUID.test(String(row.reviewId)) || !UUID.test(String(row.streamId)) ||
      !Number.isSafeInteger(row.ordinal) || Number(row.ordinal) < 1 || !SHA.test(String(row.previousSha256)) ||
      !["ui/message", "ui/update-model-context"].includes(String(row.method)) || blob(row.content).mimeType !== "application/json") throw new Error("App observation contains invalid task, review or stream identity.");
  } else throw new Error("Unsupported App artifact record.");
  return row as unknown as McpAppArtifactRecord;
}
function openDatabase(path: string, brainId: string, create = false): DatabaseSync {
  const db = new DatabaseSync(path, { readOnly: !create });
  try {
    db.exec("PRAGMA busy_timeout=1000; PRAGMA cache_size=-512; PRAGMA trusted_schema=OFF");
    if (create) db.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE IF NOT EXISTS records(record_sha TEXT PRIMARY KEY,artifact_id TEXT NOT NULL,kind TEXT NOT NULL,payload_json TEXT NOT NULL); CREATE INDEX IF NOT EXISTS record_owner ON records(artifact_id,kind); CREATE TABLE IF NOT EXISTS bindings(artifact_id TEXT PRIMARY KEY,capture_sha TEXT,resource_sha TEXT,reviewed_at TEXT); CREATE TABLE IF NOT EXISTS ancestry(parent_id TEXT NOT NULL,artifact_id TEXT NOT NULL,record_sha TEXT NOT NULL,PRIMARY KEY(parent_id,artifact_id,record_sha)); CREATE TABLE IF NOT EXISTS observation_state(record_sha TEXT PRIMARY KEY,stream_id TEXT NOT NULL,ordinal INTEGER NOT NULL,state TEXT NOT NULL,content_bytes INTEGER NOT NULL,review_id TEXT NOT NULL); CREATE INDEX IF NOT EXISTS observation_stream ON observation_state(stream_id,ordinal); CREATE TABLE IF NOT EXISTS observation_claims(record_sha TEXT PRIMARY KEY,disposition TEXT NOT NULL,base_frontier_sha TEXT); CREATE TABLE IF NOT EXISTS observation_receipts(record_sha TEXT PRIMARY KEY,receipt_json TEXT NOT NULL); CREATE TABLE IF NOT EXISTS observation_followups(record_sha TEXT PRIMARY KEY,turn_id TEXT NOT NULL,input_sha TEXT NOT NULL,state TEXT NOT NULL,error TEXT);");
    if (create) db.exec("CREATE TABLE IF NOT EXISTS observation_audits(sha256 TEXT PRIMARY KEY,descriptor_json TEXT NOT NULL,summary_json TEXT NOT NULL); CREATE TABLE IF NOT EXISTS observation_archive_claims(record_sha TEXT PRIMARY KEY,previous_audit_sha TEXT);");
    const extra = [...db.prepare("SELECT type,name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'").iterate()].some(row =>
      !["meta", "records", "record_owner", "bindings", "ancestry", "observation_state", "observation_stream", "observation_claims", "observation_receipts", "observation_followups", "observation_audits", "observation_archive_claims"].includes(String(row.name)) || !["table", "index"].includes(String(row.type)));
    if (extra) throw new Error("The App artifact ledger contains unsupported executable schema.");
    const prior = db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value;
    if (!prior && create) { db.prepare("INSERT INTO meta VALUES('format',?)").run(FORMAT); db.prepare("INSERT INTO meta VALUES('brainId',?)").run(brainId); }
    if (db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== FORMAT || db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) throw new Error("App artifacts belong to another owner or format.");
    if (db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("App artifact ledger failed SQLite integrity.");
    return db;
  } catch (error) { db.close(); throw error; }
}
const hasTable = (db: DatabaseSync, name: string): boolean => !!db.prepare("SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?").get(name);
const hasObservations = (db: DatabaseSync): boolean => hasTable(db, "observation_state");
function observationReceipt(record: McpAppObservationArtifact, sha256: string, state: McpAppObservationReceipt["state"], nativeReceipt?: NativeAppObservationResult): McpAppObservationReceipt {
  if (state === "admitted" && !nativeReceipt) throw new Error("An App observation has no genuine native admission receipt.");
  return { id: record.id, artifactId: record.artifactId, recordSha256: sha256, streamId: record.streamId, ordinal: record.ordinal,
    state, observedAt: record.observedAt, method: record.method,
    neuralAdmission: state === "admitted" ? "admitted" : state === "superseded" && nativeReceipt ? "advanced-without-observation" : "not-yet-admitted",
    ...(nativeReceipt && state !== "dormant" ? { nativeReceipt } : {}) };
}

/** UI artifacts are evidence, never an authoritative fact/answer memory. */
export class McpAppArtifactStore {
  readonly directory: string;
  constructor(readonly brainRoot: string, readonly brainId: string, private readonly hooks: McpAppArtifactHooks = {}) {
    if (!ID.test(brainId)) throw new Error("Invalid App artifact owner."); this.directory = join(brainRoot, "mcp-apps");
  }
  private async prepare(): Promise<void> {
    for (const path of [this.brainRoot, this.directory, join(this.directory, "blobs")]) {
      if (path !== this.brainRoot) await mkdir(path, { mode: 0o700 }).catch((error: NodeJS.ErrnoException) => { if (error.code !== "EEXIST") throw error; });
      const info = await lstat(path); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("App artifacts require real owned directories.");
    }
    const path = join(this.directory, "index.sqlite3");
    if (await exists(path)) { const info = await lstat(path); if (!info.isFile() || info.isSymbolicLink()) throw new Error("App ledger is not an owned regular file."); }
  }
  async initialize(): Promise<void> {
    await withMcpAppArtifactsLease(this.brainRoot, async () => { await this.prepare(); await this.hooks.checkDisk?.(65536);
      const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId, true); db.close(); });
  }
  private async putBlob(value: string, mimeType: string): Promise<McpAppBlob> {
    const bytes = Buffer.byteLength(value), sha256 = digest(value), path = join(this.directory, "blobs", `${sha256}.data`);
    await this.hooks.admit?.(bytes * 3); this.hooks.signal?.throwIfAborted();
    if (await exists(path)) { if ((await lstat(path)).size !== bytes || await fileHash(path, this.hooks.signal) !== sha256) throw new Error("A saved App blob changed."); }
    else {
      await this.hooks.checkDisk?.(bytes); const temporary = `${path}.${randomUUID()}.tmp`, file = await open(temporary, "wx", 0o600);
      try { await file.writeFile(value, "utf8"); await file.sync(); await file.close(); await rename(temporary, path); }
      finally { await file.close().catch(() => undefined); await rm(temporary, { force: true }); }
    }
    return { sha256, bytes, mimeType };
  }
  private insert(db: DatabaseSync, record: McpAppArtifactRecord): string {
    validateMcpAppArtifact(record); const payload = JSON.stringify(record), sha = digest(payload);
    const previous = db.prepare("SELECT payload_json FROM records WHERE record_sha=?").get(sha)?.payload_json;
    if (previous !== undefined && previous !== payload) throw new Error("An App artifact content hash collided.");
    db.prepare("INSERT OR IGNORE INTO records VALUES(?,?,?,?)").run(sha, record.artifactId, record.format, payload);
    return sha;
  }
  async capture(artifactId: string, call: McpCompletedAppCall): Promise<{ capture: McpAppCapture; sha256: string }> {
    return withMcpAppArtifactsLease(this.brainRoot, async () => {
      await this.prepare(); const ui = mcpToolUi(call.tool); if (!ui.resourceUri) throw new Error("The completed tool has no App resource.");
      const input = await this.putBlob(JSON.stringify(call.arguments), "application/json"), result = await this.putBlob(JSON.stringify(call.result), "application/json");
      const capture: McpAppCapture = { format: "omni-mcp-app-capture", version: 1, artifactId,
        source: { brainId: call.origin.brainId, serverId: call.origin.serverId, serverLabel: call.origin.serverLabel, endpoint: call.origin.endpoint },
        observedAt: new Date().toISOString(), resourceUri: ui.resourceUri, tool: structuredClone(call.tool), input, result, untrusted: true };
      const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId, true);
      try { db.exec("BEGIN IMMEDIATE"); const sha256 = this.insert(db, capture);
        const prior = db.prepare("SELECT capture_sha FROM bindings WHERE artifact_id=?").get(artifactId);
        db.prepare("INSERT OR REPLACE INTO bindings VALUES(?,?,NULL,NULL)").run(artifactId, !prior || prior.capture_sha === sha256 ? sha256 : null);
        db.exec("COMMIT"); return { capture, sha256 };
      } catch (error) { try { db.exec("ROLLBACK"); } catch { /* original failure */ } throw error; } finally { db.close(); }
    });
  }
  async resource(artifactId: string, captureSha256: string, source: McpAppSource, uri: string, html: string, metadata: Record<string, unknown>, declaredCsp: McpAppCsp): Promise<string> {
    return withMcpAppArtifactsLease(this.brainRoot, async () => {
      await this.prepare(); const content = await this.putBlob(html, MCP_APP_MIME);
      const record: McpAppResourceArtifact = { format: "omni-mcp-app-resource", version: 1, artifactId, captureSha256, source, uri, content,
        metadata: structuredClone(metadata), declaredCsp: structuredClone(declaredCsp), observedAt: new Date().toISOString(), untrusted: true };
      const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId, true);
      try {
        const capture = db.prepare("SELECT payload_json FROM records WHERE record_sha=? AND artifact_id=?").get(captureSha256, artifactId)?.payload_json;
        if (typeof capture !== "string" || validateMcpAppArtifact(JSON.parse(capture)).format !== "omni-mcp-app-capture") throw new Error("App resource lost its exact completed-call capture.");
        db.exec("BEGIN IMMEDIATE"); const sha = this.insert(db, record);
        db.prepare("UPDATE bindings SET resource_sha=? WHERE artifact_id=? AND capture_sha=?").run(sha, artifactId, captureSha256); db.exec("COMMIT"); return sha;
      } catch (error) { try { db.exec("ROLLBACK"); } catch { /* original failure */ } throw error; } finally { db.close(); }
    });
  }
  /** A reviewed queue receipt is not a neural admission/learning receipt. */
  async queueObservation(artifactId: string, captureSha256: string, origin: McpAppSource, reviewId: string, data: McpAppTaskData,
    maximumPendingBytes: number): Promise<McpAppObservationReceipt> {
    const checked = mcpAppTaskData(data.method, data.params), payload = JSON.stringify(checked), bytes = Buffer.byteLength(payload);
    if (!UUID.test(reviewId) || !Number.isSafeInteger(maximumPendingBytes) || maximumPendingBytes < 0) throw new Error("Invalid reviewed App observation queue budget.");
    return withMcpAppArtifactsLease(this.brainRoot, async () => {
      await this.prepare(); const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId, true);
      try {
        const capture = this.readRecord(db, captureSha256);
        if (capture.format !== "omni-mcp-app-capture" || capture.artifactId !== artifactId || origin.brainId !== this.brainId ||
          capture.source.serverId !== origin.serverId || capture.source.endpoint !== origin.endpoint) throw new Error("App task lost its exact owned call/source binding.");
        const existing = db.prepare("SELECT record_sha,state FROM observation_state WHERE review_id=?").get(reviewId);
        if (existing) {
          const record = this.readRecord(db, String(existing.record_sha));
          if (record.format !== "omni-mcp-app-observation" || record.artifactId !== artifactId || record.captureSha256 !== captureSha256 ||
            record.content.sha256 !== digest(payload)) throw new Error("An App review was reused with changed task data.");
          return this.observationReceipt(db, record, String(existing.record_sha), String(existing.state) as McpAppObservationReceipt["state"]);
        }
        const pending = Number(db.prepare("SELECT COALESCE(SUM(s.content_bytes),0) AS bytes FROM observation_state s JOIN records r ON r.record_sha=s.record_sha WHERE s.state IN ('pending-native-admission','processing-native-admission') AND NOT (?='ui/update-model-context' AND s.state='pending-native-admission' AND r.artifact_id=? AND json_extract(r.payload_json,'$.method')='ui/update-model-context')").get(data.method, artifactId)?.bytes);
        if (!Number.isSafeInteger(pending) || pending + bytes > maximumPendingBytes) throw new Error("App observations await native processing or more memory budget. Nothing was silently dropped or marked learned.");
        const content = await this.putBlob(payload, "application/json");
        let streamId = db.prepare("SELECT value FROM meta WHERE key='activeObservationStream'").get()?.value;
        if (typeof streamId !== "string") streamId = randomUUID();
        if (!UUID.test(String(streamId))) throw new Error("App observation stream identity changed.");
        const previous = db.prepare("SELECT record_sha,ordinal FROM observation_state WHERE stream_id=? ORDER BY ordinal DESC LIMIT 1").get(sql(streamId));
        const record: McpAppObservationArtifact = { format: "omni-mcp-app-observation", version: 1, id: randomUUID(), artifactId, captureSha256,
          source: structuredClone(origin), observedAt: new Date().toISOString(), reviewId, method: data.method, content,
          streamId: String(streamId), ordinal: previous ? Number(previous.ordinal) + 1 : 1, previousSha256: previous ? String(previous.record_sha) : "0".repeat(64), untrusted: true };
        await this.hooks.checkDisk?.(Buffer.byteLength(JSON.stringify(record)) * 4 + 8192);
        db.exec("BEGIN IMMEDIATE");
        if (data.method === "ui/update-model-context") db.prepare("UPDATE observation_state SET state='superseded' WHERE state='pending-native-admission' AND record_sha IN (SELECT record_sha FROM records WHERE artifact_id=? AND json_extract(payload_json,'$.method')='ui/update-model-context')").run(artifactId);
        const recordSha256 = this.insert(db, record);
        db.prepare("INSERT INTO observation_state VALUES(?,?,?,?,?,?)").run(recordSha256, record.streamId, record.ordinal, "pending-native-admission", bytes, reviewId);
        db.prepare("INSERT OR REPLACE INTO meta VALUES('activeObservationStream',?)").run(record.streamId); db.exec("COMMIT");
        return observationReceipt(record, recordSha256, "pending-native-admission");
      } catch (error) { try { db.exec("ROLLBACK"); } catch { /* original failure */ } throw error; } finally { db.close(); }
    });
  }
  private readRecord(db: DatabaseSync, sha256: string): McpAppArtifactRecord {
    if (!SHA.test(sha256)) throw new Error("Invalid App record identity."); const text = db.prepare("SELECT payload_json FROM records WHERE record_sha=?").get(sha256)?.payload_json;
    if (typeof text !== "string" || digest(text) !== sha256) throw new Error("Saved App record failed content verification."); return validateMcpAppArtifact(JSON.parse(text));
  }
  private observationReceipt(db: DatabaseSync, record: McpAppObservationArtifact, sha256: string, state: McpAppObservationReceipt["state"]): McpAppObservationReceipt {
    const raw = hasTable(db, "observation_receipts") ? db.prepare("SELECT receipt_json FROM observation_receipts WHERE record_sha=?").get(sha256)?.receipt_json : undefined;
    const claim = hasTable(db, "observation_claims") ? db.prepare("SELECT disposition,base_frontier_sha FROM observation_claims WHERE record_sha=?").get(sha256) : undefined;
    const receipt = typeof raw === "string" && claim ? checkedAppObservationResult(JSON.parse(raw), record.source.brainId, record,
      sha256, String(claim.disposition) as NativeAppObservationDisposition, String(claim.base_frontier_sha)) : undefined;
    return observationReceipt(record, sha256, state, receipt);
  }
  async claimObservation(): Promise<McpAppObservationClaim | undefined> {
    return withMcpAppArtifactsLease(this.brainRoot, async () => {
      if (!await exists(join(this.directory, "index.sqlite3"))) return undefined;
      await assertTree(this.brainRoot); const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId, true);
      try {
        const stream = db.prepare("SELECT value FROM meta WHERE key='activeObservationStream'").get()?.value; if (typeof stream !== "string") return undefined;
        const row = db.prepare("SELECT * FROM observation_state s WHERE stream_id=? AND state IN ('pending-native-admission','superseded','processing-native-admission') AND NOT EXISTS (SELECT 1 FROM observation_receipts r WHERE r.record_sha=s.record_sha) ORDER BY ordinal LIMIT 1").get(stream);
        if (!row) return undefined;
        const sha256 = String(row.record_sha), recordJson = String(db.prepare("SELECT payload_json FROM records WHERE record_sha=?").get(sha256)?.payload_json), record = this.readRecord(db, sha256);
        if (record.format !== "omni-mcp-app-observation" || record.source.brainId !== this.brainId) throw new Error("Pending App observation belongs to another source owner.");
        const dataJson = await this.readBlob(record.content); const parsed = JSON.parse(dataJson); mcpAppTaskData(parsed.method, parsed.params);
        db.exec("BEGIN IMMEDIATE"); let claim = db.prepare("SELECT disposition,base_frontier_sha FROM observation_claims WHERE record_sha=?").get(sha256);
        if (!claim) {
          if (row.state === "processing-native-admission") throw new Error("App observation lost its durable processing intent.");
          const disposition: NativeAppObservationDisposition = row.state === "superseded" ? "superseded" : "observe";
          db.prepare("INSERT INTO observation_claims VALUES(?,?,NULL)").run(sha256, disposition); claim = { disposition, base_frontier_sha: null };
        }
        if (!["observe", "superseded"].includes(String(claim.disposition))) throw new Error("App observation processing intent changed.");
        db.prepare("UPDATE observation_state SET state='processing-native-admission' WHERE record_sha=?").run(sha256); db.exec("COMMIT");
        return { record, recordJson, recordSha256: sha256, dataJson, disposition: String(claim.disposition) as NativeAppObservationDisposition,
          ...(typeof claim.base_frontier_sha === "string" ? { expectedFrontierSha256: claim.base_frontier_sha } : {}) };
      } catch (error) { try { db.exec("ROLLBACK"); } catch { /* original failure */ } throw error; } finally { db.close(); }
    });
  }
  async bindObservationFrontier(recordSha256: string, frontierSha256: string): Promise<string> {
    if (!SHA.test(frontierSha256)) throw new Error("Invalid native App frontier checksum.");
    return withMcpAppArtifactsLease(this.brainRoot, async () => {
      const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId, true);
      try { const claim = db.prepare("SELECT base_frontier_sha FROM observation_claims WHERE record_sha=?").get(recordSha256);
        if (!claim || db.prepare("SELECT state FROM observation_state WHERE record_sha=?").get(recordSha256)?.state !== "processing-native-admission") throw new Error("App observation is not claimed by this queue.");
        if (typeof claim.base_frontier_sha === "string") return claim.base_frontier_sha;
        db.prepare("UPDATE observation_claims SET base_frontier_sha=? WHERE record_sha=? AND base_frontier_sha IS NULL").run(frontierSha256, recordSha256); return frontierSha256;
      } finally { db.close(); }
    });
  }
  async archiveObservationAudit(head: NativeAppObservationHead | null, engineDirectory: string): Promise<string | undefined> {
    if (!head?.content) return undefined;
    const content = checkedAppObservationContent(head.content, head.recordSha256, head.dataSha256), audit = content.audit;
    return withMcpAppArtifactsLease(this.brainRoot, async () => {
      await this.prepare(); const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId, true);
      try {
        const record = this.readRecord(db, head.recordSha256);
        if (record.format !== "omni-mcp-app-observation" || record.source.brainId !== audit.sourceBrainId || record.content.sha256 !== audit.dataSha256) throw new Error("Native App audit lost its immutable source observation.");
        const dataJson = await this.readBlob(record.content), destination = join(this.directory, "blobs", `${audit.sha256}.data`);
        if (await exists(destination)) await verifyAppObservationAudit(destination, content, dataJson, this.hooks);
        else {
          for (const directory of [engineDirectory, join(engineDirectory, "app-observation-audits")]) {
            const info = await lstat(directory); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Native App audit directory is not its owned regular storage.");
          }
          const source = join(engineDirectory, ...audit.path.split("/")); await verifyAppObservationAudit(source, content, dataJson, this.hooks);
          await this.hooks.checkDisk?.(audit.bytes + 8192); await copyMutableFileIsolated(source, destination, { expectedSha256: audit.sha256 });
          await verifyAppObservationAudit(destination, content, dataJson, this.hooks);
        }
        const prior = db.prepare("SELECT descriptor_json,summary_json FROM observation_audits WHERE sha256=?").get(audit.sha256);
        if (prior && (flatMetadata(JSON.parse(String(prior.descriptor_json))) !== flatMetadata(audit) || flatMetadata(JSON.parse(String(prior.summary_json))) !== flatMetadata(content.summary))) throw new Error("Immutable App audit declaration changed.");
        await this.hooks.checkDisk?.(8192); db.prepare("INSERT OR IGNORE INTO observation_audits VALUES(?,?,?)").run(audit.sha256, JSON.stringify(audit), JSON.stringify(content.summary));
        return audit.sha256;
      } finally { db.close(); }
    });
  }
  async bindObservationArchive(recordSha256: string, archivedSha256: string | undefined): Promise<string | undefined> {
    if (archivedSha256 !== undefined && !SHA.test(archivedSha256)) throw new Error("Invalid archived App audit checksum.");
    return withMcpAppArtifactsLease(this.brainRoot, async () => {
      const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId, true);
      try {
        const prior = db.prepare("SELECT previous_audit_sha FROM observation_archive_claims WHERE record_sha=?").get(recordSha256);
        if (prior) return typeof prior.previous_audit_sha === "string" ? prior.previous_audit_sha : undefined;
        if (!db.prepare("SELECT 1 FROM observation_claims WHERE record_sha=?").get(recordSha256) ||
          archivedSha256 && !db.prepare("SELECT 1 FROM observation_audits WHERE sha256=?").get(archivedSha256)) throw new Error("App prior-audit proof has no durable archive/claim.");
        await this.hooks.checkDisk?.(8192); db.prepare("INSERT INTO observation_archive_claims VALUES(?,?)").run(recordSha256, archivedSha256 ?? null); return archivedSha256;
      } finally { db.close(); }
    });
  }
  async acknowledgeObservation(recordSha256: string, native: NativeAppObservationResult): Promise<McpAppObservationReceipt> {
    return withMcpAppArtifactsLease(this.brainRoot, async () => {
      const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId, true);
      try {
        const record = this.readRecord(db, recordSha256), claim = db.prepare("SELECT disposition,base_frontier_sha FROM observation_claims WHERE record_sha=?").get(recordSha256);
        if (record.format !== "omni-mcp-app-observation" || record.source.brainId !== this.brainId || !claim || typeof claim.base_frontier_sha !== "string") throw new Error("App acknowledgement lost its exact current owner/claim.");
        const checked = checkedAppObservationResult(native, this.brainId, record, recordSha256,
          String(claim.disposition) as NativeAppObservationDisposition, claim.base_frontier_sha);
        if (checked.head.content) {
          const { audit, summary } = checked.head.content, row = db.prepare("SELECT descriptor_json,summary_json FROM observation_audits WHERE sha256=?").get(audit.sha256);
          if (!row || flatMetadata(JSON.parse(String(row.descriptor_json))) !== flatMetadata(audit) || flatMetadata(JSON.parse(String(row.summary_json))) !== flatMetadata(summary) ||
            await fileHash(join(this.directory, "blobs", `${audit.sha256}.data`), this.hooks.signal) !== audit.sha256) throw new Error("App native ACK audit must be durably verified before host acknowledgement.");
        }
        const prior = db.prepare("SELECT receipt_json FROM observation_receipts WHERE record_sha=?").get(recordSha256)?.receipt_json;
        if (typeof prior === "string" && JSON.parse(prior).frontierSha256 !== checked.frontierSha256) throw new Error("A committed App acknowledgement changed.");
        const state = checked.head.state === "observed" ? "admitted" : "superseded";
        if (prior === undefined) {
          if (db.prepare("SELECT state FROM observation_state WHERE record_sha=?").get(recordSha256)?.state !== "processing-native-admission") throw new Error("Dormant App history cannot become a new admission.");
          await this.hooks.checkDisk?.(Buffer.byteLength(JSON.stringify(checked)) * 4 + 8192);
          db.exec("BEGIN IMMEDIATE"); db.prepare("INSERT INTO observation_receipts VALUES(?,?)").run(recordSha256, JSON.stringify(checked));
          db.prepare("UPDATE observation_state SET state=? WHERE record_sha=?").run(state, recordSha256); db.exec("COMMIT");
        }
        return observationReceipt(record, recordSha256, state, checked);
      } catch (error) { try { db.exec("ROLLBACK"); } catch { /* original failure */ } throw error; } finally { db.close(); }
    });
  }
  async nextFollowup(): Promise<McpAppFollowup | undefined> {
    return withMcpAppArtifactsLease(this.brainRoot, async () => {
      if (!await exists(join(this.directory, "index.sqlite3"))) return undefined;
      const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId, true);
      try {
        const stream = db.prepare("SELECT value FROM meta WHERE key='activeObservationStream'").get()?.value; if (typeof stream !== "string") return undefined;
        const row = db.prepare("SELECT s.record_sha,r.payload_json,n.receipt_json,f.turn_id,f.input_sha,f.state AS followup_state,f.error FROM observation_state s JOIN records r ON r.record_sha=s.record_sha JOIN observation_receipts n ON n.record_sha=s.record_sha LEFT JOIN observation_followups f ON f.record_sha=s.record_sha WHERE s.stream_id=? AND s.state='admitted' AND json_extract(r.payload_json,'$.method')='ui/message' AND (f.state IS NULL OR f.state<>'complete') ORDER BY s.ordinal LIMIT 1").get(stream);
        if (!row) return undefined;
        const recordSha256 = String(row.record_sha), record = this.readRecord(db, recordSha256), claim = db.prepare("SELECT disposition,base_frontier_sha FROM observation_claims WHERE record_sha=?").get(recordSha256);
        if (record.format !== "omni-mcp-app-observation" || record.source.brainId !== this.brainId || !claim || typeof claim.base_frontier_sha !== "string") throw new Error("App follow-up has no exact owned native observation.");
        const native = checkedAppObservationResult(JSON.parse(String(row.receipt_json)), this.brainId, record, recordSha256, "observe", claim.base_frontier_sha);
        const capture = this.readRecord(db, record.captureSha256); if (capture.format !== "omni-mcp-app-capture") throw new Error("App follow-up lost its original call capture.");
        return { claim: { record, recordSha256, recordJson: String(row.payload_json), dataJson: await this.readBlob(record.content), disposition: "observe", expectedFrontierSha256: claim.base_frontier_sha },
          capture, captureJson: String(db.prepare("SELECT payload_json FROM records WHERE record_sha=?").get(record.captureSha256)?.payload_json), native,
          ...(typeof row.turn_id === "string" ? { intent: { turnId: row.turn_id, inputSha256: String(row.input_sha), state: String(row.followup_state), ...(typeof row.error === "string" ? { error: row.error } : {}) } } : {}) };
      } finally { db.close(); }
    });
  }
  async reserveFollowup(recordSha256: string, turnId: string, inputSha256: string): Promise<void> {
    if (!ID.test(turnId) || !SHA.test(inputSha256)) throw new Error("Invalid App follow-up intent.");
    await withMcpAppArtifactsLease(this.brainRoot, async () => {
      const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId, true);
      try {
        const record = this.readRecord(db, recordSha256);
        if (record.format !== "omni-mcp-app-observation" || record.method !== "ui/message" || record.source.brainId !== this.brainId ||
          db.prepare("SELECT state FROM observation_state WHERE record_sha=?").get(recordSha256)?.state !== "admitted") throw new Error("Only a genuinely admitted App task can start a follow-up.");
        if (db.prepare("SELECT 1 FROM observation_followups WHERE record_sha=?").get(recordSha256)) throw new Error("This App follow-up already has a dispatch intent; its uncertain effects are not replayed.");
        await this.hooks.checkDisk?.(8192); db.prepare("INSERT INTO observation_followups VALUES(?,?,?,'dispatching',NULL)").run(recordSha256, turnId, inputSha256);
      } finally { db.close(); }
    });
  }
  async settleFollowup(recordSha256: string, state: "complete" | "failed", error?: string): Promise<void> {
    await withMcpAppArtifactsLease(this.brainRoot, async () => {
      const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId, true);
      try {
        // The actual committed stream may reconcile before send() resolves.
        // Never replace that truthful completion with a late failure callback.
        if (db.prepare("SELECT state FROM observation_followups WHERE record_sha=?").get(recordSha256)?.state === "complete") return;
        if (db.prepare("UPDATE observation_followups SET state=?,error=? WHERE record_sha=? AND state='dispatching'").run(state, error ?? null, recordSha256).changes !== 1) throw new Error("App follow-up lost its original dispatch intent.");
      }
      finally { db.close(); }
    });
  }
  /** A real controller completion can reconcile a lost/failed host ACK without
   * submitting the task again or replaying any of its external actions. */
  async reconcileFollowup(message: ChatMessage): Promise<McpAppObservationReceipt | undefined> {
    const provenance = message.inputProvenance === undefined ? undefined : checkedChatInputProvenance(message.inputProvenance);
    if (!provenance || provenance.sourceBrainId !== this.brainId || digest(message.content) !== provenance.inputSha256) return undefined;
    return withMcpAppArtifactsLease(this.brainRoot, async () => {
      const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId, true);
      try {
        const intent = db.prepare("SELECT * FROM observation_followups WHERE record_sha=?").get(provenance.recordSha256);
        if (!intent) return undefined;
        if (intent.turn_id !== message.turnId || intent.input_sha !== provenance.inputSha256) throw new Error("App follow-up completion changed its dispatched input identity.");
        const record = this.readRecord(db, provenance.recordSha256);
        if (record.format !== "omni-mcp-app-observation" || record.id !== provenance.observationId || record.artifactId !== provenance.artifactId || record.content.sha256 !== provenance.dataSha256) throw new Error("App follow-up completion lost its original observation.");
        if (intent.state !== "complete") db.prepare("UPDATE observation_followups SET state='complete',error=NULL WHERE record_sha=?").run(provenance.recordSha256);
        return this.observationReceipt(db, record, provenance.recordSha256, "admitted");
      } finally { db.close(); }
    });
  }
  async summary(artifactId: string): Promise<McpAppArtifactSummary | undefined> {
    return withMcpAppArtifactsLease(this.brainRoot, async () => {
      if (!await exists(join(this.directory, "index.sqlite3"))) return undefined;
      await assertTree(this.brainRoot);
      const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId);
      try {
        const captures: McpAppArtifactSummary["captures"] = [], resources: McpAppArtifactSummary["resources"] = [];
        let bytes = 0;
        for (const row of db.prepare("SELECT record_sha,length(payload_json) AS bytes FROM records WHERE artifact_id=? AND kind IN ('omni-mcp-app-capture','omni-mcp-app-resource') ORDER BY record_sha").iterate(artifactId)) {
          bytes += Number(row.bytes); await this.hooks.admit?.(bytes * 4); const record = this.readRecord(db, String(row.record_sha));
          if (record.format === "omni-mcp-app-capture") captures.push({ sha256: String(row.record_sha), source: record.source, observedAt: record.observedAt, resourceUri: record.resourceUri, remoteToolName: record.tool.remoteName });
          else if (record.format === "omni-mcp-app-resource") resources.push({ sha256: String(row.record_sha), captureSha256: record.captureSha256, source: record.source, observedAt: record.observedAt, content: record.content });
        }
        if (!captures.length) return undefined; const binding = db.prepare("SELECT * FROM bindings WHERE artifact_id=?").get(artifactId);
        const selectedCapture = typeof binding?.capture_sha === "string" ? binding.capture_sha : undefined, selectedResource = typeof binding?.resource_sha === "string" ? binding.resource_sha : undefined;
        if (selectedCapture && !captures.some(record => record.sha256 === selectedCapture) || selectedResource && !resources.some(record => record.sha256 === selectedResource)) throw new Error("Saved App binding lost its inert source artifact.");
        let observationQueue: McpAppArtifactSummary["observationQueue"];
        if (hasObservations(db)) {
          const counts = db.prepare("SELECT SUM(state IN ('pending-native-admission','processing-native-admission')) AS pending,SUM(state='dormant') AS dormant,SUM(state='superseded') AS superseded,SUM(state='admitted') AS admitted FROM observation_state WHERE record_sha IN (SELECT record_sha FROM records WHERE artifact_id=?)").get(artifactId);
          const latest = db.prepare("SELECT s.record_sha,s.state FROM observation_state s JOIN records r ON r.record_sha=s.record_sha WHERE r.artifact_id=? ORDER BY json_extract(r.payload_json,'$.observedAt') DESC,s.ordinal DESC,r.record_sha DESC LIMIT 1").get(artifactId);
          if (latest) { const record = this.readRecord(db, String(latest.record_sha)); if (record.format !== "omni-mcp-app-observation") throw new Error("App observation index lost its typed record.");
            observationQueue = { pending: Number(counts?.pending ?? 0), dormant: Number(counts?.dormant ?? 0), superseded: Number(counts?.superseded ?? 0), admitted: Number(counts?.admitted ?? 0), latest: this.observationReceipt(db, record, String(latest.record_sha), String(latest.state) as McpAppObservationReceipt["state"]) }; }
        }
        return { brainId: this.brainId, artifactId, captures, resources, ...(selectedCapture ? { selectedCapture } : {}), ...(selectedResource ? { selectedResource } : {}),
          needsMappingReview: !selectedCapture || resources.length > 0 && !selectedResource, ...(observationQueue ? { observationQueue } : {}) };
      } finally { db.close(); }
    });
  }
  async load(artifactId: string, selectedCapture?: string): Promise<{ capture: McpAppCapture; sha256: string; input: Record<string, unknown>; result: Record<string, unknown> }> {
    return withMcpAppArtifactsLease(this.brainRoot, async () => {
      await assertTree(this.brainRoot);
      const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId);
      try {
        const sha256 = selectedCapture ?? String(db.prepare("SELECT capture_sha FROM bindings WHERE artifact_id=?").get(artifactId)?.capture_sha ?? "");
        const capture = this.readRecord(db, sha256); if (capture.format !== "omni-mcp-app-capture" || capture.artifactId !== artifactId) throw new Error("The App capture belongs to a different artifact.");
        const input = JSON.parse(await this.readBlob(capture.input)), result = JSON.parse(await this.readBlob(capture.result));
        if (!input || typeof input !== "object" || Array.isArray(input) || !result || typeof result !== "object" || Array.isArray(result)) throw new Error("Saved App call data must be exact JSON objects.");
        return { capture, sha256, input: input as Record<string, unknown>, result: result as Record<string, unknown> };
      } finally { db.close(); }
    });
  }
  async readBlob(reference: McpAppBlob): Promise<string> {
    blob(reference); const path = join(this.directory, "blobs", `${reference.sha256}.data`), info = await lstat(path);
    await this.hooks.admit?.(reference.bytes * 4); if (info.size !== reference.bytes || await fileHash(path, this.hooks.signal) !== reference.sha256) throw new Error("Saved App blob failed its content checksum.");
    const bytes = await readFile(path); if (bytes.byteLength !== reference.bytes || digest(bytes) !== reference.sha256) throw new Error("App bytes changed after admission.");
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  }
  async select(artifactId: string, captureSha256: string, reviewedAt = new Date().toISOString()): Promise<void> {
    await withMcpAppArtifactsLease(this.brainRoot, async () => {
      const db = openDatabase(join(this.directory, "index.sqlite3"), this.brainId, true);
      try { const record = this.readRecord(db, captureSha256); if (record.artifactId !== artifactId || record.format !== "omni-mcp-app-capture") throw new Error("Selected App mapping is not an owned capture.");
        db.prepare("INSERT OR REPLACE INTO bindings VALUES(?,?,NULL,?)").run(artifactId, captureSha256, reviewedAt);
      } finally { db.close(); }
    });
  }
  async projection(artifactId: string): Promise<McpAppViewProjection | undefined> {
    const summary = await this.summary(artifactId); if (!summary) return undefined;
    const candidate = summary.selectedCapture ?? summary.captures[0]!.sha256;
    const { capture } = await this.load(artifactId, candidate);
    return { id: artifactId, brainId: this.brainId, serverId: capture.source.serverId, serverLabel: capture.source.serverLabel, resourceUri: capture.resourceUri,
      toolId: capture.tool.id, remoteToolName: capture.tool.remoteName, observedAt: capture.observedAt, untrusted: true, savedArtifact: true,
      artifactSha256: candidate, originalSource: capture.source, needsMappingReview: summary.needsMappingReview };
  }
}

/** Called on copied inert data only; never restores a request, token or session. */
export async function rebindMcpAppArtifactOwner(brainRoot: string, sourceBrainId: string, targetBrainId: string): Promise<void> {
  const path = join(brainRoot, "mcp-apps", "index.sqlite3"); if (!await exists(path)) return;
  await withMcpAppArtifactsLease(brainRoot, async () => { await assertTree(brainRoot); const db = openDatabase(path, sourceBrainId, true);
    try { db.exec("BEGIN IMMEDIATE"); db.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetBrainId);
      // A copied user review is historical provenance, never a new permission
      // to admit pending source content into another neural generation.
      if (hasObservations(db)) db.exec("UPDATE observation_state SET state='dormant' WHERE state IN ('pending-native-admission','processing-native-admission','admitted')");
      db.exec("DELETE FROM meta WHERE key='activeObservationStream'; COMMIT");
    } catch (error) { try { db.exec("ROLLBACK"); } catch { /* original failure */ } throw error; } finally { db.close(); } });
  await validateMcpAppArtifactStore(brainRoot, targetBrainId);
}
export async function validateMcpAppArtifactStore(brainRoot: string, brainId: string, hooks: McpAppArtifactHooks = {}): Promise<void> {
  const directory = join(brainRoot, "mcp-apps"), path = join(directory, "index.sqlite3"); if (!await exists(path)) return;
  await withMcpAppArtifactsLease(brainRoot, async () => {
    await assertTree(brainRoot);
    const db = openDatabase(path, brainId);
    try {
      for (const row of db.prepare("SELECT record_sha,payload_json,artifact_id,kind FROM records ORDER BY record_sha").iterate()) {
        hooks.signal?.throwIfAborted(); const text = String(row.payload_json); await hooks.admit?.(Buffer.byteLength(text) * 4);
        if (digest(text) !== row.record_sha) throw new Error("App record checksum mismatch.");
        const record = validateMcpAppArtifact(JSON.parse(text)); if (record.artifactId !== row.artifact_id || record.format !== row.kind) throw new Error("App record index changed.");
        for (const reference of record.format === "omni-mcp-app-capture" ? [record.input, record.result] : [record.content]) {
          const blobPath = join(directory, "blobs", `${reference.sha256}.data`);
          if ((await lstat(blobPath)).size !== reference.bytes || await fileHash(blobPath, hooks.signal) !== reference.sha256) throw new Error("App artifact blob checksum mismatch.");
        }
        if (record.format !== "omni-mcp-app-capture" && !db.prepare("SELECT 1 FROM records WHERE record_sha=? AND artifact_id=? AND kind='omni-mcp-app-capture'").get(record.captureSha256, record.artifactId)) throw new Error("An App resource/observation lost its capture ancestry.");
        if (record.format === "omni-mcp-app-observation") {
          const payload = JSON.parse(await new McpAppArtifactStore(brainRoot, brainId, hooks).readBlob(record.content));
          if (payload.method !== record.method) throw new Error("App observation changed its typed method."); mcpAppTaskData(payload.method, payload.params);
          const state = hasObservations(db) ? db.prepare("SELECT * FROM observation_state WHERE record_sha=?").get(sql(row.record_sha)) : undefined;
          if (!state || !["pending-native-admission", "processing-native-admission", "superseded", "dormant", "admitted"].includes(String(state.state)) || state.stream_id !== record.streamId ||
            state.ordinal !== record.ordinal || state.review_id !== record.reviewId || state.content_bytes !== record.content.bytes) throw new Error("App observation queue lost its exact source/review/progress binding.");
          if (record.ordinal === 1 ? record.previousSha256 !== "0".repeat(64) : !db.prepare("SELECT 1 FROM records WHERE record_sha=? AND kind='omni-mcp-app-observation' AND json_extract(payload_json,'$.streamId')=? AND json_extract(payload_json,'$.ordinal')=?").get(record.previousSha256, record.streamId, record.ordinal - 1)) throw new Error("App observation progress hash chain changed.");
          const claim = hasTable(db, "observation_claims") ? db.prepare("SELECT * FROM observation_claims WHERE record_sha=?").get(sql(row.record_sha)) : undefined;
          const receipt = hasTable(db, "observation_receipts") ? db.prepare("SELECT receipt_json FROM observation_receipts WHERE record_sha=?").get(sql(row.record_sha))?.receipt_json : undefined;
          if (claim && (!["observe", "superseded"].includes(String(claim.disposition)) || claim.base_frontier_sha !== null && !SHA.test(String(claim.base_frontier_sha)))) throw new Error("App processing claim changed.");
          if (state.state === "processing-native-admission" && !claim || state.state === "admitted" && typeof receipt !== "string") throw new Error("App queue claims admission without a native receipt.");
          if (typeof receipt === "string") {
            if (!claim || typeof claim.base_frontier_sha !== "string") throw new Error("Native App acknowledgement lost its original request claim.");
            checkedAppObservationResult(JSON.parse(receipt), record.source.brainId, record, String(row.record_sha), String(claim.disposition) as NativeAppObservationDisposition, claim.base_frontier_sha);
          }
        }
      }
      for (const row of db.prepare("SELECT * FROM bindings").iterate()) {
        if (!UUID.test(String(row.artifact_id))) throw new Error("Invalid App binding identity.");
        for (const [value, kind] of [[row.capture_sha, "omni-mcp-app-capture"], [row.resource_sha, "omni-mcp-app-resource"]] as const)
          if (value !== null && !db.prepare("SELECT 1 FROM records WHERE record_sha=? AND artifact_id=? AND kind=?").get(sql(value), sql(row.artifact_id), kind)) throw new Error("App binding refers to unavailable or mismatched history.");
      }
      if (hasObservations(db)) for (const row of db.prepare("SELECT record_sha FROM observation_state").iterate()) if (!db.prepare("SELECT 1 FROM records WHERE record_sha=? AND kind='omni-mcp-app-observation'").get(sql(row.record_sha))) throw new Error("An App queue entry has no immutable observation.");
      for (const name of ["observation_claims", "observation_receipts", "observation_followups"]) if (hasTable(db, name)) for (const row of db.prepare(`SELECT record_sha FROM ${name}`).iterate())
        if (!db.prepare("SELECT 1 FROM records WHERE record_sha=? AND kind='omni-mcp-app-observation'").get(sql(row.record_sha))) throw new Error("App native progress refers to unavailable history.");
      if (hasTable(db, "observation_followups")) for (const row of db.prepare("SELECT * FROM observation_followups").iterate())
        if (!ID.test(String(row.turn_id)) || !SHA.test(String(row.input_sha)) || !["dispatching", "complete", "failed"].includes(String(row.state)) || row.error !== null && typeof row.error !== "string") throw new Error("Invalid retained App follow-up intent.");
      if (hasTable(db, "observation_audits")) for (const row of db.prepare("SELECT * FROM observation_audits").iterate()) {
        const audit = JSON.parse(String(row.descriptor_json)), summary = JSON.parse(String(row.summary_json)), content = checkedAppObservationContent({ audit, summary }, audit.recordSha256, audit.dataSha256);
        if (row.sha256 !== audit.sha256) throw new Error("App archived audit index changed.");
        const recordRow = db.prepare("SELECT payload_json FROM records WHERE record_sha=? AND kind='omni-mcp-app-observation'").get(audit.recordSha256);
        if (typeof recordRow?.payload_json !== "string") throw new Error("App archived audit lost its original observation.");
        const record = validateMcpAppArtifact(JSON.parse(recordRow.payload_json));
        if (record.format !== "omni-mcp-app-observation" || record.source.brainId !== audit.sourceBrainId || record.content.sha256 !== audit.dataSha256) throw new Error("App archived audit changed its original source.");
        const store = new McpAppArtifactStore(brainRoot, brainId, hooks);
        await verifyAppObservationAudit(join(directory, "blobs", `${audit.sha256}.data`), content, await store.readBlob(record.content), hooks);
      }
      if (hasTable(db, "observation_archive_claims")) for (const row of db.prepare("SELECT * FROM observation_archive_claims").iterate())
        if (!db.prepare("SELECT 1 FROM records WHERE record_sha=? AND kind='omni-mcp-app-observation'").get(sql(row.record_sha)) ||
          row.previous_audit_sha !== null && (!SHA.test(String(row.previous_audit_sha)) || !db.prepare("SELECT 1 FROM observation_audits WHERE sha256=?").get(sql(row.previous_audit_sha)))) throw new Error("App previous-audit proof has no retained source bytes.");
      const children = await opendir(directory);
      for await (const entry of children) if (!["index.sqlite3", "index.sqlite3-wal", "index.sqlite3-shm", "blobs"].includes(entry.name) || entry.isSymbolicLink()) throw new Error("Unexpected App artifact storage entry.");
      const blobs = await opendir(join(directory, "blobs"));
      for await (const entry of blobs) {
        if (!/^[a-f0-9]{64}\.data$/.test(entry.name) || !entry.isFile() || entry.isSymbolicLink()) throw new Error("App artifacts contain an executable or unowned payload.");
        if (await fileHash(join(directory, "blobs", entry.name), hooks.signal) !== entry.name.slice(0, -5)) throw new Error("An inert App blob changed.");
      }
    } finally { db.close(); }
  });
}

/** Union immutable data only. Shared ancestry/content dedups; conflicts remain unselected. */
export async function mergeMcpAppArtifacts(sources: Array<{ root: string; brainId: string }>, targetRoot: string, targetBrainId: string,
  hooks: McpAppArtifactHooks = {}): Promise<void> {
  const present: Array<{ root: string; brainId: string }> = [];
  for (const source of sources) if (await exists(join(source.root, "mcp-apps", "index.sqlite3"))) {
    if (resolve(source.root) === resolve(targetRoot)) throw new Error("App artifact merge source cannot be its writable target.");
    await validateMcpAppArtifactStore(source.root, source.brainId, hooks); present.push(source);
  }
  if (!present.length) return;
  const target = new McpAppArtifactStore(targetRoot, targetBrainId, hooks); await target.initialize();
  const roots = [...new Set([targetRoot, ...present.map(source => source.root)].map(value => resolve(value)))].sort();
  const locked = async (index: number, action: () => Promise<void>): Promise<void> => index === roots.length ? action() : withMcpAppArtifactsLease(roots[index]!, () => locked(index + 1, action));
  await locked(0, async () => {
    const db = openDatabase(join(target.directory, "index.sqlite3"), targetBrainId, true);
    try {
      for (const source of present) {
        hooks.signal?.throwIfAborted(); const directory = join(source.root, "mcp-apps"), sourceDb = openDatabase(join(directory, "index.sqlite3"), source.brainId);
        try {
          const blobs = await opendir(join(directory, "blobs"));
          for await (const entry of blobs) {
            hooks.signal?.throwIfAborted(); if (!/^[a-f0-9]{64}\.data$/.test(entry.name) || !entry.isFile() || entry.isSymbolicLink()) throw new Error("Unsafe App merge blob.");
            const path = join(directory, "blobs", entry.name), destination = join(target.directory, "blobs", entry.name), sha = entry.name.slice(0, -5);
            if (await exists(destination)) { if (await fileHash(destination, hooks.signal) !== sha) throw new Error("App merged blob conflict."); }
            else { await hooks.checkDisk?.((await lstat(path)).size); await copyMutableFileIsolated(path, destination, { expectedSha256: sha }); }
          }
          db.exec("BEGIN IMMEDIATE");
          for (const row of sourceDb.prepare("SELECT * FROM records ORDER BY record_sha").iterate()) {
            hooks.signal?.throwIfAborted(); const payload = String(row.payload_json); await hooks.admit?.(Buffer.byteLength(payload) * 3);
            if (digest(payload) !== row.record_sha || validateMcpAppArtifact(JSON.parse(payload)).artifactId !== row.artifact_id) throw new Error("App merge record changed.");
            const old = db.prepare("SELECT payload_json FROM records WHERE record_sha=?").get(sql(row.record_sha))?.payload_json;
            if (old !== undefined && old !== payload) throw new Error("App merge hash collision.");
            db.prepare("INSERT OR IGNORE INTO records VALUES(?,?,?,?)").run(sql(row.record_sha), sql(row.artifact_id), sql(row.kind), payload);
            db.prepare("INSERT OR IGNORE INTO ancestry VALUES(?,?,?)").run(source.brainId, sql(row.artifact_id), sql(row.record_sha));
          }
          for (const row of sourceDb.prepare("SELECT * FROM ancestry").iterate()) db.prepare("INSERT OR IGNORE INTO ancestry VALUES(?,?,?)").run(sql(row.parent_id), sql(row.artifact_id), sql(row.record_sha));
          if (hasObservations(sourceDb)) for (const row of sourceDb.prepare("SELECT * FROM observation_state").iterate()) db.prepare("INSERT OR IGNORE INTO observation_state VALUES(?,?,?,?,?,?)").run(
            sql(row.record_sha), sql(row.stream_id), sql(row.ordinal), row.state === "superseded" ? "superseded" : "dormant", sql(row.content_bytes), sql(row.review_id));
          if (hasTable(sourceDb, "observation_claims")) for (const row of sourceDb.prepare("SELECT * FROM observation_claims").iterate()) {
            const prior = db.prepare("SELECT disposition,base_frontier_sha FROM observation_claims WHERE record_sha=?").get(sql(row.record_sha));
            if (prior && (prior.disposition !== row.disposition || prior.base_frontier_sha !== null && row.base_frontier_sha !== null && prior.base_frontier_sha !== row.base_frontier_sha)) throw new Error("Shared App observation has conflicting processing claims.");
            db.prepare("INSERT INTO observation_claims VALUES(?,?,?) ON CONFLICT(record_sha) DO UPDATE SET base_frontier_sha=COALESCE(observation_claims.base_frontier_sha,excluded.base_frontier_sha)").run(sql(row.record_sha), sql(row.disposition), sql(row.base_frontier_sha));
          }
          if (hasTable(sourceDb, "observation_receipts")) for (const row of sourceDb.prepare("SELECT * FROM observation_receipts").iterate()) {
            const prior = db.prepare("SELECT receipt_json FROM observation_receipts WHERE record_sha=?").get(sql(row.record_sha))?.receipt_json;
            if (typeof prior === "string" && JSON.parse(prior).frontierSha256 !== JSON.parse(String(row.receipt_json)).frontierSha256) throw new Error("Shared App observation has conflicting native acknowledgements.");
            db.prepare("INSERT OR IGNORE INTO observation_receipts VALUES(?,?)").run(sql(row.record_sha), sql(row.receipt_json));
          }
          if (hasTable(sourceDb, "observation_followups")) for (const row of sourceDb.prepare("SELECT * FROM observation_followups").iterate()) {
            const old = db.prepare("SELECT turn_id,input_sha FROM observation_followups WHERE record_sha=?").get(sql(row.record_sha));
            if (old && (old.turn_id !== row.turn_id || old.input_sha !== row.input_sha)) throw new Error("Shared App task has conflicting dispatch intents.");
            db.prepare("INSERT INTO observation_followups VALUES(?,?,?,?,?) ON CONFLICT(record_sha) DO UPDATE SET state=CASE WHEN excluded.state='complete' THEN 'complete' ELSE observation_followups.state END").run(sql(row.record_sha), sql(row.turn_id), sql(row.input_sha), sql(row.state), sql(row.error));
          }
          if (hasTable(sourceDb, "observation_audits")) for (const row of sourceDb.prepare("SELECT * FROM observation_audits").iterate()) {
            const old = db.prepare("SELECT * FROM observation_audits WHERE sha256=?").get(sql(row.sha256));
            if (old && (flatMetadata(JSON.parse(String(old.descriptor_json))) !== flatMetadata(JSON.parse(String(row.descriptor_json))) || flatMetadata(JSON.parse(String(old.summary_json))) !== flatMetadata(JSON.parse(String(row.summary_json))))) throw new Error("Shared App audit has conflicting source metadata.");
            db.prepare("INSERT OR IGNORE INTO observation_audits VALUES(?,?,?)").run(sql(row.sha256), sql(row.descriptor_json), sql(row.summary_json));
          }
          if (hasTable(sourceDb, "observation_archive_claims")) for (const row of sourceDb.prepare("SELECT * FROM observation_archive_claims").iterate()) {
            const old = db.prepare("SELECT previous_audit_sha FROM observation_archive_claims WHERE record_sha=?").get(sql(row.record_sha));
            if (old && old.previous_audit_sha !== row.previous_audit_sha) throw new Error("Shared App processing has a conflicting prior-audit proof.");
            db.prepare("INSERT OR IGNORE INTO observation_archive_claims VALUES(?,?)").run(sql(row.record_sha), sql(row.previous_audit_sha));
          }
          for (const row of sourceDb.prepare("SELECT * FROM bindings").iterate()) {
            const old = db.prepare("SELECT * FROM bindings WHERE artifact_id=?").get(sql(row.artifact_id));
            const capture = !old ? row.capture_sha : old.capture_sha === row.capture_sha ? old.capture_sha : null;
            const resource = !old ? row.resource_sha : old.resource_sha === row.resource_sha ? old.resource_sha : null;
            db.prepare("INSERT OR REPLACE INTO bindings VALUES(?,?,?,?)").run(sql(row.artifact_id), sql(capture), sql(resource), old ? null : sql(row.reviewed_at));
          }
          db.exec("COMMIT");
        } catch (error) { try { db.exec("ROLLBACK"); } catch { /* original failure */ } throw error; } finally { sourceDb.close(); }
      }
      db.exec("BEGIN IMMEDIATE; UPDATE observation_state SET state='dormant' WHERE state IN ('pending-native-admission','processing-native-admission','admitted'); DELETE FROM meta WHERE key='activeObservationStream'; COMMIT");
    } finally { db.close(); }
  });
  await validateMcpAppArtifactStore(targetRoot, targetBrainId, hooks);
}
