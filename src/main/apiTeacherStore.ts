import { createHash } from "node:crypto";
import { lstat, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { ApiTeacherProvider, ApiTeacherTrainingReport, RuntimeJob, RuntimeJobState } from "../shared/types";
import { API_TEACHER_PROVIDERS } from "./apiTeacherProviders";
import type { ApiTeacherLessonModality, ApiTeacherMediaLesson, ApiTeacherMediaCoverage, ApiTeacherResponseEnvelope } from "../shared/apiTeacherMedia";
import { apiTeacherLessonModalities } from "../shared/apiTeacherMedia";
import { checkedApiTeacherMediaCoverage, checkedApiTeacherResponseEnvelope, validateApiTeacherMediaLesson } from "./apiTeacherMedia";

export const API_TEACHER_DIRECTORY = "api-teachers";
export const API_TEACHER_DATABASE = `${API_TEACHER_DIRECTORY}/workspace.sqlite3`;
export type ApiTeacherJobState = "queued" | "running" | "paused" | "review-required" | "stopped" | "complete" | "failed" | "dormant";
export type ApiTeacherItemState = "pending" | "dispatching" | "response-staged" | "response-review" | "response-committed" | "learning" | "complete" | "failed" | "skipped" | "uncertain-review" | "learning-uncertain-review" | "dormant";
export interface ApiTeacherLearningReceipt {
  brainId: string; sourceId: string; contentHash: string; learnedResponseSha256: string; deliveryId: string;
  learnedIdeas: number; learnedConcepts: number; learnedSynapses: number; learnedParameterSteps: number; parametersChanged: boolean;
  mediaCoverage?: ApiTeacherMediaCoverage;
}
export interface ApiTeacherStoredJob {
  format: "omni-api-teacher-job"; version: 1; id: string; brainId: string; originalBrainId: string; originalJobId: string;
  requestId: string; requestSha256: string; provider: ApiTeacherProvider; model: string; maxOutputTokens: number;
  promptCount: number; learned: number; failed: number; skipped: number; nextIndex: number; state: ApiTeacherJobState;
  revision: number; createdAt: string; updatedAt: string; error?: string;
  responseModalities?: ApiTeacherLessonModality[];
}
export interface ApiTeacherStoredItem {
  format: "omni-api-teacher-item"; version: 1; jobId: string; index: number; prompt: string; promptSha256: string;
  deliveryId: string; state: ApiTeacherItemState; attemptId?: string; requestSha256?: string;
  rawResponseSha256?: string; learnedResponseSha256?: string; response?: string; experienceSha256?: string;
  receipt?: ApiTeacherLearningReceipt; error?: string;
  mediaLesson?: ApiTeacherMediaLesson;
  responseEnvelope?: ApiTeacherResponseEnvelope;
}
export interface ApiTeacherLineageRecord {
  format: "omni-api-teacher-lineage"; version: 1; sourceBrainId: string; databaseSha256: string;
  bytes: number; relativePath: string; sourceUpdatedAt: string;
}
export interface ApiTeacherWriteAdmission { (brainId: string, additionalBytes: number, signal?: AbortSignal): Promise<void> }

const ID = /^[A-Za-z0-9][A-Za-z0-9:._-]{0,255}$/;
const SHA = /^[a-f0-9]{64}$/;
const STATES = new Set<ApiTeacherJobState>(["queued", "running", "paused", "review-required", "stopped", "complete", "failed", "dormant"]);
const ITEM_STATES = new Set<ApiTeacherItemState>(["pending", "dispatching", "response-staged", "response-review", "response-committed", "learning", "complete", "failed", "skipped", "uncertain-review", "learning-uncertain-review", "dormant"]);
const codepoint = (a: string, b: string): number => a < b ? -1 : a > b ? 1 : 0;
const record = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === "object" && !Array.isArray(value);
export function apiTeacherCanonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(apiTeacherCanonical).join(",")}]`;
  if (record(value)) return `{${Object.entries(value).sort(([a], [b]) => codepoint(a, b))
    .map(([key, item]) => `${JSON.stringify(key)}:${apiTeacherCanonical(item)}`).join(",")}}`;
  return JSON.stringify(value) ?? "null";
}
export const apiTeacherSha256 = (value: string | Uint8Array): string => createHash("sha256").update(value).digest("hex");
export function apiTeacherId(value: unknown, label = "API teacher identity"): string {
  if (typeof value !== "string" || !ID.test(value)) throw new Error(`${label} is invalid.`);
  return value;
}
function integer(value: unknown, label: string, minimum = 0): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < minimum) throw new Error(`${label} is invalid.`);
  return value;
}
function instant(value: unknown, label: string): string {
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value))) throw new Error(`${label} is invalid.`);
  return value;
}
function exact(value: Record<string, unknown>, fields: readonly string[], optional: readonly string[] = []): void {
  const allowed = new Set([...fields, ...optional]);
  if (fields.some(field => !(field in value)) || Object.keys(value).some(field => !allowed.has(field)))
    throw new Error("API teacher durable record has unknown or missing fields.");
}
export function validateApiTeacherJob(value: unknown, owner: string): ApiTeacherStoredJob {
  if (!record(value)) throw new Error("Saved API teacher job is invalid.");
  exact(value, ["format", "version", "id", "brainId", "originalBrainId", "originalJobId", "requestId", "requestSha256",
    "provider", "model", "maxOutputTokens", "promptCount", "learned", "failed", "skipped", "nextIndex", "state", "revision",
    "createdAt", "updatedAt"], ["error", "responseModalities"]);
  if (value.responseModalities !== undefined) apiTeacherLessonModalities(value.responseModalities);
  if (value.format !== "omni-api-teacher-job" || value.version !== 1 || value.brainId !== owner ||
      !API_TEACHER_PROVIDERS.includes(value.provider as ApiTeacherProvider) || !STATES.has(value.state as ApiTeacherJobState) ||
      typeof value.model !== "string" || !value.model || typeof value.requestSha256 !== "string" || !SHA.test(value.requestSha256) ||
      typeof value.error !== "undefined" && typeof value.error !== "string") throw new Error("Saved API teacher job identity is invalid.");
  for (const [id, label] of [[value.id, "job ID"], [value.brainId, "brain ID"], [value.originalBrainId, "original brain ID"],
    [value.originalJobId, "original job ID"], [value.requestId, "request ID"]] as const) apiTeacherId(id, label);
  for (const [entry, label, minimum] of [[value.maxOutputTokens, "output-token budget", 1], [value.promptCount, "prompt count", 1],
    [value.learned, "learned count", 0], [value.failed, "failed count", 0], [value.skipped, "skipped count", 0],
    [value.nextIndex, "next item cursor", 0], [value.revision, "revision", 1]] as const) integer(entry, label, minimum);
  if (Number(value.learned) + Number(value.failed) + Number(value.skipped) > Number(value.promptCount) ||
      Number(value.nextIndex) > Number(value.promptCount)) throw new Error("Saved API teacher coverage is invalid.");
  instant(value.createdAt, "created timestamp"); instant(value.updatedAt, "updated timestamp");
  return structuredClone(value) as unknown as ApiTeacherStoredJob;
}
export function validateApiTeacherItem(value: unknown, jobId: string, owner?: string): ApiTeacherStoredItem {
  if (!record(value)) throw new Error("Saved API teacher item is invalid.");
  exact(value, ["format", "version", "jobId", "index", "prompt", "promptSha256", "deliveryId", "state"],
    ["attemptId", "requestSha256", "rawResponseSha256", "learnedResponseSha256", "response", "experienceSha256", "receipt", "error", "mediaLesson", "responseEnvelope"]);
  if (value.format !== "omni-api-teacher-item" || value.version !== 1 || value.jobId !== jobId ||
      typeof value.prompt !== "string" || !value.prompt || apiTeacherSha256(value.prompt) !== value.promptSha256 ||
      !ITEM_STATES.has(value.state as ApiTeacherItemState) || typeof value.error !== "undefined" && typeof value.error !== "string")
    throw new Error("Saved API teacher item identity is invalid.");
  apiTeacherId(value.jobId, "item job ID"); apiTeacherId(value.deliveryId, "item delivery ID"); integer(value.index, "item index");
  for (const name of ["attemptId", "requestSha256", "rawResponseSha256", "learnedResponseSha256", "experienceSha256"])
    if (value[name] !== undefined && (typeof value[name] !== "string" || (name === "attemptId" ? !ID.test(value[name]) : !SHA.test(value[name]))))
      throw new Error("Saved API teacher item checksum or attempt identity is invalid.");
  if (value.response !== undefined && typeof value.response !== "string") throw new Error("Saved API teacher response is invalid.");
  if (value.responseEnvelope !== undefined) {
    const envelope = checkedApiTeacherResponseEnvelope(value.responseEnvelope);
    if (!value.attemptId || !value.requestSha256 || value.rawResponseSha256 !== envelope.sha256)
      throw new Error("Teacher received response lost its exact request/byte custody.");
  }
  if (["response-staged", "response-review"].includes(String(value.state)) && !value.responseEnvelope) throw new Error("Staged teacher response has no retained byte envelope.");
  const mediaLesson = value.mediaLesson === undefined ? undefined : validateApiTeacherMediaLesson(value.mediaLesson, value.prompt as string, value.deliveryId as string);
  if (mediaLesson && (value.experienceSha256 !== undefined && value.experienceSha256 !== mediaLesson.manifestSha256 ||
      value.learnedResponseSha256 !== apiTeacherSha256(apiTeacherCanonical(mediaLesson.parts))))
    throw new Error("Saved teacher media payload differs from its exact ordered lesson.");
  if (value.receipt !== undefined) {
    const receipt = value.receipt;
    if (!record(receipt)) throw new Error("Saved API teacher learning receipt is invalid.");
    exact(receipt, ["brainId", "sourceId", "contentHash", "learnedResponseSha256", "deliveryId", "learnedIdeas", "learnedConcepts",
      "learnedSynapses", "learnedParameterSteps", "parametersChanged"], ["mediaCoverage"]);
    if (typeof receipt.brainId !== "string" || owner !== undefined && receipt.brainId !== owner ||
        typeof receipt.sourceId !== "string" || !ID.test(receipt.sourceId) ||
        typeof receipt.contentHash !== "string" || !SHA.test(receipt.contentHash) || receipt.learnedResponseSha256 !== value.learnedResponseSha256 ||
        receipt.contentHash !== value.experienceSha256 || receipt.deliveryId !== value.deliveryId ||
        typeof receipt.parametersChanged !== "boolean") throw new Error("Saved API teacher learning receipt identity is invalid.");
    for (const name of ["learnedIdeas", "learnedConcepts", "learnedSynapses", "learnedParameterSteps"])
      integer(receipt[name], `receipt ${name}`);
    if (mediaLesson) checkedApiTeacherMediaCoverage(receipt.mediaCoverage, mediaLesson);
    else if (receipt.mediaCoverage !== undefined) throw new Error("Text teacher receipts cannot claim media learning.");
  }
  if (["dispatching", "uncertain-review"].includes(String(value.state)) && (!value.attemptId || !value.requestSha256))
    throw new Error("Saved API teacher provider dispatch lost its exact attempt identity.");
  if (["response-committed", "learning", "learning-uncertain-review", "complete"].includes(String(value.state)) &&
      (!value.attemptId || !value.requestSha256 || !value.rawResponseSha256 || !value.learnedResponseSha256 || value.response === undefined))
    throw new Error("Saved API teacher committed response is incomplete.");
  if (["learning", "learning-uncertain-review", "complete"].includes(String(value.state)) && !value.experienceSha256)
    throw new Error("Saved API teacher learning delivery lost its content hash.");
  if (value.state === "complete" && !value.receipt) throw new Error("Saved API teacher completed item lost its learning receipt.");
  return structuredClone(value) as unknown as ApiTeacherStoredItem;
}

export function validateApiTeacherLineage(value: unknown): ApiTeacherLineageRecord {
  if (!record(value)) throw new Error("Saved API teacher lineage record is invalid.");
  exact(value, ["format", "version", "sourceBrainId", "databaseSha256", "bytes", "relativePath", "sourceUpdatedAt"]);
  if (value.format !== "omni-api-teacher-lineage" || value.version !== 1 || typeof value.databaseSha256 !== "string" ||
      !SHA.test(value.databaseSha256) || value.relativePath !== `${API_TEACHER_DIRECTORY}/archives/${value.databaseSha256}.sqlite3`)
    throw new Error("Saved API teacher lineage identity is invalid.");
  apiTeacherId(value.sourceBrainId, "API teacher lineage owner"); integer(value.bytes, "API teacher lineage bytes", 1);
  instant(value.sourceUpdatedAt, "API teacher lineage source timestamp");
  return structuredClone(value) as unknown as ApiTeacherLineageRecord;
}

function payload(value: unknown): { text: string; sha256: string } {
  const text = apiTeacherCanonical(value); return { text, sha256: apiTeacherSha256(text) };
}
function checkedPayload<T>(text: unknown, digest: unknown, validate: (value: unknown) => T): T {
  if (typeof text !== "string" || typeof digest !== "string" || apiTeacherSha256(text) !== digest) throw new Error("API teacher durable payload checksum failed.");
  const value: unknown = JSON.parse(text); if (apiTeacherCanonical(value) !== text) throw new Error("API teacher durable payload is not canonical.");
  return validate(value);
}
const estimate = (...values: string[]): number => values.reduce((sum, value) => sum + Buffer.byteLength(value) * 4, 64 * 1024);

export class ApiTeacherStore {
  private constructor(readonly db: DatabaseSync, readonly root: string, readonly brainId: string,
    private readonly admitWrite: ApiTeacherWriteAdmission) {}
  static async open(root: string, brainId: string, admitWrite: ApiTeacherWriteAdmission,
    options: { create?: boolean; signal?: AbortSignal } = {}): Promise<ApiTeacherStore | undefined> {
    apiTeacherId(brainId, "API teacher brain ID"); const directory = join(root, API_TEACHER_DIRECTORY), path = join(root, API_TEACHER_DATABASE);
    const create = options.create !== false;
    if (create) await mkdir(directory, { recursive: true, mode: 0o700 });
    const info = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (!info && !create) return undefined;
    if (info && (!info.isFile() || info.isSymbolicLink())) throw new Error("API teacher workspace is not an owned regular file.");
    if (!info) await admitWrite(brainId, 128 * 1024, options.signal);
    options.signal?.throwIfAborted();
    const db = new DatabaseSync(path, { readOnly: !create, allowExtension: false });
    try {
      db.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-256; PRAGMA mmap_size=0; PRAGMA busy_timeout=5000;");
      if (create) db.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; " +
        "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); " +
        "CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,request_id TEXT NOT NULL UNIQUE,request_sha TEXT NOT NULL,state TEXT NOT NULL,revision INTEGER NOT NULL,payload TEXT NOT NULL,payload_sha TEXT NOT NULL); " +
        "CREATE TABLE IF NOT EXISTS items(job_id TEXT NOT NULL,ordinal INTEGER NOT NULL,state TEXT NOT NULL,payload TEXT NOT NULL,payload_sha TEXT NOT NULL,PRIMARY KEY(job_id,ordinal)); " +
        "CREATE TABLE IF NOT EXISTS history(sequence INTEGER PRIMARY KEY AUTOINCREMENT,job_id TEXT NOT NULL,event TEXT NOT NULL,payload TEXT NOT NULL,payload_sha TEXT NOT NULL,previous_sha TEXT NOT NULL,row_sha TEXT NOT NULL UNIQUE); " +
        "CREATE TABLE IF NOT EXISTS lineage(database_sha TEXT PRIMARY KEY,source_brain_id TEXT NOT NULL,payload TEXT NOT NULL,payload_sha TEXT NOT NULL);");
      if (create) {
        db.prepare("INSERT OR IGNORE INTO meta VALUES('format','omni-api-teacher-workspace-1')").run();
        db.prepare("INSERT OR IGNORE INTO meta VALUES('brainId',?)").run(brainId);
      }
      if (db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== "omni-api-teacher-workspace-1" ||
          db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) throw new Error("API teacher workspace has another owner or format.");
      const store = new ApiTeacherStore(db, root, brainId, admitWrite); store.validate(); return store;
    } catch (error) { db.close(); throw error; }
  }
  static async validateFile(path: string, brainId: string): Promise<void> {
    apiTeacherId(brainId, "API teacher brain ID");
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink()) throw new Error("API teacher snapshot is not an owned regular file.");
    const db = new DatabaseSync(path, { readOnly: true, allowExtension: false });
    try {
      db.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-256; PRAGMA mmap_size=0; PRAGMA busy_timeout=5000;");
      const store = new ApiTeacherStore(db, join(path, ".."), brainId,
        async () => { throw new Error("Read-only API teacher validation cannot admit a write."); });
      store.validate();
    } finally { db.close(); }
  }
  close(): void { this.db.close(); }
  private transaction<T>(work: () => T): T { this.db.exec("BEGIN IMMEDIATE"); try { const result = work(); this.db.exec("COMMIT"); return result; }
    catch (error) { try { this.db.exec("ROLLBACK"); } catch { /* original error wins */ } throw error; } }
  private appendHistory(jobId: string, event: string, value: unknown): void {
    const encoded = payload(value), prior = this.db.prepare("SELECT row_sha FROM history ORDER BY sequence DESC LIMIT 1").get()?.row_sha ?? "0".repeat(64);
    const sequence = Number(this.db.prepare("SELECT COALESCE(MAX(sequence),0)+1 AS value FROM history").get()?.value ?? 1);
    const rowSha = apiTeacherSha256(apiTeacherCanonical({ sequence, jobId, event, payloadSha256: encoded.sha256, previousSha256: prior }));
    this.db.prepare("INSERT INTO history VALUES(?,?,?,?,?,?,?)").run(sequence, jobId, event, encoded.text, encoded.sha256, prior, rowSha);
  }
  validate(): void {
    if (this.db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("API teacher workspace integrity failed.");
    const allowed = new Set(["meta", "jobs", "items", "history", "lineage"]);
    for (const row of this.db.prepare("SELECT name,type FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").iterate())
      if (row.type !== "table" || !allowed.has(String(row.name))) throw new Error("API teacher workspace contains unknown schema objects.");
    const schema = {
      meta: [["key", "TEXT"], ["value", "TEXT"]],
      jobs: [["id", "TEXT"], ["request_id", "TEXT"], ["request_sha", "TEXT"], ["state", "TEXT"],
        ["revision", "INTEGER"], ["payload", "TEXT"], ["payload_sha", "TEXT"]],
      items: [["job_id", "TEXT"], ["ordinal", "INTEGER"], ["state", "TEXT"], ["payload", "TEXT"], ["payload_sha", "TEXT"]],
      history: [["sequence", "INTEGER"], ["job_id", "TEXT"], ["event", "TEXT"], ["payload", "TEXT"],
        ["payload_sha", "TEXT"], ["previous_sha", "TEXT"], ["row_sha", "TEXT"]],
      lineage: [["database_sha", "TEXT"], ["source_brain_id", "TEXT"], ["payload", "TEXT"], ["payload_sha", "TEXT"]]
    } as const;
    for (const [table, expected] of Object.entries(schema)) {
      const actual = this.db.prepare(`PRAGMA table_info('${table}')`).all();
      if (actual.length !== expected.length || expected.some(([name, type], index) =>
        actual[index]?.name !== name || String(actual[index]?.type).toUpperCase() !== type))
        throw new Error("API teacher workspace schema differs from the credential-free durable format.");
    }
    const metadata = this.db.prepare("SELECT key,value FROM meta ORDER BY key").all();
    if (metadata.length !== 2 || metadata[0]?.key !== "brainId" || metadata[0]?.value !== this.brainId ||
        metadata[1]?.key !== "format" || metadata[1]?.value !== "omni-api-teacher-workspace-1")
      throw new Error("API teacher workspace metadata is not the exact credential-free ownership envelope.");
    for (const row of this.db.prepare("SELECT id,request_id,request_sha,state,revision,payload,payload_sha FROM jobs").iterate()) {
      const job = checkedPayload(row.payload, row.payload_sha, value => validateApiTeacherJob(value, this.brainId));
      if (row.id !== job.id || row.request_id !== job.requestId || row.request_sha !== job.requestSha256 ||
          row.state !== job.state || Number(row.revision) !== job.revision)
        throw new Error("API teacher job index differs from its authenticated payload.");
    }
    for (const row of this.db.prepare("SELECT job_id,ordinal,state,payload,payload_sha FROM items").iterate()) {
      const itemJob = this.job(String(row.job_id)), itemOwner = itemJob.originalBrainId;
      const item = checkedPayload(row.payload, row.payload_sha, value => validateApiTeacherItem(value, String(row.job_id), itemOwner));
      if (itemJob.responseModalities?.some(kind => kind !== "text") && ["response-committed", "learning", "learning-uncertain-review", "complete"].includes(item.state) &&
          (!item.mediaLesson || item.mediaLesson.provider !== itemJob.provider || item.mediaLesson.model !== itemJob.model ||
           apiTeacherCanonical(item.mediaLesson.responseModalities) !== apiTeacherCanonical(itemJob.responseModalities)))
        throw new Error("Saved teacher media row differs from its original requested modalities/provider.");
      if (row.job_id !== item.jobId || Number(row.ordinal) !== item.index || row.state !== item.state ||
          !this.db.prepare("SELECT 1 FROM jobs WHERE id=?").get(item.jobId))
        throw new Error("API teacher item index differs from its authenticated job payload.");
    }
    let previous = "0".repeat(64), sequence = 0;
    for (const row of this.db.prepare("SELECT * FROM history ORDER BY sequence").iterate()) {
      if (Number(row.sequence) !== sequence + 1 || row.previous_sha !== previous || typeof row.payload !== "string" ||
          apiTeacherSha256(row.payload) !== row.payload_sha || apiTeacherCanonical(JSON.parse(row.payload)) !== row.payload ||
          apiTeacherSha256(apiTeacherCanonical({ sequence: Number(row.sequence), jobId: String(row.job_id), event: String(row.event),
            payloadSha256: String(row.payload_sha), previousSha256: String(row.previous_sha) })) !== row.row_sha)
        throw new Error("API teacher history chain failed.");
      const historical: unknown = JSON.parse(row.payload);
      if (record(historical) && historical.format === "omni-api-teacher-job") validateApiTeacherJob(historical, this.brainId);
      else if (record(historical) && historical.format === "omni-api-teacher-item")
        validateApiTeacherItem(historical, String(row.job_id), this.job(String(row.job_id)).originalBrainId);
      else throw new Error("API teacher history contains an untyped operational payload.");
      sequence += 1; previous = String(row.row_sha);
    }
    for (const row of this.db.prepare("SELECT database_sha,source_brain_id,payload,payload_sha FROM lineage").iterate()) {
      const lineage = checkedPayload(row.payload, row.payload_sha, validateApiTeacherLineage);
      if (row.database_sha !== lineage.databaseSha256 || row.source_brain_id !== lineage.sourceBrainId)
        throw new Error("API teacher lineage index differs from its authenticated payload.");
    }
  }
  job(id: string): ApiTeacherStoredJob {
    const row = this.db.prepare("SELECT payload,payload_sha FROM jobs WHERE id=?").get(apiTeacherId(id, "API teacher job ID"));
    if (!row) throw new Error("API teacher job was not found.");
    return checkedPayload(row.payload, row.payload_sha, value => validateApiTeacherJob(value, this.brainId));
  }
  jobs(): ApiTeacherStoredJob[] { return this.db.prepare("SELECT id FROM jobs ORDER BY rowid DESC").all().map(row => this.job(String(row.id))); }
  jobByRequest(requestId: string): ApiTeacherStoredJob | undefined {
    const row = this.db.prepare("SELECT id FROM jobs WHERE request_id=?").get(apiTeacherId(requestId, "API teacher request ID"));
    return row ? this.job(String(row.id)) : undefined;
  }
  item(jobId: string, index: number): ApiTeacherStoredItem {
    const row = this.db.prepare("SELECT payload,payload_sha FROM items WHERE job_id=? AND ordinal=?").get(apiTeacherId(jobId), integer(index, "item index"));
    if (!row) throw new Error("API teacher item was not found.");
    return checkedPayload(row.payload, row.payload_sha, value => validateApiTeacherItem(value, jobId, this.job(jobId).originalBrainId));
  }
  items(jobId: string): ApiTeacherStoredItem[] {
    return this.db.prepare("SELECT ordinal FROM items WHERE job_id=? ORDER BY ordinal").all(apiTeacherId(jobId)).map(row => this.item(jobId, Number(row.ordinal)));
  }
  /** Current and every authenticated historical reference, not just today's cursor. */
  mediaLessons(): Array<{ lesson: ApiTeacherMediaLesson; prompt: string }> {
    const found = new Map<string, { lesson: ApiTeacherMediaLesson; prompt: string }>();
    for (const table of ["items", "history"] as const) {
      for (const row of this.db.prepare(`SELECT payload,payload_sha FROM ${table}`).iterate()) {
        const item = checkedPayload(row.payload, row.payload_sha, value => value);
        if (!record(item) || item.format !== "omni-api-teacher-item" || item.mediaLesson === undefined) continue;
        const lesson = validateApiTeacherMediaLesson(item.mediaLesson, String(item.prompt), String(item.deliveryId));
        found.set(lesson.manifestSha256, { lesson, prompt: String(item.prompt) });
      }
    }
    return [...found.values()];
  }
  receivedEnvelopes(): ApiTeacherResponseEnvelope[] {
    const found = new Map<string, ApiTeacherResponseEnvelope>();
    for (const table of ["items", "history"] as const) for (const row of this.db.prepare(`SELECT payload,payload_sha FROM ${table}`).iterate()) {
      const item = checkedPayload(row.payload, row.payload_sha, value => value);
      if (!record(item) || item.format !== "omni-api-teacher-item" || item.responseEnvelope === undefined) continue;
      const envelope = checkedApiTeacherResponseEnvelope(item.responseEnvelope); found.set(envelope.sha256, envelope);
    }
    return [...found.values()];
  }
  lineages(): ApiTeacherLineageRecord[] {
    return this.db.prepare("SELECT payload,payload_sha FROM lineage ORDER BY database_sha").all()
      .map(row => checkedPayload(row.payload, row.payload_sha, validateApiTeacherLineage));
  }
  async addLineage(lineage: ApiTeacherLineageRecord, signal?: AbortSignal): Promise<void> {
    validateApiTeacherLineage(lineage); const encoded = payload(lineage);
    await this.admitWrite(this.brainId, estimate(encoded.text), signal); signal?.throwIfAborted();
    const current = this.db.prepare("SELECT payload,payload_sha FROM lineage WHERE database_sha=?").get(lineage.databaseSha256);
    if (current) {
      if (current.payload !== encoded.text || current.payload_sha !== encoded.sha256)
        throw new Error("API teacher lineage checksum collided with different custody metadata.");
      return;
    }
    this.transaction(() => this.db.prepare("INSERT INTO lineage VALUES(?,?,?,?)")
      .run(lineage.databaseSha256, lineage.sourceBrainId, encoded.text, encoded.sha256));
  }
  async importDormant(lineages: readonly ApiTeacherLineageRecord[], bundles: readonly {
    job: ApiTeacherStoredJob; items: readonly ApiTeacherStoredItem[];
  }[], signal?: AbortSignal): Promise<void> {
    const encodedLineages = lineages.map(lineage => { validateApiTeacherLineage(lineage); return { lineage, encoded: payload(lineage) }; });
    const encodedBundles = bundles.map(({ job, items }) => {
      validateApiTeacherJob(job, this.brainId);
      if (job.state !== "dormant" || items.length !== job.promptCount ||
          items.some((item, index) => item.jobId !== job.id || item.index !== index || item.state !== "dormant"))
        throw new Error("Imported API teacher history must be a complete dormant projection.");
      return { job, items, encodedJob: payload(job), encodedItems: items.map(item => {
        validateApiTeacherItem(item, job.id, job.originalBrainId); return payload(item);
      }) };
    });
    await this.admitWrite(this.brainId, estimate(...encodedLineages.map(value => value.encoded.text),
      ...encodedBundles.flatMap(value => [value.encodedJob.text, ...value.encodedItems.map(item => item.text)])), signal);
    signal?.throwIfAborted();
    this.transaction(() => {
      for (const { lineage, encoded } of encodedLineages) {
        const current = this.db.prepare("SELECT payload,payload_sha FROM lineage WHERE database_sha=?").get(lineage.databaseSha256);
        if (current) {
          if (current.payload !== encoded.text || current.payload_sha !== encoded.sha256)
            throw new Error("API teacher lineage checksum collided with different custody metadata.");
        } else this.db.prepare("INSERT INTO lineage VALUES(?,?,?,?)")
          .run(lineage.databaseSha256, lineage.sourceBrainId, encoded.text, encoded.sha256);
      }
      for (const value of encodedBundles) {
        const current = this.db.prepare("SELECT payload,payload_sha FROM jobs WHERE id=?").get(value.job.id);
        if (current) {
          if (current.payload !== value.encodedJob.text || current.payload_sha !== value.encodedJob.sha256)
            throw new Error("Dormant API teacher job identity collided with different history.");
          for (let index = 0; index < value.items.length; index += 1) {
            const item = this.db.prepare("SELECT payload,payload_sha FROM items WHERE job_id=? AND ordinal=?").get(value.job.id, index);
            if (item?.payload !== value.encodedItems[index]!.text || item?.payload_sha !== value.encodedItems[index]!.sha256)
              throw new Error("Dormant API teacher item identity collided with different history.");
          }
          continue;
        }
        this.db.prepare("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)").run(value.job.id, value.job.requestId,
          value.job.requestSha256, value.job.state, value.job.revision, value.encodedJob.text, value.encodedJob.sha256);
        const insert = this.db.prepare("INSERT INTO items VALUES(?,?,?,?,?)");
        value.encodedItems.forEach((item, index) => insert.run(value.job.id, index, value.items[index]!.state, item.text, item.sha256));
        this.appendHistory(value.job.id, "inherited-dormant-job", value.job);
        value.items.forEach(item => this.appendHistory(value.job.id, "inherited-dormant-item", item));
      }
    });
  }
  async createJob(job: ApiTeacherStoredJob, items: ApiTeacherStoredItem[], signal?: AbortSignal): Promise<{ job: ApiTeacherStoredJob; created: boolean }> {
    validateApiTeacherJob(job, this.brainId); if (items.length !== job.promptCount || items.some((item, index) => item.index !== index || item.jobId !== job.id))
      throw new Error("API teacher job items do not close to its complete prompt manifest.");
    const prior = this.jobByRequest(job.requestId);
    if (prior) { if (prior.requestSha256 !== job.requestSha256) throw new Error("API teacher retry changed its exact request."); return { job: prior, created: false }; }
    const encodedJob = payload(job), encodedItems = items.map(item => { validateApiTeacherItem(item, job.id, job.originalBrainId); return payload(item); });
    await this.admitWrite(this.brainId, estimate(encodedJob.text, ...encodedItems.map(item => item.text)), signal); signal?.throwIfAborted();
    let result = job, created = false;
    this.transaction(() => {
      const concurrent = this.jobByRequest(job.requestId);
      if (concurrent) {
        if (concurrent.requestSha256 !== job.requestSha256) throw new Error("API teacher retry changed its exact request.");
        result = concurrent; return;
      }
      if (this.db.prepare("SELECT 1 FROM jobs WHERE state IN ('queued','running') LIMIT 1").get())
        throw new Error("This brain already has an active API teacher job.");
      this.db.prepare("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)").run(job.id, job.requestId, job.requestSha256, job.state, job.revision, encodedJob.text, encodedJob.sha256);
      const insert = this.db.prepare("INSERT INTO items VALUES(?,?,?,?,?)");
      encodedItems.forEach((item, index) => insert.run(job.id, index, items[index]!.state, item.text, item.sha256));
      this.appendHistory(job.id, "created", job); created = true;
    }); return { job: result, created };
  }
  async updateJob(id: string, patch: Partial<ApiTeacherStoredJob>, event: string, signal?: AbortSignal): Promise<ApiTeacherStoredJob> {
    const old = this.job(id);
    if (["format", "version", "id", "brainId", "originalBrainId", "originalJobId", "requestId", "requestSha256", "provider", "model",
      "maxOutputTokens", "promptCount", "createdAt", "responseModalities"].some(key => key in patch)) throw new Error("API teacher source/request identity is immutable.");
    const next = { ...old, ...patch, revision: old.revision + 1, updatedAt: new Date().toISOString() } as ApiTeacherStoredJob;
    for (const key of Object.keys(next) as Array<keyof ApiTeacherStoredJob>) if (next[key] === undefined) delete next[key];
    validateApiTeacherJob(next, this.brainId); const encoded = payload(next);
    await this.admitWrite(this.brainId, estimate(encoded.text), signal); signal?.throwIfAborted();
    this.transaction(() => { this.db.prepare("UPDATE jobs SET state=?,revision=?,payload=?,payload_sha=? WHERE id=? AND revision=?")
      .run(next.state, next.revision, encoded.text, encoded.sha256, id, old.revision);
      if (this.db.prepare("SELECT changes() AS value").get()?.value !== 1) throw new Error("API teacher job changed concurrently.");
      this.appendHistory(id, event, next); }); return next;
  }
  async updateItem(jobId: string, index: number, patch: Partial<ApiTeacherStoredItem>, event: string,
    signal?: AbortSignal): Promise<ApiTeacherStoredItem> {
    const old = this.item(jobId, index);
    if (["format", "version", "jobId", "index", "prompt", "promptSha256", "deliveryId"].some(key => key in patch))
      throw new Error("API teacher item source/delivery identity is immutable.");
    if (old.mediaLesson && patch.mediaLesson !== undefined && apiTeacherCanonical(old.mediaLesson) !== apiTeacherCanonical(patch.mediaLesson))
      throw new Error("Committed teacher media lesson cannot be rewritten during retry.");
    const next = { ...old, ...patch } as ApiTeacherStoredItem;
    for (const key of Object.keys(next) as Array<keyof ApiTeacherStoredItem>) if (next[key] === undefined) delete next[key];
    validateApiTeacherItem(next, jobId); const encoded = payload(next);
    const job = this.job(jobId);
    if (job.responseModalities?.some(kind => kind !== "text") && ["response-committed", "learning", "learning-uncertain-review", "complete"].includes(next.state) &&
        (!next.mediaLesson || next.mediaLesson.provider !== job.provider || next.mediaLesson.model !== job.model ||
         apiTeacherCanonical(next.mediaLesson.responseModalities) !== apiTeacherCanonical(job.responseModalities)))
      throw new Error("Teacher media state does not close to its original requested provider/modalities.");
    await this.admitWrite(this.brainId, estimate(encoded.text), signal); signal?.throwIfAborted();
    this.transaction(() => { this.db.prepare("UPDATE items SET state=?,payload=?,payload_sha=? WHERE job_id=? AND ordinal=?")
      .run(next.state, encoded.text, encoded.sha256, jobId, index); this.appendHistory(jobId, event, next); }); return next;
  }
  async recover(signal?: AbortSignal): Promise<void> {
    for (const job of this.jobs()) {
      if (!["queued", "running"].includes(job.state)) continue;
      let uncertain = false, learningUncertain = false;
      for (const item of this.items(job.id)) {
        if (item.state === "response-review") uncertain = true;
        if (item.state === "dispatching") { uncertain = true; await this.updateItem(job.id, item.index,
          { state: "uncertain-review", error: "Provider dispatch outcome is unknown after restart; it was not repeated." }, "recovered-uncertain-provider", signal); }
        else if (item.state === "learning") { uncertain = true; learningUncertain = true; await this.updateItem(job.id, item.index,
          { state: "learning-uncertain-review", error: "Neural learning acknowledgement was interrupted. It was not replayed; explicit review is required." }, "recovered-learning-uncertain", signal); }
      }
      await this.updateJob(job.id, { state: uncertain ? "review-required" : "paused",
        error: uncertain ? learningUncertain
          ? "Review the uncertain neural-learning acknowledgement; no neural update was replayed."
          : "Review the uncertain provider request; no billed request was repeated."
          : "Paused after restart with committed cursor retained." }, "recovered", signal);
    }
  }
  toRuntimeJob(job: ApiTeacherStoredJob): RuntimeJob {
    const state: RuntimeJobState = job.state === "queued" ? "queued" : job.state === "running" ? "running" :
      job.state === "complete" ? "complete" : job.state === "failed" ? "failed" : "cancelled";
    const items = this.items(job.id), uncertainItemIndexes = items.filter(item =>
        item.state === "uncertain-review" || item.state === "learning-uncertain-review" || item.state === "response-review" || item.state === "dispatching" ||
      item.state === "learning" && job.state !== "running")
      .map(item => item.index);
    const report: ApiTeacherTrainingReport = { provider: job.provider, model: job.model, requested: job.promptCount,
      learned: job.learned, failed: job.failed, skipped: job.skipped, requestHashes: items.flatMap(item => item.requestSha256 ? [item.requestSha256] : []),
      responseHashes: items.flatMap(item => item.learnedResponseSha256 ? [item.learnedResponseSha256] : []),
      rawCredentialsStoredInBrain: false, hiddenBehavioralPromptUsed: false, durableState: job.state, nextIndex: job.nextIndex,
      uncertainItemIndexes, resumable: ["paused", "failed"].includes(job.state) && uncertainItemIndexes.length === 0 };
    return { id: job.id, brainId: job.brainId, kind: "teacher", state,
      progress: Math.min(1, (job.learned + job.failed + job.skipped) / job.promptCount),
      label: job.state === "review-required" ? "API teacher request needs review" : job.state === "paused" ? "API teacher learning paused" :
        job.state === "dormant" ? "Copied API teacher history · start a new reviewed job to learn" : job.state === "stopped" ? "API teacher learning stopped" :
          job.state === "complete" ? `Learned ${job.learned}/${job.promptCount} API teacher trajectories` :
            job.state === "failed" ? "API teacher learning failed" : `API teacher ${job.nextIndex + 1}/${job.promptCount}`,
      createdAt: job.createdAt, updatedAt: job.updatedAt, ...(job.error ? { error: job.error } : {}), output: report };
  }
}

/** Validate an isolated SQLite snapshot before admitting it to portable custody. */
export async function validateApiTeacherDatabaseFile(path: string, owner: string): Promise<void> {
  await ApiTeacherStore.validateFile(path, owner);
}
