import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, mkdir, open, opendir, readFile, readdir, rename, unlink } from "node:fs/promises";
import { join } from "node:path";
import { StringDecoder } from "node:string_decoder";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { DatabaseSync } from "node:sqlite";
import type { ActionEvent, ToolExecutionResult, ToolInvocation } from "../shared/types";
import { sqliteFilename } from "./sqliteFilename";

export interface CompletedActionEvidenceReceipt {
  brainId: string; evidenceId: string; executionId: string; evidenceSha256: string;
  committed: true; processed: boolean; duplicate: boolean;
  sourceIdentity?: string; commitSequence?: number;
  webImage?: import("../shared/webImages").WebImageLearningReceipt;
}
export interface CompletedActionEvidenceJob {
  format: "omni-action-result-learning-job-v1";
  brainId: string; evidenceId: string; executionId: string; evidenceSha256: string;
  state: "pending" | "complete";
  createdAt: string; updatedAt: string; attempts: number;
  provenance: Record<string, unknown>;
  route?: { eventId: string; utterance: string; toolId: string; action: string; arguments: Record<string, unknown> };
  routeComplete: boolean;
  receipt?: CompletedActionEvidenceReceipt;
  error?: string;
  /** Learning-only continuation of a retained native cursor, never an effect/grant. */
  mergeContinuation?: { pendingId: string; originalEvidenceId: string; parentId: string };
}

const digest = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");
const hex = /^[a-f0-9]{64}$/;
const uuid = /^[a-f0-9-]{36}$/i;
const CHUNK = 64 * 1024;

function effectArguments(value: Record<string, unknown>, toolId: string): string {
  const normalized = { ...value };
  if (toolId === "modality.imagine") { delete normalized.neuralActionId; delete normalized.chatTurnId; }
  const canonical = (item: unknown): string => Array.isArray(item) ? `[${item.map(canonical).join(",")}]` : item && typeof item === "object"
    ? `{${Object.keys(item).filter(key => (item as Record<string, unknown>)[key] !== undefined).sort().map(key => `${JSON.stringify(key)}:${canonical((item as Record<string, unknown>)[key])}`).join(",")}}`
    : JSON.stringify(item);
  return digest(canonical(normalized));
}

export function actualTerminalToolOutcome(value: ToolExecutionResult | undefined): value is ToolExecutionResult {
  return Boolean(value && typeof value.finishedAt === "string" && Number.isFinite(Date.parse(value.finishedAt)) &&
    (value.state === "complete" || value.state === "failed" && value.dispatchStarted === true));
}

async function readSavedEvidenceJob(path: string): Promise<CompletedActionEvidenceJob> {
  const info = await lstat(path), reserve = 64 * 1024 * 1024, need = info.size * 12;
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Saved evidence manifest is not a regular owned file.");
  if (getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed - reserve < need || freemem() - reserve < need) {
    throw new Error("Saved evidence metadata awaits physical transfer RAM.");
  }
  return validateActionEvidenceJob(JSON.parse(await readFile(path, "utf8")));
}

export async function actionEvidenceFileHash(path: string): Promise<string> {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Evidence must be a regular owned file.");
  const sha = createHash("sha256");
  for await (const chunk of createReadStream(path, { highWaterMark: CHUNK })) sha.update(chunk);
  return sha.digest("hex");
}

async function evidenceHeader(path: string, admit: (bytes: number) => Promise<void>): Promise<Record<string, unknown>> {
  const decoder = new StringDecoder("utf8"); let prefix = "";
  let position = 0, depth = 0, quotedValue = false, escaped = false, expectingKey = false,
    keyStart = -1, separator = -1;
  for await (const bytes of createReadStream(path, { highWaterMark: CHUNK })) {
    await admit((prefix.length + (bytes as Buffer).length) * 8);
    prefix += decoder.write(bytes as Buffer);
    for (; position < prefix.length; position += 1) {
      const character = prefix[position]!;
      if (quotedValue) {
        if (escaped) { escaped = false; continue; }
        if (character === "\\") { escaped = true; continue; }
        if (character !== '"') continue;
        quotedValue = false;
        if (keyStart >= 0) {
          const key = JSON.parse(prefix.slice(keyStart, position + 1)) as string;
          keyStart = -1; expectingKey = false;
          if (key === "outputPresent") return JSON.parse(`${prefix.slice(0, separator)}}`) as Record<string, unknown>;
        }
        continue;
      }
      if (character === '"') { quotedValue = true; keyStart = depth === 1 && expectingKey ? position : -1; }
      else if (character === "{" || character === "[") { depth += 1; if (depth === 1) expectingKey = true; }
      else if (character === "}" || character === "]") depth -= 1;
      else if (character === "," && depth === 1) { expectingKey = true; separator = position; }
    }
  }
  throw new Error("Evidence has no complete actual-outcome header.");
}

function* quoted(text: string): Generator<string> {
  yield '"';
  let chunk = "";
  for (const character of text) {
    const point = character.codePointAt(0)!;
    chunk += character === '"' ? '\\"' : character === "\\" ? "\\\\" :
      point < 0x20 || (point >= 0xD800 && point <= 0xDFFF) ? `\\u${point.toString(16).padStart(4, "0")}` : character;
    if (chunk.length >= CHUNK) { yield chunk; chunk = ""; }
  }
  if (chunk) yield chunk;
  yield '"';
}

/** Bounded JSON wire chunks; never stringify an entire result into RAM. */
export function* actionEvidenceJsonChunks(value: unknown, parents = new Set<object>()): Generator<string> {
  if (typeof value === "string") { yield* quoted(value); return; }
  if (value === null || typeof value === "boolean") { yield String(value); return; }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("Completed evidence contains a nonfinite value.");
    yield JSON.stringify(value); return;
  }
  if (typeof value !== "object" || parents.has(value)) throw new Error("Completed evidence must be noncyclic explicit JSON data.");
  if (!Array.isArray(value) && ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new Error("Completed evidence cannot invoke implicit object conversion.");
  if (Object.getOwnPropertySymbols(value).length) throw new Error("Completed evidence contains non-JSON keys.");
  parents.add(value);
  try {
    yield Array.isArray(value) ? "[" : "{";
    let count = 0;
    if (Array.isArray(value)) {
      for (let index = 0; index < value.length; index += 1) {
        const field = Object.getOwnPropertyDescriptor(value, String(index));
        if (!field || !("value" in field)) throw new Error("Completed evidence has a sparse/accessor array.");
        if (count++) yield ",";
        yield* actionEvidenceJsonChunks(field.value, parents);
      }
    } else {
      for (const key in value) {
        if (!Object.hasOwn(value, key)) continue;
        const field = Object.getOwnPropertyDescriptor(value, key)!;
        if (!("value" in field)) throw new Error("Completed evidence cannot evaluate accessors.");
        if (field.value === undefined) continue; // an absent optional JSON field, not invented data
        if (count++) yield ",";
        yield* quoted(key); yield ":";
        yield* actionEvidenceJsonChunks(field.value, parents);
      }
    }
    yield Array.isArray(value) ? "]" : "}";
  } finally { parents.delete(value); }
}

export function validateActionEvidenceJob(value: unknown, brainId?: string): CompletedActionEvidenceJob {
  const job = value as CompletedActionEvidenceJob;
  if (!job || job.format !== "omni-action-result-learning-job-v1" ||
      typeof job.brainId !== "string" || (brainId !== undefined && job.brainId !== brainId) ||
      !hex.test(job.evidenceId) || !uuid.test(job.executionId) || !hex.test(job.evidenceSha256) ||
      !["pending", "complete"].includes(job.state) || !job.provenance || typeof job.provenance !== "object" ||
      !Number.isSafeInteger(job.attempts) || job.attempts < 0 || typeof job.routeComplete !== "boolean") {
    throw new Error("Invalid durable completed-action evidence job.");
  }
  if (job.mergeContinuation && (!hex.test(job.mergeContinuation.pendingId) || !hex.test(job.mergeContinuation.originalEvidenceId) ||
    !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$/.test(job.mergeContinuation.parentId) ||
    Object.keys(job.mergeContinuation).sort().join(",") !== "originalEvidenceId,parentId,pendingId")) {
    throw new Error("Invalid merged action-result learning continuation.");
  }
  if (job.receipt) validateActionEvidenceReceipt(job.receipt, job);
  if (job.state === "complete" && (!job.receipt || !job.routeComplete)) throw new Error("Completed evidence job lacks exact neural/route receipts.");
  return job;
}

export { readSavedEvidenceJob as readSavedActionEvidenceJob };

/** Recover an immutable captured outcome without modifying its source queue. */
export async function readUnmanifestedActionEvidence(path: string, brainId: string, evidenceId: string,
  admit: (bytes: number) => Promise<void>): Promise<CompletedActionEvidenceJob> {
  const header = await evidenceHeader(path, admit);
  if (header.format !== "omni-completed-action-result-v1" || header.evidenceId !== evidenceId || !uuid.test(String(header.executionId)) ||
    typeof header.completedAt !== "string" || !Number.isFinite(Date.parse(header.completedAt))) throw new Error("Orphan evidence has invalid identity.");
  const { executionId, evidenceId: _id, startedAt: _started, arguments: _arguments, learningRoute, ...provenance } = header;
  const route = learningRoute as CompletedActionEvidenceJob["route"];
  return validateActionEvidenceJob({ format: "omni-action-result-learning-job-v1", brainId, evidenceId,
    executionId: String(executionId), evidenceSha256: await actionEvidenceFileHash(path), state: "pending",
    createdAt: header.completedAt, updatedAt: header.completedAt, attempts: 0, provenance, ...(route ? { route } : {}), routeComplete: !route });
}

export function validateActionEvidenceReceipt(value: unknown, expected: Pick<CompletedActionEvidenceJob,
  "brainId" | "evidenceId" | "executionId" | "evidenceSha256">): CompletedActionEvidenceReceipt {
  const result = value as CompletedActionEvidenceReceipt;
  if (!result || result.brainId !== expected.brainId || result.evidenceId !== expected.evidenceId ||
      result.executionId !== expected.executionId || result.evidenceSha256 !== expected.evidenceSha256 ||
      result.committed !== true || typeof result.processed !== "boolean" || typeof result.duplicate !== "boolean" ||
      (!result.processed && !result.duplicate)) throw new Error("Completed-action learning receipt has a different owner or is not committed.");
  if (result.webImage) {
    const image = result.webImage;
    if (image.format !== "omni-web-image-learning" || image.version !== 1 || ![image.selectionSha256, image.referenceId, image.sourceSha256, image.sourceIdentity, image.transactionId].every(value => hex.test(value)) ||
      !Number.isSafeInteger(image.sourceBytes) || image.sourceBytes < 1 || image.committed !== true || image.duplicate !== result.duplicate || image.sourceIdentity !== result.sourceIdentity) throw new Error("Selected image has no exact committed native media receipt.");
  }
  return result;
}

export class CompletedActionEvidenceStore {
  readonly directory: string;
  private readonly staged = new Map<string, Promise<CompletedActionEvidenceJob>>();
  private recoveryIndexTail: Promise<unknown> = Promise.resolve();
  private recoveryIndexFresh = false;
  constructor(readonly enginePath: string, readonly brainId: string,
    private readonly admit: (bytes: number, directory: string, writing?: boolean) => Promise<void>) {
    this.directory = join(enginePath, "action-result-learning");
  }
  evidencePath(id: string): string { if (!hex.test(id)) throw new Error("Invalid evidence ID."); return join(this.directory, "evidence", `${id}.jsonl`); }
  jobPath(id: string): string { if (!hex.test(id)) throw new Error("Invalid evidence ID."); return join(this.directory, "jobs", `${id}.json`); }
  /** Exact operational replay prevention, not neural memory or text retrieval. */
  recoverNativeEffect(invocation: ToolInvocation, owner: { turnId: string; neuralActionId: string }): Promise<ToolExecutionResult | undefined> {
    const operation = this.recoveryIndexTail.then(() => this.recoverNativeEffectOwned(invocation, owner));
    this.recoveryIndexTail = operation.catch(() => undefined); return operation;
  }
  private async recoverNativeEffectOwned(invocation: ToolInvocation, owner: { turnId: string; neuralActionId: string }): Promise<ToolExecutionResult | undefined> {
    if (invocation.brainId !== this.brainId || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(owner.turnId) || !/^[a-f0-9]{32}$/.test(owner.neuralActionId)) {
      throw new Error("Retained effect lookup requires a controller-owned native turn/action.");
    }
    const directory = join(this.directory, "evidence"), info = await lstat(directory, { bigint: true }).catch((error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") return undefined; throw error;
    });
    if (!info) return undefined;
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Retained effect directory is unsafe.");
    const indexPath = join(this.directory, "native-effect-index.sqlite3");
    const prior = await lstat(indexPath).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (prior && (!prior.isFile() || prior.isSymbolicLink())) throw new Error("Retained effect index is not an owned regular cache.");
    await this.admit(131072, this.directory, true);
    const database = new DatabaseSync(sqliteFilename(indexPath));
    try {
      database.exec("PRAGMA cache_size=-256; PRAGMA journal_mode=DELETE; CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT) WITHOUT ROWID; CREATE TABLE IF NOT EXISTS effects(evidence_id TEXT PRIMARY KEY,sha256 TEXT NOT NULL,turn_id TEXT,native_id TEXT,tool_id TEXT,action TEXT,args_sha256 TEXT) WITHOUT ROWID; CREATE INDEX IF NOT EXISTS by_native ON effects(turn_id,native_id)");
      const epoch = [info.dev, info.ino, info.mtimeNs, info.ctimeNs].join(":"), indexed = database.prepare("SELECT value FROM state WHERE key='evidence-mtime'").get();
      if (!this.recoveryIndexFresh || indexed?.value !== epoch) {
        for await (const entry of await opendir(directory)) {
          if (entry.name.endsWith(".tmp")) continue;
          if (!/^[a-f0-9]{64}\.jsonl$/.test(entry.name) || !entry.isFile() || entry.isSymbolicLink()) throw new Error("Retained effect evidence inventory is unsafe.");
          const evidenceId = entry.name.slice(0, -6);
          if (database.prepare("SELECT 1 FROM effects WHERE evidence_id=?").get(evidenceId)) continue;
          const path = this.evidencePath(evidenceId), header = await evidenceHeader(path, bytes => this.admit(bytes, this.directory, false));
          if (header.format !== "omni-completed-action-result-v1" || header.evidenceId !== evidenceId || !uuid.test(String(header.executionId)) ||
              typeof header.originBrainId !== "string" || digest(`${header.originBrainId}\0${header.executionId}\0completed-action-result-v1`) !== evidenceId) {
            throw new Error("Retained completed effect has invalid immutable provenance.");
          }
          const sha = await actionEvidenceFileHash(path), job = await this.read(evidenceId);
          if (job && job.evidenceSha256 !== sha) throw new Error("Retained effect differs from its durable queue seal.");
          // Fork/import rekeys queue ownership only. Immutable executed data
          // stays attributable to its original owner and cannot grant access.
          if (job && job.brainId !== this.brainId) throw new Error("Retained effect queue belongs to another instance.");
          const args = header.arguments;
          if (!args || typeof args !== "object" || Array.isArray(args)) throw new Error("Retained effect lost its actual operands.");
          await this.admit(4096, this.directory, true);
          database.prepare("INSERT INTO effects VALUES(?,?,?,?,?,?,?)").run(evidenceId, sha,
            typeof header.chatTurnId === "string" ? header.chatTurnId : null, typeof header.neuralActionId === "string" ? header.neuralActionId : null,
            String(header.toolId), String(header.action), effectArguments(args as Record<string, unknown>, String(header.toolId)));
        }
        database.prepare("INSERT OR REPLACE INTO state VALUES('evidence-mtime',?)").run(epoch);
        this.recoveryIndexFresh = true;
      }
      const rows = [...database.prepare("SELECT * FROM effects WHERE turn_id=? AND native_id=? LIMIT 2").iterate(owner.turnId, owner.neuralActionId)];
      if (!rows.length) return undefined;
      const argumentsSha = effectArguments(invocation.arguments, invocation.toolId);
      if (rows.some(row => row.tool_id !== invocation.toolId || row.action !== invocation.action || row.args_sha256 !== argumentsSha)) {
        throw new Error("A restored native action changed the operands of its already-performed external effect; it will not be repeated.");
      }
      if (rows.length !== 1) throw new Error("Conflicting completed receipts exist for this native action; review them instead of repeating the effect.");
      const row = rows[0]!, path = this.evidencePath(String(row.evidence_id)), stat = await lstat(path);
      if (!stat.isFile() || stat.isSymbolicLink()) throw new Error("Retained effect body is unsafe.");
      await this.admit(stat.size * 12 + 131072, this.directory, false);
      const bytes = await readFile(path);
      if (createHash("sha256").update(bytes).digest("hex") !== row.sha256) throw new Error("Retained effect body lost its exact immutable seal.");
      const body = JSON.parse(bytes.toString("utf8")) as Record<string, unknown>;
      if (body.format !== "omni-completed-action-result-v1" || body.evidenceId !== row.evidence_id ||
          digest(`${body.originBrainId}\0${body.executionId}\0completed-action-result-v1`) !== body.evidenceId ||
          !body.arguments || typeof body.arguments !== "object" || Array.isArray(body.arguments) || effectArguments(body.arguments as Record<string, unknown>, invocation.toolId) !== argumentsSha ||
          body.chatTurnId !== owner.turnId || body.neuralActionId !== owner.neuralActionId || body.toolId !== invocation.toolId || body.action !== invocation.action ||
          !uuid.test(String(body.executionId)) || !["complete", "failed"].includes(String(body.executionState)) ||
          typeof body.startedAt !== "string" || !Number.isFinite(Date.parse(body.startedAt)) || typeof body.completedAt !== "string" || !Number.isFinite(Date.parse(body.completedAt)) ||
          typeof body.outputPresent !== "boolean" || body.outputPresent !== Object.hasOwn(body, "output") ||
          body.executionState === "failed" && body.dispatchStarted !== true) throw new Error("Retained effect body is not a terminal execution receipt.");
      return { id: String(body.executionId), toolId: invocation.toolId, action: invocation.action, state: body.executionState as "complete" | "failed",
        startedAt: body.startedAt, finishedAt: body.completedAt, ...(body.dispatchStarted === true ? { dispatchStarted: true } : {}),
        ...(body.outputPresent ? { output: body.output } : {}), ...(typeof body.executionError === "string" ? { error: body.executionError } : {}),
        recoveredEffectReceipt: { evidenceId: String(row.evidence_id), originalActionEventId: String(body.actionEventId), originalBrainId: String(body.originBrainId) } };
    } finally { database.close(); }
  }
  private async prepare(): Promise<void> {
    const engine = await lstat(this.enginePath);
    if (!engine.isDirectory() || engine.isSymbolicLink()) throw new Error("Evidence owner no longer has a real engine directory.");
    for (const path of [this.directory, join(this.directory, "evidence"), join(this.directory, "jobs")]) {
      await mkdir(path).catch((error: NodeJS.ErrnoException) => { if (error.code !== "EEXIST") throw error; });
      const info = await lstat(path);
      if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Evidence state must use owned real directories.");
    }
  }
  async read(id: string): Promise<CompletedActionEvidenceJob | undefined> {
    try {
      const path = this.jobPath(id), info = await lstat(path);
      if (!info.isFile() || info.isSymbolicLink()) throw new Error("Evidence job is not an owned regular file.");
      await this.admit(info.size * 8, this.directory, false);
      return validateActionEvidenceJob(JSON.parse(await readFile(path, "utf8")), this.brainId);
    } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; }
  }
  async save(job: CompletedActionEvidenceJob): Promise<void> {
    validateActionEvidenceJob(job, this.brainId); await this.prepare();
    const path = this.jobPath(job.evidenceId), temporary = `${path}.${randomUUID()}.tmp`;
    const file = await open(temporary, "wx", 0o600);
    try {
      for (const part of actionEvidenceJsonChunks(job)) {
        const bytes = Buffer.from(part, "utf8"); await this.admit(bytes.length, this.directory, true);
        let offset = 0;
        while (offset < bytes.length) {
          const result = await file.write(bytes, offset, bytes.length - offset);
          if (result.bytesWritten < 1) throw new Error("Evidence manifest writer made no progress.");
          offset += result.bytesWritten;
        }
      }
      await file.sync(); await file.close(); await rename(temporary, path);
    } catch (error) { await file.close().catch(() => undefined); await unlink(temporary).catch(() => undefined); throw error; }
    if (process.platform !== "win32") { const parent = await open(join(this.directory, "jobs"), "r"); try { await parent.sync(); } finally { await parent.close(); } }
  }
  async pending(): Promise<CompletedActionEvidenceJob[]> {
    await this.prepare();
    // A crash between immutable evidence rename and queue-manifest publish
    // does not lose an already captured actual outcome. Recover only its
    // inert header/file hash; never execute the original effect again.
    for (const name of await readdir(join(this.directory, "evidence"))) {
      if (!/^[a-f0-9]{64}\.jsonl$/.test(name)) { if (name.endsWith(".tmp")) continue; throw new Error("Unknown evidence file."); }
      const evidenceId = name.slice(0, -6);
      if (await this.read(evidenceId)) continue;
      const path = this.evidencePath(evidenceId), header = await evidenceHeader(path,
        bytes => this.admit(bytes, this.directory, false));
      if (header.format !== "omni-completed-action-result-v1" || header.evidenceId !== evidenceId || !uuid.test(String(header.executionId))) {
        throw new Error("Orphan evidence has invalid identity.");
      }
      const { executionId, evidenceId: _id, startedAt: _started, arguments: _arguments, learningRoute, ...provenance } = header;
      const now = new Date().toISOString(), route = learningRoute as CompletedActionEvidenceJob["route"];
      await this.save({ format: "omni-action-result-learning-job-v1", brainId: this.brainId, evidenceId,
        executionId: String(executionId), evidenceSha256: await actionEvidenceFileHash(path), state: "pending",
        createdAt: now, updatedAt: now, attempts: 0, provenance, ...(route ? { route } : {}), routeComplete: !route });
    }
    const result: CompletedActionEvidenceJob[] = [];
    for (const name of await readdir(join(this.directory, "jobs"))) {
      if (!/^[a-f0-9]{64}\.json$/.test(name)) { if (name.endsWith(".tmp")) continue; throw new Error("Unknown evidence queue entry."); }
      const job = await this.read(name.slice(0, -5));
      if (job?.state === "pending") { await this.admit((result.length + 1) * 1024, this.directory, false); result.push(job); }
    }
    return result;
  }
  async pendingCount(): Promise<number> {
    // Status polls inspect inert manifests, not every historical result body.
    // Exact body hashes remain mandatory before learning and portability.
    const files = await savedActionEvidenceFiles(this.enginePath, false);
    let count = 0;
    const manifested = new Set<string>();
    for (const [name, path] of files) if (name.startsWith("jobs/")) {
      const job = await readSavedEvidenceJob(path); manifested.add(job.evidenceId);
      if (job.state === "pending") count += 1;
    }
    for (const name of files.keys()) if (name.startsWith("evidence/") && !manifested.has(name.slice(9, -6))) count += 1;
    return count;
  }
  stage(event: ActionEvent, output: unknown, route?: CompletedActionEvidenceJob["route"], chatTurnId?: string): Promise<CompletedActionEvidenceJob> {
    const execution = event.execution;
    if (event.brainId !== this.brainId || !actualTerminalToolOutcome(execution) ||
        !uuid.test(execution.id) || execution.output !== output) return Promise.reject(new Error("Only an actual completed execution can enter result learning."));
    return this.stageReceipt(execution, event.action.arguments, event.id, event.neuralActionId, route, chatTurnId);
  }
  stageTool(invocation: ToolInvocation, execution: ToolExecutionResult, chatTurnId?: string, actionEventId?: string): Promise<CompletedActionEvidenceJob> {
    if (invocation.brainId !== this.brainId || !actualTerminalToolOutcome(execution) || !uuid.test(execution.id)) {
      return Promise.reject(new Error("Only an actual completed execution can enter result learning."));
    }
    if (actionEventId !== undefined && (!actionEventId || actionEventId.includes("\0"))) return Promise.reject(new Error("Trusted completed action context is invalid."));
    return this.stageReceipt(execution, invocation.arguments, actionEventId ?? execution.id,
      typeof invocation.arguments.neuralActionId === "string" ? invocation.arguments.neuralActionId : undefined,
      undefined, chatTurnId);
  }
  private stageReceipt(execution: ToolExecutionResult, args: Record<string, unknown>, actionEventId: string,
    neuralActionId?: string, route?: CompletedActionEvidenceJob["route"], chatTurnId?: string): Promise<CompletedActionEvidenceJob> {
    this.recoveryIndexFresh = false;
    const output = execution.output;
    if (execution.state !== "complete") route = undefined; // a failed observation is not positive route supervision
    const evidenceId = digest(`${this.brainId}\0${execution.id}\0completed-action-result-v1`);
    const current = this.staged.get(evidenceId);
    if (current) return current.then(job => route && !job.route
      ? this.stageReceipt(execution, args, actionEventId, neuralActionId, route, chatTurnId) : job);
    const operation = (async () => {
      await this.prepare();
      const prior = await this.read(evidenceId);
      if (prior) {
        if (await actionEvidenceFileHash(this.evidencePath(evidenceId)) !== prior.evidenceSha256) throw new Error("Retained execution evidence changed.");
        if (route && !prior.route) { prior.route = route; prior.routeComplete = false; prior.state = "pending"; await this.save(prior); }
        return prior; // immutable completed execution identity, never another effect
      }
      const provenance = { format: "omni-completed-action-result-v1", originBrainId: this.brainId,
        actionEventId, ...(neuralActionId ? { neuralActionId } : {}),
        ...(chatTurnId ? { chatTurnId } : {}), toolId: execution.toolId, action: execution.action,
        completedAt: execution.finishedAt!, executionState: execution.state,
        ...(execution.dispatchStarted ? { dispatchStarted: true } : {}), ...(execution.error ? { executionError: execution.error } : {}) };
      const record = { ...provenance, evidenceId, executionId: execution.id, startedAt: execution.startedAt,
        ...(route ? { learningRoute: route } : {}), arguments: args, outputPresent: output !== undefined,
        ...(output === undefined ? {} : { output }) };
      const path = this.evidencePath(evidenceId), temporary = `${path}.${randomUUID()}.tmp`;
      const file = await open(temporary, "wx", 0o600), sha = createHash("sha256");
      try {
        let chunk = "";
        const write = async (text: string): Promise<void> => {
          const bytes = Buffer.from(text, "utf8"); await this.admit(bytes.length, this.directory);
          let offset = 0;
          while (offset < bytes.length) {
            const result = await file.write(bytes, offset, bytes.length - offset);
            if (result.bytesWritten < 1) throw new Error("Evidence writer made no progress.");
            offset += result.bytesWritten;
          }
          sha.update(bytes);
        };
        for (const part of actionEvidenceJsonChunks(record)) {
          chunk += part;
          if (chunk.length >= CHUNK) { await write(chunk); chunk = ""; }
        }
        await write(`${chunk}\n`); await file.sync(); await file.close();
        await rename(temporary, path);
        if (process.platform !== "win32") { const parent = await open(join(this.directory, "evidence"), "r"); try { await parent.sync(); } finally { await parent.close(); } }
      } catch (error) { await file.close().catch(() => undefined); await unlink(temporary).catch(() => undefined); throw error; }
      const now = new Date().toISOString();
      const job: CompletedActionEvidenceJob = { format: "omni-action-result-learning-job-v1", brainId: this.brainId,
        evidenceId, executionId: execution.id, evidenceSha256: sha.digest("hex"), state: "pending", createdAt: now,
        updatedAt: now, attempts: 0, provenance, ...(route ? { route } : {}), routeComplete: !route };
      await this.save(job); return job;
    })().finally(() => { this.staged.delete(evidenceId); });
    this.staged.set(evidenceId, operation); return operation;
  }
}

/** Exact operational subtree inventory for fork/export/recovery. Never replay. */
export async function savedActionEvidenceFiles(engineDirectory: string, verifyContentHashes = true): Promise<Map<string, string>> {
  const root = join(engineDirectory, "action-result-learning"), files = new Map<string, string>();
  const info = await lstat(root).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
  if (!info) return files;
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Action evidence root is unsafe.");
  for (const kind of ["jobs", "evidence"]) {
    const directory = join(root, kind), detail = await lstat(directory);
    if (!detail.isDirectory() || detail.isSymbolicLink()) throw new Error("Action evidence directory is unsafe.");
    for (const name of await readdir(directory)) {
      if (name.endsWith(".tmp")) continue;
      if (!(kind === "jobs" ? /^[a-f0-9]{64}\.json$/ : /^[a-f0-9]{64}\.jsonl$/).test(name)) throw new Error("Action evidence entry is unsafe.");
      const path = join(directory, name), file = await lstat(path);
      if (!file.isFile() || file.isSymbolicLink()) throw new Error("Action evidence entry is not a regular file.");
      files.set(`${kind}/${name}`, path);
    }
  }
  await validateSavedActionEvidenceFiles(files, verifyContentHashes);
  return files;
}

export async function validateSavedActionEvidenceFiles(files: ReadonlyMap<string, string>, verifyContentHashes = true): Promise<void> {
  for (const [name, path] of files) {
    if (!/^(?:jobs\/[a-f0-9]{64}\.json|evidence\/[a-f0-9]{64}\.jsonl)$/.test(name)) throw new Error("Unsafe saved evidence path.");
    const detail = await lstat(path);
    if (!detail.isFile() || detail.isSymbolicLink()) throw new Error("Saved evidence contains a filesystem link or nonfile.");
  }
  for (const [name, path] of files) if (name.startsWith("jobs/")) {
    const job = await readSavedEvidenceJob(path);
    if (name !== `jobs/${job.evidenceId}.json`) throw new Error("Saved evidence manifest filename changed its identity.");
    const evidence = files.get(`evidence/${job.evidenceId}.jsonl`);
    if (!evidence || verifyContentHashes && await actionEvidenceFileHash(evidence) !== job.evidenceSha256) throw new Error("Action evidence queue/file binding is invalid.");
  }
}

/** Rekey only current queue ownership/receipt; immutable executed data stays. */
export async function rekeyActionEvidenceJobs(engineDirectory: string, brainId: string): Promise<void> {
  const files = await savedActionEvidenceFiles(engineDirectory);
  for (const [name, path] of files) if (name.startsWith("jobs/")) {
    const job = await readSavedEvidenceJob(path);
    job.brainId = brainId;
    if (job.receipt) job.receipt = { ...job.receipt, brainId };
    const store = new CompletedActionEvidenceStore(engineDirectory, brainId, async () => undefined);
    await store.save(job);
  }
}
