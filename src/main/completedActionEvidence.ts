import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, mkdir, open, readFile, readdir, rename, unlink } from "node:fs/promises";
import { join } from "node:path";
import { StringDecoder } from "node:string_decoder";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import type { ActionEvent, ToolExecutionResult, ToolInvocation } from "../shared/types";

export interface CompletedActionEvidenceReceipt {
  brainId: string; evidenceId: string; executionId: string; evidenceSha256: string;
  committed: true; processed: boolean; duplicate: boolean;
  sourceIdentity?: string; commitSequence?: number;
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
}

const digest = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");
const hex = /^[a-f0-9]{64}$/;
const uuid = /^[a-f0-9-]{36}$/i;
const CHUNK = 64 * 1024;

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
  if (job.receipt) validateActionEvidenceReceipt(job.receipt, job);
  if (job.state === "complete" && (!job.receipt || !job.routeComplete)) throw new Error("Completed evidence job lacks exact neural/route receipts.");
  return job;
}

export function validateActionEvidenceReceipt(value: unknown, expected: Pick<CompletedActionEvidenceJob,
  "brainId" | "evidenceId" | "executionId" | "evidenceSha256">): CompletedActionEvidenceReceipt {
  const result = value as CompletedActionEvidenceReceipt;
  if (!result || result.brainId !== expected.brainId || result.evidenceId !== expected.evidenceId ||
      result.executionId !== expected.executionId || result.evidenceSha256 !== expected.evidenceSha256 ||
      result.committed !== true || typeof result.processed !== "boolean" || typeof result.duplicate !== "boolean" ||
      (!result.processed && !result.duplicate)) throw new Error("Completed-action learning receipt has a different owner or is not committed.");
  return result;
}

export class CompletedActionEvidenceStore {
  readonly directory: string;
  private readonly staged = new Map<string, Promise<CompletedActionEvidenceJob>>();
  constructor(readonly enginePath: string, readonly brainId: string,
    private readonly admit: (bytes: number, directory: string, writing?: boolean) => Promise<void>) {
    this.directory = join(enginePath, "action-result-learning");
  }
  evidencePath(id: string): string { if (!hex.test(id)) throw new Error("Invalid evidence ID."); return join(this.directory, "evidence", `${id}.jsonl`); }
  jobPath(id: string): string { if (!hex.test(id)) throw new Error("Invalid evidence ID."); return join(this.directory, "jobs", `${id}.json`); }
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
  stageTool(invocation: ToolInvocation, execution: ToolExecutionResult, chatTurnId?: string): Promise<CompletedActionEvidenceJob> {
    if (invocation.brainId !== this.brainId || !actualTerminalToolOutcome(execution) || !uuid.test(execution.id)) {
      return Promise.reject(new Error("Only an actual completed execution can enter result learning."));
    }
    return this.stageReceipt(execution, invocation.arguments, execution.id,
      typeof invocation.arguments.neuralActionId === "string" ? invocation.arguments.neuralActionId : undefined,
      undefined, chatTurnId);
  }
  private stageReceipt(execution: ToolExecutionResult, args: Record<string, unknown>, actionEventId: string,
    neuralActionId?: string, route?: CompletedActionEvidenceJob["route"], chatTurnId?: string): Promise<CompletedActionEvidenceJob> {
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
