import { createHash, randomUUID } from "node:crypto";
import { constants as fsConstants, createReadStream } from "node:fs";
import { copyFile, lstat, mkdir, open, opendir, readFile, readdir, realpath, rename, rm } from "node:fs/promises";
import { basename, dirname, join, relative, resolve, sep } from "node:path";
import { backup, DatabaseSync } from "node:sqlite";
import { IncompleteLearningWorkspaceError, learningWorkspaceArchive, snapshotLearningWorkspace } from "./learningWorkspacePortability";
import type { BrainConfig } from "../shared/types";
import type { FullMergeParentSeal, FullMergeRequest, FullMergeProgress } from "../shared/fullMerge";
export type { FullMergeParentSeal, FullMergeRequest, FullMergeProgress } from "../shared/fullMerge";
import { writeMutableFileIsolated } from "./mutableFileIsolation";
import { captureMcpTaskCustody } from "./mcpTaskCustody";
import { MCP_TASK_DIRECTORY } from "./mcpTaskStore";
import { mcpTaskEnginePath } from "./mcpTaskArchive";

const FORMAT = "omni-full-brain-merge";
const VERSION = 1;
const ID = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$/;
const HASH = /^[a-f0-9]{64}$/;
const originSeedingAssemblies = new WeakSet<object>();
/** Authority is scoped to the coordinator's actual unpublished assemble call,
 * not a caller-provided pathname or a serialized/imported boolean. */
export function assertFullMergeOriginSeeding(input: FullMergeAssemblyInput): void {
  if (!originSeedingAssemblies.has(input) || input.parents.some(parent => parent.brainId === input.targetBrainId) ||
    input.request.parentIds.includes(input.targetBrainId)) throw new Error("Merged-origin custody may be seeded only during a distinct, unpublished coordinator assembly.");
}

export interface FullMergeStoredFile {
  parentId: string;
  path: string;
  sha256: string;
  bytes: number;
  mode: number;
}

export interface FullMergeAssemblyInput {
  targetBrainId: string;
  targetDirectory: string;
  request: FullMergeRequest;
  parents: FullMergeParentSeal[];
  /** Complete snapshot mapping; stream it instead of making a full RAM copy. */
  files(parentId?: string): Iterable<FullMergeStoredFile>;
  blobPath(sha256: string): string;
  /** Isolated, verified source tree for native loading; never a live parent. */
  materializeParent(parentId: string): Promise<string>;
  signal?: AbortSignal;
  checkpoint(label: string): Promise<void>;
  reserve(bytes: number, purpose: string): Promise<void>;
  admit?(bytes: number, purpose: string): Promise<void> | void;
}

export interface FullMergeHooks {
  /** Stop/pause at a real checkpoint, then lock every parent in stable order. */
  withParentsFrozen<T>(parentIds: string[], action: () => Promise<T>, signal?: AbortSignal, existingSeals?: FullMergeParentSeal[]): Promise<T>;
  /** Validate native origin, current committed tensors/cursors, and ancestry. */
  sealParent(parentId: string, signal?: AbortSignal): Promise<FullMergeParentSeal>;
  validateSettings(config: BrainConfig): Promise<void>;
  reserve(bytes: number, purpose: string): Promise<void>;
  admit?(bytes: number, purpose: string): Promise<void> | void;
  /** Build the connected graph, complete history indexes and immutable origin. */
  assemble(input: FullMergeAssemblyInput): Promise<void>;
  /** Must exercise/load the actual connected graph and validate complete state. */
  verify(input: FullMergeAssemblyInput): Promise<void>;
  onProgress?(progress: FullMergeProgress): void;
}

interface Job {
  format: typeof FORMAT;
  formatVersion: typeof VERSION;
  requestSha256: string;
  request: FullMergeRequest;
  targetBrainId: string;
  state: FullMergeProgress["state"];
  parents: FullMergeParentSeal[];
  snapshotComplete: boolean;
  error?: string;
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) =>
      `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}`;
  }
  const encoded = JSON.stringify(value);
  if (encoded === undefined) throw new Error("Merge settings contain an unserializable value.");
  return encoded;
}

const digest = (value: unknown): string => createHash("sha256").update(canonical(value)).digest("hex");

function identifier(value: string): string {
  if (!ID.test(value)) throw new Error("Invalid full-merge identity.");
  return value;
}

async function exists(path: string): Promise<boolean> {
  try { await lstat(path); return true; } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

async function hashFile(path: string, signal?: AbortSignal): Promise<string> {
  const hash = createHash("sha256");
  const stream = createReadStream(path, { highWaterMark: 1024 * 1024, signal });
  for await (const value of stream) { signal?.throwIfAborted(); hash.update(value); }
  return hash.digest("hex");
}

async function copyBlob(source: string, destination: string, expected: string, signal: AbortSignal): Promise<void> {
  const temporary = `${destination}.${randomUUID()}.next`;
  const before = await lstat(source);
  if (!before.isFile() || before.isSymbolicLink()) throw new Error("Merge blob source is not a regular file.");
  try {
    // Force reflink so an unsupported filesystem falls back to our bounded,
    // cancellable stream instead of an uninterruptible whole-file copy.
    try { await copyFile(source, temporary, fsConstants.COPYFILE_FICLONE_FORCE); }
    catch (error) {
      if (!["ENOTSUP", "EOPNOTSUPP", "EINVAL", "ENOSYS", "EXDEV", "ENOTTY"].includes((error as NodeJS.ErrnoException).code ?? "")) throw error;
      await rm(temporary, { force: true });
      const output = await open(temporary, "wx", 0o600);
      try {
        for await (const bytes of createReadStream(source, { highWaterMark: 1024 * 1024, signal })) {
          signal.throwIfAborted();
          const value = bytes as Buffer;
          let written = 0;
          while (written < value.length) {
            const result = await output.write(value, written, value.length - written);
            if (!result.bytesWritten) throw new Error("A merge blob write made no progress.");
            written += result.bytesWritten;
          }
        }
        await output.sync();
      } finally { await output.close(); }
    }
    signal.throwIfAborted();
    const after = await lstat(source);
    const copied = await lstat(temporary);
    if (after.ino !== before.ino || after.dev !== before.dev || after.size !== before.size ||
        after.mtimeMs !== before.mtimeMs || copied.size !== before.size ||
        (copied.dev === before.dev && copied.ino === before.ino) ||
        await hashFile(temporary, signal) !== expected) throw new Error("A merge blob changed while copying.");
    const handle = await open(temporary, "r+");
    try { await handle.chmod(0o600); await handle.sync(); } finally { await handle.close(); }
    await rename(temporary, destination);
  } finally { await rm(temporary, { force: true }); }
}
export { copyBlob as copyFullMergeBlob };

async function snapshotSqlite(source: string, target: string, signal: AbortSignal): Promise<void> {
  const database = new DatabaseSync(source, { readOnly: true });
  try {
    database.exec("PRAGMA busy_timeout=1000; PRAGMA cache_size=-1024");
    // The asynchronous incremental API yields between pages, allowing Pause
    // while a large history/replay database is being captured.
    await backup(database, target, { rate: 128, progress: () => signal.throwIfAborted() });
    signal.throwIfAborted();
  } finally { database.close(); }
  const handle = await open(target, "r+");
  try { await handle.sync(); } finally { await handle.close(); }
}

function normalizeRequest(input: FullMergeRequest): FullMergeRequest {
  identifier(input.operationId);
  const parentIds = [...new Set(input.parentIds.map(identifier))].sort();
  if (parentIds.length < 2 || parentIds.length !== input.parentIds.length) {
    throw new Error("Full merge requires at least two different native instances.");
  }
  if (!input.name.trim() || input.name.includes("\0")) throw new Error("The merged instance needs a name.");
  return JSON.parse(canonical({ ...input, name: input.name.trim(), parentIds })) as FullMergeRequest;
}

function validateSeal(seal: FullMergeParentSeal, parentId: string): FullMergeParentSeal {
  if (seal.brainId !== parentId || !HASH.test(seal.generationSha256) ||
      !Array.isArray(seal.ancestry) || seal.ancestry.some((item) => !HASH.test(item))) {
    throw new Error("The parent native checkpoint seal is invalid.");
  }
  return JSON.parse(canonical(seal)) as FullMergeParentSeal;
}

/** Full merge is a new identity; existing reviewed overlays use a separate API. */
export class FullBrainMergeCoordinator {
  private readonly root: string;
  private readonly operations: string;
  private readonly running = new Map<string, AbortController>();

  constructor(brainsRoot: string, private readonly hooks: FullMergeHooks) {
    this.root = resolve(brainsRoot);
    this.operations = join(this.root, ".full-merges");
  }

  private operation(id: string): string { return join(this.operations, identifier(id)); }

  private async saveJob(directory: string, job: Job): Promise<void> {
    await writeMutableFileIsolated(join(directory, "job.json"), Buffer.from(canonical(job)));
  }

  private database(directory: string): DatabaseSync {
    const db = new DatabaseSync(join(directory, "files.sqlite3"));
    db.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA temp_store=FILE; PRAGMA cache_size=-1024");
    db.exec(`CREATE TABLE IF NOT EXISTS files (
      parent_id TEXT NOT NULL, path TEXT NOT NULL, source_bytes INTEGER NOT NULL, mode INTEGER NOT NULL,
      sha256 TEXT, bytes INTEGER, PRIMARY KEY(parent_id,path));
      CREATE TABLE IF NOT EXISTS blobs (sha256 TEXT PRIMARY KEY,bytes INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS discovered_parents (parent_id TEXT PRIMARY KEY);`);
    return db;
  }

  private progress(job: Job, db: DatabaseSync): FullMergeProgress {
    const row = db.prepare("SELECT COUNT(*) AS total,COUNT(sha256) AS copied,COALESCE(SUM(source_bytes),0) AS logical,COALESCE(SUM(CASE WHEN sha256 IS NOT NULL THEN bytes ELSE 0 END),0) AS bytes FROM files").get()!;
    const unique = db.prepare("SELECT COALESCE(SUM(bytes),0) AS bytes FROM blobs").get()!;
    return { operationId: job.request.operationId, targetBrainId: job.targetBrainId, state: job.state,
      name: job.request.name, reviewedSettings: structuredClone(job.request.config),
      ...(job.request.accessPolicy ? { reviewedAccess: structuredClone(job.request.accessPolicy) } : {}),
      ...(job.request.hardwareTier ? { hardwareTier: job.request.hardwareTier } : {}),
      filesDiscovered: Number(row.total), filesCopied: Number(row.copied), logicalBytes: Number(row.logical),
      copiedBytes: Number(row.bytes), uniqueBlobBytes: Number(unique.bytes), parents: job.parents,
      ...(job.error ? { error: job.error } : {}) };
  }

  async status(operationId: string): Promise<FullMergeProgress> {
    const directory = this.operation(operationId);
    const cancelled = join(this.root, ".full-merge-cancellations", `${identifier(operationId)}.json`);
    if (await exists(cancelled)) return JSON.parse(await readFile(cancelled, "utf8")) as FullMergeProgress;
    const job = JSON.parse(await readFile(join(directory, "job.json"), "utf8")) as Job;
    if (job.format !== FORMAT || job.formatVersion !== VERSION) throw new Error("Unsupported full-merge operation.");
    const db = this.database(directory);
    try {
      const result = this.progress(job, db);
      if (!this.running.has(operationId) && !["complete", "paused", "cancelled"].includes(result.state)) {
        result.state = "paused";
        result.error = "Application stopped before publication; resume the saved operation.";
      }
      return result;
    } finally { db.close(); }
  }

  async list(): Promise<FullMergeProgress[]> {
    const ids = new Set<string>();
    for (const [directory, suffix] of [[this.operations, ""], [join(this.root, ".full-merge-cancellations"), ".json"]]) {
      for (const entry of await readdir(directory!).catch((error: NodeJS.ErrnoException) => {
        if (error.code === "ENOENT") return [];
        throw error;
      })) {
        const id = suffix ? entry.slice(0, -suffix.length) : entry;
        if (ID.test(id) && (!suffix || entry.endsWith(suffix))) ids.add(id);
      }
    }
    return Promise.all([...ids].sort().map((id) => this.status(id)));
  }

  async resume(operationId: string, signal?: AbortSignal): Promise<FullMergeProgress> {
    const job = JSON.parse(await readFile(join(this.operation(operationId), "job.json"), "utf8")) as Job;
    return this.start(job.request, signal);
  }

  async start(input: FullMergeRequest, externalSignal?: AbortSignal): Promise<FullMergeProgress> {
    const request = normalizeRequest(input);
    if (this.running.has(request.operationId)) throw new Error("This merge is already running.");
    if (await exists(join(this.root, ".full-merge-cancellations", `${request.operationId}.json`))) {
      throw new Error("This merge was cancelled. Review a new operation.");
    }
    await this.hooks.validateSettings(request.config);
    const controller = new AbortController();
    const abort = (): void => controller.abort(externalSignal?.reason);
    externalSignal?.addEventListener("abort", abort, { once: true });
    if (externalSignal?.aborted) abort();
    this.running.set(request.operationId, controller);
    const signal = controller.signal;
    const directory = this.operation(request.operationId);
    let db: DatabaseSync;
    let job: Job;
    try {
      await mkdir(directory, { recursive: true });
      db = this.database(directory);
      try {
        job = await exists(join(directory, "job.json"))
          ? JSON.parse(await readFile(join(directory, "job.json"), "utf8")) as Job
          : { format: FORMAT, formatVersion: VERSION, requestSha256: digest(request), request,
              targetBrainId: randomUUID(), state: "preparing", parents: [], snapshotComplete: false };
      } catch (error) { db.close(); throw error; }
    } catch (error) {
      this.running.delete(request.operationId);
      externalSignal?.removeEventListener("abort", abort);
      throw error;
    }
    let requestVerified = false;
    try {
      if (job.format !== FORMAT || job.formatVersion !== VERSION || job.requestSha256 !== digest(request) ||
          digest(job.request) !== job.requestSha256) throw new Error("The merge request changed; review a new operation.");
      requestVerified = true;
      identifier(job.targetBrainId);
      if (job.state === "paused") job.state = job.snapshotComplete ? "assembling" : "preparing";
      await this.saveJob(directory, job);
      this.hooks.onProgress?.(this.progress(job, db));
      const destination = join(this.root, job.targetBrainId);
      if (await exists(destination)) {
        const marker = JSON.parse(await readFile(join(destination, "full-merge.json"), "utf8")) as Record<string, unknown>;
        if (marker.operationId !== request.operationId || marker.requestSha256 !== job.requestSha256 || marker.verified !== true) {
          throw new Error("The target identity already exists with a different merge.");
        }
        // Recover a crash after atomic publication but before the job receipt.
        job.state = "complete";
        delete job.error;
        await this.saveJob(directory, job);
        return this.progress(job, db);
      }
      signal.throwIfAborted();
      delete job.error;
      const target = join(directory, "target");
      const blobs = join(target, "merge", "blobs");
      await mkdir(blobs, { recursive: true });
      if (!job.snapshotComplete) {
        await this.hooks.withParentsFrozen(request.parentIds, async () => {
          const seals = [];
          for (const parentId of request.parentIds) {
            signal.throwIfAborted();
            seals.push(validateSeal(await this.hooks.sealParent(parentId, signal), parentId));
          }
          if (job.parents.length && canonical(job.parents) !== canonical(seals)) {
            throw new Error("A parent changed before its complete snapshot; review a new merge.");
          }
          job.parents = seals;
          job.state = "copying";
          await this.saveJob(directory, job);
          for (const parent of seals) {
            await this.discover(parent.brainId, db, signal, directory);
          }
          for (const row of db.prepare("SELECT parent_id,path,source_bytes,sha256,bytes FROM files ORDER BY parent_id,path").iterate()) {
            signal.throwIfAborted();
            await this.copyOne(row, blobs, directory, db, signal);
            this.hooks.onProgress?.(this.progress(job, db));
          }
          for (const parent of seals) {
            const after = validateSeal(await this.hooks.sealParent(parent.brainId, signal), parent.brainId);
            if (canonical(after) !== canonical(parent)) throw new Error("The parent changed while copying its neural state.");
          }
          job.snapshotComplete = true;
          await this.saveJob(directory, job);
        }, signal, job.parents);
      }
      if (job.snapshotComplete) await rm(join(directory, ".mcp-task-snapshots"), { recursive: true, force: true });
      const assembly = this.assemblyInput(job, db, target, blobs, signal);
      // The snapshot manifest is inside the final instance and therefore part
      // of its portable, unsanitized state. Every distinct event stays intact.
      await this.writeSnapshotIndex(job, db, target, signal);
      job.state = "assembling";
      await this.saveJob(directory, job);
      this.hooks.onProgress?.(this.progress(job, db));
      originSeedingAssemblies.add(assembly);
      try { await this.hooks.assemble(assembly); }
      finally { originSeedingAssemblies.delete(assembly); }
      signal.throwIfAborted();
      job.state = "verifying";
      await this.saveJob(directory, job);
      await this.verifyBlobs(db, blobs, signal);
      await this.hooks.verify(assembly);
      if (!await exists(join(target, "brain.json")) || !await exists(join(target, "engine", "brain.json"))) {
        throw new Error("Connected merge assembly did not create a complete instance.");
      }
      signal.throwIfAborted();
      await writeMutableFileIsolated(join(target, "full-merge.json"), Buffer.from(canonical({
        format: FORMAT, formatVersion: VERSION, operationId: request.operationId,
        requestSha256: job.requestSha256, targetBrainId: job.targetBrainId,
        parents: job.parents, verified: true,
      })));
      await this.hooks.reserve(4096, "atomic merged-instance publication");
      // The library sees no partial brain. A failed/cancelled staging job is
      // contained beneath .full-merges and never replaces a parent's path.
      await rename(target, destination);
      const handle = await open(this.root, "r");
      try { await handle.sync().catch(() => undefined); } finally { await handle.close(); }
      job.state = "complete";
      await this.saveJob(directory, job);
      const result = this.progress(job, db);
      this.hooks.onProgress?.(result);
      return result;
    } catch (error) {
      if (!requestVerified) throw error;
      job.state = "paused";
      if (signal.aborted) delete job.error;
      else job.error = error instanceof Error ? error.message : String(error);
      await this.saveJob(directory, job);
      this.hooks.onProgress?.(this.progress(job, db));
      throw signal.aborted ? signal.reason ?? error : error;
    } finally {
      db.close();
      this.running.delete(request.operationId);
      externalSignal?.removeEventListener("abort", abort);
    }
  }

  private async discover(parentId: string, db: DatabaseSync, signal: AbortSignal, stagingDirectory: string): Promise<void> {
    if (db.prepare("SELECT 1 FROM discovered_parents WHERE parent_id=?").get(parentId)) return;
    const source = join(this.root, identifier(parentId));
    const sourceInfo = await lstat(source);
    if (!sourceInfo.isDirectory() || sourceInfo.isSymbolicLink() || dirname(await realpath(source)) !== await realpath(this.root)) {
      throw new Error("A merge parent escaped its exact managed directory.");
    }
    // A restart before discovery completed rebuilds only this uncommitted
    // inventory. Previously committed immutable blobs remain reusable.
    db.prepare("DELETE FROM files WHERE parent_id=? AND sha256 IS NULL").run(parentId);
    const add = db.prepare("INSERT OR IGNORE INTO files(parent_id,path,source_bytes,mode) VALUES (?,?,?,?)");
    const taskStage = join(stagingDirectory, ".mcp-task-snapshots", identifier(parentId));
    await rm(taskStage, { recursive: true, force: true }); await mkdir(taskStage, { recursive: true, mode: 0o700 });
    const visit = async (directory: string): Promise<void> => {
      const entries = await opendir(directory);
      for await (const entry of entries) {
        signal.throwIfAborted();
        const path = join(directory, entry.name);
        const info = await lstat(path);
        if (info.isSymbolicLink()) throw new Error("A parent contains a symbolic link; resolve it before a complete merge.");
        if (info.isDirectory()) {
          if (entry.name === MCP_TASK_DIRECTORY && /(?:^|\/)engine(?:\/origin)?$/.test(relative(source, directory).split(sep).join("/"))) {
            const engineLogical = relative(source, directory).split(sep).join("/"), target = join(taskStage, ...engineLogical.split("/"));
            await mkdir(dirname(target), { recursive: true, mode: 0o700 });
            const captured = await captureMcpTaskCustody(directory, target, { signal, admit: this.hooks.admit,
              checkDisk: bytes => this.hooks.reserve(bytes, "coherent complete parent MCP task snapshot") });
            for (const [local, saved] of captured) { const stat = await lstat(saved); add.run(parentId, `${engineLogical}/${local}`, stat.size, stat.mode & 0o777); }
          } else await visit(path);
          continue;
        }
        if (!info.isFile()) throw new Error("A parent contains a nonportable special file.");
        if (/\.sqlite3?-(?:wal|shm)$/.test(entry.name)) continue; // captured by SQLite backup
        const logical = relative(source, path).split(sep).join("/");
        if (!Number.isSafeInteger(info.size)) throw new Error("A parent file size is not representable.");
        add.run(parentId, logical, info.size, info.mode & 0o777);
      }
    };
    await visit(source);
    const workspacePath = join(stagingDirectory, ".learning-workspaces", parentId);
    let workspace;
    try { workspace = await learningWorkspaceArchive(workspacePath, parentId); }
    catch (error) {
      if (!(error instanceof IncompleteLearningWorkspaceError)) throw error;
      // A process may have stopped before this private, still-undiscovered
      // inventory's final commit. Rebuild that output, not any parent data.
      await rm(workspacePath, { recursive: true, force: true });
    }
    if (!workspace) {
      try { workspace = await snapshotLearningWorkspace(source, workspacePath, parentId, {
        signal, checkDisk: async bytes => { await this.hooks.reserve(bytes, "complete parent learning source snapshot"); },
        admitMetadata: async (bytes, purpose, requestedSignal) => {
          signal.throwIfAborted(); requestedSignal?.throwIfAborted(); await this.hooks.admit?.(bytes, purpose);
          signal.throwIfAborted(); requestedSignal?.throwIfAborted();
        }
      }); } catch (error) {
        // Only this regenerable private output is removed. Parent files and
        // already committed shared blobs remain untouched for the same retry.
        await rm(workspacePath, { recursive: true, force: true }); throw error;
      }
    }
    if (workspace) for (const [path, payload] of workspace.files) {
      const logical = `portable-learning-workspace/${path}`;
      if (db.prepare("SELECT 1 FROM files WHERE parent_id=? AND path=?").get(parentId, logical)) throw new Error("A parent path collides with its portable learning snapshot.");
      add.run(parentId, logical, (await lstat(payload)).size, 0o600);
    }
    db.prepare("INSERT INTO discovered_parents(parent_id) VALUES (?)").run(parentId);
  }

  private async copyOne(row: Record<string, unknown>, blobs: string, directory: string, db: DatabaseSync, signal: AbortSignal): Promise<void> {
    const parentId = String(row.parent_id);
    const logical = String(row.path);
    const parts = logical.split("/");
    if (parts.some((p) => !p || p === "." || p === ".." || p.includes("\\"))) throw new Error("Unsafe parent snapshot path.");
    const source = mcpTaskEnginePath(logical)
      ? join(directory, ".mcp-task-snapshots", identifier(parentId), ...parts)
      : logical.startsWith("portable-learning-workspace/")
      ? join(directory, ".learning-workspaces", identifier(parentId), ...parts.slice(1))
      : join(this.root, identifier(parentId), ...parts);
    if (typeof row.sha256 === "string") {
      if (!HASH.test(row.sha256) || await hashFile(join(blobs, row.sha256), signal) !== row.sha256) {
        throw new Error("An existing merge blob failed its checksum.");
      }
      return;
    }
    const info = await lstat(source);
    if (!info.isFile() || info.isSymbolicLink() || info.size !== Number(row.source_bytes)) throw new Error("A parent file changed after discovery.");
    if (/(?:^|\/)api-teachers\//.test(logical)) {
      signal.throwIfAborted(); await this.hooks.admit?.(1024 * 1024, "teacher parent copy SQLite/hash window"); signal.throwIfAborted();
    }
    let captured = source;
    let temporary: string | undefined;
    try {
      const sealedLearning = /(?:^|\/)(?:learning\/(?:archives|source-blobs)\/[a-f0-9]{64}\.(?:sqlite3|data)|api-teachers\/archives\/[a-f0-9]{64}\.sqlite3)$/.test(logical);
      if (sealedLearning && await hashFile(source, signal) !== basename(source).split(".")[0]) throw new Error("An immutable learning archive changed.");
      if (/\.sqlite3?$/.test(logical) && !logical.startsWith("portable-learning-workspace/") && !sealedLearning) {
        temporary = join(directory, `${randomUUID()}.sqlite-snapshot`);
        const wal = await lstat(`${source}-wal`).catch(() => undefined);
        await this.hooks.reserve(info.size + (wal?.size ?? 0), "consistent parent SQLite snapshot");
        await snapshotSqlite(source, temporary, signal);
        captured = temporary;
      }
      const hash = await hashFile(captured, signal);
      const bytes = (await lstat(captured)).size;
      const destination = join(blobs, hash);
      if (!await exists(destination)) {
        await this.hooks.reserve(bytes, "complete parent state blob");
        signal.throwIfAborted();
        await copyBlob(captured, destination, hash, signal);
      } else if (await hashFile(destination, signal) !== hash) {
        throw new Error("A deduplicated parent blob failed verification.");
      }
      db.exec("BEGIN IMMEDIATE");
      try {
        db.prepare("INSERT OR IGNORE INTO blobs(sha256,bytes) VALUES (?,?)").run(hash, bytes);
        db.prepare("UPDATE files SET sha256=?,bytes=? WHERE parent_id=? AND path=?").run(hash, bytes, parentId, logical);
        db.exec("COMMIT");
      } catch (error) { db.exec("ROLLBACK"); throw error; }
    } finally { if (temporary) await rm(temporary, { force: true }); }
  }

  private assemblyInput(job: Job, db: DatabaseSync, target: string, blobs: string, signal: AbortSignal): FullMergeAssemblyInput {
    return { targetBrainId: job.targetBrainId, targetDirectory: target, request: job.request, parents: job.parents,
      files: (parentId) => (function* () {
        const sql = parentId ? "SELECT * FROM files WHERE parent_id=? ORDER BY path" : "SELECT * FROM files ORDER BY parent_id,path";
        const statement = db.prepare(sql);
        const rows = parentId ? statement.iterate(identifier(parentId)) : statement.iterate();
        for (const row of rows) yield { parentId: String(row.parent_id), path: String(row.path), sha256: String(row.sha256), bytes: Number(row.bytes), mode: Number(row.mode) };
      })(),
      blobPath: (hash) => { if (!HASH.test(hash)) throw new Error("Invalid merge blob digest."); return join(blobs, hash); },
      materializeParent: async (parentId) => {
        identifier(parentId);
        if (!job.parents.some((parent) => parent.brainId === parentId)) throw new Error("This parent is not in the reviewed merge.");
        const destinationRoot = join(target, "merge", "parents", parentId);
        await mkdir(destinationRoot, { recursive: true });
        for (const row of db.prepare("SELECT path,sha256,bytes,mode FROM files WHERE parent_id=? ORDER BY path").iterate(parentId)) {
          signal.throwIfAborted();
          const parts = String(row.path).split("/");
          if (parts.some((part) => !part || part === "." || part === ".." || part.includes("\\"))) throw new Error("Invalid parent materialization path.");
          const path = join(destinationRoot, ...parts);
          const hash = String(row.sha256);
          if (!HASH.test(hash)) throw new Error("Parent snapshot is incomplete.");
          if (await exists(path)) {
            if (await hashFile(path, signal) !== hash) throw new Error("The materialized parent snapshot changed.");
            continue;
          }
          await this.hooks.reserve(Number(row.bytes), "private native parent materialization");
          await mkdir(dirname(path), { recursive: true });
          await copyBlob(join(blobs, hash), path, hash, signal);
          const handle = await open(path, "r+");
          try { await handle.chmod(Number(row.mode)); } finally { await handle.close(); }
        }
        return destinationRoot;
      },
      signal, reserve: (bytes, purpose) => this.hooks.reserve(bytes, purpose), admit: this.hooks.admit,
      checkpoint: async () => { signal.throwIfAborted(); this.hooks.onProgress?.(this.progress(job, db)); },
    };
  }

  private async writeSnapshotIndex(job: Job, db: DatabaseSync, target: string, signal: AbortSignal): Promise<void> {
    const path = join(target, "merge", "parents.jsonl");
    const handle = await open(path + ".next", "w", 0o600);
    try {
      await handle.write(canonical({ format: "omni-full-merge-parent-index", formatVersion: 1, parents: job.parents }) + "\n");
      for (const file of this.assemblyInput(job, db, target, join(target, "merge", "blobs"), signal).files()) {
        signal.throwIfAborted();
        const line = canonical(file) + "\n";
        await this.hooks.reserve(Buffer.byteLength(line), "parent provenance index");
        await handle.write(line);
      }
      await handle.sync();
    } finally { await handle.close(); }
    await rename(path + ".next", path);
  }

  private async verifyBlobs(db: DatabaseSync, blobs: string, signal: AbortSignal): Promise<void> {
    for (const row of db.prepare("SELECT sha256,bytes FROM blobs ORDER BY sha256").iterate()) {
      const hash = String(row.sha256);
      const path = join(blobs, hash);
      if ((await lstat(path)).size !== Number(row.bytes) || await hashFile(path, signal) !== hash) {
        throw new Error("Complete merge verification found changed parent bytes.");
      }
    }
  }

  /** Pause preserves its exact copy/assembly cursor for start(same request). */
  pause(operationId: string): boolean {
    const controller = this.running.get(identifier(operationId));
    if (!controller) return false;
    controller.abort(new Error("Full merge paused by user."));
    return true;
  }

  /** Explicit cancel removes only this unpublished staging directory. */
  async cancel(operationId: string): Promise<void> {
    identifier(operationId);
    if (this.running.has(operationId)) throw new Error("Pause this merge and wait for its acknowledgement before deleting staged output.");
    const directory = this.operation(operationId);
    if (!await exists(directory)) return;
    const job = JSON.parse(await readFile(join(directory, "job.json"), "utf8")) as Job;
    if (job.format !== FORMAT || job.request.operationId !== operationId || basename(directory) !== operationId) {
      throw new Error("Refusing to remove an unrecognized merge directory.");
    }
    if (job.state === "complete" || await exists(join(this.root, identifier(job.targetBrainId)))) {
      throw new Error("This merge has a published instance; use normal instance deletion.");
    }
    const stat = await lstat(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink() || dirname(await realpath(directory)) !== await realpath(this.operations)) {
      throw new Error("Refusing to remove a merge path outside its exact staging directory.");
    }
    const receipt = { ...await this.status(operationId), state: "cancelled" as const };
    const receipts = join(this.root, ".full-merge-cancellations");
    await mkdir(receipts, { recursive: true });
    await writeMutableFileIsolated(join(receipts, `${operationId}.json`), Buffer.from(canonical(receipt)));
    await rm(directory, { recursive: true, force: false });
    this.hooks.onProgress?.(receipt);
  }
}
