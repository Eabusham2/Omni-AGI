import { createHash, randomUUID } from "node:crypto";
import { constants, createReadStream } from "node:fs";
import { copyFile, lstat, mkdir, open, opendir, readFile, realpath, rename, rm, stat } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { DatabaseSync } from "node:sqlite";
import { DatasetManifestStore } from "./dataIngestion";
import { copyMutableFileIsolated, snapshotMutableSqliteIsolated, writeMutableFileIsolated } from "./mutableFileIsolation";
import { openLearningRelocations, type LearningSourceRelocation } from "./learningSourceRelocation";
import { learningHash, learningId } from "./learningTaskStore";
import { userQuestionRequestSha256, validateUserQuestionRecord } from "./userQuestions";
import { validateWebLearningPolicy } from "./webLearningPolicy";
import type { LearningTask } from "../shared/learning";
import { rekeyClockWorkspace, validateClockWorkspace } from "./clockPortability";
import { initializeInstanceVoice, rekeyVoiceWorkspace, validateVoiceWorkspace } from "./voicePreferencesStore";
import { rekeyTaskAttachmentWorkspace, validateTaskAttachmentWorkspace } from "./taskAttachmentPortability";
import { withFileHandoffSnapshot, prepareFileHandoffCustody, rekeyFileHandoffCustody } from "./fileToolHandoffCustody";
import { safeSteerDirectionWorkspacePath, rekeySteerDirectionWorkspace, validateSteerDirectionWorkspace } from "./steerDirectionPortability";
import { rekeyAutomationWorkspace, safeAutomationWorkspacePath, validateAutomationWorkspace } from "./automationPortability";
import { safeAssemblyWorkspacePath, validateAssemblyWorkspace } from "./assemblyPortability";
import { withMcpAppArtifactsLease, rebindMcpAppArtifactOwner, validateMcpAppArtifactStore } from "./mcpAppArtifacts";
import { rekeyHtmlArtifactWorkspace, safeHtmlArtifactWorkspacePath, validateHtmlArtifactWorkspace } from "./htmlArtifactPortability";
import { rekeyImaginationWorkspace, safeImaginationWorkspacePath, validateImaginationWorkspace, withImaginationWorkspaceBoundary } from "./imaginationCaptureWorkspace";
import { rekeyEnvironmentWorkspace, safeEnvironmentWorkspacePath, validateEnvironmentWorkspace } from "./environmentPortability";
import { withEnvironmentSnapshot } from "./environmentWorkspaceLease";
import { safeReplayWorkspacePath, validateReplayWorkspace, rekeyReplayWorkspace, withReplayWorkspaceBoundary } from "./imaginationAssemblyWorkspace";
import { rekeyClipboardWorkspace, safeClipboardWorkspacePath, validateClipboardWorkspace } from "./clipboardPortability";
import { withClipboardSnapshot } from "./clipboardWorkspaceLease";
import { rekeyLocalTeacherWorkspace, validateLocalTeacherWorkspace } from "./localTeacherPortability";
import { safeLocalTeacherPath } from "./localTeacherArtifacts";
import { withLocalTeacherSnapshot } from "./localTeacherWorkspaceBoundary";
import { rekeyExternalAdviceWorkspace, safeExternalAdviceWorkspacePath, validateCapturedExternalAdviceWorkspace, validateExternalAdvicePortabilityWorkspace } from "./externalAdvicePortability";
import { rekeyPluginRuntimeWorkspace, safePluginRuntimeWorkspacePath, validateCapturedPluginRuntimeWorkspace, validatePluginRuntimeWorkspace } from "./pluginRuntimePortability";
import { rekeyPanelReceiptWorkspace, safePanelReceiptWorkspacePath, validateCapturedPanelReceipts, validatePanelReceiptWorkspace } from "./pluginPanelReceiptPortability";
import { safeSparseSubstrateTransferPath, validateSparseSubstrateTransferWorkspace, adoptSparseSubstrateTransferWorkspace } from "./sparseSubstrateTransferPortability";
import { safeApiTeacherWorkspacePath, validateApiTeacherWorkspace, validateCapturedApiTeacherWorkspace, rekeyApiTeacherWorkspace, captureApiTeacherWorkspace } from "./apiTeacherPortability";
import type { ApiTeacherMediaIo } from "./apiTeacherMedia";
import { sqliteFilename } from "./sqliteFilename";

const SHA = /^[a-f0-9]{64}$/;
const hash = (value: string): string => createHash("sha256").update(value).digest("hex");
async function fileHash(path: string): Promise<string> {
  const digest = createHash("sha256"); for await (const chunk of createReadStream(path)) digest.update(chunk); return digest.digest("hex");
}
async function exists(path: string): Promise<boolean> {
  try { await lstat(path); return true; } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; }
}
function logicalPath(value: unknown): string {
  if (typeof value !== "string" || (!/^(?:learning|datasets|questions|mcp-apps|attachments)\//.test(value) && value !== "clock/workspace.sqlite3" && value !== "voice/workspace.sqlite3" && !safeSteerDirectionWorkspacePath(value) && !safeAutomationWorkspacePath(value) && !safeAssemblyWorkspacePath(value) && !safeHtmlArtifactWorkspacePath(value) && !safeImaginationWorkspacePath(value) && !safeEnvironmentWorkspacePath(value) && !safeReplayWorkspacePath(value) && !safeClipboardWorkspacePath(value) && !safeLocalTeacherPath(value) && !safeExternalAdviceWorkspacePath(value) && !safePluginRuntimeWorkspacePath(value) && !safePanelReceiptWorkspacePath(value) && !safeSparseSubstrateTransferPath(value) && !safeApiTeacherWorkspacePath(value)) || value.includes("\\") || value.includes("\0") ||
      value.split("/").some(part => !part || part === "." || part === "..")) throw new Error("Unsafe portable learning workspace path.");
  return value;
}
async function* filesIn(root: string, prefix = "", opaque = false): AsyncGenerator<{ name: string; path: string }> {
  if (!await exists(root)) return;
  const info = await lstat(root); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Learning workspace must be an owned real directory.");
  const directory = await opendir(root);
  for await (const entry of directory) {
    const name = prefix ? `${prefix}/${entry.name}` : entry.name, path = join(root, entry.name);
    if (entry.isSymbolicLink()) throw new Error("Learning workspace contains a filesystem link.");
    if (entry.isDirectory()) yield* filesIn(path, name, opaque);
    else if (entry.isFile()) { if (opaque || !/-(?:wal|shm|journal)$/.test(entry.name)) yield { name, path }; }
    else throw new Error("Learning workspace contains a nonregular payload.");
  }
}
export interface LearningWorkspaceHooks { signal?: AbortSignal; checkDisk?(bytes: number): Promise<void>; checkpoint?(label: string): Promise<void>; targetCorePath?: string; targetMetadataPath?: string; admitMetadata?: ApiTeacherMediaIo["admitMetadata"]; retireCopiedTeacherAuthority?: true }
export interface LearningWorkspaceArchive { root: string; brainId: string; files: Map<string, string>; nativeCoreSha256?: string; nativeBindingSha256?: string; missingSources: number }
export class IncompleteLearningWorkspaceError extends Error {
  readonly code = "LEARNING_WORKSPACE_INCOMPLETE";
  constructor() { super("Portable learning inventory is incomplete; no finished workspace has been claimed."); }
}
async function neuralBinding(engine: string, metadataPath = join(engine, "brain.json")): Promise<string | undefined> {
  if (!await exists(join(engine, "core.safetensors"))) return undefined;
  let metadata: Record<string, any> = {};
  if (await exists(metadataPath)) {
    const bytes = (await lstat(metadataPath)).size, need = bytes * 12;
    if (!Number.isSafeInteger(need) || need > freemem() || need > getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) throw new Error("Learning checkpoint metadata awaits physical transfer RAM.");
    metadata = JSON.parse(await readFile(metadataPath, "utf8"));
  }
  return learningHash({ core: await fileHash(join(engine, "core.safetensors")),
    plasticity: await exists(join(engine, "plasticity.safetensors")) ? await fileHash(join(engine, "plasticity.safetensors")) : null,
    substrate: metadata.substrate?.persistence ?? null, mutable: metadata.mutable_state ?? null,
    connected: metadata.connected_regions ?? null, joint: metadata.ingestion_joint_generation ?? null });
}
export function safeLearningWorkspacePath(value: string): boolean { return value === "inventory.sqlite3" || /^blobs\/[a-f0-9]{64}\.data$/.test(value); }

export async function cloneLearningWorkspace(sourceRoot: string, targetRoot: string, sourceBrainId: string, targetBrainId: string,
  hooks: LearningWorkspaceHooks = {}): Promise<void> {
  const temporary = join(targetRoot, `.learning-copy-${randomUUID()}`);
  try {
    const archive = await snapshotLearningWorkspace(sourceRoot, temporary, sourceBrainId, hooks);
    if (archive) await adoptLearningWorkspace(archive, targetRoot, targetBrainId, hooks);
  } finally { await rm(temporary, { recursive: true, force: true }); }
}

/** Snapshot includes WAL commits, exact source bytes, questions and all raw history. */
export async function snapshotLearningWorkspace(sourceRoot: string, packageRoot: string, brainId: string,
  hooks: LearningWorkspaceHooks = {}): Promise<LearningWorkspaceArchive | undefined> {
  learningId(brainId);
  await mkdir(join(packageRoot, "blobs"), { recursive: true, mode: 0o700 });
  const indexPath = join(packageRoot, "inventory.sqlite3"), index = new DatabaseSync(sqliteFilename(indexPath));
  index.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA cache_size=-512; CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE files(path TEXT PRIMARY KEY,sha256 TEXT NOT NULL,bytes INTEGER NOT NULL); CREATE TABLE sources(manifest_id TEXT NOT NULL,entry_index INTEGER NOT NULL,body TEXT NOT NULL,blob_sha256 TEXT,error TEXT,PRIMARY KEY(manifest_id,entry_index)); CREATE TABLE generic_sources(path TEXT PRIMARY KEY,body TEXT NOT NULL,blob_sha256 TEXT NOT NULL);");
  index.prepare("INSERT INTO meta VALUES('format','omni-learning-workspace-archive-1')").run();
  index.prepare("INSERT INTO meta VALUES('brainId',?)").run(brainId);
  let missing = 0;
  const materialize = async (source: string, sqlite = false): Promise<{ sha256: string; bytes: number }> => {
    hooks.signal?.throwIfAborted();
    const info = await stat(source); if (!info.isFile()) throw new Error("Selected source is not a regular file.");
    const wal = sqlite ? await stat(`${source}-wal`).catch(() => undefined) : undefined;
    await hooks.checkDisk?.(info.size + (wal?.size ?? 0));
    const temporary = join(packageRoot, "blobs", `.copy-${randomUUID()}.data`);
    try {
      if (sqlite) {
        await snapshotMutableSqliteIsolated(source, temporary);
      } else await copyFile(source, temporary, constants.COPYFILE_FICLONE);
      const sha256 = await fileHash(temporary), bytes = (await stat(temporary)).size;
      const destination = join(packageRoot, "blobs", `${sha256}.data`);
      if (!await exists(destination)) await rename(temporary, destination);
      else if ((await stat(destination)).size !== bytes || await fileHash(destination) !== sha256) throw new Error("Portable learning blob changed.");
      return { sha256, bytes };
    } finally { await rm(temporary, { force: true }); }
  };
  try {
    // Tasks commit source progress only after its native receipt. Snapshot
    // tasks first, then sources/progress, then questions (answers commit first).
    const captureTree = async (name: string, opaque = false, capturedRoot = sourceRoot): Promise<void> => { for await (const file of filesIn(join(capturedRoot, name), name, opaque)) {
      hooks.signal?.throwIfAborted(); await hooks.checkpoint?.(`Saving ${file.name}`);
      const reader = await open(file.path, "r"); const header = Buffer.alloc(16);
      try { await reader.read(header, 0, 16, 0); } finally { await reader.close(); }
      const sealedReplayPlan = /^imagination-assemblies\/plans\/[a-f0-9]{64}\/(?:manifest|source)\.sqlite3$/.test(file.name);
      const sealed = sealedReplayPlan || /^(?:attachments\/blobs\/[a-f0-9]{64}\.data|learning\/(?:archives|source-blobs)\/[a-f0-9]{64}\.(?:sqlite3|data)|(?:directions|imagination|imagination-assemblies|local-teachers|api-teachers|plugin-runtime|plugin-panels)\/archives\/[a-f0-9]{64}\.sqlite3)$/.test(file.name);
      if (sealed && !sealedReplayPlan && await fileHash(file.path) !== basename(file.name).split(".")[0]) throw new Error("Sealed learning archive bytes changed.");
      // Bundle/plan and lineage SQLite bytes are hash-sealed payloads, not
      // mutable databases to rewrite through SQLite backup/VACUUM semantics.
      const exactSparseArtifact = safeSparseSubstrateTransferPath(file.name) && file.name !== "substrate-transfers/workspace.sqlite3";
      const payload = await materialize(file.path, !opaque && !sealed && !exactSparseArtifact && header.toString("binary") === "SQLite format 3\0");
      if (safeExternalAdviceWorkspacePath(file.name)) await validateCapturedExternalAdviceWorkspace(join(packageRoot, "blobs", `${payload.sha256}.data`), brainId);
      if (file.name === "api-teachers/workspace.sqlite3") await validateCapturedApiTeacherWorkspace(join(packageRoot, "blobs", `${payload.sha256}.data`), brainId, hooks);
      if (safePluginRuntimeWorkspacePath(file.name)) await validateCapturedPluginRuntimeWorkspace(join(packageRoot, "blobs", `${payload.sha256}.data`), file.name === "plugin-runtime/workspace.sqlite3" ? brainId : undefined);
      if (safePanelReceiptWorkspacePath(file.name)) await validateCapturedPanelReceipts(join(packageRoot, "blobs", `${payload.sha256}.data`), file.name === "plugin-panels/workspace.sqlite3" ? brainId : undefined);
      index.prepare("INSERT INTO files VALUES(?,?,?)").run(logicalPath(file.name), payload.sha256, payload.bytes);
    } };
    await validateClockWorkspace(sourceRoot, brainId);
    await validateVoiceWorkspace(sourceRoot, brainId);
    await validateTaskAttachmentWorkspace(sourceRoot, brainId);
    await validateSteerDirectionWorkspace(sourceRoot, brainId);
    await validateAutomationWorkspace(sourceRoot, brainId);
    await validateHtmlArtifactWorkspace(sourceRoot, brainId);
    for (const name of ["learning", "datasets", "questions", "clock"]) await captureTree(name);
    await captureTree("voice");
    await withFileHandoffSnapshot(sourceRoot, async () => {
      await prepareFileHandoffCustody(sourceRoot, brainId, { signal: hooks.signal,
        reserve: async (_root, bytes) => { await hooks.checkDisk?.(bytes); } });
      await captureTree("attachments");
    }, hooks.signal);
    await captureTree("directions");
    await withImaginationWorkspaceBoundary(sourceRoot, async () => { await validateImaginationWorkspace(sourceRoot, brainId); await captureTree("imagination"); });
    await withReplayWorkspaceBoundary(sourceRoot, async () => { await validateReplayWorkspace(sourceRoot, brainId); await captureTree("imagination-assemblies"); });
    await captureTree("automation");
    await withEnvironmentSnapshot(sourceRoot, async () => { await validateEnvironmentWorkspace(sourceRoot, brainId); await captureTree("environments"); });
    await withClipboardSnapshot(sourceRoot, async () => { await validateClipboardWorkspace(sourceRoot, brainId); await captureTree("clipboard"); });
    await withLocalTeacherSnapshot(sourceRoot, async () => { await validateLocalTeacherWorkspace(sourceRoot, brainId); await captureTree("local-teachers"); });
    await validateExternalAdvicePortabilityWorkspace(sourceRoot, brainId); await captureTree("external-advice");
    const teacherStage = join(packageRoot, `.api-teacher-custody-${randomUUID()}`);
    try {
      const captured = await captureApiTeacherWorkspace(sourceRoot, teacherStage, brainId, hooks);
      if (captured.size) await captureTree("api-teachers", false, teacherStage);
    } finally { await rm(teacherStage, { recursive: true, force: true }); }
    await validatePluginRuntimeWorkspace(sourceRoot, brainId); await captureTree("plugin-runtime");
    await validatePanelReceiptWorkspace(sourceRoot, brainId); await captureTree("plugin-panels");
    await validateSparseSubstrateTransferWorkspace(sourceRoot, brainId, hooks); await captureTree("substrate-transfers");
    await captureTree("html-artifacts");
    await validateAssemblyWorkspace(sourceRoot); await captureTree("assembly", true);
    await validateMcpAppArtifactStore(sourceRoot, brainId, { signal: hooks.signal, checkDisk: hooks.checkDisk });
    await withMcpAppArtifactsLease(sourceRoot, () => captureTree("mcp-apps"));
    const datasets = new DatasetManifestStore(() => sourceRoot);
    const relocations = await openLearningRelocations(sourceRoot, brainId);
    try {
    const manifests = index.prepare("SELECT path FROM files WHERE path GLOB 'datasets/*/manifest.json'");
    for (const row of manifests.iterate()) {
      const match = /^datasets\/([A-Za-z0-9._-]+)\/manifest\.json$/.exec(String(row.path));
      if (!match) continue;
      const manifest = await datasets.manifest(brainId, match[1]!);
      for await (const entry of datasets.entries(brainId, manifest.id, 0, false)) {
        if (entry.rejection) continue;
        const oldBinding = relocations?.source(manifest.id, entry.index);
        const originalPath = entry.path;
        let sourcePath = entry.path, unavailable = false;
        if (manifest.portableSources && !oldBinding) unavailable = true;
        else if (oldBinding) {
          try { sourcePath = (await relocations!.resolve(manifest.id, entry)).path; }
          catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; unavailable = true; }
        }
        const resolvedSourcePath = unavailable ? sourcePath : await realpath(sourcePath).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return sourcePath; throw error; });
        const identityPaths = [...new Set([originalPath, ...(oldBinding?.identityPaths ?? []), sourcePath, resolvedSourcePath])];
        const source: Omit<LearningSourceRelocation, "relativePath"> = { originalPath,
          originalSourceIdentity: hash(`file\0${resolvedSourcePath}`), identityPaths,
          sourceName: oldBinding?.sourceName ?? basename(originalPath), contentSha256: String(entry.contentSha256), bytes: entry.bytes,
          originalManifestId: manifest.continuationIdentity?.manifestId ?? manifest.id,
          ...(oldBinding?.parentId ? { parentId: oldBinding.parentId } : {}) };
        let blob: string | null = null, error: string | null = null;
        try {
          if (unavailable) throw Object.assign(new Error("Retained source is unavailable"), { code: "ENOENT" });
          const payload = await materialize(sourcePath);
          if (payload.sha256 !== entry.contentSha256 || payload.bytes !== entry.bytes) throw new Error("Selected source differs from its canonical manifest.");
          blob = payload.sha256;
        } catch (cause) {
          hooks.signal?.throwIfAborted();
          if ((cause as NodeJS.ErrnoException).code !== "ENOENT") throw cause;
          missing++; error = "The original selected source is unavailable; choose its exact bytes before resuming.";
        }
        index.prepare("INSERT INTO sources VALUES(?,?,?,?,?)").run(manifest.id, entry.index, JSON.stringify(source), blob, error);
      }
    }
    // Active web leases are owned immutable source spools. Preserve the URL
    // and old physical identity so a paused web/media cursor can be rekeyed.
    for (const row of index.prepare("SELECT path FROM files WHERE path GLOB 'datasets/web-cache/*.lease.json'").iterate()) {
      const name = String(row.path), lease = JSON.parse(await readFile(join(sourceRoot, ...name.split("/")), "utf8"));
      if (lease.schemaVersion !== 1 || typeof lease.url !== "string" || !SHA.test(String(lease.contentHash)) ||
          !/^[a-f0-9]{64}\.[a-z0-9]{1,12}$/.test(String(lease.file)) || hash(lease.url) !== lease.file.split(".")[0] ||
          !Number.isSafeInteger(lease.bytes) || lease.bytes < 0) throw new Error("Portable web source lease is invalid.");
      const relativePath = `${dirname(name).replaceAll("\\", "/")}/${lease.file}`, path = join(sourceRoot, ...relativePath.split("/")), old = relocations?.generic(path);
      const payload = index.prepare("SELECT sha256,bytes FROM files WHERE path=?").get(relativePath);
      if (!payload || payload.sha256 !== lease.contentHash || payload.bytes !== lease.bytes) throw new Error("Portable web lease source bytes disagree.");
      const resolvedSourcePath = await realpath(path);
      const source: LearningSourceRelocation = { originalPath: old?.originalPath ?? path, originalSourceIdentity: hash(`file\0${resolvedSourcePath}`),
        identityPaths: [...new Set([old?.originalPath ?? path, ...(old?.identityPaths ?? []), path, resolvedSourcePath])], sourceName: lease.url,
        contentSha256: lease.contentHash, bytes: lease.bytes, relativePath, ...(old?.parentId ? { parentId: old.parentId } : {}) };
      index.prepare("INSERT INTO generic_sources VALUES(?,?,?)").run(relativePath, JSON.stringify(source), payload.sha256 as string);
    }
    } finally { relocations?.close(); }
    if (await exists(join(sourceRoot, "engine", "core.safetensors"))) index.prepare("INSERT INTO meta VALUES('nativeCoreSha256',?)").run(await fileHash(join(sourceRoot, "engine", "core.safetensors")));
    const binding = await neuralBinding(join(sourceRoot, "engine"));
    if (binding) index.prepare("INSERT INTO meta VALUES('nativeBindingSha256',?)").run(binding);
    index.prepare("INSERT INTO meta VALUES('missingSources',?)").run(String(missing));
    index.prepare("INSERT INTO meta VALUES('complete','1')").run();
    index.exec("PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;");
  } finally { index.close(); }
  return learningWorkspaceArchive(packageRoot, brainId);
}

export async function learningWorkspaceArchive(root: string, expectedBrainId?: string): Promise<LearningWorkspaceArchive | undefined> {
  if (!await exists(join(root, "inventory.sqlite3"))) {
    if (await exists(root)) for await (const _ of filesIn(root)) throw new Error("Learning workspace payload has no inventory.");
    return undefined;
  }
  const files = new Map<string, string>(), inventoryPath = join(root, "inventory.sqlite3"), info = await lstat(inventoryPath);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Unsafe portable learning inventory.");
  const index = new DatabaseSync(sqliteFilename(inventoryPath), { readOnly: true });
  try {
    if (index.prepare("PRAGMA quick_check").get()?.quick_check !== "ok" || index.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== "omni-learning-workspace-archive-1") throw new Error("Portable learning inventory failed integrity validation.");
    if (index.prepare("SELECT value FROM meta WHERE key='complete'").get()?.value !== "1") throw new IncompleteLearningWorkspaceError();
    const brainId = learningId(index.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value);
    if (expectedBrainId && brainId !== expectedBrainId) throw new Error("Portable learning inventory has the wrong owner.");
    const wanted = new Map<string, number>();
    for (const row of index.prepare("SELECT path,sha256,bytes FROM files").iterate()) {
      logicalPath(row.path); if (!SHA.test(String(row.sha256)) || !Number.isSafeInteger(row.bytes) || Number(row.bytes) < 0) throw new Error("Portable learning file descriptor is invalid.");
      if (wanted.has(String(row.sha256)) && wanted.get(String(row.sha256)) !== row.bytes) throw new Error("Portable learning bytes disagree.");
      wanted.set(String(row.sha256), Number(row.bytes));
    }
    for (const row of index.prepare("SELECT body,blob_sha256 FROM sources WHERE blob_sha256 IS NOT NULL").iterate()) {
      const body = JSON.parse(String(row.body));
      if (!SHA.test(String(row.blob_sha256)) || body.contentSha256 !== row.blob_sha256 || !Number.isSafeInteger(body.bytes) || body.bytes < 0) throw new Error("Portable source descriptor is invalid.");
      wanted.set(String(row.blob_sha256), body.bytes);
    }
    files.set("inventory.sqlite3", inventoryPath);
    for (const [sha, bytes] of wanted) {
      const path = join(root, "blobs", `${sha}.data`), file = await lstat(path);
      if (!file.isFile() || file.isSymbolicLink() || file.size !== bytes || await fileHash(path) !== sha) throw new Error("Portable learning source bytes failed verification.");
      files.set(`blobs/${sha}.data`, path);
    }
    for await (const file of filesIn(root)) if (!files.has(file.name)) throw new Error("Portable learning workspace contains an unlisted payload.");
    const nativeCoreSha256 = index.prepare("SELECT value FROM meta WHERE key='nativeCoreSha256'").get()?.value;
    const nativeBindingSha256 = index.prepare("SELECT value FROM meta WHERE key='nativeBindingSha256'").get()?.value;
    if (nativeCoreSha256 !== undefined && !SHA.test(String(nativeCoreSha256))) throw new Error("Portable learning neural binding is invalid.");
    if (nativeBindingSha256 !== undefined && !SHA.test(String(nativeBindingSha256))) throw new Error("Portable learning generation binding is invalid.");
    const missingSources = Number(index.prepare("SELECT value FROM meta WHERE key='missingSources'").get()?.value ?? 0);
    if (!Number.isSafeInteger(missingSources) || missingSources < 0 || missingSources !== Number(index.prepare("SELECT COUNT(*) AS n FROM sources WHERE blob_sha256 IS NULL").get()?.n)) throw new Error("Portable source absence count is invalid.");
    return { root, brainId, files, nativeCoreSha256: nativeCoreSha256 === undefined ? undefined : String(nativeCoreSha256), nativeBindingSha256: nativeBindingSha256 === undefined ? undefined : String(nativeBindingSha256), missingSources };
  } finally { index.close(); }
}

/** Install dormant work under a new owner; no question/timer/effect is replayed. */
export async function adoptLearningWorkspace(archive: LearningWorkspaceArchive, targetRoot: string, targetBrainId: string,
  hooks: LearningWorkspaceHooks = {}): Promise<void> {
  learningId(targetBrainId);
  const index = new DatabaseSync(sqliteFilename(join(archive.root, "inventory.sqlite3")), { readOnly: true });
  const copiedAt = new Date().toISOString();
  try {
    for (const row of index.prepare("SELECT path,sha256,bytes FROM files ORDER BY path").iterate()) {
      const name = logicalPath(row.path), destination = join(targetRoot, ...name.split("/"));
      hooks.signal?.throwIfAborted(); await hooks.checkDisk?.(Number(row.bytes));
      await copyMutableFileIsolated(join(archive.root, "blobs", `${row.sha256}.data`), destination, { expectedSha256: String(row.sha256) });
    }
    await rebindMcpAppArtifactOwner(targetRoot, archive.brainId, targetBrainId);
    const taskPath = join(targetRoot, "learning", "tasks.sqlite3");
    const targetCore = hooks.targetCorePath ?? join(targetRoot, "engine", "core.safetensors");
    const nativeMatches = !archive.nativeCoreSha256 || await exists(targetCore) && await fileHash(targetCore) === archive.nativeCoreSha256 &&
      (!archive.nativeBindingSha256 || await neuralBinding(dirname(targetCore), hooks.targetMetadataPath) === archive.nativeBindingSha256);
    if (await exists(taskPath)) {
      const tasks = new DatabaseSync(sqliteFilename(taskPath));
      try {
        if (tasks.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== archive.brainId) throw new Error("Copied learning tasks lost their source owner.");
        tasks.exec("BEGIN IMMEDIATE");
        tasks.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetBrainId);
        for (const row of tasks.prepare("SELECT id,body FROM tasks").all()) {
          const task = JSON.parse(String(row.body)) as LearningTask;
          if (task.brainId !== archive.brainId || task.id !== row.id) throw new Error("Copied learning task identity failed validation.");
          validateWebLearningPolicy(task.policy); task.brainId = targetBrainId; delete task.pausedForUnload;
          if (["queued", "running", "waiting"].includes(task.state)) { task.state = "paused"; task.phase = "copied-instance"; task.label = "Saved work retained · resume when ready"; }
          if (!nativeMatches && task.state !== "complete") { task.state = "paused"; task.phase = "neural-generation-mismatch"; task.resumeBlocked = { reason: "neural-generation-mismatch", sourceCoreSha256: archive.nativeCoreSha256! }; task.label = "The selected neural origin differs from this task's learned prefix; restore its matching checkpoint before resuming"; }
          else if (nativeMatches) delete task.resumeBlocked;
          tasks.prepare("UPDATE tasks SET body=? WHERE id=?").run(JSON.stringify(task), row.id as string);
        }
        for (const row of tasks.prepare("SELECT task_id,id,body FROM sources").all()) {
          const source = JSON.parse(String(row.body)); if (source.state === "running") source.state = "paused";
          tasks.prepare("UPDATE sources SET body=? WHERE task_id=? AND id=?").run(JSON.stringify(source), row.task_id as string, row.id as string);
        }
        tasks.exec("COMMIT");
      } catch (error) { try { tasks.exec("ROLLBACK"); } catch { /* No active transaction. */ } throw error; }
      finally { tasks.close(); }
    }
    const datasets = new DatasetManifestStore(() => targetRoot);
    for (const row of index.prepare("SELECT path FROM files WHERE path GLOB 'datasets/*/manifest.json'").iterate()) {
      const match = /^datasets\/([A-Za-z0-9._-]+)\/manifest\.json$/.exec(String(row.path)); if (!match) continue;
      const path = join(targetRoot, ...String(row.path).split("/")), manifest = JSON.parse(await readFile(path, "utf8"));
      if (manifest.brainId !== archive.brainId) throw new Error("Retained dataset has the wrong owner.");
      manifest.brainId = targetBrainId; manifest.portableSources = true;
      await writeMutableFileIsolated(path, Buffer.from(JSON.stringify(manifest)));
      await datasets.manifest(targetBrainId, match[1]!);
    }
    const oldRelocations = join(targetRoot, "learning", "source-relocations.sqlite3");
    if (await exists(oldRelocations)) {
      const database = new DatabaseSync(sqliteFilename(oldRelocations));
      try { if (database.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== archive.brainId) throw new Error("Retained source bindings have the wrong owner."); database.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetBrainId); }
      finally { database.close(); }
    }
    const relocations = await openLearningRelocations(targetRoot, targetBrainId, true);
    try {
      for (const row of index.prepare("SELECT manifest_id,entry_index,body,blob_sha256 FROM sources").iterate()) {
        if (!row.blob_sha256) continue;
        const source = JSON.parse(String(row.body)), destination = join(targetRoot, "learning", "source-blobs", `${row.blob_sha256}.data`);
        await hooks.checkDisk?.(source.bytes); hooks.signal?.throwIfAborted();
        await copyMutableFileIsolated(join(archive.root, "blobs", `${row.blob_sha256}.data`), destination, { expectedSha256: String(row.blob_sha256) });
        // Root rebinding changes only physical bytes, never the canonical
        // manifest entries or native transaction identity.
        const old = relocations!.source(String(row.manifest_id), Number(row.entry_index));
        const identityPaths = [...new Set([source.originalPath, ...(source.identityPaths ?? []), ...(old?.identityPaths ?? []),
          ...(old ? [join(targetRoot, ...old.relativePath.split("/"))] : [])])];
        relocations!.adopt(String(row.manifest_id), Number(row.entry_index), { ...source, identityPaths, relativePath: `learning/source-blobs/${row.blob_sha256}.data` });
      }
      for (const row of index.prepare("SELECT path,body FROM generic_sources").iterate()) {
        const source = JSON.parse(String(row.body)); relocations!.addGeneric(source);
      }
    } finally { relocations?.close(); }
    for await (const file of filesIn(join(targetRoot, "datasets", "crawls"))) if (file.name.endsWith(".sqlite3")) {
      const crawl = new DatabaseSync(sqliteFilename(file.path));
      try {
        const owner = crawl.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value;
        if (owner !== archive.brainId) throw new Error("Copied web frontier has the wrong owner.");
        crawl.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetBrainId);
      } finally { crawl.close(); }
    }
    for await (const file of filesIn(join(targetRoot, "questions", "records"))) {
      const old = validateUserQuestionRecord(JSON.parse(await readFile(file.path, "utf8")), archive.brainId);
      const next = { ...old, brainId: targetBrainId, lineage: [...(old.lineage ?? []), { brainId: archive.brainId, questionId: old.id, copiedAt }],
        suspended: { reason: "copied-instance" as const, fromBrainId: archive.brainId, copiedAt } };
      next.requestSha256 = userQuestionRequestSha256({ ...next, requestId: next.requestId });
      validateUserQuestionRecord(next, targetBrainId, old.id);
      await writeMutableFileIsolated(file.path, Buffer.from(JSON.stringify(next)));
    }
    await rekeyClockWorkspace(targetRoot, archive.brainId, targetBrainId);
    await rekeyVoiceWorkspace(targetRoot, archive.brainId, targetBrainId);
    // Missing settings on a copied identity use Auto, not unrelated local
    // renderer-global legacy preferences.
    await initializeInstanceVoice(targetRoot, targetBrainId, "copy-auto");
    await rekeyTaskAttachmentWorkspace(targetRoot, archive.brainId, targetBrainId);
    await rekeyFileHandoffCustody(targetRoot, targetBrainId, { signal: hooks.signal,
      reserve: async (_root, bytes) => { await hooks.checkDisk?.(bytes); } });
    await rekeySteerDirectionWorkspace(targetRoot, archive.brainId, targetBrainId);
    await rekeyImaginationWorkspace(targetRoot, archive.brainId, targetBrainId);
    await rekeyReplayWorkspace(targetRoot, archive.brainId, targetBrainId);
    await rekeyAutomationWorkspace(targetRoot, archive.brainId, targetBrainId);
    await rekeyEnvironmentWorkspace(targetRoot, archive.brainId, targetBrainId);
    await rekeyClipboardWorkspace(targetRoot, archive.brainId, targetBrainId);
    await rekeyLocalTeacherWorkspace(targetRoot, archive.brainId, targetBrainId);
    await rekeyExternalAdviceWorkspace(targetRoot, archive.brainId, targetBrainId, hooks);
    await rekeyApiTeacherWorkspace(targetRoot, archive.brainId, targetBrainId, hooks);
    await rekeyPluginRuntimeWorkspace(targetRoot, archive.brainId, targetBrainId, hooks);
    await rekeyPanelReceiptWorkspace(targetRoot, archive.brainId, targetBrainId, hooks);
    await adoptSparseSubstrateTransferWorkspace(targetRoot, archive.brainId, targetBrainId,
      { signal: hooks.signal, reserveWrite: bytes => hooks.checkDisk?.(bytes) ?? Promise.resolve() });
    await rekeyHtmlArtifactWorkspace(targetRoot, archive.brainId, targetBrainId);
    await validateAssemblyWorkspace(targetRoot);
    const historyPath = join(targetRoot, "learning", "archives", `${await fileHash(join(archive.root, "inventory.sqlite3"))}.sqlite3`);
    await copyMutableFileIsolated(join(archive.root, "inventory.sqlite3"), historyPath);
    for (const [name, source] of archive.files) if (name.startsWith("blobs/")) {
      const sha = name.slice(6, -5), destination = join(targetRoot, "learning", "source-blobs", `${sha}.data`);
      if (!await exists(destination)) await copyMutableFileIsolated(source, destination, { expectedSha256: sha });
    }
  } finally { index.close(); }
}
