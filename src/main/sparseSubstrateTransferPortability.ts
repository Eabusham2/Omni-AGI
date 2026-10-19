import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, mkdir, opendir, readFile, rename, rm } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, sep } from "node:path";
import type { SparseSubstrateBundle, SparseSubstrateJob, SparseSubstratePlan } from "../shared/sparseSubstrateTransfer";
import { copyMutableFileIsolated, snapshotMutableSqliteIsolated, writeMutableFileIsolated } from "./mutableFileIsolation";
import { SPARSE_TRANSFER_DIRECTORY, SparseSubstrateTransferStore, sparseHash, sparseId,
  validateSparseTransferDatabaseFile } from "./sparseSubstrateTransferStore";
import { sqliteFilename } from "./sqliteFilename";

const WORKSPACE = `${SPARSE_TRANSFER_DIRECTORY}/workspace.sqlite3`;
const LINEAGE_FORMAT = "omni-sparse-substrate-lineage-1" as const;
const SHA256 = /^[a-f0-9]{64}$/;

export interface SparseSubstratePortabilityHooks {
  signal?: AbortSignal;
  reserveWrite?(bytes: number, purpose: string): Promise<void>;
}

export interface SparseSubstrateLineageFile {
  kind: "bundle" | "plan";
  id: string;
  relativePath: string;
  bytes: number;
  sha256: string;
}

export interface SparseSubstrateLineageManifest {
  format: typeof LINEAGE_FORMAT;
  version: 1;
  sourceBrainId: string;
  databaseOwner: string;
  workspaceSha256: string;
  workspaceBytes: number;
  files: SparseSubstrateLineageFile[];
}

export interface SparseSubstrateLineageDescriptor {
  sourceBrainId: string;
  databaseOwner: string;
  workspaceSha256: string;
  workspaceBytes: number;
  manifestSha256: string;
  relativePath: string;
  files: SparseSubstrateLineageFile[];
}

function compare(left: string, right: string): number { return left < right ? -1 : left > right ? 1 : 0; }
function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  return `{${Object.entries(value as Record<string, unknown>).sort(([left], [right]) => compare(left, right))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}`;
}
function hash(value: string | Uint8Array): string { return createHash("sha256").update(value).digest("hex"); }
async function exists(path: string): Promise<boolean> {
  return lstat(path).then(() => true, (error: NodeJS.ErrnoException) => error.code === "ENOENT" ? false : Promise.reject(error));
}
async function fileSha256(path: string, signal?: AbortSignal): Promise<string> {
  const before = await lstat(path);
  if (!before.isFile() || before.isSymbolicLink()) throw new Error("Sparse portability source is not a regular owned file.");
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path, { ...(signal ? { signal } : {}) })) {
    signal?.throwIfAborted(); digest.update(chunk as Buffer);
  }
  const after = await lstat(path);
  if (!after.isFile() || after.isSymbolicLink() || after.dev !== before.dev || after.ino !== before.ino ||
      after.size !== before.size || after.mtimeMs !== before.mtimeMs || after.ctimeMs !== before.ctimeMs) {
    throw new Error("Sparse portability source changed during complete hashing.");
  }
  signal?.throwIfAborted(); return digest.digest("hex");
}
function safeRelative(value: string): boolean {
  if (!value || value.includes("\0") || value.includes("\\") || isAbsolute(value)) return false;
  return value.split("/").every(part => part && part !== "." && part !== "..");
}
function expectedArtifactPath(kind: "bundle" | "plan", id: string): string {
  return `${SPARSE_TRANSFER_DIRECTORY}/${kind === "bundle" ? "bundles" : "plans"}/${sparseId(id)}.sqlite3`;
}
function lineageRoot(root: string, sourceBrainId: string, workspaceSha256: string): string {
  return join(root, SPARSE_TRANSFER_DIRECTORY, "lineage", sparseId(sourceBrainId), sparseHash(workspaceSha256));
}
function lineageRelative(sourceBrainId: string, workspaceSha256: string): string {
  return `${SPARSE_TRANSFER_DIRECTORY}/lineage/${sparseId(sourceBrainId)}/${sparseHash(workspaceSha256)}`;
}

export function safeSparseSubstrateTransferPath(value: string): boolean {
  if (value === WORKSPACE || /^substrate-transfers\/(?:bundles|plans)\/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.sqlite3$/.test(value)) return true;
  return /^substrate-transfers\/lineage\/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\/[a-f0-9]{64}\/(?:manifest\.json|workspace\.sqlite3|(?:bundles|plans)\/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.sqlite3)$/.test(value);
}

async function databaseOwner(path: string): Promise<string> {
  const { DatabaseSync } = await import("node:sqlite");
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Sparse portability database is unsafe.");
  const database = new DatabaseSync(sqliteFilename(path), { readOnly: true, allowExtension: false });
  try { database.exec("PRAGMA trusted_schema=OFF;"); return sparseId(database.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value); }
  finally { database.close(); }
}

async function artifactsFromDatabase(path: string, owner: string): Promise<Array<{ kind: "bundle" | "plan"; value: SparseSubstrateBundle | SparseSubstratePlan }>> {
  const store = await SparseSubstrateTransferStore.openFile(path, owner);
  try {
    const output: Array<{ kind: "bundle" | "plan"; value: SparseSubstrateBundle | SparseSubstratePlan }> = [];
    for (const row of store.db.prepare("SELECT id,kind FROM artifacts ORDER BY id").iterate()) {
      const kind = String(row.kind);
      if (kind !== "bundle" && kind !== "plan") throw new Error("Sparse lineage contains an unknown artifact kind.");
      output.push({ kind, value: store.artifact(String(row.id), kind) });
    }
    return output;
  } finally { store.close(); }
}

async function jobsFromDatabase(path: string, owner: string): Promise<SparseSubstrateJob[]> {
  const store = await SparseSubstrateTransferStore.openFile(path, owner);
  try { return [...store.db.prepare("SELECT id FROM jobs ORDER BY id").iterate()].map(row => store.job(String(row.id))); }
  finally { store.close(); }
}

async function validateLineage(root: string, sourceBrainId: string, workspaceSha256: string, signal?: AbortSignal): Promise<SparseSubstrateLineageDescriptor> {
  const directory = lineageRoot(root, sourceBrainId, workspaceSha256), manifestPath = join(directory, "manifest.json");
  const manifestBytes = await readFile(manifestPath); signal?.throwIfAborted();
  const manifest = JSON.parse(manifestBytes.toString("utf8")) as SparseSubstrateLineageManifest;
  const keys = ["databaseOwner", "files", "format", "sourceBrainId", "version", "workspaceBytes", "workspaceSha256"];
  if (!manifest || Object.keys(manifest).sort().join("\0") !== keys.sort().join("\0") ||
      `${canonical(manifest)}\n` !== manifestBytes.toString("utf8") || manifest.format !== LINEAGE_FORMAT || manifest.version !== 1 ||
      manifest.sourceBrainId !== sparseId(sourceBrainId) || manifest.workspaceSha256 !== sparseHash(workspaceSha256) ||
      sparseId(manifest.databaseOwner) !== manifest.databaseOwner || !Number.isSafeInteger(manifest.workspaceBytes) || manifest.workspaceBytes < 1 ||
      !Array.isArray(manifest.files)) throw new Error("Sparse lineage manifest identity is invalid.");
  const workspace = join(directory, "workspace.sqlite3"), workspaceInfo = await lstat(workspace);
  if (!workspaceInfo.isFile() || workspaceInfo.isSymbolicLink() || workspaceInfo.size !== manifest.workspaceBytes ||
      await fileSha256(workspace, signal) !== manifest.workspaceSha256) throw new Error("Sparse lineage database bytes changed.");
  await validateSparseTransferDatabaseFile(workspace, manifest.databaseOwner);
  const artifacts = await artifactsFromDatabase(workspace, manifest.databaseOwner), expected = new Map<string, SparseSubstrateLineageFile>();
  for (const { kind, value } of artifacts) {
    const relativePath = `${kind === "bundle" ? "bundles" : "plans"}/${value.id}.sqlite3`;
    expected.set(`${kind}\0${value.id}`, { kind, id: value.id, relativePath, bytes: value.bytes, sha256: value.sha256 });
  }
  if (manifest.files.length !== expected.size) throw new Error("Sparse lineage manifest lost an artifact descriptor.");
  const seen = new Set<string>();
  for (const file of manifest.files) {
    const key = `${file.kind}\0${file.id}`, descriptor = expected.get(key);
    if (!descriptor || seen.has(key) || canonical(file) !== canonical(descriptor) || !safeRelative(file.relativePath)) throw new Error("Sparse lineage artifact descriptor differs from its source database.");
    seen.add(key); const path = join(directory, file.relativePath), info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink() || info.size !== file.bytes || await fileSha256(path, signal) !== file.sha256) throw new Error("Sparse lineage artifact bytes changed.");
  }
  const known = new Set(["manifest.json", "workspace.sqlite3", ...manifest.files.map(file => file.relativePath)]);
  async function scan(current: string): Promise<void> {
    const entries = await opendir(current);
    for await (const entry of entries) {
      const path = join(current, entry.name), info = await lstat(path), local = relative(directory, path).split(sep).join("/");
      if (info.isSymbolicLink()) throw new Error("Sparse lineage contains a symbolic link.");
      if (info.isDirectory()) { await scan(path); continue; }
      if (!info.isFile() || !known.has(local)) throw new Error("Sparse lineage contains unindexed custody.");
    }
  }
  await scan(directory);
  return { sourceBrainId: manifest.sourceBrainId, databaseOwner: manifest.databaseOwner, workspaceSha256: manifest.workspaceSha256,
    workspaceBytes: manifest.workspaceBytes, manifestSha256: hash(manifestBytes), relativePath: lineageRelative(sourceBrainId, workspaceSha256),
    files: manifest.files.map(file => ({ ...file })) };
}

async function archiveWorkspace(sourceRoot: string, databaseBrainId: string, destinationRoot: string, sourceBrainId: string,
  hooks: SparseSubstratePortabilityHooks): Promise<SparseSubstrateLineageDescriptor> {
  const source = join(sourceRoot, WORKSPACE); hooks.signal?.throwIfAborted();
  await validateSparseSubstrateTransferWorkspace(sourceRoot, databaseBrainId, { signal: hooks.signal, validateLineage: false });
  const temporary = join(destinationRoot, SPARSE_TRANSFER_DIRECTORY, `.snapshot-${randomUUID()}.sqlite3`);
  try {
    const sourceInfo = await lstat(source); await hooks.reserveWrite?.(sourceInfo.size + 65_536, "sparse workspace lineage snapshot");
    await snapshotMutableSqliteIsolated(source, temporary); hooks.signal?.throwIfAborted();
    await validateSparseTransferDatabaseFile(temporary, databaseBrainId);
    const workspaceSha256 = await fileSha256(temporary, hooks.signal), workspaceBytes = (await lstat(temporary)).size;
    const finalDirectory = lineageRoot(destinationRoot, sourceBrainId, workspaceSha256);
    if (await exists(finalDirectory)) return validateLineage(destinationRoot, sourceBrainId, workspaceSha256, hooks.signal);
    const next = join(destinationRoot, SPARSE_TRANSFER_DIRECTORY, "lineage", sourceBrainId, `.${workspaceSha256}.${randomUUID()}.next`);
    try {
      await mkdir(next, { recursive: true, mode: 0o700 });
      await hooks.reserveWrite?.(workspaceBytes, "sparse lineage database");
      await copyMutableFileIsolated(temporary, join(next, "workspace.sqlite3"), { expectedSha256: workspaceSha256, mode: 0o600 });
      const files: SparseSubstrateLineageFile[] = [];
      for (const { kind, value } of await artifactsFromDatabase(temporary, databaseBrainId)) {
        hooks.signal?.throwIfAborted();
        const expected = expectedArtifactPath(kind, value.id);
        if (value.relativePath !== expected || !Number.isSafeInteger(value.bytes) || value.bytes < 1 || !SHA256.test(value.sha256)) {
          throw new Error("Sparse artifact descriptor escaped exact owned custody.");
        }
        const original = join(sourceRoot, value.relativePath), info = await lstat(original);
        if (!info.isFile() || info.isSymbolicLink() || info.size !== value.bytes || await fileSha256(original, hooks.signal) !== value.sha256) {
          throw new Error("Sparse artifact file differs from its immutable descriptor.");
        }
        const relativePath = `${kind === "bundle" ? "bundles" : "plans"}/${value.id}.sqlite3`, destination = join(next, relativePath);
        await hooks.reserveWrite?.(value.bytes, `sparse ${kind} lineage artifact`);
        await copyMutableFileIsolated(original, destination, { expectedSha256: value.sha256, mode: 0o600 });
        files.push({ kind, id: value.id, relativePath, bytes: value.bytes, sha256: value.sha256 });
      }
      files.sort((left, right) => compare(`${left.kind}\0${left.id}`, `${right.kind}\0${right.id}`));
      const manifest: SparseSubstrateLineageManifest = { format: LINEAGE_FORMAT, version: 1, sourceBrainId: sparseId(sourceBrainId),
        databaseOwner: sparseId(databaseBrainId), workspaceSha256, workspaceBytes, files };
      const bytes = Buffer.from(`${canonical(manifest)}\n`, "utf8");
      await hooks.reserveWrite?.(bytes.length, "sparse lineage manifest"); await writeMutableFileIsolated(join(next, "manifest.json"), bytes);
      await mkdir(dirname(finalDirectory), { recursive: true, mode: 0o700 });
      try { await rename(next, finalDirectory); }
      catch (error) { if (!await exists(finalDirectory)) throw error; }
    } finally { await rm(next, { recursive: true, force: true }); }
    return validateLineage(destinationRoot, sourceBrainId, workspaceSha256, hooks.signal);
  } finally { await rm(temporary, { force: true }); }
}

function projectedJob(job: SparseSubstrateJob, owner: string, sourceBrainId: string, namespace?: string): SparseSubstrateJob {
  const id = namespace ? hash(`sparse-job\0${namespace}\0${job.id}`) : job.id;
  const requestId = namespace ? hash(`sparse-request\0${namespace}\0${job.requestId}`) : job.requestId;
  const state = job.state === "applying" ? "uncertain" : "dormant";
  return { ...job, id, brainId: owner, originalBrainId: job.originalBrainId || sourceBrainId, requestId,
    authorizationRevision: hash(`dormant-sparse-authority\0${sourceBrainId}\0${job.authorizationRevision}\0${job.id}`),
    state, revision: job.revision + 1,
    error: `Historical ${job.state} sparse job from ${sourceBrainId}; copied authority and target plan are dormant and require a new preview/review.` };
}

async function projectJobs(databasePath: string, databaseBrainId: string, targetRoot: string, owner: string,
  sourceBrainId: string, namespace?: string): Promise<void> {
  const target = await SparseSubstrateTransferStore.open(targetRoot, owner);
  try {
    for (const job of await jobsFromDatabase(databasePath, databaseBrainId)) {
      const projected = projectedJob(job, owner, sourceBrainId, namespace), prior = target.db.prepare("SELECT payload FROM jobs WHERE id=?").get(projected.id);
      if (prior) { if (String(prior.payload) !== JSON.stringify(projected)) throw new Error("Sparse inherited job projection collision; no parent was selected silently."); continue; }
      target.createJob(projected);
    }
  } finally { target.close(); }
}

export async function validateSparseSubstrateTransferWorkspace(root: string, owner: string,
  options: { signal?: AbortSignal; validateLineage?: boolean } = {}): Promise<void> {
  sparseId(owner); const base = join(root, SPARSE_TRANSFER_DIRECTORY), workspace = join(root, WORKSPACE);
  const hasWorkspace = await exists(workspace);
  if (hasWorkspace) {
    await validateSparseTransferDatabaseFile(workspace, owner);
    const artifacts = await artifactsFromDatabase(workspace, owner), known = new Set<string>();
    for (const { kind, value } of artifacts) {
      const expected = expectedArtifactPath(kind, value.id); if (value.relativePath !== expected) throw new Error("Sparse current artifact path differs from its typed kind/id.");
      known.add(expected); const path = join(root, expected), info = await lstat(path);
      if (!info.isFile() || info.isSymbolicLink() || info.size !== value.bytes || await fileSha256(path, options.signal) !== value.sha256) throw new Error("Sparse current artifact checksum/count changed.");
    }
    for (const kind of ["bundles", "plans"] as const) {
      const directory = join(base, kind); if (!await exists(directory)) continue;
      const info = await lstat(directory); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Sparse current artifact directory is unsafe.");
      for await (const entry of await opendir(directory)) {
        const local = `${SPARSE_TRANSFER_DIRECTORY}/${kind}/${entry.name}`;
        if (!entry.isFile() || entry.isSymbolicLink() || !known.has(local)) throw new Error("Sparse current workspace contains an unindexed artifact.");
      }
    }
  } else {
    for (const kind of ["bundles", "plans"] as const) {
      const directory = join(base, kind); if (!await exists(directory)) continue;
      const info = await lstat(directory); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Sparse orphan artifact directory is unsafe.");
      for await (const _entry of await opendir(directory)) throw new Error("Sparse artifact custody exists without a current ownership database.");
    }
  }
  if (!await exists(base)) return;
  const baseInfo = await lstat(base); if (!baseInfo.isDirectory() || baseInfo.isSymbolicLink()) throw new Error("Sparse workspace root is unsafe.");
  for await (const entry of await opendir(base)) {
    if (["workspace.sqlite3", "workspace.sqlite3-journal", "workspace.sqlite3-wal", "workspace.sqlite3-shm", "bundles", "plans", "lineage"].includes(entry.name)) continue;
    throw new Error("Sparse workspace contains an unsupported custody path.");
  }
  if (options.validateLineage === false || !await exists(join(base, "lineage"))) return;
  const parents = await opendir(join(base, "lineage"));
  for await (const parent of parents) {
    if (!parent.isDirectory() || parent.isSymbolicLink()) throw new Error("Sparse lineage parent custody is unsafe.");
    sparseId(parent.name); const versions = await opendir(join(base, "lineage", parent.name));
    for await (const version of versions) {
      if (!version.isDirectory() || version.isSymbolicLink()) throw new Error("Sparse lineage version custody is unsafe.");
      await validateLineage(root, parent.name, sparseHash(version.name), options.signal);
    }
  }
}

export async function listSparseSubstrateLineage(root: string, signal?: AbortSignal): Promise<SparseSubstrateLineageDescriptor[]> {
  const lineage = join(root, SPARSE_TRANSFER_DIRECTORY, "lineage"), output: SparseSubstrateLineageDescriptor[] = [];
  if (!await exists(lineage)) return output;
  for await (const parent of await opendir(lineage)) {
    if (!parent.isDirectory() || parent.isSymbolicLink()) throw new Error("Sparse lineage parent custody is unsafe.");
    sparseId(parent.name); for await (const version of await opendir(join(lineage, parent.name))) {
      signal?.throwIfAborted(); if (!version.isDirectory() || version.isSymbolicLink()) throw new Error("Sparse lineage version custody is unsafe.");
      output.push(await validateLineage(root, parent.name, sparseHash(version.name), signal));
    }
  }
  return output.sort((left, right) => compare(`${left.sourceBrainId}\0${left.workspaceSha256}`, `${right.sourceBrainId}\0${right.workspaceSha256}`));
}

export async function resolveSparseSubstrateLineageArtifact(root: string, sourceBrainId: string, workspaceSha256: string,
  kind: "bundle" | "plan", id: string, signal?: AbortSignal): Promise<{ descriptor: SparseSubstrateLineageFile; path: string }> {
  const lineage = await validateLineage(root, sourceBrainId, workspaceSha256, signal), descriptor = lineage.files.find(file => file.kind === kind && file.id === sparseId(id));
  if (!descriptor) throw new Error("Sparse lineage artifact is not present in this exact source archive.");
  return { descriptor: { ...descriptor }, path: join(root, lineage.relativePath, descriptor.relativePath) };
}

async function copyExistingLineage(sourceRoot: string, targetRoot: string, hooks: SparseSubstratePortabilityHooks): Promise<void> {
  for (const source of await listSparseSubstrateLineage(sourceRoot, hooks.signal)) {
    hooks.signal?.throwIfAborted();
    const destination = lineageRoot(targetRoot, source.sourceBrainId, source.workspaceSha256);
    if (await exists(destination)) {
      const current = await validateLineage(targetRoot, source.sourceBrainId, source.workspaceSha256, hooks.signal);
      if (canonical(current) !== canonical(source)) throw new Error("Sparse lineage path collides with different parent custody.");
      continue;
    }
    const next = join(targetRoot, SPARSE_TRANSFER_DIRECTORY, "lineage", source.sourceBrainId,
      `.${source.workspaceSha256}.${randomUUID()}.next`);
    try {
      await mkdir(next, { recursive: true, mode: 0o700 });
      const sourceDirectory = join(sourceRoot, source.relativePath);
      const entries = [{ relativePath: "workspace.sqlite3", bytes: source.workspaceBytes, sha256: source.workspaceSha256 },
        { relativePath: "manifest.json", bytes: (await lstat(join(sourceDirectory, "manifest.json"))).size, sha256: source.manifestSha256 },
        ...source.files.map(file => ({ relativePath: file.relativePath, bytes: file.bytes, sha256: file.sha256 }))];
      for (const file of entries) {
        hooks.signal?.throwIfAborted(); const original = join(sourceDirectory, file.relativePath), info = await lstat(original);
        if (!info.isFile() || info.isSymbolicLink() || info.size !== file.bytes || await fileSha256(original, hooks.signal) !== file.sha256) {
          throw new Error("Sparse inherited lineage changed before exact copying.");
        }
        await hooks.reserveWrite?.(file.bytes, "complete inherited sparse lineage");
        await copyMutableFileIsolated(original, join(next, file.relativePath), { expectedSha256: file.sha256, mode: 0o600 });
      }
      await mkdir(dirname(destination), { recursive: true, mode: 0o700 });
      try { await rename(next, destination); }
      catch (error) { if (!await exists(destination)) throw error; }
    } finally { await rm(next, { recursive: true, force: true }); }
    const copied = await validateLineage(targetRoot, source.sourceBrainId, source.workspaceSha256, hooks.signal);
    if (canonical(copied) !== canonical(source)) throw new Error("Sparse inherited lineage copy differs from its exact source archive.");
  }
}

export async function adoptSparseSubstrateTransferWorkspace(root: string, sourceBrainId: string, targetBrainId: string,
  hooks: SparseSubstratePortabilityHooks = {}): Promise<SparseSubstrateLineageDescriptor | undefined> {
  sparseId(sourceBrainId); sparseId(targetBrainId); const workspace = join(root, WORKSPACE);
  if (!await exists(workspace)) { if (await exists(join(root, SPARSE_TRANSFER_DIRECTORY))) await validateSparseSubstrateTransferWorkspace(root, targetBrainId, { signal: hooks.signal }); return undefined; }
  const actualOwner = await databaseOwner(workspace);
  if (actualOwner === targetBrainId && actualOwner !== sourceBrainId) { await validateSparseSubstrateTransferWorkspace(root, targetBrainId, { signal: hooks.signal }); return undefined; }
  if (actualOwner !== sourceBrainId) throw new Error("Sparse copied workspace owner changed before adoption.");
  const archive = await archiveWorkspace(root, sourceBrainId, root, sourceBrainId, hooks), archivedDatabase = join(root, archive.relativePath, "workspace.sqlite3");
  const temporaryRoot = join(root, `.sparse-adopt-${randomUUID()}`);
  try {
    await projectJobs(archivedDatabase, sourceBrainId, temporaryRoot, targetBrainId, sourceBrainId);
    const next = join(temporaryRoot, WORKSPACE); await validateSparseTransferDatabaseFile(next, targetBrainId);
    await hooks.reserveWrite?.((await lstat(next)).size, "sparse adopted current workspace");
    await copyMutableFileIsolated(next, workspace, { expectedSha256: await fileSha256(next, hooks.signal), mode: 0o600 });
    await Promise.all([rm(join(root, SPARSE_TRANSFER_DIRECTORY, "bundles"), { recursive: true, force: true }),
      rm(join(root, SPARSE_TRANSFER_DIRECTORY, "plans"), { recursive: true, force: true })]);
  } finally { await rm(temporaryRoot, { recursive: true, force: true }); }
  await validateSparseSubstrateTransferWorkspace(root, targetBrainId, { signal: hooks.signal }); return archive;
}

export async function mergeSparseSubstrateTransferWorkspace(sourceRoot: string, targetRoot: string, owner: string,
  parentBrainId: string, hooks: SparseSubstratePortabilityHooks = {}): Promise<SparseSubstrateLineageDescriptor | undefined> {
  sparseId(owner); sparseId(parentBrainId);
  await copyExistingLineage(sourceRoot, targetRoot, hooks);
  const sourceWorkspace = join(sourceRoot, WORKSPACE);
  if (!await exists(sourceWorkspace)) return undefined;
  const sourceOwner = await databaseOwner(sourceWorkspace);
  const archive = await archiveWorkspace(sourceRoot, sourceOwner, targetRoot, parentBrainId, hooks), archivedDatabase = join(targetRoot, archive.relativePath, "workspace.sqlite3");
  await projectJobs(archivedDatabase, sourceOwner, targetRoot, owner, parentBrainId, `${parentBrainId}\0${archive.workspaceSha256}`);
  await validateSparseSubstrateTransferWorkspace(targetRoot, owner, { signal: hooks.signal }); return archive;
}
