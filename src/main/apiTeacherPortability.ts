import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, mkdir, opendir, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import {
  API_TEACHER_DATABASE,
  API_TEACHER_DIRECTORY,
  ApiTeacherStore,
  apiTeacherCanonical,
  apiTeacherId,
  apiTeacherSha256,
  validateApiTeacherDatabaseFile,
  type ApiTeacherLineageRecord,
  type ApiTeacherStoredItem,
  type ApiTeacherStoredJob
} from "./apiTeacherStore";
import { copyMutableFileIsolated, snapshotMutableSqliteIsolated } from "./mutableFileIsolation";
import { admitApiTeacherDatabase, apiTeacherMediaFiles, safeApiTeacherMediaPath, verifyApiTeacherMediaFile, type ApiTeacherMediaIo } from "./apiTeacherMedia";
import { sqliteFilename } from "./sqliteFilename";

export interface ApiTeacherPortabilityHooks {
  signal?: AbortSignal;
  checkDisk?(bytes: number): Promise<void>;
  admitMetadata?: ApiTeacherMediaIo["admitMetadata"];
  /** External copy/import is dormant even when its source ID is still free. */
  retireCopiedTeacherAuthority?: true;
}

const ARCHIVE = /^api-teachers\/archives\/([a-f0-9]{64})\.sqlite3$/;
const EPOCH = "1970-01-01T00:00:00.000Z";
async function exists(path: string): Promise<boolean> {
  return lstat(path).then(() => true, (error: NodeJS.ErrnoException) => error.code === "ENOENT" ? false : Promise.reject(error));
}
async function hashFile(path: string, signal?: AbortSignal): Promise<string> {
  const digest = createHash("sha256");
  for await (const block of createReadStream(path, { highWaterMark: 1024 * 1024, signal })) {
    signal?.throwIfAborted(); digest.update(block);
  }
  return digest.digest("hex");
}
function archiveRelative(sha256: string): string { return `${API_TEACHER_DIRECTORY}/archives/${sha256}.sqlite3`; }
function archivePath(root: string, sha256: string): string { return join(root, ...archiveRelative(sha256).split("/")); }
function newestSourceTime(jobs: readonly ApiTeacherStoredJob[]): string {
  return jobs.reduce((latest, job) => job.updatedAt > latest ? job.updatedAt : latest, EPOCH);
}
function lineage(sourceBrainId: string, sha256: string, bytes: number, jobs: readonly ApiTeacherStoredJob[]): ApiTeacherLineageRecord {
  return { format: "omni-api-teacher-lineage", version: 1, sourceBrainId, databaseSha256: sha256,
    bytes, relativePath: archiveRelative(sha256), sourceUpdatedAt: newestSourceTime(jobs) };
}

export function safeApiTeacherWorkspacePath(value: string): boolean {
  return value === API_TEACHER_DATABASE || ARCHIVE.test(value) || safeApiTeacherMediaPath(value);
}

/** Immutable media/envelope closure reachable from current AND exact historical databases. */
export async function apiTeacherMediaReferenceFiles(root: string, owner: string, hooks: ApiTeacherPortabilityHooks = {}): Promise<Map<string, { sha256: string; bytes: number }>> {
  const files = new Map<string, { sha256: string; bytes: number }>();
  if (!await exists(join(root, API_TEACHER_DATABASE))) return files;
  await admitApiTeacherDatabase(join(root, API_TEACHER_DATABASE), hooks);
  const current = await ApiTeacherStore.open(root, owner, async () => { throw new Error("Teacher closure is read-only."); }, { create: false });
  if (!current) return files;
  const collect = (store: ApiTeacherStore): void => {
    for (const item of store.mediaLessons()) for (const [path, expected] of apiTeacherMediaFiles(item.lesson, item.prompt)) {
      const old = files.get(path); if (old && (old.sha256 !== expected.sha256 || old.bytes !== expected.bytes)) throw new Error("Teacher media closure collided.");
      files.set(path, expected);
    }
    for (const envelope of store.receivedEnvelopes()) files.set(envelope.relativePath, { sha256: envelope.sha256, bytes: envelope.bytes });
  };
  try {
    collect(current);
    for (const lineage of current.lineages()) {
      hooks.signal?.throwIfAborted();
      const path = join(root, ...lineage.relativePath.split("/"));
      const info = await lstat(path);
      if (!info.isFile() || info.isSymbolicLink() || info.size !== lineage.bytes || await hashFile(path, hooks.signal) !== lineage.databaseSha256)
        throw new Error("Teacher media references are not from the exact sealed lineage database.");
      // Archives retain their source owner; never project them into a current
      // workspace merely to inspect their original immutable references.
      await admitApiTeacherDatabase(path, hooks);
      await validateApiTeacherDatabaseFile(path, lineage.sourceBrainId);
      const database = new DatabaseSync(sqliteFilename(path), { readOnly: true, allowExtension: false });
      try {
        database.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-256; PRAGMA mmap_size=0;");
        for (const table of ["items", "history"] as const) for (const row of database.prepare(`SELECT payload,payload_sha FROM ${table}`).iterate()) {
          if (typeof row.payload !== "string" || apiTeacherSha256(row.payload) !== row.payload_sha) throw new Error("Teacher historical payload checksum changed.");
          const item = JSON.parse(row.payload) as Partial<ApiTeacherStoredItem>;
          if (item.format !== "omni-api-teacher-item") continue;
          if (item.mediaLesson) for (const [relative, expected] of apiTeacherMediaFiles(item.mediaLesson, String(item.prompt))) files.set(relative, expected);
          if (item.responseEnvelope) files.set(item.responseEnvelope.relativePath, { sha256: item.responseEnvelope.sha256, bytes: item.responseEnvelope.bytes });
        }
      } finally { database.close(); }
    }
  } finally { current.close(); }
  return files;
}

/** SQLite's coherent snapshot pins the reference set; referenced assets are immutable. */
export async function captureApiTeacherWorkspace(sourceRoot: string, targetRoot: string, owner: string,
  hooks: ApiTeacherPortabilityHooks = {}): Promise<Map<string, string>> {
  const result = new Map<string, string>(), source = join(sourceRoot, API_TEACHER_DATABASE);
  if (!await exists(source)) return result;
  if (resolve(sourceRoot) === resolve(targetRoot)) throw new Error("Teacher custody cannot capture onto its source.");
  const target = join(targetRoot, API_TEACHER_DATABASE); await mkdir(dirname(target), { recursive: true, mode: 0o700 });
  await hooks.checkDisk?.((await lstat(source)).size + 65536); hooks.signal?.throwIfAborted();
  await snapshotMutableSqliteIsolated(source, target); await admitApiTeacherDatabase(target, hooks); await validateApiTeacherDatabaseFile(target, owner);
  result.set(API_TEACHER_DATABASE, target);
  const store = await ApiTeacherStore.open(targetRoot, owner, async () => undefined, { create: false });
  try { for (const archive of store!.lineages()) {
    const original = join(sourceRoot, ...archive.relativePath.split("/")), destination = join(targetRoot, ...archive.relativePath.split("/"));
    if ((await lstat(original)).size !== archive.bytes || await hashFile(original, hooks.signal) !== archive.databaseSha256)
      throw new Error("Teacher exact lineage changed during coherent capture.");
    await hooks.checkDisk?.(archive.bytes + 65536); await copyMutableFileIsolated(original, destination, { expectedSha256: archive.databaseSha256, mode: 0o600 });
    result.set(archive.relativePath, destination);
  } } finally { store?.close(); }
  for (const [relative, expected] of await apiTeacherMediaReferenceFiles(targetRoot, owner, hooks)) {
    await verifyApiTeacherMediaFile(sourceRoot, relative, expected, { ...hooks });
    const destination = join(targetRoot, ...relative.split("/")); await hooks.checkDisk?.(expected.bytes + 65536);
    await copyMutableFileIsolated(join(sourceRoot, ...relative.split("/")), destination, { expectedSha256: expected.sha256, mode: 0o600 });
    await verifyApiTeacherMediaFile(targetRoot, relative, expected, { ...hooks }); result.set(relative, destination);
  }
  await validateApiTeacherWorkspace(targetRoot, owner, hooks.signal, hooks.admitMetadata); return result;
}

export async function validateCapturedApiTeacherWorkspace(path: string, owner: string, hooks: ApiTeacherMediaIo = {}): Promise<void> {
  await admitApiTeacherDatabase(path, hooks);
  await validateApiTeacherDatabaseFile(path, owner);
}

/** Validate current credential-free rows and every exact source database retained in lineage. */
export async function validateApiTeacherWorkspace(root: string, owner: string, signal?: AbortSignal,
  admitMetadata?: ApiTeacherMediaIo["admitMetadata"]): Promise<void> {
  apiTeacherId(owner, "API teacher workspace owner");
  const directory = join(root, API_TEACHER_DIRECTORY), workspace = join(root, API_TEACHER_DATABASE);
  if (!await exists(directory)) return;
  const directoryInfo = await lstat(directory);
  if (!directoryInfo.isDirectory() || directoryInfo.isSymbolicLink()) throw new Error("API teacher custody needs an owned directory.");
  if (!await exists(workspace)) throw new Error("API teacher custody has no current ownership database.");
  await admitApiTeacherDatabase(workspace, { signal, admitMetadata }); await validateApiTeacherDatabaseFile(workspace, owner);
  const store = await ApiTeacherStore.open(root, owner, async () => { throw new Error("Validation cannot admit API teacher writes."); }, { create: false });
  if (!store) throw new Error("API teacher custody has no current ownership database.");
  try {
    const expected = new Map(store.lineages().map(item => [item.relativePath, item]));
    const media = await apiTeacherMediaReferenceFiles(root, owner, { signal, admitMetadata });
    const rootEntries = await opendir(directory);
    for await (const entry of rootEntries) {
      signal?.throwIfAborted();
      if (["workspace.sqlite3", "workspace.sqlite3-wal", "workspace.sqlite3-shm", "workspace.sqlite3-journal"].includes(entry.name)) {
        if (!entry.isFile() || entry.isSymbolicLink()) throw new Error("API teacher operational custody contains an unsafe file.");
        continue;
      }
      if (["assets", "envelopes"].includes(entry.name) && entry.isDirectory() && !entry.isSymbolicLink()) {
        for await (const asset of await opendir(join(directory, entry.name))) {
          const relative = `${API_TEACHER_DIRECTORY}/${entry.name}/${asset.name}`;
          if (!asset.isFile() || asset.isSymbolicLink() || !safeApiTeacherMediaPath(relative)) throw new Error("Teacher media custody contains an unknown payload.");
          const file = join(root, ...relative.split("/")), bytes = (await lstat(file)).size;
          await verifyApiTeacherMediaFile(root, relative, media.get(relative) ?? { sha256: asset.name.split(".")[0]!, bytes }, { signal, admitMetadata });
        }
        continue;
      }
      if (/^lesson-[a-f0-9]{64}\.jsonl$/.test(entry.name) && entry.isFile() && !entry.isSymbolicLink()) {
        const relative = `${API_TEACHER_DIRECTORY}/${entry.name}`, bytes = (await lstat(join(directory, entry.name))).size;
        await verifyApiTeacherMediaFile(root, relative, media.get(relative) ?? { sha256: entry.name.slice(7,-6), bytes }, { signal, admitMetadata }); continue;
      }
      if (entry.name !== "archives" || !entry.isDirectory() || entry.isSymbolicLink())
        throw new Error("API teacher custody contains an unlisted payload.");
      const archives = await opendir(join(directory, "archives"));
      for await (const archive of archives) {
        signal?.throwIfAborted(); const relativePath = `${API_TEACHER_DIRECTORY}/archives/${archive.name}`, match = ARCHIVE.exec(relativePath);
        if (!match || !archive.isFile() || archive.isSymbolicLink()) throw new Error("API teacher lineage contains an unsafe archive.");
        const record = expected.get(relativePath); if (!record) throw new Error("API teacher lineage contains an unreferenced archive.");
        const path = join(directory, "archives", archive.name), info = await lstat(path);
        if (info.size !== record.bytes || await hashFile(path, signal) !== record.databaseSha256)
          throw new Error("API teacher lineage archive checksum or byte count changed.");
        await admitApiTeacherDatabase(path, { signal, admitMetadata }); await validateApiTeacherDatabaseFile(path, record.sourceBrainId); expected.delete(relativePath);
      }
    }
    if (expected.size) throw new Error("API teacher lineage database is missing an exact archive.");
    for (const [relative, expected] of media) await verifyApiTeacherMediaFile(root, relative, expected, { signal, admitMetadata });
  } finally { store.close(); }
}

async function copyTeacherMediaReferences(sourceRoot: string, targetRoot: string, owner: string,
  hooks: ApiTeacherPortabilityHooks): Promise<string[]> {
  const copied: string[] = [];
  try {
    for (const [relative, expected] of await apiTeacherMediaReferenceFiles(sourceRoot, owner, hooks)) {
      await verifyApiTeacherMediaFile(sourceRoot, relative, expected, hooks);
      const target = join(targetRoot, ...relative.split("/"));
      if (await exists(target)) { await verifyApiTeacherMediaFile(targetRoot, relative, expected, hooks); continue; }
      await hooks.checkDisk?.(expected.bytes + 65536); hooks.signal?.throwIfAborted();
      await copyMutableFileIsolated(join(sourceRoot, ...relative.split("/")), target, { expectedSha256: expected.sha256, mode: 0o600 });
      copied.push(target); await verifyApiTeacherMediaFile(targetRoot, relative, expected, hooks);
    }
    return copied;
  } catch (error) { await Promise.all(copied.map(path => rm(path, { force: true }))); throw error; }
}

function projection(job: ApiTeacherStoredJob, sourceItems: readonly ApiTeacherStoredItem[], owner: string, sourceSha256: string,
  parentSalt = "copy"): { job: ApiTeacherStoredJob; items: ApiTeacherStoredItem[] } {
  const identity = apiTeacherCanonical({ kind: "api-teacher-dormant-projection", owner, parentSalt, sourceSha256,
    sourceJobId: job.id, originalBrainId: job.originalBrainId, originalJobId: job.originalJobId });
  const id = apiTeacherSha256(identity), requestId = apiTeacherSha256(`${identity}\0request`);
  const projected: ApiTeacherStoredJob = { ...job, id, brainId: owner, requestId, state: "dormant",
    revision: job.revision + 1, error: "Retained teacher history is dormant; start a new reviewed job with current device credentials to make provider requests." };
  const items = sourceItems.map(item => ({ ...item, jobId: id, state: "dormant" as const,
    error: "Retained source item; provider and learning authority were not copied." }));
  return { job: projected, items };
}

async function materializeArchive(sourcePath: string, root: string, sourceOwner: string,
  hooks: ApiTeacherPortabilityHooks): Promise<{
  descriptor: ApiTeacherLineageRecord; snapshot: string; created: boolean; lineages: ApiTeacherLineageRecord[];
  bundles: Array<{ job: ApiTeacherStoredJob; items: ApiTeacherStoredItem[] }>;
}> {
  hooks.signal?.throwIfAborted(); const temporaryDirectory = join(root, `.api-teacher-snapshot-${randomUUID()}`);
  const snapshot = join(temporaryDirectory, API_TEACHER_DATABASE); await mkdir(dirname(snapshot), { recursive: true, mode: 0o700 });
  try {
    const sourceInfo = await lstat(sourcePath), wal = await lstat(`${sourcePath}-wal`).catch((error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") return undefined; throw error;
    });
    if (!sourceInfo.isFile() || sourceInfo.isSymbolicLink()) throw new Error("API teacher snapshot source is not an owned database.");
    await hooks.checkDisk?.(sourceInfo.size + (wal?.size ?? 0) + 65536); hooks.signal?.throwIfAborted();
    await snapshotMutableSqliteIsolated(sourcePath, snapshot); await admitApiTeacherDatabase(snapshot, hooks); await validateApiTeacherDatabaseFile(snapshot, sourceOwner);
    const info = await lstat(snapshot), sha256 = await hashFile(snapshot, hooks.signal);
    await hooks.checkDisk?.(info.size + 65536); hooks.signal?.throwIfAborted();
    const destination = archivePath(root, sha256); let created = false;
    if (await exists(destination)) {
      const prior = await lstat(destination);
      if (!prior.isFile() || prior.isSymbolicLink() || prior.size !== info.size || await hashFile(destination, hooks.signal) !== sha256)
        throw new Error("API teacher lineage archive identity collided with different bytes.");
    } else { await copyMutableFileIsolated(snapshot, destination, { expectedSha256: sha256, mode: 0o600 }); created = true; }
    try {
      const captured = await ApiTeacherStore.open(temporaryDirectory, sourceOwner, async () => undefined, { create: false });
      if (!captured) throw new Error("API teacher source snapshot disappeared during custody capture.");
      try {
        const jobs = captured.jobs();
        return { descriptor: lineage(sourceOwner, sha256, info.size, jobs), snapshot, created, lineages: captured.lineages(),
          bundles: jobs.map(job => ({ job, items: captured.items(job.id) })) };
      } finally { captured.close(); }
    } catch (error) {
      if (created) await rm(destination, { force: true }); throw error;
    }
  } catch (error) {
    await rm(temporaryDirectory, { recursive: true, force: true }); throw error;
  }
}

async function copyKnownArchives(sourceRoot: string, targetRoot: string, records: readonly ApiTeacherLineageRecord[],
  hooks: ApiTeacherPortabilityHooks): Promise<string[]> {
  const created: string[] = [];
  try {
    for (const record of records) {
      hooks.signal?.throwIfAborted(); const source = join(sourceRoot, ...record.relativePath.split("/")), target = join(targetRoot, ...record.relativePath.split("/"));
      const info = await lstat(source);
      if (!info.isFile() || info.isSymbolicLink() || info.size !== record.bytes || await hashFile(source, hooks.signal) !== record.databaseSha256)
        throw new Error("API teacher inherited lineage archive changed before copy.");
      if (await exists(target)) {
        const current = await lstat(target);
        if (!current.isFile() || current.isSymbolicLink() || current.size !== record.bytes || await hashFile(target, hooks.signal) !== record.databaseSha256)
          throw new Error("API teacher target lineage archive collided with different bytes.");
        continue;
      }
      await hooks.checkDisk?.(record.bytes + 65536); await copyMutableFileIsolated(source, target,
        { expectedSha256: record.databaseSha256, mode: 0o600 }); created.push(target);
    }
    return created;
  } catch (error) { await Promise.all(created.map(path => rm(path, { force: true }))); throw error; }
}

async function checkpointBeforeReplacement(path: string): Promise<void> {
  const database = new DatabaseSync(sqliteFilename(path), { allowExtension: false });
  try {
    database.exec("PRAGMA trusted_schema=OFF; PRAGMA busy_timeout=30000; PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;");
  } finally { database.close(); }
  await Promise.all([rm(`${path}-wal`, { force: true }), rm(`${path}-shm`, { force: true }), rm(`${path}-journal`, { force: true })]);
}

/** Copy/adoption never carries a provider credential, live request, controller, or learning authority. */
export async function rekeyApiTeacherWorkspace(root: string, sourceOwner: string, targetOwner: string,
  hooks: ApiTeacherPortabilityHooks = {}): Promise<void> {
  apiTeacherId(sourceOwner, "source API teacher owner"); apiTeacherId(targetOwner, "target API teacher owner");
  const workspace = join(root, API_TEACHER_DATABASE); if (!await exists(workspace)) return;
  try { await validateApiTeacherWorkspace(root, sourceOwner, hooks.signal, hooks.admitMetadata); }
  catch (sourceError) {
    if (sourceOwner === targetOwner) throw sourceError;
    try { await validateApiTeacherWorkspace(root, targetOwner, hooks.signal, hooks.admitMetadata); return; } catch { throw sourceError; }
  }
  if (sourceOwner === targetOwner && !hooks.retireCopiedTeacherAuthority) return;
  const captured = await materializeArchive(workspace, root, sourceOwner, hooks);
  const temporaryRoot = join(root, `.api-teacher-adopt-${randomUUID()}`);
  let promoted = false;
  try {
    await mkdir(temporaryRoot, { recursive: false, mode: 0o700 });
    const target = await ApiTeacherStore.open(temporaryRoot, targetOwner,
      async (_brainId, bytes, signal) => { await hooks.checkDisk?.(bytes); signal?.throwIfAborted(); });
    if (!target) throw new Error("Failed to initialize copied API teacher custody.");
    try {
      await target.importDormant([...captured.lineages, captured.descriptor],
        captured.bundles.map(value => projection(value.job, value.items, targetOwner, captured.descriptor.databaseSha256)), hooks.signal);
    } finally { target.close(); }
    const next = join(temporaryRoot, API_TEACHER_DATABASE); await admitApiTeacherDatabase(next, hooks); await validateApiTeacherDatabaseFile(next, targetOwner);
    await hooks.checkDisk?.((await lstat(next)).size + 65536); hooks.signal?.throwIfAborted();
    await checkpointBeforeReplacement(workspace); hooks.signal?.throwIfAborted();
    await copyMutableFileIsolated(next, workspace, { expectedSha256: await hashFile(next, hooks.signal), mode: 0o600 });
    promoted = true;
  } finally {
    if (!promoted && captured.created) await rm(archivePath(root, captured.descriptor.databaseSha256), { force: true });
    await rm(dirname(dirname(captured.snapshot)), { recursive: true, force: true }); await rm(temporaryRoot, { recursive: true, force: true });
  }
  await validateApiTeacherWorkspace(root, targetOwner, hooks.signal, hooks.admitMetadata);
}

/** Merge preserves every exact parent database, then adds only dormant current-owner projections. */
export async function mergeApiTeacherWorkspace(sourceRoot: string, targetRoot: string, owner: string, parentBrainId: string,
  reserve: (bytes: number) => Promise<void>, signal?: AbortSignal, admitMetadata?: ApiTeacherMediaIo["admitMetadata"]): Promise<void> {
  apiTeacherId(owner, "merged API teacher owner"); apiTeacherId(parentBrainId, "API teacher parent owner");
  if (resolve(sourceRoot) === resolve(targetRoot)) throw new Error("API teacher merge cannot mutate its source workspace.");
  const sourcePath = join(sourceRoot, API_TEACHER_DATABASE); if (!await exists(sourcePath)) return;
  const hooks: ApiTeacherPortabilityHooks = { signal, checkDisk: reserve, admitMetadata };
  await validateApiTeacherWorkspace(sourceRoot, owner, signal, admitMetadata);
  const copied: string[] = [];
  let imported = false;
  let captured: Awaited<ReturnType<typeof materializeArchive>> | undefined;
  try {
    const target = await ApiTeacherStore.open(targetRoot, owner,
      async (_brainId, bytes, writeSignal) => { await reserve(bytes); writeSignal?.throwIfAborted(); });
    if (!target) throw new Error("Failed to initialize merged API teacher custody.");
    try {
      captured = await materializeArchive(sourcePath, targetRoot, owner, hooks);
      if (captured.created) copied.push(archivePath(targetRoot, captured.descriptor.databaseSha256));
      copied.push(...await copyKnownArchives(sourceRoot, targetRoot, captured.lineages, hooks));
      copied.push(...await copyTeacherMediaReferences(sourceRoot, targetRoot, owner, hooks));
      await target.importDormant([...captured.lineages, captured.descriptor], captured.bundles.map(value =>
        projection(value.job, value.items, owner, captured!.descriptor.databaseSha256, parentBrainId)), signal);
      imported = true;
    } finally { target.close(); }
  } catch (error) {
    if (!imported) await Promise.all(copied.map(path => rm(path, { force: true }))); throw error;
  } finally { if (captured) await rm(dirname(dirname(captured.snapshot)), { recursive: true, force: true }); }
  await validateApiTeacherWorkspace(targetRoot, owner, signal, admitMetadata);
}
