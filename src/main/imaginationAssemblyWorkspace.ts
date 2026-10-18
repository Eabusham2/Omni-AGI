import { createHash, randomUUID } from "node:crypto";
import { lstat, mkdir, opendir, readFile, readdir, rm } from "node:fs/promises";
import { join } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import type { ImaginationAssembly, ImaginationAssemblyManifestSummary } from "../shared/imaginationAssembly";
import { verifyReplaySpool, ReplaySpool, type ReplaySpoolSeal } from "./imaginationReplaySpool";
import { ReplayJobStore, rekeyCopiedReplayJobStore } from "./imaginationReplayJobStore";
import { copyMutableFileIsolated, snapshotMutableSqliteIsolated } from "./mutableFileIsolation";
import { replayFileHash } from "./imaginationReplayCodec";
import { replayPlanReceipts, rekeyReplayPlanRequests } from "./imaginationPlanRequests";

export const REPLAY_ROOT = "imagination-assemblies";
export const replayHash = (value: unknown): string => createHash("sha256").update(JSON.stringify(value)).digest("hex");
export interface ReplayIntent { format: "omni-replay-intent"; version: 1; id: string; brainId: string; requestId: string; requestSha256: string; manifestSha256: string; manifest: ImaginationAssemblyManifestSummary; spoolId: string; spoolSeal: ReplaySpoolSeal }
export interface ReplayStoredState { view: ImaginationAssembly }
const boundaries = new Map<string, <T>(work: () => Promise<T>) => Promise<T>>();
export function registerReplayWorkspaceBoundary(root: string, boundary: <T>(work: () => Promise<T>) => Promise<T>): () => void { boundaries.set(root, boundary); return () => { if (boundaries.get(root) === boundary) boundaries.delete(root); }; }
export function withReplayWorkspaceBoundary<T>(root: string, work: () => Promise<T>): Promise<T> { return boundaries.get(root)?.(work) ?? work(); }
export const replayId = (value: string): string => { if (!/^[a-f0-9]{32}(?:[a-f0-9]{32})?$/.test(value)) throw new Error("Invalid replay assembly identity."); return value; };
export function safeReplayWorkspacePath(value: string): boolean {
  return /^imagination-assemblies\/jobs\/[a-f0-9]{32}(?:[a-f0-9]{32})?\/(?:intent\.json|state\.sqlite3(?:-(?:wal|shm|journal))?)$/.test(value) ||
    /^imagination-assemblies\/jobs\/[a-f0-9]{32}(?:[a-f0-9]{32})?\/work\/[a-f0-9]{32}\/(?:segment-\d+\.(?:mkv|wav)|output\.(?:mp4|wav)|segments\.ffconcat)$/.test(value) ||
    /^imagination-assemblies\/(?:blobs\/[a-f0-9]{64}\.(?:data|json)|archives\/[a-f0-9]{64}\.(?:sqlite3|json)|requests\/[a-f0-9]{64}\.json|plans\/[a-f0-9]{64}\/(?:manifest\.json|manifest\.sqlite3|source\.sqlite3))$/.test(value);
}
async function json(path: string): Promise<any> {
  const stat = await lstat(path), bytes = stat.size * 12;
  if (!stat.isFile() || stat.isSymbolicLink() || !Number.isSafeInteger(bytes) || bytes > Math.min(freemem(), getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 2) throw new Error("Replay metadata requires a real file and physical parse headroom.");
  return JSON.parse(await readFile(path, "utf8"));
}
export async function readReplay(root: string, id: string, owner: string): Promise<{ intent: ReplayIntent; state: ReplayStoredState }> {
  const path = join(root, REPLAY_ROOT, "jobs", replayId(id));
  const intent = await json(join(path, "intent.json")) as ReplayIntent, store = await ReplayJobStore.open(root, id, owner, { readOnly: true });
  let view: ImaginationAssembly; try { view = store.view(); if (store.count() !== view.completedSpans) throw new Error("Replay cursor disagrees with the committed segment chain."); } finally { store.close(); }
  const state = { view };
  if (intent.format !== "omni-replay-intent" || intent.version !== 1 || intent.manifest.manifestSha256 !== intent.manifestSha256 || intent.spoolSeal.jsonSha256 !== intent.manifestSha256 || replayHash(intent.spoolSeal) !== intent.spoolId ||
      !view || view.format !== "omni-imagination-assembly" || view.version !== 1 || view.id !== id || view.brainId !== owner || view.manifestSha256 !== intent.manifestSha256 ||
      !Number.isSafeInteger(view.revision) || view.revision < 1 || !["queued", "preparing", "encoding", "complete", "cancelled", "failed", "dormant"].includes(view.state) ||
      (intent.id !== id || intent.brainId !== owner) && (view.inheritedFrom?.assemblyId !== intent.id || view.inheritedFrom?.brainId !== intent.brainId)) throw new Error("Replay source intent or current owner state changed.");
  if (view.totalSpans !== intent.manifest.spanCount || view.completedSpans > view.totalSpans) throw new Error("Replay committed segment cursor is incomplete.");
  return { intent, state };
}
export async function writeReplayState(root: string, state: ReplayStoredState): Promise<void> {
  const path = join(root, REPLAY_ROOT, "jobs", replayId(state.view.id), "state.sqlite3"), present = await lstat(path).catch(error => { if (error.code === "ENOENT") return undefined; throw error; });
  const store = await ReplayJobStore.open(root, state.view.id, state.view.brainId, present ? {} : { create: state.view });
  try { if (present) store.saveView(state.view, state.view.revision - 1); } finally { store.close(); }
}
export async function* replayJobIds(root: string): AsyncGenerator<string> {
  const directory = await opendir(join(root, REPLAY_ROOT, "jobs")).catch(error => { if (error.code === "ENOENT") return undefined; throw error; }); if (!directory) return;
  for await (const row of directory) { if (!row.isDirectory() || row.isSymbolicLink()) throw new Error("Replay job custody contains an unsafe entry."); yield replayId(row.name); }
}
async function* files(root: string, prefix = REPLAY_ROOT): AsyncGenerator<{ name: string; path: string; bytes: number }> {
  const directory = join(root, prefix), info = await lstat(directory).catch(error => { if (error.code === "ENOENT") return undefined; throw error; }); if (!info) return;
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Replay custody requires real owned directories.");
  for await (const row of await opendir(directory)) {
    const name = `${prefix}/${row.name}`, path = join(root, name); if (row.isSymbolicLink()) throw new Error("Replay custody refuses links.");
    if (row.isDirectory()) yield* files(root, name); else if (row.isFile() && safeReplayWorkspacePath(name)) yield { name, path, bytes: (await lstat(path)).size }; else throw new Error("Replay custody has undeclared payloads.");
  }
}
export async function validateReplayWorkspace(root: string, owner: string): Promise<void> {
  for await (const row of files(root)) if (/\/(?:blobs|archives)\//.test(row.name) && await replayFileHash(row.path) !== row.name.split("/").at(-1)!.split(".")[0]) throw new Error("An immutable replay artifact or archive changed.");
  for await (const receipt of replayPlanReceipts(root,owner)) if(receipt.result) await verifyReplaySpool(join(root,REPLAY_ROOT,"plans",receipt.result.plan.id),receipt.result.seal);
  const plans = await opendir(join(root, REPLAY_ROOT, "plans")).catch(error => { if (error.code === "ENOENT") return undefined; throw error; });
  if (plans) for await (const entry of plans) { if (!entry.isDirectory() || entry.isSymbolicLink() || !/^[a-f0-9]{64}$/.test(entry.name)) throw new Error("Unsafe immutable replay plan.");
    const directory = join(root, REPLAY_ROOT, "plans", entry.name), seal = { jsonSha256: await replayFileHash(join(directory, "manifest.json")), sqliteSha256: await replayFileHash(join(directory, "manifest.sqlite3")), sourceSha256: await replayFileHash(join(directory, "source.sqlite3")) };
    if (replayHash(seal) !== entry.name) throw new Error("Replay plan lost its complete immutable file seal."); }
  for await (const id of replayJobIds(root)) {
    const { intent, state } = await readReplay(root, id, owner);
    const directory = join(root, REPLAY_ROOT, "plans", intent.spoolId); await verifyReplaySpool(directory, intent.spoolSeal); const spool = new ReplaySpool(directory, intent.manifest), store = await ReplayJobStore.open(root, id, owner, { readOnly: true });
    try { store.validate(); for (const fragment of spool.fragments()) {
      if (!/^[a-f0-9]{64}$/.test(fragment.sha256)) throw new Error("Unsafe source fragment hash.");
      const path = join(root, "imagination", "blobs", `${fragment.sha256}.data`), info = await lstat(path);
      if (!info.isFile() || info.isSymbolicLink() || info.size !== fragment.bytes || await replayFileHash(path) !== fragment.sha256) throw new Error("Replay lost an exact recorded source fragment.");
    }
    for (const segment of store.segments()) { if (!safeReplayWorkspacePath(`${REPLAY_ROOT}/jobs/${id}/${segment.path}`)) throw new Error("Unsafe committed segment path."); const path = join(root, REPLAY_ROOT, "jobs", id, segment.path), info = await lstat(path);
      if (!info.isFile() || info.isSymbolicLink() || info.size !== segment.bytes || await replayFileHash(path) !== segment.sha256) throw new Error("A committed replay segment changed."); }
    if (state.view.artifactSha256) { if (!/^[a-f0-9]{64}$/.test(state.view.artifactSha256)) throw new Error("Unsafe replay artifact hash."); const path = join(root, REPLAY_ROOT, "blobs", `${state.view.artifactSha256}.data`), info = await lstat(path);
      if (!info.isFile() || info.isSymbolicLink() || info.size !== state.view.bytes || await replayFileHash(path) !== state.view.artifactSha256) throw new Error("Completed replay is missing or changed."); }
    } finally { spool.close(); store.close(); }
  }
}
export async function rekeyReplayWorkspace(root: string, oldOwner: string, newOwner: string): Promise<void> {
  await validateReplayWorkspace(root, oldOwner);
  for await (const id of replayJobIds(root)) { const store = await ReplayJobStore.open(root, id, oldOwner); try { store.rekey(newOwner); } finally { store.close(); } }
  await rekeyReplayPlanRequests(root,oldOwner,newOwner);
  await validateReplayWorkspace(root, newOwner);
}
/** Sources and every partial attempt are retained; only current owner/custody is projected. */
export async function mergeReplayWorkspace(source: string, target: string, owner: string, reserve: (bytes: number) => Promise<void>, recordingAliases?: ReadonlyMap<string,string>): Promise<void> {
  if (source === target) throw new Error("Replay adoption needs a private target."); await validateReplayWorkspace(source, owner); await validateReplayWorkspace(target, owner);
  for await (const row of files(source)) if (!row.name.startsWith(`${REPLAY_ROOT}/jobs/`)) { const destination = join(target, row.name), sha = await replayFileHash(row.path), present = await lstat(destination).catch(error => { if (error.code === "ENOENT") return undefined; throw error; });
    if (present) { if (!present.isFile() || present.isSymbolicLink()) throw new Error("Unsafe existing replay payload.");
      if (present.size !== row.bytes || await replayFileHash(destination) !== sha) { if(!row.name.startsWith(`${REPLAY_ROOT}/requests/`))throw new Error("An existing immutable replay payload changed; adoption will not overwrite it."); await reserve(row.bytes); await copyMutableFileIsolated(row.path,join(target,REPLAY_ROOT,"archives",`${sha}.json`),{expectedSha256:sha}); }
    }
    else { await reserve(row.bytes); await copyMutableFileIsolated(row.path, destination, { expectedSha256: sha }); } }
  for await (const id of replayJobIds(source)) {
    const { intent, state } = await readReplay(source, id, owner), sourceStore = await ReplayJobStore.open(source, id, owner, { readOnly: true });
    let head; try { head = sourceStore.head(); } finally { sourceStore.close(); }
    const currentRecordingId = recordingAliases?.get(state.view.currentRecordingId ?? state.view.recordingId) ?? state.view.currentRecordingId ?? state.view.recordingId;
    const digest = replayHash({ intent, state: { state: state.view.state, segments: head.segmentsSha256, artifact: state.view.artifactSha256 ?? null, currentRecordingId } });
    const nextId = digest, existing = await lstat(join(target, REPLAY_ROOT, "jobs", nextId)).catch(error => { if (error.code === "ENOENT") return undefined; throw error; });
    for await (const row of files(source, `${REPLAY_ROOT}/jobs/${id}`)) { const oldPrefix = `${REPLAY_ROOT}/jobs/${id}/`;
      if (/-(?:wal|shm|journal)$/.test(row.name) || existing) continue;
      const name = row.name.startsWith(oldPrefix) ? `${REPLAY_ROOT}/jobs/${nextId}/${row.name.slice(oldPrefix.length)}` : row.name;
      await reserve(row.bytes); if (row.name === `${oldPrefix}state.sqlite3`) { await mkdir(join(target, REPLAY_ROOT, "jobs", nextId), { recursive: true, mode: 0o700 }); await snapshotMutableSqliteIsolated(row.path, join(target, name)); }
      else await copyMutableFileIsolated(row.path, join(target, name), { expectedSha256: await replayFileHash(row.path) });
    }
    await mkdir(join(target, REPLAY_ROOT, "archives"), { recursive: true, mode: 0o700 });
    const temporary = join(target, REPLAY_ROOT, "archives", `.snapshot-${randomUUID()}.sqlite3`);
    try { await snapshotMutableSqliteIsolated(join(source, REPLAY_ROOT, "jobs", id, "state.sqlite3"), temporary); const sha = await replayFileHash(temporary); await reserve((await lstat(temporary)).size); await copyMutableFileIsolated(temporary, join(target, REPLAY_ROOT, "archives", `${sha}.sqlite3`), { expectedSha256: sha }); }
    finally { await rm(temporary, { force: true }); }
    if (!existing) { await rekeyCopiedReplayJobStore(target, nextId, owner, id, owner); const store = await ReplayJobStore.open(target, nextId, owner); try { const view = store.view(); store.saveView({ ...view, currentRecordingId, revision: view.revision + 1, updatedAt: new Date().toISOString() }, view.revision); } finally { store.close(); } }
  }
  await validateReplayWorkspace(target, owner);
}
