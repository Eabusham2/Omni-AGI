import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, mkdir, readdir, rm } from "node:fs/promises";
import { join } from "node:path";
import { captureHash, ImaginationCaptureStore } from "./imaginationCaptureStore";
import { copyMutableFileIsolated, snapshotMutableSqliteIsolated } from "./mutableFileIsolation";

const boundaries = new Map<string, <T>(work: () => Promise<T>) => Promise<T>>();
export function registerImaginationWorkspaceBoundary(root: string, boundary: <T>(work: () => Promise<T>) => Promise<T>): () => void { boundaries.set(root, boundary); return () => { if (boundaries.get(root) === boundary) boundaries.delete(root); }; }
export async function withImaginationWorkspaceBoundary<T>(root: string, work: () => Promise<T>): Promise<T> { return boundaries.get(root)?.(work) ?? work(); }
export const safeImaginationWorkspacePath = (value: string): boolean => value === "imagination/workspace.sqlite3" || /^imagination\/(?:blobs|window-blobs)\/[a-f0-9]{64}\.data$/.test(value) || /^imagination\/archives\/[a-f0-9]{64}\.sqlite3$/.test(value);
export async function imaginationFileHash(path: string): Promise<string> { const hash = createHash("sha256"); for await (const chunk of createReadStream(path)) hash.update(chunk); return hash.digest("hex"); }
export async function validateImaginationWorkspace(root: string, brainId: string): Promise<void> {
  const store = await ImaginationCaptureStore.open(root, brainId); if (!store) { const files = await readdir(join(root, "imagination")).catch(error => { if (error.code === "ENOENT") return []; throw error; });
    if (files.length) throw new Error("Imagination payload has no complete owned workspace."); return; }
  try { store.validate(); const directory = join(root, "imagination");
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      if (entry.isSymbolicLink()) throw new Error("Unsafe imagination workspace link.");
      if (entry.isDirectory() && ["blobs", "window-blobs", "archives"].includes(entry.name)) {
        for (const item of await readdir(join(directory, entry.name), { withFileTypes: true })) {
          const path = `imagination/${entry.name}/${item.name}`;
          if (!item.isFile() || item.isSymbolicLink() || !safeImaginationWorkspacePath(path) || await imaginationFileHash(join(root, ...path.split("/"))) !== item.name.split(".")[0]) throw new Error("Imagination archive/window blob changed.");
        }
      } else if (!entry.isFile() || !/^workspace\.sqlite3(?:-(?:wal|shm|journal))?$/.test(entry.name)) throw new Error("Unowned imagination workspace payload.");
    }
    const required = new Map<string, number>(); for (const row of store.db.prepare("SELECT body FROM fragments").iterate()) { const value = JSON.parse(String(row.body)); required.set(`blobs/${value.sha256}.data`, value.bytes); }
    for (const row of store.window()) required.set(`window-blobs/${row.sha256}.data`, row.bytes);
    for (const [path, bytes] of required) { const info = await lstat(join(directory, path)); if (!info.isFile() || info.isSymbolicLink() || info.size !== bytes) throw new Error("A retained imagination fragment is missing or changed."); }
  } finally { store.close(); }
}
export async function rekeyImaginationWorkspace(root: string, sourceId: string, targetId: string): Promise<void> {
  await validateImaginationWorkspace(root, sourceId); const reader = await ImaginationCaptureStore.open(root, sourceId); if (!reader) return; reader.close(); const store = (await ImaginationCaptureStore.open(root, sourceId, true))!;
  try { store.rekey(targetId); } finally { store.close(); } await validateImaginationWorkspace(root, targetId);
}
/** Complete parent custody remains in raw immutable database archives. */
export async function mergeImaginationWorkspace(sourceRoot: string, targetRoot: string, owner: string, parentId: string, reserve: (bytes: number) => Promise<void>): Promise<Map<string,string>> {
  if (sourceRoot === targetRoot) throw new Error("Capture merge requires a private destination, not source replacement.");
  const aliases = new Map<string,string>(); await validateImaginationWorkspace(sourceRoot, owner); const source = await ImaginationCaptureStore.open(sourceRoot, owner); if (!source) return aliases;
  const target = (await ImaginationCaptureStore.open(targetRoot, owner, true))!;
  const archiveRoot = join(targetRoot, "imagination", "archives"), temporary = join(archiveRoot, `${randomUUID()}.sqlite3`);
  try {
    await mkdir(archiveRoot, { recursive: true });
    await reserve((await lstat(join(sourceRoot, "imagination", "workspace.sqlite3"))).size + 65536);
    await snapshotMutableSqliteIsolated(join(sourceRoot, "imagination", "workspace.sqlite3"), temporary);
    const sha = await imaginationFileHash(temporary); await copyMutableFileIsolated(temporary, join(archiveRoot, `${sha}.sqlite3`), { expectedSha256: sha });
    await rm(temporary, { force: true });
    for (const kind of ["blobs", "window-blobs", "archives"]) {
      const directory = join(sourceRoot, "imagination", kind); const files = await readdir(directory).catch(error => { if (error.code === "ENOENT") return []; throw error; });
      for (const file of files) { const from = join(directory, file); await reserve((await lstat(from)).size); await copyMutableFileIsolated(from, join(targetRoot, "imagination", kind, file), { expectedSha256: file.split(".")[0] }); }
    }
    for (const row of source.db.prepare("SELECT id FROM recordings ORDER BY sequence").iterate()) {
      const old = source.get(String(row.id))!, digest = createHash("sha256");
      for (const fragment of source.db.prepare("SELECT body FROM fragments WHERE recording_id=? ORDER BY fragment_index").iterate(old.id)) {
        const { recordingId: _recording, index: _index, ...body } = JSON.parse(String(fragment.body)); digest.update(JSON.stringify(body));
      }
      const id = createHash("sha256").update(JSON.stringify({ originalId: old.inheritedFrom?.recordingId ?? old.id, mode: old.mode, state: old.state,
        startedAt: old.startedAt, endedAt: old.endedAt ?? null, requestedLookbackSeconds: old.requestedLookbackSeconds,
        availableLookbackSeconds: old.availableLookbackSeconds, gapCount: old.gapCount, fragments: digest.digest("hex") })).digest("hex"), requestId = `inherited-${id}`;
      const existing = target.request(requestId); if (existing) { aliases.set(old.id, existing.id); continue; }
      const adopted = target.create(requestId, old.mode, old.requestedLookbackSeconds, old.availableLookbackSeconds, old.startedAt);
      aliases.set(old.id, adopted.id);
      let after = -1; do { const page = source.fragments(old.id, after, 100); for (const fragment of page.entries) {
        const { recordingId: _recording, index: _index, ...body } = fragment; target.append(adopted.id, body); }
        if (page.nextAfterIndex === undefined) break; after = page.nextAfterIndex;
      } while (true);
      const current = target.get(adopted.id)!; target.update(current.id, current.revision, value => { value.state = old.state === "recording" ? "dormant" : old.state;
        value.gapCount = old.gapCount; value.inheritedFrom = old.inheritedFrom ?? { brainId: parentId, recordingId: old.id }; if (old.endedAt) value.endedAt = old.endedAt; if (old.timelineEndMs !== undefined) value.timelineEndMs = old.timelineEndMs; });
    }
    const windows = target.window(), ids = new Set(windows.map(value => value.id)); for (const item of source.window()) {
      const inheritedId = item.inheritedId ?? item.id, id = captureHash({ ...item, id: inheritedId, inheritedId });
      if (!ids.has(id)) { windows.push({ ...item, id, inheritedId }); ids.add(id); }
    }
    target.setWindow(windows, true, new Date().toISOString());
    const settings = target.settings(); target.db.prepare("UPDATE meta SET value=? WHERE key='settings'").run(JSON.stringify({ ...settings, dormant: true }));
    target.validate();
  } finally { source.close(); target.close(); await rm(temporary, { force: true }); }
  return aliases;
}
