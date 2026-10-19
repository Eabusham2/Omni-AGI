import { createHash, randomUUID } from "node:crypto";
import { mkdir, lstat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { StreamingZipSource } from "./streamingZip";
import { copyMutableFileIsolated } from "./mutableFileIsolation";
import { API_TEACHER_DATABASE, ApiTeacherStore } from "./apiTeacherStore";
import { safeApiTeacherWorkspacePath, apiTeacherMediaReferenceFiles, validateApiTeacherWorkspace,
  type ApiTeacherPortabilityHooks } from "./apiTeacherPortability";
import { safeApiTeacherMediaPath, admitApiTeacherMedia } from "./apiTeacherMedia";
import { sqliteFilename } from "./sqliteFilename";

const SHA = /^[a-f0-9]{64}$/;
/** Exact teacher closure inside any current/origin/history/opaque-parent learning inventory.
 * Caller owns/cleans staging; no adoption, owner projection, provider or neural work. */
export async function validateApiTeacherMediaArchiveSources(sources: Iterable<[string, StreamingZipSource]>, stagingRoot: string,
  options: ApiTeacherPortabilityHooks = {}): Promise<boolean> {
  const entries = new Map(sources); let present = false;
  const materialize = async (source: StreamingZipSource, target: string, expected?: { sha256: string; bytes: number }) => {
    options.signal?.throwIfAborted();
    await admitApiTeacherMedia(65536, "teacher archive file-transfer window", options);
    const bytes = source.sourcePath ? (await lstat(source.sourcePath)).size : source.contents?.byteLength;
    if (bytes === undefined || expected && bytes !== expected.bytes) throw new Error("Teacher archive source length differs.");
    await options.checkDisk?.(bytes + 65536); await mkdir(dirname(target), { recursive: true, mode: 0o700 });
    if (source.sourcePath) await copyMutableFileIsolated(source.sourcePath, target, { expectedSha256: expected?.sha256, mode: 0o600 });
    else {
      await admitApiTeacherMedia(bytes * 2, "teacher archive inline payload", options);
      if (expected && createHash("sha256").update(source.contents!).digest("hex") !== expected.sha256) throw new Error("Teacher archive inline hash differs.");
      await writeFile(target, source.contents!, { flag: "wx", mode: 0o600 });
    }
  };
  for (const [name, source] of entries) {
    if (!/(?:^|\/)(?:portable-)?learning-workspace\/inventory\.sqlite3$/.test(name)) continue;
    const prefix = name.slice(0, -"inventory.sqlite3".length), root = join(stagingRoot, randomUUID());
    await mkdir(root, { recursive: false, mode: 0o700 });
    const inventory = join(root, "inventory.sqlite3"); await materialize(source, inventory);
    await admitApiTeacherMedia(256 * 1024, "teacher archive SQLite inventory cache", options);
    const database = new DatabaseSync(sqliteFilename(inventory), { readOnly: true, allowExtension: false });
    let owner = ""; const declared = new Set<string>();
    try {
      database.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-256; PRAGMA mmap_size=0;");
      if (database.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== "omni-learning-workspace-archive-1")
        throw new Error("Teacher archive learning inventory has an unsupported format.");
      owner = String(database.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value ?? "");
      for (const row of database.prepare("SELECT path,sha256,bytes FROM files WHERE path GLOB 'api-teachers/*' ORDER BY path").iterate()) {
        const path = String(row.path), sha256 = String(row.sha256), bytes = Number(row.bytes);
        if (!safeApiTeacherWorkspacePath(path) || !SHA.test(sha256) || !Number.isSafeInteger(bytes) || bytes < 1 || declared.has(path))
          throw new Error("Teacher archive has an unsafe or duplicated exact teacher inventory row.");
        const payload = entries.get(`${prefix}blobs/${sha256}.data`);
        if (!payload) throw new Error("Teacher archive is missing a declared teacher custody blob.");
        await materialize(payload, join(root, ...path.split("/")), { sha256, bytes }); declared.add(path);
      }
    } finally { database.close(); }
    if (!declared.size) continue;
    await validateApiTeacherWorkspace(root, owner, options.signal, options.admitMetadata);
    const references = await apiTeacherMediaReferenceFiles(root, owner, options);
    const store = await ApiTeacherStore.open(root, owner, async () => undefined, { create: false });
    try {
      const expected = new Set([API_TEACHER_DATABASE, ...store!.lineages().map(row => row.relativePath), ...references.keys()]);
      if (declared.size !== expected.size || [...declared].some(path => !expected.has(path)))
        throw new Error("Teacher archive includes unreferenced or incomplete media custody.");
      if (store!.jobs().some(job => job.responseModalities?.some(kind => kind !== "text"))) present = true;
    } finally { store?.close(); }
    if (references.size || [...declared].some(safeApiTeacherMediaPath)) present = true;
  }
  return present;
}
