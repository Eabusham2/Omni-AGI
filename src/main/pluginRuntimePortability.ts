import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, mkdir, opendir, rename, rm } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { copyMutableFileIsolated, snapshotMutableSqliteIsolated } from "./mutableFileIsolation";
import { PLUGIN_RUNTIME_WORKSPACE, PluginRuntimeStore } from "./pluginRuntimeStore";
import { RUNTIME_SHA, runtimeId } from "./pluginRuntimeValidation";
import { sqliteFilename } from "./sqliteFilename";

export interface PluginRuntimeCustodyHooks { signal?: AbortSignal; checkDisk?(bytes: number): Promise<void>; mode?: "merge" | "history-only" }
export const safePluginRuntimeWorkspacePath = (value: string): boolean => value === PLUGIN_RUNTIME_WORKSPACE || /^plugin-runtime\/archives\/[a-f0-9]{64}\.sqlite3$/.test(value);
const exists = async (path: string) => lstat(path).then(() => true, (error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return false; throw error; });
async function digest(path: string, signal?: AbortSignal): Promise<string> { const hash = createHash("sha256"); for await (const block of createReadStream(path)) { signal?.throwIfAborted(); hash.update(block); } return hash.digest("hex"); }
async function ownerOf(path: string): Promise<string> { const info = await lstat(path); if (!info.isFile() || info.isSymbolicLink()) throw new Error("Unsafe inherited runtime database."); const db = new DatabaseSync(sqliteFilename(path), { readOnly: true, allowExtension: false });
  try { db.exec("PRAGMA trusted_schema=OFF;"); return runtimeId(db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value); } finally { db.close(); } }

export async function validateCapturedPluginRuntimeWorkspace(path: string, expectedOwner?: string): Promise<void> { const store = await PluginRuntimeStore.openFile(path, expectedOwner ?? await ownerOf(path)); try { store.validate(); } finally { store.close(); } }
export async function validatePluginRuntimeWorkspace(root: string, owner: string): Promise<void> {
  const store = await PluginRuntimeStore.open(root, owner, { readOnly: true }); if (!store) return;
  try {
    store.validate();
    for (const row of store.db.prepare("SELECT sha256,owner,bytes FROM archives").iterate()) { const path = join(root, "plugin-runtime", "archives", `${row.sha256}.sqlite3`), info = await lstat(path);
      if (!info.isFile() || info.isSymbolicLink() || info.size !== Number(row.bytes) || await digest(path) !== row.sha256) throw new Error("Inherited runtime archive checksum failed."); await validateCapturedPluginRuntimeWorkspace(path, String(row.owner)); }
    const directory = join(root, "plugin-runtime"), entries = await opendir(directory);
    for await (const entry of entries) { if (entry.name === "workspace.sqlite3" || /^workspace.sqlite3-(?:journal|wal|shm)$/.test(entry.name)) continue;
      if (entry.name !== "archives" || !entry.isDirectory() || entry.isSymbolicLink()) throw new Error("Unsupported runtime workspace payload.");
      for await (const archive of await opendir(join(directory, "archives"))) if (!archive.isFile() || archive.isSymbolicLink() || !/^[a-f0-9]{64}\.sqlite3$/.test(archive.name) || !store.db.prepare("SELECT 1 FROM archives WHERE sha256=?").get(archive.name.slice(0, 64))) throw new Error("Unindexed runtime archive; custody is incomplete.");
    }
  } finally { store.close(); }
}

/** Immutable checked source, not an executable owner or imported activation. */
export async function openPluginRuntimeSource(root: string, owner: string, sourceSha256: string): Promise<PluginRuntimeStore> {
  if (!RUNTIME_SHA.test(sourceSha256)) throw new Error("Invalid inherited runtime source SHA."); const current = await PluginRuntimeStore.open(root, owner, { readOnly: true }); if (!current) throw new Error("Runtime history is unavailable.");
  let descriptor: { owner: string; bytes: number }; try { const row = current.db.prepare("SELECT owner,bytes FROM archives WHERE sha256=?").get(sourceSha256); if (!row) throw new Error("Unknown inherited runtime source."); descriptor = { owner: runtimeId(row.owner), bytes: Number(row.bytes) }; } finally { current.close(); }
  const path = join(root, "plugin-runtime", "archives", `${sourceSha256}.sqlite3`), info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink() || info.size !== descriptor.bytes || await digest(path) !== sourceSha256) throw new Error("Inherited runtime source bytes changed.");
  const store = await PluginRuntimeStore.openFile(path, descriptor.owner); store.bindReadGuard(async () => {
    const now = await lstat(path); if (!now.isFile() || now.isSymbolicLink() || now.dev !== info.dev || now.ino !== info.ino || now.size !== info.size || now.mtimeMs !== info.mtimeMs || now.ctimeMs !== info.ctimeMs || await digest(path) !== sourceSha256) throw new Error("Inherited runtime source changed during the read.");
    const after = await lstat(path); if (after.dev !== now.dev || after.ino !== now.ino || after.size !== now.size || after.mtimeMs !== now.mtimeMs || after.ctimeMs !== now.ctimeMs) throw new Error("Inherited runtime source changed during verification.");
  });
  try { await store.verifyRead(); return store; } catch (error) { store.close(); throw error; }
}

async function archiveCurrent(root: string, owner: string, destination: string, hooks: PluginRuntimeCustodyHooks): Promise<{ sha256: string; owner: string; bytes: number }> {
  hooks.signal?.throwIfAborted(); const path = join(root, PLUGIN_RUNTIME_WORKSPACE), directory = join(destination, "plugin-runtime", "archives"); await mkdir(directory, { recursive: true, mode: 0o700 });
  const temporary = join(directory, `.snapshot-${randomUUID()}.sqlite3`);
  try { await hooks.checkDisk?.((await lstat(path)).size + 65536); await snapshotMutableSqliteIsolated(path, temporary); hooks.signal?.throwIfAborted(); await validateCapturedPluginRuntimeWorkspace(temporary, owner);
    const sha256 = await digest(temporary, hooks.signal), bytes = (await lstat(temporary)).size, target = join(directory, `${sha256}.sqlite3`);
    if (await exists(target)) { if (await digest(target, hooks.signal) !== sha256) throw new Error("Runtime archive collision."); } else await rename(temporary, target);
    return { sha256, owner, bytes };
  } finally { await rm(temporary, { force: true }); }
}

export async function rekeyPluginRuntimeWorkspace(root: string, sourceOwner: string, targetOwner: string, hooks: PluginRuntimeCustodyHooks = {}): Promise<void> {
  if (!await exists(join(root, PLUGIN_RUNTIME_WORKSPACE))) return; runtimeId(sourceOwner); runtimeId(targetOwner);
  const actualOwner = await ownerOf(join(root, PLUGIN_RUNTIME_WORKSPACE)); if (actualOwner === targetOwner && sourceOwner !== targetOwner) { await validatePluginRuntimeWorkspace(root, targetOwner); return; }
  if (actualOwner !== sourceOwner) throw new Error("Runtime copy owner changed."); await validatePluginRuntimeWorkspace(root, sourceOwner);
  const archived = await archiveCurrent(root, sourceOwner, root, hooks), store = await PluginRuntimeStore.open(root, sourceOwner, { ...hooks }); if (!store) throw new Error("Runtime copy disappeared.");
  try { await store.addArchive(archived.sha256, archived.owner, archived.bytes); await store.rekey(targetOwner); } finally { store.close(); }
  await validatePluginRuntimeWorkspace(root, targetOwner);
}

/** Preserve all parent bytes. Only equal/unambiguous variables enter default state. */
export async function mergePluginRuntimeWorkspace(stage: string, root: string, owner: string, parentId: string, hooks: PluginRuntimeCustodyHooks = {}): Promise<void> {
  runtimeId(parentId); const source = await PluginRuntimeStore.open(stage, owner, { readOnly: true }); if (!source) return;
  let target: PluginRuntimeStore | undefined;
  try {
    await validatePluginRuntimeWorkspace(stage, owner); target = await PluginRuntimeStore.open(root, owner, { create: true, ...hooks }); if (!target) throw new Error("Merged runtime custody unavailable.");
    for (const row of source.db.prepare("SELECT sha256,owner,bytes FROM archives ORDER BY sha256").iterate()) {
      hooks.signal?.throwIfAborted(); await hooks.checkDisk?.(Number(row.bytes)); const name = `${row.sha256}.sqlite3`, path = join(root, "plugin-runtime", "archives", name);
      if (!await exists(path)) await copyMutableFileIsolated(join(stage, "plugin-runtime", "archives", name), path, { expectedSha256: String(row.sha256), mode: 0o600 });
      else if (await digest(path, hooks.signal) !== row.sha256) throw new Error("Merged runtime source collision.");
      await target.addArchive(String(row.sha256), String(row.owner), Number(row.bytes));
    }
    const archived = await archiveCurrent(stage, owner, root, hooks), prior = target.db.prepare("SELECT 1 FROM archives WHERE sha256=?").get(archived.sha256);
    if (prior) return;
    // Commit variable/conflict projection before publishing the source index;
    // an interrupted staged merge remains incomplete, never an active module.
    if (hooks.mode !== "history-only") {
      for (const row of source.db.prepare("SELECT plugin_hash,module_id,key,revision,sha256,length(CAST(payload AS BLOB)) AS bytes FROM values_current ORDER BY plugin_hash,module_id,key").iterate()) {
        hooks.signal?.throwIfAborted(); const plugin = String(row.plugin_hash), module = String(row.module_id), key = String(row.key);
        // This is an I/O transfer page, not a product-data limit. Large values
        // stay complete in the source and need explicit admitted selection.
        if (Number(row.bytes) > 65536) { await target.inheritConflict(plugin, module, key); continue; }
        const payload = source.db.prepare("SELECT payload FROM values_current WHERE plugin_hash=? AND module_id=? AND key=?").get(plugin, module, key)?.payload;
        await target.mergeValue(plugin, module, key, Number(row.revision), String(payload), String(row.sha256), archived.sha256);
      }
      for (const row of source.db.prepare("SELECT * FROM state_conflicts ORDER BY plugin_hash,module_id,key").iterate()) await target.inheritConflict(String(row.plugin_hash), String(row.module_id), String(row.key));
    }
    await target.record("inherited-workspace", { parentId, sourceSha256: archived.sha256, sourceOwner: archived.owner, dormant: true }); await target.addArchive(archived.sha256, archived.owner, archived.bytes);
  } finally { source.close(); target?.close(); }
  await validatePluginRuntimeWorkspace(root, owner);
}
