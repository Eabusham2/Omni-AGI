import { lstat, mkdir, mkdtemp, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { FullMergeAssemblyInput } from "./fullBrainMerge";
import { mergeMcpAppArtifacts, validateMcpAppArtifactStore } from "./mcpAppArtifacts";
import { copyMutableFileIsolated } from "./mutableFileIsolation";
import { sqliteFilename } from "./sqliteFilename";

/** Extract only Apps from the writer-independent, lease-consistent archive.
 * The general parent tree can have been visited before this archive was sealed.
 * Reusing that raw copy would race a completed App capture with its blobs. */
async function materializeAppClosure(input: FullMergeAssemblyInput, parentRoot: string, brainId: string, target: string): Promise<void> {
  const archive = join(parentRoot, "portable-learning-workspace"), indexPath = join(archive, "inventory.sqlite3"), info = await lstat(indexPath);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Merge parent has no regular sealed workspace inventory.");
  const db = new DatabaseSync(sqliteFilename(indexPath), { readOnly: true });
  try {
    db.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-512");
    if (db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok" ||
      db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== "omni-learning-workspace-archive-1" ||
      db.prepare("SELECT value FROM meta WHERE key='complete'").get()?.value !== "1" ||
      db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) throw new Error("Merge parent App inventory is incomplete or belongs to another instance.");
    let count = 0, hasIndex = false;
    for (const row of db.prepare("SELECT path,sha256,bytes FROM files WHERE path GLOB 'mcp-apps/*' ORDER BY path").iterate()) {
      input.signal?.throwIfAborted(); const path = String(row.path), sha = String(row.sha256);
      if (!/^(?:mcp-apps\/index\.sqlite3|mcp-apps\/blobs\/[a-f0-9]{64}\.data)$/.test(path) || !/^[a-f0-9]{64}$/.test(sha) ||
        !Number.isSafeInteger(row.bytes) || Number(row.bytes) < 0) throw new Error("Merge parent App inventory contains an unsupported payload.");
      const blob = join(archive, "blobs", `${sha}.data`), file = await lstat(blob);
      if (!file.isFile() || file.isSymbolicLink() || file.size !== row.bytes) throw new Error("Merge parent App archive lost its exact source bytes.");
      await input.reserve(file.size, "sealed parent MCP App evidence"); await input.checkpoint(`Restoring retained App evidence from ${brainId}`);
      const destination = join(target, ...path.split("/")); await mkdir(dirname(destination), { recursive: true, mode: 0o700 });
      await copyMutableFileIsolated(blob, destination, { expectedSha256: sha });
      count++; if (path === "mcp-apps/index.sqlite3") hasIndex = true;
    }
    if (count && !hasIndex) throw new Error("Merge App payload has no committed artifact index.");
  } finally { db.close(); }
  await validateMcpAppArtifactStore(target, brainId, { signal: input.signal });
}

/** Retained UI evidence, never a transferred connection, grant or request. */
export async function adoptFullMergeMcpApps(input: FullMergeAssemblyInput): Promise<void> {
  const sources: Array<{ root: string; brainId: string }> = [];
  try {
    for (const parent of input.parents) {
      const parentRoot = await input.materializeParent(parent.brainId), stage = await mkdtemp(join(input.targetDirectory, ".mcp-app-parent-"));
      sources.push({ root: stage, brainId: parent.brainId }); await materializeAppClosure(input, parentRoot, parent.brainId, stage);
    }
    await mergeMcpAppArtifacts(sources, input.targetDirectory, input.targetBrainId, { signal: input.signal,
      checkDisk: bytes => input.reserve(bytes, "retained untrusted MCP App artifacts") });
  } finally { for (const source of sources) await rm(source.root, { recursive: true, force: true }); }
}
