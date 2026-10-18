import { constants } from "node:fs";
import { chmod, copyFile, lstat, mkdir, open, opendir, realpath } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { MCP_TASK_DIRECTORY, mcpTaskReadBytes, mcpTaskReadJson, withMcpTaskSnapshot, writeMcpTaskBytes, type McpTaskReadOptions } from "./mcpTaskStore";
import { checkedMcpTaskRecord, mcpTaskObject, mcpTaskSha, MCP_TASK_REF, MCP_TASK_SHA } from "./mcpTaskProtocol";

export interface McpTaskCustodyOptions extends McpTaskReadOptions { checkDisk?(bytes: number): Promise<void>; }
/** Namespace filter only; content/source/closure checks remain mandatory. */
export function safeMcpTaskCustodyPath(local: string): boolean {
  if (typeof local !== "string" || local.includes("\\") || local.includes("\0")) return false;
  const interrupted = /^(.*\.json)\.([a-f0-9-]{36})\.next$/i.exec(local);
  if (interrupted) return MCP_TASK_REF.test(interrupted[2]!) && safeMcpTaskCustodyPath(interrupted[1]!);
  const parts = local.split("/"); if (parts.shift() !== MCP_TASK_DIRECTORY || parts.some(part => !part || part === "." || part === "..")) return false;
  while (parts[0] === "lineage") { if (!MCP_TASK_SHA.test(parts[1] ?? "")) return false; parts.splice(0, 2); }
  if (parts.length === 1) return parts[0]!.endsWith(".json") && MCP_TASK_REF.test(parts[0]!.slice(0, -5));
  return parts.length === 2 && ["history", "payloads"].includes(parts[0]!) && parts[1]!.endsWith(".json") && MCP_TASK_SHA.test(parts[1]!.slice(0, -5));
}
const missing = async (path: string): Promise<boolean> => { try { await lstat(path); return false; } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return true; throw error; } };
async function metadataAdmission(options: McpTaskCustodyOptions, bytes: number): Promise<void> {
  if (!Number.isSafeInteger(bytes) || bytes < 0) throw new Error("MCP task descriptor allocation exceeds exact planning."); options.signal?.throwIfAborted();
  if (options.admit) await options.admit(bytes, "MCP task closure descriptors");
  else if (bytes > freemem() || bytes > getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) throw new Error("MCP task closure metadata awaits physical RAM.");
  options.signal?.throwIfAborted();
}

/** Read-only closure. Call while holding the source's withMcpTaskSnapshot lease. */
export async function mcpTaskCustodyFiles(engineRoot: string, options: McpTaskCustodyOptions = {}): Promise<Map<string, string>> {
  const files = new Map<string, string>(); if (await missing(engineRoot)) return files;
  const root = await realpath(engineRoot), base = join(root, MCP_TASK_DIRECTORY); if (await missing(base)) return files;
  const roots = [MCP_TASK_DIRECTORY]; let descriptorBytes = 65536;
  while (roots.length) {
    const prefix = roots.pop()!, path = join(root, prefix), stat = await lstat(path);
    if (!stat.isDirectory() || stat.isSymbolicLink() || resolve(await realpath(path)) !== path) throw new Error("MCP task custody contains a symbolic/non-directory namespace.");
    const directory = await opendir(path);
    for await (const entry of directory) {
      options.signal?.throwIfAborted(); const local = `${prefix}/${entry.name}`;
      if (entry.isSymbolicLink()) throw new Error("MCP task custody contains a symbolic entry.");
      if (entry.isDirectory()) {
        const suffix = local.slice(MCP_TASK_DIRECTORY.length + 1);
        if (!/^(?:(?:lineage\/[a-f0-9]{64}\/)*)(?:history|payloads|lineage(?:\/[a-f0-9]{64})?)$/.test(suffix)) throw new Error("MCP task custody has an unknown namespace.");
        descriptorBytes += Buffer.byteLength(local) * 4 + 512; await metadataAdmission(options, descriptorBytes); roots.push(local); continue;
      }
      if (!entry.isFile()) throw new Error("MCP task custody contains non-regular data.");
      if (!safeMcpTaskCustodyPath(local)) throw new Error("MCP task custody contains an unsafe namespace entry.");
      if (entry.name.endsWith(".next")) {
        // An interrupted exact owned publication remains raw dormant evidence;
        // never parse it as a task/result, adopt it, finish it or execute it.
        await mcpTaskReadBytes(root, local, options); descriptorBytes += Buffer.byteLength(local) * 8 + 1024;
        await metadataAdmission(options, descriptorBytes); files.set(local, join(root, local)); continue;
      }
      const saved = await mcpTaskReadJson(root, local, options), name = entry.name.slice(0, -5);
      if (prefix.endsWith("/payloads")) { if (!MCP_TASK_SHA.test(name) || saved.sha256 !== name) throw new Error("MCP task payload differs from its content address."); mcpTaskObject(saved.value, "task payload"); }
      else {
        const row = checkedMcpTaskRecord(saved.value);
        if (prefix.endsWith("/history") ? !MCP_TASK_SHA.test(name) || saved.sha256 !== name : !MCP_TASK_REF.test(name) || name !== row.taskRef) throw new Error("MCP task source record identity changed.");
        if (row.result) {
          const scope = prefix.endsWith("/history") ? prefix.slice(0, -"/history".length) : prefix;
          const payload = await mcpTaskReadJson(root, `${scope}/payloads/${row.result.payloadSha256}.json`, options);
          if (payload.sha256 !== row.result.payloadSha256 || row.result.receiptId !== mcpTaskSha(JSON.stringify([row.source, row.taskRef, row.remote!.taskId, row.result.kind, row.result.payloadSha256]))) throw new Error("MCP task final receipt lost its exact original payload/source binding.");
        }
      }
      descriptorBytes += Buffer.byteLength(local) * 8 + 1024; await metadataAdmission(options, descriptorBytes); files.set(local, join(root, local));
    }
  }
  return files;
}

/** Fast coherent operational snapshot, not a lease around the full neural export.
 * The caller owns the staging directory and removes it after consuming its paths. */
export async function captureMcpTaskCustody(sourceEngineRoot: string, stagingEngineRoot: string, options: McpTaskCustodyOptions = {}): Promise<Map<string, string>> {
  if (await missing(sourceEngineRoot)) return new Map();
  const source = await realpath(sourceEngineRoot);
  const stageExists = !await missing(stagingEngineRoot), prospective = stageExists ? await realpath(stagingEngineRoot) : join(await realpath(dirname(stagingEngineRoot)), basename(stagingEngineRoot));
  if (stageExists && (await lstat(stagingEngineRoot)).isSymbolicLink()) throw new Error("MCP task snapshot staging cannot be a symbolic directory.");
  const nested = (parent: string, child: string): boolean => { const path = relative(parent, child); return path === "" || !isAbsolute(path) && path !== ".." && !path.startsWith(`..${sep}`); };
  if (nested(source, prospective) || nested(prospective, source)) throw new Error("MCP task source and staging roots must be disjoint; source custody is read-only.");
  return withMcpTaskSnapshot(dirname(source), async () => {
    const files = await mcpTaskCustodyFiles(source, options); if (!files.size) return new Map();
    await mkdir(stagingEngineRoot, { recursive: true, mode: 0o700 }); const staging = await realpath(stagingEngineRoot);
    for (const [local, path] of files) {
      options.signal?.throwIfAborted(); const saved = await mcpTaskReadBytes(source, local, options), output = join(staging, local);
      await options.checkDisk?.(saved.bytes.length * 2 + 65536); await mkdir(dirname(output), { recursive: true, mode: 0o700 });
      if (await realpath(dirname(output)) !== dirname(output)) throw new Error("MCP task snapshot staging has symbolic ancestors.");
      // Source task/history/payload publication replaces inodes atomically;
      // FICLONE is an optimization only, never an unsafe writable hardlink.
      await copyFile(path, output, constants.COPYFILE_EXCL | constants.COPYFILE_FICLONE);
      await chmod(output, 0o600); const fd = await open(output, "r+"); try { await fd.sync(); } finally { await fd.close(); }
      if ((await mcpTaskReadBytes(staging, local, options)).sha256 !== saved.sha256) throw new Error("MCP task staging copy did not preserve exact source bytes.");
    }
    return mcpTaskCustodyFiles(staging, options);
  }, options.signal);
}

/** Copy all parent bytes into a content-addressed lineage, never merge live grants or rewrite source ids. */
export async function mergeMcpTaskCustody(sourceEngine: string, targetEngine: string, options: McpTaskCustodyOptions = {}): Promise<void> {
  const files = await mcpTaskCustodyFiles(sourceEngine, options); if (!files.size) return;
  const groups = new Map<string, Array<{ local: string; suffix: string; sha256: string }>>(); let groupBytes = 65536;
  for (const [local] of files) {
    let suffix = local.slice(MCP_TASK_DIRECTORY.length + 1), scope = "";
    // Existing ancestor groups are already independent custody, not new history
    // of the current parent. Flatten them rather than recursively duplicating.
    for (;;) { const inherited = /^lineage\/[a-f0-9]{64}\//.exec(suffix); if (!inherited) break; scope += inherited[0]; suffix = suffix.slice(inherited[0].length); }
    const saved = await mcpTaskReadBytes(sourceEngine, local, options), entries = groups.get(scope) ?? [];
    groupBytes += Buffer.byteLength(local) * 8 + 2048; await metadataAdmission(options, groupBytes);
    entries.push({ local, suffix, sha256: saved.sha256 }); groups.set(scope, entries);
  }
  for (const group of groups.values()) {
    group.sort((a, b) => a.suffix < b.suffix ? -1 : a.suffix > b.suffix ? 1 : 0);
    const closureSha = mcpTaskSha(JSON.stringify(group.map(({ suffix, sha256 }) => [suffix, sha256])));
    for (const { local, suffix, sha256: expected } of group) {
      options.signal?.throwIfAborted(); const target = `${MCP_TASK_DIRECTORY}/lineage/${closureSha}/${suffix}`;
      const saved = await mcpTaskReadBytes(sourceEngine, local, options); if (saved.sha256 !== expected) throw new Error("MCP task merge source changed after closure validation.");
      await options.checkDisk?.(saved.bytes.length * 2 + 65536);
      try { await writeMcpTaskBytes(targetEngine, target, saved.bytes); }
      catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
        if ((await mcpTaskReadBytes(targetEngine, target, options)).sha256 !== expected) throw new Error("MCP task lineage merge collided with different source bytes.");
      }
    }
  }
  await mcpTaskCustodyFiles(targetEngine, options);
}
