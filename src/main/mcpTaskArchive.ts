import { mkdir, lstat, realpath, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { captureMcpTaskCustody, mcpTaskCustodyFiles, safeMcpTaskCustodyPath, type McpTaskCustodyOptions } from "./mcpTaskCustody";
import { mcpTaskReadBytes, MCP_TASK_DIRECTORY } from "./mcpTaskStore";
import type { ExtractedZipArchive, StreamingZipSource } from "./streamingZip";
import { copyMutableFileIsolated } from "./mutableFileIsolation";

export const MCP_TASK_CUSTODY_REQUIRED_FEATURE = "mcp-task-custody-v1";
export interface McpTaskArchive { sources: StreamingZipSource[]; files: Map<string, { path: string; sha256: string; bytes: number }>; }
export function mcpTaskEnginePath(logical: string): { engine: string; local: string } | undefined {
  const marker = `/${MCP_TASK_DIRECTORY}/`, at = logical.indexOf(marker);
  if (at < 0) return undefined;
  const engine = logical.slice(0, at), local = logical.slice(at + 1);
  if (!/(?:^|\/)engine(?:\/origin)?$/.test(engine) || !safeMcpTaskCustodyPath(local)) throw new Error("Unsupported managed MCP task custody path.");
  return { engine, local };
}

/** Connected-parent payloads are opaque in ZIP but their logical names remain
 * available in the authenticated connected manifest. Validate every taskscope
 * before adoption, without materializing unrelated neural files or any grants. */
export async function validateEmbeddedMcpTaskSources(sources: Iterable<[string, StreamingZipSource]>, stagingRoot: string,
  options: McpTaskCustodyOptions = {}): Promise<boolean> {
  const scopes = new Set<string>(); let present = false;
  for (const [logical, source] of sources) {
    const scope = mcpTaskEnginePath(logical); if (!scope) continue;
    present = true; scopes.add(scope.engine); options.signal?.throwIfAborted();
    const target = join(stagingRoot, ...logical.split("/")); await mkdir(dirname(target), { recursive: true, mode: 0o700 });
    const bytes = source.sourcePath ? (await lstat(source.sourcePath)).size : source.contents?.byteLength;
    if (bytes === undefined) throw new Error("Embedded MCP task custody has no exact source bytes.");
    await options.checkDisk?.(bytes * 2 + 65536);
    if (source.sourcePath) await copyMutableFileIsolated(source.sourcePath, target);
    else await writeFile(target, source.contents!, { flag: "wx", mode: 0o600 });
  }
  for (const scope of scopes) await mcpTaskCustodyFiles(join(stagingRoot, ...scope.split("/")), options);
  return present;
}

/** The parent directory is caller-owned; each scope gets its own short coherent
 * capture. No live lease remains held while the neural/archive files stream. */
export async function captureMcpTaskArchive(scopes: { current: string; origin: string }, stagingRoot: string,
  options: McpTaskCustodyOptions = {}): Promise<McpTaskArchive> {
  const result: McpTaskArchive = { sources: [], files: new Map() };
  for (const scope of ["current", "origin"] as const) {
    const parent = join(stagingRoot, scope); await mkdir(parent, { recursive: true, mode: 0o700 });
    const engine = join(parent, "engine"), captured = await captureMcpTaskCustody(scopes[scope], engine, options);
    for (const [local, path] of captured) {
      const saved = await mcpTaskReadBytes(engine, local, options), name = `mcp-tasks/${scope}/${local}`;
      result.files.set(name, { path, sha256: saved.sha256, bytes: saved.bytes.length }); result.sources.push({ name, sourcePath: path });
    }
  }
  return result;
}

/** Structural detection only; actual closure/record/payload validation follows. */
export function hasMcpTaskArchivePayload(names: Iterable<string>): boolean {
  for (const name of names) if (name.startsWith("mcp-tasks/") || name.includes(`/${MCP_TASK_DIRECTORY}/`)) return true;
  return false;
}

/** Runs before identity reservation. Imported receipts remain raw source-owned
 * historical data; no server/session/permission grant can be reconstructed. */
export async function validateMcpTaskArchive(archive: ExtractedZipArchive, version: 1 | 2,
  features: readonly string[], options: McpTaskCustodyOptions = {}): Promise<Map<"current" | "origin", Map<string, string>>> {
  const present = hasMcpTaskArchivePayload(archive.entries.keys());
  if (present && (version !== 2 || !features.includes(MCP_TASK_CUSTODY_REQUIRED_FEATURE))) throw new Error("MCP task custody requires its declared format2 feature; no receipts were discarded or adopted.");
  const result = new Map<"current" | "origin", Map<string, string>>();
  const declared = [...archive.entries.keys()].filter(name => name.startsWith("mcp-tasks/"));
  let expected = 0;
  for (const scope of ["current", "origin"] as const) {
    const prefix = `mcp-tasks/${scope}/`, root = join(archive.root, "mcp-tasks", scope);
    const files = await mcpTaskCustodyFiles(root, options); result.set(scope, files); expected += files.size;
    for (const [local, path] of files) {
      const entry = archive.entries.get(`${prefix}${local}`);
      if (!safeMcpTaskCustodyPath(local) || !entry || entry.name !== `${prefix}${local}` || resolve(await realpath(entry.path)) !== resolve(path) ||
        entry.uncompressedBytes !== (await lstat(path)).size) throw new Error("MCP task archive entry does not match its exact validated closure.");
    }
  }
  if (declared.length !== expected || declared.some(name => {
    const scope = name.startsWith("mcp-tasks/current/") ? "current" : name.startsWith("mcp-tasks/origin/") ? "origin" : undefined;
    return !scope || !result.get(scope)?.has(name.slice(`mcp-tasks/${scope}/`.length));
  })) throw new Error("MCP task archive contains an unknown or orphan namespace.");
  return result;
}
