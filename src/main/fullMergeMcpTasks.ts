import { lstat, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { assertFullMergeOriginSeeding, type FullMergeAssemblyInput } from "./fullBrainMerge";
import { mergeMcpTaskCustody, mcpTaskCustodyFiles } from "./mcpTaskCustody";
import { mcpTaskEnginePath } from "./mcpTaskArchive";

/** Source trees came from the coordinator's coherent task captures, not a raw
 * walk of live operational files. Exact parent/history records stay dormant. */
export async function adoptFullMergeMcpTasks(input: FullMergeAssemblyInput): Promise<void> {
  const target = join(input.targetDirectory, "engine"); await mkdir(target, { recursive: true });
  const origin = join(target, "origin"); let seedOrigin = false;
  for (const parent of input.parents) {
    const scopes = new Set<string>();
    for (const file of input.files(parent.brainId)) {
      const scope = mcpTaskEnginePath(file.path); if (scope) scopes.add(scope.engine);
    }
    if (!scopes.size) continue;
    const originInfo = await lstat(origin).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (originInfo) {
      if (!originInfo.isDirectory() || originInfo.isSymbolicLink()) throw new Error("New merged-origin MCP task custody needs a real unpublished origin directory.");
      assertFullMergeOriginSeeding(input); seedOrigin = true;
    }
    const root = await input.materializeParent(parent.brainId);
    for (const scope of [...scopes].sort()) {
      input.signal?.throwIfAborted(); await input.checkpoint("Preserving exact dormant parent MCP task custody");
      await mergeMcpTaskCustody(join(root, ...scope.split("/")), target, { signal: input.signal, admit: input.admit,
        checkDisk: bytes => input.reserve(bytes, "complete dormant MCP task custody") });
    }
  }
  await mcpTaskCustodyFiles(target, { signal: input.signal, admit: input.admit });
  if (seedOrigin) {
    assertFullMergeOriginSeeding(input);
    await mergeMcpTaskCustody(target, origin, { signal: input.signal, admit: input.admit,
      checkDisk: bytes => input.reserve(bytes, "complete dormant starting-origin MCP task custody") });
  }
}
