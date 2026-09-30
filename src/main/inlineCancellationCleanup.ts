import { lstat, realpath, rm } from "node:fs/promises";
import { join, resolve } from "node:path";

/** After confirmed PID exit, remove only its app-owned, unpromoted action stage. */
export async function cleanupCancelledInlineStage(brainDirectory: string, actionId: string): Promise<boolean> {
  if (!/^[a-f0-9]{32}$/i.test(actionId)) return false;
  const engine = resolve(brainDirectory, "engine");
  const canonicalEngine = await realpath(engine);
  const parent = join(engine, ".inline-imagination");
  const parentStat = await lstat(parent).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return undefined;
    throw error;
  });
  if (!parentStat) return true;
  if (!parentStat.isDirectory() || parentStat.isSymbolicLink() ||
      await realpath(parent) !== join(canonicalEngine, ".inline-imagination")) return false;
  const stage = join(parent, actionId);
  await rm(stage, { recursive: true, force: true });
  return await lstat(stage).then(() => false, (error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return true;
    throw error;
  });
}
