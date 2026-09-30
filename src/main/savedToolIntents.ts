import { lstat, readFile, readdir } from "node:fs/promises";
import { basename, join } from "node:path";

/** Validate inert historical evidence; this never authorizes or replays an action. */
export async function validateSavedToolIntent(path: string): Promise<void> {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink() || info.size > 1024 * 1024) {
    throw new Error("Saved tool intent is not bounded, regular metadata.");
  }
  const value = JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
  const safeId = (text: unknown) => typeof text === "string" && /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(text);
  if (value.format !== "omni-authorized-tool-intent" || value.formatVersion !== 1 ||
    value.state !== "authorized-before-side-effects" || typeof value.id !== "string" ||
    !/^[a-f0-9-]{36}$/i.test(value.id) || basename(path) !== `${value.id}.json` ||
    !safeId(value.brainId) || !safeId(value.requestId) || !safeId(value.toolId) ||
    typeof value.action !== "string" || !value.action || value.action.includes("\0") ||
    typeof value.startedAt !== "string" || !Number.isFinite(Date.parse(value.startedAt)) ||
    !["ask", "auto", "full"].includes(String(value.permission)) ||
    typeof value.permissionRevision !== "string" || !value.permissionRevision ||
    !Array.isArray(value.argumentKeys) || value.argumentKeys.some(key => typeof key !== "string") ||
    typeof value.argumentSha256 !== "string" || !/^[a-f0-9]{64}$/.test(value.argumentSha256) ||
    value.rawArgumentValuesRetained !== false) throw new Error("Saved tool intent metadata is invalid.");
  // Original ownership is intentionally not rewritten on import/fork. This is
  // evidence of an ancestral action, not the new identity's current grant.
}

export async function savedToolIntentFiles(engineDirectory: string): Promise<Map<string, string>> {
  const directory = join(engineDirectory, "operational-tool-intents");
  const info = await lstat(directory).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return undefined;
    throw error;
  });
  const files = new Map<string, string>();
  if (!info) return files;
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Saved tool intent directory is unsafe.");
  for (const entry of (await readdir(directory, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isFile() || !/^[a-f0-9-]{36}\.json$/i.test(entry.name)) throw new Error("Saved tool intent entry is unsafe.");
    const path = join(directory, entry.name);
    await validateSavedToolIntent(path);
    files.set(entry.name, path);
  }
  return files;
}
