import { createHash } from "node:crypto";
import { lstat, mkdir, open, realpath, unlink } from "node:fs/promises";
import { basename, join } from "node:path";
import type { ToolInvocation, ToolPermissionLevel } from "../shared/types";

/** Immutable operational intent, independent of the live neural checkpoint. */
export async function persistAuthorizedToolIntent(
  brainDirectory: string,
  invocation: ToolInvocation,
  identity: { id: string; requestId: string; startedAt: string; permission: ToolPermissionLevel; permissionRevision: string }
): Promise<string> {
  if (!/^[a-f0-9-]{36}$/i.test(identity.id)) throw new Error("Tool intent identity is invalid.");
  if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(identity.requestId)) throw new Error("Tool intent request ownership is invalid.");
  if (!["ask", "auto", "full"].includes(identity.permission)) throw new Error("Disabled tools cannot publish authorized intent.");
  const owned = await lstat(brainDirectory);
  if (!owned.isDirectory() || owned.isSymbolicLink()) throw new Error("Tool intent brain owner must be a real directory.");
  const root = await realpath(brainDirectory);
  if (basename(root) !== invocation.brainId) throw new Error("Tool intent directory does not match its brain owner.");
  let directory = root;
  for (const name of ["engine", "operational-tool-intents"]) {
    directory = join(directory, name);
    await mkdir(directory).catch((error: NodeJS.ErrnoException) => { if (error.code !== "EEXIST") throw error; });
    const info = await lstat(directory);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Tool intent directory must be an owned real directory.");
  }
  const argumentsJson = JSON.stringify(invocation.arguments);
  const receipt = {
    format: "omni-authorized-tool-intent", formatVersion: 1,
    state: "authorized-before-side-effects", id: identity.id,
    brainId: invocation.brainId, requestId: identity.requestId,
    toolId: invocation.toolId, action: invocation.action,
    startedAt: identity.startedAt,
    permission: identity.permission, permissionRevision: identity.permissionRevision,
    ...(typeof invocation.arguments.neuralActionId === "string" && /^[a-f0-9]{32}$/i.test(invocation.arguments.neuralActionId)
      ? { neuralActionId: invocation.arguments.neuralActionId } : {}),
    ...(typeof invocation.arguments.chatTurnId === "string" && /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(invocation.arguments.chatTurnId)
      ? { chatTurnId: invocation.arguments.chatTurnId } : {}),
    argumentKeys: Object.keys(invocation.arguments).sort(),
    argumentSha256: createHash("sha256").update(argumentsJson).digest("hex"),
    rawArgumentValuesRetained: false
  };
  const path = join(directory, `${identity.id}.json`);
  const handle = await open(path, "wx", 0o600);
  try {
    await handle.writeFile(`${JSON.stringify(receipt)}\n`, "utf8");
    await handle.sync();
  } catch (error) {
    await handle.close();
    await unlink(path).catch(() => undefined);
    throw error;
  }
  await handle.close();
  // File sync is mandatory. Directory sync is additionally supported on
  // POSIX; Windows cannot open directory handles through Node's fs API.
  if (process.platform !== "win32") {
    const parent = await open(directory, "r");
    try { await parent.sync(); } finally { await parent.close(); }
  }
  return path;
}
