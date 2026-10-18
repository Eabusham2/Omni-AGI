import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdir, open, opendir, rename, rm } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import type { EnvironmentOperationReceipt, ManagedEnvironmentRecord } from "../shared/environments";
import { environmentCanonical, environmentHash } from "./environmentFormat";
import {
  ENVIRONMENT_WORKSPACE_PATH,
  EnvironmentStore,
  validateEnvironmentOperation,
  validateEnvironmentRecord
} from "./environmentStore";
import { copyMutableFileIsolated } from "./mutableFileIsolation";

const terminal = new Set(["complete", "cancelled", "failed", "interrupted"]);
async function exists(path: string): Promise<boolean> { return Boolean(await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; })); }

interface SavedFile { path: string; relativePath: string; bytes: number; sha256: string }
async function *files(root: string, current = root): AsyncGenerator<SavedFile> {
  const directory = await lstat(current); if (!directory.isDirectory() || directory.isSymbolicLink()) throw new Error("Environment workspace contains an unsafe directory.");
  const entries = await opendir(current);
  for await (const entry of entries) {
    if (!entry.name || entry.name === "." || entry.name === ".." || entry.name.includes("\\") || entry.name.includes("\0")) throw new Error("Environment workspace contains a nonportable path component.");
    const path = join(current, entry.name), info = await lstat(path);
    if (info.isSymbolicLink()) throw new Error("Environment workspace contains a symbolic link.");
    if (info.isDirectory()) { yield * files(root, path); continue; }
    if (!info.isFile()) throw new Error("Environment workspace contains an unsupported filesystem object.");
    const handle = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
    try {
      const before = await handle.stat(); if (!before.isFile() || before.dev !== info.dev || before.ino !== info.ino) throw new Error("Environment workspace file changed before verification.");
      const digest = createHash("sha256"), stream = handle.createReadStream({ autoClose: false }); for await (const chunk of stream) digest.update(chunk);
      const after = await handle.stat(); if (after.dev !== before.dev || after.ino !== before.ino || after.size !== before.size || after.mtimeMs !== before.mtimeMs) throw new Error("Environment workspace changed during verification.");
      const fromRoot = relative(root, path); if (!fromRoot || fromRoot === ".." || fromRoot.startsWith(`..${sep}`) || isAbsolute(fromRoot)) throw new Error("Environment workspace path escaped its owner.");
      yield { path, relativePath: fromRoot.split(sep).join("/"), bytes: before.size, sha256: digest.digest("hex") };
    } finally { await handle.close(); }
  }
}

export function safeEnvironmentWorkspacePath(path: string): boolean {
  if (path === ENVIRONMENT_WORKSPACE_PATH || /^environments\/instances\/[a-f0-9]{32}\/sandbox\.sb$/.test(path)) return true;
  const match = /^environments\/instances\/[a-f0-9]{32}\/workspace(?:\/(.*))?$/.exec(path);
  return Boolean(match && (match[1] === undefined || match[1].split("/").every(segment => segment && segment !== "." && segment !== ".." && !segment.includes("\0") && !segment.includes("\\"))));
}

export async function validateEnvironmentWorkspace(root: string, brainId: string): Promise<void> {
  if (!await exists(join(root, ENVIRONMENT_WORKSPACE_PATH))) {
    const base = join(root, "environments"), info = await lstat(base).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (!info) return;
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Environment subtree exists without a valid ownership database.");
    const directory = await opendir(base); for await (const _entry of directory) throw new Error("Environment files exist without their ownership database.");
    return;
  }
  const store = await EnvironmentStore.open(root, brainId, true);
  try {
    if (store.db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Environment database failed integrity verification.");
    const known = new Set<string>();
    for (const environment of store.environments()) {
      validateEnvironmentRecord(environment, brainId); known.add(environment.id);
      if (["creating", "loading", "busy", "stopping"].includes(environment.state) || environment.orphanedProcessRisk)
        throw new Error("Environment portability requires a quiesced lifecycle boundary; active or unreconciled work was not copied.");
      const directory = store.instanceDirectory(environment.id), info = await lstat(directory);
      if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Environment record lost its private instance directory.");
      for await (const file of files(directory)) {
        if (file.relativePath !== "sandbox.sb" && !file.relativePath.startsWith("workspace/")) throw new Error("Environment instance contains an unsupported executable/configuration payload.");
        if (file.relativePath === "sandbox.sb" && environment.kind !== "sandbox") throw new Error("A VM environment contains an unrelated sandbox profile.");
      }
      const operations = store.operations(environment.id); for (const operation of operations) validateEnvironmentOperation(operation, brainId);
      if (environment.lastOperationId && !operations.some(operation => operation.id === environment.lastOperationId)) throw new Error("Environment record lost its last operation owner.");
    }
    const directory = await opendir(join(root, "environments", "instances"));
    for await (const entry of directory) if (!entry.isDirectory() || entry.isSymbolicLink() || !/^[a-f0-9]{32}$/.test(entry.name) || !known.has(entry.name)) throw new Error("Environment instance storage has an unowned identity.");
    if (store.db.prepare("SELECT 1 FROM operations o LEFT JOIN environments e ON e.id=o.environment_id WHERE e.id IS NULL LIMIT 1").get()) throw new Error("Environment workspace contains an orphan operation.");
    if (store.db.prepare("SELECT 1 FROM requests r LEFT JOIN operations o ON o.id=r.operation_id WHERE o.id IS NULL LIMIT 1").get()) throw new Error("Environment workspace contains an orphan request.");
    for (const row of store.db.prepare("SELECT id,arguments_sha256,operation_id,result FROM requests").iterate()) {
      if (typeof row.id !== "string" || typeof row.arguments_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(row.arguments_sha256) ||
          !store.getOperation(String(row.operation_id))) throw new Error("Environment request ledger lost its exact operation.");
      store.request(String(row.id), String(row.arguments_sha256));
    }
  } finally { store.close(); }
}

/** Copy/import adoption retains history and workspace bytes, but never a live
 * process, external VM identity, request replay, or prior authority. */
export async function rekeyEnvironmentWorkspace(root: string, sourceBrainId: string, targetBrainId: string): Promise<void> {
  if (!await exists(join(root, ENVIRONMENT_WORKSPACE_PATH))) return;
  try { await validateEnvironmentWorkspace(root, sourceBrainId); }
  catch (sourceError) {
    if (sourceBrainId === targetBrainId) throw sourceError;
    try {
      await validateEnvironmentWorkspace(root, targetBrainId);
      const target = await EnvironmentStore.open(root, targetBrainId, true);
      try { for (const environment of target.environments()) await rm(join(root, "environments", "instances", environment.id, "sandbox.sb"), { force: true }); }
      finally { target.close(); }
      await validateEnvironmentWorkspace(root, targetBrainId); return;
    } catch { throw sourceError; }
  }
  const store = await EnvironmentStore.open(root, sourceBrainId), current = [...store.environments()], operations = current.flatMap(row => store.operations(row.id));
  const now = new Date().toISOString();
  try {
    store.transaction(() => {
      store.db.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetBrainId);
      store.db.exec("DELETE FROM requests; DELETE FROM operations; DELETE FROM environments;");
      for (const environment of current) {
        const row: ManagedEnvironmentRecord = { ...environment, brainId: targetBrainId, state: "dormant", revision: environment.revision + 1,
          updatedAt: now, externalId: undefined, orphanedProcessRisk: undefined, warning: "Copied/imported environment retained without a live process, adapter identity, or permission. Explicit review and Load are required.",
          inheritedFrom: environment.inheritedFrom ?? { brainId: sourceBrainId, environmentId: environment.id } };
        const payload = environmentCanonical(row); store.db.prepare("INSERT INTO environments VALUES(?,?,?,?,?)").run(row.id, row.kind, row.state, payload, environmentHash(payload));
      }
      for (const operation of operations) {
        const row: EnvironmentOperationReceipt = { ...operation, brainId: targetBrainId,
          ...(terminal.has(operation.state) ? {} : { state: "interrupted", updatedAt: now,
            error: "The source operation was not transferred as live execution." }) };
        const payload = environmentCanonical(row); store.db.prepare("INSERT INTO operations VALUES(?,?,?,?,?,?)")
          .run(row.id, row.environmentId, row.requestId, row.state, payload, environmentHash(payload));
      }
    });
  } finally { store.close(); }
  for (const environment of current) await rm(join(root, "environments", "instances", environment.id, "sandbox.sb"), { force: true });
  await validateEnvironmentWorkspace(root, targetBrainId);
}

/** Merge retains opaque files and receipts under a deterministic dormant local
 * identity. It never imports the parent's external VM or outstanding request. */
export async function mergeEnvironmentWorkspace(
  sourceRoot: string,
  targetRoot: string,
  sourceBrainId: string,
  targetBrainId: string,
  parentId: string,
  reserveWrite?: (bytes: number) => Promise<void>
): Promise<Map<string, string>> {
  const aliases = new Map<string, string>();
  if (!await exists(join(sourceRoot, ENVIRONMENT_WORKSPACE_PATH))) return aliases;
  await validateEnvironmentWorkspace(sourceRoot, sourceBrainId);
  const source = await EnvironmentStore.open(sourceRoot, sourceBrainId, true), target = await EnvironmentStore.open(targetRoot, targetBrainId);
  const now = new Date().toISOString();
  try {
    for (const environment of source.environments()) {
      const localId = environmentHash(environmentCanonical({ parentId, environmentId: environment.id, kind: environment.kind })).slice(0, 32);
      aliases.set(environment.id, localId);
      const existing = target.getEnvironment(localId);
      if (existing) {
        if (existing.inheritedFrom?.brainId !== parentId || existing.inheritedFrom.environmentId !== environment.id) throw new Error("Merged environment identity collided with another parent.");
        continue;
      }
      const operations = source.operations(environment.id), operationAliases = new Map(operations.map(operation => [operation.id,
        environmentHash(environmentCanonical({ parentId, operationId: operation.id, environmentId: environment.id })).slice(0, 32)]));
      const row: ManagedEnvironmentRecord = { ...environment, id: localId, brainId: targetBrainId, state: "dormant", revision: environment.revision + 1,
        updatedAt: now, externalId: undefined, orphanedProcessRisk: undefined, lastOperationId: environment.lastOperationId ? operationAliases.get(environment.lastOperationId) : undefined,
        warning: "Merged environment is dormant. No process, external VM identity, live request, credential-store entry, or permission transferred. Retained workspace/output bytes are unsanitized and may themselves contain sensitive data.",
        inheritedFrom: { brainId: parentId, environmentId: environment.id } };
      const rewritten = operations.map(operation => ({ ...operation, id: operationAliases.get(operation.id)!, brainId: targetBrainId,
        environmentId: localId, ...(terminal.has(operation.state) ? {} : { state: "interrupted" as const, updatedAt: now,
          error: "The parent operation was retained as history and not transferred as execution." }) }));
      const directory = target.instanceDirectory(localId), stage = join(targetRoot, `.environment-merge-${randomUUID()}`);
      await rm(stage, { recursive: true, force: true }); await mkdir(join(stage, "workspace"), { recursive: true, mode: 0o700 });
      try {
        for await (const file of files(source.instanceDirectory(environment.id))) {
          if (file.relativePath === "sandbox.sb") continue;
          const destination = resolve(stage, ...file.relativePath.split("/")), fromStage = relative(stage, destination);
          if (!fromStage || fromStage === ".." || fromStage.startsWith(`..${sep}`) || isAbsolute(fromStage)) throw new Error("Merged environment file escaped its target owner.");
          await reserveWrite?.(file.bytes); await mkdir(dirname(destination), { recursive: true, mode: 0o700 });
          await copyMutableFileIsolated(file.path, destination, { expectedSha256: file.sha256 });
        }
        if (await exists(directory)) await rm(directory, { recursive: true, force: true }); // Exact deterministic orphan from an interrupted pre-publication attempt.
        await rename(stage, directory);
        try { target.transaction(() => { target.putEnvironment(row); for (const operation of rewritten) target.putOperation(operation); }); }
        catch (error) { await rm(directory, { recursive: true, force: true }); throw error; }
      } finally { await rm(stage, { recursive: true, force: true }); }
    }
  } finally { source.close(); target.close(); }
  await validateEnvironmentWorkspace(targetRoot, targetBrainId); return aliases;
}
