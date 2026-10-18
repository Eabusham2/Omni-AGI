import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, readFile, unlink } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

const FORMAT = "omni-inherited-working-publication";
const MARKER = ".connected-working-publication.json";
const STAGED_ENGINE = /^\.engine-[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}\.restore$/;
const SHA256 = /^[a-f0-9]{64}$/;
const EXECUTION = /^[a-f0-9]{64}$/;
const BACKUP_NAME = /^[a-f0-9]{32}\.sqlite3$/;

interface WorkingPublicationMarker {
  format: typeof FORMAT;
  version: 1;
  brainId: string;
  executionId: string;
  beforeRootSha256: string;
  backup: string;
  sha256: string;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

async function fileSha256(path: string): Promise<string> {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest("hex");
}

async function regularFile(path: string, label: string): Promise<void> {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) {
    throw new Error(`${label} must be a regular file.`);
  }
}

/**
 * Clear a stale inherited-working publication only from an unpublished private
 * recovery stage. The live engine and selected snapshot are never arguments to
 * this helper. Invalid state is preserved for diagnosis and fails closed.
 */
export async function clearStagedConnectedWorkingPublication(
  stagedEngineDirectory: string
): Promise<{ cleared: boolean; executionId?: string }> {
  const engine = resolve(stagedEngineDirectory);
  if (!STAGED_ENGINE.test(basename(engine))) {
    throw new Error(
      "Inherited working publication cleanup requires a private recovery staging engine."
    );
  }
  const engineInfo = await lstat(engine);
  if (!engineInfo.isDirectory() || engineInfo.isSymbolicLink()) {
    throw new Error("Recovery staging engine is unsafe.");
  }

  const markerPath = join(engine, MARKER);
  let markerInfo;
  try {
    markerInfo = await lstat(markerPath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return { cleared: false };
    }
    throw error;
  }
  if (
    !markerInfo.isFile() ||
    markerInfo.isSymbolicLink() ||
    markerInfo.size < 2 ||
    markerInfo.size > 64 * 1024
  ) {
    throw new Error("Inherited working publication marker is unsafe.");
  }

  let markerValue: unknown;
  let metadataValue: unknown;
  try {
    await regularFile(
      join(engine, "brain.json"),
      "Recovery staging engine metadata"
    );
    markerValue = JSON.parse(await readFile(markerPath, "utf8"));
    metadataValue = JSON.parse(await readFile(join(engine, "brain.json"), "utf8"));
  } catch {
    throw new Error("Inherited working publication metadata is invalid.");
  }
  const value = record(markerValue);
  const metadata = record(metadataValue);
  const keys = value ? Object.keys(value).sort() : [];
  const expectedKeys = [
    "backup",
    "beforeRootSha256",
    "brainId",
    "executionId",
    "format",
    "sha256",
    "version"
  ];
  if (
    !value ||
    !metadata ||
    keys.length !== expectedKeys.length ||
    keys.some((key, index) => key !== expectedKeys[index]) ||
    value.format !== FORMAT ||
    value.version !== 1 ||
    typeof metadata.brain_id !== "string" ||
    value.brainId !== metadata.brain_id ||
    typeof value.executionId !== "string" ||
    !EXECUTION.test(value.executionId) ||
    typeof value.beforeRootSha256 !== "string" ||
    !SHA256.test(value.beforeRootSha256) ||
    typeof value.sha256 !== "string" ||
    !SHA256.test(value.sha256)
  ) {
    throw new Error("Inherited working publication ownership is invalid.");
  }
  const expectedBackup = `connected-executions/${value.executionId}/working-before/`;
  if (
    typeof value.backup !== "string" ||
    !value.backup.startsWith(expectedBackup) ||
    !BACKUP_NAME.test(value.backup.slice(expectedBackup.length))
  ) {
    throw new Error("Inherited working publication backup path is invalid.");
  }

  let current = engine;
  for (const segment of value.backup.split("/")) {
    current = join(current, segment);
    const info = await lstat(current);
    if (info.isSymbolicLink()) {
      throw new Error("Inherited working publication backup cannot follow links.");
    }
  }
  const backupPath = current;
  await regularFile(backupPath, "Inherited working publication backup");
  if ((await fileSha256(backupPath)) !== value.sha256) {
    throw new Error("Inherited working publication backup checksum failed.");
  }

  // Marker first matches native recovery semantics. A crash after this point
  // can leave only an unreachable private-stage backup; the whole stage is
  // removed on restore failure and no live/source/snapshot path was touched.
  await unlink(markerPath);
  await unlink(backupPath);
  return { cleared: true, executionId: value.executionId };
}
