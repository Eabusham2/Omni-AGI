import { createHash, randomUUID } from "node:crypto";
import { constants as fsConstants, createReadStream } from "node:fs";
import {
  access,
  copyFile,
  lstat,
  mkdir,
  open,
  rename,
  rm,
  stat,
  writeFile
} from "node:fs/promises";
import { dirname } from "node:path";
import { DatabaseSync } from "node:sqlite";

export interface IsolatedFileCopyOptions {
  expectedSha256?: string;
  mode?: number;
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function fileSha256(path: string): Promise<string> {
  const digest = createHash("sha256");
  await new Promise<void>((resolveHash, rejectHash) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => digest.update(chunk));
    stream.once("error", rejectHash);
    stream.once("end", resolveHash);
  });
  return digest.digest("hex");
}

async function syncFile(path: string): Promise<void> {
  const handle = await open(path, "r");
  try {
    await handle.sync();
  } finally {
    await handle.close();
  }
}

async function promotePrivateFile(temporary: string, destination: string): Promise<void> {
  try {
    await rename(temporary, destination);
    return;
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (!["EEXIST", "EPERM", "EACCES"].includes(code ?? "")) throw error;
  }

  const backup = `${destination}.${randomUUID()}.private-previous`;
  let movedExisting = false;
  try {
    if (await pathExists(destination)) {
      await rename(destination, backup);
      movedExisting = true;
    }
    await rename(temporary, destination);
    if (movedExisting) {
      await rm(backup, { force: true });
      movedExisting = false;
    }
  } catch (error) {
    if (movedExisting && await pathExists(backup)) {
      await rename(backup, destination).catch(() => undefined);
    }
    throw error;
  } finally {
    await rm(backup, { force: true }).catch(() => undefined);
  }
}

async function assertPrivateMaterialization(
  source: string,
  destination: string,
  expectedBytes: number
): Promise<void> {
  const [sourceInfo, destinationInfo] = await Promise.all([
    stat(source),
    stat(destination)
  ]);
  if (!destinationInfo.isFile() || destinationInfo.size !== expectedBytes) {
    throw new Error("Private mutable-state materialization is incomplete.");
  }
  if (
    sourceInfo.dev === destinationInfo.dev &&
    sourceInfo.ino === destinationInfo.ino
  ) {
    throw new Error("Private mutable-state materialization reused the source inode.");
  }
}

/**
 * Copy a logically mutable file without ever sharing its inode.
 *
 * COPYFILE_FICLONE preserves block-level copy-on-write savings on filesystems
 * that support reflinks, while still producing a distinct inode. The fallback
 * is a regular byte copy. Promotion replaces the destination path atomically.
 */
export async function copyMutableFileIsolated(
  source: string,
  destination: string,
  options: IsolatedFileCopyOptions = {}
): Promise<number> {
  const before = await lstat(source);
  if (!before.isFile() || before.isSymbolicLink()) {
    throw new Error("Mutable copy source is not a safe regular file.");
  }
  await mkdir(dirname(destination), { recursive: true });
  const temporary = `${destination}.${randomUUID()}.private-next`;
  try {
    await copyFile(source, temporary, fsConstants.COPYFILE_FICLONE);
    if (options.mode !== undefined) {
      const handle = await open(temporary, "r+");
      try {
        await handle.chmod(options.mode);
      } finally {
        await handle.close();
      }
    }
    const [after, copied] = await Promise.all([lstat(source), lstat(temporary)]);
    if (
      !after.isFile() ||
      after.isSymbolicLink() ||
      after.dev !== before.dev ||
      after.ino !== before.ino ||
      after.size !== before.size ||
      after.mtimeMs !== before.mtimeMs ||
      copied.size !== before.size
    ) {
      throw new Error("Mutable copy source changed during materialization.");
    }
    if (
      options.expectedSha256 &&
      await fileSha256(temporary) !== options.expectedSha256
    ) {
      throw new Error("Private mutable-state checksum failed.");
    }
    await syncFile(temporary);
    await promotePrivateFile(temporary, destination);
    await assertPrivateMaterialization(source, destination, before.size);
    return before.size;
  } finally {
    await rm(temporary, { force: true }).catch(() => undefined);
  }
}

/** Write and atomically promote bytes to a path that must remain independently mutable. */
export async function writeMutableFileIsolated(
  destination: string,
  contents: Uint8Array,
  mode = 0o600
): Promise<number> {
  await mkdir(dirname(destination), { recursive: true });
  const temporary = `${destination}.${randomUUID()}.private-next`;
  try {
    await writeFile(temporary, contents, { flag: "wx", mode });
    await syncFile(temporary);
    await promotePrivateFile(temporary, destination);
    const materialized = await lstat(destination);
    if (!materialized.isFile() || materialized.isSymbolicLink()) {
      throw new Error("Private mutable-state write did not produce a regular file.");
    }
    return materialized.size;
  } finally {
    await rm(temporary, { force: true }).catch(() => undefined);
  }
}

/**
 * Snapshot a live SQLite database into a private inode.
 *
 * VACUUM INTO reads one consistent transaction, including committed WAL
 * pages. WAL/SHM/journal sidecars are deliberately never copied to the new
 * identity.
 */
export async function snapshotMutableSqliteIsolated(
  source: string,
  destination: string
): Promise<number> {
  const sourceInfo = await lstat(source);
  if (!sourceInfo.isFile() || sourceInfo.isSymbolicLink()) {
    throw new Error("SQLite snapshot source is not a safe regular file.");
  }
  await mkdir(dirname(destination), { recursive: true });
  const temporary = `${destination}.${randomUUID()}.sqlite-next`;
  try {
    const database = new DatabaseSync(source, { readOnly: true });
    try {
      // The source handle is read-only. Do not also enable query_only: SQLite
      // classifies VACUUM INTO as a write statement even though it writes only
      // the separate destination database.
      database.exec("PRAGMA busy_timeout=30000;");
      database.exec(`VACUUM INTO '${temporary.replaceAll("'", "''")}'`);
    } finally {
      database.close();
    }
    const snapshot = new DatabaseSync(temporary);
    try {
      const checked = snapshot.prepare("PRAGMA quick_check").get() as
        | { quick_check?: string }
        | undefined;
      if (checked?.quick_check !== "ok") {
        throw new Error("SQLite private snapshot failed its integrity check.");
      }
      snapshot.exec("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;");
    } finally {
      snapshot.close();
    }
    await syncFile(temporary);
    await promotePrivateFile(temporary, destination);
    await assertPrivateMaterialization(source, destination, (await stat(destination)).size);
    return (await stat(destination)).size;
  } finally {
    await Promise.all([
      rm(temporary, { force: true }),
      rm(`${temporary}-wal`, { force: true }),
      rm(`${temporary}-shm`, { force: true }),
      rm(`${temporary}-journal`, { force: true })
    ]).catch(() => undefined);
  }
}
