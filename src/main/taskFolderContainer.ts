import { createHash } from "node:crypto";
import { constants, type BigIntStats } from "node:fs";
import { lstat, open, opendir, realpath, rm, type FileHandle } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { sqliteFilename } from "./sqliteFilename";
import { ensureDiskReserve } from "./streamingZip";
import type { ChatTaskAttachmentHandle, ChatTaskAttachmentReadResult, TaskFolderEntry, TaskFolderTraversal } from "../shared/taskAttachments";

export const TASK_FOLDER_MIME = "application/vnd.omni.task-folder+sqlite3";
const FORMAT = "omni-task-folder-1";
const CHUNK = 64 * 1024;
const DISK_WINDOW = 8 * 1024 * 1024;
const SCHEMA = [
  "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL) STRICT",
  "CREATE TABLE entries(ordinal INTEGER UNIQUE NOT NULL,path TEXT PRIMARY KEY,kind TEXT NOT NULL,mime TEXT NOT NULL,bytes INTEGER NOT NULL,sha256 TEXT NOT NULL) STRICT",
  "CREATE TABLE chunks(entry_ordinal INTEGER NOT NULL,position INTEGER NOT NULL,content BLOB NOT NULL,PRIMARY KEY(entry_ordinal,position),FOREIGN KEY(entry_ordinal) REFERENCES entries(ordinal)) STRICT"
];

function safeCount(value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error("Folder exceeds the runtime's exact addressable range.");
  return value;
}

function identity(info: BigIntStats): string {
  if (info.isSymbolicLink() || info.ino <= 0n) throw new Error("Folder source identity is unsafe or unavailable.");
  return [info.dev, info.ino, info.size, info.mtimeNs, info.ctimeNs].join(":");
}

function sameDirectory(before: BigIntStats, after: BigIntStats): boolean {
  return before.isDirectory() && after.isDirectory() && identity(before) === identity(after);
}

export function validatedFolderEntryPath(path: unknown): string {
  // This is a logical container key, never a filesystem pathname. Backslashes
  // are legitimate POSIX filename characters and never become host traversal.
  if (typeof path !== "string" || !path || path.includes("\0") ||
      path.split("/").some(part => !part || part === "." || part === "..")) {
    throw new Error("Choose an exact relative file entry from the saved folder manifest.");
  }
  return path;
}

interface FolderRow { ordinal: number; path: string; kind: "file" | "directory"; mime: string; bytes: number; sha256: string }
function entry(row: FolderRow): TaskFolderEntry {
  validatedFolderEntryPath(row.path);
  if (!Number.isSafeInteger(row.ordinal) || row.ordinal < 0 || !["file", "directory"].includes(row.kind) ||
      !Number.isSafeInteger(row.bytes) || row.bytes < 0 || typeof row.mime !== "string" ||
      (row.kind === "file" ? !/^[a-f0-9]{64}$/.test(row.sha256) : row.bytes !== 0 || row.sha256 !== "")) {
    throw new Error("Saved folder entry metadata is invalid.");
  }
  return { path: row.path, kind: row.kind, mimeType: row.mime, bytes: row.bytes,
    ...(row.kind === "file" ? { contentSha256: row.sha256 } : {}) };
}

/** Main-owned, streamed snapshot. Neither source traversal nor copy invokes learning. */
export async function buildTaskFolderContainer(source: string, destination: string,
  mimeForPath: (path: string) => string, signal?: AbortSignal,
  onProgress?: (progress: Readonly<TaskFolderTraversal>) => void): Promise<TaskFolderTraversal> {
  signal?.throwIfAborted();
  const selected = await lstat(resolve(source), { bigint: true });
  if (!selected.isDirectory() || selected.isSymbolicLink()) throw new Error("Selected folder must be a real directory, not a symlink.");
  const root = await realpath(resolve(source)), rootInfo = await lstat(root, { bigint: true });
  if (!sameDirectory(selected, rootInfo)) throw new Error("Selected folder identity changed before staging.");
  const destinationWithin = relative(root, resolve(destination));
  if (!destinationWithin || !isAbsolute(destinationWithin) && destinationWithin !== ".." && !destinationWithin.startsWith(`..${sep}`)) {
    throw new Error("A folder cannot contain its own attachment staging workspace.");
  }
  const initial = await open(destination, "wx", 0o600); await initial.close();
  let database: DatabaseSync | undefined;
  let nextDiskCheck = 0;
  const report: TaskFolderTraversal = { discoveredFiles: 0, processedFiles: 0, discoveredDirectories: 1,
    processedDirectories: 0, sourceBytes: 0, copiedBytes: 0, rejectedEntries: 0, complete: false,
    manifestSha256: "" };
  try {
    await ensureDiskReserve(dirname(destination), DISK_WINDOW * 2 + CHUNK);
    database = new DatabaseSync(sqliteFilename(destination), { allowExtension: false });
    database.exec("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA temp_store=FILE; PRAGMA cache_size=-2048; PRAGMA foreign_keys=ON; PRAGMA trusted_schema=OFF;");
    for (const sql of SCHEMA) database.exec(sql);
    const add = database.prepare("INSERT INTO entries VALUES(?,?,?,?,?,?)"),
      chunk = database.prepare("INSERT INTO chunks VALUES(?,?,?)"),
      seal = database.prepare("UPDATE entries SET sha256=? WHERE path=?");
    let ordinal = 0, metadataBytes = 0;
    const ancestors: Array<{ path: string; info: BigIntStats }> = [];
    const checkAncestors = async (): Promise<void> => {
      signal?.throwIfAborted();
      for (const item of ancestors) {
        const now = await lstat(item.path, { bigint: true });
        if (!now.isDirectory() || now.isSymbolicLink() || now.dev !== item.info.dev || now.ino !== item.info.ino) {
          throw new Error("Selected folder ancestry changed during staging.");
        }
      }
    };
    const checkDisk = async (): Promise<void> => {
      const estimated = safeCount(report.copiedBytes + metadataBytes);
      if (estimated >= nextDiskCheck) {
        await ensureDiskReserve(dirname(destination), DISK_WINDOW * 2 + CHUNK);
        nextDiskCheck = estimated + DISK_WINDOW;
      }
    };
    const walk = async (path: string, parts: string[]): Promise<void> => {
      signal?.throwIfAborted();
      const before = await lstat(path, { bigint: true });
      if (!before.isDirectory() || before.isSymbolicLink()) throw new Error("Selected folder contains a symlink or changed directory.");
      ancestors.push({ path, info: before });
      try {
        await checkAncestors();
        const directory = await opendir(path, { bufferSize: 32 });
        for await (const child of directory) {
          signal?.throwIfAborted(); await checkAncestors();
          const childPath = join(path, child.name), childParts = [...parts, child.name], key = childParts.join("/");
          validatedFolderEntryPath(key);
          metadataBytes = safeCount(metadataBytes + Buffer.byteLength(key) * 4 + 8192);
          const observed = await lstat(childPath, { bigint: true });
          if (observed.isSymbolicLink() || !observed.isFile() && !observed.isDirectory()) {
            throw new Error("Selected folder contains a symlink or special file; no partial folder was attached.");
          }
          safeCount(ordinal);
          if (observed.isDirectory()) {
            report.discoveredDirectories = safeCount(report.discoveredDirectories + 1);
            await checkDisk();
            add.run(ordinal++, key, "directory", "inode/directory", 0, "");
            await walk(childPath, childParts);
            continue;
          }
          report.discoveredFiles = safeCount(report.discoveredFiles + 1);
          const size = safeCount(Number(observed.size)), expected = identity(observed);
          report.sourceBytes = safeCount(report.sourceBytes + size);
          const file = await open(childPath, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
          const assertFile = async (): Promise<void> => {
            await checkAncestors();
            const [opened, current] = await Promise.all([file.stat({ bigint: true }), lstat(childPath, { bigint: true })]);
            if (!opened.isFile() || !current.isFile() || current.isSymbolicLink() || identity(opened) !== expected || identity(current) !== expected) {
              throw new Error("Selected folder file changed during staging.");
            }
          };
          try {
            await assertFile(); await checkDisk();
            database!.exec("BEGIN IMMEDIATE");
            try {
              const entryOrdinal = ordinal++;
              add.run(entryOrdinal, key, "file", mimeForPath(key), size, "");
              const digest = createHash("sha256"), buffer = Buffer.alloc(Math.min(CHUNK, Math.max(1, size)));
              let position = 0;
              while (position < size) {
                signal?.throwIfAborted(); await checkDisk();
                const wanted = Math.min(buffer.length, size - position);
                let received = 0;
                while (received < wanted) {
                  signal?.throwIfAborted();
                  const read = await file.read(buffer, received, wanted - received, position + received);
                  if (!read.bytesRead) throw new Error("Selected folder file ended before its complete byte traversal.");
                  received += read.bytesRead;
                }
                await assertFile();
                const bytes = buffer.subarray(0, received);
                digest.update(bytes); chunk.run(entryOrdinal, position, bytes); position += received;
                report.copiedBytes = safeCount(report.copiedBytes + received);
                try { onProgress?.({ ...report }); } catch { /* Diagnostics cannot undo owned copied bytes. */ }
              }
              await assertFile();
              seal.run(digest.digest("hex"), key);
              database!.exec("COMMIT");
              report.processedFiles = safeCount(report.processedFiles + 1);
              try { onProgress?.({ ...report }); } catch { /* Progress is not permission or commit authority. */ }
            } catch (error) { database!.exec("ROLLBACK"); throw error; }
          } finally { await file.close(); }
        }
        if (!sameDirectory(before, await lstat(path, { bigint: true }))) throw new Error("Selected directory changed during complete traversal.");
        report.processedDirectories = safeCount(report.processedDirectories + 1);
        try { onProgress?.({ ...report }); } catch { /* Keep zero-byte directory progress truthful. */ }
      } finally { ancestors.pop(); }
    };
    await walk(root, []); signal?.throwIfAborted();
    if (!sameDirectory(rootInfo, await lstat(root, { bigint: true }))) throw new Error("Selected folder changed before its snapshot completed.");
    const manifest = createHash("sha256");
    let manifested = 0;
    for (const row of database.prepare("SELECT * FROM entries ORDER BY ordinal").iterate() as Iterable<FolderRow>) {
      if (manifested++ % 128 === 0) {
        await new Promise<void>(proceed => setImmediate(proceed));
        signal?.throwIfAborted();
      }
      manifest.update(JSON.stringify(entry(row)) + "\n");
    }
    report.manifestSha256 = manifest.digest("hex");
    report.complete = report.discoveredFiles === report.processedFiles && report.discoveredDirectories === report.processedDirectories && report.sourceBytes === report.copiedBytes;
    if (!report.complete) throw new Error("Folder traversal did not visit every discovered source record.");
    const meta = database.prepare("INSERT INTO meta VALUES(?,?)");
    meta.run("format", FORMAT); meta.run("name", basename(root)); meta.run("traversal", JSON.stringify(report));
    if (database.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Saved folder snapshot failed SQLite integrity validation.");
    database.close(); database = undefined;
    const output = await open(destination, "r+"); try { await output.sync(); } finally { await output.close(); }
    try { onProgress?.({ ...report }); } catch { /* The completed traversal remains authoritative. */ }
    return report;
  } catch (error) {
    database?.close(); database = undefined;
    await rm(destination, { force: true }).catch(() => undefined);
    throw error;
  } finally { database?.close(); }
}

function folderDatabase(path: string): { database: DatabaseSync; traversal: TaskFolderTraversal } {
  const database = new DatabaseSync(sqliteFilename(path), { readOnly: true, allowExtension: false });
  try {
    database.exec("PRAGMA query_only=ON; PRAGMA trusted_schema=OFF; PRAGMA cache_size=-2048; PRAGMA temp_store=FILE;");
    const schemas = database.prepare("SELECT CASE WHEN length(name)<=64 THEN name END AS name,CASE WHEN length(sql)<=4096 THEN sql END AS sql FROM sqlite_schema WHERE type='table' ORDER BY name LIMIT 4").all() as Array<{ name: string; sql: string }>;
    if (schemas.length !== 3 || schemas.some(row => !SCHEMA.includes(row.sql)) ||
        database.prepare("SELECT COUNT(*) AS count FROM sqlite_schema WHERE type IN ('trigger','view')").get()?.count !== 0 ||
        database.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== FORMAT) {
      throw new Error("Saved folder container does not have the known immutable schema.");
    }
    const traversalBytes = database.prepare("SELECT length(CAST(value AS BLOB)) AS bytes FROM meta WHERE key='traversal'").get()?.bytes;
    if (typeof traversalBytes !== "number" || traversalBytes < 1 || traversalBytes > 16 * 1024) throw new Error("Saved folder traversal metadata exceeds its bounded format.");
    const traversal = JSON.parse(String(database.prepare("SELECT value FROM meta WHERE key='traversal'").get()?.value)) as TaskFolderTraversal;
    for (const key of ["discoveredFiles", "processedFiles", "discoveredDirectories", "processedDirectories", "sourceBytes", "copiedBytes", "rejectedEntries"] as const) safeCount(traversal[key]);
    if (traversal.complete !== true || traversal.rejectedEntries !== 0 || traversal.discoveredFiles !== traversal.processedFiles ||
        traversal.discoveredDirectories !== traversal.processedDirectories || traversal.sourceBytes !== traversal.copiedBytes ||
        !/^[a-f0-9]{64}$/.test(traversal.manifestSha256)) throw new Error("Saved folder traversal receipt is incomplete.");
    return { database, traversal };
  } catch (error) { database.close(); throw error; }
}

export async function readTaskFolderContainer(path: string, verified: FileHandle, offset: number, maxBytes: number,
  entryPath?: string, signal?: AbortSignal): Promise<{
    traversal: TaskFolderTraversal; entries?: TaskFolderEntry[]; nextOffset?: number;
    member?: { entryPath: string; sha256: string; bytes: number; data: Buffer; mimeType: string };
  }> {
  signal?.throwIfAborted();
  const expected = identity(await verified.stat({ bigint: true }));
  const check = async (): Promise<void> => {
    signal?.throwIfAborted();
    if (identity(await verified.stat({ bigint: true })) !== expected || identity(await lstat(path, { bigint: true })) !== expected) {
      throw new Error("Saved folder container changed during its verified page read.");
    }
  };
  await check();
  const { database, traversal } = folderDatabase(path);
  try {
    await check();
    if (entryPath === undefined) {
      const entries: TaskFolderEntry[] = [];
      let bytes = Buffer.byteLength(JSON.stringify({ format: FORMAT, entries: [], traversal, offset })), nextOffset: number | undefined;
      if (bytes > maxBytes) throw new Error("Increase maxBytes to read the full folder traversal receipt.");
      for (const row of database.prepare("SELECT * FROM entries WHERE ordinal>=? ORDER BY ordinal LIMIT 128").iterate(offset) as Iterable<FolderRow>) {
        signal?.throwIfAborted();
        const item = entry(row), size = Buffer.byteLength(JSON.stringify(item)) + 1;
        if (bytes + size > maxBytes && !entries.length) throw new Error("Increase maxBytes to read this full manifest entry; it was not truncated.");
        if (bytes + size > maxBytes) { nextOffset = row.ordinal; break; }
        entries.push(item); bytes += size; nextOffset = row.ordinal + 1;
      }
      const total = traversal.processedFiles + traversal.processedDirectories - 1;
      if (nextOffset !== undefined && nextOffset >= total) nextOffset = undefined;
      await check(); return { entries, traversal, ...(nextOffset === undefined ? {} : { nextOffset }) };
    }
    const key = validatedFolderEntryPath(entryPath);
    const row = database.prepare("SELECT * FROM entries WHERE path=?").get(key) as FolderRow | undefined;
    if (!row || entry(row).kind !== "file") throw new Error("Saved folder member is not an exact retained file.");
    if (offset > row.bytes) throw new Error("Saved member offset is outside its exact bytes.");
    const data = Buffer.alloc(Math.min(maxBytes, row.bytes - offset));
    const start = Math.floor(offset / CHUNK) * CHUNK, end = offset + data.length;
    let copied = 0;
    for (const part of database.prepare("SELECT position,content FROM chunks WHERE entry_ordinal=? AND position>=? AND position<? ORDER BY position").iterate(row.ordinal, start, end) as Iterable<{ position: number; content: Uint8Array }>) {
      signal?.throwIfAborted();
      const begin = Math.max(0, offset - part.position), length = Math.min(part.content.byteLength - begin, data.length - copied);
      if (!Number.isSafeInteger(part.position) || part.position !== start + Math.floor((offset - start + copied) / CHUNK) * CHUNK ||
          !(part.content instanceof Uint8Array) || part.content.byteLength !== Math.min(CHUNK, row.bytes - part.position) || length < 0) {
        throw new Error("Saved folder content chunks have an invalid byte boundary.");
      }
      data.set(part.content.subarray(begin, begin + length), copied); copied += length;
    }
    if (copied !== data.length) throw new Error("Saved folder member is missing committed content bytes.");
    await check();
    return { traversal, member: { entryPath: key, sha256: row.sha256, bytes: row.bytes, data, mimeType: row.mime } };
  } finally { database.close(); }
}

/** Only the already-verified saved container is queried, never originalPath. */
export async function readTaskFolderAttachment(attachment: ChatTaskAttachmentHandle, path: string, verified: FileHandle,
  offset: number, maxBytes: number, entryPath?: string, signal?: AbortSignal): Promise<ChatTaskAttachmentReadResult> {
  const page = await readTaskFolderContainer(path, verified, offset, maxBytes, entryPath, signal);
  if (!page.member) {
    const content = JSON.stringify({ format: FORMAT, entries: page.entries, traversal: page.traversal, offset });
    return { format: "omni-chat-task-attachment-read", version: 1, attachment, offset,
      returnedBytes: Buffer.byteLength(content), totalBytes: attachment.bytes, encoding: "utf8", content,
      folder: { cursorUnit: "entries", traversal: page.traversal },
      ...(page.nextOffset === undefined ? {} : { nextOffset: page.nextOffset }) };
  }
  const member = page.member;
  let data = member.data, encoding: "utf8" | "base64" = "base64", content = data.toString("base64");
  if (member.mimeType.startsWith("text/") || ["application/json", "application/x-ndjson", "application/xml", "application/javascript"].includes(member.mimeType)) {
    for (let trim = 0; trim <= Math.min(3, Math.max(0, data.length - 1)); trim++) {
      try { const candidate = data.subarray(0, data.length - trim); content = new TextDecoder("utf-8", { fatal: true }).decode(candidate); data = candidate; encoding = "utf8"; break; }
      catch { /* Preserve every undecodable byte through the exact binary page. */ }
    }
  }
  return { format: "omni-chat-task-attachment-read", version: 1, attachment, offset, returnedBytes: data.length,
    totalBytes: member.bytes, encoding, content,
    folder: { cursorUnit: "bytes", traversal: page.traversal, entryPath: member.entryPath, entrySha256: member.sha256, entryMimeType: member.mimeType },
    ...(offset + data.length < member.bytes ? { nextOffset: offset + data.length } : {}) };
}
