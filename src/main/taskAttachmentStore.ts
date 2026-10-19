import { createHash, randomUUID } from "node:crypto";
import { constants as fsConstants, createReadStream, createWriteStream } from "node:fs";
import { access, copyFile, link, lstat, mkdir, open, realpath, rm } from "node:fs/promises";
import { basename, dirname, extname, join, resolve } from "node:path";
import { Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { DatabaseSync } from "node:sqlite";
import type {
  ChatTaskAttachmentHandle,
  ChatTaskAttachmentKind,
  ChatTaskAttachmentReadResult,
  ChatTaskAttachmentResolved,
  ChatTaskAttachmentState,
  ChatTaskAttachmentSelection,
  TaskFolderTraversal
} from "../shared/taskAttachments";
import { TASK_ATTACHMENT_FORMAT, TASK_ATTACHMENT_VERSION } from "../shared/taskAttachments";
import type { ExperienceUploadKind } from "../shared/uploadSupport";
import { ensureDiskReserve } from "./streamingZip";
import { TaskAttachmentReadVerifier, taskAttachmentFileIdentity } from "./taskAttachmentReadVerifier";
import type { WebImageDownload } from "../shared/webImages";
import { checkedWebImageDownload } from "./webImageReferences";
import { sqliteFilename } from "./sqliteFilename";
import { buildTaskFolderContainer, readTaskFolderAttachment, TASK_FOLDER_MIME } from "./taskFolderContainer";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const MIME = /^[a-z0-9][a-z0-9.+-]{0,80}\/[a-z0-9][a-z0-9.+-]{0,80}$/;
export const TASK_ATTACHMENT_DATABASE = "registry.sqlite3";

export interface TaskAttachmentPrivateSource {
  kind: "local-file" | "local-folder" | "captured-audio" | "web-image";
  originalPath?: string;
  captureSessionId?: string;
  utteranceId?: string;
  sequence?: number;
  capturedAt?: string;
  selectedAt: string;
  inheritedFrom?: { brainId: string; attachmentId: string };
  webImage?: WebImageDownload;
}

export interface PersistedTaskAttachmentRow {
  id: string;
  brain_id: string;
  request_id: string;
  request_index: number;
  content_sha256: string;
  manifest_sha256: string;
  bytes: number;
  mime_type: string;
  kind: ChatTaskAttachmentKind;
  display_name: string;
  source_json: string;
  source_sha256: string;
  state: ChatTaskAttachmentState;
  turn_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface TaskAttachmentStagePath {
  path: string;
  displayName?: string;
  mimeType?: string;
  kind?: ChatTaskAttachmentKind;
  source?: TaskAttachmentPrivateSource;
}

export interface TaskAttachmentStageBytes {
  data: Uint8Array;
  displayName: string;
  mimeType: string;
  kind: ChatTaskAttachmentKind;
  source: TaskAttachmentPrivateSource;
}

function sha256(value: string | Buffer | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

export function canonicalTaskAttachmentValue(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalTaskAttachmentValue).join(",")}]`;
  return `{${Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([key, child]) => `${JSON.stringify(key)}:${canonicalTaskAttachmentValue(child)}`).join(",")}}`;
}

function requireId(value: string, label: string): string {
  if (!SAFE_ID.test(value)) throw new Error(`Invalid ${label}.`);
  return value;
}

function safeName(value: string): string {
  const name = value.replace(/\0/g, "").trim();
  if (!name || name.length > 512) throw new Error("Attachment display name is invalid.");
  return name;
}

function safeMimeType(value: string): string {
  const mimeType = value.trim().toLowerCase();
  if (!MIME.test(mimeType)) throw new Error("Attachment MIME type is invalid.");
  return mimeType;
}

function kindForSelection(selection: ExperienceUploadKind): ChatTaskAttachmentKind | undefined {
  return selection === "images" ? "image" : selection === "audio" ? "audio" : selection === "video" ? "video" : undefined;
}

const MIME_BY_EXTENSION: Record<string, string> = {
  ".txt": "text/plain", ".md": "text/markdown", ".json": "application/json", ".jsonl": "application/x-ndjson",
  ".csv": "text/csv", ".tsv": "text/tab-separated-values", ".html": "text/html", ".xml": "application/xml",
  ".js": "text/javascript", ".mjs": "text/javascript", ".ts": "text/typescript", ".tsx": "text/typescript",
  ".py": "text/x-python", ".rs": "text/x-rust", ".go": "text/x-go", ".java": "text/x-java-source",
  ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif",
  ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".opus": "audio/opus",
  ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime", ".mkv": "video/x-matroska",
  ".pdf": "application/pdf", ".zip": "application/zip", ".parquet": "application/vnd.apache.parquet"
};

function inferredMimeType(path: string): string {
  return MIME_BY_EXTENSION[extname(path).toLowerCase()] ?? "application/octet-stream";
}

function inferredKind(mimeType: string): ChatTaskAttachmentKind {
  return mimeType.startsWith("image/") ? "image" : mimeType.startsWith("audio/") ? "audio" : mimeType.startsWith("video/") ? "video" : "file";
}

export function validatedTaskAttachmentSource(value: TaskAttachmentPrivateSource): TaskAttachmentPrivateSource {
  if (!value || !["local-file", "local-folder", "captured-audio", "web-image"].includes(value.kind) ||
      typeof value.selectedAt !== "string" || !Number.isFinite(Date.parse(value.selectedAt)) ||
      (value.originalPath !== undefined && (typeof value.originalPath !== "string" || value.originalPath.includes("\0"))) ||
      (value.captureSessionId !== undefined && !SAFE_ID.test(value.captureSessionId)) ||
      (value.utteranceId !== undefined && !SAFE_ID.test(value.utteranceId)) ||
      (value.sequence !== undefined && (!Number.isSafeInteger(value.sequence) || value.sequence < 0)) ||
      (value.capturedAt !== undefined && !Number.isFinite(Date.parse(value.capturedAt))) ||
      (value.inheritedFrom !== undefined && (!SAFE_ID.test(value.inheritedFrom.brainId) ||
        !SAFE_ID.test(value.inheritedFrom.attachmentId)))) {
    throw new Error("Attachment source provenance is invalid.");
  }
  if (value.kind === "web-image") {
    checkedWebImageDownload(value.webImage);
    if (value.selectedAt !== value.webImage!.fetchedAt || value.originalPath !== undefined || value.captureSessionId !== undefined || value.utteranceId !== undefined) throw new Error("Web image source cannot impersonate a local file or device capture.");
  } else if (value.webImage !== undefined) throw new Error("Non-web attachment cannot carry web image download authority.");
  return structuredClone(value);
}

function publicManifest(row: Pick<PersistedTaskAttachmentRow, "id" | "brain_id" | "content_sha256" | "bytes" | "mime_type" | "kind" | "display_name" | "state" | "turn_id" | "created_at" | "updated_at">): Record<string, unknown> {
  return {
    format: TASK_ATTACHMENT_FORMAT,
    version: TASK_ATTACHMENT_VERSION,
    id: row.id,
    brainId: row.brain_id,
    displayName: row.display_name,
    kind: row.kind,
    mimeType: row.mime_type,
    contentSha256: row.content_sha256,
    bytes: row.bytes,
    state: row.state,
    turnId: row.turn_id,
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

export function taskAttachmentManifestSha256(row: Parameters<typeof publicManifest>[0]): string {
  return sha256(canonicalTaskAttachmentValue(publicManifest(row)));
}

export function taskAttachmentRequestItem(row: PersistedTaskAttachmentRow): Record<string, unknown> {
  return {
    contentSha256: row.content_sha256,
    bytes: row.bytes,
    mimeType: row.mime_type,
    kind: row.kind,
    displayName: row.display_name,
    source: JSON.parse(row.source_json)
  };
}

function handle(row: PersistedTaskAttachmentRow): ChatTaskAttachmentHandle {
  if (taskAttachmentManifestSha256(row) !== row.manifest_sha256) throw new Error("Attachment manifest checksum failed.");
  return {
    format: TASK_ATTACHMENT_FORMAT,
    version: TASK_ATTACHMENT_VERSION,
    id: row.id,
    brainId: row.brain_id,
    displayName: row.display_name,
    kind: row.kind,
    mimeType: row.mime_type,
    contentSha256: row.content_sha256,
    manifestSha256: row.manifest_sha256,
    bytes: row.bytes,
    state: row.state,
    ...(row.turn_id ? { turnId: row.turn_id } : {}),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    ingestionStarted: false,
    parametersChanged: false
  };
}

async function exists(path: string): Promise<boolean> {
  try { await access(path); return true; } catch { return false; }
}

async function fileSha256(path: string, signal?: AbortSignal): Promise<string> {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path, { signal })) digest.update(chunk);
  return digest.digest("hex");
}

async function syncFile(path: string): Promise<void> {
  const file = await open(path, "r+");
  try { await file.sync(); } finally { await file.close(); }
}

async function promoteContentAddressed(temporary: string, destination: string, expectedSha256: string, expectedBytes: number, signal?: AbortSignal): Promise<void> {
  signal?.throwIfAborted();
  await mkdir(dirname(destination), { recursive: true });
  try {
    await link(temporary, destination);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "EXDEV" || code === "EPERM" || code === "EACCES") {
      try { await copyFile(temporary, destination, fsConstants.COPYFILE_EXCL); }
      catch (copyError) { if ((copyError as NodeJS.ErrnoException).code !== "EEXIST") throw copyError; }
    } else if (code !== "EEXIST") throw error;
  }
  await syncFile(destination);
  const info = await lstat(destination);
  if (!info.isFile() || info.isSymbolicLink() || info.size !== expectedBytes || await fileSha256(destination, signal) !== expectedSha256) {
    throw new Error("Content-addressed attachment promotion failed integrity validation.");
  }
}

export class TaskAttachmentStore {
  private readonly root: string;
  private readonly blobs: string;
  private readonly databasePath: string;
  constructor(private readonly brainRoot: string, readonly brainId: string,
    private readonly readVerifier = new TaskAttachmentReadVerifier()) {
    requireId(brainId, "attachment brain id");
    this.root = join(brainRoot, "attachments");
    this.blobs = join(this.root, "blobs");
    this.databasePath = join(this.root, TASK_ATTACHMENT_DATABASE);
  }

  async initialize(): Promise<void> {
    await mkdir(this.blobs, { recursive: true, mode: 0o700 });
    for (const path of [this.root, this.blobs]) {
      const info = await lstat(path);
      if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Attachment workspace is unsafe.");
    }
    this.database(() => undefined);
  }

  private database<T>(operation: (database: DatabaseSync) => T): T {
    const database = new DatabaseSync(sqliteFilename(this.databasePath));
    try {
      database.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA busy_timeout=30000; PRAGMA foreign_keys=ON;");
      database.exec(`
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL) STRICT;
        CREATE TABLE IF NOT EXISTS requests(
          request_id TEXT PRIMARY KEY, request_sha256 TEXT NOT NULL, attachment_count INTEGER NOT NULL,
          created_at TEXT NOT NULL
        ) STRICT;
        CREATE TABLE IF NOT EXISTS attachments(
          id TEXT PRIMARY KEY, brain_id TEXT NOT NULL, request_id TEXT NOT NULL, request_index INTEGER NOT NULL,
          content_sha256 TEXT NOT NULL, manifest_sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,
          mime_type TEXT NOT NULL, kind TEXT NOT NULL, display_name TEXT NOT NULL,
          source_json TEXT NOT NULL, source_sha256 TEXT NOT NULL, state TEXT NOT NULL,
          turn_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          UNIQUE(request_id,request_index), FOREIGN KEY(request_id) REFERENCES requests(request_id) ON DELETE RESTRICT
        ) STRICT;
        CREATE TABLE IF NOT EXISTS turn_bindings(
          turn_id TEXT PRIMARY KEY, attachment_set_sha256 TEXT NOT NULL, bound_at TEXT NOT NULL
        ) STRICT;
        CREATE INDEX IF NOT EXISTS attachments_turn ON attachments(turn_id,state,request_id,request_index);
        CREATE INDEX IF NOT EXISTS attachments_staged ON attachments(state,created_at,id);
      `);
      database.prepare("INSERT OR IGNORE INTO meta VALUES('format','omni-chat-task-attachments-1')").run();
      database.prepare("INSERT OR IGNORE INTO meta VALUES('brainId',?)").run(this.brainId);
      const owner = database.prepare("SELECT value FROM meta WHERE key='brainId'").get() as { value?: string } | undefined;
      if (owner?.value !== this.brainId) throw new Error("Attachment workspace belongs to another brain.");
      return operation(database);
    } finally { database.close(); }
  }

  async requestHandles(requestId: string): Promise<ChatTaskAttachmentHandle[] | undefined> {
    await this.initialize();
    requireId(requestId, "attachment request id");
    return this.database((database) => {
      const request = database.prepare("SELECT attachment_count FROM requests WHERE request_id=?").get(requestId) as { attachment_count?: number } | undefined;
      if (!request) return undefined;
      const rows = database.prepare("SELECT * FROM attachments WHERE request_id=? ORDER BY request_index").all(requestId) as unknown as PersistedTaskAttachmentRow[];
      if (rows.length !== request.attachment_count) throw new Error("Attachment request receipt is incomplete.");
      return rows.map(handle);
    });
  }

  private async materializePath(source: string, signal?: AbortSignal): Promise<{ sha256: string; bytes: number; actual: string }> {
    signal?.throwIfAborted();
    const actual = await realpath(resolve(source));
    const sourceHandle = await open(actual, fsConstants.O_RDONLY | (fsConstants.O_NOFOLLOW ?? 0));
    const before = await sourceHandle.stat();
    if (!before.isFile()) { await sourceHandle.close(); throw new Error("Selected attachment is not a safe regular file."); }
    const temporary = join(this.blobs, `.stage-${randomUUID()}.data`);
    const digest = createHash("sha256");
    const tap = new Transform({ transform(chunk, _encoding, callback) { digest.update(chunk); callback(null, chunk); } });
    try {
      await ensureDiskReserve(this.root, before.size);
      await pipeline(sourceHandle.createReadStream({ autoClose: false }), tap,
        createWriteStream(temporary, { flags: "wx", mode: 0o600 }), { signal });
      await syncFile(temporary);
      const [openedAfter, pathAfter, copied] = await Promise.all([
        sourceHandle.stat(), lstat(actual), lstat(temporary)
      ]);
      if (!openedAfter.isFile() || !pathAfter.isFile() || pathAfter.isSymbolicLink() ||
          [openedAfter, pathAfter].some((after) => after.dev !== before.dev || after.ino !== before.ino ||
            after.size !== before.size || after.mtimeMs !== before.mtimeMs) || copied.size !== before.size) {
        throw new Error("Selected attachment changed during staging.");
      }
      const contentSha256 = digest.digest("hex");
      await promoteContentAddressed(temporary, join(this.blobs, `${contentSha256}.data`), contentSha256, copied.size);
      return { sha256: contentSha256, bytes: copied.size, actual };
    } finally {
      await sourceHandle.close().catch(() => undefined);
      await rm(temporary, { force: true }).catch(() => undefined);
    }
  }

  private async materializeBytes(data: Uint8Array, signal?: AbortSignal): Promise<{ sha256: string; bytes: number }> {
    signal?.throwIfAborted();
    if (!(data instanceof Uint8Array) || data.byteLength < 1) throw new Error("Attachment bytes are empty or invalid.");
    await ensureDiskReserve(this.root, data.byteLength);
    const contentSha256 = sha256(data), temporary = join(this.blobs, `.stage-${randomUUID()}.data`);
    const file = await open(temporary, "wx", 0o600);
    try { await file.writeFile(data); await file.sync(); } finally { await file.close(); }
    try {
      signal?.throwIfAborted();
      await promoteContentAddressed(temporary, join(this.blobs, `${contentSha256}.data`), contentSha256, data.byteLength);
      return { sha256: contentSha256, bytes: data.byteLength };
    } finally { await rm(temporary, { force: true }).catch(() => undefined); }
  }

  private commit(requestId: string, requestSha256: string, prepared: Array<{
    contentSha256: string; bytes: number; mimeType: string; kind: ChatTaskAttachmentKind;
    displayName: string; source: TaskAttachmentPrivateSource;
  }>): ChatTaskAttachmentHandle[] {
    for (const item of prepared) if (item.source.kind === "web-image") {
      const download = checkedWebImageDownload(item.source.webImage);
      if (download.reference.brainId !== this.brainId || download.sha256 !== item.contentSha256 ||
        download.bytes !== item.bytes || download.mimeType !== item.mimeType || item.kind !== "image") throw new Error("Selected web image source differs from its actual staged bytes.");
    }
    return this.database((database) => {
      database.exec("BEGIN IMMEDIATE");
      try {
        const prior = database.prepare("SELECT request_sha256,attachment_count FROM requests WHERE request_id=?").get(requestId) as
          { request_sha256: string; attachment_count: number } | undefined;
        if (prior) {
          if (prior.request_sha256 !== requestSha256 || prior.attachment_count !== prepared.length) {
            throw new Error("Attachment request id was reused for different source bytes.");
          }
          const rows = database.prepare("SELECT * FROM attachments WHERE request_id=? ORDER BY request_index").all(requestId) as unknown as PersistedTaskAttachmentRow[];
          if (rows.length !== prior.attachment_count) throw new Error("Attachment request lost its committed rows.");
          database.exec("COMMIT");
          return rows.map(handle);
        }
        const now = new Date().toISOString();
        database.prepare("INSERT INTO requests VALUES(?,?,?,?)").run(requestId, requestSha256, prepared.length, now);
        const insert = database.prepare(`INSERT INTO attachments
          (id,brain_id,request_id,request_index,content_sha256,manifest_sha256,bytes,mime_type,kind,display_name,
           source_json,source_sha256,state,turn_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`);
        const rows: PersistedTaskAttachmentRow[] = prepared.map((item, index) => {
          const sourceJson = canonicalTaskAttachmentValue(validatedTaskAttachmentSource(item.source));
          const row: PersistedTaskAttachmentRow = {
            id: randomUUID(), brain_id: this.brainId, request_id: requestId, request_index: index,
            content_sha256: item.contentSha256, manifest_sha256: "", bytes: item.bytes,
            mime_type: item.mimeType, kind: item.kind, display_name: item.displayName,
            source_json: sourceJson, source_sha256: sha256(sourceJson), state: "staged", turn_id: null,
            created_at: now, updated_at: now
          };
          row.manifest_sha256 = taskAttachmentManifestSha256(row);
          insert.run(row.id, row.brain_id, row.request_id, row.request_index, row.content_sha256,
            row.manifest_sha256, row.bytes, row.mime_type, row.kind, row.display_name, row.source_json,
            row.source_sha256, row.state, row.turn_id, row.created_at, row.updated_at);
          return row;
        });
        database.exec("COMMIT");
        return rows.map(handle);
      } catch (error) { database.exec("ROLLBACK"); throw error; }
    });
  }

  async stagePaths(requestId: string, sources: readonly TaskAttachmentStagePath[], selection: ChatTaskAttachmentSelection, signal?: AbortSignal,
    onProgress?: (progress: Readonly<TaskFolderTraversal>) => void): Promise<ChatTaskAttachmentHandle[]> {
    await this.initialize(); requireId(requestId, "attachment request id");
    if (!sources.length) return [];
    const selectedKind = selection === "folder" ? undefined : kindForSelection(selection), prepared = [] as Array<{
      contentSha256: string; bytes: number; mimeType: string; kind: ChatTaskAttachmentKind;
      displayName: string; source: TaskAttachmentPrivateSource;
    }>;
    try {
      for (const source of sources) {
        if (selection === "folder") {
          const temporary = join(this.blobs, `.folder-stage-${randomUUID()}.sqlite3`);
          try {
            this.readVerifier.reserveMemory(4 * 1024 * 1024, "Folder staging SQLite cache and streaming buffer");
            let lastProgress = -Infinity;
            await buildTaskFolderContainer(source.path, temporary, inferredMimeType, signal, progress => {
              const now = performance.now();
              if (progress.complete || now - lastProgress >= 100) { lastProgress = now; onProgress?.(progress); }
            });
            signal?.throwIfAborted();
            const info = await lstat(temporary);
            if (!info.isFile() || info.isSymbolicLink() || !Number.isSafeInteger(info.size)) throw new Error("Owned folder container changed before publication.");
            const contentSha256 = await fileSha256(temporary, signal);
            // This newly created, closed private container has no external
            // writer. Promote it directly; do not copy a whole folder twice.
            prepared.push({ contentSha256, bytes: info.size,
              mimeType: TASK_FOLDER_MIME, kind: "file", displayName: safeName(source.displayName ?? (basename(resolve(source.path)) || "Folder")),
              source: validatedTaskAttachmentSource({ kind: "local-folder", originalPath: resolve(source.path), selectedAt: new Date().toISOString() }) });
            await promoteContentAddressed(temporary, join(this.blobs, `${contentSha256}.data`), contentSha256, info.size, signal);
          } finally { await rm(temporary, { force: true }).catch(() => undefined); }
          continue;
        }
        const materialized = await this.materializePath(source.path, signal);
        const mimeType = safeMimeType(source.mimeType ?? inferredMimeType(materialized.actual));
        prepared.push({ contentSha256: materialized.sha256, bytes: materialized.bytes, mimeType,
          kind: source.kind ?? selectedKind ?? inferredKind(mimeType), displayName: safeName(source.displayName ?? basename(materialized.actual)),
          source: validatedTaskAttachmentSource(source.source ?? { kind: "local-file", originalPath: materialized.actual, selectedAt: new Date().toISOString() }) });
      }
      signal?.throwIfAborted();
      const requestSha256 = sha256(canonicalTaskAttachmentValue(prepared.map((item) => ({ ...item, source: item.source }))));
      return this.commit(requestId, requestSha256, prepared);
    }
    catch (error) { await this.removeUnreferencedBlobs(prepared.map((item) => item.contentSha256)); throw error; }
  }

  async stageBytes(requestId: string, source: TaskAttachmentStageBytes, signal?: AbortSignal): Promise<ChatTaskAttachmentHandle> {
    await this.initialize(); requireId(requestId, "attachment request id");
    const materialized = await this.materializeBytes(source.data, signal);
    const prepared = [{ contentSha256: materialized.sha256, bytes: materialized.bytes,
      mimeType: safeMimeType(source.mimeType), kind: source.kind, displayName: safeName(source.displayName), source: validatedTaskAttachmentSource(source.source) }];
    try {
      signal?.throwIfAborted();
      const requestSha256 = sha256(canonicalTaskAttachmentValue(prepared.map((item) => ({ ...item, source: item.source }))));
      return this.commit(requestId, requestSha256, prepared)[0]!;
    }
    catch (error) { await this.removeUnreferencedBlobs([materialized.sha256]); throw error; }
  }

  private async removeUnreferencedBlobs(contentSha256s: readonly string[]): Promise<void> {
    for (const contentSha256 of new Set(contentSha256s)) {
      const used = this.database((database) => Number(database.prepare(
        "SELECT COUNT(*) AS count FROM attachments WHERE content_sha256=?"
      ).get(contentSha256)?.count ?? 0));
      if (used === 0) await rm(join(this.blobs, `${contentSha256}.data`), { force: true }).catch(() => undefined);
    }
  }

  async list(includeRetired = false, state?: "staged"): Promise<ChatTaskAttachmentHandle[]> {
    await this.initialize();
    if (state !== undefined && state !== "staged") throw new Error("Invalid attachment state filter.");
    if (state === "staged") return this.database((database) => (database.prepare(
      "SELECT * FROM attachments WHERE state='staged' ORDER BY created_at,id"
    ).all() as unknown as PersistedTaskAttachmentRow[]).map(handle));
    return this.database((database) => (database.prepare(`SELECT * FROM attachments ${includeRetired ? "" : "WHERE state != 'retired'"} ORDER BY created_at,id`).all() as unknown as PersistedTaskAttachmentRow[]).map(handle));
  }

  async handlesForTurn(turnId: string): Promise<ChatTaskAttachmentHandle[]> {
    await this.initialize(); requireId(turnId, "attachment turn id");
    return this.database((database) => (database.prepare(
      "SELECT * FROM attachments WHERE turn_id=? AND state='bound' ORDER BY request_id,request_index"
    ).all(turnId) as unknown as PersistedTaskAttachmentRow[]).map(handle));
  }

  async handlesByIds(attachmentIds: readonly string[]): Promise<ChatTaskAttachmentHandle[]> {
    const ids = attachmentIds.map(value => requireId(value, "attachment id"));
    if (new Set(ids).size !== ids.length) throw new Error("Attachment identities are duplicated.");
    await this.initialize();
    return this.database(database => {
      database.exec("BEGIN");
      try {
        const select = database.prepare("SELECT * FROM attachments WHERE id=? AND state != 'retired'");
        const selected = ids.map(id => {
          const row = select.get(id) as unknown as PersistedTaskAttachmentRow | undefined;
          if (!row) throw new Error("Task attachment handle is unavailable.");
          return handle(row);
        });
        database.exec("COMMIT"); return selected;
      } catch (failure) { database.exec("ROLLBACK"); throw failure; }
    });
  }

  async removeStaged(attachmentId: string): Promise<boolean> {
    await this.initialize(); requireId(attachmentId, "attachment id");
    const removed = this.database((database) => {
      database.exec("BEGIN IMMEDIATE");
      try {
      const row = database.prepare("SELECT * FROM attachments WHERE id=?").get(attachmentId) as unknown as PersistedTaskAttachmentRow | undefined;
      if (!row) { database.exec("COMMIT"); return undefined; }
      if (row.state !== "staged") throw new Error("A bound or historical attachment cannot be deleted from its committed task.");
      database.prepare("DELETE FROM attachments WHERE id=? AND state='staged'").run(attachmentId);
      const remaining = database.prepare("SELECT * FROM attachments WHERE request_id=? ORDER BY request_index")
        .all(row.request_id) as unknown as PersistedTaskAttachmentRow[];
      if (!remaining.length) database.prepare("DELETE FROM requests WHERE request_id=?").run(row.request_id);
      else {
        remaining.forEach((item, index) => database.prepare("UPDATE attachments SET request_index=? WHERE id=?").run(index, item.id));
        const requestSha256 = sha256(canonicalTaskAttachmentValue(remaining.map(taskAttachmentRequestItem)));
        database.prepare("UPDATE requests SET request_sha256=?,attachment_count=? WHERE request_id=?")
          .run(requestSha256, remaining.length, row.request_id);
      }
      database.exec("COMMIT");
      return { contentSha256: row.content_sha256 };
      } catch (error) { database.exec("ROLLBACK"); throw error; }
    });
    if (!removed) return false;
    await this.removeUnreferencedBlobs([removed.contentSha256]);
    return true;
  }

  async bindToTurn(turnId: string, attachmentIds: readonly string[]): Promise<ChatTaskAttachmentResolved[]> {
    await this.initialize(); requireId(turnId, "attachment turn id");
    const ids = [...new Set(attachmentIds.map((id) => requireId(id, "attachment id")))].sort();
    if (ids.length !== attachmentIds.length || !ids.length) throw new Error("Attachment turn binding is empty or duplicated.");
    return this.database((database) => {
      database.exec("BEGIN IMMEDIATE");
      try {
        const setSha256 = sha256(canonicalTaskAttachmentValue(ids));
        const prior = database.prepare("SELECT attachment_set_sha256 FROM turn_bindings WHERE turn_id=?").get(turnId) as { attachment_set_sha256?: string } | undefined;
        if (prior?.attachment_set_sha256 && prior.attachment_set_sha256 !== setSha256) {
          throw new Error("This chat turn is already bound to a different exact attachment set.");
        }
        const rows: PersistedTaskAttachmentRow[] = [];
        for (const id of ids) {
          const row = database.prepare("SELECT * FROM attachments WHERE id=?").get(id) as unknown as PersistedTaskAttachmentRow | undefined;
          if (!row || row.brain_id !== this.brainId || row.state === "retired" ||
              (row.state === "bound" && row.turn_id !== turnId)) throw new Error("Attachment is unavailable for this chat turn.");
          rows.push(row);
        }
        if (prior) {
          database.exec("COMMIT");
          return rows.map((row) => this.resolved(row, turnId));
        }
        const now = new Date().toISOString();
        database.prepare("INSERT INTO turn_bindings VALUES(?,?,?)").run(turnId, setSha256, now);
        for (const row of rows) {
          row.state = "bound"; row.turn_id = turnId; row.updated_at = now; row.manifest_sha256 = taskAttachmentManifestSha256(row);
          database.prepare("UPDATE attachments SET state='bound',turn_id=?,updated_at=?,manifest_sha256=? WHERE id=?")
            .run(turnId, now, row.manifest_sha256, row.id);
        }
        database.exec("COMMIT");
        return rows.map((row) => this.resolved(row, turnId));
      } catch (error) { database.exec("ROLLBACK"); throw error; }
    });
  }

  private resolved(row: PersistedTaskAttachmentRow, turnId: string): ChatTaskAttachmentResolved {
    const current = handle(row);
    if (current.state !== "bound" || current.turnId !== turnId) throw new Error("Attachment is not bound to this task.");
    return { brainId: this.brainId, turnId, attachmentId: row.id, displayName: row.display_name,
      kind: row.kind, mimeType: row.mime_type, contentSha256: row.content_sha256,
      manifestSha256: row.manifest_sha256, bytes: row.bytes, relativePath: `attachments/blobs/${row.content_sha256}.data` };
  }

  async listForTurn(turnId: string): Promise<ChatTaskAttachmentResolved[]> {
    await this.initialize(); requireId(turnId, "attachment turn id");
    return this.database((database) => (database.prepare("SELECT * FROM attachments WHERE state='bound' AND turn_id=? ORDER BY request_id,request_index")
      .all(turnId) as unknown as PersistedTaskAttachmentRow[]).map((row) => this.resolved(row, turnId)));
  }

  private async readOwnerIdentity(): Promise<string> {
    const directories = [];
    for (const path of [this.brainRoot, this.root, this.blobs]) {
      const info = await lstat(path, { bigint: true });
      if (!info.isDirectory() || info.isSymbolicLink() || info.ino <= 0n) throw new Error("Attachment read workspace identity is unsafe or unsupported.");
      directories.push([resolve(path), String(info.dev), String(info.ino)]);
    }
    const registry = await lstat(this.databasePath, { bigint: true });
    if (!registry.isFile() || registry.isSymbolicLink()) throw new Error("Attachment read registry is unsafe.");
    return canonicalTaskAttachmentValue({ brainId: this.brainId, directories, registry: taskAttachmentFileIdentity(registry) });
  }

  private async readAuthority(turnId: string, attachmentId: string): Promise<{
    row: PersistedTaskAttachmentRow; descriptor: ChatTaskAttachmentResolved; authorityKey: string;
  }> {
    const before = await this.readOwnerIdentity();
    const row = this.database((database) => database.prepare("SELECT * FROM attachments WHERE id=?").get(attachmentId) as unknown as PersistedTaskAttachmentRow | undefined);
    if (!row || row.brain_id !== this.brainId || row.turn_id !== turnId || row.state !== "bound") throw new Error("Attachment is unavailable outside its exact chat task.");
    if (!/^[a-f0-9]{64}$/.test(row.content_sha256) || !/^[a-f0-9]{64}$/.test(row.source_sha256) ||
        !Number.isSafeInteger(row.bytes) || row.bytes < 1 || sha256(row.source_json) !== row.source_sha256) {
      throw new Error("Attachment source identity or provenance checksum failed.");
    }
    const source = validatedTaskAttachmentSource(JSON.parse(row.source_json) as TaskAttachmentPrivateSource);
    if (source.kind === "web-image") {
      const download = checkedWebImageDownload(source.webImage);
      if (download.sha256 !== row.content_sha256 || download.bytes !== row.bytes || download.mimeType !== row.mime_type || row.kind !== "image") throw new Error("Selected image source/attachment bytes disagree.");
    }
    const descriptor = this.resolved(row, turnId), after = await this.readOwnerIdentity();
    if (before !== after) throw new Error("Task attachment owner or registry changed while its read was pending.");
    return { row, descriptor, authorityKey: sha256(canonicalTaskAttachmentValue({ owner: before,
      brainId: this.brainId, turnId, attachmentId, manifestSha256: row.manifest_sha256,
      sourceSha256: row.source_sha256, contentSha256: row.content_sha256, bytes: row.bytes })) };
  }

  private async withVerifiedRead<T>(turnId: string, attachmentId: string,
    authority: Awaited<ReturnType<TaskAttachmentStore["readAuthority"]>>,
    operation: (file: Awaited<ReturnType<typeof open>>) => Promise<T>, signal?: AbortSignal): Promise<T> {
    const assertAuthority = async (): Promise<void> => {
      signal?.throwIfAborted();
      if ((await this.readAuthority(turnId, attachmentId)).authorityKey !== authority.authorityKey) {
        throw new Error("Task attachment changed while its verified read was pending.");
      }
    };
    return this.readVerifier.withVerifiedFile({ authorityKey: authority.authorityKey,
      path: join(this.blobs, `${authority.row.content_sha256}.data`),
      contentSha256: authority.row.content_sha256, bytes: authority.row.bytes }, assertAuthority, operation, signal);
  }

  async resolveForTurn(turnId: string, attachmentId: string, signal?: AbortSignal): Promise<ChatTaskAttachmentResolved> {
    signal?.throwIfAborted(); await this.initialize(); requireId(turnId, "attachment turn id"); requireId(attachmentId, "attachment id");
    const authority = await this.readAuthority(turnId, attachmentId);
    return this.withVerifiedRead(turnId, attachmentId, authority, async () => authority.descriptor, signal);
  }

  async readForTurn(turnId: string, attachmentId: string, offset = 0, maximumBytes = 64 * 1024, signal?: AbortSignal, entryPath?: string): Promise<ChatTaskAttachmentReadResult> {
    signal?.throwIfAborted(); await this.initialize(); requireId(turnId, "attachment turn id"); requireId(attachmentId, "attachment id");
    const authority = await this.readAuthority(turnId, attachmentId), descriptor = authority.descriptor;
    if (!Number.isSafeInteger(offset) || offset < 0 || offset > descriptor.bytes ||
        !Number.isSafeInteger(maximumBytes) || maximumBytes < 1 || maximumBytes > 1024 * 1024) {
      throw new Error("Attachment read range is invalid.");
    }
    if (descriptor.mimeType === TASK_FOLDER_MIME) {
      this.readVerifier.reserveMemory(8 * maximumBytes + 4 * 1024 * 1024, "Saved folder page and SQLite cache");
      return this.withVerifiedRead(turnId, attachmentId, authority, opened => readTaskFolderAttachment(
        handle(authority.row), join(this.blobs, `${authority.row.content_sha256}.data`), opened,
        offset, maximumBytes, entryPath, signal), signal);
    }
    if (entryPath !== undefined) throw new Error("Entry paths are available only inside a saved folder attachment.");
    const buffer = await this.withVerifiedRead(turnId, attachmentId, authority, async file => {
      const length = Math.min(maximumBytes, descriptor.bytes - offset);
      this.readVerifier.reserveMemory(8 * length + 64 * 1024, "Task attachment page and encoding buffer");
      const page = Buffer.alloc(length);
      let returned = 0;
      while (returned < length) {
        signal?.throwIfAborted();
        const read = await file.read(page, returned, length - returned, offset + returned);
        if (!read.bytesRead) throw new Error("Task attachment ended before its requested exact page.");
        returned += read.bytesRead;
      }
      return page;
    }, signal);
    let bytes = buffer;
    const textual = descriptor.mimeType.startsWith("text/") ||
      ["application/json", "application/x-ndjson", "application/xml", "application/javascript"].includes(descriptor.mimeType);
    let encoding: "utf8" | "base64" = "base64", content = bytes.toString("base64");
    if (textual) {
      for (let trim = 0; trim <= Math.min(3, Math.max(0, bytes.byteLength - 1)); trim += 1) {
        try {
          const candidate = bytes.subarray(0, bytes.byteLength - trim);
          content = new TextDecoder("utf-8", { fatal: true }).decode(candidate);
          bytes = candidate; encoding = "utf8"; break;
        } catch { /* Preserve undecodable bytes as exact base64. */ }
      }
    }
    const nextOffset = offset + bytes.byteLength < descriptor.bytes ? offset + bytes.byteLength : undefined;
    signal?.throwIfAborted();
    const attachment = handle(authority.row);
    return { format: "omni-chat-task-attachment-read", version: 1, attachment,
      offset, returnedBytes: bytes.byteLength, totalBytes: descriptor.bytes, encoding,
      content, ...(nextOffset === undefined ? {} : { nextOffset }) };
  }

  static databasePath(brainRoot: string): string { return join(brainRoot, "attachments", TASK_ATTACHMENT_DATABASE); }
}

export { inferredKind, inferredMimeType, kindForSelection };
