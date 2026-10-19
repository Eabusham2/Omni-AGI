import { createHash } from "node:crypto";
import { lstat, opendir, rm } from "node:fs/promises";
import { basename, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { copyMutableFileIsolated } from "./mutableFileIsolation";
import { mergeTaskContinuationScopes, rekeyTaskContinuationScopes, validateTaskContinuationScopes } from "./taskContinuationScope";
import { mergeInheritedTaskFileReviews, rekeyInheritedTaskFileReviews, validateInheritedTaskFileReviews } from "./inheritedTaskFileReviewStore";
import { checkedWebImageDownload, mergeWebImageReferences, retireWebImageReferences, validateWebImageReferences } from "./webImageReferences";
import {
  TASK_ATTACHMENT_DATABASE,
  TaskAttachmentStore,
  canonicalTaskAttachmentValue,
  taskAttachmentRequestItem,
  taskAttachmentManifestSha256,
  validatedTaskAttachmentSource,
  type PersistedTaskAttachmentRow,
  type TaskAttachmentPrivateSource
} from "./taskAttachmentStore";
import { sqliteFilename } from "./sqliteFilename";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const SHA256 = /^[a-f0-9]{64}$/;
const MIME = /^[a-z0-9][a-z0-9.+-]{0,80}\/[a-z0-9][a-z0-9.+-]{0,80}$/;

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

async function exists(path: string): Promise<boolean> {
  try { await lstat(path); return true; } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; }
}

async function fileSha256(path: string): Promise<string> {
  const digest = createHash("sha256");
  const input = (await import("node:fs")).createReadStream(path);
  for await (const chunk of input) digest.update(chunk);
  return digest.digest("hex");
}

function validateRow(row: PersistedTaskAttachmentRow, brainId: string): void {
  const source = JSON.parse(row.source_json) as TaskAttachmentPrivateSource;
  validatedTaskAttachmentSource(source);
  if (source.kind === "web-image") {
    const download = checkedWebImageDownload(source.webImage);
    if (download.sha256 !== row.content_sha256 || download.bytes !== row.bytes || download.mimeType !== row.mime_type || row.kind !== "image") throw new Error("Portable selected image differs from its exact source receipt.");
  }
  if (!SAFE_ID.test(row.id) || row.brain_id !== brainId || !SAFE_ID.test(row.request_id) ||
      !Number.isSafeInteger(row.request_index) || row.request_index < 0 || !SHA256.test(row.content_sha256) ||
      !SHA256.test(row.manifest_sha256) || !Number.isSafeInteger(row.bytes) || row.bytes < 1 ||
      !MIME.test(row.mime_type) || !["file", "image", "audio", "video"].includes(row.kind) ||
      typeof row.display_name !== "string" || !row.display_name || row.display_name.length > 512 ||
      sha256(row.source_json) !== row.source_sha256 || !["staged", "bound", "retired"].includes(row.state) ||
      (row.state === "bound" ? !row.turn_id || !SAFE_ID.test(row.turn_id) : row.turn_id !== null) ||
      !Number.isFinite(Date.parse(row.created_at)) || !Number.isFinite(Date.parse(row.updated_at)) ||
      taskAttachmentManifestSha256(row) !== row.manifest_sha256) {
    throw new Error("Portable task attachment row failed identity validation.");
  }
}

/** Validate the complete app-owned workspace; raw bytes are never interpreted or learned here. */
export async function validateTaskAttachmentWorkspace(brainRoot: string, brainId: string): Promise<boolean> {
  if (!SAFE_ID.test(brainId)) throw new Error("Invalid task attachment workspace owner.");
  const root = join(brainRoot, "attachments"), databasePath = join(root, TASK_ATTACHMENT_DATABASE);
  if (!await exists(databasePath)) {
    if (!await exists(root)) return false;
    const directory = await opendir(root);
    for await (const _entry of directory) throw new Error("Task attachment workspace has no registry.");
    return false;
  }
  const info = await lstat(databasePath);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Task attachment registry is unsafe.");
  const database = new DatabaseSync(sqliteFilename(databasePath), { readOnly: true });
  const referenced = new Map<string, number>();
  try {
    if (database.prepare("PRAGMA quick_check").get()?.quick_check !== "ok" ||
        database.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== "omni-chat-task-attachments-1" ||
        database.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) {
      throw new Error("Task attachment registry failed integrity validation.");
    }
    const rows = database.prepare("SELECT * FROM attachments ORDER BY request_id,request_index").all() as unknown as PersistedTaskAttachmentRow[];
    for (const row of rows) {
      validateRow(row, brainId);
      const prior = referenced.get(row.content_sha256);
      if (prior !== undefined && prior !== row.bytes) throw new Error("Task attachment content length disagrees.");
      referenced.set(row.content_sha256, row.bytes);
    }
    for (const request of database.prepare("SELECT * FROM requests ORDER BY request_id").all() as Array<Record<string, unknown>>) {
      if (!SAFE_ID.test(String(request.request_id)) || !SHA256.test(String(request.request_sha256)) ||
          !Number.isSafeInteger(request.attachment_count) || Number(request.attachment_count) < 1 ||
          !Number.isFinite(Date.parse(String(request.created_at)))) throw new Error("Task attachment request receipt is invalid.");
      const selected = rows.filter((row) => row.request_id === request.request_id);
      if (selected.length !== request.attachment_count || selected.some((row, index) => row.request_index !== index) ||
          sha256(canonicalTaskAttachmentValue(selected.map(taskAttachmentRequestItem))) !== request.request_sha256) {
        throw new Error("Task attachment request receipt differs from its exact source set.");
      }
    }
    for (const binding of database.prepare("SELECT * FROM turn_bindings ORDER BY turn_id").all() as Array<Record<string, unknown>>) {
      const turnId = String(binding.turn_id), selected = rows.filter((row) => row.state === "bound" && row.turn_id === turnId)
        .map((row) => row.id).sort();
      if (!SAFE_ID.test(turnId) || !selected.length || sha256(canonicalTaskAttachmentValue(selected)) !== binding.attachment_set_sha256 ||
          !Number.isFinite(Date.parse(String(binding.bound_at)))) throw new Error("Task attachment turn binding is invalid.");
    }
    const boundTurns = new Set(rows.filter((row) => row.state === "bound").map((row) => row.turn_id));
    if (boundTurns.size !== Number(database.prepare("SELECT COUNT(*) AS count FROM turn_bindings").get()?.count ?? 0)) {
      throw new Error("Task attachment turn bindings are incomplete.");
    }
    validateTaskContinuationScopes(database, brainId);
    validateInheritedTaskFileReviews(database, brainId);
  } finally { database.close(); }
  await validateWebImageReferences(brainRoot, brainId);
  const blobs = join(root, "blobs");
  const blobInfo = await lstat(blobs);
  if (!blobInfo.isDirectory() || blobInfo.isSymbolicLink()) throw new Error("Task attachment blob directory is unsafe.");
  const observed = new Set<string>(), directory = await opendir(blobs);
  for await (const entry of directory) {
    if (!entry.isFile() || !/^[a-f0-9]{64}\.data$/.test(entry.name)) throw new Error("Task attachment blob path is invalid.");
    const contentSha256 = basename(entry.name, ".data"), expectedBytes = referenced.get(contentSha256);
    if (expectedBytes === undefined) throw new Error("Task attachment workspace contains an unreferenced blob.");
    const path = join(blobs, entry.name), item = await lstat(path);
    if (!item.isFile() || item.isSymbolicLink() || item.size !== expectedBytes || await fileSha256(path) !== contentSha256) {
      throw new Error("Task attachment blob failed integrity validation.");
    }
    observed.add(contentSha256);
  }
  if (observed.size !== referenced.size) throw new Error("Task attachment workspace is missing source bytes.");
  return true;
}

export async function rekeyTaskAttachmentWorkspace(brainRoot: string, sourceBrainId: string, targetBrainId: string): Promise<void> {
  if (!await validateTaskAttachmentWorkspace(brainRoot, sourceBrainId)) return;
  if (!SAFE_ID.test(targetBrainId)) throw new Error("Invalid task attachment target owner.");
  const path = TaskAttachmentStore.databasePath(brainRoot), database = new DatabaseSync(sqliteFilename(path));
  try {
    database.exec("BEGIN IMMEDIATE");
    try {
      database.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetBrainId);
      const rows = database.prepare("SELECT * FROM attachments ORDER BY id").all() as unknown as PersistedTaskAttachmentRow[];
      for (const row of rows) {
        if (sourceBrainId === targetBrainId) continue; // Recovery preserves exact source manifests; its task grants still become dormant.
        row.brain_id = targetBrainId;
        if (row.state === "staged") { row.state = "retired"; row.turn_id = null; }
        row.updated_at = new Date().toISOString(); row.manifest_sha256 = taskAttachmentManifestSha256(row);
        database.prepare("UPDATE attachments SET brain_id=?,state=?,turn_id=?,updated_at=?,manifest_sha256=? WHERE id=?")
          .run(row.brain_id, row.state, row.turn_id, row.updated_at, row.manifest_sha256, row.id);
      }
      rekeyTaskContinuationScopes(database, targetBrainId);
      rekeyInheritedTaskFileReviews(database, targetBrainId);
      retireWebImageReferences(database);
      database.exec("COMMIT");
    } catch (error) { database.exec("ROLLBACK"); throw error; }
  } finally { database.close(); }
  await validateTaskAttachmentWorkspace(brainRoot, targetBrainId);
}

/** Merge retains exact parent bytes/provenance but never activates them in a new current task. */
export async function mergeTaskAttachmentWorkspace(
  sourceRoot: string,
  targetRoot: string,
  targetBrainId: string,
  parentBrainId: string,
  reserve?: (bytes: number) => Promise<void>,
  turnAliases: ReadonlyMap<string, string> = new Map()
): Promise<void> {
  if (!await validateTaskAttachmentWorkspace(sourceRoot, targetBrainId)) return;
  const target = new TaskAttachmentStore(targetRoot, targetBrainId); await target.initialize();
  const sourceDatabase = new DatabaseSync(sqliteFilename(TaskAttachmentStore.databasePath(sourceRoot)), { readOnly: true });
  const targetDatabase = new DatabaseSync(sqliteFilename(TaskAttachmentStore.databasePath(targetRoot)));
  const copied: string[] = [];
  try {
    const rows = sourceDatabase.prepare("SELECT * FROM attachments ORDER BY request_id,request_index").all() as unknown as PersistedTaskAttachmentRow[];
    for (const old of rows) {
      const sourceBlob = join(sourceRoot, "attachments", "blobs", `${old.content_sha256}.data`);
      const destinationBlob = join(targetRoot, "attachments", "blobs", `${old.content_sha256}.data`);
      if (!await exists(destinationBlob)) {
        await reserve?.(old.bytes);
        await copyMutableFileIsolated(sourceBlob, destinationBlob, { expectedSha256: old.content_sha256 });
        copied.push(old.content_sha256);
      }
    }
    targetDatabase.exec("PRAGMA foreign_keys=ON; BEGIN IMMEDIATE");
    try {
      for (const old of rows) {
        const id = sha256(canonicalTaskAttachmentValue({ parentBrainId, attachmentId: old.id }));
        const requestId = sha256(canonicalTaskAttachmentValue({ parentBrainId, requestId: old.request_id, attachmentId: old.id }));
        const source = validatedTaskAttachmentSource({
          ...(JSON.parse(old.source_json) as TaskAttachmentPrivateSource),
          inheritedFrom: { brainId: parentBrainId, attachmentId: old.id }
        });
        const sourceJson = canonicalTaskAttachmentValue(source), now = new Date().toISOString();
        const row: PersistedTaskAttachmentRow = { ...old, id, brain_id: targetBrainId, request_id: requestId,
          request_index: 0, source_json: sourceJson, source_sha256: sha256(sourceJson), state: "retired", turn_id: null,
          updated_at: now, manifest_sha256: "" };
        row.manifest_sha256 = taskAttachmentManifestSha256(row);
        const requestSha256 = sha256(canonicalTaskAttachmentValue([taskAttachmentRequestItem(row)]));
        targetDatabase.prepare("INSERT OR IGNORE INTO requests VALUES(?,?,1,?)").run(requestId, requestSha256, now);
        targetDatabase.prepare(`INSERT OR IGNORE INTO attachments
          (id,brain_id,request_id,request_index,content_sha256,manifest_sha256,bytes,mime_type,kind,display_name,
           source_json,source_sha256,state,turn_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`)
          .run(row.id,row.brain_id,row.request_id,row.request_index,row.content_sha256,row.manifest_sha256,row.bytes,row.mime_type,
            row.kind,row.display_name,row.source_json,row.source_sha256,row.state,row.turn_id,row.created_at,row.updated_at);
      }
      mergeTaskContinuationScopes(sourceDatabase, targetDatabase, targetBrainId, parentBrainId, turnAliases);
      mergeInheritedTaskFileReviews(sourceDatabase, targetDatabase, targetBrainId, parentBrainId);
      mergeWebImageReferences(sourceDatabase, targetDatabase);
      targetDatabase.exec("COMMIT");
    } catch (error) { targetDatabase.exec("ROLLBACK"); throw error; }
  } catch (error) {
    for (const contentSha256 of copied) await rm(join(targetRoot, "attachments", "blobs", `${contentSha256}.data`), { force: true }).catch(() => undefined);
    throw error;
  } finally { sourceDatabase.close(); targetDatabase.close(); }
  await validateTaskAttachmentWorkspace(targetRoot, targetBrainId);
}
