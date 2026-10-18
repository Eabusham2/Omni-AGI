import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, open, readFile } from "node:fs/promises";
import { dirname, join, relative, sep } from "node:path";
import { createInterface } from "node:readline";
import { DatabaseSync } from "node:sqlite";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import type { InstanceResumeFrontier } from "../shared/instanceLifecycle";
import type { InheritedTaskFileDisclosure } from "../shared/inheritedTaskFiles";
import { canonicalTaskAttachmentValue, taskAttachmentManifestSha256, validatedTaskAttachmentSource,
  TaskAttachmentStore, type PersistedTaskAttachmentRow } from "./taskAttachmentStore";
import { readTaskContinuationScope } from "./taskContinuationScope";
import { TaskAttachmentReadVerifier } from "./taskAttachmentReadVerifier";

const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const SHA = /^[a-f0-9]{64}$/;
export const inheritedTaskFileHash = (value: unknown): string => createHash("sha256").update(canonicalTaskAttachmentValue(value)).digest("hex");
const object = (value: unknown): Record<string, any> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Inherited file closure has an invalid declaration.");
  return value as Record<string, any>;
};

export interface InheritedTaskFileNativeContext {
  brainId: string;
  executionId: string;
  frontier: InstanceResumeFrontier;
  inputSha256: string;
  sourceBrainId: string;
  sourceTurnId: string;
  sourceAttentionEpoch: number;
  rootAttentionEpoch: number;
  descriptorSha256: string;
  regionPath: string[];
  sourceManifestSha256: string;
  sourceBindingId: string;
}

export interface InheritedTaskFileGuard { path: string; identity: string; }
export interface InheritedTaskFilePrivate {
  disclosure: InheritedTaskFileDisclosure;
  row: PersistedTaskAttachmentRow;
  sourceGuard: InheritedTaskFileGuard;
}
export interface InheritedTaskFileClosure {
  sourceClosureSha256: string;
  ownerBindingSha256: string;
  files: InheritedTaskFilePrivate[];
  guards: InheritedTaskFileGuard[];
}

function reserve(bytes: number): void {
  if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > freemem() / 2 || bytes > (getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 2) throw new Error("Inherited task file review waits for metadata transfer RAM.");
}

async function identity(path: string, file = true): Promise<string> {
  const value = await lstat(path, { bigint: true });
  if (value.isSymbolicLink() || (file ? !value.isFile() : !value.isDirectory())) throw new Error("Inherited task file closure cannot follow links or nonregular components.");
  return [value.dev, value.ino, value.size, value.mtimeNs, value.ctimeNs].map(String).join(":");
}

async function inside(root: string, logical: string): Promise<string> {
  if (!logical || logical.includes("\\") || logical.includes("\0") || logical.split("/").some(part => !part || part === "." || part === ".." || /[:\x00-\x1f]/.test(part))) throw new Error("Inherited file closure has an unsafe managed path.");
  await identity(root, false);
  const parts = logical.split("/"); let path = root;
  for (let index = 0; index < parts.length; index++) { path = join(path, parts[index]!); await identity(path, index === parts.length - 1); }
  return path;
}
export const inheritedTaskManagedFilePath = inside;

async function sealed(path: string, expected?: string, guards?: InheritedTaskFileGuard[], signal?: AbortSignal): Promise<{ sha256: string; bytes: number }> {
  signal?.throwIfAborted(); const before = await identity(path), info = await lstat(path);
  const digest = createHash("sha256");
  for await (const bytes of createReadStream(path, { highWaterMark: 64 * 1024, signal })) { signal?.throwIfAborted(); digest.update(bytes); }
  const sha256 = digest.digest("hex");
  if (await identity(path) !== before || expected !== undefined && sha256 !== expected) throw new Error("Inherited task source closure changed or failed its exact seal.");
  guards?.push({ path, identity: before }); return { sha256, bytes: info.size };
}

async function json(path: string): Promise<Record<string, any>> {
  const info = await lstat(path); reserve(info.size * 12 + 65536);
  return object(JSON.parse(await readFile(path, "utf8")));
}

/** Reads only closed host custody, never a model or the current decoder. */
export async function inheritedTaskFileClosure(root: string, context: InheritedTaskFileNativeContext, signal?: AbortSignal,
  sourceVerifier = new TaskAttachmentReadVerifier()): Promise<InheritedTaskFileClosure> {
  if (![context.brainId, context.sourceBrainId, context.sourceTurnId].every(id => ID.test(id)) ||
      ![context.executionId, context.descriptorSha256, context.inputSha256, context.sourceManifestSha256, context.sourceBindingId].every(hash => SHA.test(hash)) ||
      ![context.rootAttentionEpoch, context.sourceAttentionEpoch].every(epoch => Number.isSafeInteger(epoch) && epoch >= 0) ||
      context.frontier.inputSha256 !== context.inputSha256 || !ID.test(context.frontier.turnId) || !ID.test(context.frontier.checkpointId) ||
      context.frontier.phase !== "native-text-token-boundary" || context.frontier.exactSupportedResume !== true ||
      !context.regionPath.length || context.regionPath.some(id => !ID.test(id))) throw new Error("Inherited task files need an authenticated native source owner.");
  const guards: InheritedTaskFileGuard[] = [], chain: string[] = [], seals: unknown[] = [];
  let storage = root, metadata = await json(await inside(root, "engine/brain.json"));
  if (metadata.brain_id !== context.brainId || object(metadata.connected_regions).sha256 !== context.descriptorSha256) throw new Error("Inherited file root differs from the actual registered native owner.");
  for (const regionId of context.regionPath) {
    signal?.throwIfAborted(); const descriptor = object(metadata.connected_regions);
    const region = Array.isArray(descriptor.regions) ? descriptor.regions.map(object).find(row => row.regionId === regionId) : undefined;
    if (!region || !ID.test(region.parentId) || region.enginePath !== `merge/parents/${region.parentId}/engine` ||
        !SHA.test(region.metadataSha256) || !SHA.test(region.generationSha256)) throw new Error("Inherited task lost its registered parent region.");
    const indexPath = await inside(storage, "merge/parents.jsonl"), before = await identity(indexPath);
    const indexDigest = createHash("sha256"), declared = new Map<string, { sha256: string; bytes: number }>();
    const input = createReadStream(indexPath, { highWaterMark: 64 * 1024, signal }); input.on("data", bytes => indexDigest.update(bytes));
    const lines = createInterface({ input, crlfDelay: Infinity }); let header = false;
    try {
      for await (const line of lines) {
        signal?.throwIfAborted(); if (!line.trim()) continue; reserve(Buffer.byteLength(line) * 12 + 65536);
        const row = object(JSON.parse(line));
        if (!header) {
          header = true;
          if (row.format !== "omni-full-merge-parent-index" || row.formatVersion !== 1 || !Array.isArray(row.parents) ||
              !row.parents.some((parent: unknown) => { const value = object(parent); return value.brainId === region.parentId && value.generationSha256 === region.generationSha256; })) throw new Error("Inherited task parent index differs from its registered generation.");
        } else if (row.parentId === region.parentId && ["engine/brain.json", "portable-learning-workspace/inventory.sqlite3"].includes(row.path)) {
          if (declared.has(row.path) || !SHA.test(row.sha256) || !Number.isSafeInteger(row.bytes) || row.bytes < 0) throw new Error("Inherited task parent has an invalid or duplicated source seal.");
          declared.set(row.path, { sha256: row.sha256, bytes: row.bytes });
        }
      }
    } finally { lines.close(); input.destroy(); }
    if (!header || await identity(indexPath) !== before) throw new Error("Inherited task parent index changed during review.");
    guards.push({ path: indexPath, identity: before }); const indexSha256 = indexDigest.digest("hex");
    const parentStorage = join(storage, "merge", "parents", region.parentId);
    await identity(parentStorage, false);
    const metadataSeal = declared.get("engine/brain.json"), inventorySeal = declared.get("portable-learning-workspace/inventory.sqlite3");
    if (!metadataSeal || !inventorySeal || metadataSeal.sha256 !== region.metadataSha256) throw new Error("Inherited task parent snapshot has no sealed host workspace.");
    const metadataPath = await inside(parentStorage, "engine/brain.json");
    const measured = await sealed(metadataPath, metadataSeal.sha256, guards, signal);
    if (measured.bytes !== metadataSeal.bytes) throw new Error("Inherited source metadata length changed.");
    const inventoryPath = await inside(parentStorage, "portable-learning-workspace/inventory.sqlite3");
    const inventoryMeasured = await sealed(inventoryPath, inventorySeal.sha256, guards, signal);
    if (inventoryMeasured.bytes !== inventorySeal.bytes) throw new Error("Inherited host inventory length changed.");
    seals.push({ regionId, parentId: region.parentId, generationSha256: region.generationSha256, metadataSha256: measured.sha256,
      indexSha256, inventorySha256: inventoryMeasured.sha256 });
    chain.push(region.parentId); storage = parentStorage; metadata = await json(metadataPath);
    if (metadata.brain_id !== region.parentId) throw new Error("Inherited task parent metadata has another original owner.");
  }
  if (chain.at(-1) !== context.sourceBrainId || metadata.brain_id !== context.sourceBrainId) throw new Error("Inherited task host closure belongs to a different original native source.");
  const archive = join(storage, "portable-learning-workspace"), inventoryPath = await inside(storage, "portable-learning-workspace/inventory.sqlite3");
  const inventory = new DatabaseSync(inventoryPath, { readOnly: true }); let sourceDatabase: DatabaseSync | undefined;
  try {
    inventory.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-512");
    if (inventory.prepare("PRAGMA quick_check").get()?.quick_check !== "ok" || inventory.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== "omni-learning-workspace-archive-1" ||
        inventory.prepare("SELECT value FROM meta WHERE key='complete'").get()?.value !== "1" || inventory.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== context.sourceBrainId) throw new Error("Inherited task workspace inventory is not a complete original closure.");
    const registry = inventory.prepare("SELECT sha256,bytes FROM files WHERE path='attachments/registry.sqlite3'").get();
    if (!registry) return { sourceClosureSha256: inheritedTaskFileHash(seals), ownerBindingSha256: inheritedTaskFileHash(context), files: [], guards };
    if (!SHA.test(String(registry.sha256)) || !Number.isSafeInteger(registry.bytes) || Number(registry.bytes) < 1) throw new Error("Inherited task attachment registry seal is invalid.");
    const registryPath = await inside(archive, `blobs/${registry.sha256}.data`), measured = await sealed(registryPath, String(registry.sha256), guards, signal);
    if (measured.bytes !== registry.bytes) throw new Error("Inherited task attachment registry length changed.");
    sourceDatabase = new DatabaseSync(registryPath, { readOnly: true }); sourceDatabase.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-512");
    if (sourceDatabase.prepare("PRAGMA quick_check").get()?.quick_check !== "ok" ||
        sourceDatabase.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== context.sourceBrainId ||
        sourceDatabase.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== "omni-chat-task-attachments-1") throw new Error("Inherited task original attachment owner is invalid.");
    const ownCount = Number(sourceDatabase.prepare("SELECT COUNT(*) AS n FROM attachments WHERE state='bound' AND turn_id=?").get(context.sourceTurnId)?.n ?? 0);
    reserve(ownCount * 512 + 65536);
    const ownIds = sourceDatabase.prepare("SELECT id FROM attachments WHERE state='bound' AND turn_id=? ORDER BY request_id,request_index").all(context.sourceTurnId).map(row => String(row.id));
    if (sourceDatabase.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_continuation_scopes'").get()) {
      const bytes = Number(sourceDatabase.prepare("SELECT length(CAST(record_json AS BLOB)) AS bytes FROM task_continuation_scopes WHERE owner_brain_id=? AND successor_turn_id=?")
        .get(context.sourceBrainId, context.sourceTurnId)?.bytes ?? 0);
      const custody = Number(sourceDatabase.prepare(`SELECT COALESCE(SUM(length(CAST(custody_json AS BLOB))),0) AS bytes FROM task_continuation_files WHERE scope_id IN
        (SELECT scope_id FROM task_continuation_scopes WHERE owner_brain_id=? AND successor_turn_id=?)`).get(context.sourceBrainId, context.sourceTurnId)?.bytes ?? 0);
      reserve((bytes + custody) * 12 + 65536);
    }
    const scope = readTaskContinuationScope(sourceDatabase, context.sourceBrainId, context.sourceTurnId);
    if (scope && scope.record.attentionEpoch !== context.sourceAttentionEpoch) throw new Error("Inherited task file scope belongs to another source attention epoch.");
    const sourceIds = [...new Set([...ownIds, ...(scope?.aliases.map(alias => alias.attachmentId) ?? [])])]; reserve(sourceIds.length * 4096 + 65536);
    const current = new DatabaseSync(TaskAttachmentStore.databasePath(root), { readOnly: true });
    try {
      current.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-512");
      if (current.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== context.brainId) throw new Error("Inherited task custody registry has another current owner.");
      const files: InheritedTaskFilePrivate[] = [];
      for (const sourceId of sourceIds) {
        const sourceBytes = Number(sourceDatabase.prepare("SELECT length(CAST(source_json AS BLOB)) AS bytes FROM attachments WHERE id=?").get(sourceId)?.bytes ?? 0);
        reserve(sourceBytes * 24 + 65536);
        signal?.throwIfAborted(); const source = sourceDatabase.prepare("SELECT * FROM attachments WHERE id=?").get(sourceId) as unknown as PersistedTaskAttachmentRow | undefined;
        if (!source || source.state !== "bound" || !source.turn_id || !ID.test(source.turn_id) || !ID.test(source.id) || source.brain_id !== context.sourceBrainId ||
            ![source.content_sha256, source.manifest_sha256, source.source_sha256].every(hash => SHA.test(hash)) || !Number.isSafeInteger(source.bytes) || source.bytes < 1 ||
            !["file", "image", "audio", "video"].includes(source.kind) || !/^[a-z0-9][a-z0-9.+-]{0,80}\/[a-z0-9][a-z0-9.+-]{0,80}$/.test(source.mime_type) ||
            !source.display_name || source.display_name.length > 512 || ![source.created_at, source.updated_at].every(value => Number.isFinite(Date.parse(value))) || taskAttachmentManifestSha256(source) !== source.manifest_sha256 ||
            createHash("sha256").update(source.source_json).digest("hex") !== source.source_sha256) throw new Error("Inherited task original file binding changed.");
        validatedTaskAttachmentSource(JSON.parse(source.source_json));
        let attachmentId = source.id, sourceJson = source.source_json;
        for (const parentBrainId of [...chain].reverse()) {
          sourceJson = canonicalTaskAttachmentValue({ ...JSON.parse(sourceJson), inheritedFrom: { brainId: parentBrainId, attachmentId } });
          attachmentId = inheritedTaskFileHash({ parentBrainId, attachmentId });
        }
        const row = current.prepare("SELECT * FROM attachments WHERE id=?").get(attachmentId) as unknown as PersistedTaskAttachmentRow | undefined;
        if (!row || row.brain_id !== context.brainId || row.state !== "retired" || row.turn_id !== null || row.source_json !== sourceJson ||
            row.source_sha256 !== createHash("sha256").update(sourceJson).digest("hex") || taskAttachmentManifestSha256(row) !== row.manifest_sha256 ||
            ["content_sha256", "bytes", "mime_type", "kind", "display_name", "created_at"].some(key => row[key as keyof PersistedTaskAttachmentRow] !== source[key as keyof PersistedTaskAttachmentRow])) throw new Error("Destination retired custody does not match this exact original task file.");
        const blob = inventory.prepare("SELECT sha256,bytes FROM files WHERE path=?").get(`attachments/blobs/${source.content_sha256}.data`);
        if (!blob || blob.sha256 !== source.content_sha256 || blob.bytes !== source.bytes) throw new Error("Inherited task original file has no exact sealed bytes.");
        const sourcePath = await inside(archive, `blobs/${source.content_sha256}.data`), sourceIdentity = await identity(sourcePath);
        if ((await lstat(sourcePath)).size !== source.bytes) throw new Error("Inherited task original blob length differs.");
        await sourceVerifier.withVerifiedFile({ path: sourcePath, contentSha256: source.content_sha256, bytes: source.bytes,
          authorityKey: inheritedTaskFileHash({ brainId: context.brainId, sourceBrainId: context.sourceBrainId,
            registrySha256: registry.sha256, originalAttachmentId: source.id, originalManifestSha256: source.manifest_sha256 }) },
        () => assertInheritedTaskFileGuards(root, [...guards, { path: sourcePath, identity: sourceIdentity }], signal), async () => undefined, signal);
        files.push({ row, sourceGuard: { path: sourcePath, identity: sourceIdentity }, disclosure: {
          attachmentId: row.id, displayName: row.display_name, kind: row.kind, mimeType: row.mime_type, bytes: row.bytes,
          contentSha256: row.content_sha256, manifestSha256: row.manifest_sha256, sourceSha256: row.source_sha256,
          originalBinding: { brainId: source.brain_id, turnId: source.turn_id, attachmentId: source.id,
            manifestSha256: source.manifest_sha256, sourceSha256: source.source_sha256 } } });
      }
      for (const guard of guards) if (await identity(guard.path) !== guard.identity) throw new Error("Inherited source closure changed before review finished.");
      return { files, guards, ownerBindingSha256: inheritedTaskFileHash(context),
        sourceClosureSha256: inheritedTaskFileHash({ seals, registrySha256: registry.sha256,
          sources: files.map(file => ({ original: file.disclosure.originalBinding, contentSha256: file.disclosure.contentSha256, bytes: file.disclosure.bytes })) }) };
    } finally { current.close(); }
  } finally { sourceDatabase?.close(); inventory.close(); }
}

/** No native RPC: safe to call while the same decoder is awaiting this tool. */
export async function assertInheritedTaskFileGuards(root: string, guards: readonly InheritedTaskFileGuard[], signal?: AbortSignal): Promise<void> {
  for (const guard of guards) {
    signal?.throwIfAborted(); const logical = relative(root, guard.path).split(sep).join("/");
    if (await inside(root, logical) !== guard.path || await identity(guard.path) !== guard.identity) throw new Error("Inherited task source closure changed during this read.");
  }
}
