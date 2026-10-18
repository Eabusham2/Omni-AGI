/** Closed, authenticated regional execution dependencies; never another model. */
import { lstat, readFile } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { assertSafeArchivePath, streamFileSha256, type StreamingZipSource } from "./streamingZip";
import { nativeExecutionFiles } from "./executionCheckpointPortability";
import { DatabaseSync } from "node:sqlite";
import { validateConceptIdViewFile } from "./conceptIdViewFiles";

const SHA = /^[a-f0-9]{64}$/, ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/, CHECKPOINT = /^[a-f0-9]{32}$/;
const RUNTIME_FILES = ["brain.json", "core.safetensors", "plasticity.safetensors", "conversation.sqlite3", "state/replay.sqlite3", "state/working-memory.sqlite3"] as const;
const object = (value: unknown): Record<string, any> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid connected execution declaration.");
  return value as Record<string, any>;
};
function safe(relative: unknown): string {
  if (typeof relative !== "string") throw new Error("Connected execution path is not text.");
  assertSafeArchivePath(relative); return relative;
}
async function regular(path: string): Promise<void> {
  const value = await lstat(path);
  if (!value.isFile() || value.isSymbolicLink()) throw new Error("Connected execution component is not a regular owned file.");
}
async function inside(root: string, relative: string): Promise<string> {
  let path = root;
  const base = await lstat(root);
  if (!base.isDirectory() || base.isSymbolicLink()) throw new Error("Connected execution owner directory is unsafe.");
  for (const part of safe(relative).split("/")) {
    path = join(path, part);
    const value = await lstat(path);
    if (value.isSymbolicLink()) throw new Error("Connected execution cannot follow a filesystem link.");
  }
  return path;
}
async function fileIfPresent(root: string, relative: string): Promise<string | undefined> {
  try { return await inside(root, relative); } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; }
}
async function json(path: string): Promise<Record<string, any>> {
  await regular(path); const bytes = (await lstat(path)).size;
  if (bytes * 12 + 65536 > freemem() / 2 || bytes * 12 + 65536 > (getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 2) {
    throw new Error("Inherited execution metadata awaits enough bounded transfer RAM.");
  }
  return object(JSON.parse(await readFile(path, "utf8")));
}

export function connectedExecutionEntries(metadata: unknown): Record<string, any>[] {
  const state = object(metadata).connected_executions;
  if (state === null || state === undefined) return [];
  const value = object(state);
  if (value.format !== "omni-connected-execution-queue" || value.formatVersion !== 1 || !Array.isArray(value.entries)) {
    throw new Error("Unsupported connected execution queue.");
  }
  if (Object.keys(value).sort().join(",") !== "entries,format,formatVersion") throw new Error("Connected execution queue has unknown fields.");
  const entries = value.entries.map(object), seen = new Set<string>(), turns = new Set<string>();
  const required = ["id", "rootTurnId", "parentId", "regionPath", "sourceEnginePath", "sourceCheckpointId", "sourceManifestSha256", "sourceBindingId",
    "originalTurnId", "historyParentId", "historyHumanMessageId", "rootHumanMessageId", "status", "backend", "lineage"];
  const allowed = new Set([...required, "completion", "resumeEnginePath", "resumeCheckpointId", "resumeManifestSha256", "minimumContextTokens", "publishedText", "workerStateSha256", "runtimeFiles"]);
  for (const entry of entries) {
    if (required.some(key => !Object.hasOwn(entry, key)) || Object.keys(entry).some(key => !allowed.has(key)) ||
        !SHA.test(entry.id) || seen.has(entry.id) || turns.has(entry.rootTurnId) || !ID.test(entry.rootTurnId) || !ID.test(entry.parentId) || !ID.test(entry.originalTurnId) ||
        !CHECKPOINT.test(entry.sourceCheckpointId) || !SHA.test(entry.sourceManifestSha256) || !SHA.test(entry.sourceBindingId) ||
        !["queued", "suspended", "complete", "stopped"].includes(entry.status) || !Array.isArray(entry.regionPath) ||
        !entry.regionPath.length || entry.regionPath.some((id: unknown) => typeof id !== "string" || !ID.test(id)) || !Array.isArray(entry.lineage) ||
        ["historyParentId", "historyHumanMessageId", "rootHumanMessageId", "backend"].some(key => typeof entry[key] !== "string" || !entry[key])) {
      throw new Error("Connected execution has an invalid owner/frontier binding.");
    }
    safe(entry.sourceEnginePath);
    if (entry.resumeEnginePath !== undefined) {
      safe(entry.resumeEnginePath);
      if (!CHECKPOINT.test(entry.resumeCheckpointId) || !SHA.test(entry.resumeManifestSha256)) throw new Error("Incomplete inherited re-suspension binding.");
    }
    if (["resumeCheckpointId", "resumeManifestSha256"].some(key => Object.hasOwn(entry, key)) && entry.resumeEnginePath === undefined) throw new Error("Inherited runtime owner is absent.");
    if (entry.minimumContextTokens !== undefined && (!Number.isSafeInteger(entry.minimumContextTokens) || entry.minimumContextTokens < 0) ||
        entry.publishedText !== undefined && typeof entry.publishedText !== "string" ||
        entry.workerStateSha256 !== undefined && !SHA.test(entry.workerStateSha256)) throw new Error("Invalid inherited activity summary.");
    if (entry.runtimeFiles !== undefined) {
      const files = object(entry.runtimeFiles);
      if (Object.keys(files).sort().join(",") !== [...RUNTIME_FILES].sort().join(",") || Object.values(files).some(raw => {
        const item = object(raw); return Object.keys(item).sort().join(",") !== "bytes,sha256" || !Number.isSafeInteger(item.bytes) || item.bytes < 0 || !SHA.test(item.sha256);
      })) throw new Error("Inherited runtime activity seals are incomplete.");
    }
    if (entry.status === "complete" || entry.completion !== undefined) {
      const completion = object(entry.completion);
      safe(completion.resultPath);
      if (Object.keys(completion).sort().join(",") !== "resultPath,sha256,turnId" || !SHA.test(completion.sha256) || !ID.test(completion.turnId) ||
          completion.turnId !== entry.rootTurnId && !entry.lineage.length) throw new Error("Inherited result receipt has another owner.");
    }
    seen.add(entry.id); turns.add(entry.rootTurnId);
  }
  return entries;
}

async function registeredParentEngine(engine: string, regionId: string): Promise<string> {
  const metadata = await json(join(engine, "brain.json")), descriptor = object(metadata.connected_regions);
  const region = Array.isArray(descriptor.regions) ? descriptor.regions.map(object).find(value => value.regionId === regionId) : undefined;
  if (!region || !ID.test(region.parentId) || region.enginePath !== `merge/parents/${region.parentId}/engine`) throw new Error("Inherited region lost its immutable parent address.");
  for (const root of [dirname(engine), join(engine, "connected-inputs")]) {
    const found = await fileIfPresent(root, `${region.enginePath}/brain.json`);
    if (found) return dirname(found);
  }
  throw new Error("Inherited region parent closure is missing.");
}

async function resolveEngine(engine: string, entry: Record<string, any>, resumed: boolean): Promise<string> {
  const logical = resumed ? entry.resumeEnginePath : entry.sourceEnginePath;
  if (logical === undefined) return resolveEngine(engine, entry, false);
  for (const root of [dirname(engine), join(engine, "connected-execution-inputs")]) {
    // The fallback directory may not exist on a live (not yet snapshotted) owner.
    const found = await fileIfPresent(root, `${safe(logical)}/brain.json`);
    if (found) return dirname(found);
  }
  if (resumed && entry.lineage.length) {
    const parent = await registeredParentEngine(engine, entry.regionPath[0]), priorId = object(entry.lineage.at(-1)).executionId;
    const prior = connectedExecutionEntries(await json(join(parent, "brain.json"))).find(value => value.id === priorId);
    if (!prior) throw new Error("Nested inherited private runtime lost its queue provenance.");
    return resolveEngine(parent, prior, true);
  }
  if (resumed) throw new Error("Inherited private runtime closure is missing.");
  let current = engine;
  for (const id of entry.regionPath) current = await registeredParentEngine(current, id);
  return current;
}

async function validateNativeOwner(engine: string, entry: Record<string, any>, resumed: boolean): Promise<Map<string, string>> {
  const execution = await nativeExecutionFiles(engine, entry.parentId), checkpoint = resumed ? entry.resumeCheckpointId : entry.sourceCheckpointId;
  const manifest = execution.manifests.find(value => value.name === `.execution/${checkpoint}/m-${resumed ? entry.resumeManifestSha256 : entry.sourceManifestSha256}.json`);
  if (!manifest || manifest.value.bindingId !== entry.sourceBindingId || manifest.value.frontier.turnId !== entry.originalTurnId ||
      execution.active?.checkpointId !== checkpoint) throw new Error("Inherited source does not match its exact adopted native frontier.");
  return execution.files;
}

/** Returns engine-local aliases, keeping original regional identities intact. */
export async function collectConnectedExecutionInputs(engine: string, metadata: unknown): Promise<Map<string, string>> {
  const files = new Map<string, string>();
  for (const entry of connectedExecutionEntries(metadata)) {
    for (const resumed of entry.resumeEnginePath === undefined ? [false] : [false, true]) {
      const logical = resumed ? entry.resumeEnginePath : entry.sourceEnginePath;
      const runtime = await resolveEngine(engine, entry, resumed), prefix = `connected-execution-inputs/${logical}/`;
      const activity = await validateNativeOwner(runtime, entry, resumed);
      for (const name of RUNTIME_FILES) {
        const path = await inside(runtime, name); await regular(path);
        if (resumed && entry.runtimeFiles && ((await lstat(path)).size !== entry.runtimeFiles[name].bytes ||
            await streamFileSha256(path) !== entry.runtimeFiles[name].sha256)) throw new Error("Inherited runtime activity changed after its suspension seal.");
        for (const suffix of name.endsWith(".sqlite3") ? ["-wal", "-journal"] : []) {
          const sidecar = await fileIfPresent(runtime, name + suffix);
          if (sidecar && (await lstat(sidecar)).size > 0) throw new Error("Inherited runtime SQLite has uncommitted writes.");
        }
        files.set(prefix + name, path);
      }
      for (const [relative, path] of activity) files.set(prefix + relative, path);
      // Only declared structural ID views are transferable. Every descriptor
      // remains hash bound; source neurons are not retargeted to root memory.
      const state = await json(join(runtime, "brain.json"));
      const manifest = await nativeExecutionFiles(runtime, entry.parentId);
      const pending = manifest.manifests.find(value => value.name.endsWith(`m-${resumed ? entry.resumeManifestSha256 : entry.sourceManifestSha256}.json`));
      const visit = async (value: unknown): Promise<void> => {
        if (Array.isArray(value)) { for (const child of value) await visit(child); return; }
        if (!value || typeof value !== "object") return;
        const item = value as Record<string, unknown>;
        if (item.format === "omni-structural-concept-id-view" && typeof item.path === "string" && SHA.test(String(item.sha256))) {
          const path = await inside(runtime, safe(item.path)); await regular(path);
          if (await streamFileSha256(path) !== item.sha256) throw new Error("Inherited structural view changed checksum.");
          await validateConceptIdViewFile(path, basename(path));
          files.set(prefix + item.path, path); return;
        }
        for (const child of Object.values(item)) await visit(child);
      };
      await visit(state); await visit(pending?.value);
    }
    if (entry.completion) {
      const relative = safe(entry.completion.resultPath);
      const path = await fileIfPresent(dirname(engine), relative) ?? await fileIfPresent(join(engine, "connected-execution-inputs"), relative);
      if (!path || await streamFileSha256(path) !== entry.completion.sha256 || (await json(path)).turnId !== entry.completion.turnId) {
        throw new Error("Inherited result lost its exact durable root receipt.");
      }
      files.set(`connected-execution-inputs/${relative}`, path);
    }
  }
  return files;
}

/** Validate embedded private owners without unpacking or loading neural state. */
export async function validateConnectedExecutionArchive(metadata: unknown, engine: string,
  lookup: (relative: string) => StreamingZipSource | undefined): Promise<void> {
  const physical = (relative: string): string => {
    const source = lookup(relative); if (!source?.sourcePath) throw new Error(`Inherited private runtime is missing: ${relative}`); return source.sourcePath;
  };
  for (const entry of connectedExecutionEntries(metadata)) {
    for (const resumed of entry.resumeEnginePath === undefined ? [false] : [false, true]) {
      const logical = resumed ? entry.resumeEnginePath : entry.sourceEnginePath;
      const checkpoint = resumed ? entry.resumeCheckpointId : entry.sourceCheckpointId;
      const digest = resumed ? entry.resumeManifestSha256 : entry.sourceManifestSha256;
      const prefix = `${engine}/connected-execution-inputs/${logical}/`, inventory = new Map<string, string>();
      const pointer = await json(physical(prefix + ".execution/active.json"));
      const nativeManifest = await json(physical(`${prefix}.execution/${pointer.checkpointId}/m-${pointer.manifestSha256}.json`));
      // This archive contains the exact closed checkpoint inventory. The
      // common native validator checks every header, seal, receipt and owner.
      for (const name of [".execution/active.json", `.execution/${pointer.checkpointId}/m-${pointer.manifestSha256}.json`,
        ...Object.keys(object(nativeManifest.files)).map(name => `.execution/${pointer.checkpointId}/${safe(name)}`),
        ...["manifest.json", "controls.safetensors", "changed.sqlite"].map(name => `.execution/${pointer.checkpointId}/diagnostic/${name}`)]) {
        inventory.set(name, physical(prefix + name));
      }
      const unloaded = lookup(prefix + ".execution/unloaded.json"); if (unloaded?.sourcePath) inventory.set(".execution/unloaded.json", unloaded.sourcePath);
      const checked = await nativeExecutionFiles("archive-native-owner", entry.parentId, { executionFiles: inventory,
        metadataPath: physical(prefix + "brain.json"), neuralPaths: { "core.safetensors": physical(prefix + "core.safetensors"),
          "plasticity.safetensors": physical(prefix + "plasticity.safetensors") } });
      if (checked.active?.checkpointId !== checkpoint || checked.active?.manifestSha256 !== digest ||
          nativeManifest.bindingId !== entry.sourceBindingId) throw new Error("Archived inherited private frontier differs from its root queue.");
      if (resumed && entry.runtimeFiles) for (const name of RUNTIME_FILES) {
        const path = physical(prefix + name), seal = entry.runtimeFiles[name];
        if ((await lstat(path)).size !== seal.bytes || await streamFileSha256(path) !== seal.sha256) throw new Error("Archived inherited runtime activity seal changed.");
      }
      const validateViews = async (value: unknown): Promise<void> => {
        if (Array.isArray(value)) { for (const child of value) await validateViews(child); return; }
        if (!value || typeof value !== "object") return;
        const item = value as Record<string, unknown>;
        if (item.format === "omni-structural-concept-id-view") {
          const path = physical(prefix + safe(item.path));
          if (item.brainId !== entry.parentId || await streamFileSha256(path) !== item.sha256) throw new Error("Archived inherited argument view has another native owner.");
          await validateConceptIdViewFile(path, basename(String(item.path))); return;
        }
        for (const child of Object.values(item)) await validateViews(child);
      };
      await validateViews(nativeManifest.worker); await validateViews(nativeManifest.payload);
      for (const name of ["conversation.sqlite3", "state/replay.sqlite3", "state/working-memory.sqlite3"]) {
        const db = new DatabaseSync(physical(prefix + name), { readOnly: true });
        try { if (db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Inherited runtime SQLite failed integrity verification."); }
        finally { db.close(); }
      }
    }
    if (entry.completion) {
      const path = physical(`${engine}/connected-execution-inputs/${safe(entry.completion.resultPath)}`);
      if (await streamFileSha256(path) !== entry.completion.sha256 || (await json(path)).turnId !== entry.completion.turnId) throw new Error("Archived inherited result changed identity.");
    }
  }
}
