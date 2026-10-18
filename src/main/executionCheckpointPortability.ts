/** Authenticated immutable native token-frontier portability.
 * This copies supported activity, not a process/permission clone. Only the
 * mutable identity is rekeyed; neural bytes and completed effect identities
 * remain exact. A merged decoder cannot consume a parent cursor unchanged.
 */
import { createHash, randomUUID } from "node:crypto";
import { DatabaseSync } from "node:sqlite";
import { lstat, mkdir, readFile, readdir, rename, rm } from "node:fs/promises";
import { join } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { copyMutableFileIsolated, writeMutableFileIsolated } from "./mutableFileIsolation";
import { streamFileSha256 } from "./streamingZip";
import { rewriteNativeJsonValues } from "./nativeJsonValues";
import { nativeCheckpointMemory } from "./nativeCheckpointMemory";

const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const CHECKPOINT = /^[a-f0-9]{32}$/;
const SHA = /^[a-f0-9]{64}$/;
const hash = (text: string | Uint8Array): string => createHash("sha256").update(text).digest("hex");
const canonical = (value: unknown): string => Array.isArray(value) ? `[${value.map(canonical).join(",")}]` :
  value !== null && typeof value === "object" ? `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}` : JSON.stringify(value);
const record = (value: unknown): Record<string, any> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Native execution declaration must be an object.");
  return value as Record<string, any>;
};
async function exists(path: string): Promise<boolean> {
  try { await lstat(path); return true; } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; }
}
async function plainFile(path: string): Promise<void> {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Native execution component is not a regular private file.");
}
async function jsonFile(path: string): Promise<{ text: string; value: Record<string, any> }> {
  await plainFile(path);
  const bytes = (await lstat(path)).size;
  const reserve = Math.min(64 * 1024 * 1024, Math.floor(freemem() / 16));
  if (bytes * 12 + reserve > freemem() || bytes * 12 + reserve > getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) {
    throw new Error("Native execution metadata awaits bounded physical transfer RAM.");
  }
  const text = await readFile(path, "utf8");
  return { text, value: record(JSON.parse(text)) };
}
export function safeNativeExecutionPath(relative: string): boolean {
  return /^\.execution\/(?:active|unloaded)\.json$/.test(relative) ||
    /^\.execution\/[a-f0-9]{32}\/(?:m-[a-f0-9]{64}\.json|t-\d+\.safetensors|recall-\d+\.sqlite3|diagnostic\/(?:manifest\.json|controls\.safetensors|changed\.sqlite))$/.test(relative);
}
function snapshotFile(relative: string): boolean {
  return /^t-\d+\.safetensors$/.test(relative) || /^recall-\d+\.sqlite3$/.test(relative);
}
async function checkedPayload(path: string, descriptor: unknown, logicalName = path): Promise<void> {
  const value = record(descriptor); await plainFile(path);
  if (!SHA.test(String(value.sha256)) || !Number.isSafeInteger(value.bytes) || value.bytes < 0 ||
      (await lstat(path)).size !== value.bytes || await streamFileSha256(path) !== value.sha256) throw new Error("Native execution payload checksum/size differs.");
  if (logicalName.endsWith(".safetensors")) {
    await nativeCheckpointMemory([path], (tensors) => {
      if (!Object.hasOwn(value, "dtype")) return; // Neural/journal files own several named tensors.
      const activity = tensors.get("activity"), dtype = typeof value.dtype === "string" ? value.dtype.replace(/^torch\./, "") : "";
      const formats: Record<string, readonly [string, number]> = { bool: ["BOOL", 1], uint8: ["U8", 1], int8: ["I8", 1],
        int16: ["I16", 2], int32: ["I32", 4], int64: ["I64", 8], float16: ["F16", 2], bfloat16: ["BF16", 2],
        float32: ["F32", 4], float64: ["F64", 8] };
      const format = formats[dtype];
      if (!activity || tensors.size !== 1 || !format || activity.dtype !== format[0] ||
          canonical(activity.shape) !== canonical(value.shape) ||
          activity.bytes !== activity.shape.reduce((count, axis) => count * axis, 1) * format[1]) {
        throw new Error("Native execution activity safetensor geometry differs from its authenticated descriptor.");
      }
    });
  }
  if (/\.sqlite3?$/.test(logicalName)) {
    const database = new DatabaseSync(path, { readOnly: true });
    try {
      database.exec("PRAGMA query_only=ON");
      if (database.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Native execution SQLite integrity failed.");
    } finally { database.close(); }
    for (const suffix of ["-wal", "-shm", "-journal"]) if (await exists(path + suffix) && (await lstat(path + suffix)).size > 0) {
      throw new Error("Native execution immutable SQLite has a live sidecar.");
    }
  }
}

interface CheckedManifest { name: string; text: string; value: Record<string, any> }
export interface NativeExecutionArchive {
  files: Map<string, string>;
  manifests: CheckedManifest[];
  active?: { checkpointId: string; manifestSha256: string; brainId: string };
}
export interface NativeExecutionSourceOptions {
  metadataPath?: string;
  neuralPaths?: Partial<Record<"core.safetensors" | "plasticity.safetensors", string>>;
  /** An already hash-checked archive inventory, not a filesystem authority. */
  executionFiles?: ReadonlyMap<string, string>;
}

function validateExecutionTree(value: unknown, files: Record<string, unknown>): void {
  if (value === null || ["string", "boolean"].includes(typeof value)) return;
  if (typeof value === "number") { if (!Number.isFinite(value)) throw new Error("Execution activity metadata is not finite."); return; }
  const item = record(value);
  if (item.kind === "tensor" || item.kind === "recall-view") {
    if (typeof item.file !== "string" || !snapshotFile(item.file) || !Object.hasOwn(files, item.file)) throw new Error("Execution tree references an undeclared payload.");
    if (item.kind === "recall-view" && (!Number.isSafeInteger(item.count) || item.count < 0)) throw new Error("Execution recall count is invalid.");
    return;
  }
  if (item.kind === "map") {
    if (!Array.isArray(item.entries)) throw new Error("Execution map has no entry list.");
    const keys = new Set<string>();
    for (const pair of item.entries) {
      if (!Array.isArray(pair) || pair.length !== 2 || typeof pair[0] !== "string" || keys.has(pair[0])) throw new Error("Execution map has invalid or duplicate keys.");
      keys.add(pair[0]); validateExecutionTree(pair[1], files);
    }
    return;
  }
  if (["list", "tuple", "set"].includes(item.kind)) {
    if (!Array.isArray(item.items)) throw new Error("Execution collection is invalid.");
    for (const nested of item.items) validateExecutionTree(nested, files);
    return;
  }
  if (["paged-sequence", "saved-activity"].includes(item.kind)) {
    if (!Array.isArray(item.pages) || !Array.isArray(item.shape) || !item.pages.length ||
        item.shape.some((axis: unknown) => !Number.isSafeInteger(axis) || (axis as number) < 0)) throw new Error("Execution page geometry is invalid.");
    for (const nested of item.pages) validateExecutionTree(nested, files);
    return;
  }
  if (!["generation", "decoder-cache", "connected-cache", "attention-cache", "workspace-cache"].includes(item.kind)) throw new Error("Execution payload includes an unsupported state type.");
  validateExecutionTree(item.fields, files);
}
function executionMember(value: unknown, name: string): unknown {
  const item = record(value);
  if (item.kind !== "map" || !Array.isArray(item.entries)) throw new Error("Execution object has no closed map structure.");
  return item.entries.find(pair => Array.isArray(pair) && pair[0] === name)?.[1];
}

function rekeyObservationInbox(value: unknown, sourceId: string, targetId: string): { inbox: Record<string, any>; identities: Map<string, string> } {
  const inbox = structuredClone(record(value));
  if (inbox.format !== "omni-chat-observation-frontier" || inbox.version !== 2 || inbox.brainId !== sourceId ||
      !ID.test(String(inbox.turnId)) || !Array.isArray(inbox.seen) || !Array.isArray(inbox.pending) || !inbox.receipts ||
      canonical(Object.keys(inbox.receipts).sort()) !== canonical([...inbox.seen].sort())) throw new Error("Native execution lacks rebindable consumed-effect receipts.");
  const identities = new Map<string, string>();
  const receipts: Record<string, unknown> = {};
  for (const [oldId, raw] of Object.entries(record(inbox.receipts))) {
    const receipt = record(raw);
    const names = ["format", "brainId", "turnId", "neuralActionId", "actionEventId", "executionId", "toolId", "action", "completedAt", "payloadSha256"];
    if (receipt.brainId !== sourceId || receipt.turnId !== inbox.turnId || receipt.observationId !== oldId ||
        hash(names.map(key => String(receipt[key])).join("\0")) !== oldId) throw new Error("Consumed native evidence lost its exact original receipt.");
    receipt.brainId = targetId;
    const next = hash(names.map(key => String(receipt[key])).join("\0"));
    receipt.observationId = next; identities.set(oldId, next); receipts[next] = receipt;
  }
  for (const raw of inbox.pending) {
    const pending = record(raw), next = identities.get(pending.observationId);
    if (!next || pending.brainId !== sourceId || hash(String(pending.payloadJson)) !== pending.payloadSha256) throw new Error("Pending native evidence lost its full original payload.");
    pending.brainId = targetId; pending.observationId = next;
  }
  inbox.brainId = targetId; inbox.seen = inbox.seen.map((id: string) => identities.get(id)!); inbox.receipts = receipts;
  return { inbox, identities };
}

function encodedObservationEdits(value: unknown, path: string[], identities: ReadonlyMap<string, string>, edits: Map<string, string>): void {
  if (Array.isArray(value)) { value.forEach((item, index) => encodedObservationEdits(item, [...path, String(index)], identities, edits)); return; }
  if (!value || typeof value !== "object") return;
  const item = value as Record<string, unknown>;
  if (item.kind === "map" && Array.isArray(item.entries)) {
    item.entries.forEach((entry, index) => {
      if (Array.isArray(entry) && entry[0] === "observationId" && typeof entry[1] === "string" && identities.has(entry[1])) {
        edits.set(JSON.stringify([...path, "entries", String(index), "1"]), JSON.stringify(identities.get(entry[1])));
      } else if (Array.isArray(entry)) encodedObservationEdits(entry[1], [...path, "entries", String(index), "1"], identities, edits);
    });
    return;
  }
  for (const [key, nested] of Object.entries(item)) encodedObservationEdits(nested, [...path, key], identities, edits);
}

/** All committed historical frontiers are retained; uncommitted temp files are not. */
export async function nativeExecutionFiles(engine: string, brainId: string, options: NativeExecutionSourceOptions = {}): Promise<NativeExecutionArchive> {
  if (!ID.test(brainId)) throw new Error("Invalid native execution brain identity.");
  const root = join(engine, ".execution"), result: NativeExecutionArchive = { files: new Map(), manifests: [] };
  const archiveFiles = options.executionFiles;
  const filePath = (relative: string): string => {
    if (!archiveFiles) return join(engine, relative);
    const path = archiveFiles.get(relative); if (!path) throw new Error("Native execution archive is missing a declared component."); return path;
  };
  const list = async (relative: string): Promise<Array<{ name: string; isDirectory(): boolean; isSymbolicLink(): boolean }>> => {
    if (!archiveFiles) return readdir(join(engine, relative), { withFileTypes: true });
    const entries = new Map<string, boolean>();
    for (const name of archiveFiles.keys()) if (name.startsWith(relative + "/")) {
      const remainder = name.slice(relative.length + 1), split = remainder.indexOf("/");
      const basename = split < 0 ? remainder : remainder.slice(0, split), directory = split >= 0;
      if (entries.has(basename) && entries.get(basename) !== directory) throw new Error("Native archive path collides with a directory.");
      entries.set(basename, directory);
    }
    return [...entries].map(([name, directory]) => ({ name, isDirectory: () => directory, isSymbolicLink: () => false }));
  };
  if (archiveFiles) {
    for (const name of archiveFiles.keys()) if (!safeNativeExecutionPath(name)) throw new Error("Unsafe native execution archive component.");
    if (!archiveFiles.size) return result;
  } else {
    if (!await exists(root)) return result;
    const info = await lstat(root);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Native execution root is unsafe.");
  }
  for (const entry of await list(".execution")) {
    if (entry.name.startsWith(".")) continue;
    if (entry.isSymbolicLink()) throw new Error("Native execution contains a filesystem link.");
    if (["active.json", "unloaded.json"].includes(entry.name)) {
      const item = await jsonFile(filePath(`.execution/${entry.name}`));
      if (item.value.brainId !== brainId) throw new Error("Native execution pointer has a different brain owner.");
      if (entry.name === "active.json") {
        if (!CHECKPOINT.test(String(item.value.checkpointId)) || !SHA.test(String(item.value.manifestSha256)) ||
            Object.keys(item.value).sort().join(",") !== "brainId,checkpointId,manifestSha256") throw new Error("Invalid native execution pointer.");
        result.active = item.value as NativeExecutionArchive["active"];
      } else if (item.value.explicitLoadRequired !== true || Object.keys(item.value).sort().join(",") !== "brainId,explicitLoadRequired") {
        throw new Error("Invalid native execution unloaded marker.");
      }
      result.files.set(`.execution/${entry.name}`, filePath(`.execution/${entry.name}`)); continue;
    }
    if (!entry.isDirectory() || !CHECKPOINT.test(entry.name)) throw new Error("Unknown native execution component.");
    const directory = `.execution/${entry.name}`, expected = new Set<string>();
    for (const component of await list(directory)) {
      if (component.name.startsWith(".")) continue;
      if (component.isSymbolicLink()) throw new Error("Native execution checkpoint contains a link.");
      if (!/^m-[a-f0-9]{64}\.json$/.test(component.name)) continue;
      const item = await jsonFile(filePath(`${directory}/${component.name}`)), value = item.value;
      if (hash(item.text) !== component.name.slice(2, -5) || value.format !== "omni-native-execution-checkpoint" || value.version !== 2 ||
          value.brainId !== brainId || value.checkpointId !== entry.name || !["suspended", "completed", "stopped"].includes(value.status) ||
          !SHA.test(String(value.bindingId)) || !value.binding || hash(canonical(value.binding)) !== value.bindingId ||
          value.binding.brainId !== brainId || value.binding.turnId !== value.frontier?.turnId ||
          value.frontier?.checkpointId !== entry.name || value.frontier.exactSupportedResume !== true ||
          value.frontier.phase !== "native-text-token-boundary" || !ID.test(String(value.frontier.turnId)) ||
          !SHA.test(String(value.frontier.inputSha256)) || !Number.isSafeInteger(value.minimumContextTokens) || value.minimumContextTokens < 0 ||
          !value.worker || value.worker.brainId !== brainId || value.worker.turnId !== value.frontier.turnId ||
          value.worker.lastStreamSequence !== value.frontier.lastStreamSequence || value.presentation?.publishedText !== value.worker.publishedText ||
          hash(String(value.presentation?.input)) !== value.frontier.inputSha256 ||
          canonical(value.presentation?.actions) !== canonical(value.worker.actions)) throw new Error("Native execution manifest lost its checkpoint/presentation binding.");
      for (const [name, descriptor] of Object.entries(record(value.files))) {
        if (!snapshotFile(name)) throw new Error("Unsafe native execution activity path.");
        await checkedPayload(filePath(`${directory}/${name}`), descriptor, name); expected.add(name);
        result.files.set(`${directory}/${name}`, filePath(`${directory}/${name}`));
      }
      validateExecutionTree(value.payload, value.files);
      const rng = executionMember(value.payload, "globalRng"), cpuRng = record(executionMember(rng, "cpu"));
      const cpuDescriptor = record(value.files[cpuRng.file]);
      if (executionMember(rng, "device") !== value.backend || cpuRng.kind !== "tensor" || cpuDescriptor.device !== "cpu" ||
          cpuDescriptor.dtype !== "torch.uint8" || !Array.isArray(cpuDescriptor.shape) || cpuDescriptor.shape.length !== 1) {
        throw new Error("Execution snapshot lacks its exact global RNG frontier.");
      }
      if (value.runtimeCompatibility !== undefined) {
        const seal = record(value.runtimeCompatibility), declaration = record(seal.declaration);
        if (Object.keys(seal).sort().join(",") !== "declaration,sha256" || declaration.format !== "omni-exact-runtime-compatibility" ||
            declaration.version !== 1 || declaration.backend !== value.backend || !SHA.test(String(seal.sha256)) || hash(canonical(declaration)) !== seal.sha256) {
          throw new Error("Execution runtime/kernel compatibility seal is invalid.");
        }
        const clock = executionMember(value.payload, "activityClock"), elapsed = executionMember(clock, "elapsed");
        if (executionMember(clock, "format") !== "omni-native-activity-clock" || executionMember(clock, "version") !== 1 ||
            executionMember(clock, "paused") !== true || typeof elapsed !== "number" ||
            !Number.isFinite(elapsed) || elapsed < 0) {
          throw new Error("Execution snapshot lacks its sealed paused activity clock.");
        }
      }
      const diagnostic = await jsonFile(filePath(`${directory}/diagnostic/manifest.json`));
      if (hash(diagnostic.text) !== value.journal?.manifestSha256 || diagnostic.value.format !== "omni-core-delta-resume-1" ||
          diagnostic.value.bindingId !== value.bindingId || value.journal.bindingId !== value.bindingId ||
          Object.keys(record(diagnostic.value.files)).sort().join(",") !== "changed.sqlite,controls.safetensors") throw new Error("Native execution diagnostic baseline binding changed.");
      for (const [name, descriptor] of Object.entries(diagnostic.value.files)) {
        await checkedPayload(filePath(`${directory}/diagnostic/${name}`), descriptor, name);
        result.files.set(`${directory}/diagnostic/${name}`, filePath(`${directory}/diagnostic/${name}`));
      }
      result.files.set(`${directory}/diagnostic/manifest.json`, filePath(`${directory}/diagnostic/manifest.json`));
      const relative = `.execution/${entry.name}/${component.name}`;
      result.manifests.push({ name: relative, ...item }); result.files.set(relative, filePath(`${directory}/${component.name}`));
      expected.add(component.name); expected.add("diagnostic");
    }
    if (![...expected].some(name => name.startsWith("m-"))) throw new Error("Native execution directory has no committed manifest.");
    for (const component of await list(directory)) {
      if (!component.name.startsWith(".") && !expected.has(component.name)) throw new Error("Native execution contains an unmanifested payload.");
    }
  }
  if (archiveFiles && (archiveFiles.size !== result.files.size || [...archiveFiles.keys()].some(name => !result.files.has(name)))) {
    throw new Error("Native execution archive includes an unmanifested component.");
  }
  if (result.active && !result.files.has(`.execution/${result.active.checkpointId}/m-${result.active.manifestSha256}.json`)) throw new Error("Native execution active manifest is missing.");
  const active = result.manifests.find(item => item.name === `.execution/${result.active?.checkpointId}/m-${result.active?.manifestSha256}.json`);
  if (active?.value.status === "suspended") {
    const metadata = (await jsonFile(options.metadataPath ?? join(engine, "brain.json"))).value;
    const binding = active.value.binding;
    if (metadata.brain_id !== brainId || canonical(metadata.mutable_state) !== canonical(binding.mutable) ||
        canonical(metadata.substrate?.persistence) !== canonical(binding.substrate) || (metadata.fresh_attention_boundary?.epoch ?? 0) !== binding.attentionEpoch) {
      throw new Error("Suspended execution does not describe the selected neural generation.");
    }
    if (active.value.runtimeCompatibility) {
      const clock = executionMember(active.value.payload, "activityClock"), state = record(metadata.activity_clock);
      if (Object.keys(state).sort().join(",") !== "elapsed,format,paused,version" ||
          ["format", "version", "elapsed", "paused"].some(name => state[name] !== executionMember(clock, name))) {
        throw new Error("Suspended execution does not describe the selected paused activity clock.");
      }
    }
    for (const [name, descriptor] of Object.entries(record(active.value.neuralFiles))) {
      if (!["core.safetensors", "plasticity.safetensors"].includes(name)) throw new Error("Unsafe execution neural checkpoint file.");
      await checkedPayload(options.neuralPaths?.[name as "core.safetensors" | "plasticity.safetensors"] ?? join(engine, name), descriptor, name);
    }
    if (Object.keys(active.value.neuralFiles).sort().join(",") !== "core.safetensors,plasticity.safetensors") throw new Error("Suspended execution neural file seals are incomplete.");
  }
  return result;
}

/** Identity-safe snapshot. The caller first copies the corresponding neural generation. */
export async function snapshotNativeExecution(sourceEngine: string, targetEngine: string,
  options: NativeExecutionSourceOptions & { sourceBrainId: string; targetBrainId?: string; checkDisk?: (bytes: number) => Promise<unknown>; signal?: AbortSignal;
    /** Private recovery staging may commit metadata only after copying its selected components. */
    targetMetadataPath?: string;
    /** Recovery staging must not keep a later checkpoint when the selected snapshot has none. */
    clearIfAbsent?: boolean }): Promise<Map<string, string>> {
  const targetId = options.targetBrainId ?? options.sourceBrainId;
  if (!ID.test(targetId)) throw new Error("Invalid native execution destination identity.");
  const source = await nativeExecutionFiles(sourceEngine, options.sourceBrainId, options);
  if (!source.files.size) {
    if (options.clearIfAbsent && await exists(join(targetEngine, ".execution"))) {
      const old = await lstat(join(targetEngine, ".execution"));
      if (!old.isDirectory() || old.isSymbolicLink()) throw new Error("Recovery execution staging is unsafe.");
      await rm(join(targetEngine, ".execution"), { recursive: true, force: true });
    }
    return new Map();
  }
  const targetRoot = join(targetEngine, ".execution"), staged = join(targetEngine, `.execution-copy-${randomUUID()}`), backup = join(targetEngine, `.execution-before-${randomUUID()}`);
  await mkdir(staged, { recursive: false, mode: 0o700 });
  let moved = false, published = false;
  try {
    for (const [relative, path] of source.files) {
      options.signal?.throwIfAborted(); const bytes = (await lstat(path)).size;
      await options.checkDisk?.(bytes);
      await copyMutableFileIsolated(path, join(staged, relative.slice(".execution/".length)), { expectedSha256: await streamFileSha256(path) });
    }
    if (targetId !== options.sourceBrainId) {
      const newManifestIds = new Map<string, string>();
      const rewrittenDiagnostics = new Map<string, string>();
      for (const original of source.manifests) {
        const current = original.value, checkpointId = current.checkpointId as string;
        const binding = { ...current.binding, brainId: targetId }, bindingId = hash(canonical(binding));
        const diagnosticPath = join(staged, checkpointId, "diagnostic", "manifest.json");
        let diagnosticSha = rewrittenDiagnostics.get(checkpointId);
        if (!diagnosticSha) {
          const diagnostic = await jsonFile(diagnosticPath);
          const text = rewriteNativeJsonValues(diagnostic.text, new Map([[JSON.stringify(["bindingId"]), JSON.stringify(bindingId)]]));
          await writeMutableFileIsolated(diagnosticPath, Buffer.from(text)); diagnosticSha = hash(text); rewrittenDiagnostics.set(checkpointId, diagnosticSha);
        }
        const { inbox, identities } = rekeyObservationInbox(current.worker.inbox, options.sourceBrainId, targetId);
        const edits = new Map<string, string>([
          [JSON.stringify(["brainId"]), JSON.stringify(targetId)],
          [JSON.stringify(["binding", "brainId"]), JSON.stringify(targetId)],
          [JSON.stringify(["bindingId"]), JSON.stringify(bindingId)],
          [JSON.stringify(["journal", "bindingId"]), JSON.stringify(bindingId)],
          [JSON.stringify(["journal", "manifestSha256"]), JSON.stringify(diagnosticSha)],
          [JSON.stringify(["worker", "brainId"]), JSON.stringify(targetId)],
          [JSON.stringify(["worker", "inbox"]), canonical(inbox)]
        ]);
        encodedObservationEdits(current.payload, ["payload"], identities, edits);
        const text = rewriteNativeJsonValues(original.text, edits), nextSha = hash(text);
        const name = `m-${nextSha}.json`;
        await writeMutableFileIsolated(join(staged, checkpointId, name), Buffer.from(text));
        await rm(join(staged, checkpointId, original.name.split("/").at(-1)!), { force: true });
        newManifestIds.set(original.name, nextSha);
      }
      if (source.active) {
        const id = source.active.checkpointId, sha = newManifestIds.get(`.execution/${id}/m-${source.active.manifestSha256}.json`);
        if (!sha) throw new Error("Rekeyed execution lost its exact active pointer.");
        await writeMutableFileIsolated(join(staged, "active.json"), Buffer.from(canonical({ ...source.active, brainId: targetId, manifestSha256: sha })));
      }
      if (await exists(join(staged, "unloaded.json"))) {
        await writeMutableFileIsolated(join(staged, "unloaded.json"), Buffer.from(canonical({ brainId: targetId, explicitLoadRequired: true })));
      }
    }
    if (await exists(targetRoot)) { await rename(targetRoot, backup); moved = true; }
    await rename(staged, targetRoot); published = true;
    const copied = await nativeExecutionFiles(targetEngine, targetId, { metadataPath: options.targetMetadataPath });
    if (moved) await rm(backup, { recursive: true, force: true });
    return copied.files;
  } catch (error) {
    if (published) await rm(targetRoot, { recursive: true, force: true });
    if (moved) await rename(backup, targetRoot);
    throw error;
  } finally { await rm(staged, { recursive: true, force: true }); }
}
