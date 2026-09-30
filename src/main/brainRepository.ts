import { createHash, randomUUID } from "node:crypto";
import { constants as fsConstants, createReadStream } from "node:fs";
import { savedConceptIdViewFiles, validateConceptIdViewFile } from "./conceptIdViewFiles";
import {
  access,
  cp,
  copyFile,
  link,
  lstat,
  mkdir,
  mkdtemp,
  open,
  readFile,
  readdir,
  realpath,
  rename,
  rm,
  stat,
  writeFile
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { DatabaseSync } from "node:sqlite";
import { basename, dirname, join, resolve } from "node:path";
import { validateNativeArchitectureDescriptor } from "./nativeCoreInventory";
import { savedToolIntentFiles, validateSavedToolIntent } from "./savedToolIntents";
import { strToU8 } from "fflate";
import {
  BRAIN_SCHEMA_VERSION,
  DEFAULT_CONFIG,
  type BrainActivityLedgerSummary,
  type BrainActiveModeResult,
  type BrainConfig,
  type BrainDocument,
  type BrainExportMode,
  type BrainMetrics,
  type BrainProvenance,
  type BrainSnapshotSummary,
  type BrainSummary,
  type ActionEvent,
  type ChatDeliveryReceiptRequest,
  type ChatMessage,
  type ConversationLedgerPage,
  type DeleteInstanceRequest,
  type DeleteInstanceResult,
  type NeuralParameterAccounting,
  type PersistedSubstrateOverview,
  type JournalEntry,
  type JournalLedgerPage,
  type TrainingSource,
  type TrainingSourceLedgerPage,
  type ToolPermissionRecord
} from "../shared/types";
import {
  BrainActivityLedger,
  parseJournalExport,
  parseTrainingSourceExport,
  trainingSourceEvidenceFingerprint
} from "./brainActivityLedger";
import { ConversationLedger } from "./conversationLedger";
import {
  assertSafeArchivePath,
  ensureDiskReserve,
  extractStreamingZip,
  streamFileSha256,
  writeStreamingZip,
  type ExtractedZipArchive,
  type StreamingZipSource
} from "./streamingZip";
import { withBrainWrite } from "./brainWriteCoordinator";
import { verifyPortableReplaySqlite } from "./portableReplayIntegrity";
import {
  assertPortableWorkingMemoryCheckpoint,
  emptyPortableWorkingMemoryCheckpoint
} from "./portableWorkingMemory";
import type {
  BrainStorageOperationHooks,
  BrainStorageProgressUpdate
} from "./brainStorageOperations";
import {
  ArtifactIndexStore,
  parseArtifactIndex,
  serializeArtifactIndex,
  type PersistedArtifactIndex
} from "./mediaArtifactRegistry";
import {
  copyMutableFileIsolated,
  snapshotMutableSqliteIsolated,
  writeMutableFileIsolated
} from "./mutableFileIsolation";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const BUNDLE_FORMAT = "omni-brain";
const BUNDLE_VERSION = 1;
/** Must remain identical to engine/omni_core/vsa.py's store format. */
export const SUBSTRATE_STORE_FORMAT = "omni-substrate-shards";
const READABLE_SUBSTRATE_STORE_VERSIONS = new Set([1, 2, 3]);
/** Must remain identical to engine/omni_core/offload.py's store format. */
export const MUTABLE_STATE_STORE_FORMAT = "omni-mutable-state";
const STABLE_RELEASE_FORMAT = "stable-1.0";
const BETA_REVIEW_FILE = ".stable-v1-beta-review.json";

interface OmniManifest {
  format: typeof BUNDLE_FORMAT;
  formatVersion: number;
  releaseFormat: typeof STABLE_RELEASE_FORMAT;
  architecture: "OmniCortex";
  architectureSchemaVersion: number;
  exportedAt: string;
  brain: {
    id: string;
    name: string;
    lineage: BrainDocument["lineage"];
  };
  mode: "current-portable" | "origin-portable" | "private-archive" | "referenced-local";
  engineMaterialized: boolean;
  memoryRecipe: string;
  rawEpisodesPresent: boolean;
  quantization: "ternary-effective";
  conversationProjection: {
    historyIncluded: boolean;
    omittedLedgerRows: number;
    omittedPendingReplayJobs: number;
  };
  packedTernary?: {
    format: "omni-packed-ternary";
    formatVersion: 1;
    currentManifestSha256: string;
    originManifestSha256: string;
    currentTensorCount: number;
    originTensorCount: number;
    references?: {
      current: Record<string, string>;
      origin: Record<string, string>;
    };
  };
  secretRedaction: {
    version: 1;
    replacements: number;
  };
  /** v1: saved content is preserved, not a sanitized sharing projection. */
  savedInstance?: { version: 1; content: "unsanitized" };
  recoveryPoints?: Array<{ id: string; brainId: string; missingPayloads: string[] }>;
  licenseLedger: {
    application: "PolyForm-Noncommercial-1.0.0-or-commercial-license";
    sourceCount?: number;
    sourceLedger?: "activity/ledger.sqlite3";
    sources: Array<{
      name: string;
      provenanceUrl?: string;
      license: string;
      licenseUrl?: string;
    }>;
  };
  references?: {
    currentCore: string;
    currentPlasticity: string;
    originCore: string;
    originPlasticity: string;
  };
  files: Record<string, { sha256: string; bytes: number }>;
}

interface SavedSnapshotSummary extends BrainSnapshotSummary {
  savedContinuation?: { version: 1; componentPaths: string[] };
  exportMissingPayloads?: string[];
}

function safeSnapshotContinuationPath(path: string): string {
  if (path !== "engine/state/working-memory.sqlite3" &&
    !/^engine\/evaluation\/(?:geometry-holdouts\.json|data\/[a-f0-9]{32}(?:\.[A-Za-z0-9_-]+)?)$/.test(path) &&
    !/^engine\/state\/concept-id-views\/[a-f0-9]{64}\.jsonl$/.test(path) &&
    !/^engine\/operational-tool-intents\/[a-f0-9-]{36}\.json$/i.test(path) &&
    !/^engine\/state\/ingestion-joint\/generations\/[a-f0-9]{32}\/(?:manifest\.json|packed-vector-index\.sqlite3)$/.test(path)) {
    throw new Error("Recovery point has an unsupported continuation path.");
  }
  return path;
}

async function savedTreeFiles(directory: string): Promise<Map<string, string>> {
  const files = new Map<string, string>();
  async function visit(path: string, prefix: string): Promise<void> {
    const info = await lstat(path);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Recovery-point directory is unsafe.");
    for (const entry of await readdir(path, { withFileTypes: true })) {
      const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
      assertAllowedBundlePath(relative);
      const source = join(path, entry.name);
      if (entry.isSymbolicLink()) throw new Error("Recovery-point payload is a filesystem link.");
      if (entry.isDirectory()) await visit(source, relative);
      else if (entry.isFile()) {
        if (/(?:-wal|-shm|-journal)$/.test(entry.name)) {
          if ((await lstat(source)).size > 0) throw new Error("Immutable recovery-point payload has a live SQLite sidecar.");
          continue;
        }
        files.set(relative, source);
      } else throw new Error("Recovery-point payload is not a regular file.");
    }
  }
  await visit(directory, "");
  return files;
}

async function snapshotComponentHashes(base: string, summary: SavedSnapshotSummary): Promise<string[]> {
  const engine = join(base, "engine");
  const hashes: string[] = [];
  const metadataPath = join(engine, "brain.json");
  if (await pathExists(metadataPath)) {
    const metadata = await readFile(metadataPath);
    savedEngineState(metadata);
    const state = JSON.parse(metadata.toString("utf8")) as Record<string, unknown>;
    if (state.brain_id !== summary.brainId) throw new Error("Recovery-point neural identity does not match its owner.");
    hashes.push(sha256(metadata));
    for (const name of ["core.safetensors", "plasticity.safetensors"]) {
      const path = join(engine, name);
      if (await pathExists(path)) {
        await assertSafeTensorsFile(path, `Recovery point ${name}`);
        hashes.push(await fileSha256(path));
      }
    }
    if (await pathExists(join(engine, "packed-ternary", "manifest.json"))) {
      hashes.push((await inspectPackedTernaryDirectory(join(engine, "packed-ternary"), "Recovery point")).manifestSha256);
    }
    const substrate = await collectSubstrateSnapshot(engine, "substrate/snapshot", state);
    if (substrate) hashes.push(sha256(canonicalJson(substrate.pointer)));
    const mutable = await collectMutableStateSnapshot(engine, "mutable/snapshot", state);
    if (mutable) hashes.push(sha256(canonicalJson(mutable.pointer)));
    if (summary.durableState) {
      const artifactDirectory = join(engine, "artifacts");
      let artifacts: PersistedArtifactIndex["artifacts"] | undefined;
      const ledgerPath = ArtifactIndexStore.databasePath(artifactDirectory);
      if (await pathExists(ledgerPath)) {
        const database = new DatabaseSync(ledgerPath, { readOnly: true });
        try {
          if (database.prepare("PRAGMA quick_check").get()?.quick_check !== "ok" ||
            database.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== summary.brainId) {
            throw new Error("Recovery-point artifact ledger integrity failed.");
          }
          artifacts = parseArtifactIndex(JSON.parse(serializeArtifactIndex(summary.brainId,
            [...database.prepare("SELECT payload_json FROM artifacts ORDER BY sequence").iterate()]
              .map((row) => JSON.parse(String(row.payload_json)))
          )), summary.brainId).artifacts;
        } finally { database.close(); }
      } else if (await pathExists(join(artifactDirectory, "index.json"))) {
        artifacts = parseArtifactIndex(JSON.parse(await readFile(join(artifactDirectory, "index.json"), "utf8")), summary.brainId).artifacts;
      }
      if (summary.durableState.artifactIndex && artifacts === undefined) throw new Error("Recovery-point artifact index is missing.");
      if (artifacts !== undefined) {
        for (const artifact of artifacts) {
          const path = join(artifactDirectory, basename(artifact.relativePath));
          if ((await lstat(path)).size !== artifact.bytes || await fileSha256(path) !== artifact.sha256) {
            throw new Error("Recovery-point generated artifact checksum failed.");
          }
        }
        hashes.push(sha256(serializeArtifactIndex(summary.brainId, artifacts)));
      }
      const neural = join(engine, "conversation.sqlite3");
      if (await pathExists(neural)) {
        validateNeuralConversationLedger(neural, summary.brainId, state);
        hashes.push(await fileSha256(neural));
      } else if (summary.durableState.neuralConversationLedger) {
        throw new Error("Recovery-point neural conversation ledger is missing.");
      }
    }
    if (summary.savedContinuation) {
      if (summary.savedContinuation.version !== 1 || !Array.isArray(summary.savedContinuation.componentPaths)) {
        throw new Error("Recovery-point continuation declaration is invalid.");
      }
      for (const relative of summary.savedContinuation.componentPaths) {
        if (relative.startsWith("engine/state/concept-id-views/")) {
          await validateConceptIdViewFile(join(base, ...relative.split("/")), basename(relative));
        }
        if (relative.startsWith("engine/operational-tool-intents/")) {
          await validateSavedToolIntent(join(base, ...relative.split("/")));
        }
        hashes.push(await fileSha256(join(base, ...safeSnapshotContinuationPath(relative).split("/"))));
      }
      const declaredIntents = new Set(summary.savedContinuation.componentPaths
        .filter(path => path.startsWith("engine/operational-tool-intents/"))
        .map(path => basename(path)));
      const observedIntents = await savedToolIntentFiles(engine);
      if (observedIntents.size !== declaredIntents.size ||
        [...observedIntents.keys()].some(name => !declaredIntents.has(name))) {
        throw new Error("Recovery-point operational journal differs from its committed declaration.");
      }
      validateSavedWorkingPages(await pathExists(join(engine, "state", "working-memory.sqlite3"))
        ? join(engine, "state", "working-memory.sqlite3") : undefined, state);
      await savedJointGenerationFiles(state, (kind, relative) =>
        join(engine, "state", ...(kind === "joint" ? ["ingestion-joint"] : []), ...relative.split("/"))
      );
    }
  }
  if (summary.durableState) {
    for (const path of [join(base, "conversation", "ledger.sqlite3"), BrainActivityLedger.databasePath(base)]) {
      const database = new DatabaseSync(path, { readOnly: true });
      try {
        if (database.prepare("PRAGMA quick_check").get()?.quick_check !== "ok" ||
          database.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== summary.brainId) {
          throw new Error("Recovery-point host ledger integrity failed.");
        }
      } finally { database.close(); }
      hashes.push(await fileSha256(path));
    }
  }
  return hashes;
}

async function validateSavedSnapshot(base: string, id: string, brainId: string): Promise<{ summary: SavedSnapshotSummary; missingPayloads: string[] }> {
  const [document, summaryText] = await Promise.all([readFile(`${base}.json`), readFile(`${base}.meta.json`, "utf8")]);
  const summary = JSON.parse(summaryText) as SavedSnapshotSummary;
  const brain = normalizeBrain(JSON.parse(document.toString("utf8")));
  if (summary.id !== id || summary.brainId !== brainId || brain.id !== brainId ||
    summary.checksum !== sha256(document)) throw new Error("Recovery-point document checksum or identity failed.");
  const hashes = await snapshotComponentHashes(base, summary);
  if (summary.engineChecksum && summary.engineChecksum !== sha256(hashes.join(":"))) {
    throw new Error("Recovery-point neural and ledger checksum failed.");
  }
  const missingPayloads: string[] = [];
  const metadataPath = join(base, "engine", "brain.json");
  if (await pathExists(metadataPath)) {
    const metadata = JSON.parse(await readFile(metadataPath, "utf8")) as Record<string, unknown>;
    if (isRecord(metadata.paged_working_memory) && Number(metadata.paged_working_memory.count) > 0 &&
      !await pathExists(join(base, "engine", "state", "working-memory.sqlite3"))) {
      missingPayloads.push("engine/state/working-memory.sqlite3");
    }
    if (isRecord(metadata.ingestion_joint_generation) && typeof metadata.ingestion_joint_generation.relativeManifest === "string") {
      const reference = metadata.ingestion_joint_generation.relativeManifest;
      if (!/^generations\/[a-f0-9]{32}\/manifest\.json$/.test(reference)) {
        throw new Error("Recovery-point joint reference is unsafe.");
      }
      const relative = `engine/state/ingestion-joint/${reference}`;
      const manifestPath = join(base, ...relative.split("/"));
      if (!await pathExists(manifestPath)) missingPayloads.push(relative);
      else {
        const joint = JSON.parse(await readFile(manifestPath, "utf8")) as Record<string, unknown>;
        if (isRecord(joint.sqliteSnapshot)) {
          if (joint.sqliteSnapshot.file !== "packed-vector-index.sqlite3") throw new Error("Recovery-point joint SQLite path is unsafe.");
          const sqliteRelative = relative.replace(/manifest\.json$/, "packed-vector-index.sqlite3");
          if (!await pathExists(join(base, ...sqliteRelative.split("/")))) missingPayloads.push(sqliteRelative);
        }
        if (!missingPayloads.length) await savedJointGenerationFiles(metadata, (kind, path) =>
          join(base, "engine", "state", ...(kind === "joint" ? ["ingestion-joint"] : []), ...path.split("/"))
        );
      }
    }
  }
  return { summary, missingPayloads };
}

async function rekeySavedSnapshot(base: string, brainId: string, missingPayloads: string[]): Promise<void> {
  const summary = JSON.parse(await readFile(`${base}.meta.json`, "utf8")) as SavedSnapshotSummary;
  const document = JSON.parse(await readFile(`${base}.json`, "utf8")) as BrainDocument;
  const priorId = summary.brainId;
  document.id = brainId;
  const documentBytes = JSON.stringify(document, null, 2);
  await atomicWrite(`${base}.json`, documentBytes);
  const enginePath = join(base, "engine", "brain.json");
  if (await pathExists(enginePath)) {
    const state = JSON.parse(await readFile(enginePath, "utf8")) as Record<string, unknown>;
    state.brain_id = brainId;
    await atomicWrite(enginePath, JSON.stringify(state, null, 2));
  }
  for (const [relative, key] of [
    ["conversation/ledger.sqlite3", "brainId"], ["activity/ledger.sqlite3", "brainId"],
    ["engine/conversation.sqlite3", "brain_id"], ["engine/artifacts/index.sqlite3", "brainId"]
  ] as const) {
    const path = join(base, ...relative.split("/"));
    if (!await pathExists(path)) continue;
    const database = new DatabaseSync(path);
    try {
      if (database.prepare("SELECT value FROM meta WHERE key=?").get(key)?.value !== priorId) {
        throw new Error("Imported recovery-point ledger identity failed.");
      }
      database.prepare("UPDATE meta SET value=? WHERE key=?").run(brainId, key);
    } finally { database.close(); }
  }
  const legacyArtifacts = join(base, "engine", "artifacts", "index.json");
  if (await pathExists(legacyArtifacts)) {
    const index = parseArtifactIndex(JSON.parse(await readFile(legacyArtifacts, "utf8")), priorId, brainId);
    await atomicWrite(legacyArtifacts, serializeArtifactIndex(brainId, index.artifacts));
  }
  summary.brainId = brainId;
  summary.checksum = sha256(documentBytes);
  if (missingPayloads.length) summary.exportMissingPayloads = missingPayloads;
  const hashes = await snapshotComponentHashes(base, summary);
  if (summary.engineChecksum) summary.engineChecksum = sha256(hashes.join(":"));
  if (summary.checkpointComponentSha256) summary.checkpointComponentSha256 = hashes;
  await atomicWrite(`${base}.meta.json`, JSON.stringify(summary, null, 2));
}

interface StreamingPackedTernaryDirectory {
  manifestSha256: string;
  tensorCount: number;
  files: Map<string, string>;
}

interface CloneMaterialization {
  sourcePath?: string;
  contents?: Uint8Array;
  destination: string;
  label: string;
  bytes: number;
  shareable?: boolean;
}

export interface ManagedBetaBrain {
  id: string;
  name: string;
  path: string;
  reason: "beta-document" | "beta-engine" | "invalid-document";
}

export interface ImmutableOriginStorageReport {
  files: number;
  logicalBytes: number;
  contentHashes: number;
  sharedFiles: number;
}

export const DEFAULT_TOOL_PERMISSIONS: ToolPermissionRecord[] = [
  "system.files",
  "system.shell",
  "code.execute",
  "web.search",
  "web.fetch",
  "browser.automation",
  "device.input",
  "modality.imagine",
  "brain.history",
  "agent.fork",
  "source.self-modify"
].map((toolId) => ({
  toolId,
  label: toolId
    .split(".")
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join(" "),
  level:
    toolId === "browser.automation" || toolId === "source.self-modify"
      ? ("off" as const)
      : toolId === "modality.imagine" || toolId === "brain.history"
        ? ("auto" as const)
        : ("ask" as const),
  updatedAt: new Date(0).toISOString()
}));

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

async function assertFileContainsValidJsonText(path: string, label: string): Promise<void> {
  const decoder = new TextDecoder("utf-8", { fatal: true });
  for await (const chunk of createReadStream(path)) {
    let text: string;
    try {
      text = decoder.decode(chunk as Buffer, { stream: true });
    } catch {
      throw new Error(`${label} is not valid UTF-8 text and cannot be safely exported.`);
    }
    if (text.includes("\0")) {
      throw new Error(`${label} is not valid JSON text and cannot be safely exported.`);
    }
  }
  try {
    decoder.decode();
  } catch {
    throw new Error(`${label} is not valid UTF-8 text and cannot be safely exported.`);
  }
}

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

async function fileSha256(path: string, signal?: AbortSignal): Promise<string> {
  signal?.throwIfAborted();
  const digest = createHash("sha256");
  await new Promise<void>((resolveHash, rejectHash) => {
    const stream = createReadStream(path);
    const abort = (): void => {
      const error = new Error("Recovery-point hashing was cancelled.");
      error.name = "AbortError";
      stream.destroy(error);
    };
    const cleanup = (): void => signal?.removeEventListener("abort", abort);
    stream.on("data", (chunk) => digest.update(chunk));
    stream.once("error", (error) => {
      cleanup();
      rejectHash(error);
    });
    stream.once("end", () => {
      cleanup();
      resolveHash();
    });
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
  });
  return digest.digest("hex");
}

function validEmptySafetensors(note: string): Uint8Array {
  const header = Buffer.from(
    JSON.stringify({ __metadata__: { format: "omni-empty", note } }).padEnd(256, " "),
    "utf8"
  );
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(header.byteLength));
  return new Uint8Array(Buffer.concat([prefix, header]));
}

function safeZipPath(path: string): boolean {
  if (
    !path ||
    path.includes("\0") ||
    path.includes("\\") ||
    path.startsWith("/") ||
    /^[a-zA-Z]:/.test(path)
  ) {
    return false;
  }
  const parts = path.split("/");
  return parts.every((part) => part !== "" && part !== "." && part !== "..");
}

function assertAllowedBundlePath(path: string): void {
  if (!safeZipPath(path)) throw new Error(`Unsafe path in .omni bundle: ${path}`);
  if (
    /\.(?:exe|dll|com|bat|cmd|ps1|msi|scr|js|jse|vbs|vbe|wsf|wsh|lnk|app|dylib|so|pyc)$/i.test(path)
  ) {
    throw new Error(`Executable content is not allowed in .omni bundles: ${path}`);
  }
}

function parseChecksumFile(value: string): Map<string, string> {
  const checksums = new Map<string, string>();
  for (const line of value.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const match = /^([a-f0-9]{64})  ([^\r\n]+)$/i.exec(line);
    if (!match?.[1] || !match[2]) throw new Error("checksums.sha256 has an invalid record.");
    assertAllowedBundlePath(match[2]);
    if (checksums.has(match[2])) throw new Error(`Duplicate checksum for ${match[2]}.`);
    checksums.set(match[2], match[1].toLocaleLowerCase());
  }
  return checksums;
}

function validateNeuralConversationLedger(
  path: string,
  expectedBrainId: string,
  expectedState?: unknown
): void {
  const summary = isRecord(expectedState) ? expectedState.conversation : undefined;
  const committedSequence = isRecord(summary) ? Number(summary.headSequence) : undefined;
  if (committedSequence !== undefined && (!Number.isSafeInteger(committedSequence) || committedSequence < 0)) {
    throw new Error("Neural conversation committed head is invalid.");
  }
  const database = new DatabaseSync(path, { readOnly: true });
  try {
    const quick = database.prepare("PRAGMA quick_check").get() as
      { quick_check?: string } | undefined;
    const identity = database.prepare(
      "SELECT value FROM meta WHERE key='brain_id'"
    ).get() as { value?: string } | undefined;
    if (quick?.quick_check !== "ok" || identity?.value !== expectedBrainId) {
      throw new Error("Neural conversation ledger identity or SQLite integrity failed.");
    }
    const rows = database.prepare(`
      SELECT sequence,entry_key,kind,created_at,attention_epoch,payload_json,
        payload_sha256,previous_sha256,row_sha256
      FROM entries ORDER BY sequence
    `).iterate();
    let previous = "0".repeat(64);
    let count = 0;
    let committedEntries = 0;
    let committedSha256 = previous;
    let epoch = 0;
    const counts = { messageCount: 0, actionCount: 0, traceCount: 0 };
    for (const row of rows) {
      const payloadJson = String(row.payload_json ?? "");
      const payloadSha256 = sha256(payloadJson);
      const body = {
        sequence: Number(row.sequence),
        entryKey: String(row.entry_key),
        kind: String(row.kind),
        createdAt: String(row.created_at),
        attentionEpoch: Number(row.attention_epoch),
        payloadSha256,
        previousSha256: String(row.previous_sha256)
      };
      if (
        body.sequence !== count + 1 ||
        !["message", "action", "trace"].includes(body.kind) ||
        !Number.isSafeInteger(body.attentionEpoch) || body.attentionEpoch < 0 ||
        body.previousSha256 !== previous ||
        row.payload_sha256 !== payloadSha256 ||
        row.row_sha256 !== sha256(canonicalJson(body))
      ) {
        throw new Error("Neural conversation ledger hash chain failed.");
      }
      previous = String(row.row_sha256);
      count += 1;
      // The worker can have durable ledger rows ahead of brain.json and
      // truncates that suffix during crash recovery. Retain and validate all
      // rows in the archive; bind pending jobs only to the committed prefix.
      if (committedSequence === undefined || body.sequence <= committedSequence) {
        committedEntries += 1;
        committedSha256 = previous;
        epoch = Math.max(epoch, body.attentionEpoch);
        counts[`${body.kind}Count` as keyof typeof counts] += 1;
      }
    }
    if (summary !== undefined && canonicalJson(summary) !== canonicalJson({
      format: "omni-neural-conversation-ledger", formatVersion: 1,
      totalEntries: committedEntries, ...counts, attentionEpoch: epoch,
      headSequence: committedEntries, headSha256: committedSha256
    })) {
      throw new Error("Neural conversation head does not match its saved ledger.");
    }
    if (isRecord(expectedState) && Array.isArray(expectedState.pending_chat_slow_learning)) {
      const messages = database.prepare(
        "SELECT 1 FROM entries WHERE kind='message' AND json_extract(payload_json,'$.id')=? AND sequence<=?"
      );
      for (const pending of expectedState.pending_chat_slow_learning) {
        if (!isRecord(pending) || typeof pending.humanMessageId !== "string" ||
          !messages.get(pending.humanMessageId, committedSequence ?? count)) {
          throw new Error("Pending chat replay has no saved human message in its neural ledger.");
        }
      }
    }
  } finally {
    database.close();
  }
}

/** A consistent private snapshot includes committed WAL rows, never sidecars. */
async function snapshotSavedSqlite(
  source: string,
  destination: string,
  operation?: BrainStorageOperationHooks
): Promise<void> {
  const info = await lstat(source);
  const wal = await lstat(`${source}-wal`).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return undefined;
    throw error;
  });
  if (!info.isFile() || info.isSymbolicLink() ||
    (wal && (!wal.isFile() || wal.isSymbolicLink()))) {
    throw new Error("Saved SQLite state is not a safe regular file.");
  }
  const bytes = info.size + (wal?.size ?? 0);
  if (!Number.isSafeInteger(bytes)) throw new Error("Saved SQLite state is too large.");
  operation?.signal.throwIfAborted();
  await operation?.checkDisk(dirname(destination), bytes);
  await ensureDiskReserve(dirname(destination), bytes);
  await snapshotMutableSqliteIsolated(source, destination);
  operation?.signal.throwIfAborted();
}

function validateSavedWorkingPages(path: string | undefined, state: unknown): void {
  if (!isRecord(state)) throw new Error("Saved working-memory state is invalid.");
  const checkpoint = state.paged_working_memory;
  const expected = checkpoint === undefined ? undefined : checkpoint;
  const shape = emptyPortableWorkingMemoryCheckpoint();
  if (expected !== undefined && (!isRecord(expected) ||
    Object.keys(expected).length !== Object.keys(shape).length ||
    expected.format !== shape.format || expected.formatVersion !== 1 ||
    !Number.isSafeInteger(expected.count) || Number(expected.count) < 0 ||
    !Number.isSafeInteger(expected.highWaterId) || Number(expected.highWaterId) < 0 ||
    typeof expected.contentSha256 !== "string" || !/^[a-f0-9]{64}$/.test(expected.contentSha256) ||
    ["temporary", "runtimeReadable", "learningReadable", "pageInSupported"].some(
      (key) => expected[key] !== true
    ))) throw new Error("Saved working-memory checkpoint is invalid.");
  if (!path) {
    if (expected !== undefined && canonicalJson(expected) !== canonicalJson(shape)) {
      throw new Error("Saved working-memory pages are missing from the bundle.");
    }
    return;
  }
  const database = new DatabaseSync(path, { readOnly: true });
  try {
    const quick = database.prepare("PRAGMA quick_check").get();
    const table = database.prepare("SELECT type FROM sqlite_master WHERE name='working_pages'").get();
    if (quick?.quick_check !== "ok" || table?.type !== "table") {
      throw new Error("Saved working-memory SQLite integrity failed.");
    }
    const dtypeBytes: Record<string, number> = {
      "torch.bool": 1, "torch.uint8": 1, "torch.int8": 1,
      "torch.int16": 2, "torch.int32": 4, "torch.int64": 8,
      "torch.float16": 2, "torch.bfloat16": 2, "torch.float32": 4, "torch.float64": 8
    };
    const digest = createHash("sha256");
    let count = 0;
    let highWater = 0;
    let previous = 0;
    for (const row of database.prepare("SELECT * FROM working_pages ORDER BY sequence").iterate()) {
      const sequence = Number(row.sequence);
      const dtype = String(row.dtype);
      const payload = row.payload;
      const tensorShape = JSON.parse(String(row.shape_json)) as unknown;
      const metadata = JSON.parse(String(row.metadata_json)) as unknown;
      if (!Number.isSafeInteger(sequence) || sequence <= previous ||
        typeof row.page_id !== "string" || !/^[\x20-\x7e]+$/.test(row.page_id) ||
        typeof row.assembly_id !== "string" || !isRecord(metadata) ||
        (metadata.assemblyId !== undefined && metadata.assemblyId !== row.assembly_id) ||
        !Array.isArray(tensorShape) || tensorShape.length === 0 ||
        tensorShape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 0) ||
        !Object.hasOwn(dtypeBytes, dtype) || !(payload instanceof Uint8Array) ||
        !Number.isFinite(row.updated_at)) {
        throw new Error("Saved working-memory page schema is invalid.");
      }
      const bytes = tensorShape.reduce((total: number, dimension: number) => total * dimension, 1) * dtypeBytes[dtype]!;
      const checksum = createHash("sha256").update(dtype).update("\0")
        .update(`[${tensorShape.join(", ")}]`).update("\0").update(payload).digest("hex");
      if (!Number.isSafeInteger(bytes) || bytes !== payload.byteLength || row.sha256 !== checksum) {
        throw new Error("Saved working-memory tensor checksum failed.");
      }
      if (!isRecord(expected) || sequence <= Number(expected.highWaterId)) {
        digest.update(String(sequence)).update("\0").update(row.page_id).update("\0")
          .update(checksum).update("\n");
        count += 1;
        highWater = sequence;
      }
      previous = sequence;
    }
    if (isRecord(expected) && (count !== expected.count || highWater !== expected.highWaterId ||
      digest.digest("hex") !== expected.contentSha256)) {
      throw new Error("Saved working-memory checkpoint checksum failed.");
    }
  } finally {
    database.close();
  }
}

function rekeyNeuralConversationLedger(path: string, brainId: string): void {
  const database = new DatabaseSync(path);
  try {
    database.prepare("UPDATE meta SET value=? WHERE key='brain_id'").run(brainId);
  } finally {
    database.close();
  }
}

function assertSafeTensors(contents: Uint8Array, label: string): void {
  const buffer = Buffer.from(contents.buffer, contents.byteOffset, contents.byteLength);
  if (buffer.byteLength < 10) throw new Error(`${label} is not a valid safetensors file.`);
  const headerLength = Number(buffer.readBigUInt64LE(0));
  if (
    !Number.isSafeInteger(headerLength) ||
    headerLength < 2 ||
    headerLength > buffer.byteLength - 8
  ) {
    throw new Error(`${label} has an invalid safetensors header length.`);
  }
  let header: unknown;
  try {
    header = JSON.parse(
      buffer
        .subarray(8, 8 + headerLength)
        .toString("utf8")
        .trim()
    );
  } catch {
    throw new Error(`${label} has an invalid safetensors JSON header.`);
  }
  if (!isRecord(header)) throw new Error(`${label} has an invalid safetensors header.`);
  const dataBytes = buffer.byteLength - 8 - headerLength;
  for (const [name, descriptor] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (!isRecord(descriptor) || !Array.isArray(descriptor.data_offsets)) {
      throw new Error(`${label} contains an invalid tensor descriptor.`);
    }
    const offsets = descriptor.data_offsets;
    if (
      offsets.length !== 2 ||
      !offsets.every((offset) => typeof offset === "number" && Number.isSafeInteger(offset)) ||
      (offsets[0] as number) < 0 ||
      (offsets[1] as number) < (offsets[0] as number) ||
      (offsets[1] as number) > dataBytes
    ) {
      throw new Error(`${label} contains out-of-bounds tensor data.`);
    }
  }
}

async function assertSafeTensorsFile(path: string, label: string): Promise<void> {
  const info = await stat(path);
  if (!info.isFile() || info.size < 10) {
    throw new Error(`${label} is not a valid safetensors file.`);
  }
  const handle = await open(path, "r");
  try {
    const prefix = Buffer.alloc(8);
    const prefixRead = await handle.read(prefix, 0, prefix.byteLength, 0);
    if (prefixRead.bytesRead !== prefix.byteLength) {
      throw new Error(`${label} is not a valid safetensors file.`);
    }
    const headerLength = Number(prefix.readBigUInt64LE(0));
    if (!Number.isSafeInteger(headerLength) || headerLength < 2 || headerLength > info.size - 8) {
      throw new Error(`${label} has an invalid safetensors header length.`);
    }
    const headerBytes = Buffer.allocUnsafe(headerLength);
    let cursor = 0;
    while (cursor < headerLength) {
      const result = await handle.read(headerBytes, cursor, headerLength - cursor, 8 + cursor);
      if (result.bytesRead <= 0) {
        throw new Error(`${label} has a truncated safetensors header.`);
      }
      cursor += result.bytesRead;
    }
    let header: unknown;
    try {
      header = JSON.parse(headerBytes.toString("utf8").trim());
    } catch {
      throw new Error(`${label} has an invalid safetensors JSON header.`);
    }
    if (!isRecord(header)) throw new Error(`${label} has an invalid safetensors header.`);
    const dataBytes = info.size - 8 - headerLength;
    for (const [name, descriptor] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      if (!isRecord(descriptor) || !Array.isArray(descriptor.data_offsets)) {
        throw new Error(`${label} contains an invalid tensor descriptor.`);
      }
      const offsets = descriptor.data_offsets;
      if (
        offsets.length !== 2 ||
        !offsets.every((offset) => typeof offset === "number" && Number.isSafeInteger(offset)) ||
        (offsets[0] as number) < 0 ||
        (offsets[1] as number) < (offsets[0] as number) ||
        (offsets[1] as number) > dataBytes
      ) {
        throw new Error(`${label} contains out-of-bounds tensor data.`);
      }
    }
  } finally {
    await handle.close();
  }
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.entries(value as Record<string, unknown>)
    // Python's json.dumps(sort_keys=True) compares Unicode code points.
    // localeCompare is locale-sensitive (and orders "+1" after "-1" on
    // some hosts), which would reject the worker's canonical manifests.
    .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
    .join(",")}}`;
}

function normalizeCanonicalJsonNumbers(value: string): string {
  let normalized = "";
  let inString = false;
  let escaped = false;
  for (let index = 0; index < value.length;) {
    const character = value[index]!;
    if (inString) {
      normalized += character;
      index += 1;
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === '"') inString = false;
      continue;
    }
    if (character === '"') {
      inString = true;
      normalized += character;
      index += 1;
      continue;
    }
    if (character === "-" || /[0-9]/.test(character)) {
      const token = value
        .slice(index)
        .match(/^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/)?.[0];
      if (!token) throw new Error("Canonical JSON contains an invalid number.");
      const number = Number(token);
      if (!Number.isFinite(number)) {
        throw new Error("Canonical JSON contains a non-finite number.");
      }
      normalized += JSON.stringify(number);
      index += token.length;
      continue;
    }
    normalized += character;
    index += 1;
  }
  return normalized;
}

function canonicalManifestContentBytes(
  manifestText: string,
  claimedContentHash: string
): Buffer {
  const field = `"contentSha256":${JSON.stringify(claimedContentHash)}`;
  const fieldStart = manifestText.indexOf(field);
  if (fieldStart < 0 || manifestText.indexOf(field, fieldStart + field.length) >= 0) {
    throw new Error("Canonical manifest content checksum field is ambiguous.");
  }
  let start = fieldStart;
  let end = fieldStart + field.length;
  if (manifestText[end] === ",") end += 1;
  else if (manifestText[start - 1] === ",") start -= 1;
  else throw new Error("Canonical manifest content checksum field is malformed.");
  return Buffer.from(manifestText.slice(0, start) + manifestText.slice(end), "utf8");
}

function safeSubstrateRelativePath(path: string): string {
  if (
    path !== "manifest.json" &&
    !/^generations\/[a-f0-9]{64}\/manifest\.json$/.test(path) &&
    !/^blobs\/[a-f0-9]{64}\.(?:json|safetensors)$/.test(path)
  ) {
    throw new Error("Neural substrate manifest contains an unsupported path.");
  }
  assertSafeArchivePath(path);
  return path;
}

interface SubstrateSnapshot {
  pointer: Record<string, unknown>;
  sources: StreamingZipSource[];
  relativePaths: Set<string>;
}

async function assertSubstrateSnapshotJsonText(
  snapshot: SubstrateSnapshot | undefined,
  signal?: AbortSignal
): Promise<void> {
  for (const source of snapshot?.sources ?? []) {
    if (!source.name.endsWith(".json")) continue;
    signal?.throwIfAborted();
    // Archive names are assembled only from fixed prefixes and validated
    // content-addressed paths. Never include the source JSON in diagnostics.
    const label = `Neural substrate shard ${source.name}`;
    if (source.sourcePath) {
      await assertFileContainsValidJsonText(source.sourcePath, label);
      continue;
    }
    if (source.contents === undefined) {
      throw new Error(`${label} is empty and cannot be safely exported.`);
    }
    let text: string;
    try {
      text = new TextDecoder("utf-8", { fatal: true }).decode(source.contents);
    } catch {
      throw new Error(`${label} is not valid UTF-8 text and cannot be safely exported.`);
    }
    if (text.includes("\0")) {
      throw new Error(`${label} is not valid JSON text and cannot be safely exported.`);
    }
  }
}

function persistedSubstrateCounts(value: unknown): PersistedSubstrateOverview["totals"] {
  if (!isRecord(value)) {
    throw new Error("Persisted neural substrate counts are invalid.");
  }
  const count = (field: "neurons" | "assemblies" | "synapses"): number => {
    const candidate = value[field];
    if (!Number.isSafeInteger(candidate) || Number(candidate) < 0) {
      throw new Error("Persisted neural substrate counts are invalid.");
    }
    return Number(candidate);
  };
  return {
    neurons: count("neurons"),
    assemblies: count("assemblies"),
    synapses: count("synapses")
  };
}

function persistedParameterAccounting(
  value: unknown,
  substrateSynapses: number
): NeuralParameterAccounting | undefined {
  if (!isRecord(value)) return undefined;
  const fields = [
    "mutableDenseParameters",
    "substrateDynamicSparseSynapses",
    "dynamicSparseSynapses",
    "totalNeuralParameters"
  ] as const;
  if (
    fields.some(
      (field) =>
        !Number.isSafeInteger(value[field]) || Number(value[field]) < 0
    ) ||
    ["foundationEffectiveParameters", "sequenceDynamicSparseSynapses", "fixedSequenceStatisticalCapacity"]
      .some((field) => Object.hasOwn(value, field)) ||
    typeof value.countingRule !== "string" ||
    !value.countingRule.trim() ||
    value.countingRule.includes("\0") ||
    value.countingRule.length > 512
  ) {
    throw new Error("Persisted neural parameter accounting is invalid.");
  }
  const mutableDenseParameters = Number(value.mutableDenseParameters);
  const substrateDynamicSparseSynapses = Number(value.substrateDynamicSparseSynapses);
  const dynamicSparseSynapses = Number(value.dynamicSparseSynapses);
  const totalNeuralParameters = Number(value.totalNeuralParameters);
  const substrateVectorParameters = value.substrateVectorParameters === undefined
    ? 0 : Number(value.substrateVectorParameters);
  if (
    (value.substrateVectorParameters !== undefined &&
      (!Number.isSafeInteger(value.substrateVectorParameters) || substrateVectorParameters < 0)) ||
    substrateDynamicSparseSynapses !== substrateSynapses ||
    dynamicSparseSynapses !== substrateSynapses ||
    mutableDenseParameters + substrateVectorParameters + dynamicSparseSynapses !== totalNeuralParameters
  ) {
    throw new Error("Persisted neural parameter accounting is invalid.");
  }
  return {
    mutableDenseParameters: Number(value.mutableDenseParameters),
    ...(value.substrateVectorParameters === undefined ? {} : { substrateVectorParameters }),
    substrateDynamicSparseSynapses,
    dynamicSparseSynapses,
    totalNeuralParameters,
    countingRule: value.countingRule.trim()
  };
}

async function readPersistedSubstrateOverview(
  engineDirectory: string,
  expectedBrainId: string
): Promise<PersistedSubstrateOverview | undefined> {
  const metadataPath = join(engineDirectory, "brain.json");
  try {
    const metadataInfo = await lstat(metadataPath);
    if (!metadataInfo.isFile() || metadataInfo.isSymbolicLink()) {
      throw new Error("Persisted neural engine metadata is not a safe regular file.");
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
  const metadata = JSON.parse(await readFile(metadataPath, "utf8")) as unknown;
  if (!isRecord(metadata) || metadata.brain_id !== expectedBrainId) {
    throw new Error("Persisted neural engine metadata belongs to another brain.");
  }
  const substrate = isRecord(metadata.substrate) ? metadata.substrate : undefined;
  const embedded = substrate && isRecord(substrate.persistence)
    ? substrate.persistence
    : undefined;
  if (!embedded) return undefined;
  if (
    embedded.format !== SUBSTRATE_STORE_FORMAT ||
    !READABLE_SUBSTRATE_STORE_VERSIONS.has(embedded.formatVersion as number) ||
    typeof embedded.activeGeneration !== "string" ||
    !/^[a-f0-9]{64}$/.test(embedded.activeGeneration) ||
    embedded.contentSha256 !== embedded.activeGeneration ||
    typeof embedded.generationManifestSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(embedded.generationManifestSha256) ||
    embedded.generationManifest !==
      `generations/${embedded.activeGeneration}/manifest.json` ||
    !Number.isSafeInteger(embedded.shardCount) ||
    Number(embedded.shardCount) < 0
  ) {
    throw new Error("Persisted neural substrate pointer is invalid.");
  }
  const totals = persistedSubstrateCounts(embedded.counts);
  const store = join(engineDirectory, "substrate");
  const pointerPath = join(store, "manifest.json");
  const pointerInfo = await lstat(pointerPath);
  if (!pointerInfo.isFile() || pointerInfo.isSymbolicLink()) {
    throw new Error("Persisted neural substrate pointer is not a safe regular file.");
  }
  const pointer = JSON.parse(await readFile(pointerPath, "utf8")) as unknown;
  if (!isRecord(pointer) || canonicalJson(pointer) !== canonicalJson(embedded)) {
    throw new Error("Persisted neural substrate pointer does not match engine state.");
  }
  const generationPath = join(
    store,
    "generations",
    embedded.activeGeneration,
    "manifest.json"
  );
  const generationInfo = await lstat(generationPath);
  if (!generationInfo.isFile() || generationInfo.isSymbolicLink()) {
    throw new Error(
      "Persisted neural substrate generation is not a safe regular file."
    );
  }
  const generationBytes = await readFile(generationPath);
  if (sha256(generationBytes) !== embedded.generationManifestSha256) {
    throw new Error("Persisted neural substrate generation checksum failed.");
  }
  const generation = JSON.parse(generationBytes.toString("utf8")) as unknown;
  if (
    !isRecord(generation) ||
    generation.format !== SUBSTRATE_STORE_FORMAT ||
    generation.formatVersion !== embedded.formatVersion ||
    (generation.formatVersion === 3 && generation.schema !== "neural-substrate-2") ||
    generation.contentSha256 !== embedded.activeGeneration ||
    !Array.isArray(generation.shards) ||
    generation.shards.length !== embedded.shardCount ||
    canonicalJson(persistedSubstrateCounts(generation.counts)) !==
      canonicalJson(totals)
  ) {
    throw new Error("Persisted neural substrate generation is incompatible.");
  }
  const summed = { neurons: 0, assemblies: 0, synapses: 0 };
  for (const shard of generation.shards) {
    if (
      !isRecord(shard) ||
      !["neurons", "assemblies", "synapses"].includes(String(shard.kind)) ||
      typeof shard.bucket !== "string" ||
      !/^[a-f0-9]$/.test(shard.bucket) ||
      !Number.isSafeInteger(shard.part) ||
      Number(shard.part) < 0 ||
      !Number.isSafeInteger(shard.count) ||
      Number(shard.count) < 0
    ) {
      throw new Error("Persisted neural substrate shard table is invalid.");
    }
    summed[shard.kind as keyof typeof summed] += Number(shard.count);
    if (!Number.isSafeInteger(summed[shard.kind as keyof typeof summed])) {
      throw new Error("Persisted neural substrate counts exceed safe integers.");
    }
  }
  if (canonicalJson(summed) !== canonicalJson(totals)) {
    throw new Error("Persisted neural substrate shard totals are inconsistent.");
  }
  const generationBody = { ...generation };
  delete generationBody.contentSha256;
  if (sha256(canonicalJson(generationBody)) !== generation.contentSha256) {
    throw new Error("Persisted neural substrate content checksum failed.");
  }
  const runtimeCard = isRecord(metadata.runtime_card)
    ? metadata.runtime_card
    : undefined;
  if (Object.hasOwn(metadata, "neural_sequence_memory")) {
    throw new Error("Persisted legacy sequence memory is not part of native OmniCortex state.");
  }
  const parameterAccounting = persistedParameterAccounting(
    runtimeCard?.parameterAccounting,
    totals.synapses
  );
  return {
    brainId: expectedBrainId,
    revision: embedded.activeGeneration,
    source: "validated-persisted-substrate",
    totals,
    ...(parameterAccounting ? { parameterAccounting } : {})
  };
}

async function collectSubstrateSnapshot(
  engineDirectory: string,
  archivePrefix: string,
  engineMetadata: unknown,
  signal?: AbortSignal,
  onBoundary?: (filesInspected: number) => Promise<void>
): Promise<SubstrateSnapshot | undefined> {
  signal?.throwIfAborted();
  if (!isRecord(engineMetadata) || !isRecord(engineMetadata.substrate)) return undefined;
  const embedded = engineMetadata.substrate.persistence;
  if (!isRecord(embedded)) return undefined;
  const store = join(engineDirectory, "substrate");
  // Engine metadata is the commit record. The root pointer may legitimately
  // name an orphaned newer generation after a process interruption, so export
  // follows the embedded record and synthesizes the matching portable pointer.
  if (
    embedded.format !== SUBSTRATE_STORE_FORMAT ||
    !READABLE_SUBSTRATE_STORE_VERSIONS.has(embedded.formatVersion as number) ||
    typeof embedded.generationManifest !== "string" ||
    typeof embedded.generationManifestSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(embedded.generationManifestSha256)
  ) {
    throw new Error("Neural substrate pointer is invalid.");
  }
  const generationRelative = safeSubstrateRelativePath(embedded.generationManifest);
  if (!generationRelative.startsWith("generations/")) {
    throw new Error("Neural substrate generation path is invalid.");
  }
  const generationPath = join(store, ...generationRelative.split("/"));
  const generationBytes = await readFile(generationPath);
  if (sha256(generationBytes) !== embedded.generationManifestSha256) {
    throw new Error("Neural substrate generation manifest checksum failed.");
  }
  let generation: unknown;
  try {
    generation = JSON.parse(generationBytes.toString("utf8"));
  } catch {
    throw new Error("Neural substrate generation manifest is invalid.");
  }
  if (
    !isRecord(generation) ||
    generation.format !== SUBSTRATE_STORE_FORMAT ||
    generation.formatVersion !== embedded.formatVersion ||
    (generation.formatVersion === 3 && generation.schema !== "neural-substrate-2") ||
    !Array.isArray(generation.shards) ||
    typeof generation.contentSha256 !== "string" ||
    generation.contentSha256 !== embedded.activeGeneration ||
    embedded.contentSha256 !== embedded.activeGeneration ||
    !isRecord(embedded.counts) ||
    !isRecord(generation.counts) ||
    embedded.shardCount !== generation.shards.length ||
    canonicalJson(embedded.counts) !== canonicalJson(generation.counts)
  ) {
    throw new Error("Neural substrate generation manifest is incompatible.");
  }
  const content = { ...generation };
  delete content.contentSha256;
  if (sha256(canonicalJson(content)) !== generation.contentSha256) {
    throw new Error("Neural substrate generation content checksum failed.");
  }

  const declared = new Map<string, { sha256: string; bytes: number }>();
  for (const shard of generation.shards) {
    if (
      !isRecord(shard) ||
      !["neurons", "assemblies", "synapses"].includes(String(shard.kind)) ||
      typeof shard.bucket !== "string" ||
      !/^[a-f0-9]$/.test(shard.bucket) ||
      typeof shard.part !== "number" ||
      !Number.isSafeInteger(shard.part) ||
      shard.part < 0 ||
      typeof shard.count !== "number" ||
      !Number.isSafeInteger(shard.count) ||
      shard.count < 0
    ) {
      throw new Error("Neural substrate generation contains an invalid shard record.");
    }
    for (const key of ["records", "tensors"] as const) {
      const descriptor = shard[key];
      if (key === "tensors" && descriptor === null) continue;
      if (
        !isRecord(descriptor) ||
        typeof descriptor.path !== "string" ||
        typeof descriptor.sha256 !== "string" ||
        !/^[a-f0-9]{64}$/.test(descriptor.sha256) ||
        typeof descriptor.bytes !== "number" ||
        !Number.isSafeInteger(descriptor.bytes) ||
        descriptor.bytes < 0
      ) {
        throw new Error("Neural substrate generation contains an invalid blob descriptor.");
      }
      const relative = safeSubstrateRelativePath(descriptor.path);
      if (
        !relative.startsWith("blobs/") ||
        basename(relative).split(".")[0] !== descriptor.sha256
      ) {
        throw new Error("Neural substrate blob is not content-addressed by its checksum.");
      }
      const prior = declared.get(relative);
      const normalized = {
        sha256: descriptor.sha256,
        bytes: descriptor.bytes
      };
      if (prior && canonicalJson(prior) !== canonicalJson(normalized)) {
        throw new Error("Neural substrate generation contains conflicting blob descriptors.");
      }
      declared.set(relative, normalized);
    }
  }
  const sources: StreamingZipSource[] = [
    {
      name: `${archivePrefix}/manifest.json`,
      contents: Buffer.from(canonicalJson(embedded), "utf8")
    },
    {
      name: `${archivePrefix}/${generationRelative}`,
      sourcePath: generationPath
    }
  ];
  const relativePaths = new Set<string>(["manifest.json", generationRelative]);
  let filesInspected = 0;
  for (const [relative, descriptor] of [...declared].sort(([left], [right]) =>
    left.localeCompare(right)
  )) {
    signal?.throwIfAborted();
    if (filesInspected % 64 === 0) await onBoundary?.(filesInspected);
    const sourcePath = join(store, ...relative.split("/"));
    const info = await lstat(sourcePath);
    if (
      !info.isFile() ||
      info.isSymbolicLink() ||
      info.size !== descriptor.bytes ||
      (await streamFileSha256(sourcePath)) !== descriptor.sha256
    ) {
      throw new Error(`Neural substrate blob checksum failed: ${relative}`);
    }
    if (relative.endsWith(".safetensors")) {
      await assertSafeTensorsFile(sourcePath, `substrate shard ${relative}`);
    }
    sources.push({ name: `${archivePrefix}/${relative}`, sourcePath });
    relativePaths.add(relative);
    filesInspected += 1;
  }
  signal?.throwIfAborted();
  return { pointer: embedded, sources, relativePaths };
}

async function validateExtractedSubstrateSnapshot(
  archive: ExtractedZipArchive,
  archivePrefix: string,
  engineMetadata: unknown
): Promise<Set<string>> {
  const matching = [...archive.entries.keys()].filter((name) =>
    name.startsWith(`${archivePrefix}/`)
  );
  if (!isRecord(engineMetadata) || !isRecord(engineMetadata.substrate)) {
    if (matching.length > 0) throw new Error("The bundle contains undeclared substrate shards.");
    return new Set();
  }
  const embedded = engineMetadata.substrate.persistence;
  if (!isRecord(embedded)) {
    if (matching.length > 0) throw new Error("The bundle contains undeclared substrate shards.");
    return new Set();
  }
  const pointerEntry = archive.entries.get(`${archivePrefix}/manifest.json`);
  if (!pointerEntry) throw new Error("The bundle is missing its substrate pointer.");
  const pointer = JSON.parse(await readFile(pointerEntry.path, "utf8")) as unknown;
  if (!isRecord(pointer) || canonicalJson(pointer) !== canonicalJson(embedded)) {
    throw new Error("The bundled substrate pointer does not match engine state.");
  }
  if (
    embedded.format !== SUBSTRATE_STORE_FORMAT ||
    !READABLE_SUBSTRATE_STORE_VERSIONS.has(embedded.formatVersion as number) ||
    typeof embedded.generationManifest !== "string" ||
    typeof embedded.generationManifestSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(embedded.generationManifestSha256) ||
    typeof embedded.activeGeneration !== "string" ||
    !/^[a-f0-9]{64}$/.test(embedded.activeGeneration)
  ) {
    throw new Error("The bundled substrate pointer is invalid.");
  }
  const generationRelative = safeSubstrateRelativePath(String(embedded.generationManifest ?? ""));
  const generationEntry = archive.entries.get(`${archivePrefix}/${generationRelative}`);
  if (
    !generationEntry ||
    (await streamFileSha256(generationEntry.path)) !== embedded.generationManifestSha256
  ) {
    throw new Error("The bundled substrate generation manifest checksum failed.");
  }
  const generation = JSON.parse(await readFile(generationEntry.path, "utf8")) as unknown;
  if (
    !isRecord(generation) ||
    generation.format !== SUBSTRATE_STORE_FORMAT ||
    generation.formatVersion !== embedded.formatVersion ||
    (generation.formatVersion === 3 && generation.schema !== "neural-substrate-2") ||
    !Array.isArray(generation.shards) ||
    generation.contentSha256 !== embedded.activeGeneration ||
    embedded.contentSha256 !== embedded.activeGeneration ||
    !isRecord(embedded.counts) ||
    !isRecord(generation.counts) ||
    embedded.shardCount !== generation.shards.length ||
    canonicalJson(embedded.counts) !== canonicalJson(generation.counts)
  ) {
    throw new Error("The bundled substrate generation manifest is invalid.");
  }
  const generationBody = { ...generation };
  delete generationBody.contentSha256;
  if (sha256(canonicalJson(generationBody)) !== generation.contentSha256) {
    throw new Error("The bundled substrate generation content checksum failed.");
  }
  const declared = new Map<string, { sha256: string; bytes: number }>();
  for (const shard of generation.shards) {
    if (
      !isRecord(shard) ||
      !["neurons", "assemblies", "synapses"].includes(String(shard.kind)) ||
      typeof shard.bucket !== "string" ||
      !/^[a-f0-9]$/.test(shard.bucket) ||
      typeof shard.part !== "number" ||
      !Number.isSafeInteger(shard.part) ||
      shard.part < 0 ||
      typeof shard.count !== "number" ||
      !Number.isSafeInteger(shard.count) ||
      shard.count < 0
    ) {
      throw new Error("The bundled substrate shard table is invalid.");
    }
    for (const key of ["records", "tensors"] as const) {
      const descriptor = shard[key];
      if (key === "tensors" && descriptor === null) continue;
      if (
        !isRecord(descriptor) ||
        typeof descriptor.path !== "string" ||
        typeof descriptor.sha256 !== "string" ||
        !/^[a-f0-9]{64}$/.test(descriptor.sha256) ||
        typeof descriptor.bytes !== "number" ||
        !Number.isSafeInteger(descriptor.bytes) ||
        descriptor.bytes < 0
      ) {
        throw new Error("The bundled substrate blob descriptor is invalid.");
      }
      const relative = safeSubstrateRelativePath(descriptor.path);
      if (
        !relative.startsWith("blobs/") ||
        basename(relative).split(".")[0] !== descriptor.sha256
      ) {
        throw new Error("The bundled substrate blob is not content-addressed.");
      }
      const normalized = {
        sha256: descriptor.sha256,
        bytes: descriptor.bytes
      };
      const prior = declared.get(relative);
      if (prior && canonicalJson(prior) !== canonicalJson(normalized)) {
        throw new Error("The bundled substrate contains conflicting blob descriptors.");
      }
      declared.set(relative, normalized);
    }
  }
  const expected = new Set(["manifest.json", generationRelative, ...declared.keys()]);
  for (const relative of expected) {
    const entry = archive.entries.get(`${archivePrefix}/${relative}`);
    if (!entry) throw new Error(`The bundle is missing substrate file ${relative}.`);
    if (relative.startsWith("blobs/")) {
      const descriptor = declared.get(relative)!;
      if (
        entry.uncompressedBytes !== descriptor.bytes ||
        (await streamFileSha256(entry.path)) !== descriptor.sha256
      ) {
        throw new Error(`The bundled substrate blob checksum failed: ${relative}`);
      }
      if (relative.endsWith(".safetensors")) {
        await assertSafeTensorsFile(entry.path, `substrate shard ${relative}`);
      }
    }
  }
  if (matching.some((name) => !expected.has(name.slice(`${archivePrefix}/`.length)))) {
    throw new Error("The bundle contains an unlisted substrate shard file.");
  }
  return expected;
}

function safeMutableStateRelativePath(path: string): string {
  assertSafeArchivePath(path);
  if (
    path !== "manifest.json" &&
    path !== "replay.sqlite3" &&
    !/^generations\/[a-f0-9]{64}\/manifest\.json$/.test(path) &&
    !/^blobs\/[a-f0-9]{64}\.safetensors$/.test(path)
  ) {
    throw new Error(`Mutable neural state contains an unsupported path: ${path}`);
  }
  return path;
}

interface MutableStateSnapshot {
  pointer: Record<string, unknown>;
  sources: StreamingZipSource[];
  relativePaths: Set<string>;
}

function validateMutableGeneration(
  embedded: Record<string, unknown>,
  generation: unknown,
  generationText: string
): {
  generationRelative: string;
  blobs: Map<string, { sha256: string; bytes: number }>;
} {
  if (
    embedded.format !== MUTABLE_STATE_STORE_FORMAT ||
    embedded.formatVersion !== 1 ||
    typeof embedded.activeGeneration !== "string" ||
    !/^[a-f0-9]{64}$/.test(embedded.activeGeneration) ||
    embedded.contentSha256 !== embedded.activeGeneration ||
    typeof embedded.generationManifest !== "string" ||
    typeof embedded.generationManifestSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(embedded.generationManifestSha256)
  ) {
    throw new Error("Mutable neural state pointer is invalid.");
  }
  const generationRelative = safeMutableStateRelativePath(embedded.generationManifest);
  if (generationRelative !== `generations/${embedded.activeGeneration}/manifest.json`) {
    throw new Error("Mutable neural state generation path is invalid.");
  }
  if (
    !isRecord(generation) ||
    generation.format !== MUTABLE_STATE_STORE_FORMAT ||
    generation.formatVersion !== 1 ||
    generation.contentSha256 !== embedded.activeGeneration ||
    typeof generation.brainId !== "string" ||
    generation.brainId.length < 1 ||
    !isRecord(generation.roles) ||
    !isRecord(generation.replay) ||
    generation.replay.format !== "omni-replay-sqlite" ||
    generation.replay.formatVersion !== 1 ||
    generation.replay.path !== "replay.sqlite3" ||
    typeof generation.replay.count !== "number" ||
    !Number.isSafeInteger(generation.replay.count) ||
    generation.replay.count < 0 ||
    typeof generation.replay.highWaterId !== "number" ||
    !Number.isSafeInteger(generation.replay.highWaterId) ||
    generation.replay.highWaterId < 0 ||
    typeof generation.replay.contentSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(generation.replay.contentSha256)
  ) {
    throw new Error("Mutable neural state generation is incompatible.");
  }
  // The Python worker's canonical encoder preserves Python float lexemes such
  // as 1e-08, while JSON.parse followed by JSON.stringify emits 1e-8. Validate
  // canonical structure modulo that cross-runtime spelling difference, then
  // hash the original bytes with only the top-level checksum field removed.
  // Re-serializing the parsed object here would reject an otherwise valid
  // optimizer generation during Duplicate/Fork.
  if (normalizeCanonicalJsonNumbers(generationText) !== canonicalJson(generation)) {
    throw new Error("Mutable neural state generation manifest is not canonical.");
  }
  if (
    sha256(canonicalManifestContentBytes(generationText, generation.contentSha256)) !==
    generation.contentSha256
  ) {
    throw new Error("Mutable neural state generation content checksum failed.");
  }
  const blobs = new Map<string, { sha256: string; bytes: number }>();
  for (const role of ["core", "plasticity", "optimizer"] as const) {
    const descriptor = generation.roles[role];
    if (
      !isRecord(descriptor) ||
      typeof descriptor.path !== "string" ||
      typeof descriptor.sha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(descriptor.sha256) ||
      typeof descriptor.bytes !== "number" ||
      !Number.isSafeInteger(descriptor.bytes) ||
      descriptor.bytes < 0
    ) {
      throw new Error(`Mutable neural state ${role} descriptor is invalid.`);
    }
    const relative = safeMutableStateRelativePath(descriptor.path);
    if (relative !== `blobs/${descriptor.sha256}.safetensors`) {
      throw new Error("Mutable neural state blob is not content-addressed.");
    }
    const normalized = { sha256: descriptor.sha256, bytes: descriptor.bytes };
    const prior = blobs.get(relative);
    if (prior && canonicalJson(prior) !== canonicalJson(normalized)) {
      throw new Error("Mutable neural state has conflicting blob descriptors.");
    }
    blobs.set(relative, normalized);
  }
  return { generationRelative, blobs };
}

async function collectMutableStateSnapshot(
  engineDirectory: string,
  archivePrefix: string,
  engineMetadata: unknown,
  signal?: AbortSignal,
  onBoundary?: (filesInspected: number) => Promise<void>,
  replayStaging?: {
    destination: string;
    checkDisk?: (bytes: number) => Promise<void>;
  }
): Promise<MutableStateSnapshot | undefined> {
  signal?.throwIfAborted();
  if (!isRecord(engineMetadata) || !isRecord(engineMetadata.mutable_state)) {
    return undefined;
  }
  const embedded = engineMetadata.mutable_state;
  const store = join(engineDirectory, "state");
  if (typeof embedded.generationManifest !== "string") {
    throw new Error("Mutable neural state pointer is invalid.");
  }
  const generationRelative = safeMutableStateRelativePath(embedded.generationManifest);
  const generationPath = join(store, ...generationRelative.split("/"));
  const generationBytes = await readFile(generationPath);
  if (sha256(generationBytes) !== embedded.generationManifestSha256) {
    throw new Error("Mutable neural state generation manifest checksum failed.");
  }
  let generation: unknown;
  try {
    generation = JSON.parse(generationBytes.toString("utf8"));
  } catch {
    throw new Error("Mutable neural state generation manifest is invalid.");
  }
  const validated = validateMutableGeneration(
    embedded,
    generation,
    generationBytes.toString("utf8")
  );
  const sources: StreamingZipSource[] = [
    {
      name: `${archivePrefix}/manifest.json`,
      contents: Buffer.from(canonicalJson(embedded), "utf8")
    },
    {
      name: `${archivePrefix}/${validated.generationRelative}`,
      sourcePath: generationPath
    }
  ];
  const relativePaths = new Set<string>(["manifest.json", validated.generationRelative]);
  let filesInspected = 0;
  for (const [relative, descriptor] of [...validated.blobs].sort(([left], [right]) =>
    left.localeCompare(right)
  )) {
    signal?.throwIfAborted();
    await onBoundary?.(filesInspected);
    const sourcePath = join(store, ...relative.split("/"));
    const info = await lstat(sourcePath);
    if (
      !info.isFile() ||
      info.isSymbolicLink() ||
      info.size !== descriptor.bytes ||
      (await streamFileSha256(sourcePath)) !== descriptor.sha256
    ) {
      throw new Error(`Mutable neural state blob checksum failed: ${relative}`);
    }
    await assertSafeTensorsFile(sourcePath, `mutable state ${relative}`);
    sources.push({ name: `${archivePrefix}/${relative}`, sourcePath });
    relativePaths.add(relative);
    filesInspected += 1;
  }
  const replaySourcePath = join(store, "replay.sqlite3");
  const replayInfo = await lstat(replaySourcePath);
  if (!replayInfo.isFile() || replayInfo.isSymbolicLink()) {
    throw new Error("Mutable neural replay is not a regular SQLite file.");
  }
  let replayPath = replaySourcePath;
  if (replayStaging) {
    signal?.throwIfAborted();
    const walInfo = await lstat(`${replaySourcePath}-wal`).catch(
      (error: NodeJS.ErrnoException) => {
        if (error.code === "ENOENT") return undefined;
        throw error;
      }
    );
    const snapshotBytes = replayInfo.size + (walInfo?.size ?? 0);
    if (!Number.isSafeInteger(snapshotBytes)) {
      throw new Error("Mutable neural replay snapshot exceeds safe file size.");
    }
    await replayStaging.checkDisk?.(snapshotBytes);
    await ensureDiskReserve(dirname(replayStaging.destination), snapshotBytes);
    await snapshotMutableSqliteIsolated(replaySourcePath, replayStaging.destination);
    replayPath = replayStaging.destination;
    signal?.throwIfAborted();
  }
  try {
    verifyPortableReplaySqlite(
      replayPath,
      isRecord(generation) ? generation.replay : undefined
    );
  } catch {
    throw new Error("Mutable neural replay failed SQLite and checkpoint verification.");
  }
  sources.push({
    name: `${archivePrefix}/replay.sqlite3`,
    sourcePath: replayPath
  });
  relativePaths.add("replay.sqlite3");
  signal?.throwIfAborted();
  return { pointer: embedded, sources, relativePaths };
}

async function validateExtractedMutableStateSnapshot(
  archive: ExtractedZipArchive,
  archivePrefix: string,
  engineMetadata: unknown
): Promise<Set<string>> {
  const matching = [...archive.entries.keys()].filter((name) =>
    name.startsWith(`${archivePrefix}/`)
  );
  if (!isRecord(engineMetadata) || !isRecord(engineMetadata.mutable_state)) {
    if (matching.length > 0) {
      throw new Error("The bundle contains undeclared mutable neural state.");
    }
    return new Set();
  }
  const embedded = engineMetadata.mutable_state;
  const pointerEntry = archive.entries.get(`${archivePrefix}/manifest.json`);
  if (!pointerEntry) throw new Error("The bundle is missing its mutable-state pointer.");
  const pointer = JSON.parse(await readFile(pointerEntry.path, "utf8")) as unknown;
  if (!isRecord(pointer) || canonicalJson(pointer) !== canonicalJson(embedded)) {
    throw new Error("The bundled mutable-state pointer does not match engine state.");
  }
  if (typeof embedded.generationManifest !== "string") {
    throw new Error("The bundled mutable-state pointer is invalid.");
  }
  const generationRelative = safeMutableStateRelativePath(embedded.generationManifest);
  const generationEntry = archive.entries.get(`${archivePrefix}/${generationRelative}`);
  if (
    !generationEntry ||
    (await streamFileSha256(generationEntry.path)) !== embedded.generationManifestSha256
  ) {
    throw new Error("The bundled mutable-state generation checksum failed.");
  }
  const generationText = await readFile(generationEntry.path, "utf8");
  const generation = JSON.parse(generationText) as unknown;
  const validated = validateMutableGeneration(embedded, generation, generationText);
  const expected = new Set([
    "manifest.json",
    validated.generationRelative,
    "replay.sqlite3",
    ...validated.blobs.keys()
  ]);
  for (const relative of expected) {
    const entry = archive.entries.get(`${archivePrefix}/${relative}`);
    if (!entry) throw new Error(`The bundle is missing mutable-state file ${relative}.`);
    const descriptor = validated.blobs.get(relative);
    if (descriptor) {
      if (
        entry.uncompressedBytes !== descriptor.bytes ||
        (await streamFileSha256(entry.path)) !== descriptor.sha256
      ) {
        throw new Error(`The bundled mutable-state blob checksum failed: ${relative}`);
      }
      await assertSafeTensorsFile(entry.path, `mutable state ${relative}`);
    }
  }
  if (matching.some((name) => !expected.has(name.slice(`${archivePrefix}/`.length)))) {
    throw new Error("The bundle contains an unlisted mutable-state file.");
  }
  const replayEntry = archive.entries.get(`${archivePrefix}/replay.sqlite3`);
  if (!replayEntry) throw new Error("The bundle is missing mutable-state replay.");
  try {
    verifyPortableReplaySqlite(
      replayEntry.path,
      isRecord(generation) ? generation.replay : undefined
    );
  } catch {
    throw new Error("The bundled mutable replay failed SQLite and checkpoint verification.");
  }
  return expected;
}

function savedEngineState(contents: Buffer): Uint8Array {
  let state: unknown;
  try {
    state = JSON.parse(contents.toString("utf8"));
  } catch {
    throw new Error("The Python engine metadata is invalid.");
  }
  if (!isRecord(state)) throw new Error("The Python engine metadata is invalid.");
  if (state.release_format !== STABLE_RELEASE_FORMAT) {
    throw new Error(
      "Incompatible OmniCortex beta engine; stable v1 bundles require stable neural state."
    );
  }
  assertNativeOmniEngineState(state, "Exported neural state");
  if (
    (state.pending_chat_slow_learning !== undefined &&
      !Array.isArray(state.pending_chat_slow_learning)) ||
    (state.ingestion_checkpoints !== undefined && !isRecord(state.ingestion_checkpoints))
  ) {
    throw new Error("The saved neural continuation metadata is invalid.");
  }
  // Preserve even source paths, private text, and workspace annotations. The
  // OS credential vault is external and is never traversed by this exporter.
  return contents;
}

/** Carry only the exact protected real-data references sealed in engine state. */
async function savedGeometryHoldoutFiles(
  state: unknown,
  engineDirectory: string
): Promise<Map<string, string>> {
  const files = new Map<string, string>();
  if (!isRecord(state) || state.geometry_holdout_registration == null) return files;
  const registration = state.geometry_holdout_registration;
  if (!isRecord(registration) || registration.format !== "omni-geometry-holdout-registration" ||
    registration.formatVersion !== 1 || registration.manifestPath !== "evaluation/geometry-holdouts.json" ||
    typeof registration.manifestSha256 !== "string" || !/^[a-f0-9]{64}$/.test(registration.manifestSha256)) {
    throw new Error("Saved geometry holdout registration is invalid.");
  }
  const evaluation = join(engineDirectory, "evaluation");
  const data = join(evaluation, "data");
  for (const directory of [evaluation, data]) {
    const info = await lstat(directory);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Saved geometry holdout directory is unsafe.");
  }
  const manifest = join(evaluation, "geometry-holdouts.json");
  const manifestInfo = await lstat(manifest);
  if (!manifestInfo.isFile() || manifestInfo.isSymbolicLink() ||
    await fileSha256(manifest) !== registration.manifestSha256) {
    throw new Error("Saved geometry holdout manifest is missing or changed.");
  }
  const declaration: unknown = JSON.parse(await readFile(manifest, "utf8"));
  if (!isRecord(declaration) || declaration.format !== "omni-registered-geometry-holdouts" ||
    declaration.formatVersion !== 1 || !isRecord(declaration.categories) ||
    Object.keys(declaration.categories).sort().join(",") !== "modality,token,tool") {
    throw new Error("Saved geometry holdout manifest is invalid.");
  }
  files.set("evaluation/geometry-holdouts.json", manifest);
  for (const category of ["token", "modality", "tool"] as const) {
    const entries = declaration.categories[category];
    if (!Array.isArray(entries) || !entries.length) throw new Error("Saved geometry holdout category is empty.");
    for (const entry of entries) {
      if (!isRecord(entry) || typeof entry.path !== "string" ||
        !/^evaluation\/data\/[a-f0-9]{32}(?:\.[A-Za-z0-9_-]+)?$/.test(entry.path) ||
        typeof entry.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(entry.sha256) ||
        !Number.isSafeInteger(entry.records) || Number(entry.records) < 1 || files.has(entry.path)) {
        throw new Error("Saved geometry holdout file declaration is invalid.");
      }
      const source = join(engineDirectory, ...entry.path.split("/"));
      const info = await lstat(source);
      if (!info.isFile() || info.isSymbolicLink() || await fileSha256(source) !== entry.sha256) {
        throw new Error("Saved geometry holdout file is missing or changed.");
      }
      files.set(entry.path, source);
    }
  }
  return files;
}

/** Preserve the active v3 cursor's committed joint generation, not orphan stages. */
async function savedJointGenerationFiles(
  state: unknown,
  filePath: (kind: "joint" | "mutable", relative: string) => string
): Promise<Map<string, string>> {
  const files = new Map<string, string>();
  if (!isRecord(state)) return files;
  const cursors = isRecord(state.ingestion_checkpoints) ? Object.values(state.ingestion_checkpoints) : [];
  const v3 = cursors.filter((cursor) => isRecord(cursor) && cursor.formatVersion === 3);
  const reference = state.ingestion_joint_generation;
  if (reference === undefined || reference === null) {
    if (v3.length) throw new Error("Saved v3 ingestion cursor has no committed joint generation.");
    return files;
  }
  if (!isRecord(reference) || reference.format !== "omni-joint-checkpoint-reference" ||
    reference.formatVersion !== 2 || typeof reference.generationId !== "string" ||
    !/^[a-f0-9]{32}$/.test(reference.generationId) ||
    reference.relativeManifest !== `generations/${reference.generationId}/manifest.json` ||
    typeof reference.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(reference.sha256) ||
    v3.length !== 1 || cursors.length !== 1) {
    throw new Error("Saved ingestion joint-generation reference is invalid.");
  }
  const relative = String(reference.relativeManifest);
  const manifestPath = filePath("joint", relative);
  const info = await lstat(manifestPath);
  if (!info.isFile() || info.isSymbolicLink() || info.size > 64 * 1024 ||
    await streamFileSha256(manifestPath) !== reference.sha256) {
    throw new Error("Saved ingestion joint manifest checksum failed.");
  }
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as unknown;
  const cursor = v3[0] as Record<string, unknown>;
  const mutable = state.mutable_state;
  const substrate = isRecord(state.substrate) ? state.substrate.persistence : undefined;
  if (!isRecord(manifest) || manifest.format !== "omni-joint-checkpoint-generation" ||
    manifest.formatVersion !== 2 || manifest.generationId !== reference.generationId ||
    !isRecord(manifest.neuralState) || !isRecord(manifest.substrateState) ||
    !isRecord(mutable) || !isRecord(substrate) || !isRecord(manifest.cursor) ||
    !isRecord(manifest.coverage) ||
    manifest.neuralState.generationId !== mutable.activeGeneration ||
    manifest.neuralState.generationManifest !== mutable.generationManifest ||
    manifest.neuralState.generationManifestSha256 !== mutable.generationManifestSha256 ||
    manifest.neuralGenerationSha256 !== mutable.activeGeneration ||
    manifest.substrateState.generationId !== substrate.activeGeneration ||
    manifest.substrateState.generationManifest !== substrate.generationManifest ||
    manifest.substrateState.generationManifestSha256 !== substrate.generationManifestSha256 ||
    manifest.substrateGenerationSha256 !== substrate.activeGeneration ||
    canonicalJson(manifest.substrateState.counts) !== canonicalJson(substrate.counts) ||
    manifest.checkpointSequence !== cursor.commitSequence ||
    !Number.isSafeInteger(manifest.checkpointSequence) || Number(manifest.checkpointSequence) < 1 ||
    manifest.cursor.committedRecords !== cursor.committedRecords ||
    manifest.cursor.recordPrefixSha256 !== cursor.recordPrefixSha256 ||
    manifest.sourceManifestSha256 !== cursor.sourceManifestSha256 ||
    manifest.parserManifestSha256 !== cursor.parserManifestSha256 ||
    manifest.sourceContentSha256 !== cursor.contentHash ||
    manifest.sourceParserManifestSha256 !== sha256(canonicalJson({
      sourceManifestSha256: cursor.sourceManifestSha256,
      parserManifestSha256: cursor.parserManifestSha256
    }))) {
    throw new Error("Saved ingestion joint generation does not bind its cursor and neural state.");
  }
  for (const hash of [manifest.sourceManifestSha256, manifest.parserManifestSha256,
    manifest.sourceContentSha256, manifest.cursor.recordPrefixSha256,
    manifest.previousManifestSha256 ?? "0".repeat(64)]) {
    if (typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash)) {
      throw new Error("Saved ingestion joint manifest contains an invalid hash.");
    }
  }
  const coverage = manifest.coverage;
  for (const number of [manifest.cursor.committedRecords, coverage.visitedRecords,
    coverage.processedRecords, coverage.rejectedRecords, coverage.processedBytes,
    coverage.expectedRecords ?? 0]) {
    if (!Number.isSafeInteger(number) || Number(number) < 0) {
      throw new Error("Saved ingestion joint coverage is invalid.");
    }
  }
  if (coverage.visitedRecords !== Number(coverage.processedRecords) + Number(coverage.rejectedRecords) ||
    Number(manifest.cursor.committedRecords) > Number(coverage.visitedRecords) ||
    (coverage.expectedRecords !== null && Number(coverage.visitedRecords) > Number(coverage.expectedRecords)) ||
    typeof coverage.sourceStreamExhausted !== "boolean" ||
    (coverage.sourceStreamExhausted
      ? ((coverage.expectedRecords !== null && coverage.visitedRecords !== coverage.expectedRecords) ||
        coverage.sourceContentReverifiedSha256 !== manifest.sourceContentSha256)
      : coverage.sourceContentReverifiedSha256 !== null)) {
    throw new Error("Saved ingestion joint coverage is incomplete.");
  }
  const neuralManifest = JSON.parse(await readFile(
    filePath("mutable", safeMutableStateRelativePath(String(mutable.generationManifest))), "utf8"
  )) as Record<string, unknown>;
  if (!isRecord(neuralManifest.roles) || !isRecord(manifest.neuralState.blobs)) {
    throw new Error("Saved ingestion joint neural roles are invalid.");
  }
  for (const role of ["core", "plasticity", "optimizer"]) {
    const descriptor = neuralManifest.roles[role];
    if (!isRecord(descriptor) || canonicalJson(manifest.neuralState.blobs[role]) !== canonicalJson({
      path: descriptor.path, sha256: descriptor.sha256, bytes: descriptor.bytes
    })) throw new Error("Saved ingestion joint neural blob binding failed.");
  }
  files.set(relative, manifestPath);
  if (manifest.sqliteSnapshot !== null) {
    const snapshot = manifest.sqliteSnapshot;
    if (!isRecord(snapshot) || snapshot.file !== "packed-vector-index.sqlite3" ||
      typeof snapshot.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(snapshot.sha256) ||
      !Number.isSafeInteger(snapshot.bytes) || Number(snapshot.bytes) < 1) {
      throw new Error("Saved ingestion joint SQLite descriptor is invalid.");
    }
    const snapshotRelative = `generations/${reference.generationId}/${snapshot.file}`;
    const path = filePath("joint", snapshotRelative);
    const snapshotInfo = await lstat(path);
    if (!snapshotInfo.isFile() || snapshotInfo.isSymbolicLink() || snapshotInfo.size !== snapshot.bytes ||
      await streamFileSha256(path) !== snapshot.sha256) {
      throw new Error("Saved ingestion joint SQLite checksum failed.");
    }
    const database = new DatabaseSync(path, { readOnly: true });
    try {
      if (database.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") {
        throw new Error("Saved ingestion joint SQLite integrity failed.");
      }
    } finally { database.close(); }
    files.set(snapshotRelative, path);
  }
  return files;
}

function assertImportedNeuralConversationHead(
  engineState: unknown,
  label: string,
  hasLedger: boolean
): void {
  const state = isRecord(engineState) ? engineState : undefined;
  const summary = isRecord(state?.conversation) ? state.conversation : undefined;
  // Early stable checkpoints may omit an empty native ledger summary.
  if (state?.format === "omni-engine-unmaterialized" ||
    (summary === undefined && !hasLedger)) return;
  const countFields = [
    "totalEntries", "messageCount", "actionCount", "traceCount",
    "attentionEpoch", "headSequence"
  ] as const;
  if (
    summary?.format !== "omni-neural-conversation-ledger" ||
    summary.formatVersion !== 1 ||
    countFields.some((field) =>
      !Number.isSafeInteger(summary[field]) || Number(summary[field]) < 0
    ) ||
    typeof summary.headSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(summary.headSha256)
  ) {
    throw new Error(`${label} has an invalid native conversation head.`);
  }
  if (!hasLedger && (
    countFields.some((field) => Number(summary[field]) !== 0) ||
    summary.headSha256 !== "0".repeat(64)
  )) {
    throw new Error(`${label} claims conversation rows but omits their neural ledger.`);
  }
}


function requireSafeId(id: string, label = "brain id"): string {
  if (!SAFE_ID.test(id)) {
    throw new Error(`Invalid ${label}.`);
  }
  return id;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function boundNumber(value: unknown, fallback: number, minimum: number, maximum: number): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(minimum, Math.min(maximum, value))
    : fallback;
}

function roundedSafeNonnegativeInteger(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? Math.max(1, Math.min(Number.MAX_SAFE_INTEGER, Math.round(value)))
    : 0;
}

function normalizeConfig(value: unknown): BrainConfig {
  const config = isRecord(value) ? value : {};
  const merged = { ...DEFAULT_CONFIG } as BrainConfig;
  for (const key of Object.keys(DEFAULT_CONFIG)) {
    if (config[key] !== undefined) {
      (merged as unknown as Record<string, unknown>)[key] = config[key];
    }
  }
  merged.runtime = "adaptive-core";
  if (
    !["whole-brain", "ternary", "neuromorphic", "liquid", "symbolic", "custom"].includes(
      merged.preset
    )
  ) {
    merged.preset = "whole-brain";
  }
  if (!["summary", "standard", "research"].includes(merged.traceDetail)) {
    merged.traceDetail = "standard";
  }
  if (!["auto", "extended", "manual"].includes(merged.workingMemoryMode)) {
    merged.workingMemoryMode = "auto";
  }
  if (!["auto", "manual"].includes(merged.systemRamMode)) {
    merged.systemRamMode = "auto";
  }
  if (!["auto", "manual"].includes(merged.storagePoolMode)) {
    merged.storagePoolMode = "auto";
  }
  const importedMemoryRecipe = String(
    (merged as unknown as Record<string, unknown>).memoryRecipe ?? ""
  );
  if (["human", "human-consolidation"].includes(importedMemoryRecipe)) {
    // Read aliases remain accepted for stable-v1 bundles, but all newly
    // materialized public metadata uses the capability-oriented name.
    merged.memoryRecipe = "adaptive-retention";
  } else if (
    !["adaptive-retention", "total-recall", "synapses-only"].includes(
      importedMemoryRecipe
    )
  ) {
    merged.memoryRecipe = "adaptive-retention";
  }
  for (const key of [
    "onlineLearning",
    "extendedWorkingMemory",
    "recursiveImprovement",
    "idleCognition",
    "retainSourceText"
  ] as const) {
    if (typeof merged[key] !== "boolean") merged[key] = DEFAULT_CONFIG[key];
  }
  // This desktop field controls only optional prompt-free scheduling. The
  // engine keeps recurrent Ponder available regardless of this preference.
  merged.name =
    typeof merged.name === "string" && merged.name.trim()
      ? merged.name.trim().slice(0, 120)
      : DEFAULT_CONFIG.name;
  merged.description =
    typeof merged.description === "string"
      ? merged.description.replace(/\0/g, "").slice(0, 4_000)
      : DEFAULT_CONFIG.description;
  merged.workingMemorySlots =
    typeof merged.workingMemorySlots === "number" &&
    Number.isSafeInteger(merged.workingMemorySlots) &&
    merged.workingMemorySlots > 0
      ? merged.workingMemorySlots
      : DEFAULT_CONFIG.workingMemorySlots;
  if (config.nativeArchitecture !== undefined) {
    const descriptor = validateNativeArchitectureDescriptor(config.nativeArchitecture);
    if (descriptor.shape.workingMemoryItems !== merged.workingMemorySlots) {
      throw new Error("Saved native architecture does not match its neural workspace shape.");
    }
    merged.nativeArchitecture = descriptor;
  }
  merged.contextWindowTokens =
    typeof merged.contextWindowTokens === "number" &&
    Number.isSafeInteger(merged.contextWindowTokens) &&
    merged.contextWindowTokens >= 8
      ? merged.contextWindowTokens
      : DEFAULT_CONFIG.contextWindowTokens;
  merged.memoryOffloadBytes =
    typeof merged.memoryOffloadBytes === "number" &&
    Number.isSafeInteger(merged.memoryOffloadBytes) &&
    merged.memoryOffloadBytes >= 0
      ? merged.memoryOffloadBytes
      : 0;
  merged.memoryResidentItems =
    typeof merged.memoryResidentItems === "number" &&
    Number.isSafeInteger(merged.memoryResidentItems) &&
    merged.memoryResidentItems > 0
      ? Math.min(merged.memoryResidentItems, merged.workingMemorySlots)
      : Math.min(DEFAULT_CONFIG.memoryResidentItems, merged.workingMemorySlots);
  merged.memoryOffloadSlowdownPercent = boundNumber(
    merged.memoryOffloadSlowdownPercent,
    0,
    0,
    95
  );
  merged.systemRamSharePercent = merged.systemRamMode === "manual"
    ? boundNumber(merged.systemRamSharePercent, 65, 30, 100)
    : 0;
  merged.storagePoolBytes =
    typeof merged.storagePoolBytes === "number" &&
    Number.isSafeInteger(merged.storagePoolBytes) &&
    merged.storagePoolBytes >= 0
      ? merged.storagePoolBytes
      : 0;
  if (config.contextOffloadBudgetBytes !== undefined) {
    if (!Number.isSafeInteger(config.contextOffloadBudgetBytes) || Number(config.contextOffloadBudgetBytes) < 0) {
      throw new Error("Context offload budget must be a nonnegative exact byte count.");
    }
    merged.contextOffloadBudgetBytes = Number(config.contextOffloadBudgetBytes);
  }
  if (merged.storagePoolMode === "manual" && merged.storagePoolBytes < 1) {
    merged.storagePoolMode = "auto";
  }
  merged.storageBytesPerSecond = roundedSafeNonnegativeInteger(
    merged.storageBytesPerSecond
  );
  merged.learningRate = boundNumber(merged.learningRate, DEFAULT_CONFIG.learningRate, 0, 1);
  if (merged.memoryRecipe === "synapses-only") merged.retainSourceText = false;
  if (merged.memoryRecipe === "total-recall") merged.retainSourceText = true;
  return merged;
}

function normalizedBrainProvenance(value: unknown): BrainProvenance | undefined {
  if (!isRecord(value)) return undefined;
  const originKind = ["ground-up", "legacy-hybrid", "legacy"].includes(
    String(value.originKind)
  )
    ? value.originKind as BrainProvenance["originKind"]
    : undefined;
  if (!originKind) return undefined;
  const rawFoundation = isRecord(value.foundation) ? value.foundation : undefined;
  const modelId = typeof rawFoundation?.modelId === "string" &&
    /^[a-zA-Z0-9._-]{1,128}$/.test(rawFoundation.modelId)
      ? rawFoundation.modelId
      : undefined;
  const repository = typeof rawFoundation?.repository === "string"
    ? rawFoundation.repository.replace(/\0/g, "").trim().slice(0, 300)
    : undefined;
  const foundation = originKind === "legacy-hybrid" && modelId &&
    rawFoundation?.frozen === true
      ? {
          modelId,
          ...(repository ? { repository } : {}),
          frozen: true as const
        }
      : undefined;
  const rawInitialization = isRecord(value.randomInitialization)
    ? value.randomInitialization
    : undefined;
  const algorithm = typeof rawInitialization?.algorithm === "string"
    ? rawInitialization.algorithm.replace(/\0/g, "").trim().slice(0, 128)
    : "";
  const seed = Number.isSafeInteger(rawInitialization?.seed) &&
    Number(rawInitialization?.seed) >= 0
      ? Number(rawInitialization?.seed)
      : undefined;
  return {
    originKind: foundation ? "legacy-hybrid" : originKind === "legacy-hybrid" ? "legacy" : originKind,
    ...(foundation ? { foundation } : {}),
    ...(algorithm
      ? { randomInitialization: { algorithm, ...(seed === undefined ? {} : { seed }) } }
      : {})
  };
}

function brainProvenanceFromEngineMetadata(value: unknown): BrainProvenance | undefined {
  if (!isRecord(value)) return undefined;
  const config = isRecord(value.config) ? value.config : undefined;
  const runtime = isRecord(value.runtime_card) ? value.runtime_card : undefined;
  const origin = String(runtime?.origin_kind ?? config?.origin_kind ?? "");
  const cortex = isRecord(runtime?.pretrained_text_cortex)
    ? runtime.pretrained_text_cortex
    : undefined;
  const adapter = isRecord(cortex?.adapter) ? cortex.adapter : undefined;
  const configuredFoundation = [runtime?.foundationModelId, config?.foundation_model_id]
    .find((candidate) =>
      typeof candidate === "string" &&
      !["", "none", "auto"].includes(candidate.trim())
    );
  const modelId = typeof cortex?.id === "string" &&
    /^[a-zA-Z0-9._-]{1,128}$/.test(cortex.id)
      ? cortex.id
      : typeof configuredFoundation === "string" &&
          /^[a-zA-Z0-9._-]{1,128}$/.test(configuredFoundation.trim())
        ? configuredFoundation.trim()
        : undefined;
  const repository = typeof cortex?.repository === "string" ? cortex.repository : undefined;
  // Positive frozen-foundation evidence is stronger than a stale origin
  // label. Never allow a rewritten ground-up field to hide imported weights.
  if (
    modelId &&
    (adapter?.baseFrozen === true || runtime?.baseFrozen === true ||
      configuredFoundation !== undefined)
  ) {
    return normalizedBrainProvenance({
      originKind: "legacy-hybrid",
      foundation: { modelId, repository, frozen: true }
    });
  }
  if (origin === "ground-up") {
    if (runtime?.pretrained === true || runtime?.baseFrozen === true) {
      return { originKind: "legacy" };
    }
    const initialization = isRecord(runtime?.randomInitialization)
      ? runtime.randomInitialization
      : undefined;
    return normalizedBrainProvenance({
      originKind: "ground-up",
      randomInitialization: initialization
    });
  }
  if (origin === "starter" || origin === "blank") {
    return { originKind: "legacy" };
  }
  return undefined;
}

async function readEngineBrainProvenance(
  engineDirectory: string,
  expectedBrainId: string
): Promise<BrainProvenance | undefined> {
  const metadataPath = join(engineDirectory, "brain.json");
  try {
    const info = await lstat(metadataPath);
    if (!info.isFile() || info.isSymbolicLink()) return undefined;
    const metadata = JSON.parse(await readFile(metadataPath, "utf8")) as unknown;
    if (!isRecord(metadata) || metadata.brain_id !== expectedBrainId) return undefined;
    return brainProvenanceFromEngineMetadata(metadata);
  } catch {
    return undefined;
  }
}

function normalizedActivitySummary(value: unknown): BrainActivityLedgerSummary | undefined {
  if (!isRecord(value) || value.format !== "omni-brain-activity-ledger" || value.formatVersion !== 1) {
    return undefined;
  }
  const integerFields = [
    "journalCount",
    "journalHeadSequence",
    "trainingSourceCount",
    "trainingSourceVersionCount",
    "trainingSourceHeadSequence",
    "trainingSourceBytes",
    "learnedIdeas",
    "learnedConcepts",
    "learnedSynapses",
    "learnedRecords",
    "learnedParameterSteps",
    "parametersChangedSources"
  ] as const;
  if (
    integerFields.some((field) =>
      !Number.isSafeInteger(value[field]) || Number(value[field]) < 0
    ) ||
    value.journalCount !== value.journalHeadSequence ||
    value.trainingSourceVersionCount !== value.trainingSourceHeadSequence ||
    Number(value.trainingSourceCount) > Number(value.trainingSourceVersionCount) ||
    typeof value.journalHeadSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(value.journalHeadSha256) ||
    typeof value.trainingSourceHeadSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(value.trainingSourceHeadSha256)
  ) {
    return undefined;
  }
  const top = isRecord(value.topAdaptation) &&
    typeof value.topAdaptation.sourceId === "string" &&
    typeof value.topAdaptation.sourceLabel === "string" &&
    Number.isSafeInteger(value.topAdaptation.learnedRecords) &&
    Number(value.topAdaptation.learnedRecords) > 0
      ? {
          sourceId: value.topAdaptation.sourceId,
          sourceLabel: value.topAdaptation.sourceLabel,
          learnedRecords: Number(value.topAdaptation.learnedRecords)
        }
      : undefined;
  return {
    format: "omni-brain-activity-ledger",
    formatVersion: 1,
    journalCount: Number(value.journalCount),
    journalHeadSequence: Number(value.journalHeadSequence),
    journalHeadSha256: value.journalHeadSha256,
    trainingSourceCount: Number(value.trainingSourceCount),
    trainingSourceVersionCount: Number(value.trainingSourceVersionCount),
    trainingSourceHeadSequence: Number(value.trainingSourceHeadSequence),
    trainingSourceHeadSha256: value.trainingSourceHeadSha256,
    trainingSourceBytes: Number(value.trainingSourceBytes),
    learnedIdeas: Number(value.learnedIdeas),
    learnedConcepts: Number(value.learnedConcepts),
    learnedSynapses: Number(value.learnedSynapses),
    learnedRecords: Number(value.learnedRecords),
    learnedParameterSteps: Number(value.learnedParameterSteps),
    parametersChangedSources: Number(value.parametersChangedSources),
    ...(top ? { topAdaptation: top } : {})
  };
}

function normalizeBrain(value: unknown): BrainDocument {
  if (!isRecord(value)) throw new Error("The bundle does not contain a brain document.");
  if (value.releaseFormat !== STABLE_RELEASE_FORMAT) {
    throw new Error("Incompatible Omni AGI Studio beta brain; create or import a stable v1 brain.");
  }
  const id = requireSafeId(String(value.id ?? ""));
  const now = new Date().toISOString();
  const lineageValue = isRecord(value.lineage) ? value.lineage : {};
  const countersValue = isRecord(value.counters) ? value.counters : {};
  const liquidValue = isRecord(value.liquidState) ? value.liquidState : {};
  const readinessValue = isRecord(value.readiness) ? value.readiness : undefined;
  const readinessState = readinessValue?.state === "initializing"
    ? "initializing"
    : readinessValue?.state === "failed"
      ? "failed"
      : "ready";
  const readinessStartedAt =
    typeof readinessValue?.startedAt === "string" &&
    Number.isFinite(Date.parse(readinessValue.startedAt))
      ? new Date(readinessValue.startedAt).toISOString()
      : typeof value.createdAt === "string" && Number.isFinite(Date.parse(value.createdAt))
        ? new Date(value.createdAt).toISOString()
        : now;
  const readinessCompletedAt =
    readinessState === "ready" &&
    typeof readinessValue?.completedAt === "string" &&
    Number.isFinite(Date.parse(readinessValue.completedAt))
      ? new Date(readinessValue.completedAt).toISOString()
      : readinessState === "ready"
        ? readinessStartedAt
        : undefined;
  const readinessFailureValue = isRecord(readinessValue?.failure)
    ? readinessValue.failure
    : undefined;
  const readinessFailedAt =
    readinessState === "failed" &&
    typeof readinessFailureValue?.failedAt === "string" &&
    Number.isFinite(Date.parse(readinessFailureValue.failedAt))
      ? new Date(readinessFailureValue.failedAt).toISOString()
      : readinessState === "failed"
        ? readinessStartedAt
        : undefined;
  const readinessFailure = readinessState === "failed"
    ? {
        phase: readinessFailureValue?.phase === "initial-learning"
          ? "initial-learning" as const
          : "foundation" as const,
        message:
          typeof readinessFailureValue?.message === "string"
            ? readinessFailureValue.message.replace(/\0/g, "").trim().slice(0, 2_000) ||
              "Initialization failed."
            : "Initialization failed.",
        failedAt: readinessFailedAt!,
        retryable: true as const
      }
    : undefined;
  const readinessRecoveryValue = isRecord(readinessValue?.recovery)
    ? readinessValue.recovery
    : undefined;
  const recoveryFoundation = isRecord(readinessRecoveryValue?.foundation)
    ? readinessRecoveryValue.foundation
    : undefined;
  const recoveryResources = Array.isArray(readinessRecoveryValue?.resources)
    ? readinessRecoveryValue.resources
    : undefined;
  const readinessRecovery =
    recoveryFoundation &&
    ["micro", "personal", "gpu", "workstation"].includes(
      String(recoveryFoundation.hardwareTier)
    ) &&
    Array.isArray(recoveryFoundation.modalities) &&
    recoveryFoundation.modalities.every((value) =>
      ["vision", "image", "audio", "video"].includes(String(value))
    ) &&
    recoveryFoundation.origin === "ground-up" &&
    !Object.hasOwn(recoveryFoundation, "foundationModelId") &&
    recoveryResources &&
    recoveryResources.every((resource) =>
      isRecord(resource) &&
      ((resource.kind === "selection" &&
        typeof resource.selectionId === "string" &&
        SAFE_ID.test(resource.selectionId)) ||
        (resource.kind === "web" &&
          typeof resource.url === "string" &&
          resource.url.length <= 8_192 &&
          !resource.url.includes("\0")))
    )
      ? {
          foundation: {
            hardwareTier: recoveryFoundation.hardwareTier as "micro" | "personal" | "gpu" | "workstation",
            modalities: [...new Set(recoveryFoundation.modalities.map(String))] as Array<"vision" | "image" | "audio" | "video">,
            origin: "ground-up" as const
          },
          resources: recoveryResources.map((resource) =>
            resource.kind === "selection"
              ? { kind: "selection" as const, selectionId: String(resource.selectionId) }
              : { kind: "web" as const, url: String(resource.url) }
          )
        }
      : undefined;
  const provenance = normalizedBrainProvenance(value.provenance);
  const conversationValue = isRecord(value.conversation)
    ? value.conversation
    : undefined;
  const conversation = conversationValue &&
    conversationValue.format === "omni-conversation-ledger" &&
    conversationValue.formatVersion === 1 &&
    [
      conversationValue.totalEntries,
      conversationValue.messageCount,
      conversationValue.actionCount,
      conversationValue.traceCount,
      conversationValue.headSequence,
      conversationValue.attentionEpoch
    ].every((entry) => Number.isSafeInteger(entry) && Number(entry) >= 0) &&
    typeof conversationValue.headSha256 === "string" &&
    /^[a-f0-9]{64}$/.test(conversationValue.headSha256)
      ? conversationValue as unknown as NonNullable<BrainDocument["conversation"]>
      : undefined;
  const activity = normalizedActivitySummary(value.activity);

  const brain: BrainDocument = {
    schemaVersion: BRAIN_SCHEMA_VERSION,
    releaseFormat: STABLE_RELEASE_FORMAT,
    id,
    name:
      typeof value.name === "string"
        ? value.name.trim().slice(0, 120) || "Imported mind"
        : "Imported mind",
    createdAt: typeof value.createdAt === "string" ? value.createdAt : now,
    updatedAt: typeof value.updatedAt === "string" ? value.updatedAt : now,
    readiness: {
      state: readinessState,
      startedAt: readinessStartedAt,
      attempt: Math.max(
        1,
        Math.round(boundNumber(readinessValue?.attempt, 1, 1, 1_000_000))
      ),
      ...(readinessCompletedAt ? { completedAt: readinessCompletedAt } : {}),
      ...(readinessFailure ? { failure: readinessFailure } : {}),
      ...(readinessRecovery ? { recovery: readinessRecovery } : {})
    },
    ...(provenance ? { provenance } : {}),
    lineage: {
      parentId:
        typeof lineageValue.parentId === "string" && SAFE_ID.test(lineageValue.parentId)
          ? lineageValue.parentId
          : undefined,
      rootId:
        typeof lineageValue.rootId === "string" && SAFE_ID.test(lineageValue.rootId)
          ? lineageValue.rootId
          : id,
      generation: Math.max(0, Math.round(boundNumber(lineageValue.generation, 0, 0, 1_000_000)))
    },
    config: normalizeConfig(value.config),
    concepts: isRecord(value.concepts) ? (value.concepts as BrainDocument["concepts"]) : {},
    synapses: isRecord(value.synapses) ? (value.synapses as BrainDocument["synapses"]) : {},
    ideas: Array.isArray(value.ideas) ? (value.ideas as BrainDocument["ideas"]) : [],
    workingMemory: Array.isArray(value.workingMemory)
      ? (value.workingMemory as BrainDocument["workingMemory"])
      : [],
    liquidState: {
      values: Array.isArray(liquidValue.values)
        ? liquidValue.values.filter(
            (entry): entry is number => typeof entry === "number" && Number.isFinite(entry)
          )
        : Array.from({ length: 16 }, () => 0),
      timeConstants: Array.isArray(liquidValue.timeConstants)
        ? liquidValue.timeConstants.filter(
            (entry): entry is number => typeof entry === "number" && Number.isFinite(entry)
          )
        : Array.from({ length: 16 }, (_, index) => 0.25 + index * 0.05),
      lastUpdatedAt: typeof liquidValue.lastUpdatedAt === "string" ? liquidValue.lastUpdatedAt : now
    },
    messages: Array.isArray(value.messages) ? (value.messages as BrainDocument["messages"]) : [],
    traces: Array.isArray(value.traces) ? (value.traces as BrainDocument["traces"]) : [],
    ...(conversation ? { conversation } : {}),
    ...(activity ? { activity } : {}),
    trainingSources: Array.isArray(value.trainingSources)
      ? (value.trainingSources as BrainDocument["trainingSources"])
      : [],
    counters: {
      plasticityEvents: Math.max(
        0,
        Math.round(boundNumber(countersValue.plasticityEvents, 0, 0, Number.MAX_SAFE_INTEGER))
      ),
      inferenceCount: Math.max(
        0,
        Math.round(boundNumber(countersValue.inferenceCount, 0, 0, Number.MAX_SAFE_INTEGER))
      ),
      consolidationCycles: Math.max(
        0,
        Math.round(boundNumber(countersValue.consolidationCycles, 0, 0, Number.MAX_SAFE_INTEGER))
      )
    },
    toolPermissions: Array.isArray(value.toolPermissions)
      ? (value.toolPermissions as ToolPermissionRecord[])
      : clone(DEFAULT_TOOL_PERMISSIONS),
    journal: Array.isArray(value.journal) ? (value.journal as BrainDocument["journal"]) : [],
    originChecksum: typeof value.originChecksum === "string" ? value.originChecksum : undefined
  };
  brain.config.name = brain.name;
  return brain;
}

function originChecksumFor(brain: BrainDocument | Record<string, unknown>): string {
  // Readiness is an operational launch gate, not ancestral neural identity.
  // Excluding it also preserves origin verification for stable-v1 brains that
  // were created before the durable gate field existed.
  const {
    readiness: _readiness,
    provenance: _provenance,
    ...originState
  } = brain;
  return sha256(JSON.stringify({ ...originState, originChecksum: undefined }));
}

function assertOriginChecksum(value: unknown, label: string): string {
  if (!isRecord(value)) {
    throw new Error(`${label} is not a valid immutable state document.`);
  }
  // Verify the serialized state before normalizeBrain() applies newer defaults
  // or migrates compatibility aliases. The checksum binds the immutable state
  // that was actually written, so schema evolution must not make a legitimate
  // legacy origin unverifiable.
  const expected = originChecksumFor(value);
  if (value.originChecksum !== expected) {
    throw new Error(`${label} checksum does not match its immutable state.`);
  }
  return expected;
}

function assertUiNeuralOriginIdentity(originBrain: BrainDocument, originEngine: unknown): void {
  if (
    !isRecord(originEngine) ||
    originEngine.format !== "omni-cortex-engine" ||
    originEngine.release_format !== STABLE_RELEASE_FORMAT
  ) {
    return;
  }
  const engineConfig = isRecord(originEngine.config) ? originEngine.config : undefined;
  if (
    originEngine.brain_id !== originBrain.id ||
    originEngine.name !== originBrain.name ||
    (engineConfig?.name !== undefined && engineConfig.name !== originBrain.name)
  ) {
    throw new Error("The immutable UI origin does not match the immutable neural origin.");
  }
}

/** Reject imported weights from every non-native origin before installing files. */
export function assertNativeOmniEngineState(value: unknown, label: string): void {
  const state = isRecord(value) ? value : undefined;
  const config = isRecord(state?.config) ? state.config : undefined;
  const runtime = isRecord(state?.runtime_card) ? state.runtime_card : undefined;
  const packed = isRecord(state?.packed_ternary_manifest)
    ? state.packed_ternary_manifest
    : undefined;
  if (
    state?.format !== "omni-cortex-engine" ||
    state?.release_format !== STABLE_RELEASE_FORMAT ||
    config?.origin_kind !== "ground-up" ||
    Object.hasOwn(config ?? {}, "foundation_model_id") ||
    Object.hasOwn(config ?? {}, "foundationModelId") ||
    runtime?.origin_kind !== "ground-up" ||
    runtime?.pretrained !== false ||
    runtime?.baseFrozen !== false ||
    Object.hasOwn(runtime ?? {}, "foundationModelId") ||
    Object.hasOwn(state ?? {}, "starter_training_manifest") ||
    Object.hasOwn(runtime ?? {}, "starter_training_manifest") ||
    Object.hasOwn(runtime ?? {}, "pretrained_text_cortex") ||
    Object.hasOwn(state ?? {}, "foundation_cortex") ||
    Object.hasOwn(state ?? {}, "neural_sequence_memory") ||
    (state?.messages !== undefined && (
      !Array.isArray(state.messages) || state.messages.length > 0
    )) ||
    (state?.traces !== undefined && (
      !Array.isArray(state.traces) || state.traces.length > 0
    )) ||
    (packed !== undefined && packed.baseFrozen !== false) ||
    Object.hasOwn(packed ?? {}, "foundationModelId") ||
    Boolean(packed?.pretrainedTextCortex)
  ) {
    throw new Error(`${label} must contain only a locally initialized OmniCortex origin.`);
  }
}

export function brainMetrics(brain: BrainDocument): BrainMetrics {
  const synapses = Object.values(brain.synapses);
  const estimatedBytes = Buffer.byteLength(JSON.stringify(brain), "utf8");
  return {
    concepts: Object.keys(brain.concepts).length,
    synapses: synapses.length,
    activeSynapses: synapses.filter((synapse) => synapse.effectiveWeight !== 0).length,
    ideas: brain.ideas.length,
    messages: brain.conversation?.messageCount ?? brain.messages.length,
    trainingSources: brain.activity?.trainingSourceCount ?? brain.trainingSources.length,
    averageStability:
      synapses.length === 0
        ? 0
        : synapses.reduce((sum, synapse) => sum + synapse.stability, 0) / synapses.length,
    plasticityEvents: brain.counters.plasticityEvents,
    inferenceCount: brain.counters.inferenceCount,
    estimatedBytes
  };
}

function recoveryPointMetrics(
  brain: BrainDocument,
  persisted: PersistedSubstrateOverview | undefined,
  estimatedBytes: number
): BrainMetrics {
  const fallback = brainMetrics(brain);
  if (!persisted) return { ...fallback, estimatedBytes };
  const synapses =
    persisted.parameterAccounting?.dynamicSparseSynapses ??
    persisted.totals.synapses;
  return {
    ...fallback,
    concepts: persisted.totals.neurons,
    ideas: persisted.totals.assemblies,
    synapses,
    activeSynapses: Math.min(fallback.activeSynapses, synapses),
    estimatedBytes,
    ...(persisted.parameterAccounting
      ? { parameterAccounting: persisted.parameterAccounting }
      : {})
  };
}

async function snapshotStorageUsage(
  root: string,
  signal?: AbortSignal
): Promise<{
  files: number;
  logicalBytes: number;
  sharedBytes: number;
  physicalBytesAdded: number;
}> {
  const queue = [root];
  let files = 0;
  let logicalBytes = 0;
  let sharedBytes = 0;
  let physicalBytesAdded = 0;
  while (queue.length > 0) {
    signal?.throwIfAborted();
    const current = queue.shift()!;
    for (const entry of await readdir(current, { withFileTypes: true })) {
      signal?.throwIfAborted();
      const path = join(current, entry.name);
      const info = await lstat(path);
      if (info.isSymbolicLink()) {
        throw new Error("Recovery point materialization contains a symbolic link.");
      }
      if (info.isDirectory()) {
        queue.push(path);
        continue;
      }
      if (!info.isFile()) continue;
      files += 1;
      logicalBytes += info.size;
      if (info.nlink > 1) sharedBytes += info.size;
      else physicalBytesAdded += info.size;
    }
  }
  return { files, logicalBytes, sharedBytes, physicalBytesAdded };
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

const TRANSIENT_FILESYSTEM_CODES = new Set(["EACCES", "EBUSY", "ENOTEMPTY", "EPERM"]);
const BLOB_PROMOTION_RETRY_CODES = new Set([...TRANSIENT_FILESYSTEM_CODES, "EEXIST"]);

function filesystemErrorCode(error: unknown): string | undefined {
  return (error as NodeJS.ErrnoException | undefined)?.code;
}

async function retryFilesystemOperation<T>(
  operation: () => Promise<T>,
  retryCodes: ReadonlySet<string> = TRANSIENT_FILESYSTEM_CODES,
  attempts = 8
): Promise<T> {
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      const retryable = retryCodes.has(filesystemErrorCode(error) ?? "");
      if (!retryable || attempt + 1 >= attempts) throw error;
      await new Promise<void>((resolveDelay) => {
        setTimeout(resolveDelay, Math.min(250, 25 * (attempt + 1)));
      });
    }
  }
}

async function removeFileWithRetry(path: string): Promise<void> {
  await retryFilesystemOperation(() => rm(path, { force: true }));
}

async function removeTreeWithRetry(path: string): Promise<void> {
  await retryFilesystemOperation(() =>
    rm(path, {
      recursive: true,
      force: true,
      maxRetries: 4,
      retryDelay: 50
    })
  );
}

async function verifiedExistingBlob(path: string, expectedHash: string): Promise<boolean> {
  let info;
  try {
    info = await lstat(path);
  } catch (error) {
    if (filesystemErrorCode(error) === "ENOENT") return false;
    throw error;
  }
  if (!info.isFile() || info.isSymbolicLink() || (await fileSha256(path)) !== expectedHash) {
    throw new Error("Content-addressed blob checksum failed.");
  }
  return true;
}

async function awaitAllOrThrow(operations: readonly Promise<unknown>[]): Promise<void> {
  const settled = await Promise.allSettled(operations);
  const failed = settled.find(
    (result): result is PromiseRejectedResult => result.status === "rejected"
  );
  if (failed) throw failed.reason;
}

async function atomicWrite(path: string, contents: string | Buffer): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${randomUUID()}.next`;
  const handle = await open(temporary, "wx", 0o600);
  try {
    await handle.writeFile(contents);
    await handle.sync();
  } finally {
    await handle.close();
  }

  try {
    await rename(temporary, path);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code !== "EEXIST" && code !== "EPERM") {
      await rm(temporary, { force: true });
      throw error;
    }

    const backup = `${path}.bak`;
    if (await pathExists(path)) await rename(path, backup);
    try {
      await rename(temporary, path);
      await rm(backup, { force: true });
    } catch (replacementError) {
      if (await pathExists(backup)) await rename(backup, path);
      await rm(temporary, { force: true });
      throw replacementError;
    }
  }
}

async function inspectPackedTernaryDirectory(
  directory: string,
  label: string,
  overrides: Map<string, string> = new Map()
): Promise<StreamingPackedTernaryDirectory> {
  const directoryEntries = await readdir(directory, { withFileTypes: true });
  if (directoryEntries.some((entry) => !entry.isFile())) {
    throw new Error(`${label} packed ternary directory contains a non-file entry.`);
  }
  const inventory = new Set(directoryEntries.map((entry) => entry.name));
  const filePath = (name: string): string => {
    if (!inventory.has(name)) {
      throw new Error(`${label} packed ternary directory is missing ${name}.`);
    }
    return overrides.get(name) ?? join(directory, name);
  };
  const manifestBytes = await readFile(filePath("manifest.json"));
  const expectedManifestHash = (await readFile(filePath("manifest.sha256"), "ascii")).trim();
  if (
    !/^[a-f0-9]{64}$/.test(expectedManifestHash) ||
    sha256(manifestBytes) !== expectedManifestHash
  ) {
    throw new Error(`${label} packed ternary manifest checksum failed.`);
  }
  let parsed: unknown;
  const manifestText = manifestBytes.toString("utf8");
  try {
    parsed = JSON.parse(manifestText);
  } catch {
    throw new Error(`${label} packed ternary manifest is invalid.`);
  }
  if (
    !isRecord(parsed) ||
    parsed.format !== "omni-packed-ternary" ||
    parsed.formatVersion !== 1 ||
    parsed.architecture !== "OmniCortex" ||
    !isRecord(parsed.encoding) ||
    parsed.encoding.bitsPerValue !== 2 ||
    parsed.encoding.byteOrder !== "four-values-lsb-first" ||
    parsed.encoding.reservedCode !== 3 ||
    parsed.encoding.paddingValue !== 0 ||
    !isRecord(parsed.encoding.codes) ||
    parsed.encoding.codes["-1"] !== 0 ||
    parsed.encoding.codes["0"] !== 1 ||
    parsed.encoding.codes["+1"] !== 2 ||
    !isRecord(parsed.coverage) ||
    parsed.coverage.complete !== true ||
    !Array.isArray(parsed.coverage.eligibleTensorNames) ||
    !Array.isArray(parsed.tensors) ||
    !Array.isArray(parsed.shards)
  ) {
    throw new Error(`${label} packed ternary manifest is incompatible.`);
  }
  // Python deliberately preserves float identity (for example `1.0`) while
  // JavaScript JSON.parse represents it as the number `1`. Normalize only
  // numeric lexemes before comparing canonical structure so key ordering,
  // whitespace, string escaping, and duplicate keys remain strictly checked.
  if (normalizeCanonicalJsonNumbers(manifestText) !== canonicalJson(parsed)) {
    throw new Error(`${label} packed ternary manifest is not canonical.`);
  }
  const claimedContentHash = parsed.contentSha256;
  if (
    typeof claimedContentHash !== "string" ||
    !/^[a-f0-9]{64}$/.test(claimedContentHash) ||
    sha256(canonicalManifestContentBytes(manifestText, claimedContentHash)) !== claimedContentHash
  ) {
    throw new Error(`${label} packed ternary content checksum failed.`);
  }
  const expectedNames = parsed.coverage.eligibleTensorNames;
  if (
    expectedNames.some((name) => typeof name !== "string" || !name) ||
    new Set(expectedNames).size !== expectedNames.length ||
    parsed.coverage.eligibleTensorCount !== expectedNames.length ||
    parsed.tensors.length !== expectedNames.length
  ) {
    throw new Error(`${label} packed ternary coverage contract is invalid.`);
  }
  const shardTable = new Map<string, { byteLength: number; sha256: string; path: string }>();
  for (const descriptor of parsed.shards) {
    if (
      !isRecord(descriptor) ||
      typeof descriptor.file !== "string" ||
      !/^ternary-[0-9]{5,}-[a-f0-9]{16}\.bin$/.test(descriptor.file) ||
      typeof descriptor.byteLength !== "number" ||
      !Number.isSafeInteger(descriptor.byteLength) ||
      descriptor.byteLength < 0 ||
      typeof descriptor.sha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(descriptor.sha256) ||
      shardTable.has(descriptor.file)
    ) {
      throw new Error(`${label} packed ternary shard table is invalid.`);
    }
    const path = filePath(descriptor.file);
    const info = await lstat(path);
    if (
      !info.isFile() ||
      info.isSymbolicLink() ||
      info.size !== descriptor.byteLength ||
      (await streamFileSha256(path)) !== descriptor.sha256
    ) {
      throw new Error(`${label} packed ternary shard checksum failed.`);
    }
    shardTable.set(descriptor.file, {
      byteLength: descriptor.byteLength,
      sha256: descriptor.sha256,
      path
    });
  }
  const tensorNames: string[] = [];
  const usedShards = new Set<string>();
  for (const tensor of parsed.tensors) {
    if (
      !isRecord(tensor) ||
      typeof tensor.name !== "string" ||
      typeof tensor.shard !== "string" ||
      !shardTable.has(tensor.shard) ||
      usedShards.has(tensor.shard) ||
      !["projection", "dynamic-synapse"].includes(String(tensor.kind)) ||
      tensor.dtype !== "int8" ||
      !Array.isArray(tensor.shape) ||
      tensor.shape.some(
        (dimension) =>
          typeof dimension !== "number" || !Number.isSafeInteger(dimension) || dimension < 0
      ) ||
      typeof tensor.numel !== "number" ||
      !Number.isSafeInteger(tensor.numel) ||
      tensor.numel < 0 ||
      tensor.byteOffset !== 0 ||
      typeof tensor.byteLength !== "number" ||
      !Number.isSafeInteger(tensor.byteLength) ||
      tensor.byteLength < 0 ||
      typeof tensor.scale !== "number" ||
      !Number.isFinite(tensor.scale) ||
      tensor.scale <= 0 ||
      typeof tensor.sourceDtype !== "string" ||
      typeof tensor.packedSha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(tensor.packedSha256) ||
      typeof tensor.tensorSha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(tensor.tensorSha256)
    ) {
      throw new Error(`${label} packed ternary tensor table is invalid.`);
    }
    const shapeProduct = tensor.shape.reduce(
      (total, dimension) => total * (dimension as number),
      1
    );
    const shard = shardTable.get(tensor.shard)!;
    const expectedByteLength = Math.ceil(tensor.numel / 4);
    if (
      !Number.isSafeInteger(shapeProduct) ||
      shapeProduct !== tensor.numel ||
      tensor.byteLength !== expectedByteLength ||
      shard.byteLength !== expectedByteLength ||
      tensor.packedSha256 !== shard.sha256
    ) {
      throw new Error(`${label} packed ternary tensor shape or length is invalid.`);
    }
    const decodedDigest = createHash("sha256");
    decodedDigest.update(
      Buffer.from(JSON.stringify({ dtype: "int8", shape: tensor.shape }), "utf8")
    );
    decodedDigest.update(Buffer.from([0]));
    let decoded = 0;
    for await (const chunk of createReadStream(shard.path)) {
      const value = chunk as Buffer;
      const decodedChunk = Buffer.allocUnsafe(
        Math.min(tensor.numel - decoded, value.byteLength * 4)
      );
      let chunkDecoded = 0;
      for (let byteIndex = 0; byteIndex < value.byteLength; byteIndex += 1) {
        for (let slot = 0; slot < 4; slot += 1) {
          const code = (value[byteIndex]! >> (slot * 2)) & 0x03;
          if (decoded + chunkDecoded >= tensor.numel) {
            if (code !== 1) {
              throw new Error(`${label} packed ternary padding is non-canonical.`);
            }
          } else {
            if (code === 3) {
              throw new Error(`${label} packed ternary data uses the reserved code.`);
            }
            decodedChunk[chunkDecoded] = code === 0 ? 0xff : code === 1 ? 0 : 1;
            chunkDecoded += 1;
          }
        }
      }
      decoded += chunkDecoded;
      decodedDigest.update(decodedChunk.subarray(0, chunkDecoded));
    }
    if (decoded !== tensor.numel || decodedDigest.digest("hex") !== tensor.tensorSha256) {
      throw new Error(`${label} packed ternary decoded tensor checksum failed.`);
    }
    tensorNames.push(tensor.name);
    usedShards.add(tensor.shard);
  }
  if (
    tensorNames.length !== expectedNames.length ||
    tensorNames.some((name, index) => name !== expectedNames[index]) ||
    usedShards.size !== shardTable.size
  ) {
    throw new Error(`${label} packed ternary coverage is incomplete.`);
  }
  const allowed = new Set(["manifest.json", "manifest.sha256", ...shardTable.keys()]);
  if ([...inventory].some((name) => !allowed.has(name))) {
    throw new Error(`${label} packed ternary directory contains an unlisted file.`);
  }
  return {
    manifestSha256: expectedManifestHash,
    tensorCount: expectedNames.length,
    files: new Map([...allowed].map((name) => [name, filePath(name)]))
  };
}

export function resolveBrainDataRoot(
  userDataPath: string,
  override = process.env.OMNI_AGI_DATA_DIR
): string {
  const windowsLocal =
    process.platform === "win32" && process.env.LOCALAPPDATA
      ? join(resolve(process.env.LOCALAPPDATA), "OmniAGI")
      : undefined;
  const base = override?.trim()
    ? resolve(override.trim())
    : (windowsLocal ?? resolve(userDataPath));
  return join(base, "brains");
}

export interface BrainRepositoryCommitObserver {
  saved?(declaration: { brainId: string; storagePoolBytes: number }): void | Promise<void>;
  removed?(brainId: string): void | Promise<void>;
  error?(operation: "saved" | "removed", brainId: string, error: unknown): void;
}

export class BrainRepository {
  readonly root: string;

  constructor(root: string, private readonly commitObserver?: BrainRepositoryCommitObserver) {
    this.root = resolve(root);
  }

  private async observeCommittedBrain(brain: BrainDocument): Promise<void> {
    try {
      await this.commitObserver?.saved?.({
        brainId: brain.id,
        storagePoolBytes: brain.config.storagePoolBytes
      });
    } catch (error) {
      this.observeCommitError("saved", brain.id, error);
    }
  }

  private async observeRemovedBrain(brainId: string): Promise<void> {
    try {
      await this.commitObserver?.removed?.(brainId);
    } catch (error) {
      this.observeCommitError("removed", brainId, error);
    }
  }

  private observeCommitError(operation: "saved" | "removed", brainId: string, error: unknown): void {
    // A resource observer is not checkpoint authority. The file has already
    // committed/been removed, so a handler failure must not make a caller retry
    // the neural mutation or mistake a confirmed deletion for an intact mind.
    try {
      this.commitObserver?.error?.(operation, brainId, error);
      console.warn(`Committed brain ${operation} observer pending for ${brainId}:`, error);
    } catch {
      // Even a diagnostic failure cannot roll back a completed filesystem fact.
    }
  }

  async initialize(): Promise<void> {
    await Promise.all([
      mkdir(this.root, { recursive: true }),
      mkdir(join(this.root, ".trash"), { recursive: true }),
      mkdir(join(this.root, ".blobs"), { recursive: true }),
      mkdir(join(this.root, ".blob-leases"), { recursive: true })
    ]);
  }

  async betaReviewComplete(): Promise<boolean> {
    try {
      const value = JSON.parse(
        await readFile(join(this.root, BETA_REVIEW_FILE), "utf8")
      ) as unknown;
      return (
        isRecord(value) &&
        value.releaseFormat === STABLE_RELEASE_FORMAT &&
        typeof value.reviewedAt === "string"
      );
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
      return false;
    }
  }

  async enumerateManagedBetaBrains(): Promise<ManagedBetaBrain[]> {
    await this.initialize();
    const entries = await readdir(this.root, { withFileTypes: true });
    const candidates: ManagedBetaBrain[] = [];
    for (const entry of entries) {
      if (!entry.isDirectory() || !SAFE_ID.test(entry.name)) continue;
      const directory = join(this.root, entry.name);
      const documentPath = join(directory, "brain.json");
      if (!(await pathExists(documentPath))) continue;
      let document: Record<string, unknown> | undefined;
      try {
        const parsed = JSON.parse(await readFile(documentPath, "utf8")) as unknown;
        document = isRecord(parsed) ? parsed : undefined;
      } catch {
        document = undefined;
      }
      const name =
        typeof document?.name === "string" && document.name.trim()
          ? document.name.trim().slice(0, 120)
          : entry.name;
      if (!document) {
        candidates.push({
          id: entry.name,
          name,
          path: directory,
          reason: "invalid-document"
        });
        continue;
      }
      if (document.releaseFormat !== STABLE_RELEASE_FORMAT) {
        candidates.push({
          id: entry.name,
          name,
          path: directory,
          reason: "beta-document"
        });
        continue;
      }
      const enginePath = join(directory, "engine", "brain.json");
      if (!(await pathExists(enginePath))) continue;
      try {
        const engineState = JSON.parse(await readFile(enginePath, "utf8")) as unknown;
        if (!isRecord(engineState) || engineState.release_format !== STABLE_RELEASE_FORMAT) {
          candidates.push({
            id: entry.name,
            name,
            path: directory,
            reason: "beta-engine"
          });
        }
      } catch {
        candidates.push({
          id: entry.name,
          name,
          path: directory,
          reason: "beta-engine"
        });
      }
    }
    return candidates.sort((left, right) => left.id.localeCompare(right.id));
  }

  async deleteManagedBetaBrains(ids: string[], explicitlyConfirmed: boolean): Promise<string[]> {
    if (!explicitlyConfirmed) {
      throw new Error("Permanent beta deletion requires explicit confirmation.");
    }
    const requested = [...new Set(ids.map((id) => requireSafeId(id)))];
    const candidates = new Map(
      (await this.enumerateManagedBetaBrains()).map((candidate) => [candidate.id, candidate])
    );
    const rootPath = await realpath(this.root);
    const deleted: string[] = [];
    for (const id of requested) {
      const candidate = candidates.get(id);
      if (!candidate) {
        throw new Error(`Managed beta brain "${id}" is no longer eligible for deletion.`);
      }
      const targetPath = await realpath(candidate.path);
      if (dirname(targetPath) !== rootPath || basename(targetPath) !== id) {
        throw new Error("Managed beta deletion escaped the app data root.");
      }
      const info = await stat(targetPath);
      if (!info.isDirectory()) {
        throw new Error(`Managed beta brain "${id}" is not a directory.`);
      }
      await rm(targetPath, { recursive: true, force: false });
      await this.observeRemovedBrain(id);
      deleted.push(id);
    }
    return deleted;
  }

  async completeBetaReview(disposition: "kept" | "deleted" | "none", ids: string[]): Promise<void> {
    await this.initialize();
    await atomicWrite(
      join(this.root, BETA_REVIEW_FILE),
      JSON.stringify(
        {
          releaseFormat: STABLE_RELEASE_FORMAT,
          reviewedAt: new Date().toISOString(),
          disposition,
          managedBrainIds: [...new Set(ids.map((id) => requireSafeId(id)))]
        },
        null,
        2
      )
    );
  }

  async storeBlob(contents: Buffer): Promise<string> {
    await this.initialize();
    const hash = sha256(contents);
    const destination = join(this.root, ".blobs", hash);
    const temporary = join(this.root, ".blobs", `.incoming-${randomUUID()}`);
    try {
      // Never write directly to the digest path. A concurrent loser can see an
      // EEXIST result while the winner's write is still in progress and would
      // otherwise return a pointer to a partial file. Fully materialize a
      // private temporary and atomically promote it, validating any winner.
      await writeFile(temporary, contents, { flag: "wx", mode: 0o600 });
      await retryFilesystemOperation(async () => {
        if (await verifiedExistingBlob(destination, hash)) return;
        try {
          await rename(temporary, destination);
        } catch (error) {
          if (await verifiedExistingBlob(destination, hash)) return;
          throw error;
        }
      }, BLOB_PROMOTION_RETRY_CODES);
      if (!(await verifiedExistingBlob(destination, hash))) {
        throw new Error("Content-addressed blob promotion failed.");
      }
      await removeFileWithRetry(temporary);
      return hash;
    } catch (error) {
      await removeFileWithRetry(temporary);
      throw error;
    }
  }

  async getBlob(hash: string): Promise<Buffer> {
    if (!/^[a-f0-9]{64}$/.test(hash)) throw new Error("Invalid content-addressed blob hash.");
    const contents = await readFile(join(this.root, ".blobs", hash));
    if (sha256(contents) !== hash) throw new Error("Content-addressed blob checksum failed.");
    return contents;
  }

  async storeFileAsBlob(
    path: string,
    operation?: BrainStorageOperationHooks
  ): Promise<string> {
    await this.initialize();
    operation?.signal.throwIfAborted();
    const sourceInfo = await lstat(path);
    if (!sourceInfo.isFile() || sourceInfo.isSymbolicLink()) {
      throw new Error("Content-addressed source is not a safe regular file.");
    }
    const hash = await fileSha256(path, operation?.signal);
    operation?.signal.throwIfAborted();
    const destination = join(this.root, ".blobs", hash);
    if (await verifiedExistingBlob(destination, hash)) return hash;
    await operation?.checkDisk(this.root, sourceInfo.size);
    const temporary = join(this.root, ".blobs", `.incoming-${randomUUID()}`);
    try {
      // App-managed source files remain mutable. Copy into the immutable blob
      // store; hard-linking the source here would let a later in-place write
      // corrupt every snapshot and bundle reference sharing that inode.
      // COPYFILE_FICLONE uses filesystem COW where available and safely falls
      // back to a copy elsewhere, avoiding a second physical allocation on
      // APFS/Btrfs without sharing the mutable inode.
      await copyFile(path, temporary, fsConstants.COPYFILE_FICLONE);
      if (await fileSha256(temporary, operation?.signal) !== hash) {
        throw new Error("Content-addressed source changed during snapshotting.");
      }
      await retryFilesystemOperation(async () => {
        // Concurrent copy-on-write operations often promote the same immutable
        // tensor. Windows reports that collision as EPERM rather than EEXIST,
        // so only accept it after validating the winner's type and digest.
        if (await verifiedExistingBlob(destination, hash)) return;
        try {
          await rename(temporary, destination);
        } catch (error) {
          if (await verifiedExistingBlob(destination, hash)) return;
          throw error;
        }
      }, BLOB_PROMOTION_RETRY_CODES);
      await removeFileWithRetry(temporary);
      return hash;
    } catch (error) {
      await removeFileWithRetry(temporary);
      throw error;
    }
  }

  private async adoptFileAsBlob(
    path: string,
    operation?: BrainStorageOperationHooks
  ): Promise<{ hash: string; bytes: number; physicalBytesAdded: number }> {
    await this.initialize();
    operation?.signal.throwIfAborted();
    const sourceInfo = await lstat(path);
    if (!sourceInfo.isFile() || sourceInfo.isSymbolicLink()) {
      throw new Error("Copy-on-write source is not a safe regular file.");
    }
    const hash = await fileSha256(path);
    operation?.signal.throwIfAborted();
    const destination = join(this.root, ".blobs", hash);
    let physicalBytesAdded = 0;
    if (!(await verifiedExistingBlob(destination, hash))) {
      const temporary = join(this.root, ".blobs", `.incoming-${randomUUID()}`);
      try {
        try {
          // The source is protected by its BrainWrite lock. Promoting a hard
          // link first consumes no second copy; future neural saves atomically
          // replace their live path and therefore preserve blob immutability.
          await link(path, temporary);
        } catch (error) {
          const code = (error as NodeJS.ErrnoException).code;
          if (![
            "EXDEV",
            "EPERM",
            "EACCES",
            "ENOTSUP"
          ].includes(code ?? "")) {
            throw error;
          }
          await operation?.checkDisk(this.root, sourceInfo.size);
          await copyFile(path, temporary);
          physicalBytesAdded = sourceInfo.size;
        }
        if ((await fileSha256(temporary)) !== hash) {
          throw new Error("Copy-on-write source changed during promotion.");
        }
        try {
          await rename(temporary, destination);
        } catch (error) {
          if (!(await verifiedExistingBlob(destination, hash))) throw error;
        }
      } finally {
        await removeFileWithRetry(temporary);
      }
    }
    if (!(await verifiedExistingBlob(destination, hash))) {
      throw new Error("Copy-on-write blob promotion failed.");
    }
    const [liveInfo, blobInfo] = await Promise.all([
      stat(path),
      stat(destination)
    ]);
    if (
      liveInfo.dev !== blobInfo.dev ||
      liveInfo.ino !== blobInfo.ino ||
      liveInfo.nlink < 2 ||
      blobInfo.nlink < 2
    ) {
      await this.linkBlobTo(hash, path);
    }
    const adopted = await stat(path);
    if (adopted.size !== sourceInfo.size || adopted.nlink < 2) {
      throw new Error("Copy-on-write source adoption failed.");
    }
    return { hash, bytes: sourceInfo.size, physicalBytesAdded };
  }

  async linkBlobTo(
    hash: string,
    destination: string,
    operation?: BrainStorageOperationHooks
  ): Promise<void> {
    const source = join(this.root, ".blobs", hash);
    if (!/^[a-f0-9]{64}$/.test(hash) || (await fileSha256(source)) !== hash) {
      throw new Error("Content-addressed blob checksum failed.");
    }
    const sourceInfo = await stat(source);
    await mkdir(dirname(destination), { recursive: true });
    const temporary = `${destination}.${randomUUID()}.next`;
    try {
      await link(source, temporary);
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (!["EXDEV", "EPERM", "EACCES", "ENOTSUP"].includes(code ?? "")) throw error;
      await operation?.checkDisk(this.root, sourceInfo.size);
      await copyFile(source, temporary);
    }
    if ((await stat(temporary)).size !== sourceInfo.size) {
      await rm(temporary, { force: true });
      throw new Error("Copy-on-write blob materialization failed.");
    }
    try {
      await rename(temporary, destination);
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code !== "EEXIST" && code !== "EPERM") {
        await rm(temporary, { force: true });
        throw error;
      }
      const backup = `${destination}.bak`;
      if (await pathExists(destination)) await rename(destination, backup);
      try {
        await rename(temporary, destination);
        await rm(backup, { force: true });
      } catch (replacementError) {
        if (await pathExists(backup)) await rename(backup, destination);
        throw replacementError;
      }
    }
  }

  /**
   * Move an instance's immutable neural origin onto the repository-wide
   * content-addressed store. The live checkpoint is intentionally excluded:
   * it remains an independently writable identity. Identical origins from
   * separate builds therefore occupy one physical blob per content hash while
   * every instance keeps its own recoverable path.
   */
  async deduplicateImmutableOrigin(id: string): Promise<ImmutableOriginStorageReport> {
    await this.initialize();
    const origin = join(this.brainDirectory(id), "engine", "origin");
    const rootInfo = await lstat(origin).catch((error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") return undefined;
      throw error;
    });
    if (!rootInfo) {
      return { files: 0, logicalBytes: 0, contentHashes: 0, sharedFiles: 0 };
    }
    if (!rootInfo.isDirectory() || rootInfo.isSymbolicLink()) {
      throw new Error("The immutable neural origin is not a safe directory.");
    }

    const files: string[] = [];
    const visit = async (directory: string): Promise<void> => {
      const entries = await readdir(directory, { withFileTypes: true });
      for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
        const path = join(directory, entry.name);
        if (entry.isSymbolicLink()) {
          throw new Error("The immutable neural origin contains a symbolic link.");
        }
        if (entry.isDirectory()) {
          await visit(path);
          continue;
        }
        if (!entry.isFile()) {
          throw new Error("The immutable neural origin contains an unsupported filesystem entry.");
        }
        files.push(path);
      }
    };
    await visit(origin);

    let logicalBytes = 0;
    let sharedFiles = 0;
    const hashes = new Set<string>();
    // Process sequentially so a partial I/O failure never exposes a large fan
    // out of replacements. Each individual replacement remains atomic.
    for (const path of files) {
      const info = await lstat(path);
      if (!info.isFile() || info.isSymbolicLink()) {
        throw new Error("The immutable neural origin changed during deduplication.");
      }
      logicalBytes += info.size;
      const hash = await this.storeFileAsBlob(path);
      hashes.add(hash);
      await this.linkBlobTo(hash, path);
      const materialized = await stat(path);
      if (materialized.nlink > 1) sharedFiles += 1;
    }
    return {
      files: files.length,
      logicalBytes,
      contentHashes: hashes.size,
      sharedFiles
    };
  }

  private async recordReferencedBundleLease(
    destination: string,
    hashes: Iterable<string>
  ): Promise<void> {
    const unique = [...new Set(hashes)].sort();
    if (unique.some((hash) => !/^[a-f0-9]{64}$/.test(hash))) {
      throw new Error("A lightweight export contains an invalid local blob reference.");
    }
    const resolvedDestination = resolve(destination);
    const leaseId = sha256(resolvedDestination);
    await atomicWrite(
      join(this.root, ".blob-leases", `${leaseId}.json`),
      JSON.stringify(
        {
          format: "omni-local-blob-lease",
          formatVersion: 1,
          bundlePath: resolvedDestination,
          hashes: unique,
          createdAt: new Date().toISOString()
        },
        null,
        2
      )
    );
  }

  private async activeLeaseHashes(): Promise<Set<string> | undefined> {
    const protectedHashes = new Set<string>();
    const leaseRoot = join(this.root, ".blob-leases");
    const entries = await readdir(leaseRoot, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile() || !/^[a-f0-9]{64}\.json$/.test(entry.name)) {
        // An unknown entry could be a reference record from a newer version.
        // Disable collection instead of risking a still-needed blob.
        return undefined;
      }
      const path = join(leaseRoot, entry.name);
      let value: unknown;
      try {
        value = JSON.parse(await readFile(path, "utf8"));
      } catch {
        return undefined;
      }
      if (
        !isRecord(value) ||
        value.format !== "omni-local-blob-lease" ||
        value.formatVersion !== 1 ||
        typeof value.bundlePath !== "string" ||
        !Array.isArray(value.hashes) ||
        value.hashes.some((hash) => typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash))
      ) {
        return undefined;
      }
      const bundleInfo = await lstat(value.bundlePath).catch(() => undefined);
      if (!bundleInfo?.isFile() || bundleInfo.isSymbolicLink()) {
        await rm(path, { force: true });
        continue;
      }
      for (const hash of value.hashes as string[]) protectedHashes.add(hash);
    }
    return protectedHashes;
  }

  private async collectUnreferencedBlobs(): Promise<{
    removedSharedBlobs: number;
    reclaimedBytes: number;
  }> {
    const protectedHashes = await this.activeLeaseHashes();
    if (!protectedHashes) return { removedSharedBlobs: 0, reclaimedBytes: 0 };

    // Raw-source archives are referenced from stable brain documents rather
    // than hard-linked into every instance. Protect every exact digest found
    // in a live document in addition to filesystem link counts and export
    // leases.
    const collectStrings = (value: unknown): void => {
      if (typeof value === "string") {
        if (/^[a-f0-9]{64}$/.test(value)) protectedHashes.add(value);
        return;
      }
      if (Array.isArray(value)) {
        for (const entry of value) collectStrings(entry);
        return;
      }
      if (isRecord(value)) {
        for (const entry of Object.values(value)) collectStrings(entry);
      }
    };
    const entries = await readdir(this.root, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory() || !SAFE_ID.test(entry.name)) continue;
      try {
        collectStrings(JSON.parse(await readFile(this.documentPath(entry.name), "utf8")));
      } catch {
        // A live but unreadable instance is reason to collect nothing.
        return { removedSharedBlobs: 0, reclaimedBytes: 0 };
      }
    }

    const blobRoot = join(this.root, ".blobs");
    const blobRootPath = await realpath(blobRoot);
    let removedSharedBlobs = 0;
    let reclaimedBytes = 0;
    for (const entry of await readdir(blobRoot, { withFileTypes: true })) {
      if (!entry.isFile() || !/^[a-f0-9]{64}$/.test(entry.name)) continue;
      if (protectedHashes.has(entry.name)) continue;
      const path = join(blobRoot, entry.name);
      const info = await lstat(path);
      if (!info.isFile() || info.isSymbolicLink() || info.nlink > 1) continue;
      if (dirname(await realpath(path)) !== blobRootPath || basename(path) !== entry.name) {
        throw new Error("Shared-blob collection escaped its exact storage directory.");
      }
      await rm(path, { force: false });
      removedSharedBlobs += 1;
      reclaimedBytes += info.size;
    }
    return { removedSharedBlobs, reclaimedBytes };
  }

  private async copyPackedTernaryDirectory(
    source: string,
    destination: string,
    operation?: BrainStorageOperationHooks
  ): Promise<StreamingPackedTernaryDirectory | undefined> {
    operation?.signal.throwIfAborted();
    if (!(await pathExists(join(source, "manifest.json")))) return undefined;
    const packed = await inspectPackedTernaryDirectory(source, "Source");
    const temporary = `${destination}.${randomUUID()}.next`;
    const backup = `${destination}.${randomUUID()}.bak`;
    await mkdir(temporary, { recursive: true });
    try {
      for (const [name, sourcePath] of packed.files) {
        operation?.signal.throwIfAborted();
        await operation?.checkpoint({
          phase: "materializing",
          label: `Linking packed inference file ${name}`
        });
        const hash = await this.storeFileAsBlob(sourcePath, operation);
        await this.linkBlobTo(hash, join(temporary, name), operation);
      }
      if (await pathExists(destination)) await rename(destination, backup);
      try {
        await rename(temporary, destination);
        await rm(backup, { recursive: true, force: true });
      } catch (error) {
        if (await pathExists(backup)) await rename(backup, destination);
        throw error;
      }
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
    await inspectPackedTernaryDirectory(destination, "Copied");
    return packed;
  }

  private async copySubstrateSnapshot(
    sourceEngine: string,
    destinationEngine: string,
    metadata: unknown,
    operation?: BrainStorageOperationHooks
  ): Promise<string | undefined> {
    const prefix = "substrate/snapshot";
    const snapshot = await collectSubstrateSnapshot(
      sourceEngine,
      prefix,
      metadata,
      operation?.signal,
      operation
        ? (files) => operation.checkpoint({
            phase: "planning",
            label: `Validating persisted neural connections · ${files.toLocaleString()} files`
          })
        : undefined
    );
    if (!snapshot) return undefined;
    const destination = join(destinationEngine, "substrate");
    const temporary = `${destination}.${randomUUID()}.next`;
    const backup = `${destination}.${randomUUID()}.bak`;
    await mkdir(temporary, { recursive: true });
    try {
      for (const source of snapshot.sources) {
        operation?.signal.throwIfAborted();
        const relative = source.name.slice(`${prefix}/`.length);
        const target = join(temporary, ...relative.split("/"));
        if (relative === "manifest.json") {
          const bytes = source.sourcePath
            ? (await lstat(source.sourcePath)).size
            : source.contents!.byteLength;
          await operation?.checkDisk(this.root, bytes);
          await operation?.checkpoint({
            phase: "materializing",
            label: "Copying private neural substrate pointer"
          });
          if (source.sourcePath) {
            await copyMutableFileIsolated(source.sourcePath, target);
          } else {
            await writeMutableFileIsolated(target, source.contents!);
          }
          continue;
        }
        const hash = source.sourcePath
          ? await this.storeFileAsBlob(source.sourcePath, operation)
          : await this.storeBlob(Buffer.from(source.contents!));
        await operation?.checkpoint({
          phase: "materializing",
          label: `Linking neural substrate ${relative}`
        });
        await this.linkBlobTo(
          hash,
          target,
          operation
        );
      }
      if (await pathExists(destination)) await rename(destination, backup);
      try {
        await rename(temporary, destination);
        await rm(backup, { recursive: true, force: true });
      } catch (error) {
        if (await pathExists(backup)) await rename(backup, destination);
        throw error;
      }
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
    await collectSubstrateSnapshot(destinationEngine, prefix, metadata);
    return sha256(canonicalJson(snapshot.pointer));
  }

  private async copySavedContinuation(
    sourceEngine: string,
    destinationEngine: string,
    metadata: unknown,
    operation?: BrainStorageOperationHooks
  ): Promise<{ paths: string[]; hashes: string[] }> {
    const paths: string[] = [];
    const hashes: string[] = [];
    const working = join(sourceEngine, "state", "working-memory.sqlite3");
    const targetWorking = join(destinationEngine, "state", "working-memory.sqlite3");
    if (await pathExists(working)) {
      await snapshotSavedSqlite(working, targetWorking, operation);
      validateSavedWorkingPages(targetWorking, metadata);
      paths.push("engine/state/working-memory.sqlite3");
      hashes.push(await fileSha256(targetWorking));
    } else {
      validateSavedWorkingPages(undefined, metadata);
      await rm(targetWorking, { force: true });
    }
    const joint = await savedJointGenerationFiles(metadata, (kind, relative) =>
      join(sourceEngine, "state", ...(kind === "joint" ? ["ingestion-joint"] : []), ...relative.split("/"))
    );
    for (const [relative, source] of [...joint].sort(([left], [right]) => left.localeCompare(right))) {
      operation?.signal.throwIfAborted();
      const hash = await this.storeFileAsBlob(source, operation);
      const target = join(destinationEngine, "state", "ingestion-joint", ...relative.split("/"));
      await this.linkBlobTo(hash, target, operation);
      paths.push(`engine/state/ingestion-joint/${relative}`);
      hashes.push(hash);
    }
    for (const [name, source] of await savedToolIntentFiles(sourceEngine)) {
      operation?.signal.throwIfAborted();
      const hash = await this.storeFileAsBlob(source, operation);
      await this.linkBlobTo(hash, join(destinationEngine, "operational-tool-intents", name), operation);
      paths.push(`engine/operational-tool-intents/${name}`);
      hashes.push(hash);
    }
    for (const [relative, source] of await savedConceptIdViewFiles(sourceEngine)) {
      operation?.signal.throwIfAborted();
      const hash = await this.storeFileAsBlob(source, operation);
      await this.linkBlobTo(hash, join(destinationEngine, ...relative.split("/")), operation);
      paths.push(`engine/${relative}`);
      hashes.push(hash);
    }
    for (const [relative, source] of await savedGeometryHoldoutFiles(metadata, sourceEngine)) {
      operation?.signal.throwIfAborted();
      const hash = await this.storeFileAsBlob(source, operation);
      await this.linkBlobTo(hash, join(destinationEngine, ...relative.split("/")), operation);
      paths.push(`engine/${relative}`);
      hashes.push(hash);
    }
    return { paths, hashes };
  }

  private async copyMutableStateSnapshot(
    sourceEngine: string,
    destinationEngine: string,
    metadata: unknown,
    operation?: BrainStorageOperationHooks
  ): Promise<string | undefined> {
    const prefix = "mutable/snapshot";
    const snapshot = await collectMutableStateSnapshot(
      sourceEngine,
      prefix,
      metadata,
      operation?.signal,
      operation
        ? (files) => operation.checkpoint({
            phase: "planning",
            label: `Validating mutable neural state · ${files.toLocaleString()} files`
          })
        : undefined
    );
    if (!snapshot) return undefined;
    const destination = join(destinationEngine, "state");
    const temporary = `${destination}.${randomUUID()}.next`;
    const backup = `${destination}.${randomUUID()}.bak`;
    await mkdir(temporary, { recursive: true });
    try {
      for (const source of snapshot.sources) {
        operation?.signal.throwIfAborted();
        const relative = source.name.slice(`${prefix}/`.length);
        const target = join(temporary, ...relative.split("/"));
        if (relative === "replay.sqlite3" && source.sourcePath) {
          await snapshotSavedSqlite(source.sourcePath, target, operation);
          continue;
        }
        if (relative === "manifest.json") {
          const bytes = source.sourcePath
            ? (await lstat(source.sourcePath)).size
            : source.contents!.byteLength;
          await operation?.checkDisk(this.root, bytes);
          await operation?.checkpoint({
            phase: "materializing",
            label: "Copying private mutable-state pointer"
          });
          if (source.sourcePath) {
            await copyMutableFileIsolated(source.sourcePath, target);
          } else {
            await writeMutableFileIsolated(target, source.contents!);
          }
          continue;
        }
        const hash = source.sourcePath
          ? await this.storeFileAsBlob(source.sourcePath, operation)
          : await this.storeBlob(Buffer.from(source.contents!));
        await operation?.checkpoint({
          phase: "materializing",
          label: `Linking mutable neural state ${relative}`
        });
        await this.linkBlobTo(
          hash,
          target,
          operation
        );
      }
      if (await pathExists(destination)) await rename(destination, backup);
      try {
        await rename(temporary, destination);
        await rm(backup, { recursive: true, force: true });
      } catch (error) {
        if (await pathExists(backup)) await rename(backup, destination);
        throw error;
      }
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
    await collectMutableStateSnapshot(destinationEngine, prefix, metadata);
    return sha256(canonicalJson(snapshot.pointer));
  }

  private async copyArtifactSnapshot(
    sourceEngine: string,
    destinationEngine: string,
    brainId: string,
    operation?: BrainStorageOperationHooks
  ): Promise<string | undefined> {
    operation?.signal.throwIfAborted();
    const sourceDirectory = join(sourceEngine, "artifacts");
    const sourceStore = await ArtifactIndexStore.openExisting(
      sourceDirectory,
      brainId
    );
    if (!sourceStore) return undefined;
    let artifacts: ReturnType<ArtifactIndexStore["snapshot"]>["artifacts"];
    try {
      artifacts = sourceStore.snapshot().artifacts;
    } finally {
      sourceStore.close();
    }
    const destinationDirectory = join(destinationEngine, "artifacts");
    await rm(destinationDirectory, { recursive: true, force: true });
    await mkdir(destinationDirectory, { recursive: true });
    for (const artifact of artifacts) {
      operation?.signal.throwIfAborted();
      const name = basename(artifact.relativePath);
      const sourcePath = join(sourceDirectory, name);
      const info = await lstat(sourcePath);
      if (
        !info.isFile() ||
        info.isSymbolicLink() ||
        info.size !== artifact.bytes ||
        await fileSha256(sourcePath) !== artifact.sha256
      ) {
        throw new Error(`Generated artifact failed recovery-point integrity: ${name}`);
      }
      await operation?.checkpoint({
        phase: "materializing",
        label: `Linking generated artifact ${name}`
      });
      const hash = await this.storeFileAsBlob(sourcePath, operation);
      await this.linkBlobTo(
        hash,
        join(destinationDirectory, name),
        operation
      );
    }
    await ArtifactIndexStore.replace(
      destinationDirectory,
      brainId,
      artifacts
    );
    return sha256(serializeArtifactIndex(brainId, artifacts));
  }

  private async cloneEngineState(
    sourceBrainId: string,
    targetBrainId: string,
    targetName: string,
    operation?: BrainStorageOperationHooks
  ): Promise<void> {
    const sourceEngine = join(this.brainDirectory(sourceBrainId), "engine");
    const metadataPath = join(sourceEngine, "brain.json");
    if (!(await pathExists(metadataPath))) return;
    operation?.signal.throwIfAborted();
    const sourceMetadata = JSON.parse(await readFile(metadataPath, "utf8")) as unknown;
    if (!isRecord(sourceMetadata)) {
      throw new Error("The source engine metadata is invalid.");
    }
    const metadata = clone(sourceMetadata);
    metadata.brain_id = targetBrainId;
    metadata.name = targetName;
    if (isRecord(metadata.config)) metadata.config.name = targetName;
    const finalTargetEngine = join(this.brainDirectory(targetBrainId), "engine");
    // Materialize the complete neural clone outside its final path. The worker
    // treats brain.json as a commit record whose substrate/mutable pointers
    // must already resolve, so exposing an engine directory incrementally can
    // make an approved agent fork race a half-copied COW generation.
    const targetEngine = join(
      this.brainDirectory(targetBrainId),
      `.engine-${randomUUID()}.clone-next`
    );
    const targetOrigin = join(targetEngine, "origin");
    const sourceOrigin = join(sourceEngine, "origin");
    const sourceOriginMetadataPath = join(sourceOrigin, "brain.json");
    const hasImmutableOrigin = await pathExists(sourceOriginMetadataPath);
    const originSourceEngine = hasImmutableOrigin ? sourceOrigin : sourceEngine;
    const originMetadata = hasImmutableOrigin
      ? (JSON.parse(await readFile(sourceOriginMetadataPath, "utf8")) as unknown)
      : sourceMetadata;
    if (!isRecord(originMetadata)) {
      throw new Error("The source verified origin metadata is invalid.");
    }
    if (await pathExists(finalTargetEngine)) {
      throw new Error("The target neural engine already exists.");
    }
    const materials: CloneMaterialization[] = [];
    const addPath = async (
      sourcePath: string,
      destination: string,
      label: string,
      required = false,
      shareable = true
    ): Promise<void> => {
      const info = await lstat(sourcePath).catch((error: NodeJS.ErrnoException) => {
        if (!required && error.code === "ENOENT") return undefined;
        throw error;
      });
      if (!info) return;
      if (!info.isFile() || info.isSymbolicLink()) {
        throw new Error(`Clone source is not a safe regular file: ${label}`);
      }
      materials.push({ sourcePath, destination, label, bytes: info.size, shareable });
    };
    const addContents = (
      contents: Uint8Array,
      destination: string,
      label: string,
      shareable = true
    ): void => {
      materials.push({
        contents,
        destination,
        label,
        bytes: contents.byteLength,
        shareable
      });
    };

    for (const [source, destination] of [[sourceEngine, targetEngine], [originSourceEngine, targetOrigin]]) {
      for (const [name, path] of await savedToolIntentFiles(source!)) {
        await addPath(path, join(destination!, "operational-tool-intents", name), `historical tool intent ${name}`, true);
      }
      for (const [relative, path] of await savedConceptIdViewFiles(source!)) {
        await addPath(path, join(destination!, ...relative.split("/")), "historical structural argument view", true);
      }
      const sourceState = JSON.parse(await readFile(join(source!, "brain.json"), "utf8")) as unknown;
      for (const [relative, path] of await savedGeometryHoldoutFiles(sourceState, source!)) {
        await addPath(path, join(destination!, ...relative.split("/")), "registered geometry holdout", true);
      }
    }

    for (const name of ["core.safetensors", "plasticity.safetensors"] as const) {
      const sourcePath = join(sourceEngine, name);
      if (!(await pathExists(sourcePath))) continue;
      await addPath(
        sourcePath,
        join(targetEngine, name),
        `current ${name}`,
        true,
        false
      );
      await addPath(
        join(originSourceEngine, name),
        join(targetOrigin, name),
        `origin ${name}`,
        true,
        hasImmutableOrigin
      );
    }

    const addPacked = async (
      source: string,
      destination: string,
      label: string
    ): Promise<boolean> => {
      if (!(await pathExists(join(source, "manifest.json")))) return false;
      const packed = await inspectPackedTernaryDirectory(source, "Source");
      for (const [name, path] of packed.files) {
        await addPath(path, join(destination, name), `${label} ${name}`, true);
      }
      return true;
    };
    const livePacked = await addPacked(
      join(sourceEngine, "packed-ternary"),
      join(targetEngine, "packed-ternary"),
      "current packed ternary"
    );
    const originPacked = await addPacked(
      join(originSourceEngine, "packed-ternary"),
      join(targetOrigin, "packed-ternary"),
      "origin packed ternary"
    );

    const addSnapshot = async (
      snapshot: SubstrateSnapshot | MutableStateSnapshot | undefined,
      prefix: string,
      destination: string,
      label: string,
      privateLivePaths = false
    ): Promise<void> => {
      if (!snapshot) return;
      for (const source of snapshot.sources) {
        const relative = source.name.slice(`${prefix}/`.length);
        const target = join(destination, ...relative.split("/"));
        if (source.sourcePath) {
          await addPath(
            source.sourcePath,
            target,
            `${label} ${relative}`,
            true,
            !(
              privateLivePaths &&
              (relative === "manifest.json" || relative === "replay.sqlite3")
            )
          );
        } else if (source.contents) {
          addContents(
            source.contents,
            target,
            `${label} ${relative}`,
            !(privateLivePaths && relative === "manifest.json")
          );
        } else {
          throw new Error("Clone snapshot source has no materialization.");
        }
      }
    };
    const liveSubstrate = await collectSubstrateSnapshot(
      sourceEngine,
      "substrate/snapshot",
      metadata,
      operation?.signal,
      operation
        ? (files) => operation.checkpoint({
            phase: "planning",
            label: `Validating current substrate shards · ${files.toLocaleString()} files`
          })
        : undefined
    );
    operation?.signal.throwIfAborted();
    const originSubstrate = await collectSubstrateSnapshot(
      originSourceEngine,
      "substrate/snapshot",
      originMetadata,
      operation?.signal,
      operation
        ? (files) => operation.checkpoint({
            phase: "planning",
            label: `Validating origin substrate shards · ${files.toLocaleString()} files`
          })
        : undefined
    );
    const liveMutable = await collectMutableStateSnapshot(
      sourceEngine,
      "mutable/snapshot",
      metadata,
      operation?.signal,
      operation
        ? (files) => operation.checkpoint({
            phase: "planning",
            label: `Validating current mutable state · ${files.toLocaleString()} files`
          })
        : undefined
    );
    operation?.signal.throwIfAborted();
    const originMutable = await collectMutableStateSnapshot(
      originSourceEngine,
      "mutable/snapshot",
      originMetadata,
      operation?.signal,
      operation
        ? (files) => operation.checkpoint({
            phase: "planning",
            label: `Validating origin mutable state · ${files.toLocaleString()} files`
          })
        : undefined
    );
    await addSnapshot(
      liveSubstrate,
      "substrate/snapshot",
      join(targetEngine, "substrate"),
      "current substrate",
      true
    );
    await addSnapshot(
      originSubstrate,
      "substrate/snapshot",
      join(targetOrigin, "substrate"),
      "origin substrate"
    );
    await addSnapshot(
      liveMutable,
      "mutable/snapshot",
      join(targetEngine, "state"),
      "current mutable state",
      true
    );
    await addSnapshot(
      originMutable,
      "mutable/snapshot",
      join(targetOrigin, "state"),
      "origin mutable state",
      !hasImmutableOrigin
    );
    let sourceArtifactIndex: PersistedArtifactIndex | undefined;
    const sourceArtifactStore = await ArtifactIndexStore.openExisting(
      join(sourceEngine, "artifacts"),
      sourceBrainId
    );
    if (sourceArtifactStore) {
      try {
        sourceArtifactIndex = sourceArtifactStore.snapshot(targetBrainId);
      } finally {
        sourceArtifactStore.close();
      }
      for (const artifact of sourceArtifactIndex.artifacts) {
        const name = basename(artifact.relativePath);
        const sourcePath = join(sourceEngine, "artifacts", name);
        const info = await lstat(sourcePath);
        if (
          !info.isFile() ||
          info.isSymbolicLink() ||
          info.size !== artifact.bytes ||
          await streamFileSha256(sourcePath) !== artifact.sha256
        ) {
          throw new Error(`Generated artifact failed clone integrity: ${name}`);
        }
        await addPath(
          sourcePath,
          join(targetEngine, "artifacts", name),
          `generated artifact ${name}`,
          true
        );
      }
    }
    if (hasImmutableOrigin) {
      await addPath(
        sourceOriginMetadataPath,
        join(targetOrigin, "brain.json"),
        "origin engine metadata",
        true
      );
      await addPath(
        join(sourceOrigin, "provenance.json"),
        join(targetOrigin, "provenance.json"),
        "origin provenance"
      );
    } else {
      addContents(
        Buffer.from(JSON.stringify(sourceMetadata, null, 2)),
        join(targetOrigin, "brain.json"),
        "origin engine metadata"
      );
    }
    const currentMetadataBytes = Buffer.from(JSON.stringify(metadata, null, 2));
    const logicalBytesTotal = materials.reduce(
      (sum, material) => sum + material.bytes,
      currentMetadataBytes.byteLength
    );
    const filesTotal = materials.length + 1;
    let filesCompleted = 0;
    let logicalBytesCompleted = 0;
    let physicalBytesAdded = 0;
    let sharedBytes = 0;
    const progress = async (
      update: Partial<BrainStorageProgressUpdate> = {}
    ): Promise<void> => {
      await operation?.checkpoint({
        phase: "materializing",
        label: "Materializing verified copy-on-write files",
        targetBrainId,
        filesCompleted,
        filesTotal,
        logicalBytesCompleted,
        logicalBytesTotal,
        physicalBytesAdded,
        sharedBytes,
        ...update
      });
    };
    try {
      await mkdir(targetEngine, { recursive: false });
      await mkdir(targetOrigin, { recursive: false });
      await progress({ phase: "materializing" });
      for (const material of materials) {
        await progress({ label: material.label });
        operation?.signal.throwIfAborted();
        let hash = "";
        if (material.shareable === false) {
          await operation?.checkDisk(this.root, material.bytes);
          if (material.sourcePath) {
            await copyMutableFileIsolated(
              material.sourcePath,
              material.destination
            );
          } else {
            await writeMutableFileIsolated(
              material.destination,
              material.contents!
            );
          }
        } else if (material.sourcePath) {
            const adopted = await this.adoptFileAsBlob(
              material.sourcePath,
              operation
            );
            hash = adopted.hash;
            physicalBytesAdded += adopted.physicalBytesAdded;
        } else {
          await operation?.checkDisk(this.root, material.bytes);
          hash = await this.storeBlob(Buffer.from(material.contents!));
        }
        if (material.shareable !== false) {
          await this.linkBlobTo(hash, material.destination, operation);
        }
        const materialized = await stat(material.destination);
        if (materialized.nlink > 1) sharedBytes += material.bytes;
        else physicalBytesAdded += material.bytes;
        filesCompleted += 1;
        logicalBytesCompleted += material.bytes;
        await progress();
      }
      if (sourceArtifactIndex) {
        await operation?.checkDisk(
          this.root,
          Buffer.byteLength(serializeArtifactIndex(
            targetBrainId,
            sourceArtifactIndex.artifacts
          ))
        );
        await ArtifactIndexStore.replace(
          join(targetEngine, "artifacts"),
          targetBrainId,
          sourceArtifactIndex.artifacts
        );
      }
      const neuralConversationPath = join(sourceEngine, "conversation.sqlite3");
      if (await pathExists(neuralConversationPath)) {
        validateNeuralConversationLedger(neuralConversationPath, sourceBrainId);
        const info = await lstat(neuralConversationPath);
        await operation?.checkDisk(this.root, info.size);
        const targetConversation = join(targetEngine, "conversation.sqlite3");
        await copyFile(neuralConversationPath, targetConversation);
        rekeyNeuralConversationLedger(targetConversation, targetBrainId);
        validateNeuralConversationLedger(targetConversation, targetBrainId);
      }
      if (livePacked) {
        await inspectPackedTernaryDirectory(
          join(targetEngine, "packed-ternary"),
          "Copied"
        );
      }
      if (originPacked) {
        await inspectPackedTernaryDirectory(
          join(targetOrigin, "packed-ternary"),
          "Copied"
        );
      }
      if (liveSubstrate) {
        await collectSubstrateSnapshot(
          targetEngine,
          "substrate/snapshot",
          metadata
        );
      }
      if (originSubstrate) {
        await collectSubstrateSnapshot(
          targetOrigin,
          "substrate/snapshot",
          originMetadata
        );
      }
      if (liveMutable) {
        await collectMutableStateSnapshot(
          targetEngine,
          "mutable/snapshot",
          metadata
        );
      }
      if (originMutable) {
        await collectMutableStateSnapshot(
          targetOrigin,
          "mutable/snapshot",
          originMetadata
        );
      }
      // Current metadata is the staged clone's commit record and therefore
      // moves last. Publishing the directory is one same-volume rename, so a
      // worker can observe either no clone or every referenced generation.
      await progress({
        phase: "committing",
        label: "Validating free space before atomic promotion"
      });
      await operation?.checkDisk(this.root, currentMetadataBytes.byteLength);
      operation?.signal.throwIfAborted();
      await atomicWrite(join(targetEngine, "brain.json"), currentMetadataBytes.toString("utf8"));
      filesCompleted += 1;
      logicalBytesCompleted += currentMetadataBytes.byteLength;
      physicalBytesAdded += currentMetadataBytes.byteLength;
      await progress({
        phase: "promoting",
        label: "Promoting complete copy-on-write engine"
      });
      operation?.signal.throwIfAborted();
      await rename(targetEngine, finalTargetEngine);
    } catch (error) {
      const stagingName = basename(targetEngine);
      const stagingParent = dirname(targetEngine);
      if (
        stagingParent !== this.brainDirectory(targetBrainId) ||
        !/^\.engine-[a-f0-9-]{36}\.clone-next$/.test(stagingName)
      ) {
        throw new Error("Refusing to clean an unvalidated clone staging path.");
      }
      await removeTreeWithRetry(targetEngine);
      throw error;
    }
  }

  brainDirectory(id: string): string {
    return join(this.root, requireSafeId(id));
  }

  private documentPath(id: string): string {
    return join(this.brainDirectory(id), "brain.json");
  }

  private async compactConversationDocument(
    brain: BrainDocument,
    persistMigration = false,
    hydrateLatestTrace = false
  ): Promise<BrainDocument> {
    const ledger = await ConversationLedger.open(this.brainDirectory(brain.id), brain.id);
    try {
      const hadLegacyRows = brain.messages.length > 0 || brain.traces.length > 0;
      ledger.backfill(brain.messages, brain.traces);
      const auditedActions: ActionEvent[] = (brain.journal ?? [])
        .filter((entry) => entry.kind === "tool")
        .map((entry) => {
          const protocol = entry.summary.split(":", 1)[0] ?? "tool.action";
          const separator = protocol.lastIndexOf(".");
          const toolId = separator > 0 ? protocol.slice(0, separator) : protocol;
          const action = separator > 0 ? protocol.slice(separator + 1) : "run";
          const state = /:\s*failed\.?$/i.test(entry.summary)
            ? "failed" as const
            : /:\s*(?:cancelled|stopped)\.?$/i.test(entry.summary)
              ? "stopped" as const
              : "complete" as const;
          return {
            id: entry.id,
            brainId: brain.id,
            action: {
              kind: "tool",
              source: "human",
              toolId,
              action,
              arguments: {
                auditSummary: entry.summary,
                auditDetail: entry.detail ?? ""
              }
            },
            state,
            createdAt: entry.createdAt,
            updatedAt: entry.createdAt,
            attentionEpoch: brain.conversation?.attentionEpoch ?? 0
          };
        });
      if (auditedActions.length) {
        ledger.append(
          auditedActions.map((value) => ({ kind: "action" as const, value }))
        );
      }
      const summary = ledger.summary();
      brain.messages = [];
      brain.traces = [];
      brain.conversation = summary;
      if (persistMigration && hadLegacyRows) {
        await atomicWrite(this.documentPath(brain.id), JSON.stringify(brain, null, 2));
      }
      if (hydrateLatestTrace) {
        const latestTrace = ledger.recentEvidence(200)
          .reverse()
          .find((entry) => entry.trace)?.trace;
        if (latestTrace) brain.traces = [latestTrace];
      }
      return brain;
    } finally {
      ledger.close();
    }
  }

  private async compactActivityDocument(
    brain: BrainDocument,
    persistMigration = false,
    hydrate = true
  ): Promise<BrainDocument> {
    const ledger = await BrainActivityLedger.open(
      this.brainDirectory(brain.id),
      brain.id
    );
    try {
      const hadLegacyRows =
        (brain.journal?.length ?? 0) > 0 || brain.trainingSources.length > 0;
      const priorSummary = JSON.stringify(brain.activity ?? null);
      ledger.appendJournals(brain.journal ?? []);
      ledger.upsertTrainingSources(brain.trainingSources);
      brain.journal = [];
      brain.trainingSources = [];
      brain.activity = ledger.getSummary();
      if (
        persistMigration &&
        (hadLegacyRows || priorSummary !== JSON.stringify(brain.activity))
      ) {
        const persisted = clone(brain);
        persisted.messages = [];
        persisted.traces = [];
        persisted.journal = [];
        persisted.trainingSources = [];
        await atomicWrite(this.documentPath(brain.id), JSON.stringify(persisted, null, 2));
      }
      if (hydrate) {
        brain.journal = ledger.journalPage(undefined, 100).entries.map(
          (entry) => entry.entry
        );
        brain.trainingSources = ledger.trainingSourcePage(undefined, 100).entries.map(
          (entry) => entry.source
        );
      }
      return brain;
    } finally {
      ledger.close();
    }
  }

  async conversationPage(
    id: string,
    beforeSequence?: number,
    limit?: number
  ): Promise<ConversationLedgerPage> {
    await this.get(id);
    const ledger = await ConversationLedger.open(this.brainDirectory(id), id);
    try {
      return ledger.page(beforeSequence, limit);
    } finally {
      ledger.close();
    }
  }

  async journalPage(
    id: string,
    cursor?: string,
    limit?: number
  ): Promise<JournalLedgerPage> {
    await this.get(id, false);
    const ledger = await BrainActivityLedger.open(this.brainDirectory(id), id);
    try {
      const page = ledger.journalPage(cursor, limit);
      return {
        brainId: id,
        entries: page.entries,
        totalEntries: page.totalEntries,
        ...(page.nextCursor ? { nextCursor: page.nextCursor } : {})
      };
    } finally {
      ledger.close();
    }
  }

  async trainingSourcePage(
    id: string,
    cursor?: string,
    limit?: number
  ): Promise<TrainingSourceLedgerPage> {
    await this.get(id, false);
    const ledger = await BrainActivityLedger.open(this.brainDirectory(id), id);
    try {
      const page = ledger.trainingSourcePage(cursor, limit);
      return {
        brainId: id,
        entries: page.entries,
        totalEntries: page.totalEntries,
        ...(page.nextCursor ? { nextCursor: page.nextCursor } : {})
      };
    } finally {
      ledger.close();
    }
  }

  async trainingSourceById(id: string, sourceId: string): Promise<TrainingSource | undefined> {
    await this.get(id, false);
    const ledger = await BrainActivityLedger.open(this.brainDirectory(id), id);
    try {
      return ledger.trainingSourceById(sourceId);
    } finally {
      ledger.close();
    }
  }

  async trainingSourceByContentHash(
    id: string,
    contentHash: string
  ): Promise<TrainingSource | undefined> {
    await this.get(id, false);
    const ledger = await BrainActivityLedger.open(this.brainDirectory(id), id);
    try {
      return ledger.trainingSourceByContentHash(contentHash);
    } finally {
      ledger.close();
    }
  }

  async appendTrainingSourceProjection(
    id: string,
    sources: TrainingSource[]
  ): Promise<BrainActivityLedgerSummary> {
    if (!sources.length) {
      const brain = await this.get(id, false);
      return brain.activity!;
    }
    await access(this.documentPath(id));
    const ledger = await BrainActivityLedger.open(this.brainDirectory(id), id);
    try {
      return ledger.upsertTrainingSources(sources);
    } finally {
      ledger.close();
    }
  }

  async visitNovelTrainingSources(
    sourceBrainId: string,
    targetBrainId: string,
    visitor: (source: TrainingSource, fingerprint: string) => void | Promise<void>
  ): Promise<{ total: number; novel: number }> {
    await Promise.all([
      this.get(sourceBrainId, false),
      this.get(targetBrainId, false)
    ]);
    const sourceLedger = await BrainActivityLedger.open(
      this.brainDirectory(sourceBrainId),
      sourceBrainId
    );
    const targetLedger = await BrainActivityLedger.open(
      this.brainDirectory(targetBrainId),
      targetBrainId
    );
    let total = 0;
    let novel = 0;
    try {
      for (const source of sourceLedger.trainingSources()) {
        total += 1;
        const fingerprint = trainingSourceEvidenceFingerprint(source);
        if (targetLedger.trainingSourceByFingerprint(fingerprint)) continue;
        await visitor(source, fingerprint);
        novel += 1;
      }
      return { total, novel };
    } finally {
      sourceLedger.close();
      targetLedger.close();
    }
  }

  async journalById(id: string, journalId: string): Promise<JournalEntry | undefined> {
    await this.get(id, false);
    const ledger = await BrainActivityLedger.open(this.brainDirectory(id), id);
    try {
      return ledger.journalById(journalId);
    } finally {
      ledger.close();
    }
  }

  async searchConversation(
    id: string,
    query: string,
    beforeSequence?: number,
    limit?: number
  ): Promise<ConversationLedgerPage> {
    await this.get(id);
    const ledger = await ConversationLedger.open(this.brainDirectory(id), id);
    try {
      return ledger.search(query, beforeSequence, limit, false);
    } finally {
      ledger.close();
    }
  }

  async conversationMessage(id: string, messageId: string): Promise<ChatMessage | undefined> {
    let before: number | undefined;
    do {
      const page = await this.conversationPage(id, before, 200);
      const found = page.entries.find((entry) => entry.message?.id === messageId)?.message;
      if (found) return found;
      before = page.nextBeforeSequence;
    } while (before !== undefined);
    return undefined;
  }

  async recentConversationEvidence(id: string, limit = 500) {
    await this.get(id);
    const ledger = await ConversationLedger.open(this.brainDirectory(id), id);
    try {
      return ledger.recentEvidence(limit).filter(
        (entry) => entry.message?.deliveryReceipt?.presentationOnly !== true
      );
    } finally {
      ledger.close();
    }
  }

  async appendConversationActions(id: string, actions: ActionEvent[]): Promise<void> {
    if (!actions.length) return;
    await this.get(id);
    const ledger = await ConversationLedger.open(this.brainDirectory(id), id);
    try {
      ledger.append(actions.map((value) => ({ kind: "action" as const, value })));
    } finally {
      ledger.close();
    }
  }

  /**
   * Persist a display-only human delivery receipt without touching brain.json
   * or the worker-owned neural conversation/replay state. Each state is an
   * append-only revision; the renderer collapses revisions by turnId.
   */
  async appendChatDeliveryReceipt(
    id: string,
    receipt: ChatDeliveryReceiptRequest
  ): Promise<void> {
    await access(this.documentPath(id));
    const updatedAt = new Date().toISOString();
    const message: ChatMessage = {
      id: `delivery-${receipt.turnId}-${receipt.state}`,
      role: "human",
      content: receipt.content,
      createdAt: receipt.createdAt,
      deliveryReceipt: {
        schemaVersion: 1,
        presentationOnly: true,
        turnId: receipt.turnId,
        state: receipt.state,
        updatedAt
      }
    };
    const ledger = await ConversationLedger.open(this.brainDirectory(id), id);
    try {
      ledger.append([{ kind: "message", value: message }]);
    } finally {
      ledger.close();
    }
  }

  async create(
    config: BrainConfig,
    options: {
      initializing?: boolean;
      recovery?: NonNullable<BrainDocument["readiness"]["recovery"]>;
    } = {}
  ): Promise<BrainDocument> {
    await this.initialize();
    if (options.recovery && (
      options.recovery.foundation.origin !== "ground-up" ||
      Object.hasOwn(options.recovery.foundation, "foundationModelId")
    )) {
      throw new Error("Only locally initialized OmniCortex brains can be created.");
    }
    const id = randomUUID();
    const now = new Date().toISOString();
    const normalizedConfig = normalizeConfig(config);
    const provenance: BrainProvenance | undefined = options.recovery
      ? { originKind: "ground-up" }
      : undefined;
    const brain: BrainDocument = {
      schemaVersion: BRAIN_SCHEMA_VERSION,
      releaseFormat: STABLE_RELEASE_FORMAT,
      id,
      name: normalizedConfig.name,
      createdAt: now,
      updatedAt: now,
      readiness: options.initializing
        ? {
            state: "initializing",
            startedAt: now,
            attempt: 1,
            ...(options.recovery ? { recovery: clone(options.recovery) } : {})
          }
        : { state: "ready", startedAt: now, completedAt: now, attempt: 1 },
      ...(provenance ? { provenance } : {}),
      lineage: { rootId: id, generation: 0 },
      config: normalizedConfig,
      concepts: {},
      synapses: {},
      ideas: [],
      workingMemory: [],
      liquidState: {
        values: Array.from({ length: 16 }, () => 0),
        timeConstants: Array.from({ length: 16 }, (_, index) => 0.25 + index * 0.05),
        lastUpdatedAt: now
      },
      messages: [],
      traces: [],
      conversation: {
        format: "omni-conversation-ledger",
        formatVersion: 1,
        totalEntries: 0,
        messageCount: 0,
        actionCount: 0,
        traceCount: 0,
        headSequence: 0,
        headSha256: "0".repeat(64),
        attentionEpoch: 0
      },
      trainingSources: [],
      counters: {
        plasticityEvents: 0,
        inferenceCount: 0,
        consolidationCycles: 0
      },
      toolPermissions: clone(DEFAULT_TOOL_PERMISSIONS),
      journal: [
        {
          id: randomUUID(),
          createdAt: now,
          kind: "system",
          summary: "Immutable origin created."
        }
      ]
    };
    brain.originChecksum = originChecksumFor(brain);
    const originBrain = clone(brain);
    const directory = this.brainDirectory(id);
    await mkdir(join(directory, "snapshots"), { recursive: true });
    await this.compactActivityDocument(brain, false, false);
    await atomicWrite(this.documentPath(id), JSON.stringify(brain, null, 2));
    await writeFile(join(directory, "origin.json"), JSON.stringify(originBrain, null, 2), {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600
    });
    const created = clone(await this.compactActivityDocument(brain, false, true));
    await this.observeCommittedBrain(created);
    return normalizedConfig.idleCognition
      ? (await this.setActiveMode(id, true)).brain
      : created;
  }

  async completeInitialization(id: string): Promise<BrainDocument> {
    return withBrainWrite(this, id, async () => {
      const brain = await this.get(id);
      if (brain.readiness.state === "ready") return brain;
      if (brain.readiness.state === "failed") {
        throw new Error("Initialization failed and must be retried before chat can open.");
      }
      const completedAt = new Date().toISOString();
      brain.readiness = {
        ...brain.readiness,
        state: "ready",
        completedAt,
        failure: undefined,
        recovery: undefined
      };
      brain.journal = [
        ...(brain.journal ?? []),
        {
          id: randomUUID(),
          createdAt: completedAt,
          kind: "system",
          summary: "Initial neural learning completed; chat and organic cognition unlocked."
        }
      ];
      return this.save(brain);
    });
  }

  async failInitialization(
    id: string,
    phase: "foundation" | "initial-learning",
    error: unknown
  ): Promise<BrainDocument> {
    return withBrainWrite(this, id, async () => {
      const brain = await this.get(id);
      if (brain.readiness.state === "ready") return brain;
      const failedAt = new Date().toISOString();
      const message = String(error instanceof Error ? error.message : error)
        .replace(/\0/g, "")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 2_000) || "Initialization failed.";
      const alreadyRecorded =
        brain.readiness.state === "failed" &&
        brain.readiness.failure?.phase === phase &&
        brain.readiness.failure.message === message;
      brain.readiness = {
        state: "failed",
        startedAt: brain.readiness.startedAt,
        attempt: brain.readiness.attempt ?? 1,
        failure: { phase, message, failedAt, retryable: true },
        ...(brain.readiness.recovery ? { recovery: brain.readiness.recovery } : {})
      };
      if (!alreadyRecorded) {
        brain.journal = [
          ...(brain.journal ?? []),
          {
            id: randomUUID(),
            createdAt: failedAt,
            kind: "system",
            summary: `Initial ${phase === "foundation" ? "foundation" : "learning"} paused: ${message}`
          }
        ];
      }
      return this.save(brain);
    });
  }

  async retryInitialization(id: string): Promise<BrainDocument> {
    return withBrainWrite(this, id, async () => {
      const brain = await this.get(id);
      if (brain.readiness.state === "ready") return brain;
      const retriedAt = new Date().toISOString();
      brain.readiness = {
        state: "initializing",
        startedAt: brain.readiness.startedAt,
        attempt: (brain.readiness.attempt ?? 1) + 1,
        ...(brain.readiness.recovery ? { recovery: brain.readiness.recovery } : {})
      };
      brain.journal = [
        ...(brain.journal ?? []),
        {
          id: randomUUID(),
          createdAt: retriedAt,
          kind: "system",
          summary: "Initial neural learning retry started."
        }
      ];
      return this.save(brain);
    });
  }

  async get(id: string, hydrateActivity = true): Promise<BrainDocument> {
    const documentPath = this.documentPath(id);
    const hydrateProvenance = async (brain: BrainDocument): Promise<BrainDocument> => {
      const provenance = brain.provenance;
      if (provenance && provenance.originKind !== "ground-up") {
        throw new Error("This saved instance has a non-native origin and is not supported.");
      }
      const needsEngineProvenance =
        !provenance ||
        provenance.originKind === "ground-up";
      if (!needsEngineProvenance) return brain;
      const engineProvenance = await readEngineBrainProvenance(
        join(this.brainDirectory(id), "engine"),
        id
      );
      if (
        engineProvenance &&
        !(
          provenance?.originKind === "ground-up" &&
          engineProvenance.originKind === "ground-up" &&
          provenance.randomInitialization &&
          !engineProvenance.randomInitialization
        )
      ) {
        brain.provenance = engineProvenance;
      }
      if (brain.provenance && brain.provenance.originKind !== "ground-up") {
        throw new Error("This saved instance has a non-native origin and is not supported.");
      }
      return brain;
    };
    try {
      const raw = await readFile(documentPath, "utf8");
      const compact = await this.compactConversationDocument(
        normalizeBrain(JSON.parse(raw)),
        true,
        true
      );
      return hydrateProvenance(
        await this.compactActivityDocument(compact, true, hydrateActivity)
      );
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      const backup = `${documentPath}.bak`;
      if (!(await pathExists(backup))) throw new Error(`Brain "${id}" was not found.`);
      const recovered = await readFile(backup, "utf8");
      const brain = await this.compactConversationDocument(
        normalizeBrain(JSON.parse(recovered)),
        true,
        true
      );
      const compact = await this.compactActivityDocument(brain, true, hydrateActivity);
      const persisted = clone(compact);
      persisted.messages = [];
      persisted.traces = [];
      persisted.journal = [];
      persisted.trainingSources = [];
      await atomicWrite(documentPath, JSON.stringify(persisted, null, 2));
      return hydrateProvenance(compact);
    }
  }

  async persistedSubstrateOverview(
    id: string
  ): Promise<PersistedSubstrateOverview | undefined> {
    // Library hydration needs only the bounded, checksum-validated substrate
    // manifests. Avoid hydrating the full brain/activity ledgers for a card.
    await access(this.documentPath(id));
    return readPersistedSubstrateOverview(
      join(this.brainDirectory(id), "engine"),
      id
    );
  }

  async save(brain: BrainDocument, touch = true): Promise<BrainDocument> {
    let normalized = await this.compactConversationDocument(
      normalizeBrain(brain)
    );
    normalized = await this.compactActivityDocument(normalized, false, false);
    if (touch) normalized.updatedAt = new Date().toISOString();
    normalized.name = normalized.config.name.trim() || normalized.name;
    await atomicWrite(this.documentPath(normalized.id), JSON.stringify(normalized, null, 2));
    await this.observeCommittedBrain(normalized);
    return clone(await this.compactActivityDocument(normalized, false, true));
  }

  async list(): Promise<BrainSummary[]> {
    await this.initialize();
    const entries = await readdir(this.root, { withFileTypes: true });
    const brains = await Promise.all(
      entries
        .filter((entry) => entry.isDirectory() && SAFE_ID.test(entry.name))
        .map(async (entry): Promise<BrainDocument | undefined> => {
          try {
            return await this.get(entry.name, false);
          } catch {
            return undefined;
          }
        })
    );
    const documents = brains.filter((brain): brain is BrainDocument => brain !== undefined);
    const persistedById = new Map<string, PersistedSubstrateOverview>();
    let nextOverview = 0;
    await Promise.all(
      Array.from(
        { length: Math.min(4, documents.length) },
        async () => {
          while (nextOverview < documents.length) {
            const brain = documents[nextOverview++];
            if (!brain) continue;
            try {
              const overview = await readPersistedSubstrateOverview(
                join(this.brainDirectory(brain.id), "engine"),
                brain.id
              );
              if (overview) persistedById.set(brain.id, overview);
            } catch {
              // One corrupt/unavailable engine must not hide other local
              // identities. Its card remains on the outer-document fallback.
            }
          }
        }
      )
    );
    const originCounts = new Map<string, number>();
    for (const brain of documents) {
      const key = brain.originChecksum ?? brain.lineage.rootId;
      originCounts.set(key, (originCounts.get(key) ?? 0) + 1);
    }
    const latestLineageById = new Map<string, JournalEntry>();
    await Promise.all(documents.map(async (brain) => {
      const ledger = await BrainActivityLedger.open(this.brainDirectory(brain.id), brain.id);
      try {
        const entry = ledger.latestLineageJournal();
        if (entry) latestLineageById.set(brain.id, entry);
      } finally {
        ledger.close();
      }
    }));
    return documents
      .map((brain): BrainSummary => {
        const latestLineageEvent = latestLineageById.get(brain.id);
        let instanceKind: BrainSummary["instanceKind"] =
          brain.lineage.generation === 0 ? "original" : "fork";
        if (latestLineageEvent?.kind === "fork" && latestLineageEvent.detail) {
          try {
            const detail = JSON.parse(latestLineageEvent.detail) as unknown;
            if (isRecord(detail) && detail.operation === "duplicate") instanceKind = "duplicate";
          } catch {
            // An old journal detail is inspection metadata, never authoritative state.
          }
        } else if (latestLineageEvent && /^Imported from /i.test(latestLineageEvent.summary)) {
          instanceKind = "imported";
        }
        const adaptationSource = brain.activity?.topAdaptation;
        const originKey = brain.originChecksum ?? brain.lineage.rootId;
        return {
          id: brain.id,
          name: brain.name,
          preset: brain.config.preset,
          runtime: brain.config.runtime,
          updatedAt: brain.updatedAt,
          concepts: Object.keys(brain.concepts).length,
          synapses: Object.keys(brain.synapses).length,
          neuralUpdates: brain.counters.plasticityEvents,
          inferenceCount: brain.counters.inferenceCount,
          trainingSources:
            brain.activity?.trainingSourceCount ?? brain.trainingSources.length,
          activeMode: brain.config.idleCognition,
          ...(persistedById.get(brain.id)
            ? { substrateTotals: persistedById.get(brain.id)!.totals }
            : {}),
          generation: brain.lineage.generation,
          rootId: brain.lineage.rootId,
          parentId: brain.lineage.parentId,
          originChecksum: brain.originChecksum,
          instanceKind,
          originInstanceCount: originCounts.get(originKey) ?? 1,
          ...(brain.provenance ? { provenance: clone(brain.provenance) } : {}),
          ...(adaptationSource
            ? {
                adaptation: {
                  sourceLabel: adaptationSource.sourceLabel,
                  learnedRecords: adaptationSource.learnedRecords
                }
              }
            : {})
        };
      })
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  }

  async updateConfig(id: string, config: BrainConfig): Promise<BrainDocument> {
    return withBrainWrite(this, id, () => this.updateConfigUnlocked(id, config));
  }

  async setActiveMode(
    id: string,
    enabled: boolean
  ): Promise<BrainActiveModeResult> {
    const brainId = requireSafeId(id);
    if (typeof enabled !== "boolean") {
      throw new Error("Active Mode must be enabled or disabled.");
    }
    await this.initialize();
    const ids = [
      ...new Set([brainId, ...(await this.list()).map((brain) => brain.id)])
    ];
    return withBrainWrite(this, ids, async () => {
      const documents = await Promise.all(ids.map((candidate) => this.get(candidate)));
      let target = documents.find((brain) => brain.id === brainId);
      if (!target) throw new Error(`Brain "${brainId}" was not found.`);
      const deactivated: Array<{ id: string; name: string }> = [];

      // Disable every prior owner before enabling the target. A failed write
      // can therefore leave no owner, but can never leave two idle-mutating.
      if (enabled) {
        for (const brain of documents) {
          if (brain.id === brainId || !brain.config.idleCognition) continue;
          brain.config.idleCognition = false;
          deactivated.push({ id: brain.id, name: brain.name });
          await this.save(brain);
        }
      }
      if (target.config.idleCognition !== enabled) {
        target.config.idleCognition = enabled;
        target = await this.save(target);
      }
      return {
        schemaVersion: 1,
        brain: clone(target),
        enabled,
        ...(enabled ? { activeBrainId: brainId } : {}),
        deactivated,
        updatedAt: new Date().toISOString()
      };
    });
  }

  /** Collapse legacy multi-true files to one newest persisted process owner. */
  async reconcileActiveModeLease(): Promise<BrainActiveModeResult | undefined> {
    const active = (await this.list()).filter((brain) => brain.activeMode);
    if (active.length <= 1) return undefined;
    return this.setActiveMode(active[0]!.id, true);
  }

  /** Normalize an untrusted renderer/import config without persisting it. */
  prepareConfig(config: unknown): BrainConfig {
    return clone(normalizeConfig(config));
  }

  private async updateConfigUnlocked(id: string, config: BrainConfig): Promise<BrainDocument> {
    const brain = await this.get(id);
    brain.config = {
      ...normalizeConfig(config),
      // Active Mode is updated only through setActiveMode(), which owns the
      // cross-identity single-lease transaction.
      idleCognition: brain.config.idleCognition
    };
    brain.name = brain.config.name;
    return this.save(brain);
  }

  async fork(
    id: string,
    name?: string,
    operation?: BrainStorageOperationHooks
  ): Promise<BrainDocument> {
    return withBrainWrite(this, id, () =>
      this.copyOnWriteClone(id, name, "fork", operation)
    );
  }

  async duplicate(
    id: string,
    name?: string,
    operation?: BrainStorageOperationHooks
  ): Promise<BrainDocument> {
    return withBrainWrite(this, id, () =>
      this.copyOnWriteClone(id, name, "duplicate", operation)
    );
  }

  private async copyOnWriteClone(
    id: string,
    name: string | undefined,
    operation: "fork" | "duplicate",
    storageOperation?: BrainStorageOperationHooks
  ): Promise<BrainDocument> {
    storageOperation?.signal.throwIfAborted();
    const source = await this.get(id);
    const sourceOriginPath = join(this.brainDirectory(id), "origin.json");
    const hasSourceOrigin = await pathExists(sourceOriginPath);
    let sourceOriginBlob: string | undefined;
    if (hasSourceOrigin) {
      const sourceOriginValue: unknown = JSON.parse(
        await readFile(sourceOriginPath, "utf8")
      );
      const inheritedChecksum = assertOriginChecksum(
        sourceOriginValue,
        "The source origin"
      );
      const sourceOrigin = normalizeBrain(sourceOriginValue);
      if (source.originChecksum !== inheritedChecksum) {
        throw new Error("The source brain does not reference its immutable origin checksum.");
      }
      const neuralOriginPath = join(this.brainDirectory(id), "engine", "origin", "brain.json");
      if (await pathExists(neuralOriginPath)) {
        assertUiNeuralOriginIdentity(
          sourceOrigin,
          JSON.parse(await readFile(neuralOriginPath, "utf8"))
        );
      }
      sourceOriginBlob = (
        await this.adoptFileAsBlob(sourceOriginPath, storageOperation)
      ).hash;
    }
    const fork = clone(source);
    const now = new Date().toISOString();
    fork.id = randomUUID();
    fork.name =
      name?.trim().slice(0, 120) ||
      `${source.name}${operation === "duplicate" ? " copy" : " fork"}`;
    fork.config.name = fork.name;
    // A copied identity never inherits the source's process-owning Active
    // Mode lease. It can be enabled explicitly after creation.
    fork.config.idleCognition = false;
    fork.createdAt = now;
    fork.updatedAt = now;
    fork.lineage = {
      parentId: source.id,
      rootId: source.lineage.rootId,
      generation: source.lineage.generation + 1
    };
    await storageOperation?.checkpoint({
      phase: "planning",
      label: "Validating source lineage and neural generations",
      targetBrainId: fork.id
    });
    fork.journal = [
      ...(fork.journal ?? []),
      {
        id: randomUUID(),
        createdAt: now,
        kind: "fork",
        summary:
          operation === "duplicate"
            ? `Duplicated from ${source.name} with copy-on-write neural storage.`
            : `Forked from ${source.name}.`,
        detail: JSON.stringify({
          sourceBrainId: source.id,
          operation,
          copyOnWrite: true
        })
      }
    ];
    if (hasSourceOrigin) {
      // A duplicate/fork branches the mutable identity, not its ancestry.
      // Keep both the checksum pointer and immutable UI snapshot identical to
      // the source lineage, matching engine/origin's copy-on-write semantics.
      fork.originChecksum = source.originChecksum;
    } else {
      // Stable-v1 repositories always have origin.json. This fallback keeps a
      // recoverable behavior for an older hand-created document missing it.
      fork.originChecksum = undefined;
      fork.originChecksum = originChecksumFor(fork);
    }
    const directory = this.brainDirectory(fork.id);
    try {
      await storageOperation?.checkDisk(
        this.root,
        Buffer.byteLength(JSON.stringify(fork))
      );
      storageOperation?.signal.throwIfAborted();
      await mkdir(join(directory, "snapshots"), { recursive: true });
      await ConversationLedger.clone(
        this.brainDirectory(source.id),
        directory,
        source.id,
        fork.id
      );
      await BrainActivityLedger.clone(
        this.brainDirectory(source.id),
        directory,
        source.id,
        fork.id
      );
      await this.compactActivityDocument(fork, false, false);
      await atomicWrite(this.documentPath(fork.id), JSON.stringify(fork, null, 2));
      if (sourceOriginBlob) {
        await this.linkBlobTo(
          sourceOriginBlob,
          join(directory, "origin.json"),
          storageOperation
        );
      } else {
        await storageOperation?.checkDisk(
          this.root,
          Buffer.byteLength(JSON.stringify(fork))
        );
        await writeFile(join(directory, "origin.json"), JSON.stringify(fork, null, 2), {
          encoding: "utf8",
          flag: "wx",
          mode: 0o600
        });
      }
      await this.cloneEngineState(
        source.id,
        fork.id,
        fork.name,
        storageOperation
      );
      const completed = clone(await this.get(fork.id));
      await this.observeCommittedBrain(completed);
      return completed;
    } catch (error) {
      await removeTreeWithRetry(directory);
      throw error;
    }
  }

  async remove(id: string): Promise<boolean> {
    return withBrainWrite(this, id, () => this.removeUnlocked(id));
  }

  async permanentlyDeleteInstance(
    request: DeleteInstanceRequest
  ): Promise<DeleteInstanceResult> {
    const id = requireSafeId(request.brainId);
    return withBrainWrite(this, id, async () => {
      const brain = await this.get(id);
      if (request.acknowledgedIrreversible !== true) {
        throw new Error("Permanent deletion requires accepting the irreversible warning.");
      }
      if (request.typedName !== brain.name) {
        throw new Error("The typed instance name does not match exactly.");
      }
      const requiredFinal = `PERMANENTLY DELETE ${brain.name}`;
      if (request.finalConfirmation !== requiredFinal) {
        throw new Error(`Final confirmation must exactly match: ${requiredFinal}`);
      }

      const rootPath = await realpath(this.root);
      const source = this.brainDirectory(id);
      const sourceInfo = await lstat(source).catch(() => undefined);
      if (!sourceInfo) throw new Error(`Instance "${id}" was not found.`);
      if (!sourceInfo.isDirectory() || sourceInfo.isSymbolicLink()) {
        throw new Error("The instance path is not a safe app-managed directory.");
      }
      const exactPath = await realpath(source);
      if (dirname(exactPath) !== rootPath || basename(exactPath) !== id) {
        throw new Error("Permanent deletion escaped the exact app-managed instance directory.");
      }

      await rm(exactPath, { recursive: true, force: false });
      await this.observeRemovedBrain(id);
      const reclaimed = await this.collectUnreferencedBlobs().catch(() => ({
        removedSharedBlobs: 0,
        reclaimedBytes: 0
      }));
      return {
        deleted: true,
        brainId: id,
        name: brain.name,
        recoverable: false,
        ...reclaimed
      };
    });
  }

  private async removeUnlocked(id: string): Promise<boolean> {
    const source = this.brainDirectory(id);
    if (!(await pathExists(source))) return false;
    const trashName = `${requireSafeId(id)}-${Date.now()}`;
    await rename(source, join(this.root, ".trash", trashName));
    await this.observeRemovedBrain(id);
    return true;
  }

  async snapshot(
    id: string,
    label?: string,
    operation?: BrainStorageOperationHooks,
    prepare?: () => Promise<void>
  ): Promise<BrainSnapshotSummary> {
    return withBrainWrite(
      this,
      id,
      async () => {
        operation?.signal.throwIfAborted();
        await prepare?.();
        operation?.signal.throwIfAborted();
        return this.snapshotUnlocked(id, label, operation);
      },
      operation?.signal
    );
  }

  private async snapshotUnlocked(
    id: string,
    label?: string,
    operation?: BrainStorageOperationHooks
  ): Promise<BrainSnapshotSummary> {
    const brain = await this.get(id);
    const snapshotId = randomUUID();
    const createdAt = new Date().toISOString();
    const document = JSON.stringify(brain, null, 2);
    const brainDirectory = this.brainDirectory(id);
    const engineSource = join(brainDirectory, "engine");
    const base = join(brainDirectory, "snapshots", snapshotId);
    const engineSnapshot = join(base, "engine");
    let engineChecksum: string | undefined;
    const checkpointHashes: string[] = [];
    let neuralConversationIncluded = false;
    let artifactIndexIncluded = false;
    let savedContinuation: SavedSnapshotSummary["savedContinuation"];
    try {
      await operation?.checkpoint({
        phase: "planning",
        label: "Validating the committed neural checkpoint"
      });
      const persisted = await readPersistedSubstrateOverview(engineSource, brain.id);
      if (await pathExists(join(engineSource, "brain.json"))) {
        await mkdir(engineSnapshot, { recursive: true });
        const metadata = await readFile(join(engineSource, "brain.json"));
        const metadataValue = JSON.parse(metadata.toString("utf8")) as unknown;
        const hasNeuralConversation = await pathExists(join(engineSource, "conversation.sqlite3"));
        assertImportedNeuralConversationHead(metadataValue, "Recovery-point source", hasNeuralConversation);
        if (!hasNeuralConversation && isRecord(metadataValue) &&
          Array.isArray(metadataValue.pending_chat_slow_learning) && metadataValue.pending_chat_slow_learning.length) {
          throw new Error("Recovery-point pending chat replay is missing its neural conversation ledger.");
        }
        const hashes: string[] = [sha256(metadata)];
        for (const name of ["core.safetensors", "plasticity.safetensors"]) {
          operation?.signal.throwIfAborted();
          const sourcePath = join(engineSource, name);
          if (!(await pathExists(sourcePath))) continue;
          await operation?.checkpoint({
            phase: "materializing",
            label: `Linking ${name} into the recovery point`
          });
          const hash = await this.storeFileAsBlob(sourcePath, operation);
          await this.linkBlobTo(hash, join(engineSnapshot, name), operation);
          hashes.push(hash);
        }
        const packed = await this.copyPackedTernaryDirectory(
          join(engineSource, "packed-ternary"),
          join(engineSnapshot, "packed-ternary"),
          operation
        );
        if (packed) hashes.push(packed.manifestSha256);
        const substrateHash = await this.copySubstrateSnapshot(
          engineSource,
          engineSnapshot,
          metadataValue,
          operation
        );
        if (substrateHash) hashes.push(substrateHash);
        const mutableStateHash = await this.copyMutableStateSnapshot(
          engineSource,
          engineSnapshot,
          metadataValue,
          operation
        );
        if (mutableStateHash) hashes.push(mutableStateHash);
        const artifactHash = await this.copyArtifactSnapshot(
          engineSource,
          engineSnapshot,
          brain.id,
          operation
        );
        if (artifactHash) {
          hashes.push(artifactHash);
          artifactIndexIncluded = true;
        }
        const neuralConversation = join(engineSource, "conversation.sqlite3");
        if (await pathExists(neuralConversation)) {
          operation?.signal.throwIfAborted();
          const info = await lstat(neuralConversation);
          if (!info.isFile() || info.isSymbolicLink()) {
            throw new Error("Neural conversation ledger is not a safe regular file.");
          }
          await operation?.checkDisk(this.root, info.size);
          await operation?.checkpoint({
            phase: "materializing",
            label: "Copying the exact neural conversation ledger"
          });
          const destination = join(engineSnapshot, "conversation.sqlite3");
          await snapshotSavedSqlite(neuralConversation, destination, operation);
          validateNeuralConversationLedger(destination, brain.id, metadataValue);
          hashes.push(await fileSha256(destination));
          neuralConversationIncluded = true;
        }
        const continuation = await this.copySavedContinuation(engineSource, engineSnapshot, metadataValue, operation);
        hashes.push(...continuation.hashes);
        savedContinuation = { version: 1, componentPaths: continuation.paths };
        operation?.signal.throwIfAborted();
        // Commit metadata after every referenced shard is durable.
        await atomicWrite(join(engineSnapshot, "brain.json"), metadata);
        checkpointHashes.push(...hashes);
      }

      for (const [path, label] of [
        [join(brainDirectory, "conversation", "ledger.sqlite3"), "conversation"],
        [BrainActivityLedger.databasePath(brainDirectory), "activity"]
      ] as const) {
        const info = await lstat(path);
        if (!info.isFile() || info.isSymbolicLink()) {
          throw new Error(`The ${label} ledger is not a safe regular file.`);
        }
        await operation?.checkDisk(this.root, info.size);
      }
      await operation?.checkpoint({
        phase: "materializing",
        label: "Copying exact conversation and activity ledgers"
      });
      await ConversationLedger.clone(brainDirectory, base, brain.id, brain.id);
      operation?.signal.throwIfAborted();
      await BrainActivityLedger.clone(brainDirectory, base, brain.id, brain.id);
      checkpointHashes.push(
        await fileSha256(join(base, "conversation", "ledger.sqlite3")),
        await fileSha256(BrainActivityLedger.databasePath(base))
      );
      engineChecksum = sha256(checkpointHashes.join(":"));

      const engineUsage = await snapshotStorageUsage(
        base,
        operation?.signal
      ).catch((error: NodeJS.ErrnoException) => {
        if (error.code === "ENOENT") {
          return {
            files: 0,
            logicalBytes: 0,
            sharedBytes: 0,
            physicalBytesAdded: 0
          };
        }
        throw error;
      });
      const documentBytes = Buffer.byteLength(document, "utf8");
      const fixedLogicalBytes = engineUsage.logicalBytes + documentBytes;
      const fixedPhysicalBytes = engineUsage.physicalBytesAdded + documentBytes;
      let summary: SavedSnapshotSummary = {
        id: snapshotId,
        brainId: brain.id,
        label: label?.replace(/\0/g, "").trim().slice(0, 120) ||
          `Snapshot ${createdAt}`,
        createdAt,
        checksum: sha256(document),
        metrics: recoveryPointMetrics(brain, persisted, fixedLogicalBytes),
        engineChecksum,
        checkpointComponentSha256: checkpointHashes,
        savedContinuation,
        durableState: {
          conversationLedger: true,
          activityLedger: true,
          neuralConversationLedger: neuralConversationIncluded,
          artifactIndex: artifactIndexIncluded
        },
        storage: {
          files: engineUsage.files + 2,
          logicalBytes: fixedLogicalBytes,
          sharedBytes: engineUsage.sharedBytes,
          physicalBytesAdded: fixedPhysicalBytes
        }
      };
      let metadataText = "";
      for (let pass = 0; pass < 4; pass += 1) {
        metadataText = JSON.stringify(summary, null, 2);
        const metadataBytes = Buffer.byteLength(metadataText, "utf8");
        const logicalBytes = fixedLogicalBytes + metadataBytes;
        const physicalBytesAdded = fixedPhysicalBytes + metadataBytes;
        if (
          summary.storage?.logicalBytes === logicalBytes &&
          summary.storage.physicalBytesAdded === physicalBytesAdded &&
          summary.metrics.estimatedBytes === logicalBytes
        ) {
          break;
        }
        summary = {
          ...summary,
          metrics: recoveryPointMetrics(brain, persisted, logicalBytes),
          storage: {
            files: engineUsage.files + 2,
            logicalBytes,
            sharedBytes: engineUsage.sharedBytes,
            physicalBytesAdded
          }
        };
      }
      metadataText = JSON.stringify(summary, null, 2);
      await operation?.checkDisk(
        this.root,
        documentBytes + Buffer.byteLength(metadataText, "utf8")
      );
      operation?.signal.throwIfAborted();
      await writeFile(`${base}.json`, document, {
        encoding: "utf8",
        flag: "wx",
        mode: 0o600
      });
      await writeFile(`${base}.meta.json`, metadataText, {
        encoding: "utf8",
        flag: "wx",
        mode: 0o600
      });
      const finalStorage = summary.storage!;
      await operation?.checkpoint({
        phase: "verifying",
        label: "Recovery point saved and verified",
        filesCompleted: finalStorage.files,
        filesTotal: finalStorage.files,
        logicalBytesCompleted: finalStorage.logicalBytes,
        logicalBytesTotal: finalStorage.logicalBytes,
        physicalBytesAdded: finalStorage.physicalBytesAdded,
        sharedBytes: finalStorage.sharedBytes
      });
      return summary;
    } catch (error) {
      // Remove only this newly allocated, uncommitted recovery point. Existing
      // recovery points and live neural files are never touched.
      await rm(base, { recursive: true, force: true }).catch(() => undefined);
      await rm(`${base}.json`, { force: true }).catch(() => undefined);
      await rm(`${base}.meta.json`, { force: true }).catch(() => undefined);
      throw error;
    }
  }

  async listSnapshots(id: string): Promise<BrainSnapshotSummary[]> {
    const directory = join(this.brainDirectory(id), "snapshots");
    await mkdir(directory, { recursive: true });
    const entries = await readdir(directory, { withFileTypes: true });
    const snapshots = await Promise.all(
      entries
        .filter((entry) => entry.isFile() && entry.name.endsWith(".meta.json"))
        .map(async (entry): Promise<BrainSnapshotSummary | undefined> => {
          try {
            const value = JSON.parse(
              await readFile(join(directory, entry.name), "utf8")
            ) as unknown;
            if (!isRecord(value) || value.brainId !== id || typeof value.id !== "string")
              return undefined;
            return value as unknown as BrainSnapshotSummary;
          } catch {
            return undefined;
          }
        })
    );
    return snapshots
      .filter((snapshot): snapshot is BrainSnapshotSummary => snapshot !== undefined)
      .sort((left, right) => right.createdAt.localeCompare(left.createdAt));
  }

  async restoreSnapshot(
    id: string,
    snapshotId: string,
    operation?: BrainStorageOperationHooks,
    afterPromotion?: () => Promise<void>
  ): Promise<BrainDocument> {
    return withBrainWrite(
      this,
      id,
      () => this.restoreSnapshotUnlocked(
        id,
        snapshotId,
        operation,
        afterPromotion
      ),
      operation?.signal
    );
  }

  private async restoreSnapshotUnlocked(
    id: string,
    snapshotId: string,
    operation?: BrainStorageOperationHooks,
    afterPromotion?: () => Promise<void>
  ): Promise<BrainDocument> {
    operation?.signal.throwIfAborted();
    requireSafeId(snapshotId, "snapshot id");
    await operation?.checkpoint({
      phase: "planning",
      label: "Validating recovery-point checksums and durable ledgers"
    });
    const current = await this.get(id);
    const base = join(this.brainDirectory(id), "snapshots", snapshotId);
    const [document, metadata] = await Promise.all([
      readFile(`${base}.json`, "utf8"),
      readFile(`${base}.meta.json`, "utf8")
    ]);
    const summary = JSON.parse(metadata) as SavedSnapshotSummary;
    if (summary.brainId !== id || sha256(document) !== summary.checksum) {
      throw new Error("Snapshot checksum validation failed.");
    }
    if (
      summary.durableState &&
      (summary.durableState.conversationLedger !== true ||
        summary.durableState.activityLedger !== true ||
        typeof summary.durableState.neuralConversationLedger !== "boolean" ||
        typeof summary.durableState.artifactIndex !== "boolean" ||
        !summary.engineChecksum ||
        !Array.isArray(summary.checkpointComponentSha256) ||
        summary.checkpointComponentSha256.some(
          (hash) => typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash)
        ))
    ) {
      throw new Error("Recovery-point durable-state declaration is invalid.");
    }
    const restored = normalizeBrain(JSON.parse(document));
    restored.id = id;
    // Snapshot restore replaces neural/history state, not this identity's
    // current process-level Active Mode preference.
    restored.config.idleCognition = current.config.idleCognition;
    restored.lineage = current.lineage;
    restored.createdAt = current.createdAt;
    restored.journal = [
      ...(restored.journal ?? []),
      {
        id: randomUUID(),
        createdAt: new Date().toISOString(),
        kind: "system",
        summary: `Restored snapshot ${summary.label}.`,
        detail: snapshotId
      }
    ];
    const engineSnapshot = join(this.brainDirectory(id), "snapshots", snapshotId, "engine");
    if (await pathExists(join(engineSnapshot, "brain.json"))) {
      const engineMetadata = await readFile(join(engineSnapshot, "brain.json"));
      const engineMetadataValue = JSON.parse(engineMetadata.toString("utf8")) as unknown;
      const hashes = [sha256(engineMetadata)];
      for (const name of ["core.safetensors", "plasticity.safetensors"]) {
        const sourcePath = join(engineSnapshot, name);
        if (await pathExists(sourcePath)) hashes.push(await fileSha256(sourcePath));
      }
      if (await pathExists(join(engineSnapshot, "packed-ternary", "manifest.json"))) {
        const packed = await inspectPackedTernaryDirectory(
          join(engineSnapshot, "packed-ternary"),
          "Snapshot"
        );
        hashes.push(packed.manifestSha256);
      }
      const substrate = await collectSubstrateSnapshot(
        engineSnapshot,
        "substrate/snapshot",
        engineMetadataValue
      );
      if (substrate) hashes.push(sha256(canonicalJson(substrate.pointer)));
      const mutableState = await collectMutableStateSnapshot(
        engineSnapshot,
        "mutable/snapshot",
        engineMetadataValue
      );
      if (mutableState) hashes.push(sha256(canonicalJson(mutableState.pointer)));
      if (summary.durableState) {
        const artifactStore = await ArtifactIndexStore.openExisting(
          join(engineSnapshot, "artifacts"),
          id
        );
        if (artifactStore) {
          let artifacts: ReturnType<ArtifactIndexStore["snapshot"]>["artifacts"];
          try {
            artifacts = artifactStore.snapshot().artifacts;
          } finally {
            artifactStore.close();
          }
          for (const artifact of artifacts) {
            const path = join(
              engineSnapshot,
              "artifacts",
              basename(artifact.relativePath)
            );
            const info = await lstat(path);
            if (
              !info.isFile() ||
              info.isSymbolicLink() ||
              info.size !== artifact.bytes ||
              await fileSha256(path) !== artifact.sha256
            ) {
              throw new Error("Recovery-point artifact checksum failed.");
            }
          }
          hashes.push(sha256(serializeArtifactIndex(id, artifacts)));
        }
        const neuralConversation = join(
          engineSnapshot,
          "conversation.sqlite3"
        );
        if (await pathExists(neuralConversation)) {
          hashes.push(await fileSha256(neuralConversation));
        }
        for (const relative of summary.savedContinuation?.componentPaths ?? []) {
          hashes.push(await fileSha256(join(base, ...safeSnapshotContinuationPath(relative).split("/"))));
        }
        hashes.push(
          await fileSha256(join(base, "conversation", "ledger.sqlite3")),
          await fileSha256(BrainActivityLedger.databasePath(base))
        );
      }
      const observedSnapshotChecksum = sha256(hashes.join(":"));
      if (summary.engineChecksum && observedSnapshotChecksum !== summary.engineChecksum) {
        const changedComponent = summary.checkpointComponentSha256?.findIndex(
          (hash, index) => hash !== hashes[index]
        );
        throw new Error(
          `Neural snapshot checksum validation failed${
            changedComponent !== undefined && changedComponent >= 0
              ? ` at component ${changedComponent + 1}`
              : ""
          }.`
        );
      }
    }
    if (!(await pathExists(join(engineSnapshot, "brain.json")))) {
      await this.save(restored);
      await afterPromotion?.();
      return this.get(id);
    }

    const brainDirectory = this.brainDirectory(id);
    const targetEngine = join(brainDirectory, "engine");
    const stagedEngine = join(brainDirectory, `.engine-${randomUUID()}.restore`);
    const previousEngine = join(brainDirectory, `.engine-${randomUUID()}.previous`);
    const failedEngine = join(brainDirectory, `.engine-${randomUUID()}.failed`);
    const stagedHost = join(brainDirectory, `.host-${randomUUID()}.restore`);
    const conversationPath = join(brainDirectory, "conversation", "ledger.sqlite3");
    const activityPath = BrainActivityLedger.databasePath(brainDirectory);
    const conversationBackup = `${conversationPath}.${randomUUID()}.previous`;
    const activityBackup = `${activityPath}.${randomUUID()}.previous`;
    let conversationBackedUp = false;
    let activityBackedUp = false;
    let conversationPromoted = false;
    let activityPromoted = false;
    let documentPromoted = false;
    let previousMoved = false;
    let promoted = false;
    const rollbackHostLedgers = async (): Promise<void> => {
      if (conversationPromoted) {
        await rm(conversationPath, { force: true }).catch(() => undefined);
        conversationPromoted = false;
      }
      if (activityPromoted) {
        await rm(activityPath, { force: true }).catch(() => undefined);
        activityPromoted = false;
      }
      if (conversationBackedUp && await pathExists(conversationBackup)) {
        await rename(conversationBackup, conversationPath);
        conversationBackedUp = false;
      }
      if (activityBackedUp && await pathExists(activityBackup)) {
        await rename(activityBackup, activityPath);
        activityBackedUp = false;
      }
    };
    try {
      if (summary.durableState) {
        await mkdir(join(stagedHost, "conversation"), { recursive: true });
        await mkdir(join(stagedHost, "activity"), { recursive: true });
        await copyFile(
          join(base, "conversation", "ledger.sqlite3"),
          join(stagedHost, "conversation", "ledger.sqlite3")
        );
        await copyFile(
          BrainActivityLedger.databasePath(base),
          BrainActivityLedger.databasePath(stagedHost)
        );
        const stagedConversation = await ConversationLedger.open(
          stagedHost,
          id
        );
        try {
          stagedConversation.integrity();
        } finally {
          stagedConversation.close();
        }
        const stagedActivity = await BrainActivityLedger.open(stagedHost, id);
        try {
          stagedActivity.integrity();
        } finally {
          stagedActivity.close();
        }
      }
      if (await pathExists(targetEngine)) {
        // Preserve append-only events, immutable origin, artifacts, and other
        // non-generation state while replacing the neural generation below.
        await cp(targetEngine, stagedEngine, {
          recursive: true,
          force: false,
          errorOnExist: true,
          preserveTimestamps: true
        });
        operation?.signal.throwIfAborted();
      } else {
        await mkdir(stagedEngine, { recursive: true });
      }
      const metadata = await readFile(join(engineSnapshot, "brain.json"));
      const metadataValue = JSON.parse(metadata.toString("utf8")) as unknown;
      for (const name of ["core.safetensors", "plasticity.safetensors"]) {
        operation?.signal.throwIfAborted();
        const sourcePath = join(engineSnapshot, name);
        if (!(await pathExists(sourcePath))) continue;
        const sourceInfo = await lstat(sourcePath);
        await operation?.checkDisk(this.root, sourceInfo.size);
        await copyMutableFileIsolated(
          sourcePath,
          join(stagedEngine, name)
        );
      }
      await this.copyPackedTernaryDirectory(
        join(engineSnapshot, "packed-ternary"),
        join(stagedEngine, "packed-ternary"),
        operation
      );
      await this.copySubstrateSnapshot(
        engineSnapshot,
        stagedEngine,
        metadataValue,
        operation
      );
      await this.copyMutableStateSnapshot(
        engineSnapshot,
        stagedEngine,
        metadataValue,
        operation
      );
      await this.copySavedContinuation(engineSnapshot, stagedEngine, metadataValue, operation);
      if (summary.durableState) {
        await rm(join(stagedEngine, "artifacts"), {
          recursive: true,
          force: true
        });
        await this.copyArtifactSnapshot(
          engineSnapshot,
          stagedEngine,
          id,
          operation
        );
        const neuralConversation = join(
          engineSnapshot,
          "conversation.sqlite3"
        );
        if (
          summary.durableState?.neuralConversationLedger &&
          !(await pathExists(neuralConversation))
        ) {
          throw new Error(
            "Recovery point is missing its neural conversation ledger."
          );
        }
        if (await pathExists(neuralConversation)) {
          await copyFile(
            neuralConversation,
            join(stagedEngine, "conversation.sqlite3")
          );
        } else {
          await rm(join(stagedEngine, "conversation.sqlite3"), {
            force: true
          });
        }
      }
      // Metadata is the staged generation's final commit record.
      await atomicWrite(join(stagedEngine, "brain.json"), metadata);

      if (await pathExists(targetEngine)) {
        await rename(targetEngine, previousEngine);
        previousMoved = true;
      }
      await rename(stagedEngine, targetEngine);
      promoted = true;
      try {
        if (summary.durableState) {
          await rename(conversationPath, conversationBackup);
          conversationBackedUp = true;
          await rename(activityPath, activityBackup);
          activityBackedUp = true;
          await rename(
            join(stagedHost, "conversation", "ledger.sqlite3"),
            conversationPath
          );
          conversationPromoted = true;
          await rename(
            BrainActivityLedger.databasePath(stagedHost),
            activityPath
          );
          activityPromoted = true;
        }
        await this.save(restored);
        documentPromoted = true;
        await operation?.checkpoint({
          phase: "reloading",
          label: "Reloading the exact recovered neural state"
        });
        await afterPromotion?.();
        await rm(previousEngine, { recursive: true, force: true }).catch(
          () => undefined
        );
        previousMoved = false;
        await Promise.all([
          rm(conversationBackup, { force: true }).catch(() => undefined),
          rm(activityBackup, { force: true }).catch(() => undefined)
        ]);
        conversationBackedUp = false;
        activityBackedUp = false;
        conversationPromoted = false;
        activityPromoted = false;
        documentPromoted = false;
        return this.get(id);
      } catch (error) {
        await rollbackHostLedgers();
        await rename(targetEngine, failedEngine);
        promoted = false;
        if (previousMoved) {
          await rename(previousEngine, targetEngine);
          previousMoved = false;
        }
        await rm(failedEngine, { recursive: true, force: true });
        if (documentPromoted) {
          await this.save(current);
          documentPromoted = false;
        }
        throw error;
      }
    } catch (error) {
      await rollbackHostLedgers();
      if (promoted && (await pathExists(targetEngine))) {
        await rename(targetEngine, failedEngine).catch(() => undefined);
        promoted = false;
      }
      if (previousMoved && (await pathExists(previousEngine))) {
        await rename(previousEngine, targetEngine).catch(() => undefined);
        previousMoved = false;
      }
      if (documentPromoted) {
        await this.save(current);
        documentPromoted = false;
      }
      throw error;
    } finally {
      await Promise.all([
        rm(stagedEngine, { recursive: true, force: true }),
        rm(stagedHost, { recursive: true, force: true }),
        rm(failedEngine, { recursive: true, force: true }),
        conversationBackedUp
          ? Promise.resolve()
          : rm(conversationBackup, { force: true }),
        activityBackedUp
          ? Promise.resolve()
          : rm(activityBackup, { force: true }),
        previousMoved ? Promise.resolve() : rm(previousEngine, { recursive: true, force: true })
      ]);
    }
  }

  async exportBundle(
    id: string,
    destination: string,
    mode: BrainExportMode = "current",
    operation?: BrainStorageOperationHooks
  ): Promise<void> {
    return withBrainWrite(this, id, () => this.exportBundleUnlocked(id, destination, mode, operation), operation?.signal);
  }

  private async exportBundleUnlocked(
    id: string,
    destination: string,
    mode: BrainExportMode,
    operation?: BrainStorageOperationHooks
  ): Promise<void> {
    await operation?.checkpoint({
      phase: "planning",
      label: "Validating export sources and checksums"
    });
    const activityExportRoot = await mkdtemp(join(this.root, ".activity-export-"));
    try {
    const currentBrain = await this.get(id);
    const directory = this.brainDirectory(id);
    const engineDirectory =
      mode === "origin" ? join(directory, "engine", "origin") : join(directory, "engine");
    if (currentBrain.readiness.state !== "ready" ||
      !await pathExists(join(engineDirectory, "brain.json")) ||
      !await pathExists(join(directory, "engine", "origin", "brain.json"))) {
      throw new Error("Finish initial learning before exporting this mind; a saved native checkpoint and its materialized origin are required.");
    }
    const storedOriginBytes = await readFile(join(directory, "origin.json"));
    const storedOriginValue: unknown = JSON.parse(storedOriginBytes.toString("utf8"));
    const storedOriginChecksum = assertOriginChecksum(storedOriginValue, "The stored origin");
    const originBrain = normalizeBrain(storedOriginValue);
    const currentDocumentBytes = await readFile(this.documentPath(id));
    if (currentBrain.originChecksum !== storedOriginChecksum) {
      throw new Error("The current brain does not reference its immutable origin checksum.");
    }
    let portableBrain =
      mode === "origin"
        ? clone(storedOriginValue) as BrainDocument
        : JSON.parse(currentDocumentBytes.toString("utf8")) as BrainDocument;
    normalizeBrain(portableBrain);
    const activityExportBrainDirectory = join(activityExportRoot, "brain");
    let rawEpisodesPresent = false;
    let activitySummary: BrainActivityLedgerSummary;
    if (mode === "origin") {
      await BrainActivityLedger.replace(
        activityExportBrainDirectory,
        portableBrain.id,
        portableBrain.journal ?? [],
        portableBrain.trainingSources
      );
    } else {
      await snapshotSavedSqlite(
        BrainActivityLedger.databasePath(directory),
        BrainActivityLedger.databasePath(activityExportBrainDirectory),
        operation
      );
    }
    const activity = await BrainActivityLedger.open(activityExportBrainDirectory, portableBrain.id);
    try {
      activitySummary = activity.integrity();
    } finally {
      activity.close();
    }
    portableBrain.activity = activitySummary;
    if (mode !== "origin") {
      await snapshotSavedSqlite(
        join(directory, "conversation", "ledger.sqlite3"),
        join(activityExportBrainDirectory, "conversation", "ledger.sqlite3"),
        operation
      );
    }
    const conversation = await ConversationLedger.open(activityExportBrainDirectory, portableBrain.id);
    try {
      if (mode === "origin") conversation.backfill(portableBrain.messages, portableBrain.traces);
      portableBrain.conversation = conversation.integrity();
    } finally {
      conversation.close();
    }
    const engineStatePath = join(engineDirectory, "brain.json");
    const engineMaterialized = true;
    const selectedPackedPath = join(engineDirectory, "packed-ternary");
    const selectedPacked =
      engineMaterialized && (await pathExists(join(selectedPackedPath, "manifest.json")))
        ? await inspectPackedTernaryDirectory(selectedPackedPath, "Current")
        : undefined;
    const engineState = engineMaterialized
      ? savedEngineState(await readFile(engineStatePath))
      : strToU8(
          JSON.stringify(
            {
              format: "omni-engine-unmaterialized",
              brain_id: portableBrain.id,
              name: portableBrain.name
            },
            null,
            2
          )
        );
    if (engineMaterialized && !selectedPacked) {
      throw new Error(
        "Materialized OmniCortex state is missing its verified packed ternary inference shards."
      );
    }
    const corePath = join(engineDirectory, "core.safetensors");
    const plasticityPath = join(engineDirectory, "plasticity.safetensors");
    const coreExists = await pathExists(corePath);
    const plasticityExists = await pathExists(plasticityPath);
    const coreFallback = coreExists
      ? undefined
      : validEmptySafetensors("Neural core has not been materialized by the Python worker.");
    const plasticityFallback = plasticityExists
      ? undefined
      : validEmptySafetensors("Plastic state is represented by state/brain.json.");
    if (coreExists) await assertSafeTensorsFile(corePath, "core.safetensors");
    else assertSafeTensors(coreFallback!, "core.safetensors");
    if (plasticityExists) {
      await assertSafeTensorsFile(plasticityPath, "plastic.safetensors");
    } else assertSafeTensors(plasticityFallback!, "plastic.safetensors");
    const entries: Record<string, StreamingZipSource> = {
      "model-card.md": {
        name: "model-card.md",
        contents: strToU8(
          `# ${portableBrain.name}\n\nOmniCortex brain ${portableBrain.id}.\n\n` +
            `Preset: ${portableBrain.config.preset}\n\n` +
            `Memory recipe: ${portableBrain.config.memoryRecipe ?? "adaptive-retention"}\n\n` +
            `Unsanitized saved-instance export: chat, activity, artifacts, workspace, ` +
            `pending replay jobs and ingestion cursors are preserved. Share only with trusted recipients. ` +
            `OS-vault credentials are not included. Import starts dormant; a collision rekeys the live identity. ` +
            `Dataset continuation retains its original absolute-path/stat, content, kind/policy/epoch ` +
            `and parser bindings; relocated sources are not automatically remapped. ` +
            `This is not an in-flight process clone.\n`
        )
      },
      "state/brain.json": {
        name: "state/brain.json",
        contents: strToU8(JSON.stringify(portableBrain, null, 2))
      },
      "state/engine.json": {
        name: "state/engine.json",
        contents: engineState
      },
      "activity/ledger.sqlite3": {
        name: "activity/ledger.sqlite3",
        sourcePath: BrainActivityLedger.databasePath(activityExportBrainDirectory)
      },
      "conversation/ledger.sqlite3": {
        name: "conversation/ledger.sqlite3",
        sourcePath: join(activityExportBrainDirectory, "conversation", "ledger.sqlite3")
      },
      "tensors/core.safetensors": {
        name: "tensors/core.safetensors",
        ...(coreExists ? { sourcePath: corePath } : { contents: coreFallback })
      },
      "tensors/plastic.safetensors": {
        name: "tensors/plastic.safetensors",
        ...(plasticityExists ? { sourcePath: plasticityPath } : { contents: plasticityFallback })
      }
    };
    const currentGeometryState = JSON.parse(Buffer.from(engineState).toString("utf8")) as unknown;
    for (const [relative, sourcePath] of await savedGeometryHoldoutFiles(currentGeometryState, engineDirectory)) {
      const name = `geometry/current/${relative}`;
      entries[name] = { name, sourcePath };
    }
    const includeSavedSourceBlob = async (source: TrainingSource): Promise<void> => {
      if (source.rawTextRetained || source.rawText !== undefined) rawEpisodesPresent = true;
      if (!source.blobHash || entries[`blobs/${source.blobHash}`]) return;
      if (!/^[a-f0-9]{64}$/.test(source.blobHash)) throw new Error("Saved source blob reference is invalid.");
      const blobPath = join(this.root, ".blobs", source.blobHash);
      if (await streamFileSha256(blobPath) !== source.blobHash) throw new Error("Saved source blob checksum failed.");
      const name = `blobs/${source.blobHash}`;
      entries[name] = { name, sourcePath: blobPath };
    };
    for (const source of originBrain.trainingSources) await includeSavedSourceBlob(source);
    for (const [artifactEngine, artifactBrainId, prefix] of [
      [engineDirectory, portableBrain.id, "artifacts"],
      [join(directory, "engine", "origin"), originBrain.id, "origin/artifacts"]
    ] as const) {
      const artifactDirectory = join(artifactEngine, "artifacts");
      const artifactStore = await ArtifactIndexStore.openExisting(
        artifactDirectory,
        artifactBrainId
      );
      if (artifactStore) {
        let artifactIndex: PersistedArtifactIndex;
        try {
          artifactIndex = artifactStore.snapshot();
        } finally {
          artifactStore.close();
        }
        const artifactSnapshotDirectory = join(activityExportRoot, prefix);
        const artifactLedgerSnapshot = join(artifactSnapshotDirectory, "index.sqlite3");
        await snapshotSavedSqlite(
          ArtifactIndexStore.databasePath(artifactDirectory), artifactLedgerSnapshot, operation
        );
        const savedArtifacts = await ArtifactIndexStore.open(artifactSnapshotDirectory, artifactBrainId);
        try { artifactIndex = savedArtifacts.snapshot(); } finally { savedArtifacts.close(); }
        entries[`${prefix}/index.sqlite3`] = { name: `${prefix}/index.sqlite3`, sourcePath: artifactLedgerSnapshot };
        entries[`${prefix}/index.json`] = {
          name: `${prefix}/index.json`,
          contents: strToU8(
            serializeArtifactIndex(artifactIndex.brainId, artifactIndex.artifacts)
          )
        };
        for (const artifact of artifactIndex.artifacts) {
          const name = basename(artifact.relativePath);
          const sourcePath = join(artifactDirectory, name);
          const info = await lstat(sourcePath);
          if (
            !info.isFile() ||
            info.isSymbolicLink() ||
            info.size !== artifact.bytes ||
            await streamFileSha256(sourcePath) !== artifact.sha256
          ) {
            throw new Error(`Generated artifact failed export integrity: ${name}`);
          }
          entries[`${prefix}/files/${name}`] = {
            name: `${prefix}/files/${name}`,
            sourcePath
          };
        }
      }
    }
    if (selectedPacked) {
      for (const [name, sourcePath] of selectedPacked.files) {
        const archivePath = `packed/current/${name}`;
        entries[archivePath] = { name: archivePath, sourcePath };
      }
    }
    // The original is immutable, including its content checksum and user
    // annotations. A different exported ledger does not redefine the origin.
    portableBrain.originChecksum = storedOriginChecksum;
    entries["state/brain.json"] = {
      name: "state/brain.json",
      contents: strToU8(JSON.stringify(portableBrain, null, 2))
    };
    const immutableEngine = join(directory, "engine", "origin");
    const immutableStatePath = join(immutableEngine, "brain.json");
    const immutableProvenancePath = join(immutableEngine, "provenance.json");
    const immutableStateExists = await pathExists(immutableStatePath);
    const immutableProvenanceExists = await pathExists(immutableProvenancePath);
    if (immutableProvenanceExists) {
      throw new Error("Legacy bundled foundation provenance is not supported.");
    }
    let immutableStateBytes: Buffer | undefined;
    let immutableStateMetadata: unknown;
    if (immutableStateExists) {
      const [originDirectoryInfo, stateInfo] = await Promise.all([
        lstat(immutableEngine),
        lstat(immutableStatePath)
      ]);
      if (
        !originDirectoryInfo.isDirectory() ||
        originDirectoryInfo.isSymbolicLink() ||
        !stateInfo.isFile() ||
        stateInfo.isSymbolicLink()
      ) {
        throw new Error("The immutable origin contains an unsafe filesystem link.");
      }
      immutableStateBytes = await readFile(immutableStatePath);
      try {
        immutableStateMetadata = JSON.parse(immutableStateBytes.toString("utf8"));
      } catch {
        throw new Error("The verified origin engine metadata is invalid.");
      }
      assertUiNeuralOriginIdentity(originBrain, immutableStateMetadata);
      assertNativeOmniEngineState(immutableStateMetadata, "Immutable origin");
    }
    const immutableCorePath = join(immutableEngine, "core.safetensors");
    const immutablePlasticPath = join(immutableEngine, "plasticity.safetensors");
    const immutableCoreExists = await pathExists(immutableCorePath);
    const immutablePlasticExists = await pathExists(immutablePlasticPath);
    if (immutableCoreExists) {
      await assertSafeTensorsFile(immutableCorePath, "origin core.safetensors");
    }
    if (immutablePlasticExists) {
      await assertSafeTensorsFile(immutablePlasticPath, "origin plastic.safetensors");
    }
    const immutablePackedPath = join(immutableEngine, "packed-ternary");
    const immutablePacked =
      immutableStateExists && (await pathExists(join(immutablePackedPath, "manifest.json")))
        ? await inspectPackedTernaryDirectory(immutablePackedPath, "Origin")
        : mode === "origin"
          ? selectedPacked
          : undefined;
    if (engineMaterialized && !immutablePacked) {
      throw new Error(
        "Materialized OmniCortex state is missing its verified packed ternary shards."
      );
    }
    const immutableState = immutableStateBytes
      ? savedEngineState(immutableStateBytes)
      : mode === "origin"
        ? engineState
        : strToU8(
            JSON.stringify(
              {
                format: "omni-engine-unmaterialized",
                brain_id: originBrain.id,
                name: originBrain.name
              },
              null,
              2
            )
          );
    const references =
      mode === "referenced"
        ? {
            currentCore: coreExists
              ? await this.storeFileAsBlob(corePath)
              : await this.storeBlob(Buffer.from(coreFallback!)),
            currentPlasticity: plasticityExists
              ? await this.storeFileAsBlob(plasticityPath)
              : await this.storeBlob(Buffer.from(plasticityFallback!)),
            originCore: immutableCoreExists
              ? await this.storeFileAsBlob(immutableCorePath)
              : coreExists
                ? await this.storeFileAsBlob(corePath)
                : await this.storeBlob(Buffer.from(coreFallback!)),
            originPlasticity: immutablePlasticExists
              ? await this.storeFileAsBlob(immutablePlasticPath)
              : plasticityExists
                ? await this.storeFileAsBlob(plasticityPath)
                : await this.storeBlob(Buffer.from(plasticityFallback!))
          }
        : undefined;
    if (references) {
      entries["tensors/core.safetensors"] = {
        name: "tensors/core.safetensors",
        contents: validEmptySafetensors(`Local content reference ${references.currentCore}`)
      };
      entries["tensors/plastic.safetensors"] = {
        name: "tensors/plastic.safetensors",
        contents: validEmptySafetensors(`Local content reference ${references.currentPlasticity}`)
      };
    }
    entries["origin/state/brain.json"] = {
      name: "origin/state/brain.json",
      contents: storedOriginBytes
    };
    entries["origin/state/engine.json"] = {
      name: "origin/state/engine.json",
      contents: immutableState
    };
    const originGeometryEngine = immutableStateExists ? immutableEngine : engineDirectory;
    const originGeometryState = JSON.parse(Buffer.from(immutableState).toString("utf8")) as unknown;
    for (const [relative, sourcePath] of await savedGeometryHoldoutFiles(originGeometryState, originGeometryEngine)) {
      const name = `geometry/origin/${relative}`;
      entries[name] = { name, sourcePath };
    }
    entries["origin/tensors/core.safetensors"] = {
      name: "origin/tensors/core.safetensors",
      ...(references
        ? {
            contents: validEmptySafetensors(`Local content reference ${references.originCore}`)
          }
        : immutableCoreExists
          ? { sourcePath: immutableCorePath }
          : coreExists
            ? { sourcePath: corePath }
            : { contents: coreFallback })
    };
    entries["origin/tensors/plastic.safetensors"] = {
      name: "origin/tensors/plastic.safetensors",
      ...(references
        ? {
            contents: validEmptySafetensors(
              `Local content reference ${references.originPlasticity}`
            )
          }
        : immutablePlasticExists
          ? { sourcePath: immutablePlasticPath }
          : plasticityExists
            ? { sourcePath: plasticityPath }
            : { contents: plasticityFallback })
    };
    if (immutablePacked) {
      for (const [name, sourcePath] of immutablePacked.files) {
        const archivePath = `packed/origin/${name}`;
        entries[archivePath] = { name: archivePath, sourcePath };
      }
    }
    let packedReferences:
      | {
          current: Record<string, string>;
          origin: Record<string, string>;
        }
      | undefined;
    if (references && selectedPacked && immutablePacked) {
      packedReferences = { current: {}, origin: {} };
      for (const [scope, packed] of [
        ["current", selectedPacked],
        ["origin", immutablePacked]
      ] as const) {
        for (const [name, sourcePath] of packed.files) {
          const hash = await this.storeFileAsBlob(sourcePath);
          packedReferences[scope][name] = hash;
          const archivePath = `packed/${scope}/${name}`;
          entries[archivePath] = {
            name: archivePath,
            contents: strToU8(`Local content reference ${hash}\n`)
          };
        }
      }
    }
    {
      const portableActivity = new DatabaseSync(
        BrainActivityLedger.databasePath(activityExportBrainDirectory), { readOnly: true }
      );
      try {
        // Historical source versions are saved state too, not merely the
        // latest training-source projection.
        for (const row of portableActivity.prepare(
          "SELECT payload_json FROM training_source_versions ORDER BY sequence"
        ).iterate()) {
          const source = JSON.parse(String(row.payload_json)) as TrainingSource;
          await includeSavedSourceBlob(source);
        }
      } finally {
        portableActivity.close();
      }
    }
    const currentEngineMetadata = engineMaterialized
      ? (JSON.parse(Buffer.from(engineState).toString("utf8")) as unknown)
      : undefined;
    const originEngineMetadata = Buffer.from(immutableState)
      .toString("utf8")
      .includes("omni-cortex-engine")
      ? (JSON.parse(Buffer.from(immutableState).toString("utf8")) as unknown)
      : undefined;
    for (const [sourceEngine, metadata, scope, sourceBrainId, ledgerPrefix] of [
      [engineDirectory, currentEngineMetadata, "current", portableBrain.id, "conversation"],
      [immutableEngine, originEngineMetadata, "origin", originBrain.id, "origin/conversation"]
    ] as const) {
      if (!metadata) continue;
      for (const [relative, path] of await savedConceptIdViewFiles(sourceEngine)) {
        const name = `concept-views/${scope}/${relative.split("/").at(-1)}`;
        entries[name] = { name, sourcePath: path };
      }
      for (const [file, path] of await savedToolIntentFiles(sourceEngine)) {
        const name = `operational/${scope}/${file}`;
        entries[name] = { name, sourcePath: path };
      }
      const ledgerPath = join(sourceEngine, "conversation.sqlite3");
      const hasLedger = await pathExists(ledgerPath);
      assertImportedNeuralConversationHead(metadata, `Saved ${scope} state`, hasLedger);
      if (hasLedger) {
        const snapshot = join(activityExportRoot, `${scope}-neural-ledger.sqlite3`);
        await snapshotSavedSqlite(ledgerPath, snapshot, operation);
        validateNeuralConversationLedger(snapshot, sourceBrainId, metadata);
        const name = `${ledgerPrefix}/neural-ledger.sqlite3`;
        entries[name] = { name, sourcePath: snapshot };
      } else if (isRecord(metadata) && Array.isArray(metadata.pending_chat_slow_learning) &&
        metadata.pending_chat_slow_learning.length) {
        throw new Error("Pending chat replay is missing its saved neural conversation ledger.");
      }
      const workingPath = join(sourceEngine, "state", "working-memory.sqlite3");
      let snapshot: string | undefined;
      if (await pathExists(workingPath)) {
        snapshot = join(activityExportRoot, `${scope}-working-memory.sqlite3`);
        await snapshotSavedSqlite(workingPath, snapshot, operation);
        const name = `working/${scope}/working-memory.sqlite3`;
        entries[name] = { name, sourcePath: snapshot };
      }
      validateSavedWorkingPages(snapshot, metadata);
      const joint = await savedJointGenerationFiles(metadata, (kind, relative) =>
        join(sourceEngine, "state", ...(kind === "joint" ? ["ingestion-joint"] : []), ...relative.split("/"))
      );
      for (const [relative, sourcePath] of joint) {
        const name = `joint/${scope}/${relative}`;
        entries[name] = { name, sourcePath };
      }
    }
    const [currentSubstrate, originSubstrate] = await Promise.all([
      collectSubstrateSnapshot(
        engineDirectory,
        "substrate/current",
        currentEngineMetadata,
        operation?.signal
      ),
      collectSubstrateSnapshot(
        immutableEngine,
        "substrate/origin",
        originEngineMetadata,
        operation?.signal
      )
    ]);
    // SQLite validation sees committed WAL pages, while ZIP streams sourcePath
    // bytes. Materialize each replay into a self-contained, stable SQLite file
    // before validation, manifest hashing, or ZIP streaming.
    const currentMutableState = await collectMutableStateSnapshot(
      engineDirectory,
      "mutable/current",
      currentEngineMetadata,
      operation?.signal,
      undefined,
      {
        destination: join(activityExportRoot, "mutable-current-replay.sqlite3"),
        ...(operation ? {
          checkDisk: async (bytes: number) => {
            await operation.checkDisk(this.root, bytes);
          }
        } : {})
      }
    );
    const originMutableState = await collectMutableStateSnapshot(
      immutableEngine,
      "mutable/origin",
      originEngineMetadata,
      operation?.signal,
      undefined,
      {
        destination: join(activityExportRoot, "mutable-origin-replay.sqlite3"),
        ...(operation ? {
          checkDisk: async (bytes: number) => {
            await operation.checkDisk(this.root, bytes);
          }
        } : {})
      }
    );
    // Content-addressed JSON must retain its exact bytes and generation
    // hashes. Validate text integrity without screening private saved content.
    await assertSubstrateSnapshotJsonText(currentSubstrate, operation?.signal);
    await assertSubstrateSnapshotJsonText(originSubstrate, operation?.signal);
    for (const snapshot of [
      currentSubstrate,
      originSubstrate,
      currentMutableState,
      originMutableState
    ]) {
      for (const source of snapshot?.sources ?? []) entries[source.name] = source;
    }
    const recoveryPoints: NonNullable<OmniManifest["recoveryPoints"]> = [];
    const snapshotRoot = join(directory, "snapshots");
    if (await pathExists(snapshotRoot)) {
      for (const entry of await readdir(snapshotRoot, { withFileTypes: true })) {
        if (!entry.name.endsWith(".meta.json")) continue;
        if (!entry.isFile() || entry.isSymbolicLink()) throw new Error("Saved recovery-point record is unsafe.");
        const snapshotId = requireSafeId(entry.name.slice(0, -".meta.json".length), "recovery point id");
        const sourceBase = join(snapshotRoot, snapshotId);
        const stagedBase = join(activityExportRoot, "snapshots", snapshotId);
        const sourceFiles = await pathExists(sourceBase) ? await savedTreeFiles(sourceBase) : new Map<string, string>();
        sourceFiles.set("../document", `${sourceBase}.json`);
        sourceFiles.set("../summary", `${sourceBase}.meta.json`);
        for (const [relative, source] of sourceFiles) {
          const target = relative === "../document" ? `${stagedBase}.json`
            : relative === "../summary" ? `${stagedBase}.meta.json`
              : join(stagedBase, ...relative.split("/"));
          await operation?.checkDisk(this.root, (await lstat(source)).size);
          await copyMutableFileIsolated(source, target);
        }
        const checked = await validateSavedSnapshot(stagedBase, snapshotId, currentBrain.id);
        recoveryPoints.push({ id: snapshotId, brainId: currentBrain.id, missingPayloads: checked.missingPayloads });
        for (const relative of sourceFiles.keys()) {
          const name = relative === "../document" ? `snapshots/${snapshotId}.json`
            : relative === "../summary" ? `snapshots/${snapshotId}.meta.json`
              : `snapshots/${snapshotId}/${relative}`;
          const stagedPath = relative === "../document" ? `${stagedBase}.json`
            : relative === "../summary" ? `${stagedBase}.meta.json`
              : join(stagedBase, ...relative.split("/"));
          entries[name] = { name, sourcePath: stagedPath };
          // Historical activity can reference blobs absent from the live
          // activity generation after a restore. Preserve those references.
          if (relative === "../document") {
            for (const source of (JSON.parse(await readFile(stagedPath, "utf8")) as BrainDocument).trainingSources ?? []) {
              await includeSavedSourceBlob(source);
            }
          }
          if (relative === "activity/ledger.sqlite3") {
            const database = new DatabaseSync(stagedPath, { readOnly: true });
            try {
              for (const row of database.prepare("SELECT payload_json FROM training_source_versions ORDER BY sequence").iterate()) {
                await includeSavedSourceBlob(JSON.parse(String(row.payload_json)) as TrainingSource);
              }
            } finally { database.close(); }
          }
        }
      }
    }
    const missingHistory = recoveryPoints.filter((snapshot) => snapshot.missingPayloads.length);
    if (missingHistory.length) {
      const modelCard = Buffer.from(entries["model-card.md"]!.contents!).toString("utf8");
      entries["model-card.md"]!.contents = strToU8(modelCard + "\nAlready-missing historical recovery payloads (records retained, not claimed restorable):\n" +
        missingHistory.map((snapshot) => `${snapshot.id}: ${snapshot.missingPayloads.join(", ")}`).join("\n") + "\n");
    }
    const fileRecords: Record<string, { sha256: string; bytes: number }> = {};
    const stagedSourceRoot = join(activityExportRoot, "sources");
    for (const [path, source] of Object.entries(entries)) {
      operation?.signal.throwIfAborted();
      if (source.contents) {
        fileRecords[path] = {
          sha256: sha256(Buffer.from(source.contents)),
          bytes: source.contents.byteLength
        };
      } else if (source.sourcePath) {
        const info = await lstat(source.sourcePath);
        if (!info.isFile() || info.isSymbolicLink()) {
          throw new Error(`Bundle source ${path} is not a regular file.`);
        }
        // Streaming must read the same bytes that the manifest hashes. A
        // private reflink/byte copy pins mutable tensor mirrors and pointers
        // while preserving the content-addressed generation unchanged.
        if (!source.sourcePath.startsWith(`${activityExportRoot}/`)) {
          await operation?.checkDisk(this.root, info.size);
          await ensureDiskReserve(this.root, info.size);
          const staged = join(stagedSourceRoot, ...path.split("/"));
          await copyMutableFileIsolated(source.sourcePath, staged);
          source.sourcePath = staged;
        }
        fileRecords[path] = {
          sha256: await streamFileSha256(source.sourcePath),
          bytes: info.size
        };
      } else throw new Error(`Bundle source ${path} is empty.`);
    }
    if (!Buffer.from(await readFile(join(directory, "origin.json"))).equals(storedOriginBytes) ||
      (mode !== "origin" && !Buffer.from(await readFile(this.documentPath(id))).equals(currentDocumentBytes)) ||
      (engineMaterialized && !Buffer.from(await readFile(engineStatePath)).equals(Buffer.from(engineState))) ||
      (immutableStateBytes && !Buffer.from(await readFile(immutableStatePath)).equals(immutableStateBytes))) {
      throw new Error("Saved instance changed during export; retry after its checkpoint finishes.");
    }
    const manifest: OmniManifest = {
      format: BUNDLE_FORMAT,
      formatVersion: BUNDLE_VERSION,
      releaseFormat: STABLE_RELEASE_FORMAT,
      architecture: "OmniCortex",
      architectureSchemaVersion: portableBrain.schemaVersion,
      exportedAt: new Date().toISOString(),
      brain: {
        id: portableBrain.id,
        name: portableBrain.name,
        lineage: portableBrain.lineage
      },
      mode:
        mode === "origin"
          ? "origin-portable"
          : mode === "private-archive"
            ? "private-archive"
            : mode === "referenced"
              ? "referenced-local"
              : "current-portable",
      engineMaterialized,
      memoryRecipe: portableBrain.config.memoryRecipe ?? "adaptive-retention",
      rawEpisodesPresent,
      quantization: "ternary-effective",
      conversationProjection: {
        historyIncluded: true,
        omittedLedgerRows: 0,
        omittedPendingReplayJobs: 0
      },
      packedTernary:
        selectedPacked && immutablePacked
          ? {
              format: "omni-packed-ternary",
              formatVersion: 1,
              currentManifestSha256: selectedPacked.manifestSha256,
              originManifestSha256: immutablePacked.manifestSha256,
              currentTensorCount: selectedPacked.tensorCount,
              originTensorCount: immutablePacked.tensorCount,
              references: packedReferences
            }
          : undefined,
      secretRedaction: {
        version: 1,
        replacements: 0
      },
      savedInstance: { version: 1, content: "unsanitized" },
      recoveryPoints,
      licenseLedger: {
        application: "PolyForm-Noncommercial-1.0.0-or-commercial-license",
        sourceCount: activitySummary.trainingSourceCount,
        sourceLedger: "activity/ledger.sqlite3",
        sources: []
      },
      references,
      files: fileRecords
    };
    const manifestContents = strToU8(JSON.stringify(manifest, null, 2));
    entries["manifest.json"] = {
      name: "manifest.json",
      contents: manifestContents
    };
    const checksumRecords = {
      ...fileRecords,
      "manifest.json": {
        sha256: sha256(Buffer.from(manifestContents)),
        bytes: manifestContents.byteLength
      }
    };
    entries["checksums.sha256"] = {
      name: "checksums.sha256",
      contents: strToU8(
        Object.entries(checksumRecords)
          .map(([path, descriptor]) => `${descriptor.sha256}  ${path}`)
          .sort()
          .join("\n") + "\n"
      )
    };
    await writeStreamingZip(
      destination,
      Object.values(entries).sort((left, right) => left.name.localeCompare(right.name)),
      {
        signal: operation?.signal,
        checkDisk: operation
          ? (path, bytes) => operation.checkDisk(path, bytes)
          : undefined,
        checkpoint: operation
          ? (progress) => operation.checkpoint({
              phase: "writing",
              label:
                mode === "referenced"
                  ? "Writing verified local-reference archive"
                  : mode === "private-archive"
                    ? "Writing verified private archive"
                    : "Writing verified portable archive",
              filesCompleted: progress.filesCompleted,
              filesTotal: progress.filesTotal,
              logicalBytesCompleted: progress.bytesCompleted,
              logicalBytesTotal: progress.bytesTotal,
              physicalBytesAdded: progress.bytesCompleted,
              sharedBytes: 0
            })
          : undefined
      }
    );
    if (mode === "referenced" && references) {
      await this.recordReferencedBundleLease(destination, [
        ...Object.values(references),
        ...Object.values(packedReferences?.current ?? {}),
        ...Object.values(packedReferences?.origin ?? {})
      ]);
    }
    } finally {
      await removeTreeWithRetry(activityExportRoot);
    }
  }

  async importBundle(
    path: string,
    options: { initializing?: boolean } = {},
    operation?: BrainStorageOperationHooks
  ): Promise<BrainDocument> {
    await this.initialize();
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink()) {
      throw new Error("The selected .omni bundle is not a regular file.");
    }
    const extracted = await extractStreamingZip(
      path,
      join(this.root, ".imports"),
      {
        signal: operation?.signal,
        checkDisk: operation
          ? (directory, bytes) => operation.checkDisk(directory, bytes)
          : undefined,
        checkpoint: operation
          ? (progress) => operation.checkpoint({
              phase: "extracting",
              label: "Extracting and validating portable archive",
              filesCompleted: progress.filesCompleted,
              filesTotal: progress.filesTotal,
              logicalBytesCompleted: progress.bytesCompleted,
              logicalBytesTotal: progress.bytesTotal,
              physicalBytesAdded: progress.bytesCompleted,
              sharedBytes: 0
            })
          : undefined
      }
    );
    try {
      return await this.importExtractedBundle(
        extracted,
        basename(path),
        options,
        operation
      );
    } finally {
      await removeTreeWithRetry(extracted.root);
    }
  }

  async importBundleBuffer(
    contents: Buffer,
    sourceLabel = "download.omni",
    options: { initializing?: boolean } = {},
    operation?: BrainStorageOperationHooks
  ): Promise<BrainDocument> {
    await this.initialize();
    const temporary = await mkdtemp(join(this.root, ".omni-buffer-"));
    const bundlePath = join(temporary, "buffer.omni");
    try {
      await writeFile(bundlePath, contents, { flag: "wx", mode: 0o600 });
      const extracted = await extractStreamingZip(
        bundlePath,
        join(this.root, ".imports"),
        {
          signal: operation?.signal,
          checkDisk: operation
            ? (directory, bytes) => operation.checkDisk(directory, bytes)
            : undefined,
          checkpoint: operation
            ? (progress) => operation.checkpoint({
                phase: "extracting",
                label: "Extracting and validating portable archive",
                filesCompleted: progress.filesCompleted,
                filesTotal: progress.filesTotal,
                logicalBytesCompleted: progress.bytesCompleted,
                logicalBytesTotal: progress.bytesTotal,
                physicalBytesAdded: progress.bytesCompleted,
                sharedBytes: 0
              })
            : undefined
        }
      );
      try {
        return await this.importExtractedBundle(
          extracted,
          sourceLabel,
          options,
          operation
        );
      } finally {
        await removeTreeWithRetry(extracted.root);
      }
    } finally {
      await removeTreeWithRetry(temporary);
    }
  }

  private async importExtractedBundle(
    archive: ExtractedZipArchive,
    sourceLabel: string,
    options: { initializing?: boolean },
    operation?: BrainStorageOperationHooks
  ): Promise<BrainDocument> {
    await operation?.checkpoint({
      phase: "validating",
      label: "Validating archive checksums and neural generations"
    });
    const names = new Set(archive.entries.keys());
    const entryPath = (name: string): string => {
      const entry = archive.entries.get(name);
      if (!entry) throw new Error(`The .omni bundle is missing ${name}.`);
      return entry.path;
    };
    for (const required of [
      "manifest.json",
      "model-card.md",
      "checksums.sha256",
      "state/brain.json",
      "state/engine.json",
      "tensors/core.safetensors",
      "tensors/plastic.safetensors",
      "origin/state/brain.json",
      "origin/state/engine.json",
      "origin/tensors/core.safetensors",
      "origin/tensors/plastic.safetensors"
    ]) {
      if (!names.has(required)) {
        throw new Error(`The .omni bundle is missing ${required}.`);
      }
    }
    const checksums = parseChecksumFile(await readFile(entryPath("checksums.sha256"), "utf8"));
    for (const [path, expected] of checksums) {
      operation?.signal.throwIfAborted();
      const entry = archive.entries.get(path);
      if (!entry) throw new Error(`Checksum references missing file ${path}.`);
      if ((await streamFileSha256(entry.path)) !== expected) {
        throw new Error(`Checksum validation failed for ${path}.`);
      }
    }
    for (const path of names) {
      operation?.signal.throwIfAborted();
      assertAllowedBundlePath(path);
      if (path !== "checksums.sha256" && !checksums.has(path)) {
        throw new Error(`The .omni bundle has no checksum for ${path}.`);
      }
    }
    let manifestValue: unknown;
    try {
      manifestValue = JSON.parse(await readFile(entryPath("manifest.json"), "utf8"));
    } catch {
      throw new Error("manifest.json is invalid.");
    }
    if (
      !isRecord(manifestValue) ||
      manifestValue.format !== BUNDLE_FORMAT ||
      manifestValue.formatVersion !== BUNDLE_VERSION ||
      manifestValue.releaseFormat !== STABLE_RELEASE_FORMAT
    ) {
      throw new Error(
        "Unsupported or beta .omni bundle; stable Omni AGI Studio v1 format is required."
      );
    }
    if (
      manifestValue.architecture !== "OmniCortex" ||
      manifestValue.architectureSchemaVersion !== BRAIN_SCHEMA_VERSION ||
      manifestValue.quantization !== "ternary-effective" ||
      !["current-portable", "origin-portable", "private-archive", "referenced-local"].includes(
        String(manifestValue.mode)
      )
    ) {
      throw new Error("The .omni bundle targets an incompatible architecture schema.");
    }
    const declaredEngineMaterialized = manifestValue.engineMaterialized === true;
    let packedDeclaration: OmniManifest["packedTernary"];
    if (manifestValue.packedTernary !== undefined) {
      const packed = manifestValue.packedTernary;
      if (
        !isRecord(packed) ||
        packed.format !== "omni-packed-ternary" ||
        packed.formatVersion !== 1 ||
        typeof packed.currentManifestSha256 !== "string" ||
        !/^[a-f0-9]{64}$/.test(packed.currentManifestSha256) ||
        typeof packed.originManifestSha256 !== "string" ||
        !/^[a-f0-9]{64}$/.test(packed.originManifestSha256) ||
        typeof packed.currentTensorCount !== "number" ||
        !Number.isSafeInteger(packed.currentTensorCount) ||
        packed.currentTensorCount < 1 ||
        typeof packed.originTensorCount !== "number" ||
        !Number.isSafeInteger(packed.originTensorCount) ||
        packed.originTensorCount < 1
      ) {
        throw new Error("The .omni bundle has an invalid packed ternary declaration.");
      }
      let packedReferenceDeclaration:
        | {
            current: Record<string, string>;
            origin: Record<string, string>;
          }
        | undefined;
      if (packed.references !== undefined) {
        if (
          !isRecord(packed.references) ||
          !isRecord(packed.references.current) ||
          !isRecord(packed.references.origin)
        ) {
          throw new Error("The packed ternary local references are invalid.");
        }
        const normalizeReferences = (value: Record<string, unknown>): Record<string, string> => {
          const result: Record<string, string> = {};
          for (const [name, hash] of Object.entries(value)) {
            if (
              !["manifest.json", "manifest.sha256"].includes(name) &&
              !/^ternary-[0-9]{5,}-[a-f0-9]{16}\.bin$/.test(name)
            ) {
              throw new Error("A packed ternary local reference has an unsafe name.");
            }
            if (typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash)) {
              throw new Error("A packed ternary local reference has an invalid hash.");
            }
            result[name] = hash;
          }
          return result;
        };
        packedReferenceDeclaration = {
          current: normalizeReferences(packed.references.current),
          origin: normalizeReferences(packed.references.origin)
        };
      }
      packedDeclaration = {
        format: "omni-packed-ternary",
        formatVersion: 1,
        currentManifestSha256: packed.currentManifestSha256,
        originManifestSha256: packed.originManifestSha256,
        currentTensorCount: packed.currentTensorCount,
        originTensorCount: packed.originTensorCount,
        references: packedReferenceDeclaration
      };
    }
    if (declaredEngineMaterialized && !packedDeclaration) {
      throw new Error(
        "A materialized stable v1 brain must contain packed ternary inference shards."
      );
    }
    if (packedDeclaration?.references && manifestValue.mode !== "referenced-local") {
      throw new Error("Portable bundles may not contain packed ternary local references.");
    }
    if (
      manifestValue.mode === "referenced-local" &&
      declaredEngineMaterialized &&
      !packedDeclaration?.references
    ) {
      throw new Error("The local referenced bundle has no packed ternary references.");
    }
    if (
      !isRecord(manifestValue.secretRedaction) ||
      manifestValue.secretRedaction.version !== 1 ||
      typeof manifestValue.secretRedaction.replacements !== "number"
    ) {
      throw new Error("The .omni bundle does not declare a supported secret-redaction policy.");
    }
    const savedInstance = manifestValue.savedInstance !== undefined;
    if (savedInstance && (!isRecord(manifestValue.savedInstance) ||
      manifestValue.savedInstance.version !== 1 ||
      manifestValue.savedInstance.content !== "unsanitized" ||
      manifestValue.secretRedaction.replacements !== 0)) {
      throw new Error("The .omni bundle has an invalid saved-instance preservation contract.");
    }
    if (
      !isRecord(manifestValue.licenseLedger) ||
      typeof manifestValue.licenseLedger.application !== "string" ||
      !Array.isArray(manifestValue.licenseLedger.sources) ||
      manifestValue.licenseLedger.sources.some(
        (source) =>
          !isRecord(source) ||
          typeof source.name !== "string" ||
          typeof source.license !== "string" ||
          source.name.length > 4_000 ||
          source.license.length > 4_000 ||
          (source.provenanceUrl !== undefined && typeof source.provenanceUrl !== "string") ||
          (source.licenseUrl !== undefined && typeof source.licenseUrl !== "string")
      )
    ) {
      throw new Error("The .omni bundle does not contain a valid license ledger.");
    }
    const manifestFiles = isRecord(manifestValue.files) ? manifestValue.files : {};
    for (const [path, descriptor] of Object.entries(manifestFiles)) {
      operation?.signal.throwIfAborted();
      if (
        !isRecord(descriptor) ||
        typeof descriptor.sha256 !== "string" ||
        !/^[a-f0-9]{64}$/.test(descriptor.sha256) ||
        typeof descriptor.bytes !== "number" ||
        !Number.isSafeInteger(descriptor.bytes) ||
        descriptor.bytes < 0
      ) {
        throw new Error(`Manifest descriptor for ${path} is invalid.`);
      }
      assertAllowedBundlePath(path);
      const entry = archive.entries.get(path);
      if (
        !entry ||
        entry.uncompressedBytes !== descriptor.bytes ||
        checksums.get(path) !== descriptor.sha256
      ) {
        throw new Error(`Manifest checksum validation failed for ${path}.`);
      }
    }
    for (const path of names) {
      operation?.signal.throwIfAborted();
      if (path === "manifest.json" || path === "checksums.sha256") continue;
      if (!Object.hasOwn(manifestFiles, path)) {
        throw new Error(`Manifest is missing a descriptor for ${path}.`);
      }
    }
    const packedOverrides = {
      current: new Map<string, string>(),
      origin: new Map<string, string>()
    };
    const tensorPaths: Record<string, string> = {
      "tensors/core.safetensors": entryPath("tensors/core.safetensors"),
      "tensors/plastic.safetensors": entryPath("tensors/plastic.safetensors"),
      "origin/tensors/core.safetensors": entryPath("origin/tensors/core.safetensors"),
      "origin/tensors/plastic.safetensors": entryPath("origin/tensors/plastic.safetensors")
    };
    let resolvedReferences: OmniManifest["references"];
    if (manifestValue.mode === "referenced-local") {
      if (!isRecord(manifestValue.references)) {
        throw new Error("The local referenced bundle has no tensor references.");
      }
      const mappings = [
        ["currentCore", "tensors/core.safetensors"],
        ["currentPlasticity", "tensors/plastic.safetensors"],
        ["originCore", "origin/tensors/core.safetensors"],
        ["originPlasticity", "origin/tensors/plastic.safetensors"]
      ] as const;
      for (const [key, path] of mappings) {
        operation?.signal.throwIfAborted();
        const hash = manifestValue.references[key];
        if (typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash)) {
          throw new Error("The local referenced bundle contains an invalid tensor reference.");
        }
        const blobPath = join(this.root, ".blobs", hash);
        try {
          if ((await streamFileSha256(blobPath)) !== hash) throw new Error("checksum");
          tensorPaths[path] = blobPath;
        } catch {
          throw new Error(
            `Local tensor reference ${hash.slice(0, 12)}… is unavailable on this installation.`
          );
        }
      }
      resolvedReferences = {
        currentCore: manifestValue.references.currentCore as string,
        currentPlasticity: manifestValue.references.currentPlasticity as string,
        originCore: manifestValue.references.originCore as string,
        originPlasticity: manifestValue.references.originPlasticity as string
      };
    } else if (manifestValue.references !== undefined) {
      throw new Error("Portable bundles may not contain local tensor references.");
    }
    if (packedDeclaration?.references) {
      for (const scope of ["current", "origin"] as const) {
        const referencesForScope: Record<string, string> = packedDeclaration.references[scope];
        for (const [name, hash] of Object.entries(referencesForScope) as Array<[string, string]>) {
          const path = `packed/${scope}/${name}`;
          if (!archive.entries.has(path)) {
            throw new Error(`The local referenced bundle is missing ${path}.`);
          }
          try {
            const blobPath = join(this.root, ".blobs", hash);
            if ((await streamFileSha256(blobPath)) !== hash) throw new Error("checksum");
            packedOverrides[scope].set(name, blobPath);
          } catch {
            throw new Error(
              `Local packed ternary reference ${hash.slice(0, 12)}… is unavailable on this installation.`
            );
          }
        }
      }
    }
    let verifiedPackedCurrent: StreamingPackedTernaryDirectory | undefined;
    let verifiedPackedOrigin: StreamingPackedTernaryDirectory | undefined;
    const bundledCurrentNames = [...names].filter((name) => name.startsWith("packed/current/"));
    const bundledOriginNames = [...names].filter((name) => name.startsWith("packed/origin/"));
    if (packedDeclaration) {
      verifiedPackedCurrent = await inspectPackedTernaryDirectory(
        join(archive.root, "packed", "current"),
        "Current bundle",
        packedOverrides.current
      );
      verifiedPackedOrigin = await inspectPackedTernaryDirectory(
        join(archive.root, "packed", "origin"),
        "Origin bundle",
        packedOverrides.origin
      );
      if (
        verifiedPackedCurrent.manifestSha256 !== packedDeclaration.currentManifestSha256 ||
        verifiedPackedOrigin.manifestSha256 !== packedDeclaration.originManifestSha256 ||
        verifiedPackedCurrent.tensorCount !== packedDeclaration.currentTensorCount ||
        verifiedPackedOrigin.tensorCount !== packedDeclaration.originTensorCount
      ) {
        throw new Error("Packed ternary bundle metadata does not match its manifest.");
      }
    } else if (bundledCurrentNames.length > 0 || bundledOriginNames.length > 0) {
      throw new Error("The .omni bundle contains undeclared packed ternary data.");
    }
    await Promise.all([
      assertSafeTensorsFile(tensorPaths["tensors/core.safetensors"]!, "core.safetensors"),
      assertSafeTensorsFile(tensorPaths["tensors/plastic.safetensors"]!, "plastic.safetensors"),
      assertSafeTensorsFile(
        tensorPaths["origin/tensors/core.safetensors"]!,
        "origin core.safetensors"
      ),
      assertSafeTensorsFile(
        tensorPaths["origin/tensors/plastic.safetensors"]!,
        "origin plastic.safetensors"
      )
    ]);
    for (const path of names) {
      operation?.signal.throwIfAborted();
      if (!path.startsWith("blobs/")) continue;
      const hash = path.slice("blobs/".length);
      if (!/^[a-f0-9]{64}$/.test(hash) || (await streamFileSha256(entryPath(path))) !== hash) {
        throw new Error(`Content-addressed blob validation failed for ${path}.`);
      }
    }
    let brainValue: unknown;
    let originBrainValue: unknown;
    let engineValue: unknown;
    let originEngineValue: unknown;
    try {
      brainValue = JSON.parse(await readFile(entryPath("state/brain.json"), "utf8"));
      originBrainValue = JSON.parse(await readFile(entryPath("origin/state/brain.json"), "utf8"));
      engineValue = JSON.parse(await readFile(entryPath("state/engine.json"), "utf8"));
      originEngineValue = JSON.parse(await readFile(entryPath("origin/state/engine.json"), "utf8"));
    } catch {
      throw new Error("A required brain or engine state document is invalid.");
    }
    // Check both mutable and immutable metadata before any neural state is
    // installed. An imported foundation is never a native OmniCortex origin.
    assertNativeOmniEngineState(engineValue, "Current .omni state");
    assertNativeOmniEngineState(originEngineValue, "Origin .omni state");
    if (names.has("origin/provenance.json")) {
      throw new Error("Legacy bundled foundation provenance is not supported.");
    }

    const imported = normalizeBrain(brainValue);
    const importedOrigin = normalizeBrain(originBrainValue);
    const hasActivityLedger = names.has("activity/ledger.sqlite3");
    let bundledActivitySummary: BrainActivityLedgerSummary | undefined;
    if (hasActivityLedger) {
      const activity = await BrainActivityLedger.open(archive.root, imported.id);
      try {
        bundledActivitySummary = activity.integrity();
      } finally {
        activity.close();
      }
      if (
        imported.activity &&
        canonicalJson(imported.activity) !== canonicalJson(bundledActivitySummary)
      ) {
        throw new Error("The bundled activity summary does not match its ledger.");
      }
      if (
        isRecord(manifestValue.licenseLedger) &&
        manifestValue.licenseLedger.sourceCount !== undefined &&
        manifestValue.licenseLedger.sourceCount !== bundledActivitySummary.trainingSourceCount
      ) {
        throw new Error("The bundled license ledger count does not match activity metadata.");
      }
    }
    const importedJournalEntries = names.has("activity/journal.json")
      ? parseJournalExport(
          JSON.parse(await readFile(entryPath("activity/journal.json"), "utf8")),
          imported.id
        )
      : [...(imported.journal ?? [])];
    const importedTrainingSources = names.has("activity/training-sources.json")
      ? parseTrainingSourceExport(
          JSON.parse(
            await readFile(entryPath("activity/training-sources.json"), "utf8")
          ),
          imported.id
        )
      : [...imported.trainingSources];
    const hasConversationLedger = names.has("conversation/ledger.sqlite3");
    if (savedInstance && !hasConversationLedger) {
      throw new Error("The saved-instance bundle is missing its host conversation ledger.");
    }
    if (hasConversationLedger) {
      const conversation = await ConversationLedger.open(archive.root, imported.id);
      try {
        const summary = conversation.integrity();
        if (savedInstance && canonicalJson(imported.conversation) !== canonicalJson(summary)) {
          throw new Error("The bundled host conversation head does not match its ledger.");
        }
      } finally {
        conversation.close();
      }
    }
    const hasNeuralConversationLedger = names.has(
      "conversation/neural-ledger.sqlite3"
    );
    assertImportedNeuralConversationHead(
      engineValue,
      "Current .omni state",
      hasNeuralConversationLedger
    );
    const hasOriginNeuralConversationLedger = names.has("origin/conversation/neural-ledger.sqlite3");
    assertImportedNeuralConversationHead(originEngineValue, "Origin .omni state", hasOriginNeuralConversationLedger);
    if (savedInstance) {
      for (const [scope, state] of [["current", engineValue], ["origin", originEngineValue]] as const) {
        const working = `working/${scope}/working-memory.sqlite3`;
        const paths = [...names].filter((name) => name.startsWith(`working/${scope}/`));
        if (paths.some((name) => name !== working)) throw new Error("The bundle contains unlisted working-memory files.");
        validateSavedWorkingPages(names.has(working) ? entryPath(working) : undefined, state);
      }
    } else {
      assertPortableWorkingMemoryCheckpoint(engineValue, "Current .omni state");
      assertPortableWorkingMemoryCheckpoint(originEngineValue, "Origin .omni state");
    }
    if (hasNeuralConversationLedger) {
      validateNeuralConversationLedger(
        entryPath("conversation/neural-ledger.sqlite3"),
        imported.id,
        engineValue
      );
    }
    if (hasOriginNeuralConversationLedger) {
      validateNeuralConversationLedger(
        entryPath("origin/conversation/neural-ledger.sqlite3"), importedOrigin.id, originEngineValue
      );
    }
    for (const [state, hasLedger] of [
      [engineValue, hasNeuralConversationLedger], [originEngineValue, hasOriginNeuralConversationLedger]
    ] as const) {
      if (savedInstance && !hasLedger && isRecord(state) &&
        Array.isArray(state.pending_chat_slow_learning) && state.pending_chat_slow_learning.length) {
        throw new Error("Pending chat replay is missing its saved neural conversation ledger.");
      }
    }
    let importedArtifactIndex: PersistedArtifactIndex | undefined;
    let originArtifactIndex: PersistedArtifactIndex | undefined;
    for (const [prefix, brainId] of [["artifacts", imported.id], ["origin/artifacts", importedOrigin.id]] as const) {
    if (names.has(`${prefix}/index.json`)) {
      const artifactIndex = parseArtifactIndex(
        JSON.parse(await readFile(entryPath(`${prefix}/index.json`), "utf8")), brainId
      );
      if (prefix === "artifacts") importedArtifactIndex = artifactIndex;
      else originArtifactIndex = artifactIndex;
      if (savedInstance && !names.has(`${prefix}/index.sqlite3`)) {
        throw new Error("Saved generated artifacts are missing their exact ledger.");
      }
      if (names.has(`${prefix}/index.sqlite3`)) {
        const ledger = await ArtifactIndexStore.open(join(archive.root, prefix), brainId);
        try {
          if (canonicalJson(ledger.snapshot()) !== canonicalJson(artifactIndex)) {
            throw new Error("Saved artifact ledger does not match its bundled index.");
          }
        } finally { ledger.close(); }
      }
      const declared = new Set(
        artifactIndex.artifacts.map(
          (artifact) => `${prefix}/files/${basename(artifact.relativePath)}`
        )
      );
      const bundled = [...names].filter((name) => name.startsWith(`${prefix}/files/`));
      if (
        bundled.length !== declared.size ||
        bundled.some((name) => !declared.has(name))
      ) {
        throw new Error("Generated artifact bundle files do not match their index.");
      }
      for (const artifact of artifactIndex.artifacts) {
        const path = `${prefix}/files/${basename(artifact.relativePath)}`;
        const entry = archive.entries.get(path);
        if (
          !entry ||
          entry.uncompressedBytes !== artifact.bytes ||
          await streamFileSha256(entry.path) !== artifact.sha256
        ) {
          throw new Error(`Bundled generated artifact failed integrity: ${path}`);
        }
      }
    } else if ([...names].some((name) => name.startsWith(`${prefix}/files/`) || name === `${prefix}/index.sqlite3`)) {
      throw new Error("The bundle contains generated artifacts without an index.");
    }
    }
    const importedOriginChecksum = assertOriginChecksum(
      originBrainValue,
      "The bundled origin"
    );
    if (imported.originChecksum !== importedOriginChecksum) {
      throw new Error("The bundled brain does not reference its immutable origin checksum.");
    }
    const engineMaterialized = manifestValue.engineMaterialized === true;
    if (
      engineMaterialized &&
      (!isRecord(engineValue) ||
        engineValue.format !== "omni-cortex-engine" ||
        engineValue.schema_version !== 1 ||
        engineValue.release_format !== STABLE_RELEASE_FORMAT ||
        !isRecord(originEngineValue) ||
        originEngineValue.format !== "omni-cortex-engine" ||
        originEngineValue.schema_version !== 1 ||
        originEngineValue.release_format !== STABLE_RELEASE_FORMAT)
    ) {
      throw new Error("Materialized engine state is invalid or belongs to the beta format.");
    }
    if (engineMaterialized) {
      assertUiNeuralOriginIdentity(importedOrigin, originEngineValue);
    }
    const [
      currentSubstratePaths,
      originSubstratePaths,
      currentMutableStatePaths,
      originMutableStatePaths
    ] = await Promise.all([
      validateExtractedSubstrateSnapshot(archive, "substrate/current", engineValue),
      validateExtractedSubstrateSnapshot(archive, "substrate/origin", originEngineValue),
      validateExtractedMutableStateSnapshot(archive, "mutable/current", engineValue),
      validateExtractedMutableStateSnapshot(archive, "mutable/origin", originEngineValue)
    ]);
    const jointPaths = { current: new Set<string>(), origin: new Set<string>() };
    const operationalIntentPaths = { current: [] as string[], origin: [] as string[] };
    for (const name of names) {
      if (!name.startsWith("concept-views/")) continue;
      const match = /^concept-views\/(current|origin)\/([a-f0-9]{64}\.jsonl)$/.exec(name);
      if (!match) throw new Error("The bundle contains an unsafe structural concept view path.");
      await validateConceptIdViewFile(entryPath(name), match[2]!);
    }
    for (const name of names) {
      if (!name.startsWith("operational/")) continue;
      const match = /^operational\/(current|origin)\/([a-f0-9-]{36}\.json)$/i.exec(name);
      if (!match) throw new Error("The bundle contains an unsafe operational intent path.");
      await validateSavedToolIntent(entryPath(name));
      operationalIntentPaths[match[1] as "current" | "origin"].push(name);
    }
    for (const [scope, state] of [["current", engineValue], ["origin", originEngineValue]] as const) {
      const files = await savedJointGenerationFiles(state, (kind, relative) =>
        entryPath(`${kind}/${scope}/${relative}`)
      );
      jointPaths[scope] = new Set(files.keys());
      const matching = [...names].filter((name) => name.startsWith(`joint/${scope}/`));
      if (matching.some((name) => !jointPaths[scope].has(name.slice(`joint/${scope}/`.length)))) {
        throw new Error("The bundle contains an unlisted ingestion joint-generation file.");
      }
    }
    const recoveryPoints: NonNullable<OmniManifest["recoveryPoints"]> = [];
    if (manifestValue.recoveryPoints !== undefined) {
      if (!Array.isArray(manifestValue.recoveryPoints)) throw new Error("Saved recovery-point declaration is invalid.");
      const ids = new Set<string>();
      for (const value of manifestValue.recoveryPoints) {
        if (!isRecord(value) || typeof value.id !== "string" || !SAFE_ID.test(value.id) ||
          ids.has(value.id) || typeof value.brainId !== "string" || !SAFE_ID.test(value.brainId) ||
          !Array.isArray(value.missingPayloads) || value.missingPayloads.some((path) => typeof path !== "string")) {
          throw new Error("Saved recovery-point declaration is invalid.");
        }
        ids.add(value.id);
        const checked = await validateSavedSnapshot(join(archive.root, "snapshots", value.id), value.id, value.brainId);
        if (canonicalJson(checked.missingPayloads) !== canonicalJson(value.missingPayloads)) {
          throw new Error("Saved recovery-point missing-payload report is invalid.");
        }
        recoveryPoints.push({ id: value.id, brainId: value.brainId, missingPayloads: checked.missingPayloads });
      }
    }
    for (const path of names) {
      if (path.startsWith("snapshots/") && !recoveryPoints.some((snapshot) =>
        path === `snapshots/${snapshot.id}.json` || path === `snapshots/${snapshot.id}.meta.json` ||
        path.startsWith(`snapshots/${snapshot.id}/`))) {
        throw new Error("The bundle contains an undeclared saved recovery-point file.");
      }
    }
    const bundledBrainId = imported.id;
    const bundledGeneration = imported.lineage.generation;
    let rekeyed = false;
    let directory = this.brainDirectory(imported.id);
    for (;;) {
      try {
        // Reserving the final directory is the cross-process compare-and-set.
        // A prior pathExists check races when two imports carry the same id.
        await mkdir(directory, { recursive: false });
        break;
      } catch (error) {
        if (filesystemErrorCode(error) !== "EEXIST") throw error;
        imported.id = randomUUID();
        directory = this.brainDirectory(imported.id);
        if (!rekeyed) {
          imported.lineage = {
            parentId: bundledBrainId,
            rootId: imported.lineage.rootId,
            generation: bundledGeneration + 1
          };
          rekeyed = true;
        }
      }
    }
    if (isRecord(engineValue)) {
      engineValue.brain_id = imported.id;
      engineValue.name = imported.name;
    }
    // The immutable UI and neural origins describe the same ancestral build.
    // Import may re-key the live identity on collision, but must not rewrite
    // the origin to that new identity or it would diverge from engine/origin.
    imported.originChecksum = importedOriginChecksum;
    if (importedArtifactIndex) {
      importedArtifactIndex = parseArtifactIndex(
        JSON.parse(
          serializeArtifactIndex(
            importedArtifactIndex.brainId,
            importedArtifactIndex.artifacts
          )
        ),
        importedArtifactIndex.brainId,
        imported.id
      );
    }
    imported.name = imported.name.slice(0, 120);
    imported.config.name = imported.name;
    // Imported/recovered identities start dormant so importing a second copy
    // cannot silently create another prompt-free runtime owner.
    imported.config.idleCognition = false;
    imported.createdAt = new Date().toISOString();
    imported.updatedAt = imported.createdAt;
    if (options.initializing) {
      imported.readiness = {
        state: "initializing",
        startedAt: imported.createdAt
      };
    }
    importedJournalEntries.push({
      id: randomUUID(),
      createdAt: imported.createdAt,
      kind: "system",
      summary: `Imported from ${basename(sourceLabel)}.`
    });
    imported.journal = [];
    imported.trainingSources = [];
    let installedFiles = 0;
    let installedBytes = 0;
    let installedPhysicalBytes = 0;
    let installedSharedBytes = 0;
    const installProgress = async (label: string): Promise<void> => {
      await operation?.checkpoint({
        phase: "installing",
        label,
        targetBrainId: imported.id,
        filesCompleted: installedFiles,
        filesTotal: Math.max(installedFiles, names.size),
        logicalBytesCompleted: installedBytes,
        logicalBytesTotal: Math.max(
          installedBytes,
          [...archive.entries.values()].reduce(
            (sum, entry) => sum + entry.uncompressedBytes,
            0
          )
        ),
        physicalBytesAdded: installedPhysicalBytes,
        sharedBytes: installedSharedBytes
      });
    };
    const materializeFile = async (
      source: string,
      destination: string,
      shareable = true
    ): Promise<void> => {
      await installProgress(`Installing ${basename(destination)}`);
      operation?.signal.throwIfAborted();
      const sourceInfo = await lstat(source);
      if (!sourceInfo.isFile() || sourceInfo.isSymbolicLink()) {
        throw new Error("Import materialization source is not a safe regular file.");
      }
      await operation?.checkDisk(this.root, sourceInfo.size);
      if (!shareable) {
        await copyMutableFileIsolated(source, destination);
        installedFiles += 1;
        installedBytes += sourceInfo.size;
        installedPhysicalBytes += sourceInfo.size;
        await installProgress(`Installed private ${basename(destination)}`);
        return;
      }
      const sourceHash = await streamFileSha256(source);
      const existed = await verifiedExistingBlob(
        join(this.root, ".blobs", sourceHash),
        sourceHash
      );
      const hash = await this.storeFileAsBlob(source);
      await this.linkBlobTo(hash, destination, operation);
      const target = await stat(destination);
      installedFiles += 1;
      installedBytes += sourceInfo.size;
      if (!existed) installedPhysicalBytes += sourceInfo.size;
      if (target.nlink > 1) installedSharedBytes += sourceInfo.size;
      else installedPhysicalBytes += sourceInfo.size;
      await installProgress(`Installed ${basename(destination)}`);
    };
    const materializeReference = async (
      hash: string,
      destination: string,
      shareable = true
    ): Promise<void> => {
      await installProgress(`Linking ${basename(destination)}`);
      operation?.signal.throwIfAborted();
      const source = join(this.root, ".blobs", hash);
      const info = await stat(source);
      if (shareable) {
        await this.linkBlobTo(hash, destination, operation);
      } else {
        await operation?.checkDisk(this.root, info.size);
        await copyMutableFileIsolated(source, destination, {
          expectedSha256: hash
        });
      }
      const target = await stat(destination);
      installedFiles += 1;
      installedBytes += info.size;
      if (shareable && target.nlink > 1) installedSharedBytes += info.size;
      else installedPhysicalBytes += info.size;
      await installProgress(`Linked ${basename(destination)}`);
    };
    const installSubstrate = async (
      prefix: string,
      paths: Set<string>,
      destination: string
    ): Promise<void> => {
      for (const relative of [...paths].sort()) {
        await materializeFile(
          entryPath(`${prefix}/${relative}`),
          join(destination, ...relative.split("/")),
          // Active pointers change as learning commits later generations.
          // Content-addressed generation manifests and blobs remain shareable.
          !(prefix.endsWith("/current") && relative === "manifest.json") &&
            !(prefix.startsWith("mutable/") && relative === "replay.sqlite3")
        );
      }
    };
    try {
      await installProgress("Creating private import staging identity");
      operation?.signal.throwIfAborted();
      await awaitAllOrThrow([
        mkdir(join(directory, "snapshots"), { recursive: true }),
        mkdir(join(directory, "engine"), { recursive: true })
      ]);
      if (hasConversationLedger) {
        await ConversationLedger.clone(
          archive.root,
          directory,
          bundledBrainId,
          imported.id
        );
      }
      if (hasNeuralConversationLedger) {
        const source = entryPath("conversation/neural-ledger.sqlite3");
        const destination = join(directory, "engine", "conversation.sqlite3");
        const info = await lstat(source);
        await operation?.checkDisk(this.root, info.size);
        await copyMutableFileIsolated(source, destination);
        rekeyNeuralConversationLedger(destination, imported.id);
        validateNeuralConversationLedger(destination, imported.id, engineValue);
      }
      if (hasOriginNeuralConversationLedger) {
        await materializeFile(
          entryPath("origin/conversation/neural-ledger.sqlite3"),
          join(directory, "engine", "origin", "conversation.sqlite3"), false
        );
      }
      for (const scope of ["current", "origin"] as const) {
        for (const path of operationalIntentPaths[scope]) {
          await materializeFile(entryPath(path), join(directory, "engine",
            ...(scope === "origin" ? ["origin"] : []), "operational-tool-intents", basename(path)), false);
        }
      }
      if (hasActivityLedger) {
        await BrainActivityLedger.clone(
          archive.root,
          directory,
          bundledBrainId,
          imported.id
        );
        const importedActivity = await BrainActivityLedger.open(directory, imported.id);
        try {
          importedActivity.appendJournals(importedJournalEntries);
        } finally {
          importedActivity.close();
        }
      } else {
        await BrainActivityLedger.replace(
          directory,
          imported.id,
          importedJournalEntries,
          importedTrainingSources
        );
      }
      const importedActivity = await BrainActivityLedger.open(directory, imported.id);
      try {
        imported.activity = importedActivity.getSummary();
      } finally {
        importedActivity.close();
      }
      const importedDocument = JSON.stringify(imported, null, 2);
      await operation?.checkDisk(this.root, Buffer.byteLength(importedDocument));
      for (const [prefix, artifactIndex, artifactBrainId, artifactEngine] of [
        ["artifacts", importedArtifactIndex, imported.id, join(directory, "engine")],
        ["origin/artifacts", originArtifactIndex, importedOrigin.id, join(directory, "engine", "origin")]
      ] as const) {
      if (artifactIndex) {
        const artifactDirectory = join(artifactEngine, "artifacts");
        await mkdir(artifactDirectory, { recursive: true });
        const materialized = new Set<string>();
        for (const artifact of artifactIndex.artifacts) {
          const name = basename(artifact.relativePath);
          if (materialized.has(name)) continue;
          materialized.add(name);
          await materializeFile(
            entryPath(`${prefix}/files/${name}`),
            join(artifactDirectory, name)
          );
        }
        if (names.has(`${prefix}/index.sqlite3`)) {
          const ledgerPath = ArtifactIndexStore.databasePath(artifactDirectory);
          await materializeFile(entryPath(`${prefix}/index.sqlite3`), ledgerPath, false);
          const database = new DatabaseSync(ledgerPath);
          try { database.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(artifactBrainId); }
          finally { database.close(); }
          const ledger = await ArtifactIndexStore.open(artifactDirectory, artifactBrainId);
          try { ledger.integrity(); } finally { ledger.close(); }
        } else {
          await ArtifactIndexStore.replace(artifactDirectory, artifactBrainId, artifactIndex.artifacts);
        }
      }
      }
      if (engineMaterialized) {
        if (resolvedReferences) {
          await materializeReference(
            resolvedReferences.currentCore,
            join(directory, "engine", "core.safetensors"),
            false
          );
          await materializeReference(
            resolvedReferences.currentPlasticity,
            join(directory, "engine", "plasticity.safetensors"),
            false
          );
        } else {
          await materializeFile(
            tensorPaths["tensors/core.safetensors"]!,
            join(directory, "engine", "core.safetensors"),
            false
          );
          await materializeFile(
            tensorPaths["tensors/plastic.safetensors"]!,
            join(directory, "engine", "plasticity.safetensors"),
            false
          );
        }
      }
      if (engineMaterialized && verifiedPackedCurrent) {
        const packedDirectory = join(directory, "engine", "packed-ternary");
        await mkdir(packedDirectory, { recursive: true });
        for (const [name, sourcePath] of verifiedPackedCurrent.files) {
          const referenceHash = packedDeclaration?.references?.current[name];
          if (referenceHash) {
            await materializeReference(referenceHash, join(packedDirectory, name));
          } else {
            await materializeFile(sourcePath, join(packedDirectory, name));
          }
        }
      }
      if (currentSubstratePaths.size > 0) {
        await installSubstrate(
          "substrate/current",
          currentSubstratePaths,
          join(directory, "engine", "substrate")
        );
      }
      if (currentMutableStatePaths.size > 0) {
        await installSubstrate(
          "mutable/current",
          currentMutableStatePaths,
          join(directory, "engine", "state")
        );
      }
      for (const scope of ["current", "origin"] as const) {
        const engine = scope === "current" ? join(directory, "engine") : join(directory, "engine", "origin");
        for (const name of names) {
          if (!name.startsWith(`concept-views/${scope}/`)) continue;
          const file = name.slice(`concept-views/${scope}/`.length);
          await validateConceptIdViewFile(entryPath(name), file);
          await materializeFile(entryPath(name), join(engine, "state", "concept-id-views", file), false);
        }
        const working = `working/${scope}/working-memory.sqlite3`;
        if (names.has(working)) {
          await materializeFile(entryPath(working), join(engine, "state", "working-memory.sqlite3"), false);
        }
        if (jointPaths[scope].size) {
          await installSubstrate(`joint/${scope}`, jointPaths[scope], join(engine, "state", "ingestion-joint"));
        }
      }
      if (engineMaterialized) {
        operation?.signal.throwIfAborted();
        const engineDocument = JSON.stringify(engineValue, null, 2);
        await operation?.checkDisk(this.root, Buffer.byteLength(engineDocument));
        await atomicWrite(
          join(directory, "engine", "brain.json"),
          engineDocument
        );
      }
      if (isRecord(originEngineValue) && originEngineValue.format === "omni-cortex-engine") {
        if (resolvedReferences) {
          await materializeReference(
            resolvedReferences.originCore,
            join(directory, "engine", "origin", "core.safetensors")
          );
          await materializeReference(
            resolvedReferences.originPlasticity,
            join(directory, "engine", "origin", "plasticity.safetensors")
          );
        } else {
          await materializeFile(
            tensorPaths["origin/tensors/core.safetensors"]!,
            join(directory, "engine", "origin", "core.safetensors")
          );
          await materializeFile(
            tensorPaths["origin/tensors/plastic.safetensors"]!,
            join(directory, "engine", "origin", "plasticity.safetensors")
          );
        }
      }
      if (
        isRecord(originEngineValue) &&
        originEngineValue.format === "omni-cortex-engine" &&
        verifiedPackedOrigin
      ) {
        const packedDirectory = join(directory, "engine", "origin", "packed-ternary");
        await mkdir(packedDirectory, { recursive: true });
        for (const [name, sourcePath] of verifiedPackedOrigin.files) {
          const referenceHash = packedDeclaration?.references?.origin[name];
          if (referenceHash) {
            await materializeReference(referenceHash, join(packedDirectory, name));
          } else {
            await materializeFile(sourcePath, join(packedDirectory, name));
          }
        }
      }
      if (originSubstratePaths.size > 0) {
        await installSubstrate(
          "substrate/origin",
          originSubstratePaths,
          join(directory, "engine", "origin", "substrate")
        );
      }
      if (originMutableStatePaths.size > 0) {
        await installSubstrate(
          "mutable/origin",
          originMutableStatePaths,
          join(directory, "engine", "origin", "state")
        );
      }
      if (isRecord(originEngineValue) && originEngineValue.format === "omni-cortex-engine") {
        await materializeFile(
          entryPath("origin/state/engine.json"),
          join(directory, "engine", "origin", "brain.json")
        );
      }
      for (const [scope, state, destinationEngine] of [
        ["current", engineValue, join(directory, "engine")],
        ["origin", originEngineValue, join(directory, "engine", "origin")]
      ] as const) {
        const prefix = `geometry/${scope}/`;
        const declared = [...names].filter((name) => name.startsWith(prefix));
        for (const name of declared) {
          const relative = name.slice(prefix.length);
          if (relative !== "evaluation/geometry-holdouts.json" &&
            !/^evaluation\/data\/[a-f0-9]{32}(?:\.[A-Za-z0-9_-]+)?$/.test(relative)) {
            throw new Error("The .omni geometry holdout path is invalid.");
          }
          await materializeFile(entryPath(name), join(destinationEngine, ...relative.split("/")), false);
        }
        const verified = await savedGeometryHoldoutFiles(state, destinationEngine);
        if (verified.size !== declared.length ||
          [...verified.keys()].some((relative) => !names.has(`${prefix}${relative}`))) {
          throw new Error("The .omni geometry holdout data does not match its sealed registration.");
        }
      }
      for (const path of names) {
        if (!path.startsWith("blobs/")) continue;
        const expected = path.slice("blobs/".length);
        const stored = await this.storeFileAsBlob(entryPath(path));
        if (stored !== expected) {
          throw new Error(`Content-addressed blob validation failed for ${path}.`);
        }
      }
      for (const snapshot of recoveryPoints) {
        for (const path of [...names].filter((name) =>
          name === `snapshots/${snapshot.id}.json` || name === `snapshots/${snapshot.id}.meta.json` ||
          name.startsWith(`snapshots/${snapshot.id}/`)).sort()) {
          await materializeFile(entryPath(path), join(directory, ...path.split("/")),
            !/\.(?:json|sqlite3)$/.test(path));
        }
        await rekeySavedSnapshot(join(directory, "snapshots", snapshot.id), imported.id, snapshot.missingPayloads);
      }
      await materializeFile(entryPath("origin/state/brain.json"), join(directory, "origin.json"));
      await operation?.checkpoint({
        phase: "promoting",
        label: "Imported identity verified and ready",
        targetBrainId: imported.id,
        filesCompleted: installedFiles,
        filesTotal: installedFiles,
        logicalBytesCompleted: installedBytes,
        logicalBytesTotal: installedBytes,
        physicalBytesAdded: installedPhysicalBytes,
        sharedBytes: installedSharedBytes
      });
      operation?.signal.throwIfAborted();
      // The host document is the final identity commit record. Reserved
      // directories without it are not visible as half-installed brains.
      await atomicWrite(this.documentPath(imported.id), importedDocument);
      const completed = clone(await this.get(imported.id));
      await this.observeCommittedBrain(completed);
      return completed;
    } catch (error) {
      await removeTreeWithRetry(directory);
      throw error;
    }
  }
}
