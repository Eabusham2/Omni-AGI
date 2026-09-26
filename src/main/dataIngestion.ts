import { createHash, randomUUID } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import {
  access,
  lstat,
  mkdir,
  readFile,
  readdir,
  realpath,
  rename,
  rm,
  stat,
  statfs,
  writeFile,
} from "node:fs/promises";
import { createInterface } from "node:readline";
import {
  basename,
  dirname,
  extname,
  isAbsolute,
  join,
  relative,
  resolve,
  sep,
} from "node:path";
import { once } from "node:events";
import { DatabaseSync } from "node:sqlite";
import type {
  DatasetCursor,
  DatasetEntryReceipt,
  DatasetFormat,
  DatasetManifest,
  DatasetManifestEntry,
  DatasetProgressGeneration,
  DatasetResumeCandidate,
  PersistedDatasetCoverageSummary,
  TrainingCoverage,
  TrainingCoverageError,
} from "../shared/types";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const STREAM_TEXT_CHARS = 256 * 1024;
const COVERAGE_ERROR_SAMPLES = 256;
const PROGRESS_FORMAT = "omni-dataset-progress" as const;
const PROGRESS_POINTER_FORMAT = "omni-dataset-progress-pointer" as const;
const PROGRESS_SCHEMA_VERSION = 1;
const MAX_RECEIPT_WARNINGS = 64;
const MAX_RECEIPT_TEXT = 4_096;
const SQLITE_SNAPSHOT_RESERVE_BYTES = 1024 * 1024 * 1024;
const INTERNAL_DATASET_DIRECTORIES = new Set([
  ".git",
  ".hg",
  ".svn",
  ".mypy_cache",
  ".pytest_cache",
  ".ruff_cache",
  ".tox",
  ".nox",
  ".venv",
  "__pycache__",
  "node_modules",
]);
const AUXILIARY_DATASET_FILES = new Set([
  ".ds_store",
  ".gitattributes",
  ".gitignore",
  ".gitmodules",
  "desktop.ini",
  "thumbs.db",
]);
const AUXILIARY_DATASET_SUFFIXES = [
  ".lock",
  ".metadata",
  ".incomplete",
  ".inprogress",
  ".pending",
  ".partial",
  ".part",
  ".crdownload",
  ".download",
  ".tmp",
  ".temp",
  ".swp",
  ".swo",
] as const;

interface DatasetProgressPointer {
  schemaVersion: 1;
  format: typeof PROGRESS_POINTER_FORMAT;
  manifestId: string;
  manifestHash: string;
  generation: number;
  activeGeneration: string;
  generationFile: string;
}

export interface DatasetManifestStoreOptions {
  /** Fault-injection hook used by restart-safety tests. */
  compatibilityMaterializationHook?: (
    phase: "after-authoritative" | "after-cursor" | "after-coverage",
  ) => void | Promise<void>;
}

function canonicalJson(value: unknown): string {
  const normalize = (entry: unknown): unknown => {
    if (Array.isArray(entry)) return entry.map(normalize);
    if (entry && typeof entry === "object") {
      return Object.fromEntries(
        Object.entries(entry as Record<string, unknown>)
          .sort(([left], [right]) => comparePathNames(left, right))
          .map(([key, child]) => [key, normalize(child)]),
      );
    }
    return entry;
  };
  return JSON.stringify(normalize(value));
}

function contentSha256(value: unknown): string {
  return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

function datasetEntryTransactionKey(receipt: DatasetEntryReceipt): string {
  return createHash("sha256")
    .update(
      [
        "omni-dataset-entry-v1",
        receipt.manifestId,
        receipt.entryIndex,
        receipt.epoch,
        receipt.contentHash,
        receipt.policy,
      ].join("\0"),
    )
    .digest("hex");
}

function safeInteger(
  value: unknown,
  label: string,
  minimum = 0,
  maximum = Number.MAX_SAFE_INTEGER,
): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < minimum ||
    value > maximum
  ) {
    throw new Error(
      `${label} must be a safe integer between ${minimum} and ${maximum}.`,
    );
  }
  return value;
}

function safeProduct(left: number, right: number, label: string): number {
  const value = left * right;
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${label} exceeds the safe manifest counter range.`);
  }
  return value;
}

function validSha256(value: unknown): value is string {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}

function comparePathNames(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function auxiliaryDatasetFileRejection(path: string): string | undefined {
  const name = basename(path).toLocaleLowerCase();
  if (
    name.startsWith(".") ||
    AUXILIARY_DATASET_FILES.has(name) ||
    name.endsWith("~") ||
    AUXILIARY_DATASET_SUFFIXES.some((suffix) => name.endsWith(suffix))
  ) {
    return "Transient, operating-system, or repository metadata is not training data.";
  }
  return undefined;
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${randomUUID()}.next`;
  await writeFile(temporary, JSON.stringify(value, null, 2), {
    encoding: "utf8",
    flag: "wx",
    mode: 0o600,
  });
  await rename(temporary, path);
}

async function writeLine(
  stream: ReturnType<typeof createWriteStream>,
  value: string,
): Promise<void> {
  if (!stream.write(value)) await once(stream, "drain");
}

export function detectDatasetFormat(path: string): DatasetFormat {
  const lower = basename(path).toLocaleLowerCase();
  const extension = extname(lower);
  if (
    lower.endsWith(".tar.gz") ||
    lower.endsWith(".tar.bz2") ||
    lower.endsWith(".tar.xz")
  ) {
    return "archive";
  }
  if (
    lower.endsWith(".hf.json") ||
    lower.endsWith(".dataset.json") ||
    lower.endsWith(".manifest.json") ||
    lower === "dataset_info.json" ||
    lower === "dataset_infos.json"
  ) {
    return "huggingface";
  }
  if (extension === ".pdf") return "pdf";
  if (extension === ".epub") return "epub";
  if ([".docx", ".pptx", ".xlsx", ".odt", ".ods", ".odp"].includes(extension)) {
    return "office";
  }
  if (extension === ".csv") return "csv";
  if (extension === ".tsv") return "tsv";
  if (extension === ".json") return "json";
  if ([".jsonl", ".ndjson"].includes(extension)) return "jsonl";
  if (extension === ".parquet") return "parquet";
  if ([".arrow", ".feather", ".ipc"].includes(extension)) return "arrow";
  if ([".sqlite", ".sqlite3", ".db"].includes(extension)) return "sqlite";
  if ([".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz"].includes(extension)) {
    return extension === ".tar" || extension === ".tgz"
      ? "webdataset"
      : "archive";
  }
  if (
    [
      ".txt",
      ".md",
      ".mdx",
      ".rst",
      ".py",
      ".js",
      ".jsx",
      ".ts",
      ".tsx",
      ".rs",
      ".go",
      ".java",
      ".c",
      ".cc",
      ".cpp",
      ".h",
      ".hpp",
      ".cs",
      ".swift",
      ".kt",
      ".sql",
      ".html",
      ".htm",
      ".css",
      ".scss",
      ".yaml",
      ".yml",
      ".toml",
    ].includes(extension)
  ) {
    return "text";
  }
  if (
    [
      ".png",
      ".jpg",
      ".jpeg",
      ".webp",
      ".gif",
      ".bmp",
      ".tif",
      ".tiff",
      ".avif",
      ".heic",
      ".heif",
      ".jp2",
      ".j2k",
      ".jpf",
      ".jpx",
      ".jxl",
      ".raw",
      ".dng",
    ].includes(extension)
  ) {
    return "image";
  }
  if (
    [
      ".wav",
      ".mp3",
      ".flac",
      ".m4a",
      ".aac",
      ".ogg",
      ".oga",
      ".opus",
      ".aiff",
      ".aif",
      ".wma",
      ".caf",
      ".alac",
      ".amr",
      ".au",
      ".snd",
      ".mka",
    ].includes(extension)
  ) {
    return "audio";
  }
  if (
    [
      ".mp4",
      ".webm",
      ".mov",
      ".mkv",
      ".avi",
      ".m4v",
      ".mpeg",
      ".mpg",
      ".wmv",
      ".flv",
      ".3gp",
      ".3g2",
      ".ogv",
      ".mts",
      ".m2ts",
      ".vob",
    ].includes(extension)
  ) {
    return "video";
  }
  return "unknown";
}

async function* walkRegularFiles(roots: string[]): AsyncGenerator<{
  path: string;
  root: string;
  rootIsFile: boolean;
  bytes: number;
  lastModifiedMs: number;
  rejection?: string;
}> {
  const seen = new Set<string>();
  const queue = [...new Set(roots.map((value) => resolve(value)))]
    .sort(comparePathNames)
    .reverse()
    .map((path) => ({ path, root: path, rootIsFile: false }));
  while (queue.length > 0) {
    const next = queue.pop();
    if (!next) continue;
    let info;
    try {
      info = await lstat(next.path);
    } catch (error) {
      yield {
        ...next,
        bytes: 0,
        lastModifiedMs: 0,
        rejection: `Could not inspect dataset path: ${
          error instanceof Error ? error.message : String(error)
        }`,
      };
      continue;
    }
    const rootIsFile =
      next.path === next.root ? info.isFile() : next.rootIsFile;
    if (info.isSymbolicLink()) {
      yield {
        ...next,
        rootIsFile,
        bytes: 0,
        lastModifiedMs: info.mtimeMs,
        rejection: "Symbolic links are not followed.",
      };
      continue;
    }
    let canonical: string;
    try {
      canonical = await realpath(next.path);
    } catch (error) {
      yield {
        ...next,
        rootIsFile,
        bytes: 0,
        lastModifiedMs: info.mtimeMs,
        rejection: `Could not resolve dataset path: ${
          error instanceof Error ? error.message : String(error)
        }`,
      };
      continue;
    }
    const traversalRoot = next.path === next.root ? canonical : next.root;
    if (seen.has(canonical)) continue;
    seen.add(canonical);
    if (info.isFile()) {
      const auxiliaryRejection = auxiliaryDatasetFileRejection(canonical);
      yield {
        path: canonical,
        root: traversalRoot,
        rootIsFile,
        bytes: info.size,
        lastModifiedMs: info.mtimeMs,
        ...(auxiliaryRejection ? { rejection: auxiliaryRejection } : {}),
      };
      continue;
    }
    if (!info.isDirectory()) {
      yield {
        path: canonical,
        root: traversalRoot,
        rootIsFile,
        bytes: 0,
        lastModifiedMs: info.mtimeMs,
        rejection: "Dataset path is not a regular file or directory.",
      };
      continue;
    }
    if (
      next.path !== next.root &&
      (INTERNAL_DATASET_DIRECTORIES.has(
        basename(canonical).toLocaleLowerCase(),
      ) ||
        basename(canonical).startsWith("."))
    ) {
      continue;
    }
    let children: string[];
    try {
      children = await readdir(canonical);
    } catch (error) {
      yield {
        path: canonical,
        root: traversalRoot,
        rootIsFile,
        bytes: 0,
        lastModifiedMs: info.mtimeMs,
        rejection: `Could not read dataset directory: ${
          error instanceof Error ? error.message : String(error)
        }`,
      };
      continue;
    }
    for (const name of children.sort(comparePathNames).reverse()) {
      queue.push({
        path: join(canonical, name),
        root: traversalRoot,
        rootIsFile: false,
      });
    }
  }
}

export async function hashFile(path: string): Promise<string> {
  return hashFileWithProgress(path);
}

function previewCancelled(): Error {
  const error = new Error("Dataset preview cancelled.");
  error.name = "AbortError";
  return error;
}

function throwIfCancelled(signal?: AbortSignal): void {
  if (signal?.aborted) throw previewCancelled();
}

async function hashFileWithProgress(
  path: string,
  options: {
    signal?: AbortSignal;
    onBytes?: (bytes: number) => void;
  } = {},
): Promise<string> {
  throwIfCancelled(options.signal);
  const digest = createHash("sha256");
  const stream = createReadStream(path);
  const abort = (): void => {
    stream.destroy(previewCancelled());
  };
  options.signal?.addEventListener("abort", abort, { once: true });
  try {
    for await (const chunk of stream) {
      throwIfCancelled(options.signal);
      const bytes = chunk as Buffer;
      digest.update(bytes);
      options.onBytes?.(bytes.byteLength);
    }
  } catch (error) {
    if (options.signal?.aborted) throw previewCancelled();
    throw error;
  } finally {
    options.signal?.removeEventListener("abort", abort);
  }
  return digest.digest("hex");
}

async function snapshotSqliteDatabase(
  sourcePath: string,
  manifestDirectory: string,
  signal?: AbortSignal,
): Promise<{
  path: string;
  bytes: number;
  lastModifiedMs: number;
  contentSha256: string;
}> {
  throwIfCancelled(signal);
  const sourceInfo = await stat(sourcePath);
  const filesystem = await statfs(manifestDirectory);
  const availableBytes = filesystem.bavail * filesystem.bsize;
  const estimatedSnapshotBytes = Math.max(64 * 1024, sourceInfo.size * 2);
  if (
    !Number.isSafeInteger(availableBytes) ||
    availableBytes - SQLITE_SNAPSHOT_RESERVE_BYTES < estimatedSnapshotBytes
  ) {
    throw new Error(
      "Insufficient disk reserve for an immutable SQLite training snapshot.",
    );
  }
  const snapshots = join(manifestDirectory, "snapshots");
  await mkdir(snapshots, { recursive: true });
  const temporary = join(snapshots, `.sqlite-${randomUUID()}.next`);
  try {
    const database = new DatabaseSync(sourcePath, { readOnly: true });
    try {
      const escaped = temporary.replace(/'/g, "''");
      // VACUUM INTO reads one consistent SQLite transaction, including committed
      // WAL pages, while the source handle remains strictly read-only.
      database.exec(`VACUUM INTO '${escaped}'`);
    } finally {
      database.close();
    }
    throwIfCancelled(signal);
    const checksum = await hashFileWithProgress(temporary, { signal });
    const destination = join(snapshots, `${checksum}.sqlite`);
    if (await pathExists(destination)) {
      if ((await hashFile(destination)) !== checksum) {
        throw new Error(
          "Existing SQLite snapshot conflicts with its content hash.",
        );
      }
      await rm(temporary, { force: true });
    } else {
      await rename(temporary, destination);
    }
    const snapshotInfo = await stat(destination);
    return {
      path: destination,
      bytes: snapshotInfo.size,
      lastModifiedMs: snapshotInfo.mtimeMs,
      contentSha256: checksum,
    };
  } finally {
    await rm(temporary, { force: true });
  }
}

export async function assertManifestEntryStable(
  entry: DatasetManifestEntry,
): Promise<void> {
  if (entry.rejection) {
    throw new Error(`Dataset manifest entry rejected: ${entry.rejection}`);
  }
  const info = await stat(entry.path);
  if (!info.isFile()) throw new Error(`${entry.path} is not a regular file.`);
  if (info.size !== entry.bytes) {
    throw new Error(
      `Dataset source changed since the manifest was committed: ${entry.relativePath} size changed.`,
    );
  }
  if (
    entry.contentSha256 !== undefined &&
    !/^[a-f0-9]{64}$/.test(entry.contentSha256)
  ) {
    throw new Error(
      "Dataset manifest entry rejected: content checksum is invalid.",
    );
  }
  if (
    typeof entry.lastModifiedMs === "number" &&
    Number.isFinite(entry.lastModifiedMs) &&
    Math.abs(info.mtimeMs - entry.lastModifiedMs) > 0.001
  ) {
    throw new Error(
      `Dataset source changed since the manifest was committed: ${entry.relativePath} timestamp changed.`,
    );
  }
}

export async function* streamUtf8Text(path: string): AsyncGenerator<string> {
  const decoder = new TextDecoder("utf-8", { fatal: false });
  let pending = "";
  for await (const chunk of createReadStream(path)) {
    pending += decoder
      .decode(chunk as Buffer, { stream: true })
      .replace(/\0/g, "");
    while (pending.length >= STREAM_TEXT_CHARS) {
      const splitAt = Math.max(
        pending.lastIndexOf("\n", STREAM_TEXT_CHARS),
        pending.lastIndexOf(" ", STREAM_TEXT_CHARS),
      );
      const boundary = splitAt > 0 ? splitAt : STREAM_TEXT_CHARS;
      yield pending.slice(0, boundary);
      pending = pending.slice(boundary);
    }
  }
  pending += decoder.decode();
  if (pending) yield pending;
}

export interface DatasetManifestCreateProgress {
  phase: "discovering" | "hashing" | "committing";
  discoveredFiles: number;
  discoveredBytes: number;
  hashedFiles: number;
  hashedBytes: number;
  currentFile?: string;
  currentFileBytes?: number;
  currentFileHashedBytes?: number;
}

export interface DatasetManifestCreateOptions {
  signal?: AbortSignal;
  onProgress?: (progress: DatasetManifestCreateProgress) => void;
}

export class DatasetManifestStore {
  private readonly progressLocks = new Map<string, Promise<void>>();

  constructor(
    private readonly brainDirectory: (brainId: string) => string,
    private readonly options: DatasetManifestStoreOptions = {},
  ) {}

  private datasetRoot(brainId: string): string {
    return join(this.brainDirectory(brainId), "datasets");
  }

  private manifestDirectory(brainId: string, manifestId: string): string {
    if (!SAFE_ID.test(manifestId))
      throw new Error("Invalid dataset manifest id.");
    return join(this.datasetRoot(brainId), manifestId);
  }

  private manifestPath(brainId: string, manifestId: string): string {
    return join(this.manifestDirectory(brainId, manifestId), "manifest.json");
  }

  private cursorPath(brainId: string, manifestId: string): string {
    return join(this.manifestDirectory(brainId, manifestId), "cursor.json");
  }

  private coveragePath(brainId: string, manifestId: string): string {
    return join(this.manifestDirectory(brainId, manifestId), "coverage.json");
  }

  private progressPointerPath(brainId: string, manifestId: string): string {
    return join(this.manifestDirectory(brainId, manifestId), "progress.json");
  }

  private progressGenerationPath(
    brainId: string,
    manifestId: string,
    contentHash: string,
  ): string {
    if (!validSha256(contentHash))
      throw new Error("Invalid dataset progress generation id.");
    return join(
      this.manifestDirectory(brainId, manifestId),
      "progress",
      "generations",
      `${contentHash}.json`,
    );
  }

  private async withProgressLock<T>(
    brainId: string,
    manifestId: string,
    operation: () => Promise<T>,
  ): Promise<T> {
    const key = `${brainId}\0${manifestId}`;
    const prior = this.progressLocks.get(key) ?? Promise.resolve();
    let release = (): void => undefined;
    const gate = new Promise<void>((resolveGate) => {
      release = resolveGate;
    });
    const chain = prior.then(() => gate);
    this.progressLocks.set(key, chain);
    await prior;
    try {
      return await operation();
    } finally {
      release();
      if (this.progressLocks.get(key) === chain) this.progressLocks.delete(key);
    }
  }

  private validateCoverageCounters(
    coverage: TrainingCoverage,
    label: string,
  ): void {
    for (const field of [
      "discoveredFiles",
      "processedFiles",
      "rejectedFiles",
      "discoveredRecords",
      "processedRecords",
      "rejectedRecords",
      "discoveredBytes",
      "processedBytes",
      "shards",
    ] as const) {
      safeInteger(coverage[field], `${label}.${field}`);
    }
    if (
      coverage.processedFiles + coverage.rejectedFiles >
        coverage.discoveredFiles ||
      coverage.processedRecords + coverage.rejectedRecords !==
        coverage.discoveredRecords
    ) {
      throw new Error(`${label} traversal counters are inconsistent.`);
    }
    if (
      !coverage.modalityCounts ||
      typeof coverage.modalityCounts !== "object"
    ) {
      throw new Error(`${label}.modalityCounts is invalid.`);
    }
    for (const [kind, count] of Object.entries(coverage.modalityCounts)) {
      if (!kind || kind.length > 64)
        throw new Error(`${label} modality key is invalid.`);
      safeInteger(count, `${label}.modalityCounts.${kind}`);
    }
    if (
      !Array.isArray(coverage.errors) ||
      coverage.errors.length > COVERAGE_ERROR_SAMPLES
    ) {
      throw new Error(`${label}.errors exceeds its bounded diagnostic sample.`);
    }
    for (const error of coverage.errors) {
      if (
        !error ||
        typeof error.source !== "string" ||
        error.source.length > MAX_RECEIPT_TEXT ||
        typeof error.message !== "string" ||
        error.message.length > MAX_RECEIPT_TEXT
      ) {
        throw new Error(`${label} contains an invalid diagnostic.`);
      }
      if (error.count !== undefined)
        safeInteger(error.count, `${label}.errors.count`, 1);
    }
    const errorCount = coverage.errorCount ?? coverage.errors.length;
    safeInteger(errorCount, `${label}.errorCount`);
    if (errorCount < coverage.errors.length) {
      throw new Error(
        `${label}.errorCount is smaller than its diagnostic sample.`,
      );
    }
    if (typeof coverage.complete !== "boolean") {
      throw new Error(`${label}.complete is invalid.`);
    }
  }

  private validateReceipt(
    receipt: DatasetEntryReceipt,
    manifest: DatasetManifest,
    requestedEpochs: number,
  ): void {
    if (
      receipt.schemaVersion !== 1 ||
      receipt.manifestId !== manifest.id ||
      receipt.manifestHash !== manifest.manifestHash ||
      !validSha256(receipt.transactionKey) ||
      !validSha256(receipt.contentHash) ||
      !["encode", "consolidate", "pretrain", "archive"].includes(
        receipt.policy,
      ) ||
      !["learned", "rejected"].includes(receipt.outcome) ||
      typeof receipt.completedAt !== "string" ||
      receipt.completedAt.length > 128
    ) {
      throw new Error("Dataset entry receipt identity is invalid.");
    }
    safeInteger(
      receipt.entryIndex,
      "receipt.entryIndex",
      0,
      manifest.discoveredFiles - 1,
    );
    safeInteger(receipt.epoch, "receipt.epoch", 0, requestedEpochs - 1);
    if (receipt.transactionKey !== datasetEntryTransactionKey(receipt)) {
      throw new Error("Dataset entry receipt transaction checksum is invalid.");
    }
    for (const [field, value] of [
      ["sourceBytes", receipt.sourceBytes],
      ["learnedIdeas", receipt.learnedIdeas],
      ["learnedConcepts", receipt.learnedConcepts],
      ["learnedSynapses", receipt.learnedSynapses],
      ["learnedParameterSteps", receipt.learnedParameterSteps],
    ] as const) {
      if (value !== undefined) safeInteger(value, `receipt.${field}`);
    }
    for (const [field, value] of [
      ["sourceId", receipt.sourceId],
      ["journalId", receipt.journalId],
      ["sourceKind", receipt.sourceKind],
    ] as const) {
      if (
        value !== undefined &&
        (typeof value !== "string" || value.length > 256)
      ) {
        throw new Error(`receipt.${field} is invalid.`);
      }
    }
    if (
      receipt.warnings !== undefined &&
      (!Array.isArray(receipt.warnings) ||
        receipt.warnings.length > MAX_RECEIPT_WARNINGS ||
        receipt.warnings.some(
          (warning) =>
            typeof warning !== "string" || warning.length > MAX_RECEIPT_TEXT,
        ))
    ) {
      throw new Error(
        "receipt.warnings exceeds its bounded diagnostic sample.",
      );
    }
    if (
      receipt.workerDuplicate !== undefined &&
      typeof receipt.workerDuplicate !== "boolean"
    ) {
      throw new Error("receipt.workerDuplicate is invalid.");
    }
    if (
      receipt.parametersChanged !== undefined &&
      typeof receipt.parametersChanged !== "boolean"
    ) {
      throw new Error("receipt.parametersChanged is invalid.");
    }
    if (
      receipt.outcome === "learned" &&
      (!receipt.sourceId ||
        !receipt.sourceKind ||
        receipt.sourceBytes === undefined ||
        receipt.learnedIdeas === undefined ||
        receipt.learnedConcepts === undefined ||
        receipt.learnedSynapses === undefined)
    ) {
      throw new Error(
        "Learned dataset receipt is missing repository metadata.",
      );
    }
    if (receipt.coverage) {
      if (
        receipt.coverage.schemaVersion !== 1 ||
        receipt.coverage.manifestId !== manifest.id
      ) {
        throw new Error("Receipt coverage identity is invalid.");
      }
      this.validateCoverageCounters(receipt.coverage, "receipt.coverage");
    }
  }

  private validateProgressGeneration(
    value: DatasetProgressGeneration,
    manifest: DatasetManifest,
  ): DatasetProgressGeneration {
    const cursor = value.cursor;
    const coverage = value.coverage;
    if (
      value.schemaVersion !== PROGRESS_SCHEMA_VERSION ||
      value.format !== PROGRESS_FORMAT ||
      value.manifestId !== manifest.id ||
      value.manifestHash !== manifest.manifestHash ||
      cursor?.schemaVersion !== 1 ||
      cursor.manifestId !== manifest.id ||
      coverage?.schemaVersion !== 1 ||
      coverage.manifestId !== manifest.id ||
      typeof value.updatedAt !== "string" ||
      value.updatedAt.length > 128 ||
      !validSha256(value.contentSha256)
    ) {
      throw new Error("Dataset progress generation identity is invalid.");
    }
    safeInteger(value.generation, "progress.generation", 1);
    const requestedEpochs = safeInteger(
      cursor.requestedEpochs,
      "cursor.requestedEpochs",
      1,
    );
    if (coverage.requestedEpochs !== requestedEpochs) {
      throw new Error("Dataset cursor and coverage requested epochs disagree.");
    }
    const currentEpoch = safeInteger(
      cursor.currentEpoch,
      "cursor.currentEpoch",
      0,
      requestedEpochs,
    );
    const completedEpochs = safeInteger(
      coverage.completedEpochs,
      "coverage.completedEpochs",
      0,
      requestedEpochs,
    );
    if (completedEpochs > currentEpoch) {
      throw new Error("Dataset coverage is ahead of its cursor epoch.");
    }
    safeInteger(
      cursor.nextEntry,
      "cursor.nextEntry",
      0,
      manifest.discoveredFiles,
    );
    safeInteger(cursor.nextRecord, "cursor.nextRecord");
    safeInteger(cursor.processedFiles, "cursor.processedFiles");
    safeInteger(cursor.processedRecords, "cursor.processedRecords");
    safeInteger(cursor.processedBytes, "cursor.processedBytes");
    if (
      !["ready", "running", "paused", "complete", "failed"].includes(
        cursor.state,
      )
    ) {
      throw new Error("Dataset cursor state is invalid.");
    }
    this.validateCoverageCounters(coverage, "coverage");
    const expectedFiles = safeProduct(
      manifest.discoveredFiles,
      requestedEpochs,
      "coverage.discoveredFiles",
    );
    const expectedBytes = safeProduct(
      manifest.discoveredBytes,
      requestedEpochs,
      "coverage.discoveredBytes",
    );
    if (
      coverage.discoveredFiles !== expectedFiles ||
      coverage.discoveredBytes !== expectedBytes ||
      cursor.processedFiles !==
        coverage.processedFiles + coverage.rejectedFiles ||
      cursor.processedRecords !== coverage.processedRecords ||
      cursor.processedBytes !== coverage.processedBytes
    ) {
      throw new Error("Dataset cursor and coverage counters disagree.");
    }
    const traversalComplete =
      coverage.processedFiles + coverage.rejectedFiles ===
        coverage.discoveredFiles &&
      coverage.processedRecords + coverage.rejectedRecords ===
        coverage.discoveredRecords;
    const expectedComplete =
      expectedFiles === 0 ||
      (currentEpoch >= requestedEpochs && traversalComplete);
    if (
      coverage.complete !== expectedComplete ||
      (cursor.state === "complete") !== expectedComplete
    ) {
      throw new Error("Dataset completion state is inconsistent.");
    }
    if (value.lastEntryReceipt) {
      this.validateReceipt(value.lastEntryReceipt, manifest, requestedEpochs);
    }
    const { contentSha256: _checksum, ...body } = value;
    if (contentSha256(body) !== value.contentSha256) {
      throw new Error("Dataset progress generation checksum failed.");
    }
    return structuredClone(value);
  }

  private async progressUnlocked(
    brainId: string,
    manifest: DatasetManifest,
  ): Promise<DatasetProgressGeneration> {
    const pointer = JSON.parse(
      await readFile(this.progressPointerPath(brainId, manifest.id), "utf8"),
    ) as DatasetProgressPointer;
    safeInteger(pointer.generation, "progress pointer generation", 1);
    if (
      pointer.schemaVersion !== 1 ||
      pointer.format !== PROGRESS_POINTER_FORMAT ||
      pointer.manifestId !== manifest.id ||
      pointer.manifestHash !== manifest.manifestHash ||
      !validSha256(pointer.activeGeneration) ||
      pointer.generationFile !==
        `progress/generations/${pointer.activeGeneration}.json`
    ) {
      throw new Error(
        "Dataset progress pointer is invalid or belongs to another manifest.",
      );
    }
    const generation = JSON.parse(
      await readFile(
        this.progressGenerationPath(
          brainId,
          manifest.id,
          pointer.activeGeneration,
        ),
        "utf8",
      ),
    ) as DatasetProgressGeneration;
    const validated = this.validateProgressGeneration(generation, manifest);
    if (
      validated.generation !== pointer.generation ||
      validated.contentSha256 !== pointer.activeGeneration
    ) {
      throw new Error(
        "Dataset progress pointer does not match its generation.",
      );
    }
    return validated;
  }

  private async materializeCompatibility(
    brainId: string,
    state: DatasetProgressGeneration,
  ): Promise<void> {
    try {
      await this.options.compatibilityMaterializationHook?.(
        "after-authoritative",
      );
      await atomicJson(
        this.cursorPath(brainId, state.manifestId),
        state.cursor,
      );
      await this.options.compatibilityMaterializationHook?.("after-cursor");
      await atomicJson(
        this.coveragePath(brainId, state.manifestId),
        state.coverage,
      );
      await this.options.compatibilityMaterializationHook?.("after-coverage");
    } catch {
      // Compatibility files are disposable materializations. The checksummed
      // progress generation was already published and remains authoritative.
    }
  }

  private async publishProgressUnlocked(
    brainId: string,
    manifest: DatasetManifest,
    cursor: DatasetCursor,
    coverage: TrainingCoverage,
    lastEntryReceipt: DatasetEntryReceipt | undefined,
    expectedGeneration?: number,
  ): Promise<DatasetProgressGeneration> {
    let prior: DatasetProgressGeneration | undefined;
    if (await pathExists(this.progressPointerPath(brainId, manifest.id))) {
      prior = await this.progressUnlocked(brainId, manifest);
    }
    if (
      expectedGeneration !== undefined &&
      (prior?.generation ?? 0) !== expectedGeneration
    ) {
      throw new Error(
        "Dataset progress changed while this update was being committed.",
      );
    }
    const generation = (prior?.generation ?? 0) + 1;
    const now = new Date().toISOString();
    const body = {
      schemaVersion: 1 as const,
      format: PROGRESS_FORMAT,
      manifestId: manifest.id,
      manifestHash: manifest.manifestHash,
      generation,
      cursor: { ...structuredClone(cursor), updatedAt: now },
      coverage: { ...structuredClone(coverage), updatedAt: now },
      ...(lastEntryReceipt
        ? { lastEntryReceipt: structuredClone(lastEntryReceipt) }
        : {}),
      updatedAt: now,
    };
    const state: DatasetProgressGeneration = {
      ...body,
      contentSha256: contentSha256(body),
    };
    this.validateProgressGeneration(state, manifest);
    const generationPath = this.progressGenerationPath(
      brainId,
      manifest.id,
      state.contentSha256,
    );
    if (!(await pathExists(generationPath)))
      await atomicJson(generationPath, state);
    const pointer: DatasetProgressPointer = {
      schemaVersion: 1,
      format: PROGRESS_POINTER_FORMAT,
      manifestId: manifest.id,
      manifestHash: manifest.manifestHash,
      generation,
      activeGeneration: state.contentSha256,
      generationFile: `progress/generations/${state.contentSha256}.json`,
    };
    await atomicJson(this.progressPointerPath(brainId, manifest.id), pointer);
    await this.materializeCompatibility(brainId, state);
    return state;
  }

  async create(
    brainId: string,
    inputPaths: string[],
    options: DatasetManifestCreateOptions = {},
  ): Promise<DatasetManifest> {
    if (inputPaths.length === 0)
      throw new Error("Choose at least one dataset path.");
    throwIfCancelled(options.signal);
    const roots = [...new Set(inputPaths.map((value) => resolve(value)))].sort(
      comparePathNames,
    );
    const id = randomUUID();
    const directory = this.manifestDirectory(brainId, id);
    const entryPath = join(directory, "entries.ndjson");
    await mkdir(directory, { recursive: true });
    const output = createWriteStream(entryPath, {
      encoding: "utf8",
      flags: "wx",
      mode: 0o600,
    });
    const digest = createHash("sha256");
    let discoveredFiles = 0;
    let discoveredBytes = 0;
    let hashedFiles = 0;
    let hashedBytes = 0;
    let lastProgressAt = 0;
    let lastProgressBytes = 0;
    const publish = (
      progress: Omit<
        DatasetManifestCreateProgress,
        "discoveredFiles" | "discoveredBytes" | "hashedFiles" | "hashedBytes"
      >,
      force = false,
    ): void => {
      const now = Date.now();
      if (
        !force &&
        now - lastProgressAt < 100 &&
        hashedBytes - lastProgressBytes < 8 * 1024 * 1024
      )
        return;
      lastProgressAt = now;
      lastProgressBytes = hashedBytes;
      options.onProgress?.({
        ...progress,
        discoveredFiles,
        discoveredBytes,
        hashedFiles,
        hashedBytes,
      });
    };
    try {
      for await (const entry of walkRegularFiles(roots)) {
        throwIfCancelled(options.signal);
        discoveredFiles += 1;
        discoveredBytes += entry.bytes;
        const relativePath = entry.rootIsFile
          ? basename(entry.path)
          : relative(entry.root, entry.path).split(sep).join("/");
        const format = detectDatasetFormat(entry.path);
        let rejection = entry.rejection;
        let contentSha256: string | undefined;
        let committedPath = entry.path;
        let committedBytes = entry.bytes;
        let committedLastModifiedMs = entry.lastModifiedMs;
        let snapshotKind: DatasetManifestEntry["snapshotKind"];
        const currentFile = relativePath || basename(entry.path);
        publish(
          {
            phase: rejection ? "discovering" : "hashing",
            currentFile,
            currentFileBytes: entry.bytes,
            currentFileHashedBytes: 0,
          },
          true,
        );
        if (!rejection) {
          let currentFileHashedBytes = 0;
          try {
            if (format === "sqlite") {
              const snapshot = await snapshotSqliteDatabase(
                entry.path,
                directory,
                options.signal,
              );
              committedPath = snapshot.path;
              committedBytes = snapshot.bytes;
              committedLastModifiedMs = snapshot.lastModifiedMs;
              contentSha256 = snapshot.contentSha256;
              snapshotKind = "sqlite";
              discoveredBytes += committedBytes - entry.bytes;
              currentFileHashedBytes = committedBytes;
              hashedBytes += committedBytes;
            } else {
              contentSha256 = await hashFileWithProgress(entry.path, {
                signal: options.signal,
                onBytes: (bytes) => {
                  currentFileHashedBytes += bytes;
                  hashedBytes += bytes;
                  publish({
                    phase: "hashing",
                    currentFile,
                    currentFileBytes: entry.bytes,
                    currentFileHashedBytes,
                  });
                },
              });
              const afterHash = await stat(entry.path);
              if (
                !afterHash.isFile() ||
                afterHash.size !== entry.bytes ||
                Math.abs(afterHash.mtimeMs - entry.lastModifiedMs) > 0.001
              ) {
                rejection =
                  "Dataset source changed while its content hash was being committed; " +
                  "finish the download or write and create a new manifest.";
                contentSha256 = undefined;
              }
            }
          } catch (error) {
            if (
              options.signal?.aborted ||
              (error instanceof Error && error.name === "AbortError")
            ) {
              throw previewCancelled();
            }
            rejection = `Could not stream the dataset content hash: ${
              error instanceof Error ? error.message : String(error)
            }`;
          }
          hashedFiles += 1;
        }
        const record: DatasetManifestEntry = {
          index: discoveredFiles - 1,
          path: committedPath,
          ...(snapshotKind ? { sourcePath: entry.path, snapshotKind } : {}),
          relativePath: relativePath || basename(entry.path),
          format,
          bytes: committedBytes,
          lastModifiedMs: committedLastModifiedMs,
          ...(contentSha256 ? { contentSha256 } : {}),
          ...(rejection ? { rejection } : {}),
        };
        const line = `${JSON.stringify(record)}\n`;
        digest.update(line);
        await writeLine(output, line);
        publish(
          {
            phase: "discovering",
            currentFile,
            currentFileBytes: committedBytes,
            currentFileHashedBytes: rejection ? undefined : committedBytes,
          },
          true,
        );
      }
      throwIfCancelled(options.signal);
    } catch (error) {
      output.destroy();
      if (!output.closed) await once(output, "close");
      await rm(directory, { recursive: true, force: true });
      throw error;
    } finally {
      if (!output.destroyed) output.end();
      if (!output.closed) await once(output, "close");
    }
    publish({ phase: "committing" }, true);
    try {
      throwIfCancelled(options.signal);
    } catch (error) {
      await rm(directory, { recursive: true, force: true });
      throw error;
    }
    const now = new Date().toISOString();
    const manifest: DatasetManifest = {
      schemaVersion: 1,
      id,
      brainId,
      createdAt: now,
      updatedAt: now,
      roots,
      entryFile: "entries.ndjson",
      discoveredFiles,
      discoveredBytes,
      manifestHash: digest.digest("hex"),
    };
    const cursor: DatasetCursor = {
      schemaVersion: 1,
      manifestId: id,
      currentEpoch: 0,
      requestedEpochs: 1,
      nextEntry: 0,
      nextRecord: 0,
      processedFiles: 0,
      processedRecords: 0,
      processedBytes: 0,
      state: discoveredFiles === 0 ? "complete" : "ready",
      updatedAt: now,
    };
    const coverage: TrainingCoverage = {
      schemaVersion: 1,
      manifestId: id,
      requestedEpochs: 1,
      completedEpochs: 0,
      discoveredFiles,
      processedFiles: 0,
      rejectedFiles: 0,
      discoveredRecords: 0,
      processedRecords: 0,
      rejectedRecords: 0,
      discoveredBytes,
      processedBytes: 0,
      shards: 0,
      modalityCounts: {},
      errors: [],
      errorCount: 0,
      errorsTruncated: false,
      errorLog: "errors.ndjson",
      complete: discoveredFiles === 0,
      updatedAt: now,
    };
    try {
      await atomicJson(this.manifestPath(brainId, id), manifest);
      await this.publishProgressUnlocked(
        brainId,
        manifest,
        cursor,
        coverage,
        undefined,
        0,
      );
      throwIfCancelled(options.signal);
    } catch (error) {
      await rm(directory, { recursive: true, force: true });
      throw error;
    }
    return manifest;
  }

  async manifest(
    brainId: string,
    manifestId: string,
  ): Promise<DatasetManifest> {
    const parsed = JSON.parse(
      await readFile(this.manifestPath(brainId, manifestId), "utf8"),
    ) as DatasetManifest;
    if (
      parsed.schemaVersion !== 1 ||
      parsed.id !== manifestId ||
      parsed.brainId !== brainId ||
      !validSha256(parsed.manifestHash) ||
      parsed.entryFile !== "entries.ndjson" ||
      !Array.isArray(parsed.roots) ||
      parsed.roots.some(
        (root) => typeof root !== "string" || root.length > 32_768,
      )
    ) {
      throw new Error(
        "Dataset manifest is invalid or belongs to another brain.",
      );
    }
    safeInteger(parsed.discoveredFiles, "manifest.discoveredFiles");
    safeInteger(parsed.discoveredBytes, "manifest.discoveredBytes");
    return parsed;
  }

  async progress(
    brainId: string,
    manifestId: string,
  ): Promise<DatasetProgressGeneration> {
    const manifest = await this.manifest(brainId, manifestId);
    return this.progressUnlocked(brainId, manifest);
  }

  async cursor(brainId: string, manifestId: string): Promise<DatasetCursor> {
    return structuredClone((await this.progress(brainId, manifestId)).cursor);
  }

  async coverage(
    brainId: string,
    manifestId: string,
  ): Promise<TrainingCoverage> {
    return structuredClone((await this.progress(brainId, manifestId)).coverage);
  }

  async latestCompletedCoverage(
    brainId: string,
  ): Promise<PersistedDatasetCoverageSummary | undefined> {
    const root = this.datasetRoot(brainId);
    let entries;
    try {
      entries = await readdir(root, { withFileTypes: true });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
      throw error;
    }
    const completed: PersistedDatasetCoverageSummary[] = [];
    for (const entry of entries) {
      if (!entry.isDirectory() || !SAFE_ID.test(entry.name)) continue;
      // Crawl frontier SQLite files and attributed web-response cache entries
      // live under reserved dataset subdirectories before any dataset manifest
      // necessarily exists. Neither directory is a manifest generation; every
      // other safe-id directory remains strict so a missing manifest cannot
      // hide genuine dataset corruption.
      if (["crawls", "web-cache", "web-spool"].includes(entry.name)) continue;
      const manifest = await this.manifest(brainId, entry.name);
      const entryPath = join(
        this.manifestDirectory(brainId, manifest.id),
        manifest.entryFile,
      );
      if ((await hashFile(entryPath)) !== manifest.manifestHash) {
        throw new Error("Dataset manifest entry checksum failed.");
      }
      const progress = await this.progressUnlocked(brainId, manifest);
      const coverage = progress.coverage;
      if (!coverage.complete || progress.cursor.state !== "complete") continue;
      completed.push({
        brainId,
        manifestId: manifest.id,
        manifestHash: manifest.manifestHash,
        source: "validated-dataset-progress",
        discoveredFiles: coverage.discoveredFiles,
        processedFiles: coverage.processedFiles,
        rejectedFiles: coverage.rejectedFiles,
        discoveredRecords: coverage.discoveredRecords,
        processedRecords: coverage.processedRecords,
        rejectedRecords: coverage.rejectedRecords,
        complete: true,
        updatedAt: coverage.updatedAt,
      });
    }
    return completed.sort(
      (left, right) =>
        right.updatedAt.localeCompare(left.updatedAt) ||
        right.manifestId.localeCompare(left.manifestId),
    )[0];
  }

  async latestResumable(
    brainId: string,
  ): Promise<DatasetResumeCandidate | undefined> {
    const root = this.datasetRoot(brainId);
    let entries;
    try {
      entries = await readdir(root, { withFileTypes: true });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
      throw error;
    }
    const candidates: DatasetResumeCandidate[] = [];
    for (const entry of entries) {
      if (!entry.isDirectory() || !SAFE_ID.test(entry.name)) continue;
      if (["crawls", "web-cache", "web-spool"].includes(entry.name)) continue;
      const manifest = await this.manifest(brainId, entry.name);
      const entryPath = join(
        this.manifestDirectory(brainId, manifest.id),
        manifest.entryFile,
      );
      if ((await hashFile(entryPath)) !== manifest.manifestHash) {
        throw new Error("Dataset manifest entry checksum failed.");
      }
      const progress = await this.progressUnlocked(brainId, manifest);
      if (progress.coverage.complete || progress.cursor.state === "complete") {
        continue;
      }
      candidates.push({
        brainId,
        manifestId: manifest.id,
        manifestHash: manifest.manifestHash,
        state:
          progress.cursor.state === "running"
            ? "interrupted"
            : progress.cursor.state,
        discoveredFiles: manifest.discoveredFiles,
        processedFiles: progress.coverage.processedFiles,
        rejectedFiles: progress.coverage.rejectedFiles,
        currentEpoch: progress.cursor.currentEpoch ?? 0,
        requestedEpochs: progress.cursor.requestedEpochs ?? 1,
        updatedAt: progress.updatedAt,
      });
    }
    return candidates.sort(
      (left, right) =>
        right.updatedAt.localeCompare(left.updatedAt) ||
        right.manifestId.localeCompare(left.manifestId),
    )[0];
  }

  async *entries(
    brainId: string,
    manifestId: string,
    startIndex = 0,
  ): AsyncGenerator<DatasetManifestEntry> {
    const manifest = await this.manifest(brainId, manifestId);
    safeInteger(
      startIndex,
      "dataset entry start index",
      0,
      manifest.discoveredFiles,
    );
    const entryPath = join(
      this.manifestDirectory(brainId, manifestId),
      manifest.entryFile,
    );
    const actual = await hashFile(entryPath);
    if (actual !== manifest.manifestHash) {
      throw new Error("Dataset manifest entry checksum failed.");
    }
    const lines = createInterface({
      input: createReadStream(entryPath, { encoding: "utf8" }),
      crlfDelay: Infinity,
    });
    let expectedIndex = 0;
    for await (const line of lines) {
      if (!line.trim()) continue;
      const entry = JSON.parse(line) as DatasetManifestEntry;
      safeInteger(
        entry.index,
        "dataset manifest entry index",
        0,
        manifest.discoveredFiles - 1,
      );
      safeInteger(entry.bytes, "dataset manifest entry bytes");
      if (
        entry.index !== expectedIndex ||
        typeof entry.path !== "string" ||
        entry.path.length === 0 ||
        entry.path.length > 32_768 ||
        typeof entry.relativePath !== "string" ||
        entry.relativePath.length > 32_768 ||
        (entry.contentSha256 !== undefined && !validSha256(entry.contentSha256))
      ) {
        throw new Error("Dataset manifest entry contract is invalid.");
      }
      if (entry.snapshotKind === "sqlite") {
        const snapshots = resolve(
          this.manifestDirectory(brainId, manifestId),
          "snapshots",
        );
        const snapshot = resolve(entry.path);
        const remainder = relative(snapshots, snapshot);
        if (
          !entry.sourcePath ||
          entry.format !== "sqlite" ||
          remainder.startsWith(`..${sep}`) ||
          remainder === ".." ||
          isAbsolute(remainder) ||
          basename(snapshot) !== `${entry.contentSha256}.sqlite`
        ) {
          throw new Error("Dataset SQLite snapshot identity is invalid.");
        }
      }
      expectedIndex += 1;
      if (entry.index < startIndex) continue;
      yield entry;
    }
    if (expectedIndex !== manifest.discoveredFiles) {
      throw new Error(
        "Dataset manifest entry count does not match its declaration.",
      );
    }
  }

  async saveCursor(brainId: string, cursor: DatasetCursor): Promise<void> {
    await this.withProgressLock(brainId, cursor.manifestId, async () => {
      const current = await this.progress(brainId, cursor.manifestId);
      await this.publishProgressUnlocked(
        brainId,
        await this.manifest(brainId, cursor.manifestId),
        cursor,
        current.coverage,
        current.lastEntryReceipt,
        current.generation,
      );
    });
  }

  async saveCoverage(
    brainId: string,
    coverage: TrainingCoverage,
  ): Promise<void> {
    await this.withProgressLock(brainId, coverage.manifestId, async () => {
      const current = await this.progress(brainId, coverage.manifestId);
      await this.publishProgressUnlocked(
        brainId,
        await this.manifest(brainId, coverage.manifestId),
        current.cursor,
        coverage,
        current.lastEntryReceipt,
        current.generation,
      );
    });
  }

  async saveProgress(
    brainId: string,
    cursor: DatasetCursor,
    coverage: TrainingCoverage,
    lastEntryReceipt?: DatasetEntryReceipt,
    expectedGeneration?: number,
  ): Promise<DatasetProgressGeneration> {
    if (cursor.manifestId !== coverage.manifestId) {
      throw new Error(
        "Dataset cursor and coverage belong to different manifests.",
      );
    }
    return this.withProgressLock(brainId, cursor.manifestId, async () =>
      this.publishProgressUnlocked(
        brainId,
        await this.manifest(brainId, cursor.manifestId),
        cursor,
        coverage,
        lastEntryReceipt,
        expectedGeneration,
      ),
    );
  }

  async recordError(
    brainId: string,
    coverage: TrainingCoverage,
    error: TrainingCoverageError,
  ): Promise<void> {
    const path = join(
      this.manifestDirectory(brainId, coverage.manifestId),
      coverage.errorLog ?? "errors.ndjson",
    );
    await mkdir(dirname(path), { recursive: true });
    const stream = createWriteStream(path, {
      encoding: "utf8",
      flags: "a",
      mode: 0o600,
    });
    stream.end(`${JSON.stringify(error)}\n`);
    await once(stream, "close");
    const occurrences = Math.max(1, Math.round(error.count ?? 1));
    coverage.errorCount =
      (coverage.errorCount ?? coverage.errors.length) + occurrences;
    if (coverage.errors.length < COVERAGE_ERROR_SAMPLES)
      coverage.errors.push(error);
    else coverage.errorsTruncated = true;
  }

  async reset(
    brainId: string,
    manifestId: string,
    requestedEpochs = 1,
  ): Promise<void> {
    const manifest = await this.manifest(brainId, manifestId);
    const now = new Date().toISOString();
    const epochs = Math.max(1, Math.round(requestedEpochs));
    await writeFile(
      join(this.manifestDirectory(brainId, manifestId), "errors.ndjson"),
      "",
      { encoding: "utf8", mode: 0o600 },
    );
    await this.saveProgress(
      brainId,
      {
        schemaVersion: 1,
        manifestId,
        currentEpoch: 0,
        requestedEpochs: epochs,
        nextEntry: 0,
        nextRecord: 0,
        processedFiles: 0,
        processedRecords: 0,
        processedBytes: 0,
        state: manifest.discoveredFiles === 0 ? "complete" : "ready",
        updatedAt: now,
      },
      {
        schemaVersion: 1,
        manifestId,
        requestedEpochs: epochs,
        completedEpochs: 0,
        discoveredFiles: manifest.discoveredFiles * epochs,
        processedFiles: 0,
        rejectedFiles: 0,
        discoveredRecords: 0,
        processedRecords: 0,
        rejectedRecords: 0,
        discoveredBytes: manifest.discoveredBytes * epochs,
        processedBytes: 0,
        shards: 0,
        modalityCounts: {},
        errors: [],
        errorCount: 0,
        errorsTruncated: false,
        errorLog: "errors.ndjson",
        complete: manifest.discoveredFiles === 0,
        updatedAt: now,
      },
      undefined,
    );
  }

  async hasManifest(brainId: string, manifestId: string): Promise<boolean> {
    return pathExists(this.manifestPath(brainId, manifestId));
  }
}

export interface CrawlFrontierEntry {
  url: string;
  depth: number;
}

export interface CrawlFrontierCounts {
  queued: number;
  visited: number;
  skipped: number;
  processedBytes: number;
  resultCount: number;
  warningCount: number;
  modalityCounts: Partial<Record<DatasetFormat, number>>;
}

export interface CrawlResultReceipt {
  sourceId: string;
  sourceName: string;
  kind: DatasetFormat;
  bytes: number;
  contentHash?: string;
  learnedIdeas: number;
  learnedConcepts: number;
  learnedSynapses: number;
  warnings: string[];
}

export class CrawlFrontierStore {
  readonly id: string;
  private readonly database: DatabaseSync;

  constructor(
    brainDirectory: string,
    brainId: string,
    startUrl: string,
    requestedId?: string,
    resume = true,
  ) {
    this.id =
      requestedId && SAFE_ID.test(requestedId)
        ? requestedId
        : createHash("sha256")
            .update(`${brainId}\0${startUrl}`)
            .digest("hex")
            .slice(0, 32);
    const directory = join(brainDirectory, "datasets", "crawls");
    this.database = new DatabaseSync(join(directory, `${this.id}.sqlite3`), {
      open: false,
    });
    // DatabaseSync cannot create a missing parent directory itself.
    // mkdirSync is intentionally avoided elsewhere, so open happens after the
    // caller has prepared this known brain-local directory.
    this.prepare(directory, brainId, startUrl, resume);
  }

  private prepare(
    directory: string,
    brainId: string,
    startUrl: string,
    resume: boolean,
  ): void {
    // DatabaseSync.open is synchronous; creating only this exact app-owned
    // directory before construction is handled by CrawlFrontierStore.create.
    this.database.open();
    this.database.exec(`
      PRAGMA journal_mode=WAL;
      PRAGMA synchronous=FULL;
      CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS frontier (
        url TEXT PRIMARY KEY,
        depth INTEGER NOT NULL,
        state TEXT NOT NULL DEFAULT 'queued',
        added_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS visited (
        url TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        visited_at TEXT NOT NULL,
        error TEXT,
        bytes INTEGER NOT NULL DEFAULT 0
      );
      CREATE TABLE IF NOT EXISTS warnings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        message TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS result_receipts (
        url TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        source_name TEXT NOT NULL,
        kind TEXT NOT NULL,
        bytes INTEGER NOT NULL DEFAULT 0,
        content_hash TEXT,
        learned_ideas INTEGER NOT NULL DEFAULT 0,
        learned_concepts INTEGER NOT NULL DEFAULT 0,
        learned_synapses INTEGER NOT NULL DEFAULT 0,
        warnings_json TEXT NOT NULL DEFAULT '[]',
        recorded_at TEXT NOT NULL
      );
    `);
    const visitedColumns = this.database
      .prepare("PRAGMA table_info(visited)")
      .all() as unknown as Array<{ name: string }>;
    if (!visitedColumns.some((column) => column.name === "bytes")) {
      this.database.exec(
        "ALTER TABLE visited ADD COLUMN bytes INTEGER NOT NULL DEFAULT 0",
      );
    }
    const existing = this.database
      .prepare("SELECT value FROM meta WHERE key = 'startUrl'")
      .get() as { value?: string } | undefined;
    if (existing?.value && existing.value !== startUrl) {
      throw new Error("The crawl id belongs to a different start URL.");
    }
    if (!resume) {
      this.database.exec(
        "DELETE FROM frontier; DELETE FROM visited; DELETE FROM warnings; DELETE FROM result_receipts;",
      );
    }
    const insertMeta = this.database.prepare(
      "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
    );
    insertMeta.run("brainId", brainId);
    insertMeta.run("startUrl", startUrl);
    insertMeta.run("updatedAt", new Date().toISOString());
    this.database
      .prepare("UPDATE frontier SET state = 'queued' WHERE state = 'inflight'")
      .run();
    this.enqueue([{ url: startUrl, depth: 0 }]);
  }

  static async create(
    brainDirectory: string,
    brainId: string,
    startUrl: string,
    requestedId?: string,
    resume = true,
  ): Promise<CrawlFrontierStore> {
    await mkdir(join(brainDirectory, "datasets", "crawls"), {
      recursive: true,
    });
    return new CrawlFrontierStore(
      brainDirectory,
      brainId,
      startUrl,
      requestedId,
      resume,
    );
  }

  private transact(operation: () => void): void {
    this.database.exec("BEGIN IMMEDIATE");
    try {
      operation();
      this.database.exec("COMMIT");
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
  }

  enqueue(entries: CrawlFrontierEntry[]): void {
    const insert = this.database.prepare(`
      INSERT OR IGNORE INTO frontier(url, depth, state, added_at)
      SELECT ?, ?, 'queued', ?
      WHERE NOT EXISTS (SELECT 1 FROM visited WHERE url = ?)
    `);
    this.transact(() => {
      const now = new Date().toISOString();
      for (const entry of entries) {
        insert.run(
          entry.url,
          Math.max(0, Math.round(entry.depth)),
          now,
          entry.url,
        );
      }
    });
  }

  next(limit: number): CrawlFrontierEntry[] {
    const entries = this.database
      .prepare(
        "SELECT url, depth FROM frontier WHERE state = 'queued' ORDER BY rowid LIMIT ?",
      )
      .all(Math.max(1, Math.round(limit))) as unknown as CrawlFrontierEntry[];
    const mark = this.database.prepare(
      "UPDATE frontier SET state = 'inflight' WHERE url = ?",
    );
    this.transact(() => {
      for (const entry of entries) mark.run(entry.url);
    });
    return entries;
  }

  visited(url: string, bytes = 0, receipt?: CrawlResultReceipt): void {
    this.finish(url, "visited", undefined, bytes, receipt);
  }

  skipped(url: string, message?: string): void {
    this.finish(url, "skipped", message);
  }

  retry(url: string): void {
    this.database
      .prepare("UPDATE frontier SET state = 'queued' WHERE url = ?")
      .run(url);
  }

  private finish(
    url: string,
    status: "visited" | "skipped",
    error?: string,
    bytes = 0,
    receipt?: CrawlResultReceipt,
  ): void {
    this.transact(() => {
      this.database.prepare("DELETE FROM frontier WHERE url = ?").run(url);
      this.database
        .prepare(
          "INSERT OR REPLACE INTO visited(url, status, visited_at, error, bytes) VALUES (?, ?, ?, ?, ?)",
        )
        .run(
          url,
          status,
          new Date().toISOString(),
          error ?? null,
          Math.max(0, bytes),
        );
      if (error) {
        this.database
          .prepare("INSERT INTO warnings(url, message) VALUES (?, ?)")
          .run(url, error);
      }
      if (receipt) {
        this.database
          .prepare(
            `INSERT OR REPLACE INTO result_receipts(
              url, source_id, source_name, kind, bytes, content_hash,
              learned_ideas, learned_concepts, learned_synapses,
              warnings_json, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          )
          .run(
            url,
            receipt.sourceId,
            receipt.sourceName,
            receipt.kind,
            Math.max(0, Math.round(receipt.bytes)),
            receipt.contentHash ?? null,
            Math.max(0, Math.round(receipt.learnedIdeas)),
            Math.max(0, Math.round(receipt.learnedConcepts)),
            Math.max(0, Math.round(receipt.learnedSynapses)),
            JSON.stringify(receipt.warnings),
            new Date().toISOString(),
          );
      }
      this.database
        .prepare(
          "INSERT OR REPLACE INTO meta(key, value) VALUES ('updatedAt', ?)",
        )
        .run(new Date().toISOString());
    });
  }

  recordWarning(url: string, message: string): void {
    this.database
      .prepare("INSERT INTO warnings(url, message) VALUES (?, ?)")
      .run(url, message);
  }

  counts(): CrawlFrontierCounts {
    const queued = this.database
      .prepare("SELECT COUNT(*) AS count FROM frontier")
      .get() as { count: number };
    const counts = this.database
      .prepare(
        "SELECT status, COUNT(*) AS count, COALESCE(SUM(bytes), 0) AS bytes FROM visited GROUP BY status",
      )
      .all() as unknown as Array<{
      status: string;
      count: number;
      bytes: number;
    }>;
    const receiptCounts = this.database
      .prepare(
        "SELECT kind, COUNT(*) AS count FROM result_receipts GROUP BY kind",
      )
      .all() as unknown as Array<{ kind: DatasetFormat; count: number }>;
    const warnings = this.database
      .prepare("SELECT COUNT(*) AS count FROM warnings")
      .get() as { count: number };
    const modalityCounts: Partial<Record<DatasetFormat, number>> = {};
    for (const entry of receiptCounts) {
      modalityCounts[entry.kind] = Number(entry.count);
    }
    return {
      queued: Number(queued.count),
      visited: Number(
        counts.find((entry) => entry.status === "visited")?.count ?? 0,
      ),
      skipped: Number(
        counts.find((entry) => entry.status === "skipped")?.count ?? 0,
      ),
      processedBytes: Number(
        counts.find((entry) => entry.status === "visited")?.bytes ?? 0,
      ),
      resultCount: receiptCounts.reduce(
        (sum, entry) => sum + Number(entry.count),
        0,
      ),
      warningCount: Number(warnings.count),
      modalityCounts,
    };
  }

  warnings(limit = 64): string[] {
    const records = this.database
      .prepare("SELECT url, message FROM warnings ORDER BY id DESC LIMIT ?")
      .all(Math.max(1, Math.round(limit))) as unknown as Array<{
      url: string;
      message: string;
    }>;
    return records.reverse().map((entry) => `${entry.url}: ${entry.message}`);
  }

  close(): void {
    this.database.close();
  }
}
