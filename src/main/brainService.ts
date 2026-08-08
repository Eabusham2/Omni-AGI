import { createHash, randomUUID } from "node:crypto";
import { lookup } from "node:dns/promises";
import { createReadStream } from "node:fs";
import {
  lstat,
  mkdir,
  mkdtemp,
  open,
  readFile,
  readdir,
  realpath,
  rm,
  stat,
  statfs,
  writeFile
} from "node:fs/promises";
import { isIP } from "node:net";
import { availableParallelism } from "node:os";
import { basename, extname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { EventEmitter } from "node:events";
import { getHeapStatistics } from "node:v8";
import type {
  AgentMergeFilePreview,
  AgentMergePreview,
  AgentMergeSubstratePreview,
  BrainConfig,
  BrainDocument,
  BuildRecipe,
  ChatResult,
  CreateBrainRequest,
  DataIngestionPolicy,
  DatasetManifest,
  DatasetStartRequest,
  FeedbackRequest,
  IdleCycleResult,
  InstalledModalityPack,
  ImportUrlRequest,
  IngestWebRequest,
  IngestResult,
  ModalityGenerateRequest,
  ModalityPreview,
  RuntimeHealth,
  RuntimeJob,
  RuntimeJobEvent,
  StartTrainingRequest,
  StructuredAction,
  SubstratePage,
  SubstrateQuery,
  ToolPermissionLevel,
  ToolPermissionRecord,
  TrainingCoverage,
  TrainingSource,
  WebCrawlRequest,
  WebCrawlResult,
  WorkspaceSnapshot
} from "../shared/types";
import { recordNeuralChat } from "./presentationChat";
import {
  listInstalledPacks,
  recordInstalledPack,
  removeStagedPack,
  stageModalityPack,
  validateBuildRecipe
} from "./catalogInstaller";
import { BrainRepository } from "./brainRepository";
import { withBrainWrite } from "./brainWriteCoordinator";
import { EngineSupervisor, type EngineEvent } from "./engineSupervisor";
import { normalizeStructuredAction, parseModelActions } from "./actionProtocol";
import {
  CrawlFrontierStore,
  DatasetManifestStore,
  detectDatasetFormat,
  hashFile
} from "./dataIngestion";

const MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024;
const MAX_MERGE_FILES = 512;
const MAX_MERGE_FILE_BYTES = 128 * 1024 * 1024;
const MAX_MERGE_TOTAL_BYTES = 512 * 1024 * 1024;
const MAX_MERGE_CONFLICTS = 100;
const MAX_ROBOTS_BYTES = 2 * 1024 * 1024;
const CRAWL_DISK_CHECK_INTERVAL = 4 * 1024 * 1024;
const CRAWL_MEMORY_RESERVE_MINIMUM = 64 * 1024 * 1024;
const CRAWL_DISK_RESERVE_MINIMUM = 512 * 1024 * 1024;
const CRAWL_DIAGNOSTIC_WINDOW = 32;
const CRAWL_USER_AGENT = "OmniAGIStudio/1.0 (+local research crawler)";
const WORKING_MEMORY_SLOTS_BY_TIER = {
  micro: 128,
  personal: 256,
  gpu: 512,
  workstation: 1024
} as const;
const TOOL_ACTIONS: Readonly<Record<string, readonly string[]>> = {
  "windows.files": ["list", "read", "write"],
  "windows.powershell": ["run"],
  "code.execute": ["run"],
  "web.fetch": ["fetch"],
  "web.search": ["search"],
  "browser.automation": ["task"],
  "modality.imagine": ["generate"],
  "agent.fork": ["start"],
  "source.self-modify": ["propose", "diff", "test", "promote", "rollback"]
};

function enabledToolSchemas(brain: BrainDocument): Array<{
  id: string;
  actions: readonly string[];
  grant: ToolPermissionLevel;
}> {
  return (brain.toolPermissions ?? [])
    .filter((permission) => permission.level !== "off")
    .flatMap((permission) => {
      const actions = TOOL_ACTIONS[permission.toolId];
      return actions
        ? [
            {
              id: permission.toolId,
              actions,
              grant: permission.level
            }
          ]
        : [];
    })
    .slice(0, 100);
}

interface WorkerChatResult {
  text?: string;
  response?: string;
  content?: string;
  runtime?: string;
  trace?: {
    id?: string;
    created_at?: string;
    seed?: number;
    parameter_checksum_before?: string;
    parameter_checksum_after?: string;
    parameter_delta_norm?: number;
    stdp_update?: number;
    spike_rate?: number;
    train_loss?: number;
    generation_entropy?: number;
    ponder_steps?: number;
    steps?: Array<{ stage?: string; detail?: string; value?: string }>;
    note?: string;
  };
  metrics?: Record<string, unknown>;
  runtimeCard?: Record<string, unknown>;
  actions?: unknown;
}

export type NeuralChatStreamEvent =
  | {
      type: "chat-token";
      sequence: number;
      delta: string;
    }
  | {
      type: "chat-action";
      sequence: number;
      actionId?: string;
      action: StructuredAction;
    }
  | {
      type: "modality-preview";
      sequence: number;
      actionId?: string;
      preview: ModalityPreview;
    };

function objectRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function boundedWorkerText(value: unknown, maximum: number): string | undefined {
  if (typeof value !== "string") return undefined;
  const text = value.replace(/\0/g, "").slice(0, maximum);
  return text || undefined;
}

export function normalizeModalityPreview(
  value: unknown,
  fallbackRevision = 0,
  outerProgress?: number,
  outerMessage?: string
): ModalityPreview | undefined {
  const record = objectRecord(value);
  if (!record) return undefined;
  const mimeType = boundedWorkerText(record.mimeType ?? record.mime_type, 128);
  const supportedMime =
    mimeType && /^(?:image|audio|video)\/[a-z0-9.+-]{1,80}$/i.test(mimeType)
      ? mimeType.toLocaleLowerCase()
      : undefined;
  const rawDataUrl = boundedWorkerText(record.dataUrl ?? record.data_url, 16 * 1024 * 1024);
  const dataUrlPrefix = supportedMime ? `data:${supportedMime};base64,` : "";
  const dataUrlPayload =
    rawDataUrl && dataUrlPrefix && rawDataUrl.startsWith(dataUrlPrefix)
      ? rawDataUrl.slice(dataUrlPrefix.length)
      : "";
  const dataUrl =
    rawDataUrl &&
    supportedMime &&
    dataUrlPayload &&
    /^[a-z0-9+/]*={0,2}$/i.test(dataUrlPayload)
      ? rawDataUrl
      : undefined;
  const numberValue =
    typeof record.progress === "number" && Number.isFinite(record.progress)
      ? record.progress
      : outerProgress;
  const progress =
    typeof numberValue === "number" && Number.isFinite(numberValue)
      ? Math.max(0, Math.min(1, numberValue))
      : undefined;
  const statusLabel =
    boundedWorkerText(record.statusLabel ?? record.status_label, 200) ??
    boundedWorkerText(outerMessage, 200);
  const path = boundedWorkerText(record.path, 32_000);
  const artifactPath = boundedWorkerText(
    record.artifactPath ?? record.artifact_path,
    32_000
  );
  const rawRevision = record.revision;
  const revision =
    typeof rawRevision === "number" &&
    Number.isSafeInteger(rawRevision) &&
    rawRevision >= 0
      ? rawRevision
      : Math.max(0, fallbackRevision);
  if (
    progress === undefined &&
    !statusLabel &&
    !dataUrl &&
    !path &&
    !artifactPath
  ) {
    return undefined;
  }
  return {
    revision,
    progress,
    statusLabel,
    mimeType: supportedMime,
    dataUrl,
    path,
    artifactPath
  };
}

function normalizeChatEngineEvent(
  event: EngineEvent,
  brainId: string
): NeuralChatStreamEvent | undefined {
  if (event.brainId !== undefined && event.brainId !== brainId) return undefined;
  const sequence = event.sequence;
  if (
    typeof sequence !== "number" ||
    !Number.isSafeInteger(sequence) ||
    sequence < 0
  ) {
    return undefined;
  }
  const data = objectRecord(event.data);
  if (event.type === "chat-token") {
    const delta = boundedWorkerText(data?.delta, 64 * 1024);
    return delta ? { type: "chat-token", sequence, delta } : undefined;
  }
  if (event.type === "chat-action") {
    const action = normalizeStructuredAction(data?.action ?? event.data, "brain");
    if (!action) return undefined;
    return {
      type: "chat-action",
      sequence,
      actionId: boundedWorkerText(event.actionId ?? data?.actionId ?? data?.action_id, 128),
      action
    };
  }
  if (event.type === "modality-preview") {
    const preview = normalizeModalityPreview(
      data?.preview ?? event.data,
      sequence,
      event.progress,
      event.message
    );
    if (!preview) return undefined;
    return {
      type: "modality-preview",
      sequence,
      actionId: boundedWorkerText(event.actionId ?? data?.actionId ?? data?.action_id, 128),
      preview
    };
  }
  return undefined;
}

interface WorkerIngestResult {
  duplicate?: boolean;
  source?: {
    kind?: string;
    learned_ideas?: number;
    learned_concepts?: number;
    plasticity_events?: number;
    warnings?: string[];
    coverage?: WorkerTrainingCoverage;
  };
  warnings?: string[];
  coverage?: WorkerTrainingCoverage;
}

interface WorkerFeedbackResult {
  direction?: "up" | "down";
  stdp?: {
    stdp_update?: number;
    plasticity_events?: number;
  };
  parameterChecksumBefore?: string;
  parameterChecksumAfter?: string;
  synapseChecksumBefore?: string;
  synapseChecksumAfter?: string;
  rewardModel?: boolean;
  rlhf?: boolean;
  metrics?: Record<string, unknown>;
}

interface WorkerIdleCycleResult
  extends Omit<IdleCycleResult, "actions" | "actionEvents"> {
  actions?: unknown;
}

interface WorkerTrainingCoverage extends Partial<TrainingCoverage> {
  completedFiles?: number;
}

function coverageCount(value: unknown, fallback = 0): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  return Math.max(0, Math.trunc(value));
}

function normalizeWorkerTrainingCoverage(
  raw: WorkerTrainingCoverage,
  fileBytes: number
): TrainingCoverage {
  const discoveredFiles = coverageCount(raw.discoveredFiles, 1);
  const processedFiles = coverageCount(raw.processedFiles ?? raw.completedFiles);
  const rejectedFiles = coverageCount(raw.rejectedFiles);
  const discoveredRecords = coverageCount(raw.discoveredRecords);
  const processedRecords = coverageCount(raw.processedRecords);
  const rejectedRecords = coverageCount(raw.rejectedRecords);
  const classifiedFiles = processedFiles + rejectedFiles;
  const classifiedRecords = processedRecords + rejectedRecords;
  const complete =
    raw.complete !== false &&
    classifiedFiles === discoveredFiles &&
    classifiedRecords === discoveredRecords;
  return {
    schemaVersion: 1,
    manifestId: "",
    discoveredFiles,
    processedFiles,
    rejectedFiles,
    discoveredRecords,
    processedRecords,
    rejectedRecords,
    discoveredBytes: coverageCount(raw.discoveredBytes, fileBytes),
    processedBytes: coverageCount(raw.processedBytes),
    shards: coverageCount(raw.shards),
    modalityCounts: raw.modalityCounts ?? {},
    errors: raw.errors ?? [],
    complete,
    updatedAt: new Date().toISOString()
  };
}

function hasCompleteTraversal(
  coverage: Pick<
    TrainingCoverage,
    | "discoveredFiles"
    | "processedFiles"
    | "rejectedFiles"
    | "discoveredRecords"
    | "processedRecords"
    | "rejectedRecords"
  >
): boolean {
  return (
    coverage.processedFiles + coverage.rejectedFiles === coverage.discoveredFiles &&
    coverage.processedRecords + coverage.rejectedRecords === coverage.discoveredRecords
  );
}

interface JobRecord extends RuntimeJob {
  cancelled?: boolean;
}

interface MergeFileCandidate extends AgentMergeFilePreview {
  absoluteSourcePath?: string;
  blobHash?: string;
  evidenceFingerprint?: string;
}

interface MergePlan {
  source: BrainDocument;
  target: BrainDocument;
  evidence: Array<{ fingerprint: string; source: TrainingSource }>;
  files: MergeFileCandidate[];
  substrate: AgentMergeSubstratePreview;
  preview: AgentMergePreview;
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function mergeCountRecord(
  value: unknown,
  fields: readonly string[]
): Record<string, number> | undefined {
  const record = objectRecord(value);
  if (!record) return undefined;
  const result: Record<string, number> = {};
  for (const field of fields) {
    const count = record[field];
    if (
      typeof count !== "number" ||
      !Number.isSafeInteger(count) ||
      count < 0
    ) {
      return undefined;
    }
    result[field] = count;
  }
  return result;
}

function normalizeMergeSubstratePreview(
  value: unknown,
  sourceBrainId: string,
  targetBrainId: string
): AgentMergeSubstratePreview | undefined {
  const record = objectRecord(value);
  if (
    !record ||
    record.schemaVersion !== 1 ||
    record.engineSchemaVersion !== 1 ||
    record.sourceBrainId !== sourceBrainId ||
    record.targetBrainId !== targetBrainId ||
    record.weightsAveraged !== false
  ) {
    return undefined;
  }
  const shaFields = [
    "digest",
    "sourceStateSha256",
    "targetStateSha256",
    "sourceParameterSha256",
    "targetParameterSha256",
    "sourceConfigSha256",
    "targetConfigSha256"
  ] as const;
  const hashes = Object.fromEntries(
    shaFields.map((field) => [
      field,
      typeof record[field] === "string" &&
      /^[a-f0-9]{64}$/.test(record[field] as string)
        ? record[field]
        : undefined
    ])
  ) as Record<(typeof shaFields)[number], string | undefined>;
  if (shaFields.some((field) => !hashes[field])) return undefined;
  const countFields = [
    "neurons",
    "assemblies",
    "synapses",
    "replayExamples"
  ] as const;
  const sourceCounts = mergeCountRecord(record.sourceCounts, countFields);
  const targetCounts = mergeCountRecord(record.targetCounts, countFields);
  const additions = mergeCountRecord(record.additions, countFields);
  const duplicates = mergeCountRecord(record.duplicates, countFields);
  const divergent = mergeCountRecord(
    record.divergent,
    ["neurons", "assemblies", "synapses"]
  );
  if (!sourceCounts || !targetCounts || !additions || !duplicates || !divergent) {
    return undefined;
  }
  for (const field of countFields) {
    if (additions[field]! + duplicates[field]! !== sourceCounts[field]) {
      return undefined;
    }
  }
  return {
    schemaVersion: 1,
    engineSchemaVersion: 1,
    digest: hashes.digest!,
    sourceStateSha256: hashes.sourceStateSha256!,
    targetStateSha256: hashes.targetStateSha256!,
    sourceParameterSha256: hashes.sourceParameterSha256!,
    targetParameterSha256: hashes.targetParameterSha256!,
    sourceConfigSha256: hashes.sourceConfigSha256!,
    targetConfigSha256: hashes.targetConfigSha256!,
    sourceCounts: sourceCounts as AgentMergeSubstratePreview["sourceCounts"],
    targetCounts: targetCounts as AgentMergeSubstratePreview["targetCounts"],
    additions: additions as AgentMergeSubstratePreview["additions"],
    duplicates: duplicates as AgentMergeSubstratePreview["duplicates"],
    divergent: divergent as AgentMergeSubstratePreview["divergent"],
    weightsAveraged: false
  };
}

function safeMergeName(value: string): string {
  return (
    value
      .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "-")
      .replace(/[. ]+$/g, "")
      .slice(0, 100) || "artifact"
  );
}

function portableRelative(path: string): string {
  return path.split(sep).join("/");
}

function pathWithin(root: string, candidate: string): boolean {
  const value = relative(resolve(root), resolve(candidate));
  return value === "" || (!isAbsolute(value) && value !== ".." && !value.startsWith(`..${sep}`));
}

async function containedRegularFile(
  root: string,
  candidate: string
): Promise<string | undefined> {
  try {
    const info = await lstat(candidate);
    if (info.isSymbolicLink() || !info.isFile()) return undefined;
    const [realRoot, realCandidate] = await Promise.all([realpath(root), realpath(candidate)]);
    return pathWithin(realRoot, realCandidate) ? realCandidate : undefined;
  } catch {
    return undefined;
  }
}

async function assertSafeDestinationParents(root: string, candidate: string): Promise<void> {
  if (!pathWithin(root, candidate) || resolve(root) === resolve(candidate)) {
    throw new Error("Merge destination escapes the target brain directory.");
  }
  const relativeDestination = relative(resolve(root), resolve(candidate));
  const parts = relativeDestination.split(sep).slice(0, -1);
  let current = resolve(root);
  for (const part of parts) {
    current = join(current, part);
    try {
      const info = await lstat(current);
      if (info.isSymbolicLink() || !info.isDirectory()) {
        throw new Error("Merge destination contains a symbolic link or non-directory parent.");
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
      throw error;
    }
  }
}

function evidenceFingerprint(source: TrainingSource): string {
  if (source.contentHash && /^[a-f0-9]{64}$/i.test(source.contentHash)) {
    return `content:${source.contentHash.toLocaleLowerCase()}`;
  }
  if (source.blobHash && /^[a-f0-9]{64}$/i.test(source.blobHash)) {
    return `blob:${source.blobHash.toLocaleLowerCase()}`;
  }
  return `metadata:${sha256(
    JSON.stringify({
      name: source.name,
      kind: source.kind,
      bytes: source.bytes,
      provenanceUrl: source.provenanceUrl ?? "",
      importedAt: source.importedAt
    })
  )}`;
}

function retainsMergedBlob(target: BrainDocument, source: TrainingSource): boolean {
  const recipe = target.config.memoryRecipe ?? "human-consolidation";
  if (recipe === "synapses-only") return false;
  return (
    recipe === "total-recall" ||
    source.policy === "archive" ||
    source.kind === "image" ||
    source.kind === "audio" ||
    source.kind === "video"
  );
}

function cleanMessage(value: string): string {
  const clean = value.replace(/\0/g, "").trim();
  if (!clean) throw new Error("A chat message cannot be empty.");
  if (clean.length > 100_000) throw new Error("A chat message cannot exceed 100,000 characters.");
  return clean;
}

function normalizeToolId(toolId: string): string {
  const normalized = toolId.trim().toLocaleLowerCase();
  if (!/^[a-z][a-z0-9.-]{1,79}$/.test(normalized)) throw new Error("Invalid tool id.");
  return normalized;
}

function trainingKind(path: string): TrainingSource["kind"] {
  const datasetFormat = detectDatasetFormat(path);
  if (datasetFormat === "csv" || datasetFormat === "tsv") return "csv";
  if (datasetFormat === "parquet") return "parquet";
  if (datasetFormat === "arrow") return "arrow";
  if (
    datasetFormat === "archive" ||
    datasetFormat === "webdataset" ||
    datasetFormat === "epub" ||
    datasetFormat === "office"
  ) {
    return "archive";
  }
  if (datasetFormat === "sqlite") return "sqlite";
  if (datasetFormat === "huggingface") return "dataset";
  const extension = extname(path).toLocaleLowerCase();
  if (extension === ".pdf") return "pdf";
  if (extension === ".txt") return "text";
  if ([".md", ".mdx", ".rst"].includes(extension)) return "markdown";
  if ([".json", ".jsonl"].includes(extension)) return "json";
  if (
    [
      ".ts",
      ".tsx",
      ".js",
      ".jsx",
      ".py",
      ".rs",
      ".go",
      ".java",
      ".cs",
      ".cpp",
      ".cc",
      ".c",
      ".h",
      ".hpp",
      ".swift",
      ".kt",
      ".sql",
      ".html",
      ".css",
      ".scss",
      ".yaml",
      ".yml",
      ".toml"
    ].includes(extension)
  ) {
    return "code";
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
      ".heif"
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
      ".wma"
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
      ".flv"
    ].includes(extension)
  ) {
    return "video";
  }
  return "unknown";
}

function workerText(result: WorkerChatResult | undefined): string | undefined {
  if (!result) return undefined;
  for (const candidate of [result.text, result.response, result.content]) {
    if (typeof candidate === "string" && candidate.trim()) return candidate.trim();
  }
  return undefined;
}

function displayToolLabel(toolId: string): string {
  return toolId
    .split(/[.-]/)
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

function privateAddress(address: string): boolean {
  const normalized = address.replace(/^\[|\]$/g, "").toLocaleLowerCase();
  if (normalized === "::1" || normalized === "0:0:0:0:0:0:0:1") return true;
  if (normalized.startsWith("fc") || normalized.startsWith("fd") || /^fe[89ab]/.test(normalized)) {
    return true;
  }
  if (isIP(normalized) === 4) {
    const parts = normalized.split(".").map(Number);
    const first = parts[0] ?? 0;
    const second = parts[1] ?? 0;
    return (
      first === 0 ||
      first === 10 ||
      first === 127 ||
      (first === 169 && second === 254) ||
      (first === 172 && second >= 16 && second <= 31) ||
      (first === 192 && second === 168) ||
      first >= 224
    );
  }
  return false;
}

export async function assertSafeRemoteUrl(url: URL): Promise<void> {
  if (url.username || url.password) throw new Error("URLs containing credentials are not allowed.");
  const hostname = url.hostname.replace(/^\[|\]$/g, "").toLocaleLowerCase();
  const localhost = hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
  const allowLocal = process.env.OMNI_ALLOW_LOCAL_URLS === "1";
  if (url.protocol !== "https:" && !(allowLocal && url.protocol === "http:" && localhost)) {
    throw new Error("Remote URLs require HTTPS.");
  }
  if (localhost) {
    if (!allowLocal) throw new Error("Private or loopback network URLs are not allowed.");
    return;
  }
  const addresses = isIP(hostname)
    ? [{ address: hostname }]
    : await lookup(hostname, { all: true, verbatim: true });
  if (addresses.length === 0 || addresses.some((entry) => privateAddress(entry.address))) {
    throw new Error("Remote URL resolves to a private or reserved network address.");
  }
}

export async function safeFetch(
  initialUrl: URL,
  init: RequestInit,
  maximumRedirects = 5,
  hooks: {
    beforeRequest?: (url: URL) => Promise<void>;
    afterResponse?: (url: URL, response: Response) => Promise<void> | void;
  } = {}
): Promise<Response> {
  let current = initialUrl;
  for (let redirect = 0; redirect <= maximumRedirects; redirect += 1) {
    init.signal?.throwIfAborted();
    await assertSafeRemoteUrl(current);
    await hooks.beforeRequest?.(current);
    init.signal?.throwIfAborted();
    const response = await fetch(current, { ...init, redirect: "manual" });
    await hooks.afterResponse?.(current, response);
    if (![301, 302, 303, 307, 308].includes(response.status)) return response;
    const location = response.headers.get("location");
    if (!location) throw new Error("Remote server returned a redirect without a location.");
    if (redirect === maximumRedirects) throw new Error("Remote URL redirected too many times.");
    await response.body?.cancel("following validated redirect").catch(() => undefined);
    current = new URL(location, current);
  }
  throw new Error("Remote URL redirected too many times.");
}

export async function readResponseBounded(
  response: Response,
  maximumBytes: number
): Promise<Buffer> {
  const declared = Number(response.headers.get("content-length") ?? 0);
  if (Number.isFinite(declared) && declared > maximumBytes) {
    await response.body?.cancel("declared response size exceeds limit").catch(() => undefined);
    throw new Error("Remote content exceeds the allowed size.");
  }
  if (!response.body) return Buffer.alloc(0);
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      total += value.byteLength;
      if (total > maximumBytes) {
        await reader.cancel("response size limit exceeded");
        throw new Error("Remote content exceeds the allowed size.");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  return Buffer.concat(chunks.map((chunk) => Buffer.from(chunk)), total);
}

class CrawlResourcePause extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CrawlResourcePause";
  }
}

class CrawlPolicyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CrawlPolicyError";
  }
}

function configuredMilliseconds(
  name: string,
  fallback: number,
  minimum: number,
  maximum: number
): number {
  const raw = Number(process.env[name]);
  if (!Number.isFinite(raw)) return fallback;
  return Math.max(minimum, Math.min(maximum, Math.round(raw)));
}

function abortableDelay(milliseconds: number, signal?: AbortSignal): Promise<void> {
  if (milliseconds <= 0) return Promise.resolve();
  signal?.throwIfAborted();
  return new Promise((resolveDelay, rejectDelay) => {
    const timer = setTimeout(finish, milliseconds);
    const abort = (): void => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      rejectDelay(signal?.reason ?? new Error("The crawl request was cancelled."));
    };
    function finish(): void {
      signal?.removeEventListener("abort", abort);
      resolveDelay();
    }
    signal?.addEventListener("abort", abort, { once: true });
  });
}

interface DomainPacingState {
  nextRequestAt: number;
  backoffUntil: number;
  failures: number;
  tail: Promise<void>;
}

class DomainRequestScheduler {
  private readonly states = new Map<string, DomainPacingState>();
  private readonly minimumDelay = configuredMilliseconds(
    "OMNI_CRAWL_MIN_DELAY_MS",
    250,
    1,
    60_000
  );
  private readonly baseBackoff = configuredMilliseconds(
    "OMNI_CRAWL_BACKOFF_BASE_MS",
    1_000,
    10,
    60_000
  );

  private state(origin: string): DomainPacingState {
    const existing = this.states.get(origin);
    if (existing) return existing;
    const created: DomainPacingState = {
      nextRequestAt: 0,
      backoffUntil: 0,
      failures: 0,
      tail: Promise.resolve()
    };
    this.states.set(origin, created);
    return created;
  }

  async wait(url: URL, signal?: AbortSignal): Promise<void> {
    const state = this.state(url.origin);
    const predecessor = state.tail;
    let release = (): void => undefined;
    state.tail = new Promise<void>((resolveTail) => {
      release = resolveTail;
    });
    await predecessor;
    try {
      const waitUntil = Math.max(state.nextRequestAt, state.backoffUntil);
      await abortableDelay(Math.max(0, waitUntil - Date.now()), signal);
      state.nextRequestAt = Date.now() + this.minimumDelay;
    } finally {
      release();
    }
  }

  observe(url: URL, response: Response): void {
    const state = this.state(url.origin);
    if (response.status === 429 || response.status === 503) {
      state.failures += 1;
      const retryAfter = response.headers.get("retry-after")?.trim();
      let delay = 0;
      if (retryAfter && /^\d+$/.test(retryAfter)) {
        delay = Number(retryAfter) * 1_000;
      } else if (retryAfter) {
        const parsed = Date.parse(retryAfter);
        if (Number.isFinite(parsed)) delay = Math.max(0, parsed - Date.now());
      }
      if (delay <= 0) {
        delay = this.baseBackoff * 2 ** Math.min(6, state.failures - 1);
      }
      state.backoffUntil =
        Date.now() + Math.min(60_000, Math.max(this.minimumDelay, delay));
      return;
    }
    if (response.status < 500) {
      state.failures = 0;
      state.backoffUntil = 0;
    }
  }
}

function memoryReserveBytes(): number {
  const heapLimit = getHeapStatistics().heap_size_limit;
  return Math.max(
    CRAWL_MEMORY_RESERVE_MINIMUM,
    Math.min(256 * 1024 * 1024, Math.floor(heapLimit * 0.1))
  );
}

function hasTextMemoryHeadroom(additionalBytes: number): boolean {
  const heapLimit = getHeapStatistics().heap_size_limit;
  return (
    heapLimit -
      process.memoryUsage().heapUsed -
      Math.max(0, additionalBytes) * 2 >=
    memoryReserveBytes()
  );
}

async function assertDiskReserve(directory: string, incomingBytes = 0): Promise<void> {
  const filesystem = await statfs(directory);
  const available = Number(filesystem.bavail) * Number(filesystem.bsize);
  const total = Number(filesystem.blocks) * Number(filesystem.bsize);
  const reserve = Math.max(
    CRAWL_DISK_RESERVE_MINIMUM,
    Math.min(4 * 1024 * 1024 * 1024, Math.floor(total * 0.02))
  );
  if (!Number.isFinite(available) || available - incomingBytes < reserve) {
    throw new CrawlResourcePause(
      "Web crawling paused before exhausting the configured disk reserve."
    );
  }
}

async function streamResponseToFile(
  response: Response,
  path: string,
  directory: string,
  onProgress?: () => void
): Promise<number> {
  const declared = Number(response.headers.get("content-length") ?? 0);
  if (Number.isFinite(declared) && declared > 0) {
    try {
      await assertDiskReserve(directory, declared);
    } catch (error) {
      await response.body?.cancel("crawl response exceeds disk reserve").catch(
        () => undefined
      );
      throw error;
    }
  }
  if (!response.body) {
    const empty = await open(path, "wx", 0o600);
    await empty.close();
    return 0;
  }
  const reader = response.body.getReader();
  const output = await open(path, "wx", 0o600);
  let total = 0;
  let checkedAt = 0;
  let failed = false;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value || value.byteLength === 0) continue;
      onProgress?.();
      if (total - checkedAt >= CRAWL_DISK_CHECK_INTERVAL) {
        await assertDiskReserve(directory, value.byteLength);
        checkedAt = total;
      }
      let offset = 0;
      while (offset < value.byteLength) {
        const written = await output.write(
          value,
          offset,
          value.byteLength - offset,
          total + offset
        );
        offset += written.bytesWritten;
      }
      total += value.byteLength;
    }
    await output.sync();
    return total;
  } catch (error) {
    failed = true;
    await reader.cancel("crawl response could not be persisted").catch(() => undefined);
    throw error;
  } finally {
    reader.releaseLock();
    await output.close();
    if (failed) await rm(path, { force: true });
  }
}

async function readSpoolText(path: string, expectedBytes: number): Promise<string> {
  if (!hasTextMemoryHeadroom(expectedBytes)) {
    throw new CrawlResourcePause(
      "Web crawling paused before exhausting the configured memory reserve."
    );
  }
  const decoder = new TextDecoder("utf-8", { fatal: false });
  let text = "";
  let consumed = 0;
  for await (const chunk of createReadStream(path, { highWaterMark: 256 * 1024 })) {
    const bytes = chunk as Buffer;
    consumed += bytes.byteLength;
    if (!hasTextMemoryHeadroom(consumed)) {
      throw new CrawlResourcePause(
        "Web crawling paused before exhausting the configured memory reserve."
      );
    }
    text += decoder.decode(bytes, { stream: true });
  }
  text += decoder.decode();
  return text;
}

function htmlToText(value: string): string {
  return value
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<noscript\b[^>]*>[\s\S]*?<\/noscript>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/\s+/g, " ")
    .trim();
}

function linksFromHtml(value: string, base: URL): URL[] {
  const links: URL[] = [];
  const seen = new Set<string>();
  const pattern =
    /<(?:a|img|audio|video|source)\b[^>]*\b(?:href|src|poster)\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s>]+))/gi;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(value))) {
    const raw = match[1] ?? match[2] ?? match[3];
    if (!raw || raw.startsWith("#")) continue;
    try {
      const url = new URL(raw, base);
      url.hash = "";
      if (
        (url.protocol === "https:" || url.protocol === "http:") &&
        !seen.has(url.toString())
      ) {
        seen.add(url.toString());
        links.push(url);
      }
    } catch {
      // Ignore malformed links from untrusted pages.
    }
  }
  return links;
}

function crawledMediaKind(
  contentType: string,
  url: URL
): "image" | "audio" | "video" | undefined {
  const mime = contentType.split(";", 1)[0]?.trim().toLocaleLowerCase() ?? "";
  if (mime.startsWith("image/") && mime !== "image/svg+xml") return "image";
  if (mime.startsWith("audio/")) return "audio";
  if (mime.startsWith("video/")) return "video";
  const extension = extname(url.pathname).toLocaleLowerCase();
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
      ".heif"
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
      ".wma"
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
      ".flv"
    ].includes(extension)
  ) {
    return "video";
  }
  return undefined;
}

function crawledMediaExtension(
  kind: "image" | "audio" | "video",
  contentType: string,
  url: URL
): string {
  const extension = extname(url.pathname).toLocaleLowerCase();
  if (/^\.[a-z0-9]{1,8}$/.test(extension)) return extension;
  const mime = contentType.split(";", 1)[0]?.trim().toLocaleLowerCase() ?? "";
  const byMime: Record<string, string> = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/avif": ".avif",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov"
  };
  return byMime[mime] ?? (kind === "image" ? ".img" : kind === "audio" ? ".audio" : ".video");
}

function isCrawledText(contentType: string, url: URL): boolean {
  const mime = contentType.split(";", 1)[0]?.trim().toLocaleLowerCase() ?? "";
  if (
    !mime ||
    mime.startsWith("text/") ||
    mime === "application/json" ||
    mime.endsWith("+json") ||
    mime === "application/xml" ||
    mime.endsWith("+xml")
  ) {
    return true;
  }
  return [".html", ".htm", ".txt", ".md", ".json", ".jsonl"].includes(
    extname(url.pathname).toLocaleLowerCase()
  );
}

function robotsDisallows(value: string): string[] {
  const disallowed: string[] = [];
  let applies = false;
  for (const rawLine of value.split(/\r?\n/)) {
    const line = rawLine.replace(/#.*$/, "").trim();
    const separator = line.indexOf(":");
    if (separator < 0) continue;
    const key = line.slice(0, separator).trim().toLocaleLowerCase();
    const entry = line.slice(separator + 1).trim();
    if (key === "user-agent") applies = entry === "*";
    else if (key === "disallow" && applies && entry) disallowed.push(entry);
  }
  return disallowed;
}

export class BrainService {
  readonly datasets: DatasetManifestStore;

  constructor(
    readonly repository: BrainRepository,
    readonly engine: EngineSupervisor
  ) {
    this.datasets = new DatasetManifestStore((brainId) =>
      this.repository.brainDirectory(brainId)
    );
  }

  async create(request: CreateBrainRequest): Promise<BrainDocument> {
    if (request.origin && !["blank", "starter"].includes(request.origin)) {
      throw new Error("Invalid brain origin.");
    }
    if (
      request.hardwareTier &&
      !["micro", "personal", "gpu", "workstation"].includes(request.hardwareTier)
    ) {
      throw new Error("Invalid hardware tier.");
    }
    const modalities = [...new Set(request.modalities ?? [])];
    if (modalities.some((value) => !["vision", "image", "audio", "video"].includes(value))) {
      throw new Error("Invalid modality selection.");
    }
    if ((request.initialToolPermissions?.length ?? 0) > 100) {
      throw new Error("Too many initial tool permission records.");
    }
    for (const permission of request.initialToolPermissions ?? []) {
      normalizeToolId(permission.toolId);
      if (!["off", "ask", "auto", "full"].includes(permission.level)) {
        throw new Error("Invalid initial tool permission level.");
      }
    }
    const resolvedTier = request.hardwareTier ?? "personal";
    const resolvedConfig: BrainConfig = {
      ...request.config,
      // Neural workspace capacity is an architecture decision derived from
      // the hardware tier. Numeric values supplied by old/beta callers are
      // not behavioral controls and cannot shrink a stable-v1 brain.
      workingMemorySlots:
        WORKING_MEMORY_SLOTS_BY_TIER[resolvedTier] *
        (request.config.extendedWorkingMemory ? 2 : 1)
    };
    let brain: BrainDocument;
    // Stable v1 is starter-first at every public entry point. Blank creation
    // remains available only when a caller explicitly requests it.
    const starter = (request.origin ?? "starter") === "starter";
    const starterUrl = request.starterUrl?.trim() ?? "";
    const remoteStarter = starter && Boolean(starterUrl);
    if (remoteStarter) {
      brain = await this.importUrl({ url: starterUrl });
      try {
        const materialized = await stat(
          join(this.repository.brainDirectory(brain.id), "engine", "brain.json")
        );
        if (!materialized.isFile()) throw new Error("not a file");
      } catch {
        await this.repository.remove(brain.id).catch(() => false);
        throw new Error(
          "The selected starter contains no materialized OmniCortex checkpoint."
        );
      }
      const importedWorkingMemorySlots = brain.config.workingMemorySlots;
      const importedExtendedWorkingMemory = brain.config.extendedWorkingMemory;
      brain.config = {
        ...brain.config,
        ...resolvedConfig,
        // Decoder/global-workspace tensor shapes belong to the checkpoint.
        // A hardware choice may size a newly materialized brain, but it must
        // never rewrite a compatible imported checkpoint's recorded shape.
        workingMemorySlots: importedWorkingMemorySlots,
        extendedWorkingMemory: importedExtendedWorkingMemory,
        runtime: "adaptive-core"
      };
      brain.name = brain.config.name;
    } else {
      brain = await this.repository.create(resolvedConfig);
    }
    if (request.initialToolPermissions) {
      const selected = new Map(
        request.initialToolPermissions.map((permission) => [
          normalizeToolId(permission.toolId),
          permission.level
        ])
      );
      brain.toolPermissions = (brain.toolPermissions ?? []).map((permission) => ({
        ...permission,
        level: selected.get(permission.toolId) ?? permission.level,
        updatedAt: new Date().toISOString()
      }));
      for (const [toolId, level] of selected) {
        if (!brain.toolPermissions.some((permission) => permission.toolId === toolId)) {
          brain.toolPermissions.push({
            toolId,
            label: displayToolLabel(toolId),
            level,
            updatedAt: new Date().toISOString()
          });
        }
      }
    }
    brain.journal = [
      ...(brain.journal ?? []),
      {
        id: randomUUID(),
        createdAt: new Date().toISOString(),
        kind: "system",
        summary: `Build profile: ${request.hardwareTier ?? "automatic"}; modalities: ${(request.modalities ?? []).join(", ") || "text"}.`
      }
    ];
    brain = await this.repository.save(brain);
    const storagePath = this.repository.brainDirectory(brain.id);
    if (remoteStarter) {
      // A starter is already a complete safe-tensor checkpoint. Loading it and
      // applying only shape-preserving builder controls keeps its pretrained
      // parameters intact; calling "create" here would silently randomize it.
      await this.engine.tryRequest("unload", { brainId: brain.id }, 30_000);
      await this.engine.tryRequest(
        "load",
        {
          brainId: brain.id,
          config: brain.config,
          storagePath
        },
        300_000
      );
      await this.engine.tryRequest(
        "update_config",
        {
          brainId: brain.id,
          config: brain.config,
          storagePath
        },
        300_000
      );
    } else {
      await this.engine.request(
        "create",
        {
          brainId: brain.id,
          config: brain.config,
          hardwareTier: request.hardwareTier,
          modalities,
          origin: starter ? "starter" : "blank",
          storagePath
        },
        starter ? 300_000 : 60_000
      );
    }
    return brain;
  }

  async querySubstrate(
    brainId: string,
    query: SubstrateQuery = {}
  ): Promise<SubstratePage> {
    const brain = await this.repository.get(brainId);
    const entity = query.entity ?? "overview";
    if (!["overview", "neurons", "assemblies", "synapses"].includes(entity)) {
      throw new Error("Invalid substrate entity.");
    }
    if (
      query.cursor !== undefined &&
      (typeof query.cursor !== "string" || query.cursor.length > 2_048)
    ) {
      throw new Error("Invalid substrate cursor.");
    }
    if (
      query.pageSize !== undefined &&
      (!Number.isSafeInteger(query.pageSize) ||
        query.pageSize < 1 ||
        query.pageSize > 5_000)
    ) {
      throw new Error("Substrate page size must be between 1 and 5,000.");
    }
    if (
      query.zoom !== undefined &&
      (typeof query.zoom !== "number" || !Number.isFinite(query.zoom))
    ) {
      throw new Error("Invalid substrate zoom.");
    }
    const region =
      typeof query.region === "string"
        ? query.region.replace(/\0/g, "").trim()
        : "";
    const search =
      typeof query.search === "string"
        ? query.search.replace(/\0/g, "").trim()
        : "";
    if (region.length > 128 || search.length > 512) {
      throw new Error("Substrate filter is too long.");
    }
    return this.engine.request<SubstratePage>(
      "query_substrate",
      {
        brainId,
        config: brain.config,
        storagePath: this.repository.brainDirectory(brainId),
        query: {
          entity,
          cursor: query.cursor,
          pageSize: query.pageSize ?? 256,
          region,
          search,
          zoom: Math.max(0, Math.min(1, query.zoom ?? 0))
        }
      },
      30_000
    );
  }

  async workspace(brainId: string): Promise<WorkspaceSnapshot> {
    const brain = await this.repository.get(brainId);
    return this.engine.request<WorkspaceSnapshot>(
      "workspace",
      {
        brainId,
        config: brain.config,
        storagePath: this.repository.brainDirectory(brainId)
      },
      30_000
    );
  }

  async chat(
    id: string,
    input: string,
    signal?: AbortSignal,
    onStream?: (event: NeuralChatStreamEvent) => void,
    turnId = randomUUID(),
    responseTokenBudget?: number
  ): Promise<ChatResult> {
    return withBrainWrite(
      this.repository,
      id,
      () => this.chatUnlocked(id, input, signal, onStream, turnId, responseTokenBudget),
      signal
    );
  }

  private async chatUnlocked(
    id: string,
    input: string,
    signal?: AbortSignal,
    onStream?: (event: NeuralChatStreamEvent) => void,
    turnId = randomUUID(),
    responseTokenBudget?: number
  ): Promise<ChatResult> {
    signal?.throwIfAborted();
    if (
      responseTokenBudget !== undefined &&
      (!Number.isSafeInteger(responseTokenBudget) || responseTokenBudget < 1)
    ) {
      throw new Error("Response token budget must be a positive safe integer.");
    }
    const brain = await this.repository.get(id);
    const message = cleanMessage(input);
    const toolSchemas = enabledToolSchemas(brain);
    await this.engine.request(
      "load",
      {
        brainId: id,
        config: brain.config,
        storagePath: this.repository.brainDirectory(id)
      },
      30_000,
      signal
    );
    signal?.throwIfAborted();
    const workerResult = await this.engine.requestStream<WorkerChatResult>(
      "chat",
      {
        brainId: id,
        input: message,
        toolSchemas,
        config: brain.config,
        storagePath: this.repository.brainDirectory(id),
        ...(responseTokenBudget === undefined
          ? {}
          : { maxNewTokens: responseTokenBudget })
      },
      (event) => {
        const normalized = normalizeChatEngineEvent(event, id);
        if (normalized) onStream?.(normalized);
      },
      300_000,
      signal,
      turnId
    );
    signal?.throwIfAborted();
    const generated = workerText(workerResult);
    if (!generated) {
      throw new Error("OmniCortex returned an empty neural response.");
    }
    const result = recordNeuralChat(brain, message, generated);
    if (generated) {
      const runtime = "adaptive-core";
      result.humanMessage.runtime = runtime;
      result.brainMessage.runtime = runtime;
      result.trace.runtime = runtime;
      if (workerResult?.trace?.id) {
        result.trace.id = workerResult.trace.id;
        result.brainMessage.traceId = workerResult.trace.id;
      }
      if (workerResult?.trace?.created_at) {
        result.trace.createdAt = workerResult.trace.created_at;
      }
      if (typeof workerResult?.trace?.seed === "number") {
        result.trace.seed = workerResult.trace.seed;
      }
      if (workerResult?.trace?.steps) {
        result.trace.steps = workerResult.trace.steps
          .filter(
            (step): step is { stage: string; detail: string; value?: string } =>
              typeof step.stage === "string" && typeof step.detail === "string"
          )
          .slice(0, 100);
      }
      const trace = workerResult?.trace;
      if (trace) {
        const mutations = [
          typeof trace.parameter_delta_norm === "number"
            ? `parameter delta ${trace.parameter_delta_norm.toExponential(4)}`
            : undefined,
          typeof trace.stdp_update === "number"
            ? `STDP update ${trace.stdp_update.toExponential(4)}`
            : undefined,
          typeof trace.train_loss === "number" ? `loss ${trace.train_loss.toFixed(6)}` : undefined,
          trace.parameter_checksum_before && trace.parameter_checksum_after
            ? `${trace.parameter_checksum_before.slice(0, 12)} → ${trace.parameter_checksum_after.slice(0, 12)}`
            : undefined
        ].filter(Boolean);
        if (mutations.length > 0) {
          result.trace.steps.push({
            stage: "verified-neural-mutation",
            detail: mutations.join("; ")
          });
        }
        if (typeof trace.ponder_steps === "number") {
          result.trace.branches = Math.max(1, Math.round(trace.ponder_steps));
          result.trace.selectedBranch = result.trace.branches - 1;
        }
      }
      if (workerResult?.trace?.note) result.trace.note = workerResult.trace.note;
      result.proposedActions = parseModelActions("", workerResult?.actions);
      const metrics = workerResult?.metrics;
      if (metrics) {
        if (typeof metrics.plasticityEvents === "number") {
          result.brain.counters.plasticityEvents = metrics.plasticityEvents;
        }
        const counters =
          typeof metrics.counters === "object" && metrics.counters !== null
            ? (metrics.counters as Record<string, unknown>)
            : {};
        if (typeof counters.inference_count === "number") {
          result.brain.counters.inferenceCount = counters.inference_count;
        }
        if (typeof counters.consolidation_cycles === "number") {
          result.brain.counters.consolidationCycles = counters.consolidation_cycles;
        }
      }
    }
    result.proposedActions ??= parseModelActions("", workerResult?.actions);
    result.brain = await this.repository.save(result.brain);
    return result;
  }

  async idleCycle(
    brainId: string,
    minimumIdleSeconds = 45
  ): Promise<IdleCycleResult> {
    return withBrainWrite(this.repository, brainId, () =>
      this.idleCycleUnlocked(brainId, minimumIdleSeconds)
    );
  }

  private async idleCycleUnlocked(
    brainId: string,
    minimumIdleSeconds: number
  ): Promise<IdleCycleResult> {
    const brain = await this.repository.get(brainId);
    if (!brain.config.idleCognition) {
      return {
        brainId,
        ran: false,
        reason: "idle-cognition-disabled",
        actions: []
      };
    }
    if (
      !Number.isFinite(minimumIdleSeconds) ||
      minimumIdleSeconds < 0 ||
      minimumIdleSeconds > 86_400
    ) {
      throw new Error("Invalid idle cognition interval.");
    }
    const worker = await this.engine.tryRequest<WorkerIdleCycleResult>(
      "idle_cycle",
      {
        brainId,
        config: brain.config,
        storagePath: this.repository.brainDirectory(brainId),
        toolSchemas: enabledToolSchemas(brain),
        minimumIdleSeconds
      },
      30_000
    );
    if (!worker) {
      throw new Error("The OmniCortex neural worker is unavailable.");
    }
    const actions: StructuredAction[] = parseModelActions(
      "",
      worker.actions
    ).map((action) => ({ ...action, source: "organic" }));
    if (worker.ran) {
      const spontaneous = actions.find((action) => action.kind === "talk");
      const rawSpontaneous = spontaneous?.arguments.message;
      if (typeof rawSpontaneous === "string") {
        const content = cleanMessage(rawSpontaneous);
        brain.messages.push({
          id: randomUUID(),
          role: "brain",
          content,
          createdAt: new Date().toISOString(),
          traceId: worker.trace?.id,
          runtime: "adaptive-core",
          status: "complete"
        });
        brain.journal = [
          ...(brain.journal ?? []),
          {
            id: randomUUID(),
            createdAt: new Date().toISOString(),
            kind: "reflection",
            summary: "Spoke from prompt-free idle cognition.",
            detail:
              `trace=${worker.trace?.id ?? "none"}; hidden-behavioral-prompt=false`
          }
        ];
      }
      const plasticityEvents = worker.metrics?.plasticityEvents;
      if (typeof plasticityEvents === "number" && Number.isFinite(plasticityEvents)) {
        brain.counters.plasticityEvents = Math.max(
          brain.counters.plasticityEvents,
          Math.round(plasticityEvents)
        );
      }
      await this.repository.save(brain);
    }
    return {
      ...worker,
      brainId,
      ran: worker.ran === true,
      actions
    };
  }

  async feedback(request: FeedbackRequest): Promise<BrainDocument> {
    if (!request || !["up", "down"].includes(request.direction)) {
      throw new Error("Invalid neural feedback direction.");
    }
    return withBrainWrite(this.repository, request.brainId, () =>
      this.feedbackUnlocked(request)
    );
  }

  private async feedbackUnlocked(request: FeedbackRequest): Promise<BrainDocument> {
    if (!request || !["up", "down"].includes(request.direction)) {
      throw new Error("Invalid neural feedback direction.");
    }
    const brain = await this.repository.get(request.brainId);
    const message = brain.messages.find((entry) => entry.id === request.messageId);
    if (!message) throw new Error("The message was not found.");
    if (message.role !== "brain") {
      throw new Error("Feedback can only target a brain response.");
    }
    const worker = await this.engine.tryRequest<WorkerFeedbackResult>(
      "feedback",
      {
        brainId: brain.id,
        config: brain.config,
        storagePath: this.repository.brainDirectory(brain.id),
        messageId: message.id,
        traceId: message.traceId ?? "",
        text: message.content,
        direction: request.direction
      },
      120_000
    );
    if (!worker) {
      throw new Error("The OmniCortex neural worker is unavailable.");
    }
    if (worker.rewardModel !== false || worker.rlhf !== false) {
      throw new Error("The neural worker returned an invalid feedback contract.");
    }
    const plasticityEvents = worker.metrics?.plasticityEvents;
    if (typeof plasticityEvents === "number" && Number.isFinite(plasticityEvents)) {
      brain.counters.plasticityEvents = Math.max(
        brain.counters.plasticityEvents,
        Math.round(plasticityEvents)
      );
    }
    brain.journal = [
      ...(brain.journal ?? []),
      {
        id: randomUUID(),
        createdAt: new Date().toISOString(),
        kind: "learning",
        summary: `Integrated ${request.direction} feedback through neural STDP.`,
        detail:
          `trace=${message.traceId ?? "none"}; ` +
          `synapses=${worker.synapseChecksumBefore?.slice(0, 12) ?? "unknown"}→` +
          `${worker.synapseChecksumAfter?.slice(0, 12) ?? "unknown"}; ` +
          "reward-model=false; rlhf=false"
      }
    ];
    return this.repository.save(brain);
  }

  async consolidate(id: string): Promise<BrainDocument> {
    return withBrainWrite(this.repository, id, () => this.consolidateUnlocked(id));
  }

  private async consolidateUnlocked(id: string): Promise<BrainDocument> {
    const brain = await this.repository.get(id);
    const result = await this.engine.tryRequest(
      "consolidate",
      {
        brainId: id,
        config: brain.config,
        storagePath: this.repository.brainDirectory(id)
      },
      300_000
    );
    if (result === undefined) {
      throw new Error("The OmniCortex neural worker is unavailable.");
    }
    brain.counters.consolidationCycles += 1;
    return this.repository.save(brain);
  }

  async ingestPaths(
    brainId: string,
    paths: string[],
    policy: DataIngestionPolicy = "encode"
  ): Promise<IngestResult[]> {
    await this.repository.get(brainId);
    const manifest = await this.datasets.create(brainId, paths);
    const run = await this.ingestManifest(
      brainId,
      manifest.id,
      policy,
      () => false,
      () => undefined,
      true,
      undefined,
      false
    );
    return run.results;
  }

  async previewDataset(brainId: string, paths: string[]): Promise<DatasetManifest> {
    await this.repository.get(brainId);
    return this.datasets.create(brainId, paths);
  }

  async ingestManifest(
    brainId: string,
    manifestId: string,
    policy: DataIngestionPolicy = "encode",
    cancelled: () => boolean = () => false,
    progress: (value: number, message: string) => void = () => undefined,
    collectResults = true,
    requestedEpochs?: number,
    replayFirstEpoch = true
  ): Promise<{
    manifest: DatasetManifest;
    coverage: TrainingCoverage;
    results: IngestResult[];
    paused: boolean;
  }> {
    const manifest = await this.datasets.manifest(brainId, manifestId);
    const cursor = await this.datasets.cursor(brainId, manifestId);
    const coverage = await this.datasets.coverage(brainId, manifestId);
    const completedEpochs =
      coverage.completedEpochs ??
      (cursor.state === "complete" && coverage.complete ? 1 : 0);
    cursor.currentEpoch = Math.max(
      completedEpochs,
      Math.round(cursor.currentEpoch ?? completedEpochs)
    );
    const targetEpochs = Math.max(
      1,
      Math.round(
        Math.max(
          requestedEpochs ?? 1,
          cursor.requestedEpochs ?? 1,
          coverage.requestedEpochs ?? 1
        )
      )
    );
    cursor.requestedEpochs = targetEpochs;
    coverage.requestedEpochs = targetEpochs;
    coverage.completedEpochs = completedEpochs;
    coverage.discoveredFiles = manifest.discoveredFiles * targetEpochs;
    coverage.discoveredBytes = manifest.discoveredBytes * targetEpochs;
    coverage.complete =
      cursor.currentEpoch >= targetEpochs &&
      hasCompleteTraversal(coverage);
    if (cursor.state === "complete" && coverage.complete) {
      return { manifest, coverage, results: [], paused: false };
    }
    cursor.state = "running";
    if (cursor.nextEntry >= manifest.discoveredFiles) cursor.nextEntry = 0;
    await Promise.all([
      this.datasets.saveCursor(brainId, cursor),
      this.datasets.saveCoverage(brainId, coverage)
    ]);
    const results: IngestResult[] = [];
    while ((cursor.currentEpoch ?? 0) < targetEpochs) {
      for await (const entry of this.datasets.entries(
        brainId,
        manifestId,
        cursor.nextEntry
      )) {
        if (cancelled()) {
          cursor.state = "paused";
          await Promise.all([
            this.datasets.saveCursor(brainId, cursor),
            this.datasets.saveCoverage(brainId, coverage)
          ]);
          return { manifest, coverage, results, paused: true };
        }
        try {
          const result = await this.ingestOnePath(
            brainId,
            entry.path,
            policy,
            cursor.currentEpoch ?? 0,
            replayFirstEpoch
          );
          if (result.coverage && !result.coverage.complete) {
            throw new Error(
              `Worker traversal incomplete for ${entry.path}: ` +
                `${result.coverage.processedRecords} processed + ` +
                `${result.coverage.rejectedRecords} rejected of ` +
                `${result.coverage.discoveredRecords} discovered records`
            );
          }
          result.manifestId = manifestId;
          if (collectResults) results.push(result);
          const workerCoverage = result.coverage;
          coverage.processedFiles += 1;
          coverage.discoveredRecords += workerCoverage?.discoveredRecords ?? 1;
          coverage.processedRecords += workerCoverage?.processedRecords ?? 1;
          coverage.rejectedRecords += workerCoverage?.rejectedRecords ?? 0;
          coverage.shards += workerCoverage?.shards ?? 0;
          for (const [kind, count] of Object.entries(workerCoverage?.modalityCounts ?? {})) {
            if (typeof count !== "number") continue;
            const format = kind as keyof typeof coverage.modalityCounts;
            coverage.modalityCounts[format] =
              (coverage.modalityCounts[format] ?? 0) + count;
          }
          for (const workerError of workerCoverage?.errors ?? []) {
            await this.datasets.recordError(brainId, coverage, {
              source: workerError.source || entry.path,
              message: workerError.message
            });
          }
        } catch (error) {
          if (cancelled()) {
            cursor.state = "paused";
            await Promise.all([
              this.datasets.saveCursor(brainId, cursor),
              this.datasets.saveCoverage(brainId, coverage)
            ]);
            return { manifest, coverage, results, paused: true };
          }
          const message = error instanceof Error ? error.message : String(error);
          const permanentInputFailure =
            (error instanceof Error &&
              "code" in error &&
              (error as NodeJS.ErrnoException).code === "ENOENT") ||
            /is not a regular file|ingestion content hash mismatch/i.test(message);
          if (!permanentInputFailure) {
            // Neural, resource, worker, and training failures are resumable. Do
            // not advance the manifest cursor or misreport them as invalid data.
            cursor.state = "paused";
            await Promise.all([
              this.datasets.saveCursor(brainId, cursor),
              this.datasets.saveCoverage(brainId, coverage)
            ]);
            progress(
              coverage.discoveredFiles === 0
                ? 0
                : cursor.processedFiles / coverage.discoveredFiles,
              `Paused before ${entry.path}: ${message}`
            );
            return { manifest, coverage, results, paused: true };
          }
          coverage.rejectedFiles += 1;
          await this.datasets.recordError(brainId, coverage, {
            source: entry.path,
            message
          });
        }
        coverage.processedBytes += entry.bytes;
        cursor.nextEntry = entry.index + 1;
        cursor.nextRecord = 0;
        cursor.processedFiles = coverage.processedFiles + coverage.rejectedFiles;
        cursor.processedRecords = coverage.processedRecords;
        cursor.processedBytes = coverage.processedBytes;
        await Promise.all([
          this.datasets.saveCursor(brainId, cursor),
          this.datasets.saveCoverage(brainId, coverage)
        ]);
        progress(
          coverage.discoveredFiles === 0
            ? 1
            : cursor.processedFiles / coverage.discoveredFiles,
          `Epoch ${(cursor.currentEpoch ?? 0) + 1}/${targetEpochs}: learned ${
            cursor.processedFiles
          } of ${coverage.discoveredFiles} file visits`
        );
      }
      cursor.currentEpoch = (cursor.currentEpoch ?? 0) + 1;
      coverage.completedEpochs = cursor.currentEpoch;
      cursor.nextEntry = 0;
      cursor.nextRecord = 0;
      await Promise.all([
        this.datasets.saveCursor(brainId, cursor),
        this.datasets.saveCoverage(brainId, coverage)
      ]);
    }
    coverage.complete =
      (cursor.currentEpoch ?? 0) >= targetEpochs &&
      hasCompleteTraversal(coverage);
    cursor.state = coverage.complete ? "complete" : "failed";
    await Promise.all([
      this.datasets.saveCursor(brainId, cursor),
      this.datasets.saveCoverage(brainId, coverage)
    ]);
    return { manifest, coverage, results, paused: false };
  }

  private async ingestOnePath(
    brainId: string,
    path: string,
    policy: DataIngestionPolicy,
    epoch = 0,
    forceReplay = false
  ): Promise<IngestResult> {
    return withBrainWrite(this.repository, brainId, () =>
      this.ingestOnePathUnlocked(brainId, path, policy, epoch, forceReplay)
    );
  }

  private async ingestOnePathUnlocked(
    brainId: string,
    path: string,
    policy: DataIngestionPolicy,
    epoch: number,
    forceReplay: boolean
  ): Promise<IngestResult> {
    let brain = await this.repository.get(brainId);
    const fileInfo = await stat(path);
    if (!fileInfo.isFile()) throw new Error(`${path} is not a regular file.`);
    const contentHash = await hashFile(path);
    const duplicate = brain.trainingSources.find((source) => source.contentHash === contentHash);
    if (duplicate && !forceReplay && epoch === 0) {
      return {
        brain,
        source: duplicate,
        warnings: [`${basename(path)} was already encoded; no duplicate synapses were created.`]
      };
    }
    const kind = trainingKind(path);
    const warnings: string[] = [];
    const beforeConcepts = Object.keys(brain.concepts).length;
    const beforeSynapses = Object.keys(brain.synapses).length;
    const beforeIdeas = brain.ideas.length;
    const retainRaw =
      (brain.config.memoryRecipe ?? "human-consolidation") === "total-recall" &&
      brain.config.retainSourceText;
    const worker = await this.engine.request<WorkerIngestResult>(
      "ingest",
      {
        brainId,
        path,
        kind,
        policy,
        contentHash,
        allowReplay: forceReplay || epoch > 0,
        epoch,
        config: brain.config,
        storagePath: this.repository.brainDirectory(brainId)
      },
      86_400_000
    );
    const rawCoverage = worker?.coverage ?? worker?.source?.coverage;
    const resultCoverage = rawCoverage
      ? normalizeWorkerTrainingCoverage(rawCoverage, fileInfo.size)
      : undefined;
    if (resultCoverage && !resultCoverage.complete) {
      throw new Error(
        `Worker dataset traversal incomplete for ${basename(path)}: ` +
          `${resultCoverage.processedFiles + resultCoverage.rejectedFiles}/` +
          `${resultCoverage.discoveredFiles} files and ` +
          `${resultCoverage.processedRecords + resultCoverage.rejectedRecords}/` +
          `${resultCoverage.discoveredRecords} records were classified`
      );
    }
    warnings.push(...(worker?.warnings ?? []), ...(worker?.source?.warnings ?? []));
    const effectiveKind =
      typeof worker?.source?.kind === "string" &&
      [
        "pdf",
        "text",
        "markdown",
        "code",
        "json",
        "csv",
        "parquet",
        "arrow",
        "archive",
        "sqlite",
        "dataset",
        "image",
        "audio",
        "video",
        "unknown"
      ].includes(worker.source.kind)
        ? (worker.source.kind as TrainingSource["kind"])
        : kind;
    const memoryRecipe = brain.config.memoryRecipe ?? "human-consolidation";
    const preserveBlob =
      memoryRecipe !== "synapses-only" &&
      (policy === "archive" ||
        memoryRecipe === "total-recall" ||
        effectiveKind === "image" ||
        effectiveKind === "audio" ||
        effectiveKind === "video");
    const source: TrainingSource = {
      id: duplicate?.id ?? randomUUID(),
      name: basename(path),
      path,
      kind: effectiveKind,
      bytes: fileInfo.size,
      learnedIdeas:
        worker?.source?.learned_ideas ?? brain.ideas.length - beforeIdeas,
      learnedConcepts:
        worker?.source?.learned_concepts ??
        Object.keys(brain.concepts).length - beforeConcepts,
      learnedSynapses:
        worker?.source?.plasticity_events ??
        Object.keys(brain.synapses).length - beforeSynapses,
      importedAt: new Date().toISOString(),
      rawTextRetained: retainRaw && preserveBlob,
      contentHash,
      blobHash: preserveBlob ? await this.repository.storeFileAsBlob(path) : undefined,
      policy,
      license: "User-provided source; license not declared"
    };
    if (duplicate) {
      brain.trainingSources = brain.trainingSources.map((record) =>
        record.id === duplicate.id ? source : record
      );
    } else {
      brain.trainingSources.push(source);
    }
    brain.journal = [
      ...(brain.journal ?? []),
      {
        id: randomUUID(),
        createdAt: source.importedAt,
        kind: "learning",
        summary: `${policy === "archive" ? "Archived" : "Learned from"} ${source.name}.`,
        detail: `${source.learnedIdeas} ideas, ${source.learnedConcepts} concepts, ${source.learnedSynapses} synapses`
      }
    ];
    brain = await this.repository.save(brain);
    return { brain, source, warnings, coverage: resultCoverage };
  }

  async ingestWeb(request: IngestWebRequest): Promise<IngestResult> {
    const url = new URL(request.url);
    const response = await safeFetch(url, {
      signal: AbortSignal.timeout(120_000),
      headers: {
        Accept: "text/html, text/plain, application/json;q=0.9",
        "User-Agent": CRAWL_USER_AGENT
      }
    });
    if (!response.ok) throw new Error(`Web ingestion failed with HTTP ${response.status}.`);
    const finalUrl = new URL(response.url);
    await assertSafeRemoteUrl(finalUrl);
    const spoolDirectory = join(
      this.repository.brainDirectory(request.brainId),
      "datasets",
      "web-spool"
    );
    await mkdir(spoolDirectory, { recursive: true });
    const contentType = response.headers.get("content-type") ?? "";
    const mediaKind = crawledMediaKind(contentType, finalUrl);
    const spoolPath = join(
      spoolDirectory,
      `${randomUUID()}${
        mediaKind
          ? crawledMediaExtension(mediaKind, contentType, finalUrl)
          : ".response"
      }`
    );
    try {
      const bytes = await streamResponseToFile(response, spoolPath, spoolDirectory);
      if (mediaKind) {
        return this.ingestCrawledMedia(
          request,
          finalUrl,
          spoolPath,
          contentType,
          mediaKind
        );
      }
      if (!isCrawledText(contentType, finalUrl)) {
        throw new Error(
          `Unsupported web content type ${contentType || "(missing)"}`
        );
      }
      const raw = await readSpoolText(spoolPath, bytes);
      return this.ingestWebContent(request, finalUrl, raw, contentType);
    } finally {
      await rm(spoolPath, { force: true });
    }
  }

  private async ingestWebContent(
    request: IngestWebRequest,
    finalUrl: URL,
    raw: string,
    contentType: string
  ): Promise<IngestResult> {
    return withBrainWrite(this.repository, request.brainId, () =>
      this.ingestWebContentUnlocked(request, finalUrl, raw, contentType)
    );
  }

  private async ingestWebContentUnlocked(
    request: IngestWebRequest,
    finalUrl: URL,
    raw: string,
    contentType: string
  ): Promise<IngestResult> {
    const text = contentType.includes("html") ? htmlToText(raw) : raw.replace(/\0/g, "");
    const contentHash = sha256(text);
    let brain = await this.repository.get(request.brainId);
    const duplicate = brain.trainingSources.find((source) => source.contentHash === contentHash);
    if (duplicate) {
      return {
        brain,
        source: duplicate,
        warnings: ["This web content was already ingested; no duplicate synapses were created."]
      };
    }
    const policy = request.policy ?? "encode";
    const quarantined = request.quarantine ?? true;
    const before = {
      ideas: brain.ideas.length,
      concepts: Object.keys(brain.concepts).length,
      synapses: Object.keys(brain.synapses).length
    };
    const webCache = join(this.repository.brainDirectory(brain.id), "datasets", "web-cache");
    const temporaryPath = join(webCache, `${contentHash}.txt`);
    await mkdir(webCache, { recursive: true });
    await writeFile(temporaryPath, text, { encoding: "utf8", mode: 0o600 });
    let worker: WorkerIngestResult | undefined;
    try {
      worker = await this.engine.tryRequest<WorkerIngestResult>(
        "ingest",
        {
          brainId: brain.id,
          url: finalUrl.toString(),
          path: temporaryPath,
          name: finalUrl.toString(),
          kind: "text",
          policy: quarantined ? "archive" : policy,
          quarantine: quarantined,
          contentHash,
          storagePath: this.repository.brainDirectory(brain.id)
        },
        86_400_000
      );
    } finally {
      await rm(temporaryPath, { force: true });
    }
    if (worker === undefined) {
      throw new Error(
        "The OmniCortex neural worker is unavailable; web data was not learned."
      );
    }
    const retainRaw =
      !quarantined &&
      (brain.config.memoryRecipe ?? "human-consolidation") === "total-recall" &&
      brain.config.retainSourceText;
    const source: TrainingSource = {
      id: randomUUID(),
      name: finalUrl.hostname + finalUrl.pathname,
      kind: "text",
      bytes: Buffer.byteLength(raw),
      learnedIdeas: worker?.source?.learned_ideas ?? brain.ideas.length - before.ideas,
      learnedConcepts:
        worker?.source?.learned_concepts ??
        Object.keys(brain.concepts).length - before.concepts,
      learnedSynapses:
        worker?.source?.plasticity_events ??
        Object.keys(brain.synapses).length - before.synapses,
      importedAt: new Date().toISOString(),
      rawTextRetained: retainRaw,
      rawText: retainRaw ? text : undefined,
      contentHash,
      blobHash:
        (brain.config.memoryRecipe ?? "human-consolidation") !== "synapses-only" &&
        (quarantined || (brain.config.memoryRecipe ?? "human-consolidation") === "total-recall")
          ? await this.repository.storeBlob(Buffer.from(raw, "utf8"))
          : undefined,
      policy: quarantined ? "archive" : policy,
      provenanceUrl: finalUrl.toString(),
      license: "Web source; verify the publisher's terms",
      licenseUrl: finalUrl.toString()
    };
    brain.trainingSources.push(source);
    brain = await this.repository.save(brain);
    return {
      brain,
      source,
      warnings: quarantined
        ? ["The downloaded source is quarantined and has not changed neural parameters."]
        : []
    };
  }

  private async ingestCrawledMedia(
    request: IngestWebRequest,
    finalUrl: URL,
    path: string,
    contentType: string,
    kind: "image" | "audio" | "video"
  ): Promise<IngestResult> {
    return withBrainWrite(this.repository, request.brainId, () =>
      this.ingestCrawledMediaUnlocked(request, finalUrl, path, contentType, kind)
    );
  }

  private async ingestCrawledMediaUnlocked(
    request: IngestWebRequest,
    finalUrl: URL,
    path: string,
    contentType: string,
    kind: "image" | "audio" | "video"
  ): Promise<IngestResult> {
    const contentHash = await hashFile(path);
    let brain = await this.repository.get(request.brainId);
    const duplicate = brain.trainingSources.find(
      (source) => source.contentHash === contentHash
    );
    if (duplicate) {
      return {
        brain,
        source: duplicate,
        warnings: [
          "This crawled media was already learned; no duplicate synapses were created."
        ]
      };
    }
    const policy = request.policy ?? "encode";
    const quarantined = request.quarantine ?? true;
    const fileInfo = await stat(path);
    const before = {
      ideas: brain.ideas.length,
      concepts: Object.keys(brain.concepts).length,
      synapses: Object.keys(brain.synapses).length
    };
    const worker = await this.engine.request<WorkerIngestResult>(
      "ingest",
      {
        brainId: brain.id,
        url: finalUrl.toString(),
        path,
        name: finalUrl.toString(),
        kind,
        policy: quarantined ? "archive" : policy,
        quarantine: quarantined,
        contentHash,
        storagePath: this.repository.brainDirectory(brain.id)
      },
      86_400_000
    );
    const effectiveKind =
      worker?.source?.kind === "image" ||
      worker?.source?.kind === "audio" ||
      worker?.source?.kind === "video"
        ? worker.source.kind
        : kind;
    const memoryRecipe = brain.config.memoryRecipe ?? "human-consolidation";
    const preserveBlob = memoryRecipe !== "synapses-only";
    const source: TrainingSource = {
      id: randomUUID(),
      name: basename(finalUrl.pathname) || `${finalUrl.hostname}-${kind}`,
      kind: effectiveKind,
      bytes: fileInfo.size,
      learnedIdeas:
        worker?.source?.learned_ideas ?? brain.ideas.length - before.ideas,
      learnedConcepts:
        worker?.source?.learned_concepts ??
        Object.keys(brain.concepts).length - before.concepts,
      learnedSynapses:
        worker?.source?.plasticity_events ??
        Object.keys(brain.synapses).length - before.synapses,
      importedAt: new Date().toISOString(),
      rawTextRetained: false,
      contentHash,
      blobHash: preserveBlob
        ? await this.repository.storeFileAsBlob(path)
        : undefined,
      policy: quarantined ? "archive" : policy,
      provenanceUrl: finalUrl.toString(),
      license: "Web media source; verify the publisher's terms",
      licenseUrl: finalUrl.toString()
    };
    brain.trainingSources.push(source);
    brain.journal = [
      ...(brain.journal ?? []),
      {
        id: randomUUID(),
        createdAt: source.importedAt,
        kind: "learning",
        summary: quarantined
          ? `Quarantined crawled ${effectiveKind} ${source.name}.`
          : `Learned crawled ${effectiveKind} ${source.name}.`,
        detail:
          `${contentType || "unknown content type"}; ` +
          `${source.learnedIdeas} ideas, ${source.learnedConcepts} concepts, ` +
          `${source.learnedSynapses} synapses`
      }
    ];
    brain = await this.repository.save(brain);
    return {
      brain,
      source,
      warnings: [
        ...(worker?.warnings ?? []),
        ...(worker?.source?.warnings ?? []),
        ...(quarantined
          ? ["The crawled media is quarantined and has not changed neural parameters."]
          : [])
      ]
    };
  }

  async crawlWeb(
    request: WebCrawlRequest,
    cancelled: () => boolean = () => false,
    progress: (value: number, message: string) => void = () => undefined
  ): Promise<WebCrawlResult> {
    const start = new URL(request.url);
    await assertSafeRemoteUrl(start);
    const maximumPages =
      request.maxPages === undefined
        ? Number.POSITIVE_INFINITY
        : Math.max(1, Math.round(request.maxPages));
    const maximumDepth =
      request.maxDepth === undefined
        ? Number.POSITIVE_INFINITY
        : Math.max(0, Math.round(request.maxDepth));
    const sameOrigin =
      request.followExternalLinks === true ? false : (request.sameOrigin ?? true);
    const respectRobots = request.respectRobots ?? true;
    const concurrency = Math.max(
      1,
      Math.min(
        32,
        Math.round(request.concurrency ?? availableParallelism())
      )
    );
    const frontier = await CrawlFrontierStore.create(
      this.repository.brainDirectory(request.brainId),
      request.brainId,
      start.toString(),
      request.crawlId,
      request.resume ?? true
    );
    const scheduler = new DomainRequestScheduler();
    const robotsByOrigin = new Map<string, Promise<string[]>>();
    const rulesFor = async (url: URL): Promise<string[]> => {
      if (!respectRobots) return [];
      const existing = robotsByOrigin.get(url.origin);
      if (existing) return existing;
      const pending = (async (): Promise<string[]> => {
        let rules: string[] = [];
        try {
          const robotsUrl = new URL("/robots.txt", url.origin);
          const signal = AbortSignal.timeout(30_000);
          const robotsResponse = await safeFetch(
            robotsUrl,
            {
              signal,
              headers: {
                Accept: "text/plain",
                "User-Agent": CRAWL_USER_AGENT
              }
            },
            5,
            {
              beforeRequest: async (current) => {
                if (current.origin !== url.origin) {
                  throw new CrawlPolicyError(
                    "robots.txt redirected outside its protected origin"
                  );
                }
                await scheduler.wait(current, signal);
              },
              afterResponse: (current, response) => {
                scheduler.observe(current, response);
              }
            }
          );
          if (robotsResponse.ok) {
            rules = robotsDisallows(
              (await readResponseBounded(robotsResponse, MAX_ROBOTS_BYTES)).toString("utf8")
            );
          } else {
            await robotsResponse.body?.cancel("robots response was not successful").catch(
              () => undefined
            );
          }
        } catch (error) {
          frontier.recordWarning(
            `${url.origin}/robots.txt`,
            `robots.txt could not be read: ${
              error instanceof Error ? error.message : String(error)
            }`
          );
        }
        return rules;
      })();
      robotsByOrigin.set(url.origin, pending);
      return pending;
    };
    const isDisallowed = (url: URL, rules: readonly string[]): boolean =>
      rules.some(
        (prefix) =>
          prefix === "/" ||
          (prefix.length > 1 && `${url.pathname}${url.search}`.startsWith(prefix))
      );
    const assertCrawlPolicy = async (url: URL): Promise<void> => {
      if (sameOrigin && url.origin !== start.origin) {
        throw new CrawlPolicyError(
          `Redirect target ${url.origin} is outside the same-site crawl origin.`
        );
      }
      const disallowed = await rulesFor(url);
      if (isDisallowed(url, disallowed)) {
        throw new CrawlPolicyError("robots.txt disallows this path");
      }
    };
    const spoolDirectory = join(
      this.repository.brainDirectory(request.brainId),
      "datasets",
      "web-spool"
    );
    await mkdir(spoolDirectory, { recursive: true });
    const recentResults: IngestResult[] = [];
    let stopped = false;
    try {
      while (true) {
        const before = frontier.counts();
        if (cancelled()) {
          stopped = true;
          break;
        }
        if (before.visited >= maximumPages || before.queued === 0) break;
        const batch = frontier.next(
          Math.min(concurrency, Math.max(1, maximumPages - before.visited))
        );
        if (batch.length === 0) break;
        const fetched = await Promise.all(
          batch.map(async (entry) => {
            const pageUrl = new URL(entry.url);
            try {
              await assertSafeRemoteUrl(pageUrl);
              let response: Response | undefined;
              for (let attempt = 0; attempt < 3; attempt += 1) {
                const signal = AbortSignal.timeout(120_000);
                response = await safeFetch(
                  pageUrl,
                  {
                    signal,
                    headers: {
                      Accept:
                        "text/html, text/plain, application/json;q=0.9, " +
                        "image/*;q=0.8, audio/*;q=0.8, video/*;q=0.8",
                      "User-Agent": CRAWL_USER_AGENT
                    }
                  },
                  5,
                  {
                    beforeRequest: async (current) => {
                      await assertCrawlPolicy(current);
                      await scheduler.wait(current, signal);
                    },
                    afterResponse: (current, currentResponse) => {
                      scheduler.observe(current, currentResponse);
                    }
                  }
                );
                if (
                  attempt < 2 &&
                  [429, 502, 503, 504].includes(response.status)
                ) {
                  await response.body?.cancel("retrying transient crawl response").catch(
                    () => undefined
                  );
                  response = undefined;
                  continue;
                }
                break;
              }
              if (!response) throw new Error("Web crawl retries were exhausted.");
              if (!response.ok) {
                await response.body?.cancel("crawl response was not successful").catch(
                  () => undefined
                );
                throw new Error(`HTTP ${response.status}`);
              }
              const finalUrl = new URL(response.url);
              await assertSafeRemoteUrl(finalUrl);
              await assertCrawlPolicy(finalUrl);
              const contentType = response.headers.get("content-type") ?? "";
              const mediaKind = crawledMediaKind(contentType, finalUrl);
              const bodyPath = join(
                spoolDirectory,
                `${randomUUID()}${
                  mediaKind
                    ? crawledMediaExtension(mediaKind, contentType, finalUrl)
                    : ".response"
                }`
              );
              const bytes = await streamResponseToFile(
                response,
                bodyPath,
                spoolDirectory
              );
              return {
                entry,
                finalUrl,
                contentType,
                mediaKind,
                bodyPath,
                bytes
              };
            } catch (error) {
              return {
                entry,
                error: error instanceof Error ? error.message : String(error),
                skipped: error instanceof CrawlPolicyError,
                resourcePaused: error instanceof CrawlResourcePause
              };
            }
          })
        );
        const retryAndClean = async (
          pending: typeof fetched
        ): Promise<void> => {
          await Promise.all(
            pending.map(async (remaining) => {
              frontier.retry(remaining.entry.url);
              if ("bodyPath" in remaining && typeof remaining.bodyPath === "string") {
                await rm(remaining.bodyPath, { force: true });
              }
            })
          );
        };
        for (let index = 0; index < fetched.length; index += 1) {
          const page = fetched[index]!;
          if (cancelled()) {
            stopped = true;
            await retryAndClean(fetched.slice(index));
            break;
          }
          if ("error" in page) {
            if (page.resourcePaused) {
              stopped = true;
              frontier.recordWarning(page.entry.url, page.error);
              await retryAndClean(fetched.slice(index));
              break;
            }
            frontier.skipped(page.entry.url, page.error);
            continue;
          }
          try {
            const ingestRequest = {
              brainId: request.brainId,
              url: page.finalUrl.toString(),
              policy: request.policy,
              quarantine: request.quarantine ?? true
            };
            let raw: string | undefined;
            const result = page.mediaKind
              ? await this.ingestCrawledMedia(
                  ingestRequest,
                  page.finalUrl,
                  page.bodyPath,
                  page.contentType,
                  page.mediaKind
                )
              : await (async (): Promise<IngestResult> => {
                  if (!isCrawledText(page.contentType, page.finalUrl)) {
                    throw new Error(
                      `Unsupported crawled content type ${
                        page.contentType || "(missing)"
                      }`
                    );
                  }
                  raw = await readSpoolText(page.bodyPath, page.bytes);
                  return this.ingestWebContent(
                    ingestRequest,
                    page.finalUrl,
                    raw,
                    page.contentType
                  );
                })();
            if (recentResults.length >= CRAWL_DIAGNOSTIC_WINDOW) {
              recentResults.shift();
            }
            recentResults.push(result);
            if (
              raw !== undefined &&
              page.entry.depth < maximumDepth &&
              page.contentType.includes("html")
            ) {
              frontier.enqueue(
                linksFromHtml(raw, page.finalUrl)
                  .filter((link) => !sameOrigin || link.origin === start.origin)
                  .map((link) => ({
                    url: link.toString(),
                    depth: page.entry.depth + 1
                  }))
              );
            }
            const receiptKind =
              result.source.kind === "image" ||
              result.source.kind === "audio" ||
              result.source.kind === "video"
                ? result.source.kind
                : "text";
            frontier.visited(page.entry.url, result.source.bytes, {
              sourceId: result.source.id,
              sourceName: result.source.name,
              kind: receiptKind,
              bytes: result.source.bytes,
              contentHash: result.source.contentHash,
              learnedIdeas: result.source.learnedIdeas,
              learnedConcepts: result.source.learnedConcepts,
              learnedSynapses: result.source.learnedSynapses,
              warnings: result.warnings
            });
          } catch (error) {
            if (error instanceof CrawlResourcePause) {
              stopped = true;
              frontier.recordWarning(page.entry.url, error.message);
              await retryAndClean(fetched.slice(index));
              break;
            }
            frontier.skipped(
              page.entry.url,
              error instanceof Error ? error.message : String(error)
            );
          } finally {
            await rm(page.bodyPath, { force: true });
          }
        }
        const counts = frontier.counts();
        const denominator = Number.isFinite(maximumPages)
          ? maximumPages
          : Math.max(1, counts.visited + counts.queued);
        progress(
          Math.min(0.99, counts.visited / denominator),
          `Crawled ${counts.visited} pages; ${counts.queued} queued`
        );
        if (stopped) break;
      }
      const counts = frontier.counts();
      const warnings = frontier.warnings(CRAWL_DIAGNOSTIC_WINDOW);
      const coverage: TrainingCoverage = {
        schemaVersion: 1,
        manifestId: frontier.id,
        discoveredFiles: counts.visited + counts.skipped + counts.queued,
        processedFiles: counts.visited,
        rejectedFiles: counts.skipped,
        discoveredRecords: counts.visited + counts.skipped,
        processedRecords: counts.visited,
        rejectedRecords: counts.skipped,
        discoveredBytes: counts.processedBytes,
        processedBytes: counts.processedBytes,
        shards: 0,
        modalityCounts: counts.modalityCounts,
        errors: warnings.map((message) => ({ source: start.toString(), message })),
        errorLog: join("datasets", "crawls", `${frontier.id}.sqlite3`),
        complete:
          !stopped && (counts.queued === 0 || counts.visited >= maximumPages),
        updatedAt: new Date().toISOString()
      };
      return {
        crawlId: frontier.id,
        startUrl: start.toString(),
        visited: counts.visited,
        skipped: counts.skipped,
        results: recentResults,
        resultCount: counts.resultCount,
        resultsTruncated: counts.resultCount > recentResults.length,
        resultLog: join("datasets", "crawls", `${frontier.id}.sqlite3`),
        warnings,
        warningCount: counts.warningCount,
        warningsTruncated: counts.warningCount > warnings.length,
        frontierRemaining: counts.queued,
        stopped,
        coverage
      };
    } finally {
      frontier.close();
    }
  }

  async importUrl(request: ImportUrlRequest): Promise<BrainDocument> {
    const downloaded = await this.downloadCatalogArtifactToFile(request);
    try {
      return await this.repository.importBundle(downloaded.path);
    } finally {
      await rm(downloaded.temporaryDirectory, { recursive: true, force: true });
    }
  }

  private async downloadCatalogArtifactToFile(
    request: ImportUrlRequest
  ): Promise<{ path: string; finalUrl: URL; temporaryDirectory: string }> {
    const url = new URL(request.url);
    const expected = request.expectedSha256?.trim().toLocaleLowerCase();
    if (expected && !/^[a-f0-9]{64}$/.test(expected)) {
      throw new Error("Expected SHA-256 must be 64 hexadecimal characters.");
    }
    await this.repository.initialize();
    const controller = new AbortController();
    let idleTimer: ReturnType<typeof setTimeout> | undefined;
    const refreshIdleTimeout = (): void => {
      if (idleTimer) clearTimeout(idleTimer);
      idleTimer = setTimeout(
        () => controller.abort(new Error("Catalog download stalled.")),
        120_000
      );
      idleTimer.unref?.();
    };
    refreshIdleTimeout();
    try {
      const response = await safeFetch(url, {
        signal: controller.signal,
        headers: { Accept: "application/octet-stream, application/zip;q=0.9" }
      });
      if (!response.ok) throw new Error(`Catalog download failed with HTTP ${response.status}.`);
      const finalUrl = new URL(response.url);
      await assertSafeRemoteUrl(finalUrl);
      const temporaryDirectory = await mkdtemp(
        join(this.repository.root, ".catalog-download-")
      );
      const path = join(temporaryDirectory, "checkpoint.omni");
      try {
        await streamResponseToFile(
          response,
          path,
          temporaryDirectory,
          refreshIdleTimeout
        );
        const actual = await hashFile(path);
        if (expected && actual !== expected) {
          throw new Error("Catalog bundle checksum verification failed.");
        }
        return { path, finalUrl, temporaryDirectory };
      } catch (error) {
        await rm(temporaryDirectory, { recursive: true, force: true });
        throw error;
      }
    } finally {
      if (idleTimer) clearTimeout(idleTimer);
    }
  }

  private async downloadCatalogArtifact(
    request: ImportUrlRequest
  ): Promise<{ data: Buffer; finalUrl: URL }> {
    const url = new URL(request.url);
    const expected = request.expectedSha256?.trim().toLocaleLowerCase();
    if (expected && !/^[a-f0-9]{64}$/.test(expected)) {
      throw new Error("Expected SHA-256 must be 64 hexadecimal characters.");
    }
    const response = await safeFetch(url, {
      signal: AbortSignal.timeout(120_000),
      headers: { Accept: "application/json, application/octet-stream;q=0.9" }
    });
    if (!response.ok) throw new Error(`Catalog download failed with HTTP ${response.status}.`);
    const finalUrl = new URL(response.url);
    await assertSafeRemoteUrl(finalUrl);
    const data = await readResponseBounded(response, MAX_DOWNLOAD_BYTES);
    const actual = sha256(data);
    if (expected && actual !== expected) throw new Error("Catalog bundle checksum verification failed.");
    return { data, finalUrl };
  }

  async loadRecipeBuffer(contents: Buffer, sourceLabel: string): Promise<BuildRecipe> {
    return validateBuildRecipe(contents, sourceLabel);
  }

  async loadRecipeUrl(request: ImportUrlRequest): Promise<BuildRecipe> {
    const { data, finalUrl } = await this.downloadCatalogArtifact(request);
    return this.loadRecipeBuffer(data, finalUrl.toString());
  }

  async installModalityPackBuffer(
    brainId: string,
    contents: Buffer,
    sourceLabel: string
  ): Promise<InstalledModalityPack> {
    await this.repository.get(brainId);
    const directory = this.repository.brainDirectory(brainId);
    const staged = await stageModalityPack(directory, contents, sourceLabel);
    try {
      await this.engine.request<Record<string, unknown>>(
        "install_modality_pack",
        {
          brainId,
          storagePath: directory,
          packPath: staged.packPath,
          manifest: staged.manifest
        },
        300_000
      );
      await recordInstalledPack(directory, staged.result);
      return staged.result;
    } catch (error) {
      await removeStagedPack(staged.stagingDirectory);
      throw error;
    }
  }

  async installModalityPackUrl(
    brainId: string,
    request: ImportUrlRequest
  ): Promise<InstalledModalityPack> {
    const { data, finalUrl } = await this.downloadCatalogArtifact(request);
    return this.installModalityPackBuffer(
      brainId,
      data,
      basename(finalUrl.pathname) || "modality.omnipack"
    );
  }

  async listModalityPacks(brainId: string): Promise<InstalledModalityPack[]> {
    await this.repository.get(brainId);
    return listInstalledPacks(this.repository.brainDirectory(brainId));
  }

  async listToolPermissions(brainId: string): Promise<ToolPermissionRecord[]> {
    const brain = await this.repository.get(brainId);
    return [...(brain.toolPermissions ?? [])];
  }

  async setToolPermission(
    brainId: string,
    toolId: string,
    level: ToolPermissionLevel
  ): Promise<ToolPermissionRecord[]> {
    return withBrainWrite(this.repository, brainId, () =>
      this.setToolPermissionUnlocked(brainId, toolId, level)
    );
  }

  private async setToolPermissionUnlocked(
    brainId: string,
    toolId: string,
    level: ToolPermissionLevel
  ): Promise<ToolPermissionRecord[]> {
    if (!["off", "ask", "auto", "full"].includes(level)) throw new Error("Invalid permission level.");
    const brain = await this.repository.get(brainId);
    const id = normalizeToolId(toolId);
    const records = brain.toolPermissions ?? [];
    const record = records.find((entry) => entry.toolId === id);
    const updatedAt = new Date().toISOString();
    if (record) {
      record.level = level;
      record.updatedAt = updatedAt;
    } else {
      records.push({ toolId: id, label: displayToolLabel(id), level, updatedAt });
    }
    brain.toolPermissions = records.sort((left, right) => left.label.localeCompare(right.label));
    brain.journal = [
      ...(brain.journal ?? []),
      {
        id: randomUUID(),
        createdAt: updatedAt,
        kind: "tool",
        summary: `${displayToolLabel(id)} permission changed to ${level}.`
      }
    ];
    const saved = await this.repository.save(brain);
    return saved.toolPermissions ?? [];
  }

  private async planMergeFile(
    target: BrainDocument,
    candidate: Omit<MergeFileCandidate, "destinationPath" | "disposition">,
    conflicts: string[]
  ): Promise<MergeFileCandidate | undefined> {
    const destinationRoot =
      candidate.kind === "evidence" ? "evidence" : join("artifacts", "merged");
    const sourceKey = sha256(candidate.sourcePath).slice(0, 12);
    const destinationPath = portableRelative(
      join(destinationRoot, candidate.sha256, `${sourceKey}-${safeMergeName(basename(candidate.sourcePath))}`)
    );
    const absoluteDestination = join(
      this.repository.brainDirectory(target.id),
      ...destinationPath.split("/")
    );
    let disposition: MergeFileCandidate["disposition"] = "copy";
    try {
      const existing = await lstat(absoluteDestination);
      if (
        existing.isSymbolicLink() ||
        !existing.isFile() ||
        existing.size !== candidate.bytes ||
        sha256(await readFile(absoluteDestination)) !== candidate.sha256
      ) {
        conflicts.push(
          `Target file ${destinationPath} does not match its content address and was not overwritten.`
        );
        return undefined;
      }
      disposition = "duplicate";
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    return { ...candidate, destinationPath, disposition };
  }

  private async buildMergePlan(
    sourceBrainId: string,
    targetBrainId: string
  ): Promise<MergePlan> {
    if (sourceBrainId === targetBrainId) throw new Error("A brain cannot merge into itself.");
    const [source, target] = await Promise.all([
      this.repository.get(sourceBrainId),
      this.repository.get(targetBrainId)
    ]);
    const substrate = normalizeMergeSubstratePreview(
      await this.engine.request<unknown>(
        "preview_overlay",
        {
          targetBrainId,
          targetStoragePath: this.repository.brainDirectory(targetBrainId),
          sourceBrainId,
          sourceStoragePath: this.repository.brainDirectory(sourceBrainId)
        },
        600_000
      ),
      sourceBrainId,
      targetBrainId
    );
    if (!substrate) {
      throw new Error(
        "The authoritative neural worker did not return a verifiable fork-overlay preview."
      );
    }
    const conflicts: string[] = [];
    if (source.lineage.rootId !== target.lineage.rootId) {
      conflicts.push(
        "The brains have different immutable origins; only reviewed sparse overlays will merge."
      );
    }
    const changedConcepts = Object.entries(source.concepts).filter(
      ([id, concept]) =>
        target.concepts[id] !== undefined &&
        sha256(JSON.stringify(target.concepts[id])) !== sha256(JSON.stringify(concept))
    ).length;
    const changedSynapses = Object.entries(source.synapses).filter(
      ([id, synapse]) =>
        target.synapses[id] !== undefined &&
        sha256(JSON.stringify(target.synapses[id])) !== sha256(JSON.stringify(synapse))
    ).length;
    if (changedConcepts > 0 || changedSynapses > 0) {
      conflicts.push(
        `${changedConcepts} existing concepts and ${changedSynapses} existing synapses diverged; the target versions will be preserved.`
      );
    }
    if (
      substrate.divergent.neurons > 0 ||
      substrate.divergent.assemblies > 0 ||
      substrate.divergent.synapses > 0
    ) {
      conflicts.push(
        `${substrate.divergent.neurons} authoritative neurons, ` +
        `${substrate.divergent.assemblies} assemblies, and ` +
        `${substrate.divergent.synapses} authoritative synapses diverged; ` +
        "the target versions will be preserved."
      );
    }

    const targetEvidence = new Set(target.trainingSources.map(evidenceFingerprint));
    const evidence: MergePlan["evidence"] = [];
    for (const sourceEvidence of source.trainingSources) {
      const fingerprint = evidenceFingerprint(sourceEvidence);
      if (targetEvidence.has(fingerprint)) continue;
      targetEvidence.add(fingerprint);
      evidence.push({ fingerprint, source: sourceEvidence });
    }

    const files: MergeFileCandidate[] = [];
    let skippedFiles = 0;
    let examinedBytes = 0;
    let entryCount = 0;
    const addFile = async (
      candidate: Omit<MergeFileCandidate, "destinationPath" | "disposition">
    ): Promise<void> => {
      if (
        files.length >= MAX_MERGE_FILES ||
        candidate.bytes > MAX_MERGE_FILE_BYTES ||
        examinedBytes + candidate.bytes > MAX_MERGE_TOTAL_BYTES
      ) {
        skippedFiles += 1;
        conflicts.push(
          `Skipped ${candidate.sourcePath}: the reviewed file limit is 512 files, 128 MB each, and 512 MB total.`
        );
        return;
      }
      examinedBytes += candidate.bytes;
      const planned = await this.planMergeFile(target, candidate, conflicts);
      if (planned) files.push(planned);
      else skippedFiles += 1;
    };

    const sourceDirectory = this.repository.brainDirectory(source.id);
    const artifactRoots = [
      { directory: join(sourceDirectory, "artifacts"), label: "artifacts" },
      { directory: join(sourceDirectory, "engine", "artifacts"), label: "engine/artifacts" }
    ];
    for (const root of artifactRoots) {
      const queue: Array<{ path: string; relativePath: string; depth: number }> = [
        { path: root.directory, relativePath: "", depth: 0 }
      ];
      while (queue.length > 0) {
        const next = queue.shift();
        if (!next) break;
        let info;
        try {
          info = await lstat(next.path);
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code === "ENOENT") continue;
          throw error;
        }
        if (info.isSymbolicLink()) {
          skippedFiles += 1;
          conflicts.push(
            `Skipped symbolic link ${portableRelative(join(root.label, next.relativePath))}.`
          );
          continue;
        }
        if (info.isDirectory()) {
          if (next.depth >= 24) {
            skippedFiles += 1;
            conflicts.push(
              `Skipped deep artifact directory ${portableRelative(join(root.label, next.relativePath))}.`
            );
            continue;
          }
          const children = (await readdir(next.path)).sort();
          for (const name of children) {
            entryCount += 1;
            if (entryCount > MAX_MERGE_FILES * 8) {
              skippedFiles += 1;
              conflicts.push("Stopped artifact discovery after 4,096 branch-local entries.");
              queue.length = 0;
              break;
            }
            queue.push({
              path: join(next.path, name),
              relativePath: join(next.relativePath, name),
              depth: next.depth + 1
            });
          }
          continue;
        }
        const sourcePath = portableRelative(join(root.label, next.relativePath));
        if (!info.isFile()) {
          skippedFiles += 1;
          conflicts.push(`Skipped non-regular artifact ${sourcePath}.`);
          continue;
        }
        if (info.size > MAX_MERGE_FILE_BYTES) {
          await addFile({
            kind: "artifact",
            sourcePath,
            sha256: "0".repeat(64),
            bytes: info.size,
            absoluteSourcePath: next.path
          });
          continue;
        }
        const contents = await readFile(next.path);
        await addFile({
          kind: "artifact",
          sourcePath,
          sha256: sha256(contents),
          bytes: contents.byteLength,
          absoluteSourcePath: next.path
        });
      }
    }

    for (const record of evidence) {
      if (!retainsMergedBlob(target, record.source)) continue;
      let contents: Buffer | undefined;
      let blobHash: string | undefined;
      let absoluteSourcePath: string | undefined;
      if (record.source.blobHash) {
        try {
          contents = await this.repository.getBlob(record.source.blobHash);
          blobHash = record.source.blobHash;
        } catch {
          skippedFiles += 1;
          conflicts.push(
            `Evidence ${record.source.name} references a missing or corrupt blob; metadata only will merge.`
          );
          continue;
        }
      } else if (
        record.source.path &&
        pathWithin(sourceDirectory, record.source.path)
      ) {
        try {
          const info = await lstat(record.source.path);
          if (info.isFile() && !info.isSymbolicLink()) {
            contents = await readFile(record.source.path);
            absoluteSourcePath = record.source.path;
          }
        } catch {
          // Branch-local evidence can be absent after a user removes it. The
          // metadata remains useful and the missing bytes are reported below.
        }
      }
      if (!contents) {
        if (record.source.path || record.source.blobHash) {
          skippedFiles += 1;
          conflicts.push(
            `Evidence ${record.source.name} has no safe retained branch-local file; metadata only will merge.`
          );
        }
        continue;
      }
      await addFile({
        kind: "evidence",
        sourcePath: `evidence/${safeMergeName(record.source.name)}`,
        sha256: sha256(contents),
        bytes: contents.byteLength,
        absoluteSourcePath,
        blobHash,
        evidenceFingerprint: record.fingerprint
      });
    }

    const targetIdeaFingerprints = new Set(target.ideas.map((idea) => idea.fingerprint));
    const newConceptEntries = Object.entries(source.concepts)
      .filter(([id]) => target.concepts[id] === undefined)
      .sort(([left], [right]) => left.localeCompare(right));
    const newSynapseEntries = Object.entries(source.synapses)
      .filter(([id]) => target.synapses[id] === undefined)
      .sort(([left], [right]) => left.localeCompare(right));
    const newIdeaEntries = source.ideas.filter(
      (idea) => !targetIdeaFingerprints.has(idea.fingerprint)
    );
    const duplicateEvidence = source.trainingSources.length - evidence.length;
    const publicFiles = files.map(
      ({ kind, sourcePath, destinationPath, sha256: checksum, bytes, disposition }) => ({
        kind,
        sourcePath,
        destinationPath,
        sha256: checksum,
        bytes,
        disposition
      })
    );
    const boundedConflicts = conflicts.slice(0, MAX_MERGE_CONFLICTS);
    if (conflicts.length > boundedConflicts.length) {
      boundedConflicts.push(
        `${conflicts.length - boundedConflicts.length} additional file warnings were omitted.`
      );
    }
    const reviewDescriptor = {
      sourceBrainId,
      targetBrainId,
      sourceUpdatedAt: source.updatedAt,
      targetUpdatedAt: target.updatedAt,
      concepts: newConceptEntries.map(([id, value]) => [id, sha256(JSON.stringify(value))]),
      synapses: newSynapseEntries.map(([id, value]) => [id, sha256(JSON.stringify(value))]),
      ideas: newIdeaEntries.map((idea) => [
        idea.fingerprint,
        sha256(JSON.stringify(idea))
      ]),
      evidence: evidence
        .map(
          (entry) =>
            [entry.fingerprint, sha256(JSON.stringify(entry.source))] as const
        )
        .sort(([left], [right]) => left.localeCompare(right)),
      files: publicFiles,
      skippedFiles,
      conflicts: boundedConflicts,
      substrate
    };
    const preview: AgentMergePreview = {
      sourceBrainId,
      targetBrainId,
      reviewToken: sha256(JSON.stringify(reviewDescriptor)),
      substrate,
      newConcepts: substrate.additions.neurons,
      newIdeas: substrate.additions.assemblies,
      newSynapses: substrate.additions.synapses,
      newEvidence: evidence.length,
      duplicateEvidence,
      newFiles: files.filter((file) => file.disposition === "copy").length,
      duplicateFiles: files.filter((file) => file.disposition === "duplicate").length,
      skippedFiles,
      fileBytes: files
        .filter((file) => file.disposition === "copy")
        .reduce((sum, file) => sum + file.bytes, 0),
      files: publicFiles,
      conflicts: boundedConflicts,
      note:
        "Merge copies reviewed ideas, evidence, content-addressed files, and neural replay overlays. It never averages complete model weights."
    };
    return { source, target, evidence, files, substrate, preview };
  }

  async previewMerge(sourceBrainId: string, targetBrainId: string): Promise<AgentMergePreview> {
    return (await this.buildMergePlan(sourceBrainId, targetBrainId)).preview;
  }

  async merge(
    sourceBrainId: string,
    targetBrainId: string,
    reviewToken: string
  ): Promise<BrainDocument> {
    return withBrainWrite(this.repository, [sourceBrainId, targetBrainId], () =>
      this.mergeUnlocked(sourceBrainId, targetBrainId, reviewToken)
    );
  }

  private async mergeUnlocked(
    sourceBrainId: string,
    targetBrainId: string,
    reviewToken: string
  ): Promise<BrainDocument> {
    if (!/^[a-f0-9]{64}$/.test(reviewToken)) {
      throw new Error("A valid merge review token is required.");
    }
    const plan = await this.buildMergePlan(sourceBrainId, targetBrainId);
    if (plan.preview.reviewToken !== reviewToken) {
      throw new Error("The merge preview is stale. Review the branch overlay again.");
    }

    for (const file of plan.files) {
      let hash: string;
      if (file.blobHash) {
        const contents = await this.repository.getBlob(file.blobHash);
        hash = sha256(contents);
      } else if (file.absoluteSourcePath) {
        const sourceDirectory = this.repository.brainDirectory(plan.source.id);
        if (
          !pathWithin(sourceDirectory, file.absoluteSourcePath) ||
          (await lstat(file.absoluteSourcePath)).isSymbolicLink()
        ) {
          throw new Error(`Merge source ${file.sourcePath} is no longer a safe regular file.`);
        }
        const contents = await readFile(file.absoluteSourcePath);
        hash = await this.repository.storeBlob(contents);
      } else {
        throw new Error(`Merge source ${file.sourcePath} is unavailable.`);
      }
      if (hash !== file.sha256) {
        throw new Error(`Merge source ${file.sourcePath} changed after it was reviewed.`);
      }
      await this.repository.linkBlobTo(
        hash,
        join(
          this.repository.brainDirectory(plan.target.id),
          ...file.destinationPath.split("/")
        )
      );
    }

    const mergedOverlay = objectRecord(
      await this.engine.request<unknown>(
      "merge_overlay",
      {
        targetBrainId,
        targetStoragePath: this.repository.brainDirectory(targetBrainId),
        sourceBrainId,
        sourceStoragePath: this.repository.brainDirectory(sourceBrainId),
        expectedPreviewDigest: plan.substrate.digest
      },
      600_000
      )
    );
    if (
      !mergedOverlay ||
      mergedOverlay.weightsAveraged !== false ||
      mergedOverlay.reviewedDigest !== plan.substrate.digest
    ) {
      throw new Error(
        "The authoritative worker did not confirm the exact reviewed overlay digest."
      );
    }

    const { source, target } = plan;
    for (const [id, concept] of Object.entries(source.concepts)) {
      if (!target.concepts[id]) {
        target.concepts[id] = JSON.parse(JSON.stringify(concept)) as typeof concept;
      }
    }
    for (const [id, synapse] of Object.entries(source.synapses)) {
      if (!target.synapses[id]) {
        target.synapses[id] = JSON.parse(JSON.stringify(synapse)) as typeof synapse;
      }
    }
    const ideaFingerprints = new Set(target.ideas.map((idea) => idea.fingerprint));
    for (const idea of source.ideas) {
      if (!ideaFingerprints.has(idea.fingerprint)) {
        target.ideas.push(JSON.parse(JSON.stringify(idea)) as typeof idea);
        ideaFingerprints.add(idea.fingerprint);
      }
    }

    const targetEvidenceFingerprints = new Set(target.trainingSources.map(evidenceFingerprint));
    const evidenceIds = new Set(target.trainingSources.map((entry) => entry.id));
    for (const record of plan.evidence) {
      if (targetEvidenceFingerprints.has(record.fingerprint)) continue;
      const copied = JSON.parse(JSON.stringify(record.source)) as TrainingSource;
      if (evidenceIds.has(copied.id)) copied.id = randomUUID();
      evidenceIds.add(copied.id);
      delete copied.path;
      const evidenceFile = plan.files.find(
        (file) => file.kind === "evidence" && file.evidenceFingerprint === record.fingerprint
      );
      if (evidenceFile) {
        copied.path = join(
          this.repository.brainDirectory(target.id),
          ...evidenceFile.destinationPath.split("/")
        );
        copied.blobHash = evidenceFile.sha256;
      } else {
        delete copied.blobHash;
      }
      const retainRawText =
        (target.config.memoryRecipe ?? "human-consolidation") === "total-recall" &&
        target.config.retainSourceText;
      if (!retainRawText) {
        copied.rawTextRetained = false;
        delete copied.rawText;
      }
      target.trainingSources.push(copied);
      targetEvidenceFingerprints.add(record.fingerprint);
    }

    const mergedAt = new Date().toISOString();
    const manifest = Buffer.from(
      JSON.stringify(
        {
          schemaVersion: 1,
          sourceBrainId,
          targetBrainId,
          reviewToken,
          authoritativeSubstrateDigest: plan.substrate.digest,
          sourceStateSha256: plan.substrate.sourceStateSha256,
          targetStateSha256: plan.substrate.targetStateSha256,
          mergedAt,
          strategy: "ideas-evidence-files-replay-overlay",
          wholeModelWeightsAveraged: false,
          evidence: plan.evidence.map((entry) => entry.fingerprint),
          files: plan.preview.files
        },
        null,
        2
      ),
      "utf8"
    );
    const manifestHash = await this.repository.storeBlob(manifest);
    await this.repository.linkBlobTo(
      manifestHash,
      join(
        this.repository.brainDirectory(target.id),
        "artifacts",
        "merge-manifests",
        `${reviewToken}.json`
      )
    );
    target.journal = [
      ...(target.journal ?? []),
      {
        id: randomUUID(),
        createdAt: mergedAt,
        kind: "fork",
        summary: `Merged reviewed overlays from ${source.name}.`,
        detail: JSON.stringify({
          sourceBrainId: source.id,
          reviewToken,
          authoritativeSubstrateDigest: plan.substrate.digest,
          ideas: plan.preview.newIdeas,
          evidence: plan.preview.newEvidence,
          files: plan.preview.newFiles,
          replay: "neural-overlay",
          weightsAveraged: false
        })
      }
    ];
    return this.repository.save(target);
  }

  async runtimeHealth(id: string): Promise<RuntimeHealth> {
    const brain = await this.repository.get(id);
    const health = await this.engine.health();
    return {
      runtime: brain.config.runtime,
      ready: health.ready,
      label: health.worker === "python" ? "Omni neural worker" : "Neural worker unavailable",
      detail: health.detail
    };
  }
}

export class RuntimeJobManager extends EventEmitter {
  private readonly jobs = new Map<string, JobRecord>();

  constructor(
    private readonly service: BrainService,
    private readonly engine: EngineSupervisor
  ) {
    super();
    engine.on("event", (event: EngineEvent) => this.consumeEngineEvent(event));
  }

  list(brainId?: string): RuntimeJob[] {
    return [...this.jobs.values()]
      .filter((job) => !brainId || job.brainId === brainId)
      .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
      .map((job) => ({ ...job }));
  }

  async wait(
    jobId: string,
    signal?: AbortSignal,
    timeoutMs = 600_000
  ): Promise<RuntimeJob> {
    const terminal = new Set<RuntimeJob["state"]>([
      "complete",
      "failed",
      "cancelled"
    ]);
    const current = this.jobs.get(jobId);
    if (!current) throw new Error("The runtime job was not found.");
    if (terminal.has(current.state)) return { ...current };
    if (signal?.aborted) {
      await this.cancel(jobId);
      throw new Error("Runtime job was cancelled.");
    }

    return new Promise<RuntimeJob>((resolveJob, rejectJob) => {
      let settled = false;
      const cleanup = (): void => {
        clearTimeout(timer);
        this.off("event", onEvent);
        signal?.removeEventListener("abort", onAbort);
      };
      const finish = (
        result: { job: RuntimeJob } | { error: Error }
      ): void => {
        if (settled) return;
        settled = true;
        cleanup();
        if ("error" in result) rejectJob(result.error);
        else resolveJob({ ...result.job });
      };
      const onEvent = ({ job }: RuntimeJobEvent): void => {
        if (job.id === jobId && terminal.has(job.state)) finish({ job });
      };
      const cancelAndReject = (message: string): void => {
        if (settled) return;
        settled = true;
        cleanup();
        void this.cancel(jobId)
          .catch(() => undefined)
          .finally(() => rejectJob(new Error(message)));
      };
      const onAbort = (): void => cancelAndReject("Runtime job was cancelled.");
      const timer = setTimeout(
        () => cancelAndReject("Runtime job timed out."),
        Math.max(1_000, Math.min(3_600_000, Math.round(timeoutMs)))
      );

      this.on("event", onEvent);
      signal?.addEventListener("abort", onAbort, { once: true });

      // The job can finish between the first state check and listener setup.
      const afterSubscription = this.jobs.get(jobId);
      if (afterSubscription && terminal.has(afterSubscription.state)) {
        finish({ job: afterSubscription });
      }
    });
  }

  startTraining(request: StartTrainingRequest): RuntimeJob {
    const epochs = request.epochs ?? 1;
    if (!Number.isSafeInteger(epochs) || epochs < 1) {
      throw new Error("Training epochs must be a positive safe integer.");
    }
    const job = this.createJob(request.brainId, "training", "Training slow neural parameters");
    void this.run(job, async () => {
      return this.engine.request<unknown>(
        "train",
        {
          jobId: job.id,
          brainId: request.brainId,
          epochs,
          learningRate: request.learningRate,
          sourceIds: request.sourceIds,
          storagePath: this.service.repository.brainDirectory(request.brainId)
        },
        3_600_000
      );
    });
    return { ...job };
  }

  startIngestion(request: DatasetStartRequest): RuntimeJob {
    const epochs = request.epochs ?? 1;
    if (!Number.isSafeInteger(epochs) || epochs < 1) {
      throw new Error("Dataset epochs must be a positive safe integer.");
    }
    const job = this.createJob(
      request.brainId,
      "ingestion",
      request.resume ? "Resuming whole-dataset training" : "Training on whole dataset"
    );
    void this.run(job, async () => {
      if (request.resume === false) {
        await this.service.datasets.reset(
          request.brainId,
          request.manifestId,
          epochs
        );
      }
      return this.service.ingestManifest(
        request.brainId,
        request.manifestId,
        request.policy ?? "encode",
        () => Boolean(job.cancelled),
        (progress, message) => {
          if (job.cancelled) return;
          job.progress = Math.max(job.progress, Math.min(0.99, progress));
          job.label = message;
          job.updatedAt = new Date().toISOString();
          this.publish(job);
        },
        false,
        epochs
      );
    });
    return { ...job };
  }

  startCrawl(request: WebCrawlRequest): RuntimeJob {
    const job = this.createJob(request.brainId, "crawl", "Crawling quarantined web sources");
    void this.run(job, () =>
      this.service.crawlWeb(
        request,
        () => Boolean(job.cancelled),
        (progress, message) => {
          if (job.cancelled) return;
          job.progress = Math.max(job.progress, Math.min(0.99, progress));
          job.label = message;
          job.updatedAt = new Date().toISOString();
          this.publish(job);
        }
      )
    );
    return { ...job };
  }

  generate(request: ModalityGenerateRequest): RuntimeJob {
    const job = this.createJob(
      request.brainId,
      request.modality,
      `Generating ${request.modality}`
    );
    void this.run(job, async () => {
      const output = await this.engine.request<unknown>(
        "generate_modality",
        {
          jobId: job.id,
          ...request,
          storagePath: this.service.repository.brainDirectory(request.brainId)
        },
        3_600_000
      );
      return output;
    });
    return { ...job };
  }

  async cancel(jobId: string): Promise<RuntimeJob> {
    const job = this.jobs.get(jobId);
    if (!job) throw new Error("The runtime job was not found.");
    if (["complete", "failed", "cancelled"].includes(job.state)) return { ...job };
    job.cancelled = true;
    job.state = "cancelled";
    job.updatedAt = new Date().toISOString();
    if (job.kind === "crawl") {
      // The crawler cooperatively checks job.cancelled between bounded fetches.
    } else {
      for (const other of this.jobs.values()) {
        if (
          other.id !== job.id &&
          !other.cancelled &&
          ["queued", "running"].includes(other.state) &&
          other.kind !== "crawl"
        ) {
          other.cancelled = true;
          other.state = "failed";
          other.error = "Neural worker restarted because another running job was cancelled.";
          other.updatedAt = job.updatedAt;
          this.publish(other);
        }
      }
      await this.engine.interruptAndRestart();
    }
    this.publish(job);
    return { ...job };
  }

  private createJob(
    brainId: string,
    kind: RuntimeJob["kind"],
    label: string
  ): JobRecord {
    const now = new Date().toISOString();
    const job: JobRecord = {
      id: randomUUID(),
      brainId,
      kind,
      state: "queued",
      progress: 0,
      label,
      createdAt: now,
      updatedAt: now
    };
    this.jobs.set(job.id, job);
    this.publish(job);
    return job;
  }

  private async run(job: JobRecord, operation: () => Promise<unknown>): Promise<void> {
    if (job.cancelled) return;
    job.state = "running";
    job.progress = Math.max(job.progress, 0.01);
    job.updatedAt = new Date().toISOString();
    this.publish(job);
    try {
      const output = await operation();
      if (job.cancelled) return;
      job.state = "complete";
      job.progress = 1;
      job.output = output;
    } catch (error) {
      if (job.cancelled) return;
      job.state = "failed";
      job.error = error instanceof Error ? error.message : String(error);
    }
    job.updatedAt = new Date().toISOString();
    this.publish(job);
  }

  private consumeEngineEvent(event: EngineEvent): void {
    if (!event.jobId) return;
    const job = this.jobs.get(event.jobId);
    if (!job || job.cancelled) return;
    if (typeof event.progress === "number") {
      job.progress = Math.max(job.progress, Math.min(0.99, Math.max(0, event.progress)));
    }
    if (event.message) job.label = event.message.slice(0, 200);
    if (event.type === "modality-preview") {
      const preview = normalizeModalityPreview(
        objectRecord(event.data)?.preview ?? event.data,
        typeof event.sequence === "number" ? event.sequence : (job.preview?.revision ?? 0) + 1,
        event.progress,
        event.message
      );
      if (preview && (!job.preview || preview.revision > job.preview.revision)) {
        job.preview = preview;
      }
    }
    job.updatedAt = new Date().toISOString();
    this.publish(job);
  }

  private publish(job: JobRecord): void {
    const event: RuntimeJobEvent = { job: { ...job } };
    this.emit("event", event);
  }
}
