import { createHash, randomUUID } from "node:crypto";
import { lookup } from "node:dns/promises";
import { createReadStream, existsSync } from "node:fs";
import { normalizeConceptIdView } from "../shared/conceptIdView";
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
import { availableParallelism, freemem } from "node:os";
import { basename, extname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { performance } from "node:perf_hooks";
import { EventEmitter } from "node:events";
import { cleanupCancelledInlineStage } from "./inlineCancellationCleanup";
import { savedRuntimeShape } from "./savedRuntimeShape";
import { getHeapStatistics } from "node:v8";
import { DatabaseSync } from "node:sqlite";
import toolCatalog from "../../tools/catalog.json";
import type {
  ActionEvent,
  AgentMergeFilePreview,
  AgentMergePreview,
  AgentMergeSubstratePreview,
  BrainConfig,
  BrainDocument,
  BrainRuntimeCard,
  BrainSnapshotSummary,
  BuildResourceStartRequest,
  BuildRecipe,
  ChatCancellationState,
  ChatGenerationEnd,
  ChatNoReplyReason,
  ChatMessage,
  ChatQueueState,
  ChatResult,
  CreateBrainRequest,
  DataIngestionPolicy,
  DatasetEntryReceipt,
  DatasetManifest,
  DatasetManifestEntry,
  DatasetProgressGeneration,
  DatasetStartRequest,
  FeedbackRequest,
  FreshAttentionResult,
  HardwareTier,
  IdleCycleResult,
  InstalledModalityPack,
  ImportUrlRequest,
  IngestWebRequest,
  IngestResult,
  LiveObservationCaptureMode,
  LiveObservationControl,
  LiveObservationControlRequest,
  LiveObservationControlResolution,
  LiveObservationEvent,
  LiveObservationModality,
  LiveObservationPacket,
  LiveObservationPacketResult,
  LiveObservationSession,
  LiveObservationSessionStartRequest,
  ModalityGenerateRequest,
  NeuralSpeechGenerateRequest,
  NeuralModalityCapabilities,
  ModalityPreview,
  PersistedSubstrateOverview,
  PersistedDataIngestionPolicy,
  RuntimeHealth,
  RuntimeJob,
  RuntimeJobEvent,
  StructuredAction,
  SubstratePage,
  SubstrateQuery,
  ToolPermissionLevel,
  ToolPermissionRecord,
  TrainingCoverage,
  TrainingSource,
  WebCrawlRequest,
  WebCrawlResult,
  WorkspaceLearningStatus,
  WorkspaceSnapshot,
  WorkingMemoryPlanRequest,
  WorkingMemoryResourcePlan
} from "../shared/types";
import type { CortexPage, CortexQuery, CortexActivityQuery, CortexActivity } from "../shared/cortexInspection";
import { INLINE_MEDIA_DATA_URL_CHARACTER_LIMIT } from "../shared/mediaTransport";
import { recordNeuralChat } from "./presentationChat";
import {
  listInstalledPacks,
  recordInstalledPack,
  removeStagedPack,
  stageModalityPack,
  validateBuildRecipe
} from "./catalogInstaller";
import { BrainRepository, DEFAULT_TOOL_PERMISSIONS } from "./brainRepository";
import { withBrainWrite } from "./brainWriteCoordinator";
import { acquireCrawlSourceLease, releaseCrawlSourceLease } from "./crawlSourceLease";
import {
  BackgroundRequestDeferredError,
  ENGINE_REQUEST_NO_DEADLINE,
  EngineRequestError,
  EngineSupervisor,
  type EngineActivityTransition,
  type EngineEvent
} from "./engineSupervisor";
import { normalizeStructuredAction, parseModelActions } from "./actionProtocol";
import { ResourcePlanner } from "./resourcePlanner";
import { calculateDiskSpaceReport } from "./diskSpace";
import type { InitialFoundationPlan } from "./buildInitializationPlan";
import type { MediaArtifactRegistry } from "./mediaArtifactRegistry";
import type { BrainStorageOperationHooks } from "./brainStorageOperations";
import {
  MergeEvidencePlan,
  MergeEvidencePlanBuilder,
  type MergeEvidencePlanEntry,
  type MergeEvidencePlanSummary
} from "./mergeEvidenceStore";
import {
  assertManifestEntryStable,
  CrawlFrontierStore,
  DatasetManifestStore,
  detectDatasetFormat,
  hashFile,
  type DatasetManifestCreateOptions
} from "./dataIngestion";
import {
  foundationTelemetrySample,
  normalizePersistedTrainingTelemetry,
  resumeTrainingTelemetry,
  trainingResourceReadings,
  updateTrainingTelemetry,
  type FoundationTelemetrySample,
  type TrainingResourceReadings,
  type TrainingTelemetry
} from "../shared/trainingTelemetry";

const MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024;
const MAX_MERGE_FILES = 512;
const MAX_MERGE_FILE_BYTES = 128 * 1024 * 1024;
const MAX_MERGE_TOTAL_BYTES = 512 * 1024 * 1024;
const MAX_MERGE_CONFLICTS = 100;
const MAX_ROBOTS_BYTES = 2 * 1024 * 1024;
export const STANDARD_FOREGROUND_LOAD_TIMEOUT_MS = 5 * 60_000;
export const LARGE_FOREGROUND_LOAD_TIMEOUT_MS = 30 * 60_000;
const LARGE_SUBSTRATE_SHARD_COUNT = 2_048;

export function requireNewDataIngestionPolicy(
  value: unknown
): DataIngestionPolicy {
  if (!(["encode", "pretrain", "archive"] as const).includes(
    value as DataIngestionPolicy
  )) {
    throw new Error("New learning requests must use encode, pretrain, or archive.");
  }
  return value as DataIngestionPolicy;
}

const CRAWL_DISK_CHECK_INTERVAL = 4 * 1024 * 1024;
const CRAWL_MEMORY_RESERVE_MINIMUM = 64 * 1024 * 1024;
const CRAWL_DISK_RESERVE_MINIMUM = 512 * 1024 * 1024;
const CRAWL_DIAGNOSTIC_WINDOW = 32;
const CRAWL_USER_AGENT = "OmniAGIStudio/1.0 (+local research crawler)";
const TOOL_ACTIONS: Readonly<Record<string, readonly string[]>> = {
  "system.files": ["list", "read", "write"],
  "system.shell": ["run"],
  "code.execute": ["run"],
  "web.fetch": ["fetch"],
  "web.search": ["search"],
  "browser.automation": ["task"],
  "device.input": ["move-pointer", "click", "scroll", "key-press", "text"],
  "device.observe": ["configure", "snapshot"],
  "modality.imagine": ["generate"],
  "brain.history": ["read", "search"],
  "agent.fork": ["start"],
  "source.self-modify": ["propose", "diff", "test", "promote", "rollback"]
};

type CatalogActionInput = { input?: Record<string, string> };
type CatalogToolInput = { id: string; actions: Record<string, CatalogActionInput> };

function catalogFieldSchema(descriptor: string): Record<string, unknown> {
  const optional = descriptor.endsWith("?");
  const base = optional ? descriptor.slice(0, -1) : descriptor;
  const array = base.endsWith("[]");
  const element = array ? base.slice(0, -2) : base;
  const choices = element.split("|");
  const primitive = choices.length > 1 ? "string" : element;
  const type = ["string", "number", "boolean", "object"].includes(primitive)
    ? primitive
    : "object";
  const itemSchema: Record<string, unknown> = { type };
  if (choices.length > 1) itemSchema.enum = choices;
  return array ? { type: "array", items: itemSchema } : itemSchema;
}

const CATALOG_ACTION_INPUT_SCHEMAS = new Map<string, Record<string, Record<string, unknown>>>(
  (toolCatalog.tools as unknown as CatalogToolInput[]).map((tool) => [
    tool.id,
    Object.fromEntries(
      Object.entries(tool.actions).map(([action, definition]) => {
        const fields = Object.entries(definition.input ?? {});
        return [
          action,
          {
            type: "object",
            properties: Object.fromEntries(
              fields.map(([name, descriptor]) => [name, catalogFieldSchema(descriptor)])
            ),
            required: fields
              .filter(([, descriptor]) => !descriptor.endsWith("?"))
              .map(([name]) => name),
            additionalProperties: false
          }
        ];
      })
    )
  ])
);
const evolutionInput = CATALOG_ACTION_INPUT_SCHEMAS.get("source.self-modify")?.propose;
if (evolutionInput) {
  const properties = evolutionInput.properties as Record<string, Record<string, unknown>>;
  properties.sourceEdits = {
    type: "array", items: {
      type: "object", additionalProperties: false,
      properties: { path: { type: "string" }, content: { type: "string" }, expectedSha256: { type: ["string", "null"] } },
      required: ["path", "content", "expectedSha256"]
    }
  };
  properties.architectureChange = {
    type: "object", required: ["mutation"], additionalProperties: false,
    properties: {
      mutation: { type: "string", enum: ["grow-experts", "grow-depth", "grow-router", "grow-regions"] },
      addExperts: { type: "integer", minimum: 1 }, addLayers: { type: "integer", minimum: 1 },
      addNeurons: { type: "integer", minimum: 1 }, addRegions: { type: "integer", minimum: 1 },
      neuronsPerRegion: { type: "integer", minimum: 1 }
    }
  };
}

const LEGACY_SYSTEM_TOOL_ALIASES: Readonly<Record<string, string>> = {
  "windows.files": "system.files",
  "windows.powershell": "system.shell"
};

function canonicalToolPermissionId(toolId: string): string {
  return LEGACY_SYSTEM_TOOL_ALIASES[toolId] ?? toolId;
}

/**
 * Reversible renderer destinations are neural capabilities, but they are not
 * permission-bearing external tools. Keeping this schema outside each brain's
 * authority matrix means old and new stable brains see the same local UI
 * surface without silently widening file, process, network, or source access.
 */
const LOCAL_STUDIO_SCHEMAS = [
  {
    id: "studio.ui",
    actions: ["open-creativity"] as const,
    grant: "auto" as const
  },
  {
    id: "studio.settings",
    actions: ["inspect-access", "open-permissions"] as const,
    grant: "auto" as const
  }
] as const;

function effectiveToolPermissions(
  records: ToolPermissionRecord[] | undefined
): ToolPermissionRecord[] {
  const stored = new Map<string, ToolPermissionRecord>();
  // A current system.* record wins over its legacy alias. Otherwise migrate
  // the old grant in memory so imported brains keep authority without showing
  // Windows-only protocols on macOS/Linux.
  for (const record of records ?? []) {
    if (canonicalToolPermissionId(record.toolId) !== record.toolId) continue;
    stored.set(record.toolId, record);
  }
  for (const record of records ?? []) {
    const canonical = canonicalToolPermissionId(record.toolId);
    if (!stored.has(canonical)) {
      stored.set(canonical, { ...record, toolId: canonical });
    }
  }
  const defaults = DEFAULT_TOOL_PERMISSIONS.map((record) => ({
    ...record,
    ...(stored.get(record.toolId) ?? {}),
    toolId: record.toolId,
    label: record.label
  }));
  const custom = [...stored.values()].filter(
    (record) => !DEFAULT_TOOL_PERMISSIONS.some(
      (fallback) => fallback.toolId === record.toolId
    )
  );
  return [...defaults, ...custom]
    .map((record) => ({ ...record }))
    .sort((left, right) =>
      left.label.localeCompare(right.label) ||
      left.toolId.localeCompare(right.toolId)
    );
}

export interface ExternalToolSchema {
  id: string;
  actions: readonly string[];
  inputSchema?: Record<string, unknown>;
}

function enabledToolSchemas(
  brain: BrainDocument,
  externalSchemas: ReadonlyMap<string, ExternalToolSchema> = new Map()
): Array<{
  id: string;
  actions: readonly string[];
  grant: ToolPermissionLevel;
  inputSchema?: Record<string, unknown>;
  actionInputSchemas?: Record<string, Record<string, unknown>>;
}> {
  const permissionBearing = effectiveToolPermissions(brain.toolPermissions)
    .filter((permission) => permission.level !== "off")
    .flatMap((permission) => {
      const external = externalSchemas.get(permission.toolId);
      const actions = TOOL_ACTIONS[permission.toolId] ?? external?.actions;
      return actions
        ? [
            {
              id: permission.toolId,
              actions,
              grant: permission.level,
              ...(external?.inputSchema ? { inputSchema: external.inputSchema } : {}),
              ...(CATALOG_ACTION_INPUT_SCHEMAS.has(permission.toolId)
                ? { actionInputSchemas: CATALOG_ACTION_INPUT_SCHEMAS.get(permission.toolId)! }
                : {})
            }
          ]
        : [];
    });
  return [
    ...permissionBearing,
    ...LOCAL_STUDIO_SCHEMAS
  ];
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
    core_parameter_delta_norm?: number;
    parameter_delta_scope?: string;
    substrate_parameter_delta_norm?: null;
    substrate_parameter_delta_measured?: false;
    parameter_checksum_scope?: string;
    stdp_update?: number;
    spike_rate?: number;
    train_loss?: number;
    generation_entropy?: number;
    ponder_steps?: number;
    steps?: Array<{ stage?: string; detail?: string; value?: string }>;
    note?: string;
    attention_epoch?: number;
    generation_stop_reason?: string;
    generation_decoder_stop_reason?: string;
    generation_no_reply_reason?: string | null;
    generated_token_count?: number;
    generation_printable_text_characters?: number;
    slow_learning_job?: {
      jobId?: string;
      priority?: number;
    } | null;
  };
  metrics?: Record<string, unknown>;
  runtimeCard?: Record<string, unknown>;
  actions?: unknown;
  humanMessage?: unknown;
  message?: unknown;
  turnReceipt?: unknown;
  turnCommitted?: boolean;
  idempotentCompletion?: boolean;
  steered?: boolean;
  nativeStopped?: boolean;
  noReply?: boolean;
}

interface WorkerChatReceiptResult {
  format?: string;
  formatVersion?: number;
  brainId?: string;
  turnId?: string;
  committed?: boolean;
  turnCommitted?: boolean;
  legacyMatched?: boolean;
  inputSha256?: string;
  humanMessage?: unknown;
  brainMessage?: unknown;
  trace?: unknown;
  inferenceCount?: number;
  plasticityEvents?: number;
  consolidationCycles?: number;
  parameterChecksumAfter?: string;
  engineUpdatedAt?: string;
  substrateGeneration?: string;
  mutableStateGeneration?: string;
  idempotentCompletion?: boolean;
  generationEnd?: ChatGenerationEnd;
  noReply?: boolean;
}

export function chatSlowReplayDelayMs(priority: number): number {
  const bounded = Number.isFinite(priority)
    ? Math.max(0, Math.min(1, priority))
    : 0;
  // Replay is continuous rather than thresholded: highly recurrent/salient
  // activity returns sooner, while weak/interfered one-offs cool longer.
  return Math.round(1_000 + (1 - bounded) * 9_000);
}

interface ValidatedChatReceipt {
  humanMessage: ChatMessage;
  brainMessage: ChatMessage;
  trace: NonNullable<WorkerChatResult["trace"]>;
  inferenceCount: number;
  plasticityEvents: number;
  consolidationCycles: number;
}

interface ValidatedWorkerChatPresentation {
  humanMessage: ChatMessage;
  brainMessage: ChatMessage;
  inferenceCount: number;
}

export type NeuralChatStreamEvent =
  | {
      type: "runtime-activity";
      state: "queued" | "running" | "cancelled" | "failed";
      queue?: ChatQueueState;
      cancellation?: ChatCancellationState;
    }
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
  | { type: "inline-imagination-started"; sequence: number; actionId: string }
  | {
      type: "modality-preview";
      sequence: number;
      actionId?: string;
      preview: ModalityPreview;
    }
  | {
      type: "chat-phase";
      sequence: number;
      phase: "reply-complete-learning";
      replyComplete: true;
      turnCommitted: false;
      learning: true;
      saving: true;
    };

function neuralChatRuntimeActivity(
  transition: EngineActivityTransition
): NeuralChatStreamEvent | undefined {
  if (transition.state === "queued" && transition.queuedBehind) {
    return {
      type: "runtime-activity",
      state: "queued",
      queue: {
        position: Math.max(1, transition.queuePosition ?? 1),
        queuedBehind: { ...transition.queuedBehind }
      }
    };
  }
  if (transition.state === "running") {
    return { type: "runtime-activity", state: "running" };
  }
  if (
    (transition.state === "cancelled" || transition.state === "failed") &&
    transition.cancellationPhase
  ) {
    const phase = transition.cancellationPhase ?? "queued";
    return {
      type: "runtime-activity",
      state: transition.state,
      cancellation: {
        phase,
        unrelatedActivityContinues:
          phase === "queued" && Boolean(transition.queuedBehind),
        workerTerminationAcknowledged:
          transition.workerTerminationAcknowledged === true
      }
    };
  }
  return undefined;
}

function objectRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

export interface NeuralCheckpointReceipt {
  format: "omni-neural-checkpoint";
  formatVersion: 1;
  brainId: string;
  operationId: string;
  committed: true;
  createdAt: string;
  parameterChecksum: string;
  metadataSha256: string;
  substrateContentSha256: string;
  mutableStateContentSha256: string;
  packedManifestSha256: string;
  packedContentSha256: string;
  snapshotCreated: false;
}

export function requireNeuralCheckpointReceipt(
  value: unknown,
  brainId: string,
  operationId: string
): NeuralCheckpointReceipt {
  const receipt = objectRecord(value);
  const hashes = [
    "parameterChecksum",
    "metadataSha256",
    "substrateContentSha256",
    "mutableStateContentSha256",
    "packedManifestSha256",
    "packedContentSha256"
  ] as const;
  if (
    !receipt ||
    receipt.format !== "omni-neural-checkpoint" ||
    receipt.formatVersion !== 1 ||
    receipt.brainId !== brainId ||
    receipt.operationId !== operationId ||
    receipt.committed !== true ||
    receipt.snapshotCreated !== false ||
    typeof receipt.createdAt !== "string" ||
    !Number.isFinite(Date.parse(receipt.createdAt)) ||
    hashes.some(
      (field) =>
        typeof receipt[field] !== "string" ||
        !/^[a-f0-9]{64}$/.test(receipt[field] as string)
    )
  ) {
    throw new Error("The neural worker did not acknowledge a complete checkpoint.");
  }
  return receipt as unknown as NeuralCheckpointReceipt;
}

async function verifyNeuralCheckpointFiles(
  engineDirectory: string,
  receipt: NeuralCheckpointReceipt
): Promise<void> {
  const [metadataBytes, packedBytes] = await Promise.all([
    readFile(join(engineDirectory, "brain.json")),
    readFile(join(engineDirectory, "packed-ternary", "manifest.json"))
  ]);
  const metadata = objectRecord(JSON.parse(metadataBytes.toString("utf8")));
  const substrate = objectRecord(metadata?.substrate);
  const substratePointer = objectRecord(substrate?.persistence);
  const mutableState = objectRecord(metadata?.mutable_state);
  const packedState = objectRecord(metadata?.packed_ternary_manifest);
  const packedManifest = objectRecord(JSON.parse(packedBytes.toString("utf8")));
  if (
    sha256(metadataBytes) !== receipt.metadataSha256 ||
    sha256(packedBytes) !== receipt.packedManifestSha256 ||
    metadata?.brain_id !== receipt.brainId ||
    substratePointer?.contentSha256 !== receipt.substrateContentSha256 ||
    mutableState?.contentSha256 !== receipt.mutableStateContentSha256 ||
    packedState?.contentSha256 !== receipt.packedContentSha256 ||
    packedState?.parameterChecksum !== receipt.parameterChecksum ||
    packedManifest?.contentSha256 !== receipt.packedContentSha256
  ) {
    throw new Error(
      "The acknowledged neural checkpoint does not match its committed files."
    );
  }
}

export async function foregroundLoadTimeoutMs(
  brainDirectory: string
): Promise<number> {
  try {
    const metadata = objectRecord(
      JSON.parse(
        await readFile(join(brainDirectory, "engine", "brain.json"), "utf8")
      )
    );
    const substrate = objectRecord(metadata?.substrate);
    const persistence = objectRecord(substrate?.persistence);
    const shardCount = persistence?.shardCount;
    return Number.isSafeInteger(shardCount) &&
      Number(shardCount) >= LARGE_SUBSTRATE_SHARD_COUNT
      ? LARGE_FOREGROUND_LOAD_TIMEOUT_MS
      : STANDARD_FOREGROUND_LOAD_TIMEOUT_MS;
  } catch {
    return STANDARD_FOREGROUND_LOAD_TIMEOUT_MS;
  }
}

export async function persistedModalityCapabilities(
  brainDirectory: string,
  brainId: string
): Promise<NeuralModalityCapabilities> {
  const unavailable = (hardwareTier: HardwareTier = "personal") => ({
    brainId,
    hardwareTier,
    imagePerception: false,
    audioPerception: false,
    videoPerception: false,
    imageGeneration: false,
    audioGeneration: false,
    videoGeneration: false,
    neuralSpeechRecognition: false,
    neuralSpeechSynthesis: false,
    synchronizedVideoAudioGeneration: false,
    sameBrainSubstrate: true as const,
    hiddenBehavioralPrompt: false as const,
    detail: "Persisted modality readiness is unavailable; no neural worker was loaded for this UI inspection."
  });
  try {
    const metadata = objectRecord(JSON.parse(
      await readFile(join(brainDirectory, "engine", "brain.json"), "utf8")
    ));
    if (!metadata || metadata.brain_id !== brainId) return unavailable();
    const config = objectRecord(metadata.config) ?? {};
    const rawTier = String(config.hardware_tier ?? "");
    const hardwareTier = ["micro", "personal", "gpu", "workstation"].includes(rawTier)
      ? rawTier as HardwareTier
      : "personal";
    const training = objectRecord(metadata.modality_training) ?? {};
    const installed = new Set<string>();
    if (Array.isArray(metadata.installed_modality_packs)) {
      for (const value of metadata.installed_modality_packs) {
        const pack = objectRecord(value);
        if (!Array.isArray(pack?.modalities)) continue;
        for (const modality of pack.modalities) {
          if (["vision", "image", "audio", "video"].includes(String(modality))) {
            installed.add(String(modality));
          }
        }
      }
    }
    const trained = (modality: "vision" | "image" | "audio" | "video"): boolean =>
      (
        typeof training[modality] === "number" &&
        Number.isFinite(training[modality]) &&
        Number(training[modality]) > 0
      ) || installed.has(modality);
    const enabled = (modality: "vision" | "image" | "audio" | "video"): boolean =>
      config[`${modality}_enabled`] === true;
    const imagePerception =
      (enabled("vision") && trained("vision")) ||
      (enabled("image") && trained("image"));
    const audioGeneration = enabled("audio") && trained("audio");
    const pairedCount = training.audio_speech_pairs;
    const speechPairedExamples = typeof pairedCount === "number" && Number.isSafeInteger(pairedCount) && pairedCount >= 0
      ? pairedCount : 0;
    const videoGeneration = enabled("video") && trained("video");
    return {
      brainId,
      hardwareTier,
      imagePerception,
      audioPerception: audioGeneration,
      videoPerception: videoGeneration,
      imageGeneration: enabled("image") && trained("image"),
      audioGeneration,
      videoGeneration,
      neuralSpeechRecognition: false,
      neuralSpeechSynthesis: false,
      audioRegionAvailable: enabled("audio"),
      speechPairedExamples,
      speechQuality: speechPairedExamples > 0 ? "unverified" as const : "needs-speech-training" as const,
      synchronizedVideoAudioGeneration: videoGeneration && audioGeneration,
      sameBrainSubstrate: true,
      hiddenBehavioralPrompt: false,
      detail: "Read from committed same-brain audio metadata without loading a neural checkpoint. Own waveform output is distinct from platform STT/TTS; speech intelligibility has not been verified."
    };
  } catch {
    return unavailable();
  }
}

function isNonNegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function validIsoTimestamp(value: unknown): string | undefined {
  return typeof value === "string" && Number.isFinite(Date.parse(value))
    ? new Date(value).toISOString()
    : undefined;
}

function persistedWorkspaceLearningStatus(
  metadata: Record<string, unknown>,
  workspace: Record<string, unknown>
): WorkspaceLearningStatus | undefined {
  const rawTurns = metadata.completed_chat_turns;
  const rawPending = metadata.pending_chat_slow_learning;
  const rawCompleted = metadata.completed_chat_slow_learning;
  const counters = objectRecord(metadata.counters);
  if (
    !Array.isArray(rawTurns) &&
    !Array.isArray(rawPending) &&
    !Array.isArray(rawCompleted) &&
    !counters
  ) {
    return undefined;
  }
  const completedTurns = Array.isArray(rawTurns)
    ? rawTurns.flatMap((value) => {
        const turn = objectRecord(value);
        const committedAt = validIsoTimestamp(turn?.committedAt);
        if (
          !turn ||
          typeof turn.turnId !== "string" ||
          !turn.turnId ||
          !committedAt ||
          typeof turn.parameterChecksumAfter !== "string" ||
          !/^[a-f0-9]{64}$/i.test(turn.parameterChecksumAfter)
        ) {
          return [];
        }
        return [{
          turnId: turn.turnId,
          committedAt,
          parameterChecksumAfter: turn.parameterChecksumAfter.toLowerCase()
        }];
      })
    : [];
  completedTurns.sort((left, right) =>
    left.committedAt.localeCompare(right.committedAt)
  );
  const latestTurn = completedTurns.at(-1);
  const pending = Array.isArray(rawPending)
    ? rawPending.flatMap((value) => {
        const job = objectRecord(value);
        const queuedAt = validIsoTimestamp(job?.queuedAt);
        return job && typeof job.jobId === "string" && job.jobId && queuedAt
          ? [{ jobId: job.jobId, queuedAt }]
          : [];
      })
    : [];
  pending.sort((left, right) => left.queuedAt.localeCompare(right.queuedAt));
  const completedJobIds = Array.isArray(rawCompleted)
    ? rawCompleted.filter(
        (value): value is string =>
          typeof value === "string" && /^[a-f0-9]{64}$/i.test(value)
      )
    : [];
  const contextWindow = objectRecord(workspace.contextWindow);
  const measuredAt =
    validIsoTimestamp(metadata.updated_at) ??
    validIsoTimestamp(contextWindow?.updatedAt) ??
    validIsoTimestamp(workspace.queriedAt) ??
    new Date(0).toISOString();
  const connectionUpdatesTotal = isNonNegativeInteger(counters?.plasticity_events)
    ? Number(counters!.plasticity_events)
    : 0;
  const parameterStepsTotal = isNonNegativeInteger(counters?.training_steps)
    ? Number(counters!.training_steps)
    : 0;
  const backgroundState = pending.length > 0
    ? "pending" as const
    : completedJobIds.length > 0
      ? "complete" as const
      : "idle" as const;
  const lastPending = pending.at(-1);
  const lastCompletedJobId = completedJobIds.at(-1);
  return {
    measuredAt,
    fastNeuralMemory: {
      state: latestTurn ? "learned" : "idle",
      safelyStored: Boolean(latestTurn),
      completedTurns: completedTurns.length,
      connectionUpdatesTotal,
      parameterStepsTotal,
      ...(latestTurn
        ? {
            turnId: latestTurn.turnId,
            committedAt: latestTurn.committedAt,
            parameterChecksumAfter: latestTurn.parameterChecksumAfter
          }
        : {})
    },
    backgroundParameters: {
      state: backgroundState,
      pending: pending.length,
      completed: completedJobIds.length,
      updatedAt: lastPending?.queuedAt ?? measuredAt,
      ...(lastPending?.jobId
        ? { lastJobId: lastPending.jobId }
        : lastCompletedJobId
          ? { lastJobId: lastCompletedJobId }
          : {})
    }
  };
}

/**
 * Read the last atomically committed transparent-runtime snapshot without
 * materializing the neural model. The worker persists runtime_card.workspace
 * in the same brain.json generation as its checkpoint, so this is both more
 * stable and substantially cheaper than cold-loading a brain for UI chrome.
 */
export async function persistedWorkspaceSnapshot(
  brainDirectory: string,
  brainId: string
): Promise<WorkspaceSnapshot> {
  const metadataPath = join(brainDirectory, "engine", "brain.json");
  const info = await lstat(metadataPath);
  if (!info.isFile() || info.isSymbolicLink() || info.size > 32 * 1024 * 1024) {
    throw new Error("Committed workspace metadata is not a safe bounded file.");
  }
  const metadata = objectRecord(JSON.parse(await readFile(metadataPath, "utf8")));
  const runtimeCard = objectRecord(metadata?.runtime_card);
  const workspace = objectRecord(runtimeCard?.workspace);
  const contextWindow = objectRecord(workspace?.contextWindow);
  const latentWorkspace = objectRecord(workspace?.latentWorkspace);
  const liquidState = objectRecord(workspace?.liquidState);
  const items = latentWorkspace?.items;
  const validItem = (value: unknown): boolean => {
    const item = objectRecord(value);
    return Boolean(
      item &&
      isFiniteNumber(item.salience) &&
      isNonNegativeInteger(item.rehearsals) &&
      (item.id === undefined || typeof item.id === "string") &&
      (item.kind === undefined || typeof item.kind === "string") &&
      (item.enteredAt === undefined || typeof item.enteredAt === "string") &&
      (item.lastActiveAt === undefined || typeof item.lastActiveAt === "string")
    );
  };
  if (
    metadata?.brain_id !== brainId ||
    workspace?.brainId !== brainId ||
    typeof workspace.queriedAt !== "string" ||
    !Number.isFinite(Date.parse(workspace.queriedAt)) ||
    !contextWindow ||
    !isNonNegativeInteger(contextWindow.capacityTokens) ||
    !isNonNegativeInteger(contextWindow.tokenCount) ||
    typeof contextWindow.tokenHash !== "string" ||
    !isNonNegativeInteger(contextWindow.sensorySlots) ||
    typeof contextWindow.extended !== "boolean" ||
    typeof contextWindow.updatedAt !== "string" ||
    !Number.isFinite(Date.parse(contextWindow.updatedAt)) ||
    !latentWorkspace ||
    !isNonNegativeInteger(latentWorkspace.capacity) ||
    !isNonNegativeInteger(latentWorkspace.occupancy) ||
    !Array.isArray(items) ||
    !items.every(validItem) ||
    !isNonNegativeInteger(latentWorkspace.evictions) ||
    !isNonNegativeInteger(latentWorkspace.rehearsals) ||
    !liquidState ||
    !isNonNegativeInteger(liquidState.dimensions) ||
    !isFiniteNumber(liquidState.mean) ||
    !isFiniteNumber(liquidState.norm) ||
    typeof workspace.hiddenBehavioralPrompt !== "boolean" ||
    typeof workspace.rawLongTermTextInjected !== "boolean"
  ) {
    throw new Error("Committed workspace metadata is invalid.");
  }
  const learning = persistedWorkspaceLearningStatus(metadata, workspace);
  return {
    ...(workspace as unknown as WorkspaceSnapshot),
    runtimeCard: runtimeCard as BrainRuntimeCard,
    ...(learning ? { learning } : {})
  };
}

function workerCounter(value: unknown, label: string): number | undefined {
  if (value === undefined) return undefined;
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < 0 ||
    value > Number.MAX_SAFE_INTEGER
  ) {
    throw new Error(`The neural worker returned an invalid ${label} counter.`);
  }
  return Math.trunc(value);
}

/**
 * Mirror only bounded scalar counters into the desktop document.
 *
 * The worker owns neural tensors and the paged substrate. The desktop owns
 * user-facing source receipts and permissions, so this handoff must not copy
 * worker metadata wholesale or invent source records from a count alone.
 */
function synchronizeWorkerSummary(
  brain: BrainDocument,
  value: unknown
): boolean {
  const summary = objectRecord(value);
  if (!summary) return false;
  if (summary.brainId !== undefined && summary.brainId !== brain.id) {
    throw new Error("The neural worker returned a summary for another brain.");
  }
  const metrics = objectRecord(summary.metrics);
  if (!metrics) return false;
  const counters = objectRecord(metrics.counters) ?? {};
  const next = {
    plasticityEvents: workerCounter(
      metrics.plasticityEvents,
      "plasticity-events"
    ),
    inferenceCount: workerCounter(
      counters.inference_count,
      "inference-count"
    ),
    consolidationCycles: workerCounter(
      counters.consolidation_cycles,
      "consolidation-cycles"
    )
  };
  let changed = false;
  for (const [key, observed] of Object.entries(next) as Array<
    [keyof BrainDocument["counters"], number | undefined]
  >) {
    if (observed === undefined) continue;
    const synchronized = Math.max(brain.counters[key], observed);
    if (synchronized === brain.counters[key]) continue;
    brain.counters[key] = synchronized;
    changed = true;
  }
  return changed;
}

function boundedWorkerText(value: unknown, maximum: number): string | undefined {
  if (typeof value !== "string") return undefined;
  const text = value.replace(/\0/g, "").slice(0, maximum);
  return text || undefined;
}

/** Validate the renderer/model request before it reaches the local worker. */
export function normalizeModalityGenerateRequest(value: unknown): ModalityGenerateRequest {
  const record = objectRecord(value);
  if (!record) throw new Error("Invalid modality generation request.");
  const brainId = record.brainId;
  if (
    typeof brainId !== "string" ||
    !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(brainId)
  ) {
    throw new Error("Invalid modality generation brain id.");
  }
  const modality = record.modality;
  if (!['image', 'audio', 'video', 'vision'].includes(String(modality))) {
    throw new Error("Modality must be image, audio, video, or vision.");
  }
  const prompt = record.prompt;
  if (
    prompt !== undefined &&
    (typeof prompt !== "string" || prompt.includes("\0"))
  ) {
    throw new Error("Modality prompt must be UTF-8 text without NUL bytes; live resource admission applies.");
  }
  const inputPath = record.inputPath;
  if (
    inputPath !== undefined &&
    (typeof inputPath !== "string" || inputPath.length > 32_000 || inputPath.includes("\0"))
  ) {
    throw new Error("Invalid modality input path.");
  }
  const rawConceptIds = record.conceptIds;
  if (
    rawConceptIds !== undefined &&
    (!Array.isArray(rawConceptIds) ||
      rawConceptIds.length > 100_000 ||
      rawConceptIds.some((candidate) =>
        typeof candidate !== "string" ||
        candidate.length > 1_024 ||
        candidate.includes("\0")
      ))
  ) {
    throw new Error("Modality concept ids must be bounded text values.");
  }
  const conceptIdView = record.conceptIdView === undefined ? undefined
    : normalizeConceptIdView(record.conceptIdView, brainId, record.sourceTurnId);
  const neuralActionId = record.neuralActionId;
  if (
    neuralActionId !== undefined &&
    (typeof neuralActionId !== "string" || !/^[a-f0-9]{32}$/i.test(neuralActionId))
  ) {
    throw new Error("Invalid modality neural action id.");
  }
  const seed = record.seed;
  if (seed !== undefined && (typeof seed !== "number" || !Number.isSafeInteger(seed))) {
    throw new Error("Modality seed must be a safe integer.");
  }

  let settings: ModalityGenerateRequest["settings"];
  if (record.settings !== undefined) {
    const rawSettings = objectRecord(record.settings);
    if (!rawSettings) throw new Error("Modality settings must be an object.");
    const allowed = new Set([
      "outputMode",
      "width",
      "height",
      "durationMs",
      "sampleRate",
      "fps",
      "includeAudio",
      "targetLatencyMs",
      "previewIntervalMs"
    ]);
    const unknown = Object.keys(rawSettings).filter((key) => !allowed.has(key));
    if (unknown.length) {
      throw new Error(`Unsupported modality settings: ${unknown.sort().join(", ")}.`);
    }
    const outputMode = rawSettings.outputMode ?? "auto";
    if (!['auto', 'exact', 'legacy'].includes(String(outputMode))) {
      throw new Error("Output mode must be auto, exact, or legacy.");
    }
    const positiveInteger = (
      candidate: unknown,
      label: string,
      maximum = Number.MAX_SAFE_INTEGER
    ): number | undefined => {
      if (candidate === undefined) return undefined;
      if (
        typeof candidate !== "number" ||
        !Number.isSafeInteger(candidate) ||
        candidate < 1 ||
        candidate > maximum
      ) {
        throw new Error(`${label} must be a positive whole number within its container field.`);
      }
      return candidate;
    };
    const positiveNumber = (candidate: unknown, label: string): number | undefined => {
      if (candidate === undefined) return undefined;
      if (
        typeof candidate !== "number" ||
        !Number.isFinite(candidate) ||
        candidate <= 0 ||
        candidate > Number.MAX_SAFE_INTEGER
      ) {
        throw new Error(`${label} must be a finite positive number.`);
      }
      return candidate;
    };
    const width = positiveInteger(rawSettings.width, "Media width");
    const height = positiveInteger(rawSettings.height, "Media height");
    const durationMs = positiveNumber(rawSettings.durationMs, "Media duration");
    const sampleRate = positiveInteger(rawSettings.sampleRate, "Sample rate", 0xffff_ffff);
    const fps = positiveInteger(rawSettings.fps, "Video FPS", 65_535);
    const targetLatencyMs = positiveNumber(rawSettings.targetLatencyMs, "Target latency");
    const previewIntervalMs = positiveNumber(rawSettings.previewIntervalMs, "Preview interval");
    if (
      rawSettings.includeAudio !== undefined &&
      typeof rawSettings.includeAudio !== "boolean"
    ) {
      throw new Error("includeAudio must be boolean.");
    }
    if (outputMode === "exact") {
      const hasExactOutput = modality === "image"
        ? width !== undefined || height !== undefined
        : modality === "audio"
          ? durationMs !== undefined
          : modality === "video"
            ? width !== undefined || height !== undefined || durationMs !== undefined
            : false;
      if (!hasExactOutput) {
        throw new Error("Exact media output requires dimensions or duration.");
      }
    }
    if (modality === "vision" && Object.keys(rawSettings).length) {
      throw new Error("Vision input does not accept generation output settings.");
    }
    settings = {
      outputMode: outputMode as NonNullable<ModalityGenerateRequest["settings"]>["outputMode"],
      ...(width !== undefined ? { width } : {}),
      ...(height !== undefined ? { height } : {}),
      ...(durationMs !== undefined ? { durationMs } : {}),
      ...(sampleRate !== undefined ? { sampleRate } : {}),
      ...(fps !== undefined ? { fps } : {}),
      ...(typeof rawSettings.includeAudio === "boolean"
        ? { includeAudio: rawSettings.includeAudio }
        : {}),
      ...(targetLatencyMs !== undefined ? { targetLatencyMs } : {}),
      ...(previewIntervalMs !== undefined ? { previewIntervalMs } : {})
    };
  }
  return {
    brainId,
    modality: modality as ModalityGenerateRequest["modality"],
    ...(typeof prompt === "string" ? { prompt } : {}),
    ...(Array.isArray(rawConceptIds) ? { conceptIds: [...rawConceptIds] as string[] } : {}),
    ...(conceptIdView ? { conceptIdView, sourceTurnId: conceptIdView.turnId } : {}),
    ...(typeof inputPath === "string" ? { inputPath } : {}),
    ...(settings ? { settings } : {}),
    ...(typeof neuralActionId === "string"
      ? { neuralActionId: neuralActionId.toLocaleLowerCase() }
      : {}),
    ...(typeof seed === "number" ? { seed } : {})
  };
}

export function normalizeModalityPreview(
  value: unknown,
  fallbackRevision = 0,
  outerProgress?: number,
  outerMessage?: string
): ModalityPreview | undefined {
  const record = objectRecord(value);
  if (!record) return undefined;
  if (record.schemaVersion !== undefined && record.schemaVersion !== 1) {
    return undefined;
  }
  const mimeType = boundedWorkerText(record.mimeType ?? record.mime_type, 128);
  const supportedMime =
    mimeType && /^(?:image|audio|video)\/[a-z0-9.+-]{1,80}$/i.test(mimeType)
      ? mimeType.toLocaleLowerCase()
      : undefined;
  const rawDataUrlValue = record.dataUrl ?? record.data_url;
  const rawDataUrl = typeof rawDataUrlValue === "string" &&
    rawDataUrlValue.length <= INLINE_MEDIA_DATA_URL_CHARACTER_LIMIT
      ? boundedWorkerText(rawDataUrlValue, INLINE_MEDIA_DATA_URL_CHARACTER_LIMIT)
      : undefined;
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
  const boundedInteger = (candidate: unknown): number | undefined =>
    typeof candidate === "number" &&
    Number.isSafeInteger(candidate) &&
    candidate >= 0
      ? candidate
      : undefined;
  const boundedNumber = (candidate: unknown): number | undefined =>
    typeof candidate === "number" &&
    Number.isFinite(candidate) &&
    candidate >= 0
      ? candidate
      : undefined;
  const modality = ["image", "audio", "video"].includes(String(record.modality))
    ? record.modality as ModalityPreview["modality"]
    : undefined;
  const stage = [
    "diffusion-vq-decode",
    "codec-waveform",
    "temporal-frame-timeline"
  ].includes(String(record.stage))
    ? record.stage as ModalityPreview["stage"]
    : undefined;
  const hardwareTier = ["micro", "personal", "gpu", "workstation"].includes(
    String(record.hardwareTier)
  )
    ? record.hardwareTier as ModalityPreview["hardwareTier"]
    : undefined;
  const cadence = record.cadence === "hardware-aware-bounded-synchronous"
    ? record.cadence
    : undefined;
  const producer = record.producer === "same-brain-decoder"
    ? record.producer
    : undefined;
  const payloadSha256 = typeof record.payloadSha256 === "string" &&
    /^[a-f0-9]{64}$/i.test(record.payloadSha256)
      ? record.payloadSha256.toLocaleLowerCase()
      : undefined;
  const ideaSource = [
    "manual-prompt-and-active-assemblies",
    "manual-prompt",
    "active-assemblies",
    "active-working-memory",
    "active-liquid-state",
    "intrinsic-neural-cold-start"
  ].includes(String(record.ideaSource))
    ? record.ideaSource as ModalityPreview["ideaSource"]
    : undefined;
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
    schemaVersion: 1,
    revision,
    progress,
    statusLabel,
    mimeType: supportedMime,
    dataUrl,
    path,
    artifactPath,
    ...(modality ? { modality } : {}),
    ...(stage ? { stage } : {}),
    ...(boundedInteger(record.completedUnits) !== undefined
      ? { completedUnits: boundedInteger(record.completedUnits) }
      : {}),
    ...(boundedInteger(record.totalUnits) !== undefined
      ? { totalUnits: boundedInteger(record.totalUnits) }
      : {}),
    ...(boundedInteger(record.previewCount) !== undefined
      ? { previewCount: boundedInteger(record.previewCount) }
      : {}),
    ...(boundedInteger(record.width) !== undefined
      ? { width: boundedInteger(record.width) }
      : {}),
    ...(boundedInteger(record.height) !== undefined
      ? { height: boundedInteger(record.height) }
      : {}),
    ...(boundedInteger(record.sampleCount) !== undefined
      ? { sampleCount: boundedInteger(record.sampleCount) }
      : {}),
    ...(boundedInteger(record.totalSamples) !== undefined
      ? { totalSamples: boundedInteger(record.totalSamples) }
      : {}),
    ...(boundedNumber(record.durationMs) !== undefined
      ? { durationMs: boundedNumber(record.durationMs) }
      : {}),
    ...(boundedInteger(record.sampleRate) !== undefined &&
    Number(record.sampleRate) > 0
      ? { sampleRate: boundedInteger(record.sampleRate) }
      : {}),
    ...(boundedInteger(record.fps) !== undefined && Number(record.fps) > 0
      ? { fps: boundedInteger(record.fps) }
      : {}),
    ...(boundedInteger(record.frameCount) !== undefined
      ? { frameCount: boundedInteger(record.frameCount) }
      : {}),
    ...(boundedInteger(record.totalFrames) !== undefined
      ? { totalFrames: boundedInteger(record.totalFrames) }
      : {}),
    ...(hardwareTier ? { hardwareTier } : {}),
    ...(cadence ? { cadence } : {}),
    ...(producer ? { producer } : {}),
    ...(payloadSha256 ? { payloadSha256 } : {}),
    ...(record.actualDecoderOutput === true
      ? { actualDecoderOutput: true as const }
      : {}),
    ...(record.spatialResolutionReduced === false
      ? { spatialResolutionReduced: false as const }
      : {}),
    ...(record.hardwareScaled === true
      ? { hardwareScaled: true as const }
      : {}),
    ...(record.modelDefinedMaximum === null
      ? { modelDefinedMaximum: null }
      : {}),
    ...(typeof record.trained === "boolean"
      ? { trained: record.trained }
      : {}),
    ...(record.trainingState === "untrained-diagnostic" ||
    record.trainingState === "trained-unverified-quality"
      ? { trainingState: record.trainingState }
      : {}),
    ...(record.semanticQualityClaimed === false
      ? { semanticQualityClaimed: false as const }
      : {}),
    ...(typeof record.partialCoverage === "boolean"
      ? { partialCoverage: record.partialCoverage }
      : {}),
    ...(boundedNumber(record.coveredFraction) !== undefined &&
    Number(record.coveredFraction) <= 1
      ? { coveredFraction: boundedNumber(record.coveredFraction) }
      : {}),
    ...(ideaSource ? { ideaSource } : {}),
    ...(boundedInteger(record.activeAssemblyCount) !== undefined
      ? { activeAssemblyCount: boundedInteger(record.activeAssemblyCount) }
      : {}),
    ...(typeof record.promptProvided === "boolean"
      ? { promptProvided: record.promptProvided }
      : {})
  };
}

export function normalizeChatEngineEvent(
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
  if (event.type === "inline-imagination-started" && typeof event.actionId === "string" &&
      /^[a-f0-9]{32}$/i.test(event.actionId)) {
    return { type: "inline-imagination-started", sequence, actionId: event.actionId };
  }
  if (event.type === "chat-token") {
    const delta = boundedWorkerText(data?.delta, 64 * 1024);
    return delta ? { type: "chat-token", sequence, delta } : undefined;
  }
  if (event.type === "chat-phase") {
    if (
      data?.phase !== "reply-complete-learning" ||
      data.replyComplete !== true ||
      data.turnCommitted !== false ||
      data.learning !== true ||
      data.saving !== true
    ) {
      return undefined;
    }
    return {
      type: "chat-phase",
      sequence,
      phase: "reply-complete-learning",
      replyComplete: true,
      turnCommitted: false,
      learning: true,
      saving: true
    };
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
  transactionKey?: string;
  completedReceipt?: {
    transactionKey?: string;
  };
  recordRecovery?: {
    transactionId?: string;
    transactionKey?: string;
  };
  source?: {
    kind?: string;
    learned_ideas?: number;
    learned_concepts?: number;
    plasticity_events?: number;
    synaptic_update_events?: number;
    parameter_update_steps?: number;
    parameter_checksum_changed?: boolean;
    warnings?: string[];
    coverage?: WorkerTrainingCoverage;
  };
  warnings?: string[];
  coverage?: WorkerTrainingCoverage;
  parameterChecksumBefore?: string;
  parameterChecksumAfter?: string;
  metrics?: Record<string, unknown>;
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

export interface StructuredNeuralExperience {
  content: string;
  name: string;
  sourceLabel: string;
  provenanceUrl?: string;
  license?: string;
  licenseUrl?: string;
}

export interface ConfirmedToolRouteOutcome {
  eventId: string;
  utterance: string;
  toolId: string;
  action: string;
  arguments?: Record<string, unknown>;
}

export interface ToolRouteLearningResult {
  processed: boolean;
  applied: boolean;
  duplicate: boolean;
  ready: boolean;
  steps: number;
  reason?: string;
}

function coverageCount(value: unknown, fallback = 0): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  return Math.max(0, Math.trunc(value));
}

function workerParametersChanged(worker: WorkerIngestResult | undefined): boolean | undefined {
  if (typeof worker?.source?.parameter_checksum_changed === "boolean") {
    return worker.source.parameter_checksum_changed;
  }
  if (
    typeof worker?.parameterChecksumBefore === "string" &&
    typeof worker.parameterChecksumAfter === "string"
  ) {
    return worker.parameterChecksumBefore !== worker.parameterChecksumAfter;
  }
  return undefined;
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
    errorCount: coverageCount(
      raw.errorCount,
      (raw.errors ?? []).reduce(
        (total, error) => total + coverageCount(error.count, 1),
        0
      )
    ),
    errorsTruncated: raw.errorsTruncated === true,
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

interface DurableIngestionBaseline {
  committedRecords: number;
  coverage: TrainingCoverage;
}

async function durableIngestionBaseline(
  brainDirectory: string,
  brainId: string,
  manifestId: string,
  entry: DatasetManifestEntry,
  contentHash: string,
  policy: PersistedDataIngestionPolicy,
  epoch: number,
  expectedCommittedRecords: number
): Promise<DurableIngestionBaseline | undefined> {
  if (expectedCommittedRecords <= 0) return undefined;
  try {
    const metadata = objectRecord(
      JSON.parse(
        await readFile(join(brainDirectory, "engine", "brain.json"), "utf8")
      )
    );
    const checkpoints = objectRecord(metadata?.ingestion_checkpoints);
    if (metadata?.brain_id !== brainId || !checkpoints) return undefined;
    const matches: DurableIngestionBaseline[] = [];
    for (const [identity, value] of Object.entries(checkpoints)) {
      const checkpoint = objectRecord(value);
      const coverageAtCommit = objectRecord(checkpoint?.coverageAtCommit);
      if (
        !checkpoint ||
        !coverageAtCommit ||
        checkpoint.format !== "omni-record-ingestion-checkpoint" ||
        checkpoint.status !== "active" ||
        checkpoint.sourceIdentity !== identity ||
        !/^[a-f0-9]{64}$/iu.test(identity) ||
        checkpoint.contentHash !== contentHash ||
        checkpoint.policy !== policy ||
        checkpoint.epoch !== epoch ||
        checkpoint.sourceBytes !== entry.bytes ||
        checkpoint.resolvedKind !== datasetParserKind(entry) ||
        checkpoint.committedRecords !== expectedCommittedRecords ||
        checkpoint.visitedRecords !== coverageAtCommit.discoveredRecords ||
        checkpoint.processedRecords !== coverageAtCommit.processedRecords ||
        checkpoint.rejectedRecords !== coverageAtCommit.rejectedRecords ||
        checkpoint.processedBytes !== coverageAtCommit.processedBytes
      ) {
        continue;
      }
      const localCoverage = {
        ...normalizeWorkerTrainingCoverage(
          coverageAtCommit as WorkerTrainingCoverage,
          entry.bytes
        ),
        manifestId
      };
      if (
        localCoverage.complete ||
        localCoverage.discoveredRecords !== expectedCommittedRecords ||
        localCoverage.processedRecords + localCoverage.rejectedRecords !==
          localCoverage.discoveredRecords
      ) {
        continue;
      }
      matches.push({
        committedRecords: expectedCommittedRecords,
        coverage: localCoverage
      });
    }
    return matches.length === 1 ? matches[0] : undefined;
  } catch {
    // Telemetry recovery is non-authoritative. Missing or changing metadata
    // cannot block the worker from validating its own durable checkpoint.
    return undefined;
  }
}

interface JobRecord extends RuntimeJob {
  cancelled?: boolean;
}

interface IngestProgressDetail {
  coverage?: TrainingCoverage;
  /** Global classified-record baseline, including the current durable cursor. */
  recordsCompleted?: number;
  /** Present only when the whole manifest/epoch record total is exact. */
  recordsTotal?: number;
  currentRecord?: number;
  committedRecords?: number;
  expectedRecords?: number;
  recordTotalKnown?: boolean;
  checkpointCommitted?: boolean;
  /** Hash-bound manifest-entry transaction. Never a path or raw source id. */
  scopeId?: string;
  foundation?: FoundationTelemetrySample;
  resourceReadings?: TrainingResourceReadings;
  /** True only when processedBytes and discoveredBytes use physical source bytes. */
  physicalSourceBytesComparable?: boolean;
  /** Reconcile rollback/resume counters without deriving a speed from the jump. */
  rateBaseline?: boolean;
  /** This baseline came from the atomically committed engine checkpoint. */
  durableBaseline?: boolean;
  /** True only after the worker reports live dataset traversal measurements. */
  traversalMeasured?: boolean;
}

function telemetryClockMs(): number {
  return Math.floor(performance.timeOrigin + performance.now());
}

function resumedTelemetry(
  value: TrainingTelemetry | undefined,
  atMs: number
): TrainingTelemetry | undefined {
  const normalized = normalizePersistedTrainingTelemetry(value);
  if (!normalized) return undefined;
  try {
    return resumeTrainingTelemetry(normalized, atMs);
  } catch {
    // Telemetry is non-authoritative recovery metadata. A future/stale clock
    // must never block the durable dataset cursor from resuming.
    return undefined;
  }
}

function updateIngestionTelemetry(
  job: JobRecord,
  detail: IngestProgressDetail | undefined,
  atMs: number
): void {
  const coverage = detail?.coverage;
  const observedBeforeBaseline = job.telemetry;
  const physicalBytesComparable =
    detail?.physicalSourceBytesComparable === true;
  const previous = detail?.rateBaseline
    ? undefined
    : physicalBytesComparable || !job.telemetry
      ? job.telemetry
      : (() => {
          const {
            bytesTotal: _bytesTotal,
            bytesPerSecond: _bytesPerSecond,
            totalEtaMs: _totalEtaMs,
            ...withoutByteMeasurements
          } = job.telemetry;
          return {
            ...withoutByteMeasurements,
            sampling: {
              ...withoutByteMeasurements.sampling,
              byteSampleAtMs: withoutByteMeasurements.sampling.lastEventAtMs,
              byteSampleCount: withoutByteMeasurements.bytesCompleted
            }
          };
        })();
  const classifiedRecords = coverage
    ? coverageCount(coverage.processedRecords) + coverageCount(coverage.rejectedRecords)
    : 0;
  const recordsCompleted = Math.max(
    classifiedRecords,
    coverageCount(detail?.recordsCompleted),
    coverageCount(detail?.committedRecords),
    previous?.recordsCompleted ?? 0
  );
  const bytesCompleted = physicalBytesComparable
    ? coverage
      ? coverageCount(coverage.processedBytes)
      : previous?.bytesCompleted ?? 0
    : detail?.durableBaseline && coverage
      ? coverageCount(coverage.processedBytes)
      : observedBeforeBaseline?.bytesCompleted ??
        (coverage ? coverageCount(coverage.processedBytes) : 0);
  const bytesTotal = physicalBytesComparable && coverage
    ? coverageCount(coverage.discoveredBytes)
    : undefined;
  const checkpointScopeId =
    detail?.scopeId ?? previous?.nextCheckpoint.scopeId;
  let updated = updateTrainingTelemetry(previous, {
    atMs,
    ...(checkpointScopeId
      ? { scopeId: checkpointScopeId }
      : {}),
    recordsCompleted,
    bytesCompleted,
    ...(detail?.recordsTotal !== undefined
      ? { recordsTotal: detail.recordsTotal }
      : {}),
    ...(bytesTotal !== undefined ? { bytesTotal } : {}),
    ...(detail?.currentRecord !== undefined
      ? { currentRecord: detail.currentRecord }
      : {}),
    ...(detail?.committedRecords !== undefined
      ? { committedRecords: detail.committedRecords }
      : {}),
    ...(detail?.expectedRecords !== undefined
      ? { expectedRecords: detail.expectedRecords }
      : {}),
    ...(detail?.checkpointCommitted === true
      ? { checkpointCommitted: true }
      : {}),
    ...(detail?.foundation ? { foundation: detail.foundation } : {}),
    ...(detail?.resourceReadings
      ? { resourceReadings: detail.resourceReadings }
      : {})
  });
  if (detail?.rateBaseline && observedBeforeBaseline) {
    const elapsedMs = Math.max(
      observedBeforeBaseline.elapsedMs,
      atMs - Date.parse(observedBeforeBaseline.startedAt)
    );
    updated = {
      ...updated,
      startedAt: new Date(Math.max(0, atMs - elapsedMs)).toISOString(),
      elapsedMs
    };
  }
  job.telemetry = updated;
  job.diskSpace = updated.diskSpace;
}

function incompleteIngestionReason(
  job: JobRecord,
  output: unknown
): string | undefined {
  if (job.kind !== "ingestion" && job.kind !== "crawl") return undefined;
  const result = objectRecord(output);
  if (job.kind === "crawl") {
    if (!result) return "Web crawling ended without a complete frontier coverage receipt.";
    if (result.stopped === true) {
      return "Web crawling paused before complete frontier coverage was committed.";
    }
    if (objectRecord(result.coverage)?.complete !== true) {
      return "Web crawling ended before complete frontier coverage was committed.";
    }
    return undefined;
  }
  if (!result) {
    return "Dataset learning ended without a complete manifest coverage receipt.";
  }
  const pauseReason = typeof result.pauseReason === "string"
    ? result.pauseReason.replace(/\0/g, "").replace(/\s+/g, " ").trim().slice(0, 2_000)
    : "";
  if (result.paused === true) {
    return pauseReason || "Dataset learning paused before the current source was committed.";
  }
  const coverage = objectRecord(result.coverage);
  // A successful ingestion must prove its durable manifest traversal. Missing
  // or ambiguous coverage is not success: fail closed so Build cannot unlock
  // chat after a worker/protocol regression drops the completion receipt.
  if (!coverage || coverage.complete !== true) {
    return pauseReason ||
      "Dataset learning ended before complete manifest coverage was committed.";
  }
  return undefined;
}

interface LiveObservationRecord {
  session: LiveObservationSession;
  lastReceivedSequence: number;
  lastReceivedTimestampMs: number;
  pending: Set<Promise<LiveObservationPacketResult>>;
  controls: Map<string, LiveObservationControl>;
  controlBurstFrames: Map<string, Set<number>>;
}

interface WorkerObservationResult {
  session?: unknown;
  observation?: unknown;
  actions?: unknown;
  trace?: unknown;
}

const LIVE_VISUAL_PACKET_MIME_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp"
]);
const LIVE_AUDIO_PACKET_MIME_TYPES = new Set([
  "audio/pcm-f32le",
  "audio/x-pcm-f32le",
  "audio/pcm-s16le",
  "audio/x-pcm-s16le",
  "audio/wav",
  "audio/x-wav",
  "audio/wave",
  "audio/webm",
  "audio/ogg",
  "audio/opus",
  "audio/mp4",
  "audio/aac"
]);

function cloneObservationSession(
  session: LiveObservationSession
): LiveObservationSession {
  return {
    ...session,
    modalities: [...session.modalities],
    permission: { ...session.permission },
    capabilities: { ...session.capabilities },
    capture: { ...session.capture }
  };
}

function livePacketBytes(value: unknown): Buffer {
  if (Buffer.isBuffer(value)) return Buffer.from(value);
  if (value instanceof Uint8Array) {
    return Buffer.from(value.buffer, value.byteOffset, value.byteLength);
  }
  throw new Error("A live observation packet must contain binary bytes.");
}

function livePacketMimeType(
  modality: LiveObservationModality,
  value: unknown
): string {
  if (typeof value !== "string" || value.length > 128) {
    throw new Error("A live observation packet has an invalid MIME type.");
  }
  const normalized = value.trim().toLocaleLowerCase();
  const base = normalized.split(";", 1)[0] ?? "";
  const supported = modality === "audio"
    ? LIVE_AUDIO_PACKET_MIME_TYPES
    : LIVE_VISUAL_PACKET_MIME_TYPES;
  if (!supported.has(base)) {
    throw new Error(`Unsupported live ${modality} packet MIME type.`);
  }
  return normalized;
}

function finiteObservationNumber(
  value: unknown,
  fallback = 0
): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function optionalPositiveSafeInteger(
  value: unknown,
  label: string
): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value) || Number(value) < 1) {
    throw new Error(`${label} must be a positive safe integer.`);
  }
  return Number(value);
}

function normalizeObservationCapture(
  request: LiveObservationSessionStartRequest,
  modalities: LiveObservationModality[]
): LiveObservationSession["capture"] {
  const raw = request.capture;
  const mode = raw?.mode ?? "auto";
  const benchmarkClass = raw?.benchmarkClass ?? "fallback";
  if (!(["auto", "motion", "balanced", "detail"] as const).includes(mode)) {
    throw new Error("Invalid live observation capture mode.");
  }
  if (!(["fallback", "balanced", "high", "native"] as const).includes(benchmarkClass)) {
    throw new Error("Invalid live observation benchmark class.");
  }
  const sourceNativeWidth = optionalPositiveSafeInteger(
    raw?.sourceNativeWidth,
    "Source-native width"
  );
  const sourceNativeHeight = optionalPositiveSafeInteger(
    raw?.sourceNativeHeight,
    "Source-native height"
  );
  const sourceNativeFps = optionalPositiveSafeInteger(
    raw?.sourceNativeFps,
    "Source-native FPS"
  );
  let width = optionalPositiveSafeInteger(raw?.width, "Capture width");
  let height = optionalPositiveSafeInteger(raw?.height, "Capture height");
  let fps = optionalPositiveSafeInteger(raw?.fps, "Capture FPS");
  const audioSampleRate = optionalPositiveSafeInteger(
    raw?.audioSampleRate,
    "Capture audio sample rate"
  );
  const visual = modalities.some((value) => value !== "audio");
  if (visual && benchmarkClass === "fallback") {
    width = Math.min(320, sourceNativeWidth ?? 320);
    height = Math.min(180, sourceNativeHeight ?? 180);
    fps = Math.min(1, sourceNativeFps ?? 1);
  }
  if (
    (sourceNativeWidth !== undefined && width !== undefined && width > sourceNativeWidth) ||
    (sourceNativeHeight !== undefined && height !== undefined && height > sourceNativeHeight) ||
    (sourceNativeFps !== undefined && fps !== undefined && fps > sourceNativeFps)
  ) {
    throw new Error("Negotiated capture values exceed the source-native bounds.");
  }
  return {
    mode,
    ...(width !== undefined ? { width } : {}),
    ...(height !== undefined ? { height } : {}),
    ...(fps !== undefined ? { fps } : {}),
    ...(sourceNativeWidth !== undefined ? { sourceNativeWidth } : {}),
    ...(sourceNativeHeight !== undefined ? { sourceNativeHeight } : {}),
    ...(sourceNativeFps !== undefined ? { sourceNativeFps } : {}),
    ...(audioSampleRate !== undefined ? { audioSampleRate } : {}),
    benchmarkClass,
    revision: 0
  };
}

function liveObservationTransportEnvelope(
  capture: LiveObservationSession["capture"],
  modalities: LiveObservationModality[]
): { maxPacketBytes: number; maxInFlight: number } {
  const heap = getHeapStatistics();
  const heapAvailable = Math.max(
    1,
    heap.heap_size_limit - process.memoryUsage().heapUsed - memoryReserveBytes()
  );
  const systemAvailable = Math.max(1, freemem() - memoryReserveBytes());
  // Live transport may use only a measured fraction of both reserves. This is
  // a resource watermark, not a resolution, FPS, packet, or queue product cap.
  const resourceEnvelope = Math.max(
    1,
    // Base64 JSON-RPC temporarily expands a packet; reserve enough room for
    // the encoded line, decoded bytes, and the active neural tensor together.
    Math.floor(Math.min(heapAvailable / 6, systemAvailable / 12))
  );
  let sourcePacketBudget = 0;
  if (modalities.some((value) => value !== "audio")) {
    const width = capture.sourceNativeWidth ?? capture.width;
    const height = capture.sourceNativeHeight ?? capture.height;
    if (width !== undefined && height !== undefined) {
      const decodedBytes = BigInt(width) * BigInt(height) * 4n;
      sourcePacketBudget = Number(
        decodedBytes > BigInt(Number.MAX_SAFE_INTEGER)
          ? BigInt(Number.MAX_SAFE_INTEGER)
          : decodedBytes
      );
    }
  }
  // Independently decodable audio chunks have no image-like source dimensions;
  // the live resource envelope is their honest bound.
  if (modalities.includes("audio")) sourcePacketBudget = resourceEnvelope;
  if (sourcePacketBudget <= 0) sourcePacketBudget = resourceEnvelope;
  const maxPacketBytes = Math.max(
    1,
    Math.min(resourceEnvelope, sourcePacketBudget)
  );
  const resourceParallelism = Math.max(
    1,
    Math.floor(resourceEnvelope / Math.max(1, maxPacketBytes * 2))
  );
  const maxInFlight = capture.benchmarkClass === "fallback"
    ? 1
    : Math.max(1, Math.min(availableParallelism(), resourceParallelism));
  return { maxPacketBytes, maxInFlight };
}

interface MergeFileCandidate extends AgentMergeFilePreview {
  absoluteSourcePath?: string;
  blobHash?: string;
  evidenceFingerprint?: string;
}

interface MergePlan {
  source: BrainDocument;
  target: BrainDocument;
  evidenceDigest: string;
  newEvidence: number;
  duplicateEvidence: number;
  files: MergeFileCandidate[];
  substrate: AgentMergeSubstratePreview;
  preview: AgentMergePreview;
}

interface StoredMergePlanDescriptor {
  schemaVersion: 1;
  sourceBrainId: string;
  targetBrainId: string;
  sourceUpdatedAt: string;
  targetUpdatedAt: string;
  evidenceDigest: string;
  newEvidence: number;
  duplicateEvidence: number;
  files: MergeFileCandidate[];
  substrate: AgentMergeSubstratePreview;
  preview: AgentMergePreview;
}

function mergedEvidenceSourceId(
  targetBrainId: string,
  fingerprint: string,
  sourceId: string
): string {
  return `merge-${sha256(`${targetBrainId}\0${fingerprint}\0${sourceId}`).slice(0, 58)}`;
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function datasetTransactionKey(
  manifestId: string,
  entryIndex: number,
  epoch: number,
  contentHash: string,
  policy: PersistedDataIngestionPolicy,
  runId?: string
): string {
  return sha256(
    [
      runId ? "omni-dataset-entry-v2" : "omni-dataset-entry-v1",
      ...(runId ? [runId] : []),
      manifestId, entryIndex, epoch, contentHash, policy
    ].join("\0")
  );
}

function boundedReceiptWarnings(values: readonly string[]): string[] {
  return values
    .slice(0, 64)
    .map((value) => String(value).replace(/\0/g, "").slice(0, 4_096));
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
  const recipe = target.config.memoryRecipe ?? "adaptive-retention";
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
      ".heif",
      ".jp2",
      ".j2k",
      ".jpf",
      ".jpx",
      ".jxl",
      ".raw",
      ".dng"
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
      ".mka"
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
      ".vob"
    ].includes(extension)
  ) {
    return "video";
  }
  return "unknown";
}

/**
 * Preserve the exact streaming parser contract from the committed manifest.
 * TrainingSource.kind is intentionally coarser for the UI/ledger (for example,
 * JSONL is recorded as JSON), but sending that coarse value to the worker turns
 * a multi-gigabyte JSONL file into one JSON document and makes TSV use commas.
 */
function datasetParserKind(entry: DatasetManifestEntry): string {
  switch (entry.format) {
    case "epub":
    case "office":
    case "archive":
    case "webdataset":
      return "archive";
    case "huggingface":
      // "dataset" asks the worker to inspect the referenced manifest/shards.
      return "dataset";
    default:
      return entry.format;
  }
}

function workerText(result: WorkerChatResult | undefined): string | undefined {
  if (!result) return undefined;
  for (const candidate of [result.text, result.response, result.content]) {
    if (typeof candidate === "string" && candidate.trim()) return candidate.trim();
  }
  return undefined;
}

function receiptTimestamp(value: unknown, label: string): string {
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value))) {
    throw new Error(`Committed chat ${label} timestamp is invalid.`);
  }
  return new Date(value).toISOString();
}

function receiptIdentifier(value: unknown, label: string): string {
  if (
    typeof value !== "string" ||
    !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(value)
  ) {
    throw new Error(`Committed chat ${label} id is invalid.`);
  }
  return value;
}

function receiptCount(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || Number(value) < 0) {
    throw new Error(`Committed chat ${label} counter is invalid.`);
  }
  return Number(value);
}

function receiptAttentionEpoch(...values: unknown[]): number {
  const present = values.filter((value) => value !== undefined);
  if (!present.length) return 0;
  if (
    present.some(
      (value) => typeof value !== "number" ||
        !Number.isSafeInteger(value) ||
        value < 0
    ) ||
    present.some((value) => value !== present[0])
  ) {
    throw new Error("Committed chat attention epoch is inconsistent.");
  }
  return Number(present[0]);
}

function generationPresentationEnd(human: Record<string, unknown>, assistant: Record<string, unknown>,
  trace: Record<string, unknown>, receiptEnd?: unknown): ChatGenerationEnd | undefined {
  const markers = [human.generation_end, assistant.generation_end, receiptEnd];
  const disposition = trace.generation_stop_reason === "steered" ? "steered" :
    trace.generation_stop_reason === "native-action-stop" ? "native-stop" :
      trace.generation_stop_reason === "no-reply" ? "no-reply" : undefined;
  if (markers.some((marker) => marker !== undefined && marker !== disposition)) {
    throw new Error("Committed chat generation disposition is inconsistent.");
  }
  if (disposition === "no-reply" && (
    markers.some((marker) => marker !== "no-reply") ||
    assistant.content !== "" ||
    !["no-generated-tokens", "no-decoded-text", "whitespace-only", "no-printable-text"]
      .includes(String(trace.generation_no_reply_reason)) ||
    trace.generation_printable_text_characters !== 0 ||
    !Number.isSafeInteger(trace.generated_token_count) || Number(trace.generated_token_count) < 0 ||
    (trace.generation_no_reply_reason === "no-generated-tokens") !== (trace.generated_token_count === 0) ||
    typeof trace.generation_decoder_stop_reason !== "string" || !trace.generation_decoder_stop_reason
  )) {
    throw new Error("Committed no-reply disposition is not bound to saved zero-text evidence.");
  }
  return disposition;
}

function validateCommittedChatReceipt(
  value: WorkerChatReceiptResult,
  expected: {
    brainId: string;
    turnId: string;
    input: string;
    inputSha256: string;
    minimumInferenceCount: number;
  }
): ValidatedChatReceipt | undefined {
  if (
    value.format !== "omni-chat-turn-receipt-query" ||
    value.formatVersion !== 1 ||
    value.brainId !== expected.brainId ||
    value.turnId !== expected.turnId ||
    value.inputSha256 !== expected.inputSha256
  ) {
    throw new Error("Committed chat receipt identity is invalid.");
  }
  const inferenceCount = receiptCount(value.inferenceCount, "inference");
  if (value.committed !== true) {
    if (value.committed !== false || value.turnCommitted !== false) {
      throw new Error("Committed chat receipt state is invalid.");
    }
    return undefined;
  }
  if (
    value.turnCommitted !== true ||
    value.idempotentCompletion !== true ||
    typeof value.legacyMatched !== "boolean" ||
    inferenceCount !== expected.minimumInferenceCount + 1 ||
    typeof value.parameterChecksumAfter !== "string" ||
    !/^[a-f0-9]{64}$/.test(value.parameterChecksumAfter) ||
    typeof value.substrateGeneration !== "string" ||
    !/^[a-f0-9]{64}$/.test(value.substrateGeneration) ||
    typeof value.mutableStateGeneration !== "string" ||
    !/^[a-f0-9]{64}$/.test(value.mutableStateGeneration)
  ) {
    throw new Error("Committed chat receipt proof is invalid.");
  }
  const engineUpdatedAt = receiptTimestamp(value.engineUpdatedAt, "engine");
  const human = objectRecord(value.humanMessage);
  const assistant = objectRecord(value.brainMessage);
  const rawTrace = objectRecord(value.trace);
  if (!human || !assistant || !rawTrace) {
    throw new Error("Committed chat receipt presentation is invalid.");
  }
  const humanId = receiptIdentifier(human.id, "human message");
  const brainMessageId = receiptIdentifier(assistant.id, "brain message");
  const traceId = receiptIdentifier(rawTrace.id, "trace");
  const humanContent = human.content;
  const brainContent = assistant.content;
  const generationEnd = generationPresentationEnd(human, assistant, rawTrace, value.generationEnd);
  if ((value.noReply === true) !== (generationEnd === "no-reply")) {
    throw new Error("Committed no-reply flag is not bound to its durable receipt.");
  }
  if (
    human.role !== "human" ||
    assistant.role !== "brain" ||
    typeof humanContent !== "string" ||
    !humanContent.trim() ||
    humanContent.includes("\0") ||
    humanContent !== expected.input ||
    sha256(humanContent) !== expected.inputSha256 ||
    typeof brainContent !== "string" ||
    (generationEnd === "no-reply" ? brainContent !== "" :
      (!brainContent.length || (!generationEnd && !brainContent.trim()))) ||
    brainContent.includes("\0") ||
    assistant.traceId !== traceId ||
    rawTrace.input_sha256 !== expected.inputSha256 ||
    rawTrace.parameter_checksum_after !== value.parameterChecksumAfter ||
    typeof rawTrace.parameter_checksum_before !== "string" ||
    !/^[a-f0-9]{64}$/.test(rawTrace.parameter_checksum_before)
  ) {
    throw new Error("Committed chat receipt content binding is invalid.");
  }
  if (
    (value.legacyMatched === false && rawTrace.turn_id !== expected.turnId) ||
    (value.legacyMatched === true && rawTrace.turn_id !== undefined)
  ) {
    throw new Error("Committed chat receipt turn binding is invalid.");
  }
  const humanCreatedAt = receiptTimestamp(human.createdAt, "human message");
  const brainCreatedAt = receiptTimestamp(assistant.createdAt, "brain message");
  const traceCreatedAt = receiptTimestamp(
    rawTrace.created_at ?? rawTrace.createdAt,
    "trace"
  );
  const epoch = receiptAttentionEpoch(
    human.attention_epoch ?? human.attentionEpoch,
    assistant.attention_epoch ?? assistant.attentionEpoch,
    rawTrace.attention_epoch ?? rawTrace.attentionEpoch
  );
  if (
    Date.parse(brainCreatedAt) < Date.parse(humanCreatedAt) ||
    Date.parse(traceCreatedAt) < Date.parse(brainCreatedAt) ||
    Date.parse(engineUpdatedAt) < Date.parse(traceCreatedAt)
  ) {
    throw new Error("Committed chat receipt timestamps are inconsistent.");
  }
  const rawSteps = rawTrace.steps;
  if (rawSteps !== undefined && !Array.isArray(rawSteps)) {
    throw new Error("Committed chat trace steps are invalid.");
  }
  const steps = (rawSteps ?? []).flatMap((value) => {
    const step = objectRecord(value);
    if (
      !step ||
      typeof step.stage !== "string" ||
      typeof step.detail !== "string" ||
      step.stage.length > 200 ||
      step.detail.length > 4_000 ||
      (step.value !== undefined && typeof step.value !== "string")
    ) {
      throw new Error("Committed chat trace step is invalid.");
    }
    return [{
      stage: step.stage,
      detail: step.detail,
      ...(typeof step.value === "string"
        ? { value: step.value.slice(0, 4_000) }
        : {})
    }];
  }).slice(0, 100);
  const finiteOptional = (field: string): number | undefined => {
    const candidate = rawTrace[field];
    if (candidate === undefined) return undefined;
    if (typeof candidate !== "number" || !Number.isFinite(candidate)) {
      throw new Error(`Committed chat trace ${field} is invalid.`);
    }
    return candidate;
  };
  const note = rawTrace.note;
  if (note !== undefined && typeof note !== "string") {
    throw new Error("Committed chat trace note is invalid.");
  }
  return {
    humanMessage: {
      id: humanId,
      role: "human",
      content: humanContent,
      createdAt: humanCreatedAt,
      ...(value.legacyMatched === false ? { turnId: expected.turnId } : {}),
      runtime: "adaptive-core",
      status: "complete",
      attentionEpoch: epoch,
      ...(generationEnd ? { generationEnd } : {})
    },
    brainMessage: {
      id: brainMessageId,
      role: "brain",
      content: brainContent,
      createdAt: brainCreatedAt,
      ...(value.legacyMatched === false ? { turnId: expected.turnId } : {}),
      traceId,
      runtime: "adaptive-core",
      status: "complete",
      attentionEpoch: epoch,
      ...(generationEnd ? { generationEnd } : {})
    },
    trace: {
      id: traceId,
      created_at: traceCreatedAt,
      seed: finiteOptional("seed"),
      parameter_checksum_before: rawTrace.parameter_checksum_before,
      parameter_checksum_after: value.parameterChecksumAfter,
      parameter_delta_norm: finiteOptional("parameter_delta_norm"),
      core_parameter_delta_norm: finiteOptional("parameter_delta_norm"),
      parameter_delta_scope: "decoder, memory_bridge, idea_adapter, liquid only; signed ternary-level/control-value net L2",
      substrate_parameter_delta_norm: null,
      substrate_parameter_delta_measured: false,
      parameter_checksum_scope: "module-registered learned tensors; excludes VSA vectors and sparse substrate edges",
      stdp_update: finiteOptional("stdp_update"),
      spike_rate: finiteOptional("spike_rate"),
      train_loss: finiteOptional("train_loss"),
      generation_entropy: finiteOptional("generation_entropy"),
      ponder_steps: finiteOptional("ponder_steps"),
      steps,
      note: typeof note === "string" ? note.slice(0, 4_000) : undefined,
      attention_epoch: epoch,
      generation_stop_reason: typeof rawTrace.generation_stop_reason === "string" ? rawTrace.generation_stop_reason : undefined,
      generation_decoder_stop_reason: typeof rawTrace.generation_decoder_stop_reason === "string" ? rawTrace.generation_decoder_stop_reason : undefined,
      generation_no_reply_reason: typeof rawTrace.generation_no_reply_reason === "string" ? rawTrace.generation_no_reply_reason : undefined,
      generated_token_count: finiteOptional("generated_token_count"),
      generation_printable_text_characters: finiteOptional("generation_printable_text_characters")
    },
    inferenceCount,
    plasticityEvents: receiptCount(value.plasticityEvents, "plasticity"),
    consolidationCycles: receiptCount(
      value.consolidationCycles,
      "consolidation"
    )
  };
}

export function validateWorkerChatPresentation(
  value: WorkerChatResult,
  expected: {
    turnId: string;
    input: string;
    inputSha256: string;
    response: string;
  }
): ValidatedWorkerChatPresentation | undefined {
  const fields = [value.humanMessage, value.message, value.turnReceipt];
  if (fields.every((field) => field === undefined)) return undefined;
  const human = objectRecord(value.humanMessage);
  const assistant = objectRecord(value.message);
  const receipt = objectRecord(value.turnReceipt);
  const trace = objectRecord(value.trace);
  if (
    !human ||
    !assistant ||
    !receipt ||
    !trace ||
    value.turnCommitted !== true ||
    typeof value.idempotentCompletion !== "boolean" ||
    receipt.format !== "omni-completed-chat-turn" ||
    receipt.formatVersion !== 1 ||
    receipt.turnId !== expected.turnId ||
    receipt.inputSha256 !== expected.inputSha256 ||
    human.turn_id !== expected.turnId ||
    assistant.turn_id !== expected.turnId ||
    trace.turn_id !== expected.turnId
  ) {
    throw new Error("The neural worker returned an invalid committed turn presentation.");
  }
  const generationEnd = generationPresentationEnd(human, assistant, trace, receipt.generationEnd);
  if (generationEnd && (human.generation_end !== generationEnd ||
      assistant.generation_end !== generationEnd || receipt.generationEnd !== generationEnd)) {
    throw new Error("Neural output is not bound to its durable disposition receipt.");
  }
  if ((value.steered === true) !== (generationEnd === "steered")) {
    throw new Error("The neural worker steering flag is not bound to its saved trace.");
  }
  if ((value.nativeStopped === true) !== (generationEnd === "native-stop")) {
    throw new Error("The neural worker native stop flag is not bound to its saved trace.");
  }
  if ((value.noReply === true) !== (generationEnd === "no-reply")) {
    throw new Error("The neural worker no-reply flag is not bound to its saved trace.");
  }
  const humanId = receiptIdentifier(human.id, "worker human message");
  const brainMessageId = receiptIdentifier(
    assistant.id,
    "worker brain message"
  );
  const traceId = receiptIdentifier(trace.id, "worker trace");
  if (
    receipt.humanMessageId !== humanId ||
    receipt.brainMessageId !== brainMessageId ||
    receipt.traceId !== traceId ||
    human.role !== "human" ||
    assistant.role !== "brain" ||
    human.content !== expected.input ||
    sha256(String(human.content)) !== expected.inputSha256 ||
    assistant.content !== expected.response ||
    (generationEnd === "no-reply" ? expected.response !== "" :
      (!expected.response.length || (!generationEnd && !expected.response.trim()))) ||
    trace.input_sha256 !== expected.inputSha256 ||
    typeof receipt.parameterChecksumAfter !== "string" ||
    !/^[a-f0-9]{64}$/.test(receipt.parameterChecksumAfter) ||
    trace.parameter_checksum_after !== receipt.parameterChecksumAfter
  ) {
    throw new Error("The neural worker committed turn presentation is not receipt-bound.");
  }
  const humanCreatedAt = receiptTimestamp(
    human.created_at ?? human.createdAt,
    "worker human message"
  );
  const brainCreatedAt = receiptTimestamp(
    assistant.created_at ?? assistant.createdAt,
    "worker brain message"
  );
  const traceCreatedAt = receiptTimestamp(
    trace.created_at ?? trace.createdAt,
    "worker trace"
  );
  const committedAt = receiptTimestamp(receipt.committedAt, "worker receipt");
  const epoch = receiptAttentionEpoch(
    human.attention_epoch ?? human.attentionEpoch,
    assistant.attention_epoch ?? assistant.attentionEpoch,
    trace.attention_epoch ?? trace.attentionEpoch
  );
  if (
    Date.parse(brainCreatedAt) < Date.parse(humanCreatedAt) ||
    Date.parse(traceCreatedAt) < Date.parse(brainCreatedAt) ||
    Date.parse(committedAt) < Date.parse(traceCreatedAt)
  ) {
    throw new Error("The neural worker committed turn timestamps are inconsistent.");
  }
  const inferenceCount = receiptCount(
    receipt.inferenceCount,
    "worker inference"
  );
  if (inferenceCount < 1) {
    throw new Error("The neural worker committed turn counter is invalid.");
  }
  return {
    humanMessage: {
      id: humanId,
      role: "human",
      content: expected.input,
      createdAt: humanCreatedAt,
      turnId: expected.turnId,
      runtime: "adaptive-core",
      status: "complete",
      attentionEpoch: epoch,
      ...(generationEnd ? { generationEnd } : {})
    },
    brainMessage: {
      id: brainMessageId,
      role: "brain",
      content: expected.response,
      createdAt: brainCreatedAt,
      turnId: expected.turnId,
      traceId,
      runtime: "adaptive-core",
      status: "complete",
      attentionEpoch: epoch,
      ...(generationEnd ? { generationEnd } : {})
    },
    inferenceCount
  };
}

function applyWorkerTracePresentation(
  result: ChatResult,
  trace: NonNullable<WorkerChatResult["trace"]> | undefined
): void {
  result.humanMessage.runtime = "adaptive-core";
  result.brainMessage.runtime = "adaptive-core";
  result.trace.runtime = "adaptive-core";
  if (!trace) return;
  if (trace.id) {
    result.trace.id = trace.id;
    result.brainMessage.traceId = trace.id;
  }
  if (trace.created_at) result.trace.createdAt = trace.created_at;
  if (typeof trace.attention_epoch === "number") {
    result.trace.attentionEpoch = trace.attention_epoch;
  }
  const disposition = result.brainMessage.generationEnd;
  if (disposition || trace.generation_decoder_stop_reason !== undefined || trace.generated_token_count !== undefined) {
    result.trace.generation = {
      ...(disposition ? { disposition } : {}),
      ...(typeof trace.generation_decoder_stop_reason === "string" ? { decoderStopReason: trace.generation_decoder_stop_reason } : {}),
      ...(Number.isSafeInteger(trace.generated_token_count) && Number(trace.generated_token_count) >= 0
        ? { generatedTokenCount: trace.generated_token_count } : {}),
      ...(Number.isSafeInteger(trace.generation_printable_text_characters) && Number(trace.generation_printable_text_characters) >= 0
        ? { printableTextCharacters: trace.generation_printable_text_characters } : {}),
      ...(["no-generated-tokens", "no-decoded-text", "whitespace-only", "no-printable-text"].includes(String(trace.generation_no_reply_reason))
        ? { noReplyReason: trace.generation_no_reply_reason as ChatNoReplyReason } : {})
    };
  }
  if (typeof trace.seed === "number") result.trace.seed = trace.seed;
  if (trace.steps) {
    result.trace.steps = trace.steps
      .filter(
        (step): step is { stage: string; detail: string; value?: string } =>
          typeof step.stage === "string" && typeof step.detail === "string"
      )
      .slice(0, 100);
  }
  const mutations = [
    typeof trace.parameter_delta_norm === "number"
      ? `core-module net delta ${trace.parameter_delta_norm.toExponential(4)} (VSA/edge net delta unmeasured)`
      : undefined,
    typeof trace.stdp_update === "number"
      ? `STDP activity ${trace.stdp_update.toExponential(4)}`
      : undefined,
    typeof trace.train_loss === "number"
      ? `loss ${trace.train_loss.toFixed(6)}`
      : undefined,
    trace.parameter_checksum_before && trace.parameter_checksum_after
      ? `${trace.parameter_checksum_before.slice(0, 12)} → ${
          trace.parameter_checksum_after.slice(0, 12)
        }`
      : undefined
  ].filter((value): value is string => Boolean(value));
  result.trace.parameterDiagnostics = {
    scope: "decoder, memory_bridge, idea_adapter, liquid only",
    ...(typeof trace.parameter_delta_norm === "number" ? { coreDeltaNorm: trace.parameter_delta_norm } : {}),
    substrateDeltaNorm: null, substrateDeltaMeasured: false,
    checksumScope: "module-registered learned tensors; excludes VSA vectors and sparse substrate edges"
  };
  if (mutations.length > 0) {
    result.trace.steps.push({
      stage: "measured-core-and-stdp-activity",
      detail: mutations.join("; ")
    });
  }
  if (typeof trace.ponder_steps === "number") {
    result.trace.branches = Math.max(1, Math.round(trace.ponder_steps));
    result.trace.selectedBranch = result.trace.branches - 1;
  }
  if (trace.note) result.trace.note = trace.note;
}

function mergeCommittedChatPresentation(
  brain: BrainDocument,
  humanMessage: ChatMessage,
  brainMessage: ChatMessage,
  workerTrace: NonNullable<WorkerChatResult["trace"]>
): ChatResult {
  const existingHuman = brain.messages.find(
    (message) => message.id === humanMessage.id
  );
  const existingBrain = brain.messages.find(
    (message) => message.id === brainMessage.id
  );
  const existingTrace = brain.traces.find(
    (trace) => trace.id === workerTrace.id
  );
  const existingCount = [existingHuman, existingBrain, existingTrace].filter(
    Boolean
  ).length;
  let result: ChatResult;
  if (existingCount === 3) {
    if (
      existingHuman?.role !== "human" ||
      existingHuman.content !== humanMessage.content ||
      existingHuman.createdAt !== humanMessage.createdAt ||
      (humanMessage.generationEnd === "no-reply" && existingHuman.generationEnd !== "no-reply") ||
      existingBrain?.role !== "brain" ||
      existingBrain.content !== brainMessage.content ||
      existingBrain.createdAt !== brainMessage.createdAt ||
      (brainMessage.generationEnd === "no-reply" && existingBrain.generationEnd !== "no-reply") ||
      existingBrain.traceId !== workerTrace.id ||
      existingTrace?.input !== humanMessage.content ||
      Date.parse(existingTrace.createdAt) !== Date.parse(workerTrace.created_at ?? "")
    ) {
      throw new Error("Committed chat receipt conflicts with outer state.");
    }
    result = {
      brain,
      humanMessage: existingHuman,
      brainMessage: existingBrain,
      trace: existingTrace
    };
  } else if (existingCount === 1 && existingTrace && brain.messages.length > 0) {
    // Stable desktop builds before exact message-id adoption already stored
    // the worker trace id. Upgrade that one proven adjacent pair in place;
    // appending it would display the same committed turn twice.
    const candidates = brain.messages
      .map((message, index) => ({ message, index }))
      .filter(({ message }) => message.traceId === existingTrace.id);
    const candidate = candidates.length === 1 ? candidates[0] : undefined;
    const prior = candidate ? brain.messages[candidate.index - 1] : undefined;
    if (
      !candidate ||
      !prior ||
      prior.role !== "human" ||
      prior.content !== humanMessage.content ||
      candidate.message.role !== "brain" ||
      candidate.message.content !== brainMessage.content ||
      existingTrace.input !== humanMessage.content
    ) {
      throw new Error("Committed chat receipt conflicts with outer state.");
    }
    brain.messages.splice(
      candidate.index - 1,
      2,
      humanMessage,
      brainMessage
    );
    result = {
      brain,
      humanMessage,
      brainMessage,
      trace: existingTrace
    };
  } else if (
    existingCount === 0 ||
    (existingCount === 1 && existingTrace && brain.messages.length === 0)
  ) {
    result = recordNeuralChat(
      brain,
      humanMessage.content,
      brainMessage.content,
      brainMessage.generationEnd
    );
    result.brain.messages.splice(-2, 2, humanMessage, brainMessage);
    result.humanMessage = humanMessage;
    result.brainMessage = brainMessage;
  } else {
    throw new Error("Committed chat receipt conflicts with outer state.");
  }
  applyWorkerTracePresentation(result, workerTrace);
  return result;
}

interface ChatReconciliationRequest {
  turnId: string;
  input: string;
  inputSha256: string;
  minimumInferenceCount: number;
}

async function committedEngineMetadata(
  brainDirectory: string
): Promise<Record<string, unknown> | undefined> {
  const metadataPath = join(brainDirectory, "engine", "brain.json");
  try {
    const info = await lstat(metadataPath);
    if (!info.isFile() || info.isSymbolicLink() || info.size > 32 * 1024 * 1024) {
      throw new Error("Committed chat engine metadata is not a safe bounded file.");
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
  return objectRecord(JSON.parse(await readFile(metadataPath, "utf8")));
}

function committedLedgerPayload(
  brainDirectory: string,
  brainId: string,
  kind: "message" | "trace",
  identifier: string
): Record<string, unknown> | undefined {
  const path = join(brainDirectory, "engine", "conversation.sqlite3");
  if (!existsSync(path)) return undefined;
  let database: DatabaseSync;
  try {
    database = new DatabaseSync(path, { readOnly: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
  try {
    const identity = database.prepare(
      "SELECT value FROM meta WHERE key='brain_id'"
    ).get() as { value?: unknown } | undefined;
    if (identity?.value !== brainId) {
      throw new Error("Committed chat ledger belongs to another brain.");
    }
    const row = database.prepare(
      "SELECT payload_json,payload_sha256 FROM entries WHERE entry_key=? AND kind=?"
    ).get(`${kind}:${identifier}`, kind) as {
      payload_json?: unknown;
      payload_sha256?: unknown;
    } | undefined;
    if (!row) return undefined;
    const payloadJson = String(row.payload_json ?? "");
    if (sha256(payloadJson) !== row.payload_sha256) {
      throw new Error("Committed chat ledger payload checksum failed.");
    }
    return objectRecord(JSON.parse(payloadJson));
  } finally {
    database.close();
  }
}

function metadataPayloadById(
  metadata: Record<string, unknown>,
  field: "messages" | "traces",
  identifier: string
): Record<string, unknown> | undefined {
  const values = metadata[field];
  if (!Array.isArray(values)) return undefined;
  return values
    .map(objectRecord)
    .find((value) => value?.id === identifier);
}

function persistedChatReceipt(
  brainDirectory: string,
  brainId: string,
  metadata: Record<string, unknown>,
  request: ChatReconciliationRequest
): WorkerChatReceiptResult | undefined {
  const receipts = metadata.completed_chat_turns;
  const receipt = Array.isArray(receipts)
    ? receipts
      .map(objectRecord)
      .find((value) =>
        value?.turnId === request.turnId &&
        value.inputSha256 === request.inputSha256
      )
    : undefined;
  if (!receipt) return undefined;
  const humanId = receiptIdentifier(receipt.humanMessageId, "human message");
  const brainMessageId = receiptIdentifier(
    receipt.brainMessageId,
    "brain message"
  );
  const traceId = receiptIdentifier(receipt.traceId, "trace");
  const human = committedLedgerPayload(
    brainDirectory, brainId, "message", humanId
  ) ?? metadataPayloadById(metadata, "messages", humanId);
  const assistant = committedLedgerPayload(
    brainDirectory, brainId, "message", brainMessageId
  ) ?? metadataPayloadById(metadata, "messages", brainMessageId);
  const trace = committedLedgerPayload(
    brainDirectory, brainId, "trace", traceId
  ) ?? metadataPayloadById(metadata, "traces", traceId);
  if (!human || !assistant || !trace) {
    throw new Error("Committed chat receipt references unavailable ledger state.");
  }
  const counters = objectRecord(metadata.counters) ?? {};
  const substrate = objectRecord(metadata.substrate);
  const substratePersistence = objectRecord(substrate?.persistence);
  const mutableState = objectRecord(metadata.mutable_state);
  const externalMessage = (
    value: Record<string, unknown>,
    traceValue?: string
  ): Record<string, unknown> => ({
    id: value.id,
    role: value.role,
    content: value.content,
    createdAt: value.created_at ?? value.createdAt,
    ...(traceValue ? { traceId: traceValue } : {}),
    ...((value.attention_epoch ?? value.attentionEpoch) !== undefined
      ? { attentionEpoch: value.attention_epoch ?? value.attentionEpoch }
      : {}),
    ...(value.generation_end !== undefined ? { generation_end: value.generation_end } : {})
  });
  return {
    format: "omni-chat-turn-receipt-query",
    formatVersion: 1,
    brainId,
    turnId: request.turnId,
    committed: true,
    turnCommitted: true,
    legacyMatched: false,
    inputSha256: request.inputSha256,
    humanMessage: externalMessage(human),
    brainMessage: externalMessage(assistant, traceId),
    trace,
    inferenceCount: Number(receipt.inferenceCount),
    plasticityEvents: Number(counters.plasticity_events),
    consolidationCycles: Number(counters.consolidation_cycles),
    parameterChecksumAfter: String(receipt.parameterChecksumAfter ?? ""),
    engineUpdatedAt: String(metadata.updated_at ?? ""),
    substrateGeneration: String(substratePersistence?.activeGeneration ?? ""),
    mutableStateGeneration: String(mutableState?.activeGeneration ?? ""),
    idempotentCompletion: true,
    ...(receipt.generationEnd !== undefined ? { generationEnd: receipt.generationEnd as ChatGenerationEnd } : {}),
    noReply: receipt.generationEnd === "no-reply"
  };
}

async function latestChatReconciliationRequest(
  brain: BrainDocument,
  brainDirectory: string
): Promise<ChatReconciliationRequest | undefined> {
  const metadata = await committedEngineMetadata(brainDirectory);
  if (!metadata) return undefined;
  const messages = metadata?.messages;
  const traces = metadata?.traces;
  const counters = objectRecord(metadata?.counters);
  const inferenceCount = counters?.inference_count;
  if (
    metadata?.brain_id !== brain.id ||
    !Number.isSafeInteger(inferenceCount) ||
    Number(inferenceCount) !== brain.counters.inferenceCount + 1
  ) {
    return undefined;
  }
  const currentReceipt = Array.isArray(metadata.completed_chat_turns)
    ? metadata.completed_chat_turns.map(objectRecord).find((value) =>
      value?.inferenceCount === inferenceCount
    )
    : undefined;
  if (currentReceipt) {
    const turnId = receiptIdentifier(currentReceipt.turnId, "turn");
    const inputSha256 = String(currentReceipt.inputSha256 ?? "");
    const humanId = receiptIdentifier(
      currentReceipt.humanMessageId,
      "human message"
    );
    const human = committedLedgerPayload(
      brainDirectory, brain.id, "message", humanId
    ) ?? metadataPayloadById(metadata, "messages", humanId);
    if (
      !human ||
      typeof human.content !== "string" ||
      sha256(human.content) !== inputSha256
    ) return undefined;
    return {
      turnId,
      input: human.content,
      inputSha256,
      minimumInferenceCount: brain.counters.inferenceCount
    };
  }
  if (
    !Array.isArray(messages) ||
    !Array.isArray(traces) ||
    messages.length < 2 ||
    traces.length < 1
  ) return undefined;
  const human = objectRecord(messages.at(-2));
  const assistant = objectRecord(messages.at(-1));
  const trace = objectRecord(traces.at(-1));
  if (
    !human ||
    !assistant ||
    !trace ||
    human.role !== "human" ||
    assistant.role !== "brain" ||
    typeof human.content !== "string" ||
    !human.content.trim() ||
    human.content.length > 100_000 ||
    human.content.includes("\0")
  ) {
    return undefined;
  }
  const humanId = receiptIdentifier(human.id, "candidate human message");
  const brainMessageId = receiptIdentifier(
    assistant.id,
    "candidate brain message"
  );
  const traceId = receiptIdentifier(trace.id, "candidate trace");
  if (
    brain.messages.some((message) =>
      message.id === humanId || message.id === brainMessageId
    ) ||
    brain.traces.some((value) => value.id === traceId)
  ) {
    return undefined;
  }
  const inputSha256 = sha256(human.content);
  if (trace.input_sha256 !== inputSha256) return undefined;
  const recordedTurnId = trace.turn_id;
  const turnId =
    typeof recordedTurnId === "string" &&
    /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(recordedTurnId)
      ? recordedTurnId
      : `reconcile-${sha256(traceId)}`;
  return {
    turnId,
    input: human.content,
    inputSha256,
    minimumInferenceCount: brain.counters.inferenceCount
  };
}

function displayToolLabel(toolId: string): string {
  return toolId
    .split(/[.-]/)
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

function normalizedAddress(address: string): string {
  return address
    .replace(/^\[|\]$/g, "")
    .split("%", 1)[0]!
    .toLocaleLowerCase();
}

function loopbackAddress(address: string): boolean {
  const normalized = normalizedAddress(address);
  if (normalized === "::1" || normalized === "0:0:0:0:0:0:0:1") return true;
  if (isIP(normalized) !== 4) return false;
  return Number(normalized.split(".")[0]) === 127;
}

function privateAddress(address: string): boolean {
  const normalized = normalizedAddress(address);
  if (loopbackAddress(normalized)) return true;
  if (
    normalized === "::" ||
    normalized.startsWith("::ffff:") ||
    normalized.startsWith("fc") ||
    normalized.startsWith("fd") ||
    normalized.startsWith("ff") ||
    /^fe[89ab]/.test(normalized) ||
    normalized.startsWith("2001:db8:")
  ) {
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
      (first === 100 && second >= 64 && second <= 127) ||
      (first === 169 && second === 254) ||
      (first === 172 && second >= 16 && second <= 31) ||
      (first === 192 && second === 168) ||
      (first === 192 && second === 0) ||
      (first === 198 && (second === 18 || second === 19)) ||
      (first === 198 && second === 51) ||
      (first === 203 && second === 0) ||
      first >= 224
    );
  }
  return false;
}

export type SafeRemoteUrlClass = "loopback" | "remote";

export interface SafeRemoteUrlOptions {
  /** Explicitly scoped to user-selected local web learning, never tools/catalog. */
  allowLoopback?: boolean;
  resolveHostname?: (hostname: string) => Promise<readonly string[]>;
}

const resolveRemoteHostname = async (hostname: string): Promise<string[]> =>
  (await lookup(hostname, { all: true, verbatim: true })).map(
    (entry) => entry.address
  );

export async function assertSafeRemoteUrl(
  url: URL,
  options: SafeRemoteUrlOptions = {}
): Promise<SafeRemoteUrlClass> {
  if (url.username || url.password) throw new Error("URLs containing credentials are not allowed.");
  if (!["http:", "https:"].includes(url.protocol)) {
    throw new Error("Remote URLs require HTTPS; local web learning allows loopback HTTP only.");
  }
  const hostname = normalizedAddress(url.hostname).replace(/\.+$/, "");
  const literal = isIP(hostname) !== 0;
  const localhostName = hostname === "localhost";
  const literalLoopback = literal && loopbackAddress(hostname);
  const resolver = options.resolveHostname ?? resolveRemoteHostname;

  if (literalLoopback || localhostName) {
    if (!options.allowLoopback) {
      throw new Error("Private or loopback network URLs are not allowed.");
    }
    if (localhostName) {
      const addresses = await resolver(hostname);
      if (
        addresses.length === 0 ||
        addresses.some((address) => !loopbackAddress(address))
      ) {
        throw new Error(
          "localhost must resolve exclusively to 127/8 or ::1 for local web learning."
        );
      }
    }
    return "loopback";
  }
  if (url.protocol !== "https:") {
    throw new Error("Remote URLs require HTTPS; only verified loopback sources may use HTTP.");
  }
  const addresses = literal ? [hostname] : await resolver(hostname);
  if (addresses.length === 0 || addresses.some((address) => privateAddress(address))) {
    throw new Error("Remote URL resolves to a private or reserved network address.");
  }
  return "remote";
}

export async function safeFetch(
  initialUrl: URL,
  init: RequestInit,
  maximumRedirects = 5,
  hooks: {
    beforeRequest?: (url: URL) => Promise<void | (() => void)>;
    afterResponse?: (url: URL, response: Response) => Promise<void> | void;
    allowLoopback?: boolean;
  } = {}
): Promise<Response> {
  let current = initialUrl;
  let initialClass: SafeRemoteUrlClass | undefined;
  for (let redirect = 0; redirect <= maximumRedirects; redirect += 1) {
    init.signal?.throwIfAborted();
    const currentClass = await assertSafeRemoteUrl(current, {
      allowLoopback: hooks.allowLoopback
    });
    initialClass ??= currentClass;
    if (initialClass === "remote" && currentClass === "loopback") {
      throw new Error("A remote URL redirect cannot target a loopback address.");
    }
    const releaseRequest = await hooks.beforeRequest?.(current);
    let response: Response;
    try {
      init.signal?.throwIfAborted();
      response = await fetch(current, { ...init, redirect: "manual" });
      await hooks.afterResponse?.(current, response);
    } finally {
      releaseRequest?.();
    }
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

function crawlResourcePauseFromEngineError(
  error: unknown
): CrawlResourcePause | undefined {
  if (!(error instanceof EngineRequestError) || error.code !== -32020) {
    return undefined;
  }
  const data = objectRecord(error.data);
  const resourcePause = objectRecord(data?.resourcePause);
  if (data?.recoverable !== true && resourcePause?.recoverable !== true) {
    return undefined;
  }
  const detail = [
    error.message,
    resourcePause?.reason,
    resourcePause?.detail,
    resourcePause?.userAction
  ]
    .filter((value): value is string => typeof value === "string")
    .map((value) => value.replace(/\0/g, "").replace(/\s+/g, " ").trim())
    .filter((value, index, all) => Boolean(value) && all.indexOf(value) === index)
    .join(" ")
    .slice(0, 2_000);
  return new CrawlResourcePause(
    detail || "Neural learning paused at a recoverable resource boundary."
  );
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

  async acquire(url: URL, signal?: AbortSignal): Promise<() => void> {
    const state = this.state(url.origin);
    const predecessor = state.tail;
    let release = (): void => undefined;
    state.tail = new Promise<void>((resolveTail) => {
      release = resolveTail;
    });
    await predecessor;
    try {
      const waitUntil = Math.max(state.nextRequestAt, state.backoffUntil);
      await abortableDelay(Math.max(0, waitUntil - performance.now()), signal);
      state.nextRequestAt = performance.now() + this.minimumDelay;
    } catch (error) {
      release();
      throw error;
    }
    let released = false;
    return () => {
      if (released) return;
      released = true;
      // Anchor the next reservation to the observed end of this request as
      // well as its scheduled start. If the Electron event loop was busy
      // between `fetch()` and network dispatch, two requests must not arrive
      // together merely because the original reservation time is now old.
      state.nextRequestAt = Math.max(
        state.nextRequestAt,
        performance.now() + this.minimumDelay
      );
      release();
    };
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
        performance.now() + Math.min(60_000, Math.max(this.minimumDelay, delay));
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
      ".heif",
      ".jp2",
      ".j2k",
      ".jpf",
      ".jpx",
      ".jxl",
      ".raw",
      ".dng"
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
      ".mka"
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
      ".vob"
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
    "image/jp2": ".jp2",
    "image/jxl": ".jxl",
    "image/x-adobe-dng": ".dng",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/x-caf": ".caf",
    "audio/amr": ".amr",
    "audio/basic": ".au",
    "audio/x-matroska": ".mka",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/3gpp": ".3gp",
    "video/ogg": ".ogv",
    "video/mp2t": ".m2ts",
    "video/x-matroska": ".mkv"
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

type ChatParameterLearningRuntime = Pick<
  WorkspaceLearningStatus["backgroundParameters"],
  "state" | "updatedAt"
> & Partial<Pick<
  WorkspaceLearningStatus["backgroundParameters"],
  | "startedAt"
  | "completedAt"
  | "lastJobId"
  | "parameterChanged"
  | "corticalParametersUpdated"
  | "error"
  | "lastError"
  | "pauseReason"
>>;

export class BrainService {
  readonly datasets: DatasetManifestStore;
  private readonly externalToolSchemas = new Map<string, ExternalToolSchema>();
  private readonly chatSlowLearning = new Map<string, Promise<void>>();
  private readonly chatSlowLearningTimers = new Map<string, NodeJS.Timeout>();
  private readonly chatSlowLearningFailures = new Map<string, number>();
  private readonly pausedChatLearning = new Set<string>();
  private readonly liveChatTurns = new Map<string, {
    brainId: string; inputSha256: string; chatDispatched: boolean;
    outputEnded: boolean; steerSuccessor?: string;
  }>();
  private readonly inlineCancellationOwners = new Map<string, {
    brainId: string; turnId: string; actionId: string; pid?: number;
  }>();
  private readonly inlineCancellationListeners = new Set<(event: {
    brainId: string; turnId: string; actionId: string;
  }) => void>();
  private inlineCancellationHooksInstalled = false;
  /** Development/live-QA override; never changes a saved brain configuration. */
  private readonly backgroundLearningSuspendedForLaunch =
    process.env.OMNI_SUSPEND_BACKGROUND_LEARNING === "1";
  private readonly chatParameterLearningRuntime =
    new Map<string, ChatParameterLearningRuntime>();

  isChatParameterLearningActive(brainId: string): boolean {
    return this.chatSlowLearning.has(brainId) ||
      this.chatSlowLearningTimers.has(brainId) ||
      ["pending", "running"].includes(
        this.chatParameterLearningRuntime.get(brainId)?.state ?? "idle"
      );
  }

  constructor(
    readonly repository: BrainRepository,
    readonly engine: EngineSupervisor,
    readonly resourcePlanner?: ResourcePlanner,
    readonly mediaArtifacts?: MediaArtifactRegistry
  ) {
    this.datasets = new DatasetManifestStore((brainId) =>
      this.repository.brainDirectory(brainId)
    );
  }

  registerExternalToolSchemas(schemas: readonly ExternalToolSchema[]): void {
    for (const schema of schemas) {
      const id = normalizeToolId(schema.id);
      const actions = [...new Set(schema.actions.map((action) => action.trim()).filter(Boolean))];
      if (actions.length === 0 || actions.some((action) => action.length > 128)) {
        throw new Error("External tool schema contains invalid actions.");
      }
      this.externalToolSchemas.set(id, {
        id,
        actions,
        ...(schema.inputSchema ? { inputSchema: structuredClone(schema.inputSchema) } : {})
      });
    }
  }

  private updateChatParameterLearningRuntime(
    brainId: string,
    patch: Omit<Partial<ChatParameterLearningRuntime>, "updatedAt"> & {
      state: ChatParameterLearningRuntime["state"];
      updatedAt?: string;
    }
  ): void {
    const previous = this.chatParameterLearningRuntime.get(brainId);
    this.chatParameterLearningRuntime.set(brainId, {
      ...previous,
      ...patch,
      updatedAt: patch.updatedAt ?? new Date().toISOString()
    });
  }

  private schedulePendingChatLearning(
    brainId: string,
    priority: number,
    jobId?: string,
    preserveFailure = false,
    retryDelayMs?: number
  ): void {
    if (this.pausedChatLearning.has(brainId) || this.backgroundLearningSuspendedForLaunch) {
      this.updateChatParameterLearningRuntime(brainId, { state: "paused" });
      return;
    }
    this.updateChatParameterLearningRuntime(brainId, {
      state: preserveFailure ? "failed" : "pending",
      ...(preserveFailure ? {} : { error: undefined }),
      ...(jobId ? { lastJobId: jobId } : {})
    });
    if (
      this.chatSlowLearning.has(brainId) ||
      this.chatSlowLearningTimers.has(brainId)
    ) {
      return;
    }
    const timer = setTimeout(() => {
      this.chatSlowLearningTimers.delete(brainId);
      this.resumePendingChatLearning(brainId);
    }, retryDelayMs ?? chatSlowReplayDelayMs(priority));
    timer.unref?.();
    this.chatSlowLearningTimers.set(brainId, timer);
  }

  /**
   * Replay one durable chat episode while the serial neural worker is idle.
   * Foreground ownership may terminate this background request at any point;
   * the worker's atomic generation and pending-job tombstone make retry exact.
   */
  resumePendingChatLearning(brainId: string): void {
    if (this.pausedChatLearning.has(brainId) || this.backgroundLearningSuspendedForLaunch) {
      this.updateChatParameterLearningRuntime(brainId, { state: "paused" });
      return;
    }
    if (this.chatSlowLearning.has(brainId)) return;
    const scheduled = this.chatSlowLearningTimers.get(brainId);
    if (scheduled) {
      clearTimeout(scheduled);
      this.chatSlowLearningTimers.delete(brainId);
    }
    let operation!: Promise<void>;
    operation = (async () => {
      let retryPriority: number | undefined;
      let retryAfterFailure = false;
      let retryDelayMs: number | undefined;
      try {
        const brain = await this.repository.get(brainId);
        if (!brain.config.onlineLearning || this.pausedChatLearning.has(brainId)) {
          this.updateChatParameterLearningRuntime(brainId, { state: "paused" });
          return;
        }
        // Startup enumerates every identity. Only identities with a durable
        // pending job may claim the one neural worker; loading an idle mind
        // here used to repeatedly preempt real parameter training elsewhere.
        const queuedInThisProcess =
          this.chatParameterLearningRuntime.get(brainId)?.state === "pending" &&
          Boolean(this.chatParameterLearningRuntime.get(brainId)?.lastJobId);
        if (!queuedInThisProcess) {
          const metadata = await committedEngineMetadata(
            this.repository.brainDirectory(brainId)
          );
          const pending = metadata?.pending_chat_slow_learning;
          if (!Array.isArray(pending) || pending.length === 0) {
            this.updateChatParameterLearningRuntime(brainId, {
              state: "idle",
              error: undefined
            });
            return;
          }
        }
        // Check the current RAM/disk reserve before loading a large brain for
        // replay. Failed preflight keeps the durable job pending and reports
        // its reason; the bounded retry resumes when resources recover.
        await this.preflightStart(brainId);
        if (this.pausedChatLearning.has(brainId)) return;
        this.updateChatParameterLearningRuntime(brainId, {
          state: "running",
          startedAt: new Date().toISOString(),
          error: undefined
        });
        const result = await this.engine.request<{
          processed?: boolean;
          pending?: number;
          nextPriority?: number;
          jobId?: string;
          parameterChecksumBefore?: string;
          parameterChecksumAfter?: string;
          corticalParametersUpdated?: boolean;
        }>(
          "consolidate_chat_learning",
          {
            brainId,
            onlineLearning: true,
            storagePath: this.repository.brainDirectory(brainId)
          },
          ENGINE_REQUEST_NO_DEADLINE,
          undefined,
          "background",
          {
            requestId: randomUUID(),
            owner: "training",
            label: "Background conversation replay",
            brainId
          }
        );
        if (result.processed === true && Number(result.pending ?? 0) > 0) {
          retryPriority = Number(result.nextPriority ?? 0);
        }
        if (result.processed === true) {
          this.chatSlowLearningFailures.delete(brainId);
          const completedAt = new Date().toISOString();
          this.updateChatParameterLearningRuntime(brainId, {
            state: "complete",
            completedAt,
            ...(typeof result.jobId === "string" && result.jobId
              ? { lastJobId: result.jobId }
              : {}),
            parameterChanged:
              typeof result.parameterChecksumBefore === "string" &&
              typeof result.parameterChecksumAfter === "string" &&
              result.parameterChecksumBefore !== result.parameterChecksumAfter,
            corticalParametersUpdated:
              typeof result.corticalParametersUpdated === "boolean"
                ? result.corticalParametersUpdated
                : undefined,
            error: undefined,
            lastError: undefined
          });
        } else {
          if (Number(result.pending ?? 0) === 0) {
            this.chatSlowLearningFailures.delete(brainId);
          }
          if (Number(result.pending ?? 0) > 0) {
            retryPriority = Number(result.nextPriority ?? 0);
          }
          this.updateChatParameterLearningRuntime(brainId, {
            state: Number(result.pending ?? 0) > 0 ? "pending" : "idle",
            error: undefined
          });
        }
      } catch (error) {
        if (error instanceof BackgroundRequestDeferredError) {
          retryPriority = 1;
          this.updateChatParameterLearningRuntime(brainId, {
            state: "pending",
            error: undefined
          });
        }
        else {
          console.warn(
            `Background conversation replay paused for ${brainId}:`,
            error instanceof Error ? error.message : error
          );
          retryPriority = 0;
          retryAfterFailure = true;
          const failures = (this.chatSlowLearningFailures.get(brainId) ?? 0) + 1;
          this.chatSlowLearningFailures.set(brainId, failures);
          // A disk-pressure or worker failure must not reload a large model
          // every ten seconds indefinitely. Retry automatically with bounded
          // backoff while preserving the durable, untrained episode.
          retryDelayMs = Math.min(5 * 60_000, 30_000 * 2 ** Math.min(failures - 1, 4));
          this.updateChatParameterLearningRuntime(brainId, {
            // Keep the durable job queued for retry, but report that the
            // cortical weights have NOT changed. A pending label alone hid
            // repeated disk-pressure and worker failures from the user.
            state: "failed",
            error: error instanceof Error
              ? error.message.slice(0, 512)
              : "Background parameter learning will retry.",
            lastError: error instanceof Error
              ? error.message.slice(0, 512)
              : "Background parameter learning will retry."
          });
        }
      } finally {
        if (this.chatSlowLearning.get(brainId) === operation) {
          this.chatSlowLearning.delete(brainId);
        }
        if (this.pausedChatLearning.has(brainId)) {
          this.updateChatParameterLearningRuntime(brainId, { state: "paused" });
        } else if (retryPriority !== undefined) {
          this.schedulePendingChatLearning(
            brainId, retryPriority, undefined, retryAfterFailure, retryDelayMs
          );
        }
      }
    })();
    this.chatSlowLearning.set(brainId, operation);
  }

  unregisterExternalToolSchemas(ids: readonly string[]): void {
    for (const id of ids) this.externalToolSchemas.delete(normalizeToolId(id));
  }

  recordConversationActions(brainId: string, actions: ActionEvent[]): Promise<void> {
    return this.repository.appendConversationActions(brainId, actions);
  }

  neuralToolSchemas(brain: BrainDocument): Array<{
    id: string;
    actions: readonly string[];
    grant: ToolPermissionLevel;
    inputSchema?: Record<string, unknown>;
  }> {
    return enabledToolSchemas(brain, this.externalToolSchemas);
  }

  async planWorkingMemory(
    request: WorkingMemoryPlanRequest,
    options: { hardwareTier?: HardwareTier; brainId?: string } = {}
  ): Promise<WorkingMemoryResourcePlan> {
    if (!this.resourcePlanner) {
      throw new Error("The live device resource planner is unavailable.");
    }
    const scopedBrainId = options.brainId ?? request.brainId;
    const brain = scopedBrainId
      ? await this.repository.get(scopedBrainId)
      : undefined;
    const saved = brain ? await savedRuntimeShape(this.repository.brainDirectory(brain.id), brain.id, brain.config) : undefined;
    return this.resourcePlanner.plan(request, {
      hardwareTier: brain ? undefined : options.hardwareTier,
      config: brain ? { ...brain.config, ...saved } : undefined,
      brainDirectory: brain
        ? this.repository.brainDirectory(brain.id)
        : undefined,
      ...(brain ? { admissionScope: "existing-runtime", enforceContextFloor: false } : {})
    });
  }

  async preflightStart(brainId: string): Promise<WorkingMemoryResourcePlan | undefined> {
    if (!this.resourcePlanner) return undefined;
    const brain = await this.repository.get(brainId);
    const plan = await this.resourcePlanner.plan(
      {
        // Startup must validate the persisted selection, not silently resize
        // an Auto/Extended instance to today's recommendation.
        mode: "manual",
        requestedItems: String(brain.config.workingMemorySlots),
        requestedContextTokens: String(brain.config.contextWindowTokens),
        systemRamMode: brain.config.systemRamMode,
        ...(brain.config.systemRamMode === "manual"
          ? { systemRamSharePercent: brain.config.systemRamSharePercent }
          : {})
      },
      {
        config: brain.config,
        brainDirectory: this.repository.brainDirectory(brain.id),
        admissionScope: "existing-runtime",
        enforceContextFloor: false
      }
    );
    if (!plan.allowed) {
      throw new Error(`This mind cannot start safely: ${plan.blockers.join(" ")}`);
    }
    return plan;
  }

  /**
   * Apply Device & runtime settings as one preflighted desktop/worker update.
   * The worker is updated before the desktop commit so a rejected policy can
   * never become the configuration restored on the next app launch.
   */
  async updateConfig(brainId: string, config: BrainConfig): Promise<BrainDocument> {
    return withBrainWrite(this.repository, brainId, async () => {
      const current = await this.repository.get(brainId);
      if (typeof config !== "object" || config === null || Array.isArray(config)) {
        throw new Error("Invalid brain configuration.");
      }
      if (!["auto", "manual"].includes(config.systemRamMode)) {
        throw new Error("System RAM mode must be Auto or Manual.");
      }
      if (!["auto", "manual"].includes(config.storagePoolMode)) {
        throw new Error("Storage pool mode must be Auto or Manual.");
      }
      if (
        !Number.isSafeInteger(config.storagePoolBytes) ||
        config.storagePoolBytes < 0 ||
        (config.storagePoolMode === "manual" && config.storagePoolBytes < 1)
      ) {
        throw new Error("The shared storage pool must be a positive byte size.");
      }
      if (
        config.systemRamMode === "manual" &&
        (!Number.isFinite(config.systemRamSharePercent) ||
          config.systemRamSharePercent < 30 ||
          config.systemRamSharePercent > 100)
      ) {
        throw new Error("Manual system RAM must be between 30% and 100% of the safe pool.");
      }
      if (!["auto", "extended", "manual"].includes(config.workingMemoryMode)) {
        throw new Error("Invalid working-memory policy.");
      }
      if (!Number.isSafeInteger(config.workingMemorySlots) || config.workingMemorySlots < 1) {
        throw new Error("Working-memory capacity must be a positive safe integer.");
      }
      if (!Number.isSafeInteger(config.contextWindowTokens) || config.contextWindowTokens < 8) {
        throw new Error("Active context must contain at least eight tokens.");
      }

      let resolvedConfig = this.repository.prepareConfig({
        ...config,
        ...await savedRuntimeShape(this.repository.brainDirectory(brainId), brainId, current.config),
        // Active Mode has its own single-owner transaction. A broad device
        // settings save must never acquire or release that process lease.
        idleCognition: current.config.idleCognition,
        systemRamSharePercent:
          config.systemRamMode === "manual" ? config.systemRamSharePercent : 0
      });
      if (this.resourcePlanner) {
        const plan = await this.resourcePlanner.plan(
          {
            // Settings already carries the user's resolved context choice.
            // Revalidate that exact choice instead of silently resizing it.
            mode: "manual",
            requestedItems: String(resolvedConfig.workingMemorySlots),
            requestedContextTokens: String(resolvedConfig.contextWindowTokens),
            systemRamMode: resolvedConfig.systemRamMode,
            ...(resolvedConfig.systemRamMode === "manual"
              ? { systemRamSharePercent: resolvedConfig.systemRamSharePercent }
              : {}),
            storagePoolMode: resolvedConfig.storagePoolMode,
            ...(resolvedConfig.storagePoolMode === "manual"
              ? { storagePoolBytes: String(resolvedConfig.storagePoolBytes) }
              : {})
          },
          {
            config: resolvedConfig,
            brainDirectory: this.repository.brainDirectory(brainId),
            admissionScope: "existing-runtime",
            enforceContextFloor: false
          }
        );
        if (!plan.allowed) {
          throw new Error(`These device settings are not safe: ${plan.blockers.join(" ")}`);
        }
        resolvedConfig = this.repository.prepareConfig({
          ...resolvedConfig,
          contextWindowTokens: plan.context.selectedTokens,
          memoryOffloadBytes: plan.resources.configuredMemorySpillBytes,
          contextOffloadBudgetBytes: plan.context.evidence.contextOffloadBudgetBytes ?? 0,
          memoryResidentItems: Math.max(1, plan.offload.residentMemoryItems),
          memoryOffloadSlowdownPercent: plan.offload.estimatedSlowdownPercent,
          systemRamMode: plan.resources.systemRamMode,
          systemRamSharePercent:
            plan.resources.systemRamMode === "manual"
              ? plan.resources.systemRamSharePercent
              : 0,
          storagePoolMode: plan.resources.storagePoolMode,
          storagePoolBytes: plan.resources.sharedStoragePoolBytes,
          storageBytesPerSecond: plan.offload.benchmark.storageBytesPerSecond
        });
      }

      const storagePath = this.repository.brainDirectory(brainId);
      await this.engine.request(
        "update_config",
        { brainId, config: resolvedConfig, storagePath },
        300_000
      );
      try {
        const updated = {
          ...current,
          name: resolvedConfig.name,
          config: resolvedConfig
        };
        return await this.repository.save(updated);
      } catch (error) {
        // The worker persists its own config, so restore it if the desktop
        // document's atomic commit unexpectedly fails.
        await this.engine.tryRequest(
          "update_config",
          { brainId, config: current.config, storagePath },
          300_000
        );
        throw error;
      }
    });
  }

  /**
   * Persist the user's background-training switch without resource planning or
   * waiting for a serial worker update. Fast per-turn synaptic learning is a
   * separate mechanism and remains active. Pending slow jobs are never removed.
   */
  async setOnlineLearning(brainId: string, enabled: boolean): Promise<BrainDocument> {
    if (typeof enabled !== "boolean") {
      throw new Error("Background training must be enabled or disabled.");
    }
    if (enabled && this.backgroundLearningSuspendedForLaunch) {
      throw new Error("Background training is suspended for this app launch.");
    }
    return withBrainWrite(this.repository, brainId, async () => {
      const brain = await this.repository.get(brainId);
      if (brain.config.onlineLearning === enabled) return brain;
      const updated = await this.repository.save({
        ...brain,
        config: { ...brain.config, onlineLearning: enabled }
      });
      if (!enabled) {
        this.pausedChatLearning.add(brainId);
        const timer = this.chatSlowLearningTimers.get(brainId);
        if (timer) {
          clearTimeout(timer);
          this.chatSlowLearningTimers.delete(brainId);
        }
        const stopping = this.engine.cancelBackgroundRequest?.(
          brainId, "consolidate_chat_learning"
        ) ?? false;
        this.updateChatParameterLearningRuntime(brainId, {
          state: stopping || this.chatSlowLearning.has(brainId)
            ? "pausing" : "paused",
          error: undefined
        });
      } else {
        this.pausedChatLearning.delete(brainId);
        this.updateChatParameterLearningRuntime(brainId, {
          state: "pending",
          error: undefined
        });
        queueMicrotask(() => this.resumePendingChatLearning(brainId));
      }
      return updated;
    });
  }

  async create(
    request: CreateBrainRequest,
    onBuildEvent?: (event: EngineEvent) => void,
    onBrainPersisted?: (
      brain: BrainDocument,
      foundation: InitialFoundationPlan
    ) => Promise<void>
  ): Promise<BrainDocument> {
    const supplied = request as unknown as Record<string, unknown>;
    if (["origin", "starterUrl", "foundationModelId", "foundationRiskAcknowledged"]
      .some((field) => Object.hasOwn(supplied, field))) {
      throw new Error("Build accepts only the native OmniCortex origin. Remove legacy model fields.");
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
    const initialResources = (request.initialResources ?? []).map((resource) => {
      if (
        resource?.kind === "selection" &&
        /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(resource.selectionId)
      ) {
        return { kind: "selection" as const, selectionId: resource.selectionId };
      }
      if (resource?.kind !== "web" || typeof resource.url !== "string") {
        throw new Error("Invalid initial learning resource.");
      }
      const url = new URL(resource.url);
      if (!["http:", "https:"].includes(url.protocol) || url.username || url.password) {
        throw new Error("Initial web learning requires a credential-free HTTP(S) URL.");
      }
      url.hash = "";
      return { kind: "web" as const, url: url.toString() };
    });
    const resolvedTier = request.hardwareTier ?? "personal";
    const memoryMode = request.workingMemory?.mode ??
      request.config.workingMemoryMode ??
      (request.config.extendedWorkingMemory ? "extended" : "auto");
    if (!["auto", "extended", "manual"].includes(memoryMode)) {
      throw new Error("Invalid working-memory policy.");
    }
    if (!this.resourcePlanner) {
      throw new Error("New Build requires live resource preflight.");
    }
    const memoryPlan = await this.resourcePlanner.plan(
      {
        mode: memoryMode,
        requestedItems: request.workingMemory?.requestedItems,
        requestedContextTokens:
          request.workingMemory?.requestedContextTokens,
        acceleratorAvailable:
          request.workingMemory?.acceleratorAvailable,
        systemRamMode:
          request.workingMemory?.systemRamMode ??
          request.config.systemRamMode,
        systemRamSharePercent:
          request.workingMemory?.systemRamSharePercent ??
          request.config.systemRamSharePercent,
        storagePoolMode:
          request.workingMemory?.storagePoolMode ??
          request.config.storagePoolMode,
        storagePoolBytes:
          request.workingMemory?.storagePoolBytes ??
          (request.config.storagePoolMode === "manual"
            ? String(request.config.storagePoolBytes)
            : undefined),
        trainingSourceBytes: request.workingMemory?.trainingSourceBytes
      },
      { hardwareTier: resolvedTier, enforceContextFloor: true }
    );
    if (!memoryPlan.allowed) {
      throw new Error(`This mind cannot be built safely: ${memoryPlan.blockers.join(" ")}`);
    }
    const resolvedConfig: BrainConfig = {
      ...request.config,
      nativeArchitecture: memoryPlan.nativeArchitecture,
      // New stable instances have one non-configurable adaptive-retention
      // lifecycle: temporary working activity, durable neural learning, and
      // provenance without a routine verbatim source archive.
      onlineLearning: true,
      retainSourceText: false,
      memoryRecipe: "adaptive-retention",
      recursiveImprovement: true,
      idleCognition: true,
      // Neural workspace capacity is an architecture decision derived from
      // the hardware tier. Numeric values supplied by old/beta callers are
      // not behavioral controls and cannot shrink a stable-v1 brain.
      workingMemorySlots: memoryPlan.selectedItems,
      contextWindowTokens: memoryPlan.context.selectedTokens,
      workingMemoryMode: memoryMode,
      extendedWorkingMemory: memoryMode === "extended",
      memoryOffloadBytes: memoryPlan.resources.configuredMemorySpillBytes,
      contextOffloadBudgetBytes: memoryPlan.context.evidence.contextOffloadBudgetBytes ?? 0,
      memoryResidentItems: memoryPlan.offload.residentMemoryItems,
      memoryOffloadSlowdownPercent:
        memoryPlan.offload.estimatedSlowdownPercent,
      systemRamMode: memoryPlan.resources.systemRamMode,
      systemRamSharePercent:
        memoryPlan.resources.systemRamMode === "manual"
          ? memoryPlan.resources.systemRamSharePercent
          : 0,
      storagePoolMode: memoryPlan.resources.storagePoolMode,
      storagePoolBytes: memoryPlan.resources.sharedStoragePoolBytes,
      storageBytesPerSecond: memoryPlan.offload.benchmark.storageBytesPerSecond
    };
    // New Build has one mandatory, auditable origin: locally initialized
    // OmniCortex weights. Portable imports must have the same native origin.
    const recovery = {
      foundation: {
        hardwareTier: resolvedTier,
        modalities,
        origin: "ground-up" as const
      },
      resources: initialResources
    };
    // Persist the gate before the directory becomes visible. The engine may
    // spend minutes materializing the new neural state, during which the idle
    // scheduler must never mistake this partial mind for ready.
    let brain = await this.repository.create(resolvedConfig, {
      initializing: true,
      recovery
    });
    if (request.initialToolPermissions) {
      const selected = new Map(
        request.initialToolPermissions.map((permission) => [
          canonicalToolPermissionId(normalizeToolId(permission.toolId)),
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
        summary: `Build profile: ${request.hardwareTier ?? "automatic"}; modalities: ${(request.modalities ?? []).join(", ") || "text"}; native core: locally initialized.`
      }
    ];
    brain = await this.repository.save(brain);
    await onBrainPersisted?.(brain, {
      hardwareTier: resolvedTier,
      modalities,
      origin: "ground-up",
      diskBudget: {
        mandatoryReserveBytes:
          memoryPlan.diskSpace.mandatoryReserveBytes,
        selectedDatasetBytes:
          memoryPlan.diskSpace.selectedDatasetBytes,
        modelBytes: memoryPlan.diskSpace.modelBytes,
        checkpointBytes: memoryPlan.diskSpace.checkpointBytes,
        maximumWorkingMemorySpillBytes:
          memoryPlan.diskSpace.maximumWorkingMemorySpillBytes,
        futureGrowthBytes: memoryPlan.diskSpace.futureGrowthBytes,
        operationWriteBytes:
          memoryPlan.diskSpace.operationWriteBytes
      }
    });
    const storagePath = this.repository.brainDirectory(brain.id);
    const createParams = {
      brainId: brain.id,
      config: brain.config,
      hardwareTier: resolvedTier,
      modalities,
      origin: "ground-up" as const,
      nativeArchitecture: memoryPlan.nativeArchitecture,
      storagePath
    };
    let workerSummary: unknown;
    if (onBuildEvent) {
      workerSummary = await this.engine.requestStream(
        "create",
        createParams,
        onBuildEvent,
        300_000
      );
    } else {
      workerSummary = await this.engine.request(
        "create",
        createParams,
        300_000
      );
    }
    if (synchronizeWorkerSummary(brain, workerSummary)) {
      brain = await this.repository.save(brain);
    }
    // The live checkpoint remains independently writable, while an immutable
    // recovery origin can be shared by content hash across its own lineage.
    await this.repository.deduplicateImmutableOrigin(brain.id);
    return brain;
  }

  /**
   * Idempotently finish a foundation that was interrupted after the durable
   * desktop document and recovery plan were committed. `create` in the worker
   * loads an already committed checkpoint, or reconstructs from the same
   * recorded builder inputs when no neural commit record exists yet.
   */
  async resumeFoundation(
    brainId: string,
    foundation: InitialFoundationPlan,
    onBuildEvent?: (event: EngineEvent) => void
  ): Promise<BrainDocument> {
    if (foundation.origin !== "ground-up" || Object.hasOwn(foundation, "foundationModelId")) {
      throw new Error("Only native OmniCortex initialization can be resumed.");
    }
    let brain = await this.repository.get(brainId);
    await this.preflightStart(brainId);
    const params = {
      brainId,
      config: brain.config,
      hardwareTier: foundation.hardwareTier,
      modalities: foundation.modalities,
      origin: foundation.origin,
      nativeArchitecture: brain.config.nativeArchitecture,
      storagePath: this.repository.brainDirectory(brainId)
    };
    const workerSummary = onBuildEvent
      ? await this.engine.requestStream(
        "create",
        params,
        onBuildEvent,
        300_000
      )
      : await this.engine.request(
        "create",
        params,
        300_000
      );
    if (synchronizeWorkerSummary(brain, workerSummary)) {
      brain = await this.repository.save(brain);
    }
    await this.repository.deduplicateImmutableOrigin(brainId);
    return brain;
  }

  async queryCortex(brainId: string, query: CortexQuery = {}, signal?: AbortSignal): Promise<CortexPage> {
    await this.preflightStart(brainId);
    await this.repository.get(brainId);
    if (query.entity !== undefined && !["modules", "rows", "elements", "links", "boundaries"].includes(query.entity)) {
      throw new Error("Invalid cortical inspection entity.");
    }
    for (const field of ["row", "offset", "pageSize"] as const) {
      const value = query[field];
      if (value !== undefined && (!Number.isSafeInteger(value) || value < (field === "pageSize" ? 1 : 0))) {
        throw new Error("Invalid cortical inspection range.");
      }
    }
    for (const field of ["moduleId", "group", "search", "cursor"] as const) {
      const value = query[field];
      if (value !== undefined && (typeof value !== "string" || value.length > (field === "cursor" ? 2048 : 512))) {
        throw new Error("Invalid cortical inspection filter.");
      }
    }
    return this.engine.request<CortexPage>("query_cortex", {
      brainId, storagePath: this.repository.brainDirectory(brainId),
      query: { entity: query.entity ?? "modules", moduleId: query.moduleId, group: query.group,
        search: query.search, row: query.row, offset: query.offset, cursor: query.cursor, pageSize: query.pageSize ?? 64 }
    }, 30_000, signal);
  }

  async cortexActivity(brainId: string, query: CortexActivityQuery): Promise<CortexActivity> {
    await this.repository.get(brainId);
    if (typeof query.module !== "string" || query.module.length > 512 || typeof query.enabled !== "boolean"
        || (query.start !== undefined && (!Number.isSafeInteger(query.start) || query.start < 0))
        || (query.count !== undefined && (!Number.isSafeInteger(query.count) || query.count < 1))) {
      throw new Error("Invalid observed cortical activity viewport.");
    }
    return this.engine.request<CortexActivity>("cortex_activity", {
      brainId, storagePath: this.repository.brainDirectory(brainId), query
    }, 10_000);
  }

  async querySubstrate(
    brainId: string,
    query: SubstrateQuery = {},
    signal?: AbortSignal
  ): Promise<SubstratePage> {
    await this.engine.claimForeground?.();
    await this.preflightStart(brainId);
    const brain = await this.repository.get(brainId);
    const brainDirectory = this.repository.brainDirectory(brainId);
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
        query.pageSize < 1)
    ) {
      throw new Error("Substrate page size must be a positive safe integer.");
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
    const connectedTo =
      typeof query.connectedTo === "string"
        ? query.connectedTo.replace(/\0/g, "").trim()
        : "";
    if (region.length > 128 || search.length > 512 || connectedTo.length > 256) {
      throw new Error("Substrate filter is too long.");
    }
    if (connectedTo && entity !== "synapses") {
      throw new Error("connectedTo is valid only for synapse queries.");
    }
    const params = {
      brainId,
      config: brain.config,
      storagePath: brainDirectory,
      query: {
        entity,
        cursor: query.cursor,
        connectedTo,
        pageSize: query.pageSize ?? 256,
        region,
        search,
        zoom: Math.max(0, Math.min(1, query.zoom ?? 0))
      }
    };
    const timeout = await foregroundLoadTimeoutMs(brainDirectory);
    return signal
      ? this.engine.request<SubstratePage>(
          "query_substrate",
          params,
          timeout,
          signal
        )
      : this.engine.request<SubstratePage>(
      "query_substrate",
          params,
          timeout
        );
  }

  async queryConceptIds(brainId: string, view: unknown, sourceTurnId: string, offset = 0): Promise<import("../shared/conceptIdView").ConceptIdPage> {
    const sourceOwner = typeof view === "object" && view !== null && "brainId" in view ? String(view.brainId) : "";
    const descriptor = normalizeConceptIdView(view, sourceOwner, sourceTurnId);
    if (!Number.isSafeInteger(offset) || offset < 0) throw new Error("Invalid concept ID page offset.");
    await this.repository.get(brainId);
    return this.engine.request("query_concept_id_view", {
      brainId, storagePath: this.repository.brainDirectory(brainId),
      conceptIdView: descriptor, sourceTurnId, offset,
      historicalInspection: descriptor.brainId !== brainId
    }, 600_000);
  }

  async persistedSubstrateOverview(
    brainId: string
  ): Promise<PersistedSubstrateOverview | null> {
    const overview = await this.repository.persistedSubstrateOverview(brainId);
    if (!overview) return null;
    const latestCompletedCoverage = await this.datasets.latestCompletedCoverage(
      brainId
    );
    return {
      ...overview,
      ...(latestCompletedCoverage ? { latestCompletedCoverage } : {})
    };
  }

  async getReconciledBrain(brainId: string): Promise<BrainDocument> {
    const brain = await this.repository.get(brainId);
    const candidate = await latestChatReconciliationRequest(
      brain,
      this.repository.brainDirectory(brainId)
    );
    if (!candidate) return brain;
    const reconciled = await this.reconcileCommittedChat(brainId, candidate);
    return reconciled?.brain ?? brain;
  }

  async flushNeuralCheckpoint(
    brainId: string,
    operationId: string,
    operation?: BrainStorageOperationHooks
  ): Promise<NeuralCheckpointReceipt> {
    const brain = await this.repository.get(brainId);
    await operation?.checkpoint({
      phase: "checkpointing",
      label: "Committing current neural state and packed inference manifest"
    });
    const value = await this.engine.request<unknown>(
      "checkpoint",
      {
        brainId,
        operationId,
        config: brain.config,
        storagePath: this.repository.brainDirectory(brainId)
      },
      ENGINE_REQUEST_NO_DEADLINE,
      operation?.signal,
      "foreground",
      {
        requestId: operationId,
        owner: "system",
        label: "Saving neural recovery checkpoint",
        brainId
      }
    );
    const receipt = requireNeuralCheckpointReceipt(
      value,
      brainId,
      operationId
    );
    await verifyNeuralCheckpointFiles(
      join(this.repository.brainDirectory(brainId), "engine"),
      receipt
    );
    await operation?.checkpoint({
      phase: "checkpointed",
      label: "Neural checkpoint acknowledged; materializing one recovery point"
    });
    return receipt;
  }

  async createRecoveryPoint(
    brainId: string,
    label: string | undefined,
    operationId: string,
    operation?: BrainStorageOperationHooks
  ): Promise<BrainSnapshotSummary> {
    // Preempt optional background work before waiting for the same-brain write
    // lock; the lock then covers checkpoint acknowledgement through the final
    // repository metadata commit so no chat/learning turn can interleave.
    await this.engine.claimForeground?.();
    return this.repository.snapshot(
      brainId,
      label,
      operation,
      async () => {
        await this.flushNeuralCheckpoint(
          brainId,
          operationId,
          operation
        );
      }
    );
  }

  async restoreRecoveryPoint(
    brainId: string,
    snapshotId: string,
    operationId: string,
    operation?: BrainStorageOperationHooks
  ): Promise<BrainDocument> {
    await this.engine.claimForeground?.();
    const reload = async (
      requestId: string,
      signal?: AbortSignal
    ): Promise<void> => {
      const result = await this.engine.request<unknown>(
        "reload",
        {
          brainId,
          storagePath: this.repository.brainDirectory(brainId)
        },
        ENGINE_REQUEST_NO_DEADLINE,
        signal,
        "foreground",
        {
          requestId,
          owner: "system",
          label: "Reloading neural recovery point",
          brainId
        }
      );
      const record = objectRecord(result);
      if (record?.brainId !== brainId) {
        throw new Error(
          "The neural worker did not acknowledge the restored identity."
        );
      }
    };
    try {
      return await this.repository.restoreSnapshot(
        brainId,
        snapshotId,
        operation,
        () => reload(operationId, operation?.signal)
      );
    } catch (error) {
      // The repository rolls disk and host ledgers back before this point. If
      // reload failed after evicting the worker cache, restore the rolled-back
      // live checkpoint when cancellation did not explicitly stop the work.
      if (!operation?.signal.aborted) {
        await reload(randomUUID()).catch(() => undefined);
      }
      throw error;
    }
  }

  async workspace(brainId: string): Promise<WorkspaceSnapshot> {
    const brain = await this.repository.get(brainId);
    if (brain.readiness.state !== "ready") {
      throw new Error("This mind is still completing its initial learning.");
    }
    const brainDirectory = this.repository.brainDirectory(brainId);
    const snapshot = await persistedWorkspaceSnapshot(brainDirectory, brainId);
    const runtime = this.chatParameterLearningRuntime.get(brainId);
    if (!runtime && brain.config.onlineLearning && !this.backgroundLearningSuspendedForLaunch) {
      return snapshot;
    }
    const persistedLearning = snapshot.learning;
    const background = persistedLearning?.backgroundParameters ?? {
      state: "idle" as const,
      pending: 0,
      completed: 0,
      updatedAt: snapshot.contextWindow.updatedAt
    };
    const liveRuntime = runtime ?? {
      state: "paused" as const,
      updatedAt: new Date().toISOString()
    };
    const effectiveRuntime = (!brain.config.onlineLearning || this.backgroundLearningSuspendedForLaunch) && liveRuntime.state !== "pausing"
      ? {
          ...liveRuntime,
          state: "paused" as const,
          pauseReason: this.backgroundLearningSuspendedForLaunch
            ? "launch-override" as const : "user" as const
        }
      : liveRuntime;
    return {
      ...snapshot,
      learning: {
        measuredAt: effectiveRuntime.updatedAt,
        fastNeuralMemory: persistedLearning?.fastNeuralMemory ?? {
          state: "idle",
          safelyStored: false,
          completedTurns: 0,
          connectionUpdatesTotal: 0,
          parameterStepsTotal: 0
        },
        backgroundParameters: {
          ...background,
          ...effectiveRuntime,
          // Counts remain worker-owned durable facts. Runtime state supplies
          // only the live pending/running/completed transition around them.
          pending: background.pending,
          completed: background.completed
        }
      }
    };
  }

  async startFreshAttention(
    brainId: string,
    operationId: string = randomUUID()
  ): Promise<FreshAttentionResult> {
    await this.engine.claimForeground?.();
    return withBrainWrite(this.repository, brainId, async () => {
      await this.preflightStart(brainId);
      const brain = await this.repository.get(brainId);
      if (brain.readiness.state !== "ready") {
        throw new Error(
          "This mind is still completing its initial learning."
        );
      }
      const request = (): Promise<FreshAttentionResult> =>
        this.engine.request<FreshAttentionResult>(
          "fresh_attention",
          {
            brainId,
            operationId,
            config: brain.config,
            storagePath: this.repository.brainDirectory(brainId)
          },
          ENGINE_REQUEST_NO_DEADLINE
        );
      let result: FreshAttentionResult;
      try {
        result = await request();
      } catch (error) {
        // The worker may stop after atomically replacing brain.json but before
        // Electron receives the result. Retry the same operation identity;
        // the engine returns its persisted idempotent boundary rather than
        // advancing a second attention epoch.
        try {
          result = await request();
        } catch {
          throw error;
        }
      }
      if (
        result.format !== "omni-fresh-attention-boundary" ||
        result.formatVersion !== 1 ||
        result.brainId !== brainId ||
        result.committed !== true ||
        result.boundary?.operationId !== operationId ||
        !Number.isSafeInteger(result.boundary?.epoch) ||
        result.boundary.epoch < 1 ||
        !/^[a-f0-9]{64}$/.test(result.parameterChecksum) ||
        !/^[a-f0-9]{64}$/.test(result.fastSynapseChecksum) ||
        !/^[a-f0-9]{64}$/.test(result.substrateContentSha256) ||
        result.parameterChecksum !== result.boundary.parameterChecksum ||
        result.fastSynapseChecksum !== result.boundary.fastSynapseChecksum ||
        result.substrateContentSha256 !==
          result.boundary.substrateContentSha256 ||
        result.messagesPreserved !== result.boundary.messagesPreserved ||
        result.tracesPreserved !== result.boundary.tracesPreserved ||
        result.synapsesPreserved !== result.boundary.synapsesPreserved ||
        result.replayEntries !== result.boundary.replayEntries ||
        ![
          result.messagesPreserved,
          result.tracesPreserved,
          result.synapsesPreserved,
          result.replayEntries
        ].every((value) => Number.isSafeInteger(value) && value >= 0) ||
        result.rawPriorDialogueEligible !== false
      ) {
        throw new Error(
          "The neural worker returned an invalid fresh-attention receipt."
        );
      }
      return result;
    });
  }

  async modalityCapabilities(
    brainId: string
  ): Promise<NeuralModalityCapabilities> {
    const brain = await this.repository.get(brainId);
    const brainDirectory = this.repository.brainDirectory(brainId);
    if (brain.readiness.state !== "ready") {
      throw new Error("This mind is still completing its initial learning.");
    }
    return persistedModalityCapabilities(brainDirectory, brainId);
  }

  async chat(
    id: string,
    input: string,
    signal?: AbortSignal,
    onStream?: (event: NeuralChatStreamEvent) => void,
    turnId: string = randomUUID(),
    responseTokenBudget?: number
  ): Promise<ChatResult> {
    const message = cleanMessage(input);
    signal?.throwIfAborted();
    if (this.liveChatTurns.has(turnId)) throw new Error("This chat turn already owns the neural write boundary.");
    const ownership = { brainId: id, inputSha256: sha256(message),
      chatDispatched: false, outputEnded: false } as {
      brainId: string; inputSha256: string; chatDispatched: boolean;
      outputEnded: boolean; steerSuccessor?: string;
    };
    this.liveChatTurns.set(turnId, ownership);
    let minimumInferenceCount: number | undefined;
    let neuralRequestStarted = false;
    try {
      await this.engine.claimForeground?.();
      signal?.throwIfAborted();
      return await withBrainWrite(
        this.repository,
        id,
        async () => {
          minimumInferenceCount = (
            await this.repository.get(id)
          ).counters.inferenceCount;
          return this.chatUnlocked(
            id,
            message,
            signal,
            onStream,
            turnId,
            responseTokenBudget,
            () => {
              neuralRequestStarted = true;
            }
          );
        },
        signal
      );
    } catch (error) {
      if (error instanceof EngineRequestError && [-32801, -32802].includes(error.code ?? 0)) throw error;
      if (minimumInferenceCount !== undefined && neuralRequestStarted) {
        try {
          const reconciled = await this.reconcileCommittedChat(id, {
            turnId,
            input: message,
            inputSha256: sha256(message),
            minimumInferenceCount
          });
          if (reconciled) return reconciled;
        } catch (reconciliationError) {
          console.warn(
            `Committed chat reconciliation failed for ${id}:`,
            reconciliationError instanceof Error
              ? reconciliationError.message
              : reconciliationError
          );
        }
      }
      throw error;
    } finally {
      if (this.liveChatTurns.get(turnId) === ownership) this.liveChatTurns.delete(turnId);
    }
  }

  async steerChat(brainId: string, turnId: string, successorTurnId: string): Promise<void> {
    const ownership = this.liveChatTurns.get(turnId);
    if (!ownership || ownership.brainId !== brainId || ownership.outputEnded) return;
    ownership.steerSuccessor = successorTurnId;
    // A queued/pre-dispatch turn yields at the main admission boundary. Loading
    // may finish safely, but Steer never aborts it or starts a replacement PID.
    if (!ownership.chatDispatched) return;
    const result = await this.engine.steerChat(brainId, turnId, successorTurnId);
    if (!result.requested || !result.warm) throw new Error("The worker did not admit the exact warm steering direction.");
  }

  private chatSteerAdmissionBoundary(brainId: string, turnId: string): void {
    const ownership = this.liveChatTurns.get(turnId);
    if (!ownership?.steerSuccessor) return;
    throw new EngineRequestError("Chat yielded before decoding to a warm steering direction.", -32801, {
      brainId, turnId, inputSha256: ownership.inputSha256,
      steered: true, zeroTokenYield: true, safeBoundary: true, warm: true
    });
  }

  async cancelInlineImagination(brainId: string, turnId: string, actionId: string): Promise<{
    requested: boolean; acknowledged: boolean;
  }> {
    const key = `${brainId}:${turnId}:${actionId}`;
    this.inlineCancellationOwners.set(key, { brainId, turnId, actionId, pid: this.engine.pid });
    try {
      const result = await this.engine.cancelInlineGeneration(brainId, turnId, actionId);
      if (!result.requested) this.inlineCancellationOwners.delete(key);
      return result;
    } catch (error) {
      if (error instanceof EngineRequestError && [-32600, -32602].includes(error.code ?? 0)) {
        this.inlineCancellationOwners.delete(key);
      }
      throw error;
    }
  }

  onCodecRuntimeSetup(listener: (event: { brainId: string; turnId: string; actionId: string; message: string }) => void): () => void {
    const handle = (event: EngineEvent): void => {
      if (event.type === "video-runtime-setup" && event.brainId && event.streamId && event.actionId && event.message) {
        listener({ brainId: event.brainId, turnId: event.streamId, actionId: event.actionId, message: event.message });
      }
    };
    this.engine.on("event", handle);
    return () => { this.engine.off("event", handle); };
  }

  onInlineImaginationCancelled(listener: (event: {
    brainId: string; turnId: string; actionId: string;
  }) => void): () => void {
    this.inlineCancellationListeners.add(listener);
    const remove = (): void => { this.inlineCancellationListeners.delete(listener); };
    if (this.inlineCancellationHooksInstalled || typeof this.engine.on !== "function") return remove;
    this.inlineCancellationHooksInstalled = true;
    const handle = (event: EngineEvent): void => {
      const data = objectRecord(event.data);
      if (event.type !== "inline-imagination-cancelled" ||
          typeof event.brainId !== "string" || typeof event.streamId !== "string" ||
          typeof event.actionId !== "string" ||
          data?.acknowledged !== true || data.cleanupCompleted !== true) return;
      this.inlineCancellationOwners.delete(`${event.brainId}:${event.streamId}:${event.actionId}`);
      for (const subscriber of this.inlineCancellationListeners) {
        subscriber({ brainId: event.brainId, turnId: event.streamId, actionId: event.actionId });
      }
    };
    const closed = ({ pid }: { pid?: number }): void => {
      for (const [key, owner] of this.inlineCancellationOwners) {
        if (!pid || owner.pid !== pid) continue;
        void cleanupCancelledInlineStage(this.repository.brainDirectory(owner.brainId), owner.actionId)
          .then((cleaned) => {
            if (!cleaned || this.inlineCancellationOwners.get(key) !== owner) return;
            this.inlineCancellationOwners.delete(key);
            for (const subscriber of this.inlineCancellationListeners) subscriber(owner);
          }).catch(() => undefined); // Cleanup failure is never a false acknowledgement.
      }
    };
    this.engine.on("event", handle);
    this.engine.on("worker-closed", closed);
    return remove;
  }

  private async reconcileCommittedChat(
    brainId: string,
    request: ChatReconciliationRequest
  ): Promise<ChatResult | undefined> {
    const brainDirectory = this.repository.brainDirectory(brainId);
    const metadata = await committedEngineMetadata(brainDirectory);
    const persisted = metadata
      ? persistedChatReceipt(
          brainDirectory,
          brainId,
          metadata,
          request
        )
      : undefined;
    // Current checkpoints make absence authoritative too. Do not cold-start a
    // replacement neural worker merely to rediscover that an interrupted turn
    // never committed. Legacy checkpoints without the receipt field retain
    // the old load-free worker query for one-way migration.
    if (!persisted && Array.isArray(metadata?.completed_chat_turns)) {
      return undefined;
    }
    const raw = persisted ?? await this.engine.request<WorkerChatReceiptResult>(
      "chat_receipt",
      {
        brainId,
        storagePath: brainDirectory,
        turnId: request.turnId,
        inputSha256: request.inputSha256,
        minimumInferenceCount: request.minimumInferenceCount
      },
      30_000
    );
    const receipt = validateCommittedChatReceipt(raw, {
      brainId,
      turnId: request.turnId,
      input: request.input,
      inputSha256: request.inputSha256,
      minimumInferenceCount: request.minimumInferenceCount
    });
    if (!receipt) return undefined;
    return withBrainWrite(this.repository, brainId, async () => {
      const brain = await this.repository.get(brainId);
      const result = mergeCommittedChatPresentation(
        brain,
        receipt.humanMessage,
        receipt.brainMessage,
        receipt.trace
      );
      if (receipt.brainMessage.generationEnd) result.generationEnd = receipt.brainMessage.generationEnd;
      result.brain.counters.inferenceCount = Math.max(
        result.brain.counters.inferenceCount,
        receipt.inferenceCount
      );
      result.brain.counters.plasticityEvents = Math.max(
        result.brain.counters.plasticityEvents,
        receipt.plasticityEvents
      );
      result.brain.counters.consolidationCycles = Math.max(
        result.brain.counters.consolidationCycles,
        receipt.consolidationCycles
      );
      result.brain = await this.repository.save(result.brain);
      const slowLearningJob = receipt.trace.slow_learning_job;
      if (slowLearningJob?.jobId) {
        this.schedulePendingChatLearning(
          brainId,
          Number(slowLearningJob.priority ?? 0),
          slowLearningJob.jobId
        );
      }
      return result;
    });
  }

  private async chatUnlocked(
    id: string,
    input: string,
    signal?: AbortSignal,
    onStream?: (event: NeuralChatStreamEvent) => void,
    turnId: string = randomUUID(),
    responseTokenBudget?: number,
    onNeuralRequestStarted?: () => void
  ): Promise<ChatResult> {
    signal?.throwIfAborted();
    this.chatSteerAdmissionBoundary(id, turnId);
    if (
      responseTokenBudget !== undefined &&
      (!Number.isSafeInteger(responseTokenBudget) || responseTokenBudget < 1)
    ) {
      throw new Error("Response token budget must be a positive safe integer.");
    }
    const brain = await this.repository.get(id);
    if (brain.readiness.state !== "ready") {
      throw new Error(
        "This mind is still completing its initial learning and cannot chat yet."
      );
    }
    await this.preflightStart(id);
    const message = cleanMessage(input);
    const toolSchemas = this.neuralToolSchemas(brain);
    const brainDirectory = this.repository.brainDirectory(id);
    const onRuntimeActivity = (transition: EngineActivityTransition): void => {
      if (transition.method === "chat" && transition.state === "running") {
        const ownership = this.liveChatTurns.get(turnId);
        if (ownership) ownership.chatDispatched = true;
      }
      const normalized = neuralChatRuntimeActivity(transition);
      if (normalized) onStream?.(normalized);
    };
    const activityContext = {
      requestId: turnId,
      owner: "chat" as const,
      label: "Chat response",
      brainId: id,
      turnId,
      onTransition: onRuntimeActivity
    };
    await this.engine.request(
      "load",
      {
        brainId: id,
        config: brain.config,
        storagePath: brainDirectory
      },
      await foregroundLoadTimeoutMs(brainDirectory),
      signal,
      "foreground",
      activityContext
    );
    signal?.throwIfAborted();
    this.chatSteerAdmissionBoundary(id, turnId);
    onNeuralRequestStarted?.();
    const workerResult = await this.engine.requestStream<WorkerChatResult>(
      "chat",
      {
        brainId: id,
        input: message,
        toolSchemas,
        config: brain.config,
        onlineLearning: brain.config.onlineLearning,
        storagePath: brainDirectory,
        ...(responseTokenBudget === undefined
          ? {}
          : { maxNewTokens: responseTokenBudget })
      },
      (event) => {
        const normalized = normalizeChatEngineEvent(event, id);
        if (!normalized) return;
        if (normalized.type === "chat-phase") {
          const ownership = this.liveChatTurns.get(turnId);
          if (ownership) ownership.outputEnded = true;
        }
        if (normalized.type === "modality-preview" && this.mediaArtifacts) {
          const leased = this.mediaArtifacts.leasePreview(
            id,
            `action:${id}:${normalized.actionId ?? turnId}`,
            normalized.preview
          );
          if (!leased) return;
          normalized.preview = leased;
        }
        onStream?.(normalized);
      },
      // A response can be visible before online learning and an atomic save
      // finish. Do not discard that valid work because a large local brain
      // crosses an arbitrary wall clock; Cancel still aborts this exact turn.
      ENGINE_REQUEST_NO_DEADLINE,
      signal,
      turnId,
      "foreground",
      { ...activityContext, beforeDispatch: () => this.chatSteerAdmissionBoundary(id, turnId) }
    );
    signal?.throwIfAborted();
    const noReply = workerResult.noReply === true;
    if (noReply && [workerResult.text, workerResult.response, workerResult.content]
      .some((candidate) => candidate !== undefined && candidate !== "")) {
      throw new Error("No-reply neural completion contains fabricated response text.");
    }
    const generated = (workerResult.steered === true || workerResult.nativeStopped === true || noReply) && typeof workerResult.text === "string"
      ? workerResult.text : workerText(workerResult);
    if (generated === undefined || (!generated && !noReply)) {
      throw new Error("OmniCortex returned an empty neural response.");
    }
    const authoritativePresentation = validateWorkerChatPresentation(
      workerResult,
      {
        turnId,
        input: message,
        inputSha256: sha256(message),
        response: generated
      }
    );
    if ((workerResult.steered === true || workerResult.nativeStopped === true || noReply) && !authoritativePresentation) {
      throw new Error("Typed neural completion has no exact durable turn presentation.");
    }
    const result = authoritativePresentation
      ? mergeCommittedChatPresentation(
          brain,
          authoritativePresentation.humanMessage,
          authoritativePresentation.brainMessage,
          workerResult.trace!
        )
      : recordNeuralChat(brain, message, generated);
    if (!authoritativePresentation) {
      applyWorkerTracePresentation(result, workerResult?.trace);
    } else {
      result.brain.counters.inferenceCount = Math.max(
        result.brain.counters.inferenceCount,
        authoritativePresentation.inferenceCount
      );
    }
    result.proposedActions = parseModelActions("", workerResult?.actions);
    if (workerResult.steered === true) result.generationEnd = "steered";
    if (workerResult.nativeStopped === true) result.generationEnd = "native-stop";
    if (noReply) result.generationEnd = "no-reply";
    const metrics = workerResult?.metrics;
    if (metrics) {
      if (
        Number.isSafeInteger(metrics.plasticityEvents) &&
        Number(metrics.plasticityEvents) >= 0
      ) {
        result.brain.counters.plasticityEvents = Math.max(
          result.brain.counters.plasticityEvents,
          Number(metrics.plasticityEvents)
        );
      }
      const counters =
        typeof metrics.counters === "object" && metrics.counters !== null
          ? (metrics.counters as Record<string, unknown>)
          : {};
      if (
        Number.isSafeInteger(counters.inference_count) &&
        Number(counters.inference_count) >= 0
      ) {
        result.brain.counters.inferenceCount = Math.max(
          result.brain.counters.inferenceCount,
          Number(counters.inference_count)
        );
      }
      if (
        Number.isSafeInteger(counters.consolidation_cycles) &&
        Number(counters.consolidation_cycles) >= 0
      ) {
        result.brain.counters.consolidationCycles = Math.max(
          result.brain.counters.consolidationCycles,
          Number(counters.consolidation_cycles)
        );
      }
    }
    result.proposedActions ??= parseModelActions("", workerResult?.actions);
    result.brain = await this.repository.save(result.brain);
    const slowLearningJob = workerResult.trace?.slow_learning_job;
    if (slowLearningJob?.jobId) {
      // Weak/interfered episodes cool longer; recurrent/salient activity is
      // replayed sooner. Any following chat still preempts this optional work.
      this.schedulePendingChatLearning(
        id,
        Number(slowLearningJob.priority ?? 0),
        slowLearningJob.jobId
      );
    }
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
    if (brain.readiness.state !== "ready") {
      return {
        brainId,
        ran: false,
        reason: "initial-learning",
        actions: []
      };
    }
    await this.preflightStart(brainId);
    if (
      !Number.isFinite(minimumIdleSeconds) ||
      minimumIdleSeconds < 0 ||
      minimumIdleSeconds > 86_400
    ) {
      throw new Error("Invalid idle cognition interval.");
    }
    let worker: WorkerIdleCycleResult;
    try {
      worker = await this.engine.request<WorkerIdleCycleResult>(
        "idle_cycle",
        {
          brainId,
          config: brain.config,
          storagePath: this.repository.brainDirectory(brainId),
          toolSchemas: this.neuralToolSchemas(brain),
          minimumIdleSeconds
        },
        // Background cognition is single-flight and explicitly preemptible by
        // foreground work. Its wall-clock age must not create a worker-restart
        // loop while a large local checkpoint is healthy and still computing.
        ENGINE_REQUEST_NO_DEADLINE,
        undefined,
        "background"
      );
    } catch (error) {
      if (error instanceof BackgroundRequestDeferredError) {
        return {
          brainId,
          ran: false,
          reason: "foreground-work",
          actions: []
        };
      }
      throw error;
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
    const message = await this.repository.conversationMessage(
      request.brainId,
      request.messageId
    );
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

  async ingestPaths(
    brainId: string,
    paths: string[],
    policy: DataIngestionPolicy = "encode"
  ): Promise<IngestResult[]> {
    policy = requireNewDataIngestionPolicy(policy);
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
    if (run.paused) {
      throw new Error(
        `${run.pauseReason ?? "Dataset learning paused before the current source was committed."} ` +
        `Resume dataset manifest ${manifest.id}; no source was reported as learned.`
      );
    }
    if (!run.coverage.complete) {
      throw new Error(
        `Dataset manifest ${manifest.id} ended without complete coverage; no source was reported as learned.`
      );
    }
    return run.results;
  }

  /**
   * Learn a typed external trajectory through the same fast-synapse,
   * assembly, replay, and slow-parameter path as any other experience. The
   * plaintext is sent only to the supervised neural worker for this mutation;
   * it is not retained as prompt context or copied into a brain export.
   */
  async learnToolRouteOutcome(
    brainId: string,
    outcome: ConfirmedToolRouteOutcome,
    signal?: AbortSignal
  ): Promise<ToolRouteLearningResult> {
    return withBrainWrite(this.repository, brainId, async () => {
      if (!/^[0-9a-f]{8}-[0-9a-f-]{27,}$/i.test(outcome.eventId)) {
        throw new Error("The confirmed tool event ID is invalid.");
      }
      const utterance = outcome.utterance.replace(/\0/g, "").trim();
      const toolId = outcome.toolId.trim();
      const action = outcome.action.trim();
      const typedArguments = outcome.arguments ?? {};
      if (!utterance || !toolId || !action || toolId.length > 240 || action.length > 120) {
        throw new Error("A confirmed user utterance and tool route are required for learning.");
      }
      if (
        typeof typedArguments !== "object" ||
        typedArguments === null ||
        Array.isArray(typedArguments)
      ) {
        throw new Error("Confirmed tool arguments must be a typed object.");
      }
      await this.preflightStart(brainId);
      signal?.throwIfAborted();
      const worker = await this.engine.request<{
        brainId?: string;
        routeLearning?: ToolRouteLearningResult;
        metrics?: unknown;
      }>(
        "learn_tool_route_outcome",
        {
          brainId,
          storagePath: this.repository.brainDirectory(brainId),
          eventId: outcome.eventId,
          utterance,
          toolId,
          action,
          arguments: typedArguments,
          outcome: "success"
        },
        120_000,
        signal
      );
      const learned = worker.routeLearning;
      if (
        worker.brainId !== brainId ||
        !learned ||
        typeof learned.processed !== "boolean" ||
        typeof learned.applied !== "boolean" ||
        typeof learned.duplicate !== "boolean" ||
        typeof learned.ready !== "boolean" ||
        typeof learned.steps !== "number" ||
        !Number.isFinite(learned.steps)
      ) {
        throw new Error("The neural worker returned an invalid tool-route learning receipt.");
      }
      if (learned.applied) {
        const brain = await this.repository.get(brainId);
        synchronizeWorkerSummary(brain, worker);
        brain.journal = [
          ...(brain.journal ?? []),
          {
            id: randomUUID(),
            createdAt: new Date().toISOString(),
            kind: "learning",
            summary: "A confirmed host tool outcome trained the neural route head.",
            detail: `eventId=${outcome.eventId}; tool=${toolId}; action=${action}; steps=${learned.steps}; hiddenPrompt=false`
          }
        ];
        await this.repository.save(brain);
      }
      return learned;
    });
  }

  async learnStructuredExperience(
    brainId: string,
    experience: StructuredNeuralExperience,
    signal?: AbortSignal
  ): Promise<IngestResult> {
    return withBrainWrite(this.repository, brainId, () =>
      this.learnStructuredExperienceUnlocked(brainId, experience, signal)
    );
  }

  private async learnStructuredExperienceUnlocked(
    brainId: string,
    experience: StructuredNeuralExperience,
    signal?: AbortSignal
  ): Promise<IngestResult> {
    await this.preflightStart(brainId);
    signal?.throwIfAborted();
    const content = experience.content.replace(/\0/g, "").trim();
    if (!content || Buffer.byteLength(content) > 64 * 1024 * 1024) {
      throw new Error("Structured neural experience is empty or too large for one transaction.");
    }
    const contentHash = sha256(content);
    let brain = await this.repository.get(brainId);
    const duplicate = await this.repository.trainingSourceByContentHash(
      brainId,
      contentHash
    );
    if (duplicate) {
      return {
        brain,
        source: duplicate,
        warnings: ["This neural experience was already learned."]
      };
    }
    const worker = await this.engine.request<WorkerIngestResult>(
      "ingest",
      {
        brainId,
        text: content,
        name: experience.name.slice(0, 500),
        kind: "text",
        policy: "pretrain",
        contentHash,
        storagePath: this.repository.brainDirectory(brainId)
      },
      86_400_000,
      signal
    );
    signal?.throwIfAborted();
    synchronizeWorkerSummary(brain, worker);
    const source: TrainingSource = {
      id: randomUUID(),
      name: experience.name.replace(/\0/g, "").slice(0, 500),
      kind: "text",
      bytes: Buffer.byteLength(content),
      learnedIdeas: coverageCount(worker.source?.learned_ideas, 0),
      learnedConcepts: coverageCount(worker.source?.learned_concepts, 0),
      learnedSynapses: coverageCount(
        worker.source?.synaptic_update_events ?? worker.source?.plasticity_events,
        0
      ),
      learnedParameterSteps: coverageCount(worker.source?.parameter_update_steps, 0),
      parametersChanged: workerParametersChanged(worker),
      importedAt: new Date().toISOString(),
      rawTextRetained: false,
      contentHash,
      policy: "pretrain",
      provenanceUrl: experience.provenanceUrl,
      license: experience.license,
      licenseUrl: experience.licenseUrl
    };
    brain.trainingSources.push(source);
    brain.journal = [
      ...(brain.journal ?? []),
      {
        id: randomUUID(),
        createdAt: source.importedAt,
        kind: "learning",
        summary: `Learned ${experience.sourceLabel.slice(0, 180)} into neural state.`,
        detail:
          `contentHash=${contentHash}; rawTextRetained=false; hiddenPrompt=false; ` +
          `ideas=${source.learnedIdeas}; synapses=${source.learnedSynapses}`
      }
    ];
    brain = await this.repository.save(brain);
    return { brain, source, warnings: worker.source?.warnings ?? [] };
  }

  async previewDataset(
    brainId: string,
    paths: string[],
    options: DatasetManifestCreateOptions = {}
  ): Promise<DatasetManifest> {
    await this.repository.get(brainId);
    return this.datasets.create(brainId, paths, options);
  }

  async ingestManifest(
    brainId: string,
    manifestId: string,
    policy: PersistedDataIngestionPolicy = "encode",
    cancelled: () => boolean = () => false,
    progress: (
      value: number,
      message: string,
      detail?: IngestProgressDetail
    ) => void = () => undefined,
    collectResults = true,
    requestedEpochs?: number,
    replayFirstEpoch = true,
    jobId = "",
    signal?: AbortSignal
  ): Promise<{
    manifest: DatasetManifest;
    coverage: TrainingCoverage;
    results: IngestResult[];
    paused: boolean;
    pauseReason?: string;
  }> {
    const manifest = await this.datasets.manifest(brainId, manifestId);
    const initialProgress = await this.datasets.progress(brainId, manifestId);
    if (
      policy === "consolidate" &&
      initialProgress.lastEntryReceipt?.policy !== "consolidate"
    ) {
      throw new Error(
        "The legacy consolidate policy can resume only an exact persisted legacy receipt."
      );
    }
    const cursor = structuredClone(initialProgress.cursor);
    const coverage = structuredClone(initialProgress.coverage);
    let progressGeneration = initialProgress.generation;
    let lastEntryReceipt = initialProgress.lastEntryReceipt;
    const persistProgress = async (
      receipt: DatasetEntryReceipt | undefined = lastEntryReceipt
    ): Promise<void> => {
      const committed = await this.datasets.saveProgress(
        brainId,
        cursor,
        coverage,
        receipt,
        progressGeneration
      );
      progressGeneration = committed.generation;
      lastEntryReceipt = committed.lastEntryReceipt;
    };
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
    const committedManifestProgress = (): number => {
      if (coverage.complete && cursor.state === "complete") return 1;
      const visited = coverage.discoveredBytes > 0
        ? coverage.processedBytes / coverage.discoveredBytes
        : cursor.processedFiles / Math.max(1, coverage.discoveredFiles);
      // 100% is a durable-state assertion, not a rendering convenience. Keep
      // every in-flight or merely file-complete update below it until the final
      // epoch cursor and coverage are committed together.
      return Math.min(0.99, Math.max(0, visited));
    };
    let lastReportedProgress = committedManifestProgress();
    const reportProgress = (
      value: number,
      message: string,
      detail?: IngestProgressDetail
    ): void => {
      const durableComplete =
        coverage.complete && cursor.state === "complete" && value >= 1;
      const bounded = durableComplete
        ? 1
        : Math.min(0.99, Math.max(0, value));
      // Worker phases can report a lower local percentage after a checkpoint or
      // parser phase transition. A whole-manifest display must never move
      // backwards or reinterpret entry-local progress as a new denominator.
      lastReportedProgress = Math.max(lastReportedProgress, bounded);
      progress(lastReportedProgress, message, detail);
    };
    if (cursor.state === "complete" && coverage.complete) {
      return { manifest, coverage, results: [], paused: false };
    }
    cursor.state = "running";
    if (cursor.nextEntry >= manifest.discoveredFiles) cursor.nextEntry = 0;
    await persistProgress();
    const results: IngestResult[] = [];
    while ((cursor.currentEpoch ?? 0) < targetEpochs) {
      for await (const entry of this.datasets.entries(
        brainId,
        manifestId,
        cursor.nextEntry
      )) {
        if (cancelled()) {
          cursor.state = "paused";
          await persistProgress();
          return { manifest, coverage, results, paused: true };
        }
        const entryContentHash =
          entry.contentSha256 ?? sha256(`rejected\0${entry.path}\0${entry.rejection ?? ""}`);
        const transactionKey = datasetTransactionKey(
          manifestId,
          entry.index,
          cursor.currentEpoch ?? 0,
          entryContentHash,
          policy,
          cursor.runId
        );
        try {
          await assertManifestEntryStable(entry);
          const globalizeCoverage = (
            entryCoverage: TrainingCoverage
          ): TrainingCoverage => ({
            ...coverage,
            discoveredRecords:
              coverage.discoveredRecords + entryCoverage.discoveredRecords,
            processedRecords:
              coverage.processedRecords + entryCoverage.processedRecords,
            rejectedRecords:
              coverage.rejectedRecords + entryCoverage.rejectedRecords,
            processedBytes:
              coverage.processedBytes +
              Math.min(entry.bytes, entryCoverage.processedBytes),
            shards: coverage.shards + entryCoverage.shards,
            complete: false,
            updatedAt: new Date().toISOString()
          });
          const resumeBoundary = cursor.nextRecord;
          const durableBaseline = await durableIngestionBaseline(
            this.repository.brainDirectory(brainId),
            brainId,
            manifestId,
            entry,
            entryContentHash,
            policy,
            cursor.currentEpoch ?? 0,
            resumeBoundary
          );
          const baselineCoverage = durableBaseline
            ? globalizeCoverage(durableBaseline.coverage)
            : undefined;
          const baselineRecordsCompleted =
            coverage.processedRecords +
            coverage.rejectedRecords +
            resumeBoundary;
          let resumeRateBaselinePending = resumeBoundary > 0;
          if (resumeRateBaselinePending) {
            reportProgress(
              lastReportedProgress,
              `Resuming from durable record ${resumeBoundary.toLocaleString()}`,
              {
                ...(baselineCoverage ? { coverage: baselineCoverage } : {}),
                recordsCompleted: baselineRecordsCompleted,
                currentRecord: resumeBoundary,
                committedRecords: resumeBoundary,
                checkpointCommitted: true,
                scopeId: transactionKey,
                physicalSourceBytesComparable: false,
                rateBaseline: true,
                traversalMeasured: false,
                ...(durableBaseline ? { durableBaseline: true } : {})
              }
            );
          }
          const completed = await this.ingestOnePath(
            brainId,
            manifest,
            entry,
            policy,
            cursor.currentEpoch ?? 0,
            replayFirstEpoch,
            jobId,
            transactionKey,
            cursor.runId,
            lastEntryReceipt?.transactionKey === transactionKey
              ? lastEntryReceipt
              : undefined,
            async (receipt) => {
              await persistProgress(receipt);
            },
            async (entryProgress, workerMessage, eventData) => {
              const local = Math.min(0.99, Math.max(0, entryProgress));
              const overall = coverage.discoveredBytes > 0
                ? (coverage.processedBytes + entry.bytes * local) / coverage.discoveredBytes
                : (cursor.processedFiles + local) / Math.max(1, coverage.discoveredFiles);
              const eventRecord = objectRecord(eventData);
              const datasetProgress = objectRecord(eventRecord?.datasetProgress);
              const checkpoint = objectRecord(eventRecord?.checkpoint);
              // A live progress sample normally carries datasetProgress.coverage.
              // On recovery, the durable checkpoint snapshot is the authority
              // until the outer manifest coverage advances at file completion.
              const checkpointCoverage = objectRecord(checkpoint?.coverageAtCommit);
              const workerCoverageRaw =
                objectRecord(datasetProgress?.coverage) ?? checkpointCoverage;
              const liveWorkerCoverage = workerCoverageRaw
                ? normalizeWorkerTrainingCoverage(
                    workerCoverageRaw as WorkerTrainingCoverage,
                    entry.bytes
                  )
                : undefined;
              const liveCoverage = liveWorkerCoverage
                ? globalizeCoverage(liveWorkerCoverage)
                : undefined;
              const currentRecord = coverageCount(
                datasetProgress?.currentRecord ?? checkpoint?.committedRecords
              );
              const committedRecords = coverageCount(
                datasetProgress?.committedRecords ?? checkpoint?.committedRecords
              );
              const foundation = foundationTelemetrySample(
                eventRecord?.foundationConsolidation
              );
              const resourceReadings = trainingResourceReadings(
                eventRecord?.resourceReadings
              );
              const checkpointCommitted =
                datasetProgress?.checkpointCommitted === true ||
                checkpointCoverage !== undefined;
              const expectedRecords =
                typeof datasetProgress?.expectedRecords === "number"
                  ? coverageCount(datasetProgress.expectedRecords)
                  : undefined;
              const globalRecordsTotalCandidate = expectedRecords === undefined
                ? undefined
                : expectedRecords * targetEpochs;
              const recordsTotal =
                manifest.discoveredFiles === 1 &&
                datasetProgress?.recordTotalKnown === true &&
                Number.isSafeInteger(globalRecordsTotalCandidate)
                  ? globalRecordsTotalCandidate
                  : undefined;
              const classifiedRecords = liveCoverage
                ? liveCoverage.processedRecords + liveCoverage.rejectedRecords
                : 0;
              const rateBaseline =
                resumeRateBaselinePending &&
                (currentRecord > resumeBoundary ||
                  classifiedRecords > baselineRecordsCompleted);
              reportProgress(
                Math.min(0.99, Math.max(0, overall)),
                `Epoch ${(cursor.currentEpoch ?? 0) + 1}/${targetEpochs} · ` +
                  `source visit ${cursor.processedFiles + 1}/${coverage.discoveredFiles} · ` +
                  workerMessage,
                {
                  ...(liveCoverage ? { coverage: liveCoverage } : {}),
                  ...(liveCoverage
                    ? { recordsCompleted: classifiedRecords }
                    : {}),
                  ...(recordsTotal !== undefined ? { recordsTotal } : {}),
                  currentRecord,
                  committedRecords,
                  ...(expectedRecords !== undefined
                    ? { expectedRecords }
                    : {}),
                  recordTotalKnown: datasetProgress?.recordTotalKnown === true,
                  checkpointCommitted,
                  scopeId: transactionKey,
                  physicalSourceBytesComparable:
                    datasetProgress?.physicalSourceBytesComparable === true,
                  traversalMeasured: datasetProgress !== undefined,
                  ...(rateBaseline ? { rateBaseline: true } : {}),
                  ...(foundation ? { foundation } : {}),
                  ...(resourceReadings ? { resourceReadings } : {})
                }
              );
              if (rateBaseline) resumeRateBaselinePending = false;
              if (
                checkpointCommitted &&
                committedRecords > cursor.nextRecord
              ) {
                cursor.nextRecord = committedRecords;
                await persistProgress();
              }
            },
            signal
          );
          const result = completed.result;
          lastEntryReceipt = completed.receipt;
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
          const wholeSourceRejected = Boolean(
            workerCoverage &&
              workerCoverage.discoveredFiles > 0 &&
              workerCoverage.processedFiles === 0 &&
              workerCoverage.rejectedFiles === workerCoverage.discoveredFiles
          );
          if (wholeSourceRejected) coverage.rejectedFiles += 1;
          else coverage.processedFiles += 1;
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
          let representedWorkerErrors = 0;
          for (const workerError of workerCoverage?.errors ?? []) {
            representedWorkerErrors += Math.max(1, Math.round(workerError.count ?? 1));
            await this.datasets.recordError(brainId, coverage, {
              source: workerError.source || entry.path,
              message: workerError.message,
              ...(workerError.count && workerError.count !== 1
                ? { count: workerError.count }
                : {})
            });
          }
          const unrepresentedWorkerErrors = Math.max(
            0,
            (workerCoverage?.errorCount ?? representedWorkerErrors) - representedWorkerErrors
          );
          if (unrepresentedWorkerErrors > 0) {
            await this.datasets.recordError(brainId, coverage, {
              source: entry.path,
              message: "Additional equivalent worker rejections; see the exhaustive counters.",
              count: unrepresentedWorkerErrors
            });
          }
          coverage.errorsTruncated =
            coverage.errorsTruncated === true || workerCoverage?.errorsTruncated === true;
        } catch (error) {
          if (cancelled()) {
            cursor.state = "paused";
            await persistProgress();
            return { manifest, coverage, results, paused: true };
          }
          const message = error instanceof Error ? error.message : String(error);
          const permanentInputFailure =
            (error instanceof Error &&
              "code" in error &&
              (error as NodeJS.ErrnoException).code === "ENOENT") ||
            /is not a regular file|ingestion content hash mismatch|dataset manifest entry rejected|dataset source changed since the manifest was committed/i.test(
              message
            );
          if (!permanentInputFailure) {
            // Neural, resource, worker, and training failures are resumable. Do
            // not advance the manifest cursor or misreport them as invalid data.
            cursor.state = "paused";
            await persistProgress();
            const pauseReason =
              `Paused before ${entry.path}: ${message}`;
            reportProgress(
              committedManifestProgress(),
              pauseReason
            );
            return { manifest, coverage, results, paused: true, pauseReason };
          }
          coverage.rejectedFiles += 1;
          coverage.discoveredRecords += 1;
          coverage.rejectedRecords += 1;
          await this.datasets.recordError(brainId, coverage, {
            source: entry.path,
            message
          });
          lastEntryReceipt = {
            schemaVersion: 1,
            manifestId,
            ...(cursor.runId ? { runId: cursor.runId } : {}),
            manifestHash: manifest.manifestHash,
            transactionKey,
            entryIndex: entry.index,
            epoch: cursor.currentEpoch ?? 0,
            contentHash: entryContentHash,
            policy,
            outcome: "rejected",
            completedAt: new Date().toISOString()
          };
        }
        coverage.processedBytes += entry.bytes;
        cursor.nextEntry = entry.index + 1;
        cursor.nextRecord = 0;
        cursor.processedFiles = coverage.processedFiles + coverage.rejectedFiles;
        cursor.processedRecords = coverage.processedRecords;
        cursor.processedBytes = coverage.processedBytes;
        await persistProgress();
        reportProgress(
          committedManifestProgress(),
          `Epoch ${(cursor.currentEpoch ?? 0) + 1}/${targetEpochs}: learned ${
            cursor.processedFiles
          } of ${coverage.discoveredFiles} file visits`,
          {
            coverage: structuredClone(coverage),
            traversalMeasured: true
          }
        );
      }
      cursor.currentEpoch = (cursor.currentEpoch ?? 0) + 1;
      coverage.completedEpochs = cursor.currentEpoch;
      cursor.nextEntry = 0;
      cursor.nextRecord = 0;
      coverage.complete =
        (cursor.currentEpoch ?? 0) >= targetEpochs && hasCompleteTraversal(coverage);
      cursor.state = coverage.complete ? "complete" : "running";
      await persistProgress();
    }
    coverage.complete =
      (cursor.currentEpoch ?? 0) >= targetEpochs &&
      hasCompleteTraversal(coverage);
    cursor.state = coverage.complete ? "complete" : "failed";
    await persistProgress();
    if (coverage.complete) {
      reportProgress(
        1,
        `Whole-dataset learning complete: ${coverage.processedRecords} processed and ` +
          `${coverage.rejectedRecords} rejected records across ${targetEpochs} epoch` +
          `${targetEpochs === 1 ? "" : "s"}`,
        {
          coverage: structuredClone(coverage),
          traversalMeasured: true
        }
      );
    }
    return { manifest, coverage, results, paused: false };
  }

  private async ingestOnePath(
    brainId: string,
    manifest: DatasetManifest,
    entry: DatasetManifestEntry,
    policy: PersistedDataIngestionPolicy,
    epoch = 0,
    forceReplay = false,
    jobId = "",
    transactionKey: string,
    runId: string | undefined,
    completedReceipt?: DatasetEntryReceipt,
    onWorkerCompleted: (receipt: DatasetEntryReceipt) => Promise<void> = async () =>
      undefined,
    onWorkerProgress: (
      value: number,
      message: string,
      data?: unknown
    ) => void | Promise<void> = () => undefined,
    signal?: AbortSignal
  ): Promise<{ result: IngestResult; receipt: DatasetEntryReceipt }> {
    await this.engine.claimForeground?.();
    return withBrainWrite(this.repository, brainId, () =>
      this.ingestOnePathUnlocked(
        brainId,
        manifest,
        entry,
        policy,
        epoch,
        forceReplay,
        jobId,
        transactionKey,
        runId,
        completedReceipt,
        onWorkerCompleted,
        onWorkerProgress,
        signal
      ),
      signal
    );
  }

  private async ingestOnePathUnlocked(
    brainId: string,
    manifest: DatasetManifest,
    entry: DatasetManifestEntry,
    policy: PersistedDataIngestionPolicy,
    epoch: number,
    forceReplay: boolean,
    jobId: string,
    transactionKey: string,
    runId: string | undefined,
    completedReceipt: DatasetEntryReceipt | undefined,
    onWorkerCompleted: (receipt: DatasetEntryReceipt) => Promise<void>,
    onWorkerProgress: (
      value: number,
      message: string,
      data?: unknown
    ) => void | Promise<void>,
    signal?: AbortSignal
  ): Promise<{ result: IngestResult; receipt: DatasetEntryReceipt }> {
    await this.preflightStart(brainId);
    let brain = await this.repository.get(brainId);
    const path = entry.path;
    const fileInfo = await stat(path);
    if (!fileInfo.isFile()) throw new Error(`${path} is not a regular file.`);
    const contentHash = entry.contentSha256 ?? (await hashFile(path));
    const expectedTransactionKey = datasetTransactionKey(
      manifest.id,
      entry.index,
      epoch,
      contentHash,
      policy,
      runId
    );
    if (transactionKey !== expectedTransactionKey) {
      throw new Error("Dataset transaction identity does not match its manifest entry.");
    }
    const duplicate = await this.repository.trainingSourceByContentHash(
      brainId,
      contentHash
    );
    const kind = trainingKind(path);
    const parserKind = datasetParserKind(entry);
    let receipt = completedReceipt;
    if (receipt) {
      if (
        receipt.manifestId !== manifest.id ||
        receipt.runId !== runId ||
        receipt.manifestHash !== manifest.manifestHash ||
        receipt.transactionKey !== transactionKey ||
        receipt.entryIndex !== entry.index ||
        receipt.epoch !== epoch ||
        receipt.contentHash !== contentHash ||
        receipt.policy !== policy ||
        receipt.outcome !== "learned" ||
        !receipt.sourceId
      ) {
        throw new Error("Completed dataset receipt does not match the requested transaction.");
      }
    } else if (duplicate && !forceReplay && epoch === 0) {
      receipt = {
        schemaVersion: 1,
        manifestId: manifest.id,
        ...(runId ? { runId } : {}),
        manifestHash: manifest.manifestHash,
        transactionKey,
        entryIndex: entry.index,
        epoch,
        contentHash,
        policy,
        outcome: "learned",
        sourceId: duplicate.id,
        sourceKind: duplicate.kind,
        sourceBytes: duplicate.bytes,
        learnedIdeas: duplicate.learnedIdeas,
        learnedConcepts: duplicate.learnedConcepts,
        learnedSynapses: duplicate.learnedSynapses,
        learnedParameterSteps: duplicate.learnedParameterSteps,
        parametersChanged: duplicate.parametersChanged,
        workerDuplicate: true,
        warnings: boundedReceiptWarnings([
          `${basename(entry.sourcePath ?? path)} was already encoded; no duplicate synapses were created.`
        ]),
        completedAt: new Date().toISOString()
      };
      await onWorkerCompleted(receipt);
    } else {
      const beforeConcepts = Object.keys(brain.concepts).length;
      const beforeSynapses = Object.keys(brain.synapses).length;
      const beforeIdeas = brain.ideas.length;
      let workerProgressWrites = Promise.resolve();
      const workerEvent = (event: EngineEvent): void => {
        if (
          event.type !== "job-progress" ||
          event.jobId !== jobId ||
          typeof event.progress !== "number"
        ) return;
        workerProgressWrites = workerProgressWrites.then(() =>
          onWorkerProgress(
            event.progress!,
            event.message?.slice(0, 200) || `Training ${basename(path)}`,
            event.data
          )
        );
      };
      const eventSource = this.engine as EngineSupervisor & {
        on?: (name: "event", listener: (event: EngineEvent) => void) => unknown;
        off?: (name: "event", listener: (event: EngineEvent) => void) => unknown;
      };
      eventSource.on?.("event", workerEvent);
      let worker: WorkerIngestResult;
      try {
        worker = await this.engine.request<WorkerIngestResult>(
          "ingest",
          {
            jobId,
            brainId,
            path,
            kind: parserKind,
            policy,
            contentHash,
            transactionKey,
            idempotencyKey: transactionKey,
            manifestId: manifest.id,
            manifestHash: manifest.manifestHash,
            entryIndex: entry.index,
            committedSqliteSnapshot: entry.snapshotKind === "sqlite",
            allowReplay: forceReplay || epoch > 0,
            epoch,
            config: brain.config,
            storagePath: this.repository.brainDirectory(brainId)
          },
          // Exhaustive dataset learning can legitimately span days. Its
          // durable record checkpoints, explicit cancellation, and supervised
          // worker lifecycle are the termination controls—not wall-clock age.
          ENGINE_REQUEST_NO_DEADLINE,
          signal
        );
      } finally {
        eventSource.off?.("event", workerEvent);
        await workerProgressWrites;
      }
      synchronizeWorkerSummary(brain, worker);
      for (const returnedKey of [
        worker?.transactionKey,
        worker?.completedReceipt?.transactionKey,
        worker?.recordRecovery?.transactionKey
      ]) {
        if (returnedKey !== undefined && returnedKey !== transactionKey) {
          throw new Error("Neural worker acknowledged a different dataset transaction.");
        }
      }
      const rawCoverage = worker?.coverage ?? worker?.source?.coverage;
      const resultCoverage = rawCoverage
        ? {
            ...normalizeWorkerTrainingCoverage(rawCoverage, fileInfo.size),
            manifestId: manifest.id
          }
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
      const warnings = boundedReceiptWarnings([
        ...(worker?.warnings ?? []),
        ...(worker?.source?.warnings ?? [])
      ]);
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
      receipt = {
        schemaVersion: 1,
        manifestId: manifest.id,
        ...(runId ? { runId } : {}),
        manifestHash: manifest.manifestHash,
        transactionKey,
        entryIndex: entry.index,
        epoch,
        contentHash,
        policy,
        outcome: "learned",
        sourceId: duplicate?.id ?? randomUUID(),
        journalId: randomUUID(),
        sourceKind: effectiveKind,
        sourceBytes: fileInfo.size,
        learnedIdeas: coverageCount(
          worker?.source?.learned_ideas,
          Math.max(0, brain.ideas.length - beforeIdeas)
        ),
        learnedConcepts: coverageCount(
          worker?.source?.learned_concepts,
          Math.max(0, Object.keys(brain.concepts).length - beforeConcepts)
        ),
        learnedSynapses: coverageCount(
          worker?.source?.synaptic_update_events ?? worker?.source?.plasticity_events,
          Math.max(0, Object.keys(brain.synapses).length - beforeSynapses)
        ),
        learnedParameterSteps: coverageCount(worker?.source?.parameter_update_steps, 0),
        parametersChanged: workerParametersChanged(worker),
        workerDuplicate: worker?.duplicate === true,
        warnings,
        ...(resultCoverage ? { coverage: resultCoverage } : {}),
        completedAt: new Date().toISOString()
      };
      // Commit the bounded worker acknowledgement before mutating Electron's
      // repository. A restart can then finish bookkeeping without retraining.
      await onWorkerCompleted(receipt);
    }

    const committedSource = receipt.sourceId
      ? await this.repository.trainingSourceById(brainId, receipt.sourceId)
      : undefined;
    const committedJournal = receipt.journalId
      ? await this.repository.journalById(brainId, receipt.journalId)
      : undefined;
    const sourceAlreadyCommitted = receipt.journalId === undefined
      ? Boolean(committedSource)
      : Boolean(committedJournal);
    if (sourceAlreadyCommitted) {
      const source = committedSource ?? duplicate;
      if (!source) {
        throw new Error("Dataset receipt references a missing committed training source.");
      }
      return {
        result: {
          brain,
          source,
          warnings: receipt.warnings ?? [],
          coverage: receipt.coverage
        },
        receipt
      };
    }

    const memoryRecipe = brain.config.memoryRecipe ?? "adaptive-retention";
    // Training inputs are not artifacts. The locked human-like policy keeps
    // hashes/provenance and neural changes, then releases source bytes for all
    // modalities. Historical Total Recall callers remain explicit opt-in.
    const preserveBlob =
      memoryRecipe === "total-recall" && brain.config.retainSourceText;
    const source: TrainingSource = {
      id: receipt.sourceId!,
      name: basename(entry.sourcePath ?? path),
      path: entry.sourcePath ?? path,
      kind: receipt.sourceKind ?? kind,
      bytes: receipt.sourceBytes ?? fileInfo.size,
      learnedIdeas: receipt.learnedIdeas ?? 0,
      learnedConcepts: receipt.learnedConcepts ?? 0,
      learnedSynapses: receipt.learnedSynapses ?? 0,
      ...(receipt.coverage && receipt.coverage.processedRecords > 0
        ? { learnedRecords: receipt.coverage.processedRecords }
        : {}),
      learnedParameterSteps: receipt.learnedParameterSteps,
      parametersChanged: receipt.parametersChanged,
      importedAt: receipt.completedAt,
      rawTextRetained: preserveBlob,
      contentHash,
      blobHash: preserveBlob ? await this.repository.storeFileAsBlob(path) : undefined,
      policy,
      license: "User-provided source; license not declared"
    };
    if (duplicate) {
      brain.trainingSources = [
        ...brain.trainingSources.filter((record) => record.id !== duplicate.id),
        source
      ];
    } else {
      brain.trainingSources.push(source);
    }
    if (receipt.journalId) {
      brain.journal = [
        ...(brain.journal ?? []),
        {
          id: receipt.journalId,
          createdAt: source.importedAt,
          kind: "learning",
          summary: `${policy === "archive" ? "Archived" : "Learned from"} ${source.name}.`,
          detail: `${source.learnedIdeas} ideas, ${source.learnedConcepts} concepts, ${source.learnedSynapses} synapses`
        }
      ];
    }
    brain = await this.repository.save(brain);
    return {
      result: {
        brain,
        source,
        warnings: receipt.warnings ?? [],
        coverage: receipt.coverage
      },
      receipt
    };
  }

  async ingestWeb(request: IngestWebRequest): Promise<IngestResult> {
    request = {
      ...request,
      policy: requireNewDataIngestionPolicy(request.policy ?? "encode")
    };
    const url = new URL(request.url);
    const response = await safeFetch(
      url,
      {
        signal: AbortSignal.timeout(120_000),
        headers: {
          Accept: "text/html, text/plain, application/json;q=0.9",
          "User-Agent": CRAWL_USER_AGENT
        }
      },
      5,
      { allowLoopback: true }
    );
    if (!response.ok) throw new Error(`Web ingestion failed with HTTP ${response.status}.`);
    const finalUrl = new URL(response.url);
    await assertSafeRemoteUrl(finalUrl, { allowLoopback: true });
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
    contentType: string,
    signal?: AbortSignal,
    jobId = ""
  ): Promise<IngestResult> {
    return withBrainWrite(this.repository, request.brainId, () =>
      this.ingestWebContentUnlocked(request, finalUrl, raw, contentType, signal, jobId),
      signal
    );
  }

  private async ingestWebContentUnlocked(
    request: IngestWebRequest,
    finalUrl: URL,
    raw: string,
    contentType: string,
    signal?: AbortSignal,
    jobId = ""
  ): Promise<IngestResult> {
    await this.preflightStart(request.brainId);
    let text = contentType.includes("html") ? htmlToText(raw) : raw.replace(/\0/g, "");
    const webCache = join(this.repository.brainDirectory(request.brainId), "datasets", "web-cache");
    const lease = await acquireCrawlSourceLease({
      directory: webCache, url: finalUrl.toString(), kind: "text",
      contentHash: sha256(text), contentType, extension: ".txt", text,
    });
    if (lease.kind !== "text") {
      throw new Error("This URL has an unfinished binary learning lease; resume its original source first.");
    }
    text = await readSpoolText(lease.path, lease.bytes);
    contentType = lease.contentType;
    const contentHash = lease.contentHash;
    let brain = await this.repository.get(request.brainId);
    const policy = request.policy ?? "encode";
    const quarantined = request.quarantine ?? true;
    const duplicate = await this.repository.trainingSourceByContentHash(
      request.brainId,
      contentHash
    );
    if (duplicate && (quarantined || duplicate.policy !== "archive")) {
      await releaseCrawlSourceLease(lease);
      return {
        brain,
        source: duplicate,
        warnings: ["This web content was already ingested; no duplicate synapses were created."]
      };
    }
    const before = {
      ideas: brain.ideas.length,
      concepts: Object.keys(brain.concepts).length,
      synapses: Object.keys(brain.synapses).length
    };
    let worker: WorkerIngestResult;
    try {
      worker = await this.engine.request<WorkerIngestResult>(
        "ingest",
        {
          jobId,
          brainId: brain.id,
          url: finalUrl.toString(),
          path: lease.path,
          name: finalUrl.toString(),
          kind: "text",
          policy: quarantined ? "archive" : policy,
          quarantine: quarantined,
          allowReplay: !quarantined && duplicate?.policy === "archive",
          contentHash,
          storagePath: this.repository.brainDirectory(brain.id)
        },
        ENGINE_REQUEST_NO_DEADLINE,
        signal
      );
    } catch (error) {
      throw crawlResourcePauseFromEngineError(error) ?? error;
    }
    synchronizeWorkerSummary(brain, worker);
    const retainRaw =
      !quarantined &&
      (brain.config.memoryRecipe ?? "adaptive-retention") === "total-recall" &&
      brain.config.retainSourceText;
    const source: TrainingSource = {
      id: duplicate?.id ?? randomUUID(),
      name: finalUrl.hostname + finalUrl.pathname,
      kind: "text",
      bytes: lease.bytes,
      learnedIdeas: worker?.source?.learned_ideas ?? brain.ideas.length - before.ideas,
      learnedConcepts:
        worker?.source?.learned_concepts ??
        Object.keys(brain.concepts).length - before.concepts,
      learnedSynapses:
        worker?.source?.synaptic_update_events ??
        worker?.source?.plasticity_events ??
        Object.keys(brain.synapses).length - before.synapses,
      learnedParameterSteps: coverageCount(worker?.source?.parameter_update_steps, 0),
      parametersChanged: workerParametersChanged(worker),
      importedAt: new Date().toISOString(),
      rawTextRetained: retainRaw,
      rawText: retainRaw ? text : undefined,
      contentHash,
      blobHash:
        (brain.config.memoryRecipe ?? "adaptive-retention") !== "synapses-only" &&
        (quarantined || (brain.config.memoryRecipe ?? "adaptive-retention") === "total-recall")
          ? await this.repository.storeBlob(Buffer.from(raw, "utf8"))
          : undefined,
      policy: quarantined ? "archive" : policy,
      provenanceUrl: finalUrl.toString(),
      license: "Web source; verify the publisher's terms",
      licenseUrl: finalUrl.toString()
    };
    brain.trainingSources = [
      ...brain.trainingSources.filter((value) => value.id !== source.id), source
    ];
    brain = await this.repository.save(brain);
    await releaseCrawlSourceLease(lease);
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
    kind: "image" | "audio" | "video",
    signal?: AbortSignal,
    jobId = ""
  ): Promise<IngestResult> {
    return withBrainWrite(this.repository, request.brainId, () =>
      this.ingestCrawledMediaUnlocked(request, finalUrl, path, contentType, kind, signal, jobId),
      signal
    );
  }

  private async ingestCrawledMediaUnlocked(
    request: IngestWebRequest,
    finalUrl: URL,
    path: string,
    contentType: string,
    kind: "image" | "audio" | "video",
    signal?: AbortSignal,
    jobId = ""
  ): Promise<IngestResult> {
    await this.preflightStart(request.brainId);
    const lease = await acquireCrawlSourceLease({
      directory: join(this.repository.brainDirectory(request.brainId), "datasets", "web-cache"),
      url: finalUrl.toString(), kind, contentHash: await hashFile(path), contentType,
      extension: extname(path) || `.${kind}`, sourcePath: path,
    });
    if (lease.kind === "text") {
      throw new Error("This URL has an unfinished text learning lease; resume its original source first.");
    }
    const contentHash = lease.contentHash;
    kind = lease.kind;
    contentType = lease.contentType;
    let brain = await this.repository.get(request.brainId);
    const policy = request.policy ?? "encode";
    const quarantined = request.quarantine ?? true;
    const duplicate = await this.repository.trainingSourceByContentHash(
      request.brainId,
      contentHash
    );
    if (duplicate && (quarantined || duplicate.policy !== "archive")) {
      await releaseCrawlSourceLease(lease);
      return {
        brain,
        source: duplicate,
        warnings: [
          "This crawled media was already learned; no duplicate synapses were created."
        ]
      };
    }
    const fileInfo = await stat(lease.path);
    const before = {
      ideas: brain.ideas.length,
      concepts: Object.keys(brain.concepts).length,
      synapses: Object.keys(brain.synapses).length
    };
    let worker: WorkerIngestResult;
    try {
      worker = await this.engine.request<WorkerIngestResult>(
        "ingest",
        {
          jobId,
          brainId: brain.id,
          url: finalUrl.toString(),
          path: lease.path,
          name: finalUrl.toString(),
          kind,
          policy: quarantined ? "archive" : policy,
          quarantine: quarantined,
          allowReplay: !quarantined && duplicate?.policy === "archive",
          contentHash,
          storagePath: this.repository.brainDirectory(brain.id)
        },
        ENGINE_REQUEST_NO_DEADLINE,
        signal
      );
    } catch (error) {
      throw crawlResourcePauseFromEngineError(error) ?? error;
    }
    synchronizeWorkerSummary(brain, worker);
    const effectiveKind =
      worker?.source?.kind === "image" ||
      worker?.source?.kind === "audio" ||
      worker?.source?.kind === "video"
        ? worker.source.kind
        : kind;
    const memoryRecipe = brain.config.memoryRecipe ?? "adaptive-retention";
    // Quarantine keeps bytes only until review; accepted learning under the
    // stable policy retains provenance and neural state, not a source copy.
    const preserveBlob = quarantined ||
      (memoryRecipe === "total-recall" && brain.config.retainSourceText);
    const source: TrainingSource = {
      id: duplicate?.id ?? randomUUID(),
      name: basename(finalUrl.pathname) || `${finalUrl.hostname}-${kind}`,
      kind: effectiveKind,
      bytes: fileInfo.size,
      learnedIdeas:
        worker?.source?.learned_ideas ?? brain.ideas.length - before.ideas,
      learnedConcepts:
        worker?.source?.learned_concepts ??
        Object.keys(brain.concepts).length - before.concepts,
      learnedSynapses:
        worker?.source?.synaptic_update_events ??
        worker?.source?.plasticity_events ??
        Object.keys(brain.synapses).length - before.synapses,
      learnedParameterSteps: coverageCount(worker?.source?.parameter_update_steps, 0),
      parametersChanged: workerParametersChanged(worker),
      importedAt: new Date().toISOString(),
      rawTextRetained: false,
      contentHash,
      blobHash: preserveBlob
        ? await this.repository.storeFileAsBlob(lease.path)
        : undefined,
      policy: quarantined ? "archive" : policy,
      provenanceUrl: finalUrl.toString(),
      license: "Web media source; verify the publisher's terms",
      licenseUrl: finalUrl.toString()
    };
    brain.trainingSources = [
      ...brain.trainingSources.filter((value) => value.id !== source.id), source
    ];
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
    await releaseCrawlSourceLease(lease);
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
    progress: (value: number, message: string) => void = () => undefined,
    cancellationSignal?: AbortSignal,
    jobId = ""
  ): Promise<WebCrawlResult> {
    request = {
      ...request,
      policy: requireNewDataIngestionPolicy(request.policy ?? "encode")
    };
    const start = new URL(request.url);
    const startUrlClass = await assertSafeRemoteUrl(start, {
      allowLoopback: true
    });
    // Only a crawl explicitly started on loopback may ever visit loopback.
    // This prevents a remote HTTPS page/link from turning the crawler into an
    // internal-network proxy while keeping local development sources useful.
    const allowLoopback = startUrlClass === "loopback";
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
          const timeout = AbortSignal.timeout(30_000);
          const signal = cancellationSignal
            ? AbortSignal.any([timeout, cancellationSignal])
            : timeout;
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
              allowLoopback,
              beforeRequest: async (current) => {
                if (current.origin !== url.origin) {
                  throw new CrawlPolicyError(
                    "robots.txt redirected outside its protected origin"
                  );
                }
                return scheduler.acquire(current, signal);
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
              await assertSafeRemoteUrl(pageUrl, { allowLoopback });
              let response: Response | undefined;
              for (let attempt = 0; attempt < 3; attempt += 1) {
                const timeout = AbortSignal.timeout(120_000);
                const signal = cancellationSignal
                  ? AbortSignal.any([timeout, cancellationSignal])
                  : timeout;
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
                    allowLoopback,
                    beforeRequest: async (current) => {
                      await assertCrawlPolicy(current);
                      return scheduler.acquire(current, signal);
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
                if (response.status >= 400 && response.status < 500 &&
                    ![408, 429].includes(response.status)) {
                  throw new CrawlPolicyError(`HTTP ${response.status}: source rejected`);
                }
                throw new Error(`HTTP ${response.status}: source must be retried`);
              }
              const finalUrl = new URL(response.url);
              await assertSafeRemoteUrl(finalUrl, { allowLoopback });
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
            if (!page.skipped) {
              stopped = true;
              frontier.recordWarning(page.entry.url, page.error || "Crawl fetch failed before ingestion.");
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
                  page.mediaKind,
                  cancellationSignal,
                  jobId
                )
              : await (async (): Promise<IngestResult> => {
                  if (!isCrawledText(page.contentType, page.finalUrl)) {
                    throw new CrawlPolicyError(
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
                    page.contentType,
                    cancellationSignal,
                    jobId
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
            if (!(error instanceof CrawlPolicyError)) {
              stopped = true;
              frontier.recordWarning(
                page.entry.url,
                error instanceof Error ? error.message : String(error)
              );
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

  async importUrl(
    request: ImportUrlRequest,
    options: { initializing?: boolean } = {}
  ): Promise<BrainDocument> {
    const downloaded = await this.downloadCatalogArtifactToFile(request);
    try {
      return await this.repository.importBundle(downloaded.path, options);
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
    return effectiveToolPermissions(brain.toolPermissions);
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
    const id = canonicalToolPermissionId(normalizeToolId(toolId));
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
    return effectiveToolPermissions(saved.toolPermissions);
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

  private mergeEvidenceDirectory(targetBrainId: string): string {
    return join(
      this.repository.brainDirectory(targetBrainId),
      "activity",
      "merge-plans"
    );
  }

  private async persistMergeEvidencePlan(
    descriptor: StoredMergePlanDescriptor,
    reviewToken: string,
    descriptorJson: string
  ): Promise<void> {
    const directory = this.mergeEvidenceDirectory(descriptor.targetBrainId);
    const existing = await MergeEvidencePlan.openExisting(directory, reviewToken);
    if (existing) {
      try {
        const summary = existing.integrity();
        if (summary.descriptorSha256 !== sha256(descriptorJson)) {
          throw new Error("The reviewed merge token names another evidence plan.");
        }
        return;
      } finally {
        existing.close();
      }
    }
    const builder = await MergeEvidencePlanBuilder.begin(directory, {
      sourceBrainId: descriptor.sourceBrainId,
      targetBrainId: descriptor.targetBrainId,
      sourceUpdatedAt: descriptor.sourceUpdatedAt,
      targetUpdatedAt: descriptor.targetUpdatedAt,
      substrateDigest: descriptor.substrate.digest
    });
    const batch: MergeEvidencePlanEntry[] = [];
    let ordinal = 0;
    let sourceDigest = "0".repeat(64);
    const flush = async (): Promise<void> => {
      if (!batch.length) return;
      const bytes = batch.reduce(
        (sum, entry) => sum + Buffer.byteLength(JSON.stringify(entry.source)),
        0
      );
      await assertDiskReserve(directory, bytes * 3);
      if (!hasTextMemoryHeadroom(bytes)) {
        throw new CrawlResourcePause(
          "Merge evidence planning paused at the memory headroom watermark."
        );
      }
      builder.append(batch.splice(0));
    };
    try {
      const scan = await this.repository.visitNovelTrainingSources(
        descriptor.sourceBrainId,
        descriptor.targetBrainId,
        async (source, fingerprint) => {
          ordinal += 1;
          sourceDigest = sha256(JSON.stringify({
            ordinal,
            fingerprint,
            payloadSha256: sha256(JSON.stringify(source)),
            previousSha256: sourceDigest
          }));
          batch.push({
            sourceSequence: ordinal,
            fingerprint,
            targetSourceId: mergedEvidenceSourceId(
              descriptor.targetBrainId,
              fingerprint,
              source.id
            ),
            source
          });
          if (batch.length >= 100) await flush();
        }
      );
      await flush();
      if (
        scan.novel !== descriptor.newEvidence ||
        sourceDigest !== descriptor.evidenceDigest ||
        builder.summary().evidenceCount !== descriptor.newEvidence
      ) {
        throw new Error("The source evidence changed while its merge plan was built.");
      }
      await builder.commit(reviewToken, sha256(descriptorJson), descriptorJson);
    } catch (error) {
      await builder.abort();
      throw error;
    }
  }

  private async loadStoredMergePlan(
    sourceBrainId: string,
    targetBrainId: string,
    reviewToken: string
  ): Promise<MergePlan | undefined> {
    const evidence = await MergeEvidencePlan.openExisting(
      this.mergeEvidenceDirectory(targetBrainId),
      reviewToken
    );
    if (!evidence) return undefined;
    let descriptor: StoredMergePlanDescriptor | undefined;
    let evidenceSummary: MergeEvidencePlanSummary;
    try {
      evidenceSummary = evidence.integrity();
      descriptor = evidence.descriptor<StoredMergePlanDescriptor>();
    } finally {
      evidence.close();
    }
    if (
      !descriptor ||
      descriptor.schemaVersion !== 1 ||
      descriptor.sourceBrainId !== sourceBrainId ||
      descriptor.targetBrainId !== targetBrainId ||
      descriptor.preview.reviewToken !== reviewToken ||
      evidenceSummary.evidenceCount !== descriptor.newEvidence ||
      evidenceSummary.sourceBrainId !== sourceBrainId ||
      evidenceSummary.targetBrainId !== targetBrainId
    ) {
      throw new Error("The persisted merge evidence plan is invalid.");
    }
    const [source, target] = await Promise.all([
      this.repository.get(sourceBrainId),
      this.repository.get(targetBrainId)
    ]);
    if (
      source.updatedAt !== descriptor.sourceUpdatedAt ||
      target.updatedAt !== descriptor.targetUpdatedAt
    ) {
      throw new Error("The merge preview is stale. Review the branch overlay again.");
    }
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
    if (!substrate || substrate.digest !== descriptor.substrate.digest) {
      throw new Error("The merge preview is stale. Review the branch overlay again.");
    }
    return {
      source,
      target,
      evidenceDigest: descriptor.evidenceDigest,
      newEvidence: descriptor.newEvidence,
      duplicateEvidence: descriptor.duplicateEvidence,
      files: descriptor.files,
      substrate,
      preview: descriptor.preview
    };
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

    let evidenceDigest = "0".repeat(64);
    let evidenceOrdinal = 0;
    const sourceEvidenceScan = await this.repository.visitNovelTrainingSources(
      sourceBrainId,
      targetBrainId,
      async (sourceEvidence, fingerprint) => {
      evidenceOrdinal += 1;
      evidenceDigest = sha256(JSON.stringify({
        ordinal: evidenceOrdinal,
        fingerprint,
        payloadSha256: sha256(JSON.stringify(sourceEvidence)),
        previousSha256: evidenceDigest
      }));
      if (!retainsMergedBlob(target, sourceEvidence)) return;
      let contents: Buffer | undefined;
      let blobHash: string | undefined;
      let absoluteSourcePath: string | undefined;
      if (sourceEvidence.blobHash) {
        try {
          contents = await this.repository.getBlob(sourceEvidence.blobHash);
          blobHash = sourceEvidence.blobHash;
        } catch {
          skippedFiles += 1;
          conflicts.push(
            `Evidence ${sourceEvidence.name} references a missing or corrupt blob; metadata only will merge.`
          );
          return;
        }
      } else if (
        sourceEvidence.path &&
        pathWithin(sourceDirectory, sourceEvidence.path)
      ) {
        try {
          const info = await lstat(sourceEvidence.path);
          if (info.isFile() && !info.isSymbolicLink()) {
            contents = await readFile(sourceEvidence.path);
            absoluteSourcePath = sourceEvidence.path;
          }
        } catch {
          // Branch-local evidence can be absent after a user removes it. The
          // metadata remains useful and the missing bytes are reported below.
        }
      }
      if (!contents) {
        if (sourceEvidence.path || sourceEvidence.blobHash) {
          skippedFiles += 1;
          conflicts.push(
            `Evidence ${sourceEvidence.name} has no safe retained branch-local file; metadata only will merge.`
          );
        }
        return;
      }
      await addFile({
        kind: "evidence",
        sourcePath: `evidence/${safeMergeName(sourceEvidence.name)}`,
        sha256: sha256(contents),
        bytes: contents.byteLength,
        absoluteSourcePath,
        blobHash,
        evidenceFingerprint: fingerprint
      });
      }
    );

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
    const duplicateEvidence = sourceEvidenceScan.total - sourceEvidenceScan.novel;
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
      evidence: {
        count: sourceEvidenceScan.novel,
        digest: evidenceDigest
      },
      files: publicFiles,
      skippedFiles,
      conflicts: boundedConflicts,
      substrate
    };
    const reviewToken = sha256(JSON.stringify(reviewDescriptor));
    const preview: AgentMergePreview = {
      sourceBrainId,
      targetBrainId,
      reviewToken,
      substrate,
      newConcepts: substrate.additions.neurons,
      newIdeas: substrate.additions.assemblies,
      newSynapses: substrate.additions.synapses,
      newEvidence: sourceEvidenceScan.novel,
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
    const storedDescriptor: StoredMergePlanDescriptor = {
      schemaVersion: 1,
      sourceBrainId,
      targetBrainId,
      sourceUpdatedAt: source.updatedAt,
      targetUpdatedAt: target.updatedAt,
      evidenceDigest,
      newEvidence: sourceEvidenceScan.novel,
      duplicateEvidence,
      files,
      substrate,
      preview
    };
    await this.persistMergeEvidencePlan(
      storedDescriptor,
      reviewToken,
      JSON.stringify(storedDescriptor)
    );
    return {
      source,
      target,
      evidenceDigest,
      newEvidence: sourceEvidenceScan.novel,
      duplicateEvidence,
      files,
      substrate,
      preview
    };
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

  private async applyMergeEvidence(plan: MergePlan): Promise<void> {
    const evidence = await MergeEvidencePlan.open(
      this.mergeEvidenceDirectory(plan.target.id),
      plan.preview.reviewToken
    );
    try {
      if (evidence.summary().evidenceCount === 0) return;
      evidence.startOrResume();
      while (evidence.summary().state !== "complete") {
        const page = evidence.pendingPage(100);
        if (!page.entries.length) {
          throw new Error("The merge evidence cursor ended before its declared coverage.");
        }
        const incomingWriteBytes = page.entries.reduce(
          (sum, entry) => sum + Buffer.byteLength(JSON.stringify(entry.source)) * 3,
          0
        );
        const filesystem = await statfs(this.repository.brainDirectory(plan.target.id));
        const diskFreeBytes = Number(filesystem.bavail) * Number(filesystem.bsize);
        const diskTotalBytes = Number(filesystem.blocks) * Number(filesystem.bsize);
        const diskReserveBytes = Math.max(
          CRAWL_DISK_RESERVE_MINIMUM,
          Math.min(4 * 1024 * 1024 * 1024, Math.floor(diskTotalBytes * 0.02))
        );
        const checkpoint = evidence.checkpointResource({
          diskFreeBytes,
          diskReserveBytes,
          incomingWriteBytes,
          memoryHeadroom: hasTextMemoryHeadroom(incomingWriteBytes)
        });
        if (checkpoint.paused) {
          throw new CrawlResourcePause(
            `${checkpoint.summary.pauseReason} Retry the same reviewed merge to resume.`
          );
        }
        const sources = page.entries.map((row) => {
          const copied = JSON.parse(JSON.stringify(row.source)) as TrainingSource;
          copied.id = row.targetSourceId;
          delete copied.path;
          const evidenceFile = plan.files.find(
            (file) =>
              file.kind === "evidence" &&
              file.evidenceFingerprint === row.fingerprint
          );
          if (evidenceFile) {
            copied.path = join(
              this.repository.brainDirectory(plan.target.id),
              ...evidenceFile.destinationPath.split("/")
            );
            copied.blobHash = evidenceFile.sha256;
          } else {
            delete copied.blobHash;
          }
          const retainRawText =
            (plan.target.config.memoryRecipe ?? "adaptive-retention") === "total-recall" &&
            plan.target.config.retainSourceText;
          if (!retainRawText) {
            copied.rawTextRetained = false;
            delete copied.rawText;
          }
          return copied;
        });
        await this.repository.appendTrainingSourceProjection(plan.target.id, sources);
        evidence.recordApplied(page.entries.at(-1)!.sequence, page.entries.length);
      }
    } finally {
      evidence.close();
    }
  }

  private async validateMergeFiles(plan: MergePlan): Promise<void> {
    for (const file of plan.files) {
      let hash: string;
      if (file.blobHash) {
        hash = sha256(await this.repository.getBlob(file.blobHash));
      } else if (file.absoluteSourcePath) {
        const sourceDirectory = this.repository.brainDirectory(plan.source.id);
        const info = await lstat(file.absoluteSourcePath);
        if (
          !pathWithin(sourceDirectory, file.absoluteSourcePath) ||
          info.isSymbolicLink() ||
          !info.isFile()
        ) {
          throw new Error(`The merge preview is stale: ${file.sourcePath} is unsafe.`);
        }
        hash = sha256(await readFile(file.absoluteSourcePath));
      } else {
        throw new Error(`The merge preview is stale: ${file.sourcePath} is unavailable.`);
      }
      if (hash !== file.sha256) {
        throw new Error(
          `The merge preview is stale: source ${file.sourcePath} changed after review.`
        );
      }
    }
  }

  private async mergeUnlocked(
    sourceBrainId: string,
    targetBrainId: string,
    reviewToken: string
  ): Promise<BrainDocument> {
    if (!/^[a-f0-9]{64}$/.test(reviewToken)) {
      throw new Error("A valid merge review token is required.");
    }
    const plan =
      await this.loadStoredMergePlan(sourceBrainId, targetBrainId, reviewToken) ??
      await this.buildMergePlan(sourceBrainId, targetBrainId);
    if (plan.preview.reviewToken !== reviewToken) {
      throw new Error("The merge preview is stale. Review the branch overlay again.");
    }

    await this.validateMergeFiles(plan);
    await this.applyMergeEvidence(plan);

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
        throw new Error(
          `The merge preview is stale: source ${file.sourcePath} changed after review.`
        );
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
          evidence: {
            count: plan.newEvidence,
            digest: plan.evidenceDigest
          },
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
  private readonly observationSessions = new Map<string, LiveObservationRecord>();
  private readonly initializingBrains = new Set<string>();
  private readonly cancellationControllers = new Map<string, AbortController>();
  private readonly preparationOnlyJobs = new Set<string>();
  private readonly jobOperations = new Map<string, Promise<void>>();
  private readonly jobCancellationOperations = new Map<string, Promise<RuntimeJob>>();
  private readonly jobResumeLabels = new Map<string, string>();

  constructor(
    private readonly service: BrainService,
    private readonly engine: EngineSupervisor,
    private readonly mediaArtifacts?: MediaArtifactRegistry
  ) {
    super();
    engine.on("event", (event: EngineEvent) => this.consumeEngineEvent(event));
    engine.on("activity", (event: EngineActivityTransition) =>
      this.consumeEngineActivity(event)
    );
  }

  list(brainId?: string): RuntimeJob[] {
    return [...this.jobs.values()]
      .filter((job) => !brainId || job.brainId === brainId)
      .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
      .map((job) => ({ ...job }));
  }

  listArtifacts(brainId: string, cursor?: string, limit?: number) {
    return this.mediaArtifacts?.listArtifacts(brainId, cursor, limit) ??
      Promise.resolve({ brainId, artifacts: [], totalArtifacts: 0 });
  }

  beginInitialization(brainId: string): void {
    this.initializingBrains.add(brainId);
  }

  completeInitialization(brainId: string): void {
    this.initializingBrains.delete(brainId);
  }

  isInitializing(brainId: string): boolean {
    return this.initializingBrains.has(brainId);
  }

  isLearning(brainId: string): boolean {
    return this.isInitializing(brainId) || [...this.jobs.values()].some(
      (job) =>
        job.brainId === brainId &&
        ["queued", "running", "cancelling"].includes(job.state) &&
        ["training", "ingestion", "crawl", "image", "audio", "video"].includes(
          job.kind
        )
    );
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
        if (timer) clearTimeout(timer);
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
      const timer = timeoutMs === 0
        ? undefined
        : setTimeout(
            () => cancelAndReject("Runtime job timed out."),
            Math.max(1_000, Math.min(86_400_000, Math.round(timeoutMs)))
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

  startIngestion(
    request: DatasetStartRequest,
    seedTelemetry?: TrainingTelemetry
  ): RuntimeJob {
    const policy = requireNewDataIngestionPolicy(request.policy ?? "encode");
    return this.startIngestionWithPolicy(request, policy, seedTelemetry);
  }

  async resumeIngestion(
    request: DatasetStartRequest,
    seedTelemetry?: TrainingTelemetry
  ): Promise<RuntimeJob> {
    if (request.policy !== undefined) {
      return this.startIngestion(
        { ...request, resume: true },
        seedTelemetry
      );
    }
    const persisted = await this.service.datasets.progress(
      request.brainId,
      request.manifestId
    );
    const policy = persisted.lastEntryReceipt?.policy;
    if (!policy) {
      throw new Error(
        "Resuming this dataset requires its original ingestion policy; no completed entry records it yet."
      );
    }
    return this.startIngestionWithPolicy(
      { ...request, resume: true },
      policy,
      seedTelemetry
    );
  }

  private startIngestionWithPolicy(
    request: DatasetStartRequest,
    policy: PersistedDataIngestionPolicy,
    seedTelemetry?: TrainingTelemetry
  ): RuntimeJob {
    const epochs = request.epochs ?? 1;
    if (!Number.isSafeInteger(epochs) || epochs < 1) {
      throw new Error("Dataset epochs must be a positive safe integer.");
    }
    const job = this.createJob(
      request.brainId,
      "ingestion",
      "Preparing neural substrate",
      seedTelemetry,
      "preparing-neural-substrate"
    );
    this.launch(job, async () => {
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
        policy,
        () => Boolean(job.cancelled),
        (progress, message, detail) => {
          if (job.cancelled) return;
          const atMs = telemetryClockMs();
          const traversalMeasured =
            detail?.traversalMeasured ??
            (detail?.coverage !== undefined && detail.durableBaseline !== true);
          if (
            job.phase === "preparing-neural-substrate" &&
            !traversalMeasured
          ) {
            job.progress = 0;
            job.label = "Preparing neural substrate";
            if (detail) updateIngestionTelemetry(job, detail, atMs);
            job.output = {
              phase: "preparing-neural-substrate",
              manifestId: request.manifestId,
              status: message
            };
            job.updatedAt = new Date(atMs).toISOString();
            this.publish(job);
            return;
          }
          job.phase = "traversing-dataset";
          job.progress = Math.max(job.progress, Math.min(0.99, progress));
          job.label = message;
          updateIngestionTelemetry(job, detail, atMs);
          job.output = {
            phase: "training",
            manifestId: request.manifestId,
            ...(detail?.coverage ? { coverage: detail.coverage } : {}),
            progress: { ...detail }
          };
          job.updatedAt = new Date(atMs).toISOString();
          this.publish(job);
        },
        false,
        epochs,
        true,
        job.id,
        this.cancellationControllers.get(job.id)?.signal
      );
    });
    return { ...job };
  }

  startBuildResource(
    request: BuildResourceStartRequest,
    paths: string[],
    onManifestCommitted: (manifestId: string) => Promise<void> = async () => undefined,
    onCompleted: () => Promise<void> = async () => undefined,
    seedTelemetry?: TrainingTelemetry
  ): RuntimeJob {
    if (request.policy !== undefined) {
      requireNewDataIngestionPolicy(request.policy);
    }
    const epochs = request.epochs ?? 1;
    if (!Number.isSafeInteger(epochs) || epochs < 1) {
      throw new Error("Dataset epochs must be a positive safe integer.");
    }
    if (paths.length === 0) {
      throw new Error("The selected Build resource has no source paths.");
    }
    const job = this.createJob(
      request.brainId,
      "ingestion",
      "Preparing initial dataset snapshot",
      seedTelemetry
    );
    const controller = this.cancellationControllers.get(job.id)!;
    this.preparationOnlyJobs.add(job.id);
    this.launch(job, async () => {
      const manifest = await this.service.previewDataset(
        request.brainId,
        paths,
        {
          signal: controller.signal,
          onProgress: (preview) => {
            if (job.cancelled) return;
            const fileProgress = preview.currentFileBytes
              ? Math.max(
                  0,
                  Math.min(
                    1,
                    (preview.currentFileHashedBytes ?? 0) /
                      preview.currentFileBytes
                  )
                )
              : 0;
            job.progress = Math.max(
              job.progress,
              Math.min(0.11, 0.02 + fileProgress * 0.09)
            );
            job.label = preview.phase === "committing"
              ? "Committing initial dataset snapshot"
              : preview.phase === "hashing"
                ? `Hashing initial source ${preview.hashedFiles + 1}${
                    preview.currentFile ? ` · ${preview.currentFile.slice(-120)}` : ""
                  }`
                : `Discovering initial sources · ${preview.discoveredFiles} found`;
            job.output = { phase: "snapshot", preview };
            job.updatedAt = new Date().toISOString();
            this.publish(job);
          }
        }
      );
      if (job.cancelled) return { manifest, paused: true };
      await onManifestCommitted(manifest.id);
      if (job.cancelled) return { manifest, paused: true };
      this.preparationOnlyJobs.delete(job.id);
      job.progress = Math.max(job.progress, 0.12);
      job.label = `Training on ${manifest.discoveredFiles} committed initial source${
        manifest.discoveredFiles === 1 ? "" : "s"
      }`;
      job.output = {
        phase: "training",
        manifest: {
          id: manifest.id,
          discoveredFiles: manifest.discoveredFiles,
          discoveredBytes: manifest.discoveredBytes,
          manifestHash: manifest.manifestHash
        }
      };
      job.updatedAt = new Date().toISOString();
      this.publish(job);
      const result = await this.service.ingestManifest(
        request.brainId,
        manifest.id,
        request.policy ?? "pretrain",
        () => Boolean(job.cancelled),
        (progress, message, detail) => {
          if (job.cancelled) return;
          const atMs = telemetryClockMs();
          job.progress = Math.max(
            job.progress,
            Math.min(0.99, 0.12 + progress * 0.87)
          );
          job.label = message;
          updateIngestionTelemetry(job, detail, atMs);
          job.output = {
            ...(objectRecord(job.output) ?? {}),
            phase: "training",
            ...(detail?.coverage ? { coverage: detail.coverage } : {}),
            progress: { ...detail }
          };
          job.updatedAt = new Date(atMs).toISOString();
          this.publish(job);
        },
        false,
        epochs,
        true,
        job.id,
        controller.signal
      );
      if (
        !job.cancelled &&
        !incompleteIngestionReason(job, result)
      ) {
        await onCompleted();
      }
      return result;
    });
    return { ...job };
  }

  startCrawl(request: WebCrawlRequest): RuntimeJob {
    if (request.policy !== undefined) {
      requireNewDataIngestionPolicy(request.policy);
    }
    const job = this.createJob(
      request.brainId, "crawl",
      (request.quarantine ?? true)
        ? "Crawling quarantined web sources"
        : "Crawling web sources for neural learning"
    );
    this.launch(job, () =>
      this.service.crawlWeb(
        request,
        () => Boolean(job.cancelled),
        (progress, message) => {
          if (job.cancelled) return;
          job.progress = Math.max(job.progress, Math.min(0.99, progress));
          job.label = message;
          job.updatedAt = new Date().toISOString();
          this.publish(job);
        },
        this.cancellationControllers.get(job.id)?.signal,
        job.id
      )
    );
    return { ...job };
  }

  generate(request: ModalityGenerateRequest): RuntimeJob {
    const normalizedRequest = normalizeModalityGenerateRequest(request);
    const job = this.createJob(
      normalizedRequest.brainId,
      normalizedRequest.modality,
      `Generating ${normalizedRequest.modality}`
    );
    this.launch(job, async () => {
      await this.service.preflightStart(normalizedRequest.brainId);
      const signal = this.cancellationControllers.get(job.id)?.signal;
      const output = await this.engine.request<unknown>(
        "generate_modality",
        {
          jobId: job.id,
          ...normalizedRequest,
          storagePath: this.service.repository.brainDirectory(normalizedRequest.brainId)
        },
        3_600_000,
        signal
      );
      return output;
    });
    return { ...job };
  }

  generateSpeech(value: unknown): RuntimeJob {
    const input = objectRecord(value);
    if (!input || typeof input.brainId !== "string" || typeof input.requestId !== "string" ||
        !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(input.brainId) ||
        !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(input.requestId) ||
        typeof input.text !== "string" || !input.text.trim() || input.text.includes("\0") ||
        (input.rate !== undefined &&
          (typeof input.rate !== "number" || !Number.isFinite(input.rate) || input.rate <= 0))) {
      throw new Error("Invalid same-brain speech waveform request.");
    }
    const request: NeuralSpeechGenerateRequest = {
      brainId: input.brainId, requestId: input.requestId, text: input.text,
      rate: typeof input.rate === "number" ? input.rate : 1
    };
    const job = this.createJob(request.brainId, "audio", "Producing same-brain speech waveform");
    job.speechRequestId = request.requestId;
    this.publish(job);
    this.launch(job, async () => {
      await this.service.preflightStart(request.brainId);
      const output = await this.engine.request("generate_neural_speech", {
        jobId: job.id, brainId: request.brainId, speechRequestId: request.requestId,
        text: request.text, rate: request.rate,
        storagePath: this.service.repository.brainDirectory(request.brainId)
      }, ENGINE_REQUEST_NO_DEADLINE, this.cancellationControllers.get(job.id)?.signal);
      const artifact = objectRecord(output), speech = objectRecord(artifact?.speech);
      if (artifact?.brainId !== request.brainId || artifact?.modality !== "audio" ||
          artifact?.mimeType !== "audio/wav" || speech?.requestId !== request.requestId ||
          speech?.source !== "same-brain-audio-region" || speech?.sameBrain !== true ||
          speech?.textConditioned !== true || speech?.externalModelUsed !== false ||
          speech?.intelligibilityVerified !== false ||
          speech?.textSha256 !== createHash("sha256").update(request.text, "utf8").digest("hex")) {
        throw new Error("The generated waveform is not bound to this exact same-brain speech request.");
      }
      return output;
    });
    return { ...job };
  }

  async startObservation(
    request: LiveObservationSessionStartRequest
  ): Promise<LiveObservationSession> {
    if (
      !request ||
      typeof request.brainId !== "string" ||
      !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(request.brainId)
    ) {
      throw new Error("Invalid live observation brain id.");
    }
    const modalities = Array.isArray(request.modalities)
      ? [...new Set(request.modalities)]
      : [];
    if (
      modalities.length === 0 ||
      modalities.length !== request.modalities.length ||
      modalities.some((value) => !["image", "audio", "video"].includes(value))
    ) {
      throw new Error("Choose one or more distinct live observation modalities.");
    }
    const permission = request.permission;
    if (
      !permission ||
      permission.granted !== true ||
      permission.scope !== "session" ||
      !["camera", "microphone", "screen", "mixed"].includes(permission.source) ||
      typeof permission.grantedAt !== "string" ||
      !Number.isFinite(Date.parse(permission.grantedAt)) ||
      (permission.deviceIdHash !== undefined &&
        !/^[a-f0-9]{64}$/i.test(permission.deviceIdHash))
    ) {
      throw new Error("Trusted session capture permission is required.");
    }
    const permitted = {
      camera: new Set<LiveObservationModality>(["image", "video"]),
      microphone: new Set<LiveObservationModality>(["audio"]),
      screen: new Set<LiveObservationModality>(["image", "video"]),
      mixed: new Set<LiveObservationModality>(["image", "audio", "video"])
    }[permission.source];
    if (modalities.some((value) => !permitted.has(value))) {
      throw new Error("Capture permission does not cover every requested modality.");
    }
    const retention = request.retention ?? "neural";
    if (!(["working", "neural"] as const).includes(retention)) {
      throw new Error("Live observation retention must be working or neural.");
    }
    const capture = normalizeObservationCapture(request, modalities);
    const transport = liveObservationTransportEnvelope(capture, modalities);
    const maxInFlight = request.maxInFlight ?? transport.maxInFlight;
    if (!Number.isSafeInteger(maxInFlight) || maxInFlight < 1) {
      throw new Error("Live observation maxInFlight must be a positive safe integer.");
    }
    if (maxInFlight > transport.maxInFlight) {
      throw new Error(
        `Live observation maxInFlight exceeds the current resource envelope (${transport.maxInFlight}).`
      );
    }

    const brain = await this.service.repository.get(request.brainId);
    if (brain.readiness.state !== "ready") {
      throw new Error("This mind is still completing its initial learning.");
    }
    await this.service.preflightStart(brain.id);
    const storagePath = this.service.repository.brainDirectory(brain.id);
    await this.engine.request(
      "load",
      { brainId: brain.id, config: brain.config, storagePath },
      300_000
    );
    const sessionId = randomUUID();
    const baseToolSchemas = this.service.neuralToolSchemas(brain);
    const studioSchema = baseToolSchemas.find((schema) => schema.id === "studio.ui");
    const sessionToolSchemas = [
      ...baseToolSchemas
        .filter((schema) => schema.id !== "studio.ui" && schema.id !== "device.observe"),
      ...(studioSchema ? [studioSchema] : []),
      {
        id: "device.observe",
        actions: ["configure", "snapshot"],
        grant: "auto" as const
      }
    ];
    const worker = await this.engine.request<Record<string, unknown>>(
      "start_observation",
      {
        sessionId,
        brainId: brain.id,
        storagePath,
        modalities,
        permission: {
          ...permission,
          grantedAt: new Date(permission.grantedAt).toISOString(),
          ...(permission.deviceIdHash
            ? { deviceIdHash: permission.deviceIdHash.toLocaleLowerCase() }
            : {})
        },
        retention,
        capture,
        maxPacketBytes: transport.maxPacketBytes,
        toolSchemas: sessionToolSchemas
      },
      60_000
    );
    const workerCapabilities = objectRecord(worker.capabilities);
    const now = new Date().toISOString();
    const session: LiveObservationSession = {
      id: sessionId,
      brainId: brain.id,
      state: "active",
      modalities,
      permission: {
        ...permission,
        grantedAt: new Date(permission.grantedAt).toISOString(),
        ...(permission.deviceIdHash
          ? { deviceIdHash: permission.deviceIdHash.toLocaleLowerCase() }
          : {})
      },
      retention,
      maxInFlight,
      maxPacketBytes: transport.maxPacketBytes,
      inFlight: 0,
      packetsReceived: 0,
      packetsAccepted: 0,
      packetsDroppedBackpressure: 0,
      bytesAccepted: 0,
      lastSequence: -1,
      createdAt:
        typeof worker.createdAt === "string" &&
        Number.isFinite(Date.parse(worker.createdAt))
          ? new Date(worker.createdAt).toISOString()
          : now,
      updatedAt: now,
      capabilities: {
        imageNeural: workerCapabilities?.imageNeural === true,
        audioNeural: workerCapabilities?.audioNeural === true,
        videoNeural: workerCapabilities?.videoNeural === true
      },
      capture
    };
    this.observationSessions.set(session.id, {
      session,
      lastReceivedSequence: -1,
      lastReceivedTimestampMs: -1,
      pending: new Set(),
      controls: new Map(),
      controlBurstFrames: new Map()
    });
    this.publishObservation({
      type: "started",
      session: cloneObservationSession(session)
    });
    return cloneObservationSession(session);
  }

  async pushObservation(
    packet: LiveObservationPacket
  ): Promise<LiveObservationPacketResult> {
    if (!packet || typeof packet.sessionId !== "string") {
      throw new Error("Invalid live observation packet.");
    }
    const record = this.observationSessions.get(packet.sessionId);
    if (!record) throw new Error("The live observation session was not found.");
    if (record.session.state !== "active") {
      return {
        session: cloneObservationSession(record.session),
        accepted: false,
        reason: "session-stopped"
      };
    }
    if (!record.session.modalities.includes(packet.modality)) {
      throw new Error("The packet modality is not enabled for this session.");
    }
    if (
      !Number.isSafeInteger(packet.sequence) ||
      packet.sequence < 0 ||
      packet.sequence <= record.lastReceivedSequence
    ) {
      throw new Error("Live packet sequence numbers must increase monotonically.");
    }
    if (
      typeof packet.timestampMs !== "number" ||
      !Number.isFinite(packet.timestampMs) ||
      packet.timestampMs < 0 ||
      packet.timestampMs < record.lastReceivedTimestampMs
    ) {
      throw new Error("Live packet timestamps must not move backwards.");
    }
    const mimeType = livePacketMimeType(packet.modality, packet.mimeType);
    const bytes = livePacketBytes(packet.data);
    if (bytes.length < 1 || bytes.length > record.session.maxPacketBytes) {
      throw new Error(
        `A live observation packet must contain 1 to ${record.session.maxPacketBytes} bytes under the current source/resource envelope.`
      );
    }
    const rawSettings = packet.settings ?? {};
    const settings: Record<string, unknown> = Object.fromEntries(
      Object.entries(rawSettings).filter(([key, value]) => {
        if (
          [
            "sampleRate",
            "channels",
            "width",
            "height",
            "durationMs",
            "burstCount"
          ].includes(key)
        ) {
          return Number.isSafeInteger(value) && Number(value) > 0;
        }
        if (key === "burstIndex") {
          return Number.isSafeInteger(value) && Number(value) >= 0;
        }
        if (key === "observationControlId") {
          return typeof value === "string" && /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(value);
        }
        if (key === "resolutionMode") {
          return typeof value === "string" && ["native", "current", "custom"].includes(value);
        }
        return false;
      })
    );
    const controlId = settings.observationControlId;
    if (typeof controlId === "string") {
      const control = record.controls.get(controlId);
      const burstIndex = settings.burstIndex;
      const burstCount = settings.burstCount;
      if (
        !control ||
        control.kind !== "snapshot" ||
        control.state !== "requested" ||
        !Number.isSafeInteger(burstIndex) ||
        !Number.isSafeInteger(burstCount) ||
        Number(burstIndex) < 0 ||
        Number(burstCount) < 1 ||
        Number(burstIndex) >= Number(burstCount)
      ) {
        throw new Error("Invalid or inactive observation snapshot burst.");
      }
      const requestedCount = control.requested.burstCount;
      if (requestedCount !== undefined && requestedCount !== burstCount) {
        throw new Error("Snapshot burst count differs from the audited request.");
      }
    }
    record.lastReceivedSequence = packet.sequence;
    record.lastReceivedTimestampMs = packet.timestampMs;
    record.session.packetsReceived += 1;
    record.session.updatedAt = new Date().toISOString();

    if (record.session.inFlight >= record.session.maxInFlight) {
      record.session.packetsDroppedBackpressure += 1;
      const result: LiveObservationPacketResult = {
        session: cloneObservationSession(record.session),
        accepted: false,
        reason: "backpressure"
      };
      this.publishObservation({
        type: "backpressure",
        session: cloneObservationSession(record.session),
        packet: result
      });
      return result;
    }

    record.session.inFlight += 1;
    const operation = this.deliverObservationPacket(
      record,
      packet,
      bytes,
      mimeType,
      settings
    );
    record.pending.add(operation);
    try {
      return await operation;
    } finally {
      record.pending.delete(operation);
      record.session.inFlight = Math.max(0, record.session.inFlight - 1);
      record.session.updatedAt = new Date().toISOString();
    }
  }

  async stopObservation(sessionId: string): Promise<LiveObservationSession> {
    return this.finishObservation(sessionId, false);
  }

  async cancelObservation(sessionId: string): Promise<LiveObservationSession> {
    return this.finishObservation(sessionId, true);
  }

  async requestObservationControl(
    request: LiveObservationControlRequest
  ): Promise<LiveObservationControl> {
    const record = this.observationSessions.get(request.sessionId);
    if (!record || record.session.state !== "active") {
      throw new Error("The active live observation session was not found.");
    }
    if (!(["configure", "snapshot"] as const).includes(request.kind)) {
      throw new Error("Invalid live observation control kind.");
    }
    const actions: StructuredAction[] = [
      {
        kind: "tool",
        source: "human",
        toolId: "device.observe",
        action: request.kind,
        arguments: request.requested ?? {}
      }
    ];
    const control = this.controlsFromObservationActions(
      record,
      actions,
      "human"
    )[0];
    if (!control) throw new Error("Invalid live observation control request.");
    await this.engine.request(
      "resolve_observation_control",
      {
        brainId: record.session.brainId,
        storagePath: this.service.repository.brainDirectory(record.session.brainId),
        sessionId: record.session.id,
        control
      },
      30_000
    );
    this.publishObservation({
      type: "control",
      session: cloneObservationSession(record.session),
      control
    });
    return control;
  }

  async resolveObservationControl(
    resolution: LiveObservationControlResolution
  ): Promise<LiveObservationControl> {
    if (!resolution || typeof resolution !== "object") {
      throw new Error("Invalid live observation control resolution.");
    }
    const record = this.observationSessions.get(resolution.sessionId);
    if (!record || record.session.state !== "active") {
      throw new Error("The active live observation session was not found.");
    }
    const control = record.controls.get(resolution.controlId);
    const validTransition = Boolean(
      control &&
        ((control.state === "requested" && resolution.state !== "reverted") ||
          (control.state === "applied" && resolution.state === "reverted"))
    );
    if (!control || !validTransition) {
      throw new Error("The pending live observation control was not found.");
    }
    if (!(["applied", "rejected", "cancelled", "reverted"] as const).includes(resolution.state)) {
      throw new Error("Invalid live observation control state.");
    }
    const now = new Date().toISOString();
    if (resolution.state === "applied" || resolution.state === "reverted") {
      const raw = resolution.actual;
      if (!raw) throw new Error("Resolved capture control requires actual values.");
      const mode = raw.mode;
      if (!(["auto", "motion", "balanced", "detail"] as const).includes(mode)) {
        throw new Error("Applied capture control has an invalid mode.");
      }
      const width = optionalPositiveSafeInteger(raw.width, "Applied capture width");
      const height = optionalPositiveSafeInteger(raw.height, "Applied capture height");
      const fps = optionalPositiveSafeInteger(raw.fps, "Applied capture FPS");
      const burstCount = optionalPositiveSafeInteger(raw.burstCount, "Applied burst count");
      const intervalMs = optionalPositiveSafeInteger(raw.intervalMs, "Applied burst interval");
      const durationMs = optionalPositiveSafeInteger(raw.durationMs, "Applied control duration");
      const resolutionMode =
        typeof raw.resolutionMode === "string" &&
        ["native", "current", "custom"].includes(raw.resolutionMode)
          ? raw.resolutionMode as "native" | "current" | "custom"
          : undefined;
      const nativeWidth = record.session.capture.sourceNativeWidth;
      const nativeHeight = record.session.capture.sourceNativeHeight;
      const nativeFps = record.session.capture.sourceNativeFps;
      if (
        (nativeWidth !== undefined && width !== undefined && width > nativeWidth) ||
        (nativeHeight !== undefined && height !== undefined && height > nativeHeight) ||
        (nativeFps !== undefined && fps !== undefined && fps > nativeFps)
      ) {
        throw new Error("Applied capture control exceeds source-native bounds.");
      }
      if (
        control.kind === "snapshot" &&
        (resolutionMode === undefined ||
          width === undefined ||
          height === undefined ||
          burstCount === undefined ||
          intervalMs === undefined)
      ) {
        throw new Error(
          "Applied snapshots require a negotiated resolution and multi-frame burst."
        );
      }
      if (control.kind === "snapshot") {
        const acceptedFrames = record.controlBurstFrames.get(control.id) ?? new Set<number>();
        if (acceptedFrames.size !== burstCount) {
          throw new Error(
            "Snapshot cannot be applied until every audited burst frame enters the brain."
          );
        }
      }
      if (
        control.kind === "snapshot" &&
        resolutionMode === "native" &&
        ((nativeWidth !== undefined && width !== nativeWidth) ||
          (nativeHeight !== undefined && height !== nativeHeight))
      ) {
        throw new Error("Native snapshot dimensions must match the source.");
      }
      if (
        resolution.state === "applied" &&
        control.kind === "configure" &&
        durationMs === undefined
      ) {
        throw new Error("Applied temporary capture configuration requires a duration.");
      }
      if (resolution.state === "reverted" && control.kind !== "configure") {
        throw new Error("Only temporary capture configuration can be reverted.");
      }
      const revision = record.session.capture.revision + 1;
      control.actual = {
        mode,
        ...(width !== undefined ? { width } : {}),
        ...(height !== undefined ? { height } : {}),
        ...(fps !== undefined ? { fps } : {}),
        ...(control.kind === "snapshot" && resolutionMode === "native"
          ? { fullResolution: true as const }
          : {}),
        ...(resolutionMode !== undefined ? { resolutionMode } : {}),
        ...(burstCount !== undefined ? { burstCount } : {}),
        ...(intervalMs !== undefined ? { intervalMs } : {}),
        ...(durationMs !== undefined ? { durationMs } : {}),
        ...(resolution.state === "applied" && control.kind === "configure"
          ? { revertToRevision: record.session.capture.revision }
          : {}),
        revision
      };
      if (control.kind === "configure") {
        record.session.capture = {
          ...record.session.capture,
          mode,
          ...(width !== undefined ? { width } : {}),
          ...(height !== undefined ? { height } : {}),
          ...(fps !== undefined ? { fps } : {}),
          revision
        };
      } else {
        record.session.capture.revision = revision;
      }
    }
    control.state = resolution.state;
    control.reason = boundedWorkerText(resolution.reason, 500);
    control.updatedAt = now;
    record.session.updatedAt = now;
    await this.engine.request(
      "resolve_observation_control",
      {
        brainId: record.session.brainId,
        storagePath: this.service.repository.brainDirectory(record.session.brainId),
        sessionId: record.session.id,
        control
      },
      30_000
    );
    const resolved = {
      ...control,
      requested: { ...control.requested },
      ...(control.actual ? { actual: { ...control.actual } } : {})
    };
    this.publishObservation({
      type: "control",
      session: cloneObservationSession(record.session),
      control: resolved
    });
    return resolved;
  }

  async cancel(jobId: string): Promise<RuntimeJob> {
    const existing = this.jobCancellationOperations.get(jobId);
    if (existing) return existing;
    const operation = this.cancelJob(jobId).finally(() => {
      if (this.jobCancellationOperations.get(jobId) === operation) {
        this.jobCancellationOperations.delete(jobId);
      }
    });
    this.jobCancellationOperations.set(jobId, operation);
    return operation;
  }

  private async cancelJob(jobId: string): Promise<RuntimeJob> {
    const job = this.jobs.get(jobId);
    if (!job) throw new Error("The runtime job was not found.");
    if (["complete", "failed", "cancelled"].includes(job.state)) return { ...job };
    const preparationOnly = this.preparationOnlyJobs.has(job.id);
    const operation = this.jobOperations.get(job.id);
    job.cancelled = true;
    this.mediaArtifacts?.cancelJob(job.brainId, job.id);
    job.state = "cancelling";
    job.queue = undefined;
    job.label = `Cancelling ${job.kind} · waiting for acknowledgement`;
    job.updatedAt = new Date().toISOString();
    this.publish(job);
    const controller = this.cancellationControllers.get(job.id);
    controller?.abort(new Error(`Runtime job ${job.id} was cancelled.`));
    try {
      if (!preparationOnly) {
        const termination = await this.engine.cancelRequest(job.id);
        if (termination.phase !== "not-found" && !termination.acknowledged) {
          throw new Error(`Runtime job ${job.id} did not acknowledge cancellation.`);
        }
        if (
          termination.phase === "running" &&
          !termination.workerTerminationAcknowledged &&
          termination.codecSetupCancellationAcknowledged !== true &&
          termination.artifactCancellationAcknowledged !== true &&
          // Own-voice generation returns only after its selected cooperative
          // safe boundary. Acknowledged speech cancellation need not kill the
          // warm worker (or the already-saved reply that supplied the text).
          !job.speechRequestId
        ) {
          throw new Error(
            `Runtime worker for ${job.id} did not acknowledge process termination.`
          );
        }
      }
      await operation;
      if (!preparationOnly) {
        const acknowledgement = await this.engine.request<Record<string, unknown>>(
          "cancel",
          {
            brainId: job.brainId,
            storagePath: this.service.repository.brainDirectory(job.brainId),
            jobId: job.id,
            kind: job.kind,
            reason: "Cancelled by operator."
          },
          30_000,
          undefined,
          "foreground",
          {
            requestId: `${job.id}.cancel`,
            owner:
              job.kind === "training"
                ? "training"
                : job.kind === "ingestion" || job.kind === "crawl"
                  ? "ingestion"
                  : "modality",
            label: `Acknowledging ${job.kind} cancellation`,
            brainId: job.brainId,
            jobId: job.id
          }
        );
        if (
          acknowledgement.jobId !== job.id ||
          acknowledgement.cancelled !== true ||
          acknowledgement.acknowledged !== true
        ) {
          throw new Error("The neural worker returned an invalid job cancellation acknowledgement.");
        }
      }
    } catch (error) {
      job.error = `Cancellation is still awaiting acknowledgement: ${
        error instanceof Error ? error.message : String(error)
      }`;
      job.updatedAt = new Date().toISOString();
      this.publish(job);
      throw error;
    }
    job.state = "cancelled";
    job.label = `${job.kind} cancelled`;
    job.error = undefined;
    job.updatedAt = new Date().toISOString();
    this.cancellationControllers.delete(job.id);
    this.preparationOnlyJobs.delete(job.id);
    this.jobResumeLabels.delete(job.id);
    this.publish(job);
    return { ...job };
  }

  private async deliverObservationPacket(
    record: LiveObservationRecord,
    packet: LiveObservationPacket,
    bytes: Buffer,
    mimeType: string,
    settings: Record<string, unknown>
  ): Promise<LiveObservationPacketResult> {
    const worker = await this.engine.request<WorkerObservationResult>(
      "observe_packet",
      {
        sessionId: record.session.id,
        brainId: record.session.brainId,
        storagePath: this.service.repository.brainDirectory(record.session.brainId),
        modality: packet.modality,
        sequence: packet.sequence,
        timestampMs: packet.timestampMs,
        mimeType,
        dataBase64: bytes.toString("base64"),
        settings
      },
      120_000
    );
    const rawObservation = objectRecord(worker.observation);
    const expectedHash = createHash("sha256").update(bytes).digest("hex");
    if (
      !rawObservation ||
      rawObservation.packetSha256 !== expectedHash ||
      rawObservation.rawPacketStored !== false ||
      rawObservation.datasetCoverageCommitted !== false ||
      rawObservation.sameBrainSharedIdeaSpace !== true ||
      rawObservation.hiddenBehavioralPrompt !== false
    ) {
      throw new Error("The neural worker returned invalid live observation evidence.");
    }
    const actions = parseModelActions("", worker.actions).map(
      (action): StructuredAction => ({ ...action, source: "organic" })
    );
    const controls = this.controlsFromObservationActions(record, actions);
    for (const control of controls) {
      await this.engine.request(
        "resolve_observation_control",
        {
          brainId: record.session.brainId,
          storagePath: this.service.repository.brainDirectory(record.session.brainId),
          sessionId: record.session.id,
          control
        },
        30_000
      );
    }
    record.session.packetsAccepted += 1;
    record.session.bytesAccepted += bytes.length;
    record.session.lastSequence = packet.sequence;
    if (
      typeof settings.observationControlId === "string" &&
      Number.isSafeInteger(settings.burstIndex)
    ) {
      const frames = record.controlBurstFrames.get(settings.observationControlId) ?? new Set<number>();
      frames.add(Number(settings.burstIndex));
      record.controlBurstFrames.set(settings.observationControlId, frames);
    }
    record.session.updatedAt = new Date().toISOString();
    const assemblyId = boundedWorkerText(rawObservation.assemblyId, 128);
    const rawPerception = objectRecord(rawObservation.perception);
    const perception = rawPerception &&
      rawPerception.rawTilesStored === false &&
      rawPerception.tileSelection === "uniform-resource-scaled" &&
      rawPerception.tileBinding === "ternary-position-vsa" &&
      rawPerception.spatialTileEncoder === "same-brain-ternary-image-pack"
      ? {
          sourceWidth: finiteObservationNumber(rawPerception.sourceWidth),
          sourceHeight: finiteObservationNumber(rawPerception.sourceHeight),
          globalDecodedWidth: finiteObservationNumber(rawPerception.globalDecodedWidth),
          globalDecodedHeight: finiteObservationNumber(rawPerception.globalDecodedHeight),
          tilesAvailable: finiteObservationNumber(rawPerception.tilesAvailable),
          tilesEncoded: finiteObservationNumber(rawPerception.tilesEncoded),
          tileCoverage: finiteObservationNumber(rawPerception.tileCoverage),
          tileInputSize: finiteObservationNumber(rawPerception.tileInputSize),
          tileGridRows: finiteObservationNumber(rawPerception.tileGridRows),
          tileGridColumns: finiteObservationNumber(rawPerception.tileGridColumns),
          tileSelection: "uniform-resource-scaled" as const,
          tileBinding: "ternary-position-vsa" as const,
          spatialTileEncoder: "same-brain-ternary-image-pack" as const,
          temporalFrames: finiteObservationNumber(rawPerception.temporalFrames),
          rawTilesStored: false as const
        }
      : undefined;
    const observationControlId = boundedWorkerText(
      rawObservation.observationControlId,
      128
    );
    const resolutionMode =
      typeof rawObservation.resolutionMode === "string" &&
      ["native", "current", "custom"].includes(rawObservation.resolutionMode)
        ? rawObservation.resolutionMode as "native" | "current" | "custom"
        : undefined;
    const burstIndex = Number.isSafeInteger(rawObservation.burstIndex)
      ? Number(rawObservation.burstIndex)
      : undefined;
    const burstCount = Number.isSafeInteger(rawObservation.burstCount)
      ? Number(rawObservation.burstCount)
      : undefined;
    const result: LiveObservationPacketResult = {
      session: cloneObservationSession(record.session),
      accepted: true,
      observation: {
        ...(assemblyId ? { assemblyId } : { assemblyId: null }),
        spikeRate: finiteObservationNumber(rawObservation.spikeRate),
        novelty: finiteObservationNumber(rawObservation.novelty),
        packetSha256: expectedHash,
        rawPacketStored: false,
        datasetCoverageCommitted: false,
        sameBrainSharedIdeaSpace: true,
        hiddenBehavioralPrompt: false,
        ...(perception ? { perception } : {}),
        ...(observationControlId ? { observationControlId } : {}),
        ...(resolutionMode ? { resolutionMode } : {}),
        ...(burstIndex !== undefined ? { burstIndex } : {}),
        ...(burstCount !== undefined ? { burstCount } : {})
      },
      ...(actions.length > 0 ? { actions } : {}),
      ...(controls.length > 0 ? { controls } : {})
    };
    this.publishObservation({
      type: "packet",
      session: cloneObservationSession(record.session),
      packet: result,
      ...(actions.length > 0 ? { actions } : {})
    });
    if (actions.length > 0) {
      this.publishObservation({
        type: "action",
        session: cloneObservationSession(record.session),
        actions
      });
    }
    for (const control of controls) {
      this.publishObservation({
        type: "control",
        session: cloneObservationSession(record.session),
        control
      });
    }
    return result;
  }

  private controlsFromObservationActions(
    record: LiveObservationRecord,
    actions: StructuredAction[],
    source: LiveObservationControl["source"] = "brain"
  ): LiveObservationControl[] {
    const controls: LiveObservationControl[] = [];
    for (const action of actions) {
      if (
        action.toolId !== "device.observe" ||
        !["configure", "snapshot"].includes(action.action ?? "")
      ) {
        continue;
      }
      const kind = action.action as LiveObservationControl["kind"];
      if (
        kind === "snapshot" &&
        (!record.session.modalities.some((value) => value !== "audio") ||
          !["camera", "screen", "mixed"].includes(record.session.permission.source))
      ) {
        continue;
      }
      const raw = action.arguments;
      const rawMode = raw.mode;
      const mode =
        typeof rawMode === "string" &&
        (["auto", "motion", "balanced", "detail"] as const).includes(
          rawMode as LiveObservationCaptureMode
        )
          ? (rawMode as LiveObservationCaptureMode)
          : undefined;
      let width: number | undefined;
      let height: number | undefined;
      let fps: number | undefined;
      let burstCount: number | undefined;
      let intervalMs: number | undefined;
      let durationMs: number | undefined;
      const resolutionMode =
        kind === "snapshot" &&
        typeof raw.resolutionMode === "string" &&
        ["native", "current", "custom"].includes(raw.resolutionMode)
          ? raw.resolutionMode as "native" | "current" | "custom"
          : kind === "snapshot"
            ? "native" as const
            : undefined;
      try {
        width = optionalPositiveSafeInteger(raw.width, "Requested capture width");
        height = optionalPositiveSafeInteger(raw.height, "Requested capture height");
        fps = optionalPositiveSafeInteger(raw.fps, "Requested capture FPS");
        burstCount = optionalPositiveSafeInteger(raw.burstCount, "Requested burst count");
        intervalMs = optionalPositiveSafeInteger(raw.intervalMs, "Requested burst interval");
        durationMs = optionalPositiveSafeInteger(raw.durationMs, "Requested control duration");
      } catch {
        continue;
      }
      if (
        kind === "configure" &&
        mode === undefined &&
        width === undefined &&
        height === undefined &&
        fps === undefined
      ) {
        continue;
      }
      if (
        kind === "snapshot" &&
        resolutionMode === "custom" &&
        (width === undefined || height === undefined)
      ) {
        continue;
      }
      const now = new Date().toISOString();
      const control: LiveObservationControl = {
        id: randomUUID(),
        sessionId: record.session.id,
        kind,
        source,
        temporary: true,
        requested: {
          ...(mode !== undefined ? { mode } : {}),
          ...(width !== undefined ? { width } : {}),
          ...(height !== undefined ? { height } : {}),
          ...(fps !== undefined ? { fps } : {}),
          ...(kind === "snapshot" && resolutionMode === "native"
            ? { fullResolution: true as const }
            : {}),
          ...(resolutionMode !== undefined ? { resolutionMode } : {}),
          ...(burstCount !== undefined ? { burstCount } : {}),
          ...(intervalMs !== undefined ? { intervalMs } : {}),
          ...(durationMs !== undefined ? { durationMs } : {})
        },
        state: "requested",
        createdAt: now,
        updatedAt: now
      };
      record.controls.set(control.id, control);
      controls.push({ ...control, requested: { ...control.requested } });
    }
    return controls;
  }

  private async finishObservation(
    sessionId: string,
    cancelled: boolean
  ): Promise<LiveObservationSession> {
    if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(sessionId)) {
      throw new Error("Invalid live observation session id.");
    }
    const record = this.observationSessions.get(sessionId);
    if (!record) throw new Error("The live observation session was not found.");
    if (record.session.state !== "active") {
      return cloneObservationSession(record.session);
    }
    record.session.state = "stopping";
    record.session.updatedAt = new Date().toISOString();
    await Promise.allSettled([...record.pending]);
    for (const control of record.controls.values()) {
      if (!["requested", "applied"].includes(control.state)) continue;
      control.state = "cancelled";
      control.reason = "Live observation session ended.";
      control.updatedAt = new Date().toISOString();
      await this.engine
        .request(
          "resolve_observation_control",
          {
            brainId: record.session.brainId,
            storagePath: this.service.repository.brainDirectory(record.session.brainId),
            sessionId,
            control
          },
          30_000
        )
        .catch(() => undefined);
      this.publishObservation({
        type: "control",
        session: cloneObservationSession(record.session),
        control: {
          ...control,
          requested: { ...control.requested },
          ...(control.actual ? { actual: { ...control.actual } } : {})
        }
      });
    }
    try {
      await this.engine.request(
        cancelled ? "cancel_observation" : "stop_observation",
        {
          sessionId,
          brainId: record.session.brainId,
          storagePath: this.service.repository.brainDirectory(record.session.brainId)
        },
        30_000
      );
    } catch {
      // The local session is still closed if a restarted worker has already
      // discarded its bounded, raw-byte-free observation record.
    } finally {
      record.session.state = cancelled ? "cancelled" : "stopped";
      record.session.inFlight = 0;
      record.session.updatedAt = new Date().toISOString();
      this.observationSessions.delete(sessionId);
    }
    const session = cloneObservationSession(record.session);
    this.publishObservation({
      type: cancelled ? "cancelled" : "stopped",
      session
    });
    return session;
  }

  private publishObservation(event: LiveObservationEvent): void {
    this.emit("observation", event);
  }

  private createJob(
    brainId: string,
    kind: RuntimeJob["kind"],
    label: string,
    seedTelemetry?: TrainingTelemetry,
    phase?: RuntimeJob["phase"]
  ): JobRecord {
    const atMs = telemetryClockMs();
    const now = new Date(atMs).toISOString();
    const telemetry = resumedTelemetry(seedTelemetry, atMs);
    const job: JobRecord = {
      id: randomUUID(),
      brainId,
      kind,
      state: "queued",
      progress: 0,
      label,
      ...(phase ? { phase } : {}),
      createdAt: now,
      updatedAt: now,
      ...(telemetry ? { telemetry } : {})
    };
    this.jobs.set(job.id, job);
    this.cancellationControllers.set(job.id, new AbortController());
    this.publish(job);
    return job;
  }

  private launch(job: JobRecord, operation: () => Promise<unknown>): void {
    const running = this.run(job, operation).finally(() => {
      if (this.jobOperations.get(job.id) === running) {
        this.jobOperations.delete(job.id);
      }
    });
    this.jobOperations.set(job.id, running);
    void running;
  }

  private async run(job: JobRecord, operation: () => Promise<unknown>): Promise<void> {
    if (job.cancelled) return;
    job.state = "running";
    if (job.phase !== "preparing-neural-substrate") {
      job.progress = Math.max(job.progress, 0.01);
    }
    job.updatedAt = new Date().toISOString();
    this.publish(job);
    try {
      let output = await operation();
      if (job.cancelled) return;
      if (
        this.mediaArtifacts &&
        ["image", "audio", "video"].includes(job.kind)
      ) {
        output = await this.mediaArtifacts.completeJob(job.brainId, job.id, output);
      }
      const incompleteReason = incompleteIngestionReason(job, output);
      if (incompleteReason) {
        job.state = "failed";
        job.progress = Math.min(job.progress, 0.99);
        job.label = incompleteReason;
        job.error = incompleteReason;
        job.output = output;
      } else {
        job.state = "complete";
        job.progress = 1;
        job.output = output;
      }
    } catch (error) {
      if (job.cancelled) return;
      this.mediaArtifacts?.cancelJob(job.brainId, job.id);
      job.state = "failed";
      job.error = error instanceof Error ? error.message : String(error);
    } finally {
      this.cancellationControllers.delete(job.id);
      this.preparationOnlyJobs.delete(job.id);
    }
    job.updatedAt = new Date().toISOString();
    this.publish(job);
  }

  private consumeEngineEvent(event: EngineEvent): void {
    if (!event.jobId) return;
    const job = this.jobs.get(event.jobId);
    if (!job || job.cancelled) return;
    // Ingestion worker progress is local to the current manifest entry. The
    // service maps it to byte-weighted whole-manifest progress; consuming it
    // here made the first file's 98% event look like 98% of an entire folder.
    const entryScopedProgress =
      job.kind === "ingestion" && event.type === "job-progress";
    if (!entryScopedProgress && typeof event.progress === "number") {
      job.progress = Math.max(job.progress, Math.min(0.99, Math.max(0, event.progress)));
    }
    if (!entryScopedProgress && event.message) job.label = event.message.slice(0, 200);
    if (event.type === "modality-preview") {
      let preview = normalizeModalityPreview(
        objectRecord(event.data)?.preview ?? event.data,
        typeof event.sequence === "number" ? event.sequence : (job.preview?.revision ?? 0) + 1,
        event.progress,
        event.message
      );
      if (preview && this.mediaArtifacts) {
        preview = this.mediaArtifacts.leasePreview(
          job.brainId,
          `job:${job.brainId}:${job.id}`,
          preview
        );
      }
      if (preview && (!job.preview || preview.revision > job.preview.revision)) {
        job.preview = preview;
      }
    }
    const readings = trainingResourceReadings(
      objectRecord(event.data)?.resourceReadings
    );
    if (
      readings?.diskTotalBytes !== undefined &&
      readings.diskFreeBytes !== undefined
    ) {
      const reserve = readings.mandatoryFreeDiskBytes ?? readings.diskReserveBytes;
      if (reserve !== undefined) {
        job.diskSpace = calculateDiskSpaceReport({
          measuredAt: new Date().toISOString(),
          diskTotalBytes: readings.diskTotalBytes,
          diskFreeBytes: readings.diskFreeBytes,
          mandatoryReserveBytes: reserve,
          components: {
            operationWriteBytes: readings.estimatedWriteBytes ?? 0
          }
        });
      }
    }
    job.updatedAt = new Date().toISOString();
    this.publish(job);
  }

  private consumeEngineActivity(event: EngineActivityTransition): void {
    if (!event.jobId) return;
    const job = this.jobs.get(event.jobId);
    if (
      !job ||
      ["cancelling", "cancelled", "complete", "failed"].includes(job.state)
    ) {
      return;
    }
    if (event.state === "queued" && event.queuedBehind) {
      if (!this.jobResumeLabels.has(job.id)) {
        this.jobResumeLabels.set(job.id, job.label);
      }
      job.state = "queued";
      job.queue = {
        position: Math.max(1, event.queuePosition ?? 1),
        queuedBehind: { ...event.queuedBehind }
      };
      job.label = `Queued behind ${event.queuedBehind.label}`;
      job.updatedAt = new Date().toISOString();
      this.publish(job);
      return;
    }
    if (event.state === "running" && job.state === "queued") {
      job.state = "running";
      job.queue = undefined;
      job.label = this.jobResumeLabels.get(job.id) ?? event.label;
      this.jobResumeLabels.delete(job.id);
      job.updatedAt = new Date().toISOString();
      this.publish(job);
    }
  }

  private publish(job: JobRecord): void {
    const event: RuntimeJobEvent = { job: { ...job } };
    this.emit("event", event);
  }
}
