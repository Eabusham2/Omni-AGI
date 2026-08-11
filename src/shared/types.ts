export const BRAIN_SCHEMA_VERSION = 1;

export const APPEARANCE_SCHEMA_VERSION = 1;

/** User-owned presentation choices. These never enter a brain's neural state. */
export type AppearanceMode = "system" | "light" | "dark";
export type ResolvedColorScheme = "light" | "dark";
export type AppearancePalette = "violet" | "graphite" | "spectrum" | "aqua";
export type AppearanceLayout = "standard" | "classic" | "expressive" | "glass";

export interface AppearancePreferences {
  schemaVersion: typeof APPEARANCE_SCHEMA_VERSION;
  mode: AppearanceMode;
  palette: AppearancePalette;
  layout: AppearanceLayout;
}

/** Narrow, validated request used only to synchronize native window chrome. */
export interface NativeAppearanceRequest {
  schemaVersion: typeof APPEARANCE_SCHEMA_VERSION;
  mode: AppearanceMode;
  resolvedColorScheme: ResolvedColorScheme;
  layout: AppearanceLayout;
}

export interface NativeAppearanceState extends NativeAppearanceRequest {
  backgroundColor: string;
  symbolColor: string;
}

export type ArchitecturePreset =
  | "whole-brain"
  | "ternary"
  | "neuromorphic"
  | "liquid"
  | "symbolic"
  | "custom";

export type InferenceRuntime = "adaptive-core";
export type MemoryRecipe = "human-consolidation" | "total-recall" | "synapses-only";
export type TraceDetail = "summary" | "standard" | "research";

/**
 * Stable v1 user choices.
 *
 * Neural shape, mandatory ternary/spiking/liquid/VSA capabilities, plasticity
 * constants, organic drive state, and structural growth belong to the engine's
 * hardware-derived architecture manifest. They are deliberately not
 * behavioral controls in this public or persisted configuration.
 */
export interface BrainConfig {
  name: string;
  preset: ArchitecturePreset;
  runtime: InferenceRuntime;
  description: string;

  onlineLearning: boolean;
  extendedWorkingMemory: boolean;
  recursiveImprovement: boolean;
  idleCognition: boolean;

  /** Hardware-resolved transient neural-workspace capacity. */
  workingMemorySlots: number;
  /** Resolved slow-training rate; it is not a personality or motivation dial. */
  learningRate: number;
  traceDetail: TraceDetail;

  retainSourceText: boolean;
  memoryRecipe: MemoryRecipe;
}

export interface BrainLineage {
  parentId?: string;
  rootId: string;
  generation: number;
}

export interface ConceptNode {
  id: string;
  label: string;
  activation: number;
  importance: number;
  uncertainty: number;
  exposures: number;
  createdAt: string;
  lastActivatedAt: string;
  aliases: string[];
}

export interface Synapse {
  id: string;
  sourceId: string;
  targetId: string;
  effectiveWeight: -1 | 0 | 1;
  latentWeight: number;
  stability: number;
  plasticity: number;
  uses: number;
  lastUpdatedAt: string;
}

export type IdeaKind = "knowledge" | "preference" | "question" | "experience";
export type IdeaSource = "conversation" | "document" | "self" | "import";

export interface Idea {
  id: string;
  statement?: string;
  fingerprint: string;
  conceptIds: string[];
  kind: IdeaKind;
  source: IdeaSource;
  confidence: number;
  importance: number;
  rehearsals: number;
  createdAt: string;
  lastRecalledAt?: string;
  sourceLabel?: string;
}

export interface WorkingMemoryItem {
  conceptId: string;
  activation: number;
  enteredAt: string;
  expiresAt: string;
}

export interface LiquidState {
  values: number[];
  timeConstants: number[];
  lastUpdatedAt: string;
}

export type MessageRole = "human" | "brain";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: string;
  traceId?: string;
  runtime?: InferenceRuntime;
  status?: "complete" | "error";
}

export interface TraceStep {
  stage: string;
  detail: string;
  value?: string;
}

export interface ThoughtTrace {
  id: string;
  createdAt: string;
  input: string;
  seed: number;
  runtime: InferenceRuntime;
  activatedConcepts: Array<{
    id: string;
    label: string;
    activation: number;
  }>;
  recalledIdeas: Array<{
    id: string;
    preview: string;
    score: number;
  }>;
  driveScores: {
    novelty: number;
    coherence: number;
    curiosity: number;
  };
  branches: number;
  selectedBranch: number;
  steps: TraceStep[];
  note: string;
}

export interface TrainingSource {
  id: string;
  name: string;
  path?: string;
  kind:
    | "pdf"
    | "text"
    | "markdown"
    | "code"
    | "json"
    | "csv"
    | "parquet"
    | "arrow"
    | "archive"
    | "sqlite"
    | "dataset"
    | "image"
    | "audio"
    | "video"
    | "unknown";
  bytes: number;
  learnedIdeas: number;
  learnedConcepts: number;
  learnedSynapses: number;
  importedAt: string;
  rawTextRetained: boolean;
  rawText?: string;
  contentHash?: string;
  blobHash?: string;
  policy?: DataIngestionPolicy;
  provenanceUrl?: string;
  license?: string;
  licenseUrl?: string;
}

export interface BrainMetrics {
  concepts: number;
  synapses: number;
  activeSynapses: number;
  ideas: number;
  messages: number;
  trainingSources: number;
  averageStability: number;
  plasticityEvents: number;
  inferenceCount: number;
  estimatedBytes: number;
}

export interface BrainDocument {
  schemaVersion: number;
  /**
   * Stable v1 storage discriminator. Beta documents intentionally do not have
   * this marker and are never normalized or loaded as stable brains.
   */
  releaseFormat?: "stable-1.0";
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  lineage: BrainLineage;
  config: BrainConfig;
  concepts: Record<string, ConceptNode>;
  synapses: Record<string, Synapse>;
  ideas: Idea[];
  workingMemory: WorkingMemoryItem[];
  liquidState: LiquidState;
  messages: ChatMessage[];
  traces: ThoughtTrace[];
  trainingSources: TrainingSource[];
  counters: {
    plasticityEvents: number;
    inferenceCount: number;
    consolidationCycles: number;
  };
  toolPermissions?: ToolPermissionRecord[];
  journal?: JournalEntry[];
  originChecksum?: string;
}

export interface BrainSummary {
  id: string;
  name: string;
  preset: ArchitecturePreset;
  runtime: InferenceRuntime;
  updatedAt: string;
  concepts: number;
  synapses: number;
  generation: number;
}

export type SubstrateEntity =
  | "overview"
  | "neurons"
  | "assemblies"
  | "synapses";

export interface SubstrateQuery {
  /**
   * Overview and low zoom levels return aggregate clusters. The other entity
   * kinds expose the same unbounded substrate through cursor-based pages.
   */
  entity?: SubstrateEntity;
  cursor?: string;
  pageSize?: number;
  region?: string;
  search?: string;
  zoom?: number;
}

export interface SubstrateNeuron {
  id: string;
  label: string;
  region: string;
  activation: number;
  importance: number;
  uncertainty: number;
  exposures: number;
  createdAt?: string;
  lastActivatedAt?: string;
  aliases: string[];
}

export interface SubstrateAssembly {
  id: string;
  label: string;
  region: "assembly";
  neuronIds: string[];
  childAssemblyIds: string[];
  kind: string;
  source: string;
  confidence: number;
  importance: number;
  rehearsals: number;
  createdAt?: string;
  lastRecalledAt?: string;
  sourceLabel?: string;
  retainsSourceText: boolean;
}

export interface SubstrateSynapse {
  id: string;
  sourceId: string;
  targetId: string;
  kind: string;
  effectiveWeight: -1 | 0 | 1;
  latentWeight: number;
  eligibility: number;
  plasticity: number;
  stability: number;
  uses: number;
  lastUpdatedAt?: string;
}

export interface SubstrateCluster {
  id: string;
  label: string;
  kind: "region" | "activation-band" | "pathway";
  region?: string;
  sourceRegion?: string;
  targetRegion?: string;
  count: number;
  activeCount: number;
  meanActivation: number;
  maxActivation: number;
  effectiveWeights: {
    negative: number;
    zero: number;
    positive: number;
  };
}

export interface SubstratePage {
  brainId: string;
  queriedAt: string;
  entity: SubstrateEntity;
  zoom: number;
  totals: {
    neurons: number;
    assemblies: number;
    synapses: number;
  };
  matched: number;
  hasMore: boolean;
  nextCursor?: string;
  clusters: SubstrateCluster[];
  neurons: SubstrateNeuron[];
  assemblies: SubstrateAssembly[];
  synapses: SubstrateSynapse[];
}

export interface WorkspaceSnapshot {
  brainId: string;
  queriedAt: string;
  contextWindow: {
    capacityTokens: number;
    /** Baseline output budget; the live action state may adjust it per turn. */
    generationBudgetTokens?: number;
    capacityPolicy?: "hardware-derived-resource-guarded";
    expandable?: boolean;
    tokenCount: number;
    tokenHash: string;
    recentTokenCount?: number;
    recentTokenHash?: string;
    evictions?: number;
    sensorySlots: number;
    extended: boolean;
    updatedAt: string;
  };
  latentWorkspace: {
    capacity: number;
    occupancy: number;
    items: Array<{
      id?: string;
      kind?: string;
      salience: number;
      rehearsals: number;
      enteredAt?: string;
      lastActiveAt?: string;
    }>;
    evictions: number;
    rehearsals: number;
  };
  liquidState: {
    dimensions: number;
    mean: number;
    norm: number;
  };
  hiddenBehavioralPrompt: boolean;
  rawLongTermTextInjected: boolean;
}

export interface RecallResult {
  idea: Idea;
  score: number;
  overlap: number;
  vsaSimilarity: number;
}

export interface ChatResult {
  brain: BrainDocument;
  humanMessage: ChatMessage;
  brainMessage: ChatMessage;
  trace: ThoughtTrace;
  proposedActions?: StructuredAction[];
  actionEvents?: ActionEvent[];
}

export interface IngestResult {
  brain: BrainDocument;
  source: TrainingSource;
  warnings: string[];
  manifestId?: string;
  coverage?: TrainingCoverage;
}

export interface RuntimeHealth {
  runtime: InferenceRuntime;
  ready: boolean;
  label: string;
  detail: string;
}

export interface CreateBrainRequest {
  config: BrainConfig;
  origin?: "blank" | "starter";
  starterUrl?: string;
  hardwareTier?: HardwareTier;
  modalities?: ModalityKind[];
  initialToolPermissions?: Array<{
    toolId: string;
    level: ToolPermissionLevel;
  }>;
}

export interface ImportUrlRequest {
  url: string;
  expectedSha256?: string;
}

export interface FeedbackRequest {
  brainId: string;
  messageId: string;
  direction: "up" | "down";
}

export interface BrainSnapshotSummary {
  id: string;
  brainId: string;
  label: string;
  createdAt: string;
  checksum: string;
  metrics: BrainMetrics;
  engineChecksum?: string;
}

export type BrainExportMode = "current" | "origin" | "private-archive" | "referenced";

export type DataIngestionPolicy = "encode" | "consolidate" | "pretrain" | "archive";

export type DatasetFormat =
  | "text"
  | "pdf"
  | "epub"
  | "office"
  | "csv"
  | "tsv"
  | "json"
  | "jsonl"
  | "parquet"
  | "arrow"
  | "sqlite"
  | "archive"
  | "webdataset"
  | "huggingface"
  | "image"
  | "audio"
  | "video"
  | "unknown";

export interface DatasetManifestEntry {
  index: number;
  path: string;
  relativePath: string;
  format: DatasetFormat;
  bytes: number;
}

export interface DatasetManifest {
  schemaVersion: 1;
  id: string;
  brainId: string;
  createdAt: string;
  updatedAt: string;
  roots: string[];
  entryFile: string;
  discoveredFiles: number;
  discoveredBytes: number;
  manifestHash: string;
}

export interface DatasetCursor {
  schemaVersion: 1;
  manifestId: string;
  currentEpoch?: number;
  requestedEpochs?: number;
  nextEntry: number;
  nextRecord: number;
  processedFiles: number;
  processedRecords: number;
  processedBytes: number;
  state: "ready" | "running" | "paused" | "complete" | "failed";
  updatedAt: string;
}

export interface TrainingCoverageError {
  source: string;
  message: string;
}

export interface TrainingCoverage {
  schemaVersion: 1;
  manifestId: string;
  requestedEpochs?: number;
  completedEpochs?: number;
  discoveredFiles: number;
  processedFiles: number;
  rejectedFiles: number;
  discoveredRecords: number;
  processedRecords: number;
  rejectedRecords: number;
  discoveredBytes: number;
  processedBytes: number;
  shards: number;
  modalityCounts: Partial<Record<DatasetFormat, number>>;
  errors: TrainingCoverageError[];
  errorLog?: string;
  complete: boolean;
  updatedAt: string;
}

export interface IngestFilesRequest {
  brainId: string;
  policy?: DataIngestionPolicy;
  selection?: import("./uploadSupport").ExperienceUploadKind;
}

export interface DatasetPreviewRequest
  extends Omit<IngestFilesRequest, "selection"> {
  selection?:
    | "folder"
    | import("./uploadSupport").ExperienceUploadKind;
}

export interface DatasetStartRequest extends IngestFilesRequest {
  manifestId: string;
  epochs?: number;
  resume?: boolean;
}

export interface BuildResourceSelection {
  id: string;
  kind: "files" | "folder";
  label: string;
  itemCount: number;
}

export interface BuildResourceStartRequest extends IngestFilesRequest {
  selectionId: string;
  epochs?: number;
}

export interface IngestWebRequest {
  brainId: string;
  url: string;
  policy?: DataIngestionPolicy;
  quarantine?: boolean;
}

export interface WebCrawlRequest extends IngestWebRequest {
  crawlId?: string;
  maxPages?: number;
  maxDepth?: number;
  sameOrigin?: boolean;
  followExternalLinks?: boolean;
  respectRobots?: boolean;
  concurrency?: number;
  resume?: boolean;
}

export interface WebCrawlResult {
  crawlId: string;
  startUrl: string;
  visited: number;
  skipped: number;
  /** Recent diagnostic results; the complete receipt ledger remains in resultLog. */
  results: IngestResult[];
  resultCount: number;
  resultsTruncated: boolean;
  resultLog: string;
  warnings: string[];
  warningCount: number;
  warningsTruncated: boolean;
  frontierRemaining: number;
  stopped: boolean;
  coverage: TrainingCoverage;
}

export type ToolPermissionLevel = "off" | "ask" | "auto" | "full";

export interface ToolPermissionRecord {
  toolId: string;
  label: string;
  level: ToolPermissionLevel;
  updatedAt: string;
}

export interface JournalEntry {
  id: string;
  createdAt: string;
  kind:
    | "learning"
    | "consolidation"
    | "tool"
    | "fork"
    | "reflection"
    | "system";
  summary: string;
  detail?: string;
}

export type RuntimeJobKind =
  | "training"
  | "consolidation"
  | "ingestion"
  | "crawl"
  | "image"
  | "audio"
  | "video"
  | "vision"
  | "agent";
export type RuntimeJobState = "queued" | "running" | "complete" | "failed" | "cancelled";

export interface RuntimeJob {
  id: string;
  brainId: string;
  kind: RuntimeJobKind;
  state: RuntimeJobState;
  progress: number;
  label: string;
  createdAt: string;
  updatedAt: string;
  error?: string;
  output?: unknown;
  /** A bounded, non-authoritative preview emitted while media is decoding. */
  preview?: ModalityPreview;
}

export interface RuntimeJobEvent {
  job: RuntimeJob;
}

export interface StartTrainingRequest {
  brainId: string;
  epochs?: number;
  learningRate?: number;
  sourceIds?: string[];
}

export type ModalityKind = "image" | "audio" | "video" | "vision";

export interface ModalityGenerateRequest {
  brainId: string;
  modality: ModalityKind;
  prompt?: string;
  conceptIds?: string[];
  inputPath?: string;
  settings?: Record<string, number | string | boolean>;
  /** Worker-issued correlation for media already decoding during chat. */
  neuralActionId?: string;
  seed?: number;
}

export interface TraceQuery {
  limit?: number;
  before?: string;
}

export interface CatalogEntry {
  id: string;
  name: string;
  description: string;
  sourceUrl: string;
  license: string;
  sha256?: string;
  kind: "brain" | "recipe" | "dataset" | "modality-pack";
}

export interface BuildRecipe {
  schemaVersion: 1;
  id: string;
  name: string;
  description: string;
  source: string;
  sha256: string;
  license: string;
  provenanceUrl?: string;
  origin: "blank" | "starter";
  starterUrl?: string;
  hardwareTier: HardwareTier;
  modalities: ModalityKind[];
  toolPermissions: Array<{
    toolId: string;
    level: ToolPermissionLevel;
  }>;
  config: BrainConfig;
}

export interface ModalityPackManifest {
  format: "omni-modality-pack";
  formatVersion: 1;
  architecture: "OmniCortex";
  architectureSchemaVersion: 1;
  pack: {
    id: string;
    name: string;
    version: string;
    modalities: ModalityKind[];
  };
  compatibility: {
    dModel: number;
    modalityChannels: number;
    imageSize: number;
    audioSamples: number;
    videoFrames: number;
  };
  licenseLedger: {
    license: string;
    provenanceUrl?: string;
    sourceUrl?: string;
  };
  files: {
    "model-card.md": { sha256: string; bytes: number };
    "tensors/modality.safetensors": { sha256: string; bytes: number };
  };
}

export interface InstalledModalityPack {
  id: string;
  name: string;
  version: string;
  modalities: ModalityKind[];
  license: string;
  provenanceUrl?: string;
  sourceLabel: string;
  sha256: string;
  installedAt: string;
}

export interface InstallModalityPackUrlRequest extends ImportUrlRequest {
  brainId: string;
}

export interface AgentMergeSubstratePreview {
  schemaVersion: 1;
  engineSchemaVersion: 1;
  digest: string;
  sourceStateSha256: string;
  targetStateSha256: string;
  sourceParameterSha256: string;
  targetParameterSha256: string;
  sourceConfigSha256: string;
  targetConfigSha256: string;
  sourceCounts: {
    neurons: number;
    assemblies: number;
    synapses: number;
    replayExamples: number;
  };
  targetCounts: {
    neurons: number;
    assemblies: number;
    synapses: number;
    replayExamples: number;
  };
  additions: {
    neurons: number;
    assemblies: number;
    synapses: number;
    replayExamples: number;
  };
  duplicates: {
    neurons: number;
    assemblies: number;
    synapses: number;
    replayExamples: number;
  };
  divergent: {
    neurons: number;
    assemblies: number;
    synapses: number;
  };
  weightsAveraged: false;
}

export interface AgentMergePreview {
  sourceBrainId: string;
  targetBrainId: string;
  reviewToken: string;
  /** Authoritative worker state bound into reviewToken and rechecked at merge. */
  substrate: AgentMergeSubstratePreview;
  newConcepts: number;
  newIdeas: number;
  newSynapses: number;
  newEvidence: number;
  duplicateEvidence: number;
  newFiles: number;
  duplicateFiles: number;
  skippedFiles: number;
  fileBytes: number;
  files: AgentMergeFilePreview[];
  conflicts: string[];
  note: string;
}

export interface AgentMergeFilePreview {
  kind: "artifact" | "evidence";
  sourcePath: string;
  destinationPath: string;
  sha256: string;
  bytes: number;
  disposition: "copy" | "duplicate";
}

export interface EngineHealth {
  ready: boolean;
  worker: "python" | "unavailable";
  protocolVersion: number;
  detail: string;
  pid?: number;
}

export type HardwareTier = "micro" | "personal" | "gpu" | "workstation";

export interface HardwareProfile {
  platform: string;
  architecture: string;
  logicalCpus: number;
  totalMemoryBytes: number;
  availableMemoryBytes: number;
  gpu: {
    available: boolean;
    vendor?: string;
    device?: string;
    driver?: string;
    details?: Record<string, unknown>;
  };
  recommendedTier: HardwareTier;
  recommendation: string;
}

export interface ToolInvocation {
  brainId: string;
  toolId: string;
  action: string;
  arguments: Record<string, unknown>;
  approvalToken?: string;
}

export interface ToolExecutionResult {
  id: string;
  toolId: string;
  action: string;
  state: "approval-required" | "complete" | "failed";
  startedAt: string;
  finishedAt?: string;
  output?: unknown;
  error?: string;
  approvalToken?: string;
}

export type ActionKind =
  | "talk"
  | "tool"
  | "imagine"
  | "agent"
  | "ponder"
  | "learn"
  | "evolve"
  | "stop";

export type ActionSource = "human" | "brain" | "organic";

/**
 * A model-facing action channel. Tool capability schemas enter the neural
 * runtime as structured vectors; this value is never a behavioral prompt.
 */
export interface StructuredAction {
  kind: ActionKind;
  source: ActionSource;
  toolId?: string;
  action?: string;
  arguments: Record<string, unknown>;
  confidence?: number;
}

export interface ActionEvent {
  id: string;
  brainId: string;
  /** Opaque worker correlation; never part of the model-facing arguments. */
  neuralActionId?: string;
  action: StructuredAction;
  state:
    | "proposed"
    | "running"
    | "approval-required"
    | "complete"
    | "failed"
    | "stopped";
  createdAt: string;
  updatedAt: string;
  /** Live local-runtime progress for long imagination/tool actions. */
  progress?: number;
  statusLabel?: string;
  runtimeJobId?: string;
  execution?: ToolExecutionResult;
  evolutionRunId?: string;
  /** Latest safe progressive preview for an imagination action. */
  preview?: ModalityPreview;
  error?: string;
}

/**
 * Progressive media is display-only until the corresponding action completes.
 * The main process validates and bounds every field before it crosses preload.
 */
export interface ModalityPreview {
  revision: number;
  progress?: number;
  statusLabel?: string;
  mimeType?: string;
  dataUrl?: string;
  path?: string;
  artifactPath?: string;
}

interface ChatStreamEventBase {
  id: string;
  brainId: string;
  turnId: string;
  sequence: number;
  createdAt: string;
}

export interface ChatTokenStreamEvent extends ChatStreamEventBase {
  type: "chat-token";
  delta: string;
}

export interface ChatActionStreamEvent extends ChatStreamEventBase {
  type: "chat-action";
  actionEvent: ActionEvent;
}

export interface ChatModalityPreviewStreamEvent extends ChatStreamEventBase {
  type: "modality-preview";
  actionId: string;
  preview: ModalityPreview;
}

export interface ChatStateStreamEvent extends ChatStreamEventBase {
  type: "chat-state";
  state: "started" | "complete" | "cancelled" | "failed";
  error?: string;
}

/**
 * One typed, ordered renderer stream. No response prose, tags, or hidden
 * instructions are parsed to manufacture an action.
 */
export type ChatStreamEvent =
  | ChatTokenStreamEvent
  | ChatActionStreamEvent
  | ChatModalityPreviewStreamEvent
  | ChatStateStreamEvent;

export interface IdleCognitionTrace {
  id: string;
  createdAt: string;
  mode: "ponder" | "imagine" | "rehearse";
  seed: number;
  promptTokenCount: 0;
  hiddenBehavioralPrompt: false;
  activeAssemblyIds: string[];
  organicState: Record<string, number>;
  liquidControls: Record<string, number>;
  stdpUpdate: number;
  spikeRate: number;
  rehearsal: {
    loss: number;
    reconstructionLoss: number;
    temporalLoss: number;
    stabilityLoss: number;
  };
  parameterChecksumBefore: string;
  parameterChecksumAfter: string;
  parameterDeltaNorm: number;
  actionPolicyScores: Record<ActionKind, number>;
  proposedActionKinds: ActionKind[];
  note: string;
}

export interface IdleCycleResult {
  brainId: string;
  ran: boolean;
  reason?: "idle-cognition-disabled" | "cooldown" | "no-learned-assemblies";
  retryAfterSeconds?: number;
  trace?: IdleCognitionTrace;
  actions: StructuredAction[];
  actionEvents?: ActionEvent[];
  metrics?: Record<string, unknown>;
  runtimeCard?: Record<string, unknown>;
}

export type EvolutionRunState =
  | "experimenting"
  | "awaiting-review"
  | "promoted"
  | "rejected"
  | "stopped"
  | "failed"
  | "rolled-back";

export interface EvolutionEvaluation {
  id: string;
  createdAt: string;
  passed: boolean;
  diffSha256?: string;
  checks: Array<{
    name: string;
    passed: boolean;
    exitCode?: number;
  }>;
}

export interface PromotionRecord {
  id: string;
  candidateId: string;
  createdAt: string;
  commit: string;
  parentCommit: string;
  diffSha256: string;
  rollbackCommit?: string;
  rolledBackAt?: string;
}

/**
 * A compare-and-write source mutation for an isolated evolution worktree.
 * Existing files require their exact SHA-256; null means the path must not
 * exist. This channel never interprets generated response prose.
 */
export interface EvolutionSourceEdit {
  path: string;
  content: string;
  expectedSha256: string | null;
}

export interface EvolutionSourceEditLineage {
  path: string;
  expectedSha256: string | null;
  resultSha256: string;
  bytes: number;
}

export interface EvolutionCandidate {
  id: string;
  runId: string;
  brainId: string;
  parentCandidateId?: string;
  generation: number;
  objective: string;
  state: EvolutionRunState;
  branch?: string;
  worktree?: string;
  createdAt: string;
  updatedAt: string;
  evaluations: EvolutionEvaluation[];
  promotion?: PromotionRecord;
  sourceEditLineage?: EvolutionSourceEditLineage[];
  authoredChangedPaths?: string[];
  authoredDiffSha256?: string;
  authoredBytes?: number;
  error?: string;
}

export interface EvolutionRun {
  id: string;
  brainId: string;
  objective: string;
  state: EvolutionRunState;
  recursive: boolean;
  generation: number;
  candidateIds: string[];
  createdAt: string;
  updatedAt: string;
  error?: string;
}

export interface EvolutionStartRequest {
  brainId: string;
  objective: string;
  recursive?: boolean;
  parentCandidateId?: string;
  /**
   * Edit-free requests default to a substrate overlay. Source candidates are
   * accepted only with exact typed sourceEdits. Neural, data, substrate, and
   * compatible expert-growth architecture candidates use isolated worker-owned
   * safe-tensor overlays. Incompatible tensor-shape migration remains rejected.
   */
  candidateKind?: "source" | "neural" | "data" | "substrate" | "architecture";
  texts?: string[];
  sourceIds?: string[];
  epochs?: number;
  learningRate?: number;
  latentReplay?: boolean;
  objectives?: string[];
  provenance?: Record<string, unknown>;
  /**
   * Typed source candidate authoring. Edits are applied only inside the new
   * isolated worktree, before it becomes externally visible.
   */
  sourceEdits?: EvolutionSourceEdit[];
  architectureChange?: {
    /** Stable v1's only load-compatible architecture mutation. */
    mutation: "grow-experts";
    addExperts?: number;
  };
}

export interface EvolutionApprovalRequest {
  brainId: string;
  candidateId: string;
  tests?: Array<"typecheck" | "unit" | "build">;
  timeoutMs?: number;
}

export interface EvolutionRollbackRequest {
  brainId: string;
  candidateId: string;
}

export interface OmniApi {
  window: {
    minimize(): Promise<void>;
    maximize(): Promise<void>;
    close(): Promise<void>;
    isMaximized(): Promise<boolean>;
    openExternal(url: string): Promise<void>;
    revealDataFolder(): Promise<void>;
    platform(): Promise<string>;
    setAppearance(request: NativeAppearanceRequest): Promise<NativeAppearanceState>;
  };
  brain: {
    list(): Promise<BrainSummary[]>;
    get(id: string): Promise<BrainDocument>;
    create(request: CreateBrainRequest): Promise<BrainDocument>;
    update(id: string, config: BrainConfig): Promise<BrainDocument>;
    duplicate(id: string, name?: string): Promise<BrainDocument>;
    fork(id: string, name?: string): Promise<BrainDocument>;
    remove(id: string): Promise<boolean>;
    snapshot(id: string, label?: string): Promise<BrainSnapshotSummary>;
    listSnapshots(id: string): Promise<BrainSnapshotSummary[]>;
    restoreSnapshot(id: string, snapshotId: string): Promise<BrainDocument>;
    export(id: string, mode?: BrainExportMode): Promise<string | null>;
    importFile(): Promise<BrainDocument | null>;
    onImported(listener: (brain: BrainDocument) => void): () => void;
    health(id?: string): Promise<EngineHealth>;
    querySubstrate(id: string, query?: SubstrateQuery): Promise<SubstratePage>;
    workspace(id: string): Promise<WorkspaceSnapshot>;
  };
  chat: {
    send(id: string, input: string, turnId?: string): Promise<ChatResult>;
    cancel(id: string, turnId?: string): Promise<number>;
    list(id: string): Promise<ChatMessage[]>;
    feedback(request: FeedbackRequest): Promise<BrainDocument>;
    onAction(listener: (event: ActionEvent) => void): () => void;
    onStream(listener: (event: ChatStreamEvent) => void): () => void;
  };
  train: {
    start(request: StartTrainingRequest): Promise<RuntimeJob>;
    consolidate(id: string): Promise<BrainDocument>;
    cancel(jobId: string): Promise<RuntimeJob>;
    list(id?: string): Promise<RuntimeJob[]>;
    onEvent(listener: (event: RuntimeJobEvent) => void): () => void;
  };
  data: {
    selectBuildResources(
      kind: BuildResourceSelection["kind"],
      selection?: import("./uploadSupport").ExperienceUploadKind
    ): Promise<BuildResourceSelection | null>;
    discardBuildResource(selectionId: string): Promise<boolean>;
    startBuildResource(request: BuildResourceStartRequest): Promise<RuntimeJob>;
    preview(request: DatasetPreviewRequest): Promise<DatasetManifest | null>;
    start(request: DatasetStartRequest): Promise<RuntimeJob>;
    pause(jobId: string): Promise<RuntimeJob>;
    resume(request: DatasetStartRequest): Promise<RuntimeJob>;
    coverage(brainId: string, manifestId: string): Promise<TrainingCoverage>;
    ingestFiles(request: IngestFilesRequest): Promise<IngestResult[]>;
    ingestFolder(request: IngestFilesRequest): Promise<IngestResult[]>;
    ingestDropped(request: IngestFilesRequest, files: unknown[]): Promise<IngestResult[]>;
    ingestWeb(request: IngestWebRequest): Promise<IngestResult>;
    crawlWeb(request: WebCrawlRequest): Promise<RuntimeJob>;
    cancel(jobId: string): Promise<RuntimeJob>;
  };
  modality: {
    generate(request: ModalityGenerateRequest): Promise<RuntimeJob>;
    selectInput(
      request: Omit<ModalityGenerateRequest, "inputPath">
    ): Promise<RuntimeJob | null>;
    cancel(jobId: string): Promise<RuntimeJob>;
  };
  trace: {
    list(brainId: string, query?: TraceQuery): Promise<ThoughtTrace[]>;
  };
  tool: {
    listPermissions(brainId: string): Promise<ToolPermissionRecord[]>;
    setPermission(
      brainId: string,
      toolId: string,
      level: ToolPermissionLevel
    ): Promise<ToolPermissionRecord[]>;
    execute(request: ToolInvocation): Promise<ToolExecutionResult>;
    cancel(brainId: string): Promise<number>;
  };
  agent: {
    fork(brainId: string, name?: string): Promise<BrainDocument>;
    previewMerge(sourceBrainId: string, targetBrainId: string): Promise<AgentMergePreview>;
    merge(
      sourceBrainId: string,
      targetBrainId: string,
      reviewToken: string
    ): Promise<BrainDocument>;
  };
  evolution: {
    start(request: EvolutionStartRequest): Promise<EvolutionRun>;
    stop(brainId: string, runId: string): Promise<EvolutionRun>;
    listCandidates(brainId: string, runId?: string): Promise<EvolutionCandidate[]>;
    approve(request: EvolutionApprovalRequest): Promise<EvolutionCandidate>;
    rollback(request: EvolutionRollbackRequest): Promise<EvolutionCandidate>;
  };
  catalog: {
    list(): Promise<CatalogEntry[]>;
    importUrl(request: ImportUrlRequest): Promise<BrainDocument>;
    loadRecipeEntry(id: string): Promise<BuildRecipe>;
    loadRecipeUrl(request: ImportUrlRequest): Promise<BuildRecipe>;
    loadRecipeFile(): Promise<BuildRecipe | null>;
    installModalityPackUrl(
      request: InstallModalityPackUrlRequest
    ): Promise<InstalledModalityPack>;
    installModalityPackFile(brainId: string): Promise<InstalledModalityPack | null>;
    listModalityPacks(brainId: string): Promise<InstalledModalityPack[]>;
    hardwareProfile(): Promise<HardwareProfile>;
  };
}

export const DEFAULT_CONFIG: BrainConfig = {
  name: "New mind",
  preset: "whole-brain",
  runtime: "adaptive-core",
  description: "A persistent adaptive OmniCortex identity with a unified neural substrate.",

  onlineLearning: true,
  extendedWorkingMemory: false,
  recursiveImprovement: true,
  idleCognition: true,

  workingMemorySlots: 256,
  learningRate: 0.14,
  traceDetail: "standard",

  retainSourceText: false,
  memoryRecipe: "human-consolidation"
};

const PRESET_DESCRIPTIONS: Record<ArchitecturePreset, string> = {
  "whole-brain": DEFAULT_CONFIG.description,
  ternary:
    "A unified OmniCortex identity imported from the historical Ternary Cortex recipe.",
  neuromorphic:
    "A unified OmniCortex identity imported from the historical Neuromorphic Lab recipe.",
  liquid:
    "A unified OmniCortex identity imported from the historical Liquid Cortex recipe.",
  symbolic:
    "A unified OmniCortex identity imported from the historical VSA Idea Brain recipe.",
  custom:
    "A unified OmniCortex identity imported from a declarative architecture recipe."
};

export function createPresetConfig(
  preset: ArchitecturePreset,
  name = DEFAULT_CONFIG.name
): BrainConfig {
  return {
    ...DEFAULT_CONFIG,
    description: PRESET_DESCRIPTIONS[preset],
    name,
    preset
  };
}
