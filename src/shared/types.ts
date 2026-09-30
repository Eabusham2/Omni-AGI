import type { TrainingTelemetry } from "./trainingTelemetry";
import type { NativeArchitectureDescriptor } from "../main/nativeCoreInventory";
import type { CortexPage, CortexQuery, CortexActivityQuery, CortexActivity } from "./cortexInspection";
export type { TrainingTelemetry } from "./trainingTelemetry";

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
export type MemoryRecipe = "adaptive-retention" | "total-recall" | "synapses-only";
export type TraceDetail = "summary" | "standard" | "research";
export type WorkingMemoryMode = "auto" | "extended" | "manual";
export type SystemRamMode = "auto" | "manual";
export type StoragePoolMode = "auto" | "manual";

/**
 * Stable v1 user choices.
 *
 * Neural shape, mandatory ternary/spiking/liquid/VSA capabilities, plasticity
 * constants, organic drive state, and structural growth belong to the engine's
 * hardware-derived architecture manifest. They are deliberately not
 * behavioral controls in this public or persisted configuration.
 */
export interface BrainConfig {
  /** Main-selected exact native shape; never accepted as a renderer shape override. */
  nativeArchitecture?: NativeArchitectureDescriptor;
  name: string;
  preset: ArchitecturePreset;
  runtime: InferenceRuntime;
  description: string;

  onlineLearning: boolean;
  extendedWorkingMemory: boolean;
  recursiveImprovement: boolean;
  /**
   * Per-identity Active Mode for optional prompt-free background cognition.
   * It is an operational lease preference, not a Ponder capability or prompt.
   */
  idleCognition: boolean;

  /** Hardware-resolved transient neural-workspace capacity. */
  workingMemorySlots: number;
  /**
   * Hardware/model-resolved active token window. This is an operational
   * architecture value, not a behavior or personality control. Token indexes
   * stay resident; cold attention and recurrent state may page.
   */
  contextWindowTokens: number;
  /** Device-planned recurrent/paged capacity policy, never a personality dial. */
  workingMemoryMode: WorkingMemoryMode;
  /** Estimated on-demand disk spill reserved by the last successful preflight. */
  memoryOffloadBytes: number;
  /** Last admitted cold attention backing budget; not an extra RAM pool. */
  contextOffloadBudgetBytes?: number;
  /** Maximum active working patterns kept resident by the last preflight. */
  memoryResidentItems: number;
  /** Measured, bounded estimate shown whenever storage offload is active. */
  memoryOffloadSlowdownPercent: number;
  /** One process-wide envelope; 100% still excludes the mandatory OS reserve. */
  systemRamMode: SystemRamMode;
  /** Manual percentage of the safe pool, or 0 while Auto is selected. */
  systemRamSharePercent: number;
  /** Device-wide pool sizing policy. Every brain shares the largest reservation. */
  storagePoolMode: StoragePoolMode;
  /** This brain's required shared-pool size; the app never sums it across brains. */
  storagePoolBytes: number;
  /** Last bounded storage probe, used only to pace emergency checkpoints. */
  storageBytesPerSecond: number;
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

/** Observable completion only; no-reply does not assert intent or cancellation. */
export type ChatGenerationEnd = "steered" | "native-stop" | "no-reply";
export type ChatNoReplyReason = "no-generated-tokens" | "no-decoded-text" | "whitespace-only" | "no-printable-text";

export type ChatDeliveryReceiptState =
  | "pending"
  | "queued"
  | "steered"
  | "stopped"
  | "cancelled"
  | "failed";

/**
 * Durable renderer history for user-authored text before a neural turn commits
 * or after it ends without a commit. These rows are presentation-only: they
 * may be shown by desktop/mobile history, but must not be supplied to model
 * context, history tools, evolution evidence, or automatic queue replay.
 */
export interface ChatDeliveryReceipt {
  schemaVersion: 1;
  presentationOnly: true;
  turnId: string;
  state: ChatDeliveryReceiptState;
  updatedAt: string;
}

export interface ChatDeliveryReceiptRequest {
  schemaVersion: 1;
  turnId: string;
  content: string;
  createdAt: string;
  state: ChatDeliveryReceiptState;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: string;
  /** Correlates an atomically committed worker turn with its UI receipt. */
  turnId?: string;
  traceId?: string;
  generationEnd?: ChatGenerationEnd;
  runtime?: InferenceRuntime;
  status?: "complete" | "error";
  attentionEpoch?: number;
  deliveryReceipt?: ChatDeliveryReceipt;
}

export interface TraceStep {
  stage: string;
  detail: string;
  value?: string;
}

export interface ThoughtTrace {
  id: string;
  parameterDiagnostics?: {
    scope: string;
    coreDeltaNorm?: number;
    substrateDeltaNorm: null;
    substrateDeltaMeasured: false;
    checksumScope: string;
  };
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
  attentionEpoch?: number;
  generation?: {
    disposition?: ChatGenerationEnd;
    decoderStopReason?: string;
    generatedTokenCount?: number;
    printableTextCharacters?: number;
    noReplyReason?: ChatNoReplyReason;
  };
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
  /** Authoritative accepted records that contributed to this adaptation. */
  learnedRecords?: number;
  /** Slow neural optimizer steps, kept separate from synaptic mutation events. */
  learnedParameterSteps?: number;
  /** Composite neural checksum changed; this is not module-only weight proof. */
  parametersChanged?: boolean;
  importedAt: string;
  rawTextRetained: boolean;
  rawText?: string;
  contentHash?: string;
  blobHash?: string;
  policy?: PersistedDataIngestionPolicy;
  provenanceUrl?: string;
  license?: string;
  licenseUrl?: string;
}

export interface NeuralParameterAccounting {
  mutableDenseParameters: number;
  /** Learned packed neuron rows; assembly views alias these, counted once. */
  substrateVectorParameters?: number;
  substrateDynamicSparseSynapses: number;
  dynamicSparseSynapses: number;
  totalNeuralParameters: number;
  countingRule: string;
}

export type BrainOriginKind = "ground-up" | "legacy-hybrid" | "legacy";

/**
 * Compact origin metadata. Non-native variants exist only so stored old
 * documents can be identified and refused; they are not loadable options.
 */
export interface BrainProvenance {
  originKind: BrainOriginKind;
  foundation?: {
    modelId: string;
    repository?: string;
    frozen: true;
  };
  randomInitialization?: {
    algorithm: string;
    seed?: number;
  };
}

export interface BrainRuntimeCard extends Record<string, unknown> {
  parameterAccounting?: NeuralParameterAccounting;
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
  parameterAccounting?: NeuralParameterAccounting;
}

/**
 * Durable creation handoff. `initializing` is written before a newly-created
 * brain becomes discoverable and remains authoritative across renderer or app
 * restarts. Only the final initial-learning handoff may change it to `ready`.
 */
export interface BrainReadiness {
  state: "initializing" | "ready" | "failed";
  startedAt: string;
  completedAt?: string;
  attempt?: number;
  failure?: {
    phase: "foundation" | "initial-learning";
    message: string;
    failedAt: string;
    retryable: true;
  };
  /**
   * Minimal credential-free recovery seed committed in the same atomic brain
   * document as `initializing`. The richer local task journal is private and
   * deleted after readiness; this seed closes the crash window before that
   * journal's first write.
   */
  recovery?: {
    foundation: {
      hardwareTier: HardwareTier;
      modalities: ModalityKind[];
      origin: "ground-up";
    };
    resources: Array<
      | { kind: "selection"; selectionId: string }
      | { kind: "web"; url: string }
    >;
  };
}

export interface ConversationLedgerSummary {
  format: "omni-conversation-ledger";
  formatVersion: 1;
  totalEntries: number;
  messageCount: number;
  actionCount: number;
  traceCount: number;
  headSequence: number;
  headSha256: string;
  attentionEpoch: number;
}

export interface BrainActivityLedgerSummary {
  format: "omni-brain-activity-ledger";
  formatVersion: 1;
  journalCount: number;
  journalHeadSequence: number;
  journalHeadSha256: string;
  trainingSourceCount: number;
  trainingSourceVersionCount: number;
  trainingSourceHeadSequence: number;
  trainingSourceHeadSha256: string;
  trainingSourceBytes: number;
  learnedIdeas: number;
  learnedConcepts: number;
  learnedSynapses: number;
  learnedRecords: number;
  learnedParameterSteps: number;
  parametersChangedSources: number;
  topAdaptation?: {
    sourceId: string;
    sourceLabel: string;
    learnedRecords: number;
  };
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
  readiness: BrainReadiness;
  provenance?: BrainProvenance;
  lineage: BrainLineage;
  config: BrainConfig;
  concepts: Record<string, ConceptNode>;
  synapses: Record<string, Synapse>;
  ideas: Idea[];
  workingMemory: WorkingMemoryItem[];
  liquidState: LiquidState;
  messages: ChatMessage[];
  traces: ThoughtTrace[];
  conversation?: ConversationLedgerSummary;
  activity?: BrainActivityLedgerSummary;
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

export interface JournalLedgerEntry {
  sequence: number;
  payloadSha256: string;
  rowSha256: string;
  entry: JournalEntry;
}

export interface JournalLedgerPage {
  brainId: string;
  entries: JournalLedgerEntry[];
  totalEntries: number;
  nextCursor?: string;
}

export interface TrainingSourceLedgerEntry {
  sequence: number;
  payloadSha256: string;
  rowSha256: string;
  source: TrainingSource;
}

export interface TrainingSourceLedgerPage {
  brainId: string;
  entries: TrainingSourceLedgerEntry[];
  totalEntries: number;
  nextCursor?: string;
}

export interface BrainSummary {
  id: string;
  name: string;
  preset: ArchitecturePreset;
  runtime: InferenceRuntime;
  updatedAt: string;
  concepts: number;
  synapses: number;
  neuralUpdates?: number;
  inferenceCount?: number;
  trainingSources?: number;
  /** This identity currently owns the persisted Active Mode preference. */
  activeMode: boolean;
  substrateTotals?: {
    neurons: number;
    assemblies: number;
    synapses: number;
  };
  generation: number;
  /**
   * Instance-first lineage metadata. The immutable origin is a recovery
   * checkpoint shared by a lineage, not a second running brain.
   */
  rootId?: string;
  parentId?: string;
  originChecksum?: string;
  instanceKind?: "original" | "duplicate" | "fork" | "imported";
  originInstanceCount?: number;
  provenance?: BrainProvenance;
  adaptation?: {
    sourceLabel: string;
    learnedRecords: number;
  };
}

export interface BrainActiveModeResult {
  schemaVersion: 1;
  brain: BrainDocument;
  enabled: boolean;
  activeBrainId?: string;
  /** Identities explicitly switched off to preserve the single active lease. */
  deactivated: Array<{ id: string; name: string }>;
  updatedAt: string;
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
  /** Exact neuron/assembly endpoint filter for cursor-paged synapse inspection. */
  connectedTo?: string;
  /**
   * Preferred records per page. This is not a substrate cardinality limit;
   * the worker may return fewer only to stay inside the JSON-RPC byte envelope.
   */
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
  neuronCount?: number;
  childAssemblyCount?: number;
  /** True when relationships must be traversed with a connected synapse query. */
  relationshipsPaged?: boolean;
  kind: string;
  source: string;
  confidence: number;
  /** Measured assembly-neuron firing, never confidence or importance. */
  activation?: number;
  /** False when this inspection source has no observed firing value. */
  activationObserved?: boolean;
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
  /** Live substrate revision used to bind and invalidate continuation cursors. */
  revision: string;
  entity: SubstrateEntity;
  zoom: number;
  totals: {
    neurons: number;
    assemblies: number;
    synapses: number;
  };
  matched: number;
  /** Zero-based position of this page in the complete filtered traversal. */
  offset?: number;
  returned?: number;
  pageBytes?: number;
  transportLimited?: boolean;
  hasMore: boolean;
  nextCursor?: string;
  clusters: SubstrateCluster[];
  neurons: SubstrateNeuron[];
  assemblies: SubstrateAssembly[];
  synapses: SubstrateSynapse[];
}

/** Lightweight, checksum-validated totals from the last durable substrate. */
export interface PersistedSubstrateOverview {
  brainId: string;
  revision: string;
  source: "validated-persisted-substrate";
  totals: {
    neurons: number;
    assemblies: number;
    synapses: number;
  };
  parameterAccounting?: NeuralParameterAccounting;
  latestCompletedCoverage?: PersistedDatasetCoverageSummary;
}

export interface PersistedDatasetCoverageSummary {
  brainId: string;
  manifestId: string;
  manifestHash: string;
  source: "validated-dataset-progress";
  discoveredFiles: number;
  processedFiles: number;
  rejectedFiles: number;
  discoveredRecords: number;
  processedRecords: number;
  rejectedRecords: number;
  complete: true;
  updatedAt: string;
}

export interface FreshAttentionBoundary {
  format: "omni-fresh-attention-boundary";
  formatVersion: 1;
  operationId: string;
  epoch: number;
  createdAt: string;
  messagesPreserved: number;
  tracesPreserved: number;
  synapsesPreserved: number;
  replayEntries: number;
  parameterChecksum: string;
  fastSynapseChecksum: string;
  substrateContentSha256: string;
  cleared: Record<string, number>;
}

export interface FreshAttentionResult {
  format: "omni-fresh-attention-boundary";
  formatVersion: 1;
  brainId: string;
  committed: true;
  idempotent: boolean;
  boundary: FreshAttentionBoundary;
  parameterChecksum: string;
  fastSynapseChecksum: string;
  substrateContentSha256: string;
  messagesPreserved: number;
  tracesPreserved: number;
  synapsesPreserved: number;
  replayEntries: number;
  pagedCleanupPending: boolean;
  rawPriorDialogueEligible: false;
}

export type BackgroundParameterLearningState =
  | "idle"
  | "pending"
  | "running"
  | "pausing"
  | "paused"
  | "complete"
  | "failed";

export interface WorkspaceLearningStatus {
  /** Timestamp of the committed worker metadata used for these measurements. */
  measuredAt: string;
  fastNeuralMemory: {
    state: "idle" | "learned";
    safelyStored: boolean;
    completedTurns: number;
    connectionUpdatesTotal: number;
    parameterStepsTotal: number;
    turnId?: string;
    committedAt?: string;
    /** Composite checksum includes associative sequence tensors; not dense-weight proof. */
    parameterChecksumAfter?: string;
  };
  backgroundParameters: {
    state: BackgroundParameterLearningState;
    /** A pause may come from the saved switch or a single-launch QA override. */
    pauseReason?: "user" | "launch-override";
    pending: number;
    completed: number;
    updatedAt: string;
    startedAt?: string;
    completedAt?: string;
    lastJobId?: string;
    /** Legacy composite checksum result; includes associative sequence tensors. */
    parameterChanged?: boolean;
    /** Module-only checksum result from a completed slow replay. */
    corticalParametersUpdated?: boolean;
    /** Last genuine replay failure observed in this app process. */
    lastError?: string;
    error?: string;
  };
}

export interface WorkspaceSnapshot {
  brainId: string;
  queriedAt: string;
  /** Worker-owned transparent runtime data; never reconstructed from UI config. */
  runtimeCard?: BrainRuntimeCard;
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
    resident?: number;
    paged?: number;
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
  /**
   * Plain-language view over transient neural state. The authoritative memory
   * remains the substrate; these records contain no source text or token IDs.
   */
  memory?: {
    recentWords: {
      capacity: number;
      count: number;
      evictions: number;
      temporary: true;
    };
    activeFocus: {
      count: number;
      items: Array<{
        assemblyId: string;
        activation: number;
        currentlyFiring: boolean;
      }>;
    };
    workingThoughts: {
      capacity: number;
      count: number;
      resident?: number;
      paged?: number;
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
    afterimageTrail: {
      count: number;
      averageStrength: number;
      items: Array<Record<string, unknown>>;
      reversible: true;
      rawTextStored: false;
      rawTokenIdsStored: false;
    };
    retentionDynamics: {
      trackedAssemblies: number;
      averageScore: number;
      minimumScore: number;
      maximumScore: number;
      reinforcedSynapses: number;
      fixedStages: false;
      reversible: true;
    };
    automaticSettling: {
      requiredManualAction: false;
      cycles: number;
      experienceCycles: number;
      restCycles: number;
      expiredAfterimages: number;
      reinforcementEvents: number;
      scoresRecomputedEachCycle: true;
      lastAt: string | null;
      lastSource: string | null;
      visiblePonderForced: false;
    };
  };
  hiddenBehavioralPrompt: boolean;
  rawLongTermTextInjected: boolean;
  freshAttentionBoundary?: FreshAttentionBoundary | null;
  /** Committed fast-memory facts plus main-process background-work state. */
  learning?: WorkspaceLearningStatus;
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
  /** Renderer-authored correlation metadata; never inserted into model input. */
  turnMetadata?: ChatTurnMetadata;
  generationEnd?: ChatGenerationEnd;
}

export interface ChatTurnMetadata {
  kind: "steer";
  replacesTurnId: string;
  source: "human";
  createdAt: string;
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
  workingMemory?: WorkingMemoryPlanRequest;
  hardwareTier?: HardwareTier;
  modalities?: ModalityKind[];
  initialToolPermissions?: Array<{
    toolId: string;
    level: ToolPermissionLevel;
  }>;
  /**
   * Opaque main-process selections and credential-free web URLs that must be
   * learned before first chat. The private recovery journal resolves local
   * paths; renderer code never receives them.
   */
  initialResources?: Array<
    | { kind: "selection"; selectionId: string }
    | { kind: "web"; url: string }
  >;
}

export interface WorkingMemoryPlanRequest {
  mode: WorkingMemoryMode;
  /** Existing identity: main resolves saved geometry, never a renderer shape. */
  brainId?: string;
  /** Hardware auto-detection may provide the tier before a brain exists. */
  hardwareTier?: HardwareTier;
  /**
   * Decimal text intentionally avoids a UI-defined numeric ceiling. Manual
   * values still must pass the same live RAM/storage preflight as the slider.
   */
  requestedItems?: string;
  /** Manual active-context request. Decimal text avoids a renderer ceiling. */
  requestedContextTokens?: string;
  /** Live hardware probe result; never inferred from a user-facing mode. */
  acceleratorAvailable?: boolean;
  /** Independent of working-context sizing; controls the whole Omni process. */
  systemRamMode?: SystemRamMode;
  /** Advanced cap from 30 through 100 percent of the OS-reserved safe pool. */
  systemRamSharePercent?: number;
  /** One device-wide spill/growth pool; reservations use the largest brain, not a sum. */
  storagePoolMode?: StoragePoolMode;
  /** Manual device-wide pool size in bytes, encoded as decimal text without a UI cap. */
  storagePoolBytes?: string;
  /** Selected source bytes; sources stay in place while this sizes training scratch. */
  trainingSourceBytes?: number;
}

export interface WorkingMemoryResourcePlan {
  /** Versioned main-selected dimensions and exact symbolic packed inventory. */
  nativeArchitecture?: NativeArchitectureDescriptor;
  schemaVersion: 1;
  diskSpace: DiskSpaceTelemetry;
  /** Present during a new native build; an existing brain uses its saved architecture. */
  architecture?: {
    format: "omni-ground-up-architecture-profile";
    formatVersion: 1;
    architecture: "OmniCortex";
    origin: "ground-up-random-initialization";
    externalPretrainedWeights: false;
    hardwareTier: HardwareTier;
    dModel: number;
    layers: number;
    feedForward: number;
    vsaDimensions: number;
    routerNeurons: number;
    workingMemoryItems: number;
    workspaceLatents: number;
    exactLogicalParameterCount: number;
    exactPackedProjectionParameterCount: number;
    exactPackedTableParameterCount: number;
    exactPackedTernaryParameterCount: number;
    packedTernaryWeightBytes: number;
    packedWorkspaceTableBytes: number;
    packedMetaplasticityReserveBytes: number;
    fixedControlBufferReserveBytes: number;
    packedUpdateScratchBytes: number;
    residentInferenceStateBytes: number;
    minimumTrainingStateBytes: number;
    checkpointTensorBytes: number;
    parameterCountBasis: "architecture-logical-neural-elements";
    checkpointByteBasis: "packed-weights-plus-nonweight-state-reserve";
  };
  mode: WorkingMemoryMode;
  allowed: boolean;
  blockers: string[];
  warnings: string[];
  hardwareTier: HardwareTier;
  selectedItems: number;
  selectedItemsText: string;
  sliderMaximumItems: number;
  sliderMaximumItemsText: string;
  suitableRange: {
    minimumItems: number;
    autoItems: number;
    maximumItems: number;
    extendedItems: number;
  };
  context: {
    /** Exact active window selected for the worker configuration. */
    selectedTokens: number;
    /** Conservative tier baseline; the only tier-defined context target. */
    floorTokens: number;
    /** Live resource-derived Auto recommendation. */
    autoTokens: number;
    /** Higher resource-derived recommendation used by Extended. */
    extendedTokens: number;
    /** RAM plus designated storage/model-position boundary for the slider. */
    maximumTokens: number;
    suitableMinimumTokens: number;
    suitableMaximumTokens: number;
    evidence: {
      source: "measured-device-theoretical-allocation" | "live-device-model-measurement";
      /** Exact selected/saved geometry when supplied; conservative reference otherwise. */
      modelHiddenSize: number;
      modelLayers: number;
      /** Float32 vector plus metadata/storage allowances, not measured RSS/file usage. */
      estimatedResidentMemoryItemBytes: number;
      estimatedPagedMemoryItemBytes: number;
      /** Shared post-runtime 15% policy reserve, not memory already allocated. */
      reservedTrainingTransferBytes: number;
      modelContextLimitTokens: number;
      safeRamAfterModelBytes: number;
      contextResidentBudgetBytes: number;
      estimatedKvActivationBytesPerToken: number;
      /** Model/runtime reserve uses two live context workspaces. */
      contextWorkspaceMultiplier: 1 | 2;
      minimumContextWorkspaceBytes: number;
      selectedContextResidentBytes: number;
      selectedContextSpillBytes?: number;
      contextOffloadBudgetBytes?: number;
      residentContextMaximumTokens?: number;
      contextSpillCapacityBytes?: number;
      residentTokenBytesPerToken?: number;
      contextMetadataBytesPerToken?: number;
      workspaceResidentBytes?: number;
      pageTokens?: number;
      pagedKvBytesPerToken?: number;
      diskBlockSizeBytes?: number;
      acceleratorAvailable: boolean;
      storageClass: "slow-storage" | "moderate-storage" | "fast-storage";
      measuredStorageBytesPerSecond: number;
      autoContextRamFraction: number;
    };
  };
  resources: {
    totalMemoryBytes: number;
    availableMemoryBytes: number;
    diskTotalBytes: number;
    diskFreeBytes: number;
    ramReserveBytes: number;
    mandatoryFreeDiskBytes: number;
    checkpointHeadroomBytes: number;
    usableDiskAfterReserveBytes: number;
    modelBytes: number;
    /** Exact trainable master-element count for a new native build. */
    modelParameterCount?: number;
    modelParameterCountBasis?: "architecture-logical-neural-elements";
    modelStorageBasis?: "packed-weights-plus-nonweight-state-reserve";
    modelWorkingSetBytes: number;
    /** Estimated live learning state; raw checkpoint bytes are reported separately. */
    estimatedResidentModelBytes: number;
    /** Native model state plus Electron/worker/runtime overhead. */
    fullFoundationRuntimeBytes: number;
    /** Smallest complete model layer that must remain live per step. */
    minimumLayerResidencyBytes: number;
    /** Model/runtime bytes actually assigned to the live RAM tier. */
    residentFoundationBytes: number;
    runtimeOverheadBytes: number;
    configuredMemorySpillBytes: number;
    /** Physical capacity after the adaptive OS reserve; independent of current pressure. */
    safeRamPoolBytes: number;
    /** Reclaimable RAM available right now after the same OS reserve. */
    availableSafeRamBytes: number;
    systemRamBudgetBytes: number;
    /** Portion of the persisted budget that is currently reclaimable. */
    currentOmniAvailableBytes: number;
    /** Persisted capacity that is temporarily occupied by other processes. */
    currentOmniShortfallBytes: number;
    systemRamMode: SystemRamMode;
    systemRamSharePercent: number;
    autoSystemRamSharePercent: number;
    storagePoolMode: StoragePoolMode;
    /** Shared ceiling used by every brain; instances do not multiply the reservation. */
    sharedStoragePoolBytes: number;
    requiredStoragePoolBytes: number;
    maximumStoragePoolBytes: number;
    trainingSourceBytes: number;
    trainingScratchBytes: number;
    futureGrowthHeadroomBytes: number;
  };
  offload: {
    required: boolean;
    residentMemoryItems: number;
    pagedMemoryItems: number;
    modelSpillBytes: number;
    /** Regenerable safe-tensor cache space reserved for model layer offload. */
    modelOffloadScratchBytes: number;
    memorySpillBytes: number;
    contextSpillBytes?: number;
    estimatedSlowdownPercent: number;
    benchmark: {
      measuredAt: string;
      sampleBytes: number;
      memoryBytesPerSecond: number;
      storageBytesPerSecond: number;
      cacheHit: boolean;
    };
  };
  semantics: {
    unit: "recurrent-paged-memory-item";
    denseAttentionClaim: false;
    /** Per-device conversational floor; independent of recurrent item count. */
    contextFloorTokens: number;
    /** True when selected cold K/V pages require the designated storage pool. */
    contextPagedToStorage: boolean;
    /** The selected RAM/context capacity does not shrink when another app opens. */
    capacityPersistsAcrossPressure: true;
    /** A shared pool is budgeted once at the largest active-brain requirement. */
    storagePoolShareRule: "largest-brain-not-sum";
    hotRamPriority: string[];
    spillOrder: string[];
  };
  training: {
    policy: "ram-first-adaptive-streaming";
    physicalBatchSize: number;
    gradientAccumulation: number;
    windowTokens: number;
    allSourceBytesVisited: true;
    scratchMode: "emergency-checkpoint-only";
    scratchUsedAsVirtualRam: false;
    sequentialScratchWrites: true;
    minimumScratchIntervalSeconds: number;
    storageClass: "slow-storage" | "moderate-storage" | "fast-storage";
    measuredStorageBytesPerSecond: number;
  };
}

/**
 * Streamed, measured initialization state. A new instance is not chat-ready
 * until the final readiness event and the create request both complete.
 */
export interface BuildProgressEvent {
  brainId?: string;
  streamId?: string;
  sequence: number;
  phase:
    | "allocating"
    | "language-foundation"
    | "capability-curriculum"
    | "action-policy"
    | "modalities"
    | "packing"
    | "readiness"
    | "initial-materials"
    | "complete";
  progress: number;
  label: string;
  /** The authoritative first-learning job snapshot when this is initial-materials progress. */
  job?: RuntimeJob;
  data?: {
    recordsVisited?: number;
    recordsTotal?: number;
    neuronDelta?: number;
    assemblyDelta?: number;
    synapseDelta?: number;
    /** Composite neural checksum, including associative state. */
    parameterChecksumChanged?: boolean;
    readinessChecks?: Record<string, boolean>;
    diskSpace?: DiskSpaceTelemetry;
  };
}

export interface DeleteInstanceRequest {
  brainId: string;
  /** Confirmation 1: the irreversible warning was explicitly accepted. */
  acknowledgedIrreversible: true;
  /** Confirmation 2: exact, case-sensitive instance name. */
  typedName: string;
  /** Confirmation 3: exact phrase `PERMANENTLY DELETE <instance name>`. */
  finalConfirmation: string;
}

export interface DeleteInstanceResult {
  deleted: true;
  brainId: string;
  name: string;
  recoverable: false;
  removedSharedBlobs: number;
  reclaimedBytes: number;
}

export interface ImportUrlRequest {
  url: string;
  expectedSha256?: string;
}

export interface BrainImportFailure {
  fileName: string;
  message: string;
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
  /** Ordered hashes of the durable neural, artifact, and ledger components. */
  checkpointComponentSha256?: string[];
  durableState?: {
    conversationLedger: true;
    activityLedger: true;
    neuralConversationLedger: boolean;
    artifactIndex: boolean;
  };
  storage?: {
    files: number;
    logicalBytes: number;
    sharedBytes: number;
    physicalBytesAdded: number;
  };
}

export type BrainExportMode = "current" | "origin" | "private-archive" | "referenced";

export interface DiskSpaceTelemetry {
  schemaVersion: 1;
  measuredAt: string;
  diskTotalBytes: number;
  diskFreeBytes: number;
  mandatoryReserveBytes: number;
  selectedDatasetBytes: number;
  modelBytes: number;
  checkpointBytes: number;
  maximumWorkingMemorySpillBytes: number;
  futureGrowthBytes: number;
  operationWriteBytes: number;
  projectedRemainingBytes: number;
  projectedAboveReserveBytes: number;
  paused: boolean;
}

export type BrainStorageOperationKind =
  | "duplicate"
  | "fork"
  | "snapshot"
  | "restore"
  | "export"
  | "import";
export type BrainStorageOperationState =
  | "queued"
  | "running"
  | "paused"
  | "cancelling"
  | "cancelled"
  | "complete"
  | "failed";

export interface BrainStorageOperationEvent {
  schemaVersion: 1;
  operationId: string;
  kind: BrainStorageOperationKind;
  state: BrainStorageOperationState;
  phase: string;
  label: string;
  sourceBrainId?: string;
  targetBrainId?: string;
  filesCompleted: number;
  filesTotal: number;
  logicalBytesCompleted: number;
  logicalBytesTotal: number;
  physicalBytesAdded: number;
  sharedBytes: number;
  bytesPerSecond: number;
  elapsedMs: number;
  etaMs?: number;
  diskSpace?: DiskSpaceTelemetry;
  error?: string;
  updatedAt: string;
}

export type DataIngestionPolicy = "encode" | "pretrain" | "archive";
/** Read-only compatibility value for already hash-bound source receipts. */
export type PersistedDataIngestionPolicy = DataIngestionPolicy | "consolidate";

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
  /** Original selected path when `path` names an app-owned immutable snapshot. */
  sourcePath?: string;
  relativePath: string;
  format: DatasetFormat;
  bytes: number;
  /** Stable file identity recorded when the manifest was committed. */
  lastModifiedMs?: number;
  /** Incrementally streamed content identity for the committed source snapshot. */
  contentSha256?: string;
  /** Explicit traversal rejection; these entries never reach neural training. */
  rejection?: string;
  /** App-owned immutable snapshot kind used to prevent live-source drift. */
  snapshotKind?: "sqlite";
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

export type DatasetPreviewPhase =
  | "discovering"
  | "hashing"
  | "committing"
  | "complete"
  | "cancelled"
  | "failed";

/**
 * Live, bounded metadata for an uncapped manifest build. Counts are
 * observational while traversal is in progress; the committed manifest is
 * still the sole authoritative snapshot used by training.
 */
export interface DatasetPreviewProgress {
  schemaVersion: 1;
  requestId: string;
  brainId: string;
  phase: DatasetPreviewPhase;
  discoveredFiles: number;
  discoveredBytes: number;
  hashedFiles: number;
  hashedBytes: number;
  currentFile?: string;
  currentFileBytes?: number;
  currentFileHashedBytes?: number;
  message: string;
  startedAt: string;
  updatedAt: string;
}

export interface DatasetCursor {
  schemaVersion: 1;
  manifestId: string;
  /** Fresh on an explicit restart; stable across pause/resume. */
  runId?: string;
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
  /** Number of records represented when equivalent failures are aggregated. */
  count?: number;
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
  /** Exhaustive failure count even when `errors` contains sampled summaries. */
  errorCount?: number;
  errorsTruncated?: boolean;
  errorLog?: string;
  complete: boolean;
  updatedAt: string;
}

/**
 * Bounded durable receipt for the most recently completed manifest entry.
 * It contains only the data needed to finish Electron-side bookkeeping after
 * a worker acknowledgement; source bytes and extracted text never enter it.
 */
export interface DatasetEntryReceipt {
  schemaVersion: 1;
  manifestId: string;
  runId?: string;
  manifestHash: string;
  transactionKey: string;
  entryIndex: number;
  epoch: number;
  contentHash: string;
  policy: PersistedDataIngestionPolicy;
  outcome: "learned" | "rejected";
  sourceId?: string;
  journalId?: string;
  sourceKind?: TrainingSource["kind"];
  sourceBytes?: number;
  learnedIdeas?: number;
  learnedConcepts?: number;
  learnedSynapses?: number;
  learnedParameterSteps?: number;
  parametersChanged?: boolean;
  workerDuplicate?: boolean;
  warnings?: string[];
  coverage?: TrainingCoverage;
  completedAt: string;
}

/**
 * The sole authoritative manifest progress generation. `cursor.json` and
 * `coverage.json` are compatibility materializations and are never read back.
 */
export interface DatasetProgressGeneration {
  schemaVersion: 1;
  format: "omni-dataset-progress";
  manifestId: string;
  manifestHash: string;
  generation: number;
  cursor: DatasetCursor;
  coverage: TrainingCoverage;
  lastEntryReceipt?: DatasetEntryReceipt;
  updatedAt: string;
  contentSha256: string;
}

export interface DatasetResumeCandidate {
  brainId: string;
  manifestId: string;
  manifestHash: string;
  state: "ready" | "paused" | "failed" | "interrupted";
  discoveredFiles: number;
  processedFiles: number;
  rejectedFiles: number;
  currentEpoch: number;
  requestedEpochs: number;
  updatedAt: string;
}

export interface IngestFilesRequest {
  brainId: string;
  policy?: DataIngestionPolicy;
  selection?: import("./uploadSupport").ExperienceUploadKind;
}

export interface DatasetPreviewRequest
  extends Omit<IngestFilesRequest, "selection"> {
  /** Renderer-issued correlation used for progress and cooperative cancel. */
  requestId?: string;
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
  /** Main-process traversal result used only for resource planning. */
  bytes?: number;
  /** Regular files measured without following symbolic links. */
  fileCount?: number;
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
  | "teacher"
  | "ingestion"
  | "crawl"
  | "image"
  | "audio"
  | "video"
  | "vision"
  | "agent";
export type RuntimeJobState =
  | "queued"
  | "running"
  | "cancelling"
  | "complete"
  | "failed"
  | "cancelled";

export interface RuntimeJob {
  /** Client voice chunk correlation; never part of neural conditioning. */
  speechRequestId?: string;
  id: string;
  brainId: string;
  kind: RuntimeJobKind;
  state: RuntimeJobState;
  progress: number;
  label: string;
  /** Operational presentation phase; never a neural or dataset authority. */
  phase?: "preparing-neural-substrate" | "traversing-dataset";
  createdAt: string;
  updatedAt: string;
  error?: string;
  output?: unknown;
  /** Present while this job is waiting behind the serial neural owner. */
  queue?: ChatQueueState;
  /** Measured ingestion telemetry. Operational only; never neural authority. */
  telemetry?: TrainingTelemetry;
  /** Latest live storage reading for any long write-producing job. */
  diskSpace?: DiskSpaceTelemetry;
  /** A bounded, non-authoritative preview emitted while media is decoding. */
  preview?: ModalityPreview;
}

export interface RuntimeJobEvent {
  job: RuntimeJob;
}

export type ApiTeacherProvider = "openai" | "anthropic" | "gemini";

export interface ApiTeacherCredentialRequest {
  provider: ApiTeacherProvider;
  apiKey: string;
}

export interface ApiTeacherProviderStatus {
  provider: ApiTeacherProvider;
  configured: boolean;
  persistence: "encrypted" | "session" | "none";
  keyHint?: string;
}

export interface ApiTeacherTrainingRequest {
  brainId: string;
  provider: ApiTeacherProvider;
  model: string;
  /** Explicit user-authored learning questions. They become training data, not a prompt at chat time. */
  prompts: string[];
  maxOutputTokens?: number;
}

export interface ApiTeacherTrainingReport {
  provider: ApiTeacherProvider;
  model: string;
  requested: number;
  learned: number;
  failed: number;
  requestHashes: string[];
  responseHashes: string[];
  rawCredentialsStoredInBrain: false;
  hiddenBehavioralPromptUsed: false;
}

export type McpTransport = "http" | "stdio";

export interface McpServerRegistrationRequest {
  brainId: string;
  id: string;
  label: string;
  transport: McpTransport;
  url?: string;
  command?: string;
  args?: string[];
  bearerToken?: string;
}

export interface McpToolSummary {
  id: string;
  remoteName: string;
  actions: string[];
  inputSchema: Record<string, unknown>;
}

export interface McpServerSummary {
  brainId: string;
  id: string;
  label: string;
  transport: McpTransport;
  endpoint: string;
  connected: boolean;
  tools: McpToolSummary[];
  credentialConfigured: boolean;
  error?: string;
}

export interface ToolRuntimePreferences {
  /** Ask-level approval tokens expire after this visible, user-configured interval. */
  approvalTimeoutSeconds: number;
}

export type ModalityKind = "image" | "audio" | "video" | "vision";

export interface ModalityGenerationSettings {
  /** Legacy is accepted only for old callers; current UI/model requests use Auto or Exact. */
  outputMode?: "auto" | "exact" | "legacy";
  width?: number;
  height?: number;
  durationMs?: number;
  sampleRate?: number;
  fps?: number;
  includeAudio?: boolean;
  /** Interactive target, not a model/output ceiling. */
  targetLatencyMs?: number;
  previewIntervalMs?: number;
}

export interface ModalityGenerateRequest {
  brainId: string;
  modality: ModalityKind;
  prompt?: string;
  conceptIds?: string[];
  /** Full immutable structural ID argument set; never a diagnostic page. */
  conceptIdView?: import("./conceptIdView").ConceptIdView;
  sourceTurnId?: string;
  inputPath?: string;
  settings?: ModalityGenerationSettings;
  /** Worker-issued correlation for media already decoding during chat. */
  neuralActionId?: string;
  seed?: number;
}

export interface NeuralSpeechGenerateRequest {
  brainId: string;
  requestId: string;
  text: string;
  rate?: number;
}

export interface GeneratedArtifact {
  id: string;
  brainId: string;
  modality: Exclude<ModalityKind, "vision">;
  mimeType: string;
  sha256: string;
  bytes: number;
  createdAt: string;
  seed?: number;
  initialization?: string;
  qualityNote?: string;
  mediaUrl?: string;
  available: boolean;
  unavailableReason?: "missing" | "integrity-failed";
}

export interface GeneratedArtifactPage {
  brainId: string;
  artifacts: GeneratedArtifact[];
  totalArtifacts: number;
  nextCursor?: string;
}

/**
 * Read-only, committed neural-media readiness. Generic audio perception
 * and generation are deliberately separate from ASR/TTS: a sound decoder is
 * not advertised as an intelligible voice model.
 */
export interface NeuralModalityCapabilities {
  brainId: string;
  hardwareTier: HardwareTier;
  imagePerception: boolean;
  audioPerception: boolean;
  videoPerception: boolean;
  imageGeneration: boolean;
  audioGeneration: boolean;
  videoGeneration: boolean;
  neuralSpeechRecognition: boolean;
  neuralSpeechSynthesis: boolean;
  /** Physical same-brain waveform route, not a claim of intelligible speech. */
  audioRegionAvailable?: boolean;
  speechPairedExamples?: number;
  speechQuality?: "needs-speech-training" | "unverified";
  synchronizedVideoAudioGeneration: boolean;
  sameBrainSubstrate: true;
  hiddenBehavioralPrompt: false;
  detail: string;
}

export type LiveObservationModality = "image" | "audio" | "video";
export type LiveObservationCaptureMode =
  | "auto"
  | "motion"
  | "balanced"
  | "detail";
export type LiveObservationBenchmarkClass =
  | "fallback"
  | "balanced"
  | "high"
  | "native";
export type LiveObservationPermissionSource =
  | "camera"
  | "microphone"
  | "screen"
  | "mixed";

/**
 * Evidence supplied by the trusted Electron capture controller. This records
 * device authority; it is never model-facing behavioral text.
 */
export interface LiveObservationPermission {
  source: LiveObservationPermissionSource;
  granted: true;
  scope: "session";
  grantedAt: string;
  /** Optional one-way identifier; raw OS device ids must not cross preload. */
  deviceIdHash?: string;
}

export interface LiveObservationSessionStartRequest {
  brainId: string;
  modalities: LiveObservationModality[];
  permission: LiveObservationPermission;
  /** working = temporary recurrent state; neural = assemblies/STDP, never raw bytes. */
  retention?: "working" | "neural";
  /** Bounded transport pressure only; it never limits committed dataset traversal. */
  maxInFlight?: number;
  /**
   * Trusted capture negotiation. There is no product resolution/FPS ceiling:
   * actual values are bounded by the source, current device resources, and
   * transport backpressure. Visual fallback is 320x180 at 1 FPS.
   */
  capture?: {
    mode: LiveObservationCaptureMode;
    width?: number;
    height?: number;
    fps?: number;
    sourceNativeWidth?: number;
    sourceNativeHeight?: number;
    sourceNativeFps?: number;
    audioSampleRate?: number;
    benchmarkClass: LiveObservationBenchmarkClass;
  };
}

export interface LiveObservationPacket {
  sessionId: string;
  modality: LiveObservationModality;
  sequence: number;
  /** Monotonic capture timestamp supplied by the trusted controller. */
  timestampMs: number;
  mimeType: string;
  data: Uint8Array;
  settings?: {
    sampleRate?: number;
    channels?: number;
    width?: number;
    height?: number;
    durationMs?: number;
    observationControlId?: string;
    resolutionMode?: "native" | "current" | "custom";
    burstIndex?: number;
    burstCount?: number;
  };
}

export interface LiveObservationSession {
  id: string;
  brainId: string;
  state: "active" | "stopping" | "stopped" | "cancelled";
  modalities: LiveObservationModality[];
  permission: LiveObservationPermission;
  retention: "working" | "neural";
  maxInFlight: number;
  /** Resource/source-derived transport bound, never a fixed product limit. */
  maxPacketBytes: number;
  inFlight: number;
  packetsReceived: number;
  packetsAccepted: number;
  packetsDroppedBackpressure: number;
  bytesAccepted: number;
  lastSequence: number;
  createdAt: string;
  updatedAt: string;
  capabilities: {
    imageNeural: boolean;
    audioNeural: boolean;
    videoNeural: boolean;
  };
  capture: {
    mode: LiveObservationCaptureMode;
    width?: number;
    height?: number;
    fps?: number;
    sourceNativeWidth?: number;
    sourceNativeHeight?: number;
    sourceNativeFps?: number;
    audioSampleRate?: number;
    benchmarkClass: LiveObservationBenchmarkClass;
    revision: number;
  };
}

export interface LiveObservationControl {
  id: string;
  sessionId: string;
  kind: "configure" | "snapshot";
  source: "human" | "brain";
  temporary: true;
  requested: {
    mode?: LiveObservationCaptureMode;
    width?: number;
    height?: number;
    fps?: number;
    /** Compatibility signal for a native-resolution snapshot. */
    fullResolution?: true;
    /** Defaults to native; custom requires width and height. */
    resolutionMode?: "native" | "current" | "custom";
    /** Short burst length; applied value is resource/source bounded. */
    burstCount?: number;
    /** Inter-frame delay; applied value is resource/source bounded. */
    intervalMs?: number;
    /** Temporary configure duration; applied value is resource bounded. */
    durationMs?: number;
  };
  state: "requested" | "applied" | "rejected" | "cancelled" | "reverted";
  actual?: {
    mode: LiveObservationCaptureMode;
    width?: number;
    height?: number;
    fps?: number;
    fullResolution?: true;
    resolutionMode?: "native" | "current" | "custom";
    burstCount?: number;
    intervalMs?: number;
    durationMs?: number;
    /** Capture revision restored after a temporary configure action. */
    revertToRevision?: number;
    revision: number;
  };
  reason?: string;
  createdAt: string;
  updatedAt: string;
}

export interface LiveObservationControlResolution {
  sessionId: string;
  controlId: string;
  state: "applied" | "rejected" | "cancelled" | "reverted";
  actual?: LiveObservationControl["actual"];
  reason?: string;
}

export interface LiveObservationControlRequest {
  sessionId: string;
  kind: LiveObservationControl["kind"];
  requested: LiveObservationControl["requested"];
}

export interface LiveObservationPacketResult {
  session: LiveObservationSession;
  accepted: boolean;
  reason?: "backpressure" | "session-stopped";
  observation?: {
    assemblyId?: string | null;
    spikeRate: number;
    novelty: number;
    packetSha256: string;
    rawPacketStored: false;
    datasetCoverageCommitted: false;
    sameBrainSharedIdeaSpace: true;
    hiddenBehavioralPrompt: false;
    perception?: {
      sourceWidth: number;
      sourceHeight: number;
      globalDecodedWidth: number;
      globalDecodedHeight: number;
      tilesAvailable: number;
      tilesEncoded: number;
      tileCoverage: number;
      tileInputSize: number;
      tileGridRows: number;
      tileGridColumns: number;
      tileSelection: "uniform-resource-scaled";
      tileBinding: "ternary-position-vsa";
      spatialTileEncoder: "same-brain-ternary-image-pack";
      temporalFrames: number;
      rawTilesStored: false;
    };
    observationControlId?: string;
    resolutionMode?: "native" | "current" | "custom";
    burstIndex?: number;
    burstCount?: number;
  };
  actions?: StructuredAction[];
  controls?: LiveObservationControl[];
}

export interface LiveObservationEvent {
  type:
    | "started"
    | "packet"
    | "backpressure"
    | "action"
    | "control"
    | "stopped"
    | "cancelled";
  session: LiveObservationSession;
  packet?: LiveObservationPacketResult;
  actions?: StructuredAction[];
  control?: LiveObservationControl;
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
  origin: "ground-up";
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
  approvalExpiresAt?: string;
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
  /** Same-cortex emission identity; transport/audit only, not a model prompt. */
  actionId?: string;
  inputSchema?: Record<string, unknown>;
  selectionPhase?: string;
  selectionStep?: number;
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
  /** Exact artifact cancellation is pending; not a stopped/failed chat turn. */
  cancellationRequested?: boolean;
  inlineGenerationOwned?: boolean;
  execution?: ToolExecutionResult;
  evolutionRunId?: string;
  /** Latest safe progressive preview for an imagination action. */
  preview?: ModalityPreview;
  error?: string;
  attentionEpoch?: number;
}

/**
 * Resolve one already-visible Ask action without creating another chat turn.
 * Main owns the action arguments and completed tool output; the renderer sends
 * only opaque correlation identifiers and the single-use approval token.
 */
export interface ApprovedChatActionRequest {
  brainId: string;
  actionEventId: string;
  approvalToken: string;
}

export interface ApprovedChatActionResult {
  actionEvent: ActionEvent;
  learned: boolean;
  brain?: BrainDocument;
  learningError?: string;
}

export interface ConversationLedgerEntry {
  sequence: number;
  id: string;
  kind: "message" | "action" | "trace";
  createdAt: string;
  attentionEpoch: number;
  payloadSha256: string;
  rowSha256: string;
  message?: ChatMessage;
  action?: ActionEvent;
  trace?: ThoughtTrace;
}

export interface ConversationLedgerPage {
  brainId: string;
  entries: ConversationLedgerEntry[];
  totalEntries: number;
  hasOlder: boolean;
  beforeSequence?: number;
  nextBeforeSequence?: number;
  headSequence: number;
  headSha256: string;
}

/**
 * Progressive media is display-only until the corresponding action completes.
 * The main process validates and bounds every field before it crosses preload.
 */
export interface ModalityPreview {
  /** Versioned fields travel through the existing protocol-1 job/action stream. */
  schemaVersion?: 1;
  revision: number;
  progress?: number;
  statusLabel?: string;
  mimeType?: string;
  dataUrl?: string;
  /** Main-process lease; contains no filesystem path or media bytes. */
  mediaUrl?: string;
  path?: string;
  artifactPath?: string;
  modality?: Exclude<ModalityKind, "vision">;
  stage?:
    | "diffusion-vq-decode"
    | "codec-waveform"
    | "temporal-frame-timeline";
  completedUnits?: number;
  totalUnits?: number;
  previewCount?: number;
  width?: number;
  height?: number;
  sampleCount?: number;
  totalSamples?: number;
  durationMs?: number;
  sampleRate?: number;
  fps?: number;
  frameCount?: number;
  totalFrames?: number;
  hardwareTier?: HardwareTier;
  cadence?: "hardware-aware-bounded-synchronous";
  producer?: "same-brain-decoder";
  /** SHA-256 of the exact encoded bytes carried by dataUrl for this revision. */
  payloadSha256?: string;
  actualDecoderOutput?: true;
  spatialResolutionReduced?: false;
  hardwareScaled?: true;
  modelDefinedMaximum?: null;
  trained?: boolean;
  trainingState?: "untrained-diagnostic" | "trained-unverified-quality";
  semanticQualityClaimed?: false;
  partialCoverage?: boolean;
  coveredFraction?: number;
  ideaSource?:
    | "manual-prompt-and-active-assemblies"
    | "manual-prompt"
    | "active-assemblies"
    | "active-working-memory"
    | "active-liquid-state"
    | "intrinsic-neural-cold-start";
  activeAssemblyCount?: number;
  promptProvided?: boolean;
}

interface ChatStreamEventBase {
  id: string;
  brainId: string;
  turnId: string;
  sequence: number;
  createdAt: string;
}

export type RuntimeActivityOwner =
  | "build"
  | "chat"
  | "evolution"
  | "training"
  | "ingestion"
  | "modality"
  | "inspection"
  | "idle"
  | "system";

export interface RuntimeActivityReference {
  requestId: string;
  owner: RuntimeActivityOwner;
  label: string;
  method: string;
  brainId?: string;
  jobId?: string;
  turnId?: string;
}

export interface ChatQueueState {
  position: number;
  queuedBehind: RuntimeActivityReference;
}

export interface ChatCancellationState {
  phase: "pending" | "queued" | "running";
  /** True only when the cancelled turn never owned the worker. */
  unrelatedActivityContinues: boolean;
  /** True only after a dispatched worker has actually exited. */
  workerTerminationAcknowledged: boolean;
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

/**
 * Generation has ended and post-reply neural work remains active. During the
 * first phase the visible turn itself is not committed yet. During action
 * integration the visible turn is committed and only the completed, visible
 * action result is entering neural learning; it is never a synthetic message.
 */
export interface ChatPhaseStreamEvent extends ChatStreamEventBase {
  type: "chat-phase";
  phase: "reply-complete-learning" | "action-result-learning";
  replyComplete: true;
  turnCommitted: boolean;
  learning: true;
  saving: true;
}

/** Exact durable reply, published before optional action/result work settles. */
export interface ChatReplyCommittedStreamEvent extends ChatStreamEventBase {
  type: "chat-reply-committed";
  humanMessage: ChatMessage;
  brainMessage: ChatMessage;
  pendingActions: number;
}

export interface ChatStateStreamEvent extends ChatStreamEventBase {
  type: "chat-state";
  state: "started" | "queued" | "complete" | "steered" | "stopped" | "no-reply" | "cancelled" | "failed";
  error?: string;
  queue?: ChatQueueState;
  cancellation?: ChatCancellationState;
  /** Typed operational correlation only; it is not model-facing prose. */
  turnMetadata?: ChatTurnMetadata;
}

/**
 * One typed, ordered renderer stream. No response prose, tags, or hidden
 * instructions are parsed to manufacture an action.
 */
export type ChatStreamEvent =
  | ChatTokenStreamEvent
  | ChatActionStreamEvent
  | ChatModalityPreviewStreamEvent
  | ChatPhaseStreamEvent
  | ChatReplyCommittedStreamEvent
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
  coreParameterDeltaNorm?: number;
  parameterDeltaScope?: Record<string, unknown>;
  substrateParameterDeltaNorm?: null;
  substrateParameterDeltaMeasured?: false;
  parameterChecksumScope?: string;
  actionPolicyScores: Record<ActionKind, number>;
  proposedActionKinds: ActionKind[];
  note: string;
}

export interface IdleCycleResult {
  brainId: string;
  ran: boolean;
  reason?:
    | "idle-cognition-disabled"
    | "initial-learning"
    | "foreground-work"
    | "cooldown"
    | "no-learned-assemblies";
  retryAfterSeconds?: number;
  trace?: IdleCognitionTrace;
  actions: StructuredAction[];
  actionEvents?: ActionEvent[];
  metrics?: Record<string, unknown>;
  runtimeCard?: BrainRuntimeCard;
}

export type EvolutionRunState =
  | "experimenting"
  | "stopping"
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
   * architecture candidates use isolated worker-owned safe-tensor overlays.
   * Width/head changes require retained-state migration, training, held-out
   * evaluation and a promotion decision tied to that exact evaluated state.
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
  architectureChange?:
    | { mutation: "grow-experts"; addExperts?: number }
    | { mutation: "grow-depth"; addLayers?: number }
    | { mutation: "grow-router"; addNeurons?: number }
    | { mutation: "grow-regions"; addRegions?: number; neuronsPerRegion: number }
    | { mutation: "resize-width"; dModel: number; feedForward?: number; nHeads?: number }
    | { mutation: "repartition-heads"; nHeads: number };
  /** Explicit real-data holdouts for an isolated width/head candidate. The
   * protected evaluator, not a generated score, decides promotion. */
  geometryHoldouts?: {
    token: Array<{ path: string; records: number }>;
    modality: Array<{ path: string; kind: "image" | "audio" | "video"; conditionText: string }>;
    tool: Array<{ path: string; records: number }>;
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

export interface MobileGatewayDevice {
  id: string;
  name: string;
  createdAt: string;
  lastSeenAt: string;
}

export interface MobileGatewayStatus {
  schemaVersion: 1;
  protocolVersion: 2;
  state: "stopped" | "listening";
  allowLan: boolean;
  transportSecurity: "stopped" | "loopback-http" | "certificate-pinned-tls";
  port?: number;
  /** HTTP loopback/emulator or HTTPS LAN addresses with an out-of-band pin fragment. */
  baseUrls: string[];
  /** Public SHA-256 identity displayed and encoded into LAN pairing addresses. */
  certificateSha256?: string;
  devices: MobileGatewayDevice[];
}

export interface MobilePairingSession extends MobileGatewayStatus {
  state: "listening";
  code: string;
  expiresAt: string;
}

export interface MobileGatewayStartRequest {
  /**
   * False binds only to this computer. True is an explicit trusted-LAN opt-in;
   * every advertised LAN address uses the installation's pinned TLS identity.
   */
  allowLan: boolean;
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
    completeInitialization(id: string): Promise<BrainDocument>;
    retryInitialization(id: string): Promise<BrainDocument>;
    update(id: string, config: BrainConfig): Promise<BrainDocument>;
    setOnlineLearning(id: string, enabled: boolean): Promise<BrainDocument>;
    setActiveMode(id: string, enabled: boolean): Promise<BrainActiveModeResult>;
    duplicate(id: string, name?: string, operationId?: string): Promise<BrainDocument>;
    fork(id: string, name?: string, operationId?: string): Promise<BrainDocument>;
    pauseStorageOperation(operationId: string): Promise<BrainStorageOperationEvent>;
    resumeStorageOperation(operationId: string): Promise<BrainStorageOperationEvent>;
    cancelStorageOperation(operationId: string): Promise<BrainStorageOperationEvent>;
    onStorageOperation(listener: (event: BrainStorageOperationEvent) => void): () => void;
    remove(request: DeleteInstanceRequest): Promise<DeleteInstanceResult>;
    snapshot(
      id: string,
      label?: string,
      operationId?: string
    ): Promise<BrainSnapshotSummary>;
    listSnapshots(id: string): Promise<BrainSnapshotSummary[]>;
    restoreSnapshot(
      id: string,
      snapshotId: string,
      operationId?: string
    ): Promise<BrainDocument>;
    export(id: string, mode?: BrainExportMode, operationId?: string): Promise<string | null>;
    importFile(operationId?: string): Promise<BrainDocument | null>;
    onImported(listener: (brain: BrainDocument) => void): () => void;
    onImportFailed(listener: (failure: BrainImportFailure) => void): () => void;
    onBuild(listener: (event: BuildProgressEvent) => void): () => void;
    health(id?: string): Promise<EngineHealth>;
    persistedSubstrateOverview(id: string): Promise<PersistedSubstrateOverview | null>;
    querySubstrate(id: string, query?: SubstrateQuery): Promise<SubstratePage>;
    queryConceptIds(id: string, view: import("./conceptIdView").ConceptIdView, sourceTurnId: string, offset?: number): Promise<import("./conceptIdView").ConceptIdPage>;
    queryCortex(id: string, query?: CortexQuery): Promise<CortexPage>;
    cortexActivity(id: string, query: CortexActivityQuery): Promise<CortexActivity>;
    workspace(id: string): Promise<WorkspaceSnapshot>;
    freshAttention(id: string): Promise<FreshAttentionResult>;
    journalPage(
      id: string,
      cursor?: string,
      limit?: number
    ): Promise<JournalLedgerPage>;
  };
  chat: {
    send(
      id: string,
      input: string,
      turnId?: string,
      turnMetadata?: ChatTurnMetadata
    ): Promise<ChatResult>;
    cancel(id: string, turnId?: string): Promise<number>;
    cancelInlineAction(id: string, turnId: string, actionEventId: string): Promise<{
      actionEvent: ActionEvent; acknowledged: boolean;
    }>;
    recordDeliveryReceipt(
      id: string,
      receipt: ChatDeliveryReceiptRequest
    ): Promise<void>;
    approveAction(request: ApprovedChatActionRequest): Promise<ApprovedChatActionResult>;
    list(id: string): Promise<ChatMessage[]>;
    listPage(
      id: string,
      beforeSequence?: number,
      limit?: number
    ): Promise<ConversationLedgerPage>;
    feedback(request: FeedbackRequest): Promise<BrainDocument>;
    onAction(listener: (event: ActionEvent) => void): () => void;
    onStream(listener: (event: ChatStreamEvent) => void): () => void;
  };
  mobile: {
    status(): Promise<MobileGatewayStatus>;
    startPairing(request: MobileGatewayStartRequest): Promise<MobilePairingSession>;
    stop(): Promise<MobileGatewayStatus>;
    revoke(deviceId: string): Promise<MobileGatewayStatus>;
  };
  train: {
    cancel(jobId: string): Promise<RuntimeJob>;
    list(id?: string): Promise<RuntimeJob[]>;
    onEvent(listener: (event: RuntimeJobEvent) => void): () => void;
  };
  teacher: {
    status(): Promise<ApiTeacherProviderStatus[]>;
    saveCredential(request: ApiTeacherCredentialRequest): Promise<ApiTeacherProviderStatus[]>;
    removeCredential(provider: ApiTeacherProvider): Promise<ApiTeacherProviderStatus[]>;
    start(request: ApiTeacherTrainingRequest): Promise<RuntimeJob>;
    cancel(jobId: string): Promise<RuntimeJob>;
    list(brainId?: string): Promise<RuntimeJob[]>;
    onEvent(listener: (event: RuntimeJobEvent) => void): () => void;
  };
  mcp: {
    list(brainId: string): Promise<McpServerSummary[]>;
    add(request: McpServerRegistrationRequest): Promise<McpServerSummary>;
    remove(brainId: string, serverId: string): Promise<boolean>;
    refresh(brainId: string, serverId: string): Promise<McpServerSummary>;
  };
  data: {
    selectBuildResources(
      kind: BuildResourceSelection["kind"],
      selection?: import("./uploadSupport").ExperienceUploadKind
    ): Promise<BuildResourceSelection | null>;
    listBuildResources(): Promise<BuildResourceSelection[]>;
    discardBuildResource(selectionId: string): Promise<boolean>;
    startBuildResource(request: BuildResourceStartRequest): Promise<RuntimeJob>;
    preview(request: DatasetPreviewRequest): Promise<DatasetManifest | null>;
    cancelPreview(requestId: string): Promise<boolean>;
    onPreviewProgress(listener: (event: DatasetPreviewProgress) => void): () => void;
    start(request: DatasetStartRequest): Promise<RuntimeJob>;
    pause(jobId: string): Promise<RuntimeJob>;
    resume(request: DatasetStartRequest): Promise<RuntimeJob>;
    resumable(brainId: string): Promise<DatasetResumeCandidate | null>;
    coverage(brainId: string, manifestId: string): Promise<TrainingCoverage>;
    ingestFiles(request: IngestFilesRequest): Promise<IngestResult[]>;
    ingestFolder(request: IngestFilesRequest): Promise<IngestResult[]>;
    ingestDropped(request: IngestFilesRequest, files: unknown[]): Promise<IngestResult[]>;
    ingestWeb(request: IngestWebRequest): Promise<IngestResult>;
    crawlWeb(request: WebCrawlRequest): Promise<RuntimeJob>;
    cancel(jobId: string): Promise<RuntimeJob>;
    sources(
      brainId: string,
      cursor?: string,
      limit?: number
    ): Promise<TrainingSourceLedgerPage>;
  };
  modality: {
    capabilities(brainId: string): Promise<NeuralModalityCapabilities>;
    artifacts(
      brainId: string,
      cursor?: string,
      limit?: number
    ): Promise<GeneratedArtifactPage>;
    generate(request: ModalityGenerateRequest): Promise<RuntimeJob>;
    generateSpeech(request: NeuralSpeechGenerateRequest): Promise<RuntimeJob>;
    onGeneration(listener: (event: RuntimeJobEvent) => void): () => void;
    selectInput(
      request: Omit<ModalityGenerateRequest, "inputPath">
    ): Promise<RuntimeJob | null>;
    cancel(jobId: string): Promise<RuntimeJob>;
    startObservation(
      request: LiveObservationSessionStartRequest
    ): Promise<LiveObservationSession>;
    pushObservation(
      packet: LiveObservationPacket
    ): Promise<LiveObservationPacketResult>;
    stopObservation(sessionId: string): Promise<LiveObservationSession>;
    cancelObservation(sessionId: string): Promise<LiveObservationSession>;
    requestObservationControl(
      request: LiveObservationControlRequest
    ): Promise<LiveObservationControl>;
    resolveObservationControl(
      resolution: LiveObservationControlResolution
    ): Promise<LiveObservationControl>;
    onObservation(listener: (event: LiveObservationEvent) => void): () => void;
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
    cancel(brainId: string, requestId?: string): Promise<number>;
    preferences(): Promise<ToolRuntimePreferences>;
    setPreferences(value: ToolRuntimePreferences): Promise<ToolRuntimePreferences>;
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
    resourcePlan(request: WorkingMemoryPlanRequest): Promise<WorkingMemoryResourcePlan>;
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
  contextWindowTokens: 1024,
  workingMemoryMode: "auto",
  memoryOffloadBytes: 0,
  memoryResidentItems: 256,
  memoryOffloadSlowdownPercent: 0,
  systemRamMode: "auto",
  systemRamSharePercent: 0,
  storagePoolMode: "auto",
  storagePoolBytes: 0,
  storageBytesPerSecond: 0,
  learningRate: 0.14,
  traceDetail: "standard",

  retainSourceText: false,
  memoryRecipe: "adaptive-retention"
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
