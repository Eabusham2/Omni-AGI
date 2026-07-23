export const BRAIN_SCHEMA_VERSION = 1;

export type ArchitecturePreset =
  | "whole-brain"
  | "ternary"
  | "neuromorphic"
  | "liquid"
  | "symbolic"
  | "custom";

export type InferenceRuntime = "adaptive-core";
export type MemoryInjectionMode = "parameter-only" | "working-memory";
export type MemoryRecipe = "human-consolidation" | "total-recall" | "synapses-only";
export type GrowthPolicy = "fixed" | "elastic" | "unbounded";
export type TraceDetail = "summary" | "standard" | "research";

export interface BrainConfig {
  name: string;
  preset: ArchitecturePreset;
  runtime: InferenceRuntime;
  description: string;

  ternaryWeights: boolean;
  spikingDynamics: boolean;
  stdpPlasticity: boolean;
  liquidDynamics: boolean;
  liquidMode: "cfc" | "ltc";
  vectorSymbolicMemory: boolean;
  onlineLearning: boolean;
  consolidation: boolean;
  metaplasticity: boolean;

  workingMemorySlots: number;
  shortTermHalfLifeMinutes: number;
  longTermThreshold: number;
  initialNeuronBudget: number;
  growthPolicy: GrowthPolicy;
  maxConcepts: number;
  maxSynapses: number;

  learningRate: number;
  noise: number;
  firingThreshold: number;
  membraneLeak: number;
  stdpWindow: number;
  consolidationRate: number;
  forgettingRate: number;

  noveltyDrive: number;
  coherenceDrive: number;
  curiosityDrive: number;
  parallelThoughts: number;
  traceDetail: TraceDetail;

  storeAtomicIdeas: boolean;
  retainSourceText: boolean;
  learnFromOwnMessages: boolean;
  memoryInjection: MemoryInjectionMode;
  memoryRecipe?: MemoryRecipe;

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
}

export interface IngestResult {
  brain: BrainDocument;
  source: TrainingSource;
  warnings: string[];
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

export interface IngestFilesRequest {
  brainId: string;
  policy?: DataIngestionPolicy;
}

export interface IngestWebRequest {
  brainId: string;
  url: string;
  policy?: DataIngestionPolicy;
  quarantine?: boolean;
}

export interface WebCrawlRequest extends IngestWebRequest {
  maxPages?: number;
  maxDepth?: number;
  sameOrigin?: boolean;
  respectRobots?: boolean;
}

export interface WebCrawlResult {
  startUrl: string;
  visited: number;
  skipped: number;
  results: IngestResult[];
  warnings: string[];
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
  kind: "learning" | "consolidation" | "tool" | "fork" | "system";
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

export interface AgentMergePreview {
  sourceBrainId: string;
  targetBrainId: string;
  reviewToken: string;
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
  worker: "python" | "fallback";
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

export interface OmniApi {
  window: {
    minimize(): Promise<void>;
    maximize(): Promise<void>;
    close(): Promise<void>;
    isMaximized(): Promise<boolean>;
    openExternal(url: string): Promise<void>;
    revealDataFolder(): Promise<void>;
    platform(): Promise<string>;
  };
  brain: {
    list(): Promise<BrainSummary[]>;
    get(id: string): Promise<BrainDocument>;
    create(request: CreateBrainRequest): Promise<BrainDocument>;
    update(id: string, config: BrainConfig): Promise<BrainDocument>;
    fork(id: string, name?: string): Promise<BrainDocument>;
    remove(id: string): Promise<boolean>;
    snapshot(id: string, label?: string): Promise<BrainSnapshotSummary>;
    listSnapshots(id: string): Promise<BrainSnapshotSummary[]>;
    restoreSnapshot(id: string, snapshotId: string): Promise<BrainDocument>;
    export(id: string, mode?: BrainExportMode): Promise<string | null>;
    importFile(): Promise<BrainDocument | null>;
    onImported(listener: (brain: BrainDocument) => void): () => void;
    health(id?: string): Promise<EngineHealth>;
  };
  chat: {
    send(id: string, input: string): Promise<ChatResult>;
    list(id: string): Promise<ChatMessage[]>;
    feedback(request: FeedbackRequest): Promise<BrainDocument>;
  };
  train: {
    start(request: StartTrainingRequest): Promise<RuntimeJob>;
    consolidate(id: string): Promise<BrainDocument>;
    cancel(jobId: string): Promise<RuntimeJob>;
    list(id?: string): Promise<RuntimeJob[]>;
    onEvent(listener: (event: RuntimeJobEvent) => void): () => void;
  };
  data: {
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

  /** @deprecated Compatibility alias for the first renderer scaffold. */
  app: {
    minimize(): Promise<void>;
    maximize(): Promise<void>;
    close(): Promise<void>;
    isMaximized(): Promise<boolean>;
    openExternal(url: string): Promise<void>;
    revealDataFolder(): Promise<void>;
    platform(): Promise<string>;
  };
  /** @deprecated Compatibility alias for the first renderer scaffold. */
  brains: {
    list(): Promise<BrainSummary[]>;
    get(id: string): Promise<BrainDocument>;
    create(request: CreateBrainRequest): Promise<BrainDocument>;
    updateConfig(id: string, config: BrainConfig): Promise<BrainDocument>;
    chat(id: string, input: string): Promise<ChatResult>;
    feedback(request: FeedbackRequest): Promise<BrainDocument>;
    consolidate(id: string): Promise<BrainDocument>;
    fork(id: string, name?: string): Promise<BrainDocument>;
    remove(id: string): Promise<boolean>;
    export(id: string): Promise<string | null>;
    importFile(): Promise<BrainDocument | null>;
    importUrl(request: ImportUrlRequest): Promise<BrainDocument>;
    ingestFiles(id: string): Promise<IngestResult[]>;
    runtimeHealth(id: string): Promise<RuntimeHealth>;
  };
}

export const DEFAULT_CONFIG: BrainConfig = {
  name: "New mind",
  preset: "whole-brain",
  runtime: "adaptive-core",
  description: "A persistent local mind that grows an associative concept graph as it learns.",

  ternaryWeights: true,
  spikingDynamics: true,
  stdpPlasticity: true,
  liquidDynamics: true,
  liquidMode: "cfc",
  vectorSymbolicMemory: true,
  onlineLearning: true,
  consolidation: true,
  metaplasticity: true,

  workingMemorySlots: 24,
  shortTermHalfLifeMinutes: 45,
  longTermThreshold: 0.62,
  initialNeuronBudget: 2048,
  growthPolicy: "elastic",
  maxConcepts: 100_000,
  maxSynapses: 1_000_000,

  learningRate: 0.14,
  noise: 0.08,
  firingThreshold: 0.56,
  membraneLeak: 0.82,
  stdpWindow: 8,
  consolidationRate: 0.06,
  forgettingRate: 0.002,

  noveltyDrive: 0.72,
  coherenceDrive: 0.88,
  curiosityDrive: 0.58,
  parallelThoughts: 3,
  traceDetail: "standard",

  storeAtomicIdeas: true,
  retainSourceText: false,
  learnFromOwnMessages: true,
  memoryInjection: "parameter-only",
  memoryRecipe: "human-consolidation",

};

export const PRESET_CONFIGS: Record<ArchitecturePreset, Partial<BrainConfig>> = {
  "whole-brain": {},
  ternary: {
    spikingDynamics: false,
    stdpPlasticity: false,
    liquidDynamics: false,
    vectorSymbolicMemory: true,
    noise: 0.03,
    description: "A compact associative network with effective weights constrained to −1, 0, or +1."
  },
  neuromorphic: {
    ternaryWeights: false,
    spikingDynamics: true,
    stdpPlasticity: true,
    liquidDynamics: false,
    vectorSymbolicMemory: false,
    firingThreshold: 0.5,
    noise: 0.11,
    description: "Leaky integrate-and-fire dynamics with local spike-timing plasticity."
  },
  liquid: {
    ternaryWeights: false,
    spikingDynamics: false,
    stdpPlasticity: false,
    liquidDynamics: true,
    vectorSymbolicMemory: false,
    memoryInjection: "working-memory",
    description: "Continuous recurrent state with input-dependent time constants."
  },
  symbolic: {
    ternaryWeights: true,
    spikingDynamics: false,
    stdpPlasticity: false,
    liquidDynamics: false,
    vectorSymbolicMemory: true,
    noise: 0.02,
    parallelThoughts: 1,
    description: "Idea-first memory using compositional hypervectors and an inspectable concept graph."
  },
  custom: {
    description: "A fully configurable cortex recipe."
  }
};

export function createPresetConfig(
  preset: ArchitecturePreset,
  name = DEFAULT_CONFIG.name
): BrainConfig {
  return {
    ...DEFAULT_CONFIG,
    ...PRESET_CONFIGS[preset],
    name,
    preset
  };
}
