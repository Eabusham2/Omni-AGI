/** External local teacher data, never an alternative Omni runtime/identity. */
export type LocalTeacherRunnerKind = "llama-server" | "llama-completion" | "llama-cli";
export interface LocalTeacherGgufHeader {
  version: 2 | 3;
  tensorCount: string;
  metadataCount: string;
  metadataEnd: number;
  architecture?: string;
  name?: string;
  declaredLicense?: string;
  declaredLicenseLink?: string;
  geometry: Record<string, number>;
  scope: "header-metadata-only";
  tensorPayloadValidated: false;
}
export interface LocalTeacherFileIdentity { displayName: string; sha256: string; bytes: number }
export interface LocalTeacherSelection {
  format: "omni-local-gguf-teacher";
  version: 1;
  id: string;
  brainId: string;
  originalBrainId: string;
  originalSelectionId: string;
  createdAt: string;
  runner: LocalTeacherRunnerKind;
  runtime: LocalTeacherFileIdentity & { helpSha256: string; rawOfflineSupported: boolean; missingFlags: string[] };
  model: LocalTeacherFileIdentity & { header: LocalTeacherGgufHeader };
  license: { declaration: string | null; source: "user-declared" | "gguf-metadata" | "not-declared"; verified: false };
  /** A copied path/record is not permission to execute it. */
  currentHostAuthorized: boolean;
}
export interface LocalTeacherSelectRequest {
  brainId: string;
  requestId: string;
  runner: LocalTeacherRunnerKind;
  licenseDeclaration?: string;
  /** Reapproval still requires actual current main-owned file selections. */
  teacherId?: string;
}
export interface LocalTeacherGeneration {
  contextTokens: number;
  maxOutputTokens: number;
  threads: number;
  temperature: number;
  seed: number;
  timeoutMs: number;
  /** Explicit resource/output envelope, never silent truncation. */
  maxOutputBytes: number;
}
export interface LocalTeacherStartRequest {
  brainId: string;
  requestId: string;
  teacherId: string;
  prompts: string[];
  generation: LocalTeacherGeneration;
  epochs?: number;
}
export type LocalTeacherJobState = "queued" | "running" | "paused" | "stopped" | "waiting-load" | "complete" | "failed" | "dormant";
export interface LocalTeacherJob {
  format: "omni-local-teacher-job";
  version: 1;
  id: string;
  brainId: string;
  originalBrainId: string;
  originalJobId: string;
  requestId: string;
  requestSha256: string;
  teacherId: string;
  generation: LocalTeacherGeneration;
  epochs: number;
  state: LocalTeacherJobState;
  revision: number;
  createdAt: string;
  updatedAt: string;
  promptCount: number;
  generatedCount: number;
  learnedCount: number;
  activeIndex?: number;
  phase: string;
  error?: string;
  priorState?: LocalTeacherJobState;
  dormantReason?: "copied-instance" | "merged-history" | "recovery-history";
  /** Counts attest source cursors, never comprehension or teacher correctness. */
  proof: "attributed-generation-and-native-source-receipts";
  manifestId?: string;
  manifestSha256?: string;
  learningTaskId?: string;
  learningRequestId?: string;
  coverage?: import("./types").TrainingCoverage;
}
export interface LocalTeacherResultReceipt {
  format: "omni-local-teacher-result";
  version: 1;
  attemptId: string;
  jobId: string;
  index: number;
  inputSha256: string;
  runtimeSha256: string;
  modelSha256: string;
  argvSha256: string;
  environmentPolicySha256: string;
  exitCode: number | null;
  /** HTTP completion is not a successful process exit of a long-lived server. */
  completionProof?: "http-terminal" | "process-exit";
  state: "complete" | "interrupted" | "failed";
  stdout: { relativePath: string; sha256: string; bytes: number };
  stderr: { relativePath: string; sha256: string; bytes: number };
  elapsedNanoseconds: string;
  outputChannel: "structured-completion" | "teacher-console";
  transportDecorationsPreserved: boolean;
  rawTransport?: { relativePath: string; sha256: string; bytes: number };
  sessionId?: string;
  hiddenSystemPrompt: false;
  foreignWeightsImported: false;
  error?: string;
}
export interface LocalTeacherItem {
  jobId: string;
  index: number;
  prompt: string;
  inputSha256: string;
  state: "pending" | "generating" | "generated" | "learning" | "complete" | "interrupted";
  attemptId?: string;
  receipt?: LocalTeacherResultReceipt;
  source?: { relativePath: string; sha256: string; bytes: number };
  manifestId?: string;
  manifestSha256?: string;
  learningTaskId?: string;
  learningRequestId?: string;
  coverage?: import("./types").TrainingCoverage;
}
export interface LocalTeacherEvent { brainId: string; job: LocalTeacherJob }
export interface LocalTeacherApi {
  select(request: LocalTeacherSelectRequest): Promise<LocalTeacherSelection | null>;
  cancelSelection(brainId: string, requestId: string): Promise<boolean>;
  selectionProgress(brainId: string, requestId: string): Promise<LocalTeacherSelectionProgress | null>;
  onSelectionProgress(listener: (event: LocalTeacherSelectionProgress) => void): () => void;
  plan(request: { brainId: string; teacherId: string; prompts?: string[] }): Promise<LocalTeacherPlan>;
  selections(brainId: string): Promise<LocalTeacherSelection[]>;
  start(request: LocalTeacherStartRequest): Promise<LocalTeacherJob>;
  jobs(brainId: string): Promise<LocalTeacherJob[]>;
  get(brainId: string, jobId: string): Promise<LocalTeacherJob>;
  items(brainId: string, jobId: string, afterIndex?: number, limit?: number): Promise<LocalTeacherItem[]>;
  pause(brainId: string, jobId: string): Promise<LocalTeacherJob>;
  resume(brainId: string, jobId: string): Promise<LocalTeacherJob>;
  stop(brainId: string, jobId: string): Promise<LocalTeacherJob>;
  onEvent(listener: (event: LocalTeacherEvent) => void): () => void;
}
export interface LocalTeacherSelectionProgress {
  brainId: string;
  requestId: string;
  phase: "hashing-runtime" | "hashing-model" | "reading-header" | "probing-runtime" | "complete" | "cancelled" | "failed";
  fileName?: string;
  bytes: number;
  totalBytes: number;
  error?: string;
}
export interface LocalTeacherPlan {
  brainId: string;
  teacherId: string;
  recommendation: LocalTeacherGeneration | null;
  fit: "estimated" | "unknown" | "blocked";
  reservationBytes: number | null;
  measured: { appFamilyResidentBytes: number; selectedRamCeilingBytes: number; physicalFreeBytes: number; logicalCpuCount: number };
  declared: { modelBytes: number; kvBytesPerToken: number | null; contextLimit: number | null; tensorPayloadValidated: false };
  blockers: string[];
  warnings: string[];
}
