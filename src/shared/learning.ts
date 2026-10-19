import type { TrainingCoverage } from "./types";
import type { UserQuestionRequest, UserQuestionResolution } from "./userQuestions";

export interface WebDateRange { from?: string; through?: string }
export interface WebDomainScope { domain: string; includeSubdomains?: boolean }
export interface WebDomainDateRule extends WebDomainScope { dates: WebDateRange | null }
export interface WebLearningConstraints {
  allowedSites?: WebDomainScope[];
  excludedSites?: WebDomainScope[];
  dates?: WebDateRange;
  domainDates?: WebDomainDateRule[];
}
export interface WebLearningPolicy {
  revision: number;
  sha256: string;
  constraints: WebLearningConstraints;
  instructions: string;
  changedAt: string;
  changedBy: "user";
}
export interface WebDateEvidence {
  publishedAt?: string;
  status: "declared" | "unknown" | "conflicting";
  declarations: Array<{ value: string; source: "html-meta" | "json-ld" }>;
  /** A publisher's declaration is provenance, not an externally verified date. */
  independentlyVerified: false;
}
export interface WebSourceProvenance {
  requestedUrl: string;
  finalUrl: string;
  /** Absent when excluded before a network request. */
  fetchedAt?: string;
  checkedAt: string;
  contentType: string;
  publication: WebDateEvidence;
  policyRevision: number;
  policySha256: string;
  decision: "admitted" | "excluded-domain" | "excluded-date" | "unknown-date";
  detail?: string;
}

/** Dataset IDs come only from the existing host-owned file/folder selection. */
export type LearningInput =
  | { kind: "text"; text: string; name?: string }
  | { kind: "dataset"; manifestId: string }
  | { kind: "web"; url: string; followLinks?: boolean; followExternalLinks?: boolean; respectRobots?: boolean; quarantine?: boolean; maxPages?: number; maxDepth?: number };

export interface SpecificLearnRequest {
  brainId: string;
  requestId?: string;
  inputs: LearningInput[];
  instructions?: string;
  constraints?: WebLearningConstraints;
  epochs?: number;
  /** Optional default for web inputs only; per-source selection wins. Omission preserves ordinary learning. */
  quarantine?: boolean;
}
export type LearningTaskState = "queued" | "running" | "paused" | "stopped" | "waiting" | "complete" | "failed";
export interface LearningSourceProgress {
  id: string;
  kind: LearningInput["kind"];
  label: string;
  state: "pending" | "running" | "paused" | "complete" | "failed";
  manifestId?: string;
  crawlId?: string;
  coverage?: TrainingCoverage;
  /** Traversed with the existing web archive path, not a neural learning receipt. */
  quarantined?: boolean;
  result?: unknown;
  error?: string;
}
export interface LearningTask {
  format: "omni-learning-task";
  version: 1;
  id: string;
  brainId: string;
  requestId: string;
  requestSha256: string;
  kind: "specific-learn" | "fact-check";
  state: LearningTaskState;
  createdAt: string;
  updatedAt: string;
  revision: number;
  policy: WebLearningPolicy;
  /** Host-bound child jobs inherit live scope until the user steers them explicitly. */
  policyParentTaskId?: string;
  sources: LearningSourceProgress[];
  currentSource?: string;
  progress?: number;
  phase: string;
  /** Labels and completion are based on actual source receipts, not intelligence. */
  label: string;
  epochs: number;
  error?: string;
  pendingQuestions: string[];
  instructionReceipt?: { turnId: string; policySha256: string; committedAt: string };
  capabilityPack?: { id: string; sha256: string; receipt?: { candidateAccepted: boolean; recordsVisited: number; parameterSteps: number;
    protectedOwners: string[]; sha256: string; [key: string]: unknown } };
  resumeBlocked?: { reason: "neural-generation-mismatch" | "retained-native-frontier-unavailable"; sourceCoreSha256: string };
  /** Only lifecycle unloading may resume this task automatically on explicit Load. */
  pausedForUnload?: boolean;
  /** Exact admitted task turn which must resume before any newer work. */
  retainedNativeTurn?: { turnId: string; inputSha256: string; policySha256?: string };
  inheritedFrom?: { parentId: string; taskId: string; requestId: string; requestSha256: string; originalTurnId?: string; executionId?: string };
  /** True only after every selected source or explicitly scoped claim is settled. */
  traversalComplete: boolean;
}
export interface LearningTaskEvent { type: "created" | "updated"; task: LearningTask }
export interface LearningTaskSteerRequest {
  brainId: string;
  taskId: string;
  expectedRevision: number;
  instructions?: string;
  constraints?: WebLearningConstraints;
}

export interface FactCheckClaimInput { text: string; sources?: string[] }
export interface FactCheckRequest {
  brainId: string;
  requestId?: string;
  claims: FactCheckClaimInput[];
  instructions?: string;
  constraints?: WebLearningConstraints;
}
export interface FactCheckEvidence {
  id: string;
  claimId: string;
  url: string;
  sourceId?: string;
  contentSha256?: string;
  publication?: WebDateEvidence;
  fetchedAt: string;
  toolReceiptId?: string;
  toolId?: "web.fetch" | "web.search";
  requestedUrl?: string;
  policyRevision?: number;
  policySha256?: string;
  /** A search result is discovery evidence, not a capture of its destination. */
  capture?: "learned-source" | "fetched-content" | "search-result";
  /** Evidence can contradict another source. It is not a truth lookup table. */
  relation: "unreviewed" | "supports" | "contradicts" | "context";
}
export interface FactCheckFinding {
  id: string;
  claimId: string;
  conclusion: "supported" | "contradicted" | "mixed" | "undetermined";
  explanation: string;
  evidenceIds: string[];
  source: "model" | "user";
  createdAt: string;
  /** Model explanations are self-reports, not guaranteed reasoning or facts. */
  verifiedTruth: false;
  proposedCorrection?: string;
  /** Durable task question declaration for recovery after a lost acknowledgement. */
  question?: Omit<UserQuestionRequest, "brainId" | "requestId" | "source">;
}
export interface FactCheckClaim {
  id: string;
  text: string;
  sources: string[];
  state: "pending" | "collecting" | "reviewable" | "waiting-answer" | "reviewed";
  evidence: FactCheckEvidence[];
  findings: FactCheckFinding[];
  questionId?: string;
  answer?: UserQuestionResolution;
  correctionTaskId?: string;
  reviewError?: string;
  /** Durable current-task admission; not a truth/competence receipt. */
  assessmentReceipt?: { turnId: string; policySha256: string; committedAt: string };
}
export interface FactCheckWorkspace {
  task: LearningTask;
  claims: FactCheckClaim[];
  scope: "selected-claims-only";
  completeKnowledgeInventory: false;
}
export interface FactCheckReviewRequest {
  brainId: string; taskId: string; claimId: string; requestId: string;
  conclusion: FactCheckFinding["conclusion"];
  explanation: string;
  evidenceIds: string[];
  proposedCorrection?: string;
  question?: Omit<UserQuestionRequest, "brainId" | "requestId" | "source">;
}
export interface LearningApi {
  start(request: SpecificLearnRequest): Promise<LearningTask>;
  startFactCheck(request: FactCheckRequest): Promise<FactCheckWorkspace>;
  list(brainId: string): Promise<LearningTask[]>;
  get(brainId: string, taskId: string): Promise<LearningTask>;
  factCheck(brainId: string, taskId: string): Promise<FactCheckWorkspace>;
  pause(brainId: string, taskId: string): Promise<LearningTask>;
  resume(brainId: string, taskId: string): Promise<LearningTask>;
  stop(brainId: string, taskId: string): Promise<LearningTask>;
  steer(request: LearningTaskSteerRequest): Promise<LearningTask>;
  review(request: FactCheckReviewRequest): Promise<FactCheckWorkspace>;
  acceptCorrection(brainId: string, taskId: string, claimId: string, findingId: string): Promise<LearningTask>;
  onEvent(listener: (event: LearningTaskEvent) => void): () => void;
}
