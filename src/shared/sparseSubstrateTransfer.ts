/** Native sparse assemblies only. Not complete cortex/identity editing. */
export const SPARSE_SUBSTRATE_SCOPE = "native-sparse-assemblies" as const;
export const SPARSE_SUBSTRATE_FORMAT = "omni-sparse-substrate-transfer" as const;
export type SparseSubstrateKind = "neurons" | "assemblies" | "synapses";
export type SparseSubstrateImportMode = "append" | "replace-exact";
export interface SparseSubstrateSelection {
  all?: boolean;
  neurons?: string[];
  assemblies?: string[];
  synapses?: string[];
}
export interface SparseSubstrateBundle {
  format: typeof SPARSE_SUBSTRATE_FORMAT;
  version: 1;
  scope: typeof SPARSE_SUBSTRATE_SCOPE;
  id: string;
  sha256: string;
  bytes: number;
  manifestJson: string;
  manifestSha256: string;
  sourceBrainId: string;
  sourceGenerationSha256: string;
  rows: number;
  sourceBinding: "verified-owned-export" | "selected-native-shape-source-unverified";
  /** Main-owned relative custody; never an arbitrary caller-supplied path. */
  relativePath: string;
}
export interface SparseSubstratePlan {
  format: "omni-sparse-substrate-plan";
  version: 1;
  scope: typeof SPARSE_SUBSTRATE_SCOPE;
  id: string;
  brainId: string;
  bundleId: string;
  bundleSha256: string;
  mode: SparseSubstrateImportMode;
  sha256: string;
  bytes: number;
  manifestJson: string;
  manifestSha256: string;
  relativePath: string;
  rows: number;
  additions: number;
  replacements: number;
  identical: number;
  conflicts: number;
  /** Original target generation is audited; selected preimages are the CAS. */
  targetGenerationSha256: string;
  selectedPreimagesSha256: string;
  changesSha256: string;
  operation?: "import" | "rollback";
}
export interface SparseSubstrateNativeReceipt {
  format: "omni-sparse-substrate-applied";
  version: 1;
  scope: typeof SPARSE_SUBSTRATE_SCOPE;
  brainId: string;
  jobId: string;
  requestSha256: string;
  planSha256: string;
  bundleSha256: string;
  selectedPreimagesSha256: string;
  changesSha256: string;
  appliedRows: number;
  committedAt: string;
  /** SHA of exact UTF-8 receiptJson, never host-reserialized source records. */
  receiptJson: string;
  receiptSha256: string;
  generationSha256: string;
  duplicate: boolean;
  weightsAveraged: false;
  cortexParametersChanged: false;
  originChanged: false;
}
export type SparseSubstrateJobState = "queued" | "boundary-required" | "applying" | "complete" |
  "needs-review" | "cancelled" | "failed" | "uncertain" | "dormant";
export interface SparseSubstrateJob {
  format: "omni-sparse-substrate-job";
  version: 1;
  scope: typeof SPARSE_SUBSTRATE_SCOPE;
  id: string;
  brainId: string;
  originalBrainId: string;
  requestId: string;
  requestSha256: string;
  planId: string;
  planSha256: string;
  /** Host action/chat correlation captured at creation; never generated operands or authority. */
  actionEventId?: string;
  chatTurnId?: string;
  authorizationRevision: string;
  state: SparseSubstrateJobState;
  createdAt: string;
  updatedAt: string;
  revision: number;
  error?: string;
  receipt?: SparseSubstrateNativeReceipt;
  /** Historical data never carries an activation/Ask/session grant. */
  liveAuthorityPersisted: false;
}
export interface SparseSubstratePage {
  format: typeof SPARSE_SUBSTRATE_FORMAT;
  version: 1;
  scope: typeof SPARSE_SUBSTRATE_SCOPE;
  stateScope: "committed";
  brainId: string;
  generationSha256: string;
  rows: Array<{ kind: SparseSubstrateKind; id: string; recordJson: string; rowSha256: string;
    packedPayloadBase64?: string; updateCountDecimal?: string; controlsBase64?: string }>;
  hasMore: boolean;
  nextCursor?: string;
}

/** Renderer-safe receipt projection. Exact raw receiptJson remains in custody. */
export type SparseSubstrateReceiptView = Omit<SparseSubstrateNativeReceipt, "receiptJson">;

export interface SparseSubstrateJobView {
  format: "omni-sparse-substrate-job-view";
  version: 1;
  scope: typeof SPARSE_SUBSTRATE_SCOPE;
  id: string;
  brainId: string;
  originalBrainId: string;
  requestId: string;
  requestSha256: string;
  planId: string;
  planSha256: string;
  /** Historical provenance only. Renderers match inline cards only for the current owner. */
  actionEventId?: string;
  chatTurnId?: string;
  state: SparseSubstrateJobState;
  phase: "queued" | "waiting-for-loaded-boundary" | "native-apply" | "review-required" | "terminal" | "historical";
  createdAt: string;
  updatedAt: string;
  revision: number;
  error?: string;
  receipt?: SparseSubstrateReceiptView;
  /** UI fact only; never a serialized callback or authority revision. */
  liveAuthorityAvailable: false;
}

export interface SparseSubstrateJobQuery { after?: string; limit?: number }
export interface SparseSubstrateJobPage { entries: SparseSubstrateJobView[]; total: number; nextAfter?: string }

export interface SparseSubstratePlanView {
  format: "omni-sparse-substrate-plan-view";
  version: 1;
  scope: typeof SPARSE_SUBSTRATE_SCOPE;
  id: string;
  brainId: string;
  bundleId: string;
  bundleSha256: string;
  mode: SparseSubstrateImportMode;
  operation: "import" | "rollback";
  sha256: string;
  bytes: number;
  manifestSha256: string;
  rows: number;
  additions: number;
  replacements: number;
  identical: number;
  conflicts: number;
  fingerprintConflicts?: number;
  unprovenRemovalRows?: number;
  rollbackReceiptSha256?: string;
  targetGenerationSha256: string;
  selectedPreimagesSha256: string;
  changesSha256: string;
}

export type SparseSubstrateUiEvent = {
  format: "omni-sparse-substrate-ui-event";
  version: 1;
  brainId: string;
  kind: "job";
  job: SparseSubstrateJobView;
  observedAt: string;
} | {
  format: "omni-sparse-substrate-ui-event";
  version: 1;
  brainId: string;
  kind: "error";
  error: string;
  observedAt: string;
};

/** Main-to-preload envelope; the subscription identity never reaches product UI. */
export interface SparseSubstrateUiEventEnvelope {
  subscriptionId: string;
  brainId: string;
  event: Extract<SparseSubstrateUiEvent, { kind: "job" }>;
}

/** Main registers live delivery before taking this owner-validated initial page. */
export interface SparseSubstrateUiSubscriptionAck {
  brainId: string;
  subscriptionId: string;
  subscribed: true;
  initial: SparseSubstrateJobPage;
  observedAt: string;
}

export interface SparseSubstrateUiApi {
  jobs(brainId: string, query?: SparseSubstrateJobQuery): Promise<SparseSubstrateJobPage>;
  job(brainId: string, jobId: string): Promise<SparseSubstrateJobView>;
  plan(brainId: string, planId: string): Promise<SparseSubstratePlanView>;
  /** Trusted Studio control for stopping this exact existing job; never apply/rollback authority. */
  stop(brainId: string, jobId: string): Promise<SparseSubstrateJobView>;
  /** Listener is installed before main subscription; disposer is synchronous. */
  onEvent(brainId: string, listener: (event: SparseSubstrateUiEvent) => void): () => void;
}
