import type { DeviceResourceTopology } from "./deviceResources";

/** This is one pure decoder operation, NOT retained-KV/chat/media performance. */
export const DECODER_EXECUTION_WORK_KIND = "native.decoder.prefill-next-logits-isolated-cache" as const;
export type DecoderExecutionPhase = "idle" | "awaiting-normal-input" | "captured" | "prepared" |
  "calibrating" | "proposal" | "applied" | "cancelled" | "failed" | "unsupported";

export interface DecoderExecutionCandidate {
  planSha256: string;
  executionDevice: string;
  deviceId: string;
  memoryGroupId: string;
  backendMode: "native-cpu" | "native-metal" | "explicit-cuda-cpu-tile";
  /** Exact async compute + home-output transfer participants, not extra compute routes. */
  asyncDevices: string[];
  supported: boolean;
  fallbackAcknowledged: boolean;
}

/** Generated and rechecked by the loaded owner; renderer fields are never authority. */
export interface DecoderExecutionBinding {
  captureId: string;
  ownerSessionId: string;
  policySha256: string;
  topologySha256: string;
  registrySha256: string;
  nativeStateSha256: string;
  workloadSha256: string;
  geometrySha256: string;
  budgetSha256: string;
}

export interface DecoderExecutionProgress {
  candidate: number;
  candidates: number;
  sample: number;
  samplesPerCandidate: number;
  stage: "warming" | "measuring" | "selecting" | "restoring";
}

export interface DecoderExecutionProposal {
  proposalId: string;
  proposalSha256: string;
  workKind: typeof DECODER_EXECUTION_WORK_KIND;
  candidate: DecoderExecutionCandidate;
  comparisonSha256: string;
  outputContractSha256: string;
  baselineMeanNanoseconds: string;
  candidateMeanNanoseconds: string;
  minimumSamples: number;
  /** Exact input/state/geometry only; not a saved/global placement preference. */
  scope: "exact-isolated-prefill-operation";
}

export interface DecoderExecutionStatus {
  format: "omni-decoder-execution-status";
  version: 1;
  ownerId: string;
  ownerSessionId: string;
  phase: DecoderExecutionPhase;
  operationId: string | null;
  captureId: string | null;
  reason: string;
  workKind: typeof DECODER_EXECUTION_WORK_KIND;
  progress: DecoderExecutionProgress | null;
  proposal: DecoderExecutionProposal | null;
  measuredOperations: number;
  adoptedOperationCalls: number;
  cancellationRequested: boolean;
  /** Both remain false until separately implemented and measured. */
  retainedKvContinuationMeasured: false;
  connectedDecoderSupported: false;
  wholeBrainOrMediaMeasured: false;
  globalPlacementChanged: false;
}

export interface DecoderExecutionReview {
  format: "omni-decoder-execution-review";
  version: 1;
  ownerId: string;
  binding: DecoderExecutionBinding;
  workKind: typeof DECODER_EXECUTION_WORK_KIND;
  stageCoverage: string[];
  currentCandidate: DecoderExecutionCandidate;
  candidates: DecoderExecutionCandidate[];
  inputShape: number[];
  cueShape: number[] | null;
  minimumSamples: number;
  status: DecoderExecutionStatus;
}

/** Main-only worker parameters. A topology must be freshly host-observed. */
export interface DecoderExecutionPrepareParams { brainId: string; topology?: DeviceResourceTopology }
export interface DecoderExecutionCalibrateParams {
  brainId: string;
  /** Main-seeded identity; flags-only Stop never cancels a different call/chat. */
  operationId: string;
  reviewed: DecoderExecutionBinding;
  samplesPerCandidate?: number;
}
export interface DecoderExecutionApplyParams { brainId: string; proposalId: string; proposalSha256: string }

export interface DecoderExecutionCalibration {
  format: "omni-decoder-execution-calibration";
  version: 1;
  ownerId: string;
  operationId: string;
  binding: DecoderExecutionBinding;
  status: DecoderExecutionStatus;
  proposal: DecoderExecutionProposal | null;
  receiptEventIds: string[];
  outputEquivalence: {
    checked: boolean;
    relativeTolerance: number;
    absoluteTolerance: number;
    noSamplingOrRngDraws: true;
    maximumAbsoluteLogitError: number | null;
    maximumAbsoluteStateError: number | null;
  };
}

export interface DecoderExecutionPrepared extends DecoderExecutionReview { reviewId: string }
export interface DecoderExecutionApi {
  captureNext(request: { brainId: string; acknowledgeCudaCpuTileFallback?: true }): Promise<DecoderExecutionStatus>;
  prepare(request: { brainId: string }): Promise<DecoderExecutionPrepared>;
  calibrate(request: { brainId: string; reviewId: string; samplesPerCandidate?: number }): Promise<DecoderExecutionCalibration>;
  apply(request: DecoderExecutionApplyParams): Promise<DecoderExecutionStatus>;
  status(brainId: string): Promise<DecoderExecutionStatus>;
  stop(request: { brainId: string; operationId?: string }): Promise<DecoderExecutionStatus>;
}
