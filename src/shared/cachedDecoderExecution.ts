import type { DecoderExecutionCandidate, DecoderExecutionProgress } from "./decoderExecution";

export const CACHED_DECODER_EXECUTION_WORK_KIND = "native.decoder.cached-next-token-sampling" as const;
export interface CachedDecoderExecutionBinding {
  ownerSessionId: string;
  policySha256: string;
  topologySha256: string;
  registrySha256: string;
  budgetSha256: string;
  homeExecutionDevice: string;
}
export interface CachedDecoderExecutionStatus {
  format: "omni-cached-decoder-execution-status";
  version: 1;
  ownerId: string;
  ownerSessionId: string;
  phase: "idle" | "prepared" | "awaiting-retained-frontier" | "calibrating" | "completed" | "cancelled" | "failed-fallback" | "unsupported";
  workKind: typeof CACHED_DECODER_EXECUTION_WORK_KIND;
  operationId: string | null;
  reason: string;
  progress: DecoderExecutionProgress | null;
  cancellationRequested: boolean;
  selectedCandidate: DecoderExecutionCandidate | null;
  actualOperationConsumed: boolean;
  exactCacheAndSamplingMatched: boolean;
  globalPlacementChanged: false;
  connectedDecoderSupported: false;
  wholeBrainOrMediaMeasured: false;
  reusableAcrossWorkloads: false;
}
export interface CachedDecoderExecutionReview {
  format: "omni-cached-decoder-execution-review";
  version: 1;
  ownerId: string;
  binding: CachedDecoderExecutionBinding;
  workKind: typeof CACHED_DECODER_EXECUTION_WORK_KIND;
  currentCandidate: DecoderExecutionCandidate;
  candidates: DecoderExecutionCandidate[];
  minimumSamples: 3;
  stageCoverage: string[];
  status: CachedDecoderExecutionStatus;
}
export interface CachedDecoderExecutionPrepared extends CachedDecoderExecutionReview { reviewId: string }
export interface CachedDecoderExecutionApi {
  prepare(request: { brainId: string; acknowledgeCudaCpuTileFallback?: true }): Promise<CachedDecoderExecutionPrepared>;
  /** Explicitly authorizes one inline comparison at the next real valid KV frontier. */
  arm(request: { brainId: string; reviewId: string; samplesPerCandidate?: number }): Promise<CachedDecoderExecutionStatus>;
  status(brainId: string): Promise<CachedDecoderExecutionStatus>;
  stop(request: { brainId: string; operationId?: string }): Promise<CachedDecoderExecutionStatus>;
}
