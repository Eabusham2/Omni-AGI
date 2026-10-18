import type { DecoderExecutionCandidate, DecoderExecutionProgress } from "./decoderExecution";
import type { NativeOperationAutoCandidate } from "./nativeMixedRegions";

export const NATIVE_OPERATION_AUTO_WORK_KINDS = [
  "native.perception.image-embedding",
  "native.perception.audio-embedding",
  "native.perception.video-embedding",
  "native.decoder.cached-whole-step-sampling",
  "native.connected-decoder.cached-whole-step-sampling",
  "native.media.image-artifact",
  "native.media.audio-artifact",
  "native.media.video-artifact",
  "native.training.prepared-whole-wave",
  "native.decoder.full-prefix-sampling",
  "native.connected-decoder.full-prefix-sampling",
  "native.decoder.uncached-full-forward",
  "native.connected-decoder.uncached-full-forward"
] as const;
export type NativeOperationAutoWorkKind = typeof NATIVE_OPERATION_AUTO_WORK_KINDS[number];
export const NATIVE_OPERATION_AUTO_STAGES = [
  "operation-owner-admission", "input-activity-transfer", "complete-perception-embedding",
  "output-home-transfer", "original-owner-restore"
] as const;
export const NATIVE_CACHED_WHOLE_OPERATION_STAGES = [
  "frontier-cow-owner-admission", "input-cue-cache-activity-transfer",
  "complete-causal-workspace-conditioning", "all-decoder-regions-recurrent-exchange",
  "experts-norm-language-readout", "cache-neural-output-home-return",
  "original-owner-restore", "same-home-sampling"
] as const;
export const NATIVE_MEDIA_OPERATION_AUTO_STAGES = [
  "prepared-media-owner-admission", "home-RNG-and-activity-transfer", "all-native-media-regions",
  "overlap-composition-and-preview", "deterministic-codec-artifact-checksum", "original-owner-restore"
] as const;
export const NATIVE_TRAINING_WAVE_OPERATION_AUTO_STAGES = [
  "prepared-wave-owner-and-policy-admission", "private-native-and-graph-COW",
  "same-home-RNG-input-and-activity-transfer", "all-region-forward-backward-and-native-update",
  "exact-complete-wave-state-and-effects-proof", "original-owner-and-RNG-fence"
] as const;
export const NATIVE_PREFILL_OPERATION_AUTO_STAGES = [
  "prefix-owner-and-policy-admission", "input-cue-and-empty-cache-activity",
  "complete-prefix-causal-workspace", "all-decoder-regions-recurrent-exchange", "experts-norm-language-readout",
  "new-cache-neural-output-home-return", "original-owner-restore", "same-home-sampling"
] as const;
export const NATIVE_FORWARD_OPERATION_AUTO_STAGES = [
  "forward-owner-and-policy-admission", "input-cue-mask-activity-transfer",
  "complete-causal-and-requested-whole-workspace", "all-decoder-connected-recurrent-exchange",
  "experts-norm-language-action-readout", "all-outputs-and-activity-home-return", "original-owner-restore"
] as const;
export const NATIVE_OPERATION_AUTO_STAGE_COVERAGE = {
  "native.perception.image-embedding": NATIVE_OPERATION_AUTO_STAGES,
  "native.perception.audio-embedding": NATIVE_OPERATION_AUTO_STAGES,
  "native.perception.video-embedding": NATIVE_OPERATION_AUTO_STAGES,
  "native.decoder.cached-whole-step-sampling": NATIVE_CACHED_WHOLE_OPERATION_STAGES,
  "native.connected-decoder.cached-whole-step-sampling": NATIVE_CACHED_WHOLE_OPERATION_STAGES,
  "native.media.image-artifact": NATIVE_MEDIA_OPERATION_AUTO_STAGES,
  "native.media.audio-artifact": NATIVE_MEDIA_OPERATION_AUTO_STAGES,
  "native.media.video-artifact": NATIVE_MEDIA_OPERATION_AUTO_STAGES,
  "native.training.prepared-whole-wave": NATIVE_TRAINING_WAVE_OPERATION_AUTO_STAGES,
  "native.decoder.full-prefix-sampling": NATIVE_PREFILL_OPERATION_AUTO_STAGES,
  "native.connected-decoder.full-prefix-sampling": NATIVE_PREFILL_OPERATION_AUTO_STAGES,
  "native.decoder.uncached-full-forward": NATIVE_FORWARD_OPERATION_AUTO_STAGES,
  "native.connected-decoder.uncached-full-forward": NATIVE_FORWARD_OPERATION_AUTO_STAGES
} satisfies Record<NativeOperationAutoWorkKind, readonly string[]>;
export interface NativeOperationAutoCoverage { workKind: NativeOperationAutoWorkKind; stages: string[] }
export interface NativeOperationAutoContextRange {
  minimumTokens: number;
  maximumTokens: number;
  queryTokens: 1;
  basis: "actual-pager-and-attention-tile-transitions";
  layoutSha256: string;
}
export type NativeOperationAutoMediaModality = "image" | "audio" | "video";
export interface NativeOperationAutoMediaLayout<Modality extends NativeOperationAutoMediaModality = NativeOperationAutoMediaModality> {
  modality: Modality;
  shapeSha256: string;
}
export interface NativeOperationAutoTrainingLayout { shapeSha256: string }
export interface NativeOperationAutoPrefillLayout { shapeSha256: string }
export interface NativeOperationAutoForwardLayout { shapeSha256: string }

export interface NativeOperationAutoBinding {
  ownerSessionId: string;
  policySha256: string;
  topologySha256: string;
  registrySha256: string;
  budgetSha256: string;
  homeExecutionDevice: string;
}
interface NativeOperationAutoPlanBase {
  id: string;
  geometrySha256: string;
  qualifiedNativeStateSha256: string;
  candidate: NativeOperationAutoCandidate;
  qualifiedSamples: number;
  qualificationWorkloads: 1;
  unseenInputOrFrontierWasPreviouslyMeasured: false;
  wholeBrainOrTrainingCovered: false;
}
type NativeOperationAutoMediaPlan = {
  [Modality in NativeOperationAutoMediaModality]: {
    workKind: `native.media.${Modality}-artifact`;
    numericGate: "exact-media-neural-timeline-rng-and-encoded-artifact-equality";
    reuseBasis: "same-validated-media-operator-output-layout-current-owner-policy-inference";
    mediaLayout: NativeOperationAutoMediaLayout<Modality>;
    contextRange?: never;
    trainingLayout?: never;
    prefillLayout?: never;
    forwardLayout?: never;
  }
}[NativeOperationAutoMediaModality];
export type NativeOperationAutoPlan = NativeOperationAutoPlanBase & ({
  workKind: "native.perception.image-embedding" | "native.perception.audio-embedding" | "native.perception.video-embedding";
  numericGate: "exact-embedding-equality";
  reuseBasis: "same-validated-operator-geometry-current-owner-policy-inference";
  contextRange?: never;
  mediaLayout?: never;
  trainingLayout?: never;
  prefillLayout?: never;
  forwardLayout?: never;
} | {
  workKind: "native.decoder.cached-whole-step-sampling" | "native.connected-decoder.cached-whole-step-sampling";
  numericGate: "exact-cache-connected-activity-neural-sampling-equality";
  reuseBasis: "same-validated-operator-page-layout-range-current-owner-policy-inference";
  contextRange: NativeOperationAutoContextRange;
  mediaLayout?: never;
  trainingLayout?: never;
  prefillLayout?: never;
  forwardLayout?: never;
} | NativeOperationAutoMediaPlan | {
  workKind: "native.training.prepared-whole-wave";
  numericGate: "exact-prepared-wave-native-state-rng-and-effects-equality";
  reuseBasis: "same-prepared-wave-current-owner-policy-no-cross-wave-inference";
  trainingLayout: NativeOperationAutoTrainingLayout;
  contextRange?: never;
  mediaLayout?: never;
  prefillLayout?: never;
  forwardLayout?: never;
} | {
  workKind: "native.decoder.full-prefix-sampling" | "native.connected-decoder.full-prefix-sampling";
  numericGate: "exact-prefix-cache-connected-activity-neural-sampling-equality";
  reuseBasis: "same-prepared-prefix-current-owner-policy-no-cross-prefix-inference";
  prefillLayout: NativeOperationAutoPrefillLayout;
  contextRange?: never;
  mediaLayout?: never;
  trainingLayout?: never;
  forwardLayout?: never;
} | {
  workKind: "native.decoder.uncached-full-forward" | "native.connected-decoder.uncached-full-forward";
  numericGate: "exact-full-forward-output-workspace-action-activity-equality";
  reuseBasis: "same-prepared-full-forward-current-owner-policy-no-cross-input-inference";
  forwardLayout: NativeOperationAutoForwardLayout;
  contextRange?: never;
  mediaLayout?: never;
  trainingLayout?: never;
  prefillLayout?: never;
});
export interface NativeOperationAutoStatus {
  format: "omni-native-operation-auto-status";
  version: 2;
  ownerId: string;
  ownerSessionId: string;
  phase: "idle" | "prepared" | "awaiting-operation" | "qualifying" | "active" | "cancelled" | "failed-fallback" | "unsupported";
  operationId: string | null;
  reason: string;
  progress: DecoderExecutionProgress | null;
  cancellationRequested: boolean;
  workKind: NativeOperationAutoWorkKind | null;
  qualifiedPlanCount: number;
  lastPlan: NativeOperationAutoPlan | null;
  supportedWorkKinds: NativeOperationAutoWorkKind[];
  unsupportedRoutes: string[];
  globalPlacementChanged: false;
  calibrationOnAppOpen: false;
  wholeBrainOrTrainingMeasured: false;
  qualificationIsExact: true;
  reusableRouteIsInferenceForUnseenInputs: true;
}
export interface NativeOperationAutoReview {
  format: "omni-native-operation-auto-review";
  version: 2;
  ownerId: string;
  binding: NativeOperationAutoBinding;
  currentCandidate: DecoderExecutionCandidate;
  candidates: DecoderExecutionCandidate[];
  minimumSamples: 3;
  stageCoverageByWorkKind: NativeOperationAutoCoverage[];
  supportedWorkKinds: NativeOperationAutoWorkKind[];
  connectedPerceptionSupported: boolean;
  status: NativeOperationAutoStatus;
}
export interface NativeOperationAutoPrepared extends NativeOperationAutoReview { reviewId: string }
export interface NativeOperationAutoPlanPage {
  ownerId: string;
  ownerSessionId: string;
  revision: number;
  plans: NativeOperationAutoPlan[];
  total: number;
  nextAfter?: string;
}
export interface NativeOperationAutoApi {
  prepare(request: { brainId: string; acknowledgeCudaCpuTileFallback?: true }): Promise<NativeOperationAutoPrepared>;
  /** Explicitly arms natural-operation qualification/reuse; this method performs no model operation. */
  arm(request: { brainId: string; reviewId: string; samplesPerCandidate?: number }): Promise<NativeOperationAutoStatus>;
  status(brainId: string): Promise<NativeOperationAutoStatus>;
  plans(request: { brainId: string; after?: string; limit?: number }): Promise<NativeOperationAutoPlanPage>;
  stop(request: { brainId: string; operationId?: string }): Promise<NativeOperationAutoStatus>;
}
