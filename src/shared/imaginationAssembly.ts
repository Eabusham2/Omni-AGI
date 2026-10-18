import type { ImaginationCapturedFragment, ImaginationRecording } from "./imaginationCapture";

export interface ImaginationAssemblyOptions {
  width: number; height: number; fps: number;
  /** Required only when historical metadata has no measured monotonic end. */
  historicalTailMs?: number;
  /** Explicit source-media time at that stream's first retained delivery. Never inferred. */
  presentationOffsets?: Array<{ streamId: string; offsetMs: number }>;
}
export interface ImaginationAssemblyRequest {
  brainId: string; recordingId: string; expectedRevision: number; requestId: string;
  planId?: string;
  options?: Partial<ImaginationAssemblyOptions>;
}
export interface ImaginationAssemblyPlan {
  id: string;
  outputKind: "audio-only" | "audiovisual";
  options: ImaginationAssemblyOptions; durationMs: number; totalSpans: number; sourceFragments: number;
  estimatedPeakBytes: number; automatic: boolean;
  timingIssueCount: number; quantizationMs: number; captureGapCount: number;
}
export type ImaginationAssemblyPlanRequest = Omit<ImaginationAssemblyRequest, "planId">;
export interface ImaginationPlanEvent {
  brainId: string; requestId: string; sequence: number;
  state: "snapshotting" | "runtime" | "sources" | "spooling" | "ready" | "cancelling" | "cancelled" | "failed" | "dormant";
  completedFragments?: number; totalFragments?: number;
  runtime?: { state: string; target: string; message: string; completedBytes?: number; totalBytes?: number; fileName?: string };
  error?: string;
}
export interface ImaginationReplayTrack {
  streamId: string; slot: number; firstDeliveryMs: number; truncatedStart: boolean;
  anchor: "first-verified-main-delivery" | "explicit-retained-source-offset" | "unknown";
}
export interface ImaginationReplayState {
  streamId: string; slot: number; fragmentId: string; sourceTimeMs?: number;
  visual: "snapshot" | "play" | "hold" | "none";
  audio: "play" | "none";
  issue?: "unknown-source-timing" | "truncated-start-needs-offset" | "source-exhausted";
}
export interface ImaginationReplaySpan { startFrame: number; endFrame: number; startMs: number; endMs: number; states: ImaginationReplayState[] }
export interface ImaginationAssemblyManifest {
  format: "omni-delivered-timeline-composition"; version: 1;
  policy: "delivered-timeline-v1";
  outputKind: "audio-only" | "audiovisual";
  recording: ImaginationRecording;
  options: ImaginationAssemblyOptions;
  fragments: Omit<ImaginationCapturedFragment, "mediaUrl" | "available">[];
  tracks: ImaginationReplayTrack[]; spans: ImaginationReplaySpan[];
  durationMs: number; measuredEnd: boolean; captureGapCount: number;
  presentation: { layout: "fixed-stream-grid"; missingVisual: "neutral-black"; exhaustedVisual: "last-verified-frame-hold";
    missingAudio: "declared-silence"; independentAudio: "delivery-timed-equal-weight-mix";
    quantization: "nearest-output-frame" | "nearest-output-sample"; maximumQuantizationMs: number;
    nativeSynchronizationClaimed: false; supersededInSameFrame: string[] };
}
export interface ImaginationAssemblyManifestSummary extends Omit<ImaginationAssemblyManifest, "fragments" | "tracks" | "spans" | "presentation"> {
  fragmentCount: number; trackCount: number; spanCount: number; manifestSha256: string;
  presentation: Omit<ImaginationAssemblyManifest["presentation"], "supersededInSameFrame"> & { supersededInSameFrameCount: number };
}
export interface ImaginationAssembly {
  format: "omni-imagination-assembly"; version: 1; id: string; brainId: string; requestId: string; revision: number;
  recordingId: string; recordingRevision: number; state: "queued" | "preparing" | "encoding" | "complete" | "cancelled" | "failed" | "dormant";
  /** Main host alias only; original immutable manifest recording identity stays unchanged. */
  currentRecordingId?: string;
  createdAt: string; updatedAt: string; completedSpans: number; totalSpans: number;
  durationMs: number; manifestSha256: string; artifactSha256?: string; bytes?: number;
  mimeType?: "audio/wav" | "video/mp4";
  receiptSha256?: string;
  currentAttempt?: string; currentSpan?: number;
  codec?: { state: "ready" | "external"; artifactId?: string; artifactSha256?: string; binarySha256: string; distributionLicense: string; encoders: string[] };
  mediaUrl?: string; available?: boolean; error?: string;
  label: "Delivered timeline replay — presentation timing, not native synchronization";
  inheritedFrom?: { brainId: string; assemblyId: string };
}
export interface ImaginationAssemblyEvent { brainId: string; assemblyId: string; revision: number; state: ImaginationAssembly["state"] }
export interface ImaginationAssemblyApi {
  plan(request: ImaginationAssemblyPlanRequest): Promise<ImaginationAssemblyPlan>;
  cancelPlan(request: { brainId: string; requestId: string }): Promise<{ brainId: string; requestId: string; state: ImaginationPlanEvent["state"] | "absent" }>;
  onPlanEvent(listener: (event: ImaginationPlanEvent) => void): () => void;
  assemble(request: ImaginationAssemblyRequest): Promise<ImaginationAssembly>;
  get(brainId: string, assemblyId: string): Promise<ImaginationAssembly>;
  list(brainId: string, query?: { afterId?: string; limit?: number; recordingId?: string }): Promise<{ entries: ImaginationAssembly[]; nextAfterId?: string }>;
  manifest(brainId: string, assemblyId: string): Promise<ImaginationAssemblyManifestSummary>;
  manifestPage(brainId: string, assemblyId: string, query: { section: "fragments" | "tracks" | "spans"; afterIndex?: number; limit?: number }): Promise<{ entries: Array<ImaginationAssemblyManifest["fragments"][number] | ImaginationReplayTrack | ImaginationReplaySpan>; nextAfterIndex?: number; total: number }>;
  cancel(request: { brainId: string; assemblyId: string; expectedRevision: number }): Promise<ImaginationAssembly>;
  resume(request: { brainId: string; assemblyId: string; expectedRevision: number }): Promise<ImaginationAssembly>;
  onEvent(listener: (event: ImaginationAssemblyEvent) => void): () => void;
}
