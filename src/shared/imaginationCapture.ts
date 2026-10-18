export type ImaginationCaptureOrigin =
  | { kind: "user-requested"; authority: "main-ui" | "main-human-action"; requestId: string }
  | { kind: "spontaneous"; authority: "main-organic-action"; actionEventId: string }
  | { kind: "unclassified"; authority: "native-brain-action" | "missing-main-provenance" };

export interface ImaginationCaptureSettings {
  version: 1; revision: number; bufferSeconds: number; alwaysSave: boolean;
  /** Copies preserve preferences but require an explicit user setting action. */
  dormant: boolean;
}
export interface ImaginationRecording {
  format: "omni-imagination-recording"; version: 1; id: string; brainId: string; sequence: number;
  requestId: string; mode: "manual" | "always-save"; state: "recording" | "complete" | "cancelled" | "resource-paused" | "dormant";
  revision: number; controlRevision: number; startedAt: string; endedAt?: string; requestedLookbackSeconds: number; availableLookbackSeconds: number;
  /** Actual monotonic active recording duration; wall audit time is separate. */
  timelineEndMs?: number;
  fragmentCount: number; bytes: number; gapCount: number;
  representation: "verified-delivered-fragment-timeline";
  label: "Delivered fragments — not a completed video";
  inheritedFrom?: { brainId: string; recordingId: string };
}
export interface ImaginationCapturedFragment {
  id: string; recordingId: string; index: number; streamId: string; observedAt: string; offsetMs: number;
  origin: ImaginationCaptureOrigin; kind: "preview-fragment" | "final-artifact";
  modality: "image" | "audio" | "video"; mimeType: string; sha256: string; bytes: number;
  revision?: number; width?: number; height?: number; durationMs?: number; fps?: number; frameCount?: number;
  mediaTimeline?: import("./nativeMediaTimeline").NativeMediaTimeline;
  deliveryAnchor?: import("./nativeMediaTimeline").ImaginationDeliveryAnchor;
  mediaUrl?: string; available: boolean;
}
export interface ImaginationCaptureStatus {
  brainId: string; settings: ImaginationCaptureSettings; recording?: ImaginationRecording;
  bufferedFragments: number; bufferedBytes: number; availableLookbackSeconds: number; autoPaused: boolean;
  /** Natural chat media intent is not inferred from keywords or source:brain. */
  unclassifiedChatIntent: true;
}
export interface ImaginationCaptureEvent { brainId: string; type: "changed" | "recording" | "gap"; recordingId?: string; at: string; detail?: string }
export interface ImaginationCaptureApi {
  assembly: import("./imaginationAssembly").ImaginationAssemblyApi;
  status(brainId: string): Promise<ImaginationCaptureStatus>;
  settings(request: { brainId: string; expectedRevision: number; bufferSeconds: number; alwaysSave: boolean }): Promise<ImaginationCaptureStatus>;
  record(request: { brainId: string; requestId: string; lookbackSeconds: number }): Promise<ImaginationRecording>;
  stop(request: { brainId: string; recordingId: string; expectedRevision: number }): Promise<ImaginationRecording>;
  cancel(request: { brainId: string; recordingId: string; expectedRevision: number }): Promise<ImaginationRecording>;
  resume(request: { brainId: string; recordingId: string; expectedRevision: number }): Promise<ImaginationRecording>;
  recordings(brainId: string, query?: { beforeSequence?: number; limit?: number }): Promise<{ entries: ImaginationRecording[]; total: number; nextBeforeSequence?: number }>;
  fragments(brainId: string, recordingId: string, query?: { afterIndex?: number; limit?: number }): Promise<{ entries: ImaginationCapturedFragment[]; total: number; nextAfterIndex?: number }>;
  onEvent(listener: (event: ImaginationCaptureEvent) => void): () => void;
}
