/** Main-owned, explicitly reviewed App data; never a tool result or persona. */
export type NativeAppObservationDisposition = "observe" | "superseded";

export interface NativeAppObservationLearning {
  kind: "experience";
  source: "mcp-app";
  sourceLabel: string;
  steps: 0;
  experienceCountBefore: number;
  experienceCountAfter: number;
  assemblyId: string;
  afterimageId: string;
  textSha256: string;
  textBytes: number;
}

/** One bounded native head. The full immutable audit remains in the host. */
export interface NativeAppObservationHead {
  format: "omni-native-app-observation-head";
  version: 1;
  streamId: string;
  ordinal: number;
  recordSha256: string;
  dataSha256: string;
  previousSha256: string;
  baseFrontierSha256: string;
  disposition: NativeAppObservationDisposition;
  state: "observed" | "advanced-without-observation";
  committedAt: string;
  learning: NativeAppObservationLearning | null;
  /** Present only for an admitted ordered sensory/reference whole experience. */
  content?: { summary: NativeAppObservationContentSummary; audit: NativeAppObservationAudit };
}

export interface NativeAppObservationFrontier {
  format: "omni-native-app-observation-frontier";
  version: 1;
  brainId: string;
  frontierSha256: string;
  head: NativeAppObservationHead | null;
}

export interface NativeAppObservationInput {
  brainId: string;
  /** Exact UTF-8 source bytes, not reserialized after approval. */
  recordJson: string;
  recordSha256: string;
  dataJson: string;
  dataSha256: string;
  expectedFrontierSha256: string;
  disposition: NativeAppObservationDisposition;
  /** Required before replacing a head with an applied, unarchived ACK audit. */
  previousAuditArchivedSha256?: string;
}

export interface NativeAppObservationResult extends NativeAppObservationFrontier {
  /** An exact current-head retry never executes learning or saving again. */
  duplicate: boolean;
  /** A durable commit survived cleanup failure; warm owner was released. */
  reloadRequired?: true;
  /** The worker no longer exposes this instance as a resident neural owner. */
  nativeOwnerReleased?: true;
  head: NativeAppObservationHead;
}

export const EMPTY_NATIVE_APP_OBSERVATION_FRONTIER = "0".repeat(64);

/** Emitted only after fencing/removing the failed warm owner. */
export interface NativeAppObservationOwnerRelease {
  format: "omni-native-owner-release";
  version: 1;
  brainId: string;
  reason: "app-observation-checkpoint-boundary";
  /** null means publication could not be verified; never permission to replay. */
  committed: boolean | null;
  reloadRequired: true;
  nativeOwnerReleased: true;
  cleanupWarnings?: string[];
}

/** Constant-size coverage; detailed measured rows are in the current ACK file. */
export interface NativeAppObservationContentSummary {
  format: "omni-native-app-content-summary";
  version: 1;
  blockCount: number;
  textBlocks: number;
  resourceTextBlocks: number;
  referenceBlocks: number;
  structuredBlocks: number;
  imageBlocks: number;
  audioBlocks: number;
  decodedWindows: number;
  imageFrames: number;
  audioModelSamples: number;
  sourceMediaBytes: number;
  orderedInputSha256: string;
  wholeInputTraversed: true;
  nativeSensory: boolean;
}

export interface NativeAppObservationAudit {
  format: "omni-native-app-observation-audit";
  version: 1;
  /** Safe engine-relative path, never supplied by a remote server. */
  path: string;
  sha256: string;
  bytes: number;
  rows: number;
  sourceBrainId: string;
  recordSha256: string;
  dataSha256: string;
  orderedInputSha256: string;
}

export interface NativeAppObservationBlockReceipt {
  format: "omni-native-app-observation-block";
  version: 1;
  sourceBrainId: string;
  recordSha256: string;
  dataSha256: string;
  blockIndex: number;
  kind: "text" | "resource-text" | "resource-reference" | "structured" | "image" | "audio";
  sourceEncoding: "utf8-text" | "approved-data-json" | "decoded-inline-media";
  sourceSha256: string;
  sourceBytes: number;
  wholeInputTraversed: true;
  nativeSensory: boolean;
  /** References never imply fetching the linked resource. */
  fetched: false;
  media?: import("./nativeMcpSampling").NativeMcpSamplingMediaReceipt;
}
