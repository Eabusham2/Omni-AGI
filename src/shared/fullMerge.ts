import type { BrainConfig, ToolPermissionLevel, HardwareTier } from "./types";

export interface FullMergeParentSeal {
  brainId: string;
  name: string;
  /** Exact frozen native and host file declaration, not a display checksum. */
  generationSha256: string;
  ancestry: string[];
}

export interface FullMergeAccessPolicy {
  toolPermissions: Array<{ toolId: string; level: ToolPermissionLevel }>;
  /** Credentials and live grants require a separate destination-local review. */
  credentials: "none";
  mcp: "disabled";
}

export interface FullMergeRequest {
  operationId: string;
  parentIds: string[];
  name: string;
  config: BrainConfig;
  accessPolicy?: FullMergeAccessPolicy;
  hardwareTier?: HardwareTier;
  /** Opaque native-picker selection; installation-local and never part of BrainConfig. */
  spillPoolSelectionId?: string;
}

/** Production start requires an explicit review, never parent/default authority. */
export interface FullMergeStartRequest extends FullMergeRequest {
  accessPolicy: FullMergeAccessPolicy;
  hardwareTier: HardwareTier;
}

export interface FullMergeProgress {
  operationId: string;
  targetBrainId: string;
  name?: string;
  /** Restart-safe review data; native-picker capability tokens are never exposed here. */
  reviewedSettings?: BrainConfig;
  reviewedAccess?: FullMergeAccessPolicy;
  hardwareTier?: HardwareTier;
  state: "preparing" | "copying" | "assembling" | "verifying" | "paused" | "complete" | "cancelled";
  filesDiscovered: number;
  filesCopied: number;
  logicalBytes: number;
  copiedBytes: number;
  uniqueBlobBytes: number;
  parents: FullMergeParentSeal[];
  error?: string;
}

export interface FullMergeHistoryPage {
  brainId: string;
  entries: Array<{ sequence: number; parentId: string; path: string; kind: string;
    sourceId: string; payloadSha256: string; payload: unknown; historicalOnly: true }>;
  nextAfterSequence?: number;
}

export interface FullMergePendingIngestion {
  id: string;
  status: "queued" | "active" | "complete";
  parentId: string;
  sourceNameSha256: string;
  contentSha256: string;
  committedRecords: number;
  epoch: number;
  policy: string;
}

export interface FullMergeResumeIngestionRequest {
  brainId: string;
  pendingId: string;
  /** Exact original parser name; native activation verifies its stored hash. */
  sourceName: string;
}
