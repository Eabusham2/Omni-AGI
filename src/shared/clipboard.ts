export const CLIPBOARD_FORMAT = "omni-clipboard-1" as const;
export const CLIPBOARD_READ_PAGE_BYTES = 256 * 1024;

export type ClipboardScope = "instance" | "host";
export type ClipboardMutationAction = "write" | "clear";
export type ClipboardMutationState = "complete" | "cancelled" | "failed" | "write-uncertain";

export interface ClipboardVersionSummary {
  id: string;
  revision: number;
  action: ClipboardMutationAction | "initial";
  contentSha256: string;
  bytes: number;
  createdAt: string;
  provenance: ClipboardProvenance;
}

export interface ClipboardProvenance {
  kind: "instance-owned" | "host-os-clipboard" | "initial" | "inherited";
  capturedAt: string;
  brainId?: string;
  adapter?: string;
  parentBrainId?: string;
  parentVersionId?: string;
}

export interface ClipboardCapabilities {
  format: typeof CLIPBOARD_FORMAT;
  scope: ClipboardScope;
  implemented: boolean;
  adapter: string;
  contentTypes: ["text/plain"];
  maxReadPageBytes: number;
  readIsBounded: true;
  write: boolean;
  clear: boolean;
  authority: "brain-portable-no-os-grant" | "current-host-permission-only";
  detail: string;
  currentVersionId?: string;
  retainedVersions?: ClipboardVersionSummary[];
  nextAfterVersion?: string;
}

export interface ClipboardReadReceipt {
  format: typeof CLIPBOARD_FORMAT;
  version: 1;
  action: "read";
  requestId: string;
  scope: ClipboardScope;
  state: "complete" | "partial";
  contentType: "text/plain";
  text: string;
  chunkBytes: number;
  chunkSha256: string;
  complete: boolean;
  cursor?: string;
  totalBytes?: number;
  contentSha256?: string;
  versionId?: string;
  revision?: number;
  provenance: ClipboardProvenance;
}

export interface ClipboardMutationReceipt {
  format: typeof CLIPBOARD_FORMAT;
  version: 1;
  action: ClipboardMutationAction;
  requestId: string;
  requestSha256: string;
  scope: ClipboardScope;
  state: ClipboardMutationState;
  contentType: "text/plain";
  contentBytes: number;
  contentSha256: string;
  writeMayHaveOccurred: boolean;
  retrySafe: boolean;
  repeated?: boolean;
  versionId?: string;
  revision?: number;
  provenance: ClipboardProvenance;
  error?: string;
}

export interface ClipboardToolResult {
  capabilities?: ClipboardCapabilities;
  read?: ClipboardReadReceipt;
  mutation?: ClipboardMutationReceipt;
}
