/** Read-only cursors are live host state, not portable permissions. */
export interface FileToolSource {
  kind: "host-file" | "task-attachment";
  path?: string;
  attachmentId?: string;
  manifestSha256?: string;
  contentSha256?: string;
  displayName: string;
  bytes: number;
  /** Host metadata fence or task-manifest binding; not a host-file opening-byte hash. */
  identity: string;
}
export interface FileToolPage {
  format: "omni-file-page";
  version: 1;
  source: FileToolSource;
  offset: number;
  bytes: number;
  encoding: "utf8" | "base64";
  content: string;
  /** SHA of exactly the returned bytes, not the unread remainder. */
  sha256: string;
  /** SHA over the completed returned stream, not an atomic host-file-at-open snapshot. */
  wholeSha256?: string;
  complete: boolean;
  hasMore: boolean;
  nextCursor?: string;
  nextOffset?: number;
}
export interface FileToolSearchMatch {
  path?: string;
  attachmentId?: string;
  name: string;
  kind: "file" | "directory" | "link" | "other";
  bytes: number;
  modifiedAt?: string;
  fileIdentity: string;
  byteOffset?: number;
  /** Exact matched UTF-8 bytes, not a lossy full-file rendering. */
  matchedBytesSha256?: string;
}
export interface FileToolSearchPage {
  format: "omni-file-search-page";
  version: 1;
  matching: "name-substring-case-insensitive" | "utf8-literal-bytes" | "metadata";
  matches: FileToolSearchMatch[];
  scannedBytes: number;
  visitedEntries: number;
  hasMore: boolean;
  nextCursor?: string;
  /** End of this checked walk, not an atomic whole-filesystem snapshot. */
  complete: boolean;
  consistency: "checked-open-files-not-an-atomic-tree-snapshot";
}
export interface TaskFileHandoff {
  format: "omni-task-file-handoff";
  version: 1;
  brainId: string;
  turnId: string;
  requestId: string;
  requestSha256: string;
  source: FileToolSource;
  relativePath: string;
  bytes: number;
  sha256: string;
  createdAt: string;
  /** Historical provenance only. Never imported as a live file/code grant. */
  authority: "historical-not-live";
}
