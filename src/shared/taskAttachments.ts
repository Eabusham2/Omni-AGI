import type { ExperienceUploadKind } from "./uploadSupport";

export const TASK_ATTACHMENT_FORMAT = "omni-chat-task-attachment" as const;
export const TASK_ATTACHMENT_VERSION = 1 as const;

export type ChatTaskAttachmentKind = "file" | "image" | "audio" | "video";
export type ChatTaskAttachmentState = "staged" | "bound" | "retired";
export type ChatTaskAttachmentSelection = ExperienceUploadKind | "folder";

export interface TaskFolderEntry {
  path: string;
  kind: "file" | "directory";
  mimeType: string;
  bytes: number;
  contentSha256?: string;
}

export interface TaskFolderTraversal {
  discoveredFiles: number;
  processedFiles: number;
  discoveredDirectories: number;
  processedDirectories: number;
  sourceBytes: number;
  copiedBytes: number;
  rejectedEntries: number;
  complete: boolean;
  manifestSha256: string;
}

export interface ChatTaskAttachmentStageProgress {
  brainId: string;
  requestId: string;
  phase: "copying" | "sealing";
  traversal: TaskFolderTraversal;
}

/** Renderer-safe projection. Host paths and private source provenance never cross preload. */
export interface ChatTaskAttachmentHandle {
  format: typeof TASK_ATTACHMENT_FORMAT;
  version: typeof TASK_ATTACHMENT_VERSION;
  id: string;
  brainId: string;
  displayName: string;
  kind: ChatTaskAttachmentKind;
  mimeType: string;
  contentSha256: string;
  manifestSha256: string;
  bytes: number;
  state: ChatTaskAttachmentState;
  turnId?: string;
  createdAt: string;
  updatedAt: string;
  /** Staging alone never starts ingestion or asserts a neural mutation. */
  ingestionStarted: false;
  parametersChanged: false;
}

export interface ChatTaskAttachmentStageRequest {
  brainId: string;
  requestId: string;
  selection: ChatTaskAttachmentSelection;
}

export interface ChatTaskAttachmentDroppedRequest {
  brainId: string;
  requestId: string;
}

export interface ChatTaskAttachmentReadResult {
  format: "omni-chat-task-attachment-read";
  version: 1;
  attachment: ChatTaskAttachmentHandle;
  offset: number;
  returnedBytes: number;
  totalBytes: number;
  encoding: "utf8" | "base64";
  content: string;
  nextOffset?: number;
  /** Manifest cursors are entry ordinals; an explicit member uses byte offsets. */
  folder?: { cursorUnit: "entries" | "bytes"; traversal: TaskFolderTraversal; entryPath?: string; entrySha256?: string; entryMimeType?: string };
}

export interface ChatTaskAttachmentResolved {
  brainId: string;
  turnId: string;
  attachmentId: string;
  displayName: string;
  kind: ChatTaskAttachmentKind;
  mimeType: string;
  contentSha256: string;
  manifestSha256: string;
  bytes: number;
  /** Main/native-only path relative to the selected brain root. */
  relativePath: string;
}

export interface ChatTaskAttachmentsApi {
  stage(request: ChatTaskAttachmentStageRequest): Promise<ChatTaskAttachmentHandle[] | null>;
  stageDropped(request: ChatTaskAttachmentDroppedRequest, files: unknown[]): Promise<ChatTaskAttachmentHandle[]>;
  cancelStage(brainId: string, requestId: string): Promise<boolean>;
  onStageProgress?(listener: (event: ChatTaskAttachmentStageProgress) => void): () => void;
  list(brainId: string, state?: "staged"): Promise<ChatTaskAttachmentHandle[]>;
  forTurn(brainId: string, turnId: string): Promise<ChatTaskAttachmentHandle[]>;
  byIds(brainId: string, attachmentIds: string[]): Promise<ChatTaskAttachmentHandle[]>;
  remove(brainId: string, attachmentId: string): Promise<boolean>;
}
