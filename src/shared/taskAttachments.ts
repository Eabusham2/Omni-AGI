import type { ExperienceUploadKind } from "./uploadSupport";

export const TASK_ATTACHMENT_FORMAT = "omni-chat-task-attachment" as const;
export const TASK_ATTACHMENT_VERSION = 1 as const;

export type ChatTaskAttachmentKind = "file" | "image" | "audio" | "video";
export type ChatTaskAttachmentState = "staged" | "bound" | "retired";

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
  selection: ExperienceUploadKind;
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
  list(brainId: string, state?: "staged"): Promise<ChatTaskAttachmentHandle[]>;
  forTurn(brainId: string, turnId: string): Promise<ChatTaskAttachmentHandle[]>;
  byIds(brainId: string, attachmentIds: string[]): Promise<ChatTaskAttachmentHandle[]>;
  remove(brainId: string, attachmentId: string): Promise<boolean>;
}
