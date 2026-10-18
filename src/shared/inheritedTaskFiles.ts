import type { InstanceResumeFrontier } from "./instanceLifecycle";
import type { ChatTaskAttachmentKind, ChatTaskAttachmentReadResult } from "./taskAttachments";
import type { ToolPermissionLevel } from "./types";

export interface InheritedTaskFileDisclosure {
  attachmentId: string;
  displayName: string;
  kind: ChatTaskAttachmentKind;
  mimeType: string;
  bytes: number;
  contentSha256: string;
  manifestSha256: string;
  sourceSha256: string;
  originalBinding: {
    brainId: string;
    turnId: string;
    attachmentId: string;
    manifestSha256: string;
    sourceSha256: string;
  };
}

export interface InheritedTaskFileReviewBody {
  format: "omni-inherited-task-file-review";
  version: 1;
  reviewId: string;
  brainId: string;
  executionId: string;
  rootTurnId: string;
  sourceBrainId: string;
  sourceTurnId: string;
  frontier: InstanceResumeFrontier;
  inputSha256: string;
  rootAttentionEpoch: number;
  sourceAttentionEpoch: number;
  ownerBindingSha256: string;
  sourceClosureSha256: string;
  permission: { toolId: "task.attachments"; level: ToolPermissionLevel; revisionSha256: string };
  files: InheritedTaskFileDisclosure[];
  createdAt: string;
}

export interface InheritedTaskFileAdoptionReceipt {
  format: "omni-inherited-task-file-adoption";
  version: 1;
  brainId: string;
  reviewId: string;
  reviewSha256: string;
  executionId: string;
  rootTurnId: string;
  approvedAt: string;
  receiptSha256: string;
  permissionsChanged: false;
  sourceBindingsChanged: false;
}

export interface InheritedTaskFileReview extends InheritedTaskFileReviewBody {
  reviewSha256: string;
  state: "review-required" | "approved";
  adoption?: InheritedTaskFileAdoptionReceipt;
}

/** Honest current-owner custody reference. It does NOT assert a new bound turn or rewrite an original manifest. */
export interface InheritedTaskFileReference extends InheritedTaskFileDisclosure {
  format: "omni-inherited-task-file-reference";
  version: 1;
  brainId: string;
  custody: "retired";
  reviewSha256: string;
  executionId: string;
}

export interface InheritedTaskFileReadResult extends ChatTaskAttachmentReadResult {
  inheritedTask: {
    executionId: string;
    reviewSha256: string;
    originalBinding: InheritedTaskFileDisclosure["originalBinding"];
    custody: "retired";
  };
}

export interface InheritedTaskFilesApi {
  prepareInheritedTaskFiles(request: { brainId: string; executionId: string }): Promise<InheritedTaskFileReview | null>;
  approveInheritedTaskFiles(request: { brainId: string; reviewId: string; reviewSha256: string }): Promise<InheritedTaskFileAdoptionReceipt>;
  revokeInheritedTaskFiles(request: { brainId: string; reviewId: string; reviewSha256: string }): Promise<{ revoked: true }>;
}
