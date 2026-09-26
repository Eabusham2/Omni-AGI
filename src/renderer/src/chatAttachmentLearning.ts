import type {
  DatasetManifest,
  DatasetPreviewRequest,
  DatasetStartRequest,
  RuntimeJob
} from "../../shared/types";

export interface ChatAttachmentDataApi {
  preview(request: DatasetPreviewRequest): Promise<DatasetManifest | null>;
  start(request: DatasetStartRequest): Promise<RuntimeJob>;
}

export interface ChatAttachmentLearningRequest extends DatasetPreviewRequest {
  requestId: string;
  selection: NonNullable<DatasetPreviewRequest["selection"]>;
}

export interface ChatAttachmentLearningStart {
  manifest: DatasetManifest;
  job: RuntimeJob;
}

/**
 * Commit one selected snapshot before scheduling exactly one traversal job.
 *
 * The chat composer used to call the legacy synchronous `ingestFiles` RPC.
 * That left the picker pending and rendered no activity while a worker was
 * busy.  The manifest/job boundary is restartable, publishes hashing progress,
 * and gives cancellation a stable request or job id.
 */
export async function queueChatAttachmentLearning(
  data: ChatAttachmentDataApi,
  request: ChatAttachmentLearningRequest,
  isCurrent: () => boolean = () => true
): Promise<ChatAttachmentLearningStart | null> {
  const manifest = await data.preview(request);
  if (!manifest || !isCurrent()) return null;
  const job = await data.start({
    brainId: request.brainId,
    manifestId: manifest.id,
    policy: request.policy,
    epochs: 1,
    resume: true
  });
  return { manifest, job };
}

/** Synchronous lease prevents two picker events from scheduling duplicates. */
export class ChatAttachmentOperationGate {
  private active: symbol | null = null;

  begin(): symbol | null {
    if (this.active) return null;
    this.active = Symbol("chat-attachment-operation");
    return this.active;
  }

  isCurrent(lease: symbol): boolean {
    return this.active === lease;
  }

  finish(lease: symbol): boolean {
    if (!this.isCurrent(lease)) return false;
    this.active = null;
    return true;
  }

  reset(): void {
    this.active = null;
  }
}
