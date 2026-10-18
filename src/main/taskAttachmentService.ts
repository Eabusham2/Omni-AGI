import type { BrainRepository } from "./brainRepository";
import type {
  ChatTaskAttachmentHandle,
  ChatTaskAttachmentReadResult,
  ChatTaskAttachmentResolved
} from "../shared/taskAttachments";
import type { ExperienceUploadKind } from "../shared/uploadSupport";
import {
  TaskAttachmentStore,
  type TaskAttachmentPrivateSource,
  type TaskAttachmentStageBytes,
  type TaskAttachmentStagePath
} from "./taskAttachmentStore";
import { TaskAttachmentReadVerifier } from "./taskAttachmentReadVerifier";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;

function requireId(value: string, label: string): string {
  if (!SAFE_ID.test(value)) throw new Error(`Invalid ${label}.`);
  return value;
}

function numberArgument(value: unknown, fallback: number): number {
  if (value === undefined) return fallback;
  if (typeof value !== "number" || !Number.isSafeInteger(value)) throw new Error("Attachment read range must use safe integers.");
  return value;
}

/** Main-owned task attachment registry. Staging bytes never invokes neural ingestion. */
export class TaskAttachmentService {
  private readonly activeStages = new Map<string, AbortController>();

  constructor(private readonly repository: BrainRepository,
    private readonly readVerifier = new TaskAttachmentReadVerifier()) {}

  private async store(brainId: string): Promise<TaskAttachmentStore> {
    const brain = await this.repository.get(requireId(brainId, "attachment brain id"));
    const store = new TaskAttachmentStore(this.repository.brainDirectory(brain.id), brain.id, this.readVerifier);
    await store.initialize();
    return store;
  }

  private stageKey(brainId: string, requestId: string): string {
    return `${requireId(brainId, "attachment brain id")}\0${requireId(requestId, "attachment request id")}`;
  }

  private async stagedOperation<T>(
    brainId: string,
    requestId: string,
    operation: (store: TaskAttachmentStore, signal: AbortSignal) => Promise<T>
  ): Promise<T> {
    const key = this.stageKey(brainId, requestId), store = await this.store(brainId);
    if (this.activeStages.has(key)) throw new Error("This attachment selection is already being staged.");
    const controller = new AbortController();
    this.activeStages.set(key, controller);
    try { return await operation(store, controller.signal); }
    finally { if (this.activeStages.get(key) === controller) this.activeStages.delete(key); }
  }

  async requestHandles(brainId: string, requestId: string): Promise<ChatTaskAttachmentHandle[] | undefined> {
    return (await this.store(brainId)).requestHandles(requireId(requestId, "attachment request id"));
  }

  async stagePaths(
    brainId: string,
    requestId: string,
    paths: readonly TaskAttachmentStagePath[],
    selection: ExperienceUploadKind,
    signal?: AbortSignal
  ): Promise<ChatTaskAttachmentHandle[]> {
    signal?.throwIfAborted();
    const existing = await this.requestHandles(brainId, requestId);
    if (existing) return existing;
    return this.stagedOperation(brainId, requestId, async (store, stageSignal) => {
      const abort = (): void => { this.activeStages.get(this.stageKey(brainId, requestId))?.abort(signal?.reason); };
      signal?.addEventListener("abort", abort, { once: true }); if (signal?.aborted) abort();
      try { return await store.stagePaths(requestId, paths, selection, stageSignal); }
      finally { signal?.removeEventListener("abort", abort); }
    });
  }

  async stageBytes(
    brainId: string,
    request: Omit<TaskAttachmentStageBytes, "source"> & { requestId: string; source: TaskAttachmentPrivateSource },
    signal?: AbortSignal
  ): Promise<ChatTaskAttachmentHandle> {
    const existing = await this.requestHandles(brainId, request.requestId);
    if (existing) {
      if (existing.length !== 1) throw new Error("Attachment byte request has a different committed shape.");
      return existing[0]!;
    }
    return this.stagedOperation(brainId, request.requestId, async (store, stageSignal) => {
      const abort = (): void => {
        if (!stageSignal.aborted) (this.activeStages.get(this.stageKey(brainId, request.requestId)))?.abort(signal?.reason);
      };
      signal?.addEventListener("abort", abort, { once: true });
      if (signal?.aborted) abort();
      try { return await store.stageBytes(request.requestId, request, stageSignal); }
      finally { signal?.removeEventListener("abort", abort); }
    });
  }

  cancelStage(brainId: string, requestId: string): boolean {
    const controller = this.activeStages.get(this.stageKey(brainId, requestId));
    if (!controller || controller.signal.aborted) return false;
    controller.abort(new DOMException("Attachment staging cancelled.", "AbortError"));
    return true;
  }

  async list(brainId: string, state?: "staged"): Promise<ChatTaskAttachmentHandle[]> {
    return (await this.store(brainId)).list(false, state);
  }

  async handlesForTurn(brainId: string, turnId: string): Promise<ChatTaskAttachmentHandle[]> {
    return (await this.store(brainId)).handlesForTurn(requireId(turnId, "attachment turn id"));
  }

  async handlesByIds(brainId: string, attachmentIds: readonly string[]): Promise<ChatTaskAttachmentHandle[]> {
    const ids = attachmentIds.map((id) => requireId(id, "attachment id"));
    if (new Set(ids).size !== ids.length) throw new Error("Attachment identities are duplicated.");
    return (await this.store(brainId)).handlesByIds(ids);
  }

  async remove(brainId: string, attachmentId: string): Promise<boolean> {
    return (await this.store(brainId)).removeStaged(requireId(attachmentId, "attachment id"));
  }

  async bindToTurn(brainId: string, turnId: string, attachmentIds: readonly string[]): Promise<ChatTaskAttachmentResolved[]> {
    return (await this.store(brainId)).bindToTurn(requireId(turnId, "attachment turn id"), attachmentIds);
  }

  async listForTurn(brainId: string, turnId: string): Promise<ChatTaskAttachmentResolved[]> {
    return (await this.store(brainId)).listForTurn(requireId(turnId, "attachment turn id"));
  }

  async resolveForTurn(brainId: string, turnId: string, attachmentId: string, signal?: AbortSignal): Promise<ChatTaskAttachmentResolved> {
    signal?.throwIfAborted();
    return (await this.store(brainId)).resolveForTurn(
      requireId(turnId, "attachment turn id"), requireId(attachmentId, "attachment id"), signal
    );
  }

  async readForTurn(
    brainId: string,
    turnId: string,
    attachmentId: string,
    offset?: number,
    maxBytes?: number,
    signal?: AbortSignal
  ): Promise<ChatTaskAttachmentReadResult> {
    signal?.throwIfAborted();
    return (await this.store(brainId)).readForTurn(
      requireId(turnId, "attachment turn id"), requireId(attachmentId, "attachment id"),
      numberArgument(offset, 0), numberArgument(maxBytes, 64 * 1024), signal
    );
  }

  async execute(
    brainId: string,
    turnId: string | undefined,
    action: string,
    args: Record<string, unknown>,
    signal: AbortSignal
  ): Promise<unknown> {
    if (!turnId) throw new Error("Task attachments are available only to their owning chat turn.");
    signal.throwIfAborted();
    const attachments = await this.listForTurn(brainId, turnId);
    const publicAttachments = await this.handlesForTurn(brainId, turnId);
    signal.throwIfAborted();
    if (action === "list") return { attachments: publicAttachments };
    if (action !== "read") throw new Error("Unknown task attachment action.");
    const requested = typeof args.attachmentId === "string" ? args.attachmentId : undefined;
    const attachmentId = requested ?? (attachments.length === 1 ? attachments[0]!.attachmentId : undefined);
    if (!attachmentId) throw new Error("Choose one attachmentId from the current task before reading.");
    const result = await this.readForTurn(
      brainId, turnId, attachmentId,
      args.offset === undefined ? undefined : numberArgument(args.offset, 0),
      args.maxBytes === undefined ? undefined : numberArgument(args.maxBytes, 64 * 1024), signal
    );
    signal.throwIfAborted();
    return result;
  }
}
