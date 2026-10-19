import type { BrainRepository } from "./brainRepository";
import type { TaskAttachmentService } from "./taskAttachmentService";
import { TaskAttachmentStore } from "./taskAttachmentStore";
import { TaskContinuationScopeStore, type TaskContinuationResumeProof, type TaskContinuationScopeSnapshot,
  type WarmTaskAttachmentTransfer } from "./taskContinuationScope";
import { projectTaskAttachmentIds } from "./taskAttachmentProjection";
import type { ChatTaskAttachmentHandle, ChatTaskAttachmentReadResult, ChatTaskAttachmentResolved } from "../shared/taskAttachments";
import type { ChatMessage } from "../shared/types";
import type { InheritedTaskFiles } from "./inheritedTaskFiles";
import type { InheritedTaskFileNativeContext } from "./inheritedTaskFileClosure";
import type { InheritedTaskFileReference } from "../shared/inheritedTaskFiles";

export type TaskAttachmentVisibleReference = ChatTaskAttachmentResolved | InheritedTaskFileReference;

/** Scope is file visibility only. ToolExecutor still owns every current permission/Ask decision. */
export class TaskContinuationAttachments {
  private inheritedFiles?: InheritedTaskFiles;
  setInheritedTaskFiles(service: InheritedTaskFiles): void { this.inheritedFiles = service; }
  async activateInheritedResume(context: InheritedTaskFileNativeContext): Promise<void> {
    if (!this.inheritedFiles) throw new Error("Reviewed inherited task file adoption is unavailable.");
    await this.inheritedFiles.enter(context);
  }
  finishInheritedExecution(brainId: string, turnId: string): void { this.inheritedFiles?.finish(brainId, turnId); }
  reviewedAccess(brainId: string, turnId: string, action: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<{ revision: string } | undefined> {
    return this.inheritedFiles?.reviewedAccess(brainId, turnId, action, args, signal) ?? Promise.resolve(undefined);
  }
  private readonly projectionEvidence = new WeakMap<readonly ChatTaskAttachmentHandle[], { brainId: string; turnId: string; scope?: TaskContinuationScopeSnapshot }>();

  constructor(private readonly repository: Pick<BrainRepository, "get" | "brainDirectory">,
    private readonly base: TaskAttachmentService,
    private readonly attentionEpoch: (brainId: string) => Promise<number>) {}

  private async store(brainId: string, turnId: string): Promise<TaskContinuationScopeStore> {
    // Initialize/validate the existing workspace through its actual owner. No parallel blob registry.
    await this.base.listForTurn(brainId, turnId);
    const brain = await this.repository.get(brainId);
    return new TaskContinuationScopeStore(TaskAttachmentStore.databasePath(this.repository.brainDirectory(brain.id)), brain.id);
  }

  async transfer(request: WarmTaskAttachmentTransfer, signal: AbortSignal): Promise<void> {
    signal.throwIfAborted();
    const store = await this.store(request.brainId, request.predecessorTurnId);
    if (await this.attentionEpoch(request.brainId) !== request.attentionEpoch) throw new Error("Fresh attention retired this warm task handoff.");
    signal.throwIfAborted();
    store.transfer(request);
    if (await this.attentionEpoch(request.brainId) !== request.attentionEpoch) {
      store.retireExceptEpoch(await this.attentionEpoch(request.brainId));
      throw new Error("Fresh attention changed during the warm task handoff.");
    }
    signal.throwIfAborted();
  }

  async activateRetained(proof: TaskContinuationResumeProof): Promise<void> {
    const store = await this.store(proof.brainId, proof.turnId);
    const epoch = await this.attentionEpoch(proof.brainId);
    store.retireExceptEpoch(epoch);
    if (epoch !== proof.attentionEpoch) throw new Error("The retained task belongs to another attention epoch.");
    store.activateRetained(proof);
  }

  async hasContinuationScope(brainId: string, turnId: string): Promise<boolean> {
    return Boolean((await this.store(brainId, turnId)).snapshot(turnId));
  }

  async validateCurrentInput(brainId: string, turnId: string, inputSha256: string, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    if (this.inheritedFiles?.validateInput(brainId, turnId, inputSha256)) return;
    const { snapshot } = await this.scope(brainId, turnId, signal);
    if (snapshot && snapshot.record.successorInputSha256 !== inputSha256) throw new Error("This continued task ID belongs to a different immutable current input.");
  }

  private async scope(brainId: string, turnId: string, signal?: AbortSignal): Promise<{
    store: TaskContinuationScopeStore; snapshot?: TaskContinuationScopeSnapshot;
  }> {
    signal?.throwIfAborted();
    const store = await this.store(brainId, turnId), snapshot = store.snapshot(turnId);
    if (snapshot) {
      const epoch = await this.attentionEpoch(brainId);
      if (snapshot.record.attentionEpoch !== epoch) { store.retireExceptEpoch(epoch); throw new Error("Fresh attention retired this task continuation scope."); }
      if (snapshot.state !== "active") throw new Error("This copied or retired task scope needs its exact authenticated saved execution; a new turn cannot activate it.");
      store.assertCurrent(snapshot);
    }
    signal?.throwIfAborted();
    return { store, snapshot };
  }

  private async recheck(brainId: string, snapshot: TaskContinuationScopeSnapshot | undefined,
    store: TaskContinuationScopeStore, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    if (snapshot) {
      const epoch = await this.attentionEpoch(brainId);
      if (epoch !== snapshot.record.attentionEpoch) { store.retireExceptEpoch(epoch); throw new Error("Fresh attention changed during this task file read."); }
      store.assertCurrent(snapshot);
    }
    signal?.throwIfAborted();
  }

  // Binding and raw-audio source resolution retain their original exact owner semantics.
  bindToTurn(brainId: string, turnId: string, ids: readonly string[]) { return this.base.bindToTurn(brainId, turnId, ids); }
  list(brainId: string, state?: "staged") { return this.base.list(brainId, state); }
  resolveForTurn(brainId: string, turnId: string, id: string, signal?: AbortSignal) { return this.base.resolveForTurn(brainId, turnId, id, signal); }

  async listForTurn(brainId: string, turnId: string, signal?: AbortSignal): Promise<TaskAttachmentVisibleReference[]> {
    if (this.inheritedFiles?.has(brainId, turnId)) return this.inheritedFiles.references(brainId, turnId, signal);
    const { store, snapshot } = await this.scope(brainId, turnId, signal);
    const own = await this.base.listForTurn(brainId, turnId), results = [...own], seen = new Set(own.map(file => file.attachmentId));
    const boundTurns = new Map<string, ChatTaskAttachmentResolved[]>();
    for (const alias of snapshot?.aliases ?? []) {
      if (seen.has(alias.attachmentId)) continue;
      await this.recheck(brainId, snapshot, store, signal);
      let available = boundTurns.get(alias.owningTurnId!);
      if (!available) { available = await this.base.listForTurn(brainId, alias.owningTurnId!); boundTurns.set(alias.owningTurnId!, available); }
      const file = available.find(candidate => candidate.attachmentId === alias.attachmentId);
      if (!file || file.manifestSha256 !== alias.manifestSha256) throw new Error("The continued task lost its exact original file descriptor.");
      await this.recheck(brainId, snapshot, store, signal);
      results.push(file); seen.add(file.attachmentId); // Do not rewrite the original owning turn, source or manifest.
    }
    await this.recheck(brainId, snapshot, store, signal);
    return results;
  }

  /** Metadata projection may show historical custody; it does not grant file reads or activate a scope. */
  async handlesForTurn(brainId: string, turnId: string): Promise<ChatTaskAttachmentHandle[]> {
    const retired = await this.inheritedFiles?.handles(brainId, turnId);
    if (retired) return retired;
    const store = await this.store(brainId, turnId), scope = store.snapshot(turnId);
    const own = await this.base.handlesForTurn(brainId, turnId), seen = new Set(own.map(file => file.id));
    const inherited = await this.base.handlesByIds(brainId, (scope?.aliases ?? []).filter(alias => !seen.has(alias.attachmentId)).map(alias => alias.attachmentId));
    const handles = [...own, ...inherited];
    if (scope && store.snapshot(turnId)?.guardSha256 !== scope.guardSha256) throw new Error("Task attachment metadata changed during projection.");
    this.projectionEvidence.set(handles, { brainId, turnId, ...(scope ? { scope } : {}) });
    return handles;
  }

  projectMessage(message: ChatMessage, handles: readonly ChatTaskAttachmentHandle[]): ChatMessage {
    const retired = this.inheritedFiles?.projectMessage(message, handles);
    if (retired) return retired;
    const evidence = this.projectionEvidence.get(handles);
    if (!evidence?.scope) return projectTaskAttachmentIds(message, handles);
    if (message.role !== "human" || !handles.length) return message;
    const turnId = message.inputReceipt?.turnId ?? message.turnId;
    if (turnId !== evidence.turnId || message.turnId !== undefined && message.turnId !== evidence.turnId ||
        handles.some(handle => handle.brainId !== evidence.brainId || handle.state === "staged") ||
        new Set(handles.map(handle => handle.id)).size !== handles.length || handles.some(handle => handle.turnId !== evidence.turnId &&
          !evidence.scope!.aliases.some(alias => alias.attachmentId === handle.id && alias.manifestSha256 === handle.manifestSha256))) {
      throw new Error("Task attachment presentation has no exact continuation provenance.");
    }
    const taskAttachmentIds = handles.map(handle => handle.id);
    return { ...message, turnId, taskAttachmentIds,
      ...(message.inputReceipt ? { inputReceipt: { ...message.inputReceipt, taskAttachmentIds: [...taskAttachmentIds] } } : {}) };
  }

  async readForTask(brainId: string, turnId: string, attachmentId: string, offset?: number, maxBytes?: number,
    signal?: AbortSignal, entryPath?: string): Promise<ChatTaskAttachmentReadResult> {
    if (this.inheritedFiles?.has(brainId, turnId)) return this.inheritedFiles.read(brainId, turnId, attachmentId, offset, maxBytes, signal, entryPath);
    const { store, snapshot } = await this.scope(brainId, turnId, signal);
    const own = (await this.base.handlesForTurn(brainId, turnId)).find(file => file.id === attachmentId);
    const alias = snapshot?.aliases.find(file => file.attachmentId === attachmentId);
    if (!own && !alias?.owningTurnId) throw new Error("This attachment is outside the actual continued task; unrelated history and unsent files are unavailable.");
    await this.recheck(brainId, snapshot, store, signal);
    const result = await this.base.readForTurn(brainId, own ? turnId : alias!.owningTurnId!, attachmentId, offset, maxBytes, signal, entryPath);
    await this.recheck(brainId, snapshot, store, signal);
    return result;
  }

  async execute(brainId: string, turnId: string | undefined, action: string, args: Record<string, unknown>, signal: AbortSignal): Promise<unknown> {
    if (!turnId) throw new Error("Task attachments require the actual owning chat turn.");
    if (action !== "list" && action !== "read") throw new Error("Unknown task attachment action.");
    if (args.entryPath !== undefined && typeof args.entryPath !== "string") throw new Error("Folder entryPath must be a string.");
    if (this.inheritedFiles?.has(brainId, turnId)) {
      const references = await this.inheritedFiles.references(brainId, turnId, signal);
      if (action === "list") {
        const attachments = await this.inheritedFiles.handles(brainId, turnId);
        await this.inheritedFiles.references(brainId, turnId, signal);
        return { attachments, inheritedTaskFiles: references };
      }
      const attachmentId = typeof args.attachmentId === "string" ? args.attachmentId : references.length === 1 ? references[0]!.attachmentId : undefined;
      if (!attachmentId) throw new Error("Choose one exact reviewed inherited task attachment.");
      if (args.offset !== undefined && typeof args.offset !== "number" || args.maxBytes !== undefined && typeof args.maxBytes !== "number") throw new Error("Attachment read range must use safe integers.");
      return this.inheritedFiles.read(brainId, turnId, attachmentId, args.offset as number | undefined, args.maxBytes as number | undefined, signal, args.entryPath as string | undefined);
    }
    const { store, snapshot } = await this.scope(brainId, turnId, signal);
    const attachments = await this.handlesForTurn(brainId, turnId);
    await this.recheck(brainId, snapshot, store, signal);
    if (action === "list") return { attachments };
    const attachmentId = typeof args.attachmentId === "string" ? args.attachmentId : attachments.length === 1 ? attachments[0]!.id : undefined;
    if (!attachmentId) throw new Error("Choose one attachmentId from the current continued task before reading.");
    if (args.offset !== undefined && (typeof args.offset !== "number" || !Number.isSafeInteger(args.offset)) ||
        args.maxBytes !== undefined && (typeof args.maxBytes !== "number" || !Number.isSafeInteger(args.maxBytes))) throw new Error("Attachment read range must use safe integers.");
    return this.readForTask(brainId, turnId, attachmentId, args.offset as number | undefined, args.maxBytes as number | undefined, signal, args.entryPath as string | undefined);
  }
}
