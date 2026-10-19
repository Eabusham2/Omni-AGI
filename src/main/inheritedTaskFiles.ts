import { randomUUID, createHash } from "node:crypto";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { InheritedTaskFileAdoptionReceipt, InheritedTaskFileReadResult, InheritedTaskFileReference,
  InheritedTaskFileReview, InheritedTaskFileReviewBody } from "../shared/inheritedTaskFiles";
import type { ChatMessage, ToolPermissionRecord } from "../shared/types";
import type { ChatTaskAttachmentHandle } from "../shared/taskAttachments";
import { TaskAttachmentStore, taskAttachmentManifestSha256, canonicalTaskAttachmentValue, validatedTaskAttachmentSource } from "./taskAttachmentStore";
import { TaskAttachmentReadVerifier } from "./taskAttachmentReadVerifier";
import { InheritedTaskFileReviewStore, inheritedTaskReviewFingerprint } from "./inheritedTaskFileReviewStore";
import { assertInheritedTaskFileGuards, inheritedTaskFileClosure, inheritedTaskFileHash, inheritedTaskManagedFilePath,
  type InheritedTaskFileClosure, type InheritedTaskFileNativeContext, type InheritedTaskFilePrivate } from "./inheritedTaskFileClosure";
import { sqliteFilename } from "./sqliteFilename";
import { readTaskFolderAttachment, TASK_FOLDER_MIME } from "./taskFolderContainer";

export class InheritedTaskFileAdoptionRequired extends Error {
  readonly code = "INHERITED_TASK_FILE_ADOPTION_REQUIRED";
  constructor(readonly brainId: string, readonly executionId: string) { super("Review this inherited task's exact retained files before resuming. No file access or old permission has been restored."); }
}

interface Options {
  root(brainId: string): string;
  context(brainId: string, executionId: string): Promise<InheritedTaskFileNativeContext>;
  permissions(brainId: string): Promise<ToolPermissionRecord[]>;
  rootEpoch(brainId: string): Promise<number>;
  /** Main map/current runtime check, never a native writer RPC from an inline tool wait. */
  assertActive(context: InheritedTaskFileNativeContext): void;
  verifier?: TaskAttachmentReadVerifier;
}
interface Lease { context: InheritedTaskFileNativeContext; body: InheritedTaskFileReviewBody;
  closure: InheritedTaskFileClosure; receipt: InheritedTaskFileAdoptionReceipt; }
const key = (brainId: string, turnId: string) => `${brainId}\0${turnId}`;
function publicHandle(row: InheritedTaskFilePrivate["row"]): ChatTaskAttachmentHandle {
  if (row.state !== "retired" || row.turn_id !== null || taskAttachmentManifestSha256(row) !== row.manifest_sha256) throw new Error("Inherited file is not exact retired custody.");
  return { format: "omni-chat-task-attachment", version: 1, id: row.id, brainId: row.brain_id,
    displayName: row.display_name, kind: row.kind, mimeType: row.mime_type, contentSha256: row.content_sha256,
    manifestSha256: row.manifest_sha256, bytes: row.bytes, state: "retired", createdAt: row.created_at, updatedAt: row.updated_at,
    ingestionStarted: false, parametersChanged: false };
}

/** Explicit current-main review, not an inherited grant or a synthetic bound attachment. */
export class InheritedTaskFiles {
  private readonly pending = new Map<string, InheritedTaskFileReviewBody>();
  private readonly leases = new Map<string, Lease>();
  private readonly projection = new WeakMap<readonly ChatTaskAttachmentHandle[], Lease>();
  private readonly verifier: TaskAttachmentReadVerifier;
  constructor(private readonly options: Options) { this.verifier = options.verifier ?? new TaskAttachmentReadVerifier(); }

  private store(brainId: string): InheritedTaskFileReviewStore {
    return new InheritedTaskFileReviewStore(TaskAttachmentStore.databasePath(this.options.root(brainId)), brainId);
  }
  private async permission(brainId: string): Promise<InheritedTaskFileReviewBody["permission"]> {
    const policy = (await this.options.permissions(brainId)).find(permission => permission.toolId === "task.attachments");
    const level = policy?.level ?? "off";
    return { toolId: "task.attachments", level, revisionSha256: inheritedTaskFileHash(policy ? {
      toolId: policy.toolId, level: policy.level, updatedAt: policy.updatedAt, revision: policy.revision ?? 0
    } : { toolId: "task.attachments", level: "off", missing: true }) };
  }

  private async proposal(context: InheritedTaskFileNativeContext): Promise<{ body: InheritedTaskFileReviewBody; closure: InheritedTaskFileClosure } | null> {
    const closure = await inheritedTaskFileClosure(this.options.root(context.brainId), context, undefined, this.verifier);
    if (!closure.files.length) return null;
    const permission = await this.permission(context.brainId);
    const body: InheritedTaskFileReviewBody = { format: "omni-inherited-task-file-review", version: 1,
      reviewId: randomUUID(), brainId: context.brainId, executionId: context.executionId, rootTurnId: context.frontier.turnId,
      sourceBrainId: context.sourceBrainId, sourceTurnId: context.sourceTurnId, frontier: structuredClone(context.frontier),
      inputSha256: context.inputSha256, rootAttentionEpoch: context.rootAttentionEpoch, sourceAttentionEpoch: context.sourceAttentionEpoch,
      ownerBindingSha256: closure.ownerBindingSha256, sourceClosureSha256: closure.sourceClosureSha256, permission,
      files: closure.files.map(file => file.disclosure), createdAt: new Date().toISOString() };
    return { body, closure };
  }

  async prepare(request: { brainId: string; executionId: string }): Promise<InheritedTaskFileReview | null> {
    const proposal = await this.proposal(await this.options.context(request.brainId, request.executionId));
    if (!proposal) return null;
    const approved = this.store(request.brainId).approved(proposal.body);
    if (approved) return { ...approved.body, reviewSha256: approved.receipt.reviewSha256, state: "approved", adoption: approved.receipt };
    const fingerprint = inheritedTaskReviewFingerprint(proposal.body);
    const existing = [...this.pending.values()].find(body => body.brainId === request.brainId && inheritedTaskReviewFingerprint(body) === fingerprint);
    const body = existing ?? proposal.body; this.pending.set(body.reviewId, body);
    return { ...structuredClone(body), reviewSha256: inheritedTaskFileHash(body), state: "review-required" };
  }

  async approve(request: { brainId: string; reviewId: string; reviewSha256: string }): Promise<InheritedTaskFileAdoptionReceipt> {
    const pending = this.pending.get(request.reviewId) ?? this.store(request.brainId).approvedById(request.reviewId, request.reviewSha256)?.body;
    if (!pending || pending.brainId !== request.brainId || inheritedTaskFileHash(pending) !== request.reviewSha256) throw new Error("This review is unavailable or changed; prepare the current inherited task again.");
    const current = await this.proposal(await this.options.context(request.brainId, pending.executionId));
    if (!current || inheritedTaskReviewFingerprint(current.body) !== inheritedTaskReviewFingerprint(pending)) throw new Error("The inherited source, exact frontier, files, epoch, or current access changed while review was open.");
    const receipt = this.store(request.brainId).approve(pending); this.pending.delete(request.reviewId); return receipt;
  }

  async revoke(request: { brainId: string; reviewId: string; reviewSha256: string }): Promise<{ revoked: true }> {
    const pending = this.pending.get(request.reviewId);
    if (pending) {
      if (pending.brainId !== request.brainId || inheritedTaskFileHash(pending) !== request.reviewSha256) throw new Error("Inherited pending review changed before dismissal.");
      this.pending.delete(request.reviewId);
    } else this.store(request.brainId).revoke(request.reviewId, request.reviewSha256);
    for (const [id, lease] of this.leases) if (lease.receipt.brainId === request.brainId && lease.receipt.reviewId === request.reviewId) this.leases.delete(id);
    return { revoked: true };
  }

  /** Called from the real authenticated resume boundary, before native admission; never from renderer metadata. */
  async enter(context: InheritedTaskFileNativeContext): Promise<void> {
    context = structuredClone(context);
    const proposal = await this.proposal(context);
    if (!proposal) { this.leases.delete(key(context.brainId, context.frontier.turnId)); return; }
    const approved = this.store(context.brainId).approved(proposal.body);
    if (!approved) throw new InheritedTaskFileAdoptionRequired(context.brainId, context.executionId);
    this.leases.set(key(context.brainId, context.frontier.turnId), { context, closure: proposal.closure, body: approved.body, receipt: approved.receipt });
  }

  has(brainId: string, turnId: string): boolean { return this.leases.has(key(brainId, turnId)); }
  finish(brainId: string, turnId: string): void { this.leases.delete(key(brainId, turnId)); }
  /** Current explicit review authorizes only these retained bytes, even when
   * general attachment access is Off. It grants no original-path/host access. */
  async reviewedAccess(brainId: string, turnId: string, action: string, args: Record<string, unknown>,
    signal?: AbortSignal): Promise<{ revision: string } | undefined> {
    const lease = this.leases.get(key(brainId, turnId));
    if (!lease || !["list", "read"].includes(action)) return undefined;
    await this.recheck(lease, undefined, signal);
    if (action === "read") {
      const attachmentId = typeof args.attachmentId === "string" ? args.attachmentId
        : lease.closure.files.length === 1 ? lease.closure.files[0]!.row.id : undefined;
      const file = lease.closure.files.find(file => file.row.id === attachmentId);
      if (!file) return undefined;
      await this.recheck(lease, file, signal);
    }
    return { revision: inheritedTaskFileHash({ reviewSha256: lease.receipt.reviewSha256,
      receiptSha256: lease.receipt.receiptSha256, executionId: lease.context.executionId,
      turnId, action, attachmentId: action === "read" ? args.attachmentId ?? null : null }) };
  }
  /** Trusted-user metadata lookup only. Never exposed as a model tool, file-byte read, binding or adoption. */
  async historicalFileHandlesForUser(brainId: string, attachmentIds: readonly string[]): Promise<ChatTaskAttachmentHandle[]> {
    if (new Set(attachmentIds).size !== attachmentIds.length || attachmentIds.some(id => !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(id))) throw new Error("Invalid historical task file identities.");
    const path = await inheritedTaskManagedFilePath(this.options.root(brainId), "attachments/registry.sqlite3"), db = new DatabaseSync(sqliteFilename(path), { readOnly: true });
    try {
      db.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-512");
      if (db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId ||
          db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== "omni-chat-task-attachments-1") throw new Error("Historical attachment has another current owner or registry format.");
      return attachmentIds.map(id => {
        const length = db.prepare("SELECT length(CAST(source_json AS BLOB)) AS bytes FROM attachments WHERE id=?").get(id)?.bytes;
        if (typeof length !== "number") throw new Error("Historical task file metadata is unavailable.");
        this.verifier.reserveMemory(length * 12 + 65536, "Historical file source metadata");
        const row = db.prepare("SELECT * FROM attachments WHERE id=?").get(id) as unknown as InheritedTaskFilePrivate["row"];
        if (row.brain_id !== brainId || row.state !== "retired" || createHash("sha256").update(row.source_json).digest("hex") !== row.source_sha256) throw new Error("Historical task file source is invalid.");
        validatedTaskAttachmentSource(JSON.parse(row.source_json)); return publicHandle(row);
      });
    } finally { db.close(); }
  }
  validateInput(brainId: string, turnId: string, inputSha256: string): boolean {
    const lease = this.leases.get(key(brainId, turnId)); if (!lease) return false;
    if (lease.context.inputSha256 !== inputSha256) throw new Error("The inherited file lease belongs to another exact native input.");
    return true;
  }

  private async recheck(lease: Lease, file?: InheritedTaskFilePrivate, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    if (this.leases.get(key(lease.context.brainId, lease.context.frontier.turnId)) !== lease) throw new Error("The current inherited task file lease has ended.");
    this.options.assertActive(lease.context);
    if (await this.options.rootEpoch(lease.context.brainId) !== lease.context.rootAttentionEpoch ||
        inheritedTaskFileHash(await this.permission(lease.context.brainId)) !== inheritedTaskFileHash(lease.body.permission)) throw new Error("Fresh attention or current file permission changed during this inherited task.");
    this.store(lease.context.brainId).assertApproved(lease.receipt);
    await assertInheritedTaskFileGuards(this.options.root(lease.context.brainId), [...lease.closure.guards, ...(file ? [file.sourceGuard] : [])], signal);
    if (file) {
      const databasePath = await inheritedTaskManagedFilePath(this.options.root(lease.context.brainId), "attachments/registry.sqlite3");
      await inheritedTaskManagedFilePath(this.options.root(lease.context.brainId), `attachments/blobs/${file.row.content_sha256}.data`);
      const db = new DatabaseSync(sqliteFilename(databasePath), { readOnly: true });
      try {
        const row = db.prepare("SELECT * FROM attachments WHERE id=?").get(file.row.id);
        if (!row || db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== lease.context.brainId ||
            canonicalTaskAttachmentValue(row) !== canonicalTaskAttachmentValue(file.row) || taskAttachmentManifestSha256(file.row) !== file.row.manifest_sha256) throw new Error("The reviewed retired file custody changed during this read.");
      } finally { db.close(); }
    }
    this.options.assertActive(lease.context); signal?.throwIfAborted();
  }

  async references(brainId: string, turnId: string, signal?: AbortSignal): Promise<InheritedTaskFileReference[]> {
    const lease = this.leases.get(key(brainId, turnId)); if (!lease) return [];
    await this.recheck(lease, undefined, signal);
    for (const file of lease.closure.files) await this.recheck(lease, file, signal);
    return lease.body.files.map(file => ({ ...structuredClone(file), format: "omni-inherited-task-file-reference", version: 1,
      brainId, custody: "retired", reviewSha256: lease.receipt.reviewSha256, executionId: lease.context.executionId }));
  }

  async handles(brainId: string, turnId: string): Promise<ChatTaskAttachmentHandle[] | undefined> {
    const lease = this.leases.get(key(brainId, turnId)); if (!lease) return undefined;
    const handles = lease.closure.files.map(file => publicHandle(file.row));
    this.projection.set(handles, lease); return handles;
  }

  projectMessage(message: ChatMessage, handles: readonly ChatTaskAttachmentHandle[]): ChatMessage | undefined {
    const lease = this.projection.get(handles); if (!lease) return undefined;
    if (message.role !== "human") return message;
    if (![lease.context.frontier.turnId, lease.context.sourceTurnId].includes(message.turnId ?? "") ||
        message.inputReceipt && (message.inputReceipt.inputSha256 !== lease.context.inputSha256 ||
          ![lease.context.frontier.turnId, lease.context.sourceTurnId].includes(message.inputReceipt.turnId))) throw new Error("Inherited task attachment presentation changed its original accepted input.");
    const taskAttachmentIds = handles.map(handle => handle.id);
    return { ...message, taskAttachmentIds, ...(message.inputReceipt ? { inputReceipt: { ...message.inputReceipt, taskAttachmentIds } } : {}) };
  }

  async read(brainId: string, turnId: string, attachmentId: string, offset = 0, maxBytes = 64 * 1024, signal?: AbortSignal, entryPath?: string): Promise<InheritedTaskFileReadResult> {
    const lease = this.leases.get(key(brainId, turnId)), file = lease?.closure.files.find(file => file.row.id === attachmentId);
    if (!lease || !file) throw new Error("This retired file is not disclosed and adopted for the actual current inherited task.");
    if (!Number.isSafeInteger(offset) || offset < 0 || offset > file.row.bytes || !Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > 1024 * 1024) throw new Error("Inherited task file page range is invalid.");
    await this.recheck(lease, file, signal);
    const root = this.options.root(brainId), path = join(root, "attachments", "blobs", `${file.row.content_sha256}.data`);
    if (file.row.mime_type === TASK_FOLDER_MIME) {
      this.verifier.reserveMemory(maxBytes * 8 + 4 * 1024 * 1024, "Inherited saved folder page and SQLite cache");
      const result = await this.verifier.withVerifiedFile({ path, contentSha256: file.row.content_sha256, bytes: file.row.bytes,
        authorityKey: inheritedTaskFileHash({ brainId, executionId: lease.context.executionId, reviewSha256: lease.receipt.reviewSha256,
          inputSha256: lease.context.inputSha256, manifestSha256: file.row.manifest_sha256, sourceSha256: file.row.source_sha256 }) },
      () => this.recheck(lease, file, signal), opened => readTaskFolderAttachment(publicHandle(file.row), path, opened,
        offset, maxBytes, entryPath, signal), signal);
      await this.recheck(lease, file, signal);
      return { ...result, inheritedTask: { executionId: lease.context.executionId, reviewSha256: lease.receipt.reviewSha256,
        originalBinding: file.disclosure.originalBinding, custody: "retired" } };
    }
    if (entryPath !== undefined) throw new Error("Entry paths are available only inside a saved folder attachment.");
    const buffer = await this.verifier.withVerifiedFile({ path, contentSha256: file.row.content_sha256, bytes: file.row.bytes,
      authorityKey: inheritedTaskFileHash({ brainId, executionId: lease.context.executionId, reviewSha256: lease.receipt.reviewSha256,
        inputSha256: lease.context.inputSha256, manifestSha256: file.row.manifest_sha256, sourceSha256: file.row.source_sha256 }) },
    () => this.recheck(lease, file, signal), async opened => {
      const length = Math.min(maxBytes, file.row.bytes - offset); this.verifier.reserveMemory(length * 8 + 65536, "Inherited task file page and encoding");
      const bytes = Buffer.alloc(length); let returned = 0;
      while (returned < length) { signal?.throwIfAborted(); const read = await opened.read(bytes, returned, length - returned, offset + returned);
        if (!read.bytesRead) throw new Error("Inherited retired file ended before its exact requested page."); returned += read.bytesRead; }
      return bytes;
    }, signal);
    await this.recheck(lease, file, signal);
    let bytes = buffer, encoding: "utf8" | "base64" = "base64", content = bytes.toString("base64");
    if (file.row.mime_type.startsWith("text/") || ["application/json", "application/x-ndjson", "application/xml", "application/javascript"].includes(file.row.mime_type)) {
      for (let trim = 0; trim <= Math.min(3, Math.max(0, bytes.length - 1)); trim++) {
        try { const candidate = bytes.subarray(0, bytes.length - trim); content = new TextDecoder("utf-8", { fatal: true }).decode(candidate); bytes = candidate; encoding = "utf8"; break; } catch { /* Exact binary fallback. */ }
      }
    }
    const attachment = publicHandle(file.row);
    await this.recheck(lease, file, signal);
    return { format: "omni-chat-task-attachment-read", version: 1, attachment, offset, returnedBytes: bytes.length,
      totalBytes: file.row.bytes, encoding, content, ...(offset + bytes.length < file.row.bytes ? { nextOffset: offset + bytes.length } : {}),
      inheritedTask: { executionId: lease.context.executionId, reviewSha256: lease.receipt.reviewSha256,
        originalBinding: file.disclosure.originalBinding, custody: "retired" } };
  }
}
