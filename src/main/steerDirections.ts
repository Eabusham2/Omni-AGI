import { lstat } from "node:fs/promises";
import { join } from "node:path";
import type { ChatInputAcceptedReceipt, ChatMessage, ChatResult, ChatTurnMetadata } from "../shared/types";
import type { NativeAudioInput } from "../shared/nativeAudioChat";
import type { SteerDirectionDismissRequest, SteerDirectionEvent, SteerDirectionPageQuery, SteerDirectionRecord, SteerDirectionSource, SteerDirectionState } from "../shared/steerDirections";
import { cleanChatInput } from "../shared/chatInput";
import { chatTurnKey } from "../shared/chatTurnKey";
import { directionCanonical, directionHash, directionId, SteerDirectionStore } from "./steerDirectionStore";

export interface SteerDirectionTracker {
  received(receipt: ChatInputAcceptedReceipt, human: ChatMessage): void;
  completed(result: ChatResult): void;
  terminal(state: "failed" | "stopped" | "steered"): void;
  detached(): void;
  close(): void;
}
export interface SteerDirectionsOptions {
  brainDirectory(brainId: string): Promise<string> | string;
  reserveWrite?(root: string, bytes: number): Promise<void>;
  admit?(bytes: number): Promise<void>;
  onError?(error: unknown, brainId: string): void;
}
/** Main-owned presentation history. No worker/tool/learning/dispatch API is held. */
export class SteerDirectionsService {
  private readonly listeners = new Set<(event: SteerDirectionEvent) => void>();
  private readonly active = new Set<string>();
  private readonly recovered = new Set<string>();
  constructor(private readonly options: SteerDirectionsOptions) {}
  subscribe(listener: (event: SteerDirectionEvent) => void): () => void { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; }
  private publish(row: SteerDirectionRecord, type: SteerDirectionEvent["type"] = "changed"): void {
    const event: SteerDirectionEvent = { brainId: row.brainId, id: row.id, sequence: row.sequence, revision: row.revision, state: row.state, type, at: row.updatedAt };
    for (const listener of this.listeners) { try { listener(event); } catch (error) { this.options.onError?.(error, row.brainId); } }
  }
  private async root(brainId: string): Promise<string> {
    directionId(brainId); const root = await this.options.brainDirectory(brainId);
    for (const path of [root, join(root, "brain.json")]) { const info = await lstat(path); if (info.isSymbolicLink() || (path === root ? !info.isDirectory() : !info.isFile())) throw new Error("Direction history requires an existing managed instance."); }
    return root;
  }
  private async recover(brainId: string, root: string): Promise<void> {
    if (this.recovered.has(brainId)) return;
    const store = await SteerDirectionStore.open(root, brainId, false); if (!store) { this.recovered.add(brainId); return; }
    let count = 0;
    try { for (const row of store.db.prepare("SELECT id,turn_id FROM directions WHERE json_extract(view_json,'$.state') IN ('pending','received') AND json_extract(view_json,'$.dormant') IS NULL").iterate()) {
      if (!this.active.has(chatTurnKey(brainId, String(row.turn_id)))) count += 1;
    } } finally { store.close(); }
    if (count) {
      await this.options.reserveWrite?.(root, count * 4096 + 65536);
      const writer = (await SteerDirectionStore.open(root, brainId, true))!;
      try { for (const row of writer.db.prepare("SELECT id FROM directions WHERE json_extract(view_json,'$.state') IN ('pending','received') AND json_extract(view_json,'$.dormant') IS NULL ORDER BY sequence").iterate()) this.publish(writer.update(String(row.id), view => {
        if (!this.active.has(chatTurnKey(brainId, view.turnId))) view.dormant = { reason: "process-interrupted", fromBrainId: brainId, at: new Date().toISOString() }; return view;
      })); } finally { writer.close(); }
    }
    this.recovered.add(brainId);
  }
  async list(brainId: string, query?: SteerDirectionPageQuery) {
    const root = await this.root(brainId); await this.recover(brainId, root); const store = await SteerDirectionStore.open(root, brainId);
    try { return store?.list(query) ?? { entries: [], total: 0 }; } finally { store?.close(); }
  }
  async get(brainId: string, id: string): Promise<SteerDirectionRecord | null> {
    directionId(id); const root = await this.root(brainId); await this.recover(brainId, root); const store = await SteerDirectionStore.open(root, brainId);
    try { return store?.get(id) ?? null; } finally { store?.close(); }
  }
  async forTurns(brainId: string, turnIds: string[]): Promise<SteerDirectionRecord[]> {
    if (!Array.isArray(turnIds) || turnIds.length > 100 || new Set(turnIds).size !== turnIds.length) throw new Error("Direction turn lookup requires at most 100 distinct visible turn IDs.");
    turnIds.forEach(directionId); const root = await this.root(brainId); await this.recover(brainId, root); const store = await SteerDirectionStore.open(root, brainId);
    try { const bytes = turnIds.reduce((sum, turnId) => sum + (store?.bytesForTurn(turnId) ?? 0), 0); await this.options.admit?.(bytes * 4);
      return [...new Map(turnIds.flatMap(turnId => store?.forTurn(turnId) ?? []).map(record => [record.id, record])).values()]; } finally { store?.close(); }
  }
  async dismiss(request: SteerDirectionDismissRequest): Promise<SteerDirectionRecord> {
    if (!request || typeof request !== "object" || Object.keys(request).some(key => !["brainId", "id", "expectedRevision", "dismissed"].includes(key))) throw new Error("Invalid direction display request.");
    directionId(request.id); const root = await this.root(request.brainId); await this.options.reserveWrite?.(root, 65536);
    const store = await SteerDirectionStore.open(root, request.brainId, true);
    try { const row = store!.dismiss(request.id, request.expectedRevision, request.dismissed); this.publish(row, "dismissed"); return row; } finally { store?.close(); }
  }
  /** The controller invokes this before dispatch, not the renderer or a model tool. */
  async begin(brainId: string, turnId: string, metadata: ChatTurnMetadata | undefined, input: string,
    audio?: NativeAudioInput, continuing = false): Promise<SteerDirectionTracker | undefined> {
    if (metadata?.kind !== "steer" && !continuing) return undefined;
    const root = await this.root(brainId), key = chatTurnKey(brainId, turnId);
    if (this.active.has(key)) throw new Error("This direction already has an active host tracker.");
    let source: SteerDirectionSource | undefined;
    if (metadata?.kind === "steer") {
      const content = audio ? "" : cleanChatInput(input);
      source = { brainId, turnId, predecessorTurnId: metadata.replacesTurnId, createdAt: metadata.createdAt, content,
        ...(input !== content ? { submittedContent: input } : {}), attachmentIds: [...(metadata.attachmentIds ?? [])], inputKind: audio ? "native-audio" : "text",
        ...(audio ? { audio: { inputSha256: audio.inputSha256, attachmentId: audio.attachment.attachmentId, sourceSha256: audio.attachment.contentSha256,
          sourceBytes: audio.attachment.bytes, mimeType: audio.attachment.mimeType, utteranceId: audio.capture.utteranceId, sequence: audio.capture.sequence, capturedAt: audio.capture.capturedAt } } : {}) };
      const bytes = Buffer.byteLength(directionCanonical(source)); await this.options.admit?.(bytes * 4 + 65536); await this.options.reserveWrite?.(root, bytes + 65536);
    }
    let store = await SteerDirectionStore.open(root, brainId, !!source);
    if (!store) return undefined;
    let row: SteerDirectionRecord;
    try {
      if (source) { row = store.create(source); if (row.dormant) throw new Error("This copied direction is historical. Revise and steer with a new turn; it will not be resent automatically."); }
      else { const candidates = store.forTurn(turnId).filter(record => record.source.content === input || record.source.audio !== undefined && input === "");
        if (!candidates.length) { store.close(); return undefined; } if (candidates.length !== 1) throw new Error("Saved steering has ambiguous host provenance; it was not replayed."); row = candidates[0]!;
        store.close(); await this.options.reserveWrite?.(root, 65536); store = (await SteerDirectionStore.open(root, brainId, true))!;
      }
    } catch (error) { store.close(); throw error; }
    if (this.active.has(key)) { store.close(); throw new Error("This direction acquired a host tracker while its history was opening."); }
    this.active.add(key); this.publish(row); let closed = false;
    const update = (work: Parameters<SteerDirectionStore["update"]>[1]): void => {
      if (closed) throw new Error("Direction tracker is closed."); const next = store!.update(row.id, work); if (next.revision !== row.revision) { row = next; this.publish(row); }
    };
    const received = (receipt: ChatInputAcceptedReceipt, human: ChatMessage): void => {
      if (human.role !== "human" || human.id !== receipt.humanMessageId || human.content !== row.source.content ||
          ![row.turnId, row.source.turnId].includes(receipt.turnId)) throw new Error("Steering acceptance belongs to another actual input.");
      const ids = [...new Set([...(human.taskAttachmentIds ?? receipt.taskAttachmentIds ?? []), ...(row.source.audio ? [row.source.audio.attachmentId] : [])])];
      update(view => { const actual = { ...receipt, ...(ids.length ? { taskAttachmentIds: ids } : {}) };
        if (view.inputReceipt && directionCanonical(view.inputReceipt) !== directionCanonical(actual)) throw new Error("An accepted steering receipt cannot be replaced.");
        view.inputReceipt ??= actual; view.acceptedAttachmentIds ??= ids; view.humanMessageId ??= human.id; view.humanTurnId ??= human.turnId ?? receipt.turnId;
        if (view.state === "pending") view.state = "received"; delete view.dormant; return view;
      });
    };
    const safely = (work: () => void): void => { try { work(); } catch (error) {
      this.options.onError?.(error, brainId);
      for (const listener of this.listeners) listener({ brainId, id: row.id, sequence: row.sequence, revision: row.revision, state: row.state,
        type: "unavailable", at: new Date().toISOString(), detail: "Steering display history could not be updated. The actual turn and learned input are unchanged." });
    } };
    return {
      received: (receipt, human) => safely(() => received(receipt, human)),
      completed: result => safely(() => {
        const human = result.humanMessage, reply = result.brainMessage;
        if (human.role !== "human" || human.content !== row.source.content || reply.role !== "brain" || !human.id || !reply.id ||
            row.humanMessageId !== undefined && row.humanMessageId !== human.id ||
            row.humanTurnId !== undefined && human.turnId !== undefined && row.humanTurnId !== human.turnId ||
            human.turnId !== undefined && ![row.turnId, row.source.turnId].includes(human.turnId) ||
            reply.turnId !== undefined && ![row.turnId, row.source.turnId].includes(reply.turnId)) throw new Error("Completed steering is not bound to the actual saved human/reply pair.");
        if (!row.inputReceipt && human.inputReceipt) received(human.inputReceipt, human);
        update(view => { view.humanMessageId ??= human.id; view.humanTurnId ??= human.turnId ?? view.turnId; view.brainMessageId ??= reply.id;
          const actual = { humanMessageId: human.id, brainMessageId: reply.id, createdAt: reply.createdAt, ...(result.generationEnd ? { generationEnd: result.generationEnd } : {}) };
          if (view.reply && directionCanonical(view.reply) !== directionCanonical(actual)) throw new Error("A saved steering reply cannot be rewritten.");
          view.reply ??= actual; view.state = result.generationEnd === "steered" ? "steered" : result.generationEnd === "native-stop" ? "stopped" : "complete";
          delete view.dormant; return view;
        });
      }),
      terminal: state => safely(() => { update(view => {
        if (view.reply) { if (state === "failed") view.failure = "post-reply-work-failed"; return view; }
        view.state = state; view.failure = state === "failed" ? "dispatch-failed" : state === "stopped" ? "stopped" : undefined;
        if (view.failure === undefined) delete view.failure; return view;
      }); }),
      detached: () => safely(() => { update(view => { if (!view.reply) view.dormant = {
        reason: "saved-continuation", fromBrainId: brainId, at: new Date().toISOString() }; return view; }); }),
      close: () => { if (closed) return; closed = true; this.active.delete(key); store!.close(); }
    };
  }
  /** A recovery boundary may replace files only after controller detach/close. */
  forget(brainId: string): void { if ([...this.active].some(key => JSON.parse(key)[0] === brainId)) throw new Error("A steering tracker is still active during recovery."); this.recovered.delete(brainId); }
}
