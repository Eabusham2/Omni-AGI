import { cleanChatInput } from "../shared/chatInput";
import { join } from "node:path";
import { checkedChatInputProvenance, type NativeAppChatInput } from "../shared/chatInputProvenance";
import type { ChatResult } from "../shared/types";
import type { NativeAppObservationFrontier, NativeAppObservationInput, NativeAppObservationResult } from "../shared/nativeAppObservation";
import type { McpAppObservationEvent } from "../shared/mcpApps";
import { McpAppArtifactStore, type McpAppFollowup } from "./mcpAppArtifacts";
import { nativeAppObservationRawSha256 } from "./nativeAppObservation";
import { BackgroundRequestDeferredError, EngineRequestError } from "./engineSupervisor";
import { mcpAppTaskData } from "./mcpAppTaskData";

export function appFollowupInput(value: McpAppFollowup): { input: string; turnId: string; appInput: NativeAppChatInput } {
  const { claim, capture, native } = value, data = JSON.parse(claim.dataJson), checked = mcpAppTaskData(data.method, data.params);
  if (checked.method !== "ui/message") throw new Error("A context observation is not a task follow-up.");
  const blocks = Array.isArray(checked.params.content) ? checked.params.content : [checked.params.content];
  if (!blocks.length || blocks.some(block => block.type !== "text")) throw new Error("App task follow-ups currently require reviewed text blocks. Non-text data was not replaced with an invented prompt.");
  const input = cleanChatInput(blocks.map(block => block.type === "text" ? block.text : "").join("\n")), inputSha256 = nativeAppObservationRawSha256(input);
  if (native.head.state !== "observed" || native.head.recordSha256 !== claim.recordSha256 || native.head.dataSha256 !== claim.record.content.sha256 ||
    native.head.learning?.textSha256 !== inputSha256) throw new Error("App follow-up body does not match its genuinely admitted native experience.");
  const record = claim.record, provenance = checkedChatInputProvenance({ kind: "mcp-app", sourceBrainId: record.source.brainId,
    serverId: record.source.serverId, serverLabel: record.source.serverLabel, endpoint: record.source.endpoint, resourceUri: capture.resourceUri,
    artifactId: record.artifactId, captureSha256: record.captureSha256, originalCallBrainId: capture.source.brainId, toolId: capture.tool.id,
    reviewId: record.reviewId, observationId: record.id, streamId: record.streamId, ordinal: record.ordinal,
    recordSha256: claim.recordSha256, dataSha256: record.content.sha256, nativeFrontierSha256: native.frontierSha256, inputSha256 });
  return { input, turnId: `app-${record.id}`, appInput: { recordJson: claim.recordJson, recordSha256: claim.recordSha256,
    dataJson: claim.dataJson, dataSha256: record.content.sha256, captureJson: value.captureJson, captureSha256: record.captureSha256,
    nativeFrontierSha256: native.frontierSha256, provenance } };
}

export interface McpAppObservationDeliveryOptions {
  store(brainId: string): McpAppArtifactStore;
  directory(brainId: string): string;
  requireLoaded(brainId: string): void;
  requireSource(record: import("../shared/mcpAppArtifacts").McpAppObservationArtifact): void;
  withWriter<T>(brainId: string, action: () => Promise<T>, signal: AbortSignal): Promise<T>;
  frontier(brainId: string, storagePath: string, signal: AbortSignal, beforeDispatch: () => void): Promise<NativeAppObservationFrontier>;
  admit(input: NativeAppObservationInput, storagePath: string, signal: AbortSignal, beforeDispatch: () => void): Promise<NativeAppObservationResult>;
  followup(brainId: string, input: string, turnId: string, proof: NativeAppChatInput, signal: AbortSignal): Promise<ChatResult>;
  ownerReleased?(brainId: string): Promise<void>;
  publish(event: McpAppObservationEvent): void;
}

/** Event-driven background FIFO. Inline MCP calls only enqueue/review; they
 * never await reacquiring the neural writer which may own that same tool. */
export class McpAppObservationDelivery {
  private readonly active = new Map<string, { task: Promise<void>; controller: AbortController; again: boolean }>();
  private readonly waiting = new Set<string>();
  private closed = false;
  constructor(private readonly options: McpAppObservationDeliveryOptions) {}
  notify(brainId: string): void {
    if (this.closed) return;
    this.waiting.add(brainId);
    const running = this.active.get(brainId); if (running) { running.again = true; return; }
    const owner = { task: Promise.resolve(), controller: new AbortController(), again: false };
    this.active.set(brainId, owner);
    owner.task = this.drain(brainId, owner.controller.signal).catch(() => undefined).finally(() => {
      if (this.active.get(brainId) === owner) this.active.delete(brainId);
      if (owner.again && !this.closed) this.notify(brainId);
    });
  }
  retryWaiting(): void {
    for (const id of this.waiting) {
      if (this.active.has(id)) continue;
      try { this.options.requireLoaded(id); this.notify(id); } catch { /* explicit Load remains required */ }
    }
  }
  private async drain(brainId: string, signal: AbortSignal): Promise<void> {
    const store = this.options.store(brainId); let artifactId = "";
    try {
      // Deliberately no waitUntilLoaded/autoload here. Explicit Load events
      // notify again; copied/imported queues have no active stream to execute.
      this.options.requireLoaded(brainId);
      for (;;) {
        if (this.closed) return;
        signal.throwIfAborted();
        const followup = await store.nextFollowup();
        if (followup) {
          artifactId = followup.claim.record.artifactId; this.options.requireLoaded(brainId); this.options.requireSource(followup.claim.record);
          if (followup.intent) throw new Error(`This App follow-up has a prior ${followup.intent.state} dispatch receipt. Resume or inspect the original turn; uncertain effects were not replayed.`);
          const task = appFollowupInput(followup);
          await store.reserveFollowup(followup.claim.recordSha256, task.turnId, task.appInput.provenance.inputSha256);
          try {
            const result = await this.options.followup(brainId, task.input, task.turnId, task.appInput, signal);
            if (result.humanMessage.content !== task.input || result.humanMessage.turnId !== task.turnId ||
              JSON.stringify(checkedChatInputProvenance(result.humanMessage.inputProvenance)) !== JSON.stringify(task.appInput.provenance)) throw new Error("App follow-up has no exact source-bound native task receipt.");
            await store.settleFollowup(followup.claim.recordSha256, "complete");
          } catch (error) { await store.settleFollowup(followup.claim.recordSha256, "failed", error instanceof Error ? error.message : String(error)); throw error; }
          continue;
        }
        const progressed = await this.options.withWriter(brainId, async () => {
          this.options.requireLoaded(brainId); signal.throwIfAborted();
          const claim = await store.claimObservation(); if (!claim) return false;
          artifactId = claim.record.artifactId;
          const boundary = (): void => { this.options.requireLoaded(brainId); this.options.requireSource(claim.record); };
          boundary();
          const frontier = await this.options.frontier(brainId, this.options.directory(brainId), signal, boundary);
          const expectedFrontierSha256 = await store.bindObservationFrontier(claim.recordSha256, frontier.frontierSha256);
          const previousAudit = await store.archiveObservationAudit(frontier.head, join(this.options.directory(brainId), "engine"));
          const previousAuditArchivedSha256 = await store.bindObservationArchive(claim.recordSha256, previousAudit);
          boundary();
          let result: NativeAppObservationResult;
          try {
            result = await this.options.admit({ brainId, recordJson: claim.recordJson, recordSha256: claim.recordSha256,
              dataJson: claim.dataJson, dataSha256: claim.record.content.sha256, expectedFrontierSha256, disposition: claim.disposition,
              ...(previousAuditArchivedSha256 ? { previousAuditArchivedSha256 } : {}) },
              this.options.directory(brainId), signal, boundary);
          } catch (error) {
            const data = error instanceof EngineRequestError && error.data && typeof error.data === "object" ? error.data as Record<string, unknown> : undefined;
            if (error instanceof EngineRequestError && error.code === -32031 && data?.nativeAppObservation === true && data.nativeOwnerReleased === true && data.reloadRequired === true && data.brainId === brainId) await this.options.ownerReleased?.(brainId);
            throw error;
          }
          try {
            await store.archiveObservationAudit(result.head, join(this.options.directory(brainId), "engine"));
            const receipt = await store.acknowledgeObservation(claim.recordSha256, result);
            this.options.publish({ brainId, artifactId, receipt });
          } finally {
            // A full disk/corrupt audit cannot leave a worker-confirmed released
            // owner falsely resident in the host just because its ACK failed.
            if (result.reloadRequired) {
              if (!this.options.ownerReleased) throw new Error("Native owner was released; explicit Load is required before any follow-up.");
              await this.options.ownerReleased(brainId);
            }
          }
          return true;
        }, signal);
        if (!progressed) { this.waiting.delete(brainId); return; }
      }
    } catch (error) {
      if (error instanceof BackgroundRequestDeferredError || signal.aborted) return;
      const data = error instanceof EngineRequestError && error.data && typeof error.data === "object" ? error.data as Record<string, unknown> : undefined;
      // Foreground preemption stops only the native read phase. Keep this
      // exact durable claim waiting; a later ready event retries it once.
      if (error instanceof EngineRequestError && error.code === -32800 && data?.brainId === brainId && data.nativeAppObservation === true &&
        data.cancelled === true && data.safeBoundary === true && data.learningApplied === false && data.nativeOwnerReleased === false) return;
      if (artifactId) this.waiting.delete(brainId);
      if (artifactId) this.options.publish({ brainId, artifactId, error: error instanceof Error ? error.message : String(error) });
      // Nothing is dropped, fabricated as learned, or retried in a timer loop.
    }
  }
  async settle(brainId: string): Promise<void> { await this.active.get(brainId)?.task; }
  // Closing the host is not Stop/undo. The ordinary lifecycle checkpoint owns
  // any live App-origin chat; do not turn a saved suspension into cancellation.
  close(): void { this.closed = true; }
}
