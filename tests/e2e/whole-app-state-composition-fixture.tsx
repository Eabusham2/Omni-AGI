import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "../../src/renderer/src/App";
import { makeDemoBrain } from "../../src/renderer/src/demo";
import { createPresetConfig, type ActionEvent, type ActionReplyAnchor, type BrainDocument, type BrainImportFailure, type BrainStorageOperationEvent,
  type BrainSummary, type BuildProgressEvent, type ChatMessage, type ChatResult, type ChatStreamEvent,
  type ChatTurnMetadata, type ConversationLedgerEntry, type OmniApi, type RuntimeJobEvent, type DatasetPreviewProgress,
  type ThoughtTrace, type LiveObservationEvent } from "../../src/shared/types";
import type { PonderSelfReport, PonderSelfReportDelta } from "../../src/shared/ponderSelfReport";
import type { WholeAppCompositionControl } from "./whole-app-audit-contract";
import "../../src/renderer/src/styles.css";

/**
 * Actual production App + its ordinary event controller, not DOM card samples.
 * The only transport is this explicitly synthetic in-memory typed bridge.
 * There is no preload, IPC, worker, model/weights, filesystem, credential,
 * provider, real Create/Load/training, tool execution, device or external effect.
 * Protocol fields describe renderer fixture events, not learned behavior.
 * Unsupported operations throw; no success-shaped catch-all/proxy is installed.
 */
const ID = "whole-app-ui-metadata-only", NAME = "UI Fixture — no model";
const HASH = "d".repeat(64), ACTION = "a".repeat(32), REPORT = "b".repeat(64);
const subscribers = <T,>() => {
  const values = new Set<(value: T) => void>();
  return { subscribe(listener: (value: T) => void) { values.add(listener); return () => { values.delete(listener); }; },
    emit(value: T) { for (const listener of values) listener(structuredClone(value)); }, size: () => values.size };
};
const stream = subscribers<ChatStreamEvent>(), actions = subscribers<ActionEvent>(), builds = subscribers<BuildProgressEvent>();
const storage = subscribers<BrainStorageOperationEvent>(), imports = subscribers<BrainDocument>(), importFailures = subscribers<BrainImportFailure>();
const jobs = subscribers<RuntimeJobEvent>(), previews = subscribers<DatasetPreviewProgress>();
const observations = subscribers<LiveObservationEvent>();
const denied: string[] = [], receipts: Array<{ operation: string; turnId?: string; detail?: string }> = [];
const unsupported = (operation: string) => async (): Promise<never> => { denied.push(operation); throw new Error(`${operation} is unavailable: synthetic UI transport has no host authority.`); };
let sequence = 0, epoch = 0;
const at = (): string => new Date(Date.UTC(2026, 9, 18, 12, 0, 0) + ++epoch * 100).toISOString();
const template = makeDemoBrain(ID, NAME, createPresetConfig("whole-brain", NAME));
let documentValue: BrainDocument = { ...template, concepts: {}, synapses: {}, ideas: [], workingMemory: [], messages: [], traces: [], trainingSources: [], journal: [],
  liquidState: { values: [], timeConstants: [], lastUpdatedAt: at() }, counters: { plasticityEvents: 0, inferenceCount: 0, consolidationCycles: 0 } };
const summary = (): BrainSummary => ({ id: ID, name: NAME, preset: documentValue.config.preset, runtime: documentValue.config.runtime,
  updatedAt: documentValue.updatedAt, concepts: 0, synapses: 0, neuralUpdates: 0, inferenceCount: 0, trainingSources: 0,
  generation: 0, rootId: ID, activeMode: false });
let ledger: ConversationLedgerEntry[] = [], savedActions: ActionEvent[] = [];
let savedPonder: PonderSelfReport | undefined;
interface Pending { turnId: string; text: string; createdAt: string; output: string; metadata?: ChatTurnMetadata; resolve(value: ChatResult): void; reject(reason: Error): void }
const pending = new Map<string, Pending>();
const prefixSha256 = async (text: string): Promise<string> => Array.from(
  new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text))),
  byte => byte.toString(16).padStart(2, "0")
).join("");
const replyAnchor = async (turn: Pending): Promise<ActionReplyAnchor> => {
  // Snapshot only the actual ordinary reply already published. Self-report
  // text never enters this public prefix or its UTF-16 offset/hash.
  const prefix = turn.output;
  return { turnId: turn.turnId, utf16Offset: prefix.length, prefixSha256: await prefixSha256(prefix) };
};
const row = (value: ChatMessage | ActionEvent, kind: "message" | "action"): void => {
  const index = ledger.findIndex(item => item.id === value.id);
  if (index >= 0) { ledger[index] = { ...ledger[index]!, ...(kind === "message" ? { message: value as ChatMessage } : { action: value as ActionEvent }) }; return; }
  ledger.push({ sequence: ledger.length + 1, id: value.id, kind, createdAt: value.createdAt, attentionEpoch: 0, payloadSha256: HASH, rowSha256: HASH,
    ...(kind === "message" ? { message: value as ChatMessage } : { action: value as ActionEvent }) });
};
const current = (turnId?: string): Pending => {
  const value = turnId ? pending.get(turnId) : [...pending.values()].at(-1);
  if (!value) throw new Error("No UI submission is registered. Use the actual composer Send first.");
  return value;
};
function emit(value: Omit<ChatStreamEvent, "id" | "brainId" | "sequence" | "createdAt"> & Record<string, unknown>): void {
  stream.emit({ ...value, id: `ui-event-${++sequence}`, brainId: ID, sequence, createdAt: at() } as ChatStreamEvent);
}
function sendAction(value: ActionEvent, turnId: string): void {
  const existing = savedActions.findIndex(item => item.id === value.id);
  if (existing >= 0) savedActions[existing] = value; else savedActions.push(value);
  row(value, "action"); emit({ type: "chat-action", turnId, actionEvent: value });
}
function text(delta: string, turnId?: string): void {
  const value = current(turnId); value.output += delta;
  emit({ type: "chat-token", turnId: value.turnId, delta, outputTokens: value.output.length, tokensPerSecond: 12 });
}
const trace = (turn: Pending): ThoughtTrace => ({ id: `ui-trace-${turn.turnId}`, createdAt: at(), input: turn.text, seed: 0, runtime: "adaptive-core",
  activatedConcepts: [], recalledIdeas: [], driveScores: { novelty: 0, coherence: 0, curiosity: 0 }, branches: 0, selectedBranch: 0,
  steps: [], note: "Synthetic UI receipt only; no model, cognition, learning or measured inference ran." });
const chat: Pick<OmniApi["chat"], "send" | "recordDeliveryReceipt" | "listPage" | "list" | "onAction" | "onStream" | "cancel" | "approveAction" | "declineAction"> = {
  async send(id, input, turnId = crypto.randomUUID(), metadata) {
    if (id !== ID) throw new Error("Foreign UI fixture identity.");
    receipts.push({ operation: "synthetic-send-registered", turnId, detail: input });
    return new Promise<ChatResult>((resolve, reject) => {
      pending.set(turnId, { turnId, text: input, createdAt: at(), output: "", metadata, resolve, reject });
    });
  },
  async recordDeliveryReceipt(id, receipt) {
    if (id !== ID) throw new Error("Foreign UI fixture receipt.");
    receipts.push({ operation: `synthetic-delivery-${receipt.state}`, turnId: receipt.turnId });
  },
  async listPage(id) { if (id !== ID) throw new Error("Foreign UI fixture ledger."); return { brainId: ID, entries: structuredClone(ledger), totalEntries: ledger.length,
    hasOlder: false, headSequence: ledger.length, headSha256: HASH }; },
  async list() { return structuredClone(documentValue.messages); },
  onAction: actions.subscribe, onStream: stream.subscribe,
  cancel: unsupported("chat.cancel"), approveAction: unsupported("chat.approveAction"), declineAction: unsupported("chat.declineAction")
};
const brain: Pick<OmniApi["brain"], "list" | "get" | "health" | "onBuild" | "onStorageOperation" | "onImported" | "onImportFailed" | "create" | "load" | "importFile" | "duplicate"> = {
  async list() { return [summary()]; },
  async get(id) { if (id !== ID) throw new Error("Foreign UI fixture metadata."); return structuredClone(documentValue); },
  async health() { return { ready: false, worker: "unavailable", protocolVersion: 1, detail: "Synthetic UI events only · no neural runtime" }; },
  onBuild: builds.subscribe, onStorageOperation: storage.subscribe, onImported: imports.subscribe, onImportFailed: importFailures.subscribe,
  create: unsupported("brain.create"), load: unsupported("brain.load"), importFile: unsupported("brain.importFile"), duplicate: unsupported("brain.duplicate")
};
const train: Pick<OmniApi["train"], "list" | "onEvent" | "cancel"> = { list: async () => [], onEvent: jobs.subscribe, cancel: unsupported("train.cancel") };
const data: Pick<OmniApi["data"], "onPreviewProgress" | "cancelPreview"> = { onPreviewProgress: previews.subscribe, cancelPreview: unsupported("data.cancelPreview") };
const windowApi: Pick<OmniApi["window"], "platform" | "setAppearance"> = {
  platform: async () => "synthetic-ui-only",
  setAppearance: async request => ({ ...request, backgroundColor: getComputedStyle(document.documentElement).backgroundColor,
    symbolColor: getComputedStyle(document.documentElement).color })
};
// Deliberately omit optional capability APIs. Absent capability is not a mock
// success. Empty modality prevents optional nested-property reads; it exposes
// no generation/capture function and cannot start a device.
const traceApi: Pick<OmniApi["trace"], "list"> = { list: async () => [] };
const catalog: Pick<OmniApi["catalog"], "hardwareProfile" | "resourcePlan"> = {
  hardwareProfile: unsupported("catalog.hardwareProfile"), resourcePlan: unsupported("catalog.resourcePlan")
};
const bridge = { brain, chat, train, data, window: windowApi, trace: traceApi,
  catalog, modality: { capabilities: unsupported("modality.capabilities"), onObservation: observations.subscribe } };
Object.defineProperty(window, "omni", { value: bridge as unknown as OmniApi, configurable: false });

const controls: WholeAppCompositionControl = {
  status: () => ({ format: "synthetic-whole-app-composition", id: ID, pendingTurns: [...pending.keys()], streamSubscribers: stream.size(), buildSubscribers: builds.size(),
    receipts: structuredClone(receipts), denied: [...denied], models: 0, hostEffects: 0 }),
  start() {
    const value = current();
    const inputReceipt = { committed: true as const, turnId: value.turnId, inputSha256: HASH, humanMessageId: `ui-human-${value.turnId}`,
      afterimageId: "synthetic-ui-input-receipt-only", createdAt: value.createdAt, attentionEpoch: 0 };
    const humanMessage: ChatMessage = { id: inputReceipt.humanMessageId, role: "human", content: value.text, createdAt: value.createdAt, turnId: value.turnId, inputReceipt };
    row(humanMessage, "message");
    emit({ type: "chat-input-accepted", turnId: value.turnId, inputReceipt, humanMessage });
    emit({ type: "chat-state", state: "started", turnId: value.turnId });
  },
  openingText() { text("Synthetic opening before Ponder. 🧭 "); },
  async ponder() {
    const value = current(), createdAt = at();
    sendAction({ id: `ui-ponder-${value.turnId}`, brainId: ID, neuralActionId: ACTION, createdAt, updatedAt: createdAt,
      chatTurnId: value.turnId, replyAnchor: await replyAnchor(value),
      state: "running", action: { kind: "ponder", source: "brain", actionId: ACTION, selectionPhase: "pre-speech", arguments: {} } }, value.turnId);
    const report: PonderSelfReportDelta = { format: "omni-ponder-self-report-delta", version: 1, reportId: REPORT, actionId: ACTION, phase: "pre-speech", chunkIndex: 0,
      byteOffset: 0, delta: "Synthetic Ponder publication for layout inspection; no model ran.", publicationComplete: false,
      modelEmitted: true, selfReportNotChainOfThought: true, hiddenPromptUsed: false, inferenceCompletedBeforePublication: true };
    savedPonder = { format: "omni-ponder-self-report", version: 1, id: REPORT, actionId: ACTION, phase: report.phase,
      text: report.delta, state: "partial", emittedBytes: new TextEncoder().encode(report.delta).byteLength,
      modelEmitted: true, selfReportNotChainOfThought: true, hiddenPromptUsed: false, inferenceCompletedBeforePublication: true };
    emit({ type: "chat-ponder-self-report", turnId: value.turnId, report });
  },
  async toolDuringPonder() {
    const value = current(), createdAt = at();
    const parent = savedActions.find(item => item.id === `ui-ponder-${value.turnId}`);
    if (!parent || parent.chatTurnId !== value.turnId || parent.state !== "running") throw new Error("A running same-turn synthetic Ponder parent is required.");
    sendAction({ id: `ui-tool-${value.turnId}`, brainId: ID, neuralActionId: "e".repeat(32), createdAt, updatedAt: createdAt,
      chatTurnId: value.turnId, parentPonderActionEventId: parent.id, replyAnchor: await replyAnchor(value),
      state: "proposed", action: { kind: "tool", source: "brain", toolId: "fixture.inspect", action: "synthetic-only", selectionPhase: "ponder", selectionStep: 1,
        arguments: { fixture: "fixture.inspect is a synthetic displayed tool. No tool, file, network or host action was performed." } } }, value.turnId);
  },
  phaseOnlyUnboundAction() {
    const value = current(), createdAt = at();
    // Negative history/transport case: a phase label, sequence and matching
    // timestamp do not authorize Ponder ancestry or a position in reply text.
    sendAction({ id: `ui-unbound-${value.turnId}`, brainId: ID, neuralActionId: "f".repeat(32), createdAt, updatedAt: createdAt,
      state: "proposed", action: { kind: "tool", source: "brain", toolId: "fixture.unbound", action: "synthetic-only", selectionPhase: "ponder", selectionStep: 1,
        arguments: { fixture: "fixture.unbound is displayed history only. No ancestry, public reply anchor or host authority." } } }, value.turnId);
  },
  requestToolApproval() {
    const value = current(), tool = savedActions.find(item => item.id === `ui-tool-${value.turnId}`);
    if (!tool) throw new Error("Emit the synthetic Ponder-selected tool first.");
    sendAction({ ...tool, state: "approval-required", updatedAt: at(), execution: { id: "synthetic-exact-approval-only", toolId: "fixture.inspect", action: "synthetic-only",
      state: "approval-required", startedAt: tool.createdAt, approvalToken: "synthetic-no-authority", approvalExpiresAt: new Date(Date.now() + 3_600_000).toISOString() } }, value.turnId);
  },
  resumeText() {
    const value = current(); const ponder = savedActions.find(item => item.id === `ui-ponder-${value.turnId}`);
    if (ponder) sendAction({ ...ponder, state: "complete", updatedAt: at() }, value.turnId);
    text("Synthetic continuation after Ponder. ");
  },
  async finish(turnId) {
    const value = current(turnId), createdAt = at();
    const humanMessage: ChatMessage = { id: `ui-human-${value.turnId}`, role: "human", content: value.text, turnId: value.turnId, createdAt: value.createdAt };
    const brainMessage: ChatMessage = { id: `ui-reply-${value.turnId}`, role: "brain", content: value.output || "Synthetic completed UI reply; no model ran.", turnId: value.turnId,
      createdAt, runtime: "adaptive-core", status: "complete", ...(savedPonder ? { ponderSelfReports: [savedPonder] } : {}) };
    for (const event of [...savedActions]) {
      const anchor = event.replyAnchor;
      if (!anchor || event.chatTurnId !== value.turnId) continue;
      if (anchor.turnId !== value.turnId || !Number.isSafeInteger(anchor.utf16Offset) || anchor.utf16Offset < 0 ||
          anchor.utf16Offset > brainMessage.content.length ||
          await prefixSha256(brainMessage.content.slice(0, anchor.utf16Offset)) !== anchor.prefixSha256) {
        throw new Error("Synthetic reply binding does not match the exact canonical public prefix.");
      }
      sendAction({ ...event, replyAnchor: { ...anchor, messageId: brainMessage.id }, updatedAt: at() }, value.turnId);
    }
    documentValue = { ...documentValue, messages: [...documentValue.messages, humanMessage, brainMessage], updatedAt: createdAt };
    row(humanMessage, "message"); row(brainMessage, "message");
    emit({ type: "chat-reply-committed", turnId: value.turnId, humanMessage, brainMessage, pendingActions: 0 });
    value.resolve({ brain: structuredClone(documentValue), humanMessage, brainMessage, trace: trace(value), actionEvents: structuredClone(savedActions), turnMetadata: value.metadata });
    emit({ type: "chat-state", turnId: value.turnId, state: "complete" }); pending.delete(value.turnId);
  },
  fail() { const value = current(); emit({ type: "chat-state", turnId: value.turnId, state: "failed", error: "Synthetic transport failure · no work ran" });
    value.reject(new Error("Synthetic transport failure · no work ran")); pending.delete(value.turnId); },
  training(phase) {
    const progress = { extracting: .12, learning: .65, saving: .98, finalizing: .999, complete: 1 }[phase];
    builds.emit({ sequence: ++sequence, phase: phase === "complete" ? "complete" : phase === "extracting" || phase === "learning" ? "initial-materials" : "packing",
      progress, label: `Synthetic ${phase} overlay · no training`, elapsedMs: 12_345,
      ...(phase === "finalizing" ? {} : { etaMs: 34_567 }), data: { recordsVisited: 321, recordsTotal: 9876 } });
  },
  storage(state) { storage.emit({ schemaVersion: 1, operationId: "ui-storage-layout-only", kind: "duplicate", state, phase: "synthetic-layout",
    label: `Synthetic ${state} storage notification · no data copied`, filesCompleted: 3, filesTotal: 9, logicalBytesCompleted: 1234, logicalBytesTotal: 4321,
    physicalBytesAdded: 0, sharedBytes: 0, bytesPerSecond: 123, elapsedMs: 12_345, etaMs: 34_567, updatedAt: at() }); },
  notification() { importFailures.emit({ fileName: "synthetic-layout-only.omni", message: "Synthetic long notification for overlay layout inspection. ".repeat(12) }); }
};
window.omniWholeAppComposition = controls;

// This fixture intentionally exposes no frame preload/IPC, endpoints or keys.
// The test/readable synthetic event descriptions are the complete data source.
createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
