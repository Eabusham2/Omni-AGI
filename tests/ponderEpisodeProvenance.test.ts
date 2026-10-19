import { createHash } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it, vi } from "vitest";
import { normalizeStructuredAction } from "../src/main/actionProtocol";
import { BrainService, normalizeChatEngineEvent, type NeuralChatStreamEvent } from "../src/main/brainService";
import { ChatActionController } from "../src/main/chatActionController";
import { ConversationLedger } from "../src/main/conversationLedger";
import { PublicReplyAnchors, normalizePublicReplyTextBinding } from "../src/main/publicReplyAnchors";
import type { NativePonderEpisodeFrame } from "../src/shared/ponderEpisode";
import type { ActionEvent, ChatMessage, ChatResult, ChatStreamEvent, PublicReplyTextBinding, StructuredAction } from "../src/shared/types";

const hash = (text: string): string => createHash("sha256").update(text).digest("hex");
const parentId = "a".repeat(32), childId = "b".repeat(32);
const now = "2026-10-18T12:00:00.000Z";
const parent = (actionId = parentId, step = 3): StructuredAction => ({ kind: "ponder", source: "brain", actionId,
  selectionPhase: "mid-generation", selectionStep: step, arguments: { organic: true } });
const child = (parentPonderActionId: string | undefined = parentId): StructuredAction => ({ kind: "tool", source: "brain", actionId: childId,
  toolId: "web.search", action: "search", arguments: { query: "inert selected effect" }, ...(parentPonderActionId ? { parentPonderActionId } : {}) });
const finished = (action = parent(), reason = "learned-action-transition"): StructuredAction => ({ ...action, arguments: { ...action.arguments,
  handledInTurn: true, completedInTurn: ["converged", "learned-action-transition"].includes(reason), ponderTrace: { activated: true,
    activated_by: "learned-action-head", private: true, visibleMagicTags: false, completed: ["converged", "learned-action-transition"].includes(reason),
    passes: 2, elapsed_ms: 5, final_delta: .1, stop_reason: reason, phase: action.selectionPhase } } });
const frame = (state: NativePonderEpisodeFrame["state"], action = state === "started" ? parent() : finished()): NeuralChatStreamEvent =>
  ({ type: "chat-ponder-episode", sequence: 1, state, actionId: action.actionId!, action });
function result(content = "Opening. End.", actions: StructuredAction[] = []): ChatResult {
  return { brain: { messages: [] } as unknown as ChatResult["brain"], humanMessage: { id: "human", role: "human", turnId: "turn", content: "input", createdAt: now },
    brainMessage: { id: "reply", role: "brain", turnId: "turn", content, createdAt: now }, trace: { id: "trace" } as ChatResult["trace"], proposedActions: actions };
}
const token = (delta: string): NeuralChatStreamEvent => ({ type: "chat-token", sequence: 1, delta });
function effect() {
  return { id: "receipt", toolId: "web.search", action: "search", state: "complete" as const, startedAt: now, finishedAt: now, output: { sources: [] } };
}

describe("new native Ponder episode provenance (inert host transport)", () => {
  it("normalizes explicit lifecycle frames and rejects foreign owner, kind and unmeasured completion", () => {
    const raw = { type: "chat-ponder-episode", brainId: "brain", sequence: 1, actionId: parentId,
      data: { state: "started", actionId: parentId, action: parent() } };
    expect(normalizeChatEngineEvent(raw, "brain")).toMatchObject({ type: "chat-ponder-episode", state: "started", actionId: parentId });
    expect(normalizeChatEngineEvent(raw, "foreign")).toBeUndefined();
    expect(normalizeChatEngineEvent({ ...raw, data: { ...raw.data, action: child() } }, "brain")).toBeUndefined();
    expect(normalizeChatEngineEvent({ ...raw, data: { ...raw.data, state: "finished" } }, "brain")).toBeUndefined();
    expect(normalizeChatEngineEvent({ ...raw, data: { ...raw.data, state: "finished", action: finished() } }, "brain"))
      .toMatchObject({ state: "finished" });
    const incomplete = finished(parent(), "failed");
    incomplete.arguments.ponderTrace = { ...(incomplete.arguments.ponderTrace as object), measurementIncomplete: true, passes: null, final_delta: null };
    expect(normalizeChatEngineEvent({ ...raw, data: { ...raw.data, state: "finished", action: incomplete } }, "brain"))
      .toMatchObject({ state: "finished", action: { arguments: { ponderTrace: { completed: false, passes: null } } } });
    expect(normalizeStructuredAction({ ...child(), arguments: { parentPonderActionId: parentId } }, "human"))
      .not.toHaveProperty("parentPonderActionId");
  });

  it("links only a selected active native child; the parent finishes before a held effect and no idle cycle or duplicate dispatch runs", async () => {
    let settle!: () => void, inspected!: () => void;
    const effectHeld = new Promise<void>(resolve => { settle = resolve; });
    const episodeClosed = new Promise<void>(resolve => { inspected = resolve; });
    const seen: ChatStreamEvent[] = [];
    const persisted: ActionEvent[][] = [];
    const service = { chat: vi.fn(async (_id: string, _text: string, _signal: AbortSignal | undefined,
      emit: ((value: NeuralChatStreamEvent) => void) | undefined) => {
      emit?.(token("Opening. ")); emit?.(frame("started"));
      emit?.({ type: "chat-action", sequence: 2, actionId: childId, action: child() });
      emit?.(frame("finished")); inspected();
      emit?.(token("End.")); return result(undefined, [finished(), child()]);
    }), idleCycle: vi.fn(), recordConversationActions: vi.fn(async (_id: string, events: ActionEvent[]) => { persisted.push(structuredClone(events)); }),
      learnStructuredExperience: vi.fn(async () => ({})) };
    const tools = { execute: vi.fn(async () => { await effectHeld; return effect(); }), cancel: vi.fn(() => 0) };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    controller.on("stream", value => seen.push(structuredClone(value)));
    const sending = controller.send("brain", "input", undefined, "turn");
    await episodeClosed;
    const parentComplete = seen.find(value => value.type === "chat-action" && value.actionEvent.action.kind === "ponder" && value.actionEvent.state === "complete");
    expect(parentComplete).toBeDefined(); expect(tools.execute).toHaveBeenCalledOnce();
    settle(); const actual = await sending;
    expect(actual.actionEvents).toHaveLength(2); expect(tools.execute).toHaveBeenCalledOnce(); expect(service.idleCycle).not.toHaveBeenCalled();
    const [p, c] = actual.actionEvents!;
    expect(c).toMatchObject({ chatTurnId: "turn", parentPonderActionEventId: p!.id, neuralActionId: childId, state: "complete" });
    expect(p).toMatchObject({ state: "complete", chatTurnId: "turn", replyAnchor: { turnId: "turn", messageId: "reply", utf16Offset: 9, prefixSha256: hash("Opening. ") } });
    expect(c!.replyAnchor).toEqual(p!.replyAnchor);
    expect(persisted.at(-1)?.[1]?.parentPonderActionEventId).toBe(p!.id);
  });

  it("keeps repeated genuine episodes distinct and post-Ponder actions flat without inferring adjacency", async () => {
    const second = parent("c".repeat(32), 7);
    const service = { chat: vi.fn(async (_id: string, _text: string, _signal: AbortSignal | undefined,
      emit: ((value: NeuralChatStreamEvent) => void) | undefined) => {
      emit?.(frame("started")); emit?.(frame("finished"));
      emit?.(frame("started", second)); emit?.(frame("finished", finished(second)));
      emit?.({ type: "chat-action", sequence: 6, actionId: childId, action: child("") });
      emit?.(token("Opening. End.")); return result(undefined, [finished(), finished(second), child("")]);
    }), idleCycle: vi.fn(), learnStructuredExperience: vi.fn(async () => ({})) };
    const tools = { execute: vi.fn(async () => effect()), cancel: vi.fn(() => 0) };
    const actual = await new ChatActionController(service, tools, { start: vi.fn() }).send("brain", "input", undefined, "turn");
    expect(actual.actionEvents).toHaveLength(3); expect(tools.execute).toHaveBeenCalledOnce();
    expect(actual.actionEvents?.[2]).not.toHaveProperty("parentPonderActionEventId"); expect(service.idleCycle).not.toHaveBeenCalled();
  });

  it.each(["unknown", "finished"])("rejects %s native parent before any effect", async owner => {
    const service = { chat: vi.fn(async (_id: string, _text: string, _signal: AbortSignal | undefined,
      emit: ((value: NeuralChatStreamEvent) => void) | undefined) => {
      if (owner === "finished") { emit?.(frame("started")); emit?.(frame("finished")); }
      emit?.({ type: "chat-action", sequence: 2, actionId: childId, action: child() }); return result();
    }) };
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) };
    await expect(new ChatActionController(service, tools, { start: vi.fn() }).send("brain", "input", undefined, "turn"))
      .rejects.toThrow("active same-turn Ponder");
    expect(tools.execute).not.toHaveBeenCalled();
  });

  it("does not grant a selected child more than the unchanged Ask permission gate", async () => {
    const service = { chat: vi.fn(async (_id: string, _text: string, _signal: AbortSignal | undefined,
      emit: ((value: NeuralChatStreamEvent) => void) | undefined) => {
      emit?.(frame("started")); emit?.({ type: "chat-action", sequence: 2, actionId: childId, action: child() });
      emit?.(frame("finished")); emit?.(token("Opening. End.")); return result(undefined, [finished(), child()]);
    }) };
    const tools = { execute: vi.fn(async () => ({ ...effect(), state: "approval-required" as const })), cancel: vi.fn(() => 0) };
    const actual = await new ChatActionController(service, tools, { start: vi.fn() }).send("brain", "input", undefined, "turn");
    expect(actual.actionEvents?.[1]).toMatchObject({ state: "approval-required", parentPonderActionEventId: actual.actionEvents?.[0]?.id });
    expect(tools.execute).toHaveBeenCalledOnce();
    expect(tools.execute.mock.calls[0]).toBeDefined();
  });

  it("resumes past committed episode/effects without redispatch or inventing a final-only anchor", async () => {
    const prior = [finished(), child()];
    const service = { chat: vi.fn(async (_id: string, _text: string, _signal: AbortSignal | undefined,
      emit: ((value: NeuralChatStreamEvent) => void) | undefined) => {
      emit?.(token("End.")); return result(undefined, prior);
    }), idleCycle: vi.fn() };
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) };
    const actual = await new ChatActionController(service, tools, { start: vi.fn() }).send("brain", "input", undefined, "turn", undefined,
      { brainId: "brain", frontier: { turnId: "turn", lastStreamSequence: 4 }, input: "input", publishedText: "Opening. ",
        hostAttached: false, priorActions: prior } as unknown as import("../src/shared/instanceLifecycleApi").InstanceResumeAttachment);
    expect(actual.actionEvents).toEqual([]); expect(tools.execute).not.toHaveBeenCalled(); expect(service.idleCycle).not.toHaveBeenCalled();
    const old = { ...parent("e".repeat(32)), arguments: finished().arguments };
    const unstreamed = { chat: vi.fn(async () => result("unstreamed", [old])) };
    const returned = await new ChatActionController(unstreamed, tools, { start: vi.fn() }).send("brain", "input", undefined, "turn");
    expect(returned.actionEvents?.[0]).not.toHaveProperty("replyAnchor");
  });

  it("retains the actual cooperative Stop completion without dispatching a later child", async () => {
    let emit!: (value: NeuralChatStreamEvent) => void, rejectTurn!: (error: Error) => void;
    const held = new Promise<ChatResult>((_resolve, reject) => { rejectTurn = reject; });
    const persisted: ActionEvent[][] = [];
    const service = { chat: vi.fn(async (_id: string, _text: string, _signal: AbortSignal | undefined,
      onStream: ((value: NeuralChatStreamEvent) => void) | undefined) => {
      emit = onStream!; emit(frame("started")); return held;
    }), recordConversationActions: vi.fn(async (_id: string, events: ActionEvent[]) => { persisted.push(structuredClone(events)); }) };
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const sending = controller.send("brain", "input", undefined, "turn"); const failed = expect(sending).rejects.toThrow("stopped");
    await vi.waitFor(() => expect(service.chat).toHaveBeenCalledOnce());
    controller.cancel("brain", "turn"); emit(frame("finished", finished(parent(), "cancelled")));
    emit({ type: "chat-action", sequence: 4, actionId: childId, action: child() });
    rejectTurn(new Error("stopped")); await failed;
    await vi.waitFor(() => expect(persisted.length).toBeGreaterThan(0));
    expect(persisted.at(-1)?.[0]).toMatchObject({ state: "stopped", action: { arguments: { ponderTrace: { stop_reason: "cancelled" } } } });
    expect(persisted.at(-1)?.[0]).not.toHaveProperty("replyAnchor"); expect(tools.execute).not.toHaveBeenCalled();
  });

  it.each(["approve", "decline"] as const)("persists a mid-generation %s snapshot and later binds a new immutable canonical revision", async decision => {
    const root = await mkdtemp(join(tmpdir(), "omni-live-action-custody-"));
    const ledger = await ConversationLedger.open(root, "brain");
    let release!: () => void;
    const heldGeneration = new Promise<void>(resolve => { release = resolve; });
    const storedSnapshots: ActionEvent[][] = [];
    const repository = { appendConversationActions: async (_id: string, events: ActionEvent[]): Promise<void> => {
      storedSnapshots.push(events); await Promise.resolve();
      ledger.append(events.map(value => ({ kind: "action" as const, value })));
    } };
    const recordConversationActions = (id: string, events: ActionEvent[]): Promise<void> =>
      BrainService.prototype.recordConversationActions.call({ repository } as unknown as BrainService, id, events);
    const service = { recordConversationActions, learnStructuredExperience: vi.fn(async () => ({})),
      chat: vi.fn(async (_id: string, _text: string, _signal: AbortSignal | undefined, emit: ((value: NeuralChatStreamEvent) => void) | undefined) => {
        emit?.(token("Opening. ")); emit?.(frame("started"));
        emit?.({ type: "chat-action", sequence: 2, actionId: childId, action: child() }); emit?.(frame("finished"));
        await heldGeneration; emit?.(token("End."));
        const committed = result(undefined, [finished(), child()]);
        ledger.append([{ kind: "message", value: committed.brainMessage }]);
        return committed;
      }) };
    const tools = { execute: vi.fn().mockResolvedValueOnce({ ...effect(), state: "approval-required", approvalToken: "live-grant",
      approvalExpiresAt: new Date(Date.now() + 60_000).toISOString() }).mockResolvedValueOnce(effect()),
      invalidateApproval: vi.fn(() => true), cancel: vi.fn(() => 0) };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const updates: ActionEvent[] = [];
    controller.on("stream", (value: ChatStreamEvent) => { if (value.type === "chat-action") updates.push(structuredClone(value.actionEvent)); });
    const sending = controller.send("brain", "input", undefined, "turn");
    try {
      await vi.waitFor(() => expect(updates.some(value => value.state === "approval-required")).toBe(true));
      const pending = updates.slice().reverse().find(value => value.state === "approval-required")!;
      expect(pending.replyAnchor).not.toHaveProperty("messageId");
      const request = { brainId: "brain", actionEventId: pending.id, approvalToken: "live-grant" };
      const resolved = decision === "approve" ? (await controller.approveAction(request)).actionEvent : await controller.declineAction(request);
      expect(resolved.replyAnchor).toMatchObject({ turnId: "turn", utf16Offset: 9 });
      const prior = ledger.page().entries.slice().reverse().find(value => value.action?.id === pending.id)!.action!;
      expect(prior).not.toHaveProperty("replyAnchor");
      expect(prior.state).toBe(decision === "approve" ? "complete" : "stopped");
      const originalRevision = prior.updatedAt;
      release(); const committed = await sending;
      const final = committed.actionEvents!.find(value => value.id === pending.id)!;
      expect(final.replyAnchor).toMatchObject({ turnId: "turn", messageId: "reply", utf16Offset: 9, prefixSha256: hash("Opening. ") });
      expect(Date.parse(final.updatedAt)).toBeGreaterThan(Date.parse(originalRevision));
      const revisions = ledger.page().entries.flatMap(value => value.action?.id === pending.id ? [value.action] : []);
      expect(revisions).toHaveLength(2); expect(revisions[0]).not.toHaveProperty("replyAnchor");
      expect(revisions[1]?.parentPonderActionEventId).toBe(final.parentPonderActionEventId);
      expect(revisions[1]?.replyAnchor).toEqual(final.replyAnchor);
      expect(storedSnapshots[0]?.[0]).not.toHaveProperty("replyAnchor");
      expect(tools.execute).toHaveBeenCalledTimes(decision === "approve" ? 2 : 1);
      expect(ledger.integrity()).toBeDefined();
    } finally { release(); await sending.catch(() => undefined); ledger.close(); await rm(root, { recursive: true, force: true }); }
  });
});

const event = (id = "action"): ActionEvent => ({ id, brainId: "brain", chatTurnId: "turn", neuralActionId: childId,
  action: child(""), state: "complete", createdAt: now, updatedAt: now });
const message = (content: string): ChatMessage => ({ id: "reply", role: "brain", turnId: "turn", content, createdAt: now });
function binding(raw: string, canonical: string, prefix: number, suffix: number, normalization: PublicReplyTextBinding["normalization"] = "strip"): PublicReplyTextBinding {
  return { format: "omni-public-reply-text-binding", version: 1, normalization, rawSha256: hash(raw), canonicalSha256: hash(canonical),
    rawUtf16Length: raw.length, canonicalUtf16Length: canonical.length, removedPrefixUtf16: prefix, removedSuffixUtf16: suffix };
}

describe("exact public-reply anchors and immutable ledger (no engine)", () => {
  it("translates only an exact native trim receipt including non-BMP UTF16 boundaries", () => {
    const anchors = new PublicReplyAnchors("turn"), pre = event("pre"), middle = event("middle"), trailing = event("trailing");
    anchors.measure(pre); anchors.append(" \t🧠 "); anchors.measure(middle); anchors.append("brain\n "); anchors.measure(trailing);
    expect(middle.replyAnchor).toMatchObject({ utf16Offset: 5, prefixSha256: hash(" \t🧠 ") });
    anchors.bind([pre, middle, trailing], message("🧠 brain"), binding(" \t🧠 brain\n ", "🧠 brain", 2, 2));
    expect(pre.replyAnchor).toEqual({ turnId: "turn", messageId: "reply", utf16Offset: 0, prefixSha256: hash("") });
    expect(middle.replyAnchor).toEqual({ turnId: "turn", messageId: "reply", utf16Offset: 3, prefixSha256: hash("🧠 ") });
    expect(trailing.replyAnchor).toMatchObject({ utf16Offset: 8, prefixSha256: hash("🧠 brain") });
  });

  it("preserves identity on interrupted/resumed public prefixes and ignores externally manufactured marks", () => {
    const raw = " 🧠 raw ", anchors = new PublicReplyAnchors("turn", " 🧠 "), a = event(), manufactured = event("fake");
    anchors.measure(a); anchors.append("raw "); manufactured.replyAnchor = { turnId: "turn", utf16Offset: 1, prefixSha256: hash(" ") };
    anchors.bind([a, manufactured], message(raw), binding(raw, raw, 0, 0, "identity"));
    expect(a.replyAnchor).toMatchObject({ messageId: "reply", utf16Offset: 4, prefixSha256: hash(" 🧠 ") });
    expect(manufactured).not.toHaveProperty("replyAnchor");
  });

  it("binds the host token anchor through an exact trimmed non-BMP completion", async () => {
    const service = { chat: vi.fn(async (_id: string, _text: string, _signal: AbortSignal | undefined,
      emit: ((value: NeuralChatStreamEvent) => void) | undefined) => {
      emit?.(token(" \t🧠 ")); emit?.(frame("started")); emit?.(frame("finished"));
      emit?.(token("brain\n "));
      return { ...result("🧠 brain", [finished()]), publicTextBinding: binding(" \t🧠 brain\n ", "🧠 brain", 2, 2) };
    }), idleCycle: vi.fn() };
    const actual = await new ChatActionController(service, { execute: vi.fn(), cancel: vi.fn(() => 0) }, { start: vi.fn() })
      .send("brain", "input", undefined, "turn");
    expect(actual.actionEvents?.[0]?.replyAnchor).toEqual({ turnId: "turn", messageId: "reply", utf16Offset: 3, prefixSha256: hash("🧠 ") });
  });

  it.each(["wrong-prefix", "wrong-owner", "different-content", "invalid-receipt"])("leaves %s unanchored rather than guessing", mismatch => {
    const anchors = new PublicReplyAnchors("turn"), a = event(); anchors.append(" exact "); anchors.measure(a);
    const receipt = binding(" exact ", "exact", 1, 1), final = message("exact");
    if (mismatch === "wrong-prefix") receipt.rawSha256 = hash("other");
    if (mismatch === "wrong-owner") final.turnId = "foreign";
    if (mismatch === "different-content") final.content = "changed";
    if (mismatch === "invalid-receipt") receipt.removedPrefixUtf16 = 2;
    anchors.bind([a], final, receipt); expect(a).not.toHaveProperty("replyAnchor");
    if (mismatch === "invalid-receipt") expect(normalizePublicReplyTextBinding(receipt)).toBeUndefined();
  });

  it("persists exact canonical turn/parent/anchor and rejects idempotent provenance drift or unbound storage", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-ponder-provenance-"));
    const ledger = await ConversationLedger.open(root, "brain");
    try {
      const p = { ...event("parent-host"), neuralActionId: parentId, action: finished() };
      const c = { ...event("child-host"), parentPonderActionEventId: p.id };
      const anchors = new PublicReplyAnchors("turn"); anchors.append("Opening. "); anchors.measure(p); anchors.measure(c); anchors.append("End.");
      anchors.bind([p, c], message("Opening. End."));
      ledger.append([{ kind: "message", value: message("Opening. End.") }, { kind: "action", value: p }, { kind: "action", value: c }]);
      expect(ledger.page().entries.at(-1)?.action).toMatchObject({ chatTurnId: "turn", parentPonderActionEventId: p.id, replyAnchor: { messageId: "reply" } });
      for (const changed of [{ parentPonderActionEventId: "other" }, { chatTurnId: "other" }, { neuralActionId: "d".repeat(32) },
        { replyAnchor: { ...c.replyAnchor!, utf16Offset: 1, prefixSha256: hash("O") } }])
        expect(() => ledger.append([{ kind: "action", value: { ...c, ...changed } }])).toThrow(/idempotency|canonical/);
      expect(() => ledger.append([{ kind: "action", value: { ...event("unbound"), replyAnchor: { turnId: "turn", utf16Offset: 0, prefixSha256: hash("") } } }]))
        .toThrow("canonically bound");
      const nativeTrace = { id: "trace-binding", created_at: now, input: "input", seed: 1,
        public_reply_text_binding: binding(" Opening. End. ", "Opening. End.", 1, 1) };
      ledger.append([{ kind: "trace", value: nativeTrace as unknown as ChatResult["trace"] }]);
      expect(() => ledger.append([{ kind: "trace", value: { ...nativeTrace, public_reply_text_binding: { ...nativeTrace.public_reply_text_binding,
        rawSha256: hash("different stream") } } as unknown as ChatResult["trace"] }])).toThrow("idempotency");
      expect(ledger.integrity()).toBeDefined();
    } finally { ledger.close(); await rm(root, { recursive: true, force: true }); }
  });
});
