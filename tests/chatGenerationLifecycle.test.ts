import { describe, expect, it, vi } from "vitest";
import { ChatActionController } from "../src/main/chatActionController";
import { BrainWriteCoordinator } from "../src/main/brainWriteCoordinator";
import type { NeuralChatStreamEvent } from "../src/main/brainService";
import type { ActionEvent, ChatResult, ChatStreamEvent, ToolExecutionResult } from "../src/shared/types";
import {
  advanceChatGenerationPhase,
  reconcileUncommittedChatOutputs,
  retainUncommittedChatOutput,
  settleUncommittedChatOutput,
  type ChatGenerationPhase
} from "../src/renderer/src/chatGenerationLifecycle";
import { chatActionCancellationTarget } from "../src/renderer/src/chatActionPresentation";
import { composerTurnCapabilities, reconcileCompletedChatTurn } from "../src/renderer/src/chatTurnPresentation";

const createdAt = "2026-09-29T12:00:00.000Z";
const base = { id: "event", brainId: "brain", turnId: "first", sequence: 0, createdAt };
const action: ActionEvent = {
  id: "artifact", brainId: "brain", state: "running", createdAt, updatedAt: createdAt,
  action: { kind: "imagine", source: "brain", toolId: "modality.imagine", action: "generate", arguments: {} }
};
const phase: ChatStreamEvent = {
  ...base, type: "chat-phase", phase: "reply-complete-learning", replyComplete: true,
  turnCommitted: false, learning: true, saving: true
};

function result(turnId: string, input: string, withArtifact = false): ChatResult {
  return {
    brain: { messages: [] } as unknown as ChatResult["brain"],
    humanMessage: { id: `human-${turnId}`, role: "human", turnId, content: input, createdAt },
    brainMessage: { id: `brain-${turnId}`, role: "brain", turnId, content: `reply ${turnId}`, createdAt },
    trace: { id: `trace-${turnId}` } as ChatResult["trace"],
    proposedActions: withArtifact ? [action.action] : []
  };
}

describe("generation ends independently of reply work", () => {
  it("never reopens completed output from tokens, actions, previews or late start frames", () => {
    let lifecycle = advanceChatGenerationPhase("responding", phase);
    expect(lifecycle).toBe("reply-complete");
    for (const event of [
      { ...base, type: "chat-action", actionEvent: action },
      { ...base, type: "modality-preview", actionId: action.id, preview: { revision: 2 } },
      { ...base, type: "chat-token", delta: "late" },
      { ...base, type: "chat-state", state: "started" }
    ] as ChatStreamEvent[]) {
      lifecycle = advanceChatGenerationPhase(lifecycle, event);
      expect(lifecycle).toBe("reply-complete");
    }
    expect(composerTurnCapabilities("reply-complete-learning")).toEqual({
      turnActive: false, queueAvailable: false, steerAvailable: false
    });
    lifecycle = advanceChatGenerationPhase(lifecycle, { ...base, type: "chat-state", state: "complete" });
    expect(advanceChatGenerationPhase(lifecycle, phase)).toBe("settled");
  });

  it("retains exact final text as uncommitted through next Send and a save failure", () => {
    const first = retainUncommittedChatOutput([], "first", "same reply", createdAt);
    const both = retainUncommittedChatOutput(first, "second", "same reply", createdAt);
    expect(both).toHaveLength(2);
    expect(both.map((output) => output.state)).toEqual(["saving", "saving"]);
    const failed = settleUncommittedChatOutput(both, "first", "failed");
    expect(failed[0]).toMatchObject({ state: "failed", message: { content: "same reply" } });
    expect(reconcileUncommittedChatOutputs(failed, "second")).toEqual([failed[0]]);
    expect(reconcileUncommittedChatOutputs(failed, "first")).toEqual([failed[1]]);
  });

  it("reconciles only the correlated older human, including legacy timestamps", () => {
    const optimistic = ["steered-first", "pending-second"].map((id) => ({
      id, role: "human" as const, content: id, createdAt
    }));
    const human = { id: "saved-first", role: "human" as const, content: "steered-first", createdAt: "2026-09-29T12:00:01.000Z" };
    expect(reconcileCompletedChatTurn(optimistic, "first", human)).toEqual([optimistic[1]]);
  });

  it("uses exact artifact/tool targets and never falls back to cancelling a whole brain", () => {
    expect(chatActionCancellationTarget({ ...action, runtimeJobId: "job-one" }, "old-turn"))
      .toEqual({ kind: "modality-job", id: "job-one" });
    const tool = { ...action, action: { ...action.action, kind: "tool" as const, toolId: "web.search", action: "search" } };
    expect(chatActionCancellationTarget(tool, "old-turn"))
      .toEqual({ kind: "tool-turn", id: "old-turn" });
    expect(chatActionCancellationTarget(action, "old-turn")).toBeUndefined();
    expect(chatActionCancellationTarget({ ...action, inlineGenerationOwned: true,
      neuralActionId: "a".repeat(32) }, "old-turn"))
      .toEqual({ kind: "inline-action", id: action.id, turnId: "old-turn" });
    expect(chatActionCancellationTarget(action)).toBeUndefined();
    expect(chatActionCancellationTarget({ ...action, state: "complete" }, "old-turn")).toBeUndefined();
    expect(chatActionCancellationTarget({ ...action, action: { ...action.action, kind: "evolve" } }, "old-turn"))
      .toBeUndefined();
  });

  it("publishes no commit before save, then permits the next serialized reply while an artifact is unfinished", async () => {
    let finishSave!: () => void;
    const save = new Promise<void>((resolve) => { finishSave = resolve; });
    let finishArtifact!: (execution: ToolExecutionResult) => void;
    const artifact = new Promise<ToolExecutionResult>((resolve) => { finishArtifact = resolve; });
    const writes = new BrainWriteCoordinator();
    const neuralStarts: string[] = [];
    const service = {
      chat: vi.fn(async (brainId: string, input: string, signal?: AbortSignal,
        onStream?: (event: NeuralChatStreamEvent) => void, turnId = "first") =>
        writes.run(brainId, async () => {
          neuralStarts.push(turnId);
          if (turnId === "first") {
            onStream?.({ type: "chat-token", sequence: 0, delta: "reply first" });
            onStream?.({ type: "chat-phase", sequence: 1, phase: "reply-complete-learning",
              replyComplete: true, turnCommitted: false, learning: true, saving: true });
            await save;
          }
          return result(turnId, input, turnId === "first");
        }, signal))
    };
    const controller = new ChatActionController(service,
      { execute: vi.fn(() => artifact), cancel: vi.fn(() => 0) }, { start: vi.fn() });
    const events: ChatStreamEvent[] = [];
    const phases = new Map<string, ChatGenerationPhase>();
    controller.on("stream", (event: ChatStreamEvent) => {
      events.push(event);
      phases.set(event.turnId, advanceChatGenerationPhase(phases.get(event.turnId) ?? "responding", event));
    });
    let firstSettled = false;
    const first = controller.send("brain", "first", undefined, "first");
    void first.then(() => { firstSettled = true; });
    await vi.waitFor(() => expect(phases.get("first")).toBe("reply-complete"));
    expect(events.some((event) => event.type === "chat-reply-committed")).toBe(false);
    const second = controller.send("brain", "second", undefined, "second");
    expect(neuralStarts).toEqual(["first"]);
    finishSave();
    await second;
    expect(firstSettled).toBe(false);
    expect(neuralStarts).toEqual(["first", "second"]);
    expect(events).toContainEqual(expect.objectContaining({
      type: "chat-reply-committed", turnId: "first", pendingActions: 1,
      brainMessage: { id: "brain-first", role: "brain", turnId: "first", content: "reply first", createdAt }
    }));
    expect(phases.get("first")).toBe("reply-complete");
    expect(phases.get("second")).toBe("settled");
    finishArtifact({ id: "execution", toolId: "modality.imagine", action: "generate", state: "complete", startedAt: createdAt });
    await first;
    expect(phases.get("first")).toBe("settled");
  });

  it("cancels only an owned inline artifact, preserves its saved reply and never regenerates it", async () => {
    let ack!: (event: { brainId: string; turnId: string; actionId: string }) => void;
    let finishSave!: (value: ChatResult) => void;
    let originalSignal!: AbortSignal;
    const pending = new Promise<ChatResult>((resolve) => { finishSave = resolve; });
    const neuralActionId = "a".repeat(32);
    const service = {
      onInlineImaginationCancelled: (listener: typeof ack) => { ack = listener; return () => undefined; },
      cancelInlineImagination: vi.fn(async () => ({ requested: true, acknowledged: false })),
      recordConversationActions: vi.fn(async () => undefined),
      chat: vi.fn((_brain: string, _input: string, signal?: AbortSignal,
        onStream?: (event: NeuralChatStreamEvent) => void) => {
        originalSignal = signal!;
        onStream?.({ type: "chat-action", sequence: 0, actionId: neuralActionId, action: action.action });
        onStream?.({ type: "inline-imagination-started", sequence: 1, actionId: neuralActionId });
        onStream?.({ type: "chat-phase", sequence: 2, phase: "reply-complete-learning",
          replyComplete: true, turnCommitted: false, learning: true, saving: true });
        return pending;
      })
    };
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const stream: ChatStreamEvent[] = [];
    const settledArtifacts: ActionEvent[] = [];
    controller.on("stream", (event: ChatStreamEvent) => stream.push(event));
    controller.on("event", (event: ActionEvent) => settledArtifacts.push(event));
    const first = controller.send("brain", "keep this reply", undefined, "first");
    const running = stream.find((event) => event.type === "chat-action" && event.actionEvent.inlineGenerationOwned);
    if (!running || running.type !== "chat-action") throw new Error("missing real inline ownership event");
    await expect(controller.cancelInlineAction("other-brain", "first", running.actionEvent.id)).rejects.toThrow(/owned/);
    const cancellation = await controller.cancelInlineAction("brain", "first", running.actionEvent.id);
    expect(cancellation).toMatchObject({ acknowledged: false, actionEvent: { state: "running", cancellationRequested: true } });
    expect(service.cancelInlineImagination).toHaveBeenCalledWith("brain", "first", neuralActionId);
    expect(originalSignal.aborted).toBe(false);
    expect(tools.cancel).not.toHaveBeenCalled();
    ack({ brainId: "other-brain", turnId: "first", actionId: neuralActionId });
    expect(settledArtifacts).toEqual([]);
    finishSave(result("first", "keep this reply", true));
    await expect(first).resolves.toMatchObject({ brainMessage: { content: "reply first" } });
    expect(tools.execute).not.toHaveBeenCalled();
    ack({ brainId: "brain", turnId: "first", actionId: neuralActionId });
    expect(settledArtifacts.at(-1)).toMatchObject({ state: "stopped", cancellationRequested: false });
    expect(stream.some((event) => event.type === "chat-state" && event.state === "cancelled")).toBe(false);
    expect(originalSignal.aborted).toBe(false);
  });
});
