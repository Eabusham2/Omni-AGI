import { describe, expect, it, vi } from "vitest";
import { ChatActionController } from "../src/main/chatActionController";
import { EngineRequestError } from "../src/main/engineSupervisor";
import type { NeuralChatStreamEvent } from "../src/main/brainService";
import type { ChatResult, ChatStreamEvent, ToolExecutionResult } from "../src/shared/types";

const createdAt = "2026-09-29T23:00:00.000Z";
function reply(turnId: string, text: string, steered = false, artifact = false): ChatResult {
  return {
    brain: { messages: [] } as unknown as ChatResult["brain"],
    humanMessage: { id: `human-${turnId}`, turnId, role: "human", content: turnId, createdAt,
      ...(steered ? { generationEnd: "steered" as const } : {}) },
    brainMessage: { id: `brain-${turnId}`, turnId, role: "brain", content: text, createdAt,
      ...(steered ? { generationEnd: "steered" as const } : {}) },
    trace: { id: `trace-${turnId}` } as ChatResult["trace"],
    ...(steered ? { generationEnd: "steered" as const } : {}),
    proposedActions: artifact ? [{ kind: "imagine", source: "brain", toolId: "modality.imagine",
      action: "generate", arguments: { modality: "image" } }] : []
  };
}

describe("true warm steering boundary", () => {
  it("interrupts the old generation and starts the new direction after partial save, before old artifacts settle", async () => {
    let savePartial!: (value: ChatResult) => void;
    const original = new Promise<ChatResult>((resolve) => { savePartial = resolve; });
    let finishArtifact!: (value: ToolExecutionResult) => void;
    const artifact = new Promise<ToolExecutionResult>((resolve) => { finishArtifact = resolve; });
    let oldSignal!: AbortSignal;
    const service = {
      steerChat: vi.fn(async () => { savePartial(reply("old", "actual emitted prefix ", true, true)); }),
      chat: vi.fn((_brain: string, input: string, signal?: AbortSignal,
        _stream?: (event: NeuralChatStreamEvent) => void, turnId?: string) => {
        if (turnId === "old") { oldSignal = signal!; return original; }
        return Promise.resolve(reply(turnId!, input));
      })
    };
    const tools = { execute: vi.fn(() => artifact), cancel: vi.fn(() => 0) };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const stream: ChatStreamEvent[] = [];
    controller.on("stream", (event: ChatStreamEvent) => stream.push(event));
    let originalSettled = false;
    const first = controller.send("brain", "old", undefined, "old");
    void first.then(() => { originalSettled = true; });
    const next = controller.send("brain", "new direction verbatim", undefined, "new", {
      kind: "steer", replacesTurnId: "old", source: "human", createdAt
    });
    await expect(next).resolves.toMatchObject({ brainMessage: { content: "new direction verbatim" } });
    expect(service.steerChat).toHaveBeenCalledWith("brain", "old", "new");
    expect(service.chat.mock.calls[1]?.[1]).toBe("new direction verbatim");
    expect(oldSignal.aborted).toBe(false);
    expect(tools.cancel).not.toHaveBeenCalled();
    expect(originalSettled).toBe(false);
    expect(stream).toContainEqual(expect.objectContaining({ type: "chat-reply-committed", turnId: "old",
      brainMessage: expect.objectContaining({ content: "actual emitted prefix ", generationEnd: "steered" }) }));
    finishArtifact({ id: "artifact", toolId: "modality.imagine", action: "generate", state: "complete", startedAt: createdAt });
    await first;
  });

  it("preserves a typed zero-token yield without inventing assistant text or cancelling the worker", async () => {
    let yieldBeforeText!: (error: unknown) => void;
    const original = new Promise<ChatResult>((_resolve, reject) => { yieldBeforeText = reject; });
    let oldSignal!: AbortSignal;
    const service = {
      steerChat: vi.fn(async () => { yieldBeforeText(new EngineRequestError("warm yield", -32801,
        { brainId: "brain", turnId: "old", steered: true, zeroTokenYield: true, safeBoundary: true, warm: true })); }),
      chat: vi.fn((_brain: string, text: string, signal?: AbortSignal,
        _stream?: (event: NeuralChatStreamEvent) => void, turnId?: string) => {
        if (turnId === "old") { oldSignal = signal!; return original; }
        return Promise.resolve(reply(turnId!, text));
      })
    };
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const stream: ChatStreamEvent[] = [];
    controller.on("stream", (event: ChatStreamEvent) => stream.push(event));
    const first = controller.send("brain", "old", undefined, "old").catch((error: unknown) => error);
    await controller.send("brain", "new", undefined, "new", {
      kind: "steer", replacesTurnId: "old", source: "human", createdAt
    });
    await expect(first).resolves.toMatchObject({ code: -32801 });
    expect(stream).toContainEqual(expect.objectContaining({ type: "chat-state", turnId: "old", state: "steered" }));
    expect(stream.some((event) => event.type === "chat-reply-committed" && event.turnId === "old")).toBe(false);
    expect(oldSignal.aborted).toBe(false);
    expect(tools.cancel).not.toHaveBeenCalled();
  });
});
