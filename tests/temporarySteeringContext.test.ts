import { describe, expect, it, vi } from "vitest";
import { inheritTemporarySteeringContext, type TemporarySteeringContext } from "../src/main/temporarySteeringContext";
import { ChatActionController } from "../src/main/chatActionController";
import { EngineRequestError } from "../src/main/engineSupervisor";
import type { ChatResult } from "../src/shared/types";

const createdAt = "2026-09-30T12:00:00.000Z";
const owner = { brainId: "brain", turnId: "old", input: "Exact unfinished human direction 中文",
  attentionEpoch: 4, neuralSteered: true, earlySteeredYield: true };
const reply = (input: string, id: string): ChatResult => ({ brain: { messages: [] } as unknown as ChatResult["brain"],
  humanMessage: { id: `human-${id}`, turnId: id, role: "human", content: input, createdAt },
  brainMessage: { id: `brain-${id}`, turnId: id, role: "brain", content: "fixture reply", createdAt },
  trace: { id: `trace-${id}` } as ChatResult["trace"] });

describe("temporary actual-user warm Steer context", () => {
  it("binds the original input and carries chained zero-visible directions without a behavior prompt", () => {
    const first = inheritTemporarySteeringContext(owner, "middle", 4)!;
    expect(first.inputs[0]?.content).toBe(owner.input);
    expect(first.inputs[0]?.inputSha256).toMatch(/^[a-f0-9]{64}$/);
    const chained = inheritTemporarySteeringContext({ ...owner, turnId: "middle", input: "Actual second direction",
      temporarySteeringContext: first }, "new", 4)!;
    expect(chained.inputs.map((item) => item.content)).toEqual([owner.input, "Actual second direction"]);
    expect(chained.successorTurnId).toBe("new");
  });
  it("never duplicates a durably saved partial human turn, but retains earlier unlearned carry", () => {
    const first = inheritTemporarySteeringContext(owner, "middle", 4)!;
    const next = inheritTemporarySteeringContext({ ...owner, turnId: "middle", input: "saved partial's input",
      earlySteeredYield: false, temporarySteeringContext: first }, "new", 4)!;
    expect(next.inputs).toEqual(first.inputs);
  });
  it("Fresh, unknown attention, ordinary completion and cancellation cannot resurrect input", () => {
    expect(inheritTemporarySteeringContext(owner, "new", 5)).toBeUndefined();
    expect(inheritTemporarySteeringContext({ ...owner, attentionEpoch: undefined }, "new", 4)).toBeUndefined();
    expect(inheritTemporarySteeringContext({ ...owner, neuralSteered: false }, "new", 4)).toBeUndefined();
  });
  it("forwards only main-owned input after the actual zero-visible neural boundary without aborting the worker", async () => {
    let started!: () => void;
    const entered = new Promise<void>((resolve) => { started = resolve; });
    let yieldOld!: (error: unknown) => void;
    const old = new Promise<ChatResult>((_resolve, reject) => { yieldOld = reject; });
    const carried: Array<TemporarySteeringContext | undefined> = [];
    let oldSignal: AbortSignal | undefined;
    const service = { currentChatAttentionEpoch: vi.fn(async () => 4),
      steerChat: vi.fn(async () => yieldOld(new EngineRequestError("zero visible warm yield", -32801,
        { brainId: "brain", turnId: "old", steered: true, zeroTokenYield: true, safeBoundary: true, warm: true }))),
      chat: vi.fn((_brain: string, input: string, signal?: AbortSignal, _stream?: unknown, turnId?: string,
        _budget?: number, temporary?: TemporarySteeringContext) => {
        if (turnId === "old") { oldSignal = signal; started(); return old; }
        carried.push(temporary);
        return Promise.resolve(reply(input, turnId!));
      }) };
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const first = controller.send("brain", owner.input, undefined, "old").catch((error: unknown) => error);
    await entered;
    await controller.send("brain", "Actual new direction", undefined, "new",
      { kind: "steer", replacesTurnId: "old", source: "human", createdAt });
    await expect(first).resolves.toMatchObject({ code: -32801 });
    expect(carried[0]?.inputs[0]?.content).toBe(owner.input);
    expect(carried[0]?.successorTurnId).toBe("new");
    expect(oldSignal?.aborted).toBe(false);
    expect(tools.cancel).not.toHaveBeenCalled();
  });
});
