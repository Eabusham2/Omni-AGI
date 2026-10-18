import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import { ChatActionController, type ActionChatService } from "../src/main/chatActionController";
import { BrainWriteCoordinator } from "../src/main/brainWriteCoordinator";
import { EngineRequestError } from "../src/main/engineSupervisor";
import { BrainService } from "../src/main/brainService";
import { nativeAudioIdentityJson, nativeAudioMessageInfo, type NativeAudioInput, type NativeAudioInputReceipt } from "../src/shared/nativeAudioChat";
import { receivedChatInput } from "../src/main/chatInputReceipt";
import type { ChatResult, ChatStreamEvent } from "../src/shared/types";

const at = "2026-10-11T14:00:00Z";
function source(): NativeAudioInput {
  const data = { format: "omni-native-audio-input" as const, version: 1 as const,
    attachment: { brainId: "audio-brain", turnId: "audio-turn", attachmentId: "owned-audio", contentSha256: "a".repeat(64), bytes: 100,
      mimeType: "audio/wav", kind: "audio", relativePath: `attachments/blobs/${"a".repeat(64)}.data`, displayName: "Microphone audio", manifestSha256: "b".repeat(64) },
    capture: { sessionId: "owned-lease", utteranceId: "clip-one", sequence: 0, capturedAt: at, permissionRevision: 1 } };
  return { ...data, inputSha256: createHash("sha256").update(nativeAudioIdentityJson(data)).digest("hex") };
}
function receipt(input: NativeAudioInput): NativeAudioInputReceipt {
  return { input, assemblyId: "audio-assembly", sourceBytes: 100, sourceSha256: "a".repeat(64), decodedWindows: 1, modelSamples: 4,
    modelSampleRate: 16000, wholeInputTraversed: true, nativeSensory: true, transcriptUsed: false };
}
function result(turnId: string, input: string, audio?: NativeAudioInput): ChatResult {
  return { brain: { messages: [], traces: [] } as unknown as ChatResult["brain"],
    humanMessage: { id: `human-${turnId}`, role: "human", turnId, content: input, createdAt: at, ...(audio ? { nativeAudioInput: nativeAudioMessageInfo(receipt(audio)) } : {}) },
    brainMessage: { id: `brain-${turnId}`, role: "brain", turnId, content: "Synthetic host result; no neural inference.", createdAt: at },
    trace: { id: `trace-${turnId}` } as ChatResult["trace"], proposedActions: [] };
}
const tools = () => ({ execute: vi.fn(), cancel: vi.fn(() => 0) });

describe("typed audio normal turn controls (synthetic native boundary)", () => {
  it("queues audio on the ordinary writer and streams a safe exact accepted-input receipt", async () => {
    const audio = source(), writes = new BrainWriteCoordinator(), started: string[] = [];
    let finish!: () => void; const held = new Promise<void>(resolve => { finish = resolve; });
    const service: ActionChatService = { chat: vi.fn(async (...args) => {
      const [brainId, input, signal, stream] = args, turnId = args[4]!;
      return writes.run(brainId, async () => {
        started.push(turnId);
        if (turnId === "first-turn") await held;
        if (args[11]) {
          expect(input).toBe(""); expect(args[11]).toEqual(audio);
          const admitted = receivedChatInput({ committed: true, turnId, inputSha256: audio.inputSha256, humanMessageId: "human-audio-turn",
            afterimageId: "audio-afterimage", createdAt: at, attentionEpoch: 0, nativeAudioInput: receipt(audio) }, { turnId, input: "", nativeAudioInput: audio });
          stream?.({ type: "chat-input-accepted", sequence: 0, ...admitted });
        }
        return result(turnId, input, args[11]);
      }, signal);
    }) };
    const executor = tools(), controller = new ChatActionController(service, executor, { start: vi.fn() }), events: ChatStreamEvent[] = [];
    controller.on("stream", event => events.push(event));
    const first = controller.send("audio-brain", "Actual user text", undefined, "first-turn");
    await vi.waitFor(() => expect(started).toEqual(["first-turn"]));
    const queued = controller.send("audio-brain", "", undefined, "audio-turn", undefined, undefined, undefined, audio);
    await Promise.resolve(); expect(started).toEqual(["first-turn"]);
    finish(); await first; const output = await queued;
    expect(started).toEqual(["first-turn", "audio-turn"]); expect(output.humanMessage.content).toBe("");
    const accepted = events.find(event => event.type === "chat-input-accepted");
    expect(accepted).toMatchObject({ inputReceipt: { inputSha256: audio.inputSha256 } });
    expect(JSON.stringify(accepted)).not.toMatch(/relativePath|permissionRevision|dataBase64/);
    expect(executor.execute).not.toHaveBeenCalled();
  });
  it("Stop removes only a queued audio wait, without admitting it or cancelling another writer", async () => {
    const audio = source(), writes = new BrainWriteCoordinator(), started: string[] = [];
    let finish!: () => void; const held = new Promise<void>(resolve => { finish = resolve; });
    const service: ActionChatService = { chat: vi.fn(async (...args) => writes.run(args[0], async () => {
      const turnId = args[4]!; started.push(turnId);
      if (turnId === "first-turn") await held;
      return result(turnId, args[1], args[11]);
    }, args[2])) };
    const controller = new ChatActionController(service, tools(), { start: vi.fn() });
    const first = controller.send("audio-brain", "first", undefined, "first-turn");
    await vi.waitFor(() => expect(started).toEqual(["first-turn"]));
    const queued = controller.send("audio-brain", "", undefined, "audio-turn", undefined, undefined, undefined, audio);
    const rejected = queued.catch(error => error);
    await vi.waitFor(() => expect(service.chat).toHaveBeenCalledTimes(2));
    expect(await controller.cancelAndWait("audio-brain", "audio-turn")).toBe(1);
    expect(await rejected).toBeInstanceOf(Error); expect(started).toEqual(["first-turn"]);
    finish(); await first;
  });
  it("early audio Steer carries typed sensory input, never an empty-caption failure or a worker restart", async () => {
    const audio = source(); let reject!: (error: unknown) => void;
    const pending = new Promise<ChatResult>((_resolve, fail) => { reject = fail; });
    const service: ActionChatService = { currentChatAttentionEpoch: vi.fn(async () => 0),
      chat: vi.fn(async (...args) => {
        if (args[4] === "audio-turn") return pending;
        expect(args[1]).toBe("Continue with my correction");
        expect(args[6]).toMatchObject({ inputs: [], audioInputs: [audio], successorTurnId: "next-turn" });
        return result(args[4]!, args[1]);
      }), steerChat: vi.fn(async (_brain, turnId) => { reject(new EngineRequestError("Synthetic safe early yield", -32801,
        { brainId: "audio-brain", turnId, steered: true, zeroTokenYield: true, safeBoundary: true, warm: true })); }) };
    const controller = new ChatActionController(service, tools(), { start: vi.fn() });
    const first = controller.send("audio-brain", "", undefined, "audio-turn", undefined, undefined, undefined, audio);
    const yielded = first.catch(error => error);
    await vi.waitFor(() => expect(service.chat).toHaveBeenCalledTimes(1));
    const output = await controller.send("audio-brain", "Continue with my correction", undefined, "next-turn",
      { kind: "steer", source: "human", replacesTurnId: "audio-turn", createdAt: at });
    expect(output.humanMessage.content).toBe("Continue with my correction"); expect((await yielded).code).toBe(-32801);
    expect(service.steerChat).toHaveBeenCalledWith("audio-brain", "audio-turn", "next-turn");
  });
  it("public saved-continuation presentation retains original audio identity without a raw path", async () => {
    const audio = source(), raw = receipt(audio);
    const owner = { inspectNativeInstance: vi.fn(async () => ({ resume: { turnId: "audio-turn", checkpointId: "c".repeat(32), inputSha256: audio.inputSha256,
      exactSupportedResume: true }, presentation: { input: "", publishedText: "partial", actions: [], inputReceipt: { nativeAudioInput: raw } } })),
      liveChatTurns: new Map() } as unknown as BrainService;
    const projection = await BrainService.prototype.instanceResumePresentation.call(owner, "audio-brain");
    expect(projection?.nativeAudioInput?.inputSha256).toBe(audio.inputSha256);
    expect(projection?.nativeAudioInput?.sourceBrainId).toBe("audio-brain");
    expect(JSON.stringify(projection)).not.toMatch(/relativePath|permissionRevision|dataBase64/);
  });
  it("Stop during attachment binding cannot later create a neural writer", async () => {
    let finish!: () => void; const binding = new Promise<void>(resolve => { finish = resolve; });
    const service: ActionChatService = { bindTaskAttachments: vi.fn(async () => binding), chat: vi.fn() };
    const controller = new ChatActionController(service, tools(), { start: vi.fn() });
    const pending = controller.send("audio-brain", "Actual user request", undefined, "binding-turn",
      { kind: "task", source: "human", createdAt: at, attachmentIds: ["owned-attachment"] });
    const rejected = pending.catch(error => error);
    await vi.waitFor(() => expect(service.bindTaskAttachments).toHaveBeenCalledTimes(1));
    await controller.cancelAndWait("audio-brain", "binding-turn"); finish();
    expect(await rejected).toMatchObject({ name: "AbortError" }); expect(service.chat).not.toHaveBeenCalled();
  });
});
