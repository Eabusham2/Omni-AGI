import { mkdtemp, mkdir, rm, writeFile, readFile } from "node:fs/promises";
import { DatabaseSync } from "node:sqlite";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { receivedChatInput, receivedChatInputFromLedger, chatInputHash } from "../src/main/chatInputReceipt";
import { BrainService, normalizeChatEngineEvent, validateWorkerChatPresentation } from "../src/main/brainService";
import { ChatActionController } from "../src/main/chatActionController";
import { advanceChatGenerationPhase } from "../src/renderer/src/chatGenerationLifecycle";
import { receivedChatInputFailureStatus, reconcileCompletedChatTurn } from "../src/renderer/src/chatTurnPresentation";
import type { ChatMessage, ChatStreamEvent } from "../src/shared/types";
import type { NeuralChatStreamEvent } from "../src/main/brainService";

const roots: string[] = [], input = "actual received human input 文", turnId = "received-turn", createdAt = "2026-09-30T12:00:00.000Z";
const receipt = { committed: true as const, turnId, inputSha256: chatInputHash(input), humanMessageId: "human-received",
  afterimageId: "a".repeat(64), createdAt, attentionEpoch: 1, slowJobId: "b".repeat(64) };
const human = { id: receipt.humanMessageId, role: "human", content: input, turn_id: turnId,
  created_at: createdAt, attention_epoch: 1, input_accepted_before_reply: true };
afterEach(async () => { await Promise.all(roots.splice(0).map(path => rm(path, { recursive: true, force: true }))); });

describe("durable received input is distinct from a reply", () => {
  it("renders exactly the received human entry without a reply/trace/inference or model-facing synthetic message", () => {
    const value = receivedChatInput(receipt, { turnId, input });
    expect(value.humanMessage).toMatchObject({ id: receipt.humanMessageId, content: input, role: "human", inputReceipt: receipt });
    expect(value).not.toHaveProperty("brainMessage"); expect(value).not.toHaveProperty("trace");
    expect(value).not.toHaveProperty("turnReceipt");
    const event: ChatStreamEvent = { type: "chat-input-accepted", ...value, id: "event", brainId: "brain", turnId, createdAt, sequence: 0 };
    expect(advanceChatGenerationPhase("responding", event)).toBe("responding");
    expect(advanceChatGenerationPhase("settled", event)).toBe("settled");
  });
  it("requires exact committed identity, actual ledger marker/content/timestamp/epoch and cannot promote a notice without owner input", () => {
    expect(receivedChatInputFromLedger(receipt, human, { turnId, input }).inputReceipt).toEqual(receipt);
    for (const change of [{ committed: false }, { inputSha256: "c".repeat(64) }, { createdAt: "2026-09-30T11:00:00.000Z" }, { attentionEpoch: 2 }]) {
      expect(() => receivedChatInputFromLedger({ ...receipt, ...change }, human, { turnId, input })).toThrow();
    }
    expect(() => receivedChatInputFromLedger(receipt, { ...human, input_accepted_before_reply: false })).toThrow();
    const event = { type: "chat-input-accepted", brainId: "brain", streamId: turnId, sequence: 0, data: receipt };
    expect(normalizeChatEngineEvent(event, "brain")).toBeUndefined();
    expect(normalizeChatEngineEvent(event, "brain", { turnId, input })).toMatchObject({ type: "chat-input-accepted", inputReceipt: receipt });
  });
  it("reconciles optimistic input once and describes reply failure without unsending or claiming a completed answer", () => {
    const value = receivedChatInput(receipt, { turnId, input });
    const pending: ChatMessage = { id: `pending-${turnId}`, role: "human", content: input, createdAt };
    expect(reconcileCompletedChatTurn([pending], turnId, value.humanMessage)).toEqual([]);
    expect(receivedChatInputFailureStatus("cancelled")).toBe("Input received and saved · reply stopped.");
    expect(receivedChatInputFailureStatus("failed")).not.toContain("not sent");
  });
  it("controller publishes input acceptance before Stop and keeps it distinct from chat-reply-committed", async () => {
    let stream!: (event: NeuralChatStreamEvent) => void, reject!: (error: Error) => void;
    const service = { chat: vi.fn((_brain: string, _input: string, signal?: AbortSignal, listener?: (event: NeuralChatStreamEvent) => void) => {
      stream = listener!; return new Promise<never>((_resolve, failed) => { reject = failed; signal?.addEventListener("abort", () => reject(new Error("reply stopped"))); });
    }) };
    const controller = new ChatActionController(service, { execute: vi.fn(), cancel: vi.fn(() => 0) }, { start: vi.fn() });
    const events: ChatStreamEvent[] = []; controller.on("stream", (event: ChatStreamEvent) => events.push(event));
    const reply = controller.send("brain", input, undefined, turnId), failure = expect(reply).rejects.toThrow("reply stopped");
    stream({ type: "chat-input-accepted", sequence: 0, ...receivedChatInput(receipt, { turnId, input }) });
    controller.cancel("brain", turnId); await failure; await controller.cancelAndWait("brain", turnId);
    expect(events.filter(event => event.type === "chat-input-accepted")).toHaveLength(1);
    expect(events.some(event => event.type === "chat-reply-committed")).toBe(false);
  });
  it("allows ordinary no-reply only with immutable admitted human plus actual saved generation markers", () => {
    const checksum = "d".repeat(64), traceId = "trace", brainId = "brain-reply";
    const value = { text: "", noReply: true, turnCommitted: true, idempotentCompletion: false, humanMessage: human,
      message: { id: brainId, role: "brain", content: "", turn_id: turnId, created_at: createdAt, attention_epoch: 1, generation_end: "no-reply" },
      trace: { id: traceId, input_accepted_before_reply: true, input_sha256: receipt.inputSha256, turn_id: turnId, created_at: createdAt, attention_epoch: 1,
        generation_stop_reason: "no-reply", generation_decoder_stop_reason: "eos", generation_no_reply_reason: "no-generated-tokens", generated_token_count: 0,
        generation_printable_text_characters: 0, parameter_checksum_after: checksum },
      turnReceipt: { format: "omni-completed-chat-turn", formatVersion: 1, turnId, inputSha256: receipt.inputSha256,
        humanMessageId: human.id, brainMessageId: brainId, traceId, inferenceCount: 1, parameterChecksumAfter: checksum, committedAt: createdAt, generationEnd: "no-reply" } };
    const presentation = validateWorkerChatPresentation(value, { turnId, input, inputSha256: receipt.inputSha256, response: "" })!;
    expect(presentation.brainMessage).toMatchObject({ content: "", generationEnd: "no-reply" });
    expect(presentation.humanMessage.generationEnd).toBeUndefined(); // original input never acquires reply disposition
    expect(() => validateWorkerChatPresentation({ ...value, trace: { ...value.trace, input_accepted_before_reply: false } },
      { turnId, input, inputSha256: receipt.inputSha256, response: "" })).toThrow("no-reply");
  });
});

describe("constructor-free file/ledger human-only reconciliation", () => {
  it("keeps authoritative admitted input after restart/error, idempotently, without a brain load, completed reply or extra learning", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-received-input-")); roots.push(root); const engine = join(root, "engine"); await mkdir(engine);
    const database = new DatabaseSync(join(engine, "conversation.sqlite3"));
    database.exec("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE entries(entry_key TEXT PRIMARY KEY,kind TEXT,payload_json TEXT,payload_sha256 TEXT)");
    database.prepare("INSERT INTO meta VALUES('brain_id','brain')").run();
    const body = JSON.stringify(human), operational = JSON.stringify({ inputAccepted: receipt });
    const add = database.prepare("INSERT INTO entries VALUES(?,?,?,?)");
    add.run(`message:${human.id}`, "message", body, chatInputHash(body));
    add.run(`trace:${chatInputHash(`brain\0${turnId}\0accepted-input-v1`)}`, "trace", operational, chatInputHash(operational)); database.close();
    await writeFile(join(engine, "brain.json"), JSON.stringify({ brain_id: "brain", accepted_chat_inputs: [receipt], counters: { inference_count: 0 } }));
    const messages = new Map<string, ChatMessage>(), request = vi.fn();
    const service = Object.assign(Object.create(BrainService.prototype), { engine: { request }, repository: {
      brainDirectory: () => root, appendReceivedChatInput: vi.fn(async (_brain: string, message: ChatMessage) => { messages.set(message.id, message); }),
      get: async () => ({ id: "brain", messages: [...messages.values()], counters: { inferenceCount: 0 } })
    } }) as BrainService;
    const recovered = await service.getReconciledBrain("brain"); await service.getReconciledBrain("brain");
    expect(recovered.messages).toHaveLength(1); expect(messages.size).toBe(1); expect(recovered.counters.inferenceCount).toBe(0);
    expect(recovered.messages[0]).toMatchObject({ role: "human", content: input, inputReceipt: receipt }); expect(request).not.toHaveBeenCalled();
    // Current metadata may move on; permanent inert receipt still binds this retry.
    await writeFile(join(engine, "brain.json"), JSON.stringify({ brain_id: "brain", accepted_chat_inputs: [] }));
    const method = service as unknown as { reconcileReceivedChatInput(brain: string, expected: { turnId: string; input: string }): Promise<unknown> };
    await expect(method.reconcileReceivedChatInput("brain", { turnId, input })).resolves.toMatchObject({ inputReceipt: receipt });
    expect(request).not.toHaveBeenCalled();
  });
  it("renderer source does not restore already admitted input as an unsent draft or close the response on its input notice", async () => {
    const source = await readFile(new URL("../src/renderer/src/App.tsx", import.meta.url), "utf8");
    expect(source).toContain('event.type === "chat-input-accepted"');
    expect(source).toContain('terminalState === "failed" && !receivedInputTurnIdsRef.current.has(turnId)');
    expect(source).toContain('receivedChatInputFailureStatus(terminalState)');
  });
});
