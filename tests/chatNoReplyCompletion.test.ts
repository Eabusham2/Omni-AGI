import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { describe, expect, it, vi } from "vitest";
import { ChatActionController } from "../src/main/chatActionController";
import { validateWorkerChatPresentation } from "../src/main/brainService";
import { recordNeuralChat } from "../src/main/presentationChat";
import type { BrainDocument, ChatResult, ChatStreamEvent } from "../src/shared/types";
import { advanceChatGenerationPhase } from "../src/renderer/src/chatGenerationLifecycle";
import {
  chatMessageDeliveryState, chatNoReplyPresentation, mergeChatMessagesForPresentation,
  reconcileCompletedChatTurn
} from "../src/renderer/src/chatTurnPresentation";

const createdAt = "2026-09-30T12:00:00.000Z";
function fixture() {
  const input = "actual human input";
  const turnId = "turn-no-reply";
  const inputSha256 = createHash("sha256").update(input).digest("hex");
  const checksum = "a".repeat(64);
  const value = {
    text: "", response: "", content: "", noReply: true, steered: false, nativeStopped: false,
    turnCommitted: true, idempotentCompletion: false,
    humanMessage: { id: "human-no-reply", role: "human", content: input, turn_id: turnId,
      generation_end: "no-reply", created_at: createdAt, attention_epoch: 1 },
    message: { id: "completion-no-reply", role: "brain", content: "", turn_id: turnId,
      generation_end: "no-reply", created_at: createdAt, attention_epoch: 1 },
    trace: { id: "trace-no-reply", turn_id: turnId, input_sha256: inputSha256,
      generation_stop_reason: "no-reply", generation_decoder_stop_reason: "eos",
      generation_no_reply_reason: "no-decoded-text", generated_token_count: 1,
      generation_printable_text_characters: 0,
      parameter_checksum_after: checksum, created_at: createdAt, attention_epoch: 1 },
    turnReceipt: { format: "omni-completed-chat-turn", formatVersion: 1, turnId, inputSha256,
      humanMessageId: "human-no-reply", brainMessageId: "completion-no-reply", traceId: "trace-no-reply",
      inferenceCount: 1, parameterChecksumAfter: checksum, committedAt: createdAt, generationEnd: "no-reply" }
  };
  return { value, expected: { input, inputSha256, response: "", turnId } };
}

describe("receipt-bound ordinary no-reply completion", () => {
  it("accepts zero assistant text only with complete durable no-reply evidence", () => {
    const { value, expected } = fixture();
    const result = validateWorkerChatPresentation(value, expected)!;
    expect(result.humanMessage).toMatchObject({ content: expected.input, status: "complete", generationEnd: "no-reply" });
    expect(result.brainMessage).toMatchObject({ content: "", status: "complete", generationEnd: "no-reply" });
    expect(result.inferenceCount).toBe(1);
  });

  it.each(["text", "flag", "receipt", "trace", "count", "control"] as const)("rejects forged or conflicting %s", (field) => {
    const { value, expected } = fixture();
    if (field === "text") value.message.content = "?";
    if (field === "flag") value.noReply = false;
    if (field === "receipt") value.turnReceipt.generationEnd = "native-stop";
    if (field === "trace") value.trace.generation_no_reply_reason = "refusal";
    if (field === "count") value.trace.generation_printable_text_characters = 1;
    if (field === "control") value.steered = true;
    expect(() => validateWorkerChatPresentation(value, expected)).toThrow();
  });

  it("does not promote an untyped empty ordinary response to a completed reply", () => {
    const { value, expected } = fixture();
    const ordinary = { ...value, noReply: false,
      humanMessage: { ...value.humanMessage, generation_end: undefined },
      message: { ...value.message, generation_end: undefined },
      turnReceipt: { ...value.turnReceipt, generationEnd: undefined },
      trace: { ...value.trace, generation_stop_reason: "learned-boundary" } };
    expect(() => validateWorkerChatPresentation(ordinary, expected)).toThrow(/receipt-bound/);
  });

  it("keeps the human and typed zero-text marker in presentation across receipt reconciliation", () => {
    const { value, expected } = fixture();
    const saved = validateWorkerChatPresentation(value, expected)!;
    const optimistic = { id: `pending-${expected.turnId}`, role: "human" as const,
      content: expected.input, createdAt: "2026-09-30T11:59:59.000Z" };
    const authoritative = [saved.humanMessage, saved.brainMessage];
    expect(reconcileCompletedChatTurn([optimistic], expected.turnId, saved.humanMessage)).toEqual([]);
    expect(mergeChatMessagesForPresentation(authoritative, [optimistic])).toEqual(authoritative);
    expect(chatMessageDeliveryState(saved.brainMessage)).toBe("no-reply");
    const presentation = chatNoReplyPresentation(saved.brainMessage)!;
    expect(presentation.label).toBe("No reply · input saved");
    expect(presentation.ariaLabel).not.toMatch(/refus|cancel|ethic|chose/i);
    expect(saved.brainMessage.content).toBe("");
  });

  it("requires typed empty completion in the host presentation recorder", () => {
    const brain = { messages: [], traces: [] } as unknown as BrainDocument;
    expect(() => recordNeuralChat(brain, "input", "")).toThrow();
    expect(() => recordNeuralChat(brain, "input", "?", "no-reply")).toThrow();
    const saved = recordNeuralChat(brain, "input", "", "no-reply");
    expect(saved.brain.messages.map(({ content }) => content)).toEqual(["input", ""]);
    expect(saved.generationEnd).toBe("no-reply");
  });

  it("publishes committed completion and terminal no-reply without stopping or cancelling", async () => {
    const { value, expected } = fixture();
    const saved = validateWorkerChatPresentation(value, expected)!;
    const result: ChatResult = {
      brain: { messages: [saved.humanMessage, saved.brainMessage], traces: [] } as unknown as BrainDocument,
      humanMessage: saved.humanMessage, brainMessage: saved.brainMessage,
      trace: { id: "trace-no-reply" } as ChatResult["trace"], generationEnd: "no-reply", proposedActions: []
    };
    const service = { chat: vi.fn(async () => result) };
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) };
    const controller = new ChatActionController(service, tools, { start: vi.fn() });
    const events: ChatStreamEvent[] = [];
    controller.on("stream", (event: ChatStreamEvent) => events.push(event));
    await expect(controller.send("brain", expected.input, undefined, expected.turnId)).resolves.toBe(result);
    const committed = events.find((event) => event.type === "chat-reply-committed");
    expect(committed).toMatchObject({ humanMessage: { content: expected.input }, brainMessage: { content: "", generationEnd: "no-reply" } });
    const terminal = events.at(-1)!;
    expect(terminal).toMatchObject({ type: "chat-state", state: "no-reply" });
    expect(advanceChatGenerationPhase("responding", terminal)).toBe("settled");
    expect(events.some((event) => event.type === "chat-state" && ["cancelled", "stopped", "failed"].includes(event.state))).toBe(false);
    expect(tools.cancel).not.toHaveBeenCalled();
  });

  it("keeps the compact disposition in metadata and blocks empty voice/copy actions", async () => {
    const source = await readFile("src/renderer/src/App.tsx", "utf8");
    const bubble = source.slice(source.indexOf("function MessageBubble("), source.indexOf("function CortexOrb("));
    expect(bubble).toContain('const noReply = isBrain && deliveryState === "no-reply";');
    expect(bubble).toContain("chatNoReplyPresentation(message)");
    expect(bubble).toContain("{noReply ? null : <div className=\"message__content\">{message.content}</div>}");
    expect(bubble).toContain("!noReply && Boolean(message.content.trim())");
    expect(bubble).toContain("{message.content ? <button aria-label=\"Copy response\"");
  });
});
