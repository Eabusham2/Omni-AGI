import { createHash } from "node:crypto";
import type { ChatInputAcceptedReceipt, ChatMessage } from "../shared/types";

export const chatInputHash = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");
const record = (value: unknown): Record<string, unknown> | undefined => value && typeof value === "object" && !Array.isArray(value)
  ? value as Record<string, unknown> : undefined;
const id = (value: unknown): value is string => typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/.test(value);

/** Input admission is not a completed reply, a generated answer or an inference. */
export function receivedChatInput(value: unknown, expected: { turnId: string; input: string }): {
  inputReceipt: ChatInputAcceptedReceipt; humanMessage: ChatMessage;
} {
  const raw = record(value);
  if (!raw || raw.committed !== true || raw.turnId !== expected.turnId || raw.inputSha256 !== chatInputHash(expected.input) ||
      !/^[a-f0-9]{64}$/.test(String(raw.inputSha256)) || !id(raw.humanMessageId) || !id(raw.afterimageId) ||
      typeof raw.createdAt !== "string" || !Number.isFinite(Date.parse(raw.createdAt)) ||
      !Number.isSafeInteger(raw.attentionEpoch) || Number(raw.attentionEpoch) < 1 ||
      raw.slowJobId !== undefined && !id(raw.slowJobId)) {
    throw new Error("Received input has no exact durable human/turn binding.");
  }
  const inputReceipt: ChatInputAcceptedReceipt = { committed: true, turnId: expected.turnId,
    inputSha256: String(raw.inputSha256), humanMessageId: raw.humanMessageId, afterimageId: raw.afterimageId,
    createdAt: raw.createdAt, attentionEpoch: Number(raw.attentionEpoch),
    ...(typeof raw.slowJobId === "string" ? { slowJobId: raw.slowJobId } : {}) };
  return { inputReceipt, humanMessage: { id: inputReceipt.humanMessageId, role: "human", content: expected.input,
    createdAt: inputReceipt.createdAt, turnId: inputReceipt.turnId, attentionEpoch: inputReceipt.attentionEpoch,
    runtime: "adaptive-core", status: "complete", inputReceipt } };
}

/** The original ledger human row remains immutable when a reply later saves. */
export function receivedChatInputFromLedger(receipt: unknown, humanValue: unknown, expected?: { turnId: string; input: string }) {
  const human = record(humanValue), raw = record(receipt);
  if (!human || !raw || raw.committed !== true || human.role !== "human" || human.input_accepted_before_reply !== true ||
      human.id !== raw.humanMessageId || typeof human.content !== "string" ||
      (human.turn_id ?? human.turnId) !== raw.turnId || raw.inputSha256 !== chatInputHash(human.content) ||
      raw.createdAt !== undefined && raw.createdAt !== (human.created_at ?? human.createdAt) ||
      raw.attentionEpoch !== undefined && raw.attentionEpoch !== (human.attention_epoch ?? human.attentionEpoch)) {
    throw new Error("Received-input receipt references unavailable or different human ledger state.");
  }
  if (expected && (expected.turnId !== raw.turnId || expected.input !== human.content)) throw new Error("Received input belongs to another request.");
  return receivedChatInput({ ...raw, committed: true,
    createdAt: raw.createdAt ?? human.created_at ?? human.createdAt,
    attentionEpoch: raw.attentionEpoch ?? human.attention_epoch ?? human.attentionEpoch },
    expected ?? { turnId: String(raw.turnId), input: human.content });
}
