import { createHash } from "node:crypto";
import type { ChatInputAcceptedReceipt, ChatMessage } from "../shared/types";
import { checkedChatInputProvenance, type ChatInputProvenance } from "../shared/chatInputProvenance";
import { checkedNativeAudioInfo, nativeAudioMessageInfo, type NativeAudioInput } from "../shared/nativeAudioChat";
import { expectedChatInputHash, observedChatInputHash, verifiedNativeAudioReceipt } from "./nativeAudioIdentity";

export const chatInputHash = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");
const record = (value: unknown): Record<string, unknown> | undefined => value && typeof value === "object" && !Array.isArray(value)
  ? value as Record<string, unknown> : undefined;
const id = (value: unknown): value is string => typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/.test(value);

/** Input admission is not a completed reply, a generated answer or an inference. */
export function receivedChatInput(value: unknown, expected: { turnId: string; input: string; inputProvenance?: ChatInputProvenance; nativeAudioInput?: NativeAudioInput }): {
  inputReceipt: ChatInputAcceptedReceipt; humanMessage: ChatMessage;
} {
  const raw = record(value);
  if (!raw || raw.committed !== true || raw.turnId !== expected.turnId || raw.inputSha256 !== expectedChatInputHash(expected.input, expected.nativeAudioInput) ||
      !/^[a-f0-9]{64}$/.test(String(raw.inputSha256)) || !id(raw.humanMessageId) || !id(raw.afterimageId) ||
      typeof raw.createdAt !== "string" || !Number.isFinite(Date.parse(raw.createdAt)) ||
      !Number.isSafeInteger(raw.attentionEpoch) || Number(raw.attentionEpoch) < 0 ||
      raw.slowJobId !== undefined && !id(raw.slowJobId)) {
    throw new Error("Received input has no exact durable task/turn binding.");
  }
  const provenance = raw.inputProvenance === undefined ? undefined : checkedChatInputProvenance(raw.inputProvenance);
  const audioRecord = record(raw.nativeAudioInput);
  const audio = audioRecord === undefined ? undefined : "input" in audioRecord
    ? nativeAudioMessageInfo(verifiedNativeAudioReceipt(audioRecord)) : checkedNativeAudioInfo(audioRecord, expected.nativeAudioInput);
  if ((audio === undefined) !== (expected.nativeAudioInput === undefined) || audio && provenance) throw new Error("Received audio input differs from its owned raw capture.");
  if (audio) checkedNativeAudioInfo(audio, expected.nativeAudioInput);
  if (provenance && provenance.inputSha256 !== raw.inputSha256 || expected.inputProvenance &&
    JSON.stringify(provenance) !== JSON.stringify(checkedChatInputProvenance(expected.inputProvenance))) throw new Error("Received task source does not match the reviewed input.");
  const inputReceipt: ChatInputAcceptedReceipt = { committed: true, turnId: expected.turnId,
    inputSha256: String(raw.inputSha256), humanMessageId: raw.humanMessageId, afterimageId: raw.afterimageId,
    createdAt: raw.createdAt, attentionEpoch: Number(raw.attentionEpoch),
    ...(typeof raw.slowJobId === "string" ? { slowJobId: raw.slowJobId } : {}), ...(provenance ? { inputProvenance: provenance } : {}),
    ...(audio ? { nativeAudioInput: audio } : {}) };
  return { inputReceipt, humanMessage: { id: inputReceipt.humanMessageId, role: "human", content: expected.input,
    createdAt: inputReceipt.createdAt, turnId: inputReceipt.turnId, attentionEpoch: inputReceipt.attentionEpoch,
    runtime: "adaptive-core", status: "complete", inputReceipt, ...(provenance ? { inputProvenance: provenance } : {}),
    ...(audio ? { nativeAudioInput: audio } : {}) } };
}

/** The original ledger task row remains immutable when a reply later saves. */
export function receivedChatInputFromLedger(receipt: unknown, humanValue: unknown, expected?: { turnId: string; input: string; nativeAudioInput?: NativeAudioInput }) {
  const human = record(humanValue), raw = record(receipt);
  if (!human || !raw || raw.committed !== true || human.role !== "human" || human.input_accepted_before_reply !== true ||
      human.id !== raw.humanMessageId || typeof human.content !== "string" ||
      (human.turn_id ?? human.turnId) !== raw.turnId || raw.inputSha256 !== observedChatInputHash(human.content, human.native_audio_input ?? human.nativeAudioInput) ||
      raw.createdAt !== undefined && raw.createdAt !== (human.created_at ?? human.createdAt) ||
      raw.attentionEpoch !== undefined && raw.attentionEpoch !== (human.attention_epoch ?? human.attentionEpoch)) {
    throw new Error("Received-input receipt references unavailable or different human ledger state.");
  }
  if (expected && (expected.turnId !== raw.turnId || expected.input !== human.content)) throw new Error("Received input belongs to another request.");
  const ledgerProvenance = human.input_provenance ?? human.inputProvenance;
  const audio = raw.nativeAudioInput === undefined ? undefined : verifiedNativeAudioReceipt(raw.nativeAudioInput);
  const ledgerAudio = human.native_audio_input ?? human.nativeAudioInput;
  if (JSON.stringify(audio) !== JSON.stringify(ledgerAudio === undefined ? undefined : verifiedNativeAudioReceipt(ledgerAudio)) ||
      expected?.nativeAudioInput && JSON.stringify(audio?.input) !== JSON.stringify(expected.nativeAudioInput)) throw new Error("Raw audio attribution differs between receipt and ledger.");
  if (JSON.stringify(raw.inputProvenance === undefined ? undefined : checkedChatInputProvenance(raw.inputProvenance)) !==
    JSON.stringify(ledgerProvenance === undefined ? undefined : checkedChatInputProvenance(ledgerProvenance))) throw new Error("Task input source differs between its durable receipt and ledger.");
  return receivedChatInput({ ...raw, committed: true,
    createdAt: raw.createdAt ?? human.created_at ?? human.createdAt,
    attentionEpoch: raw.attentionEpoch ?? human.attention_epoch ?? human.attentionEpoch },
    { ...(expected ?? { turnId: String(raw.turnId), input: human.content }), ...(audio ? { nativeAudioInput: audio.input } : {}) });
}
