import { createHash } from "node:crypto";
import { checkedNativeAudioInput, checkedNativeAudioReceipt, nativeAudioIdentityJson,
  nativeAudioMessageInfo, type NativeAudioInput, type NativeAudioInputReceipt, type NativeAudioMessageInfo } from "../shared/nativeAudioChat";

const sha = (value: string): string => createHash("sha256").update(value, "utf8").digest("hex");

export function verifiedNativeAudioInput(value: unknown): NativeAudioInput {
  const input = checkedNativeAudioInput(value);
  if (sha(nativeAudioIdentityJson(input)) !== input.inputSha256) throw new Error("Native audio input checksum differs from its exact typed source.");
  return input;
}

export function verifiedNativeAudioReceipt(value: unknown): NativeAudioInputReceipt {
  const receipt = checkedNativeAudioReceipt(value);
  verifiedNativeAudioInput(receipt.input);
  return receipt;
}

export function expectedChatInputHash(text: string, audio?: NativeAudioInput): string {
  if (!audio) return sha(text);
  if (text !== "") throw new Error("Native audio cannot contain a transcript substitute.");
  return verifiedNativeAudioInput(audio).inputSha256;
}

export function observedChatInputHash(text: string, audio?: unknown): string {
  return audio === undefined || audio === null ? sha(text) : expectedChatInputHash(text, verifiedNativeAudioReceipt(audio).input);
}

export function nativeAudioMessageSource(value: Record<string, unknown>): { nativeAudioInput?: NativeAudioMessageInfo } {
  const raw = value.native_audio_input ?? value.nativeAudioInput;
  return raw === undefined || raw === null ? {} : { nativeAudioInput: nativeAudioMessageInfo(verifiedNativeAudioReceipt(raw)) };
}
