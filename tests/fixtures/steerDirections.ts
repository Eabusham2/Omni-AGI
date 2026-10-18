import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { SteerDirectionSource } from "../../src/shared/steerDirections";
import type { ChatMessage, ChatResult, ChatTurnMetadata } from "../../src/shared/types";
import { nativeAudioIdentityJson, nativeAudioMessageInfo, type NativeAudioInput } from "../../src/shared/nativeAudioChat";
import { receivedChatInput } from "../../src/main/chatInputReceipt";

export const at = "2026-10-13T13:00:00Z";
export const sha = (value: string | Uint8Array) => createHash("sha256").update(value).digest("hex");
export async function managed(root: string, owner: string): Promise<void> {
  await mkdir(root, { recursive: true }); await writeFile(join(root, "brain.json"), JSON.stringify({ id: owner }));
}
export function source(turnId = "direction-one", owner = "owner", content = "Use the blue version"): SteerDirectionSource {
  return { brainId: owner, turnId, predecessorTurnId: "original-task", createdAt: at, content, attachmentIds: [], inputKind: "text" };
}
export function metadata(turnId = "original-task", attachmentIds?: string[]): ChatTurnMetadata {
  return { kind: "steer", source: "human", replacesTurnId: turnId, createdAt: at, ...(attachmentIds ? { attachmentIds } : {}) };
}
export function accepted(turnId = "direction-one", input = "Use the blue version", attachmentIds: string[] = [], audio?: NativeAudioInput) {
  const received = receivedChatInput({ committed: true, turnId, inputSha256: audio?.inputSha256 ?? sha(input),
    humanMessageId: `human-${turnId}`, afterimageId: `afterimage-${turnId}`, createdAt: at, attentionEpoch: 3,
    ...(audio ? { nativeAudioInput: nativeAudioMessageInfo({ input: audio, assemblyId: "sensory-assembly", sourceBytes: audio.attachment.bytes,
      sourceSha256: audio.attachment.contentSha256, decodedWindows: 1, modelSamples: 4, modelSampleRate: 16000,
      wholeInputTraversed: true, nativeSensory: true, transcriptUsed: false }) } : {}) }, { turnId, input, ...(audio ? { nativeAudioInput: audio } : {}) });
  if (attachmentIds.length) { received.inputReceipt.taskAttachmentIds = [...attachmentIds]; received.humanMessage.taskAttachmentIds = [...attachmentIds]; }
  return received;
}
export function result(humanMessage: ChatMessage, ending?: ChatResult["generationEnd"]): ChatResult {
  return { brain: { messages: [], traces: [] } as unknown as ChatResult["brain"], humanMessage,
    brainMessage: { id: `brain-${humanMessage.turnId}`, turnId: humanMessage.turnId, role: "brain", content: "Synthetic saved reply; no model invoked", createdAt: at },
    trace: { id: `trace-${humanMessage.turnId}` } as ChatResult["trace"], proposedActions: [], ...(ending ? { generationEnd: ending } : {}) };
}
export function audio(turnId = "audio-direction", owner = "owner"): NativeAudioInput {
  const value = { format: "omni-native-audio-input" as const, version: 1 as const,
    attachment: { brainId: owner, turnId, attachmentId: "original-audio", contentSha256: "a".repeat(64), bytes: 100,
      mimeType: "audio/wav", kind: "audio", relativePath: `attachments/blobs/${"a".repeat(64)}.data`, displayName: "Microphone audio", manifestSha256: "b".repeat(64) },
    capture: { sessionId: "private-owned-lease", utteranceId: "utterance-one", sequence: 2, capturedAt: at, permissionRevision: 7 } };
  return { ...value, inputSha256: sha(nativeAudioIdentityJson(value)) };
}
