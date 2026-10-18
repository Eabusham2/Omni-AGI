import { lstat } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join } from "node:path";
import type { ChatHostSuspensionReceipt } from "../shared/chatSuspension";
import type { ChatInputAcceptedReceipt, ChatMessage } from "../shared/types";
import { InstanceContinuationDetachedError } from "./instanceContinuationDetached";
import { ConversationLedger } from "./conversationLedger";

/** Only a main-owned acknowledged detach can create this disposition. */
export function chatHostSuspensionForDetach(error: unknown, expected: {
  brainId: string; turnId: string; inputSha256: string; inputReceipt?: ChatInputAcceptedReceipt;
}): ChatHostSuspensionReceipt | undefined {
  if (!(error instanceof InstanceContinuationDetachedError) || error.brainId !== expected.brainId || error.turnId !== expected.turnId) return undefined;
  if (!/^[a-f0-9]{64}$/.test(expected.inputSha256) || expected.inputReceipt &&
      (expected.inputReceipt.turnId !== expected.turnId || expected.inputReceipt.inputSha256 !== expected.inputSha256 || expected.inputReceipt.committed !== true)) throw new Error("Detached host input has inconsistent acceptance evidence.");
  const frontier = error.frontier;
  if (frontier && (frontier.turnId !== expected.turnId || frontier.inputSha256 !== expected.inputSha256 || frontier.exactSupportedResume !== true ||
      frontier.phase !== "native-text-token-boundary" || !/^(?:[a-f0-9]{32}|[a-f0-9]{64})$/.test(frontier.checkpointId) ||
      ![frontier.lastStreamSequence, frontier.sampledTokens, frontier.publishedTokens].every(value => Number.isSafeInteger(value) && value >= 0) ||
      frontier.publishedTokens > frontier.sampledTokens)) throw new Error("Detached host frontier differs from this saved turn.");
  return { format: "omni-chat-host-suspension", version: 1, presentationOnly: true, brainId: expected.brainId, turnId: expected.turnId,
    state: frontier ? "suspended" : "detached", inputSha256: expected.inputSha256, acknowledgedUnload: true, createdAt: new Date().toISOString(),
    ...(frontier ? { frontier: { checkpointId: frontier.checkpointId, turnId: frontier.turnId, inputSha256: frontier.inputSha256,
      lastStreamSequence: frontier.lastStreamSequence, sampledTokens: frontier.sampledTokens, publishedTokens: frontier.publishedTokens,
      phase: frontier.phase, exactSupportedResume: frontier.exactSupportedResume } } : {}),
    ...(error.inheritedExecutionId ? { inheritedExecutionId: error.inheritedExecutionId } : {}),
    ...(expected.inputReceipt ? { inputReceipt: structuredClone(expected.inputReceipt) } : {}) };
}

export interface ChatSuspensionSource { content: string; createdAt: string; attachmentIds: string[] }
/** Display-only append. It never touches native input, task scopes or effects. */
export async function persistChatHostSuspension(root: string, receipt: ChatHostSuspensionReceipt, source: ChatSuspensionSource): Promise<void> {
  for (const path of [root, join(root, "brain.json")]) { const info = await lstat(path);
    if (info.isSymbolicLink() || (path === root ? !info.isDirectory() : !info.isFile())) throw new Error("Suspension display history requires its existing managed owner."); }
  const message: ChatMessage = { id: `delivery-suspension-${createHash("sha256").update(JSON.stringify(receipt)).digest("hex")}`, role: "human", turnId: receipt.turnId,
    content: source.content, createdAt: source.createdAt, ...(source.attachmentIds.length ? { taskAttachmentIds: [...source.attachmentIds] } : {}),
    ...(receipt.inputReceipt?.nativeAudioInput ? { nativeAudioInput: receipt.inputReceipt.nativeAudioInput } : {}),
    deliveryReceipt: { schemaVersion: 1, presentationOnly: true, turnId: receipt.turnId, state: receipt.state, updatedAt: receipt.createdAt,
      ...(source.attachmentIds.length ? { attachmentIds: [...source.attachmentIds] } : {}), suspension: receipt } };
  const ledger = await ConversationLedger.open(root, receipt.brainId);
  try { ledger.append([{ kind: "message", value: message }]); } finally { ledger.close(); }
}
