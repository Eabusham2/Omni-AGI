export interface InheritedInputAlias {
  humanMessageId: string; originalTurnId: string; attentionEpoch: number; inputSha256: string;
}
const object = (value: unknown): Record<string, unknown> | undefined =>
  value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined;

/** A committed native root receipt, never authority to rerun an old action. */
export function completedInheritedTurn(metadata: unknown, brainId: string, executionId: string): {
  turnId: string; humanMessageId: string; inputSha256: string; inferenceCount: number;
} | undefined {
  if (!/^[a-f0-9]{64}$/.test(executionId)) throw new Error("Invalid inherited completion identity.");
  const root = object(metadata), queue = object(root?.connected_executions);
  if (!root || root.brain_id !== brainId) return undefined;
  if (!queue || queue.format !== "omni-connected-execution-queue" || queue.formatVersion !== 1 || !Array.isArray(queue.entries)) return undefined;
  const matches = queue.entries.map(object).filter(entry => entry?.id === executionId);
  if (matches.length > 1) throw new Error("Inherited completion has conflicting root queue owners.");
  const entry = matches[0]; if (!entry || entry.status !== "complete") return undefined;
  const completion = object(entry.completion), turnId = entry.rootTurnId;
  if (typeof turnId !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(turnId) ||
    completion?.turnId !== turnId || typeof completion.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(completion.sha256) ||
    typeof completion.resultPath !== "string" || completion.resultPath.startsWith("/") || completion.resultPath.includes("\\") || completion.resultPath.includes("\0") ||
    completion.resultPath.split("/").some(part => !part || part === "." || part === "..")) {
    throw new Error("Inherited completion is not bound to its current root turn.");
  }
  const receipts = Array.isArray(root.completed_chat_turns) ? root.completed_chat_turns.map(object).filter(receipt => receipt?.turnId === turnId) : [];
  const receipt = receipts[0];
  if (receipts.length !== 1 || receipt?.format !== "omni-completed-chat-turn" || receipt.formatVersion !== 1 ||
    receipt.humanMessageId !== entry.rootHumanMessageId || typeof receipt.humanMessageId !== "string" || !receipt.humanMessageId ||
    typeof receipt.inputSha256 !== "string" || !/^[a-f0-9]{64}$/.test(receipt.inputSha256) ||
    !Number.isSafeInteger(receipt.inferenceCount) || Number(receipt.inferenceCount) < 1) {
    throw new Error("Inherited completion lost its exact committed root receipt.");
  }
  return { turnId, humanMessageId: receipt.humanMessageId, inputSha256: receipt.inputSha256, inferenceCount: Number(receipt.inferenceCount) };
}
