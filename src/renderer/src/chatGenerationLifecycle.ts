import type { ChatMessage, ChatStreamEvent } from "../../shared/types";

export type ChatGenerationPhase = "responding" | "reply-complete" | "settled";

/** Reply completion is monotonic. Artifact updates cannot reopen generation. */
export function advanceChatGenerationPhase(
  phase: ChatGenerationPhase,
  event: ChatStreamEvent
): ChatGenerationPhase {
  if (phase === "settled") return phase;
  if (event.type === "chat-state" &&
      ["complete", "steered", "stopped", "no-reply", "failed", "cancelled"].includes(event.state)) return "settled";
  if (event.type === "chat-phase" || event.type === "chat-reply-committed") {
    return "reply-complete";
  }
  return phase;
}

export interface UncommittedChatOutput {
  turnId: string;
  message: ChatMessage;
  state: "saving" | "steering" | "failed" | "cancelled";
}

/** Final output stays visible, explicitly provisional, until its exact save. */
export function retainUncommittedChatOutput(
  outputs: UncommittedChatOutput[],
  turnId: string,
  content: string,
  createdAt: string
): UncommittedChatOutput[] {
  if (!content || outputs.some((output) => output.turnId === turnId)) return outputs;
  return [...outputs, {
    turnId,
    state: "saving",
    message: { id: `reply-output-${turnId}`, turnId, role: "brain", content, createdAt }
  }];
}

/** Identity, not matching prose, retires a provisional completed output. */
export function reconcileUncommittedChatOutputs(
  outputs: UncommittedChatOutput[],
  committedTurnId: string
): UncommittedChatOutput[] {
  return outputs.filter((output) => output.turnId !== committedTurnId);
}

export function settleUncommittedChatOutput(
  outputs: UncommittedChatOutput[],
  turnId: string,
  state: "failed" | "cancelled"
): UncommittedChatOutput[] {
  return outputs.map((output) => output.turnId === turnId ? { ...output, state } : output);
}
