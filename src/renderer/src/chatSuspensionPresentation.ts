import type { ChatMessage } from "../../shared/types";
import type { ChatHostSuspensionReceipt } from "../../shared/chatSuspension";

export function pausedOutputLabel(state?: string): string | undefined {
  if (state === "suspended") return "Reply paused · checkpoint retained";
  if (state === "detached") return "Host detached · partial output retained";
  return undefined;
}

/** Lost stream delivery cannot make a main-certified saved cursor a send failure. */
export function savedChatSuspension(messages: ChatMessage[], brainId: string, turnId: string): ChatHostSuspensionReceipt | undefined {
  // Ledger pages arrive in append order. Never revive an older suspension
  // after a later Stop, real saved reply, or other current disposition.
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]!;
    if ((message.turnId ?? message.deliveryReceipt?.turnId) !== turnId) continue;
    const delivery = message.deliveryReceipt, suspension = delivery?.suspension;
    if (message.role !== "human" || delivery?.presentationOnly !== true || delivery.turnId !== turnId ||
        !suspension || suspension.format !== "omni-chat-host-suspension" || suspension.version !== 1 ||
        suspension.presentationOnly !== true || suspension.acknowledgedUnload !== true || suspension.brainId !== brainId ||
        suspension.turnId !== turnId || suspension.state !== delivery.state ||
        !["suspended", "detached"].includes(suspension.state)) return undefined;
    if (suspension.state === "suspended" && (!suspension.frontier?.exactSupportedResume ||
        suspension.frontier.turnId !== turnId || suspension.frontier.inputSha256 !== suspension.inputSha256)) return undefined;
    return suspension;
  }
  return undefined;
}
