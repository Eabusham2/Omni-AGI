import type { InstanceResumeFrontier } from "./instanceLifecycle";
import type { ChatInputAcceptedReceipt } from "./types";

/** A host attachment ended after acknowledged Unload, not native completion. */
export interface ChatHostSuspensionReceipt {
  format: "omni-chat-host-suspension";
  version: 1;
  presentationOnly: true;
  brainId: string;
  turnId: string;
  state: "suspended" | "detached";
  inputSha256: string;
  acknowledgedUnload: true;
  createdAt: string;
  /** Present only when the actual saved cursor has this exact input/turn. */
  frontier?: InstanceResumeFrontier;
  inheritedExecutionId?: string;
  /** Acceptance remains separate from a saved cursor or an assistant reply. */
  inputReceipt?: ChatInputAcceptedReceipt;
}

export function chatHostSuspensionStatus(receipt: ChatHostSuspensionReceipt): string {
  if (receipt.state === "suspended") return receipt.inputReceipt
    ? "Input saved · reply suspended at its retained checkpoint."
    : "Turn suspended at its retained checkpoint.";
  return receipt.inputReceipt
    ? "Input remains saved · host attachment detached."
    : "Host attachment detached · this pending input was not confirmed received.";
}
