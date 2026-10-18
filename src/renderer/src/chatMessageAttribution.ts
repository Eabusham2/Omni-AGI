import { checkedChatInputProvenance } from "../../shared/chatInputProvenance";
import type { ChatMessage } from "../../shared/types";

/** Presentation only; the native receipt and main-owned ledger bind authority. */
export function chatMessageAttribution(message: ChatMessage, brainName: string): {
  name: string; connectedApp: boolean; verifiedShape: boolean;
} {
  if (message.role === "brain") return { name: brainName, connectedApp: false, verifiedShape: true };
  if (!message.inputProvenance) return { name: "You", connectedApp: false, verifiedShape: true };
  try {
    const source = checkedChatInputProvenance(message.inputProvenance);
    // Isolate presentation-only directional controls, preserving original data.
    const name = (source.serverLabel.trim() || source.serverId).replace(/\p{Bidi_Control}/gu, "");
    return { name: name || source.serverId, connectedApp: true, verifiedShape: true };
  } catch {
    return { name: "Unverified app source", connectedApp: true, verifiedShape: false };
  }
}
