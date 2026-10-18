import type { SteerDirectionRecord } from "../../shared/steerDirections";
import type { ChatMessage } from "../../shared/types";
/** View data only. The main-owned direction record owns receipt/authority. */
export interface ChatDirectionStripItem {
  id: string;
  content: string;
  /** Actual editable text only; an audio display label is never submitted. */
  revisionText?: string;
  state: "pending" | "received" | "complete" | "failed" | "stopped" | "steered";
  inputReceived: boolean;
  dormant: boolean;
  dormantReason?: NonNullable<SteerDirectionRecord["dormant"]>["reason"];
  dismissed: boolean;
}

export function chatDirectionStatus(item: ChatDirectionStripItem): string {
  if (item.dormantReason === "saved-continuation") return "Saved turn · paused";
  if (item.dormantReason === "process-interrupted") return "Interrupted · not resent";
  if (item.dormant) return "Saved copy · not resent";
  if (item.state === "pending") return "Waiting";
  if (item.state === "received") return "Received";
  if (item.state === "complete") return "Completed";
  if (item.state === "steered") return "Replaced by a later direction";
  if (item.state === "stopped") return item.inputReceived ? "Received · reply stopped" : "Stopped before receipt";
  return item.inputReceived ? "Received · reply did not finish" : "Not received";
}

export function mergeChatDirections(current: SteerDirectionRecord[], incoming: SteerDirectionRecord[]): SteerDirectionRecord[] {
  const records = new Map(current.map(value => [value.id, value]));
  for (const value of incoming) {
    const previous = records.get(value.id);
    if (previous && (previous.brainId !== value.brainId || previous.bindingSha256 !== value.bindingSha256)) {
      throw new Error("A recorded steering direction changed its immutable identity.");
    }
    if (!previous || value.revision >= previous.revision) records.set(value.id, value);
  }
  return [...records.values()].sort((a, b) => a.sequence - b.sequence);
}

/** Never hide the predecessor task just because its reply was steered. */
export function isSteerDirectionMessage(message: ChatMessage, directions: SteerDirectionRecord[]): boolean {
  const messageTurnId = message.turnId ?? message.deliveryReceipt?.turnId;
  if (message.role !== "human" || !messageTurnId) return false;
  return directions.some(direction => {
    if ((direction.humanTurnId ?? direction.turnId) !== messageTurnId) return false;
    if (direction.humanMessageId === message.id) return true;
    const pending = message.deliveryReceipt?.presentationOnly === true || message.id === `pending-${direction.turnId}` || message.id === `queued-${direction.turnId}`;
    return pending && message.content === (direction.source.submittedContent ?? direction.source.content) &&
      message.createdAt === direction.source.createdAt;
  });
}

export function chatDirectionStripItem(record: SteerDirectionRecord): ChatDirectionStripItem {
  return { id: record.id, content: record.source.content || (record.source.inputKind === "native-audio" ? "Voice direction" : ""),
    revisionText: record.source.content,
    state: record.state, inputReceived: record.inputReceipt?.committed === true,
    dormant: Boolean(record.dormant), dormantReason: record.dormant?.reason, dismissed: Boolean(record.dismissedAt) };
}

export function chatDirectionRevisionText(item: ChatDirectionStripItem): string {
  return item.revisionText ?? item.content;
}
