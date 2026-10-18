import type { ChatMessage } from "../shared/types";
import type { ChatTaskAttachmentHandle } from "../shared/taskAttachments";

/** Public conversation projection contains stable IDs only; raw paths remain main/native-owned. */
export function projectTaskAttachmentIds(
  message: ChatMessage,
  attachments: readonly ChatTaskAttachmentHandle[]
): ChatMessage {
  if (message.role !== "human" || !attachments.length) return message;
  const taskAttachmentIds = attachments.map((item) => item.id);
  const turnIds = new Set(attachments.map((item) => item.turnId).filter((value): value is string => Boolean(value)));
  if (new Set(taskAttachmentIds).size !== taskAttachmentIds.length) {
    throw new Error("Task attachment projection contains duplicate identities.");
  }
  if (turnIds.size !== 1 || attachments.some((item) => item.state !== "bound") ||
      message.turnId !== undefined && !turnIds.has(message.turnId) ||
      message.inputReceipt && !turnIds.has(message.inputReceipt.turnId)) {
    throw new Error("Task attachment projection belongs to another or unbound turn.");
  }
  const [turnId] = [...turnIds];
  return {
    ...message,
    turnId,
    taskAttachmentIds,
    ...(message.inputReceipt
      ? { inputReceipt: { ...message.inputReceipt, taskAttachmentIds: [...taskAttachmentIds] } }
      : {})
  };
}
