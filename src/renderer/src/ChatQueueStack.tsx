import { useState } from "react";
import { Icon } from "./icons";
import type { ChatTaskAttachmentHandle } from "../../shared/taskAttachments";
import { TaskAttachmentStrip } from "./TaskAttachmentStrip";

export interface QueueStackItem { id: string; text: string; attachments?: ChatTaskAttachmentHandle[] }

export function ChatQueueStack({ items, busy, onEdit, onRemove, onMove, onSendNow }: {
  items: QueueStackItem[];
  busy: boolean;
  onEdit: (id: string, text: string) => void;
  onRemove: (id: string) => void;
  onMove: (id: string, direction: -1 | 1) => void;
  onSendNow: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);
  if (!items.length) return null;
  return <section className="chat-queue-stack" aria-label="Queued messages">
    <header>{items.length} queued</header>
    <ol>{items.map((item, index) => <li key={item.id}>
      <div className="chat-queue-stack__row">
        <button type="button" className="chat-queue-stack__text" title={item.text}
          aria-expanded={expanded === item.id} aria-label={`Edit queued message ${index + 1}`}
          onClick={() => setExpanded(current => current === item.id ? null : item.id)}>
          <span>{item.text}</span>
        </button>
        <button type="button" title={busy ? "Steer the current reply with this message" : "Send this message now"}
          aria-label={`${busy ? "Steer with" : "Send"} queued message ${index + 1}`}
          disabled={!item.text.trim()} onClick={() => onSendNow(item.id)}>
          {busy ? "Steer" : "Send now"}
        </button>
        <button type="button" aria-label={`Move queued message ${index + 1} earlier`} title="Move earlier"
          disabled={index === 0} onClick={() => onMove(item.id, -1)}>↑</button>
        <button type="button" aria-label={`Move queued message ${index + 1} later`} title="Move later"
          disabled={index === items.length - 1} onClick={() => onMove(item.id, 1)}>↓</button>
        <button type="button" aria-label={`Remove queued message ${index + 1}`} title="Remove from queue"
          onClick={() => onRemove(item.id)}><Icon name="close" size={13} /></button>
      </div>
      {item.attachments?.length ? <TaskAttachmentStrip entries={item.attachments} /> : null}
      {expanded === item.id ? <textarea aria-label={`Queued message ${index + 1}`} value={item.text}
        onChange={event => onEdit(item.id, event.target.value)} rows={3} /> : null}
    </li>)}</ol>
  </section>;
}
