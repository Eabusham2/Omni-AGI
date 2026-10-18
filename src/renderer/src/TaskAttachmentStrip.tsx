import type { ChatTaskAttachmentHandle } from "../../shared/taskAttachments";
import { Icon } from "./icons";
import "./taskAttachments.css";

export function TaskAttachmentStrip({ entries, onRemove }: {
  entries: ChatTaskAttachmentHandle[]; onRemove?(id: string): void;
}) {
  if (!entries.length) return null;
  return <ul className="task-attachment-strip" aria-label={onRemove ? "Attachments for the next message" : "Message attachments"}>
    {entries.map(value => <li key={value.id}>
      <details><summary title={value.displayName}><Icon name={value.kind === "image" ? "image" : value.kind === "audio" ? "volume" : value.kind === "video" ? "video" : "file"} size={14} /><span>{value.displayName}</span></summary>
        <p>{value.displayName}</p><p>{value.mimeType} · {value.bytes.toLocaleString()} bytes</p><p>Attached for the task; attachment alone does not start training.</p>
        <p className="task-attachment-strip__hash">SHA-256: {value.contentSha256}</p></details>
      {onRemove ? <button type="button" aria-label={`Remove attachment ${value.displayName}`} title="Remove unsent attachment" onClick={() => onRemove(value.id)}><Icon name="close" size={13} /></button> : null}
    </li>)}
  </ul>;
}
