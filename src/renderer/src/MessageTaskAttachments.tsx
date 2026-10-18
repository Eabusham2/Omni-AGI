import { useEffect, useState } from "react";
import type { ChatTaskAttachmentHandle } from "../../shared/taskAttachments";
import { TaskAttachmentStrip } from "./TaskAttachmentStrip";

/** One exact turn query, not an unbounded scan of attachment history. */
export function MessageTaskAttachments({ brainId, turnId, attachmentIds, received }: { brainId: string; turnId: string; attachmentIds: string[]; received: boolean }) {
  const [entries, setEntries] = useState<ChatTaskAttachmentHandle[]>([]);
  const [error, setError] = useState("");
  const idsKey = JSON.stringify(attachmentIds);
  useEffect(() => {
    let active = true; setEntries([]); setError("");
    const api = window.omni?.chat.attachments;
    if (!api?.forTurn) return;
    const request = received ? api.forTurn(brainId, turnId) : api.byIds(brainId, JSON.parse(idsKey) as string[]);
    void request.then(value => { if (active) setEntries(value); }, failure => {
      if (active) setError(failure instanceof Error ? failure.message : String(failure));
    });
    return () => { active = false; };
  }, [brainId, turnId, received, idsKey]);
  return <><TaskAttachmentStrip entries={entries} />{error ? <p role="status">Attachment details unavailable: {error}</p> : null}</>;
}
