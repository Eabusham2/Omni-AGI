import { useEffect, useRef, useState } from "react";
import type { ChatTaskAttachmentHandle, ChatTaskAttachmentsApi } from "../../shared/taskAttachments";
import type { ExperienceUploadKind } from "../../shared/uploadSupport";

export function mergeTaskAttachments(current: ChatTaskAttachmentHandle[], added: ChatTaskAttachmentHandle[]): ChatTaskAttachmentHandle[] {
  const byId = new Map(current.map(value => [value.id, value]));
  added.forEach(value => byId.set(value.id, value));
  return [...byId.values()];
}

/** Only stages immutable task bytes. Explicit Learn uses a different API. */
export function useTaskAttachmentDraft(brainId: string, api?: ChatTaskAttachmentsApi) {
  const [entries, setEntries] = useState<ChatTaskAttachmentHandle[]>([]);
  const [busy, setBusy] = useState(false), [status, setStatus] = useState("");
  const scope = useRef({ brainId, changes: 0 });
  if (scope.current.brainId !== brainId) scope.current = { brainId, changes: 0 };
  const operation = useRef<{ brainId: string; requestId: string } | undefined>(undefined);
  const entriesRef = useRef(entries);
  entriesRef.current = entries;

  useEffect(() => {
    const owner = scope.current, changes = owner.changes; let active = true;
    setEntries([]); setBusy(false); setStatus("");
    void api?.list(brainId, "staged").then(value => {
      if (active && scope.current === owner && owner.changes === changes) setEntries(value.filter(row => row.state === "staged"));
    }, failure => { if (active && scope.current === owner) setStatus(failure instanceof Error ? failure.message : String(failure)); });
    return () => {
      active = false;
      const request = operation.current;
      if (request?.brainId === brainId) { operation.current = undefined; void api?.cancelStage(brainId, request.requestId).catch(() => undefined); }
    };
  }, [api, brainId]);

  const stage = async (selection: ExperienceUploadKind = "files", files?: File[]): Promise<void> => {
    if (!api) { setStatus("Task attachments require the desktop runtime."); return; }
    if (operation.current) return;
    const owner = scope.current;
    owner.changes += 1;
    const request = { brainId, requestId: crypto.randomUUID() };
    operation.current = request; setBusy(true); setStatus("Preparing task attachments…");
    try {
      const added = files ? await api.stageDropped(request, files) : await api.stage({ ...request, selection });
      if (scope.current !== owner || operation.current !== request) return;
      if (added?.length) {
        setEntries(current => mergeTaskAttachments(current, added));
        setStatus("Attached for this task · not learned by attaching.");
      } else setStatus("No attachment selected.");
    } catch (failure) {
      if (scope.current === owner && operation.current === request) setStatus(failure instanceof Error ? failure.message : String(failure));
    } finally {
      if (operation.current === request) { operation.current = undefined; setBusy(false); }
    }
  };
  const remove = async (id: string): Promise<void> => {
    if (!api) return;
    const owner = scope.current;
    try {
      const removed = await api.remove(brainId, id);
      if (scope.current !== owner) return;
      if (removed) {
        owner.changes += 1;
        const next = entriesRef.current.filter(value => value.id !== id);
        entriesRef.current = next; setEntries(next);
        setStatus(next.length ? "Attached for this task · not learned by attaching." : "Attachment removed; no learning was started.");
      }
      else setStatus("This file could not be removed from the task draft.");
    } catch (failure) { if (scope.current === owner) setStatus(failure instanceof Error ? failure.message : String(failure)); }
  };
  const take = (): ChatTaskAttachmentHandle[] => {
    scope.current.changes += 1;
    const result = entriesRef.current.filter(value => value.brainId === brainId && value.state === "staged");
    entriesRef.current = []; setEntries([]); setStatus(""); return result;
  };
  const restore = async (ids: string[]): Promise<void> => {
    if (!api || !ids.length) return;
    const owner = scope.current;
    owner.changes += 1;
    const values = await api.byIds(brainId, ids);
    if (scope.current !== owner) return;
    const selected = new Set(ids);
    setEntries(current => mergeTaskAttachments(current, values.filter(value => selected.has(value.id) && value.state === "staged")));
    if (values.some(value => selected.has(value.id) && value.state === "bound")) {
      setStatus("Sent attachments remain with their original turn; they are not attached again automatically.");
    }
  };
  const cancel = (): void => {
    const request = operation.current;
    const owner = scope.current;
    if (request && api) { setStatus("Stopping attachment preparation…"); void api.cancelStage(request.brainId, request.requestId).catch(failure => {
      if (scope.current === owner) setStatus(failure instanceof Error ? failure.message : String(failure));
    }); }
  };
  return { entries: entries.filter(value => value.brainId === brainId), busy, status, stage, remove, take, restore, cancel };
}
