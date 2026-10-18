import { useLayoutEffect, useRef, useState } from "react";
import { Icon } from "./icons";
import type { ChatDirectionStripItem } from "./chatDirectionPresentation";
import { chatDirectionRevisionText, chatDirectionStatus } from "./chatDirectionPresentation";
export type { ChatDirectionStripItem } from "./chatDirectionPresentation";
export { chatDirectionStatus } from "./chatDirectionPresentation";
import "./chatDirections.css";

export function ChatDirectionStrip({ items, total, canSteer, busy, hasOlder, onOlder, onDismiss, onRevise }: {
  items: ChatDirectionStripItem[]; total: number; canSteer: boolean; busy: boolean; hasOlder: boolean;
  onOlder(): void; onDismiss(id: string): Promise<void>; onRevise(text: string): Promise<void>;
}) {
  const [expanded, setExpanded] = useState<string>();
  const [draft, setDraft] = useState("");
  const [operation, setOperation] = useState<string>();
  const [error, setError] = useState("");
  const shown = items.filter(value => !value.dismissed);
  const list = useRef<HTMLOListElement>(null), following = useRef(true);
  const previous = useRef<{ first?: string; last?: string; height: number; top: number } | undefined>(undefined);
  useLayoutEffect(() => {
    const element = list.current;
    if (!element) return;
    const first = shown[0]?.id, last = shown.at(-1)?.id, prior = previous.current;
    if (prior && first !== prior.first && last === prior.last) {
      // Earlier paging prepends rows without moving the entry being read.
      element.scrollTop = prior.top + element.scrollHeight - prior.height;
    } else if ((!prior || last !== prior.last) && following.current) {
      element.scrollTop = element.scrollHeight;
    }
    previous.current = { first, last, height: element.scrollHeight, top: element.scrollTop };
  });
  if (!shown.length && !hasOlder) return null;
  const act = async (id: string, action: () => Promise<void>): Promise<void> => {
    if (operation) return;
    setOperation(id); setError("");
    try { await action(); setExpanded(undefined); }
    catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setOperation(undefined); }
  };
  return <section className="chat-queue-stack chat-direction-strip" aria-label="Steering directions">
    <header>Steering{total ? ` · ${total.toLocaleString()} recorded` : ""}</header>
    <ol ref={list} onScroll={event => {
      const element = event.currentTarget;
      following.current = element.scrollHeight - element.clientHeight - element.scrollTop < 24;
      if (previous.current) previous.current.top = element.scrollTop;
    }}>{shown.map(item => <li key={item.id}>
      <div className="chat-queue-stack__row">
        <button type="button" className="chat-queue-stack__text" title={item.content} aria-expanded={expanded === item.id}
          onClick={() => { setExpanded(current => current === item.id ? undefined : item.id); setDraft(chatDirectionRevisionText(item)); setError(""); }}>
          <span>{item.content}</span></button>
        <small>{chatDirectionStatus(item)}</small>
        <button type="button" title="Hide this direction; its recorded input and learning remain saved"
          aria-label="Dismiss steering direction" disabled={busy || Boolean(operation)}
          onClick={() => void act(item.id, () => onDismiss(item.id))}><Icon name="close" size={13} /></button>
      </div>
      {expanded === item.id ? <div className="chat-direction-strip__expanded">
        <p>{item.content}</p>
        <label>Revise as a new direction<textarea rows={3} value={draft}
          onChange={event => setDraft(event.currentTarget.value)} /></label>
        <button type="button" className="button button--ghost" disabled={busy || Boolean(operation) || !draft.trim()}
          onClick={() => void act(item.id, () => onRevise(draft))}>{canSteer ? "Steer with revision" : "Use in composer"}</button>
      </div> : null}
    </li>)}</ol>
    {hasOlder ? <button type="button" className="button button--ghost" disabled={busy || Boolean(operation)} onClick={onOlder}>Earlier directions</button> : null}
    {error ? <p role="alert">{error}</p> : null}
  </section>;
}
