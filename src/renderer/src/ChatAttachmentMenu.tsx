import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { attachmentMenuPosition } from "./attachmentMenuPosition";
import { Icon } from "./icons";

/** Native top-layer popup: never clipped by the compact composer's scroller. */
export function ChatAttachmentMenu({ children, active = true }: { children: ReactNode; active?: boolean }) {
  const id = useId(), trigger = useRef<HTMLButtonElement>(null), menu = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const place = useCallback(() => {
    const source = trigger.current, popup = menu.current;
    if (!source || !popup?.matches(":popover-open")) return;
    const position = attachmentMenuPosition(source.getBoundingClientRect(), popup.getBoundingClientRect(),
      { width: window.innerWidth, height: window.innerHeight });
    Object.assign(popup.style, { left: `${position.left}px`, top: `${position.top}px`,
      maxHeight: `${position.maxHeight}px`, visibility: "visible" });
  }, []);
  useLayoutEffect(() => {
    if (!active && menu.current?.matches(":popover-open")) menu.current.hidePopover();
  }, [active]);
  useEffect(() => {
    if (!open) return;
    window.addEventListener("resize", place);
    document.addEventListener("scroll", place, true);
    return () => { window.removeEventListener("resize", place); document.removeEventListener("scroll", place, true); };
  }, [open, place]);
  return <section className="composer__attach-menu">
    <button ref={trigger} type="button" className="composer__attach-trigger" popoverTarget={id}
      aria-label="Add files" title="Add files" disabled={!active}><Icon name="file" size={18} /></button>
    <div ref={menu} id={id} popover="auto" className="composer__attach-options"
      aria-label="Attachment choices"
      onToggle={event => {
        const visible = event.currentTarget.matches(":popover-open"); setOpen(visible);
        if (visible) place(); else event.currentTarget.style.visibility = "hidden";
      }}
      onClickCapture={event => {
        if (event.target instanceof Element && event.target.closest("button")) menu.current?.hidePopover();
      }}>
      {children}
    </div>
  </section>;
}
