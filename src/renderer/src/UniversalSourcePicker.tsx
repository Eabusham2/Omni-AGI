import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { attachmentMenuPosition } from "./attachmentMenuPosition";
import { Icon, type IconName } from "./icons";

export type UniversalSourceSelection = "files" | "folder";

/** One entry point for documents, datasets and media, with native folder selection. */
export function UniversalSourcePicker({ label, icon = "file", disabled = false, active = true, onSelect }: {
  label: string;
  icon?: IconName;
  disabled?: boolean;
  active?: boolean;
  onSelect: (selection: UniversalSourceSelection) => void;
}) {
  const id = useId();
  const trigger = useRef<HTMLButtonElement>(null), menu = useRef<HTMLDivElement>(null);
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
    if ((!active || disabled) && menu.current?.matches(":popover-open")) menu.current.hidePopover();
  }, [active, disabled]);
  useEffect(() => {
    if (!open) return;
    window.addEventListener("resize", place);
    document.addEventListener("scroll", place, true);
    return () => { window.removeEventListener("resize", place); document.removeEventListener("scroll", place, true); };
  }, [open, place]);
  const choose = (selection: UniversalSourceSelection): void => {
    menu.current?.hidePopover();
    if (active && !disabled) onSelect(selection);
  };
  return <div className="universal-source-picker">
    <button ref={trigger} type="button" className="universal-source-picker__trigger" popoverTarget={id}
      aria-label={label} title={label} disabled={!active || disabled}>
      <Icon name={icon} size={16} /><span>{label}</span>
    </button>
    <div ref={menu} id={id} popover="auto" className="composer__attach-options universal-source-picker__menu"
      aria-label={`${label} source`}
      onToggle={event => {
        const visible = event.currentTarget.matches(":popover-open"); setOpen(visible);
        if (visible) place(); else event.currentTarget.style.visibility = "hidden";
      }}>
      <button type="button" onClick={() => choose("files")} disabled={!active || disabled}>
        <Icon name="file" size={16} /> Files &amp; media
      </button>
      <button type="button" onClick={() => choose("folder")} disabled={!active || disabled}>
        <Icon name="archive" size={16} /> Folder
      </button>
    </div>
  </div>;
}
