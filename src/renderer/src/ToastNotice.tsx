import { useLayoutEffect, useRef } from "react";
import { Icon } from "./icons";
import { createToastExpiry, toastReadingTime, type ToastExpiry } from "./toastExpiry";

/** Mount with a fresh key for each notice, even when its text is identical. */
export function ToastNotice({ text, onDismiss }: { text: string; onDismiss: () => void }) {
  const expiry = useRef<ToastExpiry | undefined>(undefined), dismiss = useRef(onDismiss);
  const hovered = useRef(false), focused = useRef(false); dismiss.current = onDismiss;
  useLayoutEffect(() => {
    const owner = createToastExpiry(toastReadingTime(text), () => dismiss.current(), () => performance.now(), (work, delay) => window.setTimeout(work, delay), handle => window.clearTimeout(handle));
    expiry.current = owner; if (!hovered.current && !focused.current) owner.resume();
    return () => { owner.dispose(); if (expiry.current === owner) expiry.current = undefined; };
  }, [text]);
  const resume = () => { if (!hovered.current && !focused.current) expiry.current?.resume(); };
  return <div className="toast" role="status" onPointerEnter={() => { hovered.current = true; expiry.current?.pause(); }} onPointerLeave={() => { hovered.current = false; resume(); }}
    onFocusCapture={() => { focused.current = true; expiry.current?.pause(); }} onBlurCapture={event => { if (!event.currentTarget.contains(event.relatedTarget)) { focused.current = false; resume(); } }}>
    <span className="toast__icon"><Icon name="info" size={16} /></span>
    <div className="toast__message" tabIndex={0} aria-label="Notification text">{text}</div>
    <button type="button" onClick={onDismiss} aria-label="Dismiss notification"><Icon name="close" size={14} /></button>
  </div>;
}
