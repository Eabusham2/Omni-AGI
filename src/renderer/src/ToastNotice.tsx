import { useId, useLayoutEffect, useRef } from "react";
import { Icon } from "./icons";
import { createToastExpiry, toastReadingTime, type ToastExpiry } from "./toastExpiry";

type NoticeBox = Pick<DOMRectReadOnly, "left" | "right" | "top" | "bottom">;

/** Measured visual overlap, not a fixed extra margin or a model allocation. */
export function conversationNoticeSpace(conversation: NoticeBox, notice: NoticeBox): number {
  if (![conversation.left, conversation.right, conversation.top, conversation.bottom,
      notice.left, notice.right, notice.top, notice.bottom].every(Number.isFinite) ||
      conversation.right <= conversation.left || conversation.bottom <= conversation.top ||
      notice.right <= notice.left || notice.bottom <= notice.top ||
      notice.right <= conversation.left || notice.left >= conversation.right ||
      notice.bottom <= conversation.top || notice.top >= conversation.bottom) return 0;
  return Math.ceil(conversation.bottom - notice.top) + 8;
}

/** Mount with a fresh key for each notice, even when its text is identical. */
export function ToastNotice({ text, onDismiss }: { text: string; onDismiss: () => void }) {
  const notice = useRef<HTMLDivElement>(null), reservationId = useId();
  const expiry = useRef<ToastExpiry | undefined>(undefined), dismiss = useRef(onDismiss);
  const hovered = useRef(false), focused = useRef(false); dismiss.current = onDismiss;
  useLayoutEffect(() => {
    const element = notice.current;
    if (!element) return;
    const app = element.closest(".app-shell");
    if (!app) return; // Standalone notices have no conversation to reserve.
    let disposed = false, frame: number | undefined;
    let reserved: { node: HTMLElement; composer: HTMLElement; previousOwner: string | null; margin: string; marginPriority: string; space: string; spacePriority: string } | undefined;
    const observer = typeof ResizeObserver === "function" ? new ResizeObserver(() => schedule()) : undefined;
    const release = (): void => {
      if (!reserved) return;
      const value = reserved; reserved = undefined;
      observer?.unobserve(value.node); observer?.unobserve(value.composer);
      if (value.node.dataset.noticeReservation !== reservationId) return;
      if (value.previousOwner === null) value.node.removeAttribute("data-notice-reservation");
      else value.node.setAttribute("data-notice-reservation", value.previousOwner);
      for (const [name, previous, priority] of [
        ["--notice-composer-base-margin", value.margin, value.marginPriority],
        ["--conversation-notice-space", value.space, value.spacePriority]
      ]) {
        if (previous) value.node.style.setProperty(name!, previous, priority);
        else value.node.style.removeProperty(name!);
      }
    };
    const update = (): void => {
      if (disposed) return;
      const conversation = app.querySelector<HTMLElement>(".chat-layout:not([hidden]) .conversation");
      const composer = conversation?.querySelector<HTMLElement>(":scope > .call-layout > .call-layout__conversation > .composer-wrap");
      if (!conversation || !composer || !conversation.getClientRects().length) { release(); return; }
      const space = conversationNoticeSpace(conversation.getBoundingClientRect(), element.getBoundingClientRect());
      if (space === 0) { release(); return; }
      if (reserved?.node !== conversation || reserved.composer !== composer) {
        release();
        reserved = { node: conversation, composer, previousOwner: conversation.getAttribute("data-notice-reservation"),
          margin: conversation.style.getPropertyValue("--notice-composer-base-margin"),
          marginPriority: conversation.style.getPropertyPriority("--notice-composer-base-margin"),
          space: conversation.style.getPropertyValue("--conversation-notice-space"),
          spacePriority: conversation.style.getPropertyPriority("--conversation-notice-space") };
        // The reservation is outside compact composer overflow. Padding inside
        // that scrollport can be scrolled away, putting Add files back under
        // the fixed notice; an external margin preserves the gap at any scroll.
        // Existing padding/safe-area and selected-layout margin remain intact.
        conversation.style.setProperty("--notice-composer-base-margin", getComputedStyle(composer).marginBottom);
        conversation.dataset.noticeReservation = reservationId;
        observer?.observe(conversation); observer?.observe(composer);
      }
      const value = `${space}px`;
      if (conversation.style.getPropertyValue("--conversation-notice-space") !== value) conversation.style.setProperty("--conversation-notice-space", value);
    };
    function schedule(): void {
      if (!disposed && frame === undefined) frame = window.requestAnimationFrame(() => { frame = undefined; update(); });
    }
    observer?.observe(element);
    // Navigation toggles the existing ChatWorkspace's hidden attribute; page
    // replacement is a direct child change. Do not scan on every token/text
    // mutation or attach an unbounded observer to the whole chat subtree.
    const hidden = new MutationObserver(schedule), pages = new MutationObserver(schedule);
    hidden.observe(app, { attributes: true, subtree: true, attributeFilter: ["hidden"] });
    pages.observe(app, { childList: true });
    window.addEventListener("resize", schedule);
    element.addEventListener("animationend", schedule);
    window.visualViewport?.addEventListener("resize", schedule);
    window.visualViewport?.addEventListener("scroll", schedule);
    update();
    return () => {
      disposed = true;
      if (frame !== undefined) window.cancelAnimationFrame(frame);
      hidden.disconnect(); pages.disconnect(); observer?.disconnect();
      window.removeEventListener("resize", schedule);
      element.removeEventListener("animationend", schedule);
      window.visualViewport?.removeEventListener("resize", schedule);
      window.visualViewport?.removeEventListener("scroll", schedule);
      release();
    };
  }, [reservationId]);
  useLayoutEffect(() => {
    const owner = createToastExpiry(toastReadingTime(text), () => dismiss.current(), () => performance.now(), (work, delay) => window.setTimeout(work, delay), handle => window.clearTimeout(handle));
    expiry.current = owner; if (!hovered.current && !focused.current) owner.resume();
    return () => { owner.dispose(); if (expiry.current === owner) expiry.current = undefined; };
  }, [text]);
  const resume = () => { if (!hovered.current && !focused.current) expiry.current?.resume(); };
  return <div ref={notice} className="toast" role="status" onPointerEnter={() => { hovered.current = true; expiry.current?.pause(); }} onPointerLeave={() => { hovered.current = false; resume(); }}
    onFocusCapture={() => { focused.current = true; expiry.current?.pause(); }} onBlurCapture={event => { if (!event.currentTarget.contains(event.relatedTarget)) { focused.current = false; resume(); } }}>
    <span className="toast__icon"><Icon name="info" size={16} /></span>
    <div className="toast__message" tabIndex={0} aria-label="Notification text">{text}</div>
    <button type="button" onClick={onDismiss} aria-label="Dismiss notification"><Icon name="close" size={14} /></button>
  </div>;
}
