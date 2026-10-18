import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ToastNotice } from "../src/renderer/src/ToastNotice";
import { createToastExpiry, toastReadingTime } from "../src/renderer/src/toastExpiry";
function fixture(expire = () => {}) { let now = 0, next = 0; const pending = new Map<number, { work: () => void; delay: number }>();
  const expiry = createToastExpiry(5000, expire, () => now, (work, delay) => { const id = ++next; pending.set(id, { work, delay }); return id; }, handle => { pending.delete(handle); });
  return { expiry, pending, advance: (ms: number) => { now += ms; } };
}
describe("notification owner and reading time", () => {
  it("pauses and resumes the actual monotonic remainder without resetting or multiplying timers", () => {
    const value = fixture(); value.expiry.resume(); value.expiry.resume(); expect(value.pending.size).toBe(1);
    value.advance(1200); value.expiry.pause(); expect(value.pending.size).toBe(0); value.advance(60000); value.expiry.resume();
    expect([...value.pending.values()][0]!.delay).toBe(3800);
  });
  it("retires the old owner even when a cancelled stale callback arrives after an identical new notice", () => {
    let old = 0, fresh = 0; const first = fixture(() => old++), second = fixture(() => fresh++);
    first.expiry.resume(); const late = [...first.pending.values()][0]!.work; first.expiry.dispose(); second.expiry.resume(); late();
    expect(old).toBe(0); expect(fresh).toBe(0); [...second.pending.values()][0]!.work(); expect(fresh).toBe(1);
  });
  it("gives longer notices more reading time without attaching that timer to model work", () => {
    expect(toastReadingTime("Saved")).toBe(4000); expect(toastReadingTime("x".repeat(320))).toBeGreaterThan(10000);
    for (const duration of [-1, NaN, Infinity]) expect(() => createToastExpiry(duration, () => {}, () => 0, () => 1, () => {})).toThrow();
  });
  it("uses neutral information, an accessible reading region and explicit dismiss rather than claiming every result succeeded", () => {
    const html = renderToStaticMarkup(<ToastNotice text="An operation could not finish." onDismiss={() => { throw new Error("Unexpected dismiss during render"); }} />);
    expect(html).toContain("Notification text"); expect(html).toContain("Dismiss notification"); expect(html).toContain("An operation could not finish."); expect(html).toContain('role="status"');
  });
  it("ignores a late cancelled reading timer without erasing a resumed timer on the same notice", () => {
    let expired = 0; const value = fixture(() => expired++); value.expiry.resume();
    const stale = [...value.pending.values()][0]!.work; value.advance(500); value.expiry.pause(); value.expiry.resume(); stale();
    expect(expired).toBe(0); expect(value.pending.size).toBe(1); [...value.pending.values()][0]!.work(); expect(expired).toBe(1);
  });
});
