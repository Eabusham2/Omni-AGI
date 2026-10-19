import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { conversationNoticeSpace } from "../src/renderer/src/ToastNotice";

// New geometry cases only; the unchanged expiry/reading mechanism stays closed.

describe("measured notice/composer separation", () => {
  it("reserves the actual compact bottom notice height and gap", () => {
    expect(conversationNoticeSpace({ left: 0, right: 390, top: 94, bottom: 791 }, { left: 10, right: 380, top: 740, bottom: 782 })).toBe(59);
  });
  it("does not change a conversation beside a nonoverlapping notice", () => {
    expect(conversationNoticeSpace({ left: 58, right: 800, top: 94, bottom: 900 }, { left: 900, right: 1300, top: 830, bottom: 880 })).toBe(0);
    expect(conversationNoticeSpace({ left: 58, right: 1300, top: 94, bottom: 400 }, { left: 900, right: 1300, top: 830, bottom: 880 })).toBe(0);
  });
  it("recomputes from changed notice height rather than keeping a decorative constant", () => {
    const conversation = { left: 0, right: 390, top: 94, bottom: 791 };
    expect(conversationNoticeSpace(conversation, { left: 10, right: 380, top: 700, bottom: 782 })).toBe(99);
    expect(conversationNoticeSpace(conversation, { left: 10, right: 380, top: 760, bottom: 782 })).toBe(39);
    expect(conversationNoticeSpace(conversation, { left: 10, right: 380, top: NaN, bottom: 782 })).toBe(0);
  });
  it("keeps the measured gap outside scrolling composer content", () => {
    const styles = readFileSync(new URL("../src/renderer/src/styles.css", import.meta.url), "utf8");
    const rule = styles.match(/\.conversation\[data-notice-reservation\] > \.call-layout > \.call-layout__conversation > \.composer-wrap\s*\{([^}]+)\}/)?.[1];
    expect(rule).toMatch(/margin-bottom:\s*calc\(var\(--notice-composer-base-margin/);
    expect(rule).not.toMatch(/padding-bottom:/);
    const source = readFileSync(new URL("../src/renderer/src/ToastNotice.tsx", import.meta.url), "utf8");
    expect(source).toContain('":scope > .call-layout > .call-layout__conversation > .composer-wrap"');
    const conversation = { left: 0, right: 390, top: 94, bottom: 791 };
    const notice = { left: 10, right: 380, top: 740, bottom: 782 };
    const externalMargin = conversationNoticeSpace(conversation, notice);
    // The scrollport's own bottom is lifted by an exterior margin; no amount
    // of child scrolling consumes that gap. Padding would not guarantee this.
    for (const scrollTop of [0, 20, 120]) {
      const scrollportBottom = conversation.bottom - externalMargin;
      const lastControlBottom = Math.min(scrollportBottom, scrollportBottom + 120 - scrollTop);
      expect(lastControlBottom).toBeLessThanOrEqual(notice.top - 8);
    }
  });
});
