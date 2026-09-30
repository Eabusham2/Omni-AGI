import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { chatInputCapacity, cleanChatInput, utf8InputTokenCount } from "../src/shared/chatInput";

describe("user chat input admission", () => {
  it("preserves full Unicode text above the removed 100,000-character ceiling", () => {
    const text = "ordinary experience مرحبا ".repeat(6_000);
    expect(text.length).toBeGreaterThan(100_000);
    expect(cleanChatInput(text)).toBe(text.trim());
  });

  it("keeps the existing empty-input and NUL normalization boundary", () => {
    expect(cleanChatInput("  hello\0 world  ")).toBe("hello world");
    expect(() => cleanChatInput("\0 \n ")).toThrow("cannot be empty");
  });

  it("does not reject an otherwise valid committed user receipt at the old ceiling", () => {
    const source = readFileSync(new URL("../src/main/brainService.ts", import.meta.url), "utf8");
    expect(source).not.toContain("human.content.length > 100_000");
    expect(source).not.toContain("A chat message cannot exceed 100,000 characters");
    expect(source).toContain("return cleanChatInput(value)");
  });

  it("uses actual UTF-8 boundary cost, not old context occupancy or character counts", () => {
    for (const text of ["hello", "مرحبا", "中文🙂", "\ud800", "\u0000\t"]) {
      expect(utf8InputTokenCount(text)).toBe(new TextEncoder().encode(text).length);
    }
    const payload = "中文🙂";
    const required = utf8InputTokenCount(payload) + 3;
    expect(chatInputCapacity(payload, required).fits).toBe(true);
    expect(chatInputCapacity(payload, required - 1).fits).toBe(false);
    expect(chatInputCapacity(payload, NaN).fits).toBe(false);
    expect(chatInputCapacity("a".repeat(150_000), 200_000).fits).toBe(true);
  });

  it("blocks Send, Queue and Steer before receipts while retaining the large draft", () => {
    const app = readFileSync(new URL("../src/renderer/src/App.tsx", import.meta.url), "utf8");
    expect(app).toContain("disabled={!input.trim() || !window.omni || !draftCapacity.fits}");
    expect(app.match(/disabled=\{!draftCapacity\.fits\}/g)?.length).toBe(2);
    expect(app).toContain('id="chat-input-capacity-error"');
    const steer = app.slice(app.indexOf("const steerCurrentTurn ="), app.indexOf("const queueCurrentTurn ="));
    expect(steer.indexOf("chatInputCapacity(")).toBeLessThan(steer.indexOf("persistDeliveryReceipt("));
    const host = readFileSync(new URL("../src/main/brainService.ts", import.meta.url), "utf8");
    const chat = host.slice(host.indexOf("  async chat("), host.indexOf("  async chatUnlocked("));
    expect(chat.indexOf("chatInputCapacity(")).toBeLessThan(chat.indexOf("claimForeground"));
  });
});
