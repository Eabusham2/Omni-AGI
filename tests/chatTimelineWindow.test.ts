import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { describe, expect, it } from "vitest";
import {
  anchoredTimelineScrollTop,
  latestChatTimelineWindow,
  newerChatTimelineWindow,
  olderChatTimelineWindow,
  reconcileChatTimelineWindow
} from "../src/renderer/src/chatTimelineWindow";

describe("bounded dynamic chat timeline", () => {
  it("keeps a hundred-thousand-entry history to at most 120 DOM rows", () => {
    const started = performance.now();
    const latest = latestChatTimelineWindow(100_000);
    expect(latest).toEqual({ start: 99_880, end: 100_000, total: 100_000 });
    expect(latest.end - latest.start).toBe(120);
    let older = olderChatTimelineWindow(latest);
    expect(older).toEqual({ start: 99_840, end: 99_960, total: 100_000 });
    older = olderChatTimelineWindow(older);
    expect(older.end - older.start).toBe(120);
    expect(newerChatTimelineWindow(older)).toEqual({
      start: 99_840,
      end: 99_960,
      total: 100_000
    });
    expect(performance.now() - started).toBeLessThan(100);
  });

  it("does not drag an older window to appended streaming activity", () => {
    const older = { start: 2_000, end: 2_120, total: 10_000 };
    expect(reconcileChatTimelineWindow(older, 10_500, false)).toEqual({
      start: 2_000,
      end: 2_120,
      total: 10_500
    });
    expect(reconcileChatTimelineWindow(older, 10_500, true)).toEqual({
      start: 10_380,
      end: 10_500,
      total: 10_500
    });
  });

  it("preserves a surviving variable-height row offset after prepend", () => {
    expect(anchoredTimelineScrollTop(440, 24, 318)).toBe(734);
    expect(anchoredTimelineScrollTop(20, 200, 10)).toBe(0);
  });

  it("loads persisted pages instead of mapping the monolithic BrainDocument", () => {
    const app = readFileSync(
      resolve(import.meta.dirname, "../src/renderer/src/App.tsx"),
      "utf8"
    );
    expect(app).toContain("window.omni.chat.listPage");
    expect(app).toContain("persistedConversation.flatMap");
    expect(app).toContain("conversationHasOlder");
    expect(app).toContain("pinnedIds");
    expect(app).not.toContain(
      "const authoritativeAndPendingMessages = [...brain.messages, ...optimisticHumans]"
    );
  });
});
