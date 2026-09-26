import { describe, expect, it } from "vitest";
import { pendingChatOutputPresentation } from "../src/renderer/src/chatLoadingPresentation";

describe("pending chat output presentation", () => {
  it("names the actual activity ahead of a queued chat without a fake load timer", () => {
    expect(pendingChatOutputPresentation({
      outputTokens: 0,
      startedAtMs: 1_000,
      nowMs: 361_000,
      queue: {
        position: 1,
        queuedBehind: {
          requestId: "evolution-run-1",
          owner: "evolution",
          label: "Evolution proposal",
          method: "evolution.propose",
          brainId: "brain-1"
        }
      }
    })).toEqual({
      phase: "queued",
      label: "Queued #1 behind Evolution proposal",
      ariaLabel:
        "Queued #1 behind Evolution proposal. Stopping this message does not cancel the activity ahead of it.",
      outputTokens: 0
    });
  });

  it("shows honest loading copy before the first output token", () => {
    const presented = pendingChatOutputPresentation({
      outputTokens: 0,
      startedAtMs: 10_000,
      nowMs: 22_345
    });

    expect(presented).toEqual({
      phase: "loading",
      label: "Loading brain · no output tokens yet · 12s elapsed",
      ariaLabel: "Loading brain · no output tokens yet · 12s elapsed",
      outputTokens: 0,
      elapsedMs: 12_345
    });
    expect(presented).not.toHaveProperty("percent");
    expect(JSON.stringify(presented)).not.toMatch(/shard|progress/i);
  });

  it("does not invent elapsed time without a valid measured interval", () => {
    expect(pendingChatOutputPresentation({
      outputTokens: 0,
      startedAtMs: undefined,
      nowMs: 22_345
    })).toEqual({
      phase: "loading",
      label: "Loading brain · no output tokens yet",
      ariaLabel: "Loading brain · no output tokens yet",
      outputTokens: 0
    });
    expect(pendingChatOutputPresentation({
      outputTokens: 0,
      startedAtMs: 30_000,
      nowMs: 22_345
    })).not.toHaveProperty("elapsedMs");
  });

  it("switches to live output usage on the first streamed token", () => {
    expect(pendingChatOutputPresentation({
      outputTokens: 1,
      startedAtMs: 10_000,
      nowMs: 80_000
    })).toEqual({
      phase: "streaming",
      label: "1 output token used",
      ariaLabel: "Live response output: 1 output token used",
      outputTokens: 1
    });
    expect(pendingChatOutputPresentation({ outputTokens: 12_345 })).toMatchObject({
      phase: "streaming",
      label: "12,345 output tokens used",
      outputTokens: 12_345
    });
  });

  it("formats a long measured wait without implying completion", () => {
    expect(pendingChatOutputPresentation({
      outputTokens: 0,
      startedAtMs: 1_000,
      nowMs: 126_678
    }).label).toBe("Loading brain · no output tokens yet · 2m 05s elapsed");
  });
});
