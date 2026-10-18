import React from "react";
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { normalizeCognitionProgress } from "../src/shared/cognition";
import { normalizeChatEngineEvent } from "../src/main/brainService";
import { advanceChatGenerationPhase } from "../src/renderer/src/chatGenerationLifecycle";
import { CognitionProgressView } from "../src/renderer/src/CognitionProgressView";

describe("live native cognition", () => {
  it("keeps progress and cognition completion separate from reply completion", () => {
    for (const active of [true, false]) {
      const progress = normalizeCognitionProgress({ phase: "ponder", active, passes: 8,
        elapsedMs: 720, delta: .03, selectedAction: "talk", schedulerYield: true });
      expect(progress).toBeDefined();
      expect(normalizeChatEngineEvent({ type: "chat-cognition", brainId: "b", sequence: 3, data: progress }, "b"))
        .toEqual({ type: "chat-cognition", sequence: 3, progress });
      expect(advanceChatGenerationPhase("responding", {
        type: "chat-cognition", brainId: "b", turnId: "t", id: "c", sequence: 3,
        createdAt: "2026-10-07T00:00:00Z", progress: progress!
      })).toBe("responding");
    }
  });
  it("rejects invalid measurements and never forwards made-up reply flags", () => {
    for (const patch of [{ passes: -1 }, { elapsedMs: Infinity }, { delta: NaN },
      { selectedAction: "execute anything" }, { concurrency: "parallel-gpu" },
      { admittedBranches: 2, completedBranches: 3 }, { active: "yes" }]) {
      expect(normalizeCognitionProgress({ phase: "ponder", ...patch })).toBeUndefined();
    }
    expect(normalizeCognitionProgress({ phase: "best-work", branch: 2, admittedBranches: 4,
      concurrency: "cooperative-interleaved", replyComplete: true, toolPermissions: "full" }))
      .toEqual({ phase: "best-work", active: true, branch: 2, admittedBranches: 4, concurrency: "cooperative-interleaved" });
  });
  it("uses an expandable non-narrative activity receipt", () => {
    const html = renderToStaticMarkup(<CognitionProgressView progress={{ phase: "ponder", active: false,
      passes: 0, stopReason: "resource-pressure" }} />);
    expect(html).toContain("<details");
    expect(html).not.toContain("<details open");
    expect(html).toContain("resource pressure");
    expect(html).toContain("Recurrent passes");
  });
});
