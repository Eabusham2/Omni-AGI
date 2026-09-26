import { describe, expect, it } from "vitest";
import {
  chatActionSearchActivity,
  chatActionStateLabel,
  chatActionTitle,
  cleanChatActionStatus
} from "../src/renderer/src/chatActionPresentation";
import type { StructuredAction } from "../src/shared/types";

const ponder: StructuredAction = {
  kind: "ponder",
  source: "organic",
  toolId: "ponder",
  action: "ponder",
  arguments: {}
};

describe("plain chat action presentation", () => {
  it("never exposes legacy ponder or creativity protocol strings", () => {
    expect(chatActionTitle(ponder)).toBe("Pondering");
    expect(cleanChatActionStatus("ponder.ponder: running.", ponder)).toBe(
      "Pondering: running."
    );
    expect(cleanChatActionStatus("studio.ui/open-creativity: complete.", {
      ...ponder,
      kind: "tool",
      toolId: "studio.ui",
      action: "open-creativity"
    })).toBe("Creativity: complete.");
  });

  it("uses human action and state names", () => {
    expect(chatActionTitle({
      ...ponder,
      kind: "imagine",
      toolId: "modality.imagine",
      action: "generate",
      arguments: { modality: "video" }
    })).toBe("Video imagination");
    expect(chatActionStateLabel("approval-required")).toBe("Needs approval");
  });

  it("shows the exact bounded web search in the collapsed activity card", () => {
    const action = {
      kind: "tool" as const,
      toolId: "web.search",
      action: "search",
      arguments: { query: "liquid neural network stability" },
      source: "brain" as const
    };
    expect(chatActionSearchActivity(action, "running")).toBe(
      "Searching “liquid neural network stability”"
    );
    expect(chatActionSearchActivity(action, "complete")).toBe(
      "Searched “liquid neural network stability”"
    );
  });

});
