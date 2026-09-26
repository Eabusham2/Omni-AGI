import { describe, expect, it } from "vitest";
import {
  ACTION_RESULT_LEARNING_PHASE,
  REPLY_COMPLETE_LEARNING_PHASE,
  replyCompleteLearningPresentation
} from "../src/renderer/src/chatPhasePresentation";
import { composerSubmitIntent } from "../src/renderer/src/chatTurnPresentation";

describe("post-reply learning presentation", () => {
  const phase = {
    phase: REPLY_COMPLETE_LEARNING_PHASE,
    replyComplete: true,
    turnCommitted: false,
    learning: true,
    saving: true
  } as const;

  it("keeps the reply provisional while learning and saving", () => {
    const presented = replyCompleteLearningPresentation(phase);
    expect(presented).toEqual({
      phase: REPLY_COMPLETE_LEARNING_PHASE,
      label: "Reply complete · learning/saving this experience",
      pendingLabel: "Learning and saving this experience…",
      cortexLabel: "Learning & saving reply",
      ariaLabel:
        "Reply generation is complete. The brain is learning and saving this experience; the turn is not committed yet.",
      preserveAssistantOutput: true,
      allowQueuedInput: true,
      allowParallelSend: false,
      turnCommitted: false
    });
    expect(presented).not.toHaveProperty("percent");
  });

  it("truthfully labels finite structured learning after a visible action", () => {
    expect(replyCompleteLearningPresentation({
      ...phase,
      phase: ACTION_RESULT_LEARNING_PHASE,
      turnCommitted: true
    })).toEqual({
      phase: ACTION_RESULT_LEARNING_PHASE,
      label: "Action complete · integrating result",
      pendingLabel: "Integrating the completed action result…",
      cortexLabel: "Integrating action result",
      ariaLabel:
        "The reply and visible action are complete. The chat turn is committed; the brain is learning the action result as structured neural experience.",
      preserveAssistantOutput: true,
      allowQueuedInput: true,
      allowParallelSend: false,
      turnCommitted: true
    });
  });

  it("presents ordinary Send while the runtime preserves atomic ordering", () => {
    expect(composerSubmitIntent({
      key: "Enter",
      shiftKey: false,
      ctrlKey: false,
      metaKey: false,
      turnActive: true,
      queueAvailable: false,
      steerAvailable: false
    })).toBe("send");
  });

  it("rejects any phase that overclaims commit or omits active saving", () => {
    expect(replyCompleteLearningPresentation({
      ...phase,
      turnCommitted: true
    } as never)).toBeUndefined();
    expect(replyCompleteLearningPresentation({
      ...phase,
      saving: false
    } as never)).toBeUndefined();
    expect(replyCompleteLearningPresentation({
      ...phase,
      phase: ACTION_RESULT_LEARNING_PHASE,
      turnCommitted: false
    })).toBeUndefined();
  });
});
