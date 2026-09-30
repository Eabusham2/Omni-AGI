export const REPLY_COMPLETE_LEARNING_PHASE = "reply-complete-learning" as const;
export const ACTION_RESULT_LEARNING_PHASE = "action-result-learning" as const;

export interface ReplyCompleteLearningPhase {
  phase:
    | typeof REPLY_COMPLETE_LEARNING_PHASE
    | typeof ACTION_RESULT_LEARNING_PHASE;
  replyComplete: true;
  turnCommitted: boolean;
  learning: true;
  saving: true;
}

export interface ReplyCompleteLearningPresentation {
  phase: ReplyCompleteLearningPhase["phase"];
  label:
    | "Reply complete · learning/saving this experience"
    | "Action complete · integrating result";
  pendingLabel:
    | "Learning and saving this experience…"
    | "Integrating the completed action result…";
  cortexLabel: "Learning & saving reply" | "Integrating action result";
  ariaLabel: string;
  preserveAssistantOutput: true;
  allowNextSend: true;
  /** Accepting a next Send never means two concurrent neural mutations. */
  parallelNeuralMutation: false;
  turnCommitted: boolean;
}

/**
 * Describe the synchronous post-generation phase without claiming that neural
 * learning or the atomic turn save has committed. No progress percentage is
 * exposed because the stream reports state, not remaining work.
 */
export function replyCompleteLearningPresentation(
  phase: ReplyCompleteLearningPhase
): ReplyCompleteLearningPresentation | undefined {
  if (
    phase.replyComplete !== true ||
    phase.learning !== true ||
    phase.saving !== true
  ) {
    return undefined;
  }
  if (
    phase.phase === ACTION_RESULT_LEARNING_PHASE &&
    phase.turnCommitted === true
  ) {
    return {
      phase: ACTION_RESULT_LEARNING_PHASE,
      label: "Action complete · integrating result",
      pendingLabel: "Integrating the completed action result…",
      cortexLabel: "Integrating action result",
      ariaLabel:
        "The reply and visible action are complete. The chat turn is committed; the brain is learning the action result as structured neural experience.",
      preserveAssistantOutput: true,
      allowNextSend: true,
      parallelNeuralMutation: false,
      turnCommitted: true
    };
  }
  if (
    phase.phase !== REPLY_COMPLETE_LEARNING_PHASE ||
    phase.turnCommitted !== false
  ) {
    return undefined;
  }
  return {
    phase: REPLY_COMPLETE_LEARNING_PHASE,
    label: "Reply complete · learning/saving this experience",
    pendingLabel: "Learning and saving this experience…",
    cortexLabel: "Learning & saving reply",
    ariaLabel:
      "Reply generation is complete. The brain is learning and saving this experience; the turn is not committed yet.",
    preserveAssistantOutput: true,
    allowNextSend: true,
    parallelNeuralMutation: false,
    turnCommitted: false
  };
}
