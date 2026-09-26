import { describe, expect, it } from "vitest";
import {
  chatCancellationStatus,
  chatQueueStatus
} from "../src/renderer/src/chatRuntimePresentation";

const queue = {
  position: 1,
  queuedBehind: {
    requestId: "evolution-one",
    owner: "evolution" as const,
    label: "Evolution proposal",
    method: "evolution.propose",
    brainId: "brain-one"
  }
};

describe("typed chat runtime activity presentation", () => {
  it("names the activity that actually owns the worker", () => {
    expect(chatQueueStatus(queue)).toBe(
      "Message queued #1 behind Evolution proposal. Stop cancels only this message."
    );
  });

  it("distinguishes queued cancellation from acknowledged running termination", () => {
    expect(chatCancellationStatus({
      phase: "queued",
      unrelatedActivityContinues: true,
      workerTerminationAcknowledged: false
    }, queue)).toBe(
      "Queued message cancelled. Evolution proposal continues independently."
    );
    expect(chatCancellationStatus({
      phase: "running",
      unrelatedActivityContinues: false,
      workerTerminationAcknowledged: true
    }, null)).toBe(
      "Response stopped after the neural worker acknowledged termination."
    );
  });
});
