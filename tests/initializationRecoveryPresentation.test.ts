import { describe, expect, it } from "vitest";
import type {
  BrainReadiness,
  BuildProgressEvent
} from "../src/shared/types";
import {
  activeInitializationTrainingJob,
  initializationIsRunning,
  initializationRecoveryPresentation,
  initializationRunningPresentation
} from "../src/renderer/src/initializationRecoveryPresentation";

function failedReadiness(
  phase: "foundation" | "initial-learning",
  message: string
): BrainReadiness {
  return {
    state: "failed",
    startedAt: "2026-01-01T00:00:00.000Z",
    failure: {
      phase,
      message,
      failedAt: "2026-01-01T00:01:00.000Z",
      retryable: true
    }
  };
}

describe("initialization recovery presentation", () => {
  it("treats only the active brain's correlated retry as running", () => {
    const event: BuildProgressEvent = {
      brainId: "brain-a",
      sequence: 0,
      phase: "allocating",
      progress: 0.01,
      label: "Resuming initial neural learning"
    };
    const failed = failedReadiness(
      "initial-learning",
      "Dataset learning paused before the current source was committed."
    );

    expect(initializationIsRunning("brain-a", failed, true, event)).toBe(true);
    expect(initializationIsRunning("brain-a", failed, false, event)).toBe(false);
    expect(initializationIsRunning("brain-b", failed, true, event)).toBe(false);
    expect(
      initializationIsRunning(
        "brain-a",
        { state: "initializing", startedAt: failed.startedAt },
        false,
        undefined
      )
    ).toBe(true);
  });

  it("shows correlated RuntimeJob record and checkpoint progress during cold recovery", () => {
    const event: BuildProgressEvent = {
      brainId: "brain-a",
      sequence: 4,
      phase: "initial-materials",
      progress: 0.937,
      label: "Learning record 37 of 100",
      job: {
        id: "job-a",
        brainId: "brain-a",
        kind: "ingestion",
        state: "running",
        progress: 0.37,
        label: "Learning record 37 of 100",
        createdAt: "2026-01-01T00:00:00.000Z",
        updatedAt: "2026-01-01T00:00:01.000Z",
        output: {
          progress: {
            currentRecord: 37,
            committedRecords: 32,
            expectedRecords: 100,
            checkpointCommitted: true
          }
        }
      }
    };

    expect(initializationRunningPresentation("brain-a", event)).toEqual({
      title: "Learning record 37 of 100",
      detail: "Record 37 of 100 · 32 committed",
      guidance:
        "Chat and organic activity stay locked until every selected first-learning task completes.",
      progress: 0.37,
      percent: 37
    });
    expect(initializationRunningPresentation("brain-a", event).detail).not.toContain("…");
    const unrelated = initializationRunningPresentation("brain-b", event);
    expect(unrelated.title).toBe("Resuming initial learning");
    expect(unrelated.detail).toContain("committed OmniCortex native core");
    expect(unrelated.detail).not.toContain("ground-up");
    expect(unrelated.progress).toBeUndefined();
    expect(unrelated.percent).toBeUndefined();
  });

  it("exposes controls only for the active mind's live initial-training job", () => {
    const runningJob: NonNullable<BuildProgressEvent["job"]> = {
      id: "job-a",
      brainId: "brain-a",
      kind: "ingestion",
      state: "running",
      progress: 0.42,
      label: "Training",
      createdAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:01.000Z"
    };
    const event: BuildProgressEvent = {
      brainId: "brain-a",
      sequence: 3,
      phase: "initial-materials",
      progress: 0.42,
      label: "Training",
      job: runningJob
    };

    expect(activeInitializationTrainingJob("brain-a", event, undefined)).toBe(runningJob);
    expect(activeInitializationTrainingJob("brain-b", event, undefined)).toBeUndefined();
    expect(
      activeInitializationTrainingJob("brain-a", undefined, {
        ...runningJob,
        id: "crawl-a",
        kind: "crawl"
      })?.id
    ).toBe("crawl-a");
    expect(
      activeInitializationTrainingJob("brain-a", {
        ...event,
        job: { ...runningJob, state: "cancelled" }
      }, undefined)
    ).toBeUndefined();
    expect(
      activeInitializationTrainingJob("brain-a", {
        ...event,
        job: { ...runningJob, kind: "training" }
      }, undefined)
    ).toBeUndefined();
  });

  it.each([
    "Paused before /dataset/part.parquet: worker stopped before commit",
    "Dataset learning paused before the current source was committed.",
    "Dataset learning ended before complete manifest coverage was committed.",
    "Initial learning was cancelled and can be retried."
  ])("presents a resumable learning interruption as paused: %s", (message) => {
    expect(
      initializationRecoveryPresentation(failedReadiness("initial-learning", message))
    ).toMatchObject({
      tone: "paused",
      title: "Learning paused",
      actionLabel: "Resume training",
      busyLabel: "Resuming training…"
    });
  });

  it("keeps a permanent foundation failure visibly distinct", () => {
    expect(
      initializationRecoveryPresentation(
        failedReadiness("foundation", "Packed checkpoint checksum failed.")
      )
    ).toMatchObject({
      tone: "error",
      title: "Setup failed",
      detail: "Packed checkpoint checksum failed.",
      actionLabel: "Retry setup",
      busyLabel: "Retrying…"
    });
  });

  it("uses neutral native-core wording when setup has no safe detail", () => {
    expect(initializationRecoveryPresentation(failedReadiness("foundation", ""))).toMatchObject({
      detail: "The OmniCortex native core could not be completed.",
      guidance: expect.stringContaining("verifies the OmniCortex native core")
    });
  });

  it("does not mislabel an unknown initial-learning failure as a safe pause", () => {
    expect(
      initializationRecoveryPresentation(
        failedReadiness("initial-learning", "Dataset manifest signature is invalid.")
      )
    ).toMatchObject({
      tone: "error",
      title: "Initial learning failed",
      actionLabel: "Retry initial learning"
    });
  });

  it("keeps raw worker tracebacks out of the compact recovery card", () => {
    const presented = initializationRecoveryPresentation(
      failedReadiness(
        "initial-learning",
        "Paused before /dataset/part.parquet: Worker traceback:\nTraceback (most recent call last):\nFile \"worker.py\", line 1\nRuntimeError: allocation interrupted"
      )
    );
    expect(presented.tone).toBe("paused");
    expect(presented.detail).not.toMatch(/traceback|worker\.py/i);
    expect(presented.detail.length).toBeLessThanOrEqual(320);
  });

  it("keeps cancellation visible until acknowledgement and exposes failed acknowledgement", () => {
    const cancelling: BuildProgressEvent = {
      sequence: 5,
      phase: "initial-materials",
      progress: 0.4,
      label: "Initial learning",
      brainId: "brain-one",
      job: {
        id: "job-one",
        brainId: "brain-one",
        kind: "ingestion",
        state: "cancelling",
        progress: 0.4,
        label: "Cancelling ingestion",
        createdAt: "2026-09-12T00:00:00.000Z",
        updatedAt: "2026-09-12T00:00:01.000Z"
      }
    };

    expect(activeInitializationTrainingJob("brain-one", cancelling, null)?.state)
      .toBe("cancelling");
    expect(initializationRunningPresentation("brain-one", cancelling)).toMatchObject({
      title: "Stopping initial learning",
      detail: expect.stringMatching(/acknowledge termination/i)
    });
    expect(initializationRunningPresentation("brain-one", {
      ...cancelling,
      job: {
        ...cancelling.job!,
        error: "Cancellation acknowledgement timed out."
      }
    })).toMatchObject({
      title: "Initial-learning stop needs retry",
      detail: "Cancellation acknowledgement timed out."
    });
  });
});
