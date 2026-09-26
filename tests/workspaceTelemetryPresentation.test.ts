import { describe, expect, it } from "vitest";
import {
  workspaceTelemetryPresentation
} from "../src/renderer/src/workspaceTelemetryPresentation";
import type { WorkspaceSnapshot } from "../src/shared/types";

function snapshot(
  background: NonNullable<WorkspaceSnapshot["learning"]>["backgroundParameters"]
): WorkspaceSnapshot {
  return {
    brainId: "nova-copy",
    queriedAt: "2026-09-20T00:24:25Z",
    contextWindow: {
      capacityTokens: 2048,
      tokenCount: 1170,
      tokenHash: "input-hash",
      recentTokenCount: 1138,
      sensorySlots: 0,
      extended: false,
      updatedAt: "2026-09-20T00:24:25Z"
    },
    latentWorkspace: {
      capacity: 256,
      occupancy: 0,
      items: [],
      evictions: 0,
      rehearsals: 0
    },
    liquidState: { dimensions: 32, mean: 0, norm: 0 },
    hiddenBehavioralPrompt: false,
    rawLongTermTextInjected: false,
    learning: {
      measuredAt: "2026-09-20T00:24:25Z",
      fastNeuralMemory: {
        state: "learned",
        safelyStored: true,
        completedTurns: 8,
        connectionUpdatesTotal: 17_623_348,
        parameterStepsTotal: 2870,
        committedAt: "2026-09-20T00:24:25Z",
        parameterChecksumAfter: "a".repeat(64)
      },
      backgroundParameters: background
    }
  };
}

describe("workspace learning truth", () => {
  it("shows fast learning and 8 pending/0 completed without claiming dense weights changed", () => {
    const presentation = workspaceTelemetryPresentation({
      workspace: snapshot({
        state: "pending",
        pending: 8,
        completed: 0,
        updatedAt: "2026-09-20T00:24:25Z",
        parameterChanged: true
      }),
      currentExperience: "idle"
    });

    expect(presentation.experienceText).toContain("fast synapses and episodes");
    expect(presentation.backgroundText).toContain("8 pending · 0 completed");
    expect(presentation.corticalWeightsText).toContain("not independently verified");
    expect(presentation.backgroundAriaLabel).toContain(
      "composite parameter checksum includes associative memory"
    );
    expect(presentation.backgroundText).not.toContain("weights changed");
  });

  it("requires the module-only result and a completed replay for a weight claim", () => {
    const pending = workspaceTelemetryPresentation({
      workspace: snapshot({
        state: "running",
        pending: 1,
        completed: 0,
        updatedAt: "2026-09-20T00:24:25Z",
        corticalParametersUpdated: true
      }),
      currentExperience: "idle"
    });
    expect(pending.corticalWeightsText).toContain("not independently verified");

    const complete = workspaceTelemetryPresentation({
      workspace: snapshot({
        state: "complete",
        pending: 0,
        completed: 1,
        updatedAt: "2026-09-20T00:25:25Z",
        completedAt: "2026-09-20T00:25:25Z",
        corticalParametersUpdated: true
      }),
      currentExperience: "idle"
    });
    expect(complete.corticalWeightsText).toContain(
      "Trainable module weights changed in the last completed slow replay"
    );
    expect(complete.backgroundText).toContain("module weights changed");

    const unchanged = workspaceTelemetryPresentation({
      workspace: snapshot({
        state: "complete",
        pending: 0,
        completed: 1,
        updatedAt: "2026-09-20T00:25:25Z",
        parameterChanged: true,
        corticalParametersUpdated: false
      }),
      currentExperience: "idle"
    });
    expect(unchanged.corticalWeightsText).toContain(
      "No trainable module weight change"
    );
    expect(unchanged.backgroundText).toContain("module weights unchanged");
  });

  it("keeps the last slow-replay failure visible with the durable job counts", () => {
    const presentation = workspaceTelemetryPresentation({
      workspace: snapshot({
        state: "running",
        pending: 8,
        completed: 0,
        updatedAt: "2026-09-20T00:24:25Z",
        lastError: "  Foundation replay  failed: worker timed out  "
      }),
      currentExperience: "idle"
    });

    expect(presentation.backgroundText).toContain("8 pending · 0 completed");
    expect(presentation.backgroundText).toContain(
      "last error: Foundation replay failed: worker timed out"
    );
    expect(presentation.backgroundErrorText).toBe(
      "Last slow-replay error: Foundation replay failed: worker timed out"
    );
  });

  it("distinguishes a stopping replay from paused jobs without implying a weight update", () => {
    const stopping = workspaceTelemetryPresentation({
      workspace: snapshot({
        state: "pausing",
        pending: 8,
        completed: 0,
        updatedAt: "2026-09-20T00:24:25Z"
      }),
      currentExperience: "idle"
    });
    expect(stopping.backgroundText).toContain("stopping current update");

    const paused = workspaceTelemetryPresentation({
      workspace: snapshot({
        state: "paused",
        pending: 8,
        completed: 0,
        updatedAt: "2026-09-20T00:24:25Z"
      }),
      currentExperience: "idle"
    });
    expect(paused.backgroundText).toContain("paused; pending updates retained");
    expect(paused.corticalWeightsText).toContain("not independently verified");

    const qaOverride = workspaceTelemetryPresentation({
      workspace: snapshot({
        state: "paused",
        pauseReason: "launch-override",
        pending: 8,
        completed: 0,
        updatedAt: "2026-09-20T00:24:25Z"
      }),
      currentExperience: "idle"
    });
    expect(qaOverride.backgroundText).toContain("suspended for this app launch");
  });
});
