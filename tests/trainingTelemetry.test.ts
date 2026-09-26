import { describe, expect, it } from "vitest";
import {
  foundationTelemetrySample,
  normalizePersistedTrainingTelemetry,
  presentTrainingTelemetry,
  resumeTrainingTelemetry,
  updateTrainingTelemetry,
  trainingResourceReadings,
  type TrainingTelemetry
} from "../src/shared/trainingTelemetry";

function sample(
  atMs: number,
  recordsCompleted: number,
  bytesCompleted: number,
  overrides: Partial<Parameters<typeof updateTrainingTelemetry>[1]> = {}
) {
  return {
    atMs,
    recordsCompleted,
    bytesCompleted,
    recordsTotal: 100,
    bytesTotal: 9_000,
    currentRecord: recordsCompleted,
    committedRecords: 0,
    expectedRecords: 100,
    ...overrides
  };
}

describe("measured training telemetry", () => {
  it("sanitizes worker progress without forwarding raw foundation fields", () => {
    expect(
      foundationTelemetrySample({
        batchIndex: 2,
        batchCount: 4,
        targetWindowIndex: 3,
        targetWindowCount: 7,
        completedTargetTokens: 128,
        targetTokens: 512,
        lossChunksCompleted: 1,
        lossChunksTotal: 4,
        rawText: "must not cross the app boundary"
      })
    ).toEqual({
      batchIndex: 2,
      batchCount: 4,
      targetWindowIndex: 3,
      targetWindowCount: 7,
      completedTokens: 128,
      targetTokens: 512,
      completedChunks: 1,
      totalChunks: 4
    });
    expect(
      trainingResourceReadings({
        processMemoryBytes: 2_000,
        processPeakMemoryBytes: 2_500,
        totalMemoryBytes: 16_000,
        unrelated: 99,
        availableMemoryBytes: Number.NaN
      })
    ).toEqual({
      processMemoryBytes: 2_000,
      processPeakMemoryBytes: 2_500,
      totalMemoryBytes: 16_000
    });
    expect(
      foundationTelemetrySample({
        batchIndex: 1,
        batchCount: 1,
        targetWindowIndex: 1,
        completedTargetTokens: 1,
        targetTokens: 1,
        lossChunksCompleted: 1,
        lossChunksTotal: 1
      })
    ).toBeUndefined();
  });

  it("waits for a real positive counter delta before reporting speed or ETA", () => {
    const first = updateTrainingTelemetry(undefined, sample(1_000, 10, 1_000));
    expect(first.recordsPerSecond).toBeUndefined();
    expect(first.bytesPerSecond).toBeUndefined();
    expect(first.totalEtaMs).toBeUndefined();
    expect(first.nextCheckpoint.etaMs).toBeUndefined();
    expect(presentTrainingTelemetry(first)).toMatchObject({
      stageLabel: "Learning",
      throughput: [],
      memory: "Worker memory unavailable"
    });

    const noDelta = updateTrainingTelemetry(first, sample(2_000, 10, 1_000));
    expect(noDelta.recordsPerSecond).toBeUndefined();
    expect(noDelta.bytesPerSecond).toBeUndefined();
    expect(noDelta.totalEtaMs).toBeUndefined();
    expect(() =>
      updateTrainingTelemetry(noDelta, sample(1_999, 11, 1_100))
    ).toThrow(/timestamps must be monotonic/i);
  });

  it("accumulates sub-sample deltas instead of starving high-frequency rates", () => {
    let telemetry = updateTrainingTelemetry(undefined, sample(1_000, 0, 0));
    telemetry = updateTrainingTelemetry(telemetry, sample(1_100, 1, 100));
    expect(telemetry.recordsPerSecond).toBeUndefined();
    telemetry = updateTrainingTelemetry(telemetry, sample(1_200, 2, 200));
    expect(telemetry.recordsPerSecond).toBeUndefined();
    telemetry = updateTrainingTelemetry(telemetry, sample(1_300, 3, 300));
    expect(telemetry.recordsPerSecond).toBe(10);
    expect(telemetry.bytesPerSecond).toBe(1_000);

    let foundation = updateTrainingTelemetry(undefined, sample(2_000, 3, 300, {
      foundation: {
        batchIndex: 1,
        batchCount: 1,
        completedTokens: 0,
        targetTokens: 256,
        completedChunks: 0,
        totalChunks: 8
      }
    }));
    for (const [offset, tokens] of [[100, 32], [200, 64], [300, 96]] as const) {
      foundation = updateTrainingTelemetry(
        foundation,
        sample(2_000 + offset, 3, 300, {
          foundation: {
            batchIndex: 1,
            batchCount: 1,
            completedTokens: tokens,
            targetTokens: 256,
            completedChunks: tokens / 32,
            totalChunks: 8
          }
        })
      );
    }
    expect(foundation.foundation?.tokensPerSecond).toBe(320);
  });

  it("smooths measured rates while keeping counters monotonic", () => {
    const first = updateTrainingTelemetry(undefined, sample(1_000, 10, 1_000));
    const second = updateTrainingTelemetry(first, sample(2_000, 30, 5_000));
    expect(second.recordsPerSecond).toBe(20);
    expect(second.bytesPerSecond).toBe(4_000);
    expect(second.totalEtaMs).toBe(1_000);
    expect(second.nextCheckpoint).toMatchObject({
      currentRecord: 30,
      targetRecord: 100,
      etaMs: 3_500
    });
    const expectedFinish = new Intl.DateTimeFormat(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit"
    }).format(new Date(Date.parse(second.updatedAt) + second.totalEtaMs!));
    expect(presentTrainingTelemetry(second).timing).toEqual([
      "1s elapsed",
      "1s remaining",
      "2s projected total",
      `Finish around ${expectedFinish}`
    ]);

    const third = updateTrainingTelemetry(second, sample(3_000, 40, 7_000));
    expect(third.recordsPerSecond).toBeCloseTo(17);
    expect(third.bytesPerSecond).toBeCloseTo(3_400);
    const regressed = updateTrainingTelemetry(third, sample(4_000, 39, 6_000));
    expect(regressed.recordsCompleted).toBe(40);
    expect(regressed.bytesCompleted).toBe(7_000);
    expect(regressed.recordsPerSecond).toBeCloseTo(17);
    expect(regressed.bytesPerSecond).toBeCloseTo(3_400);
    expect(regressed.recordsTotal).toBe(100);
    expect(regressed.bytesTotal).toBe(9_000);
    expect(regressed.totalEtaMs).toBe(third.totalEtaMs);
    expect(regressed.nextCheckpoint.currentRecord).toBe(40);
  });

  it("finishes a one-record manifest without extrapolating to record 512", () => {
    const started = updateTrainingTelemetry(undefined, sample(1_000, 0, 0, {
      recordsTotal: 1,
      bytesTotal: 71,
      currentRecord: 1,
      committedRecords: 0,
      expectedRecords: undefined
    }));
    expect(started.nextCheckpoint).toMatchObject({
      currentRecord: 1,
      targetRecord: 1
    });
    expect(presentTrainingTelemetry(started).timing).toEqual([
      "0s elapsed",
      "Finishing current record…"
    ]);
    const finishing = updateTrainingTelemetry(
      started,
      sample(24_000, 1, 71, {
        recordsTotal: 1,
        bytesTotal: 71,
        currentRecord: 1,
        committedRecords: 0,
        expectedRecords: undefined
      })
    );

    expect(finishing.recordsPerSecond).toBeCloseTo(1 / 23);
    expect(finishing.nextCheckpoint).toMatchObject({
      currentRecord: 1,
      targetRecord: 1
    });
    expect(finishing.nextCheckpoint.etaMs).toBeUndefined();
    expect(finishing.totalEtaMs).toBeUndefined();
    expect(presentTrainingTelemetry(finishing)).toMatchObject({
      throughput: ["Final record neural work in progress"],
      timing: ["23s elapsed", "Finishing current record…"]
    });
    expect(presentTrainingTelemetry(finishing).timing.join(" ")).not.toMatch(
      /512|checkpoint/i
    );
    expect(normalizePersistedTrainingTelemetry(finishing)).toEqual(finishing);
  });

  it("retains checkpoint ETA only for a genuinely unknown streaming source", () => {
    const started = updateTrainingTelemetry(undefined, sample(1_000, 0, 0, {
      recordsTotal: undefined,
      bytesTotal: undefined,
      currentRecord: 0,
      committedRecords: 0,
      expectedRecords: undefined
    }));
    const streaming = updateTrainingTelemetry(
      started,
      sample(2_000, 1, 71, {
        recordsTotal: undefined,
        bytesTotal: undefined,
        currentRecord: 1,
        committedRecords: 0,
        expectedRecords: undefined
      })
    );

    expect(streaming).not.toHaveProperty("recordsTotal");
    expect(streaming).not.toHaveProperty("bytesTotal");
    expect(streaming.nextCheckpoint).toMatchObject({
      currentRecord: 1,
      targetRecord: 512,
      etaMs: 511_000
    });
    expect(presentTrainingTelemetry(streaming).timing).toContain(
      "8m 31s to record 512 checkpoint"
    );
  });

  it("uses the finite multi-record manifest remainder as the completion ETA", () => {
    const started = updateTrainingTelemetry(undefined, sample(1_000, 0, 0, {
      recordsTotal: 100,
      bytesTotal: 10_000,
      currentRecord: 0,
      committedRecords: 0,
      expectedRecords: undefined
    }));
    const progressing = updateTrainingTelemetry(
      started,
      sample(2_000, 20, 2_000, {
        recordsTotal: 100,
        bytesTotal: 10_000,
        currentRecord: 20,
        committedRecords: 0,
        expectedRecords: undefined
      })
    );

    expect(progressing.recordsTotal).toBe(100);
    expect(progressing.bytesTotal).toBe(10_000);
    expect(progressing.totalEtaMs).toBe(4_000);
    expect(progressing.nextCheckpoint).toMatchObject({
      currentRecord: 20,
      targetRecord: 100,
      etaMs: 4_000
    });
    const timing = presentTrainingTelemetry(progressing).timing;
    expect(timing).toContain("4s remaining");
    expect(timing.join(" ")).not.toMatch(/record 512|checkpoint/i);
  });

  it("keeps an uncommitted crossed checkpoint boundary pending", () => {
    const first = updateTrainingTelemetry(undefined, sample(1_000, 500, 4_000, {
      recordsTotal: 2_000,
      bytesTotal: 16_000,
      expectedRecords: 2_000,
      currentRecord: 500,
      committedRecords: 0
    }));
    const crossed = updateTrainingTelemetry(first, sample(2_000, 520, 4_160, {
      recordsTotal: 2_000,
      bytesTotal: 16_000,
      expectedRecords: 2_000,
      currentRecord: 520,
      committedRecords: 0
    }));
    expect(crossed.nextCheckpoint).toMatchObject({
      currentRecord: 520,
      committedRecords: 0,
      targetRecord: 512
    });
    expect(crossed.nextCheckpoint.etaMs).toBeUndefined();

    const committed = updateTrainingTelemetry(crossed, sample(3_000, 520, 4_160, {
      recordsTotal: 2_000,
      bytesTotal: 16_000,
      expectedRecords: 2_000,
      currentRecord: 520,
      committedRecords: 512,
      checkpointCommitted: true
    }));
    expect(committed.nextCheckpoint.targetRecord).toBe(1_024);
  });

  it("does not manufacture a final record total from a durable resume cursor", () => {
    const resumed = updateTrainingTelemetry(undefined, sample(1_000, 5_632, 0, {
      recordsTotal: undefined,
      bytesTotal: undefined,
      currentRecord: 5_632,
      committedRecords: 5_632,
      expectedRecords: undefined,
      checkpointCommitted: true
    }));
    expect(resumed.nextCheckpoint).toMatchObject({
      currentRecord: 5_632,
      committedRecords: 5_632,
      targetRecord: 6_144
    });
    expect(resumed.nextCheckpoint.expectedRecords).toBeUndefined();
  });

  it("resets only entry-local checkpoint cursors when a second file starts", () => {
    const firstScope = "a".repeat(64);
    const secondScope = "b".repeat(64);
    const first = updateTrainingTelemetry(undefined, sample(1_000, 590, 5_900, {
      scopeId: firstScope,
      recordsTotal: undefined,
      bytesTotal: 20_000,
      currentRecord: 590,
      committedRecords: 512,
      expectedRecords: 600
    }));
    const completedFirst = updateTrainingTelemetry(first, sample(2_000, 600, 6_000, {
      scopeId: firstScope,
      recordsTotal: undefined,
      bytesTotal: 20_000,
      currentRecord: 600,
      committedRecords: 512,
      expectedRecords: 600
    }));
    expect(completedFirst.nextCheckpoint.targetRecord).toBe(600);

    const secondFile = updateTrainingTelemetry(
      completedFirst,
      sample(3_000, 610, 6_100, {
        scopeId: secondScope,
        recordsTotal: undefined,
        bytesTotal: 20_000,
        currentRecord: 10,
        committedRecords: 0,
        expectedRecords: 1_000
      })
    );
    expect(secondFile.recordsCompleted).toBe(610);
    expect(secondFile.bytesCompleted).toBe(6_100);
    expect(secondFile.recordsPerSecond).toBe(10);
    expect(secondFile.nextCheckpoint).toMatchObject({
      scopeId: secondScope,
      currentRecord: 10,
      committedRecords: 0,
      expectedRecords: 1_000,
      targetRecord: 512,
      etaMs: 50_200
    });
  });

  it("reports foundation token speed only from correlated samples in one batch", () => {
    const first = updateTrainingTelemetry(undefined, sample(1_000, 12, 2_000, {
      foundation: {
        batchIndex: 2,
        batchCount: 4,
        targetWindowIndex: 1,
        targetWindowCount: 2,
        completedTokens: 128,
        targetTokens: 512,
        completedChunks: 1,
        totalChunks: 4
      }
    }));
    expect(first.stage).toBe("foundation");
    expect(first.foundation?.tokensPerSecond).toBeUndefined();

    const second = updateTrainingTelemetry(first, sample(2_000, 12, 2_000, {
      foundation: {
        batchIndex: 2,
        batchCount: 4,
        targetWindowIndex: 1,
        targetWindowCount: 2,
        completedTokens: 384,
        targetTokens: 512,
        completedChunks: 3,
        totalChunks: 4
      }
    }));
    expect(second.foundation?.tokensPerSecond).toBe(256);
    expect(presentTrainingTelemetry(second)).toMatchObject({
      stageLabel: "Answering cortex",
      throughput: [
        "Smoothed avg 256 foundation tokens/s",
        "window 1/2",
        "3 / 4 chunks"
      ]
    });

    const nextWindow = updateTrainingTelemetry(second, sample(3_000, 12, 2_000, {
      foundation: {
        batchIndex: 2,
        batchCount: 4,
        targetWindowIndex: 2,
        targetWindowCount: 2,
        completedTokens: 64,
        targetTokens: 512,
        completedChunks: 1,
        totalChunks: 8
      }
    }));
    expect(nextWindow.foundation?.tokensPerSecond).toBeUndefined();

    const nextBatch = updateTrainingTelemetry(nextWindow, sample(4_000, 12, 2_000, {
      foundation: {
        batchIndex: 3,
        batchCount: 4,
        completedTokens: 64,
        targetTokens: 512,
        completedChunks: 1,
        totalChunks: 8
      }
    }));
    expect(nextBatch.foundation?.tokensPerSecond).toBeUndefined();
  });

  it("does not correlate a reused foundation batch index across learning work", () => {
    const first = updateTrainingTelemetry(undefined, sample(1_000, 12, 2_000, {
      foundation: {
        batchIndex: 1,
        batchCount: 1,
        completedTokens: 128,
        targetTokens: 512,
        completedChunks: 1,
        totalChunks: 4
      }
    }));
    const measured = updateTrainingTelemetry(first, sample(2_000, 12, 2_000, {
      foundation: {
        batchIndex: 1,
        batchCount: 1,
        completedTokens: 384,
        targetTokens: 512,
        completedChunks: 3,
        totalChunks: 4
      }
    }));
    expect(measured.foundation?.tokensPerSecond).toBe(256);

    const learning = updateTrainingTelemetry(measured, sample(3_000, 13, 2_100));
    const restarted = updateTrainingTelemetry(learning, sample(4_000, 13, 2_100, {
      foundation: {
        batchIndex: 1,
        batchCount: 1,
        completedTokens: 64,
        targetTokens: 512,
        completedChunks: 1,
        totalChunks: 8
      }
    }));
    expect(restarted.foundation?.tokensPerSecond).toBeUndefined();
    expect(restarted.sampling.foundationSampleTokens).toBe(64);
    const advanced = updateTrainingTelemetry(restarted, sample(5_000, 13, 2_100, {
      foundation: {
        batchIndex: 1,
        batchCount: 1,
        completedTokens: 128,
        targetTokens: 512,
        completedChunks: 2,
        totalChunks: 8
      }
    }));
    expect(advanced.foundation?.tokensPerSecond).toBe(64);
  });

  it("uses only worker-reported physical memory and preserves the measured peak", () => {
    const first = updateTrainingTelemetry(undefined, sample(1_000, 1, 100, {
      resourceReadings: {
        processMemoryBytes: 2_000,
        processPeakMemoryBytes: 2_500,
        availableMemoryBytes: 8_000,
        totalMemoryBytes: 16_000
      }
    }));
    expect(first.memory).toEqual({
      source: "worker-process",
      currentPhysicalBytes: 2_000,
      peakPhysicalBytes: 2_500,
      availablePhysicalBytes: 8_000,
      totalPhysicalBytes: 16_000
    });
    const second = updateTrainingTelemetry(first, sample(2_000, 2, 200, {
      resourceReadings: {
        processMemoryBytes: 1_800,
        processPeakMemoryBytes: 2_200
      }
    }));
    expect(second.memory?.currentPhysicalBytes).toBe(1_800);
    expect(second.memory?.peakPhysicalBytes).toBe(2_500);
  });

  it("never labels a peak-only reading as current worker memory", () => {
    const peakOnly = updateTrainingTelemetry(undefined, sample(1_000, 1, 100, {
      resourceReadings: { processPeakMemoryBytes: 2_500 }
    }));
    expect(peakOnly.memory).toBeUndefined();
    expect(presentTrainingTelemetry(peakOnly).memory).toBe(
      "Worker memory unavailable"
    );

    const measured = updateTrainingTelemetry(peakOnly, sample(2_000, 2, 200, {
      resourceReadings: {
        processMemoryBytes: 2_000,
        processPeakMemoryBytes: 2_100
      }
    }));
    const nextPeak = updateTrainingTelemetry(measured, sample(3_000, 3, 300, {
      resourceReadings: { processPeakMemoryBytes: 3_000 }
    }));
    expect(nextPeak.memory?.currentPhysicalBytes).toBe(2_000);
    expect(nextPeak.memory?.peakPhysicalBytes).toBe(3_000);
  });

  it("labels checkpoint events honestly and resets stale rates on recovery", () => {
    const first = updateTrainingTelemetry(undefined, sample(1_000, 500, 4_000, {
      recordsTotal: 2_000,
      bytesTotal: 16_000,
      expectedRecords: 2_000,
      currentRecord: 500,
      committedRecords: 0
    }));
    const measured = updateTrainingTelemetry(first, sample(2_000, 512, 4_096, {
      recordsTotal: 2_000,
      bytesTotal: 16_000,
      expectedRecords: 2_000,
      currentRecord: 512,
      committedRecords: 512,
      checkpointCommitted: true
    }));
    expect(measured.stage).toBe("checkpoint");
    expect(presentTrainingTelemetry(measured).stageLabel).toBe("Checkpoint commit");

    const recovered = resumeTrainingTelemetry(measured, 10_000);
    expect(recovered.elapsedMs).toBe(measured.elapsedMs);
    expect(recovered.recordsPerSecond).toBeUndefined();
    expect(recovered.bytesPerSecond).toBeUndefined();
    expect(recovered.totalEtaMs).toBeUndefined();
    expect(recovered.nextCheckpoint.etaMs).toBeUndefined();
    expect(recovered.sampling.recordSampleCount).toBe(512);
    expect(recovered.sampling).not.toHaveProperty("foundationBatchIndex");
    expect(recovered.sampling).not.toHaveProperty("foundationSampleAtMs");
    expect(recovered.sampling).not.toHaveProperty("foundationSampleTokens");
    const persistedRecovery = JSON.parse(JSON.stringify(recovered)) as unknown;
    expect(normalizePersistedTrainingTelemetry(persistedRecovery)).toEqual(
      persistedRecovery
    );
  });

  it("includes foundation pauses in the next positive record and byte sample", () => {
    const first = updateTrainingTelemetry(undefined, sample(1_000, 10, 1_000));
    let duringFoundation = first;
    for (let index = 1; index <= 6; index += 1) {
      duringFoundation = updateTrainingTelemetry(
        duringFoundation,
        sample(1_000 + index * 10_000, 10, 1_000, {
          foundation: {
            batchIndex: 1,
            batchCount: 1,
            completedTokens: index * 64,
            targetTokens: 384,
            completedChunks: index,
            totalChunks: 6
          }
        })
      );
    }
    const advanced = updateTrainingTelemetry(
      duringFoundation,
      sample(71_000, 20, 2_000)
    );
    // Ten new records/1,000 bytes over the full 70 seconds since the prior
    // positive sample, not only the ten seconds since the final foundation event.
    expect(advanced.recordsPerSecond).toBeCloseTo(10 / 70);
    expect(advanced.bytesPerSecond).toBeCloseTo(1_000 / 70);
    expect(presentTrainingTelemetry(advanced).throughput[0]).toBe(
      "Smoothed avg 0.14 records/s · 8.6/min"
    );
  });

  it("keeps persisted telemetry structurally cloneable", () => {
    const telemetry: TrainingTelemetry = updateTrainingTelemetry(
      undefined,
      sample(1_000, 1, 100)
    );
    expect(structuredClone(telemetry)).toEqual(telemetry);
    expect(normalizePersistedTrainingTelemetry(telemetry)).toEqual(telemetry);
    const sanitized = normalizePersistedTrainingTelemetry({
      ...telemetry,
      rawText: "must not survive",
      sampling: {
        ...telemetry.sampling,
        rawTokenIds: [1, 2, 3]
      }
    });
    expect(sanitized).toEqual(telemetry);
    expect(sanitized).not.toHaveProperty("rawText");
    expect(sanitized?.sampling).not.toHaveProperty("rawTokenIds");
    expect(
      normalizePersistedTrainingTelemetry({
        ...telemetry,
        recordsPerSecond: Number.NaN
      })
    ).toBeUndefined();
    for (const targetRecord of [99, 999]) {
      expect(
        normalizePersistedTrainingTelemetry({
          ...telemetry,
          nextCheckpoint: {
            ...telemetry.nextCheckpoint,
            targetRecord
          }
        })
      ).toBeUndefined();
    }
    expect(
      normalizePersistedTrainingTelemetry({
        ...telemetry,
        sampling: {
          ...telemetry.sampling,
          recordSampleAtMs: Date.parse(telemetry.startedAt) - 1
        }
      })
    ).toBeUndefined();
    const measured = updateTrainingTelemetry(
      telemetry,
      sample(2_000, 11, 1_100)
    );
    expect(measured.nextCheckpoint.etaMs).toBeDefined();
    expect(measured.totalEtaMs).toBeDefined();
    expect(
      normalizePersistedTrainingTelemetry({
        ...measured,
        nextCheckpoint: {
          ...measured.nextCheckpoint,
          etaMs: measured.nextCheckpoint.etaMs! + 1
        }
      })
    ).toBeUndefined();
    expect(
      normalizePersistedTrainingTelemetry({
        ...measured,
        totalEtaMs: measured.totalEtaMs! + 1
      })
    ).toBeUndefined();
    expect(
      normalizePersistedTrainingTelemetry({
        ...telemetry,
        nextCheckpoint: {
          ...telemetry.nextCheckpoint,
          scopeId: "raw/source/path"
        }
      })
    ).toBeUndefined();
    expect(
      normalizePersistedTrainingTelemetry({
        ...telemetry,
        recordsTotal: 0
      })
    ).toBeUndefined();
    expect(
      normalizePersistedTrainingTelemetry({
        ...telemetry,
        sampling: {
          ...telemetry.sampling,
          recordSampleAtMs: telemetry.sampling.lastEventAtMs + 1
        }
      })
    ).toBeUndefined();
    expect(
      normalizePersistedTrainingTelemetry({
        ...telemetry,
        memory: {
          source: "worker-process",
          currentPhysicalBytes: 2_000,
          peakPhysicalBytes: 1_000
        }
      })
    ).toBeUndefined();
  });

  it("carries live disk space-left readings through progress, persistence, and presentation", () => {
    const gib = 1024 ** 3;
    const first = updateTrainingTelemetry(undefined, sample(1_000, 1, 100, {
      resourceReadings: {
        diskTotalBytes: 100 * gib,
        diskFreeBytes: 40 * gib,
        diskReserveBytes: 20 * gib,
        mandatoryFreeDiskBytes: 20 * gib,
        projectedDiskFreeBytes: 35 * gib,
        estimatedWriteBytes: 5 * gib,
        diskPressure: false
      }
    }));
    expect(first.diskSpace).toMatchObject({
      diskTotalBytes: 100 * gib,
      diskFreeBytes: 40 * gib,
      mandatoryReserveBytes: 20 * gib,
      operationWriteBytes: 5 * gib,
      projectedRemainingBytes: 35 * gib,
      projectedAboveReserveBytes: 15 * gib,
      paused: false
    });
    expect(presentTrainingTelemetry(first).storage).toMatch(
      /40 GB free.*35 GB projected.*20 GB reserve/i
    );
    expect(normalizePersistedTrainingTelemetry(first)?.diskSpace)
      .toEqual(first.diskSpace);

    const atReserve = updateTrainingTelemetry(first, sample(2_000, 2, 200, {
      resourceReadings: {
        diskTotalBytes: 100 * gib,
        diskFreeBytes: 24 * gib,
        diskReserveBytes: 20 * gib,
        projectedDiskFreeBytes: 20 * gib,
        estimatedWriteBytes: 4 * gib,
        diskPressure: true
      }
    }));
    expect(atReserve.diskSpace).toMatchObject({
      diskFreeBytes: 24 * gib,
      projectedRemainingBytes: 20 * gib,
      projectedAboveReserveBytes: 0,
      paused: true
    });
    expect(presentTrainingTelemetry(atReserve).storage).toContain("paused");
  });
});
