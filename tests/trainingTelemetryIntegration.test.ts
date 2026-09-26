import { EventEmitter } from "node:events";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService, RuntimeJobManager } from "../src/main/brainService";
import {
  BuildInitializationPlanStore,
  type InitialFoundationPlan
} from "../src/main/buildInitializationPlan";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { initializationRunningPresentation } from "../src/renderer/src/initializationRecoveryPresentation";
import {
  updateTrainingTelemetry,
  type TrainingTelemetry
} from "../src/shared/trainingTelemetry";
import type {
  BuildProgressEvent,
  RuntimeJobEvent,
  TrainingCoverage
} from "../src/shared/types";

const foundation: InitialFoundationPlan = {
  hardwareTier: "micro",
  modalities: [],
  origin: "ground-up",
};

function coverage(
  processedRecords: number,
  rejectedRecords: number,
  processedBytes: number,
  complete = false
): TrainingCoverage {
  return {
    schemaVersion: 1,
    manifestId: "manifest-a",
    discoveredFiles: 1,
    processedFiles: complete ? 1 : 0,
    rejectedFiles: 0,
    discoveredRecords: complete ? processedRecords + rejectedRecords : 0,
    processedRecords,
    rejectedRecords,
    discoveredBytes: 10_000,
    processedBytes,
    shards: 1,
    modalityCounts: { parquet: processedRecords },
    errors: [],
    complete,
    updatedAt: new Date().toISOString()
  };
}

function telemetrySeed(atMs = Date.now() - 2_000): TrainingTelemetry {
  const first = updateTrainingTelemetry(undefined, {
    atMs,
    scopeId: "a".repeat(64),
    recordsCompleted: 10,
    bytesCompleted: 1_000,
    bytesTotal: 10_000,
    currentRecord: 10,
    committedRecords: 0,
    expectedRecords: 1_000
  });
  return updateTrainingTelemetry(first, {
    atMs: atMs + 1_000,
    scopeId: "a".repeat(64),
    recordsCompleted: 20,
    bytesCompleted: 2_000,
    bytesTotal: 10_000,
    currentRecord: 20,
    committedRecords: 0,
    expectedRecords: 1_000
  });
}

describe("training telemetry runtime integration", () => {
  it("resumes an exact legacy policy only from its persisted receipt", async () => {
    const ingestManifest = vi.fn(async () => ({
      paused: false,
      coverage: { complete: true }
    }));
    const progress = vi.fn(async () => ({
      manifestHash: "a".repeat(64),
      lastEntryReceipt: {
        manifestHash: "a".repeat(64),
        policy: "consolidate"
      }
    }));
    const service = {
      datasets: { progress },
      ingestManifest
    } as unknown as BrainService;
    const manager = new RuntimeJobManager(
      service,
      new EventEmitter() as EngineSupervisor
    );

    const resumed = await manager.resumeIngestion({
      brainId: "legacy-brain",
      manifestId: "legacy-manifest"
    });
    await expect(manager.wait(resumed.id)).resolves.toMatchObject({
      state: "complete"
    });
    expect(progress).toHaveBeenCalledWith("legacy-brain", "legacy-manifest");
    expect(ingestManifest.mock.calls[0]?.slice(0, 3)).toEqual([
      "legacy-brain",
      "legacy-manifest",
      "consolidate"
    ]);

    await expect(manager.resumeIngestion({
      brainId: "legacy-brain",
      manifestId: "legacy-manifest",
      policy: "consolidate" as never
    })).rejects.toThrow(/encode, pretrain, or archive/i);
    expect(progress).toHaveBeenCalledTimes(1);
  });

  it("publishes measured global rates, scoped checkpoints, foundation work, and worker memory", async () => {
    const events: RuntimeJobEvent[] = [];
    const service = {
      ingestManifest: vi.fn(async (...args: unknown[]) => {
        const report = args[4] as (
          progress: number,
          message: string,
          detail: Record<string, unknown>
        ) => void;
        report(0.2, "Foundation pass", {
          coverage: coverage(18, 2, 2_000),
          currentRecord: 20,
          committedRecords: 0,
          expectedRecords: 1_000,
          scopeId: "b".repeat(64),
          foundation: {
            batchIndex: 1,
            batchCount: 2,
            targetWindowIndex: 1,
            targetWindowCount: 2,
            completedTokens: 64,
            targetTokens: 256,
            completedChunks: 1,
            totalChunks: 4
          },
          resourceReadings: {
            processMemoryBytes: 2_000,
            processPeakMemoryBytes: 2_500
          },
          physicalSourceBytesComparable: true
        });
        await new Promise((resolve) => setTimeout(resolve, 275));
        report(0.3, "Committed checkpoint", {
          coverage: coverage(28, 2, 3_000),
          currentRecord: 30,
          committedRecords: 30,
          expectedRecords: 1_000,
          checkpointCommitted: true,
          scopeId: "b".repeat(64),
          foundation: {
            batchIndex: 1,
            batchCount: 2,
            targetWindowIndex: 1,
            targetWindowCount: 2,
            completedTokens: 128,
            targetTokens: 256,
            completedChunks: 2,
            totalChunks: 4
          },
          resourceReadings: {
            processMemoryBytes: 2_200,
            processPeakMemoryBytes: 2_700
          },
          physicalSourceBytesComparable: true
        });
        return { paused: false, coverage: { complete: true } };
      })
    } as unknown as BrainService;
    const manager = new RuntimeJobManager(
      service,
      new EventEmitter() as EngineSupervisor
    );
    manager.on("event", (event: RuntimeJobEvent) => events.push(event));

    const started = manager.startIngestion({
      brainId: "brain-a",
      manifestId: "manifest-a",
      policy: "pretrain",
      epochs: 1,
      resume: true
    });
    const finished = await manager.wait(started.id);

    expect(finished.telemetry).toMatchObject({
      stage: "checkpoint",
      recordsCompleted: 30,
      bytesCompleted: 3_000,
      bytesTotal: 10_000,
      nextCheckpoint: {
        scopeId: "b".repeat(64),
        currentRecord: 30,
        committedRecords: 30,
        expectedRecords: 1_000,
        targetRecord: 512
      },
      foundation: {
        completedTokens: 128,
        completedChunks: 2
      },
      memory: {
        source: "worker-process",
        currentPhysicalBytes: 2_200,
        peakPhysicalBytes: 2_700
      }
    });
    expect(finished.telemetry?.recordsTotal).toBeUndefined();
    expect(finished.telemetry?.recordsPerSecond).toBeGreaterThan(0);
    expect(finished.telemetry?.bytesPerSecond).toBeGreaterThan(0);
    expect(finished.telemetry?.foundation?.tokensPerSecond).toBeGreaterThan(0);
    const measuredEvent = events.find(
      ({ job }) => job.label === "Committed checkpoint"
    );
    expect(JSON.stringify(measuredEvent?.job.output)).not.toContain(
      "estimatedRemainingMs"
    );
  });

  it("resumes persisted counters but resets stale speed and ETA samples", async () => {
    let finish = (): void => undefined;
    const gate = new Promise<void>((resolve) => {
      finish = resolve;
    });
    const service = {
      ingestManifest: vi.fn(async () => {
        await gate;
        return { paused: false, coverage: { complete: true } };
      })
    } as unknown as BrainService;
    const manager = new RuntimeJobManager(
      service,
      new EventEmitter() as EngineSupervisor
    );
    const seed = telemetrySeed();

    const job = manager.startIngestion({
      brainId: "brain-a",
      manifestId: "manifest-a",
      resume: true
    }, seed);
    expect(job.telemetry).toMatchObject({
      recordsCompleted: seed.recordsCompleted,
      bytesCompleted: seed.bytesCompleted,
      elapsedMs: seed.elapsedMs
    });
    expect(job.telemetry?.recordsPerSecond).toBeUndefined();
    expect(job.telemetry?.bytesPerSecond).toBeUndefined();
    expect(job.telemetry?.totalEtaMs).toBeUndefined();
    expect(job.telemetry?.nextCheckpoint.etaMs).toBeUndefined();

    finish();
    await expect(manager.wait(job.id)).resolves.toMatchObject({ state: "complete" });
  });

  it("keeps cold substrate validation indeterminate until live traversal metrics arrive", async () => {
    let report: ((
      progress: number,
      message: string,
      detail?: Record<string, unknown>
    ) => void) | undefined;
    let finish = (): void => undefined;
    const gate = new Promise<void>((resolve) => {
      finish = resolve;
    });
    const service = {
      ingestManifest: vi.fn(async (...args: unknown[]) => {
        report = args[4] as typeof report;
        await gate;
        return { paused: false, coverage: { complete: true } };
      })
    } as unknown as BrainService;
    const manager = new RuntimeJobManager(
      service,
      new EventEmitter() as EngineSupervisor
    );

    const started = manager.startIngestion({
      brainId: "brain-a",
      manifestId: "manifest-a",
      policy: "pretrain",
      resume: true
    }, telemetrySeed());
    expect(started).toMatchObject({
      state: "running",
      phase: "preparing-neural-substrate",
      progress: 0,
      label: "Preparing neural substrate"
    });
    await vi.waitFor(() => expect(report).toBeTypeOf("function"));

    report!(0.98, "Validating 9,474 neural shards", {
      traversalMeasured: false
    });
    expect(manager.list("brain-a")[0]).toMatchObject({
      phase: "preparing-neural-substrate",
      progress: 0,
      label: "Preparing neural substrate"
    });

    report!(0.25, "Epoch 1/1 · record 1/4", {
      traversalMeasured: true,
      coverage: {
        ...coverage(1, 0, 250),
        discoveredRecords: 4
      },
      recordsCompleted: 1,
      recordsTotal: 4,
      currentRecord: 1,
      committedRecords: 0,
      expectedRecords: 4,
      scopeId: "d".repeat(64),
      physicalSourceBytesComparable: true
    });
    expect(manager.list("brain-a")[0]).toMatchObject({
      phase: "traversing-dataset",
      progress: 0.25,
      label: "Epoch 1/1 · record 1/4",
      output: {
        phase: "training",
        coverage: { discoveredRecords: 4 }
      }
    });

    finish();
    await expect(manager.wait(started.id)).resolves.toMatchObject({
      state: "complete",
      progress: 1
    });
  });

  it("reconciles a durable resume before measuring and rejects decoded Parquet byte rates", async () => {
    const events: RuntimeJobEvent[] = [];
    const resumeCoverage = (
      records: number,
      processedBytes: number
    ): TrainingCoverage => ({
      ...coverage(records, 0, processedBytes),
      discoveredRecords: records,
      discoveredBytes: 9_989_127,
      modalityCounts: { parquet: records }
    });
    const service = {
      ingestManifest: vi.fn(async (...args: unknown[]) => {
        const report = args[4] as (
          progress: number,
          message: string,
          detail: Record<string, unknown>
        ) => void;
        report(0.256, "Durable resume baseline", {
          coverage: resumeCoverage(5_632, 4_284_377),
          recordsCompleted: 5_632,
          recordsTotal: 21_990,
          currentRecord: 5_632,
          committedRecords: 5_632,
          checkpointCommitted: true,
          scopeId: "c".repeat(64),
          physicalSourceBytesComparable: false,
          durableBaseline: true,
          rateBaseline: true
        });
        report(0.2565, "First live record after replay", {
          coverage: resumeCoverage(5_637, 4_300_000),
          recordsCompleted: 5_637,
          recordsTotal: 21_990,
          currentRecord: 5_637,
          committedRecords: 5_632,
          expectedRecords: 21_990,
          scopeId: "c".repeat(64),
          physicalSourceBytesComparable: false,
          rateBaseline: true
        });
        await new Promise((resolve) => setTimeout(resolve, 275));
        report(0.2567, "Subsequent measured record", {
          coverage: resumeCoverage(5_641, 4_320_000),
          recordsCompleted: 5_641,
          recordsTotal: 21_990,
          currentRecord: 5_641,
          committedRecords: 5_632,
          expectedRecords: 21_990,
          scopeId: "c".repeat(64),
          physicalSourceBytesComparable: false
        });
        return { paused: false, coverage: { complete: true } };
      })
    } as unknown as BrainService;
    const manager = new RuntimeJobManager(
      service,
      new EventEmitter() as EngineSupervisor
    );
    manager.on("event", (event: RuntimeJobEvent) => events.push(event));

    const started = manager.startIngestion({
      brainId: "brain-a",
      manifestId: "manifest-a",
      policy: "pretrain",
      resume: true
    });
    const finished = await manager.wait(started.id);
    const durable = events.find(({ job }) =>
      job.phase === "preparing-neural-substrate" &&
      job.telemetry?.recordsCompleted === 5_632
    )?.job;
    const firstLive = events.find(
      ({ job }) => job.label === "First live record after replay"
    )?.job;

    expect(durable?.telemetry).toMatchObject({
      recordsCompleted: 5_632,
      bytesCompleted: 4_284_377,
      recordsTotal: 21_990,
      nextCheckpoint: {
        currentRecord: 5_632,
        committedRecords: 5_632,
        targetRecord: 6_144
      }
    });
    expect(durable).toMatchObject({
      progress: 0,
      label: "Preparing neural substrate",
      output: {
        phase: "preparing-neural-substrate",
        status: "Durable resume baseline"
      }
    });
    expect(firstLive?.telemetry).toMatchObject({
      recordsCompleted: 5_637,
      bytesCompleted: 4_284_377,
      recordsTotal: 21_990
    });
    expect(firstLive?.telemetry?.recordsPerSecond).toBeUndefined();
    expect(firstLive?.telemetry?.totalEtaMs).toBeUndefined();
    expect(firstLive?.telemetry?.nextCheckpoint.etaMs).toBeUndefined();
    expect(finished.telemetry?.recordsPerSecond).toBeGreaterThan(0);
    expect(finished.telemetry?.totalEtaMs).toBeGreaterThan(0);
    expect(finished.telemetry?.nextCheckpoint.etaMs).toBeGreaterThan(0);
    expect(finished.telemetry?.bytesPerSecond).toBeUndefined();
    expect(finished.telemetry?.bytesTotal).toBeUndefined();
  });
});

describe("initialization telemetry recovery", () => {
  const roots: string[] = [];

  afterEach(async () => {
    await Promise.all(
      roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))
    );
  });

  it("persists valid snapshots, retains them across retry, and drops invalid metadata", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-training-telemetry-"));
    roots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const plans = new BuildInitializationPlanStore(repository);
    await plans.create("brain-a", foundation, [
      { kind: "selection", selectionId: "selection-a" }
    ]);
    await plans.foundationReady("brain-a");
    const telemetry = telemetrySeed();

    await plans.taskTelemetry("brain-a", "selection-a", telemetry);
    await plans.taskState("brain-a", "selection-a", "failed", {
      error: "recoverable pause"
    });
    await plans.fail("brain-a", "initial-learning", "recoverable pause");
    await plans.retry("brain-a");
    expect((await plans.get("brain-a"))?.tasks[0]?.telemetry).toEqual(telemetry);

    const path = join(repository.brainDirectory("brain-a"), "initialization.json");
    const persisted = JSON.parse(await readFile(path, "utf8")) as {
      tasks: Array<{ telemetry?: { nextCheckpoint?: { targetRecord?: number } } }>;
    };
    persisted.tasks[0]!.telemetry!.nextCheckpoint!.targetRecord = 999;
    await writeFile(path, JSON.stringify(persisted), "utf8");
    const sanitized = await plans.get("brain-a");
    expect(sanitized?.tasks[0]).toMatchObject({ id: "selection-a" });
    expect(sanitized?.tasks[0]?.telemetry).toBeUndefined();
  });

  it("presents telemetry only for the correlated initialization job", () => {
    const telemetry = telemetrySeed();
    const event: BuildProgressEvent = {
      brainId: "brain-a",
      sequence: 2,
      phase: "initial-materials",
      progress: 0.2,
      label: "Learning",
      job: {
        id: "job-a",
        brainId: "brain-a",
        kind: "ingestion",
        state: "running",
        progress: 0.2,
        label: "Learning",
        createdAt: telemetry.startedAt,
        updatedAt: telemetry.updatedAt,
        telemetry
      }
    };

    expect(initializationRunningPresentation("brain-a", event).telemetry).toMatchObject({
      stageLabel: "Learning",
      throughput: expect.arrayContaining([
        expect.stringMatching(/records\/s/),
        expect.stringMatching(/\/s/)
      ]),
      timing: expect.arrayContaining([expect.stringMatching(/elapsed/)]),
      memory: "Worker memory unavailable"
    });
    expect(
      initializationRunningPresentation("brain-b", event).telemetry
    ).toBeUndefined();
  });

  it("keeps coordinator resume wiring and both renderer surfaces explicit", async () => {
    const [coordinator, renderer] = await Promise.all([
      readFile("src/main/buildInitializationCoordinator.ts", "utf8"),
      readFile("src/renderer/src/App.tsx", "utf8")
    ]);
    expect(coordinator).toContain("task.telemetry");
    expect(coordinator).toContain("this.plans.taskTelemetry");
    expect(renderer.match(/<TrainingTelemetryMetrics/g)).toHaveLength(2);
    expect(renderer).toContain("Measuring throughput…");
    expect(renderer).toContain("Worker-reported only");
  });
});
