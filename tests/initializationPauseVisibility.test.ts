import { EventEmitter } from "node:events";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService, RuntimeJobManager } from "../src/main/brainService";
import {
  BuildInitializationCoordinator,
  type InitialLearningProgress
} from "../src/main/buildInitializationCoordinator";
import {
  BuildInitializationPlanStore,
  type InitialFoundationPlan
} from "../src/main/buildInitializationPlan";
import { BuildResourceSelectionStore } from "../src/main/buildResourceSelections";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import {
  DEFAULT_CONFIG,
  type BrainDocument,
  type CreateBrainRequest,
  type DatasetManifest,
  type RuntimeJobEvent
} from "../src/shared/types";

const foundation: InitialFoundationPlan = {
  hardwareTier: "micro",
  modalities: [],
  origin: "ground-up",
};

function fixtureManifest(brainId: string): DatasetManifest {
  const now = new Date().toISOString();
  return {
    schemaVersion: 1,
    id: "preserved-fixture-manifest",
    brainId,
    createdAt: now,
    updatedAt: now,
    entryFile: "entries.ndjson",
    discoveredFiles: 1,
    discoveredBytes: 9_989_127,
    roots: ["/fixture"],
    manifestHash: "a".repeat(64)
  };
}

interface ColdStartSplitFixture {
  repository: BrainRepository;
  selectionsPath: string;
  brainId: string;
  selectionId: string;
  manifestId: string;
  sourcePath: string;
  sourceBytes: number;
  staleJobId: string;
}

async function coldStartSplitFixture(root: string): Promise<ColdStartSplitFixture> {
  const repository = new BrainRepository(join(root, "brains"));
  await repository.initialize();
  const selectionsPath = join(root, "selections.json");
  const selections = new BuildResourceSelectionStore(selectionsPath);
  const plans = new BuildInitializationPlanStore(repository);
  const selectionId = "cold-start-selection";
  const staleJobId = "stale-runtime-job";
  const sourcePath = join(root, "validation.parquet");
  const source = "A tiny story.\nA second tiny story.\n";
  await writeFile(sourcePath, source, "utf8");
  await selections.put({
    id: selectionId,
    kind: "files",
    label: "Cold-start validation fixture",
    itemCount: 1,
    bytes: Buffer.byteLength(source),
    fileCount: 1,
    paths: [sourcePath],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    state: "selected"
  });
  const brain = await repository.create(
    { ...DEFAULT_CONFIG, name: "Cold-start split fixture" },
    {
      initializing: true,
      recovery: {
        foundation,
        resources: [{ kind: "selection", selectionId }]
      }
    }
  );
  await plans.create(brain.id, foundation, [
    { kind: "selection", selectionId }
  ]);
  await plans.foundationReady(brain.id);
  await selections.claim(selectionId, brain.id);
  const setupService = new BrainService(
    repository,
    new EventEmitter() as EngineSupervisor
  );
  const manifest = await setupService.previewDataset(brain.id, [sourcePath]);
  const initialProgress = await setupService.datasets.progress(brain.id, manifest.id);
  const pausedAt = new Date().toISOString();
  await setupService.datasets.saveProgress(
    brain.id,
    { ...initialProgress.cursor, state: "paused", updatedAt: pausedAt },
    { ...initialProgress.coverage, complete: false, updatedAt: pausedAt },
    initialProgress.lastEntryReceipt,
    initialProgress.generation
  );
  await selections.commitManifest(selectionId, manifest.id);
  await selections.attachRuntimeJob(selectionId, staleJobId);

  // Complete one ordinary failure/retry transition so the persisted split is
  // exactly the second-attempt crash window observed in the live brain: the
  // task failure was committed, but the process exited before plan/outer
  // failure convergence.
  const firstFailure = new Error("first recoverable worker interruption");
  await plans.taskState(brain.id, selectionId, "failed", {
    manifestId: manifest.id,
    error: firstFailure
  });
  await plans.fail(brain.id, "initial-learning", firstFailure);
  await repository.failInitialization(brain.id, "initial-learning", firstFailure);
  await plans.retry(brain.id);
  await repository.retryInitialization(brain.id);
  await plans.taskState(brain.id, selectionId, "failed", {
    manifestId: manifest.id,
    error: new Error("process exited after the runtime job failed")
  });

  return {
    repository,
    selectionsPath,
    brainId: brain.id,
    selectionId,
    manifestId: manifest.id,
    sourcePath,
    sourceBytes: Buffer.byteLength(source),
    staleJobId
  };
}

function completeWorkerResult(
  brainId: string,
  sourceBytes: number
): Record<string, unknown> {
  const coverage = {
    discoveredFiles: 1,
    processedFiles: 1,
    rejectedFiles: 0,
    discoveredRecords: 100,
    processedRecords: 100,
    rejectedRecords: 0,
    discoveredBytes: sourceBytes,
    processedBytes: sourceBytes,
    shards: 1,
    modalityCounts: { parquet: 100 },
    errors: [],
    errorCount: 0,
    errorsTruncated: false,
    complete: true
  };
  return {
    brainId,
    metrics: {
      plasticityEvents: 130_000,
      counters: {
        inference_count: 3,
        consolidation_cycles: 2
      }
    },
    source: {
      kind: "parquet",
      learned_ideas: 100,
      learned_concepts: 40,
      synaptic_update_events: 3_824,
      parameter_update_steps: 5,
      parameter_checksum_changed: true,
      warnings: [],
      coverage
    },
    coverage,
    parameterChecksumBefore: "a".repeat(64),
    parameterChecksumAfter: "b".repeat(64)
  };
}

function emitRecordProgress(
  engine: EventEmitter,
  brainId: string,
  jobId: string,
  sourceBytes: number
): void {
  engine.emit("event", {
    type: "job-progress",
    brainId,
    jobId,
    progress: 0.37,
    message: "Learning record 37 of 100",
    data: {
      datasetProgress: {
        currentRecord: 37,
        committedRecords: 32,
        expectedRecords: 100,
        recordTotalKnown: true,
        checkpointCommitted: true,
        coverage: {
          discoveredFiles: 1,
          processedFiles: 0,
          rejectedFiles: 0,
          discoveredRecords: 100,
          processedRecords: 37,
          rejectedRecords: 0,
          discoveredBytes: sourceBytes,
          processedBytes: Math.floor(sourceBytes * 0.37),
          shards: 1,
          modalityCounts: { parquet: 37 },
          errors: [],
          complete: false
        }
      }
    }
  });
}

describe("paused initial-learning visibility and recovery", () => {
  const temporaryRoots: string[] = [];

  afterEach(async () => {
    await Promise.all(
      temporaryRoots.splice(0).map((root) =>
        rm(root, { recursive: true, force: true })
      )
    );
  });

  it("preserves the source manifest, gates chat, and resumes the same dataset", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-paused-initial-learning-"));
    temporaryRoots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const selections = new BuildResourceSelectionStore(join(root, "selections.json"));
    const plans = new BuildInitializationPlanStore(repository);
    const sourcePath = join(root, "validation.parquet");
    await selections.put({
      id: "preserved-selection",
      kind: "files",
      label: "TinyStories validation fixture",
      itemCount: 1,
      paths: [sourcePath],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      state: "selected"
    });

    let ingestionAttempts = 0;
    const service = {
      repository,
      preflightStart: vi.fn(async () => undefined),
      create: vi.fn(async (
        request: CreateBrainRequest,
        _onProgress?: unknown,
        onPersisted?: (
          brain: BrainDocument,
          plan: InitialFoundationPlan
        ) => Promise<void>
      ) => {
        const brain = await repository.create(request.config, {
          initializing: true,
          recovery: {
            foundation,
            resources: request.initialResources ?? []
          }
        });
        await onPersisted?.(brain, foundation);
        return brain;
      }),
      resumeFoundation: vi.fn(async (brainId: string) => repository.get(brainId)),
      previewDataset: vi.fn(async (brainId: string) => fixtureManifest(brainId)),
      ingestManifest: vi.fn(async () => {
        ingestionAttempts += 1;
        if (ingestionAttempts === 1) {
          return {
            paused: true,
            pauseReason: "Learning paused before the current source was committed.",
            coverage: { complete: false }
          };
        }
        return { paused: false, coverage: { complete: true } };
      }),
      crawlWeb: vi.fn(async () => ({ stopped: false }))
    };
    const jobs = new RuntimeJobManager(
      service as unknown as BrainService,
      new EventEmitter() as EngineSupervisor
    );
    const coordinator = new BuildInitializationCoordinator(
      repository,
      service as unknown as BrainService,
      jobs,
      selections,
      plans
    );
    const request: CreateBrainRequest = {
      config: { ...DEFAULT_CONFIG, name: "Paused recovery fixture" },
      hardwareTier: "micro",
      initialResources: [
        { kind: "selection", selectionId: "preserved-selection" }
      ]
    };

    await expect(coordinator.create(request)).rejects.toThrow(
      /learning paused before the current source was committed/i
    );

    const [summary] = await repository.list();
    const pausedBrain = await repository.get(summary!.id);
    const [visiblePausedJob] = jobs.list(pausedBrain.id);
    expect(visiblePausedJob).toMatchObject({
      kind: "ingestion",
      state: "failed",
      error: expect.stringMatching(/learning paused/i),
      output: {
        paused: true,
        pauseReason: expect.stringMatching(/learning paused/i),
        coverage: { complete: false }
      }
    });
    expect(visiblePausedJob!.progress).toBeLessThan(1);
    expect(pausedBrain.readiness).toMatchObject({
      state: "failed",
      failure: {
        phase: "initial-learning",
        message: expect.stringMatching(/learning paused/i),
        retryable: true
      },
      recovery: {
        resources: [
          { kind: "selection", selectionId: "preserved-selection" }
        ]
      }
    });
    expect(await selections.get("preserved-selection")).toMatchObject({
      state: "retryable",
      paths: [sourcePath],
      manifestId: "preserved-fixture-manifest"
    });
    expect(
      (await selections.get("preserved-selection"))?.runtimeJobId
    ).toBeUndefined();
    expect(await plans.get(pausedBrain.id)).toMatchObject({
      phase: "failed",
      tasks: [
        {
          id: "preserved-selection",
          state: "failed",
          manifestId: "preserved-fixture-manifest",
          error: expect.stringMatching(/learning paused/i)
        }
      ]
    });

    const engineRequest = vi.fn();
    const gatedService = new BrainService(
      repository,
      { request: engineRequest } as unknown as EngineSupervisor
    );
    await expect(
      gatedService.chat(pausedBrain.id, "chat must remain unavailable")
    ).rejects.toThrow(/initial learning/i);
    expect(engineRequest).not.toHaveBeenCalled();

    const recovered = await coordinator.retry(pausedBrain.id);
    expect(recovered.readiness.state).toBe("ready");
    expect(service.previewDataset).toHaveBeenCalledTimes(1);
    expect(service.ingestManifest).toHaveBeenCalledTimes(2);
    expect(await selections.get("preserved-selection")).toBeUndefined();
  });

  it("cold-starts the exact split state from the same manifest with visible progress and synchronized counters", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-cold-start-split-success-"));
    temporaryRoots.push(root);
    const split = await coldStartSplitFixture(root);
    const plans = new BuildInitializationPlanStore(split.repository);
    const selections = new BuildResourceSelectionStore(split.selectionsPath);

    expect(await split.repository.get(split.brainId)).toMatchObject({
      readiness: { state: "initializing", attempt: 2 },
      counters: {
        plasticityEvents: 0,
        inferenceCount: 0,
        consolidationCycles: 0
      }
    });
    expect(await plans.get(split.brainId)).toMatchObject({
      phase: "initial-learning",
      attempt: 2,
      tasks: [{
        id: split.selectionId,
        state: "failed",
        manifestId: split.manifestId
      }]
    });
    expect(await selections.get(split.selectionId)).toMatchObject({
      state: "training",
      manifestId: split.manifestId,
      runtimeJobId: split.staleJobId
    });

    const eventBus = new EventEmitter();
    const resumedJobIds: string[] = [];
    const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
      if (method !== "ingest") throw new Error(`Unexpected worker method ${method}`);
      const jobId = String(params.jobId);
      resumedJobIds.push(jobId);
      emitRecordProgress(eventBus, split.brainId, jobId, split.sourceBytes);
      return completeWorkerResult(split.brainId, split.sourceBytes);
    });
    const engine = Object.assign(eventBus, { request }) as unknown as EngineSupervisor;
    const service = new BrainService(split.repository, engine);
    const preview = vi.spyOn(service, "previewDataset");
    expect(await service.datasets.cursor(split.brainId, split.manifestId)).toMatchObject({
      state: "paused",
      currentEpoch: 0,
      nextEntry: 0,
      nextRecord: 0,
      processedFiles: 0,
      processedRecords: 0,
      processedBytes: 0
    });
    const jobs = new RuntimeJobManager(service, engine);
    const events: RuntimeJobEvent[] = [];
    jobs.on("event", (event: RuntimeJobEvent) => events.push(event));
    const restarted = new BuildInitializationCoordinator(
      split.repository,
      service,
      jobs,
      selections,
      plans
    );
    const recoveryErrors: unknown[] = [];
    const recoveryProgress: InitialLearningProgress[] = [];

    await restarted.recoverAll(
      (_brainId, error) => recoveryErrors.push(error),
      undefined,
      (_brainId, event) => recoveryProgress.push(event)
    );

    expect(recoveryErrors).toEqual([]);
    expect(preview).not.toHaveBeenCalled();
    expect(request).toHaveBeenCalledTimes(1);
    expect(request.mock.calls[0]?.[1]).toMatchObject({
      brainId: split.brainId,
      manifestId: split.manifestId,
      jobId: resumedJobIds[0]
    });
    expect(resumedJobIds).toHaveLength(1);
    expect(resumedJobIds[0]).not.toBe(split.staleJobId);
    expect(events.some(({ job }) =>
      job.state === "running" &&
      job.label.includes("Learning record 37 of 100") &&
      job.output !== undefined &&
      (job.output as { progress?: { currentRecord?: number } }).progress
        ?.currentRecord === 37
    )).toBe(true);
    expect(recoveryProgress.some(({ job }) =>
      job.state === "running" &&
      job.label.includes("Learning record 37 of 100") &&
      (job.output as { progress?: { committedRecords?: number } } | undefined)
        ?.progress?.committedRecords === 32
    )).toBe(true);

    const ready = await split.repository.get(split.brainId);
    expect(ready).toMatchObject({
      readiness: { state: "ready", attempt: 2 },
      counters: {
        plasticityEvents: 130_000,
        inferenceCount: 3,
        consolidationCycles: 2
      },
      trainingSources: [{
        kind: "parquet",
        learnedIdeas: 100,
        learnedConcepts: 40,
        learnedSynapses: 3_824,
        learnedParameterSteps: 5,
        parametersChanged: true
      }]
    });
    expect(jobs.isInitializing(split.brainId)).toBe(false);
    expect(await plans.get(split.brainId)).toBeUndefined();
    expect(await selections.get(split.selectionId)).toBeUndefined();
  });

  it("cold-starts the exact split state and converges both outer and plan failures when learning pauses again", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-cold-start-split-failure-"));
    temporaryRoots.push(root);
    const split = await coldStartSplitFixture(root);
    const plans = new BuildInitializationPlanStore(split.repository);
    const selections = new BuildResourceSelectionStore(split.selectionsPath);
    const eventBus = new EventEmitter();
    const resumedJobIds: string[] = [];
    const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
      if (method !== "ingest") throw new Error(`Unexpected worker method ${method}`);
      const jobId = String(params.jobId);
      resumedJobIds.push(jobId);
      emitRecordProgress(eventBus, split.brainId, jobId, split.sourceBytes);
      throw new Error("non-finite batch training loss after cold start");
    });
    const engine = Object.assign(eventBus, { request }) as unknown as EngineSupervisor;
    const service = new BrainService(split.repository, engine);
    const preview = vi.spyOn(service, "previewDataset");
    const jobs = new RuntimeJobManager(service, engine);
    const events: RuntimeJobEvent[] = [];
    jobs.on("event", (event: RuntimeJobEvent) => events.push(event));
    const restarted = new BuildInitializationCoordinator(
      split.repository,
      service,
      jobs,
      selections,
      plans
    );
    const recoveryErrors: Array<{ brainId: string; error: unknown }> = [];
    const recoveryProgress: InitialLearningProgress[] = [];

    await restarted.recoverAll(
      (brainId, error) => {
        recoveryErrors.push({ brainId, error });
      },
      undefined,
      (_brainId, event) => recoveryProgress.push(event)
    );

    expect(recoveryErrors).toHaveLength(1);
    expect(recoveryErrors[0]?.brainId).toBe(split.brainId);
    expect(String(recoveryErrors[0]?.error)).toMatch(/non-finite batch training loss/i);
    expect(preview).not.toHaveBeenCalled();
    expect(request).toHaveBeenCalledTimes(1);
    expect(resumedJobIds).toHaveLength(1);
    expect(resumedJobIds[0]).not.toBe(split.staleJobId);
    expect(events.some(({ job }) =>
      job.state === "running" && job.label.includes("Learning record 37 of 100")
    )).toBe(true);
    expect(recoveryProgress.some(({ job }) =>
      job.state === "running" && job.label.includes("Learning record 37 of 100")
    )).toBe(true);
    expect(jobs.list(split.brainId)[0]).toMatchObject({
      id: resumedJobIds[0],
      state: "failed",
      error: expect.stringMatching(/non-finite batch training loss/i)
    });

    const failedOuter = await split.repository.get(split.brainId);
    const failedPlan = await plans.get(split.brainId);
    expect(failedOuter.readiness).toMatchObject({
      state: "failed",
      attempt: 2,
      failure: {
        phase: "initial-learning",
        message: expect.stringMatching(/non-finite batch training loss/i),
        retryable: true
      }
    });
    expect(failedPlan).toMatchObject({
      phase: "failed",
      attempt: 2,
      failure: {
        phase: "initial-learning",
        message: expect.stringMatching(/non-finite batch training loss/i),
        retryable: true
      },
      tasks: [{
        id: split.selectionId,
        state: "failed",
        manifestId: split.manifestId,
        error: expect.stringMatching(/non-finite batch training loss/i)
      }]
    });
    expect(await selections.get(split.selectionId)).toMatchObject({
      state: "retryable",
      brainId: split.brainId,
      manifestId: split.manifestId,
      paths: [split.sourcePath]
    });
    expect(
      (await selections.get(split.selectionId))?.runtimeJobId
    ).toBeUndefined();
    expect(jobs.isInitializing(split.brainId)).toBe(false);
  });
});
