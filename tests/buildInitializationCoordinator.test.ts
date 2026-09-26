import { EventEmitter } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService, RuntimeJobManager } from "../src/main/brainService";
import { BuildResourceSelectionStore } from "../src/main/buildResourceSelections";
import {
  BuildInitializationPlanStore,
  type InitialFoundationPlan
} from "../src/main/buildInitializationPlan";
import {
  BuildInitializationCoordinator,
  type InitialLearningProgress
} from "../src/main/buildInitializationCoordinator";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { initializationRecoveryPresentation } from "../src/renderer/src/initializationRecoveryPresentation";
import {
  DEFAULT_CONFIG,
  type BrainDocument,
  type CreateBrainRequest,
  type DatasetManifest
} from "../src/shared/types";

const foundation: InitialFoundationPlan = {
  hardwareTier: "micro",
  modalities: [],
  origin: "ground-up",
};

const request = (selectionId?: string): CreateBrainRequest => ({
  config: { ...DEFAULT_CONFIG, name: "Recovery fixture" },
  hardwareTier: "micro",
  ...(selectionId
    ? { initialResources: [{ kind: "selection" as const, selectionId }] }
    : {})
});

function manifest(brainId: string, id = "fixture-manifest"): DatasetManifest {
  const now = new Date().toISOString();
  return {
    schemaVersion: 1,
    id,
    brainId,
    createdAt: now,
    updatedAt: now,
    entryFile: "entries.ndjson",
    discoveredFiles: 1,
    discoveredBytes: 12,
    roots: ["/fixture"],
    manifestHash: "a".repeat(64)
  };
}

interface Harness {
  root: string;
  repository: BrainRepository;
  selections: BuildResourceSelectionStore;
  plans: BuildInitializationPlanStore;
  jobs: RuntimeJobManager;
  coordinator: BuildInitializationCoordinator;
  service: {
    create: ReturnType<typeof vi.fn>;
    resumeFoundation: ReturnType<typeof vi.fn>;
    previewDataset: ReturnType<typeof vi.fn>;
    ingestManifest: ReturnType<typeof vi.fn>;
    crawlWeb: ReturnType<typeof vi.fn>;
    preflightStart: ReturnType<typeof vi.fn>;
  };
}

async function makeHarness(options: {
  previewDelayMs?: number;
  foundationDelayMs?: number;
  failIngestionOnce?: boolean;
  pauseIngestionOnce?: boolean;
  pauseCrawlOnce?: boolean;
} = {}): Promise<Harness> {
  const root = await mkdtemp(join(tmpdir(), "omni-init-coordinator-"));
  const repository = new BrainRepository(join(root, "brains"));
  await repository.initialize();
  const selections = new BuildResourceSelectionStore(join(root, "selections.json"));
  const plans = new BuildInitializationPlanStore(repository);
  let ingestAttempts = 0;
  let crawlAttempts = 0;
  const service = {
    repository,
    preflightStart: vi.fn(async () => undefined),
    create: vi.fn(async (
      buildRequest: CreateBrainRequest,
      _onProgress?: unknown,
      onPersisted?: (brain: BrainDocument, plan: InitialFoundationPlan) => Promise<void>
    ) => {
      const brain = await repository.create(buildRequest.config, {
        initializing: true,
        recovery: {
          foundation,
          resources: buildRequest.initialResources ?? []
        }
      });
      await onPersisted?.(brain, foundation);
      return brain;
    }),
    resumeFoundation: vi.fn(async (brainId: string) => {
      if (options.foundationDelayMs) {
        await new Promise((resolve) => setTimeout(resolve, options.foundationDelayMs));
      }
      return repository.get(brainId);
    }),
    previewDataset: vi.fn(async (brainId: string) => {
      if (options.previewDelayMs) {
        await new Promise((resolve) => setTimeout(resolve, options.previewDelayMs));
      }
      return manifest(brainId);
    }),
    ingestManifest: vi.fn(async () => {
      ingestAttempts += 1;
      if (options.failIngestionOnce && ingestAttempts === 1) {
        throw new Error("fixture ingestion interruption");
      }
      if (options.pauseIngestionOnce && ingestAttempts === 1) {
        return {
          paused: true,
          pauseReason: "Paused before fixture.parquet: recoverable worker interruption",
          coverage: { complete: false }
        };
      }
      return { paused: false, coverage: { complete: true } };
    }),
    crawlWeb: vi.fn(async () => {
      crawlAttempts += 1;
      if (options.pauseCrawlOnce && crawlAttempts === 1) {
        return { stopped: true, coverage: { complete: false } };
      }
      return { stopped: false, coverage: { complete: true } };
    })
  };
  const engine = new EventEmitter() as EngineSupervisor;
  const jobs = new RuntimeJobManager(service as unknown as BrainService, engine);
  const coordinator = new BuildInitializationCoordinator(
    repository,
    service as unknown as BrainService,
    jobs,
    selections,
    plans
  );
  return { root, repository, selections, plans, jobs, coordinator, service };
}

describe("main-authoritative initialization recovery", () => {
  const roots: string[] = [];

  afterEach(async () => {
    await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
  });

  it("reports shared-worker ownership from the first create edge until handoff", async () => {
    const harness = await makeHarness();
    roots.push(harness.root);

    expect(harness.coordinator.isBusy()).toBe(false);
    const creating = harness.coordinator.create(request());
    expect(harness.coordinator.isBusy()).toBe(true);

    await expect(creating).resolves.toMatchObject({
      readiness: { state: "ready" }
    });
    expect(harness.coordinator.isBusy()).toBe(false);
  });

  it("waits beyond one second for first-learning work without timing out or opening chat", async () => {
    const harness = await makeHarness({ previewDelayMs: 1_100 });
    roots.push(harness.root);
    await harness.selections.put({
      id: "slow-selection",
      kind: "files",
      label: "slow fixture",
      itemCount: 1,
      paths: [join(harness.root, "slow.txt")],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      state: "selected"
    });

    const started = performance.now();
    const creating = harness.coordinator.create(request("slow-selection"));
    try {
      const visibilityDeadline = performance.now() + 1_000;
      let summaries = await harness.repository.list();
      while (summaries.length === 0 && performance.now() < visibilityDeadline) {
        await new Promise((resolve) => setTimeout(resolve, 10));
        summaries = await harness.repository.list();
      }
      expect(summaries).toHaveLength(1);
      expect(
        (await harness.repository.get(summaries[0]!.id)).readiness.state
      ).toBe("initializing");

      const brain = await creating;
      expect(performance.now() - started).toBeGreaterThan(1_000);
      expect(brain.readiness.state).toBe("ready");
      expect(harness.service.previewDataset).toHaveBeenCalledTimes(1);
      expect(harness.service.ingestManifest).toHaveBeenCalledTimes(1);
    } finally {
      // Do not let an early assertion leave background initialization running
      // while Vitest tears down the repository fixture.
      await creating.catch(() => undefined);
    }
  });

  it("recovers interrupted foundations sequentially instead of loading them in parallel", async () => {
    const harness = await makeHarness();
    roots.push(harness.root);
    const ids: string[] = [];
    for (const name of ["first", "second"]) {
      const brain = await harness.repository.create(
        { ...DEFAULT_CONFIG, name },
        {
          initializing: true,
          recovery: { foundation, resources: [] }
        }
      );
      ids.push(brain.id);
    }
    let active = 0;
    let peak = 0;
    const order: string[] = [];
    harness.service.resumeFoundation.mockImplementation(async (brainId: string) => {
      order.push(brainId);
      active += 1;
      peak = Math.max(peak, active);
      await new Promise((resolve) => setTimeout(resolve, 35));
      active -= 1;
      return harness.repository.get(brainId);
    });
    const expectedOrder = (await harness.repository.list()).map((summary) => summary.id);

    await harness.coordinator.recoverAll();

    expect(peak).toBe(1);
    expect(order).toEqual(expectedOrder);
    for (const id of ids) {
      expect((await harness.repository.get(id)).readiness.state).toBe("ready");
    }
  });

  it("resumes a committed first-dataset cursor without rehashing or duplicate learning", async () => {
    const harness = await makeHarness();
    roots.push(harness.root);
    const brain = await harness.repository.create(
      { ...DEFAULT_CONFIG, name: "Dataset crash" },
      { initializing: true }
    );
    await harness.selections.put({
      id: "resume-selection",
      kind: "folder",
      label: "resume fixture",
      itemCount: 1,
      paths: [join(harness.root, "dataset")],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      state: "selected"
    });
    await harness.plans.create(brain.id, foundation, [
      { kind: "selection", selectionId: "resume-selection" }
    ]);
    await harness.plans.foundationReady(brain.id);
    await harness.selections.claim("resume-selection", brain.id);
    await harness.selections.commitManifest("resume-selection", "fixture-manifest");
    await harness.plans.taskState(
      brain.id,
      "resume-selection",
      "running",
      { manifestId: "fixture-manifest" }
    );

    await harness.coordinator.recoverAll();
    expect((await harness.repository.get(brain.id)).readiness.state).toBe("ready");
    expect(harness.service.previewDataset).not.toHaveBeenCalled();
    expect(harness.service.ingestManifest).toHaveBeenCalledTimes(1);

    const restarted = new BuildInitializationCoordinator(
      harness.repository,
      harness.service as unknown as BrainService,
      new RuntimeJobManager(
        harness.service as unknown as BrainService,
        new EventEmitter() as EngineSupervisor
      ),
      harness.selections,
      harness.plans
    );
    await restarted.recoverAll();
    expect(harness.service.ingestManifest).toHaveBeenCalledTimes(1);
  });

  it("resumes the same persisted web frontier without renderer state", async () => {
    const harness = await makeHarness();
    roots.push(harness.root);
    const brain = await harness.repository.create(
      { ...DEFAULT_CONFIG, name: "Web crash" },
      { initializing: true }
    );
    const plan = await harness.plans.create(brain.id, foundation, [
      { kind: "web", url: "https://example.com/learn#fragment" }
    ]);
    await harness.plans.foundationReady(brain.id);
    const webTask = plan.tasks[0]!;
    await harness.plans.taskState(brain.id, webTask.id, "running");

    await harness.coordinator.recoverAll();

    expect((await harness.repository.get(brain.id)).readiness.state).toBe("ready");
    expect(harness.service.crawlWeb).toHaveBeenCalledTimes(1);
    expect(harness.service.crawlWeb.mock.calls[0]?.[0]).toMatchObject({
      brainId: brain.id,
      url: "https://example.com/learn",
      crawlId: (webTask.kind === "web" ? webTask.request.crawlId : ""),
      resume: true,
      respectRobots: true
    });
    await harness.coordinator.recoverAll();
    expect(harness.service.crawlWeb).toHaveBeenCalledTimes(1);
  });

  it("never unlocks chat when an initial web crawl pauses with incomplete coverage", async () => {
    const harness = await makeHarness({ pauseCrawlOnce: true });
    roots.push(harness.root);
    const buildRequest: CreateBrainRequest = {
      ...request(),
      initialResources: [{ kind: "web", url: "https://example.com/learn" }]
    };

    await expect(harness.coordinator.create(buildRequest)).rejects.toThrow(
      /complete crawl coverage/i
    );

    const [summary] = await harness.repository.list();
    const failed = await harness.repository.get(summary!.id);
    expect(failed.readiness).toMatchObject({
      state: "failed",
      failure: {
        phase: "initial-learning",
        message: expect.stringMatching(/complete crawl coverage/i),
        retryable: true
      }
    });
    expect(await harness.plans.get(failed.id)).toMatchObject({
      phase: "failed",
      tasks: [{ kind: "web", state: "failed" }]
    });

    const recovered = await harness.coordinator.retry(failed.id);
    expect(recovered.readiness.state).toBe("ready");
    expect(harness.service.crawlWeb).toHaveBeenCalledTimes(2);
    expect(harness.service.crawlWeb.mock.calls[1]?.[0]).toMatchObject({
      crawlId: harness.service.crawlWeb.mock.calls[0]?.[0].crawlId,
      resume: true
    });
  });

  it("records a failed initial job, keeps chat gated, and retries from its manifest", async () => {
    const harness = await makeHarness({ failIngestionOnce: true });
    roots.push(harness.root);
    await harness.selections.put({
      id: "retry-selection",
      kind: "files",
      label: "retry fixture",
      itemCount: 1,
      paths: [join(harness.root, "retry.txt")],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      state: "selected"
    });

    await expect(
      harness.coordinator.create(request("retry-selection"))
    ).rejects.toThrow(/fixture ingestion interruption/i);
    const [summary] = await harness.repository.list();
    const failed = await harness.repository.get(summary!.id);
    expect(failed.readiness).toMatchObject({
      state: "failed",
      failure: { phase: "initial-learning", retryable: true }
    });
    expect(harness.service.previewDataset).toHaveBeenCalledTimes(1);
    expect(harness.service.ingestManifest).toHaveBeenCalledTimes(1);

    const recovered = await harness.coordinator.retry(failed.id);
    expect(recovered.readiness.state).toBe("ready");
    expect(harness.service.previewDataset).toHaveBeenCalledTimes(1);
    expect(harness.service.ingestManifest).toHaveBeenCalledTimes(2);
  });

  it("keeps retry record progress visible and converges promptly to a failed card", async () => {
    const harness = await makeHarness({ failIngestionOnce: true });
    roots.push(harness.root);
    await harness.selections.put({
      id: "visible-retry-selection",
      kind: "files",
      label: "visible retry fixture",
      itemCount: 1,
      paths: [join(harness.root, "retry.parquet")],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      state: "selected"
    });
    await expect(
      harness.coordinator.create(request("visible-retry-selection"))
    ).rejects.toThrow(/fixture ingestion interruption/i);
    const [summary] = await harness.repository.list();
    const failed = await harness.repository.get(summary!.id);

    let releaseFailure = (): void => undefined;
    const failureGate = new Promise<void>((resolveFailure) => {
      releaseFailure = resolveFailure;
    });
    let resolveRecordProgress = (): void => undefined;
    const recordProgress = new Promise<void>((resolveProgress) => {
      resolveRecordProgress = resolveProgress;
    });
    harness.service.ingestManifest.mockImplementation(async (...args: unknown[]) => {
      const report = args[4] as (
        progress: number,
        message: string,
        detail: Record<string, unknown>
      ) => void;
      report(0.37, "Learning record 37 of 100", {
        currentRecord: 37,
        committedRecords: 32,
        expectedRecords: 100,
        recordTotalKnown: true,
        coverage: {
          complete: false,
          discoveredRecords: 37,
          processedRecords: 37,
          rejectedRecords: 0
        }
      });
      resolveRecordProgress();
      await failureGate;
      throw new Error(
        "Paused before retry.parquet: RuntimeError: non-finite batch training loss"
      );
    });
    const events: InitialLearningProgress[] = [];
    const retry = harness.coordinator.retry(
      failed.id,
      undefined,
      (event) => events.push(event)
    );
    const retryFailure = expect(retry).rejects.toThrow(
      /non-finite batch training loss/i
    );

    await recordProgress;
    await vi.waitFor(() => {
      expect(events.some((event) =>
        event.label === "Learning record 37 of 100" &&
        event.job.state === "running"
      )).toBe(true);
    });
    expect(harness.jobs.isInitializing(failed.id)).toBe(true);
    expect((await harness.repository.get(failed.id)).readiness.state).toBe("initializing");
    const liveProgress = [...events].reverse().find(
      (event) => event.label === "Learning record 37 of 100"
    );
    expect(liveProgress).toMatchObject({
      progress: 0.37,
      job: {
        state: "running",
        output: {
          progress: {
            currentRecord: 37,
            committedRecords: 32,
            expectedRecords: 100,
            recordTotalKnown: true
          }
        }
      }
    });

    releaseFailure();
    await retryFailure;
    const converged = await harness.repository.get(failed.id);
    expect(converged.readiness).toMatchObject({
      state: "failed",
      failure: {
        phase: "initial-learning",
        message: expect.stringMatching(/non-finite batch training loss/i),
        retryable: true
      }
    });
    expect(harness.jobs.isInitializing(failed.id)).toBe(false);
    expect(harness.jobs.list(failed.id)[0]).toMatchObject({
      state: "failed",
      error: expect.stringMatching(/non-finite batch training loss/i)
    });
    expect(await harness.plans.get(failed.id)).toMatchObject({
      phase: "failed",
      tasks: [
        {
          id: "visible-retry-selection",
          state: "failed",
          error: expect.stringMatching(/non-finite batch training loss/i)
        }
      ]
    });
    expect(initializationRecoveryPresentation(converged.readiness)).toMatchObject({
      tone: "paused",
      title: "Learning paused",
      actionLabel: "Resume training"
    });
  });

  it("never unlocks chat when initial ingestion returns a resumable pause", async () => {
    const harness = await makeHarness({ pauseIngestionOnce: true });
    roots.push(harness.root);
    await harness.selections.put({
      id: "paused-selection",
      kind: "files",
      label: "paused fixture",
      itemCount: 1,
      paths: [join(harness.root, "fixture.parquet")],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      state: "selected"
    });

    await expect(
      harness.coordinator.create(request("paused-selection"))
    ).rejects.toThrow(/recoverable worker interruption/i);

    const [summary] = await harness.repository.list();
    const failed = await harness.repository.get(summary!.id);
    expect(failed.readiness).toMatchObject({
      state: "failed",
      failure: {
        phase: "initial-learning",
        message: expect.stringMatching(/recoverable worker interruption/i),
        retryable: true
      },
      recovery: expect.any(Object)
    });
    const plan = await harness.plans.get(failed.id);
    expect(plan).toMatchObject({
      phase: "failed",
      tasks: [
        {
          id: "paused-selection",
          state: "failed",
          manifestId: "fixture-manifest",
          error: expect.stringMatching(/recoverable worker interruption/i)
        }
      ]
    });
    expect(await harness.selections.get("paused-selection")).toMatchObject({
      state: "retryable",
      manifestId: "fixture-manifest"
    });
    expect(
      (await harness.selections.get("paused-selection"))?.runtimeJobId
    ).toBeUndefined();

    const recovered = await harness.coordinator.retry(failed.id);
    expect(recovered.readiness.state).toBe("ready");
    expect(harness.service.previewDataset).toHaveBeenCalledTimes(1);
    expect(harness.service.ingestManifest).toHaveBeenCalledTimes(2);
  });
});
