import { EventEmitter } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  RuntimeJobManager,
  BrainService
} from "../src/main/brainService";
import { BrainRepository } from "../src/main/brainRepository";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { IdleCognitionScheduler } from "../src/main/idleCognitionScheduler";
import { GIB, ResourcePlanner } from "../src/main/resourcePlanner";
import { DEFAULT_CONFIG, type IdleCycleResult } from "../src/shared/types";

function deferred(): { promise: Promise<void>; resolve(): void } {
  let resolve = (): void => undefined;
  const promise = new Promise<void>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

function testResourcePlanner(root: string): ResourcePlanner {
  return new ResourcePlanner(root, {
    readResources: async () => ({
      totalMemoryBytes: 16 * GIB,
      availableMemoryBytes: 13 * GIB,
      diskTotalBytes: 500 * GIB,
      diskFreeBytes: 240 * GIB
    }),
    benchmark: async () => ({
      measuredAt: "2026-09-07T00:00:00.000Z",
      sampleBytes: 16 * 1024 * 1024,
      memoryBytesPerSecond: 12 * GIB,
      storageBytesPerSecond: 734_003_201,
      cacheHit: false
    })
  });
}

describe("initial learning gate", () => {
  const temporaryRoots: string[] = [];

  afterEach(async () => {
    await Promise.all(
      temporaryRoots.splice(0).map((root) => rm(root, { recursive: true, force: true }))
    );
  });

  it("counts foundation-to-first-data initialization as learning until explicit handoff", () => {
    const engine = new EventEmitter() as EngineSupervisor;
    const jobs = new RuntimeJobManager({} as BrainService, engine);

    expect(jobs.isLearning("new-mind")).toBe(false);
    jobs.beginInitialization("new-mind");
    expect(jobs.isInitializing("new-mind")).toBe(true);
    expect(jobs.isLearning("new-mind")).toBe(true);

    jobs.completeInitialization("new-mind");
    expect(jobs.isInitializing("new-mind")).toBe(false);
    expect(jobs.isLearning("new-mind")).toBe(false);
  });

  it("persists the gate before foundation work and keeps chat and idle blocked after restart", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-initialization-gate-"));
    temporaryRoots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const foundationStarted = deferred();
    const releaseFoundation = deferred();
    const requestStream = vi.fn(async () => {
      foundationStarted.resolve();
      await releaseFoundation.promise;
      return {};
    });
    const engine = {
      requestStream,
      request: vi.fn(),
      tryRequest: vi.fn()
    } as unknown as EngineSupervisor;
    const service = new BrainService(
      repository,
      engine,
      testResourcePlanner(repository.root)
    );
    const creating = service.create(
      {
        config: { ...DEFAULT_CONFIG, name: "Persisted build gate" },
        hardwareTier: "micro"
      },
      () => undefined
    );

    await foundationStarted.promise;
    const [summary] = await repository.list();
    expect(summary).toBeDefined();
    const whileBuilding = await repository.get(summary!.id);
    expect(whileBuilding.readiness).toMatchObject({ state: "initializing" });

    // A fresh repository represents a full main-process restart: no in-memory
    // RuntimeJobManager state survives, but the authoritative gate does.
    const afterRestart = new BrainRepository(join(root, "brains"));
    await afterRestart.initialize();
    expect((await afterRestart.get(summary!.id)).readiness.state).toBe("initializing");

    const actions = {
      idle: vi.fn<() => Promise<IdleCycleResult>>()
    };
    const scheduler = new IdleCognitionScheduler(afterRestart, actions, {
      isLearning: () => false
    });
    await expect(scheduler.tick()).resolves.toBeUndefined();
    expect(actions.idle).not.toHaveBeenCalled();
    await expect(service.chat(summary!.id, "hello")).rejects.toThrow(/initial learning/i);
    await expect(service.idleCycle(summary!.id, 0)).resolves.toMatchObject({
      ran: false,
      reason: "initial-learning"
    });

    releaseFoundation.resolve();
    await expect(creating).resolves.toMatchObject({
      readiness: { state: "initializing" }
    });
  });

  it("atomically unlocks a persisted build once and remains ready after restart", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-initialization-ready-"));
    temporaryRoots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    const created = await repository.create(
      { ...DEFAULT_CONFIG, name: "Ready handoff" },
      { initializing: true }
    );

    const completed = await repository.completeInitialization(created.id);
    expect(completed.readiness.state).toBe("ready");
    expect(completed.readiness.completedAt).toBeTruthy();
    const journalCount = completed.journal?.filter((event) =>
      event.summary.startsWith("Initial neural learning completed")
    ).length;

    const repeated = await repository.completeInitialization(created.id);
    expect(repeated.readiness).toEqual(completed.readiness);
    expect(repeated.journal?.filter((event) =>
      event.summary.startsWith("Initial neural learning completed")
    )).toHaveLength(journalCount ?? 0);

    const afterRestart = new BrainRepository(join(root, "brains"));
    expect((await afterRestart.get(created.id)).readiness).toEqual(completed.readiness);
  });
});
