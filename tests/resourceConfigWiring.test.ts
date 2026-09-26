import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { GIB, MIB, ResourcePlanner } from "../src/main/resourcePlanner";
import { DEFAULT_CONFIG } from "../src/shared/types";

describe("resource configuration wiring", () => {
  let temporaryRoot: string;
  let repository: BrainRepository;

  beforeEach(async () => {
    temporaryRoot = await mkdtemp(join(tmpdir(), "omni-resource-config-"));
    repository = new BrainRepository(join(temporaryRoot, "brains"));
    await repository.initialize();
  });

  afterEach(async () => {
    await rm(temporaryRoot, { recursive: true, force: true });
  });

  it("persists RAM mode, manual share, and measured storage through duplicate and restart", async () => {
    const source = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Resource source",
      systemRamMode: "manual",
      systemRamSharePercent: 47,
      storagePoolMode: "manual",
      storagePoolBytes: 18 * GIB,
      storageBytesPerSecond: 734_003_200.4
    });

    const duplicate = await repository.duplicate(source.id, "Resource duplicate");
    const bundle = join(temporaryRoot, "resource-source.omni");
    await repository.exportBundle(source.id, bundle);
    await expect(repository.importBundle(bundle)).rejects.toThrow(/locally initialized OmniCortex origin/i);

    for (const brain of [source, duplicate]) {
      expect(brain.config).toMatchObject({
        systemRamMode: "manual",
        systemRamSharePercent: 47,
        storagePoolMode: "manual",
        storagePoolBytes: 18 * GIB,
        storageBytesPerSecond: 734_003_200
      });
    }

    const restarted = new BrainRepository(repository.root);
    await restarted.initialize();
    await expect(restarted.get(duplicate.id)).resolves.toMatchObject({
      config: {
        systemRamMode: "manual",
        systemRamSharePercent: 47,
        storagePoolMode: "manual",
        storagePoolBytes: 18 * GIB,
        storageBytesPerSecond: 734_003_200
      }
    });

    const automatic = await restarted.updateConfig(source.id, {
      ...source.config,
      systemRamMode: "auto",
      // A stale slider value must never become an Auto cap.
      systemRamSharePercent: 99
    });
    expect(automatic.config.systemRamSharePercent).toBe(0);
  });

  it("keeps GPU Auto semantic while persisting a rounded measured storage rate", async () => {
    const measuredStorageRate = 734_003_200.6;
    const roundedStorageRate = 734_003_201;
    const planner = new ResourcePlanner(repository.root, {
      readResources: async () => ({
        totalMemoryBytes: 16 * GIB,
        availableMemoryBytes: 13 * GIB,
        diskTotalBytes: 500 * GIB,
        diskFreeBytes: 240 * GIB
      }),
      benchmark: async () => ({
        measuredAt: "2026-08-30T00:00:00.000Z",
        sampleBytes: 16 * 1024 * 1024,
        memoryBytesPerSecond: 12 * GIB,
        storageBytesPerSecond: measuredStorageRate,
        cacheHit: false
      })
    });
    const plan = await planner.plan(
      { mode: "auto", systemRamMode: "auto" },
      { hardwareTier: "gpu" }
    );
    expect(plan.resources.systemRamMode).toBe("auto");
    expect(plan.resources.autoSystemRamSharePercent).toBe(75);
    expect(plan.offload.benchmark.storageBytesPerSecond).toBe(
      roundedStorageRate
    );

    const request = vi.fn(async (_method: string) => ({}));
    const service = new BrainService(
      repository,
      {
        request,
        tryRequest: vi.fn(async () => undefined)
      } as unknown as EngineSupervisor,
      planner
    );
    const built = await service.create({
      hardwareTier: "gpu",
      config: {
        ...DEFAULT_CONFIG,
        name: "GPU Auto resource mind",
        systemRamMode: "auto",
        // Auto remains a mode/sentinel; the resolved 75% stays measured output.
        systemRamSharePercent: 0
      }
    });

    expect(built.config).toMatchObject({
      systemRamMode: "auto",
      systemRamSharePercent: 0,
      storageBytesPerSecond: roundedStorageRate
    });
    await expect(repository.get(built.id)).resolves.toMatchObject({
      config: {
        systemRamMode: "auto",
        systemRamSharePercent: 0,
        storageBytesPerSecond: roundedStorageRate
      }
    });
    expect(request).toHaveBeenCalledWith(
      "create",
      expect.objectContaining({
        hardwareTier: "gpu",
        config: expect.objectContaining({
          systemRamMode: "auto",
          systemRamSharePercent: 0,
          storageBytesPerSecond: roundedStorageRate
        })
      }),
      300_000
    );
  });

  it("preflights settings before commit and forwards the measured plan to the worker", async () => {
    let diskFreeBytes = 240 * GIB;
    let storageBytesPerSecond = 860_000_000;
    let benchmarkCalls = 0;
    const planner = new ResourcePlanner(repository.root, {
      readResources: async () => ({
        totalMemoryBytes: 16 * GIB,
        availableMemoryBytes: 12 * GIB,
        diskTotalBytes: 500 * GIB,
        diskFreeBytes
      }),
      benchmark: async () => {
        benchmarkCalls += 1;
        return {
          measuredAt: new Date(benchmarkCalls * 1_000).toISOString(),
          sampleBytes: 16 * 1024 * 1024,
          memoryBytesPerSecond: 12 * GIB,
          storageBytesPerSecond,
          cacheHit: false
        };
      }
    });
    const request = vi.fn(async () => ({}));
    const tryRequest = vi.fn(async () => undefined);
    const service = new BrainService(
      repository,
      { request, tryRequest } as unknown as EngineSupervisor,
      planner
    );
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Preflighted resource mind"
    });

    const manual = await service.updateConfig(brain.id, {
      ...brain.config,
      systemRamMode: "manual",
      systemRamSharePercent: 43,
      // Renderer values are advisory; the main-process probe is authoritative.
      storageBytesPerSecond: 1
    });

    expect(manual.config).toMatchObject({
      systemRamMode: "manual",
      systemRamSharePercent: 43,
      storageBytesPerSecond: 860_000_000
    });
    expect(request).toHaveBeenCalledWith(
      "update_config",
      expect.objectContaining({
        brainId: brain.id,
        config: expect.objectContaining({
          systemRamMode: "manual",
          systemRamSharePercent: 43,
          storageBytesPerSecond: 860_000_000
        })
      }),
      300_000
    );

    request.mockClear();
    const automatic = await service.updateConfig(brain.id, {
      ...manual.config,
      systemRamMode: "auto",
      systemRamSharePercent: 88
    });
    expect(automatic.config.systemRamSharePercent).toBe(0);
    expect(request).toHaveBeenCalledWith(
      "update_config",
      expect.objectContaining({
        config: expect.objectContaining({
          systemRamMode: "auto",
          systemRamSharePercent: 0
        })
      }),
      300_000
    );

    request.mockClear();
    await expect(
      service.updateConfig(brain.id, {
        ...automatic.config,
        systemRamMode: "manual",
        systemRamSharePercent: 29
      })
    ).rejects.toThrow(/between 30% and 100%/i);
    expect(request).not.toHaveBeenCalled();
    await expect(repository.get(brain.id)).resolves.toMatchObject({
      config: { systemRamMode: "auto", systemRamSharePercent: 0 }
    });

    diskFreeBytes = 10 * GIB;
    await expect(service.preflightStart(brain.id)).rejects.toThrow(/cannot start safely/i);
    expect(benchmarkCalls).toBeGreaterThanOrEqual(3);

    diskFreeBytes = 240 * GIB;
    storageBytesPerSecond = 42 * 1024 * 1024;
    const startupPlan = await service.preflightStart(brain.id);
    expect(startupPlan?.offload.benchmark.storageBytesPerSecond).toBe(
      storageBytesPerSecond
    );
    expect(startupPlan?.resources.systemRamMode).toBe("auto");
  });

  it("uses packed resident-size estimates for a constrained micro preflight", async () => {
    const planner = new ResourcePlanner(repository.root, {
      readResources: async () => ({
        totalMemoryBytes: 4 * GIB,
        availableMemoryBytes: 2.5 * GIB,
        diskTotalBytes: 128 * GIB,
        diskFreeBytes: 80 * GIB
      }),
      benchmark: async () => ({
        measuredAt: "2026-08-22T00:00:00.000Z",
        sampleBytes: 16 * 1024 * 1024,
        memoryBytesPerSecond: 8 * GIB,
        storageBytesPerSecond: 700_000_000,
        cacheHit: false
      })
    });

    const plan = await planner.plan(
      { mode: "auto", systemRamMode: "auto" },
      { hardwareTier: "micro" }
    );

    expect(plan.resources.modelWorkingSetBytes).toBeLessThan(1.2 * GIB);
    expect(plan.allowed).toBe(true);
  });

  it("starts a fitting persisted brain when only its Auto pool high-water mark is stale", async () => {
    const planner = new ResourcePlanner(repository.root, {
      readResources: async () => ({
        totalMemoryBytes: 32 * GIB,
        availableMemoryBytes: 24 * GIB,
        diskTotalBytes: 500 * GIB,
        diskFreeBytes: 35 * GIB
      }),
      benchmark: async () => ({
        measuredAt: "2026-08-30T00:00:00.000Z",
        sampleBytes: 16 * 1024 * 1024,
        memoryBytesPerSecond: 20 * GIB,
        storageBytesPerSecond: 800_000_000,
        cacheHit: false
      })
    });
    const request = vi.fn(async () => ({}));
    const tryRequest = vi.fn(async () => undefined);
    const service = new BrainService(
      repository,
      { request, tryRequest } as unknown as EngineSupervisor,
      planner
    );
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Stale Auto storage mind",
      storagePoolMode: "auto",
      storagePoolBytes: 18 * GIB
    });

    const plan = await service.preflightStart(brain.id);

    expect(plan?.allowed).toBe(true);
    expect(plan?.resources.mandatoryFreeDiskBytes).toBe(20 * GIB);
    expect(plan?.resources.maximumStoragePoolBytes).toBe(15 * GIB);
    expect(plan?.resources.sharedStoragePoolBytes).toBe(18 * GIB);
    expect(plan?.resources.requiredStoragePoolBytes).toBeLessThan(18 * GIB);
    expect(plan?.warnings.join(" ")).toMatch(/did not reserve future growth/i);
    expect(request).not.toHaveBeenCalled();
    await expect(repository.get(brain.id)).resolves.toMatchObject({
      config: { storagePoolMode: "auto", storagePoolBytes: 18 * GIB }
    });
  });

  it("cold-loads and steers an existing brain without reserving future growth again", async () => {
    const planner = new ResourcePlanner(repository.root, {
      readResources: async () => ({
        totalMemoryBytes: 16 * GIB,
        availableMemoryBytes: 12 * GIB,
        diskTotalBytes: 500 * GIB,
        diskFreeBytes: 20 * GIB + 676 * MIB
      }),
      benchmark: async () => ({
        measuredAt: "2026-09-19T00:00:00.000Z",
        sampleBytes: 16 * MIB,
        memoryBytesPerSecond: 20 * GIB,
        storageBytesPerSecond: 800_000_000,
        cacheHit: true
      })
    });
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Existing low-headroom mind",
      workingMemorySlots: 65_536,
      contextWindowTokens: 14_848,
      storagePoolMode: "auto",
      storagePoolBytes: 2_884_068_301
    });
    const engineDirectory = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engineDirectory, { recursive: true });
    await Promise.all([
      writeFile(join(engineDirectory, "brain.json"), JSON.stringify({
        brain_id: brain.id,
        config: {
          hardware_tier: "gpu",
          d_model: 256,
          n_layers: 8
        }
      })),
      writeFile(join(engineDirectory, "core.safetensors"), Buffer.alloc(MIB)),
      writeFile(join(engineDirectory, "plasticity.safetensors"), Buffer.alloc(MIB))
    ]);
    let turn = 0;
    const request = vi.fn(async (_method: string) => ({}));
    const requestStream = vi.fn(async (_method: string, params: Record<string, unknown>) => ({
      text: `reply ${++turn}: ${String(params.input)}`,
      trace: { id: `trace-${turn}` },
      metrics: {}
    }));
    const service = new BrainService(
      repository,
      {
        claimForeground: vi.fn(async () => undefined),
        request,
        requestStream
      } as unknown as EngineSupervisor,
      planner
    );

    const admission = await service.preflightStart(brain.id);
    expect(admission?.allowed).toBe(true);
    expect(admission?.resources.diskFreeBytes).toBe(20 * GIB + 676 * MIB);
    expect(admission?.resources.maximumStoragePoolBytes).toBe(676 * MIB);
    expect(admission?.resources.requiredStoragePoolBytes).toBeGreaterThan(2 * GIB);
    expect(admission?.diskSpace.futureGrowthBytes).toBe(0);
    expect(admission?.diskSpace.checkpointBytes).toBeGreaterThan(0);

    await expect(service.chat(brain.id, "first turn")).resolves.toMatchObject({
      brainMessage: { content: "reply 1: first turn" }
    });
    await expect(service.chat(brain.id, "steer toward saved evidence")).resolves.toMatchObject({
      brainMessage: { content: "reply 2: steer toward saved evidence" }
    });
    expect(request.mock.calls.filter(([method]) => method === "load")).toHaveLength(2);
    expect(requestStream).toHaveBeenCalledTimes(2);
  });
});
