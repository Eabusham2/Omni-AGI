import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  DISK_BYTES_PER_MEMORY_ITEM,
  GIB,
  MIB,
  MANDATORY_FREE_DISK_BYTES,
  MODEL_CHECKPOINT_HEADROOM_RATIO,
  RAM_BYTES_PER_MEMORY_ITEM,
  ResourcePlanner,
  adaptiveRamReserve,
  applyMemoryLimit,
  parseLinuxMemoryInfo,
  parseMacMemory,
  parseMacMemoryPressure,
  planWorkingMemory,
  type ResourceSnapshot,
  type StorageBenchmark
} from "../src/main/resourcePlanner";

const benchmark: StorageBenchmark = {
  measuredAt: "2026-08-13T00:00:00.000Z",
  sampleBytes: 4 * 1024 * 1024,
  memoryBytesPerSecond: 20 * GIB,
  storageBytesPerSecond: 1 * GIB,
  cacheHit: true
};

const resources: ResourceSnapshot = {
  totalMemoryBytes: 32 * GIB,
  availableMemoryBytes: 24 * GIB,
  diskTotalBytes: 500 * GIB,
  diskFreeBytes: 200 * GIB
};

describe("hardware-safe recurrent/paged memory planning", () => {
  it("parses reclaimable macOS pages instead of trusting the tiny free-page count", () => {
    const parsed = parseMacMemory(
      [
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)",
        "Pages free:                               1000.",
        "Pages active:                           99999.",
        "Pages inactive:                          8000.",
        "Pages speculative:                        500.",
        "Pages purgeable:                         1500."
      ].join("\n"),
      16 * GIB
    );
    expect(parsed?.availableMemoryBytes).toBe((1000 + 8000 + 1500) * 16384);
    expect(parsed?.totalMemoryBytes).toBe(16 * GIB);
  });

  it("uses macOS memory-pressure reclaimability without exceeding physical RAM", () => {
    expect(
      parseMacMemoryPressure(
        "System-wide memory free percentage: 33%",
        16 * GIB
      )
    ).toBe(Math.floor((16 * GIB * 33) / 100));
    expect(parseMacMemoryPressure("unavailable", 16 * GIB)).toBeUndefined();
    expect(
      parseMacMemoryPressure(
        "System-wide memory free percentage: 101%",
        16 * GIB
      )
    ).toBeUndefined();
  });

  it("honors a Linux cgroup limit below physical RAM", () => {
    expect(applyMemoryLimit(
      { totalMemoryBytes: 64 * GIB, availableMemoryBytes: 40 * GIB },
      4 * GIB,
      1.5 * GIB
    )).toEqual({
      totalMemoryBytes: 4 * GIB,
      availableMemoryBytes: 2.5 * GIB
    });
    expect(applyMemoryLimit(
      { totalMemoryBytes: 8 * GIB, availableMemoryBytes: 3 * GIB },
      16 * GIB,
      1 * GIB
    )).toEqual({
      totalMemoryBytes: 8 * GIB,
      availableMemoryBytes: 3 * GIB
    });
  });

  it("scales OS reserve down for constrained devices and up for workstations", () => {
    expect(adaptiveRamReserve(2 * GIB)).toBe(192 * MIB);
    expect(adaptiveRamReserve(4 * GIB)).toBe(384 * MIB);
    expect(adaptiveRamReserve(8 * GIB)).toBe(512 * MIB);
    expect(adaptiveRamReserve(16 * GIB)).toBe(GIB);
    expect(adaptiveRamReserve(32 * GIB)).toBe(2 * GIB);
    expect(adaptiveRamReserve(128 * GIB)).toBe(2 * GIB);
  });

  it("uses Linux MemAvailable rather than MemFree alone", () => {
    const parsed = parseLinuxMemoryInfo([
      "MemTotal:       16384000 kB",
      "MemFree:          100000 kB",
      "MemAvailable:    6200000 kB",
      "Cached:          4000000 kB"
    ].join("\n"));
    expect(parsed?.availableMemoryBytes).toBe(6_200_000 * 1024);
    expect(parsed?.totalMemoryBytes).toBe(16_384_000 * 1024);
  });

  it("runs the real read/write storage probe and then reuses its measured cache", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-resource-probe-"));
    try {
      const planner = new ResourcePlanner(root, {
        readResources: async () => resources
      });
      const first = await planner.plan(
        { mode: "auto" },
        { hardwareTier: "personal", modelBytes: 1.8 * GIB }
      );
      const second = await planner.plan(
        { mode: "auto" },
        { hardwareTier: "personal", modelBytes: 1.8 * GIB }
      );
      expect(first.allowed).toBe(true);
      expect(first.offload.benchmark.storageBytesPerSecond).toBeGreaterThan(0);
      expect(first.offload.benchmark.cacheHit).toBe(false);
      expect(second.offload.benchmark.cacheHit).toBe(true);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("derives the slider maximum only after model, checkpoint, RAM, and 20 GiB disk reserves", () => {
    const modelBytes = 3 * GIB;
    const plan = planWorkingMemory({
      request: { mode: "auto" },
      resources,
      benchmark,
      hardwareTier: "workstation",
      modelBytes
    });

    const ramReserve = adaptiveRamReserve(resources.totalMemoryBytes);
    const safePool = resources.totalMemoryBytes - ramReserve;
    const safeRam = Math.floor(safePool * 0.8);
    const residentItems = Math.floor(
      Math.max(
        0,
        safeRam -
          plan.resources.residentFoundationBytes -
          plan.context.evidence.selectedContextResidentBytes
      ) / RAM_BYTES_PER_MEMORY_ITEM
    );
    const diskAfterReserve =
      resources.diskFreeBytes -
      MANDATORY_FREE_DISK_BYTES -
      Math.ceil(modelBytes * MODEL_CHECKPOINT_HEADROOM_RATIO) -
      plan.offload.modelOffloadScratchBytes;
    const pagedItems = Math.floor(
      Math.max(0, diskAfterReserve) /
        DISK_BYTES_PER_MEMORY_ITEM
    );

    expect(plan.sliderMaximumItems).toBe(residentItems + pagedItems);
    expect(plan.resources.mandatoryFreeDiskBytes).toBe(20 * GIB);
    expect(plan.resources.checkpointHeadroomBytes).toBe(
      Math.ceil(modelBytes * 0.2)
    );
    expect(plan.semantics.denseAttentionClaim).toBe(false);
    expect(plan.semantics.contextFloorTokens).toBe(16_384);
    expect(plan.semantics.contextPagedToStorage).toBe(false);
    expect(plan.allowed).toBe(true);
  });

  it("materially adapts Auto training and RAM residency to fast/high versus slow/low hardware", () => {
    const fast = planWorkingMemory({
      request: { mode: "auto", systemRamMode: "auto" },
      resources: {
        totalMemoryBytes: 64 * GIB,
        availableMemoryBytes: 54 * GIB,
        diskTotalBytes: 2_000 * GIB,
        diskFreeBytes: 1_500 * GIB
      },
      benchmark: {
        ...benchmark,
        storageBytesPerSecond: 2 * GIB
      },
      hardwareTier: "workstation",
      modelBytes: 3 * GIB
    });
    const slow = planWorkingMemory({
      request: { mode: "auto", systemRamMode: "auto" },
      resources: {
        totalMemoryBytes: 8 * GIB,
        availableMemoryBytes: 5 * GIB,
        diskTotalBytes: 500 * GIB,
        diskFreeBytes: 200 * GIB
      },
      benchmark: {
        ...benchmark,
        storageBytesPerSecond: 40 * 1024 * 1024
      },
      hardwareTier: "micro",
      modelBytes: 1.8 * GIB
    });

    expect(fast.training.storageClass).toBe("fast-storage");
    expect(slow.training.storageClass).toBe("slow-storage");
    expect(fast.training.physicalBatchSize).toBeGreaterThan(slow.training.physicalBatchSize);
    expect(fast.training.windowTokens).toBeGreaterThan(slow.training.windowTokens);
    expect(slow.training.minimumScratchIntervalSeconds).toBe(3600);
    expect(fast.resources.systemRamSharePercent).toBe(80);
    expect(slow.resources.systemRamSharePercent).toBeGreaterThanOrEqual(65);
    expect(fast.resources.systemRamBudgetBytes).toBeGreaterThan(
      slow.resources.systemRamBudgetBytes
    );
    expect(slow.training.allSourceBytesVisited).toBe(true);
  });

  it("keeps Auto capacity stable while a strict manual cap can use verified live-layer offload", () => {
    const constrained: ResourceSnapshot = {
      totalMemoryBytes: 16 * GIB,
      availableMemoryBytes: 5 * GIB,
      diskTotalBytes: 500 * GIB,
      diskFreeBytes: 200 * GIB
    };
    const automatic = planWorkingMemory({
      request: { mode: "auto", systemRamMode: "auto" },
      resources: constrained,
      benchmark,
      hardwareTier: "personal",
      modelBytes: 945_000_000
    });
    const manual = planWorkingMemory({
      request: {
        mode: "auto",
        systemRamMode: "manual",
        systemRamSharePercent: 30
      },
      resources: constrained,
      benchmark,
      hardwareTier: "personal",
      modelBytes: 3 * GIB
    });

    expect(automatic.resources.systemRamSharePercent).toBe(65);
    expect(automatic.resources.systemRamSharePercent).toBeLessThanOrEqual(90);
    expect(automatic.allowed).toBe(true);
    expect(manual.resources.systemRamSharePercent).toBe(30);
    expect(manual.allowed).toBe(true);
    expect(manual.offload.required).toBe(true);
    expect(manual.offload.modelSpillBytes).toBeGreaterThan(0);
    expect(manual.offload.modelOffloadScratchBytes).toBeGreaterThan(0);
    expect(manual.warnings.join(" ")).toMatch(/storage offload is required/i);
  });

  it("derives persistent capacity from total RAM and reports current pressure without shrinking or blocking", () => {
    const base = {
      totalMemoryBytes: 16 * GIB,
      diskTotalBytes: 500 * GIB,
      diskFreeBytes: 200 * GIB
    };
    const free = planWorkingMemory({
      request: { mode: "auto", systemRamMode: "auto" },
      resources: { ...base, availableMemoryBytes: 14 * GIB },
      benchmark,
      hardwareTier: "personal",
      modelBytes: 945_000_000
    });
    const pressured = planWorkingMemory({
      request: { mode: "auto", systemRamMode: "auto" },
      resources: { ...base, availableMemoryBytes: 2 * GIB },
      benchmark,
      hardwareTier: "personal",
      modelBytes: 945_000_000
    });

    expect(pressured.allowed).toBe(true);
    expect(pressured.context).toMatchObject({
      selectedTokens: free.context.selectedTokens,
      autoTokens: free.context.autoTokens,
      maximumTokens: free.context.maximumTokens
    });
    expect(pressured.resources.systemRamBudgetBytes).toBe(
      free.resources.systemRamBudgetBytes
    );
    expect(pressured.resources.currentOmniShortfallBytes).toBeGreaterThan(0);
    expect(pressured.warnings.join(" ")).toMatch(/close memory-heavy applications/i);
    expect(pressured.semantics.capacityPersistsAcrossPressure).toBe(true);
  });

  it("sizes one future-proof shared pool from selected training bytes and validates a manual pool", () => {
    const automatic = planWorkingMemory({
      request: {
        mode: "auto",
        storagePoolMode: "auto",
        trainingSourceBytes: 67 * GIB
      },
      resources,
      benchmark,
      hardwareTier: "personal",
      modelBytes: 945_000_000
    });
    const tooSmall = planWorkingMemory({
      request: {
        mode: "auto",
        storagePoolMode: "manual",
        storagePoolBytes: String(GIB),
        trainingSourceBytes: 67 * GIB
      },
      resources,
      benchmark,
      hardwareTier: "personal",
      modelBytes: 945_000_000
    });

    expect(automatic.allowed).toBe(true);
    expect(automatic.resources.trainingSourceBytes).toBe(67 * GIB);
    expect(automatic.resources.trainingScratchBytes).toBeGreaterThan(6 * GIB);
    expect(automatic.resources.sharedStoragePoolBytes).toBe(
      automatic.resources.requiredStoragePoolBytes
    );
    expect(automatic.semantics.storagePoolShareRule).toBe("largest-brain-not-sum");
    expect(tooSmall.allowed).toBe(false);
    expect(tooSmall.blockers.join(" ")).toMatch(/shared storage pool must be at least/i);
  });

  it("preserves an existing Auto storage reservation while it remains physically safe", async () => {
    const existingPool = 18 * GIB;
    const planner = new ResourcePlanner("/tmp/omni-resource-test", {
      readResources: async () => resources,
      benchmark: async () => benchmark
    });
    const plan = await planner.plan(
      { mode: "auto", storagePoolMode: "auto" },
      {
        hardwareTier: "personal",
        modelBytes: 945_000_000,
        config: {
          ...({} as import("../src/shared/types").BrainConfig),
          storagePoolMode: "auto",
          storagePoolBytes: existingPool
        }
      }
    );

    expect(plan.allowed).toBe(true);
    expect(plan.resources.sharedStoragePoolBytes).toBe(existingPool);
    expect(plan.resources.sharedStoragePoolBytes).toBeGreaterThan(
      plan.resources.requiredStoragePoolBytes
    );
    expect(plan.warnings.join(" ")).toMatch(/existing shared storage reservation/i);
  });

  it("right-sizes a stale Auto reservation instead of blocking a fitting brain", async () => {
    const stalePool = 18 * GIB;
    const constrainedResources: ResourceSnapshot = {
      ...resources,
      diskFreeBytes: 35 * GIB
    };
    const planner = new ResourcePlanner("/tmp/omni-resource-test", {
      readResources: async () => constrainedResources,
      benchmark: async () => benchmark
    });
    const plan = await planner.plan(
      { mode: "auto", storagePoolMode: "auto" },
      {
        hardwareTier: "personal",
        modelBytes: 945_000_000,
        config: {
          ...({} as import("../src/shared/types").BrainConfig),
          storagePoolMode: "auto",
          storagePoolBytes: stalePool
        }
      }
    );

    expect(plan.allowed).toBe(true);
    expect(plan.resources.mandatoryFreeDiskBytes).toBe(20 * GIB);
    expect(plan.resources.maximumStoragePoolBytes).toBe(15 * GIB);
    expect(plan.resources.sharedStoragePoolBytes).toBe(
      plan.resources.requiredStoragePoolBytes
    );
    expect(plan.resources.sharedStoragePoolBytes).toBeLessThan(stalePool);
    expect(plan.warnings.join(" ")).toMatch(/safely right-sized/i);
    expect(plan.warnings.join(" ")).not.toMatch(/reservation is retained/i);
  });

  it("allows custom safe RAM shares while warning about low, slower, near-boundary, and slow-storage spill choices", () => {
    const slowBenchmark: StorageBenchmark = {
      ...benchmark,
      storageBytesPerSecond: 35 * 1024 * 1024
    };
    const low = planWorkingMemory({
      request: {
        mode: "manual",
        requestedItems: "2000000",
        systemRamMode: "manual",
        systemRamSharePercent: 30
      },
      resources: {
        totalMemoryBytes: 8 * GIB,
        availableMemoryBytes: 7 * GIB,
        diskTotalBytes: 500 * GIB,
        diskFreeBytes: 200 * GIB
      },
      benchmark: slowBenchmark,
      hardwareTier: "micro",
      modelBytes: 800 * MIB
    });
    const high = planWorkingMemory({
      request: {
        mode: "auto",
        systemRamMode: "manual",
        systemRamSharePercent: 92
      },
      resources,
      benchmark,
      hardwareTier: "personal",
      modelBytes: 1.8 * GIB
    });

    expect(low.allowed).toBe(true);
    expect(low.warnings.join(" ")).toMatch(/very low/i);
    expect(low.warnings.join(" ")).toMatch(/heavy spill.*slow storage/i);
    expect(high.allowed).toBe(true);
    expect(high.warnings.join(" ")).toMatch(/near the safe-pool boundary/i);
  });

  it("accepts no manual product cap but rejects a physically impossible decimal without parsing it", () => {
    const impossible = "9".repeat(120);
    const plan = planWorkingMemory({
      request: { mode: "manual", requestedItems: impossible },
      resources,
      benchmark,
      hardwareTier: "personal",
      modelBytes: 1.8 * GIB
    });
    expect(plan.allowed).toBe(false);
    expect(plan.blockers.join(" ")).toMatch(/physical RAM and storage/i);
    expect(plan.sliderMaximumItems).toBeGreaterThan(0);
  });

  it("requires and visibly prices storage offload when model plus full memory cannot stay in RAM", () => {
    const constrained: ResourceSnapshot = {
      totalMemoryBytes: 8 * GIB,
      availableMemoryBytes: 5 * GIB,
      diskTotalBytes: 500 * GIB,
      diskFreeBytes: 100 * GIB
    };
    const plan = planWorkingMemory({
      request: { mode: "manual", requestedItems: "1000000" },
      resources: constrained,
      benchmark,
      hardwareTier: "micro",
      modelBytes: 1 * GIB
    });
    expect(plan.allowed).toBe(true);
    expect(plan.offload.required).toBe(true);
    expect(plan.resources.configuredMemorySpillBytes).toBeGreaterThan(0);
    expect(plan.offload.estimatedSlowdownPercent).toBeGreaterThan(0);
    expect(plan.warnings.join(" ")).toMatch(/storage offload is required/i);
  });

  it("locks Build/start before crossing the model headroom or free-space boundary", () => {
    const plan = planWorkingMemory({
      request: { mode: "auto" },
      resources: {
        ...resources,
        diskFreeBytes: 20.1 * GIB
      },
      benchmark,
      hardwareTier: "personal",
      modelBytes: 1.8 * GIB
    });
    expect(plan.allowed).toBe(false);
    expect(plan.blockers.join(" ")).toMatch(
      /checkpoint headroom.*device safety reserve/i
    );
  });

  it("places Auto in the suitable band and Extended in the higher warning band", () => {
    const automatic = planWorkingMemory({
      request: { mode: "auto" },
      resources,
      benchmark,
      hardwareTier: "gpu",
      modelBytes: 3 * GIB
    });
    const extended = planWorkingMemory({
      request: { mode: "extended" },
      resources,
      benchmark,
      hardwareTier: "gpu",
      modelBytes: 3 * GIB
    });
    expect(automatic.selectedItems).toBe(automatic.suitableRange.autoItems);
    expect(extended.selectedItems).toBe(automatic.suitableRange.extendedItems);
    expect(extended.selectedItems).toBeGreaterThan(automatic.selectedItems);
  });

  it("derives active context from live RAM, model cost, accelerator, storage, and the model limit", () => {
    const constrained = planWorkingMemory({
      request: { mode: "auto", acceleratorAvailable: false },
      resources: {
        totalMemoryBytes: 16 * GIB,
        availableMemoryBytes: 10 * GIB,
        diskTotalBytes: 500 * GIB,
        diskFreeBytes: 200 * GIB
      },
      benchmark: {
        ...benchmark,
        storageBytesPerSecond: 45 * MIB
      },
      hardwareTier: "personal",
      modelBytes: 945_000_000
    });
    const capable = planWorkingMemory({
      request: { mode: "auto", acceleratorAvailable: true },
      resources: {
        totalMemoryBytes: 32 * GIB,
        availableMemoryBytes: 26 * GIB,
        diskTotalBytes: 1_000 * GIB,
        diskFreeBytes: 700 * GIB
      },
      benchmark: {
        ...benchmark,
        storageBytesPerSecond: 1.5 * GIB
      },
      hardwareTier: "personal",
      modelBytes: 945_000_000,
      modelContextLimitTokens: 32_768
    });

    expect(constrained.allowed).toBe(true);
    expect(capable.allowed).toBe(true);
    expect(constrained.context.floorTokens).toBe(4_096);
    expect(capable.context.autoTokens).toBeGreaterThan(
      constrained.context.autoTokens
    );
    expect(capable.context.autoTokens).toBeLessThanOrEqual(32_768);
    expect(capable.context.maximumTokens).toBe(32_768);
    expect(capable.context.selectedTokens).toBe(capable.context.autoTokens);
    expect(capable.context.evidence).toMatchObject({
      source: "live-device-model-measurement",
      acceleratorAvailable: true,
      storageClass: "fast-storage",
      modelContextLimitTokens: 32_768
    });
    expect(
      capable.context.evidence.estimatedKvActivationBytesPerToken
    ).toBeGreaterThan(0);
    expect(capable.semantics.contextPagedToStorage).toBe(false);
  });

  it("keeps manual context resident, enforces the new-build floor, and permits legacy recorded windows", () => {
    const input = {
      resources,
      benchmark,
      hardwareTier: "personal" as const,
      modelBytes: 945_000_000
    };
    const below = planWorkingMemory({
      ...input,
      request: { mode: "manual", requestedContextTokens: "2048" }
    });
    const exact = planWorkingMemory({
      ...input,
      request: { mode: "manual", requestedContextTokens: "12288" }
    });
    const impossible = planWorkingMemory({
      ...input,
      request: { mode: "manual", requestedContextTokens: "999999999" }
    });
    const legacy = planWorkingMemory({
      ...input,
      request: { mode: "manual", requestedContextTokens: "1024" },
      enforceContextFloor: false
    });

    expect(below.allowed).toBe(false);
    expect(below.blockers.join(" ")).toMatch(/below the .* baseline/i);
    expect(exact.allowed).toBe(true);
    expect(exact.context.selectedTokens).toBe(12_288);
    expect(exact.context.evidence.selectedContextResidentBytes).toBe(
      12_288 * exact.context.evidence.estimatedKvActivationBytesPerToken * 2
    );
    expect(impossible.allowed).toBe(false);
    expect(impossible.blockers.join(" ")).toMatch(/cannot stay resident/i);
    expect(legacy.allowed).toBe(true);
    expect(legacy.warnings.join(" ")).toMatch(/checkpoint compatibility/i);
  });

  it("allows low-RAM operation when one live layer plus the exact context fits and disk scratch is safe", () => {
    const plan = planWorkingMemory({
      request: {
        mode: "manual",
        requestedContextTokens: "2048",
        systemRamMode: "manual",
        systemRamSharePercent: 30
      },
      resources: {
        totalMemoryBytes: 8 * GIB,
        availableMemoryBytes: 6 * GIB,
        diskTotalBytes: 500 * GIB,
        diskFreeBytes: 100 * GIB
      },
      benchmark,
      hardwareTier: "micro",
      modelBytes: 2 * GIB
    });

    expect(plan.allowed).toBe(true);
    expect(plan.offload.required).toBe(true);
    expect(plan.offload.modelSpillBytes).toBeGreaterThan(0);
    expect(plan.offload.modelOffloadScratchBytes).toBeGreaterThan(0);
    expect(
      plan.resources.residentFoundationBytes +
        plan.context.evidence.selectedContextResidentBytes
    ).toBeLessThanOrEqual(plan.resources.systemRamBudgetBytes);
    expect(plan.context.selectedTokens).toBe(2_048);
  });

  it("blocks when no resident tier can hold one complete layer plus active context", () => {
    const plan = planWorkingMemory({
      request: {
        mode: "manual",
        requestedContextTokens: "256",
        systemRamMode: "manual",
        systemRamSharePercent: 30
      },
      resources: {
        totalMemoryBytes: 2 * GIB,
        availableMemoryBytes: 900 * MIB,
        diskTotalBytes: 500 * GIB,
        diskFreeBytes: 100 * GIB
      },
      benchmark,
      hardwareTier: "micro",
      modelBytes: 2 * GIB,
      enforceContextFloor: false
    });

    expect(plan.allowed).toBe(false);
    expect(plan.blockers.join(" ")).toMatch(/one complete neural layer/i);
  });

  it("blocks layer offload when its safe-tensor scratch crosses the adaptive reserve", () => {
    const plan = planWorkingMemory({
      request: {
        mode: "manual",
        requestedContextTokens: "2048",
        systemRamMode: "manual",
        systemRamSharePercent: 30
      },
      resources: {
        totalMemoryBytes: 8 * GIB,
        availableMemoryBytes: 6 * GIB,
        diskTotalBytes: 30 * GIB,
        diskFreeBytes: 7 * GIB
      },
      benchmark,
      hardwareTier: "micro",
      modelBytes: 2 * GIB
    });

    expect(plan.offload.modelOffloadScratchBytes).toBeGreaterThan(0);
    expect(plan.allowed).toBe(false);
    expect(plan.blockers.join(" ")).toMatch(
      /mandatory free-space reserve|device-adaptive free-space boundary/i
    );
  });

  it("admits an existing runtime without reserving future growth a second time", () => {
    const common = {
      request: {
        mode: "manual" as const,
        requestedItems: "65536",
        requestedContextTokens: "14848",
        systemRamMode: "auto" as const,
        storagePoolMode: "auto" as const
      },
      resources: {
        totalMemoryBytes: 16 * GIB,
        availableMemoryBytes: 12 * GIB,
        diskTotalBytes: 500 * GIB,
        // Exact live incident shape: only 676 MiB remains above 20 GiB.
        diskFreeBytes: 20 * GIB + 676 * MIB
      },
      benchmark,
      hardwareTier: "gpu" as const,
      modelBytes: 76 * MIB,
      minimumStoragePoolBytes: 2_884_068_301,
      enforceContextFloor: false
    };
    const capacity = planWorkingMemory(common);
    const runtime = planWorkingMemory({
      ...common,
      admissionScope: "existing-runtime"
    });

    expect(capacity.allowed).toBe(false);
    expect(capacity.blockers.join(" ")).toMatch(
      /shared storage pool|working memory.*storage/i
    );
    expect(runtime.allowed).toBe(true);
    expect(runtime.resources.diskFreeBytes).toBe(common.resources.diskFreeBytes);
    expect(runtime.resources.mandatoryFreeDiskBytes).toBe(20 * GIB);
    expect(runtime.resources.requiredStoragePoolBytes).toBeGreaterThan(2 * GIB);
    expect(runtime.resources.maximumStoragePoolBytes).toBe(676 * MIB);
    expect(runtime.diskSpace.futureGrowthBytes).toBe(0);
    expect(runtime.diskSpace.operationWriteBytes).toBe(0);
    expect(runtime.diskSpace.checkpointBytes).toBeGreaterThan(0);
    expect(runtime.warnings.join(" ")).toMatch(/actual checkpoint.*reserve-gated/i);
  });
});
