/** Config/arithmetic fixtures; no native worker or model construction. */
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { ResourcePlanner, GIB } from "../src/main/resourcePlanner";
import { DEFAULT_CONFIG } from "../src/shared/types";
import type { DeviceResourceSettings, DeviceResourceTopology } from "../src/shared/deviceResources";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
const settings: DeviceResourceSettings = { version: 1, allocations: [
  { deviceId: "ram:system", enabled: true, mode: "manual", percent: 47, autoIncrease: false }
], allowStorageSpill: false, autoIncreaseStorage: false, storageDirectories: [] };

it("persists device selections and their RAM ceiling across repository reload", async () => {
  const root = await mkdtemp(join(tmpdir(), "omni-device-config-")); roots.push(root);
  const repository = new BrainRepository(root); await repository.initialize();
  const saved = await repository.create({ ...DEFAULT_CONFIG, deviceResources: settings });
  expect(saved.config.systemRamMode).toBe("manual");
  expect(saved.config.systemRamSharePercent).toBe(47);
  const restarted = new BrainRepository(root); await restarted.initialize();
  expect((await restarted.get(saved.id)).config.deviceResources).toEqual(settings);
});

it("does not credit projected multiple-GPU capacity beyond the actual current pager", async () => {
  const source: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1, measuredAt: "2026-10-06T00:00:00Z",
    platform: "fixture", architecture: "fixture", links: [], measurementRequested: false, neuralModelConstructed: false, warnings: [], devices: [
      { id: "ram:system", label: "RAM", kind: "ram", backend: "cpu", executionDevice: "cpu", memoryGroupId: "ram:system",
        totalBytes: 32 * GIB, availableBytes: 24 * GIB, allocatedBytes: GIB, reservedBytes: null, unifiedMemory: false, nativeComputeSupported: true },
      ...[0, 1].map((index) => ({ id: `cuda:${index}`, label: `GPU ${index}`, kind: "gpu" as const, backend: "cuda" as const,
        executionDevice: `cuda:${index}`, memoryGroupId: `cuda:${index}`, totalBytes: 80 * GIB, availableBytes: 72 * GIB,
        allocatedBytes: 0, reservedBytes: 0, unifiedMemory: false, nativeComputeSupported: true }))
    ] };
  const planner = new ResourcePlanner("/fixture", {
    readResources: async () => ({ totalMemoryBytes: 32 * GIB, availableMemoryBytes: 24 * GIB, diskTotalBytes: 500 * GIB, diskFreeBytes: 300 * GIB }),
    benchmark: async () => ({ measuredAt: "2026-10-06T00:00:00Z", sampleBytes: 4096, memoryBytesPerSecond: 20 * GIB, storageBytesPerSecond: GIB, cacheHit: true }),
    deviceTopology: async () => source
  });
  const result = await planner.plan({ mode: "auto", deviceResources: settings }, { hardwareTier: "micro", config: DEFAULT_CONFIG });
  expect(result.deviceResources).toBeDefined();
  expect(result.deviceResources!.selectedMemoryCapacityBytes).toBeGreaterThan(32 * GIB);
  expect(result.deviceResources!.admittedContextMaximumTokens).toBeLessThanOrEqual(result.context.maximumTokens);
  expect(result.deviceResources!.admittedContextMaximumTokens).toBe(result.context.maximumTokens);
});
