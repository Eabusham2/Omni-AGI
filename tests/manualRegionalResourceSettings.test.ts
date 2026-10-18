import { expect, it } from "vitest";
import { planDeviceResources, validateDeviceResourceSettings, validateRegionalExecutionSettings } from "../src/main/deviceResourcePlan";
import type { DeviceResourceSettings, DeviceResourceTopology } from "../src/shared/deviceResources";

const original: DeviceResourceSettings = { version: 1, allocations: [], allowStorageSpill: false, autoIncreaseStorage: false, storageDirectories: [] };

it("preserves absent settings and requires exact manual routes, device enablement and per-route fallback review", () => {
  expect(validateDeviceResourceSettings(original)).toEqual(original);
  expect(validateDeviceResourceSettings(original)).not.toHaveProperty("regionalExecution");
  const route = { regionId: "decoder.regions.0.decoder.blocks.1", deviceId: "cuda:GPU-A", executionMode: "packed-cpu-tile-fallback" as const,
    acknowledgePackedCpuFallback: true as const };
  const regional = { version: 1 as const, enabled: true, mode: "manual" as const, routes: [route] };
  expect(() => validateDeviceResourceSettings({ ...original, regionalExecution: regional })).toThrow("enabled device");
  const saved = validateDeviceResourceSettings({ ...original, allocations: [{ deviceId: route.deviceId, enabled: true, mode: "manual", percent: 50, autoIncrease: false }], regionalExecution: regional });
  expect(saved.regionalExecution).toEqual(regional);
  expect(saved.regionalExecution).not.toHaveProperty("epoch");
  for (const changed of [{ ...regional, epoch: "imported-active-authority" }, { ...regional, mode: "auto" },
    { ...regional, routes: [{ ...route, acknowledgePackedCpuFallback: undefined }] }, { ...regional, routes: [route, route] }]) {
    expect(() => validateRegionalExecutionSettings(changed)).toThrow();
  }
});

it("keeps unsupported native CUDA false while acknowledging only that explicit route, without crediting real context", () => {
  const topology: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1, measuredAt: "2026-10-15T00:00:00Z",
    platform: "fixture", architecture: "fixture", measurementRequested: false, neuralModelConstructed: false, warnings: [], links: [],
    devices: [{ id: "ram:system", label: "RAM", kind: "ram", backend: "cpu", executionDevice: "cpu", memoryGroupId: "ram:system",
      totalBytes: 16 * 1024 ** 3, availableBytes: 12 * 1024 ** 3, allocatedBytes: 0, reservedBytes: 0, unifiedMemory: false, nativeComputeSupported: true },
      { id: "cuda:GPU-A", label: "CUDA fixture", kind: "gpu", backend: "cuda", executionDevice: "cuda:0", memoryGroupId: "physical:A",
        totalBytes: 8 * 1024 ** 3, availableBytes: 6 * 1024 ** 3, allocatedBytes: 0, reservedBytes: 0, unifiedMemory: false,
        nativeComputeSupported: false, unavailableReason: "native-packed-compute-kernel-unavailable" }] };
  const allocation = { deviceId: "cuda:GPU-A", enabled: true, mode: "manual" as const, percent: 50, autoIncrease: false };
  const demand = { modelBytes: 1024, fixedRuntimeBytes: 1024, contextBytesPerToken: 128, requestedContextTokens: 32,
    systemRamBudgetBytes: 8 * 1024 ** 3, diskReserveBytes: 20 * 1024 ** 3 };
  const without = planDeviceResources(topology, { ...original, allocations: [allocation] }, demand);
  expect(without.executionBlockers.join(" ")).toContain("native compute backend is unavailable");
  const acknowledged = planDeviceResources(topology, { ...original, allocations: [allocation], regionalExecution: {
    version: 1, enabled: true, mode: "manual", routes: [{ regionId: "decoder.blocks.0", deviceId: allocation.deviceId,
      executionMode: "packed-cpu-tile-fallback", acknowledgePackedCpuFallback: true }] } }, demand);
  expect(acknowledged.executionBlockers).toEqual([]);
  expect(acknowledged.admittedContextMaximumTokens).toBeUndefined();
  expect(topology.devices[1]!.nativeComputeSupported).toBe(false);
});
