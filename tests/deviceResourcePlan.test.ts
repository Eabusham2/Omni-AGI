import { describe, expect, it } from "vitest";
import { planDeviceResources, validateDeviceResourceSettings, validateDeviceResourceTopology } from "../src/main/deviceResourcePlan";
import type { DeviceResourceDemand, DeviceResourceSettings, DeviceResourceTopology, ResourceDevice } from "../src/shared/deviceResources";

const GiB = 1024 ** 3;
const selected: DeviceResourceSettings = { version: 1, allocations: [], storageDirectories: [], allowStorageSpill: true, autoIncreaseStorage: false };
const demand: DeviceResourceDemand = { modelBytes: GiB, fixedRuntimeBytes: GiB, contextBytesPerToken: 4096,
  requestedContextTokens: 8192, systemRamBudgetBytes: 8 * GiB, diskReserveBytes: 20 * GiB };
const device = (patch: Partial<ResourceDevice>): ResourceDevice => ({ id: "ram:system", label: "RAM", kind: "ram", backend: "cpu",
  memoryGroupId: "ram:system", executionDevice: "cpu", totalBytes: 16 * GiB, availableBytes: 12 * GiB,
  allocatedBytes: GiB, reservedBytes: GiB, unifiedMemory: false, nativeComputeSupported: true, ...patch });
const topology = (devices: ResourceDevice[] = [device({})]): DeviceResourceTopology => ({ format: "omni-device-resource-topology", version: 1,
  measuredAt: "2026-10-06T00:00:00Z", platform: "fixture", architecture: "arm64", devices, links: [],
  measurementRequested: false, neuralModelConstructed: false, warnings: [] });

describe("measured device resource planning", () => {
  it("counts unified CPU/Metal memory and shared disk containers only once", () => {
    const source = topology([device({}), device({ id: "mps:0", label: "Metal", kind: "gpu", backend: "mps", executionDevice: "mps", unifiedMemory: true }),
      device({ id: "storage:a", label: "A", kind: "storage", backend: "filesystem", executionDevice: undefined, memoryGroupId: "apfs:disk3", totalBytes: 500 * GiB, availableBytes: 100 * GiB, nativeComputeSupported: false }),
      device({ id: "storage:b", label: "B", kind: "storage", backend: "filesystem", executionDevice: undefined, memoryGroupId: "apfs:disk3", totalBytes: 500 * GiB, availableBytes: 100 * GiB, nativeComputeSupported: false })]);
    const plan = planDeviceResources(source, selected, demand);
    expect(plan.physicalMemoryCapacityBytes).toBe(16 * GiB);
    expect(plan.selectedMemoryCapacityBytes).toBe(8 * GiB);
    expect(plan.selectedStorageCapacityBytes).toBe(72 * GiB);
    expect(plan.admittedContextMaximumTokens).toBeUndefined();
    expect(plan.maximumBasis).toBe("weighted-state-and-reserves-deduplicated-physical-pools");
  });

  it("keeps saved capacities independent of current free memory and does not inflate a host ceiling", () => {
    const first = planDeviceResources(topology(), selected, demand);
    const pressured = planDeviceResources(topology([device({ availableBytes: 0 })]), selected, demand);
    expect(pressured.projectedContextMaximumTokens).toBe(first.projectedContextMaximumTokens);
    expect(pressured.budgets[0]!.currentlyAvailableBytes).toBe(0);
    expect(planDeviceResources(topology(), selected, { ...demand, systemRamBudgetBytes: 100 * GiB }).budgets[0]!.ceilingBytes).toBe(16 * GiB);
  });

  it("rejects a model larger than selected capacity even with a zero-token request", () => {
    const result = planDeviceResources(topology(), selected, { ...demand, modelBytes: 100 * GiB, requestedContextTokens: 0 });
    expect(result.fits).toBe(false);
    expect(result.projectedContextMaximumTokens).toBe(0);
    expect(result.blockers.join(" ")).toMatch(/weighted model and fixed runtime/);
  });

  it("keeps slow storage opt-in and rejects unavailable selected devices", () => {
    const source = topology([device({}), device({ id: "storage:a", label: "A", kind: "storage", backend: "filesystem", executionDevice: undefined,
      memoryGroupId: "volume:1", totalBytes: 500 * GiB, availableBytes: 100 * GiB, nativeComputeSupported: false })]);
    const noSpill = planDeviceResources(source, { ...selected, allowStorageSpill: false }, demand);
    expect(noSpill.selectedStorageCapacityBytes).toBe(0);
    const missing = planDeviceResources(source, { ...selected, allocations: [
      { deviceId: "cuda:missing", enabled: true, mode: "manual", percent: 50, autoIncrease: false }
    ] }, demand);
    expect(missing.fits).toBe(false);
    expect(missing.blockers.join(" ")).toMatch(/unavailable/);
  });

  it("orders devices by measured transfer-plus-local rate rather than GPU branding", () => {
    const measurement = (bytesPerSecond: number) => ({ bytesPerSecond, sampleBytes: 4096, runs: 3,
      medianNanoseconds: 1000, measuredAt: "2026-10-06T00:00:00Z", scope: "memory-copy" as const });
    const source = topology([device({ measurement: measurement(1000) }),
      device({ id: "cuda:fast", label: "GPU", kind: "gpu", backend: "cuda", executionDevice: "cuda:0", memoryGroupId: "cuda:fast", measurement: measurement(5000) }),
      device({ id: "cuda:unknown", label: "Other GPU", kind: "gpu", backend: "cuda", executionDevice: "cuda:1", memoryGroupId: "cuda:unknown" })]);
    source.links = [{ from: "ram:system", to: "cuda:fast", peerAccess: null, measurement: { ...measurement(100), scope: "device-transfer" } }];
    expect(planDeviceResources(source, selected, demand).placementOrder).toEqual(["ram:system", "cuda:fast", "cuda:unknown"]);
  });

  it("validates unknown input fields, all required counts, exact arithmetic, and resource identities", () => {
    expect(() => validateDeviceResourceSettings({ ...selected, allocations: [{ deviceId: "ram:system", enabled: true, mode: "manual", percent: 29, autoIncrease: false }] })).toThrow();
    expect(() => validateDeviceResourceSettings({ ...selected, hostCapacityBytes: 999999 })).toThrow();
    expect(() => validateDeviceResourceTopology(topology([device({}), device({})]))).toThrow();
    const partial = { ...demand }; delete (partial as Partial<DeviceResourceDemand>).fixedRuntimeBytes;
    expect(() => planDeviceResources(topology(), selected, partial)).toThrow();
    expect(() => planDeviceResources(topology(), selected, { ...demand, modelBytes: Number.MAX_SAFE_INTEGER, fixedRuntimeBytes: 1 })).toThrow(/exact integer/);
  });
});
