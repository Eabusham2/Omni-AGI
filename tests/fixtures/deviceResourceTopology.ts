import type { DeviceResourcePlan, DeviceResourceTopology, ResourceDevice } from "../../src/shared/deviceResources";
const GIB = 1024 ** 3;
const measuredAt = "2026-10-07T12:00:00.000Z";
const device = (value: Partial<ResourceDevice> & Pick<ResourceDevice, "id" | "label" | "kind" | "backend" | "memoryGroupId">): ResourceDevice => ({
  totalBytes: null, availableBytes: null, allocatedBytes: null, reservedBytes: null, unifiedMemory: false, nativeComputeSupported: false, ...value
});
export function deviceTopologyFixture(extra = false): DeviceResourceTopology {
  return { format: "omni-device-resource-topology", version: 1, measuredAt, platform: "fixture", architecture: "test",
    measurementRequested: false, neuralModelConstructed: false, warnings: [], devices: [
      device({ id: "ram:system", label: "System RAM", kind: "ram", backend: "cpu", memoryGroupId: "ram:system",
        totalBytes: 16 * GIB, availableBytes: 12 * GIB, allocatedBytes: GIB, reservedBytes: 2 * GIB, nativeComputeSupported: true,
        measurement: { bytesPerSecond: 40 * GIB, sampleBytes: 16 * 1024 ** 2, runs: 3, medianNanoseconds: 390625, measuredAt, scope: "memory-copy" } }),
      device({ id: "gpu:mps", label: "Unified GPU", kind: "gpu", backend: "mps", executionDevice: "mps", memoryGroupId: "ram:system",
        totalBytes: 16 * GIB, availableBytes: 12 * GIB, allocatedBytes: 0, reservedBytes: 0, unifiedMemory: true, nativeComputeSupported: true }),
      device({ id: "gpu:cuda:0", label: "Discrete GPU", kind: "gpu", backend: "cuda", executionDevice: "cuda:0", memoryGroupId: "gpu:discrete",
        totalBytes: 8 * GIB, availableBytes: 6 * GIB, allocatedBytes: GIB, reservedBytes: GIB, nativeComputeSupported: true }),
      device({ id: "npu:unavailable", label: "Unavailable NPU", kind: "npu", backend: "npu", memoryGroupId: "npu:unknown", unavailableReason: "Native packed kernels are not available." }),
      device({ id: "storage:primary", label: "Primary drive", kind: "storage", backend: "filesystem", memoryGroupId: "disk:primary",
        totalBytes: 128 * GIB, availableBytes: 64 * GIB, allocatedBytes: 0, reservedBytes: 0, poolDirectory: "/fixture/primary", filesystemDeviceId: "1", readOnly: false }),
      ...(extra ? [device({ id: "storage:chosen", label: "Chosen drive", kind: "storage", backend: "filesystem", memoryGroupId: "disk:chosen",
        totalBytes: 256 * GIB, availableBytes: 144 * GIB, allocatedBytes: 0, reservedBytes: 0, poolDirectory: "/fixture/chosen", filesystemDeviceId: "2", readOnly: false })] : [])
    ], links: [{ from: "ram:system", to: "gpu:cuda:0", peerAccess: false, measurement: {
      bytesPerSecond: 8 * GIB, sampleBytes: 16 * 1024 ** 2, runs: 3, medianNanoseconds: 1953125, measuredAt, scope: "device-transfer"
    } }] };
}

/** Deliberately different diagnostic/free-space and applied/pinned quotas. */
export function devicePlanFixture(): DeviceResourcePlan {
  return { version: 1, topologyMeasuredAt: measuredAt, budgets: [
    { deviceId: "storage:primary", memoryGroupId: "disk:primary", kind: "storage", enabled: true,
      capacityBytes: 128 * GIB, ceilingBytes: 32 * GIB, currentlyAvailableBytes: 64 * GIB,
      reserveBytes: 20 * GIB, selection: "auto", autoIncrease: false },
    { deviceId: "storage:chosen", memoryGroupId: "disk:chosen", kind: "storage", enabled: true,
      capacityBytes: 256 * GIB, ceilingBytes: 96 * GIB, currentlyAvailableBytes: 144 * GIB,
      reserveBytes: 20 * GIB, selection: "auto", autoIncrease: false }
  ], placementOrder: [], memoryGroupCeilings: {}, physicalMemoryCapacityBytes: 24 * GIB,
    selectedMemoryCapacityBytes: 8 * GIB, selectedStorageCapacityBytes: 96 * GIB,
    projectedContextMaximumTokens: 999999, admittedContextMaximumTokens: 8192,
    maximumBasis: "weighted-state-and-reserves-deduplicated-physical-pools", automaticContextTokens: 4096,
    fits: true, blockers: [], executionBlockers: [], appliedStorage: {
      largestSharedPoolBytes: 48 * GIB, chargedBytes: 2 * GIB, pendingBytes: GIB, groups: [
        { memoryGroupId: "disk:primary", ceilingBytes: 8 * GIB, availableAfterReserveBytes: 44 * GIB,
          chargedBytes: 0, pendingBytes: 0, unmaterializedPromiseBytes: 0 },
        { memoryGroupId: "disk:chosen", ceilingBytes: 48 * GIB, availableAfterReserveBytes: 124 * GIB,
          chargedBytes: 2 * GIB, pendingBytes: GIB, unmaterializedPromiseBytes: 4 * GIB }
      ]
    }
  };
}
