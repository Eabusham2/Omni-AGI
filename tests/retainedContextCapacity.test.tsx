import React, { isValidElement, type ReactElement, type ReactNode } from "react";
import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { contextCapacity, contextColdSpillBytes, contextSnapshotBytes, retainedContextPools,
  checkpointPackedOwnerMinimum, type ContextCapacityInput, type RetainedContextPool, type RetainedContextMinimum } from "../src/main/retainedContextCapacity";
import { ResourcePlanner, GIB } from "../src/main/resourcePlanner";
import type { DeviceResourceBudget, DeviceResourceSettings, DeviceResourceTopology, ResourceDevice } from "../src/shared/deviceResources";
import { DeviceAllocationCards, deviceAllocationChoice, deviceAvailability } from "../src/renderer/src/DeviceAllocationSelector";

const fixture = JSON.parse(readFileSync(new URL("./fixtures/retained-context-capacity.json", import.meta.url), "utf8")) as {
  topology: DeviceResourceTopology; budgets: DeviceResourceBudget[]; expectedPools: RetainedContextPool[];
  manualMinimums: Record<string, RetainedContextMinimum>; manualExpectedBudgets: Record<string, number>;
  capacityCases: Array<{ name: string; input: ContextCapacityInput; expected: ReturnType<typeof contextCapacity>; snapshotTokens: number; snapshotBytes: number }>;
};

function elements(node: ReactNode): ReactElement<Record<string, unknown>>[] {
  if (Array.isArray(node)) return node.flatMap(elements);
  if (!isValidElement<Record<string, unknown>>(node)) return [];
  return [node, ...elements(node.props.children as ReactNode)];
}

const gpu = (patch: Partial<ResourceDevice> = {}): ResourceDevice => ({ id: "cuda:fixture", label: "Inert CUDA memory",
  kind: "gpu", backend: "cuda", executionDevice: "cuda:0", memoryGroupId: "gpu:fixture", totalBytes: 4 * GIB,
  availableBytes: 3 * GIB, allocatedBytes: 0, reservedBytes: 0, unifiedMemory: false, nativeComputeSupported: false,
  unavailableReason: "native-packed-compute-kernel-unavailable", ...patch });
const topology = (device: ResourceDevice): DeviceResourceTopology => ({ format: "omni-device-resource-topology", version: 1,
  measuredAt: "2026-10-18T00:00:00Z", platform: "inert", architecture: "inert", neuralModelConstructed: false,
  measurementRequested: false, warnings: [], links: [], devices: [{ id: "ram:system", label: "Inert RAM", kind: "ram",
    backend: "cpu", executionDevice: "cpu", memoryGroupId: "ram:system", totalBytes: 16 * GIB,
    availableBytes: 12 * GIB, allocatedBytes: 0, reservedBytes: 0, unifiedMemory: false, nativeComputeSupported: true }, device] });
const settings = (enabled = true): DeviceResourceSettings => ({ version: 1, allowStorageSpill: false, autoIncreaseStorage: false,
  storageDirectories: [], allocations: [{ deviceId: "ram:system", enabled: true, mode: "auto", percent: 70, autoIncrease: false },
    { deviceId: "cuda:fixture", enabled, mode: "manual", percent: 90, autoIncrease: false }] });

describe("actual retained-context capacity contracts, not hardware speed evidence", () => {
  it("uses the identical literal physical-pool fixture as Python and never counts unified aliases twice", () => {
    expect(retainedContextPools(fixture.topology, fixture.budgets)).toEqual(fixture.expectedPools);
    expect(fixture.expectedPools.reduce((total, item) => total + item.budgetBytes, 0)).toBe(6000);
  });

  it("matches Python for simultaneous RAM/VRAM/cold disk, host metadata and separate paused snapshot bytes", () => {
    for (const item of fixture.capacityCases) {
      expect(contextCapacity(item.input), item.name).toEqual(item.expected);
      expect(contextSnapshotBytes(item.snapshotTokens, item.input), item.name).toBe(item.snapshotBytes);
    }
    const simultaneous = fixture.capacityCases.find(item => item.name.startsWith("simultaneous"))!;
    expect(contextColdSpillBytes(simultaneous.expected.maximumTokens, simultaneous.input)).toBe(28160);
    expect(simultaneous.expected.maximumTokens).toBeGreaterThan(simultaneous.expected.residentMaximumTokens);
    expect(simultaneous.expected.maximumTokens).toBeGreaterThan(simultaneous.input.spillCapacityBytes / simultaneous.input.pagedKvBytesPerToken);
  });

  it("rebalances Manual from real minima, keeps missing minima conservative and does not aggregate indivisible pages", () => {
    const pools = retainedContextPools(fixture.topology, fixture.budgets, { manual: true,
      minimums: fixture.manualMinimums, pageBytes: 1000 });
    expect(Object.fromEntries(pools.map(item => [item.executionDevice, item.budgetBytes]))).toEqual(fixture.manualExpectedBudgets);
    expect(pools[0]!.minimumPackedOwnerBytes).toBe(0);
    expect(pools[1]!.minimumPackedOwnerBytes).toBe(3000);
    expect(pools.every(item => item.partitionMode === "manual-admitted-minimums")).toBe(true);
    expect(retainedContextPools(fixture.topology, fixture.budgets, { manual: true }).map(item => item.budgetBytes)).toEqual([2400, 3600]);
    expect(retainedContextPools(fixture.topology, fixture.budgets, { pageBytes: 4096 }).reduce((total, item) => total + item.budgetBytes, 0)).toBe(0);
  });

  it("reserves complete direct packed owners rather than the largest separate sign tensor or child namespace", () => {
    const tensor = (bytes: number) => ({ dtype: "U8", shape: [bytes], bytes });
    expect(checkpointPackedOwnerMinimum(new Map([
      ["decoder.block._packed_forward_weight", tensor(8)], ["decoder.block._connection_state", tensor(32)],
      ["decoder.block._row_stability", tensor(4)], ["decoder.block.parallel_connections.child._packed_forward_weight", tensor(2)],
      ["decoder.block.parallel_connections.child._connection_state", tensor(8)]
    ]))).toBe(48);
    expect(checkpointPackedOwnerMinimum(new Map([["nonpacked", tensor(4096)]]))).toBeUndefined();
  });

  it("the real main planner credits an enabled storage-only GPU behind actual pool math without compute acknowledgement", async () => {
    const planner = new ResourcePlanner("/inert/no-file-read", {
      readResources: async () => ({ totalMemoryBytes: 16 * GIB, availableMemoryBytes: 12 * GIB,
        diskTotalBytes: 500 * GIB, diskFreeBytes: 200 * GIB, diskBlockSizeBytes: 4096 }),
      benchmark: async () => ({ measuredAt: "2026-10-18T00:00:00Z", sampleBytes: 4096,
        memoryBytesPerSecond: 10 * GIB, storageBytesPerSecond: GIB, cacheHit: true }),
      deviceTopology: async () => topology(gpu())
    });
    const options = { modelBytes: 512 * 1024 ** 2, maximumPackedOwnerBytes: 65536, nonpackedControlBytes: 0, modelHiddenSize: 64,
      modelLayers: 2, modelWorkspaceSlots: 8, fixedWorkingMemoryItems: 8, modelContextLimitTokens: 10000000 };
    const without = await planner.plan({ mode: "manual", requestedContextTokens: "8192", deviceResources: settings(false) }, options);
    const withGpu = await planner.plan({ mode: "manual", requestedContextTokens: "8192", deviceResources: settings(true) }, options);
    expect(withGpu.context.maximumTokens).toBeGreaterThan(without.context.maximumTokens);
    expect(withGpu.context.evidence.retainedDeviceContextBytes).toBeGreaterThan(0);
    expect(withGpu.context.evidence.manualCoreRebalance).toBe(true);
    expect(withGpu.context.evidence.contextCheckpointReserveBytes).toBeGreaterThan(8192 * 1024);
    expect(withGpu.deviceResources?.executionBlockers).toEqual([]);
    expect(withGpu.deviceResources?.admittedContextMaximumTokens).toBe(withGpu.context.maximumTokens);
  });

  it("admits a primary snapshot larger than extra-only spill capacity without combining aliases or bypassing the global quota", async () => {
    const extra = "/inert/extra-spill", extraBytes = 10 * GIB;
    let alias = false;
    const selected = settings(false); selected.allowStorageSpill = true; selected.storageDirectories = [extra];
    selected.allocations[0] = { deviceId: "ram:system", enabled: true, mode: "manual", percent: 70, autoIncrease: false };
    selected.allocations.push({ deviceId: "storage:extra", enabled: true, mode: "manual", percent: 100, autoIncrease: false });
    const planner = new ResourcePlanner("/inert/no-file-read", {
      readResources: async () => ({ totalMemoryBytes: 64 * GIB, availableMemoryBytes: 64 * GIB,
        diskTotalBytes: 500 * GIB, diskFreeBytes: 80 * GIB, diskBlockSizeBytes: 4096 }),
      benchmark: async () => ({ measuredAt: "2026-10-18T00:00:00Z", sampleBytes: 4096,
        memoryBytesPerSecond: 20 * GIB, storageBytesPerSecond: GIB, cacheHit: true }),
      deviceTopology: async () => {
        const current = topology(gpu()); current.devices[0]!.totalBytes = 64 * GIB; current.devices[0]!.availableBytes = 64 * GIB;
        current.devices.push({ id: "storage:extra", label: "Inert authorized spill", kind: "storage", backend: "filesystem",
          memoryGroupId: alias ? "primary:fixture" : "extra:fixture", totalBytes: 200 * GIB, availableBytes: 30 * GIB,
          allocatedBytes: 0, reservedBytes: 0, unifiedMemory: false, nativeComputeSupported: false,
          poolDirectory: extra, filesystemDeviceId: "42", readOnly: false });
        return current;
      },
      spillCapacity: async () => ({ authorized: true, largestSharedPoolBytes: 0, sharedUsedBytes: 0, sharedPendingBytes: 0, ownerBytes: 0,
        groups: [{ physicalGroup: alias ? "primary:fixture" : "extra:fixture", ceilingBytes: extraBytes,
          availableBytes: extraBytes, allocatedBytes: 0, pendingBytes: 0, unmaterializedPromiseBytes: 0,
          ownerBytes: 0, primaryVolume: alias, allocationUnitBytes: 4096 }] })
    });
    const options = { modelBytes: 128 * 1024 ** 2, maximumPackedOwnerBytes: 65536, nonpackedControlBytes: 0,
      modelHiddenSize: 512, modelLayers: 2, modelWorkspaceSlots: 8, fixedWorkingMemoryItems: 8, modelContextLimitTokens: 10000000 };
    const request = { mode: "manual" as const, requestedContextTokens: "5000000", deviceResources: selected };
    const plan = await planner.plan(request, options);
    expect(plan.context.selectedTokens).toBe(5000000);
    expect(plan.context.evidence.contextCheckpointReserveBytes).toBeGreaterThan(extraBytes);
    expect(plan.context.evidence.contextCheckpointReserveBytes).toBeLessThan(60 * GIB);
    expect(plan.context.evidence.contextOffloadBudgetBytes).toBeGreaterThan(0);
    expect(plan.context.evidence.contextOffloadBudgetBytes).toBeLessThan(extraBytes);
    expect(plan.resources.maximumStoragePoolBytes).toBe(70 * GIB);
    expect(plan.allowed, plan.blockers.join("\n")).toBe(true);
    const quota = await planner.plan({ ...request, storagePoolMode: "manual", storagePoolBytes: String(extraBytes) }, options);
    expect(quota.allowed).toBe(false);
    expect(quota.context.maximumTokens).toBeLessThan(plan.context.selectedTokens);
    alias = true;
    const samePhysicalVolume = await planner.plan(request, options);
    expect(samePhysicalVolume.resources.maximumStoragePoolBytes).toBe(60 * GIB);
  });

  it("existing card elements allow explicit verified storage-only CUDA and keep unknown/XPU disabled with defaults unchanged", () => {
    const onChange = vi.fn(), valid = gpu(), unknown = gpu({ id: "cuda:unknown", label: "Unknown CUDA", availableBytes: null }),
      unsupported = gpu({ id: "xpu:unknown", label: "Unknown XPU", backend: "xpu", executionDevice: "xpu:0" });
    const current = topology(valid); current.devices.push(unknown, unsupported);
    const tree = DeviceAllocationCards({ topology: current, ramMode: "auto", ramPercent: 70, onChange });
    const inputs = elements(tree).filter(node => node.type === "input");
    const checkbox = inputs.find(node => node.props["aria-label"] === "Enable Inert CUDA memory")!;
    expect(checkbox.props.disabled).toBeFalsy(); expect(checkbox.props.checked).toBe(false);
    expect(inputs.find(node => node.props["aria-label"] === "Enable Unknown CUDA")!.props.disabled).toBe(true);
    expect(inputs.find(node => node.props["aria-label"] === "Enable Unknown XPU")!.props.disabled).toBe(true);
    expect(renderToStaticMarkup(tree)).toContain("Retained working-memory storage is supported");
    (checkbox.props.onChange as (event: { target: { checked: boolean } }) => void)({ target: { checked: true } });
    expect(onChange).toHaveBeenCalledOnce();
    const selected = onChange.mock.calls[0]![0] as DeviceResourceSettings;
    expect(selected.allocations.find(item => item.deviceId === valid.id)?.enabled).toBe(true);
    expect(selected.regionalExecution).toBeUndefined();
    expect(deviceAvailability(unknown)).toBeTruthy();
    expect(deviceAllocationChoice(valid, undefined, "auto", 70).enabled).toBe(false);
  });
});
