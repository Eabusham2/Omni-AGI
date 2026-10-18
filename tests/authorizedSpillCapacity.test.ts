import { describe, expect, it } from "vitest";
import { authorizedSpillCapacity } from "../src/main/authorizedSpillCapacity";
import { planWorkingMemory, GIB, MIB } from "../src/main/resourcePlanner";
import type { AuthorizedSpillStorageSnapshot } from "../src/shared/deviceResources";

const snapshot = (): AuthorizedSpillStorageSnapshot => ({ authorized: true,
  largestSharedPoolBytes: 0, sharedUsedBytes: 0, sharedPendingBytes: 0, ownerBytes: 0,
  groups: ["a", "b"].map((physicalGroup) => ({ physicalGroup, ceilingBytes: 8 * GIB,
    availableBytes: 10 * GIB, allocatedBytes: 0, pendingBytes: 0, unmaterializedPromiseBytes: 0,
    ownerBytes: 0, primaryVolume: false, allocationUnitBytes: 4096 })) });
const plan = (spillStorage?: AuthorizedSpillStorageSnapshot, allowStorageSpill = true) => planWorkingMemory({
  request: { mode: "manual", requestedContextTokens: "8192", systemRamMode: "manual", systemRamSharePercent: 65 },
  resources: { totalMemoryBytes: 16 * GIB, availableMemoryBytes: 14 * GIB, diskTotalBytes: 256 * GIB, diskFreeBytes: 25 * GIB },
  benchmark: { measuredAt: "2026-10-07T00:00:00Z", sampleBytes: 4096, memoryBytesPerSecond: 20 * GIB, storageBytesPerSecond: GIB, cacheHit: true },
  hardwareTier: "personal", modelBytes: 256 * MIB, modelHiddenSize: 256, modelLayers: 8,
  modelWorkspaceSlots: 8, fixedWorkingMemoryItems: 8, spillStorage, allowStorageSpill, enforceContextFloor: false
});

describe("actual selected-drive capacity admission", () => {
  it("adds distinct authorized volumes but respects the global largest pool", () => {
    const pools = snapshot();
    expect(authorizedSpillCapacity(pools, { selectedPoolBytes: 12 * GIB, primaryRequiredBytes: 2 * GIB }).capacityBytes).toBe(10 * GIB);
    pools.largestSharedPoolBytes = 30 * GIB;
    expect(authorizedSpillCapacity(pools, { selectedPoolBytes: 12 * GIB, primaryRequiredBytes: 2 * GIB }).capacityBytes).toBe(16 * GIB);
    pools.groups.push({ ...pools.groups[0]! });
    expect(() => authorizedSpillCapacity(pools, { selectedPoolBytes: 30 * GIB, primaryRequiredBytes: 0 })).toThrow(/twice/);
  });
  it("retains quotas under external pressure and excludes outstanding sparse promises", () => {
    const pools = snapshot(); pools.groups[0]!.availableBytes = GIB;
    pools.groups[0]!.unmaterializedPromiseBytes = GIB;
    pools.groups[1]!.pendingBytes = 2 * GIB; pools.sharedPendingBytes = 2 * GIB;
    const result = authorizedSpillCapacity(pools, { selectedPoolBytes: 20 * GIB, primaryRequiredBytes: GIB });
    expect(result.capacityBytes).toBe(16 * GIB);
    expect(result.currentlyAvailableBytes).toBe(6 * GIB);
  });
  it("reserves primary checkpoint/training room once and credits only this owner's existing pages", () => {
    const pools = snapshot(); pools.groups[0]!.primaryVolume = true;
    pools.groups[1]!.allocatedBytes = 3 * GIB; pools.groups[1]!.ownerBytes = GIB;
    pools.sharedUsedBytes = 3 * GIB; pools.ownerBytes = GIB;
    const result = authorizedSpillCapacity(pools, { selectedPoolBytes: 30 * GIB, primaryRequiredBytes: 2 * GIB });
    expect(result.capacityBytes).toBe(14 * GIB);
    expect(result.currentlyAvailableBytes).toBe(12 * GIB);
  });
  it("raises the real context maximum with drives, not its RAM-first automatic sweetspot", () => {
    const before = plan(), after = plan(snapshot());
    expect(after.context.maximumTokens).toBeGreaterThan(before.context.maximumTokens);
    expect(after.context.autoTokens).toBe(before.context.autoTokens);
    expect(after.context.evidence.selectedDriveCapacityBytes).toBeGreaterThan(0);
    expect(after.diskSpace.maximumWorkingMemorySpillBytes).toBe(0);
  });
  it("temporary occupied drives pause without changing configured context or theoretical limit", () => {
    const pools = snapshot(), before = plan(pools);
    pools.groups.forEach((group) => { group.availableBytes = 0; });
    const after = plan(pools);
    expect(after.context.maximumTokens).toBe(before.context.maximumTokens);
    expect(after.context.selectedTokens).toBe(before.context.selectedTokens);
    expect(after.allowed).toBe(false);
    expect(after.blockers.join(" ")).toMatch(/temporarily occupied/);
  });
  it("disabled spill admits only resident context and does not reserve cold backing", () => {
    const result = plan(undefined, false);
    expect(result.context.maximumTokens).toBe(result.context.evidence.residentContextMaximumTokens);
    expect(result.context.evidence.contextOffloadBudgetBytes).toBe(0);
  });
  it("does not silently raise Auto's RAM share to fit a large saved model", () => {
    const input = { request: { mode: "auto" as const }, resources: { totalMemoryBytes: 16 * GIB, availableMemoryBytes: 14 * GIB,
      diskTotalBytes: 500 * GIB, diskFreeBytes: 300 * GIB }, benchmark: { measuredAt: "2026-10-07T00:00:00Z", sampleBytes: 4096,
      memoryBytesPerSecond: 20 * GIB, storageBytesPerSecond: GIB, cacheHit: true }, hardwareTier: "personal" as const };
    expect(planWorkingMemory({ ...input, modelBytes: 256 * MIB }).resources.systemRamSharePercent).toBe(65);
    expect(planWorkingMemory({ ...input, modelBytes: 32 * GIB }).resources.systemRamSharePercent).toBe(65);
    expect(planWorkingMemory({ ...input, modelBytes: 32 * GIB, benchmark: { ...input.benchmark, storageBytesPerSecond: 0 } }).resources.systemRamSharePercent).toBe(65);
  });
});
