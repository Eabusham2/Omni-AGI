import { describe, expect, it } from "vitest";
import { ResourcePlanner, GIB } from "../src/main/resourcePlanner";
import { nativeCoreInventory } from "../src/main/nativeCoreInventory";

const mib = 1024 ** 2;
const benchmark = { measuredAt: "2026-09-30T12:00:00Z", sampleBytes: 16 * mib,
  memoryBytesPerSecond: 12 * GIB, storageBytesPerSecond: 40 * mib, cacheHit: true };
function planner(ram: number, available = ram) {
  return new ResourcePlanner("/fixture-no-probe", {
    readResources: async () => ({ totalMemoryBytes: ram, availableMemoryBytes: available,
      diskTotalBytes: 2 * 1024 * GIB, diskFreeBytes: 1536 * GIB }),
    benchmark: async () => benchmark,
    measurementCollector: async () => { throw new Error("default must not run a compute probe"); }
  });
}

describe("fresh total-RAM-first baseline with working-set headroom", () => {
  it("scales real native dimensions beyond the old tiny tiers without a TPS/parameter ceiling", async () => {
    const small = await planner(8 * GIB).plan({ mode: "auto" }, { hardwareTier: "personal" });
    const large = await planner(32 * GIB).plan({ mode: "auto" }, { hardwareTier: "personal" });
    expect(small.allowed).toBe(true);
    expect(large.allowed).toBe(true);
    expect(small.nativeArchitecture!.shape.dModel).toBeGreaterThan(64);
    expect(large.nativeArchitecture!.shape.dModel).toBeGreaterThan(small.nativeArchitecture!.shape.dModel);
    expect(large.nativeArchitecture!.shape.layers).toBeGreaterThan(small.nativeArchitecture!.shape.layers);
    expect(large.nativeArchitecture!.inventory).toEqual(nativeCoreInventory(large.nativeArchitecture!.shape));
    expect(large.nativeArchitecture!.sizing).toMatchObject({ isAutomaticDefault: true,
      transientAvailableRamUsedForShape: false, ramPercentageIsCeilingNotUsageTarget: true,
      workingSetIsOsRssGuarantee: false, selectionMode: "ram-first-headroom-default" });
    expect(large.nativeArchitecture!.sizing).not.toHaveProperty("maximumTraversalMacsProxy");
  });

  it("keeps Auto model/context/fast activity resident on an HDD baseline with free slack and training room", async () => {
    for (const [ram, tier] of [[4 * GIB, "micro"], [8 * GIB, "personal"], [32 * GIB, "gpu"], [64 * GIB, "workstation"]] as const) {
      const plan = await planner(ram).plan({ mode: "auto" }, { hardwareTier: tier });
      expect(plan.allowed, `${tier}: ${plan.blockers.join(";")}`).toBe(true);
      expect(plan.offload.modelSpillBytes).toBe(0);
      expect(plan.offload.memorySpillBytes).toBe(0);
      expect(plan.offload.contextSpillBytes).toBe(0);
      expect(plan.offload.required).toBe(false);
      expect(plan.context.selectedTokens).toBeGreaterThanOrEqual(plan.context.floorTokens);
      expect(plan.context.selectedTokens).toBeGreaterThanOrEqual(Number(plan.nativeArchitecture!.sizing.baselineContextTokens));
      expect(Number(plan.nativeArchitecture!.sizing.ramFirstUnusedCeilingBytes)).toBeGreaterThan(0);
      expect(Number(plan.nativeArchitecture!.sizing.ramFirstBaselineEstimatedWorkingSetBytes))
        .toBeLessThanOrEqual(plan.resources.systemRamBudgetBytes);
      expect(Number(plan.nativeArchitecture!.sizing.boundedTrainingWithHeadroomReserveBytes)).toBeGreaterThan(0);
    }
  });

  it("does not shrink chosen baseline for busy applications, and asks to close them", async () => {
    const open = await planner(32 * GIB).plan({ mode: "auto" }, { hardwareTier: "personal" });
    const busy = await planner(32 * GIB, GIB).plan({ mode: "auto" }, { hardwareTier: "personal" });
    expect(busy.nativeArchitecture).toEqual(open.nativeArchitecture);
    expect(busy.context.selectedTokens).toBe(open.context.selectedTokens);
    expect(busy.selectedItems).toBe(open.selectedItems);
    expect(busy.warnings.join(" ")).toMatch(/Close memory-heavy applications/);
  });

  it("keeps physical geometry/item-price/reserve evidence finite on fresh, blocked and saved-runtime branches", async () => {
    const plans = await Promise.all([
      planner(16 * GIB).plan({ mode: "auto" }, { hardwareTier: "personal" }),
      planner(16 * GIB).plan({ mode: "auto", storagePoolMode: "manual", storagePoolBytes: String(mib) }, { hardwareTier: "personal" }),
      planner(16 * GIB).plan({ mode: "auto" }, { hardwareTier: "personal", modelBytes: 512 * mib,
        modelHiddenSize: 1536, modelLayers: 11, fixedWorkingMemoryItems: 16384, admissionScope: "existing-runtime" })
    ]);
    for (const plan of plans) {
      for (const key of ["modelHiddenSize", "modelLayers", "estimatedResidentMemoryItemBytes", "estimatedPagedMemoryItemBytes", "reservedTrainingTransferBytes"] as const) {
        expect(Number.isFinite(plan.context.evidence[key]), key).toBe(true);
        expect(plan.context.evidence[key], key).toBeGreaterThan(0);
      }
      expect(plan.context.evidence.estimatedResidentMemoryItemBytes)
        .toBe(Math.max(4096, 4 * plan.context.evidence.modelHiddenSize + 512));
    }
    expect(plans[2]!.context.evidence.modelHiddenSize).toBe(1536);
    expect(plans[2]!.context.evidence.modelLayers).toBe(11);
    expect(plans[2]!.nativeArchitecture).toBeUndefined();
  });

  it("lets Auto pool grow instead of shrinking the selected cortex to a bootstrap pool", async () => {
    const auto = await planner(32 * GIB).plan({ mode: "auto" }, { hardwareTier: "personal" });
    const constrained = await planner(32 * GIB).plan({ mode: "auto", storagePoolMode: "manual", storagePoolBytes: String(64 * mib) }, { hardwareTier: "personal" });
    expect(auto.resources.sharedStoragePoolBytes).toBeGreaterThan(64 * mib);
    expect(constrained.nativeArchitecture!.shape).toEqual(auto.nativeArchitecture!.shape);
    expect(constrained.allowed).toBe(false);
    expect(constrained.blockers.join(" ")).toMatch(/storage pool|storage-pool/);
  });

  it("keeps manual context maximum physical and permits declared spill without changing saved shapes", async () => {
    const auto = await planner(32 * GIB).plan({ mode: "auto" }, { hardwareTier: "personal" });
    expect(auto.context.maximumTokens).toBeGreaterThan(32768);
    const manual = await planner(32 * GIB).plan({ mode: "manual", requestedContextTokens: String(auto.context.selectedTokens * 4),
      storagePoolMode: "manual", storagePoolBytes: String(1024 * GIB) }, {
      hardwareTier: "personal", modelBytes: auto.resources.modelBytes,
      modelHiddenSize: auto.nativeArchitecture!.shape.dModel, modelLayers: auto.nativeArchitecture!.shape.layers,
      fixedWorkingMemoryItems: auto.selectedItems, modelWorkspaceSlots: auto.nativeArchitecture!.shape.workspaceLatents,
      enforceContextFloor: false });
    expect(manual.nativeArchitecture).toBeUndefined();
    expect(manual.context.maximumTokens).toBeGreaterThan(32768);
    expect(manual.allowed, manual.blockers.join(";")).toBe(true);
    expect(manual.offload.contextSpillBytes).toBeGreaterThan(0);
    expect(manual.selectedItems).toBe(auto.selectedItems);
  });
});
