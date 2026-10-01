import { describe, expect, it } from "vitest";
import {
  defaultGroundUpWorkingMemoryItems,
  groundUpArchitectureProfile
} from "../src/main/omniArchitectureProfile";
import { nativeCoreInventory } from "../src/main/nativeCoreInventory";
import {
  GIB,
  ResourcePlanner,
  type ResourceSnapshot,
  type StorageBenchmark
} from "../src/main/resourcePlanner";

const resources: ResourceSnapshot = {
  totalMemoryBytes: 32 * GIB,
  availableMemoryBytes: 24 * GIB,
  diskTotalBytes: 500 * GIB,
  diskFreeBytes: 200 * GIB
};

const benchmark: StorageBenchmark = {
  measuredAt: "2026-09-07T00:00:00.000Z",
  sampleBytes: 16 * 1024 * 1024,
  memoryBytesPerSecond: 20 * GIB,
  storageBytesPerSecond: GIB,
  cacheHit: true
};

describe("ground-up OmniCortex architecture accounting", () => {
  it("reports the exact Personal parameter inventory without an upstream model", () => {
    const profile = groundUpArchitectureProfile("personal");

    expect(profile).toMatchObject({
      architecture: "OmniCortex",
      origin: "ground-up-random-initialization",
      externalPretrainedWeights: false,
      hardwareTier: "personal",
      dModel: 64,
      layers: 2,
      workingMemoryItems: 32_768,
      workspaceLatents: 8_192,
      exactLogicalParameterCount: 1_301_695,
      exactPackedProjectionParameterCount: 736_991,
      exactPackedTableParameterCount: 560_608,
      exactPackedTernaryParameterCount: 1_301_695,
      packedTernaryWeightBytes: 325_454,
      packedWorkspaceTableBytes: 131_072,
      packedMetaplasticityReserveBytes: 72_908,
      fixedControlBufferReserveBytes: 16_384,
      packedUpdateScratchBytes: 4_194_304,
      residentInferenceStateBytes: 455_706,
      minimumTrainingStateBytes: 4_650_010,
      checkpointTensorBytes: 1_504_282,
      parameterCountBasis: "architecture-logical-neural-elements"
    });
  });

  it("scales only the disclosed native architecture and its workspace", () => {
    const tiers = (["micro", "personal", "gpu", "workstation"] as const)
      .map((tier) => groundUpArchitectureProfile(tier));

    expect(tiers.map((profile) => profile.exactLogicalParameterCount)).toEqual([
      403_895,
      1_301_695,
      4_096_927,
      9_907_583
    ]);
    expect(
      groundUpArchitectureProfile(
        "personal",
        defaultGroundUpWorkingMemoryItems("personal", "extended")
    ).exactLogicalParameterCount
    ).toBe(2_874_559);
  });

  it("does not charge dense masters, gradients, or Adam moments for packed weights or tables", () => {
    const profile = groundUpArchitectureProfile("personal");
    expect(profile.exactLogicalParameterCount).toBe(
      profile.exactPackedTernaryParameterCount
    );
    expect(profile).not.toHaveProperty("higherPrecisionParameterCount");
    expect(profile).not.toHaveProperty("higherPrecisionWeightBytes");
    expect(profile.exactPackedTableParameterCount).toBe(
      36_320 + profile.workspaceLatents * profile.dModel
    );
    expect(profile.packedWorkspaceTableBytes).toBe(
      profile.workspaceLatents * Math.ceil(profile.dModel / 4)
    );
    expect(profile.minimumTrainingStateBytes).toBe(
      profile.residentInferenceStateBytes +
        profile.packedUpdateScratchBytes
    );
    expect(profile.checkpointTensorBytes).toBe(
      profile.residentInferenceStateBytes +
        1024 ** 2
    );
    expect(profile.residentInferenceStateBytes).toBeLessThan(
      profile.exactLogicalParameterCount * 4
    );
    expect(profile.packedMetaplasticityReserveBytes).toBeLessThan(
      profile.packedTernaryWeightBytes
    );
  });

  it("uses the native profile for a new Build but not for an explicit legacy checkpoint", async () => {
    const planner = new ResourcePlanner("/tmp/omni-ground-up-plan", {
      readResources: async () => resources,
      benchmark: async () => benchmark
    });

    const nativePlan = await planner.plan(
      { mode: "auto", storagePoolMode: "manual", storagePoolBytes: String(32 * GIB) },
      { hardwareTier: "personal", nativeSizingMode: "physical-capacity" }
    );
    expect(nativePlan.allowed, JSON.stringify(nativePlan.blockers)).toBe(true);
    expect(nativePlan.architecture).toMatchObject({
      origin: "ground-up-random-initialization",
      externalPretrainedWeights: false,
      workingMemoryItems: nativePlan.selectedItems
    });
    expect(nativePlan.nativeArchitecture?.shape.dModel).toBeGreaterThan(64);
    const exact = nativeCoreInventory(nativePlan.nativeArchitecture!.shape,
      nativePlan.nativeArchitecture!.sizing.routerStorageLayout as "block-sparse-v1" | undefined);
    expect(nativePlan.architecture!.exactLogicalParameterCount).toBe(exact.logicalParameters);
    expect(nativePlan.resources).toMatchObject({
      modelBytes: nativePlan.architecture!.checkpointTensorBytes,
      modelParameterCount: exact.logicalParameters,
      modelParameterCountBasis: "architecture-logical-neural-elements",
      modelStorageBasis: "packed-weights-plus-nonweight-state-reserve"
    });
    expect(nativePlan.resources.modelBytes).not.toBe(665_041_488);
    expect(nativePlan.resources.modelBytes).not.toBe(998_568_704);
    expect(nativePlan.resources.modelWorkingSetBytes).toBe(
      Math.ceil(nativePlan.architecture!.residentInferenceStateBytes * 1.2)
    );
    expect(nativePlan.resources.estimatedResidentModelBytes).toBe(
      Math.max(
        8 * 1024 ** 2,
        nativePlan.architecture!.residentInferenceStateBytes +
          nativePlan.architecture!.packedUpdateScratchBytes
      )
    );
    const disk = nativePlan.diskSpace;
    const projectedWrites =
      disk.modelBytes +
      disk.checkpointBytes +
      disk.maximumWorkingMemorySpillBytes +
      disk.futureGrowthBytes +
      disk.operationWriteBytes;
    expect(nativePlan.resources.requiredStoragePoolBytes).toBe(projectedWrites);
    expect(disk.projectedRemainingBytes).toBe(
      disk.diskFreeBytes - projectedWrites
    );
    expect(disk.selectedDatasetBytes).toBe(0);

    const legacyPlan = await planner.plan(
      { mode: "auto" },
      { hardwareTier: "personal", modelBytes: 665_041_488 }
    );
    expect(legacyPlan.architecture).toBeUndefined();
    expect(legacyPlan.resources.modelBytes).toBe(665_041_488);
  });
});
