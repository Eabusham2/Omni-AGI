import { describe, expect, it } from "vitest";
import { nativeCoreInventory, nativeArchitectureSha256,
  type NativeArchitectureShape, type NativeArchitectureDescriptor } from "../src/main/nativeCoreInventory";
import { capacityDerivedNativeArchitectureProfile } from "../src/main/omniArchitectureProfile";

const shape: NativeArchitectureShape = { dModel: 64, layers: 2, feedForward: 192, nHeads: 4,
  vsaDimensions: 256, routerNeurons: 64, modalityChannels: 16, imageSize: 16, audioSamples: 256,
  videoFrames: 4, workingMemoryItems: 32768, workspaceLatents: 8192, vocabSize: 261, liquidMode: "cfc" };
const gib = 1024 ** 3;
const input = { hardwareTier: "personal" as const, workingMemoryItems: 32768,
  selectedSystemRamBudgetBytes: 8 * gib, runtimeBaselineReserveBytes: 256 * 1024 ** 2,
  selectedStoragePoolBytes: 16 * gib, measuredStorageBytesPerSecond: gib,
  estimatedTrainingSourceBytes: 0, acceleratorAvailable: false };

describe("source-derived native shape arithmetic without models", () => {
  it("matches the Python canonical descriptor and exact default inventory", () => {
    const inventory = nativeCoreInventory(shape);
    expect(inventory.logicalParameters).toBe(1301695);
    expect(inventory.packedWeightBytes).toBe(325454);
    expect(inventory.resistanceBytes).toBe(16584);
    expect(inventory.packedOwners).toBe(104);
    const descriptor: Omit<NativeArchitectureDescriptor, "sha256"> = {
      format: "omni-main-selected-native-architecture", formatVersion: 1,
      architecture: "OmniCortex", externalPretrainedWeights: false,
      hardwareTier: "personal", shape, inventory,
      sizing: { policy: "fixture", selectedSystemRamBudgetBytes: 8589934592 },
      qualityEvidence: "unmeasured-native-quality-deferred"
    };
    expect(nativeArchitectureSha256(descriptor)).toBe("7caef23bd630bb834fe0c03bad59472466ebe4814e423060b872f1cb9e9dc7a2");
  });
  it("sizes real native width/depth from selected resources, not only workspace rows", () => {
    const small = capacityDerivedNativeArchitectureProfile({ ...input, selectedSystemRamBudgetBytes: 2 * gib });
    const large = capacityDerivedNativeArchitectureProfile(input);
    expect(large.dModel).toBeGreaterThan(small.dModel);
    expect(large.layers).toBeGreaterThan(small.layers);
    expect(large.dModel).toBeGreaterThan(64);
    expect(large.nativeArchitecture?.shape.workspaceLatents).toBe(shape.workspaceLatents);
    expect(large.nativeArchitecture?.sizing.fitsPolicyTarget).toBe(true);
    expect(large.nativeArchitecture?.qualityEvidence).toBe("unmeasured-native-quality-deferred");
    expect(large.nativeArchitecture?.sizing.computeEvidence).toContain("not-throughput-benchmark");
  });
  it("responds to supplied source and storage estimates without quality claims", () => {
    const plain = capacityDerivedNativeArchitectureProfile(input);
    const data = capacityDerivedNativeArchitectureProfile({ ...input, estimatedTrainingSourceBytes: 32 * gib });
    expect(data.exactLogicalParameterCount).toBe(plain.exactLogicalParameterCount);
    expect(data.nativeArchitecture?.sizing).not.toHaveProperty("modelCoreFractionPercent");
    expect(data.nativeArchitecture?.sizing.corePartitionBytes).toBeGreaterThan(data.packedTernaryWeightBytes);
    const storageLimited = capacityDerivedNativeArchitectureProfile({ ...input, selectedStoragePoolBytes: 32 * 1024 ** 2 });
    expect(storageLimited.dModel).toBeLessThan(plain.dModel);
    expect(data.nativeArchitecture?.sizing.sourceEvidence).toBe("source-byte-estimate-not-token-coverage");
    expect(data.checkpointByteBasis).toBe("packed-weights-plus-nonweight-state-reserve");
  });
});
