import { describe, expect, it, vi } from "vitest";
import {
  createNativeComputeMeasurementCollector, nativeProjectionTensorBudget, validateNativeProjectionComputeProfile,
  NATIVE_COMPUTE_ADMITTED_BYTES, NATIVE_COMPUTE_KERNEL_REVISION, NATIVE_COMPUTE_TENSOR_BUDGET,
  type NativeProjectionComputeProfile,
} from "../src/main/nativeComputeMeasurement";
import { groundUpArchitectureProfile, balancedMeasuredNativeArchitectureProfile } from "../src/main/omniArchitectureProfile";
import { nativeCoreInventory } from "../src/main/nativeCoreInventory";
import { planWorkingMemory } from "../src/main/resourcePlanner";
import type { EngineSupervisor } from "../src/main/engineSupervisor";

function profile(): NativeProjectionComputeProfile {
  return {
    format: "omni-bounded-packed-projection-compute-profile", formatVersion: 2,
    measuredAt: "2026-10-15T00:00:00Z", requestedDevice: "cpu", actualDevice: "cpu",
    kernel: "omni_core.model._packed_ternary_forward", kernelRevision: NATIVE_COMPUTE_KERNEL_REVISION,
    activityDtype: "float32", packedDtype: "uint8", quantizedActivityDtype: "int8", integerResultDtype: "int64",
    intermediateAccumulationDtypeVerified: false, outputDtype: "float32", connectionStateDtype: "uint8",
    connectionStateEncoding: "strength-low4-resistance-high4", repeatedContacts: 0, repeatedContactPages: 0,
    repeatedContactTimingMeasured: false, activityRows: 8, inputFeatures: 64, outputFeatures: 64,
    ...NATIVE_COMPUTE_TENSOR_BUDGET, scratchReserveBytes: 1048576, admittedBytes: NATIVE_COMPUTE_ADMITTED_BYTES,
    tensorReserveIsExactPeak: false, resourceByteBasis: "known-forward-tensors-plus-allocator-initialization-reserve",
    runs: 4, projectionMacsPerRun: 32768, medianRunMicroseconds: 1000, projectionMacsPerSecond: 32768000,
    elapsedMicroseconds: 4000, samplingBudgetMicroseconds: 250000, hardWallClockDeadline: false, fallbackReason: null,
    warmupPerformed: true, implementation: "native-packed-kernel", integerBackend: "runtime-dispatch-may-use-bounded-cpu-fallback",
    backendInitializationIncluded: false, neuralModelConstructed: false, neuralQualityMeasured: false,
    fullNeuralThroughputMeasured: false, evidence: "cache-tile-packed-primitive-not-complete-neural-throughput",
  };
}

describe("new weighted int64 primitive byte contract, without hardware", () => {
  it("counts contact bytes, int64 blocks+cat and row padding; old evidence/budget cannot be cached", async () => {
    expect(NATIVE_COMPUTE_TENSOR_BUDGET).toMatchObject({ sampleTensorBytes: 7172, connectionStateBytes: 4096,
      integerRowBlockBytes: 4096, integerConcatenationBytes: 4096, forwardTensorReserveBytes: 16940 });
    expect(NATIVE_COMPUTE_ADMITTED_BYTES).toBe(1072688);
    expect(nativeProjectionTensorBudget(2, 65, 3).packedSignBytes).toBe(51);
    expect(() => nativeProjectionTensorBudget(Number.MAX_SAFE_INTEGER, 64, 64)).toThrow("exact integer");
    for (const changed of [{ formatVersion: 1 }, { integerResultDtype: "int32" }, { admittedBytes: 1053700 },
      { repeatedContacts: 1 }, { connectionStateBytes: 0 }, { integerConcatenationBytes: 0 }]) {
      expect(() => validateNativeProjectionComputeProfile({ ...profile(), ...changed })).toThrow();
    }
    const request = vi.fn(async () => profile());
    const collector = createNativeComputeMeasurementCollector({ request } as unknown as Pick<EngineSupervisor, "request">);
    expect(await collector({ device: "cpu", hardwareTier: "personal", ramBudgetBytes: 1053700 })).toBeUndefined();
    expect(request).not.toHaveBeenCalled();
    expect(await collector({ device: "cpu", hardwareTier: "personal", ramBudgetBytes: NATIVE_COMPUTE_ADMITTED_BYTES })).toEqual(profile());
    expect(request).toHaveBeenCalledOnce();
    expect(await collector({ device: "cpu", hardwareTier: "personal", ramBudgetBytes: 1053700 })).toBeUndefined();
    expect(request).toHaveBeenCalledOnce();
  });

  it("reserves actual declared modality-head forward geometry without changing native inventory", () => {
    const value = groundUpArchitectureProfile("personal");
    expect(value.packedForwardTensorReserveBytes).toBe(201260);
    expect(value.minimumTrainingStateBytes).toBe(value.residentInferenceStateBytes + value.packedUpdateScratchBytes + 201260);
    const candidate = balancedMeasuredNativeArchitectureProfile({ hardwareTier: "personal", workingMemoryItems: 32,
      selectedSystemRamBudgetBytes: 8 * 1024 ** 3, runtimeBaselineReserveBytes: 256 * 1024 ** 2,
      selectedStoragePoolBytes: 64 * 1024 ** 3, measuredStorageBytesPerSecond: 1024 ** 3,
      estimatedTrainingSourceBytes: 0, acceleratorAvailable: false }, profile(), 100000);
    const descriptor = candidate.nativeArchitecture!;
    let expectedReserve = 0, sawModalityHead = false;
    const inventory = nativeCoreInventory(descriptor.shape, "block-sparse-v1", 2, layout => {
      if (!layout.table) {
        expectedReserve = Math.max(expectedReserve, nativeProjectionTensorBudget(8, layout.inputFeatures, layout.outputFeatures).forwardTensorReserveBytes);
        if (layout.outputFeatures === descriptor.shape.modalityChannels * descriptor.shape.videoFrames * (descriptor.shape.imageSize / 4) ** 2) sawModalityHead = true;
      }
    });
    expect(inventory).toEqual(nativeCoreInventory(descriptor.shape, "block-sparse-v1"));
    expect(inventory).toEqual(descriptor.inventory);
    expect(sawModalityHead).toBe(true);
    expect(candidate.packedForwardTensorReserveBytes).toBe(expectedReserve);
    expect(descriptor.sizing.minimumPackedForwardTensorReserveBytes).toBe(expectedReserve);
    expect(descriptor.sizing.minimumTransferComputeReserveBytes).toBe(Number(descriptor.sizing.packedRowScratchReserveBytes)
      + Number(descriptor.sizing.minimumAttentionTileBytes) + 4 * 1024 ** 2 + expectedReserve);
    expect(descriptor.sizing).toMatchObject({ primitiveIntegerResultDtype: "int64", primitiveAdmittedBytes: 1072688,
      primitiveRepeatedContacts: 0, primitiveRepeatedContactTimingMeasured: false, isAutomaticDefault: false });
  });

  it("carries the forward reserve into real resident and minimum-layer planner arithmetic", () => {
    const current = groundUpArchitectureProfile("personal");
    const omitted = { ...current, packedForwardTensorReserveBytes: 0,
      minimumTrainingStateBytes: current.minimumTrainingStateBytes - current.packedForwardTensorReserveBytes! };
    const plan = (architectureProfile: typeof current) => planWorkingMemory({ architectureProfile, hardwareTier: "personal",
      modelBytes: architectureProfile.checkpointTensorBytes, modelHiddenSize: current.dModel, modelLayers: current.layers,
      request: { mode: "manual", requestedItems: "32", storagePoolMode: "manual", storagePoolBytes: String(64 * 1024 ** 3) },
      resources: { totalMemoryBytes: 32 * 1024 ** 3, availableMemoryBytes: 30 * 1024 ** 3,
        diskTotalBytes: 500 * 1024 ** 3, diskFreeBytes: 400 * 1024 ** 3 },
      benchmark: { measuredAt: "2026-10-15T00:00:00Z", sampleBytes: 4096,
        memoryBytesPerSecond: 20 * 1024 ** 3, storageBytesPerSecond: 1024 ** 3, cacheHit: true } });
    const actual = plan(current), without = plan(omitted);
    expect(actual.resources.estimatedResidentModelBytes - without.resources.estimatedResidentModelBytes).toBe(201260);
    expect(actual.resources.minimumLayerResidencyBytes - without.resources.minimumLayerResidencyBytes).toBe(201260);
  });
});
