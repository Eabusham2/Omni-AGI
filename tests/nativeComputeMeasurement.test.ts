import { describe, expect, it, vi } from "vitest";
import { readFile } from "node:fs/promises";
import {
  createNativeComputeMeasurementCollector, validateNativeProjectionComputeProfile,
  type NativeProjectionComputeProfile
} from "../src/main/nativeComputeMeasurement";
import { balancedMeasuredNativeArchitectureProfile, capacityDerivedNativeArchitectureProfile,
  legacyNativeArchitectureDescriptor } from "../src/main/omniArchitectureProfile";
import { nativeCoreInventory } from "../src/main/nativeCoreInventory";
import { ResourcePlanner } from "../src/main/resourcePlanner";
import type { EngineSupervisor } from "../src/main/engineSupervisor";

const gib = 1024 ** 3;
const date = "2026-09-30T12:00:00.000Z";
function measurement(medianRunMicroseconds = 1000): NativeProjectionComputeProfile {
  return {
    format: "omni-bounded-packed-projection-compute-profile", formatVersion: 1,
    measuredAt: date, requestedDevice: "cpu", actualDevice: "cpu",
    kernel: "omni_core.model._packed_ternary_forward",
    kernelRevision: "float32-activity-int8-quantization-packed-rows-integer-dispatch-v1",
    activityDtype: "float32", packedDtype: "uint8", quantizedActivityDtype: "int8", integerResultDtype: "int32", intermediateAccumulationDtypeVerified: false, outputDtype: "float32",
    activityRows: 8, inputFeatures: 64, outputFeatures: 64, sampleTensorBytes: 5124,
    scratchReserveBytes: 1048576, admittedBytes: 1053700, runs: 4, projectionMacsPerRun: 32768,
    medianRunMicroseconds, projectionMacsPerSecond: Math.floor(32768 * 1_000_000 / medianRunMicroseconds),
    elapsedMicroseconds: medianRunMicroseconds * 4, fallbackReason: null,
    samplingBudgetMicroseconds: 250000, hardWallClockDeadline: false,
    warmupPerformed: true, implementation: "native-packed-kernel", backendInitializationIncluded: false,
    integerBackend: "runtime-dispatch-may-use-bounded-cpu-fallback",
    neuralModelConstructed: false, neuralQualityMeasured: false, fullNeuralThroughputMeasured: false,
    evidence: "cache-tile-packed-primitive-not-complete-neural-throughput"
  };
}
const input = { hardwareTier: "personal" as const, workingMemoryItems: 32768,
  selectedSystemRamBudgetBytes: 8 * gib, runtimeBaselineReserveBytes: 256 * 1024 ** 2,
  selectedStoragePoolBytes: 64 * gib, measuredStorageBytesPerSecond: gib,
  estimatedTrainingSourceBytes: 0, acceleratorAvailable: false };
const dependencies = {
  readResources: async () => ({ totalMemoryBytes: 16 * gib, availableMemoryBytes: 12 * gib,
    diskTotalBytes: 512 * gib, diskFreeBytes: 400 * gib }),
  benchmark: async () => ({ measuredAt: date, sampleBytes: 16 * 1024 ** 2,
    memoryBytesPerSecond: 12 * gib, storageBytesPerSecond: gib, cacheHit: true })
};
const request = { mode: "manual" as const, requestedItems: "32768", storagePoolMode: "manual" as const,
  storagePoolBytes: String(64 * gib) };

describe("trusted optional actual packed-primitive profiling, without probes", () => {
  it("validates exact kernel, dtypes, geometry, timing and no-neural evidence", () => {
    expect(validateNativeProjectionComputeProfile(measurement()).projectionMacsPerRun).toBe(32768);
    for (const changed of [{ kernel: "int32-matmul" }, { activityDtype: "float16" }, { neuralQualityMeasured: true },
      { projectionMacsPerSecond: 1 }, { implementation: "injected-fixture" }, { actualDevice: "cuda:0" }]) {
      expect(() => validateNativeProjectionComputeProfile({ ...measurement(), ...changed })).toThrow();
    }
  });

  it("does no RPC on construction, measures only explicit hardware calls and caches validated copies", async () => {
    const rpc = vi.fn(async (_method: string, _params: Record<string, unknown>) => measurement());
    const collector = createNativeComputeMeasurementCollector({ request: rpc } as unknown as EngineSupervisor, () => new Date(date));
    expect(rpc).not.toHaveBeenCalled();
    const selection = { device: "cpu", hardwareTier: "personal" as const, ramBudgetBytes: 8 * gib };
    const first = await collector(selection);
    const second = await collector(selection);
    expect(rpc).toHaveBeenCalledOnce();
    expect(rpc.mock.calls[0]?.[0]).toBe("hardware_projection_profile");
    expect(rpc.mock.calls[0]?.[1]).toEqual(selection);
    expect(second).toEqual(first);
    expect(second).not.toBe(first);
  });

  it("deduplicates in-flight primitive work and validates resource-only requests before RPC", async () => {
    let finish!: (profile: NativeProjectionComputeProfile) => void;
    const running = new Promise<NativeProjectionComputeProfile>((resolve) => { finish = resolve; });
    const rpc = vi.fn(async () => running);
    const collector = createNativeComputeMeasurementCollector({ request: rpc } as unknown as EngineSupervisor, () => new Date(date));
    const selection = { device: "cpu", hardwareTier: "personal" as const, ramBudgetBytes: 8 * gib };
    const first = collector(selection), second = collector(selection);
    expect(rpc).toHaveBeenCalledOnce();
    finish(measurement());
    expect(await first).toEqual(await second);
    for (const invalid of [{ ...selection, ramBudgetBytes: 0 }, { ...selection, ramBudgetBytes: Infinity },
      { ...selection, device: "other-model" }, { ...selection, modelPath: "/forged" }]) {
      expect(await collector(invalid)).toBeUndefined();
    }
    expect(rpc).toHaveBeenCalledOnce();
  });

  it("does not fabricate a rate for resource deferral, unavailable kernels or invalid observations", async () => {
    for (const value of [{ available: false }, { ...measurement(), projectionMacsPerSecond: -1 }]) {
      const collector = createNativeComputeMeasurementCollector({ request: vi.fn(async () => value) } as unknown as EngineSupervisor);
      expect(await collector({ device: "cpu", hardwareTier: "personal", ramBudgetBytes: 8 * gib })).toBeUndefined();
    }
  });

  it("defers occupied production workers without queuing or preempting a profile", async () => {
    const request = vi.fn(async (..._args: unknown[]) => { throw new Error("background work deferred"); });
    const collector = createNativeComputeMeasurementCollector({ request } as unknown as EngineSupervisor);
    expect(await collector({ device: "cpu", hardwareTier: "personal", ramBudgetBytes: 8 * gib })).toBeUndefined();
    expect(request).toHaveBeenCalledOnce();
    expect(request.mock.calls[0]?.[4]).toBe("background");
  });

  it("uses measured primitive rate only under an explicit main work-proxy budget", () => {
    const slow = balancedMeasuredNativeArchitectureProfile(input, measurement(2000), 100000);
    const fast = balancedMeasuredNativeArchitectureProfile(input, measurement(500), 100000);
    const physical = capacityDerivedNativeArchitectureProfile(input);
    expect(fast.dModel).toBeGreaterThan(slow.dModel);
    expect(fast.dModel).toBeLessThan(physical.dModel);
    expect(fast.nativeArchitecture?.inventory).toEqual(nativeCoreInventory(fast.nativeArchitecture!.shape));
    expect(fast.nativeArchitecture?.sizing).toMatchObject({ isAutomaticDefault: false, qualityOptimizationVerified: false,
      neuralTokensPerSecondMeasured: false, sourceCorpusTimeEstimated: false,
      selectionMode: "balanced-measured-primitive-candidate", mainSelectedPrimitiveWorkBudgetMicroseconds: 100000 });
    expect(fast.nativeArchitecture?.sizing).not.toHaveProperty("modelCoreFractionPercent");
    expect(fast.nativeArchitecture?.sizing.traversalWorkBasis).toContain("not-actual-chat-path");
  });

  it("preserves current Build shape/availability and never profiles default or existing checkpoints", async () => {
    const collector = vi.fn(async () => measurement());
    const planner = new ResourcePlanner("/fixture-not-probed", { ...dependencies, measurementCollector: collector });
    const ordinary = await planner.plan(request, { hardwareTier: "personal" });
    expect(ordinary.nativeArchitecture?.shape.dModel).toBeGreaterThan(legacyNativeArchitectureDescriptor("personal", ordinary.selectedItems).shape.dModel);
    expect(ordinary.nativeArchitecture?.sizing.selectionMode).toBe("ram-first-headroom-default");
    expect(collector).not.toHaveBeenCalled();
    const existing = await planner.plan(request, { hardwareTier: "personal", modelBytes: 64 * 1024 ** 2,
      nativeSizingMode: "balanced-measured", nativePrimitiveWorkBudgetMicroseconds: 100000 });
    expect(existing.nativeArchitecture).toBeUndefined();
    expect(collector).not.toHaveBeenCalled();
  });

  it("falls back to shipped policy instead of a new Build blockade when optional measurement is absent", async () => {
    const planner = new ResourcePlanner("/fixture-not-probed", { ...dependencies, measurementCollector: async () => undefined });
    const ordinary = await planner.plan(request, { hardwareTier: "personal" });
    const missing = await planner.plan(request, { hardwareTier: "personal", nativeSizingMode: "balanced-measured",
      nativePrimitiveWorkBudgetMicroseconds: 100000 });
    expect(missing.allowed).toBe(ordinary.allowed);
    expect(missing.nativeArchitecture?.shape).toEqual(ordinary.nativeArchitecture?.shape);
    expect(missing.warnings.join(" ")).toContain("shipped native shape policy remains available");
  });

  it("does not invent a default work target or block Build on an unmet optional proxy target", async () => {
    const collector = vi.fn(async () => measurement());
    const planner = new ResourcePlanner("/fixture-not-probed", { ...dependencies, measurementCollector: collector });
    const ordinary = await planner.plan(request, { hardwareTier: "personal" });
    const noBudget = await planner.plan(request, { hardwareTier: "personal", nativeSizingMode: "balanced-measured" });
    expect(collector).not.toHaveBeenCalled();
    expect(noBudget.nativeArchitecture?.shape).toEqual(ordinary.nativeArchitecture?.shape);
    const tooTight = await planner.plan(request, { hardwareTier: "personal", nativeSizingMode: "balanced-measured",
      nativePrimitiveWorkBudgetMicroseconds: 1 });
    expect(tooTight.allowed).toBe(ordinary.allowed);
    expect(tooTight.nativeArchitecture?.shape).toEqual(ordinary.nativeArchitecture?.shape);
    expect(tooTight.warnings.join(" ")).toContain("shipped native policy is retained, not blocked");
  });

  it("carries valid main-collected measurements into an explicitly selected fresh candidate", async () => {
    const collector = vi.fn(async () => measurement(500));
    const planner = new ResourcePlanner("/fixture-not-probed", { ...dependencies, measurementCollector: collector });
    const selected = await planner.plan(request, { hardwareTier: "personal", nativeSizingMode: "balanced-measured",
      nativeComputeDevice: "cpu", nativePrimitiveWorkBudgetMicroseconds: 100000 });
    expect(collector).toHaveBeenCalledOnce();
    expect(selected.nativeArchitecture?.sizing.selectionMode).toBe("balanced-measured-primitive-candidate");
    expect(selected.nativeArchitecture?.sizing.primitiveKernel).toBe("omni_core.model._packed_ternary_forward");
    expect(selected.nativeArchitecture?.sizing.isAutomaticDefault).toBe(false);
  });

  it("wires explicit future device profiling through trusted main, never renderer sizing fields", async () => {
    const [index, ipc] = await Promise.all([readFile("src/main/index.ts", "utf8"), readFile("src/main/ipc.ts", "utf8")]);
    expect(index).toContain("measurementCollector: createNativeComputeMeasurementCollector(engine)");
    expect(index).toContain("collectHardwareCompute:");
    expect(ipc).toContain("collectHardwareCompute");
    const route = ipc.slice(ipc.indexOf("handle(IPC.catalog.hardwareProfile"), ipc.indexOf("handle(IPC.catalog.resourcePlan"));
    expect(route).toContain("hardwareProfile()");
    expect(route).toContain("collectHardwareCompute");
    expect(route).toContain("void dependencies.collectHardwareCompute");
    expect(route).not.toContain("await dependencies.collectHardwareCompute");
    expect(route).not.toContain("requireWorkingMemoryPlanRequest");
  });
});
