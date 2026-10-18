import { randomUUID } from "node:crypto";
import type { HardwareTier } from "../shared/types";
import type { EngineSupervisor } from "./engineSupervisor";

export const NATIVE_COMPUTE_PROFILE_FORMAT = "omni-bounded-packed-projection-compute-profile";
export const NATIVE_COMPUTE_KERNEL = "omni_core.model._packed_ternary_forward";
export const NATIVE_COMPUTE_KERNEL_REVISION = "float32-activity-int8-quantization-weighted-packed-rows-int64-dispatch-v2";

/** Mirrored by native_compute_profile.native_projection_tensor_budget, without a probe.
 * Base weighted contacts only: dynamic repeated-contact pages remain owner-admitted.
 * Known buffers are summed conservatively; this is not an exact allocator/RSS peak.
 */
export function nativeProjectionTensorBudget(activityRows: number, inputFeatures: number, outputFeatures: number) {
  if (![activityRows, inputFeatures, outputFeatures].every(value => Number.isSafeInteger(value) && value > 0)) {
    throw new Error("Native projection tensor geometry requires positive safe integers.");
  }
  const values = {
    inputActivityBytes: activityRows * inputFeatures * 4,
    packedSignBytes: outputFeatures * Math.ceil(inputFeatures / 4),
    connectionStateBytes: outputFeatures * inputFeatures,
    weightScaleBytes: 4,
    quantizedActivityBytes: activityRows * inputFeatures,
    activationScaleBytes: activityRows * 4,
    integerRowBlockBytes: activityRows * outputFeatures * 8,
    integerConcatenationBytes: activityRows * outputFeatures * 8,
    floatingOutputWorkingBytes: activityRows * outputFeatures * 4 * 2,
    quantizationWorkingBytes: activityRows * inputFeatures * 4 * 2,
    backendDescriptorBytes: 12,
  };
  const sampleTensorBytes = values.inputActivityBytes + values.packedSignBytes + values.connectionStateBytes + values.weightScaleBytes;
  const forwardTensorReserveBytes = values.quantizedActivityBytes + values.activationScaleBytes + values.integerRowBlockBytes
    + values.integerConcatenationBytes + values.floatingOutputWorkingBytes + values.quantizationWorkingBytes + values.backendDescriptorBytes;
  const budget = { ...values, sampleTensorBytes, forwardTensorReserveBytes };
  if (!Object.values(budget).every(Number.isSafeInteger)) throw new Error("Native projection tensor bytes exceed exact integer accounting.");
  return budget;
}

export const NATIVE_COMPUTE_TENSOR_BUDGET = Object.freeze(nativeProjectionTensorBudget(8, 64, 64));
export const NATIVE_COMPUTE_SCRATCH_RESERVE_BYTES = 1048576;
export const NATIVE_COMPUTE_ADMITTED_BYTES = NATIVE_COMPUTE_TENSOR_BUDGET.sampleTensorBytes
  + NATIVE_COMPUTE_TENSOR_BUDGET.forwardTensorReserveBytes + NATIVE_COMPUTE_SCRATCH_RESERVE_BYTES;

/** Trusted main dependency only. No renderer shapes, prompts or model paths. */
export interface NativeComputeMeasurementRequest {
  device: string;
  ramBudgetBytes: number;
  hardwareTier: HardwareTier;
}

export interface NativeProjectionComputeProfile extends ReturnType<typeof nativeProjectionTensorBudget> {
  format: typeof NATIVE_COMPUTE_PROFILE_FORMAT;
  formatVersion: 2;
  measuredAt: string;
  requestedDevice: string;
  actualDevice: string;
  kernel: typeof NATIVE_COMPUTE_KERNEL;
  kernelRevision: typeof NATIVE_COMPUTE_KERNEL_REVISION;
  activityDtype: "float32";
  packedDtype: "uint8";
  quantizedActivityDtype: "int8";
  integerResultDtype: "int64";
  intermediateAccumulationDtypeVerified: false;
  outputDtype: "float32";
  connectionStateDtype: "uint8";
  connectionStateEncoding: "strength-low4-resistance-high4";
  repeatedContacts: 0;
  repeatedContactPages: 0;
  repeatedContactTimingMeasured: false;
  activityRows: 8;
  inputFeatures: 64;
  outputFeatures: 64;
  scratchReserveBytes: typeof NATIVE_COMPUTE_SCRATCH_RESERVE_BYTES;
  admittedBytes: number;
  tensorReserveIsExactPeak: false;
  resourceByteBasis: "known-forward-tensors-plus-allocator-initialization-reserve";
  runs: number;
  projectionMacsPerRun: 32768;
  medianRunMicroseconds: number;
  projectionMacsPerSecond: number;
  elapsedMicroseconds: number;
  samplingBudgetMicroseconds: 250000;
  hardWallClockDeadline: false;
  fallbackReason: string | null;
  warmupPerformed: true;
  implementation: "native-packed-kernel";
  integerBackend: "runtime-dispatch-may-use-bounded-cpu-fallback";
  backendInitializationIncluded: false;
  neuralModelConstructed: false;
  neuralQualityMeasured: false;
  fullNeuralThroughputMeasured: false;
  evidence: "cache-tile-packed-primitive-not-complete-neural-throughput";
}

export type NativeComputeMeasurementCollector =
  (request: NativeComputeMeasurementRequest) => Promise<NativeProjectionComputeProfile | undefined>;

export function validateNativeProjectionComputeProfile(value: unknown): NativeProjectionComputeProfile {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("Missing native compute measurement.");
  const record = value as Record<string, unknown>;
  const exact = {
    format: NATIVE_COMPUTE_PROFILE_FORMAT, formatVersion: 2, kernel: NATIVE_COMPUTE_KERNEL,
    kernelRevision: NATIVE_COMPUTE_KERNEL_REVISION, activityDtype: "float32", packedDtype: "uint8",
    quantizedActivityDtype: "int8", integerResultDtype: "int64", intermediateAccumulationDtypeVerified: false, outputDtype: "float32",
    connectionStateDtype: "uint8", connectionStateEncoding: "strength-low4-resistance-high4",
    repeatedContacts: 0, repeatedContactPages: 0, repeatedContactTimingMeasured: false,
    activityRows: 8, inputFeatures: 64, outputFeatures: 64, ...NATIVE_COMPUTE_TENSOR_BUDGET,
    scratchReserveBytes: NATIVE_COMPUTE_SCRATCH_RESERVE_BYTES, admittedBytes: NATIVE_COMPUTE_ADMITTED_BYTES, projectionMacsPerRun: 32768,
    tensorReserveIsExactPeak: false, resourceByteBasis: "known-forward-tensors-plus-allocator-initialization-reserve",
    warmupPerformed: true, implementation: "native-packed-kernel",
    integerBackend: "runtime-dispatch-may-use-bounded-cpu-fallback", backendInitializationIncluded: false,
    neuralModelConstructed: false, neuralQualityMeasured: false, fullNeuralThroughputMeasured: false,
    evidence: "cache-tile-packed-primitive-not-complete-neural-throughput",
    samplingBudgetMicroseconds: 250000, hardWallClockDeadline: false
  };
  if (Object.entries(exact).some(([key, expected]) => record[key] !== expected)) throw new Error("Native compute kernel/scope evidence is invalid.");
  if (typeof record.measuredAt !== "string" || !Number.isFinite(Date.parse(record.measuredAt)) ||
      typeof record.requestedDevice !== "string" || !/^(cpu|mps|directml|cuda(?::[0-9]+)?)$/.test(record.requestedDevice) ||
      typeof record.actualDevice !== "string" || !/^(cpu|mps(?::[0-9]+)?|privateuseone(?::[0-9]+)?|cuda(?::[0-9]+)?)$/.test(record.actualDevice) ||
      !["runs", "medianRunMicroseconds", "projectionMacsPerSecond", "elapsedMicroseconds"]
        .every((key) => typeof record[key] === "number" && Number.isSafeInteger(record[key]) && Number(record[key]) > 0) ||
      Number(record.runs) > 4 ||
      record.projectionMacsPerSecond !== Math.max(1, Math.floor(32768 * 1_000_000 / Number(record.medianRunMicroseconds))) ||
      Number(record.elapsedMicroseconds) < Number(record.medianRunMicroseconds) ||
      !(record.fallbackReason === null || typeof record.fallbackReason === "string" && record.fallbackReason.length > 0)) {
    throw new Error("Native compute measurement geometry/timing is invalid.");
  }
  const requested = String(record.requestedDevice), actual = String(record.actualDevice);
  const matched = requested === "directml" ? actual.startsWith("privateuseone") :
    requested === "mps" ? actual.startsWith("mps") : requested === actual;
  if (!matched && !(actual === "cpu" && typeof record.fallbackReason === "string" && record.fallbackReason)) {
    throw new Error("Native compute requested/actual device evidence is invalid.");
  }
  return { ...record } as unknown as NativeProjectionComputeProfile;
}

/** Explicit future profiling only; constructing this adapter performs no RPC. */
export function createNativeComputeMeasurementCollector(
  engine: Pick<EngineSupervisor, "request">,
  now: () => Date = () => new Date()
): NativeComputeMeasurementCollector {
  const cache = new Map<string, NativeProjectionComputeProfile>();
  const inFlight = new Map<string, Promise<NativeProjectionComputeProfile | undefined>>();
  return async (request) => {
    if (typeof request !== "object" || request === null || Array.isArray(request) ||
        Object.keys(request).some((key) => !["device", "ramBudgetBytes", "hardwareTier"].includes(key)) ||
        typeof request.device !== "string" || !/^(cpu|mps|directml|cuda(?::[0-9]+)?)$/.test(request.device) ||
        !["micro", "personal", "gpu", "workstation"].includes(request.hardwareTier) ||
        !Number.isSafeInteger(request.ramBudgetBytes) || request.ramBudgetBytes < NATIVE_COMPUTE_ADMITTED_BYTES) return undefined;
    const key = `${NATIVE_COMPUTE_KERNEL_REVISION}:${request.device}:${request.hardwareTier}`;
    const prior = cache.get(key);
    const age = prior ? now().getTime() - Date.parse(prior.measuredAt) : Infinity;
    if (prior && age >= 0 && age < 24 * 60 * 60 * 1000) return { ...prior };
    let pending = inFlight.get(key);
    if (!pending) {
      pending = (async () => {
        try {
          const context = { requestId: `compute-profile-${randomUUID()}`, owner: "system" as const,
            label: "Bounded native projection hardware profiling" };
          // Background request admission refuses occupied workers before
          // queueing. Catch here instead of tryRequest's global lastError so
          // an expected calibration deferral is not a brain/runtime failure.
          const value = await engine.request<unknown>("hardware_projection_profile", { ...request }, 30_000,
            undefined, "background", context);
          const profile = validateNativeProjectionComputeProfile(value);
          if (profile.requestedDevice !== request.device) return undefined;
          cache.set(key, { ...profile });
          return profile;
        } catch {
          // Missing/deferred measurements never become fabricated speed or
          // a new Build readiness gate. A later request may retry safely.
          return undefined;
        }
      })();
      inFlight.set(key, pending);
    }
    try {
      const profile = await pending;
      return profile ? { ...profile } : undefined;
    } finally {
      if (inFlight.get(key) === pending) inFlight.delete(key);
    }
  };
}
