/** New pure mixed-plan inspection contracts. No worker, model, Load or device probing. */
import { describe, expect, it } from "vitest";
import { checkedNativeOperationCandidate } from "../src/main/nativeMixedRegionCandidate";
import { checkedNativeOperationPlan } from "../src/main/nativeOperationAutoService";
import { nativeComputeSha256 as hash } from "../src/main/cachedDecoderExecutionService";
import { nativeAutoCandidateLabel } from "../src/renderer/src/NativeOperationAutoControls";
import { NATIVE_OPERATION_AUTO_WORK_KINDS, type NativeOperationAutoWorkKind } from "../src/shared/nativeOperationAuto";
import type { DecoderExecutionCandidate } from "../src/shared/decoderExecution";
import type { NativeMixedRegionCandidate } from "../src/shared/nativeMixedRegions";

function base(device = "cpu"): DecoderExecutionCandidate {
  const payload = { executionDevice: device, deviceId: device + ":interface", memoryGroupId: device.startsWith("cuda:") ? "vram:0" : "ram:system",
    backendMode: device === "cpu" ? "native-cpu" as const : device === "mps" ? "native-metal" as const : "explicit-cuda-cpu-tile" as const,
    asyncDevices: device === "cpu" ? [] : [device], supported: true, fallbackAcknowledged: device.startsWith("cuda:") };
  return { ...payload, planSha256: hash(payload) };
}
function mixed(): NativeMixedRegionCandidate {
  const payload = { ...base(), backendMode: "native-mixed-sequential" as const, asyncDevices: ["cuda:0", "mps"], fallbackAcknowledged: true,
    regionSchedule: { version: 1 as const, registrySha256: "3".repeat(64), regionCount: 4, outerCandidate: base(),
      segments: [{ first: 0, count: 2, candidate: base("mps") }, { first: 2, count: 2, candidate: base("cuda:0") }] } };
  const { planSha256: _old, ...descriptor } = payload;
  return { ...descriptor, planSha256: hash(descriptor) };
}
function rehash(value: NativeMixedRegionCandidate): NativeMixedRegionCandidate {
  const { planSha256: _old, ...descriptor } = value;
  return { ...descriptor, planSha256: hash(descriptor) };
}
function plan(workKind: NativeOperationAutoWorkKind): Record<string, unknown> {
  const cached = workKind.includes("cached-whole-step"), prefix = workKind.endsWith(".full-prefix-sampling"), forward = workKind.endsWith(".uncached-full-forward"),
    media = workKind.startsWith("native.media."), training = workKind === "native.training.prepared-whole-wave";
  return { id: "a".repeat(64), workKind, geometrySha256: "b".repeat(64), qualifiedNativeStateSha256: "c".repeat(64), candidate: mixed(), qualifiedSamples: 3,
    qualificationWorkloads: 1, unseenInputOrFrontierWasPreviouslyMeasured: false, wholeBrainOrTrainingCovered: false,
    numericGate: forward ? "exact-full-forward-output-workspace-action-activity-equality"
      : prefix ? "exact-prefix-cache-connected-activity-neural-sampling-equality"
        : cached ? "exact-cache-connected-activity-neural-sampling-equality"
          : training ? "exact-prepared-wave-native-state-rng-and-effects-equality"
            : media ? "exact-media-neural-timeline-rng-and-encoded-artifact-equality" : "exact-embedding-equality",
    reuseBasis: forward ? "same-prepared-full-forward-current-owner-policy-no-cross-input-inference"
      : prefix ? "same-prepared-prefix-current-owner-policy-no-cross-prefix-inference"
        : cached ? "same-validated-operator-page-layout-range-current-owner-policy-inference"
          : training ? "same-prepared-wave-current-owner-policy-no-cross-wave-inference"
            : media ? "same-validated-media-operator-output-layout-current-owner-policy-inference" : "same-validated-operator-geometry-current-owner-policy-inference",
    ...(cached ? { contextRange: { minimumTokens: 1, maximumTokens: 128, queryTokens: 1,
      basis: "actual-pager-and-attention-tile-transitions", layoutSha256: "d".repeat(64) } } : {}),
    ...(prefix ? { prefillLayout: { shapeSha256: "b".repeat(64) } } : {}),
    ...(forward ? { forwardLayout: { shapeSha256: "b".repeat(64) } } : {}),
    ...(training ? { trainingLayout: { shapeSha256: "d".repeat(64) } } : {}),
    ...(media ? { mediaLayout: { modality: workKind.split(".")[2]!.split("-")[0], shapeSha256: "d".repeat(64) } } : {}) };
}

describe("operation-bound mixed sequential inspection descriptor", () => {
  it("preserves exact home identity, participant union and independent nested copies", () => {
    const value = mixed(), checked = checkedNativeOperationCandidate(value);
    expect(checked).toEqual(value); expect(checked.executionDevice).toBe("cpu"); expect(checked.memoryGroupId).toBe("ram:system");
    value.regionSchedule.segments[0]!.candidate.memoryGroupId = "wrong";
    expect(checked).not.toEqual(value);
    expect(checkedNativeOperationCandidate(base())).toEqual(base());
  });

  it("rejects home/participant/fallback substitutions even with recomputed public hashes", () => {
    for (const mutate of [
      (v: NativeMixedRegionCandidate) => { v.executionDevice = "mps"; },
      (v: NativeMixedRegionCandidate) => { v.memoryGroupId = "vram:fake"; },
      (v: NativeMixedRegionCandidate) => { v.asyncDevices = ["mps"]; },
      (v: NativeMixedRegionCandidate) => { v.fallbackAcknowledged = false; },
      (v: NativeMixedRegionCandidate) => { v.regionSchedule.segments[0]!.candidate.backendMode = "native-cpu"; },
    ]) {
      const value = mixed(); mutate(value);
      expect(() => checkedNativeOperationCandidate(rehash(value))).toThrow();
    }
    const raw = { ...mixed(), sourcePath: "/not-a-dispatch-grant" };
    expect(() => checkedNativeOperationCandidate(raw)).toThrow(/shape/);
  });

  it("requires nonempty exact bounded ordered ranges and actual integer geometry", () => {
    for (const [first, count] of [[-1, 1], [0, 0], [0, 5], [0.5, 1], [Number.MAX_SAFE_INTEGER, 1]]) {
      const value = mixed(); value.regionSchedule.segments[0] = { ...value.regionSchedule.segments[0]!, first: first!, count: count! };
      expect(() => checkedNativeOperationCandidate(rehash(value))).toThrow(/overlaps|exceeds/);
    }
    const overlap = mixed(); overlap.regionSchedule.segments[1]!.first = 1;
    expect(() => checkedNativeOperationCandidate(rehash(overlap))).toThrow(/overlaps/);
    const empty = mixed(); empty.regionSchedule.segments = [];
    expect(() => checkedNativeOperationCandidate(rehash(empty))).toThrow(/ranges/);
    const redundant = mixed(); redundant.regionSchedule.segments[0]!.candidate = base();
    expect(() => checkedNativeOperationCandidate(rehash(redundant))).toThrow(/redundant/);
  });

  it("admits only the six currently wired text kinds, not fictitious mixed modalities or training", () => {
    expect(NATIVE_OPERATION_AUTO_WORK_KINDS).toHaveLength(13);
    const text = NATIVE_OPERATION_AUTO_WORK_KINDS.filter(kind => kind.startsWith("native.decoder.") || kind.startsWith("native.connected-decoder."));
    expect(text).toHaveLength(6);
    for (const kind of NATIVE_OPERATION_AUTO_WORK_KINDS) {
      const raw = plan(kind);
      if (text.includes(kind as never)) expect(checkedNativeOperationPlan(raw)).toEqual(raw);
      else expect(() => checkedNativeOperationPlan(raw)).toThrow(/not qualified/);
    }
  });

  it("shows actual sequential participants without presenting a route as concurrent or all-brain", () => {
    expect(nativeAutoCandidateLabel(mixed())).toBe("Mixed sequential · CPU / Metal / CUDA device 0");
    expect(nativeAutoCandidateLabel(base())).toBe("CPU");
    expect(nativeAutoCandidateLabel(base("mps"))).toBe("Metal");
  });
});
