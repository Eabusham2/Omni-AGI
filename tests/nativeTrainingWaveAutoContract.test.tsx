/** NEW inert host/presentation contracts only. No worker, model, training or benchmark. */
import React from "react";
import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { NativeOperationAutoService, checkedNativeOperationPlan, checkedNativeOperationStatus,
  type NativeOperationAutoDependencies } from "../src/main/nativeOperationAutoService";
import { nativeComputeSha256 as hash } from "../src/main/cachedDecoderExecutionService";
import { NATIVE_OPERATION_AUTO_WORK_KINDS, NATIVE_OPERATION_AUTO_STAGE_COVERAGE, NATIVE_TRAINING_WAVE_OPERATION_AUTO_STAGES,
  type NativeOperationAutoApi, type NativeOperationAutoPlan, type NativeOperationAutoReview,
  type NativeOperationAutoStatus } from "../src/shared/nativeOperationAuto";
import { NativeOperationAutoControls, nativeAutoStatusText, nativeAutoWorkKindLabel,
  NATIVE_AUTO_MEASUREMENT_SCOPE_TEXT, NATIVE_AUTO_REUSE_SCOPE_TEXT } from "../src/renderer/src/NativeOperationAutoControls";
import type { DeviceResourceTopology } from "../src/shared/deviceResources";

const KIND = "native.training.prepared-whole-wave" as const;
const topology: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1,
  measuredAt: "2026-10-18T00:00:00Z", platform: "inert-fixture", architecture: "inert-fixture",
  devices: [], links: [], measurementRequested: false, neuralModelConstructed: false, warnings: [] };
const descriptor = { executionDevice: "cpu", deviceId: "cpu:0", memoryGroupId: "ram:0", backendMode: "native-cpu" as const,
  asyncDevices: [] as string[], supported: true, fallbackAcknowledged: false };
const candidate = { ...descriptor, planSha256: hash(descriptor) };
const binding = { ownerSessionId: "wave-session", policySha256: "1".repeat(64), topologySha256: hash({ devices: [], links: [] }),
  registrySha256: "2".repeat(64), budgetSha256: "3".repeat(64), homeExecutionDevice: "cpu" };
const base = () => ({ id: "a".repeat(64), geometrySha256: "b".repeat(64), qualifiedNativeStateSha256: "c".repeat(64), candidate,
  qualifiedSamples: 3, qualificationWorkloads: 1 as const, unseenInputOrFrontierWasPreviouslyMeasured: false as const,
  wholeBrainOrTrainingCovered: false as const });
function plan(): Extract<NativeOperationAutoPlan, { workKind: typeof KIND }> {
  return { ...base(), workKind: KIND, numericGate: "exact-prepared-wave-native-state-rng-and-effects-equality",
    reuseBasis: "same-prepared-wave-current-owner-policy-no-cross-wave-inference", trainingLayout: { shapeSha256: "d".repeat(64) } };
}
function status(phase: NativeOperationAutoStatus["phase"], lastPlan: NativeOperationAutoPlan | null = null): NativeOperationAutoStatus {
  return { format: "omni-native-operation-auto-status", version: 2, ownerId: "brain-a", ownerSessionId: binding.ownerSessionId,
    phase, operationId: phase === "prepared" ? null : "wave-operation", reason: "inert prepared-wave source", progress: null,
    cancellationRequested: false, workKind: lastPlan?.workKind ?? null, qualifiedPlanCount: lastPlan ? 1 : 0,
    lastPlan, supportedWorkKinds: [KIND], unsupportedRoutes: ["unfenced-cross-rank-or-complete-training-checkpoint-pipeline"],
    globalPlacementChanged: false, calibrationOnAppOpen: false, wholeBrainOrTrainingMeasured: false,
    qualificationIsExact: true, reusableRouteIsInferenceForUnseenInputs: true };
}
function review(): NativeOperationAutoReview {
  return { format: "omni-native-operation-auto-review", version: 2, ownerId: "brain-a", binding,
    currentCandidate: candidate, candidates: [candidate], minimumSamples: 3,
    stageCoverageByWorkKind: [{ workKind: KIND, stages: [...NATIVE_TRAINING_WAVE_OPERATION_AUTO_STAGES] }],
    supportedWorkKinds: [KIND], connectedPerceptionSupported: false, status: status("prepared") };
}
function host() {
  let current = status("prepared");
  const dispatch = vi.fn<NativeOperationAutoDependencies["dispatch"]>(async (method, params, signal, context) => {
    await context.beforeDispatch?.(); signal.throwIfAborted();
    if (method === "native_operation_auto_prepare") return structuredClone(review()) as never;
    if (method === "native_operation_auto_arm") {
      current = { ...status("awaiting-operation"), operationId: String(params.operationId) };
      return structuredClone(current) as never;
    }
    throw new Error("Only metadata Prepare/Arm is allowed in this inert host.");
  });
  const control = vi.fn<NativeOperationAutoDependencies["control"]>(async method => {
    if (method === "native_operation_auto_plans") return { ownerId: "brain-a", ownerSessionId: binding.ownerSessionId,
      revision: 1, total: current.lastPlan ? 1 : 0, plans: current.lastPlan ? [current.lastPlan] : [] } as never;
    return structuredClone(current) as never;
  });
  const service = new NativeOperationAutoService({ requireLoaded: () => undefined,
    owner: async brainId => ({ brainId, storagePath: "/owned/brain-a", originSha256: "4".repeat(64),
      configSha256: "5".repeat(64), identitySha256: "6".repeat(64) }), topology: async () => structuredClone(topology), dispatch, control });
  return { service, dispatch, control, setStatus: (value: NativeOperationAutoStatus) => { current = value; } };
}

describe("native prepared learner-wave Auto host contract", () => {
  it("accepts and privately copies only the exact prepared-wave descriptor and gate", () => {
    const raw = plan(), checked = checkedNativeOperationPlan(raw);
    expect(checked).toEqual(raw);
    expect(checked.wholeBrainOrTrainingCovered).toBe(false);
    raw.trainingLayout.shapeSha256 = "f".repeat(64);
    expect(checked.trainingLayout?.shapeSha256).toBe("d".repeat(64));
    expect(checked).not.toHaveProperty("contextRange"); expect(checked).not.toHaveProperty("mediaLayout");
  });

  it("rejects missing, malformed and additional training layout fields rather than passing them through", () => {
    const raw = plan(), { trainingLayout: _layout, ...missing } = raw;
    expect(() => checkedNativeOperationPlan(missing)).toThrow(/shape/);
    for (const trainingLayout of [null, [], {}, "d".repeat(64), { shapeSha256: 7 }, { shapeSha256: "short" },
      { shapeSha256: "D".repeat(64) }, { shapeSha256: "d".repeat(64), modality: "image" },
      { shapeSha256: "d".repeat(64), sourcePath: "/not-authority" }, { shapeSha256: "d".repeat(64), physicalBatchRecords: 1 }])
      expect(() => checkedNativeOperationPlan({ ...raw, trainingLayout })).toThrow(/shape|layout/);
  });

  it("forbids media layouts, cached context ranges and added authority on a training plan", () => {
    const raw = plan();
    for (const changed of [{ mediaLayout: { modality: "image", shapeSha256: "d".repeat(64) } },
      { contextRange: { minimumTokens: 1, maximumTokens: 128, queryTokens: 1 } },
      { storagePath: "/not-authority" }, { ownerSessionId: "invented" }, { worldSize: 2 }])
      expect(() => checkedNativeOperationPlan({ ...raw, ...changed })).toThrow(/shape/);
  });

  it("refuses weaker gates, cross-wave inference and whole-training measurement claims", () => {
    const raw = plan();
    for (const changed of [{ numericGate: "exact-embedding-equality" },
      { numericGate: "exact-media-neural-timeline-rng-and-encoded-artifact-equality" },
      { reuseBasis: "same-validated-operator-geometry-current-owner-policy-inference" },
      { wholeBrainOrTrainingCovered: true }, { unseenInputOrFrontierWasPreviouslyMeasured: true },
      { qualificationWorkloads: 2 }, { qualifiedSamples: 2 }])
      expect(() => checkedNativeOperationPlan({ ...raw, ...changed })).toThrow(/overclaims/);
  });

  it("keeps trainingLayout forbidden on every existing perception, cached and media discriminator", () => {
    for (const kind of NATIVE_OPERATION_AUTO_WORK_KINDS.filter(kind => kind !== KIND &&
      !kind.endsWith(".full-prefix-sampling") && !kind.endsWith(".uncached-full-forward"))) {
      const media = kind.startsWith("native.media."), cached = kind.includes("cached-whole-step");
      const prior = { ...base(), workKind: kind,
        numericGate: media ? "exact-media-neural-timeline-rng-and-encoded-artifact-equality"
          : cached ? "exact-cache-connected-activity-neural-sampling-equality" : "exact-embedding-equality",
        reuseBasis: media ? "same-validated-media-operator-output-layout-current-owner-policy-inference"
          : cached ? "same-validated-operator-page-layout-range-current-owner-policy-inference"
            : "same-validated-operator-geometry-current-owner-policy-inference",
        ...(media ? { mediaLayout: { modality: kind.split(".")[2]!.split("-")[0], shapeSha256: "d".repeat(64) } } : {}),
        ...(cached ? { contextRange: { minimumTokens: 65, maximumTokens: 128, queryTokens: 1,
          basis: "actual-pager-and-attention-tile-transitions", layoutSha256: "e".repeat(64) } } : {}) };
      expect(() => checkedNativeOperationPlan({ ...prior, trainingLayout: { shapeSha256: "d".repeat(64) } })).toThrow(/shape/);
    }
  });

  it("prepares and arms the existing loaded-owner control for the training-only supported subset without running a job", async () => {
    const setup = host(), prepared = await setup.service.prepare({ brainId: "brain-a" });
    expect(prepared.supportedWorkKinds).toEqual([KIND]);
    expect(prepared.stageCoverageByWorkKind).toEqual([{ workKind: KIND, stages: [...NATIVE_TRAINING_WAVE_OPERATION_AUTO_STAGES] }]);
    await expect(setup.service.arm({ brainId: "brain-a", reviewId: prepared.reviewId })).resolves.toMatchObject({ phase: "awaiting-operation" });
    expect(setup.dispatch.mock.calls.map(call => call[0])).toEqual(["native_operation_auto_prepare", "native_operation_auto_arm"]);
    expect(setup.dispatch.mock.calls[1]![1]).toMatchObject({ reviewed: binding, storagePath: "/owned/brain-a" });
    expect(setup.dispatch.mock.calls[1]![1]).not.toHaveProperty("train");
  });

  it("rejects shortened, reordered or substituted training stage coverage", async () => {
    const stages = [...NATIVE_TRAINING_WAVE_OPERATION_AUTO_STAGES];
    for (const invalid of [stages.slice(0, -1), [...stages].reverse(), [...stages, stages[0]!],
      [...stages.slice(0, -1), "original-owner-restore"]]) {
      const setup = host(); setup.dispatch.mockResolvedValueOnce({ ...review(), stageCoverageByWorkKind: [{ workKind: KIND, stages: invalid }] } as never);
      await expect(setup.service.prepare({ brainId: "brain-a" })).rejects.toThrow(/stages/);
      expect(setup.dispatch).toHaveBeenCalledOnce();
    }
  });

  it("validates training plans through current-owner status and plan pagination", async () => {
    const setup = host(), current = status("active", plan()); setup.setStatus(current);
    await expect(setup.service.status("brain-a")).resolves.toMatchObject({ workKind: KIND, lastPlan: plan() });
    await expect(setup.service.plans({ brainId: "brain-a", limit: 1 })).resolves.toMatchObject({ plans: [plan()], total: 1 });
    setup.setStatus({ ...current, lastPlan: { ...plan(), trainingLayout: { shapeSha256: "wrong" } } });
    await expect(setup.service.status("brain-a")).rejects.toThrow(/layout/);
    expect(setup.dispatch).not.toHaveBeenCalled();
  });

  it("keeps whole-training and global-placement claims forbidden on training statuses", () => {
    for (const changed of [{ wholeBrainOrTrainingMeasured: true }, { globalPlacementChanged: true },
      { trainingLayout: { shapeSha256: "d".repeat(64) } }])
      expect(() => checkedNativeOperationStatus({ ...status("active", plan()), ...changed }, "brain-a")).toThrow(/shape|scope/);
  });

  it("matches the actual frozen native ninth-kind declaration without executing Python", () => {
    const source = readFileSync(new URL("../engine/omni_core/operation_auto_registry.py", import.meta.url), "utf8");
    const kinds = JSON.parse(source.match(/TRAINING_KINDS = (\[[^\n]+\])/)![1]!) as string[];
    const stages = JSON.parse(source.match(/TRAINING_STAGES = (\[[\s\S]*?\])\n/)![1]!) as string[];
    expect(kinds).toEqual([KIND]); expect(stages).toEqual(NATIVE_TRAINING_WAVE_OPERATION_AUTO_STAGES);
    expect(NATIVE_OPERATION_AUTO_WORK_KINDS.filter(kind => !kind.endsWith(".full-prefix-sampling") && !kind.endsWith(".uncached-full-forward"))).toHaveLength(9);
    expect(NATIVE_OPERATION_AUTO_STAGE_COVERAGE[KIND]).toEqual(stages);
  });
});

describe("prepared learner-wave Auto presentation only", () => {
  it("labels the single-wave operation and does not imply reuse or finished full training", () => {
    expect(nativeAutoWorkKindLabel(KIND)).toBe("Prepared training wave");
    expect(nativeAutoStatusText(status("qualifying", plan()))).toBe("Checking and applying one prepared training wave.");
    expect(nativeAutoStatusText(status("active", plan()))).toContain("future waves need a new check");
    expect(NATIVE_AUTO_REUSE_SCOPE_TEXT).toContain("no cross-wave reuse");
    expect(NATIVE_AUTO_MEASUREMENT_SCOPE_TEXT).toContain("Full training pipelines, distributed training, whole-brain and complete-reply placement are not covered");
    expect(NATIVE_AUTO_MEASUREMENT_SCOPE_TEXT).toContain("do not establish a total speedup");
  });

  it("keeps the existing collapsed control without a new mode, host call or automatic workload", () => {
    const api = new Proxy({}, { get() { throw new Error("Rendering cannot invoke host operations"); } }) as NativeOperationAutoApi;
    const html = renderToStaticMarkup(<NativeOperationAutoControls api={api} brainId="brain-a" loaded={false}
      foregroundTurnActive={false} dirty={false} />);
    expect(html).toContain("automatic operation placement"); expect(html).toContain("Route reuse depends on the operation");
    expect(html).not.toContain("<details open"); expect(html).not.toContain("Train now"); expect(html).not.toContain("training mode");
  });
});
