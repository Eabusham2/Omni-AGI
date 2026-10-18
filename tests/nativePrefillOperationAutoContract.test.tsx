/** NEW inert host/label contracts only. No Python execution, worker, model, Load or benchmark. */
import React from "react";
import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { NativeOperationAutoService, checkedNativeOperationPlan, checkedNativeOperationStatus,
  type NativeOperationAutoDependencies } from "../src/main/nativeOperationAutoService";
import { nativeComputeSha256 as hash } from "../src/main/cachedDecoderExecutionService";
import { NATIVE_OPERATION_AUTO_WORK_KINDS, NATIVE_OPERATION_AUTO_STAGE_COVERAGE, NATIVE_PREFILL_OPERATION_AUTO_STAGES,
  type NativeOperationAutoApi, type NativeOperationAutoPlan, type NativeOperationAutoReview,
  type NativeOperationAutoStatus } from "../src/shared/nativeOperationAuto";
import { NativeOperationAutoControls, nativeAutoStatusText, nativeAutoWorkKindLabel,
  NATIVE_AUTO_MEASUREMENT_SCOPE_TEXT, NATIVE_AUTO_REUSE_SCOPE_TEXT } from "../src/renderer/src/NativeOperationAutoControls";
import type { DeviceResourceTopology } from "../src/shared/deviceResources";

const KINDS = ["native.decoder.full-prefix-sampling", "native.connected-decoder.full-prefix-sampling"] as const;
type Kind = typeof KINDS[number];
const topology: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1,
  measuredAt: "2026-10-18T00:00:00Z", platform: "inert-fixture", architecture: "inert-fixture",
  devices: [], links: [], measurementRequested: false, neuralModelConstructed: false, warnings: [] };
const descriptor = { executionDevice: "cpu", deviceId: "cpu:0", memoryGroupId: "ram:0", backendMode: "native-cpu" as const,
  asyncDevices: [] as string[], supported: true, fallbackAcknowledged: false };
const candidate = { ...descriptor, planSha256: hash(descriptor) };
const binding = { ownerSessionId: "prefix-session", policySha256: "1".repeat(64), topologySha256: hash({ devices: [], links: [] }),
  registrySha256: "2".repeat(64), budgetSha256: "3".repeat(64), homeExecutionDevice: "cpu" };
const base = () => ({ id: "a".repeat(64), geometrySha256: "b".repeat(64), qualifiedNativeStateSha256: "c".repeat(64), candidate,
  qualifiedSamples: 3, qualificationWorkloads: 1 as const, unseenInputOrFrontierWasPreviouslyMeasured: false as const,
  wholeBrainOrTrainingCovered: false as const });
function plan(kind: Kind = KINDS[0]): Extract<NativeOperationAutoPlan, { workKind: Kind }> {
  return { ...base(), workKind: kind, numericGate: "exact-prefix-cache-connected-activity-neural-sampling-equality",
    reuseBasis: "same-prepared-prefix-current-owner-policy-no-cross-prefix-inference", prefillLayout: { shapeSha256: "b".repeat(64) } };
}
function status(phase: NativeOperationAutoStatus["phase"], lastPlan: NativeOperationAutoPlan | null = null): NativeOperationAutoStatus {
  return { format: "omni-native-operation-auto-status", version: 2, ownerId: "brain-a", ownerSessionId: binding.ownerSessionId,
    phase, operationId: phase === "prepared" ? null : "prefix-operation", reason: "inert full-prefix source", progress: null,
    cancellationRequested: false, workKind: lastPlan?.workKind ?? null, qualifiedPlanCount: lastPlan ? 1 : 0,
    lastPlan, supportedWorkKinds: [...KINDS], unsupportedRoutes: ["general-retained-KV-whole-turn", "router-STDP-and-fast-memory-learning"],
    globalPlacementChanged: false, calibrationOnAppOpen: false, wholeBrainOrTrainingMeasured: false,
    qualificationIsExact: true, reusableRouteIsInferenceForUnseenInputs: true };
}
function review(): NativeOperationAutoReview {
  return { format: "omni-native-operation-auto-review", version: 2, ownerId: "brain-a", binding,
    currentCandidate: candidate, candidates: [candidate], minimumSamples: 3,
    stageCoverageByWorkKind: KINDS.map(workKind => ({ workKind, stages: [...NATIVE_PREFILL_OPERATION_AUTO_STAGES] })),
    supportedWorkKinds: [...KINDS], connectedPerceptionSupported: false, status: status("prepared") };
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

describe("native actual full-prefix Auto host contract", () => {
  it.each(KINDS)("accepts and privately copies exact %s prefix geometry only", kind => {
    const raw = plan(kind), checked = checkedNativeOperationPlan(raw);
    expect(checked).toEqual(raw); expect(checked.wholeBrainOrTrainingCovered).toBe(false);
    raw.prefillLayout.shapeSha256 = "f".repeat(64);
    expect(checked.prefillLayout?.shapeSha256).toBe("b".repeat(64));
    expect(checked).not.toHaveProperty("contextRange"); expect(checked).not.toHaveProperty("mediaLayout");
    expect(checked).not.toHaveProperty("trainingLayout");
  });

  it("rejects missing, malformed, additional or mismatched prefix geometry", () => {
    const raw = plan(), { prefillLayout: _layout, ...missing } = raw;
    expect(() => checkedNativeOperationPlan(missing)).toThrow(/shape/);
    for (const prefillLayout of [null, [], {}, "b".repeat(64), { shapeSha256: 7 }, { shapeSha256: "short" },
      { shapeSha256: "B".repeat(64) }, { shapeSha256: "d".repeat(64) }, { shapeSha256: "b".repeat(64), queryTokens: 1 },
      { shapeSha256: "b".repeat(64), sourcePath: "/not-authority" }])
      expect(() => checkedNativeOperationPlan({ ...raw, prefillLayout })).toThrow(/shape|geometry/);
  });

  it("forbids cached, media and training layouts or authority on prefix plans", () => {
    for (const kind of KINDS) for (const changed of [{ contextRange: { minimumTokens: 1, maximumTokens: 128, queryTokens: 1 } },
      { mediaLayout: { modality: "image", shapeSha256: "b".repeat(64) } }, { trainingLayout: { shapeSha256: "b".repeat(64) } },
      { storagePath: "/not-authority" }, { ownerSessionId: "invented" }, { applyToFuturePrefixes: true }])
      expect(() => checkedNativeOperationPlan({ ...plan(kind), ...changed })).toThrow(/shape/);
  });

  it("refuses weaker gates, cross-prefix inference and whole-reply claims", () => {
    for (const changed of [{ numericGate: "exact-cache-connected-activity-neural-sampling-equality" },
      { numericGate: "exact-embedding-equality" }, { reuseBasis: "same-validated-operator-page-layout-range-current-owner-policy-inference" },
      { wholeBrainOrTrainingCovered: true }, { unseenInputOrFrontierWasPreviouslyMeasured: true },
      { qualificationWorkloads: 2 }, { qualifiedSamples: 2 }])
      expect(() => checkedNativeOperationPlan({ ...plan(), ...changed })).toThrow(/overclaims/);
  });

  it("keeps prefillLayout forbidden on all nine prior work kinds", () => {
    for (const kind of NATIVE_OPERATION_AUTO_WORK_KINDS.filter(kind => !KINDS.includes(kind as Kind) && !kind.endsWith(".uncached-full-forward"))) {
      const media = kind.startsWith("native.media."), cached = kind.includes("cached-whole-step"), training = kind === "native.training.prepared-whole-wave";
      const old = { ...base(), workKind: kind,
        numericGate: training ? "exact-prepared-wave-native-state-rng-and-effects-equality"
          : media ? "exact-media-neural-timeline-rng-and-encoded-artifact-equality"
            : cached ? "exact-cache-connected-activity-neural-sampling-equality" : "exact-embedding-equality",
        reuseBasis: training ? "same-prepared-wave-current-owner-policy-no-cross-wave-inference"
          : media ? "same-validated-media-operator-output-layout-current-owner-policy-inference"
            : cached ? "same-validated-operator-page-layout-range-current-owner-policy-inference"
              : "same-validated-operator-geometry-current-owner-policy-inference",
        ...(media ? { mediaLayout: { modality: kind.split(".")[2]!.split("-")[0], shapeSha256: "d".repeat(64) } } : {}),
        ...(cached ? { contextRange: { minimumTokens: 1, maximumTokens: 128, queryTokens: 1,
          basis: "actual-pager-and-attention-tile-transitions", layoutSha256: "e".repeat(64) } } : {}),
        ...(training ? { trainingLayout: { shapeSha256: "d".repeat(64) } } : {}) };
      expect(checkedNativeOperationPlan(old)).toEqual(old);
      expect(() => checkedNativeOperationPlan({ ...old, prefillLayout: { shapeSha256: "b".repeat(64) } })).toThrow(/shape/);
    }
  });

  it("prepares and arms the two current-owner prefix kinds using metadata only", async () => {
    const setup = host(), prepared = await setup.service.prepare({ brainId: "brain-a" });
    expect(prepared.supportedWorkKinds).toEqual(KINDS);
    expect(prepared.stageCoverageByWorkKind).toEqual(KINDS.map(workKind => ({ workKind, stages: [...NATIVE_PREFILL_OPERATION_AUTO_STAGES] })));
    await expect(setup.service.arm({ brainId: "brain-a", reviewId: prepared.reviewId })).resolves.toMatchObject({ phase: "awaiting-operation" });
    expect(setup.dispatch.mock.calls.map(call => call[0])).toEqual(["native_operation_auto_prepare", "native_operation_auto_arm"]);
    expect(setup.dispatch.mock.calls[1]![1]).toMatchObject({ reviewed: binding, storagePath: "/owned/brain-a" });
    expect(setup.dispatch.mock.calls[1]![1]).not.toHaveProperty("generate");
  });

  it("rejects missing, reordered or foreign prefix stages", async () => {
    const complete = [...NATIVE_PREFILL_OPERATION_AUTO_STAGES];
    for (const stages of [complete.slice(0, -1), [...complete].reverse(), [...complete, complete[0]!],
      [...complete.slice(0, -1), "same-home-RNG-input-and-activity-transfer"]]) {
      const setup = host(); setup.dispatch.mockResolvedValueOnce({ ...review(), stageCoverageByWorkKind:
        [{ workKind: KINDS[0], stages }, { workKind: KINDS[1], stages: complete }] } as never);
      await expect(setup.service.prepare({ brainId: "brain-a" })).rejects.toThrow(/stages/);
      expect(setup.dispatch).toHaveBeenCalledOnce();
    }
  });

  it("passes exact current-prefix plans through owner-bound status and pagination", async () => {
    for (const kind of KINDS) {
      const setup = host(); setup.setStatus(status("active", plan(kind)));
      await expect(setup.service.status("brain-a")).resolves.toMatchObject({ workKind: kind, lastPlan: plan(kind) });
      await expect(setup.service.plans({ brainId: "brain-a", limit: 1 })).resolves.toMatchObject({ plans: [plan(kind)], total: 1 });
      setup.setStatus({ ...status("active", plan(kind)), lastPlan: { ...plan(kind), prefillLayout: { shapeSha256: "wrong" } } });
      await expect(setup.service.status("brain-a")).rejects.toThrow(/geometry/);
      expect(setup.dispatch).not.toHaveBeenCalled();
    }
  });

  it("keeps global, whole-reply and pipeline measurement scope false", () => {
    for (const changed of [{ wholeBrainOrTrainingMeasured: true }, { globalPlacementChanged: true },
      { prefillLayout: { shapeSha256: "b".repeat(64) } }])
      expect(() => checkedNativeOperationStatus({ ...status("active", plan()), ...changed }, "brain-a")).toThrow(/shape|scope/);
  });

  it("matches current native kinds and both exact eight-stage prefix declarations by source read only", () => {
    const source = readFileSync(new URL("../engine/omni_core/operation_auto_registry.py", import.meta.url), "utf8");
    const kinds = JSON.parse(source.match(/PREFILL_KINDS = (\[[^\n]+\])/)![1]!) as string[];
    const stages = JSON.parse(source.match(/PREFILL_STAGES = (\[[\s\S]*?\])\n/)![1]!) as string[];
    expect(kinds).toEqual(KINDS); expect(stages).toEqual(NATIVE_PREFILL_OPERATION_AUTO_STAGES);
    expect(NATIVE_OPERATION_AUTO_WORK_KINDS).toHaveLength(13);
    for (const kind of KINDS) expect(NATIVE_OPERATION_AUTO_STAGE_COVERAGE[kind]).toEqual(stages);
    expect(source).toContain("kind: PREFILL_STAGES for kind in PREFILL_KINDS");
    const nativePolicy = readFileSync(new URL("../engine/omni_core/prefill_whole_operation_auto.py", import.meta.url), "utf8");
    expect(nativePolicy).toContain(`PREFILL_NUMERIC_GATE = "${plan().numericGate}"`);
    expect(nativePolicy).toContain(`PREFILL_REUSE_BASIS = "${plan().reuseBasis}"`);
  });
});

describe("full-prefix Auto existing-control presentation", () => {
  it("labels prefix-only checks separately from retained steps and complete replies", () => {
    expect(nativeAutoWorkKindLabel(KINDS[0])).toBe("Full decoder prefix");
    expect(nativeAutoWorkKindLabel(KINDS[1])).toBe("Full connected prefix");
    expect(nativeAutoStatusText(status("qualifying", plan()))).toBe("Comparing one actual full decoder prefix.");
    expect(nativeAutoStatusText(status("active", plan(KINDS[1])))).toContain("another prefix needs a new check");
    expect(NATIVE_AUTO_REUSE_SCOPE_TEXT).toContain("no cross-prefix reuse");
    expect(NATIVE_AUTO_REUSE_SCOPE_TEXT).toContain("no cross-wave reuse");
    expect(NATIVE_AUTO_MEASUREMENT_SCOPE_TEXT).toContain("complete-reply placement are not covered");
    expect(NATIVE_AUTO_MEASUREMENT_SCOPE_TEXT).toContain("do not establish a total speedup");
  });

  it("adds no mode, implicit workload or host calls to the existing collapsed panel", () => {
    const api = new Proxy({}, { get() { throw new Error("Rendering cannot invoke host operations"); } }) as NativeOperationAutoApi;
    const html = renderToStaticMarkup(<NativeOperationAutoControls api={api} brainId="brain-a" loaded={false}
      foregroundTurnActive={false} dirty={false} />);
    expect(html).toContain("automatic operation placement"); expect(html).not.toContain("<details open");
    expect(html).not.toContain("Generate now"); expect(html).not.toContain("prefix mode");
    expect(html.match(/type="checkbox"/g)).toHaveLength(1);
  });
});
