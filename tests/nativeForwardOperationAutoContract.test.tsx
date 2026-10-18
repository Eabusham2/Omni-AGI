/** New inert full-forward contracts. No worker, Python execution, model, Load or benchmark. */
import React from "react";
import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { NativeOperationAutoService, checkedNativeOperationPlan, checkedNativeOperationStatus,
  type NativeOperationAutoDependencies } from "../src/main/nativeOperationAutoService";
import { nativeComputeSha256 as hash } from "../src/main/cachedDecoderExecutionService";
import { NATIVE_FORWARD_OPERATION_AUTO_STAGES, NATIVE_OPERATION_AUTO_STAGE_COVERAGE, NATIVE_OPERATION_AUTO_WORK_KINDS,
  type NativeOperationAutoApi, type NativeOperationAutoPlan, type NativeOperationAutoReview,
  type NativeOperationAutoStatus, type NativeOperationAutoWorkKind } from "../src/shared/nativeOperationAuto";
import { NativeOperationAutoControls, nativeAutoStatusText, nativeAutoWorkKindLabel,
  NATIVE_AUTO_REUSE_SCOPE_TEXT, NATIVE_AUTO_MEASUREMENT_SCOPE_TEXT } from "../src/renderer/src/NativeOperationAutoControls";
import type { DeviceResourceTopology } from "../src/shared/deviceResources";

const KINDS = ["native.decoder.uncached-full-forward", "native.connected-decoder.uncached-full-forward"] as const;
type Kind = typeof KINDS[number];
const topology: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1,
  measuredAt: "2026-10-18T00:00:00Z", platform: "inert-fixture", architecture: "inert-fixture", devices: [], links: [],
  measurementRequested: false, neuralModelConstructed: false, warnings: [] };
const descriptor = { executionDevice: "cpu", deviceId: "cpu:0", memoryGroupId: "ram:0", backendMode: "native-cpu" as const,
  asyncDevices: [] as string[], supported: true, fallbackAcknowledged: false };
const candidate = { ...descriptor, planSha256: hash(descriptor) };
const binding = { ownerSessionId: "forward-session", policySha256: "1".repeat(64), topologySha256: hash({ devices: [], links: [] }),
  registrySha256: "2".repeat(64), budgetSha256: "3".repeat(64), homeExecutionDevice: "cpu" };
const base = () => ({ id: "a".repeat(64), geometrySha256: "b".repeat(64), qualifiedNativeStateSha256: "c".repeat(64), candidate,
  qualifiedSamples: 3, qualificationWorkloads: 1 as const, unseenInputOrFrontierWasPreviouslyMeasured: false as const,
  wholeBrainOrTrainingCovered: false as const });
function plan(kind: Kind = KINDS[0]): Extract<NativeOperationAutoPlan, { workKind: Kind }> {
  return { ...base(), workKind: kind, numericGate: "exact-full-forward-output-workspace-action-activity-equality",
    reuseBasis: "same-prepared-full-forward-current-owner-policy-no-cross-input-inference", forwardLayout: { shapeSha256: "b".repeat(64) } };
}
function status(phase: NativeOperationAutoStatus["phase"], lastPlan: NativeOperationAutoPlan | null = null,
  kinds: readonly NativeOperationAutoWorkKind[] = KINDS): NativeOperationAutoStatus {
  return { format: "omni-native-operation-auto-status", version: 2, ownerId: "brain-a", ownerSessionId: binding.ownerSessionId,
    phase, operationId: phase === "prepared" ? null : "full-forward-operation", reason: "inert metadata contract", progress: null,
    cancellationRequested: false, workKind: lastPlan?.workKind ?? null, qualifiedPlanCount: lastPlan ? 1 : 0, lastPlan,
    supportedWorkKinds: [...kinds], unsupportedRoutes: ["general-retained-KV-whole-turn", "router-STDP-and-fast-memory-learning"],
    globalPlacementChanged: false, calibrationOnAppOpen: false, wholeBrainOrTrainingMeasured: false,
    qualificationIsExact: true, reusableRouteIsInferenceForUnseenInputs: true };
}
function review(kinds: readonly NativeOperationAutoWorkKind[] = KINDS): NativeOperationAutoReview {
  return { format: "omni-native-operation-auto-review", version: 2, ownerId: "brain-a", binding,
    currentCandidate: candidate, candidates: [candidate], minimumSamples: 3,
    stageCoverageByWorkKind: kinds.map(workKind => ({ workKind, stages: [...NATIVE_OPERATION_AUTO_STAGE_COVERAGE[workKind]] })),
    supportedWorkKinds: [...kinds], connectedPerceptionSupported: false, status: status("prepared", null, kinds) };
}
function host(kinds: readonly NativeOperationAutoWorkKind[] = KINDS) {
  let current = status("prepared", null, kinds);
  const dispatch = vi.fn<NativeOperationAutoDependencies["dispatch"]>(async (method, params, signal, context) => {
    await context.beforeDispatch?.(); signal.throwIfAborted();
    if (method === "native_operation_auto_prepare") return structuredClone(review(kinds)) as never;
    if (method === "native_operation_auto_arm") {
      current = { ...status("awaiting-operation", null, kinds), operationId: String(params.operationId) };
      return structuredClone(current) as never;
    }
    throw new Error("This inert host only supports metadata review and arm.");
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
function priorPlan(kind: NativeOperationAutoWorkKind): Record<string, unknown> {
  const media = kind.startsWith("native.media."), cached = kind.includes("cached-whole-step"),
    training = kind === "native.training.prepared-whole-wave", prefix = kind.endsWith(".full-prefix-sampling");
  return { ...base(), workKind: kind,
    numericGate: prefix ? "exact-prefix-cache-connected-activity-neural-sampling-equality"
      : training ? "exact-prepared-wave-native-state-rng-and-effects-equality"
        : media ? "exact-media-neural-timeline-rng-and-encoded-artifact-equality"
          : cached ? "exact-cache-connected-activity-neural-sampling-equality" : "exact-embedding-equality",
    reuseBasis: prefix ? "same-prepared-prefix-current-owner-policy-no-cross-prefix-inference"
      : training ? "same-prepared-wave-current-owner-policy-no-cross-wave-inference"
        : media ? "same-validated-media-operator-output-layout-current-owner-policy-inference"
          : cached ? "same-validated-operator-page-layout-range-current-owner-policy-inference"
            : "same-validated-operator-geometry-current-owner-policy-inference",
    ...(media ? { mediaLayout: { modality: kind.split(".")[2]!.split("-")[0], shapeSha256: "d".repeat(64) } } : {}),
    ...(cached ? { contextRange: { minimumTokens: 1, maximumTokens: 128, queryTokens: 1,
      basis: "actual-pager-and-attention-tile-transitions", layoutSha256: "e".repeat(64) } } : {}),
    ...(training ? { trainingLayout: { shapeSha256: "d".repeat(64) } } : {}),
    ...(prefix ? { prefillLayout: { shapeSha256: "b".repeat(64) } } : {}) };
}

describe("complete uncached full-forward Auto contract", () => {
  it.each(KINDS)("copies only the exact complete %s layout", kind => {
    const raw = plan(kind), checked = checkedNativeOperationPlan(raw);
    expect(checked).toEqual(raw); raw.forwardLayout.shapeSha256 = "d".repeat(64);
    expect(checked.forwardLayout?.shapeSha256).toBe("b".repeat(64));
    expect(checked).not.toHaveProperty("contextRange"); expect(checked).not.toHaveProperty("prefillLayout");
  });

  it("requires exact geometry and rejects incomplete, malformed and authority-bearing layouts", () => {
    const { forwardLayout: _layout, ...missing } = plan();
    expect(() => checkedNativeOperationPlan(missing)).toThrow(/shape/);
    for (const forwardLayout of [null, [], {}, "b".repeat(64), { shapeSha256: 1 }, { shapeSha256: "B".repeat(64) },
      { shapeSha256: "bad" }, { shapeSha256: "d".repeat(64) }, { shapeSha256: "b".repeat(64), storagePath: "/other" }])
      expect(() => checkedNativeOperationPlan({ ...plan(), forwardLayout })).toThrow(/shape|geometry/);
    for (const extra of [{ contextRange: {} }, { mediaLayout: {} }, { trainingLayout: {} }, { prefillLayout: {} },
      { sourcePath: "/other" }, { ownerSessionId: "injected" }, { tokenSampler: "injected" }])
      expect(() => checkedNativeOperationPlan({ ...plan(), ...extra })).toThrow(/shape/);
  });

  it("refuses weaker numeric gates, cross-input reuse and whole-reply claims", () => {
    for (const extra of [{ numericGate: "exact-prefix-cache-connected-activity-neural-sampling-equality" },
      { numericGate: "exact-embedding-equality" }, { reuseBasis: "same-validated-operator-geometry-current-owner-policy-inference" },
      { qualificationWorkloads: 2 }, { qualifiedSamples: 2 }, { unseenInputOrFrontierWasPreviouslyMeasured: true },
      { wholeBrainOrTrainingCovered: true }])
      expect(() => checkedNativeOperationPlan({ ...plan(), ...extra })).toThrow(/overclaims/);
  });

  it("forbids forwardLayout on each of the prior eleven discriminators", () => {
    const prior = NATIVE_OPERATION_AUTO_WORK_KINDS.filter(kind => !kind.endsWith(".uncached-full-forward"));
    expect(prior).toHaveLength(11);
    for (const kind of prior) {
      const raw = priorPlan(kind); expect(checkedNativeOperationPlan(raw)).toEqual(raw);
      expect(() => checkedNativeOperationPlan({ ...raw, forwardLayout: { shapeSha256: "b".repeat(64) } })).toThrow(/shape/);
    }
  });

  it("prepares and arms only the two declared current-owner kinds without workloads", async () => {
    const setup = host(), prepared = await setup.service.prepare({ brainId: "brain-a" });
    expect(prepared.supportedWorkKinds).toEqual(KINDS);
    expect(prepared.stageCoverageByWorkKind).toEqual(KINDS.map(workKind => ({ workKind, stages: [...NATIVE_FORWARD_OPERATION_AUTO_STAGES] })));
    await expect(setup.service.arm({ brainId: "brain-a", reviewId: prepared.reviewId })).resolves.toMatchObject({ phase: "awaiting-operation" });
    expect(setup.dispatch.mock.calls.map(value => value[0])).toEqual(["native_operation_auto_prepare", "native_operation_auto_arm"]);
    expect(setup.dispatch.mock.calls[1]![1]).toMatchObject({ reviewed: binding, storagePath: "/owned/brain-a" });
    expect(setup.dispatch.mock.calls[1]![1]).not.toHaveProperty("input_ids");
  });

  it("requires all seven exact ordered stages and matching supported ownership", async () => {
    const stages = [...NATIVE_FORWARD_OPERATION_AUTO_STAGES];
    for (const changed of [stages.slice(1), [...stages].reverse(), [...stages, stages[0]!], [...stages.slice(0, -1), "same-home-sampling"]]) {
      const setup = host(); setup.dispatch.mockResolvedValueOnce({ ...review(), stageCoverageByWorkKind:
        [{ workKind: KINDS[0], stages: changed }, { workKind: KINDS[1], stages }] } as never);
      await expect(setup.service.prepare({ brainId: "brain-a" })).rejects.toThrow(/stages/);
    }
    const setup = host(); setup.dispatch.mockResolvedValueOnce({ ...review(), status: status("prepared", null, [KINDS[0]]) } as never);
    await expect(setup.service.prepare({ brainId: "brain-a" })).rejects.toThrow(/owner\/topology/);
  });

  it("cannot enlarge an older review during Arm or report a foreign operation in status or pages", async () => {
    const setup = host(["native.perception.image-embedding"]), prepared = await setup.service.prepare({ brainId: "brain-a" });
    setup.dispatch.mockImplementationOnce(async (_method, params) => ({ ...status("awaiting-operation"), operationId: String(params.operationId) }) as never);
    await expect(setup.service.arm({ brainId: "brain-a", reviewId: prepared.reviewId })).rejects.toThrow(/reviewed supported kinds/);
    expect(() => checkedNativeOperationStatus(status("active", plan(), ["native.perception.image-embedding"]), "brain-a"))
      .toThrow(/reviewed supported kinds/);
    const pages = host(["native.perception.image-embedding"]);
    pages.setStatus(status("awaiting-operation", null, ["native.perception.image-embedding"]));
    pages.control.mockImplementation(async method => method === "native_operation_auto_plans"
      ? { ownerId: "brain-a", ownerSessionId: binding.ownerSessionId, revision: 1, total: 1, plans: [plan()] } as never
      : status("awaiting-operation", null, ["native.perception.image-embedding"]) as never);
    await expect(pages.service.plans({ brainId: "brain-a" })).rejects.toThrow(/current reviewed kinds/);
  });

  it("keeps exact forward plans owner-bound through status and metadata pagination", async () => {
    for (const kind of KINDS) {
      const setup = host(); setup.setStatus(status("active", plan(kind)));
      await expect(setup.service.status("brain-a")).resolves.toMatchObject({ workKind: kind, lastPlan: plan(kind) });
      await expect(setup.service.plans({ brainId: "brain-a", limit: 1 })).resolves.toMatchObject({ plans: [plan(kind)], total: 1 });
      setup.setStatus({ ...status("active", plan(kind)), ownerId: "other" });
      await expect(setup.service.status("brain-a")).rejects.toThrow(/status/);
      expect(setup.dispatch).not.toHaveBeenCalled();
    }
  });

  it("matches all thirteen source declarations and exact forward gate/reuse without executing Python", () => {
    const source = readFileSync(new URL("../engine/omni_core/operation_auto_registry.py", import.meta.url), "utf8");
    const kinds = JSON.parse(source.match(/FORWARD_KINDS = (\[[^\n]+\])/)![1]!) as string[];
    const stages = JSON.parse(source.match(/FORWARD_STAGES = (\[[\s\S]*?\])\n/)![1]!) as string[];
    expect(kinds).toEqual(KINDS); expect(stages).toEqual(NATIVE_FORWARD_OPERATION_AUTO_STAGES);
    expect(NATIVE_OPERATION_AUTO_WORK_KINDS).toHaveLength(13);
    for (const kind of KINDS) expect(NATIVE_OPERATION_AUTO_STAGE_COVERAGE[kind]).toEqual(stages);
    expect(source).toContain("kind: FORWARD_STAGES for kind in FORWARD_KINDS");
    const policy = readFileSync(new URL("../engine/omni_core/forward_whole_operation_auto.py", import.meta.url), "utf8");
    expect(policy).toContain(`FORWARD_NUMERIC_GATE = "${plan().numericGate}"`);
    expect(policy).toContain(`FORWARD_REUSE_BASIS = "${plan().reuseBasis}"`);
  });
});

describe("complete-forward labels in the existing Auto control", () => {
  it("shows only a complete-forward check and keeps reply/training scope explicit", () => {
    expect(nativeAutoWorkKindLabel(KINDS[0])).toBe("Full decoder forward");
    expect(nativeAutoWorkKindLabel(KINDS[1])).toBe("Full connected forward");
    expect(nativeAutoStatusText(status("qualifying", plan()))).toBe("Comparing one complete decoder forward.");
    expect(nativeAutoStatusText(status("active", plan(KINDS[1])))).toContain("another input needs a new check");
    expect(NATIVE_AUTO_REUSE_SCOPE_TEXT).toContain("no cross-input reuse");
    expect(NATIVE_AUTO_MEASUREMENT_SCOPE_TEXT).toContain("complete-reply placement are not covered");
    const api = new Proxy({}, { get() { throw new Error("Rendering cannot invoke host operations"); } }) as NativeOperationAutoApi;
    const html = renderToStaticMarkup(<NativeOperationAutoControls api={api} brainId="brain-a" loaded={false}
      foregroundTurnActive={false} dirty={false} />);
    expect(html).toContain("automatic operation placement"); expect(html).not.toContain("<details open");
    expect(html.match(/type="checkbox"/g)).toHaveLength(1);
  });
});
