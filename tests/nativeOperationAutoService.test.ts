/** Fake loaded-host controls only: no brain, worker, calibration or provider. */
import { describe, expect, it, vi } from "vitest";
import { NativeOperationAutoService, checkedNativeOperationPlan, type NativeOperationAutoDependencies } from "../src/main/nativeOperationAutoService";
import { nativeComputeSha256 as hash } from "../src/main/cachedDecoderExecutionService";
import { NATIVE_MEDIA_OPERATION_AUTO_STAGES, NATIVE_OPERATION_AUTO_STAGE_COVERAGE, NATIVE_OPERATION_AUTO_WORK_KINDS,
  type NativeOperationAutoStatus } from "../src/shared/nativeOperationAuto";
import type { DeviceResourceTopology } from "../src/shared/deviceResources";

const topology: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1, measuredAt: "2026-10-15T00:00:00Z",
  platform: "fixture", architecture: "fixture", devices: [], links: [], measurementRequested: false, neuralModelConstructed: false, warnings: [] };
const descriptor = { executionDevice: "cpu", deviceId: "cpu:0", memoryGroupId: "ram:0", backendMode: "native-cpu" as const,
  asyncDevices: [] as string[], supported: true, fallbackAcknowledged: false };
const candidate = { ...descriptor, planSha256: hash(descriptor) };
const binding = { ownerSessionId: "operation-session", policySha256: "1".repeat(64), topologySha256: hash({ devices: [], links: [] }),
  registrySha256: "2".repeat(64), budgetSha256: "3".repeat(64), homeExecutionDevice: "cpu" };
function status(phase: NativeOperationAutoStatus["phase"], operationId: string | null = null): NativeOperationAutoStatus {
  return { format: "omni-native-operation-auto-status", version: 2, ownerId: "brain-a", ownerSessionId: binding.ownerSessionId,
    phase, operationId, reason: "inert test", progress: null, cancellationRequested: phase === "cancelled", workKind: null,
    qualifiedPlanCount: 0, lastPlan: null, supportedWorkKinds: [...NATIVE_OPERATION_AUTO_WORK_KINDS], unsupportedRoutes: ["training"],
    globalPlacementChanged: false, calibrationOnAppOpen: false, wholeBrainOrTrainingMeasured: false,
    qualificationIsExact: true, reusableRouteIsInferenceForUnseenInputs: true };
}
function review() { return { format: "omni-native-operation-auto-review", version: 2, ownerId: "brain-a", binding,
  currentCandidate: candidate, candidates: [candidate], minimumSamples: 3,
  stageCoverageByWorkKind: NATIVE_OPERATION_AUTO_WORK_KINDS.map(workKind => ({ workKind, stages: [...NATIVE_OPERATION_AUTO_STAGE_COVERAGE[workKind]] })),
  supportedWorkKinds: [...NATIVE_OPERATION_AUTO_WORK_KINDS], connectedPerceptionSupported: true, status: status("prepared") }; }
function fixture() {
  let configSha256 = "4".repeat(64), loaded = true, current = status("idle");
  const dispatch = vi.fn<NativeOperationAutoDependencies["dispatch"]>(async (method, params, signal, context) => {
    await context.beforeDispatch?.(); signal.throwIfAborted();
    if (method === "native_operation_auto_prepare") { current = status("prepared"); return review() as never; }
    if (method === "native_operation_auto_arm") current = status("awaiting-operation", String(params.operationId));
    return structuredClone(current) as never;
  });
  const control = vi.fn<NativeOperationAutoDependencies["control"]>(async (method, params) => {
    if (method === "native_operation_auto_stop" && (params.operationId === undefined || params.operationId === current.operationId))
      current = status("cancelled", current.operationId);
    if (method === "native_operation_auto_plans") return { ownerId: "brain-a", ownerSessionId: binding.ownerSessionId,
      revision: 0, plans: [], total: 0 } as never;
    return structuredClone(current) as never;
  });
  const onCleanupError = vi.fn();
  const dependencies: NativeOperationAutoDependencies = {
    requireLoaded: () => { if (!loaded) throw new Error("Explicit Load required"); },
    owner: async brainId => ({ brainId, storagePath: "/owned/brain-a", originSha256: "5".repeat(64), configSha256, identitySha256: "6".repeat(64) }),
    topology: async () => structuredClone(topology),
    dispatch: dispatch as unknown as NativeOperationAutoDependencies["dispatch"],
    control: control as unknown as NativeOperationAutoDependencies["control"], onCleanupError
  };
  return { service: new NativeOperationAutoService(dependencies), dependencies, dispatch, control, onCleanupError,
    changeConfig: () => { configSha256 = "7".repeat(64); }, unload: () => { loaded = false; }, setStatus: (value: NativeOperationAutoStatus) => { current = value; } };
}
function mediaPlan(modality: "image" | "audio" | "video") {
  return { id: "a".repeat(64), geometrySha256: "b".repeat(64), qualifiedNativeStateSha256: "c".repeat(64), candidate,
    qualifiedSamples: 3, qualificationWorkloads: 1, unseenInputOrFrontierWasPreviouslyMeasured: false, wholeBrainOrTrainingCovered: false,
    workKind: `native.media.${modality}-artifact`, numericGate: "exact-media-neural-timeline-rng-and-encoded-artifact-equality",
    reuseBasis: "same-validated-media-operator-output-layout-current-owner-policy-inference",
    mediaLayout: { modality, shapeSha256: "d".repeat(64) } };
}

describe("native media Auto contract without neural execution", () => {
  it.each(["image", "audio", "video"] as const)("accepts exact media %s layout and preserves the measured scope", modality => {
    const raw = mediaPlan(modality), plan = checkedNativeOperationPlan(raw);
    expect(plan).toEqual(raw);
    raw.mediaLayout.shapeSha256 = "e".repeat(64);
    expect(plan.mediaLayout?.shapeSha256).toBe("d".repeat(64));
    expect(NATIVE_OPERATION_AUTO_STAGE_COVERAGE[plan.workKind]).toEqual([
      "prepared-media-owner-admission", "home-RNG-and-activity-transfer", "all-native-media-regions",
      "overlap-composition-and-preview", "deterministic-codec-artifact-checksum", "original-owner-restore"
    ]);
  });
  it("rejects media layouts with missing, foreign, malformed or extra fields instead of passing them through", () => {
    const raw = mediaPlan("image"), { mediaLayout: _layout, ...missing } = raw;
    expect(() => checkedNativeOperationPlan(missing)).toThrow(/shape/);
    for (const mediaLayout of [null, [], {}, { ...raw.mediaLayout, modality: "audio" },
      { ...raw.mediaLayout, modality: { value: "image" } }, { ...raw.mediaLayout, shapeSha256: "short" },
      { ...raw.mediaLayout, shapeSha256: "D".repeat(64) }, { ...raw.mediaLayout, width: 1024 }])
      expect(() => checkedNativeOperationPlan({ ...raw, mediaLayout })).toThrow(/shape|layout/);
    expect(() => checkedNativeOperationPlan({ ...raw, contextRange: {} })).toThrow(/shape/);
    expect(() => checkedNativeOperationPlan({ ...raw, sourcePath: "/foreign/output" })).toThrow(/shape/);
  });
  it("refuses weakened media equality or broader measured claims", () => {
    const raw = mediaPlan("video");
    for (const changed of [{ numericGate: "exact-embedding-equality" },
      { reuseBasis: "same-validated-operator-geometry-current-owner-policy-inference" },
      { unseenInputOrFrontierWasPreviouslyMeasured: true }, { wholeBrainOrTrainingCovered: true },
      { qualificationWorkloads: 2 }, { qualifiedSamples: 2 }])
      expect(() => checkedNativeOperationPlan({ ...raw, ...changed })).toThrow(/overclaims/);
  });
  it("keeps media layouts forbidden on the unchanged perception and cached contracts", () => {
    const base = mediaPlan("audio");
    const perception = { ...base, workKind: "native.perception.audio-embedding", numericGate: "exact-embedding-equality",
      reuseBasis: "same-validated-operator-geometry-current-owner-policy-inference" };
    expect(() => checkedNativeOperationPlan(perception)).toThrow(/shape/);
    const cached = { ...base, workKind: "native.decoder.cached-whole-step-sampling",
      numericGate: "exact-cache-connected-activity-neural-sampling-equality",
      reuseBasis: "same-validated-operator-page-layout-range-current-owner-policy-inference",
      contextRange: { minimumTokens: 65, maximumTokens: 128, queryTokens: 1,
        basis: "actual-pager-and-attention-tile-transitions", layoutSha256: "e".repeat(64) } };
    expect(() => checkedNativeOperationPlan(cached)).toThrow(/shape/);
  });
  it("accepts only the actual supported media subset with its complete six-stage review", async () => {
    const supportedWorkKinds = ["native.media.image-artifact", "native.media.audio-artifact"] as const;
    const mediaReview = { ...review(), supportedWorkKinds: [...supportedWorkKinds],
      stageCoverageByWorkKind: supportedWorkKinds.map(workKind => ({ workKind, stages: [...NATIVE_MEDIA_OPERATION_AUTO_STAGES] })),
      status: { ...status("prepared"), supportedWorkKinds: [...supportedWorkKinds] } };
    const accepted = fixture(); accepted.dispatch.mockResolvedValueOnce(mediaReview as never);
    await expect(accepted.service.prepare({ brainId: "brain-a" })).resolves.toMatchObject({ supportedWorkKinds });
    expect(accepted.dispatch.mock.calls.map(call => call[0])).toEqual(["native_operation_auto_prepare"]);
    for (const stages of [NATIVE_MEDIA_OPERATION_AUTO_STAGES.slice(0, -1),
      [...NATIVE_MEDIA_OPERATION_AUTO_STAGES.slice(0, -1), "same-home-sampling"]]) {
      const rejected = fixture(); rejected.dispatch.mockResolvedValueOnce({ ...mediaReview,
        stageCoverageByWorkKind: [{ workKind: supportedWorkKinds[0], stages }, mediaReview.stageCoverageByWorkKind[1]] } as never);
      await expect(rejected.service.prepare({ brainId: "brain-a" })).rejects.toThrow(/stages/);
    }
  });
  it("checks media layout contracts through owner-bound status and plan pagination", async () => {
    const setup = fixture(), plan = checkedNativeOperationPlan(mediaPlan("video"));
    const current: NativeOperationAutoStatus = { ...status("active", "media-op"), workKind: plan.workKind,
      qualifiedPlanCount: 1, lastPlan: plan, supportedWorkKinds: [plan.workKind] };
    setup.setStatus(current);
    await expect(setup.service.status("brain-a")).resolves.toMatchObject({ lastPlan: plan });
    setup.control.mockResolvedValueOnce(current as never).mockResolvedValueOnce({ ownerId: "brain-a", ownerSessionId: binding.ownerSessionId,
      revision: 1, total: 1, plans: [plan] } as never);
    await expect(setup.service.plans({ brainId: "brain-a" })).resolves.toMatchObject({ plans: [plan], total: 1 });
    setup.setStatus({ ...current, lastPlan: { ...plan, mediaLayout: { modality: "image", shapeSha256: "d".repeat(64) } } } as never);
    await expect(setup.service.status("brain-a")).rejects.toThrow(/layout/);
  });
});

describe("session-owned natural-operation Auto", () => {
  it("requires each registered operation's complete v2 stage map instead of reusing perception stages for cached work", async () => {
    const setup = fixture(); setup.dispatch.mockResolvedValueOnce({ ...review(), stageCoverageByWorkKind: [review().stageCoverageByWorkKind[0]] } as never);
    await expect(setup.service.prepare({ brainId: "brain-a" })).rejects.toThrow(/omits/);
  });
  it("accepts cached range inference only with the exact cache/activity/sampling gate and actual layout range", () => {
    const plan = { id: "a".repeat(64), geometrySha256: "b".repeat(64), qualifiedNativeStateSha256: "c".repeat(64), candidate,
      qualifiedSamples: 3, qualificationWorkloads: 1, unseenInputOrFrontierWasPreviouslyMeasured: false, wholeBrainOrTrainingCovered: false,
      workKind: "native.decoder.cached-whole-step-sampling", numericGate: "exact-cache-connected-activity-neural-sampling-equality",
      reuseBasis: "same-validated-operator-page-layout-range-current-owner-policy-inference",
      contextRange: { minimumTokens: 65, maximumTokens: 128, queryTokens: 1, basis: "actual-pager-and-attention-tile-transitions", layoutSha256: "d".repeat(64) } };
    expect(checkedNativeOperationPlan(plan)).toEqual(plan);
    expect(() => checkedNativeOperationPlan({ ...plan, numericGate: "exact-embedding-equality" })).toThrow(/overclaims/);
    expect(() => checkedNativeOperationPlan({ ...plan, contextRange: { ...plan.contextRange, maximumTokens: 64 } })).toThrow(/layout range/);
    expect(() => checkedNativeOperationPlan({ ...plan, unseenInputOrFrontierWasPreviouslyMeasured: true })).toThrow(/overclaims/);
  });
  it("prepares metadata and arms a one-use review without invoking any model operation", async () => {
    const setup = fixture(), prepared = await setup.service.prepare({ brainId: "brain-a" });
    prepared.binding.policySha256 = "f".repeat(64);
    const armed = await setup.service.arm({ brainId: "brain-a", reviewId: prepared.reviewId });
    expect(armed).toMatchObject({ phase: "awaiting-operation", wholeBrainOrTrainingMeasured: false });
    const arm = setup.dispatch.mock.calls[1]!;
    expect(arm[1]).toMatchObject({ reviewed: binding, storagePath: "/owned/brain-a" });
    expect(arm[1].operationId).toBe(arm[3].requestId);
    await expect(setup.service.arm({ brainId: "brain-a", reviewId: prepared.reviewId })).rejects.toThrow(/Review/);
    expect(setup.dispatch.mock.calls.map(call => call[0])).toEqual(["native_operation_auto_prepare", "native_operation_auto_arm"]);
  });

  it("retires changed owners and rejects a foreign process session or fabricated coverage", async () => {
    const changed = fixture(), prepared = await changed.service.prepare({ brainId: "brain-a" }); changed.changeConfig();
    await expect(changed.service.arm({ brainId: "brain-a", reviewId: prepared.reviewId })).rejects.toThrow(/changed|retired/);
    expect(changed.dispatch).toHaveBeenCalledOnce();
    const session = fixture(); await session.service.status("brain-a");
    session.setStatus({ ...status("idle"), ownerSessionId: "another-session" });
    await expect(session.service.status("brain-a")).rejects.toThrow(/session changed/);
    const scope = fixture(); scope.control.mockResolvedValueOnce({ ...status("active"), wholeBrainOrTrainingMeasured: true } as never);
    await expect(scope.service.status("brain-a")).rejects.toThrow(/measurement scope/);
    scope.unload(); await expect(scope.service.status("brain-a")).rejects.toThrow(/Explicit Load/);
  });

  it("stops flags immediately and serially releases only the same stopped policy", async () => {
    const setup = fixture(), prepared = await setup.service.prepare({ brainId: "brain-a" });
    const armed = await setup.service.arm({ brainId: "brain-a", reviewId: prepared.reviewId });
    let cleanupEntered!: () => void, release!: () => void;
    const entered = new Promise<void>(resolve => { cleanupEntered = resolve; }), gate = new Promise<void>(resolve => { release = resolve; });
    setup.dispatch.mockImplementationOnce(async (method, _params, _signal, context) => {
      expect(method).toBe("native_operation_auto_release_stopped"); cleanupEntered(); await gate; await context.beforeDispatch?.(); return status("cancelled") as never;
    });
    await expect(setup.service.stop({ brainId: "brain-a", operationId: armed.operationId! })).resolves.toMatchObject({ phase: "cancelled" });
    await entered;
    setup.setStatus(status("prepared")); release();
    await vi.waitFor(() => expect(setup.onCleanupError).toHaveBeenCalledOnce());
    expect(String(setup.onCleanupError.mock.calls[0]![1])).toMatch(/superseded/);
    expect(setup.control.mock.calls.map(call => call[0])).not.toContain("chat_cancel" as never);
  });

  it("does not let a late stopped Prepare publish a new review", async () => {
    const setup = fixture(); let entered!: () => void, finish!: (value: never) => void;
    const started = new Promise<void>(resolve => { entered = resolve; });
    setup.dispatch.mockImplementationOnce(async () => { entered(); return await new Promise<never>(resolve => { finish = resolve; }); });
    const preparing = setup.service.prepare({ brainId: "brain-a" }); await started;
    await setup.service.stop({ brainId: "brain-a" }); finish(review() as never);
    await expect(preparing).rejects.toThrow(/stopped/);
  });

  it("paginates metadata without accepting extra authority or non-progressing/foreign cursors", async () => {
    const setup = fixture();
    await expect(setup.service.plans({ brainId: "brain-a" })).resolves.toMatchObject({ plans: [], total: 0 });
    expect(setup.control).toHaveBeenLastCalledWith("native_operation_auto_plans", { brainId: "brain-a", storagePath: "/owned/brain-a", limit: 32 });
    await expect(setup.service.plans({ brainId: "brain-a", after: "x".repeat(513) })).rejects.toThrow(/cursor/);
    await expect(setup.service.prepare({ brainId: "brain-a", storagePath: "/foreign" } as never)).rejects.toThrow(/shape/);
    setup.control.mockResolvedValueOnce(status("idle") as never).mockResolvedValueOnce({ ownerId: "brain-a", ownerSessionId: "foreign",
      revision: 0, plans: [], total: 0 } as never);
    await expect(setup.service.plans({ brainId: "brain-a" })).rejects.toThrow(/session/);
  });
});
