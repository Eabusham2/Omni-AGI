/** Main/service fake-host coverage only; no worker, model, Load, sampling or calibration. */
import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import { CachedDecoderExecutionService, type CachedDecoderExecutionDependencies } from "../src/main/cachedDecoderExecutionService";
import type { EngineRequestContext } from "../src/main/engineSupervisor";
import { CACHED_DECODER_EXECUTION_WORK_KIND, type CachedDecoderExecutionBinding,
  type CachedDecoderExecutionStatus } from "../src/shared/cachedDecoderExecution";
import type { DecoderExecutionCandidate } from "../src/shared/decoderExecution";
import type { DeviceResourceTopology } from "../src/shared/deviceResources";

const canonical = (value: unknown): string => Array.isArray(value) ? `[${value.map(canonical).join(",")}]` :
  value && typeof value === "object" ? `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}` : JSON.stringify(value) ?? "null";
const digest = (value: unknown): string => createHash("sha256").update(canonical(value)).digest("hex");
const topology: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1,
  measuredAt: "2026-10-15T00:00:00Z", platform: "fixture", architecture: "fixture", devices: [], links: [],
  measurementRequested: false, neuralModelConstructed: false, warnings: [] };
const topologySha256 = digest({ devices: [], links: [] });
const candidateDescriptor = { executionDevice: "cpu", deviceId: "cpu:0", memoryGroupId: "ram:0",
  backendMode: "native-cpu" as const, asyncDevices: [] as string[], supported: true, fallbackAcknowledged: false };
const candidate: DecoderExecutionCandidate = { planSha256: digest(candidateDescriptor), ...candidateDescriptor };
const cudaDescriptor = { executionDevice: "cuda:0", deviceId: "gpu:0", memoryGroupId: "vram:0",
  backendMode: "explicit-cuda-cpu-tile" as const, asyncDevices: ["cuda:0"], supported: true, fallbackAcknowledged: true };
const cudaCandidate: DecoderExecutionCandidate = { planSha256: digest(cudaDescriptor), ...cudaDescriptor };
const binding: CachedDecoderExecutionBinding = { ownerSessionId: "cached-session-1", policySha256: "1".repeat(64),
  topologySha256, registrySha256: "2".repeat(64), budgetSha256: "3".repeat(64), homeExecutionDevice: "cpu" };
function status(phase: CachedDecoderExecutionStatus["phase"], operationId: string | null = null): CachedDecoderExecutionStatus {
  return { format: "omni-cached-decoder-execution-status", version: 1, ownerId: "brain-a", ownerSessionId: binding.ownerSessionId,
    phase, workKind: CACHED_DECODER_EXECUTION_WORK_KIND, operationId, reason: "fake host only", progress: null,
    cancellationRequested: phase === "cancelled", selectedCandidate: phase === "idle" || phase === "unsupported" ? null : candidate,
    actualOperationConsumed: phase === "completed" || phase === "failed-fallback",
    exactCacheAndSamplingMatched: phase === "completed", globalPlacementChanged: false,
    connectedDecoderSupported: false, wholeBrainOrMediaMeasured: false, reusableAcrossWorkloads: false };
}
function review() {
  return { format: "omni-cached-decoder-execution-review" as const, version: 1 as const, ownerId: "brain-a", binding,
    workKind: CACHED_DECODER_EXECUTION_WORK_KIND, currentCandidate: candidate, candidates: [candidate], minimumSamples: 3 as const,
    stageCoverage: ["cache-cow-snapshot", "cached-workspace", "routed-decoder-blocks", "home-readout",
      "same-home-sampling", "retained-cache-return"], status: status("prepared") };
}
function fixture() {
  let configSha256 = "4".repeat(64), loaded = true;
  const dispatch = vi.fn(async (method: string, params: Record<string, unknown>, signal: AbortSignal,
    context: EngineRequestContext) => {
    await context.beforeDispatch?.();
    if (method === "cached_decoder_execution_prepare") return review() as never;
    return status("awaiting-retained-frontier", String(params.operationId)) as never;
  });
  const control = vi.fn(async (method: string, params: Record<string, unknown>) =>
    (method === "cached_decoder_execution_stop" ? status("cancelled", String(params.operationId ?? "stopped-current"))
      : status("awaiting-retained-frontier", "known-operation")) as never);
  const dependencies: CachedDecoderExecutionDependencies = {
    requireLoaded: () => { if (!loaded) throw new Error("Explicit Load required"); },
    owner: async brainId => ({ brainId, storagePath: "/owned/brain-a", originSha256: "5".repeat(64),
      configSha256, identitySha256: "6".repeat(64) }),
    topology: async (_owner, _signal, context) => { await context.beforeDispatch?.(); return structuredClone(topology); },
    dispatch: dispatch as unknown as CachedDecoderExecutionDependencies["dispatch"],
    control: control as unknown as CachedDecoderExecutionDependencies["control"]
  };
  return { service: new CachedDecoderExecutionService(dependencies), dependencies, dispatch, control,
    changeConfig: () => { configSha256 = "7".repeat(64); }, unload: () => { loaded = false; } };
}

describe("main-owned cached decoder comparison", () => {
  it("binds fresh topology to one reviewed arm without running a calibration", async () => {
    const setup = fixture();
    const prepared = await setup.service.prepare({ brainId: "brain-a" });
    prepared.binding.policySha256 = "f".repeat(64);
    const armed = await setup.service.arm({ brainId: "brain-a", reviewId: prepared.reviewId, samplesPerCandidate: 4 });
    expect(armed).toMatchObject({ phase: "awaiting-retained-frontier", actualOperationConsumed: false,
      reusableAcrossWorkloads: false });
    const prepare = setup.dispatch.mock.calls[0]!;
    expect(prepare[0]).toBe("cached_decoder_execution_prepare");
    expect((prepare[1].topology as DeviceResourceTopology).neuralModelConstructed).toBe(false);
    const arm = setup.dispatch.mock.calls[1]!;
    expect(arm[0]).toBe("cached_decoder_execution_arm");
    expect(arm[1]).toMatchObject({ brainId: "brain-a", storagePath: "/owned/brain-a", reviewed: binding,
      samplesPerCandidate: 4 });
    expect(arm[1].operationId).toBe(arm[3].requestId);
    await expect(setup.service.arm({ brainId: "brain-a", reviewId: prepared.reviewId })).rejects.toThrow(/Prepare/);
    expect(setup.dispatch.mock.calls.map(call => call[0])).not.toContain("decoder_execution_calibrate");
  });

  it("invalidates review custody on owner/config change and requires the same loaded owner", async () => {
    const first = fixture(), prepared = await first.service.prepare({ brainId: "brain-a" }); first.changeConfig();
    await expect(first.service.arm({ brainId: "brain-a", reviewId: prepared.reviewId })).rejects.toThrow(/owner changed/);
    expect(first.dispatch).toHaveBeenCalledOnce();
    const second = fixture(); second.unload();
    await expect(second.service.status("brain-a")).rejects.toThrow(/Explicit Load/);
  });

  it("does not accept a CUDA CPU-tile candidate without this review's explicit acknowledgment", async () => {
    const refused = fixture(); refused.dispatch.mockResolvedValueOnce({ ...review(), candidates: [candidate, cudaCandidate] } as never);
    await expect(refused.service.prepare({ brainId: "brain-a" })).rejects.toThrow(/explicit acknowledgment/);
    const allowed = fixture(); allowed.dispatch.mockResolvedValueOnce({ ...review(), candidates: [candidate, cudaCandidate] } as never);
    await expect(allowed.service.prepare({ brainId: "brain-a", acknowledgeCudaCpuTileFallback: true }))
      .resolves.toMatchObject({ candidates: [{ backendMode: "native-cpu" }, { backendMode: "explicit-cuda-cpu-tile" }] });
  });

  it("stops only the main-seeded exact in-flight arm and never cancels a chat or creates a worker", async () => {
    const setup = fixture(), prepared = await setup.service.prepare({ brainId: "brain-a" });
    let began!: (operationId: string) => void;
    const started = new Promise<string>(resolve => { began = resolve; });
    setup.dispatch.mockImplementationOnce(async (method, params, signal) => {
      expect(method).toBe("cached_decoder_execution_arm"); began(String(params.operationId));
      return await new Promise<never>((_resolve, reject) => signal.addEventListener("abort", () => reject(signal.reason), { once: true }));
    });
    const arming = setup.service.arm({ brainId: "brain-a", reviewId: prepared.reviewId });
    const operationId = await started;
    await expect(setup.service.stop({ brainId: "brain-a", operationId })).resolves.toMatchObject({ phase: "cancelled", operationId });
    await expect(arming).rejects.toThrow(/stopped/);
    expect(setup.control).toHaveBeenCalledWith("cached_decoder_execution_stop",
      { brainId: "brain-a", storagePath: "/owned/brain-a", operationId });
  });

  it("uses an observed same-owner armed operation for an exact Stop without persisting authority", async () => {
    const setup = fixture();
    await expect(setup.service.status("brain-a")).resolves.toMatchObject({ operationId: "known-operation" });
    await setup.service.stop({ brainId: "brain-a" });
    expect(setup.control).toHaveBeenLastCalledWith("cached_decoder_execution_stop",
      { brainId: "brain-a", storagePath: "/owned/brain-a", operationId: "known-operation" });
    setup.service.forget("brain-a");
  });

  it("retires only the exact current Prepare across pre-dispatch and late-success Stop races", async () => {
    const before = fixture();
    let ownerEntered!: () => void, releaseOwner!: () => void, firstOwner = true;
    const enteredOwner = new Promise<void>(resolve => { ownerEntered = resolve; });
    const ownerGate = new Promise<void>(resolve => { releaseOwner = resolve; });
    const originalOwner = before.dependencies.owner;
    const preDispatch = new CachedDecoderExecutionService({ ...before.dependencies, owner: async brainId => {
      if (firstOwner) { firstOwner = false; ownerEntered(); await ownerGate; }
      return originalOwner(brainId);
    } });
    const preparingBeforeOwner = preDispatch.prepare({ brainId: "brain-a" });
    await enteredOwner;
    await expect(preDispatch.stop({ brainId: "brain-a" })).resolves.toMatchObject({ phase: "cancelled" });
    releaseOwner();
    await expect(preparingBeforeOwner).rejects.toThrow(/stopped/);
    expect(before.dispatch).not.toHaveBeenCalled();

    const late = fixture();
    let dispatchEntered!: () => void, finishDispatch!: (value: ReturnType<typeof review>) => void;
    const enteredDispatch = new Promise<void>(resolve => { dispatchEntered = resolve; });
    late.dispatch.mockImplementationOnce(async () => {
      dispatchEntered(); return await new Promise<ReturnType<typeof review>>(resolve => { finishDispatch = resolve; }) as never;
    });
    const latePrepare = late.service.prepare({ brainId: "brain-a" });
    await enteredDispatch;
    await late.service.stop({ brainId: "brain-a" });
    finishDispatch(review());
    await expect(latePrepare).rejects.toThrow(/stopped/);

    const unrelated = fixture();
    let unrelatedEntered!: () => void, finishUnrelated!: (value: ReturnType<typeof review>) => void;
    const enteredUnrelated = new Promise<void>(resolve => { unrelatedEntered = resolve; });
    unrelated.dispatch.mockImplementationOnce(async () => {
      unrelatedEntered(); return await new Promise<ReturnType<typeof review>>(resolve => { finishUnrelated = resolve; }) as never;
    });
    const replacement = unrelated.service.prepare({ brainId: "brain-a" });
    await enteredUnrelated;
    await unrelated.service.stop({ brainId: "brain-a", operationId: "older-operation" });
    finishUnrelated(review());
    await expect(replacement).resolves.toMatchObject({ status: { phase: "prepared" } });
  });
});
