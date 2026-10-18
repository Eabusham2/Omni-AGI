import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import { DecoderExecutionService, type DecoderExecutionDependencies } from "../src/main/decoderExecutionService";
import type { EngineRequestContext } from "../src/main/engineSupervisor";
import { DECODER_EXECUTION_WORK_KIND, type DecoderExecutionBinding, type DecoderExecutionCandidate,
  type DecoderExecutionProposal, type DecoderExecutionStatus } from "../src/shared/decoderExecution";
import type { DeviceResourceTopology } from "../src/shared/deviceResources";

const topology: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1,
  measuredAt: "2026-10-15T00:00:00Z", platform: "fixture", architecture: "fixture", devices: [], links: [],
  measurementRequested: false, neuralModelConstructed: false, warnings: [] };
const canonical = (value: unknown): string => Array.isArray(value) ? `[${value.map(canonical).join(",")}]` :
  value && typeof value === "object" ? `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}` : JSON.stringify(value) ?? "null";
const topologySha = createHash("sha256").update(canonical({ devices: [], links: [] })).digest("hex");
const candidateDescriptor = { executionDevice: "cpu", deviceId: "cpu:0", memoryGroupId: "ram:0",
  backendMode: "native-cpu" as const, asyncDevices: [] as string[], supported: true, fallbackAcknowledged: false };
const candidate: DecoderExecutionCandidate = { planSha256: createHash("sha256").update(canonical(candidateDescriptor)).digest("hex"),
  ...candidateDescriptor };
const binding: DecoderExecutionBinding = { captureId: "capture-1", ownerSessionId: "session-1", policySha256: "2".repeat(64),
  topologySha256: topologySha, registrySha256: "3".repeat(64), nativeStateSha256: "4".repeat(64),
  workloadSha256: "5".repeat(64), geometrySha256: "6".repeat(64), budgetSha256: "7".repeat(64) };
const proposal: DecoderExecutionProposal = { proposalId: "proposal-1", proposalSha256: "8".repeat(64),
  workKind: DECODER_EXECUTION_WORK_KIND, candidate, comparisonSha256: "9".repeat(64), outputContractSha256: "a".repeat(64),
  baselineMeanNanoseconds: "200", candidateMeanNanoseconds: "100", minimumSamples: 3,
  scope: "exact-isolated-prefill-operation" };
function status(phase: DecoderExecutionStatus["phase"], operationId: string | null = null): DecoderExecutionStatus {
  return { format: "omni-decoder-execution-status", version: 1, ownerId: "brain-a", ownerSessionId: "session-1",
    phase, operationId, captureId: "capture-1", reason: "fixture", workKind: DECODER_EXECUTION_WORK_KIND,
    progress: null, proposal: phase === "proposal" ? proposal : null, measuredOperations: phase === "proposal" ? 1 : 0,
    adoptedOperationCalls: phase === "applied" ? 1 : 0, cancellationRequested: phase === "cancelled",
    retainedKvContinuationMeasured: false, connectedDecoderSupported: false, wholeBrainOrMediaMeasured: false,
    globalPlacementChanged: false };
}
function review() {
  return { format: "omni-decoder-execution-review" as const, version: 1 as const, ownerId: "brain-a", binding,
    workKind: DECODER_EXECUTION_WORK_KIND, stageCoverage: ["exact-placement-admission", "input-and-cue-transfer", "native-prefill",
      "output-home-transfer", "private-cache-release", "home-placement-restore"], currentCandidate: candidate,
    candidates: [candidate], inputShape: [1, 4], cueShape: null, minimumSamples: 3, status: status("prepared") };
}
function fixture() {
  let configSha256 = "b".repeat(64), loaded = true;
  const dispatch = vi.fn(async (method: string, params: Record<string, unknown>, signal: AbortSignal, context: EngineRequestContext) => {
    await context.beforeDispatch?.();
    if (method === "decoder_execution_capture_next") return status("awaiting-normal-input") as never;
    if (method === "decoder_execution_prepare") return review() as never;
    if (method === "decoder_execution_calibrate") return { format: "omni-decoder-execution-calibration", version: 1,
      ownerId: "brain-a", operationId: params.operationId, binding, status: status("proposal", params.operationId as string), proposal,
      receiptEventIds: ["receipt.1"], outputEquivalence: { checked: true, relativeTolerance: 0,
        absoluteTolerance: 0, noSamplingOrRngDraws: true, maximumAbsoluteLogitError: 0,
        maximumAbsoluteStateError: 0 } } as never;
    return status("applied") as never;
  });
  const control = vi.fn(async (method: string) =>
    (method === "decoder_execution_stop" ? status("cancelled") : status("idle")) as never);
  const dependencies: DecoderExecutionDependencies = {
    requireLoaded: () => { if (!loaded) throw new Error("Explicit Load required"); },
    owner: async brainId => ({ brainId, storagePath: "/owned/brain-a", originSha256: "c".repeat(64),
      configSha256, identitySha256: "d".repeat(64) }),
    topology: async (_owner, _signal, context) => { await context.beforeDispatch?.(); return structuredClone(topology); },
    dispatch: dispatch as unknown as DecoderExecutionDependencies["dispatch"],
    control: control as unknown as DecoderExecutionDependencies["control"]
  };
  return { service: new DecoderExecutionService(dependencies), dispatch, control,
    changeConfig: () => { configSha256 = "e".repeat(64); }, unload: () => { loaded = false; } };
}

describe("main-owned isolated decoder execution", () => {
  it("uses fresh topology plus one-use review/proposal custody through capture, calibration and apply", async () => {
    const { service, dispatch } = fixture();
    await expect(service.captureNext({ brainId: "brain-a" })).resolves.toMatchObject({ phase: "awaiting-normal-input" });
    const prepared = await service.prepare({ brainId: "brain-a" });
    prepared.binding.policySha256 = "f".repeat(64);
    const calibration = await service.calibrate({ brainId: "brain-a", reviewId: prepared.reviewId, samplesPerCandidate: 3 });
    expect(calibration).toMatchObject({ proposal: { proposalId: "proposal-1" }, outputEquivalence: { noSamplingOrRngDraws: true } });
    const calibrateParams = dispatch.mock.calls[2]![1];
    expect(calibrateParams).toMatchObject({ brainId: "brain-a", storagePath: "/owned/brain-a", reviewed: binding, samplesPerCandidate: 3 });
    expect(calibrateParams.operationId).toBe(dispatch.mock.calls[2]![3].requestId);
    await expect(service.calibrate({ brainId: "brain-a", reviewId: prepared.reviewId })).rejects.toThrow(/Prepare/);
    await expect(service.apply({ brainId: "brain-a", proposalId: proposal.proposalId,
      proposalSha256: proposal.proposalSha256 })).resolves.toMatchObject({ phase: "applied" });
    await expect(service.apply({ brainId: "brain-a", proposalId: proposal.proposalId,
      proposalSha256: proposal.proposalSha256 })).rejects.toThrow(/Calibrate/);
    expect((dispatch.mock.calls[1]![1].topology as DeviceResourceTopology).neuralModelConstructed).toBe(false);
  });

  it("invalidates opaque custody when loaded config/path identity changes or unloads", async () => {
    const first = fixture(), prepared = await first.service.prepare({ brainId: "brain-a" }); first.changeConfig();
    await expect(first.service.calibrate({ brainId: "brain-a", reviewId: prepared.reviewId })).rejects.toThrow(/owner changed/);
    expect(first.dispatch).toHaveBeenCalledOnce();
    const second = fixture(), old = await second.service.prepare({ brainId: "brain-a" }); second.service.forget("brain-a");
    await expect(second.service.calibrate({ brainId: "brain-a", reviewId: old.reviewId })).rejects.toThrow(/Prepare/);
    second.unload(); await expect(second.service.status("brain-a")).rejects.toThrow(/Explicit Load/);
  });

  it("stops the exact active operation and aborts a queued/running main reservation without broader authority", async () => {
    const setup = fixture(), prepared = await setup.service.prepare({ brainId: "brain-a" });
    let started!: (value: string) => void; const operation = new Promise<string>(resolve => { started = resolve; });
    setup.dispatch.mockImplementationOnce(async (method, params, signal) => {
      expect(method).toBe("decoder_execution_calibrate"); started(params.operationId as string);
      return await new Promise<never>((_resolve, reject) => signal.addEventListener("abort", () => reject(signal.reason), { once: true }));
    });
    const calibrating = setup.service.calibrate({ brainId: "brain-a", reviewId: prepared.reviewId });
    const operationId = await operation;
    await expect(setup.service.stop({ brainId: "brain-a", operationId })).resolves.toMatchObject({ phase: "cancelled" });
    await expect(calibrating).rejects.toThrow(/stopped/);
    expect(setup.control).toHaveBeenCalledWith("decoder_execution_stop",
      { brainId: "brain-a", storagePath: "/owned/brain-a", operationId });
  });
});
