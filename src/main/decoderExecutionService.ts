import { createHash, randomUUID } from "node:crypto";
import type { EngineRequestContext } from "./engineSupervisor";
import { validateDeviceResourceTopology } from "./deviceResourcePlan";
import { DECODER_EXECUTION_WORK_KIND, type DecoderExecutionApi, type DecoderExecutionApplyParams,
  type DecoderExecutionBinding, type DecoderExecutionCalibration, type DecoderExecutionCandidate,
  type DecoderExecutionPrepared, type DecoderExecutionProgress, type DecoderExecutionProposal,
  type DecoderExecutionReview, type DecoderExecutionStatus } from "../shared/decoderExecution";
import type { DeviceResourceTopology } from "../shared/deviceResources";

export interface DecoderExecutionOwner {
  brainId: string;
  storagePath: string;
  originSha256: string;
  configSha256: string;
  identitySha256: string;
}

type SerialMethod = "decoder_execution_capture_next" | "decoder_execution_prepare" |
  "decoder_execution_calibrate" | "decoder_execution_apply" | "decoder_execution_release_stopped";
type ControlMethod = "decoder_execution_status" | "decoder_execution_stop";

export interface DecoderExecutionDependencies {
  owner(brainId: string): Promise<DecoderExecutionOwner>;
  requireLoaded(brainId: string): void;
  topology(owner: DecoderExecutionOwner, signal: AbortSignal, context: EngineRequestContext): Promise<DeviceResourceTopology>;
  dispatch<T>(method: SerialMethod, params: Record<string, unknown>, signal: AbortSignal, context: EngineRequestContext): Promise<T>;
  control<T>(method: ControlMethod, params: Record<string, unknown>): Promise<T>;
  diagnostic?(message: string): void;
}

interface ReviewLease { owner: DecoderExecutionOwner; generation: object; review: DecoderExecutionPrepared }
interface ProposalLease { owner: DecoderExecutionOwner; generation: object; proposal: DecoderExecutionProposal }
interface ActiveCalibration { operationId: string; controller: AbortController; generation: object }

const SHA = /^[a-f0-9]{64}$/;
const ID = /^[A-Za-z0-9][A-Za-z0-9:._-]{0,255}$/;
const PHASES = new Set(["idle", "awaiting-normal-input", "captured", "prepared", "calibrating",
  "proposal", "applied", "cancelled", "failed", "unsupported"]);
const BACKENDS = new Set(["native-cpu", "native-metal", "explicit-cuda-cpu-tile"]);
const PROGRESS_STAGES = new Set(["warming", "measuring", "selecting", "restoring"]);
const STAGE_COVERAGE = ["exact-placement-admission", "input-and-cue-transfer", "native-prefill",
  "output-home-transfer", "private-cache-release", "home-placement-restore"] as const;
const codepoint = (a: string, b: string): number => a < b ? -1 : a > b ? 1 : 0;
const record = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === "object" && !Array.isArray(value);
const count = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;

function asciiJson(value: unknown): string {
  return (JSON.stringify(value) ?? "null").replace(/[\u007f-\uffff]/g,
    character => `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`);
}
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (record(value)) return `{${Object.entries(value).sort(([a], [b]) => codepoint(a, b))
    .map(([key, item]) => `${asciiJson(key)}:${canonical(item)}`).join(",")}}`;
  return asciiJson(value);
}
function sha(value: unknown): string { return createHash("sha256").update(canonical(value)).digest("hex"); }
function topologyIdentity(topology: DeviceResourceTopology): { devices: Record<string, unknown>[]; links: Record<string, unknown>[] } {
  const devices = topology.devices.filter(device => device.kind !== "storage").map(device => {
    if (typeof device.executionDevice !== "string" || !device.executionDevice)
      throw new Error("Decoder topology needs an exact execution device for each compute interface.");
    return { id: device.id, kind: device.kind, backend: device.backend, memoryGroupId: device.memoryGroupId,
      totalBytes: device.totalBytes, recommendedLimitBytes: device.recommendedLimitBytes ?? null,
      unifiedMemory: device.unifiedMemory, nativeComputeSupported: device.nativeComputeSupported,
      unavailableReason: device.unavailableReason ?? null,
      executionDevice: device.executionDevice === "mps:0" ? "mps" : device.executionDevice };
  }).sort((a, b) => codepoint(String(a.id), String(b.id)));
  const ids = new Set(devices.map(device => device.id));
  const links = topology.links.filter(link => ids.has(link.from) && ids.has(link.to)).map(link => ({
    from: link.from, to: link.to, peerAccess: link.peerAccess,
    unavailableReason: link.unavailableReason ?? null
  })).sort((a, b) => codepoint(a.from, b.from) || codepoint(a.to, b.to));
  return { devices, links };
}
function identifier(value: unknown, label: string): string {
  if (typeof value !== "string" || !ID.test(value)) throw new Error(`Invalid decoder ${label}.`);
  return value;
}
function exact(value: unknown, allowed: readonly string[]): Record<string, unknown> {
  if (!record(value) || Object.keys(value).some(key => !allowed.includes(key)))
    throw new Error("Decoder execution accepts typed user intent only; owner paths, hashes and topology remain main-owned.");
  return value;
}
function checkedOwner(owner: DecoderExecutionOwner, brainId: string): DecoderExecutionOwner {
  if (!record(owner) || owner.brainId !== brainId || typeof owner.storagePath !== "string" || !owner.storagePath || owner.storagePath.includes("\0") ||
      !SHA.test(owner.originSha256) || !SHA.test(owner.configSha256) || !SHA.test(owner.identitySha256))
    throw new Error("Decoder loaded-owner metadata is invalid.");
  return structuredClone(owner);
}
function sameOwner(a: DecoderExecutionOwner, b: DecoderExecutionOwner): boolean {
  return a.brainId === b.brainId && a.storagePath === b.storagePath && a.originSha256 === b.originSha256 &&
    a.configSha256 === b.configSha256 && a.identitySha256 === b.identitySha256;
}
function checkedBinding(value: unknown): DecoderExecutionBinding {
  if (!record(value) || typeof value.captureId !== "string" || !ID.test(value.captureId) ||
      typeof value.ownerSessionId !== "string" || !ID.test(value.ownerSessionId) ||
      ![value.policySha256, value.topologySha256, value.registrySha256, value.nativeStateSha256,
        value.workloadSha256, value.geometrySha256, value.budgetSha256].every(item => typeof item === "string" && SHA.test(item)))
    throw new Error("Invalid decoder execution binding.");
  return structuredClone(value) as unknown as DecoderExecutionBinding;
}
function checkedCandidate(value: unknown): DecoderExecutionCandidate {
  if (!record(value) || typeof value.planSha256 !== "string" || !SHA.test(value.planSha256) ||
      typeof value.executionDevice !== "string" || !value.executionDevice || typeof value.deviceId !== "string" || !ID.test(value.deviceId) ||
      typeof value.memoryGroupId !== "string" || !ID.test(value.memoryGroupId) || !BACKENDS.has(String(value.backendMode)) ||
      !Array.isArray(value.asyncDevices) || value.asyncDevices.some(item => typeof item !== "string" || !/^(?:mps|cuda:[0-9]+)$/.test(item)) ||
      new Set(value.asyncDevices).size !== value.asyncDevices.length || typeof value.supported !== "boolean" ||
      typeof value.fallbackAcknowledged !== "boolean") throw new Error("Invalid decoder execution candidate.");
  if (canonical(value.asyncDevices) !== canonical([...value.asyncDevices].sort(codepoint)) ||
      value.backendMode === "native-cpu" && (value.executionDevice !== "cpu" || value.fallbackAcknowledged) ||
      value.backendMode === "native-metal" && (value.executionDevice !== "mps" || !value.asyncDevices.includes("mps") || value.fallbackAcknowledged) ||
      value.backendMode === "explicit-cuda-cpu-tile" && (!/^cuda:[0-9]+$/.test(String(value.executionDevice)) ||
        !value.asyncDevices.includes(value.executionDevice) || value.fallbackAcknowledged !== true))
    throw new Error("Decoder candidate cannot invent a native accelerator or inherit fallback authority.");
  if (sha({ executionDevice: value.executionDevice, deviceId: value.deviceId,
    memoryGroupId: value.memoryGroupId, backendMode: value.backendMode,
    asyncDevices: value.asyncDevices, supported: value.supported,
    fallbackAcknowledged: value.fallbackAcknowledged }) !== value.planSha256)
    throw new Error("Decoder candidate plan digest differs from its exact descriptor.");
  return structuredClone(value) as unknown as DecoderExecutionCandidate;
}
function checkedProposal(value: unknown): DecoderExecutionProposal {
  if (!record(value) || typeof value.proposalId !== "string" || !ID.test(value.proposalId) || typeof value.proposalSha256 !== "string" || !SHA.test(value.proposalSha256) ||
      value.workKind !== DECODER_EXECUTION_WORK_KIND || typeof value.comparisonSha256 !== "string" || !SHA.test(value.comparisonSha256) ||
      typeof value.outputContractSha256 !== "string" || !SHA.test(value.outputContractSha256) ||
      typeof value.baselineMeanNanoseconds !== "string" || !/^(0|[1-9][0-9]*)$/.test(value.baselineMeanNanoseconds) ||
      typeof value.candidateMeanNanoseconds !== "string" || !/^(0|[1-9][0-9]*)$/.test(value.candidateMeanNanoseconds) ||
      !count(value.minimumSamples) || value.minimumSamples < 1 || value.scope !== "exact-isolated-prefill-operation")
    throw new Error("Invalid decoder execution proposal.");
  return { proposalId: value.proposalId as string, proposalSha256: value.proposalSha256,
    workKind: DECODER_EXECUTION_WORK_KIND, candidate: checkedCandidate(value.candidate), comparisonSha256: value.comparisonSha256,
    outputContractSha256: value.outputContractSha256, baselineMeanNanoseconds: value.baselineMeanNanoseconds,
    candidateMeanNanoseconds: value.candidateMeanNanoseconds, minimumSamples: value.minimumSamples,
    scope: "exact-isolated-prefill-operation" };
}
function checkedProgress(value: unknown): DecoderExecutionProgress | null {
  if (value === null) return null;
  if (!record(value) || !count(value.candidate) || value.candidate < 1 || !count(value.candidates) || value.candidates < 1 || value.candidate > value.candidates ||
      !count(value.sample) || !count(value.samplesPerCandidate) || value.samplesPerCandidate < 1 || value.sample > value.samplesPerCandidate ||
      !PROGRESS_STAGES.has(String(value.stage))) throw new Error("Invalid decoder calibration progress.");
  return structuredClone(value) as unknown as DecoderExecutionProgress;
}
function checkedStatus(value: unknown, brainId: string): DecoderExecutionStatus {
  if (!record(value) || value.format !== "omni-decoder-execution-status" || value.version !== 1 || value.ownerId !== brainId ||
      !PHASES.has(String(value.phase)) || !(value.operationId === null || typeof value.operationId === "string" && ID.test(value.operationId)) ||
      !(value.captureId === null || typeof value.captureId === "string" && ID.test(value.captureId)) || typeof value.reason !== "string" ||
      value.workKind !== DECODER_EXECUTION_WORK_KIND || !count(value.measuredOperations) || !count(value.adoptedOperationCalls) ||
      typeof value.cancellationRequested !== "boolean" || value.retainedKvContinuationMeasured !== false ||
      value.connectedDecoderSupported !== false || value.wholeBrainOrMediaMeasured !== false || value.globalPlacementChanged !== false)
    throw new Error("Invalid decoder execution status receipt.");
  return { format: "omni-decoder-execution-status", version: 1, ownerId: brainId,
    ownerSessionId: identifier(value.ownerSessionId, "owner session"), phase: value.phase as DecoderExecutionStatus["phase"],
    operationId: value.operationId as string | null, captureId: value.captureId as string | null, reason: value.reason,
    workKind: DECODER_EXECUTION_WORK_KIND, progress: checkedProgress(value.progress),
    proposal: value.proposal === null ? null : checkedProposal(value.proposal), measuredOperations: value.measuredOperations,
    adoptedOperationCalls: value.adoptedOperationCalls, cancellationRequested: value.cancellationRequested,
    retainedKvContinuationMeasured: false, connectedDecoderSupported: false,
    wholeBrainOrMediaMeasured: false, globalPlacementChanged: false };
}
function checkedReview(value: unknown, brainId: string, topologySha256: string): DecoderExecutionReview {
  if (!record(value) || value.format !== "omni-decoder-execution-review" || value.version !== 1 || value.ownerId !== brainId ||
      value.workKind !== DECODER_EXECUTION_WORK_KIND || !Array.isArray(value.stageCoverage) ||
      value.stageCoverage.some(stage => typeof stage !== "string" || !stage || stage.includes("\0")) ||
      canonical(value.stageCoverage) !== canonical(STAGE_COVERAGE) ||
      !Array.isArray(value.candidates) || !value.candidates.length || !Array.isArray(value.inputShape) ||
      value.inputShape.some(part => !count(part) || part < 1) || !(value.cueShape === null || Array.isArray(value.cueShape) && value.cueShape.every(part => count(part) && part > 0)) ||
      !count(value.minimumSamples) || value.minimumSamples < 1) throw new Error("Invalid current decoder execution review.");
  const binding = checkedBinding(value.binding);
  if (binding.topologySha256 !== topologySha256) throw new Error("Decoder review differs from the fresh main-observed topology.");
  const candidates = value.candidates.map(checkedCandidate), currentCandidate = checkedCandidate(value.currentCandidate);
  const inputShape = value.inputShape as number[], cueShape = value.cueShape as number[] | null;
  if (!candidates.some(candidate => canonical(candidate) === canonical(currentCandidate))) throw new Error("Decoder current candidate is outside its reviewed set.");
  for (const candidate of candidates) {
    const expectedAsync = [...new Set([candidate.executionDevice, currentCandidate.executionDevice]
      .filter(device => /^(?:mps|cuda:[0-9]+)$/.test(device)))].sort(codepoint);
    if (canonical(candidate.asyncDevices) !== canonical(expectedAsync))
      throw new Error("Decoder candidate async devices differ from its execution target and observed home.");
  }
  return { format: "omni-decoder-execution-review", version: 1, ownerId: brainId, binding,
    workKind: DECODER_EXECUTION_WORK_KIND, stageCoverage: [...value.stageCoverage] as string[], currentCandidate, candidates,
    inputShape: [...inputShape], cueShape: cueShape === null ? null : [...cueShape],
    minimumSamples: value.minimumSamples, status: checkedStatus(value.status, brainId) };
}
function checkedCalibration(value: unknown, brainId: string, operationId: string,
  binding: DecoderExecutionBinding): DecoderExecutionCalibration {
  if (!record(value) || value.format !== "omni-decoder-execution-calibration" || value.version !== 1 || value.ownerId !== brainId ||
      value.operationId !== operationId || canonical(value.binding) !== canonical(binding) || !Array.isArray(value.receiptEventIds) ||
      value.receiptEventIds.some(item => typeof item !== "string" || !ID.test(item)) || !record(value.outputEquivalence) ||
      typeof value.outputEquivalence.checked !== "boolean" || typeof value.outputEquivalence.relativeTolerance !== "number" ||
      !Number.isFinite(value.outputEquivalence.relativeTolerance) || value.outputEquivalence.relativeTolerance < 0 ||
      typeof value.outputEquivalence.absoluteTolerance !== "number" || !Number.isFinite(value.outputEquivalence.absoluteTolerance) ||
      value.outputEquivalence.absoluteTolerance < 0 || value.outputEquivalence.noSamplingOrRngDraws !== true ||
      !(value.outputEquivalence.maximumAbsoluteLogitError === null || typeof value.outputEquivalence.maximumAbsoluteLogitError === "number" && Number.isFinite(value.outputEquivalence.maximumAbsoluteLogitError) && value.outputEquivalence.maximumAbsoluteLogitError >= 0) ||
      !(value.outputEquivalence.maximumAbsoluteStateError === null || typeof value.outputEquivalence.maximumAbsoluteStateError === "number" && Number.isFinite(value.outputEquivalence.maximumAbsoluteStateError) && value.outputEquivalence.maximumAbsoluteStateError >= 0))
    throw new Error("Invalid decoder calibration receipt.");
  return { format: "omni-decoder-execution-calibration", version: 1, ownerId: brainId, operationId,
    binding: checkedBinding(value.binding), status: checkedStatus(value.status, brainId),
    proposal: value.proposal === null ? null : checkedProposal(value.proposal),
    receiptEventIds: [...value.receiptEventIds] as string[], outputEquivalence: structuredClone(value.outputEquivalence) as DecoderExecutionCalibration["outputEquivalence"] };
}

/** Main-owned process-local custody for one explicitly loaded decoder owner. */
export class DecoderExecutionService implements DecoderExecutionApi {
  private readonly generations = new Map<string, object>();
  private readonly reviews = new Map<string, ReviewLease>();
  private readonly proposals = new Map<string, ProposalLease>();
  private readonly active = new Map<string, ActiveCalibration>();
  constructor(private readonly dependencies: DecoderExecutionDependencies) {}

  forget(brainId?: string): void {
    if (brainId === undefined) {
      for (const value of this.active.values()) value.controller.abort(new Error("Decoder execution owner retired."));
      this.generations.clear(); this.reviews.clear(); this.proposals.clear(); this.active.clear(); return;
    }
    this.active.get(brainId)?.controller.abort(new Error("Decoder execution owner retired."));
    this.active.delete(brainId); this.generations.delete(brainId);
    for (const [id, value] of this.reviews) if (value.owner.brainId === brainId) this.reviews.delete(id);
    for (const [id, value] of this.proposals) if (value.owner.brainId === brainId) this.proposals.delete(id);
  }
  private invalidateCustody(brainId: string): void {
    for (const [id, value] of this.reviews) if (value.owner.brainId === brainId) this.reviews.delete(id);
    for (const [id, value] of this.proposals) if (value.owner.brainId === brainId) this.proposals.delete(id);
  }
  private async capture(brainId: string, label: string): Promise<{ owner: DecoderExecutionOwner; generation: object; context: EngineRequestContext }> {
    this.dependencies.requireLoaded(brainId);
    const owner = checkedOwner(await this.dependencies.owner(brainId), brainId);
    this.dependencies.requireLoaded(brainId);
    let generation = this.generations.get(brainId);
    if (!generation) { generation = {}; this.generations.set(brainId, generation); }
    const captured = generation;
    const context: EngineRequestContext = { requestId: `decoder-${randomUUID()}`, owner: "system", label, brainId,
      beforeDispatch: async () => {
        this.dependencies.requireLoaded(brainId);
        const current = checkedOwner(await this.dependencies.owner(brainId), brainId);
        this.dependencies.requireLoaded(brainId);
        if (this.generations.get(brainId) !== captured || !sameOwner(current, owner))
          throw new Error("Decoder owner/origin/config changed while waiting; review again.");
      } };
    return { owner, generation, context };
  }
  private async recheck(owner: DecoderExecutionOwner, generation: object): Promise<void> {
    this.dependencies.requireLoaded(owner.brainId);
    const current = checkedOwner(await this.dependencies.owner(owner.brainId), owner.brainId);
    this.dependencies.requireLoaded(owner.brainId);
    if (this.generations.get(owner.brainId) !== generation || !sameOwner(current, owner))
      throw new Error("Decoder owner/origin/config changed; review again.");
  }
  async captureNext(input: { brainId: string; acknowledgeCudaCpuTileFallback?: true }): Promise<DecoderExecutionStatus> {
    const row = exact(input, ["brainId", "acknowledgeCudaCpuTileFallback"]), brainId = identifier(row.brainId, "brain id");
    if (row.acknowledgeCudaCpuTileFallback !== undefined && row.acknowledgeCudaCpuTileFallback !== true)
      throw new Error("Explicit CUDA CPU-tile fallback acknowledgment must be true when supplied.");
    const { owner, generation, context } = await this.capture(brainId, "Arming one isolated decoder capture");
    this.invalidateCustody(brainId);
    const value = await this.dependencies.dispatch<DecoderExecutionStatus>("decoder_execution_capture_next",
      { brainId, storagePath: owner.storagePath, ...(row.acknowledgeCudaCpuTileFallback === true ? { acknowledgeCudaCpuTileFallback: true } : {}) },
      new AbortController().signal, context);
    await this.recheck(owner, generation); return checkedStatus(value, brainId);
  }
  async prepare(input: { brainId: string }): Promise<DecoderExecutionPrepared> {
    const row = exact(input, ["brainId"]), brainId = identifier(row.brainId, "brain id");
    const { owner, generation, context } = await this.capture(brainId, "Preparing isolated decoder execution review");
    const signal = new AbortController().signal;
    const topology = validateDeviceResourceTopology(await this.dependencies.topology(owner, signal, context));
    await context.beforeDispatch?.();
    const review = checkedReview(await this.dependencies.dispatch<DecoderExecutionReview>("decoder_execution_prepare",
      { brainId, storagePath: owner.storagePath, topology }, signal, context), brainId, sha(topologyIdentity(topology)));
    await this.recheck(owner, generation); this.invalidateCustody(brainId);
    const prepared: DecoderExecutionPrepared = { ...review, reviewId: randomUUID() };
    this.reviews.set(prepared.reviewId, { owner, generation, review: structuredClone(prepared) });
    return structuredClone(prepared);
  }
  async calibrate(input: { brainId: string; reviewId: string; samplesPerCandidate?: number }): Promise<DecoderExecutionCalibration> {
    const row = exact(input, ["brainId", "reviewId", "samplesPerCandidate"]), brainId = identifier(row.brainId, "brain id");
    const reviewId = identifier(row.reviewId, "review id"), samples = row.samplesPerCandidate;
    if (samples !== undefined && (!count(samples) || samples < 3)) throw new Error("Decoder calibration requires at least three real samples per candidate.");
    const lease = this.reviews.get(reviewId);
    if (!lease || lease.owner.brainId !== brainId || this.generations.get(brainId) !== lease.generation)
      throw new Error("Prepare the current loaded decoder operation before calibration.");
    const { owner, generation, context } = await this.capture(brainId, "Calibrating reviewed isolated decoder execution");
    if (this.reviews.get(reviewId) !== lease || !sameOwner(owner, lease.owner) || generation !== lease.generation) {
      this.invalidateCustody(brainId); throw new Error("Decoder owner changed after review; prepare again.");
    }
    this.reviews.delete(reviewId);
    const operationId = randomUUID(), controller = new AbortController();
    const active = { operationId, controller, generation }; this.active.set(brainId, active);
    const calibratedContext = { ...context, requestId: operationId, jobId: operationId };
    try {
      const value = await this.dependencies.dispatch<DecoderExecutionCalibration>("decoder_execution_calibrate",
        { brainId, storagePath: owner.storagePath, operationId, reviewed: structuredClone(lease.review.binding),
          ...(samples === undefined ? {} : { samplesPerCandidate: samples }) }, controller.signal, calibratedContext);
      controller.signal.throwIfAborted();
      await this.recheck(owner, generation);
      const calibration = checkedCalibration(value, brainId, operationId, lease.review.binding);
      if (calibration.proposal) this.proposals.set(calibration.proposal.proposalId,
        { owner, generation, proposal: structuredClone(calibration.proposal) });
      return calibration;
    } finally {
      if (this.active.get(brainId) === active) this.active.delete(brainId);
    }
  }
  async apply(input: DecoderExecutionApplyParams): Promise<DecoderExecutionStatus> {
    const row = exact(input, ["brainId", "proposalId", "proposalSha256"]), brainId = identifier(row.brainId, "brain id");
    const proposalId = identifier(row.proposalId, "proposal id");
    if (typeof row.proposalSha256 !== "string" || !SHA.test(row.proposalSha256)) throw new Error("Invalid decoder proposal digest.");
    const lease = this.proposals.get(proposalId);
    if (!lease || lease.owner.brainId !== brainId || lease.proposal.proposalSha256 !== row.proposalSha256 ||
        this.generations.get(brainId) !== lease.generation) throw new Error("Calibrate and review the current decoder proposal before applying it.");
    const { owner, generation, context } = await this.capture(brainId, "Applying reviewed isolated decoder execution");
    if (this.proposals.get(proposalId) !== lease || !sameOwner(owner, lease.owner) || generation !== lease.generation) {
      this.invalidateCustody(brainId); throw new Error("Decoder owner changed after calibration; calibrate again.");
    }
    this.proposals.delete(proposalId);
    const value = await this.dependencies.dispatch<DecoderExecutionStatus>("decoder_execution_apply",
      { brainId, storagePath: owner.storagePath, proposalId, proposalSha256: row.proposalSha256 },
      new AbortController().signal, context);
    await this.recheck(owner, generation); return checkedStatus(value, brainId);
  }
  async status(brainId: string): Promise<DecoderExecutionStatus> {
    identifier(brainId, "brain id"); const { owner, generation } = await this.capture(brainId, "Reading decoder execution status");
    const value = await this.dependencies.control<DecoderExecutionStatus>("decoder_execution_status", { brainId, storagePath: owner.storagePath });
    await this.recheck(owner, generation); return checkedStatus(value, brainId);
  }
  async stop(input: { brainId: string; operationId?: string }): Promise<DecoderExecutionStatus> {
    const row = exact(input, ["brainId", "operationId"]), brainId = identifier(row.brainId, "brain id");
    const requested = row.operationId === undefined ? undefined : identifier(row.operationId, "operation id");
    const { owner, generation, context } = await this.capture(brainId, "Stopping exact decoder calibration");
    const active = this.active.get(brainId), operationId = requested ?? active?.operationId;
    if (active && (!requested || requested === active.operationId) && !active.controller.signal.aborted)
      active.controller.abort(new Error("This exact decoder calibration was stopped."));
    const value = await this.dependencies.control<DecoderExecutionStatus>("decoder_execution_stop",
      { brainId, storagePath: owner.storagePath, ...(operationId ? { operationId } : {}) });
    await this.recheck(owner, generation); const result = checkedStatus(value, brainId);
    // The flags-only control cannot release retained GPU tensors. Queue exact
    // cleanup behind the current serial owner without delaying the Stop ACK.
    const cleanup = new AbortController();
    void this.dependencies.dispatch<DecoderExecutionStatus>("decoder_execution_release_stopped",
      { brainId, storagePath: owner.storagePath, ...(operationId ? { operationId } : {}) }, cleanup.signal,
      { ...context, requestId: `decoder-cleanup-${randomUUID()}`, label: "Releasing stopped decoder inspection state" })
      .then(value => { checkedStatus(value, brainId); return this.recheck(owner, generation); })
      .catch(error => this.dependencies.diagnostic?.(`Decoder Stop cleanup remains pending/refused: ${error instanceof Error ? error.message : String(error)}`));
    return result;
  }
}
