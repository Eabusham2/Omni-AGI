import { createHash, randomUUID } from "node:crypto";
import type { EngineRequestContext } from "./engineSupervisor";
import { validateDeviceResourceTopology } from "./deviceResourcePlan";
import { CACHED_DECODER_EXECUTION_WORK_KIND, type CachedDecoderExecutionApi,
  type CachedDecoderExecutionBinding, type CachedDecoderExecutionPrepared,
  type CachedDecoderExecutionReview, type CachedDecoderExecutionStatus } from "../shared/cachedDecoderExecution";
import type { DecoderExecutionCandidate, DecoderExecutionProgress } from "../shared/decoderExecution";
import type { DeviceResourceTopology } from "../shared/deviceResources";

export interface CachedDecoderExecutionOwner {
  brainId: string;
  storagePath: string;
  originSha256: string;
  configSha256: string;
  identitySha256: string;
}

type SerialMethod = "cached_decoder_execution_prepare" | "cached_decoder_execution_arm";
type ControlMethod = "cached_decoder_execution_status" | "cached_decoder_execution_stop";
export interface CachedDecoderExecutionDependencies {
  owner(brainId: string): Promise<CachedDecoderExecutionOwner>;
  requireLoaded(brainId: string): void;
  topology(owner: CachedDecoderExecutionOwner, signal: AbortSignal, context: EngineRequestContext): Promise<DeviceResourceTopology>;
  dispatch<T>(method: SerialMethod, params: Record<string, unknown>, signal: AbortSignal, context: EngineRequestContext): Promise<T>;
  control<T>(method: ControlMethod, params: Record<string, unknown>): Promise<T>;
}

interface ReviewLease { owner: CachedDecoderExecutionOwner; generation: object; review: CachedDecoderExecutionPrepared }
interface ActivePrepare { controller: AbortController; generation?: object }
interface ActiveArm { operationId: string; controller: AbortController; generation: object }
interface ArmedOperation { operationId: string; owner: CachedDecoderExecutionOwner; generation: object }

const SHA = /^[a-f0-9]{64}$/;
const ID = /^[A-Za-z0-9][A-Za-z0-9:._-]{0,255}$/;
const DEVICE = /^(?:cpu|mps|cuda:[0-9]+)$/;
const PHASES = new Set(["idle", "prepared", "awaiting-retained-frontier", "calibrating", "completed",
  "cancelled", "failed-fallback", "unsupported"]);
const BACKENDS = new Set(["native-cpu", "native-metal", "explicit-cuda-cpu-tile"]);
const PROGRESS_STAGES = new Set(["warming", "measuring", "selecting", "restoring"]);
const STAGE_COVERAGE = ["cache-cow-snapshot", "cached-workspace", "routed-decoder-blocks", "home-readout",
  "same-home-sampling", "retained-cache-return"] as const;
const codepoint = (a: string, b: string): number => a < b ? -1 : a > b ? 1 : 0;
const record = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === "object" && !Array.isArray(value);
const count = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
const exactShape = (value: Record<string, unknown>, names: readonly string[], label: string): void => {
  if (Object.keys(value).sort(codepoint).join("\0") !== [...names].sort(codepoint).join("\0"))
    throw new Error(`${label} shape differs.`);
};
const exact = (value: unknown, names: readonly string[]): Record<string, unknown> => {
  if (!record(value) || Object.keys(value).some(name => !names.includes(name)))
    throw new Error("Cached decoder execution accepts typed intent only; owner paths, hashes and topology remain main-owned.");
  return value;
};
const identifier = (value: unknown, label: string): string => {
  if (typeof value !== "string" || !ID.test(value)) throw new Error(`Invalid cached decoder ${label}.`);
  return value;
};
function asciiJson(value: unknown): string {
  return (JSON.stringify(value) ?? "null").replace(/[\u007f-\uffff]/g,
    character => `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`);
}
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (record(value)) return `{${Object.entries(value).sort(([a], [b]) => codepoint(a, b))
    .map(([name, item]) => `${asciiJson(name)}:${canonical(item)}`).join(",")}}`;
  return asciiJson(value);
}
const sha = (value: unknown): string => createHash("sha256").update(canonical(value)).digest("hex");
// Shared receipt identities only; importing these helpers performs no worker or model operation.
export { canonical as nativeComputeCanonical, sha as nativeComputeSha256,
  topologyIdentity as nativeComputeTopologyIdentity, checkedCandidate as checkedNativeComputeCandidate,
  checkedProgress as checkedNativeComputeProgress, checkedBinding as checkedNativeComputeBinding };
function topologyIdentity(topology: DeviceResourceTopology): { devices: Record<string, unknown>[]; links: Record<string, unknown>[] } {
  const devices = topology.devices.filter(device => device.kind !== "storage").map(device => {
    if (typeof device.executionDevice !== "string" || !device.executionDevice)
      throw new Error("Cached decoder topology needs an exact execution device for every compute interface.");
    return { id: device.id, kind: device.kind, backend: device.backend, memoryGroupId: device.memoryGroupId,
      totalBytes: device.totalBytes, recommendedLimitBytes: device.recommendedLimitBytes ?? null,
      unifiedMemory: device.unifiedMemory, nativeComputeSupported: device.nativeComputeSupported,
      unavailableReason: device.unavailableReason ?? null,
      executionDevice: device.executionDevice === "mps:0" ? "mps" : device.executionDevice };
  }).sort((a, b) => codepoint(String(a.id), String(b.id)));
  const ids = new Set(devices.map(device => device.id));
  const links = topology.links.filter(link => ids.has(link.from) && ids.has(link.to)).map(link => ({
    from: link.from, to: link.to, peerAccess: link.peerAccess, unavailableReason: link.unavailableReason ?? null
  })).sort((a, b) => codepoint(a.from, b.from) || codepoint(a.to, b.to));
  return { devices, links };
}
function checkedOwner(value: CachedDecoderExecutionOwner, brainId: string): CachedDecoderExecutionOwner {
  if (!record(value) || value.brainId !== brainId || typeof value.storagePath !== "string" || !value.storagePath || value.storagePath.includes("\0") ||
      !SHA.test(value.originSha256) || !SHA.test(value.configSha256) || !SHA.test(value.identitySha256))
    throw new Error("Cached decoder loaded-owner metadata is invalid.");
  return structuredClone(value);
}
function sameOwner(left: CachedDecoderExecutionOwner, right: CachedDecoderExecutionOwner): boolean {
  return left.brainId === right.brainId && left.storagePath === right.storagePath &&
    left.originSha256 === right.originSha256 && left.configSha256 === right.configSha256 && left.identitySha256 === right.identitySha256;
}
function checkedCandidate(value: unknown): DecoderExecutionCandidate {
  if (!record(value)) throw new Error("Invalid cached decoder candidate.");
  exactShape(value, ["planSha256", "executionDevice", "deviceId", "memoryGroupId", "backendMode", "asyncDevices",
    "supported", "fallbackAcknowledged"], "Cached decoder candidate");
  if (typeof value.planSha256 !== "string" || !SHA.test(value.planSha256) ||
      typeof value.executionDevice !== "string" || !DEVICE.test(value.executionDevice) ||
      typeof value.deviceId !== "string" || !ID.test(value.deviceId) || typeof value.memoryGroupId !== "string" || !ID.test(value.memoryGroupId) ||
      !BACKENDS.has(String(value.backendMode)) || !Array.isArray(value.asyncDevices) ||
      value.asyncDevices.some(item => typeof item !== "string" || !/^(?:mps|cuda:[0-9]+)$/.test(item)) ||
      new Set(value.asyncDevices).size !== value.asyncDevices.length || typeof value.supported !== "boolean" ||
      typeof value.fallbackAcknowledged !== "boolean") throw new Error("Invalid cached decoder candidate.");
  if (canonical(value.asyncDevices) !== canonical([...value.asyncDevices].sort(codepoint)) ||
      value.backendMode === "native-cpu" && (value.executionDevice !== "cpu" || value.fallbackAcknowledged) ||
      value.backendMode === "native-metal" && (value.executionDevice !== "mps" || !value.asyncDevices.includes("mps") || value.fallbackAcknowledged) ||
      value.backendMode === "explicit-cuda-cpu-tile" && (!/^cuda:[0-9]+$/.test(value.executionDevice) ||
        !value.asyncDevices.includes(value.executionDevice) || value.fallbackAcknowledged !== true))
    throw new Error("Cached decoder candidate cannot invent accelerator authority.");
  if (sha({ executionDevice: value.executionDevice, deviceId: value.deviceId, memoryGroupId: value.memoryGroupId,
    backendMode: value.backendMode, asyncDevices: value.asyncDevices, supported: value.supported,
    fallbackAcknowledged: value.fallbackAcknowledged }) !== value.planSha256)
    throw new Error("Cached decoder candidate digest differs from its descriptor.");
  return { planSha256: value.planSha256, executionDevice: value.executionDevice, deviceId: value.deviceId,
    memoryGroupId: value.memoryGroupId, backendMode: value.backendMode as DecoderExecutionCandidate["backendMode"],
    asyncDevices: [...value.asyncDevices] as string[], supported: value.supported, fallbackAcknowledged: value.fallbackAcknowledged };
}
function checkedProgress(value: unknown): DecoderExecutionProgress | null {
  if (value === null) return null;
  if (!record(value)) throw new Error("Invalid cached decoder comparison progress.");
  exactShape(value, ["candidate", "candidates", "sample", "samplesPerCandidate", "stage"], "Cached decoder progress");
  if (!count(value.candidate) || value.candidate < 1 || !count(value.candidates) || value.candidates < 1 ||
      value.candidate > value.candidates || !count(value.sample) || !count(value.samplesPerCandidate) ||
      value.samplesPerCandidate < 1 || value.sample > value.samplesPerCandidate || !PROGRESS_STAGES.has(String(value.stage)))
    throw new Error("Invalid cached decoder comparison progress.");
  return structuredClone(value) as unknown as DecoderExecutionProgress;
}
function checkedBinding(value: unknown): CachedDecoderExecutionBinding {
  if (!record(value) || Object.keys(value).sort(codepoint).join("\0") !== ["ownerSessionId", "policySha256", "topologySha256",
    "registrySha256", "budgetSha256", "homeExecutionDevice"].sort(codepoint).join("\0") ||
      typeof value.ownerSessionId !== "string" || !ID.test(value.ownerSessionId) ||
      ![value.policySha256, value.topologySha256, value.registrySha256, value.budgetSha256]
        .every(item => typeof item === "string" && SHA.test(item)) ||
      typeof value.homeExecutionDevice !== "string" || !DEVICE.test(value.homeExecutionDevice))
    throw new Error("Invalid cached decoder execution binding.");
  return structuredClone(value) as unknown as CachedDecoderExecutionBinding;
}
function checkedStatus(value: unknown, brainId: string, expectedOperationId?: string): CachedDecoderExecutionStatus {
  if (!record(value)) throw new Error("Invalid cached decoder execution status receipt.");
  exactShape(value, ["format", "version", "ownerId", "ownerSessionId", "phase", "workKind", "operationId", "reason",
    "progress", "cancellationRequested", "selectedCandidate", "actualOperationConsumed", "exactCacheAndSamplingMatched",
    "globalPlacementChanged", "connectedDecoderSupported", "wholeBrainOrMediaMeasured", "reusableAcrossWorkloads"],
  "Cached decoder status");
  if (value.format !== "omni-cached-decoder-execution-status" || value.version !== 1 || value.ownerId !== brainId ||
      !PHASES.has(String(value.phase)) || !(value.operationId === null || typeof value.operationId === "string" && ID.test(value.operationId)) ||
      expectedOperationId !== undefined && value.operationId !== expectedOperationId || typeof value.reason !== "string" ||
      value.workKind !== CACHED_DECODER_EXECUTION_WORK_KIND || typeof value.cancellationRequested !== "boolean" ||
      typeof value.actualOperationConsumed !== "boolean" || typeof value.exactCacheAndSamplingMatched !== "boolean" ||
      value.globalPlacementChanged !== false || value.connectedDecoderSupported !== false ||
      value.wholeBrainOrMediaMeasured !== false || value.reusableAcrossWorkloads !== false)
    throw new Error("Invalid cached decoder execution status receipt.");
  if (value.exactCacheAndSamplingMatched && !value.actualOperationConsumed)
    throw new Error("Cached decoder cannot claim exact equivalence without consuming the reviewed operation.");
  return { format: "omni-cached-decoder-execution-status", version: 1, ownerId: brainId,
    ownerSessionId: identifier(value.ownerSessionId, "owner session"), phase: value.phase as CachedDecoderExecutionStatus["phase"],
    workKind: CACHED_DECODER_EXECUTION_WORK_KIND, operationId: value.operationId as string | null,
    reason: value.reason, progress: checkedProgress(value.progress), cancellationRequested: value.cancellationRequested,
    selectedCandidate: value.selectedCandidate === null ? null : checkedCandidate(value.selectedCandidate),
    actualOperationConsumed: value.actualOperationConsumed, exactCacheAndSamplingMatched: value.exactCacheAndSamplingMatched,
    globalPlacementChanged: false, connectedDecoderSupported: false, wholeBrainOrMediaMeasured: false, reusableAcrossWorkloads: false };
}
function checkedReview(value: unknown, brainId: string, topologySha256: string,
  cudaCpuTileAcknowledged: boolean): CachedDecoderExecutionReview {
  if (!record(value)) throw new Error("Invalid current cached decoder review.");
  exactShape(value, ["format", "version", "ownerId", "binding", "workKind", "currentCandidate", "candidates",
    "minimumSamples", "stageCoverage", "status"], "Cached decoder review");
  if (value.format !== "omni-cached-decoder-execution-review" || value.version !== 1 || value.ownerId !== brainId ||
      value.workKind !== CACHED_DECODER_EXECUTION_WORK_KIND || !Array.isArray(value.stageCoverage) ||
      canonical(value.stageCoverage) !== canonical(STAGE_COVERAGE) || !Array.isArray(value.candidates) || !value.candidates.length ||
      value.minimumSamples !== 3) throw new Error("Invalid current cached decoder review.");
  const binding = checkedBinding(value.binding), status = checkedStatus(value.status, brainId);
  if (binding.topologySha256 !== topologySha256 || status.ownerSessionId !== binding.ownerSessionId || status.phase !== "prepared")
    throw new Error("Cached decoder review differs from its current topology/session.");
  const currentCandidate = checkedCandidate(value.currentCandidate), candidates = value.candidates.map(checkedCandidate);
  if (binding.homeExecutionDevice !== currentCandidate.executionDevice ||
      !candidates.some(candidate => canonical(candidate) === canonical(currentCandidate)))
    throw new Error("Cached decoder current candidate is outside its reviewed home placement.");
  if (!cudaCpuTileAcknowledged && candidates.some(candidate => candidate.backendMode === "explicit-cuda-cpu-tile"))
    throw new Error("Cached decoder CUDA CPU-tile fallback requires this review's explicit acknowledgment.");
  for (const candidate of candidates) {
    const expectedAsync = [...new Set([candidate.executionDevice, currentCandidate.executionDevice]
      .filter(device => /^(?:mps|cuda:[0-9]+)$/.test(device)))].sort(codepoint);
    if (canonical(candidate.asyncDevices) !== canonical(expectedAsync))
      throw new Error("Cached decoder candidate async devices differ from target plus observed home.");
  }
  return { format: "omni-cached-decoder-execution-review", version: 1, ownerId: brainId, binding,
    workKind: CACHED_DECODER_EXECUTION_WORK_KIND, currentCandidate, candidates, minimumSamples: 3,
    stageCoverage: [...STAGE_COVERAGE], status };
}

/** Main-owned process-local custody for one explicitly loaded decoder owner. */
export class CachedDecoderExecutionService implements CachedDecoderExecutionApi {
  private readonly generations = new Map<string, object>();
  private readonly reviews = new Map<string, ReviewLease>();
  private readonly preparing = new Map<string, ActivePrepare>();
  private readonly active = new Map<string, ActiveArm>();
  private readonly armed = new Map<string, ArmedOperation>();
  constructor(private readonly dependencies: CachedDecoderExecutionDependencies) {}

  forget(brainId?: string): void {
    if (brainId === undefined) {
      for (const value of this.preparing.values()) value.controller.abort(new Error("Cached decoder owner retired."));
      for (const value of this.active.values()) value.controller.abort(new Error("Cached decoder owner retired."));
      this.generations.clear(); this.reviews.clear(); this.preparing.clear(); this.active.clear(); this.armed.clear(); return;
    }
    this.preparing.get(brainId)?.controller.abort(new Error("Cached decoder owner retired."));
    this.active.get(brainId)?.controller.abort(new Error("Cached decoder owner retired."));
    this.preparing.delete(brainId); this.active.delete(brainId); this.armed.delete(brainId); this.generations.delete(brainId);
    for (const [id, value] of this.reviews) if (value.owner.brainId === brainId) this.reviews.delete(id);
  }
  private invalidateReviews(brainId: string): void {
    for (const [id, value] of this.reviews) if (value.owner.brainId === brainId) this.reviews.delete(id);
  }
  private async capture(brainId: string, label: string): Promise<{
    owner: CachedDecoderExecutionOwner; generation: object; context: EngineRequestContext
  }> {
    this.dependencies.requireLoaded(brainId);
    const owner = checkedOwner(await this.dependencies.owner(brainId), brainId);
    this.dependencies.requireLoaded(brainId);
    let generation = this.generations.get(brainId);
    if (!generation) { generation = {}; this.generations.set(brainId, generation); }
    const captured = generation;
    const context: EngineRequestContext = { requestId: `cached-decoder-${randomUUID()}`, owner: "system", label, brainId,
      beforeDispatch: async () => {
        this.dependencies.requireLoaded(brainId);
        const current = checkedOwner(await this.dependencies.owner(brainId), brainId);
        this.dependencies.requireLoaded(brainId);
        if (this.generations.get(brainId) !== captured || !sameOwner(current, owner))
          throw new Error("Cached decoder owner/origin/config changed while waiting; prepare again.");
      } };
    return { owner, generation, context };
  }
  private async recheck(owner: CachedDecoderExecutionOwner, generation: object): Promise<void> {
    this.dependencies.requireLoaded(owner.brainId);
    const current = checkedOwner(await this.dependencies.owner(owner.brainId), owner.brainId);
    this.dependencies.requireLoaded(owner.brainId);
    if (this.generations.get(owner.brainId) !== generation || !sameOwner(current, owner))
      throw new Error("Cached decoder owner/origin/config changed; prepare again.");
  }
  private rememberStatus(status: CachedDecoderExecutionStatus, owner: CachedDecoderExecutionOwner, generation: object): void {
    if (status.operationId && ["awaiting-retained-frontier", "calibrating"].includes(status.phase))
      this.armed.set(owner.brainId, { operationId: status.operationId, owner, generation });
    else this.armed.delete(owner.brainId);
  }

  async prepare(input: { brainId: string; acknowledgeCudaCpuTileFallback?: true }): Promise<CachedDecoderExecutionPrepared> {
    const row = exact(input, ["brainId", "acknowledgeCudaCpuTileFallback"]), brainId = identifier(row.brainId, "brain id");
    if (row.acknowledgeCudaCpuTileFallback !== undefined && row.acknowledgeCudaCpuTileFallback !== true)
      throw new Error("Explicit CUDA CPU-tile fallback acknowledgment must be true when supplied.");
    if (this.preparing.has(brainId)) throw new Error("Cached decoder review preparation is already active for this instance.");
    // Register retirement before owner/topology awaits so a same-owner Stop or
    // lifecycle close cannot be overtaken by a late Prepare response.
    const controller = new AbortController(), active: ActivePrepare = { controller };
    this.preparing.set(brainId, active);
    try {
      const { owner, generation, context } = await this.capture(brainId, "Preparing cached decoder execution review");
      controller.signal.throwIfAborted();
      if (this.preparing.get(brainId) !== active) throw new Error("Cached decoder review preparation was retired.");
      active.generation = generation;
      const topology = validateDeviceResourceTopology(await this.dependencies.topology(owner, controller.signal, context));
      controller.signal.throwIfAborted(); await context.beforeDispatch?.();
      const review = checkedReview(await this.dependencies.dispatch<CachedDecoderExecutionReview>("cached_decoder_execution_prepare",
        { brainId, storagePath: owner.storagePath, topology,
          ...(row.acknowledgeCudaCpuTileFallback === true ? { acknowledgeCudaCpuTileFallback: true } : {}) },
        controller.signal, context), brainId, sha(topologyIdentity(topology)), row.acknowledgeCudaCpuTileFallback === true);
      controller.signal.throwIfAborted();
      if (this.preparing.get(brainId) !== active) throw new Error("Cached decoder review preparation was retired.");
      await this.recheck(owner, generation); controller.signal.throwIfAborted(); this.invalidateReviews(brainId);
      const prepared: CachedDecoderExecutionPrepared = { ...review, reviewId: randomUUID() };
      this.reviews.set(prepared.reviewId, { owner, generation, review: structuredClone(prepared) });
      return structuredClone(prepared);
    } finally { if (this.preparing.get(brainId) === active) this.preparing.delete(brainId); }
  }

  async arm(input: { brainId: string; reviewId: string; samplesPerCandidate?: number }): Promise<CachedDecoderExecutionStatus> {
    const row = exact(input, ["brainId", "reviewId", "samplesPerCandidate"]), brainId = identifier(row.brainId, "brain id");
    const reviewId = identifier(row.reviewId, "review id"), samples = row.samplesPerCandidate;
    if (samples !== undefined && (!count(samples) || samples < 3))
      throw new Error("Cached decoder comparison requires at least three real samples per candidate.");
    const lease = this.reviews.get(reviewId);
    if (!lease || lease.owner.brainId !== brainId || this.generations.get(brainId) !== lease.generation)
      throw new Error("Prepare the current loaded cached decoder before arming it.");
    const { owner, generation, context } = await this.capture(brainId, "Arming one reviewed cached decoder comparison");
    if (this.reviews.get(reviewId) !== lease || !sameOwner(owner, lease.owner) || generation !== lease.generation) {
      this.invalidateReviews(brainId); throw new Error("Cached decoder owner changed after review; prepare again.");
    }
    this.reviews.delete(reviewId);
    const operationId = randomUUID(), controller = new AbortController();
    const active = { operationId, controller, generation }; this.active.set(brainId, active);
    const armedContext = { ...context, requestId: operationId, jobId: operationId };
    try {
      const value = await this.dependencies.dispatch<CachedDecoderExecutionStatus>("cached_decoder_execution_arm",
        { brainId, storagePath: owner.storagePath, operationId, reviewed: structuredClone(lease.review.binding),
          ...(samples === undefined ? {} : { samplesPerCandidate: samples }) }, controller.signal, armedContext);
      controller.signal.throwIfAborted(); await this.recheck(owner, generation);
      const status = checkedStatus(value, brainId, operationId); this.rememberStatus(status, owner, generation); return status;
    } finally { if (this.active.get(brainId) === active) this.active.delete(brainId); }
  }

  async status(brainId: string): Promise<CachedDecoderExecutionStatus> {
    identifier(brainId, "brain id"); const { owner, generation } = await this.capture(brainId, "Reading cached decoder execution status");
    const value = await this.dependencies.control<CachedDecoderExecutionStatus>("cached_decoder_execution_status",
      { brainId, storagePath: owner.storagePath });
    await this.recheck(owner, generation); const status = checkedStatus(value, brainId);
    this.rememberStatus(status, owner, generation); return status;
  }

  async stop(input: { brainId: string; operationId?: string }): Promise<CachedDecoderExecutionStatus> {
    const row = exact(input, ["brainId", "operationId"]), brainId = identifier(row.brainId, "brain id");
    const requested = row.operationId === undefined ? undefined : identifier(row.operationId, "operation id");
    // A no-ID user Stop retires the exact review/Prepare that was current when
    // Stop began. A Stop for an older explicit operation must not cancel a
    // replacement review that has no such operation identity.
    const preparingAtRequest = requested === undefined ? this.preparing.get(brainId) : undefined;
    if (requested === undefined) this.invalidateReviews(brainId);
    if (preparingAtRequest && !preparingAtRequest.controller.signal.aborted)
      preparingAtRequest.controller.abort(new Error("Cached decoder review preparation was stopped."));
    const { owner, generation } = await this.capture(brainId, "Stopping exact cached decoder comparison");
    const active = this.active.get(brainId), known = this.armed.get(brainId);
    if (known && (known.generation !== generation || !sameOwner(known.owner, owner))) this.armed.delete(brainId);
    const operationId = requested ?? (this.armed.get(brainId)?.operationId ?? active?.operationId);
    if (active && (!operationId || operationId === active.operationId) && !active.controller.signal.aborted)
      active.controller.abort(new Error("This exact cached decoder comparison was stopped."));
    const value = await this.dependencies.control<CachedDecoderExecutionStatus>("cached_decoder_execution_stop",
      { brainId, storagePath: owner.storagePath, ...(operationId ? { operationId } : {}) });
    await this.recheck(owner, generation); const status = checkedStatus(value, brainId);
    this.rememberStatus(status, owner, generation); return status;
  }
}
