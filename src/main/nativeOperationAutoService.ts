import { randomUUID } from "node:crypto";
import type { EngineRequestContext } from "./engineSupervisor";
import { validateDeviceResourceTopology } from "./deviceResourcePlan";
import { checkedNativeComputeBinding, checkedNativeComputeCandidate, checkedNativeComputeProgress,
  nativeComputeCanonical as canonical, nativeComputeSha256 as hash, nativeComputeTopologyIdentity,
  type CachedDecoderExecutionOwner } from "./cachedDecoderExecutionService";
import { NATIVE_OPERATION_AUTO_STAGE_COVERAGE, NATIVE_OPERATION_AUTO_WORK_KINDS,
  type NativeOperationAutoApi, type NativeOperationAutoPrepared, type NativeOperationAutoReview,
  type NativeOperationAutoStatus, type NativeOperationAutoPlan, type NativeOperationAutoPlanPage } from "../shared/nativeOperationAuto";
import type { DeviceResourceTopology } from "../shared/deviceResources";
import { checkedNativeOperationCandidate } from "./nativeMixedRegionCandidate";

type SerialMethod = "native_operation_auto_prepare" | "native_operation_auto_arm" | "native_operation_auto_release_stopped";
type ControlMethod = "native_operation_auto_status" | "native_operation_auto_stop" | "native_operation_auto_plans";
export interface NativeOperationAutoDependencies {
  owner(brainId: string): Promise<CachedDecoderExecutionOwner>;
  requireLoaded(brainId: string): void;
  topology(owner: CachedDecoderExecutionOwner, signal: AbortSignal, context: EngineRequestContext): Promise<DeviceResourceTopology>;
  dispatch<T>(method: SerialMethod, params: Record<string, unknown>, signal: AbortSignal, context: EngineRequestContext): Promise<T>;
  control<T>(method: ControlMethod, params: Record<string, unknown>): Promise<T>;
  onCleanupError?(brainId: string, error: unknown): void;
}
const ID = /^[A-Za-z0-9][A-Za-z0-9:._-]{0,255}$/, SHA = /^[a-f0-9]{64}$/;
const PHASES = ["idle", "prepared", "awaiting-operation", "qualifying", "active", "cancelled", "failed-fallback", "unsupported"];
const record = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === "object" && !Array.isArray(value);
const count = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
function exact(value: unknown, keys: readonly string[], complete = true): Record<string, unknown> {
  if (!record(value) || Object.keys(value).some(key => !keys.includes(key)) || complete && keys.some(key => !(key in value)))
    throw new Error("Native operation Auto received an unexpected shape; paths, topology and authority remain main-owned.");
  return value;
}
function id(value: unknown): string { if (typeof value !== "string" || !ID.test(value)) throw new Error("Invalid native operation Auto identity."); return value; }
function cursor(value: unknown): string { if (typeof value !== "string" || !value || value.length > 512 || value.includes("\0")) throw new Error("Invalid native operation Auto page cursor."); return value; }
function workKinds(value: unknown): NativeOperationAutoStatus["supportedWorkKinds"] {
  if (!Array.isArray(value) || value.some(kind => !NATIVE_OPERATION_AUTO_WORK_KINDS.includes(kind)) || new Set(value).size !== value.length)
    throw new Error("Native operation Auto advertised an unsupported operation.");
  return [...value] as NativeOperationAutoStatus["supportedWorkKinds"];
}
function owner(value: CachedDecoderExecutionOwner, brainId: string): CachedDecoderExecutionOwner {
  if (!record(value) || value.brainId !== brainId || typeof value.storagePath !== "string" || !value.storagePath || value.storagePath.includes("\0") ||
      ![value.originSha256, value.configSha256, value.identitySha256].every(item => typeof item === "string" && SHA.test(item)))
    throw new Error("Native operation Auto loaded-owner metadata is invalid.");
  return structuredClone(value);
}
function sameOwner(left: CachedDecoderExecutionOwner, right: CachedDecoderExecutionOwner): boolean { return canonical(left) === canonical(right); }
export function checkedNativeOperationPlan(value: unknown): NativeOperationAutoPlan {
  const cached = record(value) && (value.workKind === "native.decoder.cached-whole-step-sampling" || value.workKind === "native.connected-decoder.cached-whole-step-sampling");
  const media = record(value) && (value.workKind === "native.media.image-artifact" || value.workKind === "native.media.audio-artifact" || value.workKind === "native.media.video-artifact");
  const training = record(value) && value.workKind === "native.training.prepared-whole-wave";
  const prefill = record(value) && (value.workKind === "native.decoder.full-prefix-sampling" || value.workKind === "native.connected-decoder.full-prefix-sampling");
  const forward = record(value) && (value.workKind === "native.decoder.uncached-full-forward" || value.workKind === "native.connected-decoder.uncached-full-forward");
  const row = exact(value, ["id", "workKind", "geometrySha256", "qualifiedNativeStateSha256", "candidate", "qualifiedSamples", "qualificationWorkloads",
    "numericGate", "reuseBasis", "unseenInputOrFrontierWasPreviouslyMeasured", "wholeBrainOrTrainingCovered", ...(cached ? ["contextRange"] : []),
    ...(media ? ["mediaLayout"] : []), ...(training ? ["trainingLayout"] : []), ...(prefill ? ["prefillLayout"] : []),
    ...(forward ? ["forwardLayout"] : [])]);
  if (![row.id, row.geometrySha256, row.qualifiedNativeStateSha256].every(item => typeof item === "string" && SHA.test(item)) ||
      !NATIVE_OPERATION_AUTO_WORK_KINDS.includes(row.workKind as never) || !count(row.qualifiedSamples) || row.qualifiedSamples < 3 ||
      row.qualificationWorkloads !== 1 || row.numericGate !== (forward ? "exact-full-forward-output-workspace-action-activity-equality"
        : prefill ? "exact-prefix-cache-connected-activity-neural-sampling-equality"
        : training ? "exact-prepared-wave-native-state-rng-and-effects-equality"
        : media ? "exact-media-neural-timeline-rng-and-encoded-artifact-equality"
        : cached ? "exact-cache-connected-activity-neural-sampling-equality" : "exact-embedding-equality") ||
      row.reuseBasis !== (forward ? "same-prepared-full-forward-current-owner-policy-no-cross-input-inference"
        : prefill ? "same-prepared-prefix-current-owner-policy-no-cross-prefix-inference"
        : training ? "same-prepared-wave-current-owner-policy-no-cross-wave-inference"
        : media ? "same-validated-media-operator-output-layout-current-owner-policy-inference"
        : cached ? "same-validated-operator-page-layout-range-current-owner-policy-inference" : "same-validated-operator-geometry-current-owner-policy-inference") || row.unseenInputOrFrontierWasPreviouslyMeasured !== false ||
      row.wholeBrainOrTrainingCovered !== false) throw new Error("Native operation Auto plan overclaims its measured scope.");
  if (cached) {
    const range = exact(row.contextRange, ["minimumTokens", "maximumTokens", "queryTokens", "basis", "layoutSha256"]);
    if (!count(range.minimumTokens) || range.minimumTokens < 1 || !count(range.maximumTokens) || range.maximumTokens < range.minimumTokens ||
        range.queryTokens !== 1 || range.basis !== "actual-pager-and-attention-tile-transitions" || typeof range.layoutSha256 !== "string" || !SHA.test(range.layoutSha256))
      throw new Error("Cached Auto plan lacks its actual admitted page/tile layout range.");
  }
  if (media) {
    const layout = exact(row.mediaLayout, ["modality", "shapeSha256"]);
    if (typeof layout.modality !== "string" || row.workKind !== `native.media.${layout.modality}-artifact` ||
        typeof layout.shapeSha256 !== "string" || !SHA.test(layout.shapeSha256))
      throw new Error("Media Auto plan differs from its exact native modality/output layout.");
  }
  if (training) {
    const layout = exact(row.trainingLayout, ["shapeSha256"]);
    if (typeof layout.shapeSha256 !== "string" || !SHA.test(layout.shapeSha256))
      throw new Error("Prepared training-wave Auto plan lacks its exact frozen input/schedule layout.");
  }
  if (prefill) {
    const layout = exact(row.prefillLayout, ["shapeSha256"]);
    if (typeof layout.shapeSha256 !== "string" || !SHA.test(layout.shapeSha256) || layout.shapeSha256 !== row.geometrySha256)
      throw new Error("Full-prefix Auto plan lacks its exact complete prefix/page geometry.");
  }
  if (forward) {
    const layout = exact(row.forwardLayout, ["shapeSha256"]);
    if (typeof layout.shapeSha256 !== "string" || !SHA.test(layout.shapeSha256) || layout.shapeSha256 !== row.geometrySha256)
      throw new Error("Full-forward Auto plan lacks its exact complete input/output geometry.");
  }
  const candidate = checkedNativeOperationCandidate(row.candidate);
  if (candidate.backendMode === "native-mixed-sequential" && !cached && !prefill && !forward)
    throw new Error("Mixed region placement is not qualified for this media/perception/learner operation.");
  if (!candidate.supported) throw new Error("An unsupported candidate cannot become a qualified Auto route.");
  return { ...structuredClone(row), candidate } as unknown as NativeOperationAutoPlan;
}
export function checkedNativeOperationStatus(value: unknown, brainId: string, operationId?: string): NativeOperationAutoStatus {
  const row = exact(value, ["format", "version", "ownerId", "ownerSessionId", "phase", "operationId", "reason", "progress", "cancellationRequested",
    "workKind", "qualifiedPlanCount", "lastPlan", "supportedWorkKinds", "unsupportedRoutes", "globalPlacementChanged", "calibrationOnAppOpen",
    "wholeBrainOrTrainingMeasured", "qualificationIsExact", "reusableRouteIsInferenceForUnseenInputs"]);
  if (row.format !== "omni-native-operation-auto-status" || row.version !== 2 || row.ownerId !== brainId || !PHASES.includes(String(row.phase)) ||
      row.operationId !== null && (typeof row.operationId !== "string" || !ID.test(row.operationId)) ||
      operationId !== undefined && row.operationId !== operationId || typeof row.reason !== "string" ||
      row.workKind !== null && !NATIVE_OPERATION_AUTO_WORK_KINDS.includes(row.workKind as never) || !count(row.qualifiedPlanCount) ||
      typeof row.cancellationRequested !== "boolean" || !Array.isArray(row.unsupportedRoutes) || row.unsupportedRoutes.some(item => typeof item !== "string") ||
      row.globalPlacementChanged !== false || row.calibrationOnAppOpen !== false || row.wholeBrainOrTrainingMeasured !== false ||
      row.qualificationIsExact !== true || row.reusableRouteIsInferenceForUnseenInputs !== true)
    throw new Error("Invalid native operation Auto status or measurement scope.");
  const lastPlan = row.lastPlan === null ? null : checkedNativeOperationPlan(row.lastPlan);
  const kinds = workKinds(row.supportedWorkKinds);
  if (lastPlan && row.qualifiedPlanCount === 0) throw new Error("Native operation Auto plan/count differs.");
  if (row.workKind !== null && !kinds.includes(row.workKind as typeof kinds[number]) ||
      lastPlan && !kinds.includes(lastPlan.workKind))
    throw new Error("Native operation Auto status contains an operation outside its reviewed supported kinds.");
  return { ...structuredClone(row), ownerSessionId: id(row.ownerSessionId), progress: checkedNativeComputeProgress(row.progress),
    supportedWorkKinds: kinds, lastPlan } as unknown as NativeOperationAutoStatus;
}
function review(value: unknown, brainId: string, topologySha256: string, acknowledged: boolean): NativeOperationAutoReview {
  const row = exact(value, ["format", "version", "ownerId", "binding", "currentCandidate", "candidates", "minimumSamples", "stageCoverageByWorkKind",
    "supportedWorkKinds", "connectedPerceptionSupported", "status"]);
  if (row.format !== "omni-native-operation-auto-review" || row.version !== 2 || row.ownerId !== brainId || row.minimumSamples !== 3 ||
      !Array.isArray(row.stageCoverageByWorkKind) || typeof row.connectedPerceptionSupported !== "boolean" ||
      !Array.isArray(row.candidates) || !row.candidates.length) throw new Error("Invalid native operation Auto review.");
  const kinds = workKinds(row.supportedWorkKinds), seen = new Set<string>();
  for (const item of row.stageCoverageByWorkKind) {
    const coverage = exact(item, ["workKind", "stages"]);
    if (!kinds.includes(coverage.workKind as never) || seen.has(String(coverage.workKind)) ||
        canonical(coverage.stages) !== canonical(NATIVE_OPERATION_AUTO_STAGE_COVERAGE[coverage.workKind as typeof kinds[number]]))
      throw new Error("Native operation review has mixed, omitted or invented operation stages.");
    seen.add(String(coverage.workKind));
  }
  if (seen.size !== kinds.length) throw new Error("Native operation review omits a supported operation's complete stages.");
  const binding = checkedNativeComputeBinding(row.binding), status = checkedNativeOperationStatus(row.status, brainId);
  const currentCandidate = checkedNativeComputeCandidate(row.currentCandidate), candidates = row.candidates.map(checkedNativeComputeCandidate);
  if (binding.topologySha256 !== topologySha256 || binding.ownerSessionId !== status.ownerSessionId || status.phase !== "prepared" ||
      canonical(kinds) !== canonical(status.supportedWorkKinds) ||
      binding.homeExecutionDevice !== currentCandidate.executionDevice || !candidates.some(candidate => canonical(candidate) === canonical(currentCandidate)) ||
      !acknowledged && candidates.some(candidate => candidate.backendMode === "explicit-cuda-cpu-tile"))
    throw new Error("Native operation Auto review differs from its current owner/topology or fallback acknowledgment.");
  for (const candidate of candidates) {
    const expected = [...new Set([currentCandidate.executionDevice, candidate.executionDevice].filter(device => /^(?:mps|cuda:[0-9]+)$/.test(device)))].sort();
    if (canonical(candidate.asyncDevices) !== canonical(expected)) throw new Error("Native operation Auto candidate omits target/home synchronization.");
  }
  return { ...structuredClone(row), binding, currentCandidate, candidates, supportedWorkKinds: kinds, status } as unknown as NativeOperationAutoReview;
}
interface ReviewLease { owner: CachedDecoderExecutionOwner; generation: object; prepared: NativeOperationAutoPrepared }
interface Active { controller: AbortController; operationId?: string }

/** Explicit loaded-owner intent; opening a panel or reading status never runs calibration. */
export class NativeOperationAutoService implements NativeOperationAutoApi {
  private generations = new Map<string, object>();
  private reviews = new Map<string, ReviewLease>();
  private preparing = new Map<string, Active>();
  private arming = new Map<string, Active>();
  private nativeSessions = new Map<string, { generation: object; sessionId: string }>();
  private cleanups = new Map<string, Set<AbortController>>();
  constructor(private readonly dependencies: NativeOperationAutoDependencies) {}
  forget(brainId?: string): void {
    if (brainId === undefined) { for (const key of new Set([...this.generations.keys(), ...this.preparing.keys(), ...this.arming.keys(), ...this.cleanups.keys()])) this.forget(key); return; }
    this.preparing.get(brainId)?.controller.abort(new Error("Native operation Auto owner retired."));
    this.arming.get(brainId)?.controller.abort(new Error("Native operation Auto owner retired."));
    for (const controller of this.cleanups.get(brainId) ?? []) controller.abort(new Error("Native operation Auto cleanup owner retired."));
    this.cleanups.delete(brainId);
    this.preparing.delete(brainId); this.arming.delete(brainId); this.generations.delete(brainId); this.nativeSessions.delete(brainId); this.invalidate(brainId);
  }
  private invalidate(brainId: string): void { for (const [key, lease] of this.reviews) if (lease.owner.brainId === brainId) this.reviews.delete(key); }
  private async capture(brainId: string): Promise<{ owner: CachedDecoderExecutionOwner; generation: object; context: EngineRequestContext }> {
    id(brainId); this.dependencies.requireLoaded(brainId);
    const captured = owner(await this.dependencies.owner(brainId), brainId); this.dependencies.requireLoaded(brainId);
    let generation = this.generations.get(brainId); if (!generation) { generation = {}; this.generations.set(brainId, generation); }
    const bound = generation;
    return { owner: captured, generation: bound, context: { brainId, requestId: randomUUID(), owner: "system", label: "Native operation Auto controls",
      beforeDispatch: () => this.recheck(captured, bound) } };
  }
  private async recheck(captured: CachedDecoderExecutionOwner, generation: object): Promise<void> {
    this.dependencies.requireLoaded(captured.brainId);
    const current = owner(await this.dependencies.owner(captured.brainId), captured.brainId); this.dependencies.requireLoaded(captured.brainId);
    if (this.generations.get(captured.brainId) !== generation || !sameOwner(captured, current))
      throw new Error("Native operation Auto owner/config/origin changed; review again.");
  }
  private checkSession(brainId: string, generation: object, sessionId: string): void {
    const previous = this.nativeSessions.get(brainId);
    if (previous && (previous.generation !== generation || previous.sessionId !== sessionId))
      throw new Error("Native operation Auto process session changed; the old review cannot authorize this worker.");
    this.nativeSessions.set(brainId, { generation, sessionId });
  }
  private releaseStopped(captured: Awaited<ReturnType<NativeOperationAutoService["capture"]>>, stopped: NativeOperationAutoStatus): void {
    // Stop's flags receipt is immediate. Releasing observation leases belongs
    // to the existing serial owner, after any current operation has settled.
    // A newer Prepare/Arm is never cleanup authority for this old Stop.
    const brainId = captured.owner.brainId, controller = new AbortController();
    const controllers = this.cleanups.get(brainId) ?? new Set<AbortController>();
    controllers.add(controller); this.cleanups.set(brainId, controllers);
    const context: EngineRequestContext = { ...captured.context, requestId: randomUUID(), label: "Release stopped native Auto observations",
      beforeDispatch: async () => {
        await this.recheck(captured.owner, captured.generation); controller.signal.throwIfAborted();
        const current = checkedNativeOperationStatus(await this.dependencies.control<unknown>("native_operation_auto_status",
          { brainId, storagePath: captured.owner.storagePath }), brainId);
        if (current.ownerSessionId !== stopped.ownerSessionId || current.operationId !== stopped.operationId || !current.cancellationRequested)
          throw new Error("Stopped native Auto cleanup was superseded by a newer policy.");
        await this.recheck(captured.owner, captured.generation); controller.signal.throwIfAborted();
      } };
    void this.dependencies.dispatch("native_operation_auto_release_stopped", { brainId, storagePath: captured.owner.storagePath,
      ...(stopped.operationId ? { operationId: stopped.operationId } : {}) }, controller.signal, context)
      .catch(error => { if (!controller.signal.aborted) this.dependencies.onCleanupError?.(brainId, error); })
      .finally(() => { controllers.delete(controller); if (!controllers.size && this.cleanups.get(brainId) === controllers) this.cleanups.delete(brainId); });
  }
  async prepare(input: Parameters<NativeOperationAutoApi["prepare"]>[0]): Promise<NativeOperationAutoPrepared> {
    const row = exact(input, ["brainId", "acknowledgeCudaCpuTileFallback"], false), brainId = id(row.brainId);
    if (row.acknowledgeCudaCpuTileFallback !== undefined && row.acknowledgeCudaCpuTileFallback !== true) throw new Error("Explicit fallback acknowledgment must be true.");
    if (this.preparing.has(brainId)) throw new Error("This native operation Auto review is already preparing.");
    const active: Active = { controller: new AbortController() }; this.preparing.set(brainId, active);
    try {
      const captured = await this.capture(brainId); active.controller.signal.throwIfAborted();
      const topology = validateDeviceResourceTopology(await this.dependencies.topology(captured.owner, active.controller.signal, captured.context));
      active.controller.signal.throwIfAborted(); await captured.context.beforeDispatch?.();
      const raw = await this.dependencies.dispatch<unknown>("native_operation_auto_prepare", { brainId, storagePath: captured.owner.storagePath, topology,
        ...(row.acknowledgeCudaCpuTileFallback === true ? { acknowledgeCudaCpuTileFallback: true } : {}) }, active.controller.signal, captured.context);
      active.controller.signal.throwIfAborted(); await this.recheck(captured.owner, captured.generation); active.controller.signal.throwIfAborted();
      if (this.preparing.get(brainId) !== active) throw new Error("Native operation Auto preparation was retired.");
      const value = review(raw, brainId, hash(nativeComputeTopologyIdentity(topology)), row.acknowledgeCudaCpuTileFallback === true);
      this.checkSession(brainId, captured.generation, value.binding.ownerSessionId);
      this.invalidate(brainId); const prepared = { ...value, reviewId: randomUUID() };
      this.reviews.set(prepared.reviewId, { owner: captured.owner, generation: captured.generation, prepared }); return structuredClone(prepared);
    } finally { if (this.preparing.get(brainId) === active) this.preparing.delete(brainId); }
  }
  async arm(input: Parameters<NativeOperationAutoApi["arm"]>[0]): Promise<NativeOperationAutoStatus> {
    const row = exact(input, ["brainId", "reviewId", "samplesPerCandidate"], false), brainId = id(row.brainId), lease = this.reviews.get(id(row.reviewId));
    if (!lease || lease.owner.brainId !== brainId || row.samplesPerCandidate !== undefined && (!count(row.samplesPerCandidate) || row.samplesPerCandidate < 3))
      throw new Error("Review the current loaded native operation before arming Auto.");
    const captured = await this.capture(brainId);
    if (this.reviews.get(String(row.reviewId)) !== lease || lease.generation !== captured.generation || !sameOwner(lease.owner, captured.owner))
      throw new Error("Native operation Auto review was replaced or retired.");
    this.invalidate(brainId); const operationId = randomUUID(), active = { controller: new AbortController(), operationId }; this.arming.set(brainId, active);
    try {
      const raw = await this.dependencies.dispatch<unknown>("native_operation_auto_arm", { brainId, storagePath: captured.owner.storagePath, operationId,
        reviewed: structuredClone(lease.prepared.binding), ...(row.samplesPerCandidate === undefined ? {} : { samplesPerCandidate: row.samplesPerCandidate }) },
      active.controller.signal, { ...captured.context, requestId: operationId, jobId: operationId });
      active.controller.signal.throwIfAborted(); await this.recheck(captured.owner, captured.generation); active.controller.signal.throwIfAborted();
      const value = checkedNativeOperationStatus(raw, brainId, operationId);
      if (value.ownerSessionId !== lease.prepared.binding.ownerSessionId) throw new Error("Native operation Auto Arm belongs to another process session.");
      if (canonical(value.supportedWorkKinds) !== canonical(lease.prepared.supportedWorkKinds))
        throw new Error("Native operation Auto Arm changed its reviewed supported kinds; review the current owner again.");
      this.checkSession(brainId, captured.generation, value.ownerSessionId); return value;
    } finally { if (this.arming.get(brainId) === active) this.arming.delete(brainId); }
  }
  async status(brainId: string): Promise<NativeOperationAutoStatus> {
    const captured = await this.capture(brainId), raw = await this.dependencies.control<unknown>("native_operation_auto_status", { brainId, storagePath: captured.owner.storagePath });
    await this.recheck(captured.owner, captured.generation); const value = checkedNativeOperationStatus(raw, brainId);
    this.checkSession(brainId, captured.generation, value.ownerSessionId); return value;
  }
  async plans(input: Parameters<NativeOperationAutoApi["plans"]>[0]): Promise<NativeOperationAutoPlanPage> {
    const row = exact(input, ["brainId", "after", "limit"], false), brainId = id(row.brainId);
    if (row.after !== undefined) cursor(row.after);
    if (row.limit !== undefined && (!count(row.limit) || row.limit < 1 || row.limit > 128)) throw new Error("Choose a plan-page window from1 to128.");
    const status = await this.status(brainId), captured = await this.capture(brainId);
    const raw = exact(await this.dependencies.control<unknown>("native_operation_auto_plans",
      { brainId, storagePath: captured.owner.storagePath, limit: row.limit ?? 32, ...(row.after === undefined ? {} : { after: row.after }) }),
    ["ownerId", "ownerSessionId", "revision", "plans", "total", "nextAfter"], false);
    await this.recheck(captured.owner, captured.generation);
    if (raw.ownerId !== brainId || !count(raw.revision) || !count(raw.total) || !Array.isArray(raw.plans) || raw.plans.length > Number(row.limit ?? 32) || raw.plans.length > raw.total)
      throw new Error("Native operation Auto plan page differs from its current owner/window.");
    const plans = raw.plans.map(checkedNativeOperationPlan);
    if (new Set(plans.map(plan => plan.id)).size !== plans.length) throw new Error("Native operation Auto plan page repeats an identity.");
    if (plans.some(plan => !status.supportedWorkKinds.includes(plan.workKind)))
      throw new Error("Native operation Auto plan page contains an operation outside its current reviewed kinds.");
    const sessionId = id(raw.ownerSessionId);
    if (sessionId !== status.ownerSessionId || raw.nextAfter !== undefined && (!plans.length || raw.nextAfter === row.after))
      throw new Error("Native operation Auto page crossed its session or made no cursor progress.");
    this.checkSession(brainId, captured.generation, sessionId);
    return { ownerId: brainId, ownerSessionId: sessionId, revision: raw.revision, total: raw.total, plans,
      ...(raw.nextAfter === undefined ? {} : { nextAfter: cursor(raw.nextAfter) }) };
  }
  async stop(input: Parameters<NativeOperationAutoApi["stop"]>[0]): Promise<NativeOperationAutoStatus> {
    const row = exact(input, ["brainId", "operationId"], false), brainId = id(row.brainId), requested = row.operationId === undefined ? undefined : id(row.operationId);
    if (requested === undefined) { this.invalidate(brainId); this.preparing.get(brainId)?.controller.abort(new Error("Native operation Auto review stopped.")); }
    const active = this.arming.get(brainId);
    if (active && (requested === undefined || requested === active.operationId)) active.controller.abort(new Error("Native operation Auto arming stopped."));
    const captured = await this.capture(brainId), raw = await this.dependencies.control<unknown>("native_operation_auto_stop", { brainId,
      storagePath: captured.owner.storagePath, ...(requested === undefined ? {} : { operationId: requested }) });
    await this.recheck(captured.owner, captured.generation); const value = checkedNativeOperationStatus(raw, brainId);
    this.checkSession(brainId, captured.generation, value.ownerSessionId);
    if (value.cancellationRequested && (requested === undefined || requested === value.operationId)) this.releaseStopped(captured, value);
    return value;
  }
}
