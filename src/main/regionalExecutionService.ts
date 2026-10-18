import { randomUUID } from "node:crypto";
import type { EngineRequestContext } from "./engineSupervisor";
import type { DeviceResourceSettings, DeviceResourceTopology, RegionalExecutionReview, RegionalExecutionStatus } from "../shared/deviceResources";
import type { RegionalExecutionApi, RegionalExecutionPrepared } from "../shared/regionalExecution";
import { validateDeviceResourceTopology } from "./deviceResourcePlan";

interface RegionalOwner { brainId: string; storagePath: string; identitySha256: string; settings?: DeviceResourceSettings }
interface Dependencies {
  owner(brainId: string): Promise<RegionalOwner>;
  requireLoaded(brainId: string): void;
  topology(owner: RegionalOwner, signal: AbortSignal, context: EngineRequestContext): Promise<DeviceResourceTopology>;
  dispatch<T>(method: string, params: Record<string, unknown>, signal: AbortSignal, context: EngineRequestContext): Promise<T>;
}
interface ReviewLease { owner: RegionalOwner; generation: object; review: RegionalExecutionPrepared }
const SHA = /^[a-f0-9]{64}$/;
const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}`;
  return JSON.stringify(value) ?? "null";
}
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Regional execution accepts typed intent only.");
  return value as Record<string, unknown>;
}
function exact(value: unknown, allowed: string[]): Record<string, unknown> {
  const row = record(value);
  if (Object.keys(row).some(key => !allowed.includes(key))) throw new Error("Regional topology, policy hashes and paths are main-owned observations, not caller authority.");
  return row;
}
function identity(value: unknown): string {
  if (typeof value !== "string" || !ID.test(value)) throw new Error("Invalid regional instance/review identity.");
  return value;
}
function checkedStatus(value: RegionalExecutionStatus, brainId: string): RegionalExecutionStatus {
  if (value?.format !== "omni-manual-regional-execution-status" || value.version !== 1 || value.ownerId !== brainId
    || typeof value.active !== "boolean" || typeof value.leaseValid !== "boolean"
    || value.trainingSupported !== false || value.automaticDefault !== false
    || value.contextGpuCapacityCredited !== false || value.wholePathPerformanceMeasured !== false
    || !Array.isArray(value.routes) || !Array.isArray(value.trace)
    || ![value.registeredRegions, value.activeCalls, value.completedRegionDispatches, value.actualActivationTransfers,
      value.actualActivationTransferBytes, value.trainingInapplicableCalls, value.unknownTransferCosts].every(number => Number.isSafeInteger(number) && number >= 0)
    || !Number.isFinite(value.estimatedKnownTransferSeconds) || value.estimatedKnownTransferSeconds < 0)
    throw new Error("Invalid loaded regional status receipt.");
  return structuredClone(value);
}

/** Process-local review custody only. Saved preferences never activate a route. */
export class RegionalExecutionService implements RegionalExecutionApi {
  private readonly generations = new Map<string, object>();
  private readonly reviews = new Map<string, ReviewLease>();
  constructor(private readonly dependencies: Dependencies) {}

  forget(brainId?: string): void {
    if (brainId === undefined) { this.generations.clear(); this.reviews.clear(); return; }
    this.generations.delete(brainId);
    for (const [id, lease] of this.reviews) if (lease.owner.brainId === brainId) this.reviews.delete(id);
  }
  private async capture(brainId: string): Promise<{ owner: RegionalOwner; generation: object; context: EngineRequestContext }> {
    this.dependencies.requireLoaded(brainId);
    const owner = await this.dependencies.owner(brainId);
    this.dependencies.requireLoaded(brainId);
    if (owner.brainId !== brainId || !SHA.test(owner.identitySha256)) throw new Error("Regional owner metadata changed.");
    let generation = this.generations.get(brainId);
    if (!generation) { generation = {}; this.generations.set(brainId, generation); }
    const captured = generation;
    const context: EngineRequestContext = { requestId: `regional-${randomUUID()}`, brainId, owner: "system", label: "Reviewing manual regional execution",
      beforeDispatch: async () => {
        this.dependencies.requireLoaded(brainId);
        const current = await this.dependencies.owner(brainId);
        this.dependencies.requireLoaded(brainId);
        if (this.generations.get(brainId) !== captured || current.brainId !== brainId || current.storagePath !== owner.storagePath
          || current.identitySha256 !== owner.identitySha256) throw new Error("Regional owner/settings changed while waiting; review again. No route was silently remapped.");
      } };
    return { owner: structuredClone(owner), generation, context };
  }
  async prepare(input: { brainId: string; start?: number; count?: number }): Promise<RegionalExecutionPrepared> {
    const row = exact(input, ["brainId", "start", "count"]), brainId = identity(row.brainId);
    const start = row.start ?? 0, count = row.count ?? 128;
    if (typeof start !== "number" || !Number.isSafeInteger(start) || start < 0 || typeof count !== "number" || !Number.isSafeInteger(count) || count < 1 || count > 256)
      throw new Error("Regional descriptors use a nonnegative cursor and 1–256 records per page; all regions remain pageable.");
    const { owner, generation, context } = await this.capture(brainId), signal = new AbortController().signal;
    const topology = validateDeviceResourceTopology(await this.dependencies.topology(owner, signal, context));
    await context.beforeDispatch?.();
    const review = await this.dependencies.dispatch<RegionalExecutionReview>("regional_execution_prepare",
      { brainId, storagePath: owner.storagePath, start, count, topology }, signal, context);
    await context.beforeDispatch?.();
    if (review?.format !== "omni-manual-regional-execution-review" || review.version !== 1 || review.ownerId !== brainId
      || ![review.policySha256, review.topologyIdentitySha256, review.registrySha256].every(value => typeof value === "string" && SHA.test(value))
      || review.start !== start || !Number.isSafeInteger(review.total) || review.total < 0 || !Array.isArray(review.regions) || review.regions.length > count)
      throw new Error("Invalid current regional registry review.");
    checkedStatus(review.status, brainId);
    // Worker revalidates this exact main-observed snapshot; callers receive only
    // inspection data and an opaque token, never activation authority from it.
    if (canonical(review.topology) !== canonical(topology)) throw new Error("Regional review did not retain the main-observed topology.");
    this.forgetReviews(brainId);
    const prepared = { ...structuredClone(review), reviewId: randomUUID() };
    this.reviews.set(prepared.reviewId, { owner, generation, review: prepared });
    return structuredClone(prepared);
  }
  private forgetReviews(brainId: string): void {
    for (const [id, lease] of this.reviews) if (lease.owner.brainId === brainId) this.reviews.delete(id);
  }
  async status(brainId: string): Promise<RegionalExecutionStatus> {
    identity(brainId); const { owner, context } = await this.capture(brainId);
    const result = await this.dependencies.dispatch<RegionalExecutionStatus>("regional_execution_status",
      { brainId, storagePath: owner.storagePath }, new AbortController().signal, context);
    await context.beforeDispatch?.(); return checkedStatus(result, brainId);
  }
  async activate(input: { brainId: string; reviewId: string }): Promise<RegionalExecutionStatus> {
    const row = exact(input, ["brainId", "reviewId"]), brainId = identity(row.brainId), reviewId = identity(row.reviewId);
    const lease = this.reviews.get(reviewId);
    if (!lease || lease.owner.brainId !== brainId || this.generations.get(brainId) !== lease.generation) throw new Error("Prepare the current loaded regional routes before activation.");
    const { owner, context } = await this.capture(brainId);
    if (owner.identitySha256 !== lease.owner.identitySha256 || owner.storagePath !== lease.owner.storagePath) {
      this.forgetReviews(brainId); throw new Error("Regional settings changed after review; prepare again.");
    }
    this.reviews.delete(reviewId); // An uncertain ACK is resolved by Status, never an automatic second activation.
    const review = lease.review;
    const result = await this.dependencies.dispatch<RegionalExecutionStatus>("regional_execution_activate", { brainId, storagePath: owner.storagePath,
      reviewed: { policySha256: review.policySha256, topologyIdentitySha256: review.topologyIdentitySha256, registrySha256: review.registrySha256, topology: review.topology } },
      new AbortController().signal, { ...context, label: "Activating reviewed manual regional execution" });
    return checkedStatus(result, brainId);
  }
  async deactivate(input: { brainId: string; epoch?: string }): Promise<RegionalExecutionStatus> {
    const row = exact(input, ["brainId", "epoch"]), brainId = identity(row.brainId);
    if (row.epoch !== undefined && (typeof row.epoch !== "string" || !row.epoch || row.epoch.includes("\0"))) throw new Error("Invalid regional activation epoch.");
    const { owner, context } = await this.capture(brainId);
    const result = await this.dependencies.dispatch<RegionalExecutionStatus>("regional_execution_deactivate", { brainId, storagePath: owner.storagePath,
      ...(row.epoch === undefined ? {} : { epoch: row.epoch }) }, new AbortController().signal, { ...context, label: "Deactivating manual regional execution" });
    this.forgetReviews(brainId); return checkedStatus(result, brainId);
  }
}
