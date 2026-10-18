import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdir, open, rm } from "node:fs/promises";
import { join } from "node:path";
import type { HostActionOriginAuthority } from "../shared/hostActionAuthority";
import { SPARSE_SUBSTRATE_FORMAT, SPARSE_SUBSTRATE_SCOPE, type SparseSubstrateBundle, type SparseSubstrateImportMode,
  type SparseSubstrateJob, type SparseSubstrateJobPage, type SparseSubstrateJobQuery, type SparseSubstrateJobView,
  type SparseSubstrateNativeReceipt, type SparseSubstratePage, type SparseSubstratePlan, type SparseSubstratePlanView,
  type SparseSubstrateSelection, type SparseSubstrateUiEvent } from "../shared/sparseSubstrateTransfer";
import { SparseSubstrateTransferStore, SPARSE_TRANSFER_DIRECTORY, sparseHash, sparseId } from "./sparseSubstrateTransferStore";
import { taskAttachmentFileIdentity } from "./taskAttachmentReadVerifier";

export interface SparseSubstrateNativeHead { receiptJson: string; receiptSha256: string; generationSha256: string }
type ExportResult = Pick<SparseSubstrateBundle, "manifestJson" | "manifestSha256" | "sha256" | "bytes" | "rows">;
type PrepareResult = Pick<SparseSubstratePlan, "manifestJson" | "manifestSha256" | "sha256" | "bytes" | "rows" | "additions" | "replacements" | "identical" | "conflicts" | "targetGenerationSha256" | "selectedPreimagesSha256" | "changesSha256">;
export interface SparseSubstrateNativeBridge {
  /** Persisted inspection lane: MUST NOT call _get/Load/checkpoint. */
  inspect(brainId: string, query: Record<string, unknown>, signal: AbortSignal): Promise<SparseSubstratePage>;
  export(brainId: string, selection: SparseSubstrateSelection, destination: string, signal: AbortSignal): Promise<ExportResult>;
  prepare(brainId: string, bundle: string, bundleSha256: string, destination: string, mode: SparseSubstrateImportMode, signal: AbortSignal): Promise<PrepareResult>;
  validateBundle(brainId: string, path: string, expectedSha256: string, signal: AbortSignal): Promise<ExportResult>;
  prepareRollback(brainId: string, originalPlan: string, originalPlanSha256: string, receiptJson: string, receiptSha256: string, destination: string, signal: AbortSignal): Promise<PrepareResult>;
  /** Last atomic metadata head, no implicit Load. */
  head(brainId: string, signal: AbortSignal): Promise<SparseSubstrateNativeHead | null>;
  /** Actual current loaded owner only, under the already-acquired host FIFO. */
  apply(brainId: string, plan: string, input: { jobId: string; requestSha256: string; planSha256: string; acknowledgedReceiptSha256?: string }, signal: AbortSignal): Promise<SparseSubstrateNativeReceipt>;
}
export interface SparseSubstrateTransferDependencies {
  brainDirectory(brainId: string): Promise<string>;
  assertOwner(brainId: string): Promise<void>;
  checkResources(brainId: string, bytes: number, purpose: string): Promise<void>;
  checkDisk(brainId: string, bytes: number): Promise<void>;
  native: SparseSubstrateNativeBridge;
  /** Ordinary FIFO, in a detached tracked job. NEVER awaited by enqueue/apply. */
  deferWrite<T>(brainId: string, operation: () => Promise<T>, signal: AbortSignal): Promise<T>;
  assertLoadedBoundary(brainId: string): Promise<void>;
  selectedBundle?(brainId: string, chatTurnId: string, attachmentId: string, signal: AbortSignal): Promise<{ path: string; sha256: string; bytes: number }>;
  changed?(job: SparseSubstrateJob): void;
  ownerReleased?(brainId: string, error: unknown): Promise<void>;
}
export class SparseTransferBoundaryError extends Error {
  constructor(message: string, readonly code: "boundary-required" | "needs-review" | "reload-required" | "cancelled-unapplied", readonly committed?: boolean | null) { super(message); }
}
const hash = (value: string | Uint8Array): string => createHash("sha256").update(value).digest("hex");
function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  return `{${Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}`;
}
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Sparse tool arguments must be typed data.");
  return value as Record<string, unknown>;
}
function keys(value: Record<string, unknown>, allowed: readonly string[]): void { if (Object.keys(value).some(key => !allowed.includes(key))) throw new Error("Unknown sparse tool argument; source paths/brain ownership are not caller authority."); }
function whole(value: unknown, fallback: number, maximum: number): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || (value as number) < 1 || (value as number) > maximum) throw new Error("Invalid sparse page bound.");
  return value as number;
}
function checkedManifest(value: ExportResult | PrepareResult): Record<string, unknown> {
  sparseHash(value.sha256); sparseHash(value.manifestSha256);
  if (typeof value.manifestJson !== "string" || hash(value.manifestJson) !== value.manifestSha256 || !Number.isSafeInteger(value.bytes) || value.bytes < 1 || !Number.isSafeInteger(value.rows) || value.rows < 0) throw new Error("Native sparse artifact receipt is unverifiable.");
  const manifest = object(JSON.parse(value.manifestJson));
  if (manifest.scope !== SPARSE_SUBSTRATE_SCOPE || manifest.version !== 1 || manifest.rows !== value.rows) throw new Error("Native sparse artifact scope/coverage differs.");
  return manifest;
}
export function checkedSparseNativeReceipt(value: SparseSubstrateNativeReceipt, job: SparseSubstrateJob, plan: SparseSubstratePlan): SparseSubstrateNativeReceipt {
  const body = object(JSON.parse(value.receiptJson));
  if (hash(value.receiptJson) !== sparseHash(value.receiptSha256) || body.format !== "omni-sparse-substrate-applied" || body.version !== 1 || body.scope !== SPARSE_SUBSTRATE_SCOPE ||
      body.brainId !== job.brainId || body.jobId !== job.id || body.requestSha256 !== job.requestSha256 || body.planSha256 !== plan.sha256 || body.bundleSha256 !== plan.bundleSha256 ||
      body.selectedPreimagesSha256 !== plan.selectedPreimagesSha256 || body.changesSha256 !== plan.changesSha256 ||
      body.weightsAveraged !== false || body.cortexParametersChanged !== false || body.originChanged !== false ||
      !Number.isSafeInteger(body.appliedRows) || (body.appliedRows as number) < 0 || !Number.isFinite(Date.parse(String(body.committedAt))) ||
      Object.entries(body).some(([key, item]) => value[key as keyof SparseSubstrateNativeReceipt] !== item)) throw new Error("Native sparse applied receipt does not bind this exact reviewed job.");
  sparseHash(value.generationSha256);
  return structuredClone(value);
}
interface LiveJob { controller: AbortController; authority: HostActionOriginAuthority; promise?: Promise<void>; unlink(): void }

function publicJob(job: SparseSubstrateJob): SparseSubstrateJobView {
  const phase: SparseSubstrateJobView["phase"] = job.state === "queued" ? "queued" :
    job.state === "boundary-required" ? "waiting-for-loaded-boundary" : job.state === "applying" ? "native-apply" :
      job.state === "needs-review" ? "review-required" : job.state === "dormant" ? "historical" : "terminal";
  let receipt: SparseSubstrateJobView["receipt"];
  if (job.receipt) { const { receiptJson: _privateRawReceipt, ...safe } = job.receipt; receipt = structuredClone(safe); }
  return { format: "omni-sparse-substrate-job-view", version: 1, scope: SPARSE_SUBSTRATE_SCOPE,
    id: job.id, brainId: job.brainId, originalBrainId: job.originalBrainId, requestId: job.requestId,
    requestSha256: job.requestSha256, planId: job.planId, planSha256: job.planSha256,
    ...(job.actionEventId ? { actionEventId: job.actionEventId } : {}), ...(job.chatTurnId ? { chatTurnId: job.chatTurnId } : {}),
    state: job.state, phase,
    createdAt: job.createdAt, updatedAt: job.updatedAt, revision: job.revision,
    ...(job.error ? { error: job.error } : {}), ...(receipt ? { receipt } : {}), liveAuthorityAvailable: false };
}

function optionalCount(value: unknown): number | undefined {
  return Number.isSafeInteger(value) && (value as number) >= 0 ? value as number : undefined;
}
function publicPlan(plan: SparseSubstratePlan): SparseSubstratePlanView {
  let manifest: Record<string, unknown> = {};
  try { const value: unknown = JSON.parse(plan.manifestJson); if (value && typeof value === "object" && !Array.isArray(value)) manifest = value as Record<string, unknown>; }
  catch { throw new Error("Sparse plan manifest is no longer valid typed JSON."); }
  const rollback = manifest.rollbackOf && typeof manifest.rollbackOf === "object" && !Array.isArray(manifest.rollbackOf)
    ? manifest.rollbackOf as Record<string, unknown> : undefined;
  const fingerprintConflicts = optionalCount(manifest.fingerprintConflicts), unprovenRemovalRows = optionalCount(manifest.unprovenRemovalRows);
  return { format: "omni-sparse-substrate-plan-view", version: 1, scope: SPARSE_SUBSTRATE_SCOPE,
    id: plan.id, brainId: plan.brainId, bundleId: plan.bundleId, bundleSha256: plan.bundleSha256, mode: plan.mode,
    operation: plan.operation ?? "import", sha256: plan.sha256, bytes: plan.bytes, manifestSha256: plan.manifestSha256,
    rows: plan.rows, additions: plan.additions, replacements: plan.replacements, identical: plan.identical, conflicts: plan.conflicts,
    ...(fingerprintConflicts === undefined ? {} : { fingerprintConflicts }),
    ...(unprovenRemovalRows === undefined ? {} : { unprovenRemovalRows }),
    ...(typeof rollback?.receiptSha256 === "string" ? { rollbackReceiptSha256: sparseHash(rollback.receiptSha256) } : {}),
    targetGenerationSha256: plan.targetGenerationSha256, selectedPreimagesSha256: plan.selectedPreimagesSha256,
    changesSha256: plan.changesSha256 };
}

/** Staging/read is independent; a detached FIFO job owns every neural mutation. */
export class SparseSubstrateTransferService {
  private readonly live = new Map<string, LiveJob>();
  private readonly recovered = new Set<string>();
  private readonly requestTails = new Map<string, { action: string; sha: string; promise: Promise<unknown> }>();
  /** Bounded main-only identity attestations, not learned/model data. */
  private readonly verifiedBundles = new Map<string, true>();
  private readonly uiListeners = new Set<(event: Extract<SparseSubstrateUiEvent, { kind: "job" }>) => void>();
  private closing = false;
  constructor(private readonly dependencies: SparseSubstrateTransferDependencies) {}
  private async store(brainId: string): Promise<SparseSubstrateTransferStore> {
    await this.dependencies.assertOwner(brainId);
    const store = await SparseSubstrateTransferStore.open(await this.dependencies.brainDirectory(brainId), brainId);
    if (!this.recovered.has(brainId)) {
      const retiring = [...store.db.prepare("SELECT id,payload FROM jobs").iterate()].filter(row => {
        const state = (JSON.parse(String(row.payload)) as { state?: unknown }).state; return ["queued", "boundary-required", "applying"].includes(String(state));
      }).map(row => String(row.id));
      store.retirePending(); this.recovered.add(brainId);
      for (const id of retiring) this.publish(store.job(id));
    }
    return store;
  }
  private publish(job: SparseSubstrateJob): void {
    try { this.dependencies.changed?.(job); } catch { /* Observers cannot own mutation/cancellation. */ }
    const event: Extract<SparseSubstrateUiEvent, { kind: "job" }> = { format: "omni-sparse-substrate-ui-event", version: 1,
      brainId: job.brainId, kind: "job", job: publicJob(job), observedAt: new Date().toISOString() };
    for (const listener of this.uiListeners) { try { listener(structuredClone(event)); } catch { /* Presentation cannot own state. */ } }
  }
  onUiEvent(listener: (event: Extract<SparseSubstrateUiEvent, { kind: "job" }>) => void): () => void {
    this.uiListeners.add(listener); return () => { this.uiListeners.delete(listener); };
  }
  async uiJobs(brainIdValue: string, query: SparseSubstrateJobQuery = {}): Promise<SparseSubstrateJobPage> {
    const brainId = sparseId(brainIdValue), after = query.after === undefined ? "" : sparseId(query.after), limit = query.limit ?? 50;
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 200) throw new Error("Invalid sparse UI job page size.");
    const store = await this.store(brainId);
    try { const page = store.jobs(after, limit), total = Number(store.db.prepare("SELECT COUNT(*) AS n FROM jobs").get()?.n ?? 0);
      return { entries: page.entries.map(publicJob), total, ...(page.nextAfter ? { nextAfter: page.nextAfter } : {}) }; }
    finally { store.close(); }
  }
  async uiJob(brainIdValue: string, jobId: string): Promise<SparseSubstrateJobView> {
    const store = await this.store(sparseId(brainIdValue)); try { return publicJob(store.job(sparseId(jobId))); } finally { store.close(); }
  }
  async uiPlan(brainIdValue: string, planId: string): Promise<SparseSubstratePlanView> {
    const store = await this.store(sparseId(brainIdValue)); try { return publicPlan(store.artifact(sparseId(planId), "plan")); } finally { store.close(); }
  }
  async userStop(brainIdValue: string, jobId: string): Promise<SparseSubstrateJobView> {
    return publicJob(await this.stopOwned(sparseId(brainIdValue), sparseId(jobId), "User stopped this exact sparse transfer job."));
  }
  private async stopOwned(brainId: string, id: string, reason: string): Promise<SparseSubstrateJob> {
    const live = this.live.get(`${brainId}\0${id}`), store = await this.store(brainId); let job: SparseSubstrateJob;
    try { job = store.job(id); } finally { store.close(); }
    if (live) { live.controller.abort(new Error(reason)); await live.promise; }
    else if (["queued", "boundary-required", "dormant", "needs-review"].includes(job.state)) await this.update(brainId, id, { state: "cancelled", error: reason });
    const current = await this.store(brainId); try { return current.job(id); } finally { current.close(); }
  }
  private async checked(authority: HostActionOriginAuthority, brainId: string, signal: AbortSignal): Promise<void> {
    signal.throwIfAborted(); authority.signal.throwIfAborted();
    if (this.closing) throw new Error("Sparse transfer service is closing.");
    await this.dependencies.assertOwner(brainId); await authority.assertCurrent(); signal.throwIfAborted(); authority.signal.throwIfAborted();
  }
  private async destination(root: string, kind: "bundles" | "plans", id: string): Promise<{ path: string; relativePath: string }> {
    const directory = join(root, SPARSE_TRANSFER_DIRECTORY, kind); await mkdir(directory, { recursive: true, mode: 0o700 });
    const info = await lstat(directory); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Sparse artifact directory is unsafe.");
    const relativePath = `${SPARSE_TRANSFER_DIRECTORY}/${kind}/${sparseId(id)}.sqlite3`; return { path: join(root, relativePath), relativePath };
  }
  async execute(brainIdValue: string, action: string, argsValue: Record<string, unknown>, requestIdValue: string,
    signal: AbortSignal, authority: HostActionOriginAuthority, context?: { chatTurnId?: string; actionEventId?: string }): Promise<unknown> {
    const brainId = sparseId(brainIdValue), requestId = sparseId(requestIdValue), args = structuredClone(object(argsValue));
    if (!authority || !authority.revision || typeof authority.assertCurrent !== "function") throw new Error("Sparse transfer needs current main-owned operation authority.");
    await this.checked(authority, brainId, signal);
    if (action === "inspect") {
      keys(args, ["entity", "cursor", "maxBytes", "stateScope"]);
      if (args.stateScope !== undefined && args.stateScope !== "committed") throw new Error("Live-owned substrate inspection is not advertised by this sparse committed reader.");
      const result = await this.dependencies.native.inspect(brainId, args, signal); await this.checked(authority, brainId, signal);
      if (result.brainId !== brainId || result.scope !== SPARSE_SUBSTRATE_SCOPE || result.stateScope !== "committed") throw new Error("Sparse inspection owner/scope is unverifiable.");
      return result;
    }
    if (action === "status" || action === "list") {
      keys(args, action === "status" ? ["jobId"] : ["after", "limit"]);
      const store = await this.store(brainId); try { return action === "status" ? store.job(sparseId(args.jobId)) : store.jobs(args.after === undefined ? "" : sparseId(args.after), whole(args.limit, 50, 200)); } finally { store.close(); }
    }
    if (action === "cancel") {
      keys(args, ["jobId"]); return this.stopOwned(brainId, sparseId(args.jobId), "Permissioned tool action cancelled this exact sparse transfer job.");
    }
    if (action === "read") return this.readBundle(brainId, args, signal, authority);
    if (!["export", "register-bundle", "prepare-import", "prepare-rollback", "apply"].includes(action)) throw new Error("Unsupported sparse transfer action; complete cortex/candidate editing is a separate boundary.");
    const requestSha = hash(canonical({ action, args })), key = `${brainId}\0${requestId}`, previous = this.requestTails.get(key);
    if (previous) { if (previous.action !== action || previous.sha !== requestSha) throw new Error("Concurrent sparse request changed reviewed operands."); return previous.promise; }
    const operation = this.request(brainId, action, args, requestId, requestSha, signal, authority, context);
    const pending = { action, sha: requestSha, promise: operation }; this.requestTails.set(key, pending);
    try { return await operation; } finally { if (this.requestTails.get(key) === pending) this.requestTails.delete(key); }
  }
  private async request(brainId: string, action: string, args: Record<string, unknown>, requestId: string, requestSha: string,
    signal: AbortSignal, authority: HostActionOriginAuthority, context?: { chatTurnId?: string; actionEventId?: string }): Promise<unknown> {
    let store = await this.store(brainId);
    const retained = store.request(requestId, action, requestSha);
    if (retained) {
      try {
        if (!retained.result) throw new Error("An earlier sparse request has no terminal receipt; it was not silently repeated.");
        return action === "apply" ? store.job(sparseId(object(retained.result).id)) : retained.result;
      } finally { store.close(); }
    }
    store.beginRequest(requestId, action, requestSha); const root = store.root; store.close();
    const id = SparseSubstrateTransferStore.id();
    await this.dependencies.checkResources(brainId, Buffer.byteLength(canonical(args)) * 16 + 65536, "sparse typed request");
    await this.dependencies.checkDisk(brainId, 131072); await this.checked(authority, brainId, signal);
    if (action === "export" || action === "register-bundle") {
      keys(args, action === "export" ? ["selection"] : ["attachmentId"]);
      const target = await this.destination(root, "bundles", id);
      try {
        let result: ExportResult;
        if (action === "export") result = await this.dependencies.native.export(brainId, object(args.selection) as SparseSubstrateSelection, target.path, signal);
        else {
          if (!context?.chatTurnId || !this.dependencies.selectedBundle) throw new Error("Register bundle requires the actual main-owned current task attachment scope.");
          const selected = await this.dependencies.selectedBundle(brainId, context.chatTurnId, sparseId(args.attachmentId), signal);
          await this.copySelected(brainId, selected, target.path, signal, authority);
          const current = await this.dependencies.selectedBundle(brainId, context.chatTurnId, sparseId(args.attachmentId), signal);
          if (current.path !== selected.path || current.sha256 !== selected.sha256 || current.bytes !== selected.bytes) throw new Error("Selected task attachment/scope changed during bundle custody.");
          result = await this.dependencies.native.validateBundle(brainId, target.path, selected.sha256, signal);
        }
        const manifest = checkedManifest(result);
        if (manifest.format !== SPARSE_SUBSTRATE_FORMAT || action === "export" && manifest.sourceBrainId !== brainId || manifest.cortexStateIncluded !== false || manifest.sourceAuthorityIncluded !== false) throw new Error("Native sparse export provenance differs.");
        await this.checked(authority, brainId, signal);
        const artifact: SparseSubstrateBundle = { format: SPARSE_SUBSTRATE_FORMAT, version: 1, scope: SPARSE_SUBSTRATE_SCOPE, id, ...result,
          sourceBrainId: sparseId(manifest.sourceBrainId), sourceGenerationSha256: sparseHash(manifest.sourceGenerationSha256), relativePath: target.relativePath,
          sourceBinding: action === "export" ? "verified-owned-export" : "selected-native-shape-source-unverified" };
        store = await this.store(brainId); try { store.putArtifact("bundle", artifact); store.finishRequest(requestId, artifact); } finally { store.close(); } return artifact;
      } catch (error) { await rm(target.path, { force: true }).catch(() => undefined); throw error; }
    }
    if (action === "prepare-rollback") {
      keys(args, ["jobId"]); store = await this.store(brainId);
      let original: SparseSubstrateJob, originalPlan: SparseSubstratePlan;
      try { original = store.job(sparseId(args.jobId)); originalPlan = store.artifact(original.planId, "plan"); } finally { store.close(); }
      if (original.state !== "complete" || !original.receipt) throw new Error("An inverse preview requires the original complete native applied receipt.");
      checkedSparseNativeReceipt(original.receipt, original, originalPlan);
      const target = await this.destination(root, "plans", id);
      try {
        const result = await this.dependencies.native.prepareRollback(brainId, join(root, originalPlan.relativePath), originalPlan.sha256,
          original.receipt.receiptJson, original.receipt.receiptSha256, target.path, signal), manifest = checkedManifest(result);
        if (manifest.format !== "omni-sparse-substrate-plan" || manifest.brainId !== brainId || manifest.operation !== "rollback" ||
            object(manifest.rollbackOf).receiptSha256 !== original.receipt.receiptSha256 || manifest.bundleSha256 !== originalPlan.bundleSha256) throw new Error("Inverse native review lost its original applied receipt.");
        await this.checked(authority, brainId, signal);
        const inverse: SparseSubstratePlan = { format: "omni-sparse-substrate-plan", version: 1, scope: SPARSE_SUBSTRATE_SCOPE, id, brainId,
          bundleId: originalPlan.bundleId, bundleSha256: originalPlan.bundleSha256, mode: "replace-exact", operation: "rollback", ...result, relativePath: target.relativePath };
        store = await this.store(brainId); try { store.putArtifact("plan", inverse); store.finishRequest(requestId, inverse); } finally { store.close(); }
        return inverse;
      } catch (error) { await rm(target.path, { force: true }).catch(() => undefined); throw error; }
    }
    keys(args, action === "prepare-import" ? ["bundleId", "mode"] : ["planId", "reviewSha256"]);
    store = await this.store(brainId);
    let plan: SparseSubstratePlan;
    try {
      if (action === "apply") plan = store.artifact<SparseSubstratePlan>(sparseId(args.planId), "plan");
      else {
        const bundle = store.artifact<SparseSubstrateBundle>(sparseId(args.bundleId), "bundle");
        if (args.mode !== "append" && args.mode !== "replace-exact") throw new Error("Select append or replace-exact explicitly.");
        const target = await this.destination(root, "plans", id);
        try {
          const result = await this.dependencies.native.prepare(brainId, join(root, bundle.relativePath), bundle.sha256, target.path, args.mode, signal), manifest = checkedManifest(result);
          if (manifest.format !== "omni-sparse-substrate-plan" || manifest.brainId !== brainId || manifest.bundleSha256 !== bundle.sha256 || manifest.mode !== args.mode) throw new Error("Native sparse preview is not bound to this owner/bundle.");
          await this.checked(authority, brainId, signal);
          plan = { format: "omni-sparse-substrate-plan", version: 1, scope: SPARSE_SUBSTRATE_SCOPE, id, brainId, bundleId: bundle.id, bundleSha256: bundle.sha256,
            mode: args.mode, ...result, relativePath: target.relativePath };
          store.putArtifact("plan", plan); store.finishRequest(requestId, plan); return plan;
        } catch (error) { await rm(target.path, { force: true }).catch(() => undefined); throw error; }
      }
    } finally { store.close(); }
    if (sparseHash(args.reviewSha256) !== plan.sha256 || plan.brainId !== brainId || plan.conflicts !== 0) throw new Error("Apply needs the exact current reviewed conflict-free native plan.");
    const timestamp = new Date().toISOString();
    const job: SparseSubstrateJob = { format: "omni-sparse-substrate-job", version: 1, scope: SPARSE_SUBSTRATE_SCOPE, id, brainId, originalBrainId: brainId,
      requestId, requestSha256: requestSha, planId: plan.id, planSha256: plan.sha256, authorizationRevision: authority.revision,
      ...(context?.actionEventId ? { actionEventId: sparseId(context.actionEventId) } : {}),
      ...(context?.chatTurnId ? { chatTurnId: sparseId(context.chatTurnId) } : {}),
      state: "queued", createdAt: timestamp, updatedAt: timestamp, revision: 1, liveAuthorityPersisted: false };
    store = await this.store(brainId); try { store.createJob(job); store.finishRequest(requestId, job); } finally { store.close(); }
    this.publish(job);
    const controller = new AbortController(), abort = (): void => controller.abort(authority.signal.reason ?? signal.reason);
    authority.signal.addEventListener("abort", abort, { once: true }); signal.addEventListener("abort", abort, { once: true });
    if (authority.signal.aborted || signal.aborted) abort();
    this.live.set(`${brainId}\0${id}`, { controller, authority, unlink: () => { authority.signal.removeEventListener("abort", abort); signal.removeEventListener("abort", abort); } });
    this.dispatch(job); return job;
  }
  private async update(brainId: string, id: string, patch: Partial<Pick<SparseSubstrateJob, "state" | "error" | "receipt">>): Promise<SparseSubstrateJob> {
    const store = await this.store(brainId); let job: SparseSubstrateJob;
    try { job = store.updateJob(id, patch); } finally { store.close(); }
    this.publish(job);
    return job;
  }
  private dispatch(job: SparseSubstrateJob): void {
    const key = `${job.brainId}\0${job.id}`, live = this.live.get(key);
    if (!live || live.promise) return;
    let nativeDispatched = false;
    const run = this.dependencies.deferWrite(job.brainId, async () => {
      await this.checked(live.authority, job.brainId, live.controller.signal);
      if (live.authority.revision !== job.authorizationRevision) throw new SparseTransferBoundaryError("Sparse authorization revision changed; review a new action.", "needs-review");
      await this.dependencies.assertLoadedBoundary(job.brainId);
      const store = await this.store(job.brainId); let plan: SparseSubstratePlan, root: string;
      try {
        const current = store.job(job.id);
        if (!["queued", "boundary-required"].includes(current.state)) throw new Error("This sparse job is not dispatchable.");
        plan = store.artifact(current.planId, "plan"); root = store.root;
      } finally { store.close(); }
      await this.checked(live.authority, job.brainId, live.controller.signal);
      const head = await this.dependencies.native.head(job.brainId, live.controller.signal);
      if (head) {
        if (hash(head.receiptJson) !== sparseHash(head.receiptSha256)) throw new Error("Current native sparse head is unverifiable.");
        const body = object(JSON.parse(head.receiptJson));
        const archive = await this.store(job.brainId); try { archive.archiveHead(sparseId(body.brainId), head.receiptSha256, head.receiptJson); } finally { archive.close(); }
      }
      await this.dependencies.checkResources(job.brainId, 131072, "native sparse import boundary");
      await this.dependencies.checkDisk(job.brainId, plan.bytes * 2 + 131072); await this.checked(live.authority, job.brainId, live.controller.signal);
      await this.update(job.brainId, job.id, { state: "applying" });
      nativeDispatched = true;
      const receipt = checkedSparseNativeReceipt(await this.dependencies.native.apply(job.brainId, join(root, plan.relativePath), {
        jobId: job.id, requestSha256: job.requestSha256, planSha256: plan.sha256,
        ...(head ? { acknowledgedReceiptSha256: head.receiptSha256 } : {}) }, live.controller.signal), job, plan);
      // A genuine committed receipt remains complete even if cancellation or
      // access revocation arrives after the atomic native publication.
      const archive = await this.store(job.brainId); try { archive.archiveHead(job.brainId, receipt.receiptSha256, receipt.receiptJson); } finally { archive.close(); }
      await this.update(job.brainId, job.id, { state: "complete", receipt });
    }, live.controller.signal);
    live.promise = run.catch(async error => {
      let state: SparseSubstrateJob["state"] = nativeDispatched ? "uncertain" : live.controller.signal.aborted ? "cancelled" : "failed";
      let knownUnapplied = false;
      if (error instanceof SparseTransferBoundaryError) {
        knownUnapplied = error.code === "cancelled-unapplied";
        state = knownUnapplied ? "cancelled" : error.code === "boundary-required" ? "boundary-required" : error.code === "needs-review" ? "needs-review" : "uncertain";
        if (error.code === "reload-required") await this.dependencies.ownerReleased?.(job.brainId, error);
      }
      const store = await this.store(job.brainId); try { if (store.job(job.id).state === "applying" && state === "cancelled" && !knownUnapplied) state = "uncertain"; } finally { store.close(); }
      await this.update(job.brainId, job.id, { state, error: error instanceof Error ? error.message : String(error) });
    }).catch(() => undefined).finally(async () => {
      if (!live.controller.signal.aborted && !this.closing) {
        // Only boundary-required jobs retain this process-local lease. No
        // timer/automatic Load or copied grant is manufactured here.
        try { const store = await this.store(job.brainId); try { if (store.job(job.id).state !== "boundary-required") { live.unlink(); this.live.delete(key); } } finally { store.close(); } }
        catch { live.unlink(); this.live.delete(key); }
      } else { live.unlink(); this.live.delete(key); }
      live.promise = undefined;
    });
  }
  /** Root calls on explicit Load/current boundary availability; no polling. */
  async boundaryAvailable(brainId: string): Promise<void> {
    if (![...this.live].some(([key, live]) => key.startsWith(`${brainId}\0`) && !live.promise && !live.controller.signal.aborted)) return;
    const store = await this.store(brainId);
    try { for (const [key, live] of this.live) if (key.startsWith(`${brainId}\0`) && !live.promise && !live.controller.signal.aborted) {
      const id = key.slice(brainId.length + 1), job = store.job(id); if (job.state === "boundary-required") this.dispatch(job);
    } } finally { store.close(); }
  }
  private async readBundle(brainId: string, args: Record<string, unknown>, signal: AbortSignal, authority: HostActionOriginAuthority): Promise<unknown> {
    keys(args, ["bundleId", "offset", "maxBytes"]); const maximum = whole(args.maxBytes, 65536, 262144), offset = args.offset === undefined ? 0 : args.offset;
    if (!Number.isSafeInteger(offset) || (offset as number) < 0) throw new Error("Invalid sparse artifact byte offset.");
    const store = await this.store(brainId); let bundle: SparseSubstrateBundle, root: string;
    try { bundle = store.artifact(sparseId(args.bundleId), "bundle"); root = store.root; } finally { store.close(); }
    if ((offset as number) > bundle.bytes) throw new Error("Sparse page offset is beyond the exact artifact.");
    await this.dependencies.checkResources(brainId, maximum * 8 + 131072, "packed sparse artifact byte page");
    const path = join(root, bundle.relativePath), before = await lstat(path, { bigint: true });
    if (!before.isFile() || before.isSymbolicLink() || before.size !== BigInt(bundle.bytes)) throw new Error("Sparse bundle custody changed.");
    const file = await open(path, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
    try {
      const identity = taskAttachmentFileIdentity(before), cachePrefix = `${brainId}\0${bundle.id}\0`, cacheKey = `${cachePrefix}${bundle.sha256}\0${identity}`;
      const guard = async (): Promise<void> => { await this.checked(authority, brainId, signal);
        if (taskAttachmentFileIdentity(await file.stat({ bigint: true })) !== identity || taskAttachmentFileIdentity(await lstat(path, { bigint: true })) !== identity) {
          for (const key of this.verifiedBundles.keys()) if (key.startsWith(cachePrefix)) this.verifiedBundles.delete(key);
          throw new Error("Sparse bundle FD/path changed during read."); } };
      await guard();
      if (!this.verifiedBundles.has(cacheKey)) {
        for (const key of this.verifiedBundles.keys()) if (key.startsWith(cachePrefix)) this.verifiedBundles.delete(key);
        await this.dependencies.checkResources(brainId, 65536 * 4, "verify exact sparse bundle bytes");
        const digest = createHash("sha256"), scratch = Buffer.alloc(65536); let cursor = 0;
        while (cursor < bundle.bytes) { const row = await file.read(scratch, 0, Math.min(scratch.length, bundle.bytes - cursor), cursor);
          if (!row.bytesRead) throw new Error("Sparse bundle ended before its exact bytes."); digest.update(scratch.subarray(0, row.bytesRead)); cursor += row.bytesRead; await guard(); }
        if (digest.digest("hex") !== bundle.sha256) throw new Error("Sparse bundle content hash changed.");
        await guard(); this.verifiedBundles.set(cacheKey, true);
        while (this.verifiedBundles.size > 32) this.verifiedBundles.delete(this.verifiedBundles.keys().next().value!);
      } else { this.verifiedBundles.delete(cacheKey); this.verifiedBundles.set(cacheKey, true); }
      const bytes = Buffer.alloc(Math.min(maximum, Math.max(0, bundle.bytes - (offset as number))));
      const read = await file.read(bytes, 0, bytes.length, offset as number); await guard();
      if (read.bytesRead !== bytes.length) throw new Error("Sparse byte page was truncated.");
      return { scope: SPARSE_SUBSTRATE_SCOPE, bundleId: bundle.id, sha256: bundle.sha256, totalBytes: bundle.bytes, offset,
        dataBase64: bytes.toString("base64"), pageSha256: hash(bytes), complete: (offset as number) + bytes.length >= bundle.bytes,
        ...((offset as number) + bytes.length < bundle.bytes ? { nextOffset: (offset as number) + bytes.length } : {}) };
    } finally { await file.close(); }
  }
  private async copySelected(brainId: string, selected: { path: string; sha256: string; bytes: number }, destination: string,
    signal: AbortSignal, authority: HostActionOriginAuthority): Promise<void> {
    sparseHash(selected.sha256); if (!Number.isSafeInteger(selected.bytes) || selected.bytes < 1) throw new Error("Selected sparse source has no exact byte count.");
    await this.dependencies.checkResources(brainId, 65536 * 4, "selected sparse bundle copy"); await this.dependencies.checkDisk(brainId, selected.bytes + 65536);
    const before = await lstat(selected.path, { bigint: true });
    if (!before.isFile() || before.isSymbolicLink() || before.size !== BigInt(selected.bytes)) throw new Error("Selected sparse source is not an exact regular attachment.");
    const source = await open(selected.path, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
    let output: Awaited<ReturnType<typeof open>> | undefined;
    try {
      output = await open(destination, "wx", 0o600); const identity = taskAttachmentFileIdentity(before), digest = createHash("sha256"), scratch = Buffer.alloc(65536); let offset = 0;
      while (offset < selected.bytes) {
        await this.checked(authority, brainId, signal);
        if (taskAttachmentFileIdentity(await source.stat({ bigint: true })) !== identity || taskAttachmentFileIdentity(await lstat(selected.path, { bigint: true })) !== identity) throw new Error("Selected sparse source FD/path changed.");
        const row = await source.read(scratch, 0, Math.min(scratch.length, selected.bytes - offset), offset);
        if (!row.bytesRead) throw new Error("Selected sparse attachment ended early.");
        await this.dependencies.checkDisk(brainId, row.bytesRead + 65536); digest.update(scratch.subarray(0, row.bytesRead));
        let written = 0; while (written < row.bytesRead) { const part = await output.write(scratch, written, row.bytesRead - written, offset + written);
          if (!part.bytesWritten) throw new Error("Sparse owned copy made no write progress."); written += part.bytesWritten; }
        offset += row.bytesRead;
      }
      await output.sync(); await this.checked(authority, brainId, signal);
      if (taskAttachmentFileIdentity(await source.stat({ bigint: true })) !== identity || taskAttachmentFileIdentity(await lstat(selected.path, { bigint: true })) !== identity || digest.digest("hex") !== selected.sha256) throw new Error("Selected sparse source changed or failed its complete checksum.");
    } finally { try { await output?.close(); } finally { await source.close(); } }
  }
  async authorityChanged(brainId?: string): Promise<void> {
    const jobs = [...this.live].filter(([key]) => brainId === undefined || key.startsWith(`${brainId}\0`));
    for (const [, live] of jobs) live.controller.abort(new Error("Sparse operation authority was revoked or changed."));
    await Promise.allSettled(jobs.map(([, live]) => live.promise));
    for (const [key, live] of jobs) { if (!live.promise) { const [owner, id] = key.split("\0"); const store = await this.store(owner!);
      try { if (["queued", "boundary-required"].includes(store.job(id!).state)) this.publish(store.updateJob(id!, { state: "needs-review", error: "Current permission/runtime authority changed." })); } finally { store.close(); }
      live.unlink(); this.live.delete(key); } }
  }
  async forget(brainId: string): Promise<void> {
    for (const key of this.verifiedBundles.keys()) if (key.startsWith(`${brainId}\0`)) this.verifiedBundles.delete(key);
    const pending = [...this.live].filter(([key]) => key.startsWith(`${brainId}\0`));
    for (const [, live] of pending) live.controller.abort(new Error("Sparse instance/runtime ownership was released."));
    await Promise.allSettled(pending.map(([, live]) => live.promise));
    for (const [key, live] of pending) { live.unlink(); this.live.delete(key); }
    this.recovered.delete(brainId);
  }
  async shutdown(): Promise<void> { this.closing = true; for (const live of this.live.values()) live.controller.abort(new Error("Studio is closing."));
    await Promise.allSettled([...this.live.values()].map(live => live.promise)); for (const live of this.live.values()) live.unlink(); this.live.clear(); this.verifiedBundles.clear(); this.uiListeners.clear(); }
}
