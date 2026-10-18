import { createHash, randomUUID } from "node:crypto";
import type { EvolutionController, EvolutionRunRecord } from "./evolutionController";
import type { HostActionOriginAuthority } from "../shared/hostActionAuthority";
import {
  nativeCandidateArgumentsJson,
  nativeCandidateToolRequest,
  type NativeCandidateToolJob,
  type NativeCandidateToolState
} from "../shared/nativeCandidateTool";

/** Only a main-owned, observed lifecycle state may construct a boundary hold. */
export class NativeCandidateToolBoundaryError extends Error {
  constructor(readonly code: "waiting-load" | "waiting-boundary", message: string) { super(message); }
}

export interface NativeCandidateToolDependencies {
  controller: Pick<EvolutionController, "startAuthorized" | "nativeToolJob" | "nativeToolJobForRequest" | "nativeToolRun" |
    "putNativeToolJob" | "updateNativeToolJob" | "deliverNativeToolReassessment" | "stop">;
  /** Existing tracked FIFO; must resolve only after its writer is released. */
  deferWrite<T>(brainId: string, operation: () => Promise<T>, signal: AbortSignal): Promise<T>;
  assertOwner(brainId: string): Promise<void>;
  /** Never starts a worker or Loads. A known unavailable frontier throws the typed hold. */
  assertLoadedBoundary(brainId: string): Promise<void>;
  checkResources(brainId: string, bytes: number, purpose: string): Promise<void>;
  changed?(job: NativeCandidateToolJob): void;
}

interface LiveJob {
  job: NativeCandidateToolJob;
  controller: AbortController;
  authority: HostActionOriginAuthority;
  dispose(): void;
  work?: Promise<void>;
  stopWork?: Promise<void>;
  controllerStarted: boolean;
  pendingReassessment?: string;
}
const pendingStates = new Set<NativeCandidateToolState>(["queued", "waiting-load", "waiting-boundary", "running", "stopping"]);
const identityPattern = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
function identity(value: unknown, label: string): string {
  if (typeof value !== "string" || identityPattern.exec(value)?.[0] !== value) throw new Error(`Invalid ${label}.`);
  return value;
}
function message(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function hash(value: string): string { return createHash("sha256").update(value).digest("hex"); }

/**
 * Advertised native proposals use the existing native candidate evaluator,
 * not Git edits and not a second model. Only inert evidence is durable; work
 * requires the exact still-live main permission/origin lease on every boundary.
 */
export class NativeCandidateToolService {
  private readonly live = new Map<string, LiveJob>();
  private readonly requestTails = new Map<string, { sha: string; promise: Promise<NativeCandidateToolJob> }>();
  private closed = false;
  constructor(private readonly dependencies: NativeCandidateToolDependencies) {}

  async execute(brainId: string, action: string, args: Record<string, unknown>, requestId: string,
    signal: AbortSignal, authority?: HostActionOriginAuthority): Promise<NativeCandidateToolJob> {
    identity(brainId, "native candidate owner"); identity(requestId, "native candidate request");
    if (!authority) throw new Error("Native candidate tools require the current main-owned permission/origin scope.");
    if (this.closed) throw new Error("Native candidate tools are shutting down.");
    const raw = JSON.parse(nativeCandidateArgumentsJson(args)) as Record<string, unknown>;
    const source = { revision: authority.revision, signal: authority.signal, assertCurrent: authority.assertCurrent };
    const assertAction = async (): Promise<void> => {
      signal.throwIfAborted(); source.signal.throwIfAborted();
      if (authority.revision !== source.revision || authority.signal !== source.signal || authority.assertCurrent !== source.assertCurrent) throw new Error("Native tool origin scope changed.");
      await source.assertCurrent.call(authority);
      signal.throwIfAborted(); source.signal.throwIfAborted();
      if (authority.revision !== source.revision || authority.signal !== source.signal || authority.assertCurrent !== source.assertCurrent) throw new Error("Native tool origin scope changed during validation.");
    };
    await assertAction();
    await this.dependencies.assertOwner(brainId);
    await assertAction();
    if (action === "native-status" || action === "native-stop") {
      if (Object.keys(raw).length !== 1 || !Object.hasOwn(raw, "jobId")) throw new Error("Native status/Stop requires only an exact jobId.");
      const jobId = identity(raw.jobId, "native candidate job");
      await assertAction();
      return action === "native-status" ? this.status(brainId, jobId) : this.stop(brainId, jobId, {
        revision: source.revision, signal: source.signal,
        assertCurrent: async () => { await assertAction(); await this.dependencies.assertOwner(brainId); await this.dependencies.assertLoadedBoundary(brainId); await assertAction(); }
      });
    }
    if (action !== "propose") throw new Error("Unknown native candidate tool action.");
    // Validate and copy before the first admission await. The persisted hash
    // binds exact semantic operands, not mutable caller-owned objects.
    const request = nativeCandidateToolRequest(brainId, raw);
    const sha = hash(nativeCandidateArgumentsJson({ brainId, arguments: raw }));
    const key = `${brainId}\0${requestId}`;
    const previous = this.requestTails.get(key);
    if (previous) {
      if (previous.sha !== sha) throw new Error("The concurrent native request changed operands.");
      return previous.promise;
    }
    const promise = this.propose(brainId, requestId, sha, raw, request, signal, authority).finally(() => {
      if (this.requestTails.get(key)?.promise === promise) this.requestTails.delete(key);
    });
    this.requestTails.set(key, { sha, promise });
    return promise;
  }

  private async propose(brainId: string, requestId: string, requestSha256: string, raw: Record<string, unknown>,
    request: ReturnType<typeof nativeCandidateToolRequest>, signal: AbortSignal,
    source: HostActionOriginAuthority): Promise<NativeCandidateToolJob> {
    const original = { revision: source.revision, signal: source.signal, assertCurrent: source.assertCurrent };
    const retained = await this.dependencies.controller.nativeToolJobForRequest(brainId, requestId);
    if (retained) {
      if (retained.requestSha256 !== requestSha256) throw new Error("The saved native request changed operands.");
      return this.retireWithoutGrant(brainId, retained);
    }
    const control = new AbortController();
    const abortFromSource = (): void => { if (!control.signal.aborted) control.abort(original.signal.reason ?? new Error("Native source authority ended.")); };
    const abortFromTool = (): void => { if (!control.signal.aborted) control.abort(signal.reason ?? new Error("Native proposal was stopped.")); };
    original.signal.addEventListener("abort", abortFromSource, { once: true });
    signal.addEventListener("abort", abortFromTool, { once: true });
    if (original.signal.aborted) abortFromSource(); if (signal.aborted) abortFromTool();
    const dispose = (): void => { original.signal.removeEventListener("abort", abortFromSource); signal.removeEventListener("abort", abortFromTool); };
    const baseGuard = async (): Promise<void> => {
      if (this.closed) throw new Error("Native candidate tools are shutting down.");
      control.signal.throwIfAborted();
      if (source.revision !== original.revision || source.signal !== original.signal || source.assertCurrent !== original.assertCurrent) throw new Error("The native proposal's origin scope was substituted.");
      await original.assertCurrent.call(source); control.signal.throwIfAborted();
      if (source.revision !== original.revision || source.signal !== original.signal || source.assertCurrent !== original.assertCurrent) throw new Error("The native proposal's origin scope changed during validation.");
      await this.dependencies.assertOwner(brainId); control.signal.throwIfAborted();
    };
    try {
      const id = randomUUID(), now = new Date().toISOString();
      const job: NativeCandidateToolJob = {
        format: "omni-native-candidate-tool-job", version: 1, id, brainId, requestId, requestSha256,
        candidateKind: request.candidateKind, arguments: raw,
        request: { ...request, provenance: { source: "permissioned-native-candidate-tool", jobId: id, requestId,
          ...(request.provenance === undefined ? {} : { declared: structuredClone(request.provenance) }) } },
        authorizationRevision: original.revision, state: "queued", createdAt: now, updatedAt: now, revision: 1,
        liveAuthorityPersisted: false
      };
      await baseGuard();
      await this.dependencies.checkResources(brainId, Buffer.byteLength(JSON.stringify(job)), "Native candidate request and durable job evidence");
      await baseGuard();
      const saved = await this.dependencies.controller.putNativeToolJob(brainId, job);
      if (saved.id !== id) { dispose(); return this.retireWithoutGrant(brainId, saved); }
      const live: LiveJob = { job: saved, controller: control, dispose, controllerStarted: false,
        authority: { revision: original.revision, signal: control.signal, assertCurrent: async () => {
          await baseGuard();
          if (this.live.get(id) !== live) throw new Error("The native proposal no longer owns its deferred operation.");
          await this.dependencies.assertLoadedBoundary(brainId);
          control.signal.throwIfAborted(); await baseGuard();
        } }
      };
      this.live.set(id, live); this.publish(saved); this.schedule(live);
      return structuredClone(saved);
    } catch (error) { dispose(); throw error; }
  }

  private publish(job: NativeCandidateToolJob): void {
    try { this.dependencies.changed?.(structuredClone(job)); } catch { /* Observers never own cancellation or durability. */ }
  }
  private async update(live: LiveJob, patch: Parameters<NativeCandidateToolDependencies["controller"]["updateNativeToolJob"]>[2]): Promise<void> {
    live.job = await this.dependencies.controller.updateNativeToolJob(live.job.brainId, live.job.id, patch);
    this.publish(live.job);
  }
  private schedule(live: LiveJob): void {
    if (live.work || this.live.get(live.job.id) !== live || this.closed) return;
    const work = this.run(live).finally(() => {
      if (live.work === work) delete live.work;
      if (!["waiting-load", "waiting-boundary"].includes(live.job.state) && !live.stopWork) {
        live.dispose(); if (this.live.get(live.job.id) === live) this.live.delete(live.job.id);
      }
    });
    live.work = work;
    // run() records terminal uncertainty. A persistence failure remains visible
    // in the live status and never starts a second operation.
    void work.catch(error => { live.job = { ...live.job, state: "uncertain", error: `Native job evidence could not be updated: ${message(error)}` }; this.publish(live.job); });
  }

  private async run(live: LiveJob): Promise<void> {
    try {
      await this.dependencies.deferWrite(live.job.brainId, async () => {
        await live.authority.assertCurrent();
        await this.update(live, { state: "running", error: undefined });
        await live.authority.assertCurrent();
        live.controllerStarted = true;
        const run = await this.dependencies.controller.startAuthorized(structuredClone(live.job.request), {
          authority: live.authority,
          onRunCreated: async (runId, candidateId) => { await this.update(live, { runId, candidateId }); },
          deferReassessment: async event => {
            live.pendingReassessment = event.parentCandidateId;
            await this.update(live, { reassessmentPending: true });
          }
        });
        await this.recordRun(live, run);
      }, live.controller.signal);
      // The existing result-learning handler acquires its own writer. Never
      // await it from the candidate's FIFO operation (that would deadlock).
      if (live.pendingReassessment) {
        try {
          await this.dependencies.controller.deliverNativeToolReassessment(live.job.brainId, live.pendingReassessment, {
            signal: live.controller.signal, assertCurrent: () => live.authority.assertCurrent(), loadedOnly: true
          });
          await this.update(live, { reassessmentPending: false, reassessmentError: undefined });
        } catch (error) {
          await this.update(live, { reassessmentPending: true, reassessmentError: message(error) });
        }
      }
    } catch (error) {
      if (!live.controllerStarted && error instanceof NativeCandidateToolBoundaryError && !live.controller.signal.aborted && !this.closed) {
        await this.update(live, { state: error.code, error: message(error) }); return;
      }
      const run = live.job.runId ? await this.dependencies.controller.nativeToolRun(live.job.brainId, live.job.runId) : undefined;
      if (run?.state === "promoted" || run?.state === "rolled-back") { await this.recordRun(live, run); return; }
      await this.update(live, { ...(run ? { run } : {}),
        state: live.controllerStarted ? run && ["failed", "stopped", "rejected"].includes(run.state) ? run.state === "stopped" ? "cancelled" : "failed" : "uncertain" : live.controller.signal.aborted ? "cancelled" : "failed",
        error: message(error) });
    }
  }
  private async recordRun(live: LiveJob, run: EvolutionRunRecord): Promise<void> {
    await this.update(live, { run, runId: run.id,
      state: run.state === "stopped" ? "cancelled" : run.state === "failed" ? "failed" : "complete", error: run.error });
  }
  private async retireWithoutGrant(brainId: string, job: NativeCandidateToolJob): Promise<NativeCandidateToolJob> {
    const live = this.live.get(job.id);
    if (live?.job.brainId === brainId && job.brainId === brainId) return structuredClone(live.job);
    if (!pendingStates.has(job.state)) return structuredClone(job);
    if (job.brainId !== brainId) return { ...structuredClone(job), state: "dormant", error: "Imported native operation history carries no executable source authority." };
    const state = ["running", "stopping"].includes(job.state) ? "uncertain" : "dormant";
    const saved = await this.dependencies.controller.updateNativeToolJob(brainId, job.id, { state,
      error: state === "uncertain" ? "The previous process ended without a terminal native-operation receipt; inspect the actual candidate before any new request." : "Deferred native authority was not retained. A new reviewed request is required." });
    this.publish(saved); return saved;
  }
  private async status(brainId: string, jobId: string): Promise<NativeCandidateToolJob> {
    const live = this.live.get(jobId);
    if (live && live.job.brainId === brainId) return structuredClone(live.job);
    return this.retireWithoutGrant(brainId, await this.dependencies.controller.nativeToolJob(brainId, jobId));
  }

  private async stop(brainId: string, jobId: string, stopAuthority: HostActionOriginAuthority): Promise<NativeCandidateToolJob> {
    const live = this.live.get(jobId);
    if (!live || live.job.brainId !== brainId) return this.status(brainId, jobId);
    if (!pendingStates.has(live.job.state)) return structuredClone(live.job);
    if (live.stopWork) return structuredClone(live.job);
    if (!live.controller.signal.aborted) live.controller.abort(new Error("Native candidate Stop requested."));
    await this.update(live, { state: live.controllerStarted ? "stopping" : "cancelled", error: "Native candidate Stop requested." });
    // Only this live operation's actual controller-created run can be stopped.
    // A copied ID or a caller-supplied runId never enters the cancellation API.
    const stopWork = (async () => {
      try {
        if (live.job.runId && live.controllerStarted) {
          await stopAuthority.assertCurrent();
          await this.dependencies.controller.stop(brainId, live.job.runId, stopAuthority);
        }
        await live.work;
        const run = live.job.runId ? await this.dependencies.controller.nativeToolRun(brainId, live.job.runId) : undefined;
        if (run) await this.recordRun(live, run);
        else await this.update(live, { state: "cancelled" });
      } catch (error) {
        const run = live.job.runId ? await this.dependencies.controller.nativeToolRun(brainId, live.job.runId) : undefined;
        if (run?.state === "promoted" || run?.state === "rolled-back") await this.recordRun(live, run);
        else await this.update(live, { ...(run ? { run } : {}), state: "uncertain", error: `Stop is awaiting a verifiable terminal outcome: ${message(error)}` });
      } finally { live.dispose(); if (this.live.get(jobId) === live) this.live.delete(jobId); }
    })();
    live.stopWork = stopWork; void stopWork.catch(error => { live.job = { ...live.job, state: "uncertain", error: message(error) }; this.publish(live.job); });
    return structuredClone(live.job);
  }

  async boundaryAvailable(brainId: string): Promise<void> {
    if (this.closed) return;
    for (const live of this.live.values()) if (live.job.brainId === brainId && ["waiting-load", "waiting-boundary"].includes(live.job.state) && !live.work && !live.controller.signal.aborted) {
      await this.update(live, { state: "queued", error: undefined }); this.schedule(live);
    }
  }
  async authorityChanged(brainId?: string): Promise<void> {
    for (const live of [...this.live.values()]) if (brainId === undefined || live.job.brainId === brainId) {
      if (!live.controller.signal.aborted) live.controller.abort(new Error("Native candidate permission/origin changed; no new operation is authorized."));
      if (!live.work && !live.stopWork) {
        try { await this.update(live, { state: "cancelled", error: "Deferred source authority was revoked." }); }
        finally { live.dispose(); this.live.delete(live.job.id); }
      }
    }
  }
  async forget(brainId: string): Promise<void> { await this.authorityChanged(brainId); }
  async shutdown(): Promise<void> {
    if (this.closed) return; this.closed = true;
    await this.authorityChanged();
    await Promise.allSettled([...this.live.values()].flatMap(live => [live.work, live.stopWork].filter((work): work is Promise<void> => Boolean(work))));
  }
}
