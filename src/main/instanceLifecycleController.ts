import { randomUUID } from "node:crypto";
import { lstat, mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { join } from "node:path";
import type { InstanceInheritedExecutions, InstanceLifecycleState, InstanceReconnectRequirement, InstanceResumeFrontier } from "../shared/instanceLifecycle";
import type { InstanceLoadFit, NativeInstanceStatus } from "../shared/instanceLifecycleApi";

interface LifecycleDependencies {
  inspect(brainId: string): Promise<NativeInstanceStatus>;
  fit(brainId: string, native: NativeInstanceStatus): Promise<InstanceLoadFit>;
  load(brainId: string): Promise<void>;
  suspend(brainId: string, update: (waitingFor: InstanceLifecycleState["waitingFor"]) => void): Promise<NativeInstanceStatus>;
  resumeBackground?(brainId: string): void;
  releaseDispatch?(brainId: string): void;
}

/** Main-only explicit admission. Preferences are installation-local, not model authority. */
export class InstanceLifecycleController {
  private readonly states = new Map<string, InstanceLifecycleState>();
  private readonly operations = new Map<string, Promise<InstanceLifecycleState>>();
  private readonly operationKinds = new Map<string, "load" | "unload">();
  private readonly listeners = new Set<(state: InstanceLifecycleState) => void>();
  private readonly waiters = new Map<string, Set<() => void>>();
  private readonly admitted = new Set<string>();
  private readonly externalReconnect = new Map<string, Map<string, InstanceReconnectRequirement>>();
  constructor(private readonly root: string, private readonly dependencies: LifecycleDependencies) {}

  onChange(listener: (state: InstanceLifecycleState) => void): () => void {
    this.listeners.add(listener); return () => { this.listeners.delete(listener); };
  }
  isLoaded(brainId: string): boolean {
    return this.admitted.has(brainId) && this.states.get(brainId)?.phase === "loaded";
  }
  canRun(brainId: string, turnId?: string): boolean {
    const state = this.states.get(brainId), frontier = state?.resume;
    const inherited = state?.inheritedExecutions;
    const next = inherited?.entries.find((entry) => entry.status === "queued" || entry.status === "suspended");
    return this.isLoaded(brainId) && (!(frontier || inherited?.rootMutationHeld) || state?.resumeCompatibility?.compatible !== false) && (!frontier || frontier.turnId === turnId) &&
      (!inherited?.rootMutationHeld || next?.turnId === turnId);
  }
  requireLoaded(brainId: string, turnId?: string): void {
    if (this.states.get(brainId)?.resumeCompatibility?.compatible === false) throw new Error(this.states.get(brainId)!.resumeCompatibility!.reason ?? "The exact saved execution is incompatible with this runtime.");
    if (!this.canRun(brainId, turnId)) throw new Error("Load this instance and continue its saved turn before starting new neural work. Unload preserves its supported paused turn.");
  }
  rememberReconnect(brainId: string, requirement: InstanceReconnectRequirement): void {
    const entries = this.externalReconnect.get(brainId) ?? new Map<string, InstanceReconnectRequirement>();
    entries.set(requirement.id, requirement); this.externalReconnect.set(brainId, entries);
  }
  acknowledgeReconnect(brainId: string, ids: readonly string[]): void {
    for (const id of ids) this.externalReconnect.get(brainId)?.delete(id);
    const current = this.states.get(brainId);
    if (current && ids.length) this.update(brainId, { reconnect: current.reconnect.filter((entry) => !ids.includes(entry.id)) });
  }
  private reconnect(brainId: string, native: NativeInstanceStatus): InstanceReconnectRequirement[] {
    return [...new Map([...native.reconnect, ...(this.externalReconnect.get(brainId)?.values() ?? [])].map((entry) => [entry.id, entry])).values()];
  }
  private preferencePath(brainId: string): string {
    if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(brainId)) throw new Error("Invalid instance identity.");
    return join(this.root, ".instance-lifecycle", `${brainId}.json`);
  }
  private async initial(brainId: string): Promise<InstanceLifecycleState> {
    const existing = this.states.get(brainId); if (existing) return existing;
    let autoLoad = false;
    try {
      const path = this.preferencePath(brainId), info = await lstat(path);
      if (!info.isFile() || info.isSymbolicLink() || info.size > 4096) throw new Error("Invalid local lifecycle preference.");
      const value = JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
      if (value.version !== 1 || value.brainId !== brainId || typeof value.autoLoad !== "boolean") throw new Error("Invalid local lifecycle preference.");
      autoLoad = value.autoLoad;
    } catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
    const state: InstanceLifecycleState = { brainId, phase: "unloaded", autoLoad, revision: 0,
      updatedAt: new Date().toISOString(), learnedRuntimeResident: false, pausedJobIds: [], reconnect: [] };
    // Concurrent no-load reads must not overwrite a newer transition.
    if (!this.states.has(brainId)) this.states.set(brainId, state);
    return this.states.get(brainId)!;
  }
  private update(brainId: string, patch: Partial<InstanceLifecycleState>): InstanceLifecycleState {
    const prior = this.states.get(brainId)!;
    const next = { ...prior, ...patch, brainId, revision: prior.revision + 1, updatedAt: new Date().toISOString() };
    this.states.set(brainId, next);
    for (const listener of this.listeners) listener(structuredClone(next));
    return structuredClone(next);
  }
  async status(brainId: string): Promise<InstanceLifecycleState> {
    const prior = await this.initial(brainId);
    if (this.operations.has(brainId)) return structuredClone(this.states.get(brainId)!);
    const native = await this.dependencies.inspect(brainId);
    if (this.operations.has(brainId) || this.states.get(brainId)?.revision !== prior.revision) return structuredClone(this.states.get(brainId)!);
    if (native.brainId !== brainId) throw new Error("Native lifecycle owner mismatch.");
    if (!native.learnedRuntimeResident) this.admitted.delete(brainId);
    const state = this.update(brainId, { learnedRuntimeResident: native.learnedRuntimeResident,
      phase: this.isLoaded(brainId) ? "loaded" : native.learnedRuntimeResident && (native.waitingFor || prior.phase === "suspending")
        ? "suspending" : (native.resume || native.inheritedExecutions?.pending) && !native.learnedRuntimeResident ? "suspended" : "unloaded",
      resume: native.resume, resumeCompatibility: native.resumeCompatibility, inheritedExecutions: native.inheritedExecutions,
      reconnect: this.reconnect(brainId, native), waitingFor: native.waitingFor });
    for (const wake of this.waiters.get(brainId) ?? []) wake();
    if ((prior.resume || prior.inheritedExecutions?.rootMutationHeld) && this.canRun(brainId)) {
      this.dependencies.releaseDispatch?.(brainId); this.dependencies.resumeBackground?.(brainId);
    }
    return state;
  }
  async fit(brainId: string): Promise<InstanceLoadFit> {
    await this.initial(brainId);
    return this.dependencies.fit(brainId, await this.dependencies.inspect(brainId));
  }
  async setAutoLoad(brainId: string, enabled: boolean): Promise<InstanceLifecycleState> {
    if (typeof enabled !== "boolean") throw new Error("Auto-load must be enabled or disabled.");
    await this.initial(brainId);
    const path = this.preferencePath(brainId), temporary = `${path}.${randomUUID()}.tmp`;
    await mkdir(join(this.root, ".instance-lifecycle"), { recursive: true, mode: 0o700 });
    const file = await open(temporary, "wx", 0o600);
    try { await file.writeFile(JSON.stringify({ version: 1, brainId, autoLoad: enabled })); await file.sync(); }
    finally { await file.close(); }
    try { await rename(temporary, path); } catch (error) { await rm(temporary, { force: true }); throw error; }
    return this.update(brainId, { autoLoad: enabled });
  }
  async load(brainId: string, expectedCheckpointId?: string): Promise<InstanceLifecycleState> {
    const ongoing = this.operations.get(brainId);
    if (ongoing) { if (this.operationKinds.get(brainId) === "load") return ongoing; await ongoing; return this.load(brainId, expectedCheckpointId); }
    await this.initial(brainId);
    if (this.operations.has(brainId)) return this.load(brainId, expectedCheckpointId);
    const operation = (async () => {
      this.update(brainId, { phase: "checking", error: undefined, waitingFor: { kind: "resources", label: "Checking the saved shape and current resource envelope" } });
      try {
        const native = await this.dependencies.inspect(brainId);
        if (native.resumeCompatibility?.compatible === false) throw new Error(native.resumeCompatibility.reason ?? "This saved frontier cannot resume exactly on this runtime.");
        if (expectedCheckpointId && native.resume?.checkpointId !== expectedCheckpointId) throw new Error("The saved execution checkpoint changed. Review its current resume receipt.");
        const fit = await this.dependencies.fit(brainId, native);
        if (!fit.allowed) return this.update(brainId, { phase: native.resume || native.inheritedExecutions?.pending ? "suspended" : "unloaded",
          learnedRuntimeResident: native.learnedRuntimeResident, resume: native.resume, resumeCompatibility: native.resumeCompatibility, inheritedExecutions: native.inheritedExecutions,
          waitingFor: { kind: "resources", label: fit.blockers.join(" ") },
          error: { code: "RESOURCE_FIT", message: fit.blockers.join(" "), recoverable: true } });
        this.update(brainId, { phase: native.resume ? "resuming" : "loading", resume: native.resume, resumeCompatibility: native.resumeCompatibility, inheritedExecutions: native.inheritedExecutions,
          reconnect: this.reconnect(brainId, native), waitingFor: undefined });
        await this.dependencies.load(brainId);
        const loaded = await this.dependencies.inspect(brainId);
        if (!loaded.learnedRuntimeResident) throw new Error("The worker has not acknowledged a resident learned runtime.");
        this.admitted.add(brainId);
        const state = this.update(brainId, { phase: "loaded", learnedRuntimeResident: true, resume: loaded.resume, resumeCompatibility: loaded.resumeCompatibility, inheritedExecutions: loaded.inheritedExecutions,
          reconnect: this.reconnect(brainId, loaded), waitingFor: undefined, error: undefined });
        for (const wake of this.waiters.get(brainId) ?? []) wake();
        if (!loaded.resume && !loaded.inheritedExecutions?.rootMutationHeld) { this.dependencies.releaseDispatch?.(brainId); this.dependencies.resumeBackground?.(brainId); }
        return state;
      } catch (error) {
        this.admitted.delete(brainId);
        return this.update(brainId, { phase: "error", waitingFor: undefined,
          error: { code: "LOAD_FAILED", message: error instanceof Error ? error.message : "Instance load failed.", recoverable: true } });
      }
    })();
    this.operations.set(brainId, operation);
    this.operationKinds.set(brainId, "load");
    try { return await operation; } finally { if (this.operations.get(brainId) === operation) { this.operations.delete(brainId); this.operationKinds.delete(brainId); } }
  }
  async unload(brainId: string): Promise<InstanceLifecycleState> {
    const ongoing = this.operations.get(brainId);
    if (ongoing) { if (this.operationKinds.get(brainId) === "unload") return ongoing; await ongoing; return this.unload(brainId); }
    await this.initial(brainId);
    if (this.operations.has(brainId)) return this.unload(brainId);
    this.admitted.delete(brainId);
    const operation = (async () => {
      this.update(brainId, { phase: "suspending", error: undefined,
        waitingFor: { kind: "native-boundary", label: "Waiting for a supported checkpoint boundary" } });
      try {
        const result = await this.dependencies.suspend(brainId, (waitingFor) => this.update(brainId, { waitingFor }));
        if (result.unloaded !== true || result.learnedRuntimeResident) {
          return this.update(brainId, { phase: "suspending", learnedRuntimeResident: result.learnedRuntimeResident,
            resume: result.resume, resumeCompatibility: result.resumeCompatibility, inheritedExecutions: result.inheritedExecutions, reconnect: this.reconnect(brainId, result),
            waitingFor: result.waitingFor ?? { kind: "native-boundary", label: "Native runtime release is not acknowledged" } });
        }
        return this.update(brainId, { phase: result.resume || result.inheritedExecutions?.pending ? "suspended" : "unloaded", learnedRuntimeResident: false,
          resume: result.resume, resumeCompatibility: result.resumeCompatibility, inheritedExecutions: result.inheritedExecutions, reconnect: this.reconnect(brainId, result), waitingFor: undefined });
      } catch (error) {
        return this.update(brainId, { phase: "error", error: { code: "SUSPEND_FAILED",
          message: error instanceof Error ? error.message : "Instance suspension failed.", recoverable: true } });
      }
    })();
    this.operations.set(brainId, operation);
    this.operationKinds.set(brainId, "unload");
    try { return await operation; } finally { if (this.operations.get(brainId) === operation) { this.operations.delete(brainId); this.operationKinds.delete(brainId); } }
  }
  async waitUntilLoaded(brainId: string, signal?: AbortSignal, turnId?: string): Promise<void> {
    signal?.throwIfAborted(); if (this.canRun(brainId, turnId)) return;
    await new Promise<void>((resolve, reject) => {
      const waiting = this.waiters.get(brainId) ?? new Set<() => void>(); this.waiters.set(brainId, waiting);
      const cleanup = () => { waiting.delete(wake); signal?.removeEventListener("abort", abort); if (!waiting.size) this.waiters.delete(brainId); };
      const wake = () => { if (!this.canRun(brainId, turnId)) return; cleanup(); resolve(); };
      const abort = () => { cleanup(); reject(signal?.reason ?? new DOMException("Stopped", "AbortError")); };
      waiting.add(wake); signal?.addEventListener("abort", abort, { once: true });
      if (signal?.aborted) abort(); else wake();
    });
  }
  noteFrontier(brainId: string, resume: InstanceResumeFrontier): void {
    this.admitted.delete(brainId);
    if (this.states.has(brainId)) this.update(brainId, { resume, resumeCompatibility: undefined });
  }
  noteInheritedFrontier(brainId: string, inheritedExecutions: InstanceInheritedExecutions): void {
    if (!inheritedExecutions.rootMutationHeld || inheritedExecutions.pending < 1) throw new Error("The inherited native frontier is not durably pending.");
    this.admitted.delete(brainId);
    if (this.states.has(brainId)) this.update(brainId, { resume: undefined, resumeCompatibility: undefined, inheritedExecutions });
  }
  completed(brainId: string): void {
    if (this.states.has(brainId)) {
      this.update(brainId, { resume: undefined, resumeCompatibility: undefined });
      for (const wake of this.waiters.get(brainId) ?? []) wake();
      if (this.canRun(brainId)) { this.dependencies.releaseDispatch?.(brainId); this.dependencies.resumeBackground?.(brainId); }
    }
  }
  workerClosed(): void {
    this.admitted.clear();
    for (const brainId of this.states.keys()) this.update(brainId, { phase: "unloaded", learnedRuntimeResident: false,
      waitingFor: undefined, error: { code: "WORKER_CLOSED", message: "Runtime disconnected. Load checks the durable continuation before restarting work.", recoverable: true } });
  }
  /** Main-only, correlated native release under the same brain writer. This
   * is not an invented Unload checkpoint or a late unowned transport event. */
  async confirmedNativeOwnerRelease(brainId: string): Promise<InstanceLifecycleState> {
    await this.initial(brainId); this.admitted.delete(brainId);
    return this.update(brainId, { phase: "unloaded", learnedRuntimeResident: false, waitingFor: undefined,
      error: { code: "NATIVE_OWNER_RELEASED", message: "The native transaction released its RAM owner. Explicitly Load the durable checkpoint before continuing.", recoverable: true } });
  }
}
