import { randomUUID } from "node:crypto";
import type { ToolPermissionLevel } from "../shared/types";
import { PROCESS_MEMORY_FORMAT, PROCESS_MEMORY_MAX_BYTES, PROCESS_MEMORY_MAX_REGIONS,
  type ProcessMemoryCapabilities, type ProcessMemoryNativeRequest, type ProcessMemoryPreferenceUpdate,
  type ProcessMemoryPreferences, type ProcessMemoryReceipt, type ProcessMemoryTarget } from "../shared/processMemory";
import { memoryAddress, memoryBytes, memoryId, memoryKeys, memoryObject, memorySha, memoryWhole,
  memoryReceipt, processMemoryCanonical, processMemoryHash } from "./processMemoryFormat";
import { ProcessMemoryStore } from "./processMemoryStore";
import { ProcessMemoryOperationError, uncertainMemoryReceipt,
  type ProcessMemoryBridgeContext } from "./processMemoryBridge";

export interface ProcessMemoryDependencies {
  /** Must be outside every instance/export/import/recovery/merge directory. */
  installationDirectory: string;
  assertBrainOwner(brainId: string): Promise<void>;
  /** Actual host lifecycle; settings/inspection alone never load it. */
  requireActiveOwner?(brainId: string): Promise<void>;
  currentPermission(brainId: string): Promise<{ level: ToolPermissionLevel; revision: string }>;
  protectedPids(): readonly number[];
  checkResources(brainId: string, bytes: number, purpose: string, childPids?: readonly number[]): Promise<void>;
  reserveWrite(bytes: number): Promise<void>;
  run(request: ProcessMemoryNativeRequest, context: ProcessMemoryBridgeContext): Promise<ProcessMemoryReceipt>;
  platform?: NodeJS.Platform;
}
interface TargetLease { target: ProcessMemoryTarget; permissionRevision: string; generation: number }
interface ActiveOperation { brainId: string; controller: AbortController; promise: Promise<unknown> }

export function processMemoryCapabilities(platform: string): ProcessMemoryCapabilities {
  const adapter = platform === "darwin" ? "mach-task-port" : platform === "win32" ? "windows-process-handle" : platform === "linux" ? "proc-mem-fd" : "unsupported";
  return { platform, adapter, implemented: adapter !== "unsupported", targetAccess: "not-probed", maxTransferBytes: PROCESS_MEMORY_MAX_BYTES,
    maxRegionPage: PROCESS_MEMORY_MAX_REGIONS, environment: "host", physicalRamAccess: false, elevationPerformed: false, protectionChanges: false,
    detail: "Advanced process-memory access for capable instances/hosts. Explicit PID and current OS rights are required; many protected applications are unavailable. No access to physical RAM or automatic elevation. Transfers are bounded pages, not a whole-process dump. A write preimage check is not atomic against a running target." };
}

/** Process-private target leases. There is deliberately no model/renderer setter. */
export class ProcessMemoryService {
  private readonly targets = new Map<string, TargetLease>();
  private readonly generations = new Map<string, number>();
  private readonly active = new Map<string, ActiveOperation>();
  private readonly writeQueues = new Map<string, Promise<unknown>>();
  private readonly forcedOff = new Set<string>();
  private closing = false;
  constructor(private readonly dependencies: ProcessMemoryDependencies) {}
  private async store(): Promise<ProcessMemoryStore> { return ProcessMemoryStore.open(this.dependencies.installationDirectory); }
  private generation(brainId: string): number { return this.generations.get(brainId) ?? 0; }
  private async settings(brainId: string): Promise<Pick<ProcessMemoryPreferences, "enabled" | "revision" | "updatedAt">> {
    const store = await ProcessMemoryStore.open(this.dependencies.installationDirectory, false);
    try { const saved = store?.preferences(brainId) ?? { enabled: false, revision: 0, updatedAt: null };
      return this.forcedOff.has(brainId) ? { ...saved, enabled: false } : saved;
    } finally { store?.close(); }
  }
  async preferences(brainId: string): Promise<ProcessMemoryPreferences> {
    memoryId(brainId); await this.dependencies.assertBrainOwner(brainId);
    return { brainId, ...await this.settings(brainId), authority: "current-installation-only",
      capabilities: processMemoryCapabilities(this.dependencies.platform ?? process.platform) };
  }
  /** Trusted main IPC only. Tool calls can never enable this switch themselves. */
  async update(value: ProcessMemoryPreferenceUpdate): Promise<ProcessMemoryPreferences> {
    const row = memoryObject(value); memoryKeys(row, ["brainId", "expectedRevision", "enabled"]);
    const brainId = memoryId(row.brainId);
    await this.dependencies.assertBrainOwner(brainId);
    if (typeof row.enabled !== "boolean") throw new Error("DMA must be explicitly enabled or disabled.");
    const expected = memoryWhole(row.expectedRevision, Number.MAX_SAFE_INTEGER - 1, 0);
    if ((await this.settings(brainId)).revision !== expected) throw new Error("Host-memory preferences changed; reload them before saving.");
    if (!row.enabled) {
      // Revocation is immediate even if storage is full. Failure to persist
      // must not leave an already-running process-memory grant usable.
      this.forcedOff.add(brainId); this.invalidate(brainId, "DMA was disabled; host process leases were revoked.");
    }
    try {
      await this.dependencies.reserveWrite(65536);
      const store = await this.store();
      try { store.update(brainId, expected, row.enabled); }
      finally { store.close(); }
    } catch (error) {
      if (!row.enabled) throw new Error(`DMA is disabled in this running Studio session, but the change could not be saved. Recheck after restarting: ${error instanceof Error ? error.message : String(error)}`);
      throw error;
    }
    if (row.enabled) this.forcedOff.delete(brainId);
    this.invalidate(brainId, "The DMA preference changed; reattach an exact target under the new setting.");
    return this.preferences(brainId);
  }
  private invalidate(brainId: string, reason: string): void {
    this.generations.set(brainId, this.generation(brainId) + 1);
    for (const [key, lease] of this.targets) if (lease.target.brainId === brainId) this.targets.delete(key);
    for (const item of this.active.values()) if (item.brainId === brainId) item.controller.abort(new Error(reason));
  }
  permissionChanged(brainId: string): void { this.invalidate(brainId, "The process-memory tool permission changed."); }
  private async retainWrite(brainId: string, requestId: string, receipt: ProcessMemoryReceipt): Promise<void> {
    try { const store = await this.store(); try { store.terminal(brainId, requestId, receipt); } finally { store.close(); } }
    catch (error) {
      throw new ProcessMemoryOperationError(`The actual memory result is retained here, but its local effect journal could not finish; do not repeat the write: ${error instanceof Error ? error.message : String(error)}`,
        { ...receipt, error: `${receipt.error ? `${receipt.error}; ` : ""}The local effect-journal save failed; do not automatically retry.` });
    }
  }
  async forget(brainId: string): Promise<void> {
    this.invalidate(brainId, "The instance was unloaded or its ownership changed; process leases were released.");
    await Promise.allSettled([...this.active.values()].filter(item => item.brainId === brainId).map(item => item.promise));
  }
  async shutdown(): Promise<void> {
    this.closing = true;
    for (const brainId of new Set([...this.targets.values()].map(item => item.target.brainId).concat([...this.active.values()].map(item => item.brainId))))
      this.invalidate(brainId, "Studio is closing; no new process-memory operation is admitted.");
    await Promise.allSettled([...this.active.values()].map(item => item.promise));
  }
  private protected(pid: number): void {
    if (pid < 2 || pid === process.pid || this.dependencies.protectedPids().includes(pid)) throw new Error("The kernel/system or Omni authority/worker process is a protected target.");
  }
  async execute(brainId: string, action: string, value: Record<string, unknown>, requestId: string, signal: AbortSignal): Promise<unknown> {
    memoryId(brainId); memoryId(requestId); signal.throwIfAborted();
    const args = memoryObject(value);
    if (action === "capabilities") { memoryKeys(args, []); return this.preferences(brainId); }
    if (action === "detach") {
      memoryKeys(args, ["targetId"]); const key = memoryId(args.targetId), lease = this.targets.get(key);
      if (!lease || lease.target.brainId !== brainId) throw new Error("The process-memory lease is not owned by this instance/session.");
      this.targets.delete(key); return { targetId: key, detached: true, osHandleRetained: false };
    }
    if (!["attach", "regions", "read", "write"].includes(action)) throw new Error("Unsupported process-memory action.");
    const key = `${brainId}\0${requestId}`;
    if (this.active.has(key)) throw new Error("This exact process-memory request is already in flight; do not duplicate it.");
    const controller = new AbortController(), abort = () => controller.abort(signal.reason);
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
    const prior = action === "write" ? this.writeQueues.get(brainId) : undefined;
    const promise = (async () => {
      if (prior) await prior.catch(() => undefined);
      controller.signal.throwIfAborted();
      return this.perform(brainId, action as ProcessMemoryNativeRequest["operation"], args, requestId, controller.signal);
    })().finally(() => {
      signal.removeEventListener("abort", abort); this.active.delete(key);
      if (this.writeQueues.get(brainId) === promise) this.writeQueues.delete(brainId);
    });
    this.active.set(key, { brainId, controller, promise });
    if (action === "write") this.writeQueues.set(brainId, promise);
    return promise;
  }
  private async perform(brainId: string, operation: ProcessMemoryNativeRequest["operation"], args: Record<string, unknown>,
    requestId: string, signal: AbortSignal): Promise<unknown> {
    const initial = await this.preferences(brainId), permission = await this.dependencies.currentPermission(brainId), generation = this.generation(brainId);
    signal.throwIfAborted();
    if (this.closing || !initial.enabled || permission.level === "off") throw new Error("DMA access is off. Both the user setting and current tool permission must allow this operation.");
    if (!initial.capabilities.implemented) throw new Error("This operating system has no process-memory adapter.");
    let lease: TargetLease | undefined;
    const request: ProcessMemoryNativeRequest = { format: PROCESS_MEMORY_FORMAT, operationId: requestId, operation,
      pid: 0, access: "read" };
    if (operation === "attach") {
      memoryKeys(args, ["pid", "access", "environment"]);
      if (args.environment !== undefined && args.environment !== "host") throw new Error("Process-memory access inside a sandbox/VM needs an actual supported adapter; host fallback is not permitted.");
      if (args.access !== "read" && args.access !== "read-write") throw new Error("Choose exact read or read-write process access.");
      request.pid = memoryWhole(args.pid, 0x7fffffff, 2); request.access = args.access;
    } else {
      memoryKeys(args, operation === "regions" ? ["targetId", "afterAddress", "limit"] : operation === "read"
        ? ["targetId", "address", "length"] : ["targetId", "address", "dataBase64", "expectedBeforeSha256"]);
      lease = this.targets.get(memoryId(args.targetId));
      if (!lease || lease.target.brainId !== brainId || lease.target.preferenceRevision !== initial.revision ||
          lease.permissionRevision !== permission.revision || lease.generation !== generation) throw new Error("The exact process-memory lease expired. Reattach under the current permission and host session.");
      request.pid = lease.target.identity.pid; request.access = lease.target.access; request.expectedIdentity = lease.target.identity;
      if (operation === "regions") {
        request.afterAddress = args.afterAddress === undefined ? "0x0" : memoryAddress(args.afterAddress);
        request.regionLimit = args.limit === undefined ? 50 : memoryWhole(args.limit, PROCESS_MEMORY_MAX_REGIONS);
      } else {
        request.address = memoryAddress(args.address);
        if (operation === "read") request.length = memoryWhole(args.length, PROCESS_MEMORY_MAX_BYTES);
        else {
          if (request.access !== "read-write") throw new Error("The target lease is read-only; no write was attempted.");
          const bytes = memoryBytes(args.dataBase64); request.dataBase64 = bytes.toString("base64"); request.length = bytes.length;
          request.expectedBeforeSha256 = memorySha(args.expectedBeforeSha256);
        }
        if (BigInt(request.address) + BigInt(request.length) > 1n << 64n) throw new Error("The process-memory address range overflows.");
      }
    }
    this.protected(request.pid);
    const requireCurrent = async (childPid?: number): Promise<void> => {
      signal.throwIfAborted(); this.protected(request.pid);
      await this.dependencies.assertBrainOwner(brainId);
      await this.dependencies.requireActiveOwner?.(brainId);
      const current = await this.settings(brainId), actual = await this.dependencies.currentPermission(brainId);
      signal.throwIfAborted();
      if (this.closing || this.generation(brainId) !== generation || !current.enabled || current.revision !== initial.revision ||
          actual.level !== permission.level || actual.level === "off" || actual.revision !== permission.revision ||
          (lease && this.targets.get(lease.target.id) !== lease)) throw new Error("Process-memory authority changed before/after the awaited operation.");
      await this.dependencies.checkResources(brainId, 32 * 1024 * 1024 + (request.length ?? 0) * 6, "bounded OS process-memory adapter", childPid ? [childPid] : []);
      signal.throwIfAborted();
      // Re-read after the asynchronous resource measurement too.
      const end = await this.settings(brainId), endPermission = await this.dependencies.currentPermission(brainId);
      await this.dependencies.requireActiveOwner?.(brainId);
      signal.throwIfAborted(); this.protected(request.pid);
      if (this.closing || this.generation(brainId) !== generation || !end.enabled || end.revision !== initial.revision ||
          endPermission.level !== permission.level || endPermission.revision !== permission.revision ||
          (lease && this.targets.get(lease.target.id) !== lease)) throw new Error("Process-memory authority changed during resource admission.");
    };
    await requireCurrent();
    const argsSha = processMemoryHash(processMemoryCanonical(request));
    if (operation === "write") {
      await this.dependencies.reserveWrite(65536);
      const store = await this.store();
      try {
        const previous = store.write(brainId, requestId);
        if (previous) {
          if (previous.argumentSha256 !== argsSha) throw new Error("A memory write request ID cannot be reused with different bytes/target.");
          if (previous.receipt) throw new ProcessMemoryOperationError("This exact memory write already has a durable result and was not repeated.", memoryReceipt(previous.receipt, request));
          throw new ProcessMemoryOperationError("This earlier memory write has no terminal receipt; it was not repeated.",
            uncertainMemoryReceipt(request, previous.phase === "dispatched", "An earlier attempt lost its final receipt. Inspect the exact target before considering a new operation."));
        }
        if (!store.prepareWrite(brainId, requestId, argsSha)) throw new Error("The memory-write request was concurrently claimed.");
      } finally { store.close(); }
    }
    let receipt: ProcessMemoryReceipt, commitAuthorized = false;
    try {
      receipt = await this.dependencies.run(request, { signal, beforeCommit: async (ready, childPid) => {
        await requireCurrent(childPid);
        if (ready.requestSha256 !== argsSha || ready.identity.pid !== request.pid ||
            (request.expectedIdentity && processMemoryCanonical(ready.identity) !== processMemoryCanonical(request.expectedIdentity))) throw new Error("The prepared native operation changed its exact target/input.");
        if (operation === "write") { const store = await this.store(); try { store.dispatched(brainId, requestId); } finally { store.close(); } }
        // Persist-before-dispatch can await filesystem work; check again before
        // the bridge sends its one-use commit nonce to the native helper.
        await requireCurrent(childPid);
        commitAuthorized = true;
      } });
    } catch (error) {
      receipt = error instanceof ProcessMemoryOperationError ? error.receipt : uncertainMemoryReceipt(request, commitAuthorized,
        error instanceof Error ? error.message : String(error));
      if (operation === "write") await this.retainWrite(brainId, requestId, receipt);
      throw new ProcessMemoryOperationError(receipt.error ?? "The process-memory operation did not finish.", receipt);
    }
    if (operation === "write") await this.retainWrite(brainId, requestId, receipt);
    try { await requireCurrent(); }
    catch (error) {
      // A post-effect revocation cannot erase the real write receipt. Reads
      // retain only a hash/count, not newly disclosed bytes, after revocation.
      const { dataBase64: _bytes, ...safe } = receipt;
      throw new ProcessMemoryOperationError(error instanceof Error ? error.message : String(error),
        { ...safe, ...(operation === "write" ? {} : { state: "cancelled" as const }), error: "Current authority changed; no memory bytes were delivered after revocation." });
    }
    if (receipt.state !== "complete") throw new ProcessMemoryOperationError(receipt.error ?? `Process-memory ${receipt.state}.`, receipt);
    if (operation === "attach") {
      if (!receipt.identity) throw new Error("Native attach returned no exact process identity.");
      const target: ProcessMemoryTarget = { id: randomUUID(), brainId, identity: receipt.identity, access: request.access,
        preferenceRevision: initial.revision, attachedAt: new Date().toISOString(), lease: "current-host-process-only" };
      this.targets.set(target.id, { target, permissionRevision: permission.revision, generation });
      return { target, receipt };
    }
    return receipt;
  }
}
