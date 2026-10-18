import { randomUUID } from "node:crypto";
import { chmod, lstat, mkdir, rm } from "node:fs/promises";
import { join } from "node:path";
import type {
  EnvironmentAdapterCapability,
  EnvironmentOperationReceipt,
  EnvironmentPage,
  EnvironmentProcessReceipt,
  EnvironmentToolResult,
  ManagedEnvironmentKind,
  ManagedEnvironmentRecord
} from "../shared/environments";
import {
  ENVIRONMENT_OPERATION_FORMAT,
  ENVIRONMENT_RECORD_FORMAT
} from "../shared/environments";
import {
  createLimaVm,
  environmentAdapterCapabilities,
  inspectLimaVm,
  inspectWindowsSandbox,
  loadLimaVm,
  prepareLimaVm,
  prepareSandboxEnvironment,
  runLimaVm,
  runSandboxEnvironment,
  runWindowsSandbox,
  startWindowsSandbox,
  stopLimaVm,
  stopWindowsSandbox,
  type EnvironmentAdapterDependencies,
  type EnvironmentProcessRunner,
  type ResolvedEnvironmentImage
} from "./environmentAdapters";
import {
  environmentCanonical,
  environmentHash,
  environmentId,
  environmentObject,
  environmentRequestId,
  environmentSha
} from "./environmentFormat";
import { EnvironmentStore } from "./environmentStore";
import { withEnvironmentMutation } from "./environmentWorkspaceLease";

export interface EnvironmentImageSource extends ResolvedEnvironmentImage {
  attachmentId: string;
}

export interface EnvironmentServiceDependencies {
  brainDirectory(brainId: string): Promise<string>;
  run: EnvironmentProcessRunner;
  platform?: NodeJS.Platform;
  arch?: string;
  findExecutable?(name: string): Promise<string | undefined>;
  windowsSandboxExecutable?: EnvironmentAdapterDependencies["windowsSandboxExecutable"];
  checkResources(brainId: string, bytes: number, purpose: string, childPids?: readonly number[]): Promise<void>;
  reserveWrite(root: string, bytes: number): Promise<void>;
  vmRuntimeDirectory?(brainId: string, environmentId: string): Promise<string>;
  deleteVmRuntime?(brainId: string): Promise<void>;
  resolveVmImage?(brainId: string, turnId: string, attachmentId: string, expectedSha256: string, architecture: "aarch64" | "x86_64", signal: AbortSignal): Promise<EnvironmentImageSource>;
  onError?(error: Error, brainId: string): void;
}

export class EnvironmentOperationError extends Error {
  constructor(message: string, readonly result: EnvironmentToolResult) {
    super(message); this.name = "EnvironmentOperationError";
  }
}

interface ActiveOperation {
  brainId: string;
  environmentId: string;
  requestId: string;
  action: EnvironmentOperationReceipt["action"];
  controller: AbortController;
  promise: Promise<EnvironmentToolResult>;
}

const terminalStates = new Set(["complete", "cancelled", "failed", "interrupted"]);
const exactKeys = (value: Record<string, unknown>, allowed: readonly string[]): void => {
  const unknown = Object.keys(value).filter(key => !allowed.includes(key));
  if (unknown.length) throw new Error(`Unsupported environment operands: ${unknown.join(", ")}.`);
};
const safeName = (value: unknown): string => {
  if (typeof value !== "string" || !value.trim() || value.length > 96 || /[\0-\x1f\x7f]/.test(value)) throw new Error("Environment name must be visible text from 1 to 96 characters.");
  return value.trim();
};
const timeout = (value: unknown, fallback: number): number => {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || Number(value) < 100 || Number(value) > 600_000) throw new Error("Environment timeout must be an integer from 100 to 600000 milliseconds.");
  return Number(value);
};
const whole = (value: unknown, label: string, minimum: number, maximum: number, fallback: number): number => {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || Number(value) < minimum || Number(value) > maximum) throw new Error(`${label} must be an integer from ${minimum} to ${maximum}.`);
  return Number(value);
};
const id = (): string => randomUUID().replaceAll("-", "");
const now = (): string => new Date().toISOString();
const MAX_PROCESS_TEXT_BYTES = 2 * 1024 * 1024;
const ENVIRONMENT_RECEIPT_RESERVE_BYTES = 64 * 1024 * 1024;
const WINDOWS_SANDBOX_MEMORY_BYTES = 2 * 1024 ** 3;
function boundedText(value: unknown, maximumBytes: number): { text: string; truncated: boolean } {
  const source = typeof value === "string" ? value : String(value ?? ""), bytes = Buffer.from(source, "utf8");
  if (bytes.byteLength <= maximumBytes) return { text: source, truncated: false };
  return { text: bytes.subarray(0, maximumBytes).toString("utf8"), truncated: true };
}
function processReceipt(value: EnvironmentProcessReceipt): EnvironmentProcessReceipt {
  if (!Number.isSafeInteger(value.exitCode)) throw new Error("Environment process returned an invalid exit code.");
  const stdout = boundedText(value.stdout, MAX_PROCESS_TEXT_BYTES), stderr = boundedText(value.stderr, MAX_PROCESS_TEXT_BYTES);
  return { exitCode: value.exitCode, stdout: stdout.text, stderr: stderr.text, truncated: Boolean(value.truncated || stdout.truncated || stderr.truncated),
    ...(value.guestOutput === "guest-output-unavailable" ? { guestOutput: value.guestOutput } : {}),
    ...(["already-absent", "stopped", "unconfirmed"].includes(String(value.sandboxDisposition)) ? { sandboxDisposition: value.sandboxDisposition as "already-absent" | "stopped" | "unconfirmed" } : {}) };
}
function errorText(value: unknown, maximum: number): string {
  const source = (value instanceof Error ? value.message : String(value)).replaceAll("\0", "�");
  return source.length <= maximum ? source : `${source.slice(0, Math.max(0, maximum - 1))}…`;
}

export class EnvironmentService {
  private readonly recovered = new Map<string, Promise<void>>();
  private readonly activeRequests = new Map<string, { argumentsSha256: string; promise: Promise<EnvironmentToolResult> }>();
  private readonly activeEnvironments = new Map<string, ActiveOperation>();
  private readonly activeControllers = new Map<string, { brainId: string; controller: AbortController }>();

  constructor(private readonly dependencies: EnvironmentServiceDependencies) {}

  private requestKey(brainId: string, requestId: string): string { return `${brainId}\0${requestId}`; }
  private environmentKey(brainId: string, environmentIdValue: string): string { return `${brainId}\0${environmentIdValue}`; }
  private async root(brainId: string): Promise<string> { return this.dependencies.brainDirectory(brainId); }
  private async store(brainId: string): Promise<EnvironmentStore> { return EnvironmentStore.open(await this.root(brainId), brainId); }
  private async ensureRecovered(brainId: string): Promise<void> {
    let pending = this.recovered.get(brainId);
    if (!pending) {
      pending = (async () => { const root = await this.root(brainId); await withEnvironmentMutation(root, async () => {
        const store = await EnvironmentStore.open(root, brainId); try { store.recoverInterrupted(); } finally { store.close(); }
      }); })();
      this.recovered.set(brainId, pending);
    }
    try { await pending; } catch (error) { this.recovered.delete(brainId); throw error; }
  }
  async forget(brainId: string): Promise<void> {
    await this.cancelBrain(brainId);
    this.recovered.delete(brainId);
  }
  async deleteInstallationRuntime(brainId: string): Promise<void> {
    const deleteVmRuntime = this.dependencies.deleteVmRuntime;
    if (!deleteVmRuntime) throw new Error("Private VM runtime deletion is unavailable; the instance was not deleted.");
    await this.cancelBrain(brainId); await this.ensureRecovered(brainId);
    const root = await this.root(brainId);
    await withEnvironmentMutation(root, async () => {
      const store = await EnvironmentStore.open(root, brainId);
      try {
        for (const environment of [...store.environments()]) {
          if (environment.adapter === "windows-sandbox-cli" && environment.externalId) {
            const signal = AbortSignal.timeout(120_000), directory = store.instanceDirectory(environment.id);
            const state = await inspectWindowsSandbox(this.adapterDependencies(), environment, directory, signal).catch(() => "unknown" as const);
            let disposition: "already-absent" | "stopped" = state === "stopped" || state === "missing" ? "already-absent" : "stopped";
            if (state !== "stopped" && state !== "missing") {
              const result = processReceipt(await stopWindowsSandbox(this.adapterDependencies(), environment, directory, signal));
              if (result.exitCode !== 0) throw new Error(`The owned Windows Sandbox could not be stopped before instance deletion (exit ${result.exitCode}).`);
              disposition = result.sandboxDisposition === "already-absent" ? "already-absent" : "stopped";
            }
            store.putEnvironment({ ...environment, externalId: undefined, orphanedProcessRisk: undefined, state: "stopped", revision: environment.revision + 1, updatedAt: now(),
              warning: disposition === "already-absent"
                ? "The exact owned Windows Sandbox UUID was already absent during confirmed instance deletion; no other session was stopped."
                : "The owned disposable Windows Sandbox was stopped for confirmed instance deletion; its guest state was deleted." });
            continue;
          }
          if (environment.kind !== "vm" || !environment.externalId) continue;
          const runtime = await this.vmRuntime(brainId, environment.id), signal = AbortSignal.timeout(120_000);
          const state = await inspectLimaVm(this.adapterDependencies(), environment, store.instanceDirectory(environment.id), runtime, signal).catch(() => "unknown" as const);
          if (state !== "stopped" && state !== "missing") {
            const result = processReceipt(await stopLimaVm(this.adapterDependencies(), environment, store.instanceDirectory(environment.id), runtime, signal));
            if (result.exitCode !== 0) throw new Error(`The managed VM could not be stopped before instance deletion (exit ${result.exitCode}).`);
          }
          store.putEnvironment({ ...environment, state: "stopped", revision: environment.revision + 1, updatedAt: now(),
            warning: "The managed VM was stopped for confirmed instance deletion; private runtime removal is pending." });
        }
      } finally { store.close(); }
      await deleteVmRuntime(brainId);
    });
    this.recovered.delete(brainId);
  }
  async shutdown(): Promise<void> {
    for (const operation of this.activeControllers.values()) operation.controller.abort(new Error("Studio is closing; the environment operation is retained as interrupted/cancelled."));
    await Promise.allSettled([...this.activeRequests.values()].map(item => item.promise));
  }
  activeCount(brainId: string): number {
    return [...this.activeEnvironments.values()].filter(item => item.brainId === brainId).length;
  }
  private async cancelBrain(brainId: string): Promise<void> {
    const controllers = [...this.activeControllers.values()].filter(item => item.brainId === brainId);
    for (const item of controllers) item.controller.abort(new Error("Environment ownership changed; the active operation was cancelled."));
    await Promise.allSettled([...this.activeRequests.entries()].filter(([key]) => key.startsWith(`${brainId}\0`)).map(([, item]) => item.promise));
  }

  private adapterDependencies(onSpawn?: (pid: number) => void): EnvironmentAdapterDependencies {
    return { platform: this.dependencies.platform, arch: this.dependencies.arch, run: this.dependencies.run,
      findExecutable: this.dependencies.findExecutable, windowsSandboxExecutable: this.dependencies.windowsSandboxExecutable, onSpawn };
  }
  async capabilities(brainId: string): Promise<{ adapters: EnvironmentAdapterCapability[]; installationPerformed: false }> {
    await this.ensureRecovered(brainId); await this.dependencies.checkResources(brainId, 65536, "environment runtime inspection");
    return { adapters: await environmentAdapterCapabilities(this.adapterDependencies(), Boolean(this.dependencies.resolveVmImage && this.dependencies.vmRuntimeDirectory)), installationPerformed: false };
  }

  private async vmRuntime(brainId: string, environmentIdValue: string): Promise<string> {
    if (!this.dependencies.vmRuntimeDirectory) throw new Error("A private installation VM runtime directory is unavailable.");
    const path = await this.dependencies.vmRuntimeDirectory(brainId, environmentIdValue); await mkdir(path, { recursive: true, mode: 0o700 });
    let info = await lstat(path); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("VM runtime directory is not a private real directory.");
    if (process.platform !== "win32") {
      const uid = typeof process.getuid === "function" ? process.getuid() : info.uid;
      if (info.uid !== uid) throw new Error("VM runtime directory has another operating-system owner.");
      if ((info.mode & 0o077) !== 0) { await chmod(path, 0o700); info = await lstat(path); }
      if (info.uid !== uid || (info.mode & 0o077) !== 0) throw new Error("VM runtime directory is not private to the current operating-system owner.");
    }
    return path;
  }

  async execute(
    brainId: string,
    toolId: "runtime.sandbox" | "runtime.vm",
    action: string,
    argsValue: Record<string, unknown>,
    requestIdValue: string,
    signal: AbortSignal,
    context?: { chatTurnId?: string }
  ): Promise<unknown> {
    const args = structuredClone(environmentObject(argsValue, "environment arguments")),
      ownedContext = context?.chatTurnId === undefined ? undefined : { chatTurnId: String(context.chatTurnId) },
      kind: ManagedEnvironmentKind = toolId === "runtime.vm" ? "vm" : "sandbox";
    await this.ensureRecovered(brainId); signal.throwIfAborted();
    if (action === "capabilities") { exactKeys(args, []); return this.capabilities(brainId); }
    if (action === "list") { exactKeys(args, ["after", "limit"]); const store = await this.store(brainId);
      try { return store.page(args.after === undefined ? "" : environmentId(args.after), whole(args.limit, "Environment page size", 1, 200, 50)); }
      finally { store.close(); } }
    if (action === "status") { exactKeys(args, ["environmentId"]); return this.status(brainId, kind, environmentId(args.environmentId), signal); }
    if (!["create", "load", "run", "stop"].includes(action)) throw new Error("Unknown environment lifecycle action.");
    if (action === "load" || action === "stop") exactKeys(args, ["environmentId"]);
    else if (action === "run") exactKeys(args, ["environmentId", "program", "arguments", "timeoutMs"]);
    const requestId = environmentRequestId(requestIdValue), argsSha = environmentHash(environmentCanonical({ toolId, action, args }));
    const requestKey = this.requestKey(brainId, requestId), active = this.activeRequests.get(requestKey);
    if (active) {
      if (active.argumentsSha256 !== argsSha) throw new Error("Environment request ID is already active with different operands.");
      return active.promise;
    }
    const promise = this.mutate(brainId, kind, action as EnvironmentOperationReceipt["action"], args, requestId, argsSha, signal, ownedContext)
      .finally(() => { if (this.activeRequests.get(requestKey)?.promise === promise) this.activeRequests.delete(requestKey); });
    this.activeRequests.set(requestKey, { argumentsSha256: argsSha, promise }); return promise;
  }

  private async status(brainId: string, kind: ManagedEnvironmentKind, environmentIdValue: string, signal: AbortSignal): Promise<{ environment: ManagedEnvironmentRecord; operations: EnvironmentOperationReceipt[] }> {
    const root = await this.root(brainId);
    return withEnvironmentMutation(root, () => this.statusOwned(brainId, kind, environmentIdValue, signal));
  }
  private async statusOwned(brainId: string, kind: ManagedEnvironmentKind, environmentIdValue: string, signal: AbortSignal): Promise<{ environment: ManagedEnvironmentRecord; operations: EnvironmentOperationReceipt[] }> {
    const store = await this.store(brainId);
    try {
      let environment = store.getEnvironment(environmentIdValue); if (!environment || environment.kind !== kind) throw new Error("Managed environment was not found for this tool.");
      if (environment.kind === "vm" && ["loaded", "stopped", "review-required", "error"].includes(environment.state)) {
        let actual: "loaded" | "stopped" | "missing" | "unknown" = "unknown";
        try { actual = await inspectLimaVm(this.adapterDependencies(), environment, store.instanceDirectory(environment.id),
          await this.vmRuntime(brainId, environment.id), signal); } catch { signal.throwIfAborted(); }
        const current = store.getEnvironment(environment.id);
        if (!current) throw new Error("Managed VM disappeared during status inspection.");
        if (current.revision === environment.revision && current.lastOperationId === environment.lastOperationId) {
          if (actual === "loaded" || actual === "stopped") {
            environment = { ...current, state: actual, revision: current.revision + 1, updatedAt: now(), warning: undefined };
            store.putEnvironment(environment);
          } else if (actual === "missing" && current.state !== "review-required") {
            environment = { ...current, state: "review-required", revision: current.revision + 1, updatedAt: now(), warning: "The recorded VM is absent from the current Lima installation. No replacement was created." };
            store.putEnvironment(environment);
          } else environment = current;
        } else {
          environment = current;
        }
      } else if (environment.adapter === "windows-sandbox-cli" && environment.externalId &&
          ["loaded", "stopped", "review-required", "error"].includes(environment.state)) {
        let actual: "loaded" | "stopped" | "missing" | "unknown" = "unknown";
        try { actual = await inspectWindowsSandbox(this.adapterDependencies(), environment, store.instanceDirectory(environment.id), signal); }
        catch { signal.throwIfAborted(); }
        const current = store.getEnvironment(environment.id);
        if (!current) throw new Error("Managed Windows Sandbox disappeared during status inspection.");
        if (current.revision === environment.revision && current.lastOperationId === environment.lastOperationId) {
          if (actual === "loaded" && current.orphanedProcessRisk) {
            environment = { ...current, state: "review-required", revision: current.revision + 1, updatedAt: now(),
              warning: "A guest command may have survived the interrupted host. Stop this exact disposable Windows Sandbox before another Load or Run." };
            store.putEnvironment(environment);
          } else if (actual === "loaded") {
            environment = { ...current, state: "loaded", revision: current.revision + 1, updatedAt: now(), warning: undefined };
            store.putEnvironment(environment);
          } else if (actual === "stopped" || actual === "missing") {
            environment = { ...current, externalId: undefined, orphanedProcessRisk: undefined, state: "stopped", revision: current.revision + 1, updatedAt: now(),
              warning: "The disposable Windows Sandbox is not running; its prior guest state is unavailable." };
            store.putEnvironment(environment);
          } else environment = current;
        } else environment = current;
      }
      return { environment, operations: store.operations(environment.id) };
    } finally { store.close(); }
  }

  private async mutate(
    brainId: string,
    kind: ManagedEnvironmentKind,
    action: EnvironmentOperationReceipt["action"],
    args: Record<string, unknown>,
    requestId: string,
    argumentsSha256: string,
    outerSignal: AbortSignal,
    context?: { chatTurnId?: string }
  ): Promise<EnvironmentToolResult> {
    const root = await this.root(brainId);
    return withEnvironmentMutation(root, () => this.mutateOwned(brainId, kind, action, args, requestId, argumentsSha256, outerSignal, context));
  }
  private async mutateOwned(
    brainId: string,
    kind: ManagedEnvironmentKind,
    action: EnvironmentOperationReceipt["action"],
    args: Record<string, unknown>,
    requestId: string,
    argumentsSha256: string,
    outerSignal: AbortSignal,
    context?: { chatTurnId?: string }
  ): Promise<EnvironmentToolResult> {
    let store = await this.store(brainId);
    const prior = store.request(requestId, argumentsSha256);
    if (prior?.result) {
      store.close();
      if (prior.result.operation.state !== "complete") throw new EnvironmentOperationError(prior.result.operation.error ?? "Retained environment operation did not complete.", prior.result);
      return prior.result;
    }
    if (prior) {
      const operation = store.getOperation(prior.operationId), environment = operation && store.getEnvironment(operation.environmentId);
      if (!operation || !environment) { store.close(); throw new Error("Environment request lost its retained operation."); }
      if (!terminalStates.has(operation.state)) { store.close(); throw new Error("The retained environment operation has no terminal acknowledgement; inspect it before a new attempt."); }
      const result = { environment, operation }; store.finishRequest(requestId, result); store.close();
      if (operation.state !== "complete") throw new EnvironmentOperationError(operation.error ?? "Retained environment operation did not complete.", result);
      return result;
    }
    const environmentIdValue = action === "create" ? id() : environmentId(args.environmentId), operationId = id(), createdAt = now();
    let environment = action === "create"
      ? this.newEnvironment(brainId, kind, environmentIdValue, args, operationId, createdAt)
      : store.getEnvironment(environmentIdValue);
    if (!environment || environment.kind !== kind) { store.close(); throw new Error("Managed environment was not found for this tool."); }
    const key = this.environmentKey(brainId, environment.id), priorOwner = this.activeEnvironments.get(key);
    if (priorOwner && (action !== "stop" || priorOwner.action === "stop")) { store.close(); throw new Error("Another operation already owns this environment."); }
    const noEffectError = action === "load"
      ? environment.orphanedProcessRisk ? "This sandbox has an unreconciled crashed process and cannot be loaded."
        : environment.state === "dormant" && environment.kind === "vm" ? "A copied/imported VM must be recreated from a newly reviewed local image."
          : !["ready", "stopped", "dormant"].includes(environment.state) ? "This environment is not at a loadable boundary." : undefined
      : action === "run" && environment.state !== "loaded" ? "Load this environment before using it."
        : action === "stop" && environment.kind === "sandbox" && environment.adapter !== "windows-sandbox-cli" && environment.orphanedProcessRisk && !priorOwner ? "A crashed sandbox has no durable process identity that can be safely stopped."
          : action === "stop" && !priorOwner && ["ready", "stopped", "dormant"].includes(environment.state) ? "This environment has no active runtime to stop." : undefined;
    if (noEffectError) { store.close(); throw new Error(noEffectError); }
    const operation: EnvironmentOperationReceipt = { format: ENVIRONMENT_OPERATION_FORMAT, version: 1, id: operationId, brainId,
      environmentId: environment.id, requestId, action, state: "intent", argumentsSha256, dispatchStarted: false, createdAt, updatedAt: createdAt };
    store.transaction(() => { store.putEnvironment(environment!); store.putOperation(operation); store.beginRequest(requestId, argumentsSha256, operationId); });
    store.close();
    const controller = new AbortController(), forward = () => controller.abort(outerSignal.reason);
    const requestKey = this.requestKey(brainId, requestId); this.activeControllers.set(requestKey, { brainId, controller });
    outerSignal.addEventListener("abort", forward, { once: true }); if (outerSignal.aborted) forward();
    const running = this.perform(brainId, environment, operation, args, controller, context, priorOwner);
    const active: ActiveOperation = { brainId, environmentId: environment.id, requestId, action, controller, promise: running };
    this.activeEnvironments.set(key, active);
    try { return await running; }
    finally {
      outerSignal.removeEventListener("abort", forward);
      if (this.activeControllers.get(requestKey)?.controller === controller) this.activeControllers.delete(requestKey);
      if (this.activeEnvironments.get(key) === active) this.activeEnvironments.delete(key);
    }
  }

  private newEnvironment(brainId: string, kind: ManagedEnvironmentKind, environmentIdValue: string, args: Record<string, unknown>, operationId: string, createdAt: string): ManagedEnvironmentRecord {
    if (kind === "sandbox") {
      exactKeys(args, ["name", "adapter"]); const platform = this.dependencies.platform ?? process.platform;
      const expected = platform === "darwin" ? "darwin-sandbox-exec" : platform === "linux" ? "linux-bwrap"
        : platform === "win32" ? "windows-sandbox-cli" : undefined;
      const adapter = args.adapter ?? expected;
      if (!expected || adapter !== expected) throw new Error("No supported sandbox adapter is available for this operating system; host execution was not substituted.");
      return { format: ENVIRONMENT_RECORD_FORMAT, version: 1, id: environmentIdValue, brainId, kind, name: safeName(args.name), adapter: expected,
        state: "creating", revision: 1, network: "none", createdAt, updatedAt: createdAt, lastOperationId: operationId,
        ...(expected === "darwin-sandbox-exec" ? { warning: "Apple marks sandbox-exec deprecated; this profile is not App Sandbox certification." }
          : expected === "windows-sandbox-cli" ? { warning: "Windows Sandbox is disposable and single-instance. Load starts a clean guest; Stop deletes its guest state, and guest stdout/stderr are unavailable." } : {}) };
    }
    exactKeys(args, ["name", "adapter", "imageAttachmentId", "imageSha256", "architecture", "cpus", "memoryGiB", "diskGiB", "network"]);
    if ((args.adapter ?? "lima") !== "lima") throw new Error("Only the explicit Lima VM adapter is supported.");
    if (!this.dependencies.resolveVmImage || !this.dependencies.vmRuntimeDirectory) throw new Error("VM creation awaits a configured hash-verified local-image resolver and private runtime store; no download or template was used.");
    const imageSha256 = environmentSha(args.imageSha256, "VM image checksum"), architecture = args.architecture;
    if (!['aarch64', 'x86_64'].includes(String(architecture)) || args.network !== "guest-outbound") throw new Error("VM creation requires an exact guest architecture and explicit guest-outbound network review; network-disabled Lima is not claimed.");
    if (typeof args.imageAttachmentId !== "string") throw new Error("VM creation requires one current-task image attachment ID.");
    const cpus = whole(args.cpus, "VM CPUs", 1, 64, 2), memoryGiB = whole(args.memoryGiB, "VM memory", 1, 512, 2), diskGiB = whole(args.diskGiB, "VM disk", 4, 2048, 20);
    return { format: ENVIRONMENT_RECORD_FORMAT, version: 1, id: environmentIdValue, brainId, kind, name: safeName(args.name), adapter: "lima",
      state: "creating", revision: 1, network: "guest-outbound", createdAt, updatedAt: createdAt, lastOperationId: operationId,
      externalId: `omni-${brainId.replaceAll("-", "").slice(0, 12)}-${environmentIdValue.slice(0, 12)}`,
      imageSha256, guestArchitecture: architecture as "aarch64" | "x86_64",
      resourcePlan: { cpus, memoryBytes: memoryGiB * 1024 ** 3, diskBytes: diskGiB * 1024 ** 3 } };
  }

  private async perform(
    brainId: string,
    initialEnvironment: ManagedEnvironmentRecord,
    initialOperation: EnvironmentOperationReceipt,
    args: Record<string, unknown>,
    controller: AbortController,
    context?: { chatTurnId?: string },
    priorOwner?: ActiveOperation
  ): Promise<EnvironmentToolResult> {
    const root = await this.root(brainId), directory = join(root, "environments", "instances", initialEnvironment.id);
    const pids: number[] = [], adapterDeps = this.adapterDependencies(pid => { pids.push(pid); void this.dependencies.checkResources(brainId, 65536, "environment child process", pids).catch(error => controller.abort(error)); });
    const monitor = setInterval(() => { void this.dependencies.checkResources(brainId, 65536, "environment operation", pids).catch(error => controller.abort(error)); }, 250); monitor.unref();
    let environment = initialEnvironment, operation = initialOperation;
    const commit = async (): Promise<void> => { const store = await this.store(brainId); try { store.transaction(() => { store.putEnvironment(environment); store.putOperation(operation); }); } finally { store.close(); } };
    try {
      await this.dependencies.checkResources(brainId, Buffer.byteLength(environmentCanonical(args)) * 12 + 65536, "environment arguments", pids);
      // stdout/stderr can require JSON escaping and are retained both in the
      // operation ledger and its lost-ACK request result. Reserve the bounded
      // worst-case durable receipt before any external effect starts.
      await this.dependencies.reserveWrite(root, ENVIRONMENT_RECEIPT_RESERVE_BYTES); controller.signal.throwIfAborted();
      if (operation.action === "stop") {
        if (environment.kind === "sandbox" && environment.adapter !== "windows-sandbox-cli" && environment.orphanedProcessRisk && !priorOwner) throw new Error("A crashed sandbox has no durable process identity that can be safely killed or resumed. Retain its files for inspection and create a new sandbox.");
        environment = { ...environment,
          ...(environment.adapter === "windows-sandbox-cli" && priorOwner?.action === "run" ? { orphanedProcessRisk: true as const } : {}),
          state: "stopping", revision: environment.revision + 1, updatedAt: now(), lastOperationId: operation.id };
        operation = { ...operation, state: "running", dispatchStarted: true, updatedAt: now() }; await commit();
        if (priorOwner && priorOwner.requestId !== operation.requestId) { priorOwner.controller.abort(new Error("The environment was explicitly stopped.")); await Promise.allSettled([priorOwner.promise]); }
        let process;
        if (environment.kind === "vm") {
          process = processReceipt(await stopLimaVm(adapterDeps, environment, directory, await this.vmRuntime(brainId, environment.id), controller.signal));
          if (process.exitCode !== 0) { operation = { ...operation, process }; throw new Error(`Lima stop failed with exit code ${process.exitCode}: ${process.stderr || process.stdout}`); }
        } else if (environment.adapter === "windows-sandbox-cli" && environment.externalId) {
          process = processReceipt(await stopWindowsSandbox(adapterDeps, environment, directory, controller.signal));
          if (process.exitCode !== 0) { operation = { ...operation, process }; throw new Error(`Windows Sandbox stop failed with exit code ${process.exitCode}: ${process.stderr || process.stdout}`); }
        }
        environment = { ...environment, ...(environment.adapter === "windows-sandbox-cli" ? { externalId: undefined, orphanedProcessRisk: undefined,
          warning: process?.sandboxDisposition === "already-absent"
            ? "The exact owned Windows Sandbox UUID was already absent; Studio did not stop or adopt another session."
            : "The disposable Windows Sandbox was stopped; its guest state was deleted." } : {}),
          state: "stopped", revision: environment.revision + 1, updatedAt: now(), lastOperationId: operation.id };
        operation = { ...operation, state: "complete", updatedAt: now(), ...(process ? { process } : {}) };
      } else if (operation.action === "create") {
        await mkdir(join(directory, "workspace", "tmp"), { recursive: true, mode: 0o700 });
        const base = await lstat(directory); if (!base.isDirectory() || base.isSymbolicLink()) throw new Error("Environment instance storage is unsafe.");
        operation = { ...operation, state: "running", dispatchStarted: true, updatedAt: now() }; await commit();
        if (environment.kind === "sandbox") {
          const capability = (await environmentAdapterCapabilities(adapterDeps)).find(item => item.id === environment.adapter);
          if (!capability?.available) throw new Error(capability?.warning ?? "Sandbox adapter is unavailable.");
          await prepareSandboxEnvironment(environment, directory); controller.signal.throwIfAborted();
          environment = { ...environment, state: "ready", revision: environment.revision + 1, updatedAt: now() };
          operation = { ...operation, state: "complete", updatedAt: now() };
        } else {
          const turnId = context?.chatTurnId; if (!turnId) throw new Error("VM images are available only from the current owning chat turn.");
          if (!environment.resourcePlan) throw new Error("VM resource plan is missing.");
          const image = await this.dependencies.resolveVmImage!(brainId, turnId, String(args.imageAttachmentId), environment.imageSha256!, environment.guestArchitecture!, controller.signal);
          if (image.attachmentId !== args.imageAttachmentId || image.sha256 !== environment.imageSha256 || image.architecture !== environment.guestArchitecture) throw new Error("Resolved VM image identity differs from the reviewed request.");
          await this.dependencies.checkResources(brainId, environment.resourcePlan.memoryBytes, "VM reviewed memory", pids);
          const limaHome = await this.vmRuntime(brainId, environment.id); await this.dependencies.reserveWrite(limaHome, environment.resourcePlan.diskBytes);
          const resources = { cpus: environment.resourcePlan.cpus, memoryGiB: environment.resourcePlan.memoryBytes / 1024 ** 3, diskGiB: environment.resourcePlan.diskBytes / 1024 ** 3 };
          const config = await prepareLimaVm(limaHome, environment, image, resources);
          try {
            const process = processReceipt(await createLimaVm(adapterDeps, environment, directory, limaHome, controller.signal));
            if (process.exitCode !== 0) { operation = { ...operation, process }; throw new Error(`Lima create failed with exit code ${process.exitCode}: ${process.stderr || process.stdout}`); }
            environment = { ...environment, state: "ready", revision: environment.revision + 1, updatedAt: now() };
            operation = { ...operation, state: "complete", updatedAt: now(), process };
          } finally { await rm(config, { force: true }).catch(() => undefined); }
        }
      } else {
        if (environment.state === "review-required" || environment.orphanedProcessRisk) throw new Error(environment.kind === "sandbox"
          ? environment.adapter === "windows-sandbox-cli" ? "Reconcile this exact Windows Sandbox identity with Status before another use."
            : "This sandbox was interrupted without a durable process identity. Retain its files for inspection and create a new sandbox; it cannot be safely resumed."
          : "Inspect the actual VM adapter state before a new use.");
        if (operation.action === "load") {
          if (!["ready", "stopped", "dormant"].includes(environment.state)) throw new Error("This environment is not at a loadable boundary.");
          if (environment.state === "dormant" && environment.kind === "vm") throw new Error("A copied/imported VM is retained as dormant evidence. Recreate it from a newly reviewed local image; the source host VM and grants were not transferred.");
          environment = { ...environment, ...(environment.adapter === "windows-sandbox-cli" && !environment.externalId ? { externalId: randomUUID() } : {}),
            state: "loading", revision: environment.revision + 1, updatedAt: now(), lastOperationId: operation.id };
          operation = { ...operation, state: "running", dispatchStarted: environment.kind === "vm" || environment.adapter === "windows-sandbox-cli", updatedAt: now() }; await commit();
          let process; if (environment.kind === "vm") { if (!environment.resourcePlan) throw new Error("VM resource plan is missing.");
            await this.dependencies.checkResources(brainId, environment.resourcePlan.memoryBytes, "VM reviewed memory", pids);
            process = processReceipt(await loadLimaVm(adapterDeps, environment, directory, await this.vmRuntime(brainId, environment.id), controller.signal)); if (process.exitCode !== 0) { operation = { ...operation, process }; throw new Error(`Lima start failed with exit code ${process.exitCode}: ${process.stderr || process.stdout}`); } }
          else { const capability = (await environmentAdapterCapabilities(adapterDeps)).find(item => item.id === environment.adapter); if (!capability?.available) throw new Error(capability?.warning ?? "Sandbox adapter is unavailable.");
            await prepareSandboxEnvironment(environment, directory);
            if (environment.adapter === "windows-sandbox-cli") {
              await this.dependencies.checkResources(brainId, WINDOWS_SANDBOX_MEMORY_BYTES, "Windows Sandbox minimum guest memory", pids);
              process = processReceipt(await startWindowsSandbox(adapterDeps, environment, directory, controller.signal));
              if (process.exitCode !== 0) { operation = { ...operation, process }; throw new Error(`Windows Sandbox start failed with exit code ${process.exitCode}: ${process.stderr || process.stdout}`); }
            } }
          environment = { ...environment, state: "loaded", revision: environment.revision + 1, updatedAt: now(), warning: environment.warning };
          operation = { ...operation, state: "complete", dispatchStarted: true, updatedAt: now(), ...(process ? { process } : {}) };
        } else {
          exactKeys(args, ["environmentId", "program", "arguments", "timeoutMs"]);
          if (environment.state !== "loaded") throw new Error("Load this environment before using it.");
          const milliseconds = timeout(args.timeoutMs, 60_000);
          environment = { ...environment, state: "busy", revision: environment.revision + 1, updatedAt: now(), lastOperationId: operation.id };
          operation = { ...operation, state: "running", dispatchStarted: true, updatedAt: now() }; await commit();
          const process = processReceipt(environment.kind === "sandbox"
            ? environment.adapter === "windows-sandbox-cli"
              ? await runWindowsSandbox(adapterDeps, environment, directory, args.program, args.arguments, milliseconds, controller.signal)
              : await runSandboxEnvironment(adapterDeps, environment, directory, args.program, args.arguments, milliseconds, controller.signal)
            : await runLimaVm(adapterDeps, environment, directory, await this.vmRuntime(brainId, environment.id), args.program, args.arguments, milliseconds, controller.signal));
          environment = { ...environment, state: "loaded", revision: environment.revision + 1, updatedAt: now() };
          operation = { ...operation, state: process.exitCode === 0 ? "complete" : "failed", updatedAt: now(), process,
            ...(process.exitCode === 0 ? {} : { error: `Environment program exited with code ${process.exitCode}.` }) };
        }
      }
    } catch (error) {
      const message = errorText(error, 8000), cancelled = controller.signal.aborted;
      const explicitStop = cancelled && controller.signal.reason instanceof Error && controller.signal.reason.message === "The environment was explicitly stopped.";
      let windowsSandboxStopped = false, windowsSandboxAlreadyAbsent = false, windowsCleanupError: string | undefined;
      if (environment.adapter === "windows-sandbox-cli" && environment.externalId && operation.action !== "stop" && !explicitStop) {
        try {
          const cleanup = processReceipt(await stopWindowsSandbox(this.adapterDependencies(), environment, directory, AbortSignal.timeout(120_000)));
          windowsSandboxStopped = cleanup.exitCode === 0;
          windowsSandboxAlreadyAbsent = cleanup.sandboxDisposition === "already-absent";
          if (!windowsSandboxStopped) windowsCleanupError = `Windows Sandbox cleanup exited ${cleanup.exitCode}: ${cleanup.stderr || cleanup.stdout}`;
        } catch (cleanupError) { windowsCleanupError = errorText(cleanupError, 1000); }
      }
      const stickySandboxRisk = environment.kind === "sandbox" && environment.orphanedProcessRisk;
      const cancelledSandboxState = operation.action === "create" && !operation.dispatchStarted ? "error" : "stopped";
      const windowsState = windowsSandboxStopped ? "stopped" : "review-required";
      environment = { ...environment, ...(windowsSandboxStopped ? { externalId: undefined, orphanedProcessRisk: undefined }
        : environment.adapter === "windows-sandbox-cli" && operation.action === "run" && environment.externalId ? { orphanedProcessRisk: true as const } : {}),
        state: environment.adapter === "windows-sandbox-cli" && environment.externalId ? windowsState
          : stickySandboxRisk ? "review-required" : cancelled ? (environment.kind === "sandbox" ? cancelledSandboxState : "review-required")
        : operation.action === "stop" && environment.kind === "vm" ? "review-required" : "error",
        revision: environment.revision + 1, updatedAt: now(), lastOperationId: operation.id,
        warning: errorText(windowsSandboxStopped
          ? windowsSandboxAlreadyAbsent ? `${message} The fresh owned UUID was confirmed absent; no Windows Sandbox was stopped or adopted.`
            : `${message} The exact owned disposable Windows Sandbox was stopped and its guest state was deleted.`
          : windowsCleanupError ? `${message} Windows Sandbox cleanup could not be confirmed: ${windowsCleanupError}` : message, 2000) };
      operation = { ...operation, state: cancelled ? "cancelled" : "failed", updatedAt: now(), error: message };
    } finally { clearInterval(monitor); }
    await commit(); const result = { environment, operation }; const store = await this.store(brainId);
    try { store.finishRequest(operation.requestId, result); } finally { store.close(); }
    if (operation.state === "failed" || operation.state === "cancelled") throw new EnvironmentOperationError(operation.error ?? "Environment operation did not complete.", result);
    return result;
  }
}
