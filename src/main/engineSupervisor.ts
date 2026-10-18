import { EventEmitter } from "node:events";
import { existsSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { randomUUID } from "node:crypto";
import { AsyncLocalStorage } from "node:async_hooks";
import { waitHostComputePredecessors } from "./hostComputeWait";
import type { EngineHealth } from "../shared/types";
import { VideoRuntimeUnavailableError, type PreparedVideoRuntime, type VideoRuntimeProgress } from "./videoRuntimeProvisioner";
import { CodecRuntimeSetupBridge, codecRuntimeChallenge, sameCodecOwner, type CodecRuntimeOwner } from "./codecRuntimeBridge";
import { observationReceipt, type ChatToolObservation, type ChatToolObservationReceipt } from "./chatToolObservation";
import type { NativeInheritedPresentation, NativeInstanceStatus } from "../shared/instanceLifecycleApi";
import type { InstanceLifecycleState } from "../shared/instanceLifecycle";
import type { ApprovedNativeMcpSamplingInput, NativeMcpSamplingResult } from "../shared/nativeMcpSampling";
import { performNativeMcpSampling } from "./nativeMcpSampling";
import type { NativeAppObservationFrontier, NativeAppObservationInput, NativeAppObservationResult } from "../shared/nativeAppObservation";
import { checkedAppObservationFrontier, checkedAppObservationResult } from "./mcpAppObservationReceipt";
import type { McpAppObservationArtifact } from "../shared/mcpAppArtifacts";
import type { RouterSpikeQuery } from "../shared/routerActivity";

const PROTOCOL_VERSION = 1;
const MAX_PROTOCOL_LINE = 32 * 1024 * 1024;
const COOPERATIVE_CANCEL_GRACE_MS = 180_000;
// Native safetensors loading can hold the GIL and delay Python's signal
// handler. Before a chat has produced a correlated token or completion phase,
// its fast experience is still uncommitted; Stop gets a short fallback deadline.
const PRE_OUTPUT_CANCEL_GRACE_MS = 5_000;
const COOPERATIVE_CANCEL_METHODS = new Set([
  "hardware_topology",
  "hardware_projection_profile",
  "load",
  "chat",
  "resume_chat",
  "resume_inherited_chat",
  "consolidate_chat_learning",
  "learn_action_result",
  "learn_tool_route_outcome",
  "app_observation_frontier",
  "admit_app_observation",
  "configure_video_runtime",
  "generate_neural_speech",
  "generate_modality"
]);
/** Explicit opt-out used only by durable, cancellable multi-day jobs. */
export const ENGINE_REQUEST_NO_DEADLINE = 0 as const;
const HOST_COMPUTE_SCOPE = new AsyncLocalStorage<object>();
const NATIVE_START_SCOPE = new AsyncLocalStorage<object>();

interface PendingRequest {
  child: ChildProcessWithoutNullStreams;
  chatStreamId?: string;
  chatOutputOrCommitObserved?: boolean;
  resolve(value: unknown): void;
  reject(error: Error): void;
  timeout?: NodeJS.Timeout;
  cancelRequested?: boolean;
  codecOwner?: CodecRuntimeOwner;
  codecSignal?: AbortSignal;
  codecChallenges?: Set<string>;
  cleanup(): void;
}

interface RequestQueueEntry {
  ready: Promise<void>;
  release(): void;
}

export type EngineRequestPriority = "foreground" | "background";

export type EngineActivityOwner =
  | "build"
  | "chat"
  | "evolution"
  | "training"
  | "ingestion"
  | "modality"
  | "inspection"
  | "idle"
  | "system";

export type EngineActivityState =
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "complete"
  | "failed";

export interface EngineActivityReference {
  requestId: string;
  owner: EngineActivityOwner;
  label: string;
  method: string;
  brainId?: string;
  jobId?: string;
  turnId?: string;
}

export interface EngineActivityTransition extends EngineActivityReference {
  state: EngineActivityState;
  queuePosition?: number;
  queuedBehind?: EngineActivityReference;
  cancellationPhase?: "queued" | "running";
  workerTerminationAcknowledged?: boolean;
}

/**
 * Main-process-only ownership metadata. It is never sent to, or interpreted by,
 * the neural worker. Stable request ids let independent UI operations observe
 * and cancel exactly their own reservation in the serial runtime.
 */
export interface EngineRequestContext {
  requestId: string;
  owner: EngineActivityOwner;
  label: string;
  brainId?: string;
  jobId?: string;
  turnId?: string;
  onTransition?(transition: EngineActivityTransition): void;
  /** Main-only safe-boundary admission check; never model-facing input. */
  beforeDispatch?(): void | Promise<void>;
}

export interface EngineCancellationAcknowledgement {
  requestId: string;
  acknowledged: boolean;
  phase: "queued" | "running" | "not-found";
  workerTerminationAcknowledged: boolean;
  codecSetupCancellationAcknowledged?: true;
  artifactCancellationAcknowledged?: true;
}

/** A request-correlated JSON-RPC failure with the worker's typed data intact. */
export class EngineRequestError extends Error {
  readonly code?: number;
  readonly data?: unknown;

  constructor(message: string, code?: number, data?: unknown) {
    super(message);
    this.name = "EngineRequestError";
    this.code = code;
    this.data = data;
  }
}

/**
 * Expected control flow when optional prompt-free work yields the one neural
 * worker to an interactive or Build request. Callers may suppress this exact
 * condition without hiding genuine worker crashes, RPC failures, or timeouts.
 */
export class BackgroundRequestDeferredError extends Error {
  constructor(method: string) {
    super(`Background worker request "${method}" yielded to foreground work.`);
    this.name = "BackgroundRequestDeferredError";
  }
}

interface RequestReservation {
  readonly method: string;
  readonly priority: EngineRequestPriority;
  readonly requestId: string;
  readonly owner: EngineActivityOwner;
  readonly label: string;
  readonly brainId?: string;
  readonly jobId?: string;
  readonly turnId?: string;
  readonly controller: AbortController;
  readonly onTransition?: (transition: EngineActivityTransition) => void;
  readonly beforeDispatch?: () => void | Promise<void>;
  readonly externalSignal?: AbortSignal;
  readonly externalAbort?: () => void;
  readonly settled: Promise<void>;
  readonly resolveSettled: () => void;
  state: EngineActivityState;
  queuePosition?: number;
  queuedBehind?: EngineActivityReference;
  cancellationPhase?: "queued" | "running";
  workerTerminationAcknowledged: boolean;
  codecSetupCancellationAcknowledged?: true;
  artifactCancellationAcknowledged?: true;
  inlineScopeCancellationRequested?: true;
  dispatched: boolean;
  cancelled: boolean;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: string;
  result?: unknown;
  error?: {
    code?: number;
    message?: string;
    data?: unknown;
  };
}

interface JsonRpcNotification {
  jsonrpc: "2.0";
  method: string;
  params?: unknown;
}

interface PythonCandidate {
  command: string;
  prefix: string[];
}

interface ChildLifecycle {
  child: ChildProcessWithoutNullStreams;
  closed: Promise<void>;
  resolveClosed(): void;
  stdoutBuffer: string;
  stdoutListener(chunk: string): void;
  stderrListener(chunk: string): void;
  stderr: string[];
  processError?: string;
  forcedClose?: NodeJS.Timeout;
  settled: boolean;
}

export interface EngineEvent {
  type: string;
  brainId?: string;
  jobId?: string;
  /** Correlates ordered chat notifications with the request that created them. */
  streamId?: string;
  sequence?: number;
  actionId?: string;
  progress?: number;
  message?: string;
  data?: unknown;
}

export interface EngineSupervisorOptions {
  appPath: string;
  resourcesPath?: string;
  workerPath?: string;
  pythonCommand?: string;
  /** Trusted app-owned path shared by neural and inspection worker children. */
  sharedResourceLedgerPath?: string;
  sendSignal?: (pid: number, signal: NodeJS.Signals) => void;
  videoRuntimeCacheRoot?: string;
  prepareVideoRuntime?: (
    signal: AbortSignal,
    onProgress: (progress: VideoRuntimeProgress) => void
  ) => Promise<PreparedVideoRuntime>;
}

/** Health/code checks and text-only jobs never initiate dependency downloads. */
export function requestNeedsVideoRuntime(method: string, params: Record<string, unknown>): boolean {
  if (method === "generate_modality") return params.modality === "video";
  if (method === "ingest") {
    const path = typeof params.path === "string" ? params.path : "";
    // These inputs use the existing Pillow/PCM paths, not a video runtime.
    if (/\.(?:gif|wav|wave|pcm)$/i.test(path)) return false;
    return params.kind === "video" || params.kind === "audio" ||
      /\.(?:mp4|webm|mov|mkv|avi|m4v|mp3|m4a|aac|ogg|opus)$/i.test(path);
  }
  if (method === "observe_packet") {
    const mime = typeof params.mimeType === "string" ? params.mimeType.split(";", 1)[0] ?? "" : "";
    return /^(?:video\/(?:mp4|webm|quicktime|ogg|mp2t|x-matroska)|audio\/(?:webm|ogg|mp4|aac|opus))$/i.test(mime);
  }
  return false;
}

function workerCandidates(options: EngineSupervisorOptions): string[] {
  return [
    options.workerPath,
    options.resourcesPath ? join(options.resourcesPath, "engine", "worker.py") : undefined,
    join(options.appPath, "engine", "worker.py"),
    resolve(process.cwd(), "engine", "worker.py")
  ].filter((path, index, paths): path is string => Boolean(path) && paths.indexOf(path) === index);
}

function pythonCandidates(options: EngineSupervisorOptions): PythonCandidate[] {
  const configured =
    options.pythonCommand?.trim() ||
    process.env.OMNI_PYTHON?.trim() ||
    process.env.OMNI_AGI_PYTHON?.trim();
  const embedded = options.resourcesPath
    ? join(options.resourcesPath, "python", process.platform === "win32" ? "python.exe" : "bin/python3")
    : undefined;
  const candidates: PythonCandidate[] = [];
  if (configured) candidates.push({ command: configured, prefix: [] });
  if (embedded && existsSync(embedded)) candidates.push({ command: embedded, prefix: [] });
  if (process.platform === "win32") {
    candidates.push(
      { command: "py", prefix: ["-3"] },
      { command: "python", prefix: [] },
      { command: "python3", prefix: [] }
    );
  } else {
    candidates.push({ command: "python3", prefix: [] }, { command: "python", prefix: [] });
  }
  return candidates.filter(
    (candidate, index, all) =>
      all.findIndex(
        (entry) =>
          entry.command === candidate.command && entry.prefix.join("\0") === candidate.prefix.join("\0")
      ) === index
  );
}

export function packagedEnginePath(
  resourcesPath: string,
  platform: NodeJS.Platform = process.platform
): string {
  return join(
    resourcesPath,
    "engine-runtime",
    platform === "win32" ? "omni-engine.exe" : "omni-engine"
  );
}

function messageFromError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function workerTraceback(value: unknown): string | undefined {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return undefined;
  }
  const traceback = (value as Record<string, unknown>).traceback;
  if (typeof traceback !== "string") return undefined;
  const bounded = traceback.replace(/\0/g, "").trim().slice(-8_000);
  return bounded || undefined;
}

const SPARSE_COMMITTED_METHODS = new Set(["sparse_substrate_inspect", "sparse_substrate_export",
  "sparse_substrate_validate_bundle", "sparse_substrate_prepare", "sparse_substrate_prepare_rollback", "sparse_substrate_head"]);

function inferredActivityOwner(
  method: string,
  priority: EngineRequestPriority,
  workerRole: "neural" | "inspection"
): EngineActivityOwner {
  if (workerRole === "inspection" || method === "query_whole_brain" || method === "query_substrate" || method === "query_cortex" || method === "query_concept_id_view" || SPARSE_COMMITTED_METHODS.has(method)) return "inspection";
  if (priority === "background" || method === "idle_cycle") return "idle";
  if (method === "create") return "build";
  if (["chat", "resume_chat", "resume_inherited_chat", "chat_receipt"].includes(method)) return "chat";
  if (method.startsWith("evolution.")) return "evolution";
  if (method === "train") return "training";
  if (method === "ingest") return "ingestion";
  if (method === "generate_modality" || method === "generate_neural_speech") return "modality";
  return "system";
}

function inferredActivityLabel(method: string, owner: EngineActivityOwner): string {
  if (owner === "chat") return method === "chat_receipt" ? "Recovering chat turn" : "Chat response";
  if (owner === "evolution") return "Neural evolution";
  if (owner === "build") return "Building brain";
  if (owner === "training") return "Neural training";
  if (owner === "ingestion") return "Dataset learning";
  if (owner === "modality") return "Neural media generation";
  if (owner === "inspection") return "Brain inspection";
  if (owner === "idle") return "Background cognition";
  return method.replace(/[._-]+/g, " ");
}

export class EngineSupervisor extends EventEmitter {
  private readonly options: EngineSupervisorOptions;
  private readonly workerRole: "neural" | "inspection";
  private inspectionWorker?: EngineSupervisor;
  private child?: ChildProcessWithoutNullStreams;
  private pending = new Map<string, PendingRequest>();
  private starting?: Promise<boolean>;
  private terminating?: Promise<void>;
  private terminatingChild?: ChildProcessWithoutNullStreams;
  private unacknowledgedChild?: ChildProcessWithoutNullStreams;
  private lifecycle?: ChildLifecycle;
  private stopping = false;
  private lastError = "Python worker has not been started.";
  private recentStderr: string[] = [];
  private requestQueueTail: Promise<void> = Promise.resolve();
  private readonly requestReservations = new Set<RequestReservation>();
  private readonly requestReservationsById = new Map<string, RequestReservation>();
  private readonly hostComputeReservations = new Set<RequestReservation>();
  private activeRequest?: RequestReservation;
  private readonly instanceDispatchHolds = new Set<string>();
  private readonly instanceDispatchWaiters = new Map<string, Set<() => void>>();
  private backgroundPreemption?: Promise<void>;
  private readonly codecBridge: CodecRuntimeSetupBridge;
  private readonly inlineCodecOwners = new Map<string, { owner: CodecRuntimeOwner; controller: AbortController; child: ChildProcessWithoutNullStreams }>();

  constructor(
    options: EngineSupervisorOptions,
    workerRole: "neural" | "inspection" = "neural"
  ) {
    super();
    this.options = options;
    this.workerRole = workerRole;
    this.codecBridge = new CodecRuntimeSetupBridge(async (signal, progress) => {
      if (!this.options.prepareVideoRuntime) throw new Error("Trusted pinned codec setup is unavailable in this runtime.");
      return this.options.prepareVideoRuntime(signal, progress);
    });
  }

  private activityReference(reservation: RequestReservation): EngineActivityReference {
    return {
      requestId: reservation.requestId,
      owner: reservation.owner,
      label: reservation.label,
      method: reservation.method,
      ...(reservation.brainId ? { brainId: reservation.brainId } : {}),
      ...(reservation.jobId ? { jobId: reservation.jobId } : {}),
      ...(reservation.turnId ? { turnId: reservation.turnId } : {})
    };
  }

  holdInstanceDispatch(brainId: string): void { this.instanceDispatchHolds.add(brainId); }
  releaseInstanceDispatch(brainId: string): void {
    this.instanceDispatchHolds.delete(brainId);
    for (const wake of this.instanceDispatchWaiters.get(brainId) ?? []) wake();
  }
  private instanceDispatchHeld(reservation: RequestReservation): boolean {
    return Boolean(reservation.brainId && this.instanceDispatchHolds.has(reservation.brainId) &&
      !["load", "resume_chat", "resume_inherited_chat", "suspend_instance", "stop_suspended_chat", "stop_inherited_chat", "chat_receipt"].includes(reservation.method));
  }
  private async waitForInstanceDispatch(reservation: RequestReservation): Promise<void> {
    if (!this.instanceDispatchHeld(reservation)) return;
    const brainId = reservation.brainId!, signal = reservation.controller.signal;
    reservation.queuedBehind = { requestId: `instance:${brainId}`, owner: "system", label: "Explicit Load and retained turn", method: "instance-lifecycle", brainId };
    reservation.queuePosition = 1; this.transitionActivity(reservation, "queued");
    await new Promise<void>((resolve, reject) => {
      const waiting = this.instanceDispatchWaiters.get(brainId) ?? new Set<() => void>(); this.instanceDispatchWaiters.set(brainId, waiting);
      const cleanup = () => { waiting.delete(wake); signal.removeEventListener("abort", abort); if (!waiting.size) this.instanceDispatchWaiters.delete(brainId); };
      const wake = () => { if (this.instanceDispatchHeld(reservation)) return; cleanup(); resolve(); };
      const abort = () => { cleanup(); reject(signal.reason ?? new Error("The queued request was cancelled.")); };
      waiting.add(wake); signal.addEventListener("abort", abort, { once: true });
      if (signal.aborted) abort(); else wake();
    });
  }

  private transitionActivity(
    reservation: RequestReservation,
    state: EngineActivityState
  ): void {
    reservation.state = state;
    const transition: EngineActivityTransition = {
      ...this.activityReference(reservation),
      state,
      ...((state === "queued" || reservation.cancellationPhase === "queued") &&
      reservation.queuePosition !== undefined
        ? { queuePosition: reservation.queuePosition }
        : {}),
      ...((state === "queued" || reservation.cancellationPhase === "queued") &&
      reservation.queuedBehind
        ? { queuedBehind: { ...reservation.queuedBehind } }
        : {}),
      ...(reservation.cancellationPhase
        ? { cancellationPhase: reservation.cancellationPhase }
        : {}),
      ...(["cancelled", "failed"].includes(state)
        ? {
            workerTerminationAcknowledged:
              reservation.workerTerminationAcknowledged
          }
        : {})
    };
    try {
      reservation.onTransition?.(transition);
    } catch (error) {
      this.emit(
        "diagnostic",
        `Engine activity observer failed: ${messageFromError(error)}`
      );
    }
    this.emit("activity", transition);
  }

  private createRequestReservation(
    method: string,
    params: Record<string, unknown>,
    priority: EngineRequestPriority,
    signal?: AbortSignal,
    context?: EngineRequestContext,
    hostCompute = false
  ): RequestReservation {
    const requestId =
      context?.requestId?.trim() ||
      (typeof params.jobId === "string" ? params.jobId.trim() : "") ||
      randomUUID();
    if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(requestId)) {
      throw new Error("Engine activity requestId is invalid.");
    }
    if (this.requestReservationsById.has(requestId)) {
      throw new Error(`Engine activity requestId "${requestId}" is already active.`);
    }
    const owner = context?.owner ?? inferredActivityOwner(method, priority, this.workerRole);
    const label = (context?.label ?? inferredActivityLabel(method, owner))
      .replace(/\0/g, "")
      .trim()
      .slice(0, 200);
    if (!label) throw new Error("Engine activity label is invalid.");
    const controller = new AbortController();
    let resolveSettled!: () => void;
    const settled = new Promise<void>((resolve) => {
      resolveSettled = resolve;
    });
    let reservation!: RequestReservation;
    const externalAbort = signal
      ? (): void => {
          reservation.cancelled = true;
          if (!reservation.cancellationPhase) {
            reservation.cancellationPhase =
              reservation.state === "running" ? "running" : "queued";
          }
          if (!controller.signal.aborted) {
            controller.abort(
              signal.reason instanceof Error
                ? signal.reason
                : new Error(`Worker request "${method}" was cancelled.`)
            );
          }
        }
      : undefined;
    reservation = {
      method,
      priority,
      requestId,
      owner,
      label,
      brainId:
        context?.brainId ??
        (typeof params.brainId === "string" ? params.brainId : undefined),
      jobId:
        context?.jobId ??
        (typeof params.jobId === "string" ? params.jobId : undefined),
      turnId: context?.turnId,
      controller,
      onTransition: context?.onTransition,
      beforeDispatch: context?.beforeDispatch,
      externalSignal: signal,
      externalAbort,
      settled,
      resolveSettled,
      state: "queued",
      workerTerminationAcknowledged: false,
      dispatched: false,
      cancelled: false
    };
    const preceding = [...this.requestReservations].filter((entry) => !entry.cancelled);
    if (preceding.length > 0) {
      reservation.queuePosition = preceding.length;
      reservation.queuedBehind = this.activityReference(
        this.activeRequest && !this.activeRequest.cancelled
          ? this.activeRequest
          : preceding[0]!
      );
    }
    this.requestReservations.add(reservation);
    if (hostCompute) this.hostComputeReservations.add(reservation);
    this.requestReservationsById.set(requestId, reservation);
    signal?.addEventListener("abort", externalAbort!, { once: true });
    if (signal?.aborted) externalAbort?.();
    if (reservation.queuedBehind) this.transitionActivity(reservation, "queued");
    return reservation;
  }

  private finishRequestReservation(reservation: RequestReservation): void {
    reservation.externalSignal?.removeEventListener(
      "abort",
      reservation.externalAbort as () => void
    );
    this.requestReservations.delete(reservation);
    if (this.requestReservationsById.get(reservation.requestId) === reservation) {
      this.requestReservationsById.delete(reservation.requestId);
    }
    reservation.resolveSettled();
  }

  /**
   * Cancel exactly one queued or running runtime reservation and do not resolve
   * until a dispatched worker process has actually exited. A queued request is
   * removed without touching the unrelated activity that currently owns the
   * serial worker.
   */
  async cancelRequest(requestId: string): Promise<EngineCancellationAcknowledgement> {
    const reservation = this.requestReservationsById.get(requestId);
    if (!reservation) {
      return {
        requestId,
        acknowledged: false,
        phase: "not-found",
        workerTerminationAcknowledged: false
      };
    }
    const phase = reservation.dispatched ? "running" : "queued";
    reservation.cancelled = true;
    reservation.cancellationPhase = phase;
    this.transitionActivity(reservation, "cancelling");
    if (!reservation.controller.signal.aborted) {
      reservation.controller.abort(
        new Error(`Worker request "${reservation.method}" was cancelled.`)
      );
    }
    await reservation.settled;
    return {
      requestId,
      acknowledged: reservation.state === "cancelled",
      phase,
      workerTerminationAcknowledged: reservation.workerTerminationAcknowledged,
      ...(reservation.codecSetupCancellationAcknowledged ? { codecSetupCancellationAcknowledged: true as const } : {}),
      ...(reservation.artifactCancellationAcknowledged ? { artifactCancellationAcknowledged: true as const } : {})
    };
  }

  private inspectionSupervisor(): EngineSupervisor {
    if (this.workerRole === "inspection") return this;
    if (!this.inspectionWorker) {
      this.inspectionWorker = new EngineSupervisor(
        this.options,
        "inspection"
      );
    }
    return this.inspectionWorker;
  }

  get pid(): number | undefined {
    return this.child?.pid;
  }

  /** Current children owned by this supervisor; never scans host processes. */
  get managedProcessIds(): readonly number[] {
    return [...new Set([this.child?.pid, this.unacknowledgedChild?.pid,
      ...(this.inspectionWorker?.managedProcessIds ?? [])].filter((pid): pid is number => typeof pid === "number"))];
  }

  async start(): Promise<boolean> {
    if (HOST_COMPUTE_SCOPE.getStore() === this) throw new Error("Release the local teacher host-compute reservation before starting or loading native work.");
    if (NATIVE_START_SCOPE.getStore() !== this && this.hostComputeReservations.size) {
      await waitHostComputePredecessors([...this.hostComputeReservations].map(entry => entry.settled));
    }
    if (this.terminating) await this.terminating;
    if (this.unacknowledgedChild) {
      if (
        this.unacknowledgedChild.exitCode === null &&
        this.unacknowledgedChild.signalCode === null
      ) {
        throw new Error(
          "The previous Python worker has not acknowledged termination; a replacement will not be started."
        );
      }
      this.unacknowledgedChild = undefined;
    }
    if (this.child && !this.child.killed && this.child.exitCode === null) return true;
    const lifecycle = this.lifecycle;
    if (this.child && lifecycle?.child === this.child) {
      // An exited process is not replaced until all of its stdio streams have
      // closed. This keeps its pending requests and stderr diagnostics from
      // crossing into the replacement worker.
      await lifecycle.closed;
    }
    if (this.terminating) await this.terminating;
    if (this.child && !this.child.killed && this.child.exitCode === null) return true;
    if (this.starting) return this.starting;
    this.starting = this.startCandidates().finally(() => {
      this.starting = undefined;
    });
    return this.starting;
  }

  private async startCandidates(): Promise<boolean> {
    const worker = workerCandidates(this.options).find(existsSync);
    const configured =
      this.options.pythonCommand?.trim() ||
      process.env.OMNI_PYTHON?.trim() ||
      process.env.OMNI_AGI_PYTHON?.trim();
    const packagedExecutable = this.options.resourcesPath
      ? packagedEnginePath(this.options.resourcesPath)
      : undefined;
    const packagedRequired = process.env.OMNI_PACKAGED_ENGINE_REQUIRED === "1";
    const candidates: Array<{ candidate: PythonCandidate; direct: boolean }> = [];
    // A packaged build must exercise its self-contained, non-pickle worker first.
    // A source-Python worker is a development/recovery launch form of the
    // same authoritative OmniCortex engine, never an alternate brain.
    if (packagedExecutable && existsSync(packagedExecutable)) {
      candidates.push({ candidate: { command: packagedExecutable, prefix: [] }, direct: true });
    }
    if (!packagedRequired && configured && worker) {
      candidates.push({ candidate: { command: configured, prefix: [] }, direct: false });
    }
    if (!packagedRequired && worker) {
      candidates.push(
        ...pythonCandidates({ ...this.options, pythonCommand: undefined })
          .filter((candidate) => candidate.command !== configured)
          .map((candidate) => ({ candidate, direct: false }))
      );
    }
    if (candidates.length === 0) {
      this.lastError =
        packagedRequired
          ? "The required packaged OmniCortex engine was not found."
          : "No packaged engine or engine/worker.py source was found.";
      return false;
    }
    for (const launch of candidates) {
      try {
        await this.spawnCandidate(launch.candidate, worker, launch.direct);
        return true;
      } catch (error) {
        this.lastError = `${launch.candidate.command}: ${messageFromError(error)}`;
        await this.terminateChild();
      }
    }
    return false;
  }

  private async spawnCandidate(
    candidate: PythonCandidate,
    worker: string | undefined,
    direct: boolean
  ): Promise<void> {
    if (!direct && !worker) throw new Error("Python worker source was not found.");
    const args = direct ? candidate.prefix : [...candidate.prefix, "-u", worker as string];
    const child = spawn(candidate.command, args, {
      cwd: direct ? dirname(candidate.command) : dirname(worker as string),
      env: {
        ...managedWorkerMemoryEnvironment(process.env, this.options.sharedResourceLedgerPath),
        PYTHONUNBUFFERED: "1",
        OMNI_PROTOCOL_VERSION: String(PROTOCOL_VERSION),
        OMNI_WORKER_ROLE: this.workerRole,
        ...(this.options.videoRuntimeCacheRoot
          ? { OMNI_VIDEO_RUNTIME_CACHE_ROOT: resolve(this.options.videoRuntimeCacheRoot) }
          : {})
      },
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
      shell: false
    });

    await new Promise<void>((resolveSpawn, rejectSpawn) => {
      const onError = (error: Error): void => {
        child.removeListener("spawn", onSpawn);
        rejectSpawn(error);
      };
      const onSpawn = (): void => {
        child.removeListener("error", onError);
        resolveSpawn();
      };
      child.once("error", onError);
      child.once("spawn", onSpawn);
    });

    this.superviseChild(child);

    try {
      await this.rawRequest("health", {}, 30_000);
      this.lastError = "";
    } catch (error) {
      throw new Error(`Worker health handshake failed: ${messageFromError(error)}`);
    }
  }

  private superviseChild(child: ChildProcessWithoutNullStreams): ChildLifecycle {
    let resolveClosed!: () => void;
    const closed = new Promise<void>((resolveClose) => {
      resolveClosed = resolveClose;
    });
    const lifecycle = {} as ChildLifecycle;
    Object.assign(lifecycle, {
      child,
      closed,
      resolveClosed,
      stdoutBuffer: "",
      stdoutListener: (chunk: string) => this.consumeStdout(lifecycle, chunk),
      stderrListener: (chunk: string) => {
        lifecycle.stderr.push(chunk.trim().slice(-4_000));
        lifecycle.stderr.splice(0, Math.max(0, lifecycle.stderr.length - 12));
      },
      stderr: [],
      settled: false
    } satisfies Omit<ChildLifecycle, "processError" | "forcedClose">);
    this.child = child;
    this.lifecycle = lifecycle;
    child.once("spawn", () => {
      if (this.child === child && typeof child.pid === "number") this.emit("worker-started", { pid: child.pid });
    });
    this.recentStderr = lifecycle.stderr;
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", lifecycle.stdoutListener);
    child.stderr.on("data", lifecycle.stderrListener);
    child.once("error", (error) => {
      lifecycle.processError = `Worker process error: ${error.message}`;
      this.rejectPendingForChild(child, new Error(lifecycle.processError));
    });
    child.once("exit", (code, signal) => {
      const detail = `Worker exited with code ${String(code)} and signal ${String(signal)}.`;
      if (this.child === child) this.lastError = detail;
      // `close` can be delayed indefinitely by inherited/open stdio handles.
      // A request with no wall-clock deadline must still settle as soon as its
      // supervised PID exits. Keep close-time handling for complete stderr and
      // replacement ordering, but reject this worker generation immediately.
      this.rejectPendingForChild(child, new Error(detail));
      lifecycle.forcedClose = setTimeout(() => {
        this.handleClose(lifecycle, code, signal);
      }, 3_100);
    });
    // `close` runs after the stdio streams close, so crash diagnostics include
    // the complete bounded stderr tail rather than racing the final writes.
    child.once("close", (code, signal) => {
      this.handleClose(lifecycle, code, signal);
    });
    return lifecycle;
  }

  private rejectPendingForChild(
    child: ChildProcessWithoutNullStreams,
    error: Error
  ): void {
    const retiredRequests = new Set([...this.pending.entries()].filter(([_id, request]) => request.child === child).map(([id]) => id));
    this.codecBridge.cancel((owner) => retiredRequests.has(owner.requestId));
    for (const [key, entry] of this.inlineCodecOwners) if (entry.child === child) {
      entry.controller.abort(new Error("The owning codec worker closed."));
      this.inlineCodecOwners.delete(key);
    }
    for (const [id, request] of [...this.pending.entries()]) {
      if (request.child !== child) continue;
      this.pending.delete(id);
      request.cleanup();
      request.reject(error);
    }
  }

  private consumeStdout(lifecycle: ChildLifecycle, chunk: string): void {
    if (this.lifecycle !== lifecycle || this.child !== lifecycle.child || lifecycle.settled) {
      return;
    }
    lifecycle.stdoutBuffer += chunk;
    if (lifecycle.stdoutBuffer.length > MAX_PROTOCOL_LINE && !lifecycle.stdoutBuffer.includes("\n")) {
      this.lastError = "Worker emitted an oversized protocol line.";
      void this.terminateChild();
      return;
    }
    let newline = lifecycle.stdoutBuffer.indexOf("\n");
    while (newline >= 0) {
      const line = lifecycle.stdoutBuffer.slice(0, newline).trim();
      lifecycle.stdoutBuffer = lifecycle.stdoutBuffer.slice(newline + 1);
      if (line) this.consumeLine(line);
      newline = lifecycle.stdoutBuffer.indexOf("\n");
    }
  }

  private consumeLine(line: string): void {
    let message: unknown;
    try {
      message = JSON.parse(line);
    } catch {
      this.recentStderr.push(`Ignored non-JSON stdout: ${line.slice(0, 500)}`);
      this.recentStderr = this.recentStderr.slice(-12);
      return;
    }
    if (typeof message !== "object" || message === null) return;
    const record = message as Partial<JsonRpcResponse & JsonRpcNotification>;
    if (record.jsonrpc !== "2.0") return;
    if (typeof record.id === "string") {
      const pending = this.pending.get(record.id);
      if (!pending) return;
      this.pending.delete(record.id);
      pending.cleanup();
      if (record.error) {
        const message =
          typeof record.error.message === "string"
            ? record.error.message
            : "The Python worker returned an error.";
        const traceback = workerTraceback(record.error.data);
        pending.reject(new EngineRequestError(
          traceback
            ? `${message}\nWorker traceback:\n${traceback}`
            : message,
          typeof record.error.code === "number" && Number.isFinite(record.error.code)
            ? record.error.code
            : undefined,
          record.error.data
        ));
      } else {
        pending.resolve(record.result);
      }
      return;
    }
    if (record.method === "event") {
      const event =
        typeof record.params === "object" && record.params !== null
          ? (record.params as EngineEvent)
          : ({ type: "worker-event", data: record.params } satisfies EngineEvent);
      if (this.handleCodecRuntimeEvent(event)) return;
      const emittedVisibleOutput = (event.type === "chat-token" || event.type === "chat-ponder-self-report") &&
        typeof event.data === "object" && event.data !== null &&
        typeof (event.data as Record<string, unknown>).delta === "string" &&
        (event.data as Record<string, unknown>).delta !== "";
      const completedReply = event.type === "chat-phase" &&
        typeof event.data === "object" && event.data !== null &&
        (event.data as Record<string, unknown>).phase === "reply-complete-learning";
      if (
        typeof event.streamId === "string" && event.streamId &&
        (emittedVisibleOutput || completedReply)
      ) {
        for (const pending of this.pending.values()) {
          if (pending.child === this.child && pending.chatStreamId === event.streamId) {
            pending.chatOutputOrCommitObserved = true;
          }
        }
      }
      this.emit("event", event);
    }
  }

  private handleCodecRuntimeEvent(event: EngineEvent): boolean {
    const data = event.data && typeof event.data === "object" && !Array.isArray(event.data)
      ? event.data as Record<string, unknown> : undefined;
    const key = (requestId: string, actionId: string) => `${requestId}:${actionId}`;
    if (event.type === "inline-imagination-started" && data && typeof data.requestId === "string" &&
        typeof event.actionId === "string" && /^[a-f0-9]{32}$/.test(event.actionId) && this.child) {
      const pending = this.pending.get(data.requestId);
      const owner = pending?.codecOwner;
      if (pending?.child === this.child && owner && owner.brainId === event.brainId &&
          owner.streamId === event.streamId && pending.chatStreamId === event.streamId) {
        this.inlineCodecOwners.set(key(data.requestId, event.actionId), {
          owner: { ...owner, jobId: "", actionId: event.actionId }, controller: new AbortController(), child: this.child
        });
      }
    }
    if (event.type === "inline-imagination-finished" || event.type === "inline-imagination-cancelled") {
      for (const [entryKey, entry] of this.inlineCodecOwners) {
        if (entry.owner.brainId === event.brainId && entry.owner.streamId === event.streamId && entry.owner.actionId === event.actionId &&
            (event.type === "inline-imagination-cancelled" || data?.requestId === entry.owner.requestId)) {
          entry.controller.abort(new Error("This exact inline codec owner ended."));
          this.inlineCodecOwners.delete(entryKey);
        }
      }
    }
    if (event.type === "codec-runtime-released") {
      const owner = codecRuntimeChallenge({ ...data, purpose: "decode-video" });
      if (owner) {
        this.codecBridge.release(owner.challengeId, owner);
        this.pending.get(owner.requestId)?.codecChallenges?.delete(owner.challengeId);
      }
      return true;
    }
    if (event.type !== "codec-runtime-needed") return false;
    const challenge = codecRuntimeChallenge(data);
    const pending = challenge ? this.pending.get(challenge.requestId) : undefined;
    const direct = challenge && pending && pending.child === this.child && pending.codecOwner && sameCodecOwner(pending.codecOwner, challenge);
    const inline = challenge ? this.inlineCodecOwners.get(key(challenge.requestId, challenge.actionId)) : undefined;
    if (!challenge || !this.child || challenge.brainId !== (event.brainId ?? "") || challenge.jobId !== (event.jobId ?? "") ||
        challenge.streamId !== (event.streamId ?? "") || challenge.actionId !== (event.actionId ?? "") ||
        (!direct && (!inline || inline.child !== this.child || !sameCodecOwner(inline.owner, challenge)))) {
      this.emit("diagnostic", "Refused an unowned or malformed worker codec challenge.");
      return true;
    }
    if (direct) pending?.codecChallenges?.add(challenge.challengeId);
    const signal = direct ? pending?.codecSignal ?? new AbortController().signal : inline!.controller.signal;
    try {
      this.codecBridge.accept(challenge, signal, async (receipt) => {
        if (receipt.outcome === "ready" && !this.options.videoRuntimeCacheRoot) throw new Error("Codec setup has no fixed main-owned cache root.");
        const ack = await this.rawRequest("resolve_codec_runtime", receipt as unknown as Record<string, unknown>, 10_000, undefined, true) as Record<string, unknown>;
        if (ack.acknowledged !== true || ack.challengeId !== challenge.challengeId) throw new Error("Worker codec receipt acknowledgement is invalid.");
      }, (progress) => this.emit("event", {
        type: "video-runtime-setup", brainId: challenge.brainId, jobId: challenge.jobId,
        streamId: challenge.streamId, actionId: challenge.actionId, message: progress.message,
        data: { ...progress, setupOnly: true, requestId: challenge.requestId, challengeId: challenge.challengeId }
      } satisfies EngineEvent));
    } catch (error) { this.emit("diagnostic", `Codec challenge failed: ${messageFromError(error)}`); }
    return true;
  }

  private signalCooperativeCancellation(
    child: ChildProcessWithoutNullStreams
  ): boolean {
    const pid = child.pid;
    if (!pid) return false;
    const signal = (
      process.platform === "win32" ? "SIGBREAK" : "SIGUSR1"
    ) as NodeJS.Signals;
    try {
      (this.options.sendSignal ?? ((target, value) => process.kill(target, value)))(
        pid,
        signal
      );
      return true;
    } catch {
      return false;
    }
  }

  private rawRequest(
    method: string,
    params: Record<string, unknown>,
    timeoutMs: number,
    signal?: AbortSignal,
    control = false
  ): Promise<unknown> {
    const child = this.child;
    if (!child || child.killed || child.exitCode !== null || !child.stdin.writable) {
      return Promise.reject(new Error("Python worker is unavailable."));
    }
    if (signal?.aborted) {
      return Promise.reject(new Error(`Worker request "${method}" was cancelled.`));
    }
    const id = randomUUID();
    const loadedMetadata = method === "native_operation_auto_prepare" ||
      method === "native_operation_auto_release_stopped" || method === "hardware_topology" && params.measure !== true &&
      typeof params.brainId === "string" && typeof params.storagePath === "string";
    return new Promise((resolveRequest, rejectRequest) => {
      let cancellationTimeout: NodeJS.Timeout | undefined;
      const timeout = timeoutMs === ENGINE_REQUEST_NO_DEADLINE
        ? undefined
        : setTimeout(() => {
            const pending = this.pending.get(id);
            if (!pending) return;
            this.pending.delete(id);
            pending.cleanup();
            pending.reject(new Error(`Worker request "${method}" timed out.`));
            // A missing artifact-control receipt is not authority to kill the
            // neural transaction or replace the worker owning a saved reply.
            if (!control && !loadedMetadata) void this.terminateChild();
          }, timeoutMs);
      const abort = (): void => {
        const pending = this.pending.get(id);
        if (!pending || pending.cancelRequested) return;
        pending.cancelRequested = true;
        if (timeout) clearTimeout(timeout);
        const forceAfterGrace = (): void => {
          const current = this.pending.get(id);
          if (!current) return;
          this.pending.delete(id);
          current.cleanup();
          void this.terminateChild(true).then(
            () => current.reject(
              new Error(`Worker request "${method}" was cancelled.`)
            ),
            (error: unknown) => current.reject(
              new Error(
                `Worker request "${method}" cancellation was not acknowledged: ${messageFromError(error)}`
              )
            )
          );
        };
        if (loadedMetadata) {
          // A discarded metadata review is not permission to kill the current
          // neural owner. Drain its response; the serial caller rejects the
          // retired intent after the terminal metadata arrives.
          cancellationTimeout = setTimeout(() => this.emit("diagnostic",
            "Loaded-owner metadata is still draining; the warm worker was not terminated."), COOPERATIVE_CANCEL_GRACE_MS);
          return;
        }
        if (method === "native_operation_auto_arm") {
          const brainId = typeof params.brainId === "string" ? params.brainId : "";
          const operationId = typeof params.operationId === "string" ? params.operationId : "";
          if (brainId && operationId) void this.rawRequest("native_operation_auto_stop", { brainId, operationId,
            ...(typeof params.storagePath === "string" ? { storagePath: params.storagePath } : {}) }, 10_000, undefined, true)
            .catch(error => this.emit("diagnostic", `Exact native operation Auto Stop failed: ${messageFromError(error)}`));
          else this.emit("diagnostic", "Native operation Auto cancellation lacked its main-owned operation identity.");
          cancellationTimeout = setTimeout(() => this.emit("diagnostic",
            "Native operation Auto Arm is settling its exact Stop boundary; the warm worker was not terminated."), COOPERATIVE_CANCEL_GRACE_MS);
          return;
        }
        if (method === "cached_decoder_execution_prepare") {
          // Preparation only returns metadata for a possible later human
          // review. Drain its terminal response, then let the main caller
          // discard it; cancellation is never authority to replace the loaded
          // neural owner.
          cancellationTimeout = setTimeout(() => this.emit("diagnostic",
            "Cached decoder preparation is still draining; the warm worker was not terminated."),
          COOPERATIVE_CANCEL_GRACE_MS);
          return;
        }
        if (method === "cached_decoder_execution_arm") {
          // Arm owns one main-seeded future operation identity. Stop only that
          // operation, including the worker's queued-ID race, and never signal,
          // kill or restart the warm worker.
          const brainId = typeof params.brainId === "string" ? params.brainId : "";
          const operationId = typeof params.operationId === "string" ? params.operationId : "";
          if (brainId && operationId) {
            void this.rawRequest("cached_decoder_execution_stop", {
              brainId, operationId,
              ...(typeof params.storagePath === "string" ? { storagePath: params.storagePath } : {})
            }, 10_000, undefined, true).catch((error: unknown) =>
              this.emit("diagnostic", `Exact cached decoder Stop failed: ${messageFromError(error)}`));
          } else {
            this.emit("diagnostic", "Cached decoder cancellation lacked its exact main-owned operation identity.");
          }
          cancellationTimeout = setTimeout(() => this.emit("diagnostic",
            "Cached decoder arm is still settling its exact Stop boundary; the warm worker was not terminated."),
          COOPERATIVE_CANCEL_GRACE_MS);
          return;
        }
        if (method === "decoder_execution_calibrate") {
          // Calibration has its own exact operation event and bytes-only
          // control handler. Never turn a slow cancellation into authority to
          // terminate or replace the warm neural worker.
          const brainId = typeof params.brainId === "string" ? params.brainId : "";
          const operationId = typeof params.operationId === "string" ? params.operationId : "";
          if (brainId && operationId) {
            void this.rawRequest("decoder_execution_stop", {
              brainId, operationId,
              ...(typeof params.storagePath === "string" ? { storagePath: params.storagePath } : {})
            }, 10_000, undefined, true).catch((error: unknown) =>
              this.emit("diagnostic", `Exact decoder calibration Stop failed: ${messageFromError(error)}`));
          } else {
            this.emit("diagnostic", "Decoder calibration cancellation lacked its exact main-owned operation identity.");
          }
          cancellationTimeout = setTimeout(() => this.emit("diagnostic",
            "Decoder calibration is still settling its exact Stop boundary; the warm worker was not terminated."),
          COOPERATIVE_CANCEL_GRACE_MS);
          return;
        }
        if (method === "admit_app_observation") {
          // Native read/decode has an exact active-request cancellation flag.
          // Once ordinary mutation starts its atomic save must settle; a slow
          // disk or codec unwind is never authority to kill this warm owner.
          if (pending.codecChallenges?.size) this.codecBridge.cancel(owner => owner.requestId === id);
          if (!this.signalCooperativeCancellation(child)) this.emit("diagnostic", "App observation cancellation signal was unavailable; its current save boundary remains owned.");
          cancellationTimeout = setTimeout(() => this.emit("diagnostic",
            "App observation is still settling its read/save boundary; the warm worker was not terminated."), COOPERATIVE_CANCEL_GRACE_MS);
          return;
        }
        const inline = pending.codecOwner?.actionId
          ? [...this.inlineCodecOwners.values()].find((entry) => entry.child === child &&
            entry.owner.brainId === pending.codecOwner?.brainId && entry.owner.actionId === pending.codecOwner?.actionId)
          : undefined;
        if (method === "generate_modality" && inline) {
          // A host job can claim a decode whose codec setup is still bound to
          // its original chat/action scope. Cancel that exact artifact, not the
          // current brain or the saved reply belonging to the retired text RPC.
          if (this.activeRequest?.method === method && this.activeRequest.brainId === inline.owner.brainId) {
            this.activeRequest.inlineScopeCancellationRequested = true;
          }
          this.codecBridge.cancel((owner) => sameCodecOwner(owner, inline.owner));
          void this.rawRequest("cancel_inline_generation", { brainId: inline.owner.brainId,
            streamId: inline.owner.streamId, neuralActionId: inline.owner.actionId }, 10_000, undefined, true)
            .catch((error: unknown) => this.emit("diagnostic", `Exact inline cancel control failed: ${messageFromError(error)}`));
          cancellationTimeout = setTimeout(() => this.emit("diagnostic",
            "Exact inline artifact cleanup remains pending; its warm worker was not terminated."), COOPERATIVE_CANCEL_GRACE_MS);
          return;
        }
        if (pending.codecChallenges?.size) {
          // Setup is an out-of-band file operation, not a reason to terminate
          // the active neural writer. Its handler returns after actual unwind.
          this.codecBridge.cancel((owner) => owner.requestId === id);
          cancellationTimeout = setTimeout(() => this.emit("diagnostic",
            "Owned codec setup cleanup remains pending; its warm worker was not terminated."), COOPERATIVE_CANCEL_GRACE_MS);
          return;
        }
        if ((method === "generate_modality" || method === "generate_neural_speech") && pending.codecOwner) {
          // Strict flags-only control is portable and names the exact raw RPC,
          // brain, job and action. It cannot accidentally signal a sibling.
          void this.rawRequest("cancel_artifact_request", pending.codecOwner as unknown as Record<string, unknown>, 10_000, undefined, true)
            .catch((error: unknown) => this.emit("diagnostic", `Exact artifact cancel control failed: ${messageFromError(error)}`));
          cancellationTimeout = setTimeout(() => this.emit("diagnostic",
            "Exact neural artifact cancellation remains pending; the warm worker was not terminated."), COOPERATIVE_CANCEL_GRACE_MS);
          return;
        }
        if (
          COOPERATIVE_CANCEL_METHODS.has(method) &&
          this.signalCooperativeCancellation(child)
        ) {
          if (method === "generate_modality" || method === "generate_neural_speech") {
            cancellationTimeout = setTimeout(() => this.emit("diagnostic",
              "Exact neural artifact cancellation remains pending; the warm worker was not terminated."), COOPERATIVE_CANCEL_GRACE_MS);
            return;
          }
          const preOutputChat = method === "chat" &&
            Boolean(pending.chatStreamId) &&
            !pending.chatOutputOrCommitObserved;
          if (method === "load" || preOutputChat) {
            cancellationTimeout = setTimeout(() => {
              const current = this.pending.get(id);
              if (!current) return;
              if (preOutputChat && current.chatOutputOrCommitObserved) {
                // A token/phase arrived while cancellation was pending. Its
                // fast experience may be committing; retain the original
                // cooperative window and reconcile any durable receipt.
                cancellationTimeout = setTimeout(
                  forceAfterGrace,
                  COOPERATIVE_CANCEL_GRACE_MS - PRE_OUTPUT_CANCEL_GRACE_MS
                );
                return;
              }
              forceAfterGrace();
            }, PRE_OUTPUT_CANCEL_GRACE_MS);
          } else {
            cancellationTimeout = setTimeout(
              forceAfterGrace,
              COOPERATIVE_CANCEL_GRACE_MS
            );
          }
        } else {
          forceAfterGrace();
        }
      };
      const cleanup = (): void => {
        if (timeout) clearTimeout(timeout);
        if (cancellationTimeout) clearTimeout(cancellationTimeout);
        signal?.removeEventListener("abort", abort);
      };
      this.pending.set(id, {
        child,
        ...(!control ? { codecOwner: { requestId: id, brainId: typeof params.brainId === "string" ? params.brainId : "",
          jobId: typeof params.jobId === "string" ? params.jobId : "", streamId: typeof params.streamId === "string" ? params.streamId : "",
          actionId: typeof params.neuralActionId === "string" ? params.neuralActionId : "" }, codecSignal: signal, codecChallenges: new Set<string>() } : {}),
        ...(["chat", "resume_chat", "resume_inherited_chat"].includes(method) && typeof params.streamId === "string" && params.streamId
          ? { chatStreamId: params.streamId }
          : {}),
        resolve: resolveRequest,
        reject: rejectRequest,
        timeout,
        cleanup
      });
      signal?.addEventListener("abort", abort, { once: true });
      if (signal?.aborted) {
        abort();
        return;
      }
      child.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`, (error) => {
        if (!error) return;
        const pending = this.pending.get(id);
        if (!pending) return;
        this.pending.delete(id);
        pending.cleanup();
        pending.reject(error);
      });
    });
  }

  /**
   * Reserve one turn in the worker's FIFO protocol loop.
   *
   * The Python worker intentionally executes one neural mutation at a time.
   * Previously Electron still wrote concurrent RPC lines and started every
   * deadline immediately. A harmless 30-second workspace read queued behind a
   * long dataset update could therefore time out before Python had even read
   * it, killing the healthy worker and pausing training at its last cursor.
   *
   * Mirror the worker's serial contract here. Deadlines now begin only after
   * an entry owns the execution turn. A queued request can be cancelled
   * without interrupting the active neural transaction; cancellation after
   * dispatch retains the existing fail-closed worker restart semantics.
   */
  private reserveRequestTurn(signal?: AbortSignal): RequestQueueEntry {
    const predecessor = this.requestQueueTail.catch(() => undefined);
    let release = (): void => undefined;
    const barrier = new Promise<void>((resolveBarrier) => {
      release = resolveBarrier;
    });
    const tail = predecessor.then(() => barrier);
    this.requestQueueTail = tail;

    const ready = signal
      ? new Promise<void>((resolveReady, rejectReady) => {
          let settled = false;
          const finish = (error?: Error): void => {
            if (settled) return;
            settled = true;
            signal.removeEventListener("abort", onAbort);
            if (error) rejectReady(error);
            else resolveReady();
          };
          const onAbort = (): void => {
            const reason = signal.reason;
            finish(
              reason instanceof Error
                ? reason
                : new Error(`Worker request was cancelled while queued.`)
            );
          };
          signal.addEventListener("abort", onAbort, { once: true });
          void predecessor.then(() => finish());
          if (signal.aborted) onAbort();
        })
      : predecessor;

    return {
      ready,
      release: () => {
        release();
        void tail.then(() => {
          if (this.requestQueueTail === tail) {
            this.requestQueueTail = Promise.resolve();
          }
        });
      }
    };
  }

  async request<T>(
    method: string,
    params: Record<string, unknown> = {},
    timeoutMs = 120_000,
    signal?: AbortSignal,
    priority: EngineRequestPriority = "foreground",
    context?: EngineRequestContext
  ): Promise<T> {
    if (HOST_COMPUTE_SCOPE.getStore() === this) throw new Error("Native ingestion/inference cannot re-enter an owned local teacher host-compute slot; release it first.");
    if (method === "configure_video_runtime" || method === "resolve_codec_runtime" || method === "cancel_artifact_request") {
      throw new Error("Video runtime configuration is an internal verified-main operation.");
    }
    if (
      (method === "query_whole_brain" || method === "query_substrate" || method === "query_cortex" || method === "query_concept_id_view" || SPARSE_COMMITTED_METHODS.has(method) || method === "cancel") &&
      this.workerRole === "neural"
    ) {
      if (this.hostComputeReservations.size) await waitHostComputePredecessors([...this.hostComputeReservations].map(entry => entry.settled), signal);
      // Brain Map reads run in their own supervised process. A cold index
      // backfill or wide cursor traversal can be cancelled independently and
      // can never occupy, restart, or queue behind the transactional writer.
      return this.inspectionSupervisor().request<T>(
        method,
        params,
        timeoutMs,
        signal,
        priority,
        context
      );
    }
    if (
      priority === "background" &&
      [...this.requestReservations].some((entry) => !entry.cancelled)
    ) {
      throw new BackgroundRequestDeferredError(method);
    }
    const reservation = this.createRequestReservation(
      method,
      params,
      priority,
      signal,
      context
    );
    const requestSignal = reservation.controller.signal;
    const precedingHosts: Promise<void>[] = [];
    for (const entry of this.requestReservations) {
      if (entry === reservation) break;
      if (this.hostComputeReservations.has(entry)) precedingHosts.push(entry.settled);
    }
    try {
      await waitHostComputePredecessors(precedingHosts, requestSignal);
      if (requestSignal.aborted) {
        throw new Error(`Worker request "${method}" was cancelled.`);
      }
      if (priority === "foreground") await NATIVE_START_SCOPE.run(this, () => this.claimForegroundWorker());
      if (!(await NATIVE_START_SCOPE.run(this, () => this.start()))) throw new Error(this.lastError);
      // A background request can be waiting on the same cold worker startup
      // when foreground work arrives. Re-check ownership after startup before
      // either request enters the serial protocol queue.
      if (priority === "foreground") await NATIVE_START_SCOPE.run(this, () => this.claimForegroundWorker());
      if (requestSignal.aborted) {
        throw new Error(`Worker request "${method}" was cancelled.`);
      }
      if (reservation.cancelled) throw new BackgroundRequestDeferredError(method);
      for (;;) {
      await this.waitForInstanceDispatch(reservation);
      const turn = this.reserveRequestTurn(requestSignal);
      try {
        await turn.ready;
        if (requestSignal.aborted) {
          throw new Error(`Worker request "${method}" was cancelled.`);
        }
        if (reservation.cancelled) throw new BackgroundRequestDeferredError(method);
        // Unload may have arrived while this request was behind a chat.
        // Give up the serial slot before parking, so its Load/resume can run.
        if (this.instanceDispatchHeld(reservation)) continue;
        // The preceding reservation may have been actively cancelled, which
        // intentionally terminates the one serial worker. A different queued
        // request owns a different operation and must continue on a clean
        // replacement rather than fail with "worker unavailable".
        if (!(await NATIVE_START_SCOPE.run(this, () => this.start()))) throw new Error(this.lastError);
        if (requestSignal.aborted) {
          throw new Error(`Worker request "${method}" was cancelled.`);
        }
        this.activeRequest = reservation;
        reservation.queuePosition = undefined;
        reservation.queuedBehind = undefined;
        this.transitionActivity(reservation, "running");
        try {
          try {
            if (requestNeedsVideoRuntime(method, params)) {
              await this.prepareRequestedVideoRuntime(reservation, requestSignal);
            }
            // Optional background cognition may legitimately spend a long time
            // restoring or evaluating a large local brain. Its caller remains
            // single-flight, and foreground ownership explicitly preempts it;
            // a background wall clock must never restart an otherwise healthy
            // worker in a loop.
            const effectiveTimeoutMs = priority === "background"
              ? ENGINE_REQUEST_NO_DEADLINE
              : timeoutMs;
            await reservation.beforeDispatch?.();
            requestSignal.throwIfAborted();
            reservation.dispatched = true;
            return (await this.rawRequest(
              method,
              params,
              effectiveTimeoutMs,
              requestSignal
            )) as T;
          } catch (error) {
            // Foreground ownership deliberately terminates an active optional
            // cold load. Reclassify only that reservation; unrelated worker
            // exits and protocol errors must continue to surface normally.
            if (
              priority === "background" &&
              reservation.cancelled
            ) {
              throw new BackgroundRequestDeferredError(method);
            }
            throw error;
          }
        } finally {
          if (this.activeRequest === reservation) this.activeRequest = undefined;
        }
      } finally {
        turn.release();
      }
      }
    } catch (error) {
      if (requestSignal.aborted || reservation.cancelled) {
        if (error instanceof EngineRequestError && error.code === -32800 &&
            error.data && typeof error.data === "object" &&
            (error.data as Record<string, unknown>).codecRuntimeCancelled === true &&
            (error.data as Record<string, unknown>).safeBoundary === true) {
          reservation.codecSetupCancellationAcknowledged = true;
        }
        if (reservation.inlineScopeCancellationRequested && error instanceof EngineRequestError && error.code === -32800) {
          reservation.artifactCancellationAcknowledged = true;
        }
        if (error instanceof EngineRequestError && error.code === -32800 && error.data && typeof error.data === "object" &&
            (error.data as Record<string, unknown>).modalityCancelled === true && (error.data as Record<string, unknown>).safeBoundary === true) {
          reservation.artifactCancellationAcknowledged = true;
        }
        if (
          reservation.cancellationPhase === "running" &&
          reservation.dispatched &&
          !(error instanceof EngineRequestError && error.code === -32800) &&
          /(?:was cancelled|Python worker was stopped)/i.test(messageFromError(error))
        ) {
          // rawRequest reports cancellation only after terminateChild confirms
          // process close. Queued cancellation never enters rawRequest and must
          // not claim that the unrelated owner was terminated.
          reservation.workerTerminationAcknowledged = true;
        }
        this.transitionActivity(
          reservation,
          reservation.cancellationPhase === "running" &&
            !reservation.workerTerminationAcknowledged &&
            /not acknowledged/i.test(messageFromError(error))
            ? "failed"
            : "cancelled"
        );
      } else {
        this.transitionActivity(reservation, "failed");
      }
      throw error;
    } finally {
      if (reservation.state === "running") {
        this.transitionActivity(reservation, "complete");
      }
      this.finishRequestReservation(reservation);
    }
  }

  private async prepareRequestedVideoRuntime(
    reservation: RequestReservation,
    signal: AbortSignal
  ): Promise<void> {
    if (!this.options.prepareVideoRuntime) return;
    const onProgress = (progress: VideoRuntimeProgress): void => {
      // Job owners already expose Stop through this same reservation/signal.
      // Setup bytes are not misreported as neural-generation progress.
      this.emit("event", {
        type: "video-runtime-setup",
        brainId: reservation.brainId,
        jobId: reservation.jobId,
        message: progress.message,
        data: progress
      } satisfies EngineEvent);
    };
    let prepared: PreparedVideoRuntime;
    try {
      prepared = await this.options.prepareVideoRuntime(signal, onProgress);
    } catch (error) {
      if (signal.aborted || !(error instanceof VideoRuntimeUnavailableError)) throw error;
      // An empty vetted catalog does not disable existing GIF/APNG/WAV paths,
      // masquerade as successful setup, or bypass a hash/license failure.
      this.emit("diagnostic", error.message);
      return;
    }
    if (signal.aborted) throw new Error("Video runtime setup was cancelled.");
    if (prepared.state === "external") return;
    if (!this.options.videoRuntimeCacheRoot || !prepared.artifactSha256 ||
        !prepared.binarySha256 || !prepared.binarySizeBytes || !prepared.target) {
      throw new Error("Video runtime preparation did not provide its verified binary identity.");
    }
    // Serialize this tiny RPC with the media request. It selects the verified
    // executable in an already-warm worker without restarting or loading it.
    await this.rawRequest("configure_video_runtime", {
      executablePath: prepared.executablePath,
      artifactSha256: prepared.artifactSha256,
      binarySha256: prepared.binarySha256,
      binarySizeBytes: prepared.binarySizeBytes,
      target: prepared.target
    }, ENGINE_REQUEST_NO_DEADLINE, signal);
  }

  /**
   * Preempt optional worker activity before a same-brain foreground caller
   * waits on a higher-level repository write lock. Without this early claim,
   * the caller cannot reach request() to trigger normal foreground ownership.
   */
  async claimForeground(): Promise<void> {
    if (HOST_COMPUTE_SCOPE.getStore() === this) throw new Error("Release the local teacher host-compute slot before claiming native foreground work.");
    if (this.hostComputeReservations.size) await waitHostComputePredecessors([...this.hostComputeReservations].map(entry => entry.settled));
    await this.claimForegroundWorker();
  }

  /** Main-owned external compute shares native startup/dispatch ordering.
   * Never starts/loads/RPCs a model. Caller acquires its brain writer FIRST,
   * then releases this slot and that writer before native source ingestion. */
  async withHostCompute<T>(context: Pick<EngineRequestContext, "requestId" | "brainId" | "jobId" | "label">,
    signal: AbortSignal, operation: (signal: AbortSignal) => Promise<T>): Promise<T> {
    if (this.workerRole !== "neural" || HOST_COMPUTE_SCOPE.getStore() === this || this.stopping) throw new Error("A local teacher needs a nonreentrant current main runtime reservation.");
    signal.throwIfAborted();
    const preceding = [...this.requestReservations, ...(this.inspectionWorker?.requestReservations ?? [])].map(entry => entry.settled);
    const startups = [this.starting, this.terminating, this.inspectionWorker?.starting, this.inspectionWorker?.terminating].filter((entry): entry is Promise<boolean> | Promise<void> => !!entry);
    const reservation = this.createRequestReservation("local_teacher_compute", { brainId: context.brainId }, "foreground", signal,
      { ...context, owner: "training" }, true);
    const ownedSignal = reservation.controller.signal;
    let turn: RequestQueueEntry | undefined;
    try {
      await waitHostComputePredecessors([...preceding, ...startups], ownedSignal);
      turn = this.reserveRequestTurn(ownedSignal); await turn.ready; ownedSignal.throwIfAborted();
      if (this.stopping) throw new Error("Studio is shutting down; no local teacher was started.");
      this.activeRequest = reservation; reservation.dispatched = true;
      reservation.queuePosition = undefined; reservation.queuedBehind = undefined;
      this.transitionActivity(reservation, "running");
      const result = await HOST_COMPUTE_SCOPE.run(this, () => operation(ownedSignal));
      this.transitionActivity(reservation, "complete"); return result;
    } catch (error) {
      this.transitionActivity(reservation, ownedSignal.aborted ? "cancelled" : "failed"); throw error;
    } finally {
      if (this.activeRequest === reservation) this.activeRequest = undefined;
      turn?.release(); this.hostComputeReservations.delete(reservation); this.finishRequestReservation(reservation);
    }
  }

  /** Configure on the existing serial writer; live polls/stops read only the
   * already admitted byte ring on its metadata control lane. Never start,
   * Load, preempt or timeout-restart a neural worker for a graph observer. */
  async routerActivity(brainId: string, storagePath: string, query: RouterSpikeQuery): Promise<unknown> {
    const child = this.child;
    if (this.workerRole !== "neural" || !child || child.killed || child.exitCode !== null || !child.stdin.writable || this.starting || this.terminating)
      throw new Error("Router activity requires an already explicitly loaded native worker.");
    const params = { brainId, storagePath, query };
    if (query.epoch !== undefined) return await this.rawRequest("router_activity_control", params, 10_000, undefined, true);
    const reservation = this.createRequestReservation("router_activity", params, "background", undefined, {
      requestId: `router-capture-${randomUUID()}`, owner: "system", label: "Configuring exact LIF channel observation", brainId
    });
    const turn = this.reserveRequestTurn(reservation.controller.signal);
    try {
      await turn.ready; reservation.controller.signal.throwIfAborted();
      if (this.child !== child || child.killed || child.exitCode !== null || !child.stdin.writable || this.starting || this.terminating || this.instanceDispatchHeld(reservation))
        throw new Error("Router capture's loaded owner changed; no replacement was started.");
      this.activeRequest = reservation; reservation.dispatched = true;
      this.transitionActivity(reservation, "running");
      const value = await this.rawRequest("router_activity", params, 10_000, undefined, true);
      this.transitionActivity(reservation, "complete"); return value;
    } catch (error) {
      this.transitionActivity(reservation, "failed"); throw error;
    } finally {
      if (this.activeRequest === reservation) this.activeRequest = undefined;
      turn.release(); this.finishRequestReservation(reservation);
    }
  }

  /** Reviewed, same-brain private sampling; never re-enters an inline chat. */
  async sampleNativeMcp(approved: ApprovedNativeMcpSamplingInput, signal?: AbortSignal): Promise<NativeMcpSamplingResult> {
    if (this.workerRole !== "neural") throw new Error("Native MCP sampling requires the neural worker.");
    return performNativeMcpSampling({
      control: (method, params) => this.rawRequest(method, params, 10_000, undefined, true),
      runIdle: (owner) => this.request("mcp_sample", { ...owner }, ENGINE_REQUEST_NO_DEADLINE, undefined, "foreground", {
        requestId: `mcp-sample-${randomUUID()}`, owner: "system", label: "Reviewed native MCP sampling",
        brainId: owner.brainId
      })
    }, approved, signal);
  }

  /** Existing warm worker only; these methods still run on its serial writer,
   * never the flags-only stdin control reader. A queued observation cannot
   * spawn/replace a worker or implicitly Load an instance. Once dispatched the
   * native read phase can cancel cooperatively; any entered atomic save settles
   * before cancellation/Unload releases its owner. */
  private async loadedAppObservationRequest(method: "app_observation_frontier" | "admit_app_observation" | "sparse_substrate_apply",
    params: Record<string, unknown>, signal?: AbortSignal, beforeDispatch?: () => void): Promise<unknown> {
    const child = this.child;
    if (this.workerRole !== "neural" || !child || child.killed || child.exitCode !== null || !child.stdin.writable || this.starting || this.terminating) throw new Error("App observations wait for an explicitly loaded existing neural worker.");
    if ([...this.requestReservations].some(entry => !entry.cancelled)) throw new BackgroundRequestDeferredError(method);
    const reservation = this.createRequestReservation(method, params, "background", signal, {
      requestId: `app-observation-${randomUUID()}`,
      owner: "system", label: method === "sparse_substrate_apply" ? "Applying reviewed sparse connections" : method === "app_observation_frontier" ? "Checking reviewed App data frontier" : "Admitting reviewed App data",
      brainId: String(params.brainId), beforeDispatch
    });
    const turn = this.reserveRequestTurn(reservation.controller.signal);
    try {
      await turn.ready; reservation.controller.signal.throwIfAborted();
      if (this.child !== child || child.killed || child.exitCode !== null || !child.stdin.writable || this.starting || this.terminating || this.instanceDispatchHeld(reservation)) throw new Error("App observation's loaded worker boundary changed; no replacement was started.");
      await reservation.beforeDispatch?.(); reservation.controller.signal.throwIfAborted();
      if (this.child !== child || child.killed || child.exitCode !== null || !child.stdin.writable || this.starting || this.terminating || this.instanceDispatchHeld(reservation))
        throw new Error("App observation's loaded owner changed during admission; no replacement was started.");
      this.activeRequest = reservation; reservation.dispatched = true;
      this.transitionActivity(reservation, "running");
      const result = await this.rawRequest(method, params, ENGINE_REQUEST_NO_DEADLINE,
        method !== "app_observation_frontier" ? reservation.controller.signal : undefined);
      this.transitionActivity(reservation, "complete"); return result;
    } catch (error) {
      const data = error instanceof EngineRequestError && error.data && typeof error.data === "object" ? error.data as Record<string, unknown> : undefined;
      const safelyUnapplied = method === "admit_app_observation" && error instanceof EngineRequestError && error.code === -32800 && data !== undefined &&
        data.brainId === params.brainId && data.nativeAppObservation === true && data.cancelled === true && data.safeBoundary === true && data.learningApplied === false && data.nativeOwnerReleased === false;
      this.transitionActivity(reservation, safelyUnapplied || reservation.controller.signal.aborted && !reservation.dispatched ? "cancelled" : "failed");
      throw error;
    }
    finally { if (this.activeRequest === reservation) this.activeRequest = undefined; turn.release(); this.finishRequestReservation(reservation); }
  }
  async appObservationFrontier(brainId: string, storagePath: string, signal?: AbortSignal, beforeDispatch?: () => void): Promise<NativeAppObservationFrontier> {
    return checkedAppObservationFrontier(await this.loadedAppObservationRequest("app_observation_frontier", { brainId, storagePath }, signal, beforeDispatch), brainId);
  }
  /** Explicit already-loaded owner only; never spawn/restart/Load for a paste. */
  async applySparseSubstrate(params: Record<string, unknown>, signal: AbortSignal, beforeDispatch: () => void): Promise<import("../shared/sparseSubstrateTransfer").SparseSubstrateNativeReceipt> {
    return await this.loadedAppObservationRequest("sparse_substrate_apply", params, signal, beforeDispatch) as import("../shared/sparseSubstrateTransfer").SparseSubstrateNativeReceipt;
  }
  /** Detached, permissioned candidates may use only the current native owner.
   * Unlike ordinary foreground work, waiting for the serial slot never gives
   * this operation permission to start or replace a worker. */
  async requestLoadedNativeCandidate<T>(method: string, params: Record<string, unknown>, timeoutMs: number,
    signal: AbortSignal, context: EngineRequestContext): Promise<T> {
    if (!["evolution.propose", "evolution.evaluate", "evolution.list", "evolution.promote", "evolution.reject", "evolution.rollback", "cancel"].includes(method))
      throw new Error("Unsupported loaded native candidate operation.");
    return this.loadedOwnerSerialRequest<T>(method, params, timeoutMs, signal, context, "Native candidate");
  }
  /** Main-owned regional review never implicitly Starts, Loads or replaces a worker. */
  async requestLoadedRegionalExecution<T>(method: string, params: Record<string, unknown>, timeoutMs: number,
    signal: AbortSignal, context: EngineRequestContext): Promise<T> {
    if (!["hardware_topology", "regional_execution_prepare", "regional_execution_status", "regional_execution_activate", "regional_execution_deactivate"].includes(method))
      throw new Error("Unsupported loaded regional execution operation.");
    if (typeof params.brainId !== "string" || params.brainId !== context.brainId)
      throw new Error("Regional execution requires the exact main-owned loaded instance.");
    return this.loadedOwnerSerialRequest<T>(method, params, timeoutMs, signal, context, "Regional device");
  }
  /** Main-owned isolated decoder work on the same explicit loaded owner. */
  async requestLoadedDecoderExecution<T>(method: string, params: Record<string, unknown>,
    signal: AbortSignal, context: EngineRequestContext): Promise<T> {
    if (!["decoder_execution_capture_next", "decoder_execution_prepare", "decoder_execution_calibrate", "decoder_execution_apply", "decoder_execution_release_stopped"].includes(method))
      throw new Error("Unsupported loaded decoder execution operation.");
    if (typeof params.brainId !== "string" || params.brainId !== context.brainId)
      throw new Error("Decoder execution requires the exact main-owned loaded instance.");
    if (method === "decoder_execution_calibrate" &&
        (typeof params.operationId !== "string" || params.operationId !== context.requestId))
      throw new Error("Decoder calibration requires its exact main-seeded operation identity.");
    return this.loadedOwnerSerialRequest<T>(method, params,
      method === "decoder_execution_calibrate" ? ENGINE_REQUEST_NO_DEADLINE : 30_000,
      signal, context, "Decoder execution");
  }
  /** Flags/metadata only: usable while one exact calibration owns the serial loop. */
  async controlLoadedDecoderExecution<T>(method: string, params: Record<string, unknown>): Promise<T> {
    if (!["decoder_execution_status", "decoder_execution_stop"].includes(method))
      throw new Error("Unsupported decoder execution control operation.");
    if (typeof params.brainId !== "string" || !params.brainId || params.brainId.includes("\0"))
      throw new Error("Decoder control requires an exact main-owned instance identity.");
    if (method === "decoder_execution_stop" && params.operationId !== undefined &&
        (typeof params.operationId !== "string" || !params.operationId || params.operationId.includes("\0")))
      throw new Error("Decoder Stop operation identity is invalid.");
    const child = this.child;
    const requireOwner = (): void => {
      if (this.workerRole !== "neural" || !child || this.child !== child || child.killed || child.exitCode !== null ||
          !child.stdin.writable || this.starting || this.terminating || this.stopping)
        throw new Error("Decoder control requires the same explicitly loaded existing worker; no replacement was started.");
    };
    requireOwner();
    const result = await this.rawRequest(method, params, 10_000, undefined, true) as T;
    requireOwner(); return result;
  }
  /** Main-owned cached-decoder metadata/arm work on the same explicit owner. */
  async requestLoadedCachedDecoderExecution<T>(method: string, params: Record<string, unknown>,
    signal: AbortSignal, context: EngineRequestContext): Promise<T> {
    if (!["cached_decoder_execution_prepare", "cached_decoder_execution_arm"].includes(method))
      throw new Error("Unsupported loaded cached decoder execution operation.");
    if (typeof params.brainId !== "string" || params.brainId !== context.brainId)
      throw new Error("Cached decoder execution requires the exact main-owned loaded instance.");
    if (method === "cached_decoder_execution_arm" &&
        (typeof params.operationId !== "string" || params.operationId !== context.requestId))
      throw new Error("Cached decoder arm requires its exact main-seeded operation identity.");
    return this.loadedOwnerSerialRequest<T>(method, params, ENGINE_REQUEST_NO_DEADLINE,
      signal, context, "Cached decoder execution");
  }
  /** Flags/metadata only: remains available while the retained frontier runs. */
  async controlLoadedCachedDecoderExecution<T>(method: string, params: Record<string, unknown>): Promise<T> {
    if (!["cached_decoder_execution_status", "cached_decoder_execution_stop"].includes(method))
      throw new Error("Unsupported cached decoder execution control operation.");
    if (typeof params.brainId !== "string" || !params.brainId || params.brainId.includes("\0"))
      throw new Error("Cached decoder control requires an exact main-owned instance identity.");
    if (method === "cached_decoder_execution_stop" && params.operationId !== undefined &&
        (typeof params.operationId !== "string" || !params.operationId || params.operationId.includes("\0")))
      throw new Error("Cached decoder Stop operation identity is invalid.");
    const child = this.child;
    const requireOwner = (): void => {
      if (this.workerRole !== "neural" || !child || this.child !== child || child.killed || child.exitCode !== null ||
          !child.stdin.writable || this.starting || this.terminating || this.stopping)
        throw new Error("Cached decoder control requires the same explicitly loaded existing worker; no replacement was started.");
    };
    requireOwner();
    const result = await this.rawRequest(method, params, 10_000, undefined, true) as T;
    requireOwner(); return result;
  }
  async requestLoadedNativeOperationAuto<T>(method: string, params: Record<string, unknown>, signal: AbortSignal,
    context: EngineRequestContext): Promise<T> {
    if (!["native_operation_auto_prepare", "native_operation_auto_arm", "native_operation_auto_release_stopped"].includes(method) ||
        typeof params.brainId !== "string" || params.brainId !== context.brainId)
      throw new Error("Native operation Auto requires an exact main-owned loaded instance and serial operation.");
    if (method === "native_operation_auto_arm" && (typeof params.operationId !== "string" || params.operationId !== context.requestId))
      throw new Error("Native operation Auto Arm requires its exact main-seeded operation identity.");
    return this.loadedOwnerSerialRequest<T>(method, params, ENGINE_REQUEST_NO_DEADLINE, signal, context, "Native operation Auto");
  }
  async controlLoadedNativeOperationAuto<T>(method: string, params: Record<string, unknown>): Promise<T> {
    if (!["native_operation_auto_status", "native_operation_auto_stop", "native_operation_auto_plans"].includes(method) ||
        typeof params.brainId !== "string" || !params.brainId || params.brainId.includes("\0"))
      throw new Error("Native operation Auto accepts only current-owner flags/metadata controls.");
    if (method === "native_operation_auto_stop" && params.operationId !== undefined &&
        (typeof params.operationId !== "string" || !params.operationId || params.operationId.includes("\0"))) throw new Error("Native operation Auto Stop identity is invalid.");
    const child = this.child;
    const current = () => {
      if (this.workerRole !== "neural" || !child || this.child !== child || child.killed || child.exitCode !== null ||
          !child.stdin.writable || this.starting || this.terminating || this.stopping)
        throw new Error("Native operation Auto requires the same explicitly loaded worker; no replacement was started.");
    };
    current(); const value = await this.rawRequest(method, params, 10_000, undefined, true) as T; current(); return value;
  }
  private async loadedOwnerSerialRequest<T>(method: string, params: Record<string, unknown>, timeoutMs: number,
    signal: AbortSignal, context: EngineRequestContext, activity: string): Promise<T> {
    const child = this.child;
    const requireOwner = (): void => {
      if (this.workerRole !== "neural" || !child || this.child !== child || child.killed || child.exitCode !== null ||
          !child.stdin.writable || this.starting || this.terminating || this.stopping)
        throw new Error(`${activity} work requires the same explicitly loaded existing worker; no replacement was started.`);
    };
    signal.throwIfAborted(); requireOwner();
    const reservation = this.createRequestReservation(method, params, "foreground", signal, context);
    const turn = this.reserveRequestTurn(reservation.controller.signal);
    try {
      await turn.ready; reservation.controller.signal.throwIfAborted(); requireOwner();
      if (this.instanceDispatchHeld(reservation)) throw new Error(`${activity} work waits for its current loaded execution boundary.`);
      await reservation.beforeDispatch?.();
      reservation.controller.signal.throwIfAborted(); requireOwner();
      this.activeRequest = reservation; reservation.dispatched = true;
      reservation.queuePosition = undefined; reservation.queuedBehind = undefined;
      this.transitionActivity(reservation, "running");
      const result = await this.rawRequest(method, params, timeoutMs, reservation.controller.signal) as T;
      if ((method === "cached_decoder_execution_prepare" || method === "native_operation_auto_prepare" ||
          method === "hardware_topology" && params.measure !== true) && reservation.controller.signal.aborted) {
        const reason = reservation.controller.signal.reason;
        throw reason instanceof Error ? reason : new Error("Loaded-owner preparation was cancelled after its terminal metadata drained.");
      }
      // A real committed ACK remains evidence even if Stop arrived afterward.
      this.transitionActivity(reservation, "complete"); return result;
    } catch (error) {
      this.transitionActivity(reservation, reservation.controller.signal.aborted &&
        (!reservation.dispatched || method === "cached_decoder_execution_prepare" || method === "native_operation_auto_prepare" ||
          method === "hardware_topology" && params.measure !== true) ? "cancelled" : "failed");
      throw error;
    } finally {
      if (this.activeRequest === reservation) this.activeRequest = undefined;
      turn.release(); this.finishRequestReservation(reservation);
    }
  }
  async admitAppObservation(input: NativeAppObservationInput, storagePath: string, signal?: AbortSignal, beforeDispatch?: () => void): Promise<NativeAppObservationResult> {
    const result = await this.loadedAppObservationRequest("admit_app_observation", { ...input, storagePath }, signal, beforeDispatch);
    return checkedAppObservationResult(result, input.brainId, JSON.parse(input.recordJson) as McpAppObservationArtifact,
      input.recordSha256, input.disposition, input.expectedFrontierSha256);
  }

  async cancelInlineGeneration(brainId: string, turnId: string, actionId: string): Promise<{
    requested: boolean; acknowledged: boolean;
  }> {
    if (this.workerRole !== "neural") throw new Error("Inline control requires the neural worker.");
    this.codecBridge.cancel((owner) => owner.brainId === brainId && owner.streamId === turnId && owner.actionId === actionId);
    // This narrow flags-only RPC is handled by the worker's stdin reader even
    // while its one neural dispatch is busy. Never start/reload a brain here.
    return await this.rawRequest("cancel_inline_generation", {
      brainId, streamId: turnId, neuralActionId: actionId
    }, 10_000, undefined, true) as { requested: boolean; acknowledged: boolean };
  }

  async steerChat(brainId: string, turnId: string, successorTurnId: string): Promise<{
    requested: boolean; warm: boolean;
  }> {
    if (this.workerRole !== "neural") throw new Error("Warm Steer requires the neural worker.");
    return await this.rawRequest("steer_chat", { brainId, streamId: turnId, successorTurnId },
      10_000, undefined, true) as { requested: boolean; warm: boolean };
  }

  async suspendChat(brainId: string, turnId: string): Promise<{ requested: boolean; waitingFor?: InstanceLifecycleState["waitingFor"] }> {
    if (this.workerRole !== "neural") throw new Error("Instance suspension requires the neural worker.");
    // A control timeout is not permission to terminate the original turn.
    const result = await this.rawRequest("suspend_chat", { brainId, streamId: turnId }, 10_000, undefined, true) as Record<string, unknown>;
    if (!result || typeof result.requested !== "boolean") throw new Error("Invalid native suspension acknowledgement.");
    return result as unknown as { requested: boolean; waitingFor?: InstanceLifecycleState["waitingFor"] };
  }

  async instanceStatus(brainId: string, storagePath: string): Promise<NativeInstanceStatus> {
    if (this.workerRole !== "neural") throw new Error("Instance status requires the neural worker.");
    if (!await this.start()) throw new Error(this.lastError || "Native status is unavailable.");
    const result = await this.rawRequest("instance_status", { brainId, storagePath }, 30_000, undefined, true) as NativeInstanceStatus;
    if (!result || result.brainId !== brainId || typeof result.learnedRuntimeResident !== "boolean" ||
        !Number.isSafeInteger(result.minimumContextTokens) || result.minimumContextTokens < 0 || !Array.isArray(result.reconnect) ||
        result.resumeCompatibility !== undefined && (typeof result.resumeCompatibility?.compatible !== "boolean" ||
          result.resumeCompatibility.reason !== undefined && typeof result.resumeCompatibility.reason !== "string")) {
      throw new Error("Invalid native instance status.");
    }
    return result;
  }
  async inheritedExecutionPresentation(brainId: string, storagePath: string, executionId: string): Promise<NativeInheritedPresentation> {
    if (this.workerRole !== "neural") throw new Error("Inherited execution status requires the neural worker.");
    if (!await this.start()) throw new Error(this.lastError || "Native inherited status is unavailable.");
    const result = await this.rawRequest("inherited_execution_presentation", { brainId, storagePath, executionId }, 30_000, undefined, true) as NativeInheritedPresentation;
    if (!result || result.brainId !== brainId || result.executionId !== executionId || !result.frontier || !result.presentation) {
      throw new Error("Invalid native inherited execution presentation.");
    }
    return result;
  }

  async observeChatAction(observation: ChatToolObservation): Promise<ChatToolObservationReceipt> {
    if (this.workerRole !== "neural") throw new Error("Live action observations require the neural worker.");
    const active = this.activeRequest;
    if (!active || !["chat", "resume_chat", "resume_inherited_chat"].includes(active.method) || active.brainId !== observation.brainId ||
        active.turnId !== observation.turnId || active.cancelled || active.controller.signal.aborted) {
      return { brainId: observation.brainId, turnId: observation.turnId, observationId: observation.observationId,
        accepted: false, reason: "turn-not-active" };
    }
    // Deposit structured evidence through the reader, not behind the neural
    // RPC that needs to consume it. A missing control ACK never kills/reloads
    // that worker; ordinary Stop/Steer remains the current turn's authority.
    const receipt = await this.rawRequest("observe_chat_action", {
      brainId: observation.brainId, streamId: observation.turnId, observation
    }, 10_000, undefined, true);
    return observationReceipt(receipt, observation);
  }

  async authorizeInlineImagination(params: Record<string, unknown>): Promise<{ accepted: boolean }> {
    if (this.workerRole !== "neural") throw new Error("Inline authorization requires the neural worker.");
    const result = await this.rawRequest("authorize_inline_imagination", params, 10_000, undefined, true) as Record<string, unknown>;
    if (!result || typeof result.accepted !== "boolean" || result.accepted &&
      (result.brainId !== params.brainId || result.streamId !== params.streamId || result.neuralActionId !== params.neuralActionId)) {
      throw new Error("Inline authorization receipt has a different artifact owner.");
    }
    return result as { accepted: boolean };
  }

  async awaitInlineArtifact(params: Record<string, unknown>, signal?: AbortSignal): Promise<boolean> {
    if (typeof params.neuralActionId !== "string") return false;
    let owned: { streamId?: string } | undefined;
    try {
      while (true) {
        signal?.throwIfAborted();
        const state = await this.rawRequest("inline_generation_status", params, 10_000, undefined, true) as {
          exists: boolean; brainId?: string; neuralActionId?: string; started?: boolean; ready?: boolean; cancelled?: boolean; streamId?: string;
        };
        if (!state || typeof state.exists !== "boolean" || state.exists &&
          (state.brainId !== params.brainId || state.neuralActionId !== params.neuralActionId ||
           typeof state.streamId !== "string" || typeof state.ready !== "boolean" || typeof state.cancelled !== "boolean")) {
          throw new Error("Inline artifact status has a different owned request.");
        }
        if (!state.exists) return false;
        owned = state;
        if (state.cancelled) throw new Error("This exact inline artifact was cancelled.");
        if (state.ready) return true;
        await new Promise<void>((resolve, reject) => {
          const stop = (): void => { clearTimeout(timer); signal?.removeEventListener("abort", stop); reject(new Error("Inline artifact wait cancelled.")); };
          const timer = setTimeout(() => { signal?.removeEventListener("abort", stop); resolve(); }, 100);
          signal?.addEventListener("abort", stop, { once: true });
          if (signal?.aborted) stop();
        });
      }
    } catch (error) {
      if (signal?.aborted && owned?.streamId && typeof params.brainId === "string") {
        await this.cancelInlineGeneration(params.brainId, owned.streamId, String(params.neuralActionId)).catch(() => undefined);
      }
      throw error;
    }
  }

  /**
   * Ask one optional background operation to stop without waiting behind the
   * worker's serial RPC queue. The existing cooperative cancellation path
   * preserves its last atomic checkpoint and pending replay record.
   */
  cancelBackgroundRequest(brainId: string, method: string): boolean {
    let cancelled = false;
    for (const reservation of this.requestReservations) {
      if (
        reservation.priority !== "background" ||
        reservation.brainId !== brainId ||
        reservation.method !== method ||
        reservation.cancelled
      ) continue;
      reservation.cancelled = true;
      reservation.cancellationPhase = reservation.dispatched ? "running" : "queued";
      reservation.controller.abort(
        new Error(`Worker request "${method}" was cancelled by learning pause.`)
      );
      cancelled = true;
    }
    return cancelled;
  }

  /**
   * Give interactive/build work precedence over optional prompt-free activity.
   *
   * Background requests never accumulate behind other reservations. If one is
   * already restoring a cold checkpoint, terminate that supervised process;
   * the brain's last atomic checkpoint remains authoritative and the
   * foreground request starts on a clean replacement worker.
   */
  private async claimForegroundWorker(): Promise<void> {
    const background = [...this.requestReservations].filter(
      (reservation) => reservation.priority === "background"
    );
    for (const reservation of background) {
      reservation.cancelled = true;
      reservation.cancellationPhase = reservation.dispatched
        ? "running"
        : "queued";
      if (!reservation.controller.signal.aborted) {
        reservation.controller.abort(
          new Error(`Worker request "${reservation.method}" was cancelled.`)
        );
      }
    }
    if (this.activeRequest?.priority !== "background") return;
    if (COOPERATIVE_CANCEL_METHODS.has(this.activeRequest.method)) {
      await this.activeRequest.settled;
      return;
    }
    if (!this.backgroundPreemption) {
      const operation = this.interruptAndRestart()
        .then(() => undefined)
        .finally(() => {
          if (this.backgroundPreemption === operation) {
            this.backgroundPreemption = undefined;
          }
        });
      this.backgroundPreemption = operation;
    }
    await this.backgroundPreemption;
  }

  /**
   * Run a request while receiving only its correlated worker notifications.
   *
   * Worker contract: the request contains `streamId`; notifications must echo
   * it and use a strictly increasing non-negative integer `sequence`. Duplicate
   * or out-of-order notifications are ignored before reaching application code.
   */
  async requestStream<T>(
    method: string,
    params: Record<string, unknown>,
    onEvent: (event: EngineEvent) => void,
    timeoutMs = 120_000,
    signal?: AbortSignal,
    requestedStreamId?: string,
    priority: EngineRequestPriority = "foreground",
    context?: EngineRequestContext
  ): Promise<T> {
    const streamId = requestedStreamId?.trim() || randomUUID();
    const brainId = typeof params.brainId === "string" ? params.brainId : undefined;
    let lastSequence = -1;
    let callbackFailure: unknown;
    let callbackFailed = false;
    const listener = (event: EngineEvent): void => {
      if (event.streamId !== streamId || brainId !== undefined && event.brainId !== brainId) return;
      if (
        typeof event.sequence !== "number" ||
        !Number.isSafeInteger(event.sequence) ||
        event.sequence < 0 ||
        event.sequence <= lastSequence
      ) {
        return;
      }
      lastSequence = event.sequence;
      if (callbackFailed) return;
      try { onEvent(event); } catch (error) { callbackFailure = error; callbackFailed = true; }
    };
    this.on("event", listener);
    try {
      const result = await this.request<T>(
        method,
        { ...params, streamId },
        timeoutMs,
        signal,
        priority,
        context
      );
      // A malformed observer event must reject this owned operation, never
      // escape EventEmitter as a process-wide uncaught exception. Finish the
      // existing native receipt first so the caller can reconcile it without
      // killing a worker or replaying its already completed effects.
      if (callbackFailed) throw callbackFailure;
      return result;
    } finally {
      this.off("event", listener);
    }
  }

  async tryRequest<T>(
    method: string,
    params: Record<string, unknown> = {},
    timeoutMs = 120_000,
    signal?: AbortSignal,
    priority: EngineRequestPriority = "foreground",
    context?: EngineRequestContext
  ): Promise<T | undefined> {
    try {
      return await this.request<T>(method, params, timeoutMs, signal, priority, context);
    } catch (error) {
      this.lastError = messageFromError(error);
      return undefined;
    }
  }

  async tryRequestStream<T>(
    method: string,
    params: Record<string, unknown>,
    onEvent: (event: EngineEvent) => void,
    timeoutMs = 120_000,
    signal?: AbortSignal,
    streamId?: string,
    priority: EngineRequestPriority = "foreground",
    context?: EngineRequestContext
  ): Promise<T | undefined> {
    try {
      return await this.requestStream<T>(
        method,
        params,
        onEvent,
        timeoutMs,
        signal,
        streamId,
        priority,
        context
      );
    } catch (error) {
      this.lastError = messageFromError(error);
      return undefined;
    }
  }

  async health(): Promise<EngineHealth> {
    let restarting = Boolean(this.terminating || this.starting);
    const started = await this.start();
    if (started) {
      const result = await this.tryRequest<Record<string, unknown>>("health", {}, 10_000);
      if (result) {
        return {
          ready: true,
          worker: "python",
          protocolVersion: PROTOCOL_VERSION,
          detail:
            typeof result.detail === "string"
              ? result.detail
              : "Python neural worker is ready.",
          pid: this.child?.pid
        };
      }
      restarting = true;
    }
    return {
      ready: false,
      worker: "unavailable",
      protocolVersion: PROTOCOL_VERSION,
      // Full process stderr and traceback remain in lastError/recentStderr and
      // the emitted exit diagnostic. Health is a compact user-facing surface,
      // not a log transport or multi-line progress report.
      detail:
        restarting || /(?:stopped|exited|cancelled|killed)/i.test(this.lastError)
          ? "Neural engine restarting…"
          : "Neural engine unavailable."
    };
  }

  async stop(): Promise<void> {
    if (HOST_COMPUTE_SCOPE.getStore() === this) throw new Error("The local teacher must release its host-compute owner before native shutdown.");
    if (this.stopping) return;
    this.stopping = true;
    try {
      const hosts = [...this.hostComputeReservations];
      for (const reservation of hosts) reservation.controller.abort(new Error("Studio is shutting down; stop the owned local teacher child before releasing compute."));
      await Promise.allSettled(hosts.map(entry => entry.settled));
      const inspection = this.inspectionWorker;
      this.inspectionWorker = undefined;
      if (inspection) await inspection.stop();
      if (this.child && !this.child.killed && this.child.exitCode === null) {
        await this.rawRequest("shutdown", {}, 1_500).catch(() => undefined);
      }
      await this.terminateChild();
    } finally {
      this.stopping = false;
    }
  }

  async interruptAndRestart(): Promise<boolean> {
    if (this.stopping) return false;
    this.stopping = true;
    try {
      // Long neural methods occupy the worker's serial JSON-RPC loop, so a
      // cancel RPC cannot be observed until the work is already finished.
      // Terminating the supervised process discards the unpromoted in-memory
      // candidate while the last atomic safe-tensor checkpoint stays intact.
      await this.terminateChild();
    } finally {
      this.stopping = false;
    }
    return this.start();
  }

  private handleClose(
    lifecycle: ChildLifecycle,
    code: number | null,
    signal: NodeJS.Signals | null
  ): void {
    const { child } = lifecycle;
    if (lifecycle.settled) return;
    lifecycle.settled = true;
    if (lifecycle.forcedClose) clearTimeout(lifecycle.forcedClose);
    child.stdout.removeListener("data", lifecycle.stdoutListener);
    child.stderr.removeListener("data", lifecycle.stderrListener);
    const exitDetail = `Worker exited with code ${String(code)} and signal ${String(signal)}.`;
    const detail = lifecycle.processError
      ? `${lifecycle.processError}\n${exitDetail}`
      : exitDetail;
    const stderr = lifecycle.stderr.filter(Boolean).join("\n").slice(-8_000);
    const diagnostic = stderr ? `${detail}\nWorker stderr:\n${stderr}` : detail;
    if (stderr || lifecycle.processError) {
      // Internal-only diagnostic channel. The bootstrap logger records this;
      // health/UI responses deliberately never include it.
      this.emit("diagnostic", diagnostic);
    }
    try {
      if (typeof child.exitCode === "number" || typeof child.signalCode === "string") {
        this.emit("worker-closed", { pid: child.pid });
      }
      if (this.child !== child) return;
      this.lastError = diagnostic;
      this.rejectPendingForChild(child, new Error(diagnostic));
      this.child = undefined;
      if (this.lifecycle === lifecycle) this.lifecycle = undefined;
      if (!this.stopping) this.emit("exit", diagnostic);
    } finally {
      lifecycle.resolveClosed();
    }
  }

  private terminateChild(force = false): Promise<void> {
    if (this.terminating) {
      if (
        force &&
        this.terminatingChild &&
        this.terminatingChild.exitCode === null
      ) {
        this.terminatingChild.kill("SIGKILL");
      }
      return this.terminating;
    }
    const child = this.child;
    if (!child) return Promise.resolve();
    const lifecycle = this.lifecycle?.child === child ? this.lifecycle : undefined;
    this.child = undefined;
    this.terminatingChild = child;
    this.rejectPendingForChild(child, new Error("Python worker was stopped."));
    const terminate = async (): Promise<void> => {
      if (child.exitCode === null && (force || !child.killed)) {
        if (force) child.kill("SIGKILL");
        else child.kill();
      }
      await new Promise<void>((resolveClose, rejectClose) => {
        let finished = false;
        let forceTimeout: NodeJS.Timeout | undefined;
        let gracefulTimeout: NodeJS.Timeout;
        const finish = (): void => {
          if (finished) return;
          finished = true;
          clearTimeout(gracefulTimeout);
          if (forceTimeout) clearTimeout(forceTimeout);
          if (lifecycle) {
            this.handleClose(lifecycle, child.exitCode, child.signalCode);
          } else {
            child.removeListener("close", finish);
          }
          resolveClose();
        };
        gracefulTimeout = setTimeout(() => {
          if (child.exitCode === null) child.kill("SIGKILL");
          forceTimeout = setTimeout(() => {
            if (child.exitCode === null && child.signalCode === null) {
              this.unacknowledgedChild = child;
              child.removeListener("close", finish);
              rejectClose(
                new Error(
                  `Python worker PID ${String(child.pid ?? "unknown")} did not acknowledge termination.`
                )
              );
              return;
            }
            finish();
          }, 1_000);
        }, force ? 250 : 2_000);
        if (lifecycle) {
          void lifecycle.closed.then(finish);
        } else {
          child.once("close", finish);
        }
      });
    };
    const operation = terminate().finally(() => {
      if (this.lifecycle === lifecycle) this.lifecycle = undefined;
      if (this.terminatingChild === child) this.terminatingChild = undefined;
      if (this.terminating === operation) this.terminating = undefined;
    });
    this.terminating = operation;
    return operation;
  }
}
/** Trusted launch metadata only; user/renderer environment cannot pick its owner. */
export function managedWorkerMemoryEnvironment(
  environment: NodeJS.ProcessEnv,
  sharedResourceLedgerPath?: string
): NodeJS.ProcessEnv {
  const result: NodeJS.ProcessEnv = { ...environment, OMNI_MEMORY_OWNER_PID: String(process.pid) };
  // An inherited shell/renderer variable cannot redirect app quota accounting.
  delete result.OMNI_SHARED_RESOURCE_LEDGER;
  if (sharedResourceLedgerPath !== undefined) {
    if (!isAbsolute(sharedResourceLedgerPath)) throw new Error("Shared resource ledger must be an absolute trusted path.");
    result.OMNI_SHARED_RESOURCE_LEDGER = resolve(sharedResourceLedgerPath);
  }
  return result;
}
