import { EventEmitter } from "node:events";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { randomUUID } from "node:crypto";
import type { EngineHealth } from "../shared/types";

const PROTOCOL_VERSION = 1;
const MAX_PROTOCOL_LINE = 32 * 1024 * 1024;

interface PendingRequest {
  resolve(value: unknown): void;
  reject(error: Error): void;
  timeout: NodeJS.Timeout;
  cleanup(): void;
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

export interface EngineEvent {
  type: string;
  brainId?: string;
  jobId?: string;
  progress?: number;
  message?: string;
  data?: unknown;
}

export interface EngineSupervisorOptions {
  appPath: string;
  resourcesPath?: string;
  workerPath?: string;
  pythonCommand?: string;
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

function messageFromError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export class EngineSupervisor extends EventEmitter {
  private readonly options: EngineSupervisorOptions;
  private child?: ChildProcessWithoutNullStreams;
  private pending = new Map<string, PendingRequest>();
  private stdoutBuffer = "";
  private starting?: Promise<boolean>;
  private terminating?: Promise<void>;
  private stopping = false;
  private lastError = "Python worker has not been started.";
  private recentStderr: string[] = [];

  constructor(options: EngineSupervisorOptions) {
    super();
    this.options = options;
  }

  get pid(): number | undefined {
    return this.child?.pid;
  }

  async start(): Promise<boolean> {
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
      ? join(this.options.resourcesPath, "engine-runtime", "omni-engine.exe")
      : undefined;
    const packagedRequired = process.env.OMNI_PACKAGED_ENGINE_REQUIRED === "1";
    const candidates: Array<{ candidate: PythonCandidate; direct: boolean }> = [];
    // A packaged build must exercise its self-contained, non-pickle worker first.
    // Source-Python candidates remain a development/recovery fallback.
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
          ? "The required packaged OmniCortex engine was not found; the deterministic local fallback is active."
          : "No packaged engine or engine/worker.py source was found; the deterministic local fallback is active.";
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
        ...process.env,
        PYTHONUNBUFFERED: "1",
        OMNI_PROTOCOL_VERSION: String(PROTOCOL_VERSION)
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

    this.child = child;
    this.stdoutBuffer = "";
    this.recentStderr = [];
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => this.consumeStdout(chunk));
    child.stderr.on("data", (chunk: string) => {
      this.recentStderr.push(chunk.trim().slice(-4_000));
      this.recentStderr = this.recentStderr.filter(Boolean).slice(-12);
    });
    child.once("error", (error) =>
      this.handleExit(child, `Worker process error: ${error.message}`)
    );
    child.once("exit", (code, signal) => {
      this.handleExit(
        child,
        `Worker exited with code ${String(code)} and signal ${String(signal)}.`
      );
    });

    try {
      await this.rawRequest("health", {}, 30_000);
      this.lastError = "";
    } catch (error) {
      throw new Error(`Worker health handshake failed: ${messageFromError(error)}`);
    }
  }

  private consumeStdout(chunk: string): void {
    this.stdoutBuffer += chunk;
    if (this.stdoutBuffer.length > MAX_PROTOCOL_LINE && !this.stdoutBuffer.includes("\n")) {
      this.lastError = "Worker emitted an oversized protocol line.";
      void this.terminateChild();
      return;
    }
    let newline = this.stdoutBuffer.indexOf("\n");
    while (newline >= 0) {
      const line = this.stdoutBuffer.slice(0, newline).trim();
      this.stdoutBuffer = this.stdoutBuffer.slice(newline + 1);
      if (line) this.consumeLine(line);
      newline = this.stdoutBuffer.indexOf("\n");
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
        pending.reject(
          new Error(
            typeof record.error.message === "string"
              ? record.error.message
              : "The Python worker returned an error."
          )
        );
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
      this.emit("event", event);
    }
  }

  private rawRequest(
    method: string,
    params: Record<string, unknown>,
    timeoutMs: number,
    signal?: AbortSignal
  ): Promise<unknown> {
    const child = this.child;
    if (!child || child.killed || child.exitCode !== null || !child.stdin.writable) {
      return Promise.reject(new Error("Python worker is unavailable."));
    }
    if (signal?.aborted) {
      return Promise.reject(new Error(`Worker request "${method}" was cancelled.`));
    }
    const id = randomUUID();
    return new Promise((resolveRequest, rejectRequest) => {
      const timeout = setTimeout(() => {
        const pending = this.pending.get(id);
        if (!pending) return;
        this.pending.delete(id);
        pending.cleanup();
        pending.reject(new Error(`Worker request "${method}" timed out.`));
        void this.terminateChild();
      }, timeoutMs);
      const abort = (): void => {
        const pending = this.pending.get(id);
        if (!pending) return;
        this.pending.delete(id);
        pending.cleanup();
        pending.reject(new Error(`Worker request "${method}" was cancelled.`));
        void this.terminateChild();
      };
      const cleanup = (): void => {
        clearTimeout(timeout);
        signal?.removeEventListener("abort", abort);
      };
      this.pending.set(id, {
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

  async request<T>(
    method: string,
    params: Record<string, unknown> = {},
    timeoutMs = 120_000,
    signal?: AbortSignal
  ): Promise<T> {
    if (signal?.aborted) throw new Error(`Worker request "${method}" was cancelled.`);
    if (!(await this.start())) throw new Error(this.lastError);
    if (signal?.aborted) throw new Error(`Worker request "${method}" was cancelled.`);
    return (await this.rawRequest(method, params, timeoutMs, signal)) as T;
  }

  async tryRequest<T>(
    method: string,
    params: Record<string, unknown> = {},
    timeoutMs = 120_000,
    signal?: AbortSignal
  ): Promise<T | undefined> {
    try {
      return await this.request<T>(method, params, timeoutMs, signal);
    } catch (error) {
      this.lastError = messageFromError(error);
      return undefined;
    }
  }

  async health(): Promise<EngineHealth> {
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
    }
    const stderr = this.recentStderr.at(-1);
    return {
      ready: true,
      worker: "fallback",
      protocolVersion: PROTOCOL_VERSION,
      detail: [this.lastError, stderr, "The TypeScript adaptive core remains available."]
        .filter(Boolean)
        .join(" ")
    };
  }

  async stop(): Promise<void> {
    if (this.stopping) return;
    this.stopping = true;
    try {
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

  private handleExit(child: ChildProcessWithoutNullStreams, detail: string): void {
    if (this.child !== child) return;
    this.lastError = detail;
    const error = new Error(detail);
    for (const request of this.pending.values()) {
      request.cleanup();
      request.reject(error);
    }
    this.pending.clear();
    this.child = undefined;
    if (!this.stopping) this.emit("exit", detail);
  }

  private terminateChild(): Promise<void> {
    if (this.terminating) return this.terminating;
    const child = this.child;
    if (!child) return Promise.resolve();
    this.child = undefined;
    if (this.pending.size > 0) {
      const error = new Error("Python worker was stopped.");
      for (const request of this.pending.values()) {
        request.cleanup();
        request.reject(error);
      }
      this.pending.clear();
    }
    const terminate = async (): Promise<void> => {
      if (child.exitCode !== null) return;
      child.kill();
      await new Promise<void>((resolveExit) => {
        let finished = false;
        let forceTimeout: NodeJS.Timeout | undefined;
        let gracefulTimeout: NodeJS.Timeout;
        const finish = (): void => {
          if (finished) return;
          finished = true;
          clearTimeout(gracefulTimeout);
          if (forceTimeout) clearTimeout(forceTimeout);
          child.removeListener("exit", finish);
          resolveExit();
        };
        gracefulTimeout = setTimeout(() => {
          if (child.exitCode === null) child.kill("SIGKILL");
          forceTimeout = setTimeout(finish, 1_000);
        }, 2_000);
        child.once("exit", finish);
        if (child.exitCode !== null) finish();
      });
    };
    const operation = terminate().finally(() => {
      if (this.terminating === operation) this.terminating = undefined;
    });
    this.terminating = operation;
    return operation;
  }
}
