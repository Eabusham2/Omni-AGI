import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import type { ProcessMemoryNativeRequest, ProcessMemoryReady, ProcessMemoryReceipt } from "../shared/processMemory";
import { PROCESS_MEMORY_FORMAT } from "../shared/processMemory";
import { memoryIdentity, memoryObject, memoryReceipt, processMemoryCanonical, processMemoryHash } from "./processMemoryFormat";

export interface ProcessMemoryLaunch { command: string; arguments: string[] }
const helperPids = new Set<number>();
/** Current app-owned helper children only; no OS process enumeration. */
export const processMemoryHelperPids = (): readonly number[] => [...helperPids];
export interface ProcessMemoryBridgeOptions {
  appPath: string;
  resourcesPath?: string;
  packagedRequired?: boolean;
  pythonCommand?: string;
}
export interface ProcessMemoryBridgeContext {
  signal: AbortSignal;
  /** Latest permission/settings/resource checks. No commit is sent until this settles. */
  beforeCommit(ready: ProcessMemoryReady, helperPid: number): Promise<void>;
}
export class ProcessMemoryOperationError extends Error {
  constructor(message: string, readonly receipt: ProcessMemoryReceipt) { super(message); this.name = "ProcessMemoryOperationError"; }
}
export function uncertainMemoryReceipt(request: ProcessMemoryNativeRequest, committed: boolean, message: string): ProcessMemoryReceipt {
  return { format: PROCESS_MEMORY_FORMAT, operationId: request.operationId, operation: request.operation,
    requestSha256: processMemoryHash(processMemoryCanonical(request)), state: committed && request.operation === "write" ? "write-uncertain" : "cancelled",
    ...(request.expectedIdentity ? { identity: request.expectedIdentity } : {}), requestedBytes: request.length ?? 0,
    transferredBytes: committed && request.operation === "write" ? null : 0, writeMayHaveOccurred: committed && request.operation === "write",
    retrySafe: !(committed && request.operation === "write"), compareAtomic: false, error: message.slice(0, 4096) };
}
/** No neural EngineSupervisor start, torch import, dependency install or shell. */
export function processMemoryLaunch(options: ProcessMemoryBridgeOptions): ProcessMemoryLaunch {
  const binary = options.resourcesPath ? join(options.resourcesPath, "engine-runtime", process.platform === "win32" ? "omni-engine.exe" : "omni-engine") : undefined;
  if (binary && existsSync(binary)) return { command: binary, arguments: ["--process-memory-helper"] };
  if (options.packagedRequired) throw new Error("The packaged process-memory adapter is unavailable. No external interpreter fallback was used.");
  const worker = [options.resourcesPath ? join(options.resourcesPath, "engine", "process_memory_helper.py") : undefined,
    join(options.appPath, "engine", "process_memory_helper.py")].find((file): file is string => !!file && existsSync(file));
  if (!worker) throw new Error("The standard-library process-memory helper is unavailable.");
  const embedded = options.resourcesPath ? join(options.resourcesPath, "python", process.platform === "win32" ? "python.exe" : "bin/python3") : undefined;
  const command = options.pythonCommand?.trim() || process.env.OMNI_PYTHON?.trim() || process.env.OMNI_AGI_PYTHON?.trim() ||
    (embedded && existsSync(embedded) ? embedded : process.platform === "win32" ? "python" : "python3");
  return { command, arguments: ["-u", worker] };
}

/** The only killed process on timeout is our own helper, never the selected target. */
export function runProcessMemoryHelper(launch: ProcessMemoryLaunch, request: ProcessMemoryNativeRequest,
  context: ProcessMemoryBridgeContext, timeoutMs = 15_000): Promise<ProcessMemoryReceipt> {
  context.signal.throwIfAborted();
  const requestSha256 = processMemoryHash(processMemoryCanonical(request));
  return new Promise((resolve, reject) => {
    const child = spawn(launch.command, launch.arguments, { stdio: ["pipe", "pipe", "pipe"], windowsHide: true });
    if (child.pid) helperPids.add(child.pid);
    child.once("close", () => { if (child.pid) helperPids.delete(child.pid); });
    let settled = false, committed = false, readySeen = false, output = "", stderr = "", parsedBytes = 0;
    const finish = (receipt?: ProcessMemoryReceipt, error?: unknown): void => {
      if (settled) return; settled = true; clearTimeout(timer); context.signal.removeEventListener("abort", abort);
      child.stdin?.end();
      if (!child.killed && child.exitCode === null) child.kill();
      if (error) reject(error); else resolve(receipt!);
    };
    const fail = (error: unknown): void => {
      const message = error instanceof Error ? error.message : String(error);
      finish(undefined, new ProcessMemoryOperationError(message, uncertainMemoryReceipt(request, committed, message)));
    };
    const abort = (): void => {
      // After a write commit, allow the bounded adapter to return the actual
      // result even if Stop/revocation arrives. The target is never signalled.
      if (!committed || request.operation !== "write") fail(context.signal.reason ?? new Error("Process-memory operation stopped before commit."));
    };
    const timer = setTimeout(() => fail(new Error(committed && request.operation === "write"
      ? "Memory write receipt timed out; effects are uncertain and must not be automatically retried."
      : "Process-memory helper timed out before a confirmed result.")), Math.max(100, Math.min(30_000, timeoutMs)));
    context.signal.addEventListener("abort", abort, { once: true });
    child.on("error", fail);
    child.stdin?.on("error", fail);
    child.stderr?.on("data", (bytes: Buffer) => { stderr = (stderr + bytes.toString("utf8")).slice(-4096); });
    child.stdout?.setEncoding("utf8");
    child.stdout?.on("data", (chunk: string) => {
      if (settled) return;
      parsedBytes += Buffer.byteLength(chunk);
      if (parsedBytes > 2 * 1024 * 1024) { fail(new Error("Process-memory adapter exceeded its bounded output envelope.")); return; }
      output += chunk;
      for (;;) {
        const end = output.indexOf("\n"); if (end < 0) return;
        const line = output.slice(0, end); output = output.slice(end + 1);
        let row: Record<string, unknown>;
        try { row = memoryObject(JSON.parse(line)); } catch (error) { fail(error); return; }
        if (row.type === "ready") {
          if (readySeen || row.format !== PROCESS_MEMORY_FORMAT || row.operationId !== request.operationId || row.requestSha256 !== requestSha256 ||
              typeof row.nonce !== "string" || !/^[a-f0-9]{48}$/.test(row.nonce) || !child.pid) { fail(new Error("Native memory preparation does not match its exact request.")); return; }
          let identity;
          try { identity = memoryIdentity(row.identity); if (identity.pid !== request.pid || (request.expectedIdentity && processMemoryCanonical(identity) !== processMemoryCanonical(request.expectedIdentity))) throw new Error("Native memory preparation changed its process identity."); }
          catch (error) { fail(error); return; }
          readySeen = true;
          const ready: ProcessMemoryReady = { format: PROCESS_MEMORY_FORMAT, type: "ready", operationId: request.operationId, requestSha256, nonce: row.nonce, identity };
          void context.beforeCommit(ready, child.pid).then(() => {
            if (settled) return; context.signal.throwIfAborted();
            // Mark uncertainty before handing bytes to the pipe. An EPIPE or
            // lost ACK cannot prove that a target write never happened.
            committed = true;
            child.stdin?.end(`${JSON.stringify({ commit: ready.nonce, requestSha256 })}\n`);
          }).catch(fail);
        } else if (row.type === "protocol-error") {
          fail(new Error(typeof row.error === "string" ? row.error : "Native memory protocol was rejected.")); return;
        } else {
          try {
            const receipt = memoryReceipt(row, request);
            if (receipt.state === "complete" && (!readySeen || !committed)) throw new Error("The adapter reported success without host authorization.");
            finish(receipt);
          } catch (error) { fail(error); }
        }
      }
    });
    child.on("close", (code) => { if (!settled) fail(new Error(`Process-memory helper exited without its complete receipt (${code ?? "unknown"}). ${stderr}`)); });
    child.stdin?.write(`${processMemoryCanonical(request)}\n`);
  });
}
