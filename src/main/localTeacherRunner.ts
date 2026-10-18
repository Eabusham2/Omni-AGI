import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createHash } from "node:crypto";
import { lstat, open } from "node:fs/promises";
import { join } from "node:path";
import type { Readable } from "node:stream";
import type { LocalTeacherGeneration, LocalTeacherRunnerKind } from "../shared/localTeacher";
import { localTeacherDirectory } from "./localTeacherArtifacts";
import { localTeacherCanonical, localTeacherHash } from "./localTeacherFormat";
const ownedTeacherPids = new Set<number>();
export const localTeacherChildPids = (): number[] => [...ownedTeacherPids];

export const LOCAL_TEACHER_ENV_POLICY_SHA256 = localTeacherHash(localTeacherCanonical({ version: 1,
  allowedInheritedKeys: ["PATH", "SystemRoot", "SYSTEMROOT", "WINDIR", "LANG", "LC_ALL", "COMSPEC", "PATHEXT"],
  privateHomeAndTemp: true, offline: true, inheritedProviderAndLlamaDefaults: false }));
/** Unlike runProcess, this never merges all of process.env. */
export function cleanLocalTeacherEnvironment(privateDirectory: string): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const key of ["PATH", "SystemRoot", "SYSTEMROOT", "WINDIR", "LANG", "LC_ALL", "COMSPEC", "PATHEXT"]) if (typeof process.env[key] === "string") env[key] = process.env[key];
  Object.assign(env, { HOME: privateDirectory, USERPROFILE: privateDirectory, TMPDIR: privateDirectory, TEMP: privateDirectory,
    TMP: privateDirectory, XDG_CACHE_HOME: privateDirectory, APPDATA: privateDirectory, LOCALAPPDATA: privateDirectory, LLAMA_ARG_OFFLINE: "1" });
  return env;
}
const commonRequired = ["--offline", "--model", "--ctx-size", "--threads", "--gpu-layers", "--no-op-offload", "--no-kv-offload",
  "--cache-type-k", "--cache-type-v", "--fit", "--no-warmup", "--no-context-shift", "--verbosity"];
export function localTeacherRequiredFlags(kind: LocalTeacherRunnerKind): string[] {
  return [...commonRequired, ...(kind === "llama-server" ? ["--host", "--port", "--api-key-file", "--parallel", "--no-agent", "--no-webui", "--no-slots", "--no-cache-prompt"]
    : ["--no-conversation", "--no-display-prompt", "--simple-io", "--no-escape", "--file", "--predict", "--temp", "--seed", "--color"])];
}
export function localTeacherProbeFlags(help: string, kind: LocalTeacherRunnerKind): { supported: boolean; missingFlags: string[] } {
  const missingFlags = localTeacherRequiredFlags(kind).filter(flag => !new RegExp(`(?:^|[\\s,])${flag}(?=[\\s,=]|$)`, "m").test(help));
  if (kind === "llama-server" && process.platform !== "win32" && !/UNIX socket|socket paths|\.sock/.test(help)) missingFlags.push("documented-private-unix-socket-host");
  return { supported: !missingFlags.length, missingFlags };
}
export function localTeacherBaseArguments(modelPath: string, generation: LocalTeacherGeneration): string[] {
  return ["--offline", "--model", modelPath, "--ctx-size", String(generation.contextTokens), "--threads", String(generation.threads),
    "--gpu-layers", "0", "--no-op-offload", "--no-kv-offload", "--cache-type-k", "f16", "--cache-type-v", "f16",
    "--fit", "off", "--no-warmup", "--no-context-shift"];
}
export interface LocalTeacherCapturedFile { relativePath: string; sha256: string; bytes: number; coverage: "complete" | "prefix-only" }
export interface LocalTeacherChildOptions {
  root: string;
  relativeDirectory: string;
  executable: string;
  arguments: string[];
  signal: AbortSignal;
  /** null streams all operational logs to disk under current write admission. */
  maxLogBytes: number | null;
  reserveLogWrite?(bytes: number): Promise<void>;
  timeoutMs: number;
  requireCurrent(): Promise<void>;
  checkResources(childPids: readonly number[]): Promise<void>;
  onLog?(text: string): void;
}
export interface LocalTeacherChild {
  child: ChildProcessWithoutNullStreams;
  stop(reason?: Error): Promise<void>;
  closed: Promise<{ exitCode: number | null; stdout: LocalTeacherCapturedFile; stderr: LocalTeacherCapturedFile; elapsedNanoseconds: string; error?: string }>;
  alive(): boolean;
}

/** Completion is acknowledged only after the exact child and both pipes close. */
export async function startLocalTeacherChild(options: LocalTeacherChildOptions): Promise<LocalTeacherChild> {
  if (options.maxLogBytes === null && !options.reserveLogWrite) throw new Error("Uncapped operational logs require current bounded disk-write admission.");
  options.signal.throwIfAborted(); await options.requireCurrent(); await options.checkResources([]);
  const directory = await localTeacherDirectory(options.root, options.relativeDirectory), home = join(directory, "home");
  await localTeacherDirectory(options.root, `${options.relativeDirectory}/home`);
  const stdoutFile = await open(join(directory, "process.stdout.bin"), "wx", 0o600);
  let stderrFile: Awaited<ReturnType<typeof open>>;
  try { stderrFile = await open(join(directory, "process.stderr.bin"), "wx", 0o600); }
  catch (error) { await stdoutFile.close(); throw error; }
  let child: ChildProcessWithoutNullStreams, started: bigint;
  try {
    // Resource admission and owned-file preparation awaited above. Revalidate
    // at the actual effect boundary, including a Stop arriving during policy.
    options.signal.throwIfAborted(); await options.requireCurrent(); options.signal.throwIfAborted();
    started = process.hrtime.bigint();
    child = spawn(options.executable, options.arguments, { cwd: directory,
      env: cleanLocalTeacherEnvironment(home), shell: false, windowsHide: true, detached: process.platform !== "win32", stdio: ["pipe", "pipe", "pipe"] });
  } catch (error) { await Promise.allSettled([stdoutFile.close(), stderrFile.close()]); throw error; }
  child.stdin.end();
  if (child.pid) ownedTeacherPids.add(child.pid);
  let reason: Error | undefined, forceTimer: NodeJS.Timeout | undefined, deadline: NodeJS.Timeout | undefined, checking = false;
  let exitResolve!: (code: number | null) => void;
  const exit = new Promise<number | null>(yes => { exitResolve = yes; });
  const alive = () => child.pid !== undefined && child.exitCode === null && child.signalCode === null && !child.killed;
  const terminate = (error?: Error): void => {
    reason ??= error;
    if (child.exitCode !== null || child.signalCode !== null || !child.pid) return;
    try { if (process.platform === "win32") child.kill(); else process.kill(-child.pid, "SIGTERM"); } catch (error) { if ((error as NodeJS.ErrnoException).code !== "ESRCH") reason ??= error as Error; }
    forceTimer ??= setTimeout(() => { if (child.exitCode !== null || child.signalCode !== null || !child.pid) return;
      try { if (process.platform === "win32") child.kill("SIGKILL"); else process.kill(-child.pid, "SIGKILL"); } catch (error) { if ((error as NodeJS.ErrnoException).code !== "ESRCH") reason ??= error as Error; }
    }, 500);
  };
  child.once("error", error => { reason ??= error; }); child.once("close", code => exitResolve(code));
  const capture = async (stream: Readable, file: Awaited<ReturnType<typeof open>>, name: string): Promise<LocalTeacherCapturedFile> => {
    const digest = createHash("sha256"), decoder = new TextDecoder(); let bytes = 0, truncated = false;
    try {
      for await (const value of stream) {
        const data = Buffer.isBuffer(value) ? value : Buffer.from(value as Uint8Array);
        const available = options.maxLogBytes === null ? data.length : Math.max(0, options.maxLogBytes - bytes), keep = data.subarray(0, available);
        if (keep.length) { await options.reserveLogWrite?.(keep.length); await file.writeFile(keep); digest.update(keep); bytes += keep.length; }
        if (keep.length !== data.length) { truncated = true; terminate(new Error("Teacher log/output exceeded its explicit byte envelope; the prefix is retained but will not be learned.")); }
        options.onLog?.(decoder.decode(data, { stream: true }));
      }
      options.onLog?.(decoder.decode()); await file.sync();
    } finally { await file.close(); }
    return { relativePath: `${options.relativeDirectory}/${name}`, sha256: digest.digest("hex"), bytes, coverage: truncated ? "prefix-only" : "complete" };
  };
  const captureFailure = (error: unknown): never => { terminate(error instanceof Error ? error : new Error(String(error))); throw error; };
  const stdout = capture(child.stdout, stdoutFile, "process.stdout.bin").catch(captureFailure), stderr = capture(child.stderr, stderrFile, "process.stderr.bin").catch(captureFailure);
  const abort = () => terminate(options.signal.reason instanceof Error ? options.signal.reason : new Error("Local teacher paused/stopped."));
  options.signal.addEventListener("abort", abort, { once: true }); if (options.signal.aborted) abort();
  const monitor = setInterval(() => { if (checking || !alive()) return; checking = true;
    void options.requireCurrent().then(() => options.checkResources(child.pid ? [child.pid] : [])).catch(error => terminate(error instanceof Error ? error : new Error(String(error)))).finally(() => { checking = false; });
  }, 250);
  if (options.timeoutMs > 0) deadline = setTimeout(() => terminate(new Error("The explicit local-teacher wall-clock budget elapsed; incomplete output is retained, not learned.")), options.timeoutMs);
  const closed = (async () => {
    try { const [result, out, err] = await Promise.allSettled([exit, stdout, stderr]);
      if (result.status !== "fulfilled" || out.status !== "fulfilled" || err.status !== "fulfilled") throw reason ?? new Error("Teacher capture could not be sealed after the exact child closed.");
      return { exitCode: result.value, stdout: out.value, stderr: err.value, elapsedNanoseconds: (process.hrtime.bigint() - started).toString(), ...(reason ? { error: reason.message } : {}) };
    } finally { if (child.pid) ownedTeacherPids.delete(child.pid); clearInterval(monitor); if (forceTimer) clearTimeout(forceTimer); if (deadline) clearTimeout(deadline); options.signal.removeEventListener("abort", abort); }
  })();
  return { child, alive, closed, stop: async error => { terminate(error); await closed; } };
}
export async function readBoundedTeacherText(path: string, maximum: number): Promise<string> {
  const item = await lstat(path); if (!item.isFile() || item.isSymbolicLink() || item.size > maximum) throw new Error("Teacher protocol metadata exceeds its admitted envelope.");
  const file = await open(path, "r"); try { return new TextDecoder("utf-8", { fatal: true }).decode(await file.readFile()); } finally { await file.close(); }
}
