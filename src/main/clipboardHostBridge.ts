import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { access, lstat } from "node:fs/promises";
import { spawn, type ChildProcess } from "node:child_process";
import { Readable } from "node:stream";
import { CLIPBOARD_READ_PAGE_BYTES, type ClipboardCapabilities } from "../shared/clipboard";
import { clipboardHash } from "./clipboardFormat";

export interface HostClipboardCursor { afterBytes: number; prefixSha256: string }
export interface HostClipboardPage { adapter: string; text: string; bytes: number; chunkSha256: string; complete: boolean; capturedAt: string; next?: HostClipboardCursor }
export interface HostClipboardHolderLease { id: string; pid: number; adapter: string; startedAt: string; exit: Promise<void>; stop(): Promise<void> }
export interface HostClipboardMutation { adapter: string; completedAt: string; holder?: HostClipboardHolderLease }
export interface HostClipboardContext {
  signal: AbortSignal;
  onSpawn(pid: number): void;
  beforeDispatch?(): Promise<void>;
  onRetainedHolder?(holder: HostClipboardHolderLease): void;
}
export interface HostClipboardAdapter {
  capabilities(): Promise<ClipboardCapabilities>;
  read(cursor: HostClipboardCursor | undefined, maximumBytes: number, context: HostClipboardContext): Promise<HostClipboardPage>;
  write(text: string, context: HostClipboardContext): Promise<HostClipboardMutation>;
  clear(context: HostClipboardContext): Promise<HostClipboardMutation>;
  shutdown?(): Promise<void>;
}
export interface ClipboardProcessLaunch { command: string; args: string[]; environment: Record<string, string> }
export type ClipboardAdapterId = "macos-pasteboard" | "windows-powershell" | "linux-wl-clipboard" | "linux-xclip";
export interface ClipboardHostBridgeTestSeam { adapter: ClipboardAdapterId; launch(adapter: ClipboardAdapterId, action: "read" | "write" | "clear"): ClipboardProcessLaunch }
const helperPids = new Set<number>();
export const clipboardHostHelperPids = (): readonly number[] => [...helperPids];

function safeSessionEnvironment(platform: NodeJS.Platform): Record<string, string> {
  const copy = (name: string, target: Record<string, string>): void => { const value = process.env[name]; if (value && !value.includes("\0")) target[name] = value; };
  if (platform === "win32") return { SystemRoot: "C:\\Windows", WINDIR: "C:\\Windows", PATH: "C:\\Windows\\System32;C:\\Windows\\System32\\WindowsPowerShell\\v1.0",
    PSModulePath: "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\Modules", PATHEXT: ".COM;.EXE;.BAT;.CMD" };
  const result: Record<string, string> = { PATH: "/usr/bin:/bin:/usr/local/bin", LANG: "C.UTF-8", LC_ALL: "C.UTF-8" };
  if (platform === "linux") for (const name of ["DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"]) copy(name, result);
  return result;
}

export function clipboardHostLaunch(adapter: ClipboardAdapterId, action: "read" | "write" | "clear"): ClipboardProcessLaunch {
  if (adapter === "macos-pasteboard") {
    const script = action === "read"
      ? "ObjC.import('AppKit');ObjC.import('Foundation');const value=$.NSPasteboard.generalPasteboard.stringForType($.NSPasteboardTypeString);if(value){const text=ObjC.unwrap(value);$.NSFileHandle.fileHandleWithStandardOutput.writeData($(text).dataUsingEncoding($.NSUTF8StringEncoding));}"
      : "ObjC.import('AppKit');ObjC.import('Foundation');const data=$.NSFileHandle.fileHandleWithStandardInput.readDataToEndOfFile;const text=$.NSString.alloc.initWithDataEncoding(data,$.NSUTF8StringEncoding);const board=$.NSPasteboard.generalPasteboard;board.clearContents;board.setStringForType(text,$.NSPasteboardTypeString);";
    return { command: "/usr/bin/osascript", args: ["-l", "JavaScript", "-e", script], environment: safeSessionEnvironment("darwin") };
  }
  if (adapter === "windows-powershell") {
    const command = "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe";
    const script = action === "read"
      ? "$ErrorActionPreference='Stop';[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);[Console]::Out.Write([string](Get-Clipboard -Raw -TextFormatType UnicodeText))"
      : "$ErrorActionPreference='Stop';[Console]::InputEncoding=[Text.UTF8Encoding]::new($false);$value=[Console]::In.ReadToEnd();Set-Clipboard -Value $value";
    return { command, args: ["-NoLogo", "-NoProfile", "-NonInteractive", "-STA", "-Command", script], environment: safeSessionEnvironment("win32") };
  }
  if (adapter === "linux-wl-clipboard") return { command: action === "read" ? "/usr/bin/wl-paste" : "/usr/bin/wl-copy",
    args: action === "read" ? ["--no-newline", "--type", "text/plain;charset=utf-8"]
      : action === "clear" ? ["--clear"] : ["--foreground", "--type", "text/plain;charset=utf-8"], environment: safeSessionEnvironment("linux") };
  return { command: "/usr/bin/xclip", args: action === "read" ? ["-selection", "clipboard", "-out"]
    : ["-selection", "clipboard", "-in", "-quiet", "-loops", "0"], environment: safeSessionEnvironment("linux") };
}

async function trustedExecutable(path: string): Promise<boolean> {
  try { await access(path, constants.X_OK); const info = await lstat(path); if (!info.isFile() || info.isSymbolicLink()) return false;
    if (process.platform !== "win32" && ((info.mode & 0o022) !== 0 || info.uid !== 0)) return false; return true; }
  catch { return false; }
}
export async function detectClipboardHostAdapter(platform: NodeJS.Platform = process.platform): Promise<ClipboardAdapterId | undefined> {
  if (platform === "darwin" && await trustedExecutable("/usr/bin/osascript")) return "macos-pasteboard";
  if (platform === "win32" && await trustedExecutable("C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe")) return "windows-powershell";
  const sessionValue = (name: string): boolean => { const value = process.env[name]; return Boolean(value && !value.includes("\0")); };
  if (platform === "linux" && sessionValue("WAYLAND_DISPLAY") && sessionValue("XDG_RUNTIME_DIR") &&
      await trustedExecutable("/usr/bin/wl-paste") && await trustedExecutable("/usr/bin/wl-copy")) return "linux-wl-clipboard";
  if (platform === "linux" && sessionValue("DISPLAY") && await trustedExecutable("/usr/bin/xclip")) return "linux-xclip";
  return undefined;
}

function boundedUtf8(bytes: Buffer, maximum: number, complete: boolean): { bytes: Buffer; text: string } {
  const start = complete ? bytes.length : Math.min(maximum, bytes.length);
  for (let drop = 0; drop <= 4 && start - drop >= 0; drop++) {
    const page = bytes.subarray(0, start - drop); try { return { bytes: page, text: new TextDecoder("utf-8", { fatal: true }).decode(page) }; } catch { /* boundary */ }
  }
  throw new Error("Host clipboard did not produce bounded valid UTF-8 text.");
}

export class SystemClipboardHostAdapter implements HostClipboardAdapter {
  private readonly holders = new Map<string, HostClipboardHolderLease>();
  private mutationTail: Promise<void> = Promise.resolve();
  private closing = false;
  constructor(private readonly platform: NodeJS.Platform = process.platform,
    private readonly spawnProcess: typeof spawn = spawn, private readonly injected?: ClipboardHostBridgeTestSeam) {}
  async capabilities(): Promise<ClipboardCapabilities> {
    const adapter = this.injected?.adapter ?? await detectClipboardHostAdapter(this.platform);
    return { format: "omni-clipboard-1", scope: "host", implemented: Boolean(adapter), adapter: adapter ?? "unsupported", contentTypes: ["text/plain"],
      maxReadPageBytes: CLIPBOARD_READ_PAGE_BYTES, readIsBounded: true, write: Boolean(adapter), clear: Boolean(adapter), authority: "current-host-permission-only",
      detail: adapter ? "Text-only current-user OS clipboard access. Reads stream through a monitored helper page; writes/clear replace clipboard formats and require current host policy."
        : "No trusted local text clipboard adapter is available. Studio does not install one or fall back to renderer clipboard APIs." };
  }
  private async adapter(): Promise<ClipboardAdapterId> { const value = this.injected?.adapter ?? await detectClipboardHostAdapter(this.platform); if (!value) throw new Error("No trusted host clipboard adapter is available."); return value; }
  async read(cursor: HostClipboardCursor | undefined, maximumBytes: number, context: HostClipboardContext): Promise<HostClipboardPage> {
    const adapter = await this.adapter(), launch = this.injected?.launch(adapter, "read") ?? clipboardHostLaunch(adapter, "read"), after = cursor?.afterBytes ?? 0;
    if (!Number.isSafeInteger(after) || after < 0 || !Number.isSafeInteger(maximumBytes) || maximumBytes < 1 || maximumBytes > CLIPBOARD_READ_PAGE_BYTES) throw new Error("Invalid host clipboard page request.");
    if (cursor && after === 0 && cursor.prefixSha256 !== clipboardHash(Buffer.alloc(0))) throw new Error("Host clipboard continuation prefix is invalid.");
    return new Promise<HostClipboardPage>((resolve, reject) => {
      context.signal.throwIfAborted(); const child = this.spawnProcess(launch.command, launch.args, { shell: false, windowsHide: true, env: launch.environment, stdio: ["ignore", "pipe", "pipe"] });
      if (!child.pid) { child.kill(); reject(new Error("Host clipboard helper has no process identity.")); return; }
      helperPids.add(child.pid); child.once("close", () => helperPids.delete(child.pid!));
      context.onSpawn(child.pid); let settled = false, prefixRemaining = after, page = Buffer.alloc(0), stoppedForPage = false, stderr = "";
      const prefix = createHash("sha256"), finish = (error?: unknown): void => { if (settled) return; settled = true; clearTimeout(timer); context.signal.removeEventListener("abort", abort);
        if (!child.killed && child.exitCode === null) child.kill(); if (error) reject(error); };
      const abort = (): void => finish(context.signal.reason ?? new Error("Host clipboard read was cancelled."));
      const timer = setTimeout(() => finish(new Error("Host clipboard read exceeded its bounded helper deadline.")), 15_000);
      context.signal.addEventListener("abort", abort, { once: true }); child.on("error", finish);
      child.stderr?.on("data", (chunk: Buffer) => { stderr = `${stderr}${chunk.toString("utf8")}`.slice(-4096); });
      child.stdout?.on("data", (chunk: Buffer) => {
        if (settled || stoppedForPage) return; let offset = 0;
        if (prefixRemaining) { const take = Math.min(prefixRemaining, chunk.length); prefix.update(chunk.subarray(0, take)); prefixRemaining -= take; offset = take;
          if (!prefixRemaining && cursor && prefix.copy().digest("hex") !== cursor.prefixSha256) { finish(new Error("Host clipboard changed before the requested continuation cursor.")); return; } }
        if (offset < chunk.length) { const need = maximumBytes + 4 - page.length; if (need > 0) page = Buffer.concat([page, chunk.subarray(offset, offset + need)]); }
        if (!prefixRemaining && page.length >= maximumBytes + 4) { stoppedForPage = true; child.kill(); }
      });
      child.on("close", code => {
        if (settled) return; if (prefixRemaining) { finish(new Error("Host clipboard continuation is beyond the current text.")); return; }
        if (!stoppedForPage && code !== 0) { finish(new Error(`Host clipboard helper failed (${code ?? "unknown"}): ${stderr}`)); return; }
        try {
          const selected = boundedUtf8(page, maximumBytes, !stoppedForPage), digest = prefix.copy(); digest.update(selected.bytes);
          const complete = !stoppedForPage, result: HostClipboardPage = { adapter, text: selected.text, bytes: selected.bytes.length,
            chunkSha256: clipboardHash(selected.bytes), complete, capturedAt: new Date().toISOString(), ...(!complete ? { next: { afterBytes: after + selected.bytes.length, prefixSha256: digest.digest("hex") } } : {}) };
          settled = true; clearTimeout(timer); context.signal.removeEventListener("abort", abort); resolve(result);
        } catch (error) { finish(error); }
      });
    });
  }
  private queuedMutation(action: "write" | "clear", text: string, context: HostClipboardContext): Promise<HostClipboardMutation> {
    const run = this.mutationTail.catch(() => undefined).then(() => this.mutate(action, text, context));
    this.mutationTail = run.then(() => undefined, () => undefined);
    return run;
  }
  private spawn(launch: ClipboardProcessLaunch, stdio: ["pipe" | "ignore", "pipe" | "ignore", "pipe"]): ChildProcess {
    const child = this.spawnProcess(launch.command, launch.args, { shell: false, windowsHide: true, env: launch.environment, stdio });
    // spawn(2) failures report asynchronously even when no PID was assigned.
    // Operation-specific listeners still receive the same event.
    child.once("error", () => undefined); return child;
  }
  private register(child: ChildProcess, adapter: ClipboardAdapterId, context: HostClipboardContext): HostClipboardHolderLease {
    if (!child.pid) { child.kill(); throw new Error("Host clipboard helper has no process identity."); }
    const pid = child.pid, id = randomUUID(), startedAt = new Date().toISOString(); helperPids.add(pid); context.onSpawn(pid);
    let resolveExit!: () => void; const exit = new Promise<void>(resolve => { resolveExit = resolve; });
    let stopPromise: Promise<void> | undefined;
    const lease: HostClipboardHolderLease = { id, pid, adapter, startedAt, exit, stop: async (): Promise<void> => {
      if (!stopPromise) stopPromise = (async () => {
        if (child.exitCode === null && !child.killed) child.kill("SIGTERM");
        let settled = await Promise.race([exit.then(() => true), new Promise<false>(resolve => setTimeout(() => resolve(false), 750))]);
        if (!settled && child.exitCode === null) child.kill("SIGKILL");
        settled = settled || await Promise.race([exit.then(() => true), new Promise<false>(resolve => setTimeout(() => resolve(false), 750))]);
        if (!settled) throw new Error(`Owned clipboard holder ${pid} did not exit after TERM/KILL.`);
      })();
      await stopPromise;
    } };
    this.holders.set(id, lease);
    child.once("close", () => { helperPids.delete(pid); if (this.holders.get(id) === lease) this.holders.delete(id); resolveExit(); });
    context.onRetainedHolder?.(lease); return lease;
  }
  private async stopOwned(except?: string, bestEffort = false): Promise<void> {
    const results = await Promise.allSettled([...this.holders.values()].filter(holder => holder.id !== except).map(holder => holder.stop()));
    const failures = results.filter((result): result is PromiseRejectedResult => result.status === "rejected");
    if (failures.length && !bestEffort) throw new Error(`Could not reconcile ${failures.length} prior owned clipboard holder(s): ${failures
      .map(result => result.reason instanceof Error ? result.reason.message : String(result.reason)).join("; ")}`);
  }
  private async pipeInput(child: ChildProcess, bytes: Buffer, signal: AbortSignal): Promise<void> {
    if (!child.stdin) throw new Error("Host clipboard helper has no input pipe.");
    signal.throwIfAborted();
    const chunks = function* (): Generator<Buffer> { for (let offset = 0; offset < bytes.length; offset += 64 * 1024) yield bytes.subarray(offset, Math.min(bytes.length, offset + 64 * 1024)); };
    const input = Readable.from(chunks());
    await new Promise<void>((resolve, reject) => {
      let settled = false;
      const finish = (error?: unknown): void => { if (settled) return; settled = true; signal.removeEventListener("abort", abort); input.removeListener("error", fail); child.stdin?.removeListener("error", fail);
        if (error) reject(error); else resolve(); };
      const fail = (error: unknown): void => finish(error);
      const abort = (): void => { const error = signal.reason ?? new Error("Host clipboard mutation was cancelled after dispatch; its effect is uncertain."); input.destroy(error as Error); child.stdin?.destroy(error as Error); finish(error); };
      signal.addEventListener("abort", abort, { once: true }); input.once("error", fail); child.stdin!.once("error", fail); child.stdin!.once("finish", () => finish()); input.pipe(child.stdin!);
    });
  }
  private async provePublished(adapter: ClipboardAdapterId, expected: Buffer, context: HostClipboardContext): Promise<void> {
    const launch = this.injected?.launch(adapter, "read") ?? clipboardHostLaunch(adapter, "read");
    await new Promise<void>((resolve, reject) => {
      context.signal.throwIfAborted(); const child = this.spawn(launch, ["ignore", "pipe", "pipe"]);
      if (!child.pid) { child.kill(); reject(new Error("Clipboard readback helper has no process identity.")); return; }
      const pid = child.pid; helperPids.add(pid); context.onSpawn(pid); child.once("close", () => helperPids.delete(pid));
      let settled = false, count = 0, stderr = ""; const digest = createHash("sha256");
      const finish = (error?: unknown): void => { if (settled) return; settled = true; clearTimeout(timer); context.signal.removeEventListener("abort", abort);
        if (child.exitCode === null && !child.killed) child.kill(); if (error) reject(error); else resolve(); };
      const abort = (): void => finish(context.signal.reason ?? new Error("Clipboard publication proof was cancelled; the write is uncertain."));
      const timer = setTimeout(() => finish(new Error("Clipboard publication proof timed out; the write is uncertain.")), 15_000);
      context.signal.addEventListener("abort", abort, { once: true }); child.on("error", finish);
      child.stderr?.on("data", (chunk: Buffer) => { stderr = `${stderr}${chunk.toString("utf8")}`.slice(-4096); });
      child.stdout?.on("data", (chunk: Buffer) => { if (settled) return; count += chunk.length;
        if (count > expected.length) { finish(new Error("Clipboard publication readback contained more bytes than the dispatched text; the write is uncertain.")); return; } digest.update(chunk); });
      child.on("close", code => { if (settled) return; if (code !== 0) { finish(new Error(`Clipboard publication readback failed (${code ?? "unknown"}): ${stderr}`)); return; }
        if (count !== expected.length || digest.digest("hex") !== clipboardHash(expected)) { finish(new Error("Clipboard publication readback did not match the dispatched text; the write is uncertain.")); return; }
        finish(); });
    });
  }
  private async retainedLinuxMutation(adapter: "linux-wl-clipboard" | "linux-xclip", action: "write" | "clear", text: string,
    context: HostClipboardContext): Promise<HostClipboardMutation> {
    const bytes = Buffer.from(action === "clear" ? "" : text, "utf8"), launch = this.injected?.launch(adapter, action) ?? clipboardHostLaunch(adapter, action);
    const child = this.spawn(launch, ["pipe", "ignore", "pipe"]), lease = this.register(child, adapter, context); let stderr = "";
    child.stderr?.on("data", (chunk: Buffer) => { stderr = `${stderr}${chunk.toString("utf8")}`.slice(-4096); });
    await this.pipeInput(child, bytes, context.signal);
    await new Promise<void>((resolve, reject) => { let settled = false;
      const finish = (error?: Error): void => { if (settled) return; settled = true; clearTimeout(timer); child.removeListener("close", closed); if (error) reject(error); else resolve(); };
      const closed = (): void => finish(new Error(`Linux clipboard holder exited before publication proof: ${stderr}`));
      const timer = setTimeout(() => finish(), 50); child.once("close", closed); });
    context.signal.throwIfAborted();
    if (child.exitCode !== null || child.killed) throw new Error(`Linux clipboard holder exited before publication proof: ${stderr}`);
    await this.provePublished(adapter, bytes, context); context.signal.throwIfAborted();
    if (child.exitCode !== null || child.killed) throw new Error("Linux clipboard ownership was lost during publication proof; the write is uncertain.");
    await this.stopOwned(lease.id);
    return { adapter, completedAt: new Date().toISOString(), holder: lease };
  }
  private async oneShotMutation(adapter: ClipboardAdapterId, action: "write" | "clear", text: string, context: HostClipboardContext): Promise<HostClipboardMutation> {
    const launch = this.injected?.launch(adapter, action) ?? clipboardHostLaunch(adapter, action);
    return new Promise<HostClipboardMutation>((resolve, reject) => {
      const child = this.spawn(launch, ["pipe", "ignore", "pipe"]);
      if (!child.pid) { child.kill(); reject(new Error("Host clipboard helper has no process identity.")); return; }
      const pid = child.pid; helperPids.add(pid); child.once("close", () => helperPids.delete(pid)); context.onSpawn(pid); let settled = false, stderr = "";
      const finish = (error?: unknown): void => { if (settled) return; settled = true; clearTimeout(timer); context.signal.removeEventListener("abort", abort);
        if (child.exitCode === null && !child.killed) child.kill(); if (error) reject(error); };
      const abort = (): void => finish(context.signal.reason ?? new Error("Host clipboard mutation was cancelled after dispatch; its effect is uncertain."));
      const timer = setTimeout(() => finish(new Error("Host clipboard mutation timed out after dispatch; its effect is uncertain.")), 15_000);
      context.signal.addEventListener("abort", abort, { once: true }); child.on("error", finish); child.stdin?.on("error", finish);
      child.stderr?.on("data", (chunk: Buffer) => { stderr = `${stderr}${chunk.toString("utf8")}`.slice(-4096); });
      child.on("close", code => { if (settled) return; if (code !== 0) { finish(new Error(`Host clipboard helper failed after dispatch (${code ?? "unknown"}): ${stderr}`)); return; }
        settled = true; clearTimeout(timer); context.signal.removeEventListener("abort", abort); resolve({ adapter, completedAt: new Date().toISOString() }); });
      const bytes = Buffer.from(action === "clear" ? "" : text, "utf8");
      if (!child.stdin) { finish(new Error("Host clipboard helper has no input pipe.")); return; }
      const chunks = function* (): Generator<Buffer> { for (let offset = 0; offset < bytes.length; offset += 64 * 1024) yield bytes.subarray(offset, Math.min(bytes.length, offset + 64 * 1024)); };
      Readable.from(chunks()).pipe(child.stdin);
    });
  }
  private async mutate(action: "write" | "clear", text: string, context: HostClipboardContext): Promise<HostClipboardMutation> {
    if (this.closing) throw new Error("Host clipboard adapter is closing.");
    const adapter = await this.adapter(); await context.beforeDispatch?.(); context.signal.throwIfAborted();
    if (adapter === "linux-xclip" || adapter === "linux-wl-clipboard" && action === "write") return this.retainedLinuxMutation(adapter, action, text, context);
    const result = await this.oneShotMutation(adapter, action, text, context);
    if (adapter === "linux-wl-clipboard" && action === "clear") await this.stopOwned();
    return result;
  }
  async shutdown(): Promise<void> { this.closing = true; await this.stopOwned(undefined, true); await this.mutationTail.catch(() => undefined); await this.stopOwned(undefined, true); }
  write(text: string, context: HostClipboardContext): Promise<HostClipboardMutation> { return this.queuedMutation("write", text, context); }
  clear(context: HostClipboardContext): Promise<HostClipboardMutation> { return this.queuedMutation("clear", "", context); }
}
