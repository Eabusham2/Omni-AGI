import { constants, createReadStream } from "node:fs";
import { createHash } from "node:crypto";
import { access, chmod, lstat, realpath } from "node:fs/promises";
import { isAbsolute, join, relative, resolve, sep, win32 } from "node:path";
import { pathToFileURL } from "node:url";
import type {
  EnvironmentAdapterCapability,
  EnvironmentProcessReceipt,
  ManagedEnvironmentRecord
} from "../shared/environments";
import { environmentHash } from "./environmentFormat";
import { writeMutableFileIsolated } from "./mutableFileIsolation";

export type EnvironmentProcessRunner = (
  command: string,
  args: string[],
  cwd: string,
  timeoutMs: number,
  environment?: Record<string, string>,
  signal?: AbortSignal,
  onSpawn?: (pid: number) => void
) => Promise<EnvironmentProcessReceipt>;

export interface EnvironmentAdapterDependencies {
  platform?: NodeJS.Platform;
  arch?: string;
  run: EnvironmentProcessRunner;
  findExecutable?(name: string): Promise<string | undefined>;
  /** Test/host seam that must itself obtain exact Appx/AuthentiCode evidence. */
  windowsSandboxExecutable?(): Promise<WindowsSandboxExecutableEvidence | undefined>;
  onSpawn?(pid: number): void;
}

export interface WindowsSandboxExecutableEvidence {
  path: string;
  installationLocation: string;
  packageFamilyName: "MicrosoftWindows.WindowsSandbox_cw5n1h2txyewy";
  signatureStatus: "Valid";
}

export interface ResolvedEnvironmentImage {
  path: string;
  sha256: string;
  bytes: number;
  architecture: "aarch64" | "x86_64";
}

const supportedProgramRoots = ["/bin", "/sbin", "/usr/bin", "/usr/sbin", "/usr/local/bin", "/opt/homebrew/bin"];
const safeEnvironment = (workspace: string): Record<string, string> => ({
  HOME: workspace,
  TMPDIR: join(workspace, "tmp"),
  PATH: "/usr/bin:/bin:/usr/sbin:/sbin",
  LANG: "C.UTF-8",
  LC_ALL: "C.UTF-8"
});

async function defaultFindExecutable(name: string): Promise<string | undefined> {
  const candidates = name === "sandbox-exec" && process.platform === "darwin" ? ["/usr/bin/sandbox-exec"]
    : name === "bwrap" && process.platform === "linux" ? ["/usr/bin/bwrap", "/bin/bwrap", "/usr/local/bin/bwrap"]
      : name === "limactl" && process.platform !== "win32" ? ["/opt/homebrew/bin/limactl", "/usr/local/bin/limactl", "/usr/bin/limactl"] : [];
  for (const candidate of candidates) {
    try {
      if (!isAbsolute(candidate)) continue;
      await access(candidate, constants.X_OK);
      const actual = await realpath(candidate), info = await lstat(actual);
      const uid = typeof process.getuid === "function" ? process.getuid() : info.uid;
      if (info.isFile() && !info.isSymbolicLink() && (info.mode & 0o022) === 0 && (info.uid === 0 || info.uid === uid)) return actual;
    } catch { /* Continue with explicit platform installation paths only. */ }
  }
  return undefined;
}

function validatedWindowsSandboxExecutable(evidence: WindowsSandboxExecutableEvidence | undefined): string | undefined {
  if (!evidence || evidence.packageFamilyName !== "MicrosoftWindows.WindowsSandbox_cw5n1h2txyewy" || evidence.signatureStatus !== "Valid" ||
      !win32.isAbsolute(evidence.path) || !win32.isAbsolute(evidence.installationLocation) || /[\0\r\n]/.test(evidence.path) || /[\0\r\n]/.test(evidence.installationLocation)) return undefined;
  const packageRoot = "C:\\Program Files\\WindowsApps", installedPackage = win32.relative(packageRoot, evidence.installationLocation);
  const inside = win32.relative(evidence.installationLocation, evidence.path);
  if (!installedPackage || installedPackage === ".." || installedPackage.startsWith(`..${win32.sep}`) || win32.isAbsolute(installedPackage) ||
      !inside || inside === ".." || inside.startsWith(`..${win32.sep}`) || win32.isAbsolute(inside) || win32.basename(evidence.path).toLocaleLowerCase() !== "wsb.exe") return undefined;
  return evidence.path;
}

async function trustedWindowsSandboxExecutable(deps: EnvironmentAdapterDependencies): Promise<string | undefined> {
  if ((deps.platform ?? process.platform) !== "win32" || process.platform !== "win32") return undefined;
  // A nonstandard Windows-directory installation is reported unsupported
  // rather than trusting an environment-controlled SystemRoot executable.
  const powershell = "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe";
  try {
    const info = await lstat(powershell);
    if (!info.isFile() || info.isSymbolicLink()) return undefined;
    // Microsoft publishes the `wsb` command but no fixed executable path or
    // signer subject. Resolve it from the exact installed Appx package and
    // require Windows' own Authenticode verifier to report a valid signature.
    const script = [
      "$ErrorActionPreference='Stop'",
      "$package=Get-AppxPackage -Name 'MicrosoftWindows.WindowsSandbox' | Select-Object -First 1",
      "if($null -eq $package){exit 20}",
      "$candidate=Join-Path $package.InstallLocation 'wsb.exe'",
      "$signature=Get-AuthenticodeSignature -LiteralPath $candidate",
      "[ordered]@{path=[IO.Path]::GetFullPath($candidate);installationLocation=[IO.Path]::GetFullPath($package.InstallLocation);packageFamilyName=$package.PackageFamilyName;signatureStatus=$signature.Status.ToString()} | ConvertTo-Json -Compress"
    ].join(";");
    const result = await deps.run(powershell, ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script], process.cwd(), 8_000,
      { SystemRoot: "C:\\Windows", WINDIR: "C:\\Windows", PATH: "C:\\Windows\\System32;C:\\Windows\\System32\\WindowsPowerShell\\v1.0",
        PSModulePath: "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\Modules", PATHEXT: ".COM;.EXE;.BAT;.CMD" }, AbortSignal.timeout(8_000));
    if (result.exitCode !== 0) return undefined;
    const parsed = JSON.parse(result.stdout.trim()) as WindowsSandboxExecutableEvidence;
    const executable = validatedWindowsSandboxExecutable(parsed); if (!executable) return undefined;
    const executableInfo = await lstat(executable);
    return executableInfo.isFile() && !executableInfo.isSymbolicLink() ? executable : undefined;
  } catch { return undefined; }
}

async function adapterExecutable(deps: EnvironmentAdapterDependencies, name: string): Promise<string | undefined> {
  if (name === "wsb.exe") return deps.windowsSandboxExecutable
    ? validatedWindowsSandboxExecutable(await deps.windowsSandboxExecutable())
    : trustedWindowsSandboxExecutable(deps);
  if (deps.findExecutable) return deps.findExecutable(name);
  return defaultFindExecutable(name);
}

async function commandVersion(deps: EnvironmentAdapterDependencies, executable: string, args: string[]): Promise<string | undefined> {
  const signal = AbortSignal.timeout(3_000);
  try {
    const result = await deps.run(executable, args, process.cwd(), 3_000, { PATH: process.env.PATH ?? "" }, signal);
    const line = `${result.stdout}\n${result.stderr}`.split(/\r?\n/).map(item => item.trim()).find(Boolean);
    return line?.slice(0, 500);
  } catch { return undefined; }
}

export async function environmentAdapterCapabilities(
  deps: EnvironmentAdapterDependencies,
  vmImageResolverConfigured = false
): Promise<EnvironmentAdapterCapability[]> {
  const platform = deps.platform ?? process.platform, find = (name: string): Promise<string | undefined> => adapterExecutable(deps, name);
  const result: EnvironmentAdapterCapability[] = [];
  if (platform === "darwin") {
    const executable = await find("sandbox-exec");
    result.push({ id: "darwin-sandbox-exec", kind: "sandbox", available: Boolean(executable), ...(executable ? { executable } : {}),
      supportsCreate: Boolean(executable), supportsRun: Boolean(executable), requiresLocalImage: false, deprecated: true,
      warning: executable
        ? "Apple marks sandbox-exec deprecated. It is an actual profile boundary, not App Sandbox certification or a future-OS guarantee."
        : "No supported local sandbox adapter is installed; unrestricted host execution is not substituted." });
  } else if (platform === "linux") {
    const executable = await find("bwrap");
    result.push({ id: "linux-bwrap", kind: "sandbox", available: Boolean(executable), ...(executable ? { executable } : {}),
      supportsCreate: Boolean(executable), supportsRun: Boolean(executable), requiresLocalImage: false,
      warning: executable ? undefined : "bubblewrap is not installed; unrestricted host execution is not substituted." });
  } else if (platform === "win32") {
    const executable = await find("wsb.exe");
    let ready = false, diagnostic = "";
    if (executable) {
      try {
        const probe = await deps.run(executable, ["list", "--raw"], process.cwd(), 8_000, {}, AbortSignal.timeout(8_000));
        ready = probe.exitCode === 0 && parseWindowsSandboxList(probe.stdout, probe.truncated).recognized;
        diagnostic = `${probe.stderr}\n${probe.stdout}`.trim().slice(0, 500);
      } catch (error) { diagnostic = error instanceof Error ? error.message.slice(0, 500) : String(error).slice(0, 500); }
    }
    result.push({ id: "windows-sandbox-cli", kind: "sandbox", available: Boolean(executable && ready), ...(executable ? { executable } : {}),
      supportsCreate: Boolean(executable && ready), supportsRun: Boolean(executable && ready), requiresLocalImage: false,
      warning: !executable
        ? "A trusted installed Windows Sandbox CLI was not found. Windows 11 24H2+, a supported edition, virtualization, and the already-enabled optional feature are required; Studio does not install or enable them."
        : !ready ? `The trusted Windows Sandbox CLI is installed but its read-only runtime probe failed${diagnostic ? `: ${diagnostic}` : "."}`
          : "Windows Sandbox is disposable and single-instance. Stop or cancellation deletes guest state; the CLI exposes a guest exit code but no guest stdout/stderr." });
  } else {
    result.push({ id: "linux-bwrap", kind: "sandbox", available: false, supportsCreate: false, supportsRun: false, requiresLocalImage: false,
      warning: "No supported sandbox adapter is implemented for this operating system; the normal host runner is not treated as isolation." });
  }
  const lima = platform === "darwin" || platform === "linux" ? await find("limactl") : undefined;
  result.push({ id: "lima", kind: "vm", available: Boolean(lima), ...(lima ? { executable: lima, version: await commandVersion(deps, lima, ["--version"]) } : {}),
    supportsCreate: Boolean(lima && vmImageResolverConfigured), supportsRun: Boolean(lima), requiresLocalImage: true,
    warning: !lima ? "Lima is not installed; no host command is substituted for a VM."
      : !vmImageResolverConfigured ? "Lima is installed, but VM creation awaits a hash-verified local image attachment. Downloads and templates are disabled."
        : "VM creation accepts only the exact reviewed local image. No remote URL/template download, host mount, user provisioning script, credential forwarding, or elevated helper is used." });
  return result;
}

function sandboxProfile(workspace: string): string {
  const quoted = (value: string): string => `"${value.replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"`;
  return `(version 1)\n(deny default)\n(allow process*)\n` +
    `(allow file-read*)\n(deny file-read*${["/Users", "/Volumes", "/Network", "/private/tmp", "/private/var/folders", "/var/tmp"].map(path => ` (subpath ${quoted(path)})`).join("")})\n` +
    `(allow file-read* (subpath ${quoted(workspace)}))\n` +
    `(allow file-write* (subpath ${quoted(workspace)}))\n(deny network*)\n`;
}

export async function prepareSandboxEnvironment(record: ManagedEnvironmentRecord, directory: string): Promise<{ profileSha256?: string }> {
  const workspacePath = join(directory, "workspace"), tmp = join(workspacePath, "tmp");
  for (const path of [directory, workspacePath, tmp]) {
    let info = await lstat(path); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Sandbox workspace is not a private real directory.");
    if (process.platform !== "win32") {
      const uid = typeof process.getuid === "function" ? process.getuid() : info.uid;
      if (info.uid !== uid) throw new Error("Sandbox workspace has another operating-system owner.");
      if ((info.mode & 0o077) !== 0) { await chmod(path, 0o700); info = await lstat(path); }
      if (info.uid !== uid || (info.mode & 0o077) !== 0) throw new Error("Sandbox workspace is not private to the current operating-system owner.");
    }
  }
  if (record.adapter !== "darwin-sandbox-exec") return {};
  const workspace = await realpath(workspacePath);
  const profile = sandboxProfile(workspace), profileSha256 = environmentHash(profile);
  await writeMutableFileIsolated(join(directory, "sandbox.sb"), Buffer.from(profile, "utf8"));
  return { profileSha256 };
}

async function safeProgram(program: unknown, workspace: string): Promise<{ host: string; guest: string }> {
  if (typeof program !== "string" || !program || program.includes("\0") || program.includes("\n")) throw new Error("Sandbox program must be one exact executable path.");
  const candidate = isAbsolute(program) ? program : resolve(workspace, program);
  const actual = await realpath(candidate), info = await lstat(actual);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Sandbox program is not a real executable file.");
  await access(actual, process.platform === "win32" ? constants.F_OK : constants.X_OK);
  const inside = relative(workspace, actual);
  if (inside && inside !== ".." && !inside.startsWith(`..${sep}`) && !isAbsolute(inside)) return { host: actual, guest: `/workspace/${inside.split(/[\\/]/).join("/")}` };
  const requestedRoot = supportedProgramRoots.find(root => candidate === root || candidate.startsWith(`${root}/`));
  const resolvedRequestedRoot = requestedRoot ? await realpath(requestedRoot).catch(() => requestedRoot) : undefined;
  const resolvedRoots = requestedRoot === "/opt/homebrew/bin" ? [requestedRoot, resolvedRequestedRoot!, "/opt/homebrew/Cellar"]
    : requestedRoot === "/usr/local/bin" ? [requestedRoot, resolvedRequestedRoot!, "/usr/local/Cellar"] : requestedRoot ? [requestedRoot, resolvedRequestedRoot!] : [];
  if (!resolvedRoots.some(root => actual === root || actual.startsWith(`${root}/`))) throw new Error("Sandbox programs must be inside its workspace or a verified installed system executable directory.");
  return { host: actual, guest: actual };
}

function exactArguments(value: unknown): string[] {
  if (value === undefined) return [];
  if (!Array.isArray(value) || value.some(item => typeof item !== "string" || item.includes("\0"))) throw new Error("Every environment argument must be exact NUL-free text.");
  return [...value] as string[];
}

const WINDOWS_SANDBOX_CONFIGURATION = "<Configuration><vGPU>Disable</vGPU><Networking>Disable</Networking><AudioInput>Disable</AudioInput><VideoInput>Disable</VideoInput><ProtectedClient>Enable</ProtectedClient><PrinterRedirection>Disable</PrinterRedirection><ClipboardRedirection>Disable</ClipboardRedirection><MemoryInMB>2048</MemoryInMB></Configuration>";
const windowsSandboxIdPattern = /^[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i;
const discoveredWindowsSandboxIdPattern = /^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/i;
const windowsSandboxIdGlobalPattern = /\b[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}\b/gi;
let windowsSandboxMutationTail: Promise<void> = Promise.resolve();

async function withWindowsSandboxMutation<T>(operation: () => Promise<T>): Promise<T> {
  // Electron holds the installation's single-instance lock. This process-wide
  // lease closes the remaining cross-brain list→start/stop race; the Windows
  // Sandbox service still arbitrates races from unrelated current-user apps.
  let release!: () => void;
  const predecessor = windowsSandboxMutationTail;
  windowsSandboxMutationTail = new Promise<void>(resolveLease => { release = resolveLease; });
  await predecessor;
  try { return await operation(); } finally { release(); }
}

export function windowsSandboxConfiguration(): string { return WINDOWS_SANDBOX_CONFIGURATION; }

function windowsSandboxId(value: unknown): string {
  if (typeof value !== "string" || !windowsSandboxIdPattern.test(value)) throw new Error("Windows Sandbox adapter identity is not an exact UUID.");
  return value.toLocaleLowerCase();
}

interface ParsedWindowsSandboxSession { id: string; status?: "running" | "stopped" }
interface ParsedWindowsSandboxList { recognized: boolean; sessions: ParsedWindowsSandboxSession[] }
function parseWindowsSandboxList(value: string, truncated = false): ParsedWindowsSandboxList {
  if (truncated) return { recognized: false, sessions: [] };
  try {
    const parsed: unknown = JSON.parse(value);
    if (!parsed || typeof parsed !== "object") return { recognized: false, sessions: [] };
    let entries: unknown;
    if (Array.isArray(parsed)) entries = parsed;
    else {
      const record = parsed as Record<string, unknown>, list = Object.entries(record)
        .find(([key, child]) => Array.isArray(child) && /sandbox|session|environment|items|result|value/i.test(key));
      if (list) entries = list[1];
      else return { recognized: false, sessions: [] };
    }
    if (!Array.isArray(entries)) return { recognized: false, sessions: [] };
    const sessions: ParsedWindowsSandboxSession[] = [];
    for (const entry of entries) {
      if (typeof entry === "string" && discoveredWindowsSandboxIdPattern.test(entry)) { sessions.push({ id: entry.toLocaleLowerCase() }); continue; }
      if (!entry || typeof entry !== "object" || Array.isArray(entry)) return { recognized: false, sessions: [] };
      const fields = Object.entries(entry as Record<string, unknown>), id = fields.find(([key, child]) =>
        /^(?:id|sandboxId|environmentId)$/i.test(key) && typeof child === "string" && discoveredWindowsSandboxIdPattern.test(child));
      if (!id) return { recognized: false, sessions: [] };
      const status = fields.find(([key, child]) => /^status$/i.test(key) && typeof child === "string" && /^(?:running|stopped)$/i.test(child));
      sessions.push({ id: String(id[1]).toLocaleLowerCase(), ...(status ? { status: String(status[1]).toLocaleLowerCase() as "running" | "stopped" } : {}) });
    }
    return { recognized: true, sessions };
  } catch {
    const lines = value.split(/\r?\n/).map(item => item.trim()).filter(Boolean);
    if (!lines.length || lines.some(item => !discoveredWindowsSandboxIdPattern.test(item))) return { recognized: false, sessions: [] };
    return { recognized: true, sessions: [...new Set(lines.map(item => item.toLocaleLowerCase()))].map(id => ({ id })) };
  }
}

export function windowsSandboxListState(value: string, expectedId: string, truncated = false): "loaded" | "stopped" | "missing" | "unknown" {
  const id = windowsSandboxId(expectedId), list = parseWindowsSandboxList(value, truncated);
  if (!list.recognized) return "unknown";
  const session = list.sessions.find(item => item.id === id); if (!session) return "missing";
  if (session.status === "stopped") return "stopped";
  if (session.status === "running" || session.status === undefined) return "loaded";
  return "unknown";
}

function quoteWindowsCommandLineArgument(value: string): string {
  if (!/[\s"]/.test(value) && value.length) return value;
  let output = '"', slashes = 0;
  for (const character of value) {
    if (character === "\\") { slashes += 1; continue; }
    if (character === '"') { output += "\\".repeat(slashes * 2 + 1) + '"'; slashes = 0; continue; }
    output += "\\".repeat(slashes) + character; slashes = 0;
  }
  return `${output}${"\\".repeat(slashes * 2)}"`;
}

export function windowsSandboxCommand(programValue: unknown, argumentsValue: unknown): string {
  if (typeof programValue !== "string" || programValue.length > 1024 || !/^[A-Za-z]:\\/.test(programValue) || /[\0\r\n]/.test(programValue) ||
      programValue.startsWith("\\\\") || programValue.split("\\").some(segment => !segment || segment === "." || segment === ".." || segment.endsWith(" ") || segment.endsWith(".")) ||
      !/\.(?:exe|com)$/i.test(programValue)) throw new Error("Windows Sandbox program must be one exact absolute guest executable path.");
  const args = exactArguments(argumentsValue);
  const command = [programValue, ...args].map(quoteWindowsCommandLineArgument).join(" ");
  if (command.length > 30_000) throw new Error("Windows Sandbox command line exceeds the reviewed guest command limit.");
  return command;
}

async function windowsSandboxCommandRunner(
  deps: EnvironmentAdapterDependencies,
  args: string[],
  cwd: string,
  timeoutMs: number,
  signal: AbortSignal
): Promise<EnvironmentProcessReceipt> {
  const capability = (await environmentAdapterCapabilities(deps)).find(row => row.id === "windows-sandbox-cli");
  if (!capability?.available || !capability.executable) throw new Error(capability?.warning ?? "Windows Sandbox CLI is unavailable.");
  return deps.run(capability.executable, args, cwd, timeoutMs, {}, signal, deps.onSpawn);
}

export async function inspectWindowsSandbox(
  deps: EnvironmentAdapterDependencies,
  record: ManagedEnvironmentRecord,
  directory: string,
  signal: AbortSignal
): Promise<"loaded" | "stopped" | "missing" | "unknown"> {
  if (record.adapter !== "windows-sandbox-cli" || !record.externalId) return "missing";
  const result = await windowsSandboxCommandRunner(deps, ["list", "--raw"], directory, 15_000, signal);
  if (result.exitCode !== 0) return "unknown";
  return windowsSandboxListState(result.stdout, record.externalId, result.truncated);
}

export async function startWindowsSandbox(
  deps: EnvironmentAdapterDependencies,
  record: ManagedEnvironmentRecord,
  directory: string,
  signal: AbortSignal
): Promise<EnvironmentProcessReceipt> {
  return withWindowsSandboxMutation(async () => {
    if (record.adapter !== "windows-sandbox-cli" || !record.externalId) throw new Error("Windows Sandbox start requires an exact owned UUID.");
    const id = windowsSandboxId(record.externalId);
    const listed = await windowsSandboxCommandRunner(deps, ["list", "--raw"], directory, 15_000, signal);
    if (listed.exitCode !== 0) throw new Error(`Windows Sandbox runtime inspection failed before Start (exit ${listed.exitCode}).`);
    const parsed = parseWindowsSandboxList(listed.stdout, listed.truncated);
    if (!parsed.recognized) throw new Error("Windows Sandbox returned an unrecognized or truncated session list; Studio did not Start or adopt anything.");
    const existing = parsed.sessions.map(session => session.id);
    if (existing.some(candidate => candidate !== id)) throw new Error("Windows Sandbox supports one session and an unrelated current-user sandbox is already active; Studio did not adopt or stop it.");
    if (existing.includes(id)) throw new Error("The owned Windows Sandbox identity is already active; inspect its retained lifecycle before another Load.");
    const process = await windowsSandboxCommandRunner(deps,
      ["start", "--raw", "--id", id, "--config", WINDOWS_SANDBOX_CONFIGURATION], directory, 120_000, signal);
    if (process.exitCode !== 0) return process;
    const state = await inspectWindowsSandbox(deps, record, directory, signal);
    if (state !== "loaded") throw new Error("Windows Sandbox Start returned success but the exact owned UUID was not reported running.");
    return process;
  });
}

export async function runWindowsSandbox(
  deps: EnvironmentAdapterDependencies,
  record: ManagedEnvironmentRecord,
  directory: string,
  programValue: unknown,
  argumentsValue: unknown,
  timeoutMs: number,
  signal: AbortSignal
): Promise<EnvironmentProcessReceipt> {
  if (record.adapter !== "windows-sandbox-cli" || !record.externalId) throw new Error("Windows Sandbox run requires an exact owned UUID.");
  if (await inspectWindowsSandbox(deps, record, directory, signal) !== "loaded") throw new Error("The exact owned Windows Sandbox is not running.");
  const result = await windowsSandboxCommandRunner(deps,
    ["exec", "--raw", "--id", windowsSandboxId(record.externalId), "--command", windowsSandboxCommand(programValue, argumentsValue), "--run-as", "System"],
    directory, timeoutMs, signal);
  return { ...result, guestOutput: "guest-output-unavailable" };
}

export async function stopWindowsSandbox(
  deps: EnvironmentAdapterDependencies,
  record: ManagedEnvironmentRecord,
  directory: string,
  signal: AbortSignal
): Promise<EnvironmentProcessReceipt> {
  return withWindowsSandboxMutation(async () => {
    if (record.adapter !== "windows-sandbox-cli" || !record.externalId) throw new Error("Windows Sandbox stop requires an exact owned UUID.");
    const id = windowsSandboxId(record.externalId), before = await inspectWindowsSandbox(deps, record, directory, signal);
    if (before === "missing" || before === "stopped") return { exitCode: 0, stdout: "", stderr: "", truncated: false, sandboxDisposition: "already-absent" };
    if (before !== "loaded") throw new Error("Windows Sandbox state is unknown; Studio did not issue an unverified Stop.");
    const result = await windowsSandboxCommandRunner(deps, ["stop", "--raw", "--id", id], directory, 120_000, signal);
    if (result.exitCode !== 0) return { ...result, sandboxDisposition: "unconfirmed" };
    const after = await inspectWindowsSandbox(deps, record, directory, signal);
    if (after !== "missing" && after !== "stopped") throw new Error("Windows Sandbox Stop returned success but the exact owned UUID still appears active.");
    return { ...result, sandboxDisposition: "stopped" };
  });
}

export async function runSandboxEnvironment(
  deps: EnvironmentAdapterDependencies,
  record: ManagedEnvironmentRecord,
  directory: string,
  programValue: unknown,
  argumentsValue: unknown,
  timeoutMs: number,
  signal: AbortSignal
): Promise<EnvironmentProcessReceipt> {
  await prepareSandboxEnvironment(record, directory);
  const workspace = await realpath(join(directory, "workspace")), program = await safeProgram(programValue, workspace), args = exactArguments(argumentsValue);
  signal.throwIfAborted();
  const capability = (await environmentAdapterCapabilities(deps)).find(row => row.id === record.adapter);
  if (!capability?.available || !capability.executable) throw new Error(capability?.warning ?? "The saved sandbox adapter is unavailable.");
  if (record.adapter === "darwin-sandbox-exec") {
    const profile = join(directory, "sandbox.sb"), info = await lstat(profile);
    if (!info.isFile() || info.isSymbolicLink()) throw new Error("Sandbox profile is missing or unsafe.");
    // runProcess deliberately preserves the host environment for ordinary
    // tools. Enter through env -i so no host credential/value reaches the
    // sandboxed command even though sandbox-exec itself is a trusted host tool.
    return deps.run("/usr/bin/env", ["-i", ...Object.entries(safeEnvironment(workspace)).map(([key, value]) => `${key}=${value}`),
      capability.executable, "-f", profile, program.host, ...args], workspace, timeoutMs, {}, signal, deps.onSpawn);
  }
  if (record.adapter === "linux-bwrap") {
    const systemBinds: string[] = [], systemLinks: string[] = [];
    for (const path of ["/usr", "/bin", "/sbin", "/lib", "/lib64"]) {
      const info = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
      if (info?.isDirectory() && !info.isSymbolicLink()) systemBinds.push("--ro-bind", path, path);
      else if (info?.isSymbolicLink()) {
        const target = await realpath(path), fromRoot = relative("/", target);
        if (!target.startsWith("/usr/") || !fromRoot || fromRoot === ".." || fromRoot.startsWith("../") || isAbsolute(fromRoot)) throw new Error(`Unsupported merged-/usr link for ${path}.`);
        systemLinks.push("--symlink", fromRoot.split(/[\\/]/).join("/"), path);
      }
    }
    const guestProgram = program.guest;
    return deps.run(capability.executable, ["--die-with-parent", "--new-session", "--unshare-all", "--unshare-net", "--clearenv",
      "--proc", "/proc", "--dev", "/dev", ...systemBinds, ...systemLinks, "--bind", workspace, "/workspace", "--chdir", "/workspace",
      "--setenv", "HOME", "/workspace", "--setenv", "TMPDIR", "/workspace/tmp", "--setenv", "PATH", "/usr/bin:/bin:/usr/sbin:/sbin",
      guestProgram, ...args], workspace, timeoutMs, {}, signal, deps.onSpawn);
  }
  throw new Error("The saved environment is not a sandbox adapter.");
}

function yamlString(value: string): string { return JSON.stringify(value); }
export async function prepareLimaVm(
  directory: string,
  record: ManagedEnvironmentRecord,
  image: ResolvedEnvironmentImage,
  resources: { cpus: number; memoryGiB: number; diskGiB: number }
): Promise<string> {
  if (record.adapter !== "lima" || record.network !== "guest-outbound" || !record.externalId) throw new Error("Lima creation requires an exact outbound-network VM record.");
  const imageInfo = await lstat(image.path); if (!imageInfo.isFile() || imageInfo.isSymbolicLink() || imageInfo.size !== image.bytes) throw new Error("The reviewed VM image is unavailable or changed.");
  const digest = await new Promise<string>((resolveHash, rejectHash) => {
    const hash = createHash("sha256"), stream = createReadStream(image.path);
    stream.on("data", chunk => hash.update(chunk)); stream.once("error", rejectHash); stream.once("end", () => resolveHash(hash.digest("hex")));
  });
  if (digest !== image.sha256 || image.sha256 !== record.imageSha256 || image.architecture !== record.guestArchitecture) throw new Error("The local VM image differs from its reviewed checksum/architecture.");
  const body = [
    `images:`, `- location: ${yamlString(pathToFileURL(image.path).href)}`, `  arch: ${image.architecture}`, `  digest: ${yamlString(`sha256:${image.sha256}`)}`,
    `arch: ${image.architecture}`, `cpus: ${resources.cpus}`, `memory: ${yamlString(`${resources.memoryGiB}GiB`)}`, `disk: ${yamlString(`${resources.diskGiB}GiB`)}`,
    `mounts: []`, `networks: []`, `portForwards: []`, `provision: []`, `probes: []`, `containerd:`, `  system: false`, `  user: false`,
    `ssh:`, `  loadDotSSHPubKeys: false`, `  forwardAgent: false`, `  forwardX11: false`, `plain: true`, ``
  ].join("\n");
  await writeMutableFileIsolated(join(directory, "lima-create.yaml"), Buffer.from(body, "utf8"));
  return join(directory, "lima-create.yaml");
}

async function limaCommand(
  deps: EnvironmentAdapterDependencies,
  args: string[],
  cwd: string,
  limaHome: string,
  timeoutMs: number,
  signal: AbortSignal
): Promise<EnvironmentProcessReceipt> {
  const capability = (await environmentAdapterCapabilities(deps, true)).find(row => row.id === "lima");
  if (!capability?.available || !capability.executable) throw new Error(capability?.warning ?? "Lima is unavailable.");
  const ssh = await realpath("/usr/bin/ssh"), sshInfo = await lstat(ssh);
  if (!sshInfo.isFile() || sshInfo.isSymbolicLink() || (sshInfo.mode & 0o022) !== 0 || sshInfo.uid !== 0) throw new Error("A trusted system SSH executable is unavailable for the private Lima runtime.");
  const clean = { HOME: limaHome, LIMA_HOME: limaHome, PATH: "/usr/bin:/bin:/usr/sbin:/sbin", SSH: ssh, LANG: "C.UTF-8", LC_ALL: "C.UTF-8" };
  return deps.run("/usr/bin/env", ["-i", ...Object.entries(clean).map(([key, value]) => `${key}=${value}`), capability.executable, "--tty=false", ...args],
    cwd, timeoutMs, {}, signal, deps.onSpawn);
}

export function limaCreateArguments(record: ManagedEnvironmentRecord, configPath: string): string[] {
  if (!record.externalId) throw new Error("VM adapter identity is unavailable.");
  return ["create", `--name=${record.externalId}`, "--plain", "--mount-none", "--containerd=none", configPath];
}

export async function createLimaVm(deps: EnvironmentAdapterDependencies, record: ManagedEnvironmentRecord, directory: string, limaHome: string, signal: AbortSignal): Promise<EnvironmentProcessReceipt> {
  const config = join(limaHome, "lima-create.yaml"), info = await lstat(config);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("The reviewed local VM configuration is missing.");
  return limaCommand(deps, limaCreateArguments(record, config), directory, limaHome, 600_000, signal);
}
export async function loadLimaVm(deps: EnvironmentAdapterDependencies, record: ManagedEnvironmentRecord, directory: string, limaHome: string, signal: AbortSignal): Promise<EnvironmentProcessReceipt> {
  if (!record.externalId) throw new Error("VM adapter identity is unavailable.");
  return limaCommand(deps, ["start", record.externalId], directory, limaHome, 600_000, signal);
}
export async function runLimaVm(deps: EnvironmentAdapterDependencies, record: ManagedEnvironmentRecord, directory: string, limaHome: string, programValue: unknown, argumentsValue: unknown, timeoutMs: number, signal: AbortSignal): Promise<EnvironmentProcessReceipt> {
  if (!record.externalId || typeof programValue !== "string" || !/^\/[A-Za-z0-9_./+-]+$/.test(programValue) || programValue.includes("..")) throw new Error("VM program must be one exact absolute guest path.");
  const args = exactArguments(argumentsValue);
  return limaCommand(deps, ["shell", "--workdir=/tmp", record.externalId, programValue, ...args], directory, limaHome, timeoutMs, signal);
}
export async function stopLimaVm(deps: EnvironmentAdapterDependencies, record: ManagedEnvironmentRecord, directory: string, limaHome: string, signal: AbortSignal): Promise<EnvironmentProcessReceipt> {
  if (!record.externalId) throw new Error("VM adapter identity is unavailable.");
  return limaCommand(deps, ["stop", record.externalId], directory, limaHome, 120_000, signal);
}

export async function inspectLimaVm(deps: EnvironmentAdapterDependencies, record: ManagedEnvironmentRecord, directory: string, limaHome: string, signal: AbortSignal): Promise<"loaded" | "stopped" | "missing" | "unknown"> {
  if (!record.externalId) return "missing";
  const result = await limaCommand(deps, ["list", "--json", record.externalId], directory, limaHome, 15_000, signal);
  if (result.exitCode !== 0) return /no instance|not found/i.test(`${result.stdout}\n${result.stderr}`) ? "missing" : "unknown";
  const line = result.stdout.split(/\r?\n/).map(value => value.trim()).find(Boolean); if (!line) return "missing";
  try {
    const value = JSON.parse(line) as Record<string, unknown>;
    if (value.name !== record.externalId) return "unknown";
    const status = String(value.status).toLocaleLowerCase();
    return status === "running" ? "loaded" : status === "stopped" ? "stopped" : "unknown";
  } catch { return "unknown"; }
}
