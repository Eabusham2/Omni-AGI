import { createHash, randomUUID } from "node:crypto";
import { createReadStream, constants } from "node:fs";
import { access, lstat, mkdir, open, readFile, realpath, writeFile } from "node:fs/promises";
import { delimiter, join, resolve } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { assemblyArtifactPage, boundAssemblyOriginalId } from "./assemblyPortability";

export interface AssemblyProcessResult { exitCode: number; stdout: string; stderr: string; truncated: boolean }
export type AssemblyRunner = (command: string, args: string[], cwd: string, timeoutMs: number, environment?: Record<string, string>, signal?: AbortSignal, onSpawn?: (pid: number) => void) => Promise<AssemblyProcessResult>;
export interface AssemblyDependencies {
  root: string; platform?: NodeJS.Platform; arch?: string; run: AssemblyRunner;
  findExecutable?(name: string): Promise<string | undefined>;
  checkResources(bytes: number, purpose: string, childPids?: readonly number[]): Promise<void>;
  reserveWrite(bytes: number): Promise<void>;
}
interface AssemblyFile { name: string; bytes: number; sha256: string }
export interface AssemblyReceipt {
  format: "omni-assembly-receipt"; version: 1; artifactId: string; operation: "compile" | "run";
  platform: string; architecture: string; kind: string; state: string; directory: string; receiptPath: string;
  sourceSha256?: string; artifact?: AssemblyFile; partialFiles: AssemblyFile[]; processes: AssemblyProcessResult[];
  createdAt: string; error?: string; stdoutSha256?: string; stderrSha256?: string; outputHashScope?: "captured-bounded-output";
  commands?: Array<{ executable: string; arguments: string[] }>; sdkPath?: string;
}
export class AssemblyExecutionError extends Error {
  constructor(message: string, readonly receipt: AssemblyReceipt) { super(`${message} Retained receipt: ${receipt.receiptPath}`); this.name = "AssemblyExecutionError"; }
}
const hash = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");
const id = (value: unknown): string => { if (typeof value !== "string" || !/^[a-f0-9]{32}$/.test(value)) throw new Error("Invalid managed assembly artifact ID."); return value; };
const text = (value: unknown, label: string): string => { if (typeof value !== "string" || value.includes("\0") || /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(value)) throw new Error(`Invalid ${label}; exact UTF-8 text is required.`); return value; };
function object(value: unknown): Record<string, unknown> { if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Assembly arguments must be an object."); return value as Record<string, unknown>; }
function known(args: Record<string, unknown>, fields: string[]): void { if (Object.keys(args).some(key => !fields.includes(key))) throw new Error("Unsupported assembly operands; no argument was silently removed."); }
function timeout(value: unknown): number { if (value === undefined) return 60_000; if (!Number.isSafeInteger(value) || Number(value) < 100 || Number(value) > 600_000) throw new Error("Assembly timeout must be an integer from 100 to 600000 milliseconds."); return Number(value); }
async function directory(path: string): Promise<void> { const row = await lstat(path); if (!row.isDirectory() || row.isSymbolicLink()) throw new Error("Assembly output requires a real managed directory, not a link."); }
async function file(path: string): Promise<AssemblyFile> {
  const before = await lstat(path); if (!before.isFile() || before.isSymbolicLink()) throw new Error("Assembly source/output must be a regular file, not a link.");
  const digest = createHash("sha256"); for await (const chunk of createReadStream(path)) digest.update(chunk);
  const after = await lstat(path); if (before.dev !== after.dev || before.ino !== after.ino || before.size !== after.size || before.mtimeMs !== after.mtimeMs) throw new Error("Assembly file changed while hashing.");
  return { name: path.split(/[\\/]/).at(-1)!, bytes: after.size, sha256: digest.digest("hex") };
}
async function findExecutable(name: string): Promise<string | undefined> {
  const candidates = [...(process.platform === "darwin" && name === "clang" ? ["/Library/Developer/CommandLineTools/usr/bin/clang", "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang"] : []),
    ...(process.env.PATH ?? "").split(delimiter).filter(Boolean).map(entry => resolve(entry, name))];
  for (const path of candidates) {
    try { await access(path, process.platform === "win32" ? constants.F_OK : constants.X_OK); const actual = await realpath(path); if ((await lstat(actual)).isFile()) return actual; } catch { /* Try the next installed PATH entry; no setup/download. */ }
  }
  return undefined;
}
async function bounded(deps: AssemblyDependencies, command: string, args: string[], cwd: string, milliseconds: number, signal: AbortSignal, purpose: string): Promise<AssemblyProcessResult> {
  await deps.checkResources(65536, purpose); signal.throwIfAborted();
  const local = new AbortController(), children: number[] = []; let pressure: Error | undefined, checking = false;
  const abort = () => local.abort(signal.reason); signal.addEventListener("abort", abort, { once: true }); if (signal.aborted) abort();
  const check = async () => { if (checking || local.signal.aborted) return; checking = true;
    try { await deps.checkResources(65536, purpose, children); if (purpose !== "toolchain inspection") await deps.reserveWrite(8 * 1024 * 1024); }
    catch (error) { pressure = error instanceof Error ? error : new Error(String(error)); local.abort(pressure); }
    finally { checking = false; }
  };
  const timer = setInterval(() => { void check(); }, 250); timer.unref();
  try { return await deps.run(command, args, cwd, milliseconds, undefined, local.signal, pid => { children.push(pid); void check(); }); }
  catch (error) { throw pressure ?? error; }
  finally { clearInterval(timer); signal.removeEventListener("abort", abort); }
}
interface Toolchain { id: string; available: boolean; path?: string; version?: string; target?: string; probeError?: string }
async function installedSdk(deps: AssemblyDependencies, signal: AbortSignal): Promise<string | undefined> {
  if ((deps.platform ?? process.platform) !== "darwin") return undefined;
  const xcrun = await (deps.findExecutable ?? findExecutable)("xcrun"); if (!xcrun) return undefined;
  const result = await bounded(deps, xcrun, ["--show-sdk-path"], deps.root, 3000, signal, "toolchain inspection");
  if (result.exitCode !== 0) return undefined;
  const value = result.stdout.trim(); if (!value.startsWith("/") || value.includes("\0") || value.includes("\n")) throw new Error("Installed SDK discovery did not return one absolute directory.");
  const path = await realpath(value); await directory(path); return path;
}
async function toolchains(deps: AssemblyDependencies, signal: AbortSignal): Promise<Toolchain[]> {
  const platform = deps.platform ?? process.platform, found: Toolchain[] = [];
  for (const [name, binary] of [["clang", platform === "win32" ? "clang.exe" : "clang"], ["gcc", platform === "win32" ? "gcc.exe" : "gcc"], ["nasm", platform === "win32" ? "nasm.exe" : "nasm"], ["msvc", "ml64.exe"]]) {
    if (name === "msvc" && platform !== "win32") { found.push({ id: name!, available: false }); continue; }
    const path = await (deps.findExecutable ?? findExecutable)(binary!); if (!path) { found.push({ id: name!, available: false }); continue; }
    try { const version = await bounded(deps, path, name === "msvc" ? ["/?"] : ["--version"], deps.root, 3000, signal, "toolchain inspection");
      const target = ["clang", "gcc"].includes(name!) ? await bounded(deps, path, ["-dumpmachine"], deps.root, 3000, signal, "toolchain inspection") : undefined;
      found.push({ id: name!, available: true, path, version: version.stdout || version.stderr, ...(target ? { target: target.stdout.trim() } : {}) });
    } catch (error) { signal.throwIfAborted(); found.push({ id: name!, available: true, path, probeError: error instanceof Error ? error.message : String(error) }); }
  }
  return found;
}
async function partial(directoryPath: string, names: string[]): Promise<AssemblyFile[]> { const result: AssemblyFile[] = [];
  for (const name of new Set(names)) try { result.push(await file(join(directoryPath, name))); } catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; } return result;
}
async function saveReceipt(deps: AssemblyDependencies, receipt: AssemblyReceipt): Promise<void> { const body = JSON.stringify(receipt); await deps.reserveWrite(Buffer.byteLength(body)); await writeFile(receipt.receiptPath, body, { flag: "wx", mode: 0o600 }); }
function targetTriple(platform: string, architecture: string): string {
  const machine = architecture === "arm64" ? "aarch64" : architecture === "ia32" ? "i686" : "x86_64";
  return platform === "darwin" ? `${architecture === "arm64" ? "arm64" : machine}-apple-darwin` : platform === "win32" ? `${machine}-pc-windows-msvc` : `${machine}-unknown-linux-gnu`;
}
/** Exact operands, fresh managed outputs, ordinary native toolchain—not an OS sandbox. */
export async function executeAssembly(deps: AssemblyDependencies, action: string, argsValue: unknown, signal: AbortSignal): Promise<unknown> {
  const args = object(argsValue), platform = deps.platform ?? process.platform, hostArch = deps.arch ?? process.arch;
  signal.throwIfAborted(); await directory(deps.root);
  if (action === "inspect") { known(args, ["artifactId", "after", "limit"]);
    if (args.artifactId !== undefined && (args.after !== undefined || args.limit !== undefined)) throw new Error("Choose one artifact or a saved-artifact page.");
    if (args.artifactId !== undefined) { const root = join(deps.root, "assembly", id(args.artifactId)); await directory(join(deps.root, "assembly")); await directory(root);
      const stored = await lstat(join(root, "compile.json")).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
      if (!stored) return { artifactId: args.artifactId, state: "incomplete", nativeProcessResumable: false, currentFiles: await partial(root, ["source.s", "source.asm", "object.o", "object.obj", "program", "program.exe", "program.bin"]) };
      const info = await file(join(root, "compile.json")); if (info.bytes * 12 > Math.min(freemem(), getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 2) throw new Error("Assembly receipt awaits parse headroom.");
      const receipt = JSON.parse(await readFile(join(root, "compile.json"), "utf8")) as AssemblyReceipt; if (receipt.format !== "omni-assembly-receipt" || receipt.version !== 1) throw new Error("Unsupported assembly receipt format."); await boundAssemblyOriginalId(root, id(args.artifactId), receipt); return { artifactId: args.artifactId, receipt, currentFiles: await partial(root, ["source.s", "source.asm", "object.o", "object.obj", "program", "program.exe", "program.bin"]) }; }
    await deps.checkResources(65536, "toolchain inspection"); return { platform, architecture: hostArch, toolchains: await toolchains(deps, signal), sdkPath: await installedSdk(deps, signal), artifacts: await assemblyArtifactPage(deps.root, args.after as string | undefined, args.limit as number | undefined), nativeExecutionIsSandboxed: false, installationPerformed: false };
  }
  if (action === "compile") {
    known(args, ["source", "syntax", "kind", "toolchain", "architecture", "entryPoint", "libraries", "timeoutMs"]);
    const source = text(args.source, "assembly source"); if (!source.length) throw new Error("Enter assembly source."); const milliseconds = timeout(args.timeoutMs);
    const syntax = args.syntax ?? "gas", kind = args.kind ?? "executable", selected = args.toolchain ?? "auto", architecture = args.architecture ?? (syntax === "gas" ? hostArch : "x64");
    if (!["gas", "nasm", "masm"].includes(String(syntax)) || !["executable", "object", "flat-binary"].includes(String(kind)) || !["auto", "clang", "gcc", "nasm", "msvc"].includes(String(selected)) || !["x64", "arm64", "ia32"].includes(String(architecture))) throw new Error("Unsupported exact assembly syntax, output kind, toolchain or architecture.");
    if (kind === "flat-binary" && syntax !== "nasm" || syntax === "nasm" && (!["x64", "ia32"].includes(String(architecture)) || kind === "executable" && architecture !== hostArch) || syntax === "masm" && (platform !== "win32" || architecture !== "x64" || kind === "flat-binary")) throw new Error("This toolchain/output architecture combination is not supported; foreign NASM output may be compiled as an object, not silently linked for another host.");
    const libraries = args.libraries === undefined ? [] : args.libraries; if (!Array.isArray(libraries) || libraries.some(value => typeof value !== "string" || !/^[A-Za-z0-9_][A-Za-z0-9_.+-]*$/.test(value))) throw new Error("Libraries must be exact library names, not paths or flags.");
    const entry = args.entryPoint; if (entry !== undefined && (typeof entry !== "string" || !/^[A-Za-z_.$][A-Za-z0-9_.$]*$/.test(entry))) throw new Error("Entry point must be one exact symbol.");
    if (kind !== "executable" && (libraries.length || entry !== undefined)) throw new Error("Link operands apply only to executable output.");
    const payload = Buffer.byteLength(source) + Buffer.byteLength(JSON.stringify(args)); await deps.checkResources(payload * 12 + 65536, "assembly source"); await deps.reserveWrite(payload + 8 * 1024 * 1024);
    const installed = await toolchains(deps, signal), compatible = syntax === "gas" ? ["clang", "gcc"] : syntax === "nasm" ? ["nasm"] : ["msvc"];
    const compiler = installed.find(row => compatible.includes(row.id) && (selected === "auto" || selected === row.id) && row.available && !row.probeError);
    if (!compiler?.path) return { state: "toolchain-unavailable", platform, architecture, installed, installationPerformed: false };
    const sdkPath = await installedSdk(deps, signal), sdkArguments = sdkPath ? ["-isysroot", sdkPath] : [];
    if (compiler.id === "gcc" && architecture !== hostArch && !(["x64", "ia32"].includes(String(architecture)) && ["x64", "ia32"].includes(hostArch))) throw new Error("GCC cross-architecture assembly requires an explicitly installed matching compiler; no target was silently substituted.");
    const base = join(deps.root, "assembly"); await mkdir(base, { recursive: true, mode: 0o700 }); await directory(base); const artifactId = randomUUID().replaceAll("-", ""), root = join(base, artifactId); await mkdir(root, { mode: 0o700 });
    const sourceName = syntax === "gas" ? "source.s" : "source.asm", sourcePath = join(root, sourceName), objectName = platform === "win32" ? "object.obj" : "object.o", outputName = kind === "object" ? objectName : kind === "flat-binary" ? "program.bin" : platform === "win32" ? "program.exe" : "program";
    const receipt: AssemblyReceipt = { format: "omni-assembly-receipt", version: 1, artifactId, operation: "compile", platform, architecture: String(architecture), kind: String(kind), state: "compiling", directory: root, receiptPath: join(root, "compile.json"), sourceSha256: hash(source), partialFiles: [], processes: [], commands: [], ...(sdkPath ? { sdkPath } : {}), createdAt: new Date().toISOString() };
    await writeFile(sourcePath, source, { flag: "wx", mode: 0o600 }); await writeFile(join(root, "compile.intent.json"), JSON.stringify({ artifactId, sourceSha256: receipt.sourceSha256, sourceBytes: Buffer.byteLength(source), operands: args, compiler, platform, architecture, ...(sdkPath ? { sdkPath } : {}) }), { flag: "wx", mode: 0o600 });
    try {
      signal.throwIfAborted(); let argv: string[];
      if (compiler.id === "nasm") { const format = kind === "flat-binary" ? "bin" : platform === "darwin" ? architecture === "ia32" ? "macho32" : "macho64" : platform === "win32" ? architecture === "ia32" ? "win32" : "win64" : architecture === "ia32" ? "elf32" : "elf64";
        argv = ["-f", format, sourcePath, "-o", join(root, kind === "executable" ? objectName : outputName)]; }
      else if (compiler.id === "msvc") argv = ["/nologo", "/c", `/Fo${join(root, objectName)}`, sourcePath];
      else argv = [...sdkArguments, ...(architecture !== hostArch ? compiler.id === "clang" ? ["-target", targetTriple(platform, String(architecture))] : [architecture === "ia32" ? "-m32" : "-m64"] : []), "-x", "assembler", ...(kind === "object" ? ["-c"] : []), sourcePath, "-o", join(root, outputName), ...(entry ? [`-Wl,-e,${entry}`] : []), ...libraries.map(value => `-l${value}`)];
      receipt.commands!.push({ executable: compiler.path, arguments: [...argv] });
      receipt.processes.push(await bounded(deps, compiler.path, argv, root, milliseconds, signal, "assembly compile"));
      if (receipt.processes.at(-1)!.exitCode === 0 && kind === "executable" && ["nasm", "msvc"].includes(compiler.id)) {
        const linker = compiler.id === "msvc" ? await (deps.findExecutable ?? findExecutable)("link.exe") : installed.find(row => ["clang", "gcc"].includes(row.id) && row.available && !row.probeError)?.path;
        if (!linker) receipt.error = "Assembly object retained; no compatible installed linker is available.";
        else { const linkerArgs = compiler.id === "msvc" ? ["/NOLOGO", `/OUT:${join(root, outputName)}`, "/SUBSYSTEM:CONSOLE", ...(entry ? [`/ENTRY:${entry}`] : []), join(root, objectName), ...(libraries.length ? libraries : ["kernel32.lib"])] :
          [...sdkArguments, join(root, objectName), "-o", join(root, outputName), ...(entry ? [`-Wl,-e,${entry}`] : []), ...libraries.map(value => `-l${value}`)];
          receipt.commands!.push({ executable: linker, arguments: [...linkerArgs] }); receipt.processes.push(await bounded(deps, linker, linkerArgs, root, milliseconds, signal, "assembly link")); }
      }
      receipt.partialFiles = await partial(root, [sourceName, objectName, outputName]);
      const output = receipt.partialFiles.find(row => row.name === outputName);
      receipt.state = receipt.error || receipt.processes.some(row => row.exitCode !== 0) || !output ? "compile-failed" : "compiled"; if (receipt.state === "compiled") receipt.artifact = output;
      await saveReceipt(deps, receipt); return receipt;
    } catch (error) {
      receipt.state = signal.aborted ? "cancelled" : /timed out/i.test(String(error)) ? "timed-out" : "compile-failed"; receipt.error = error instanceof Error ? error.message : String(error); receipt.partialFiles = await partial(root, [sourceName, objectName, outputName]); await saveReceipt(deps, receipt); throw new AssemblyExecutionError(receipt.error, receipt);
    }
  }
  if (action === "run") {
    known(args, ["artifactId", "artifactSha256", "arguments", "timeoutMs"]); const artifactId = id(args.artifactId), expected = text(args.artifactSha256, "artifact hash"); if (!/^[a-f0-9]{64}$/.test(expected)) throw new Error("Supply the exact compiled artifact hash.");
    const argv = args.arguments ?? []; if (!Array.isArray(argv) || argv.some(value => typeof value !== "string")) throw new Error("Every native argument must be a string; none are truncated or removed."); for (const value of argv) text(value, "native argument"); const milliseconds = timeout(args.timeoutMs);
    const root = join(deps.root, "assembly", artifactId); await directory(join(deps.root, "assembly")); await directory(root);
    const metadata = await file(join(root, "compile.json")); await deps.checkResources(metadata.bytes * 12 + Buffer.byteLength(JSON.stringify(argv)) * 12 + 65536, "assembly receipt");
    const compiled = object(JSON.parse(await readFile(join(root, "compile.json"), "utf8"))) as unknown as AssemblyReceipt;
    if (compiled.format !== "omni-assembly-receipt" || compiled.version !== 1 || compiled.state !== "compiled" || compiled.kind !== "executable" || !compiled.artifact) throw new Error("Only a successfully compiled executable may be run.");
    await boundAssemblyOriginalId(root, artifactId, compiled);
    if (compiled.platform !== platform || compiled.architecture !== hostArch) throw new Error("Native execution on this platform/architecture is unsupported; no emulator or loader was substituted.");
    const outputName = platform === "win32" ? "program.exe" : "program"; if (compiled.artifact.name !== outputName) throw new Error("Compiled artifact inventory is unsafe.");
    const binary = await file(join(root, outputName)); if (binary.sha256 !== expected || compiled.artifact.sha256 !== expected) throw new Error("Compiled artifact hash changed; it was not executed.");
    const sourceName = compiled.partialFiles.find(row => ["source.s", "source.asm"].includes(row.name))?.name; if (!sourceName || (await file(join(root, sourceName))).sha256 !== compiled.sourceSha256) throw new Error("Assembly source changed after compilation.");
    await deps.reserveWrite(8 * 1024 * 1024); const executionId = randomUUID().replaceAll("-", ""), receipt: AssemblyReceipt = { ...compiled, artifactId, operation: "run", state: "running", directory: root, receiptPath: join(root, `run-${executionId}.json`), processes: [], commands: [{ executable: join(root, outputName), arguments: [...argv] }], createdAt: new Date().toISOString() };
    await writeFile(join(root, `run-${executionId}.intent.json`), JSON.stringify({ executionId, artifactId, artifactSha256: expected, arguments: argv, timeoutMs: milliseconds }), { flag: "wx", mode: 0o600 });
    try { if (platform !== "win32") { const executable = await open(join(root, outputName), constants.O_RDONLY | constants.O_NOFOLLOW); try { const current = await executable.stat(); if (!current.isFile()) throw new Error("Native executable is no longer a regular file."); await executable.chmod(0o700); } finally { await executable.close(); } }
      const result = await bounded(deps, join(root, outputName), argv, root, milliseconds, signal, "native assembly execution"); receipt.processes.push(result); receipt.state = "exited"; receipt.stdoutSha256 = hash(result.stdout); receipt.stderrSha256 = hash(result.stderr); receipt.outputHashScope = "captured-bounded-output";
      if ((await file(join(root, outputName))).sha256 !== expected) receipt.error = "The executed native program changed its binary; future execution requires a new inspected artifact.";
      await saveReceipt(deps, receipt); return receipt;
    } catch (error) { receipt.state = signal.aborted ? "cancelled" : /timed out/i.test(String(error)) ? "timed-out" : "run-failed"; receipt.error = error instanceof Error ? error.message : String(error); await saveReceipt(deps, receipt); throw new AssemblyExecutionError(receipt.error, receipt); }
  }
  throw new Error("Unknown assembly operation.");
}
