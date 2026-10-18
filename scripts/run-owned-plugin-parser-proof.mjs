/** One owned non-neural Electron run with an outside wall/RSS watchdog. */
import { spawn, execFile } from "node:child_process";
import { promisify } from "node:util";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, statfs, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
if (process.argv.slice(2).some(value => value !== "--reviewed-availability-retry")) throw new Error("Only the exact approved affected retry flag is supported.");
const build = process.argv.includes("--reviewed-availability-retry") ? ".runtime/plugin-runtime-parser-owned-retry" : ".runtime/plugin-runtime-parser-owned";
const run = promisify(execFile), compiled = resolve(`${build}/main.cjs`), preload = resolve(`${build}/preload.cjs`);
const directory = await mkdtemp(join(tmpdir(), "omni-runtime-parser-owned-")), report = join(directory, "app-report.json"), started = performance.now(), limits = { wallMs: 20000, familyRssBytes: 768 * 1024 ** 2, freeDiskBytes: 20 * 1024 ** 3 };
await mkdir(join(directory, "electron-profile"), { mode: 0o700 }); const initialDisk = await statfs(directory); if (initialDisk.bavail * initialDisk.bsize < limits.freeDiskBytes) throw new Error("Owned parser run not started:20GiB free reserve unavailable.");
const sourcePaths = ["tests/e2e/plugin-runtime-parser-owned-main.ts", "src/main/pluginRuntimeService.ts", "src/main/pluginRuntimeStreamParser.ts", "src/main/pluginRuntimeProcess.ts", "src/main/pluginRuntimeBootstrap.ts", "src/main/pluginRuntimeResources.ts", "src/main/pluginRuntimeStore.ts", "src/main/pluginRuntimeValidation.ts", "src/preload/pluginRuntime.ts", "src/shared/pluginRuntime.ts", "src/shared/plugins.ts", "src/main/resourcePlanner.ts", "src/main/automationResources.ts", "src/main/index.ts", "src/shared/types.ts"];
const hashes = async () => Object.fromEntries(await Promise.all([...sourcePaths, compiled, preload].map(async path => [path, createHash("sha256").update(await readFile(path)).digest("hex")]))), before = await hashes();
const env = { ...process.env, OMNI_OWNED_PARSER_ROOT: directory, OMNI_OWNED_PARSER_PRELOAD: preload, OMNI_OWNED_PARSER_REPORT: report }; delete env.ELECTRON_RUN_AS_NODE;
const child = spawn(resolve("node_modules/.bin/electron"), [compiled], { env, shell: false, detached: true, stdio: ["ignore", "pipe", "pipe"] });
let stdout = "", stderr = "", reason, peak = 0, checking = false, finished = false; const samples = [];
child.stdout.on("data", bytes => { stdout += bytes; process.stdout.write(bytes); }); child.stderr.on("data", bytes => { stderr += bytes; });
process.stdout.write(`Owned non-neural parser fixture: ${directory}\n`);
async function family() { const { stdout: text } = await run("/bin/ps", ["-axo", "pid=,ppid=,rss="], { timeout: 750, maxBuffer: 1024 * 1024 }); const rows = text.trim().split("\n").map(line => line.trim().split(/\s+/).map(Number)), ids = new Set([child.pid]); let changed = true;
  while (changed) { changed = false; for (const [pid, parent] of rows) if (ids.has(parent) && !ids.has(pid)) { ids.add(pid); changed = true; } } const owned = rows.filter(([pid]) => ids.has(pid));
  return { rows: owned, bytes: rows.filter(([pid]) => pid === process.pid || ids.has(pid)).reduce((total, row) => total + row[2] * 1024, 0) }; }
async function kill(label) { reason ??= label; try { const current = await family(); for (const [pid] of current.rows.reverse()) try { process.kill(pid, "SIGKILL"); } catch {} } finally { try { child.kill("SIGKILL"); } catch {} } }
const timer = setTimeout(() => { void kill("20-second-hard-wall-watchdog"); }, 19000), monitor = setInterval(async () => { if (finished || checking) return; checking = true; try { const current = await family(), disk = await statfs(directory), free = disk.bavail * disk.bsize; peak = Math.max(peak, current.bytes); samples.push({ elapsedMs: performance.now() - started, familyRssBytes: current.bytes, freeDiskBytes: free });
    if (current.bytes > limits.familyRssBytes || free < limits.freeDiskBytes) await kill("resource-envelope"); } catch (error) { if (!finished) await kill(`resource-monitor-unverified:${error.message}`); } finally { checking = false; } }, 100);
const terminal = await new Promise(resolveExit => { child.once("exit", (code, signal) => resolveExit({ code, signal })); child.once("error", error => resolveExit({ error: error.message })); }); finished = true; clearTimeout(timer); clearInterval(monitor); while (checking) await new Promise(resolveWait => setTimeout(resolveWait, 5));
let actual; try { actual = JSON.parse(await readFile(report, "utf8")); } catch {}
const after = await hashes(), preservedSources = JSON.stringify(before) === JSON.stringify(after), complete = terminal.code === 0 && actual?.passed === true && !reason && preservedSources;
const receipt = { format: "omni-owned-runtime-parser-watchdog", version: 1, complete, directory, limits, terminal, reason: reason ?? null, elapsedMs: performance.now() - started, peakSampledFamilyRssBytes: peak, preservedSources, before, after, samples, appReport: actual ?? null };
await writeFile(join(directory, "watchdog-report.json"), JSON.stringify(receipt, null, 2), { flag: "wx", mode: 0o600 }); await writeFile(join(directory, "stdout.log"), stdout, { flag: "wx", mode: 0o600 }); await writeFile(join(directory, "stderr.log"), stderr, { flag: "wx", mode: 0o600 });
process.stdout.write(`Retained parser evidence: ${directory}\n`); process.exitCode = complete ? 0 : 1;
