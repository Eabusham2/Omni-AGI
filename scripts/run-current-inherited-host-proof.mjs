/** One externally bounded opt-in host proof. Retains private state on every outcome. */
import { spawn, execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdtemp, statfs, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const run = promisify(execFile), source = process.env.OMNI_CURRENT_INHERITED_SOURCE;
if (!source || !/^omni-connected-execution-proof-/.test(resolve(source).split("/").at(-1))) {
  throw new Error("Set OMNI_CURRENT_INHERITED_SOURCE to the retained private trained fixture; no model is created.");
}
const root = await mkdtemp(join(tmpdir(), "omni-connected-current-seal-host-"));
const started = performance.now(), limits = { wallSeconds: 120, cooperativeSeconds: 110, familyRssBytes: 768 * 1024 ** 2,
  privateAllocatedBytes: 512 * 1024 ** 2, freeDiskBytes: 20 * 1024 ** 3 };
let reason, peakRss = 0, peakPrivate = 0, minimumFree = Infinity, checking = false, finished = false;
const workerPids = new Set(), samples = [], stages = [];
// Refuse before starting any host/native child, not merely on the first monitor
// tick. Keep this private preparation directory and a truthful refusal receipt.
const initialDisk = await statfs(root), initialFree = initialDisk.bavail * initialDisk.bsize;
if (!Number.isFinite(initialFree) || initialFree < limits.freeDiskBytes) {
  await writeFile(join(root, "bounded-attempt.json"), JSON.stringify({
    format: "omni-bounded-current-inherited-host-attempt", version: 1, root, source: resolve(source), limits,
    elapsedSeconds: (performance.now() - started) / 1000, terminal: { notSpawned: true },
    reason: "pre-launch-disk-floor", complete: false,
    minimumFreeDiskBytes: Number.isFinite(initialFree) ? initialFree : null, workerPids: [], stages, samples
  }, null, 2), { flag: "wx", mode: 0o600 });
  throw new Error(`Host/native proof not started: the verified 20 GiB free floor is unavailable. Receipt: ${root}`);
}
minimumFree = initialFree;
samples.push({ elapsedSeconds: (performance.now() - started) / 1000, rssBytes: process.memoryUsage().rss,
  privateBytes: 0, freeDiskBytes: initialFree, phase: "before-spawn" });
const child = spawn(process.execPath, ["node_modules/vitest/vitest.mjs", "run", "tests/currentInheritedNativeHost.test.ts", "--maxWorkers=1"], {
  cwd: process.cwd(), env: { ...process.env, OMNI_CURRENT_INHERITED_SOURCE: resolve(source), OMNI_CURRENT_INHERITED_HOST_ROOT: root,
    OMNI_CURRENT_INHERITED_HOST_PROOF: "1", OMNI_CURRENT_INHERITED_EXTERNAL_CAP: "120s-768MiB-512MiB-20GiB" },
  stdio: ["ignore", "pipe", "pipe"], shell: false
});
process.stdout.write(`Owned current inherited host proof: ${root}\n`);
let stdout = "", stderr = "", pending = "";
child.stdout.on("data", bytes => {
  process.stdout.write(bytes); stdout += bytes; pending += bytes;
  let end;
  while ((end = pending.indexOf("\n")) >= 0) {
    const line = pending.slice(0, end); pending = pending.slice(end + 1);
    const at = line.indexOf("CURRENT_INHERITED_PROOF ");
    if (at < 0) continue;
    try { const stage = JSON.parse(line.slice(at + "CURRENT_INHERITED_PROOF ".length)); stages.push(stage);
      if (Number.isSafeInteger(stage.workerPid) && stage.workerPid > 1) workerPids.add(stage.workerPid); } catch { /* Log remains exact. */ }
  }
});
child.stderr.on("data", bytes => { process.stderr.write(bytes); stderr += bytes; });

async function family() {
  const { stdout: output } = await run("/bin/ps", ["-axo", "pid=,ppid=,rss="], { maxBuffer: 1024 * 1024, timeout: 1500 });
  const rows = output.trim().split("\n").map(line => line.trim().split(/\s+/).map(Number));
  const ids = new Set([child.pid]); let changed = true;
  while (changed) { changed = false; for (const [pid, parent] of rows) if (ids.has(parent) && !ids.has(pid)) { ids.add(pid); changed = true; } }
  const owned = rows.filter(([pid]) => ids.has(pid));
  return { ids, rows: owned, bytes: rows.filter(([pid]) => pid === process.pid || ids.has(pid)).reduce((sum, row) => sum + row[2] * 1024, 0) };
}
async function cooperative(label) {
  reason ??= label;
  try { const current = await family(); for (const pid of workerPids) if (current.ids.has(pid)) process.kill(pid, "SIGUSR1"); } catch { /* Hard cap still owns termination. */ }
}
async function terminate(label) {
  reason ??= label;
  try {
    const current = await family();
    // Only currently verified descendants of this exact spawned proof. Never
    // a broad name match, user application, or stale recycled PID inventory.
    for (const [pid] of current.rows.reverse()) if (pid !== child.pid) try { process.kill(pid, "SIGKILL"); } catch {}
  } finally { try { child.kill("SIGKILL"); } catch {} }
}
const monitor = setInterval(async () => {
  if (checking || finished) return; checking = true;
  try {
    const current = await family(), disk = await statfs(root), free = disk.bavail * disk.bsize;
    const { stdout: allocated } = await run("/usr/bin/du", ["-sk", root], { maxBuffer: 4096, timeout: 2000 });
    const bytes = Number(allocated.trim().split(/\s+/, 1)[0]) * 1024;
    if (!Number.isFinite(bytes)) throw new Error("private footprint could not be measured");
    peakRss = Math.max(peakRss, current.bytes); peakPrivate = Math.max(peakPrivate, bytes); minimumFree = Math.min(minimumFree, free);
    samples.push({ elapsedSeconds: (performance.now() - started) / 1000, rssBytes: current.bytes, privateBytes: bytes, freeDiskBytes: free });
    if (current.bytes > limits.familyRssBytes || bytes > limits.privateAllocatedBytes || free < limits.freeDiskBytes) {
      await terminate("resource-cap");
    }
  } catch (error) { if (!finished) await terminate(`unverified-resource-monitor:${error.message}`); }
  finally { checking = false; }
}, 500);
const soft = setTimeout(() => { void cooperative("110-second-safe-boundary-request"); }, 110_000);
// Begin verified-family termination with two seconds left for the bounded ps
// read; do not spend the final wall budget waiting for cooperative teardown.
const hard = setTimeout(() => { void terminate("120-second-hard-wall-cap"); }, 118_000);
process.on("SIGINT", () => { void cooperative("user-interrupted"); setTimeout(() => { void terminate("user-interrupted"); }, 3000).unref(); });
const terminal = await new Promise(resolve => { child.once("exit", (code, signal) => resolve({ code, signal })); child.once("error", error => resolve({ error: error.message })); });
finished = true; clearInterval(monitor); clearTimeout(soft); clearTimeout(hard);
while (checking) await new Promise(resolve => setTimeout(resolve, 10));
const complete = stages.some(stage => stage.stage === "current-host-worker-proof-complete");
const receipt = { format: "omni-bounded-current-inherited-host-attempt", version: 1, root, source: resolve(source), limits,
  elapsedSeconds: (performance.now() - started) / 1000, terminal, reason: reason ?? null, complete,
  peakRssBytes: peakRss, peakPrivateAllocatedBytes: peakPrivate, minimumFreeDiskBytes: Number.isFinite(minimumFree) ? minimumFree : null,
  workerPids: [...workerPids], stages, samples };
await writeFile(join(root, "bounded-attempt.json"), JSON.stringify(receipt, null, 2), { flag: "wx", mode: 0o600 });
await writeFile(join(root, "stdout.log"), stdout, { flag: "wx", mode: 0o600 });
await writeFile(join(root, "stderr.log"), stderr, { flag: "wx", mode: 0o600 });
process.stdout.write(`Retained bounded attempt: ${root}\n`);
process.exitCode = terminal.code === 0 && complete && !reason ? 0 : 1;
