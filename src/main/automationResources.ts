import { execFile } from "node:child_process";
import { freemem } from "node:os";
import { promisify } from "node:util";
import { getHeapStatistics } from "node:v8";

const execute = promisify(execFile);
/** Main-owned PIDs only; no user string is interpolated into a host command. */
export async function automationResidentBytes(pids: readonly number[], platform: NodeJS.Platform = process.platform): Promise<number> {
  const ids = [...new Set(pids)];
  if (!ids.length || ids.some(id => !Number.isSafeInteger(id) || id < 1)) throw new Error("Host residency requires actual owned process IDs.");
  const { stdout } = platform === "win32" ? await execute("powershell.exe", ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
    `$ids=@(${ids.join(",")}); Get-Process -Id $ids -ErrorAction SilentlyContinue | ForEach-Object { '{0} {1}' -f $_.Id,$_.WorkingSet64 }`], { timeout: 3000, maxBuffer: 65536 })
    : await execute("ps", ["-o", "pid=,rss=", "-p", ids.join(",")], { timeout: 3000, maxBuffer: 65536 });
  const observed = new Map<number, number>();
  for (const line of stdout.trim().split(/\r?\n/)) {
    if (!line.trim()) continue;
    const parts = /^(\d+)\s+(\d+)$/.exec(line.trim()); if (!parts) throw new Error("Host RSS measurement returned an unsupported record.");
    const id = Number(parts[1]), bytes = Number(parts[2]) * (platform === "win32" ? 1 : 1024);
    if (!ids.includes(id) || !Number.isSafeInteger(bytes) || bytes < 0 || observed.has(id)) throw new Error("Host RSS measurement lost its owner."); observed.set(id, bytes);
  }
  for (const id of ids) if (!observed.has(id)) {
    try { process.kill(id, 0); throw new Error("A live owned process has no residency measurement."); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error; }
  }
  return [...observed.values()].reduce((sum, value) => sum + value, 0);
}
/** Cooperative measured admission, not a claim of an OS hard RSS sandbox. */
export function requireAutomationHeadroom(bytes: number, readings: { residentBytes: number; ceilingBytes: number; freeBytes?: number; heapAvailableBytes?: number }): void {
  if (![bytes, readings.residentBytes, readings.ceilingBytes].every(value => Number.isSafeInteger(value) && value >= 0)) throw new Error("Action resource measurements are unavailable.");
  const free = readings.freeBytes ?? freemem(), heap = readings.heapAvailableBytes ?? getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed;
  const reserve = Math.min(64 * 1024 * 1024, Math.floor(Math.max(0, free) / 16));
  if (readings.residentBytes + bytes > readings.ceilingBytes || bytes + reserve > free || bytes + reserve > heap) throw new Error("Action execution is paused before exceeding selected app RAM or physical transfer headroom. No next effect was dispatched.");
}
