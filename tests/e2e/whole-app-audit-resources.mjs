/** UI-audit admission only. Reads OS pressure/space; never controls other apps. */
import { appendFileSync, existsSync, mkdirSync, statfsSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { join } from "node:path";

export class AuditResourcePauseError extends Error {
  constructor(snapshot) { super("UI audit paused at its existing disk reserve or OS memory-pressure boundary."); this.name = "AuditResourcePauseError"; this.snapshot = snapshot; }
}
/** Latest user policy admits normal and warning pressure (swap may be used),
 * but never critical/unknown pressure or the existing free-disk boundary. */
export function permitsAuditResources(availableDiskBytes, osMemoryPressureLevel) {
  return Number.isFinite(availableDiskBytes) && availableDiskBytes >= 20 * 1024 ** 3 && [1, 2].includes(osMemoryPressureLevel);
}
export function auditResourceSnapshot(repository) {
  const capturedAt = new Date().toISOString(), diskReserveBytes = 20 * 1024 ** 3;
  try {
    const disk = statfsSync(repository), availableDiskBytes = disk.bavail * disk.bsize;
    if (process.platform !== "darwin") return { capturedAt, diskReserveBytes, availableDiskBytes, platform: process.platform,
      canContinue: false, reason: "This local full-cohort guard requires Darwin's OS memory-pressure indicator; no unsupported safe-memory claim is made." };
    const osMemoryPressureLevel = Number(execFileSync("sysctl", ["-n", "kern.memorystatus_vm_pressure_level"], { encoding: "utf8", timeout: 5000 }).trim());
    const swapUsageRaw = execFileSync("sysctl", ["vm.swapusage"], { encoding: "utf8", timeout: 5000 }).trim();
    return { capturedAt, diskReserveBytes, availableDiskBytes, platform: process.platform, osMemoryPressureLevel, swapUsageRaw,
      allowedMemoryPressureLevels: [1, 2], memoryPressureWarning: osMemoryPressureLevel === 2, criticalMemoryPressure: osMemoryPressureLevel === 4,
      admissionPolicy: "warning-pressure-and-swap-permitted; critical-or-unknown-pressure-pauses",
      canContinue: permitsAuditResources(availableDiskBytes, osMemoryPressureLevel) };
  } catch (error) { return { capturedAt, diskReserveBytes, platform: process.platform, canContinue: false,
    reason: error instanceof Error ? error.message : String(error) }; }
}
export function assertAuditResources(repository, output, label) {
  mkdirSync(output, { recursive: true });
  const snapshot = auditResourceSnapshot(repository), record = { format: "omni-ui-resource-gate", version: 1, label, ...snapshot };
  appendFileSync(join(output, "resource-checks.jsonl"), `${JSON.stringify(record)}\n`, { flush: true });
  if (snapshot.canContinue) return snapshot;
  const pause = join(output, "RESOURCE_PAUSE.json");
  if (!existsSync(pause)) {
    try { writeFileSync(pause, `${JSON.stringify({ state: "paused-before-next-cell-or-batch", ...record }, null, 2)}\n`, { flag: "wx", mode: 0o600 }); }
    catch (error) { if (error.code !== "EEXIST") throw error; }
  }
  throw new AuditResourcePauseError(snapshot);
}
