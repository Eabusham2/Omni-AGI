import { availableParallelism, arch, cpus, loadavg, release, uptime } from "node:os";
import { isAbsolute } from "node:path";
import { measuredSystemMemory } from "./resourcePlanner";
import { readDiskSpace, type DiskSpaceReader } from "./diskSpace";
import { readSystemTemperatures } from "./systemTemperatureReader";
import type { CachedSystemDeviceTelemetry, SystemStatsFamilyMeasurement, SystemStatsHostReading, SystemStatsProcessReading, SystemStatsReading, SystemStatsSnapshot, SystemTemperatureReading } from "../shared/systemStats";

export type { SystemStatsSnapshot } from "../shared/systemStats";

export interface SystemStatsRequest {
  /** Main supplies this exact owner directory; it is never a neural tool argument. */
  dataDirectory: string;
  signal?: AbortSignal;
  isCurrent?: () => boolean;
  requireCurrent?: () => void | Promise<void>;
  cachedDeviceTelemetry?: () => CachedSystemDeviceTelemetry | undefined;
  /** Main-owned PID measurement only; neither IDs nor commands come from tool arguments. */
  managedFamilyMeasurement?: (signal?: AbortSignal) => Promise<SystemStatsFamilyMeasurement>;
}

/** Dependency injection is for owned inert fixtures, never renderer input. */
export interface SystemStatsReaders {
  platform?: NodeJS.Platform;
  now?: () => Date;
  host?: () => SystemStatsHostReading;
  memory?: typeof measuredSystemMemory;
  loadAverage?: () => number[];
  process?: () => SystemStatsProcessReading;
  storage?: DiskSpaceReader;
  temperatures?: (signal?: AbortSignal) => Promise<SystemTemperatureReading | undefined>;
}

const READ_DEADLINE_MS = 3_000;
class InvalidReading extends Error {}
class ReadDeadline extends Error {}
const count = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
const nonnegative = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0;
function check(request: SystemStatsRequest): void {
  if (request.signal?.aborted || request.isCurrent && !request.isCurrent()) throw new DOMException("System snapshot cancelled or owner changed.", "AbortError");
}
function readWithinDeadline<T>(request: SystemStatsRequest, read: (signal: AbortSignal) => T | Promise<T>): Promise<T> {
  check(request);
  return new Promise<T>((resolve, reject) => {
    let settled = false; const reading = new AbortController();
    const finish = (failure: boolean, value: unknown): void => {
      if (settled) return; settled = true; clearTimeout(timer); request.signal?.removeEventListener("abort", aborted);
      if (failure) reject(value); else resolve(value as T);
    };
    const aborted = () => { reading.abort(); finish(true, new DOMException("System snapshot cancelled.", "AbortError")); };
    const timer = setTimeout(() => { reading.abort(); finish(true, new ReadDeadline()); }, READ_DEADLINE_MS);
    request.signal?.addEventListener("abort", aborted, { once: true });
    if (request.signal?.aborted) { aborted(); return; }
    Promise.resolve().then(() => { check(request); return read(reading.signal); }).then(value => finish(false, value), error => finish(true, error));
  });
}

/** Read-only metadata: never engine discovery, model admission, capture or benchmarking. */
export async function readSystemStatsSnapshot(request: SystemStatsRequest, readers: SystemStatsReaders = {}): Promise<SystemStatsSnapshot> {
  if (!request || typeof request.dataDirectory !== "string" || !isAbsolute(request.dataDirectory) || request.dataDirectory.includes("\0")) {
    throw new Error("System snapshot requires a main-owned absolute data directory.");
  }
  check(request);
  if (request.requireCurrent) await readWithinDeadline(request, request.requireCurrent);
  const now = () => (readers.now?.() ?? new Date()).toISOString();
  const platform = readers.platform ?? process.platform;
  const capture = async <T>(source: string, read: (signal: AbortSignal) => T | Promise<T>): Promise<SystemStatsReading<T>> => {
    let result: SystemStatsReading<T>;
    try { const value = await readWithinDeadline(request, read); result = { state: "observed", observedAt: now(), source, value }; }
    catch (error) {
      check(request);
      if (error instanceof DOMException && error.name === "AbortError") throw error;
      const name = error instanceof Error && /^[A-Za-z][A-Za-z0-9]{0,79}$/.test(error.name) ? error.name : "Error";
      result = { state: "unavailable", reason: error instanceof ReadDeadline ? "timed-out" : error instanceof InvalidReading ? "invalid-reading" : "measurement-failed", causeClass: name };
    }
    check(request); return result;
  };
  const [host, memory, loadAverage, mainProcess, storage, devices, temperatures, managedFamily] = await Promise.all([
    capture("node-os-passive", () => {
      const cpu = readers.host ? undefined : cpus();
      const value = readers.host?.() ?? { platform, architecture: arch(), osRelease: release(), uptimeSeconds: uptime(),
        logicalProcessors: cpu!.length, availableParallelism: availableParallelism(), processorModel: cpu!.find(row => row.model.trim())?.model ?? null };
      if (!count(value.logicalProcessors) || value.logicalProcessors < 1 || !count(value.availableParallelism) || value.availableParallelism < 1 || !nonnegative(value.uptimeSeconds)) throw new InvalidReading();
      return value;
    }),
    capture("existing-os-physical-memory-reader", async () => {
      const value = await (readers.memory ?? measuredSystemMemory)();
      if (!count(value.totalMemoryBytes) || !value.totalMemoryBytes || !count(value.availableMemoryBytes) || value.availableMemoryBytes > value.totalMemoryBytes) throw new InvalidReading();
      return { totalMemoryBytes: value.totalMemoryBytes, availableMemoryBytes: value.availableMemoryBytes };
    }),
    platform === "win32" ? Promise.resolve({ state: "unavailable", reason: "unsupported" } as const) : capture("node-os-loadavg", () => {
      const value = (readers.loadAverage ?? loadavg)();
      if (value.length !== 3 || !value.every(nonnegative)) throw new InvalidReading();
      return { oneMinute: value[0]!, fiveMinutes: value[1]!, fifteenMinutes: value[2]!, unit: "runnable-or-waiting-system-tasks" as const };
    }),
    capture("node-current-process-counters", () => {
      const value = readers.process?.() ?? (() => {
        const memory = process.memoryUsage(), cpu = process.cpuUsage();
        return { scope: "current-main-process-only" as const, pid: process.pid, uptimeSeconds: process.uptime(),
          residentBytes: memory.rss, heapTotalBytes: memory.heapTotal, heapUsedBytes: memory.heapUsed, externalBytes: memory.external,
          cpuTimeMicroseconds: { user: cpu.user, system: cpu.system } };
      })();
      if (value.scope !== "current-main-process-only" || !count(value.pid) || value.pid < 1 || !nonnegative(value.uptimeSeconds) ||
        ![value.residentBytes, value.heapTotalBytes, value.heapUsedBytes, value.externalBytes, value.cpuTimeMicroseconds.user, value.cpuTimeMicroseconds.system].every(count)) throw new InvalidReading();
      return value;
    }),
    capture("existing-data-volume-statfs", async () => {
      const value = await (readers.storage ?? readDiskSpace)(request.dataDirectory);
      if (!count(value.diskTotalBytes) || !value.diskTotalBytes || !count(value.diskFreeBytes) || value.diskFreeBytes > value.diskTotalBytes) throw new InvalidReading();
      return { scope: "brain-data-volume" as const, diskTotalBytes: value.diskTotalBytes, diskFreeBytes: value.diskFreeBytes };
    }),
    request.cachedDeviceTelemetry ? capture("cached-device-topology-no-refresh", () => {
      const cached = request.cachedDeviceTelemetry!();
      if (!cached) return undefined;
      const age = Date.parse(now()) - Date.parse(cached.measuredAt);
      if (!nonnegative(age)) throw new InvalidReading();
      return { measuredAt: cached.measuredAt, cached: true as const, ageMilliseconds: age, devices: cached.devices.map(device => {
        if (![device.totalBytes, device.availableBytes, device.allocatedBytes, device.reservedBytes].every(value => value === null || count(value))) throw new InvalidReading();
        return { id: device.id, label: device.label, kind: device.kind, backend: device.backend, memoryGroupId: device.memoryGroupId,
          totalBytes: device.totalBytes, availableBytes: device.availableBytes, allocatedBytes: device.allocatedBytes,
          reservedBytes: device.reservedBytes, unifiedMemory: device.unifiedMemory };
      }) };
    }) : Promise.resolve({ state: "unavailable", reason: "not-observed" } as const),
    capture("passive-linux-sysfs-thermal-zones", signal => readers.temperatures ? readers.temperatures(signal) : readSystemTemperatures(platform, signal)),
    request.managedFamilyMeasurement ? capture("existing-owned-process-residency-reader", async signal => {
      const value = await request.managedFamilyMeasurement!(signal);
      if (!count(value.residentBytes) || !count(value.sampledProcessCount) || value.sampledProcessCount < 1 ||
        !count(value.requestedProcessCount) || value.requestedProcessCount < value.sampledProcessCount) throw new InvalidReading();
      return { residentBytes: value.residentBytes, sampledProcessCount: value.sampledProcessCount, requestedProcessCount: value.requestedProcessCount,
        scope: "main-owned-managed-process-rss-sum" as const, sharedPagesMayBeCountedMoreThanOnce: true as const };
    }) : Promise.resolve({ state: "unavailable", reason: "not-observed" } as const)
  ]);
  check(request);
  if (request.requireCurrent) await readWithinDeadline(request, request.requireCurrent);
  check(request);
  return { format: "omni-system-stats", version: 1, observedAt: now(),
    scope: { readOnly: true, workerStartedBySnapshot: false, modelLoadedBySnapshot: false, benchmarkPerformedBySnapshot: false, deviceCapturePerformedBySnapshot: false },
    host, memory, loadAverage, process: mainProcess, managedFamily, storage,
    devices: devices.state === "observed" && devices.value === undefined ? { state: "unavailable", reason: "not-observed" } : devices as SystemStatsSnapshot["devices"],
    sensors: { temperatures: temperatures.state === "observed" && temperatures.value === undefined ? { state: "unavailable", reason: "not-observed" } : temperatures as SystemStatsSnapshot["sensors"]["temperatures"] } };
}
