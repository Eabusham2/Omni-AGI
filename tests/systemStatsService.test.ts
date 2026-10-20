/** Inert reader/cache fixtures only; no worker, model, OS sensor, benchmark or provider. */
import { afterEach, expect, it, vi } from "vitest";
import { readSystemStatsSnapshot, type SystemStatsReaders } from "../src/main/systemStatsService";
import { DeviceResourceService } from "../src/main/deviceResourceService";
import type { DeviceResourceTopology } from "../src/shared/deviceResources";
import type { CachedSystemDeviceTelemetry } from "../src/shared/systemStats";
import { readSystemTemperatures } from "../src/main/systemTemperatureReader";

const observedAt = "2026-10-20T12:00:00.000Z";
function fixtures(): Required<SystemStatsReaders> {
  return { platform: "linux", now: () => new Date(observedAt),
    host: vi.fn(() => ({ platform: "linux", architecture: "arm64", osRelease: "fixture", uptimeSeconds: 300,
      logicalProcessors: 8, availableParallelism: 6, processorModel: "Fixture CPU" })),
    memory: vi.fn(async () => ({ totalMemoryBytes: 8192, availableMemoryBytes: 4096 })),
    loadAverage: vi.fn(() => [1.25, 1.5, 1.75]),
    process: vi.fn(() => ({ scope: "current-main-process-only" as const, pid: 321, uptimeSeconds: 10,
      residentBytes: 128, heapTotalBytes: 64, heapUsedBytes: 32, externalBytes: 16, cpuTimeMicroseconds: { user: 10000, system: 2500 } })),
    storage: vi.fn(async () => ({ diskTotalBytes: 100000, diskFreeBytes: 30000 })), temperatures: vi.fn(async () => undefined) };
}
function cached(): CachedSystemDeviceTelemetry {
  return { measuredAt: "2026-10-20T11:59:58.000Z", devices: [{ id: "gpu:fixture", label: "Fixture GPU", kind: "gpu", backend: "cuda",
    memoryGroupId: "gpu:fixture", totalBytes: 4096, availableBytes: 2048, allocatedBytes: 1024, reservedBytes: null, unifiedMemory: false }] };
}
afterEach(() => { vi.useRealTimers(); });

it("returns actual injected observations with honest units, scope and original cached-device age", async () => {
  const readers = fixtures(), requireCurrent = vi.fn(), cache = vi.fn(() => cached());
  const value = await readSystemStatsSnapshot({ dataDirectory: "/owned/brain", requireCurrent, cachedDeviceTelemetry: cache }, readers);
  expect(value).toMatchObject({ format: "omni-system-stats", version: 1, observedAt,
    scope: { readOnly: true, workerStartedBySnapshot: false, modelLoadedBySnapshot: false, benchmarkPerformedBySnapshot: false, deviceCapturePerformedBySnapshot: false },
    host: { state: "observed", observedAt, value: { logicalProcessors: 8, availableParallelism: 6 } },
    memory: { state: "observed", observedAt, value: { totalMemoryBytes: 8192, availableMemoryBytes: 4096 } },
    loadAverage: { state: "observed", value: { oneMinute: 1.25, fiveMinutes: 1.5, fifteenMinutes: 1.75, unit: "runnable-or-waiting-system-tasks" } },
    process: { state: "observed", value: { scope: "current-main-process-only", residentBytes: 128, cpuTimeMicroseconds: { user: 10000, system: 2500 } } },
    storage: { state: "observed", value: { scope: "brain-data-volume", diskTotalBytes: 100000, diskFreeBytes: 30000 } },
    devices: { state: "observed", source: "cached-device-topology-no-refresh", value: { measuredAt: cached().measuredAt, cached: true, ageMilliseconds: 2000 } } });
  expect(readers.storage).toHaveBeenCalledWith("/owned/brain"); expect(cache).toHaveBeenCalledOnce(); expect(requireCurrent).toHaveBeenCalledTimes(2);
  expect(value.sensors.temperatures).toEqual({ state: "unavailable", reason: "not-observed" });
  expect(JSON.stringify(value)).not.toContain("/owned/brain"); expect(JSON.stringify(value)).not.toContain("percentCPUUsage");
});

it("marks unsupported Windows load, missing cache and failed/invalid readings unavailable instead of inventing zero", async () => {
  const readers = fixtures(); readers.platform = "win32";
  readers.memory = vi.fn(async () => { throw new Error("PRIVATE_SOURCE_PATH_OR_CREDENTIAL"); });
  readers.storage = vi.fn(async () => ({ diskTotalBytes: 100, diskFreeBytes: 101 }));
  readers.process = vi.fn(() => ({ ...fixtures().process(), cpuTimeMicroseconds: { user: 0, system: 0 } }));
  const value = await readSystemStatsSnapshot({ dataDirectory: "/owned/brain", cachedDeviceTelemetry: () => undefined }, readers);
  expect(value.loadAverage).toEqual({ state: "unavailable", reason: "unsupported" }); expect(readers.loadAverage).not.toHaveBeenCalled();
  expect(value.devices).toEqual({ state: "unavailable", reason: "not-observed" });
  expect(value.memory).toEqual({ state: "unavailable", reason: "measurement-failed", causeClass: "Error" });
  expect(value.storage).toMatchObject({ state: "unavailable", reason: "invalid-reading" });
  expect(value.process).toMatchObject({ state: "observed", value: { cpuTimeMicroseconds: { user: 0, system: 0 } } });
  expect(JSON.stringify(value)).not.toContain("PRIVATE_SOURCE_PATH_OR_CREDENTIAL");
});

it("rejects cancellation and invalid host scope before any reader can run", async () => {
  const readers = fixtures(), abort = new AbortController(); abort.abort();
  await expect(readSystemStatsSnapshot({ dataDirectory: "/owned/brain", signal: abort.signal }, readers)).rejects.toMatchObject({ name: "AbortError" });
  await expect(readSystemStatsSnapshot({ dataDirectory: "model-supplied-relative-path" }, readers)).rejects.toThrow("main-owned absolute");
  for (const read of [readers.host, readers.memory, readers.loadAverage, readers.process, readers.storage]) expect(read).not.toHaveBeenCalled();
});

it("cancels a pending read without waiting for or later publishing its result", async () => {
  const readers = fixtures(), abort = new AbortController();
  let entered!: () => void, resolve!: (value: { totalMemoryBytes: number; availableMemoryBytes: number }) => void;
  const started = new Promise<void>(done => { entered = done; });
  readers.memory = vi.fn(() => { entered(); return new Promise<{ totalMemoryBytes: number; availableMemoryBytes: number }>(done => { resolve = done; }); });
  const pending = readSystemStatsSnapshot({ dataDirectory: "/owned/brain", signal: abort.signal }, readers);
  const rejected = expect(pending).rejects.toMatchObject({ name: "AbortError" });
  await started; abort.abort(); await rejected;
  resolve({ totalMemoryBytes: 8192, availableMemoryBytes: 4096 }); await Promise.resolve();
});

it("refuses stale ownership or permission revocation after asynchronous measurement", async () => {
  const readers = fixtures(); let owner = true;
  readers.memory = vi.fn(async () => { owner = false; return { totalMemoryBytes: 8192, availableMemoryBytes: 4096 }; });
  await expect(readSystemStatsSnapshot({ dataDirectory: "/owned/brain", isCurrent: () => owner }, readers)).rejects.toMatchObject({ name: "AbortError" });
  const current = vi.fn().mockResolvedValueOnce(undefined).mockRejectedValueOnce(new Error("Stats permission revoked"));
  await expect(readSystemStatsSnapshot({ dataDirectory: "/owned/brain", requireCurrent: current }, fixtures())).rejects.toThrow("Stats permission revoked");
  expect(current).toHaveBeenCalledTimes(2);
});

it("bounds a stalled read and preserves other observations without a late success claim", async () => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  const readers = fixtures(); readers.memory = vi.fn(() => new Promise<{ totalMemoryBytes: number; availableMemoryBytes: number }>(() => {}));
  const pending = readSystemStatsSnapshot({ dataDirectory: "/owned/brain" }, readers);
  await vi.advanceTimersByTimeAsync(3001);
  const value = await pending;
  expect(value.memory).toMatchObject({ state: "unavailable", reason: "timed-out" });
  expect(value.storage.state).toBe("observed"); expect(value.process.state).toBe("observed"); expect(vi.getTimerCount()).toBe(0);
});

it("reads only the exact cached directory owner, strips host paths and returns independent copies without discovery", () => {
  const request = vi.fn(() => { throw new Error("Cache inspection must not dispatch a worker request"); });
  const service = new DeviceResourceService({ request } as never, "/owned");
  expect(service.cachedTopologyForDirectory("/owned")).toBeUndefined();
  const topology: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1, measuredAt: cached().measuredAt,
    platform: "fixture", architecture: "fixture", measurementRequested: false, neuralModelConstructed: false, warnings: [], links: [],
    devices: [{ ...cached().devices[0]!, nativeComputeSupported: false }, { id: "storage:fixture", label: "/PRIVATE/storage",
      kind: "storage", backend: "filesystem", memoryGroupId: "volume:fixture", totalBytes: 1000, availableBytes: 500,
      allocatedBytes: null, reservedBytes: null, unifiedMemory: false, nativeComputeSupported: false, poolDirectory: "/PRIVATE/storage" }] };
  (service as unknown as { snapshots: Map<string, DeviceResourceTopology> }).snapshots.set(JSON.stringify(["/owned"]), topology);
  for (const directory of ["/other", "/owned/child", "relative"]) expect(service.cachedTopologyForDirectory(directory)).toBeUndefined();
  const first = service.cachedTopologyForDirectory("/owned")!;
  expect(first.devices).toHaveLength(1); expect(JSON.stringify(first)).not.toContain("/PRIVATE");
  first.devices[0]!.availableBytes = 1; first.devices.push({ ...first.devices[0]!, id: "fake" });
  expect(service.cachedTopologyForDirectory("/owned")).toEqual(cached()); expect(request).not.toHaveBeenCalled();
});

it("publishes actual passive thermal values through the stats path rather than an unconditional unavailable state", async () => {
  const readers = fixtures(); readers.temperatures = signal => readSystemTemperatures("linux", signal, {
    async *entries() { yield "thermal_zone0"; }, async read(_zone, field) { return field === "temp" ? "43500\n" : "cpu_thermal\n"; }
  });
  const value = await readSystemStatsSnapshot({ dataDirectory: "/owned/brain" }, readers);
  expect(value.sensors.temperatures).toMatchObject({ state: "observed", observedAt, source: "passive-linux-sysfs-thermal-zones",
    value: { interface: "linux-sysfs-thermal-zones", unit: "celsius", readings: [{ sensorId: "thermal_zone0", label: "cpu_thermal", celsius: 43.5 }] } });
});

it("signals a timed-out sensor traversal to stop while keeping timeout distinct from caller cancellation", async () => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  const readers = fixtures(); let sensorSignal: AbortSignal | undefined;
  readers.temperatures = signal => { sensorSignal = signal; return new Promise((_resolve, reject) => {
    signal!.addEventListener("abort", () => reject(new DOMException("Fixture traversal aborted", "AbortError")), { once: true });
  }); };
  const pending = readSystemStatsSnapshot({ dataDirectory: "/owned/brain" }, readers);
  await vi.advanceTimersByTimeAsync(3001);
  const value = await pending;
  expect(sensorSignal?.aborted).toBe(true); expect(value.sensors.temperatures).toMatchObject({ state: "unavailable", reason: "timed-out" });
  expect(value.memory.state).toBe("observed"); expect(vi.getTimerCount()).toBe(0);
});

it("reports independently measured managed-family RSS with exact sample counts instead of relabeling main-process memory", async () => {
  const managedFamilyMeasurement = vi.fn(async (_signal?: AbortSignal) => ({ residentBytes: 6 * 1024 ** 3, sampledProcessCount: 7, requestedProcessCount: 8 }));
  const value = await readSystemStatsSnapshot({ dataDirectory: "/owned/brain", managedFamilyMeasurement }, fixtures());
  expect(value.managedFamily).toMatchObject({ state: "observed", source: "existing-owned-process-residency-reader", value: {
    residentBytes: 6 * 1024 ** 3, sampledProcessCount: 7, requestedProcessCount: 8,
    scope: "main-owned-managed-process-rss-sum", sharedPagesMayBeCountedMoreThanOnce: true } });
  expect(value.process).toMatchObject({ state: "observed", value: { residentBytes: 128, scope: "current-main-process-only" } });
  expect(managedFamilyMeasurement.mock.calls[0]![0]).toBeInstanceOf(AbortSignal);
  const missing = await readSystemStatsSnapshot({ dataDirectory: "/owned/brain" }, fixtures());
  expect(missing.managedFamily).toEqual({ state: "unavailable", reason: "not-observed" });
});
