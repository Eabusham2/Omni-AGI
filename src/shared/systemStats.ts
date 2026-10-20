import type { ResourceDevice } from "./deviceResources";

export type SystemStatsReading<T> =
  | { state: "observed"; observedAt: string; source: string; value: T }
  | { state: "unavailable"; reason: "unsupported" | "not-observed" | "measurement-failed" | "timed-out" | "invalid-reading"; causeClass?: string };

/** A cached observation, never an instruction to discover or benchmark devices. */
export interface CachedSystemDeviceTelemetry {
  measuredAt: string;
  devices: Array<Pick<ResourceDevice, "id" | "label" | "backend" | "memoryGroupId" | "totalBytes" | "availableBytes" | "allocatedBytes" | "reservedBytes" | "unifiedMemory"> & {
    kind: "ram" | "gpu" | "npu";
  }>;
}

export interface SystemStatsHostReading {
  platform: string;
  architecture: string;
  osRelease: string;
  uptimeSeconds: number;
  logicalProcessors: number;
  availableParallelism: number;
  processorModel: string | null;
}

export interface SystemStatsProcessReading {
  scope: "current-main-process-only";
  pid: number;
  uptimeSeconds: number;
  residentBytes: number;
  heapTotalBytes: number;
  heapUsedBytes: number;
  externalBytes: number;
  /** Lifetime CPU time, NOT utilization percentage or token-generation speed. */
  cpuTimeMicroseconds: { user: number; system: number };
}

export interface SystemTemperatureReading {
  interface: "linux-sysfs-thermal-zones";
  unit: "celsius";
  readings: Array<{ sensorId: string; label: string | null; celsius: number }>;
  unavailableSensors: Array<{ sensorId: string; reason: "unreadable-or-invalid-reading" }>;
  traversalComplete: true;
}

export interface SystemStatsFamilyMeasurement {
  residentBytes: number;
  sampledProcessCount: number;
  requestedProcessCount: number;
}

export interface SystemStatsSnapshot {
  format: "omni-system-stats";
  version: 1;
  observedAt: string;
  scope: { readOnly: true; workerStartedBySnapshot: false; modelLoadedBySnapshot: false; benchmarkPerformedBySnapshot: false; deviceCapturePerformedBySnapshot: false };
  host: SystemStatsReading<SystemStatsHostReading>;
  memory: SystemStatsReading<{ totalMemoryBytes: number; availableMemoryBytes: number }>;
  loadAverage: SystemStatsReading<{ oneMinute: number; fiveMinutes: number; fifteenMinutes: number; unit: "runnable-or-waiting-system-tasks" }>;
  process: SystemStatsReading<SystemStatsProcessReading>;
  managedFamily: SystemStatsReading<SystemStatsFamilyMeasurement & {
    scope: "main-owned-managed-process-rss-sum";
    sharedPagesMayBeCountedMoreThanOnce: true;
  }>;
  storage: SystemStatsReading<{ scope: "brain-data-volume"; diskTotalBytes: number; diskFreeBytes: number }>;
  devices: SystemStatsReading<CachedSystemDeviceTelemetry & { cached: true; ageMilliseconds: number }>;
  sensors: { temperatures: SystemStatsReading<SystemTemperatureReading> };
}
