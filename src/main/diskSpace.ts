import { statfs } from "node:fs/promises";
import { resolve } from "node:path";
import type { DiskSpaceTelemetry } from "../shared/types";

export const MANDATORY_FREE_DISK_BYTES = 20 * 1024 ** 3;
const GIB = 1024 ** 3;
const MIB = 1024 ** 2;

export interface DiskSpaceComponents {
  selectedDatasetBytes?: number;
  modelBytes?: number;
  checkpointBytes?: number;
  maximumWorkingMemorySpillBytes?: number;
  futureGrowthBytes?: number;
  operationWriteBytes?: number;
}

export interface DiskSpaceMeasurement {
  diskTotalBytes: number;
  diskFreeBytes: number;
}

export interface DiskReservePolicyInput {
  platform?: string;
  diskTotalBytes: number;
  diskFreeBytes?: number;
  operationWriteBytes?: number;
  userMinimumReserveBytes?: number;
}

export type DiskSpaceReader = (path: string) => Promise<DiskSpaceMeasurement>;

function safeBytes(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? Math.min(Number.MAX_SAFE_INTEGER, Math.floor(value))
    : 0;
}

/**
 * Keep constrained/mobile volumes usable while preserving substantial OS and
 * recovery headroom on desktop storage. Twenty GiB is the desktop ceiling and
 * recommendation for volumes large enough to support it, not a mobile floor.
 */
export function adaptiveDiskReserve(input: DiskReservePolicyInput): number {
  const total = safeBytes(input.diskTotalBytes);
  const platformName = String(input.platform ?? process.platform).toLowerCase();
  const mobile = platformName === "android" || platformName === "ios";
  const proportional = Math.ceil(total * (mobile ? 0.04 : 0.05));
  const base = mobile
    ? Math.min(2 * GIB, Math.max(256 * MIB, proportional))
    : Math.min(MANDATORY_FREE_DISK_BYTES, Math.max(GIB, proportional));
  const amplificationMargin = Math.min(
    mobile ? 512 * MIB : 2 * GIB,
    Math.ceil(safeBytes(input.operationWriteBytes) * 0.05)
  );
  return Math.min(
    Math.max(0, total),
    Math.max(
      safeBytes(input.userMinimumReserveBytes),
      base + amplificationMargin
    )
  );
}

/**
 * Project physical writes without double-counting selected sources that
 * already occupy the measured volume. The source size remains visible in the
 * report because it still describes the workload being trained.
 */
export function calculateDiskSpaceReport(input: {
  measuredAt?: string;
  diskTotalBytes: number;
  diskFreeBytes: number;
  mandatoryReserveBytes?: number;
  components?: DiskSpaceComponents;
}): DiskSpaceTelemetry {
  const diskTotalBytes = safeBytes(input.diskTotalBytes);
  const diskFreeBytes = Math.min(
    diskTotalBytes || Number.MAX_SAFE_INTEGER,
    safeBytes(input.diskFreeBytes)
  );
  const mandatoryReserveBytes = input.mandatoryReserveBytes === undefined
    ? adaptiveDiskReserve({
        diskTotalBytes,
        diskFreeBytes,
        operationWriteBytes: input.components?.operationWriteBytes
      })
    : safeBytes(input.mandatoryReserveBytes);
  const selectedDatasetBytes = safeBytes(
    input.components?.selectedDatasetBytes
  );
  const modelBytes = safeBytes(input.components?.modelBytes);
  const checkpointBytes = safeBytes(input.components?.checkpointBytes);
  const maximumWorkingMemorySpillBytes = safeBytes(
    input.components?.maximumWorkingMemorySpillBytes
  );
  const futureGrowthBytes = safeBytes(
    input.components?.futureGrowthBytes
  );
  const operationWriteBytes = safeBytes(
    input.components?.operationWriteBytes
  );
  const projectedPhysicalWriteBytes = [
    modelBytes,
    checkpointBytes,
    maximumWorkingMemorySpillBytes,
    futureGrowthBytes,
    operationWriteBytes
  ].reduce((sum, value) => Math.min(Number.MAX_SAFE_INTEGER, sum + value), 0);
  const projectedRemainingBytes = Math.max(
    0,
    diskFreeBytes - projectedPhysicalWriteBytes
  );
  const projectedAboveReserveBytes = Math.max(
    0,
    projectedRemainingBytes - mandatoryReserveBytes
  );
  const measuredAt = input.measuredAt ?? new Date().toISOString();
  if (!Number.isFinite(Date.parse(measuredAt))) {
    throw new Error("Disk-space measurement timestamp is invalid.");
  }
  return {
    schemaVersion: 1,
    measuredAt: new Date(measuredAt).toISOString(),
    diskTotalBytes,
    diskFreeBytes,
    mandatoryReserveBytes,
    selectedDatasetBytes,
    modelBytes,
    checkpointBytes,
    maximumWorkingMemorySpillBytes,
    futureGrowthBytes,
    operationWriteBytes,
    projectedRemainingBytes,
    projectedAboveReserveBytes,
    paused: projectedRemainingBytes <= mandatoryReserveBytes
  };
}

export const readDiskSpace: DiskSpaceReader = async (path) => {
  const reading = await statfs(resolve(path));
  return {
    diskTotalBytes: safeBytes(reading.blocks * reading.bsize),
    diskFreeBytes: safeBytes(reading.bavail * reading.bsize)
  };
};

export class DiskReservePauseError extends Error {
  readonly diskSpace: DiskSpaceTelemetry;

  constructor(diskSpace: DiskSpaceTelemetry) {
    super(
      `This operation would enter the mandatory ${diskSpace.mandatoryReserveBytes.toLocaleString()}-byte free-space reserve.`
    );
    this.name = "DiskReservePauseError";
    this.diskSpace = diskSpace;
  }
}

export async function requireDiskWrite(
  path: string,
  components: DiskSpaceComponents,
  dependencies: {
    read?: DiskSpaceReader;
    now?: () => Date;
    platform?: string;
    userMinimumReserveBytes?: number;
  } = {}
): Promise<DiskSpaceTelemetry> {
  const measurement = await (dependencies.read ?? readDiskSpace)(path);
  const diskSpace = calculateDiskSpaceReport({
    measuredAt: (dependencies.now?.() ?? new Date()).toISOString(),
    ...measurement,
    mandatoryReserveBytes: adaptiveDiskReserve({
      platform: dependencies.platform,
      ...measurement,
      operationWriteBytes: components.operationWriteBytes,
      userMinimumReserveBytes: dependencies.userMinimumReserveBytes
    }),
    components
  });
  if (diskSpace.paused) throw new DiskReservePauseError(diskSpace);
  return diskSpace;
}
