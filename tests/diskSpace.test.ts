import { describe, expect, it } from "vitest";
import {
  DiskReservePauseError,
  MANDATORY_FREE_DISK_BYTES,
  adaptiveDiskReserve,
  calculateDiskSpaceReport,
  requireDiskWrite
} from "../src/main/diskSpace";

const GIB = 1024 ** 3;

describe("global disk space-left contract", () => {
  it("reports every component without charging referenced source bytes twice", () => {
    const report = calculateDiskSpaceReport({
      measuredAt: "2026-09-07T00:00:00.000Z",
      diskTotalBytes: 500 * GIB,
      diskFreeBytes: 200 * GIB,
      mandatoryReserveBytes: 20 * GIB,
      components: {
        selectedDatasetBytes: 80 * GIB,
        modelBytes: 2 * GIB,
        checkpointBytes: 3 * GIB,
        maximumWorkingMemorySpillBytes: 4 * GIB,
        futureGrowthBytes: 5 * GIB,
        operationWriteBytes: 6 * GIB
      }
    });

    expect(report).toMatchObject({
      schemaVersion: 1,
      diskTotalBytes: 500 * GIB,
      diskFreeBytes: 200 * GIB,
      mandatoryReserveBytes: 20 * GIB,
      selectedDatasetBytes: 80 * GIB,
      modelBytes: 2 * GIB,
      checkpointBytes: 3 * GIB,
      maximumWorkingMemorySpillBytes: 4 * GIB,
      futureGrowthBytes: 5 * GIB,
      operationWriteBytes: 6 * GIB,
      projectedRemainingBytes: 180 * GIB,
      projectedAboveReserveBytes: 160 * GIB,
      paused: false
    });
  });

  it("pauses exactly at the one global reserve boundary", () => {
    const exact = calculateDiskSpaceReport({
      diskTotalBytes: 100 * GIB,
      diskFreeBytes: 25 * GIB,
      mandatoryReserveBytes: 20 * GIB,
      components: { operationWriteBytes: 5 * GIB }
    });
    const above = calculateDiskSpaceReport({
      diskTotalBytes: 100 * GIB,
      diskFreeBytes: 25 * GIB + 1,
      mandatoryReserveBytes: 20 * GIB,
      components: { operationWriteBytes: 5 * GIB }
    });

    expect(exact.projectedRemainingBytes).toBe(MANDATORY_FREE_DISK_BYTES);
    expect(exact.paused).toBe(true);
    expect(above.paused).toBe(false);
  });

  it("returns the measured projection or a typed retryable pause", async () => {
    const read = async () => ({
      diskTotalBytes: 100 * GIB,
      diskFreeBytes: 30 * GIB
    });
    await expect(
      requireDiskWrite("/unused-in-injected-test", {
        operationWriteBytes: GIB
      }, { read, now: () => new Date("2026-09-07T00:00:00.000Z") })
    ).resolves.toMatchObject({
      projectedRemainingBytes: 29 * GIB,
      paused: false
    });
    await expect(
      requireDiskWrite("/unused-in-injected-test", {
        operationWriteBytes: 26 * GIB
      }, { read })
    ).rejects.toBeInstanceOf(DiskReservePauseError);
  });

  it("keeps the local-brain disk minimum independent of the RAM/device reserve", () => {
    expect(adaptiveDiskReserve({
      platform: "linux",
      diskTotalBytes: 16 * GIB
    })).toBe(MANDATORY_FREE_DISK_BYTES);
    expect(adaptiveDiskReserve({
      platform: "android",
      diskTotalBytes: 16 * GIB
    })).toBe(MANDATORY_FREE_DISK_BYTES);
    expect(adaptiveDiskReserve({
      platform: "ios",
      diskTotalBytes: 256 * GIB
    })).toBe(MANDATORY_FREE_DISK_BYTES);
    expect(adaptiveDiskReserve({
      platform: "darwin",
      diskTotalBytes: 500 * GIB
    })).toBe(MANDATORY_FREE_DISK_BYTES);
    expect(adaptiveDiskReserve({
      platform: "linux",
      diskTotalBytes: 16 * GIB,
      operationWriteBytes: 4 * GIB
    })).toBe(MANDATORY_FREE_DISK_BYTES);
  });

  it("does not clamp the required reserve to an undersized volume or an override", () => {
    expect(adaptiveDiskReserve({ diskTotalBytes: 0 })).toBe(MANDATORY_FREE_DISK_BYTES);
    expect(adaptiveDiskReserve({
      diskTotalBytes: 16 * GIB,
      userMinimumReserveBytes: 24 * GIB
    })).toBe(24 * GIB);
    const report = calculateDiskSpaceReport({
      diskTotalBytes: 16 * GIB,
      diskFreeBytes: 15 * GIB,
      mandatoryReserveBytes: GIB
    });
    expect(report.mandatoryReserveBytes).toBe(MANDATORY_FREE_DISK_BYTES);
    expect(report.paused).toBe(true);
  });
});
