import { describe, expect, it, vi } from "vitest";
import {
  BrainStorageOperationSession
} from "../src/main/brainStorageOperations";
import { DiskReservePauseError } from "../src/main/diskSpace";
import type { DiskSpaceTelemetry } from "../src/shared/types";

function disk(paused: boolean): DiskSpaceTelemetry {
  return {
    schemaVersion: 1,
    measuredAt: "2026-09-07T06:00:00.000Z",
    diskTotalBytes: 100,
    diskFreeBytes: paused ? 20 : 80,
    mandatoryReserveBytes: 20,
    selectedDatasetBytes: 0,
    modelBytes: 0,
    checkpointBytes: 0,
    maximumWorkingMemorySpillBytes: 0,
    futureGrowthBytes: 0,
    operationWriteBytes: paused ? 1 : 10,
    projectedRemainingBytes: paused ? 19 : 70,
    projectedAboveReserveBytes: paused ? 0 : 50,
    paused
  };
}

describe("brain storage operation control", () => {
  it("pauses at a durable file boundary and resumes the same progress cursor", async () => {
    const session = new BrainStorageOperationSession(
      "pause-resume-clone",
      "duplicate",
      "source-brain"
    );
    const events: string[] = [];
    session.on("event", (event) => events.push(event.state));
    await session.hooks.checkpoint({
      filesCompleted: 3,
      filesTotal: 10,
      logicalBytesCompleted: 300,
      logicalBytesTotal: 1_000
    });
    session.pause();
    let passed = false;
    const checkpoint = session.hooks.checkpoint({
      filesCompleted: 4,
      logicalBytesCompleted: 400
    }).then(() => {
      passed = true;
    });
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(passed).toBe(false);
    expect(session.current()).toMatchObject({
      state: "paused",
      filesCompleted: 4,
      filesTotal: 10,
      logicalBytesCompleted: 400,
      logicalBytesTotal: 1_000
    });

    session.resume();
    await checkpoint;
    expect(passed).toBe(true);
    expect(events).toContain("paused");
    expect(session.current().state).toBe("running");
  });

  it("pauses before disk reserve and rechecks after explicit resume", async () => {
    let enoughSpace = false;
    const check = vi.fn(async () => {
      if (!enoughSpace) throw new DiskReservePauseError(disk(true));
      return disk(false);
    });
    const session = new BrainStorageOperationSession(
      "disk-pressure-clone",
      "duplicate",
      "source-brain",
      check
    );
    const pending = session.hooks.checkDisk("/fixture", 10);
    await vi.waitFor(() => expect(session.current().state).toBe("paused"));
    expect(session.current().diskSpace).toMatchObject({ paused: true });

    enoughSpace = true;
    session.resume();
    await expect(pending).resolves.toMatchObject({ paused: false });
    expect(check).toHaveBeenCalledTimes(2);
  });

  it("cancels a paused cursor without allowing the next materialization", async () => {
    const session = new BrainStorageOperationSession(
      "cancel-clone",
      "duplicate",
      "source-brain"
    );
    session.pause();
    const pending = session.hooks.checkpoint({ filesCompleted: 2, filesTotal: 8 });
    await new Promise<void>((resolve) => setImmediate(resolve));
    session.cancel();
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(session.fail(new Error("ignored after cancel"))).toMatchObject({
      state: "cancelled",
      filesCompleted: 2,
      filesTotal: 8
    });
  });
});
