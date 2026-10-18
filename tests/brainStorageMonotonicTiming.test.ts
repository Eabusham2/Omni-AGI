import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainStorageOperationSession } from "../src/main/brainStorageOperations";

afterEach(() => vi.useRealTimers());

describe("storage-operation monotonic timing", () => {
  it("keeps elapsed and rate stable across a backward wall-clock jump", async () => {
    vi.useFakeTimers(); vi.setSystemTime(new Date("2035-01-01T00:00:00.000Z"));
    let monotonic = 1_000;
    const session = new BrainStorageOperationSession("clock-jump", "duplicate", "source", async () => {
      throw new Error("Disk check is not part of this timing fixture.");
    }, () => monotonic);
    expect(session.start().elapsedMs).toBe(0);
    vi.setSystemTime(new Date("1995-01-01T00:00:00.000Z")); monotonic = 3_000;
    await session.hooks.checkpoint({ logicalBytesCompleted: 2_000, logicalBytesTotal: 4_000 });
    expect(session.current()).toMatchObject({ elapsedMs: 2_000, bytesPerSecond: 1_000, etaMs: 2_000 });
  });
});
