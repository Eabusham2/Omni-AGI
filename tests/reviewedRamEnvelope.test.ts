import { expect, it, vi } from "vitest";
import { GIB, MIB, ResourcePlanner } from "../src/main/resourcePlanner";
import { DEFAULT_CONFIG } from "../src/shared/types";

function fixture(availableMemoryBytes = 12 * GIB) {
  const benchmark = vi.fn(async () => { throw new Error("budget selection must not benchmark or infer model geometry"); });
  return { benchmark, planner: new ResourcePlanner("/fixture", {
    readResources: async () => ({ totalMemoryBytes: 16 * GIB, availableMemoryBytes, diskTotalBytes: 500 * GIB, diskFreeBytes: 300 * GIB }), benchmark
  }) };
}

it("selects the reviewed manual family ceiling independently of model geometry or current pressure", async () => {
  const { planner, benchmark } = fixture(2 * GIB);
  const value = await planner.reviewedRamEnvelope({ ...DEFAULT_CONFIG, systemRamMode: "manual", systemRamSharePercent: 50,
    contextWindowTokens: 1_000_000 }, "micro");
  expect(value.systemRamBudgetBytes).toBe(15 * GIB / 2);
  expect(value.availableSafeRamBytes).toBe(GIB);
  expect(value.modelFitEvaluated).toBe(false);
  expect(benchmark).not.toHaveBeenCalled();
});

it("matches the native worker Auto policy including unknown and measured drive speed", async () => {
  const { planner } = fixture();
  for (const [speed, share] of [[0, 65], [50 * MIB, 75], [300 * MIB, 70], [GIB, 65]]) {
    const value = await planner.reviewedRamEnvelope({ ...DEFAULT_CONFIG, systemRamMode: "auto", storageBytesPerSecond: speed! }, "personal");
    expect(value.systemRamSharePercent).toBe(share);
    expect(value.systemRamBudgetBytes).toBe(Math.floor(15 * GIB * share! / 100));
  }
});

it("uses the explicit RAM-device override and rejects invalid review inputs", async () => {
  const { planner } = fixture();
  const config = { ...DEFAULT_CONFIG, deviceResources: { version: 1 as const, allocations: [{ deviceId: "ram:system", enabled: true,
    mode: "manual" as const, percent: 47.5, autoIncrease: false }], storageDirectories: [], allowStorageSpill: true, autoIncreaseStorage: false } };
  expect((await planner.reviewedRamEnvelope(config)).systemRamBudgetBytes).toBe(Math.floor(15 * GIB * 0.475));
  await expect(planner.reviewedRamEnvelope({ ...DEFAULT_CONFIG, systemRamMode: "manual", systemRamSharePercent: 29 })).rejects.toThrow();
});
