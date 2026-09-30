import { describe, expect, it } from "vitest";
import { requireWorkingMemoryPlanRequest } from "../src/main/resourcePlanRequest";

describe("working-memory resource request boundary", () => {
  it("preserves storage sizing and selected training bytes for the main-process probe", () => {
    const request = requireWorkingMemoryPlanRequest({
      mode: "manual",
      hardwareTier: "gpu",
      requestedItems: "120000",
      requestedContextTokens: "32768",
      acceleratorAvailable: true,
      systemRamMode: "manual",
      systemRamSharePercent: 64,
      storagePoolMode: "manual",
      storagePoolBytes: "32212254720",
      trainingSourceBytes: 12_345_678_901,
    });

    expect(request).toEqual({
      mode: "manual",
      hardwareTier: "gpu",
      requestedItems: "120000",
      requestedContextTokens: "32768",
      acceleratorAvailable: true,
      systemRamMode: "manual",
      systemRamSharePercent: 64,
      storagePoolMode: "manual",
      storagePoolBytes: "32212254720",
      trainingSourceBytes: 12_345_678_901,
    });
  });

  it.each([
    { mode: "auto", brainId: "../different-brain" },
    { mode: "auto", brainId: 42 },
    { mode: "auto", storagePoolMode: "fixed" },
    { mode: "auto", storagePoolBytes: 30 },
    { mode: "auto", trainingSourceBytes: -1 },
    { mode: "auto", trainingSourceBytes: 1.5 },
    { mode: "auto", trainingSourceBytes: Number.MAX_SAFE_INTEGER + 1 },
  ])("rejects malformed privileged storage input: %o", (value) => {
    expect(() => requireWorkingMemoryPlanRequest(value)).toThrow(
      /invalid working-memory resource request/i,
    );
  });
});
