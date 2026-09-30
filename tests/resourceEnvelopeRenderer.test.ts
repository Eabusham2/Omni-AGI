import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  DEFAULT_CONFIG,
  type WorkingMemoryResourcePlan
} from "../src/shared/types";
import {
  MAX_SYSTEM_RAM_SHARE_PERCENT,
  MIN_SYSTEM_RAM_SHARE_PERCENT,
  STORAGE_BOUNDARY_COPY,
  clampSystemRamSharePercent,
  configWithResourceEnvelope,
  contextCapacityBandStyle
} from "../src/renderer/src/resourceEnvelope";

const root = resolve(import.meta.dirname, "..");
const read = (path: string): string => readFileSync(resolve(root, path), "utf8");

function reviewedPlan(): WorkingMemoryResourcePlan {
  return {
    mode: "extended",
    selectedItems: 65_536,
    context: {
      selectedTokens: 16_384,
      floorTokens: 8_192,
      autoTokens: 12_288,
      extendedTokens: 16_384,
      maximumTokens: 24_576,
      suitableMinimumTokens: 8_192,
      suitableMaximumTokens: 15_360
    },
    resources: {
      configuredMemorySpillBytes: 4_096,
      storagePoolMode: "manual",
      sharedStoragePoolBytes: 18 * 1_073_741_824
    },
    offload: {
      residentMemoryItems: 52_000,
      estimatedSlowdownPercent: 9,
      benchmark: {
        storageBytesPerSecond: 321_000_000
      }
    }
  } as WorkingMemoryResourcePlan;
}

describe("renderer resource envelope", () => {
  it("enforces the 30–100 percent safe-pool boundary", () => {
    expect(MIN_SYSTEM_RAM_SHARE_PERCENT).toBe(30);
    expect(MAX_SYSTEM_RAM_SHARE_PERCENT).toBe(100);
    expect(clampSystemRamSharePercent(-1)).toBe(30);
    expect(clampSystemRamSharePercent(64.6)).toBe(65);
    expect(clampSystemRamSharePercent(101)).toBe(100);
    expect(clampSystemRamSharePercent(Number.NaN)).toBe(30);
  });

  it("persists the exact reviewed plan and keeps Auto unpinned", () => {
    const plan = reviewedPlan();
    const manual = configWithResourceEnvelope(DEFAULT_CONFIG, plan, "manual", 74);
    expect(manual).toMatchObject({
      workingMemorySlots: DEFAULT_CONFIG.workingMemorySlots,
      contextWindowTokens: 16_384,
      workingMemoryMode: "extended",
      extendedWorkingMemory: true,
      memoryOffloadBytes: 4_096,
      memoryResidentItems: Math.min(DEFAULT_CONFIG.workingMemorySlots, 52_000),
      memoryOffloadSlowdownPercent: 9,
      systemRamMode: "manual",
      systemRamSharePercent: 74,
      storagePoolMode: "manual",
      storagePoolBytes: 18 * 1_073_741_824,
      storageBytesPerSecond: 321_000_000
    });

    expect(
      configWithResourceEnvelope(manual, plan, "auto", 99).systemRamSharePercent
    ).toBe(0);
  });

  it("centers the capacity colors on the measured suitable range", () => {
    const first = reviewedPlan();
    const second = {
      ...reviewedPlan(),
      context: {
        ...reviewedPlan().context,
        suitableMinimumTokens: 2_048,
        suitableMaximumTokens: 8_192
      }
    };
    const firstStyle = contextCapacityBandStyle(first);
    const secondStyle = contextCapacityBandStyle(second);

    expect(firstStyle["--capacity-green-start"]).toBe("33.33333333333333%");
    expect(firstStyle["--capacity-green-end"]).toBe("62.5%");
    expect(secondStyle["--capacity-green-start"]).toBe("8.333333333333332%");
    expect(secondStyle).not.toEqual(firstStyle);
  });

  it("renders Build and Device settings with accurate benchmark and storage semantics", () => {
    const app = read("src/renderer/src/App.tsx");
    const css = read("src/renderer/src/styles.css");
    const presentation = read("src/renderer/src/resourceEnvelope.ts");

    expect(app).toContain('view === "settings"');
    expect(app).toContain("<DeviceRuntimeWorkspace");
    expect(app).toContain("Advanced cap · 30–100%");
    expect(app).toContain("30% stability floor");
    expect(app).toContain("100% safe-pool maximum");
    expect(app.match(/Measured RAM-copy throughput/g)?.length).toBeGreaterThanOrEqual(2);
    expect(app.match(/Measured durable storage-write throughput/g)?.length).toBeGreaterThanOrEqual(2);
    expect(app).toContain("Custom cap warnings");
    expect(app).toContain("Change this instance&apos;s active context");
    expect(app).toContain("Choose up to the live physical/model maximum");
    expect(app).toContain("One reusable pool covers the largest active instance");
    expect(app).toContain('aria-label="Active context tokens"');
    expect(app).toContain('aria-label="Shared pool capacity"');
    expect(app).toContain("storageSliderMinimumGiB");
    expect(app).toContain("storageSliderUnavailable");
    expect(app).toContain("Saved context and cortex capacity are derived from physical RAM");
    expect(app).toContain("Temporarily occupied by other apps");
    expect(app.includes("RAM first")).toBe(true);
    expect(app).toContain("Core, active-context, and neural-memory placement");
    expect(app).toContain("All initial learned weights use 2-bit packed ternary codes");
    expect(app).toContain("contextCapacityBandStyle(memoryPlan)");
    expect(app).not.toContain("maximum 100% of the currently safe pool");
    expect(presentation).toContain(STORAGE_BOUNDARY_COPY);
    expect(STORAGE_BOUNDARY_COPY).toContain("cold memory, replay batches, and checkpoints");
    expect(STORAGE_BOUNDARY_COPY).toContain("Cold attention and saved activity");
    expect(css).toContain(".resource-settings-layout");
    expect(css).toContain(".working-memory-placement");
    expect(css).toContain("var(--capacity-green-start)");
    expect(css).toContain("@media (max-width: 520px)");
  });
});
