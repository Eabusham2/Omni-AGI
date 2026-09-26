import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  GIB_BYTES,
  INITIAL_WEB_CRAWL_RESERVE_BYTES,
  plannedTrainingSourceBytes
} from "../src/renderer/src/buildResourcePlanning";
import {
  MANDATORY_FREE_DISK_BYTES,
  adaptiveDiskReserve
} from "../src/main/diskSpace";

const root = resolve(import.meta.dirname, "..");
const app = readFileSync(resolve(root, "src/renderer/src/App.tsx"), "utf8");
const css = readFileSync(resolve(root, "src/renderer/src/styles.css"), "utf8");

describe("stable Build wizard resource flow", () => {
  it("accounts for every selected local byte and gives unknown web crawls a nonzero reserve", () => {
    expect(INITIAL_WEB_CRAWL_RESERVE_BYTES).toBe(GIB_BYTES);
    expect(plannedTrainingSourceBytes([
      { kind: "files", bytes: 6 * GIB_BYTES },
      { kind: "folder", bytes: 12 * GIB_BYTES },
      { kind: "web" },
      { kind: "web" }
    ])).toBe(20 * GIB_BYTES);
    expect(plannedTrainingSourceBytes([
      { kind: "files", bytes: Number.NaN },
      { kind: "folder", bytes: -50 }
    ])).toBe(0);
  });

  it("orders data before the single combined Memory & storage calculation", () => {
    expect(app).toContain(
      "const stageContent = [identityStage, dataStage, memoryStorageStage, accessStage]"
    );
    expect(app).toContain('["Memory & storage", "Size from this device and data"]');
    expect(app).toContain('aria-label="Memory and storage calculation"');
    expect(app).toContain('aria-label="Runtime memory order"');
    expect(app).toContain("RAM first");
    expect(app).toContain("Drive spill only when needed");
    expect(app).toContain("trainingSourceBytes: selectedTrainingBytes");
  });

  it("lets Identity and Data advance while enforcing the plan at Memory and Create", () => {
    expect(app).toContain(
      'disabled={!name.trim() || (step === 2 && memoryPlanBlocksProgress)}'
    );
    expect(app).toContain(
      'disabled={!name.trim() || building || memoryPlanBlocksProgress}'
    );
    expect(app).not.toContain(
      'disabled={!name.trim() || memoryPlanPending || (memoryPlan ? !memoryPlan.allowed'
    );
  });

  it("keeps Auto plus a seeded Custom pool outside an adaptive device reserve", () => {
    expect(app).toContain('className="system-ram-budget storage-pool-budget"');
    expect(app).toContain('onClick={() => chooseStoragePoolMode("manual")}');
    expect(app).toContain("memoryPlan?.resources.sharedStoragePoolBytes");
    expect(app).toContain("requiredStoragePoolBytes");
    expect(app).toContain("maximumStoragePoolBytes");
    expect(app).toContain("adaptive device/OS reserve");
    expect(app).toContain("memoryPlan.diskSpace.projectedRemainingBytes");
    expect(app).toContain("memoryPlan.diskSpace.projectedAboveReserveBytes");
    expect(app).toContain("memoryPlan.diskSpace.mandatoryReserveBytes");
    expect(app).not.toContain("mandatory 20 GiB free-space reserve");
    expect(app).toContain("storagePoolBytes: wholeGiBTextToBytes(manualStoragePoolGiB)");

    expect(adaptiveDiskReserve({
      platform: "android",
      diskTotalBytes: 16 * GIB_BYTES
    })).toBe(Math.ceil(16 * GIB_BYTES * 0.04));
    expect(adaptiveDiskReserve({
      platform: "linux",
      diskTotalBytes: 16 * GIB_BYTES
    })).toBe(GIB_BYTES);
    expect(adaptiveDiskReserve({
      platform: "darwin",
      diskTotalBytes: 500 * GIB_BYTES
    })).toBe(MANDATORY_FREE_DISK_BYTES);
  });

  it("keeps the four-stage flow accessible on compact screens", () => {
    expect(app).toContain('aria-label="Brain Library"');
    expect(app).toContain("aria-label={`Step ${index + 1}: ${title}. ${copy}`}");
    expect(app).toContain('role="status" aria-live="polite"');
    expect(css).toMatch(/@media \(max-width: 760px\)[\s\S]*?\.memory-storage-flow\s*\{[\s\S]*?grid-template-columns: minmax\(0, 1fr\)/);
    expect(css).toMatch(/\.step-list button\s*\{[\s\S]*?min-height: 44px/);
    expect(css).toContain(".build-resource-actions .button");
  });
});
