import { describe, expect, it } from "vitest";
import { automationResidentBytes, requireAutomationHeadroom } from "../src/main/automationResources";
describe("measured host action admission", () => {
  it("uses the actual owned process residency without starting a neural worker", async () => { expect(await automationResidentBytes([process.pid, process.pid])).toBeGreaterThan(0); });
  it("refuses selected-ceiling, physical and heap pressure before the next effect", () => {
    const readings = { residentBytes: 100, ceilingBytes: 1000, freeBytes: 1000, heapAvailableBytes: 1000 };
    expect(() => requireAutomationHeadroom(100, readings)).not.toThrow();
    expect(() => requireAutomationHeadroom(901, readings)).toThrow(/paused/);
    expect(() => requireAutomationHeadroom(100, { ...readings, freeBytes: 99 })).toThrow(/paused/);
    expect(() => requireAutomationHeadroom(100, { ...readings, heapAvailableBytes: 99 })).toThrow(/paused/);
    expect(() => requireAutomationHeadroom(1, { ...readings, residentBytes: NaN })).toThrow(/unavailable/);
  });
});
