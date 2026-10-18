import { describe, expect, it } from "vitest";
import { requireCatalogMemoryHeadroom } from "../src/main/catalogMemoryAdmission";

describe("catalog allocation admission", () => {
  const reading = { residentBytes: 300, selectedCeilingBytes: 500, freeBytes: 2000, heapAvailableBytes: 2000 };
  it("honors the current selected app-family ceiling without rounding it upwards", () => {
    expect(() => requireCatalogMemoryHeadroom(200, reading)).not.toThrow();
    expect(() => requireCatalogMemoryHeadroom(201, reading)).toThrow(/paused/);
    expect(() => requireCatalogMemoryHeadroom(100, { ...reading, selectedCeilingBytes: 350 })).toThrow(/paused/);
  });
  it("uses measured physical and heap headroom when no instance has a ceiling", () => {
    const unloaded = { ...reading, selectedCeilingBytes: null };
    expect(() => requireCatalogMemoryHeadroom(1000, unloaded)).not.toThrow();
    expect(() => requireCatalogMemoryHeadroom(1000, { ...unloaded, freeBytes: 900 })).toThrow(/paused/);
    expect(() => requireCatalogMemoryHeadroom(1000, { ...unloaded, heapAvailableBytes: 900 })).toThrow(/paused/);
  });
  it("fails closed on unavailable or invalid resource measurements", () => {
    for (const patch of [{ freeBytes: NaN }, { heapAvailableBytes: -1 }, { selectedCeilingBytes: NaN }, { residentBytes: Infinity }]) {
      expect(() => requireCatalogMemoryHeadroom(1, { ...reading, ...patch })).toThrow(/unavailable/);
    }
  });
});
