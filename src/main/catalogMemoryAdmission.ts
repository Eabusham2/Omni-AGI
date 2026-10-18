import { requireAutomationHeadroom } from "./automationResources";

export interface CatalogMemoryReadings {
  residentBytes: number;
  selectedCeilingBytes: number | null;
  freeBytes: number;
  heapAvailableBytes: number;
}

/** Cooperative admission for catalog parsing, not a whole-app OS memory lock. */
export function requireCatalogMemoryHeadroom(bytes: number, readings: CatalogMemoryReadings): void {
  const { residentBytes, selectedCeilingBytes, freeBytes, heapAvailableBytes } = readings;
  if (![freeBytes, heapAvailableBytes].every(value => Number.isSafeInteger(value) && value >= 0) ||
    (selectedCeilingBytes !== null && (!Number.isSafeInteger(selectedCeilingBytes) || selectedCeilingBytes < 0))) {
    throw new Error("Brain archive memory measurements are unavailable.");
  }
  // With no loaded instance there is no selected family ceiling. Bound this
  // transfer by measured physical and JS-heap headroom, not a made-up RAM share.
  const ceilingBytes = selectedCeilingBytes ?? residentBytes + Math.min(freeBytes, heapAvailableBytes);
  requireAutomationHeadroom(bytes, { residentBytes, ceilingBytes, freeBytes, heapAvailableBytes });
}
