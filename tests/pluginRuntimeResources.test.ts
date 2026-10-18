import { expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { PluginRuntimeResources } from "../src/main/pluginRuntimeResources";

const mib = 1024 * 1024;
it("uses the reviewed available RAM pool in production composition without raising its selected ceiling", () => {
  const source = readFileSync(new URL("../src/main/index.ts", import.meta.url), "utf8");
  const start = source.indexOf("const runtimeMemory = new PluginRuntimeResources"), end = source.indexOf("const runtimeProcesses", start);
  const composition = source.slice(start, end);
  expect(start >= 0 && end > start).toBe(true);
  expect(composition.includes("freeBytes: envelope.availableSafeRamBytes")).toBe(true);
  expect(composition.includes("Math.min(envelope.systemRamBudgetBytes, selected ?? envelope.systemRamBudgetBytes)")).toBe(true);
  expect(composition.includes("heapAvailableBytes:")).toBe(true);
  expect(composition.includes("freeBytes: Math.floor(freemem())")).toBe(false);
});
it("admits read transfers before parsing and retains simultaneous reservations until release", async () => {
  const resources = new PluginRuntimeResources(async () => ({ residentBytes: 20 * mib, ceilingBytes: 100 * mib, freeBytes: 200 * mib, heapAvailableBytes: 200 * mib }));
  const first = await resources.admitTransfer("brain", 10 * mib);
  await expect(resources.admitTransfer("brain", 10 * mib)).rejects.toThrow("selected app RAM");
  await first.release(); const next = await resources.admitTransfer("brain", 10 * mib); await next.release();
  await expect(resources.admitTransfer("brain", 100 * mib)).rejects.toThrow("selected app RAM");
});
it("serializes concurrent startup reservations and releases only its own admitted owner", async () => {
  const measure = vi.fn(async () => ({ residentBytes: 20 * mib, ceilingBytes: 120 * mib, freeBytes: 200 * mib, heapAvailableBytes: 200 * mib }));
  const resources = new PluginRuntimeResources(measure);
  const results = await Promise.allSettled([resources.admit("brain", 32), resources.admit("brain", 32)]);
  expect(results[0]!.status).toBe("fulfilled"); expect(results[1]!.status).toBe("rejected");
  if (results[0]!.status !== "fulfilled") throw new Error("First admission missing");
  await results[0]!.value.release();
  await expect(results[0]!.value.check()).rejects.toThrow("released");
  const next = await resources.admit("brain", 32); await next.release();
});
it("counts measured renderer residency without duplicating its startup estimate and detects pressure", async () => {
  let reading = { residentBytes: 20 * mib, ceilingBytes: 160 * mib, freeBytes: 200 * mib, heapAvailableBytes: 200 * mib };
  const resources = new PluginRuntimeResources(async () => reading);
  const lease = await resources.admit("brain", 32);
  reading = { ...reading, residentBytes: 70 * mib }; await lease.check(100);
  reading = { ...reading, residentBytes: 159 * mib };
  await expect(lease.check(100)).rejects.toThrow("selected app RAM"); await lease.release();
});
