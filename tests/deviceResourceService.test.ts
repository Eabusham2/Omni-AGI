import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DeviceResourceService } from "../src/main/deviceResourceService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import type { DeviceResourceTopology } from "../src/shared/deviceResources";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function directory() { const root = await mkdtemp(join(tmpdir(), "omni-device-service-")); roots.push(root); return root; }
function topology(directories: string[], measured = false): DeviceResourceTopology {
  return { format: "omni-device-resource-topology", version: 1, measuredAt: "2026-10-06T00:00:00Z", platform: "fixture", architecture: "fixture",
    devices: directories.map((path, index) => ({ id: `storage:${index}`, label: path, kind: "storage", backend: "filesystem", memoryGroupId: `volume:${index}`,
      totalBytes: 500 * 1024 ** 3, availableBytes: 100 * 1024 ** 3, allocatedBytes: null, reservedBytes: null, unifiedMemory: false,
      nativeComputeSupported: false, poolDirectory: path, ...(measured ? { measurement: { bytesPerSecond: 1000, sampleBytes: 4096, runs: 3,
        medianNanoseconds: 4096000000, measuredAt: "2026-10-06T00:00:00Z", scope: "durable-file-write" as const } } : {}) })),
    links: [], measurementRequested: measured, neuralModelConstructed: false, warnings: [] };
}

describe("main-owned device discovery", () => {
  it("does not satisfy an explicit measurement with an in-flight unmeasured discovery", async () => {
    const root = await directory();
    const pending: Array<{ measured: boolean; resolve: (value: DeviceResourceTopology) => void }> = [];
    const request = vi.fn((_method: string, args: { measure: boolean }) => new Promise<DeviceResourceTopology>((resolve) => pending.push({ measured: args.measure, resolve })));
    const service = new DeviceResourceService({ request } as unknown as Pick<EngineSupervisor, "request">, root, { platform: "linux" });
    const discovered = service.topology();
    await vi.waitFor(() => expect(pending.length).toBe(1));
    const measured = service.topology({ measure: true });
    const duplicate = service.topology({ measure: true });
    await vi.waitFor(() => expect(pending.length).toBe(2));
    expect(pending.map((entry) => entry.measured)).toEqual([false, true]);
    pending[0]!.resolve(topology([root])); pending[1]!.resolve(topology([root], true));
    expect((await discovered).measurementRequested).toBe(false);
    expect((await measured).measurementRequested).toBe(true);
    expect((await duplicate).measurementRequested).toBe(true);
    expect(request).toHaveBeenCalledTimes(2);
  });

  it("counts APFS containers once and fails conservatively if any selected identity is unknown", async () => {
    const root = await directory(); const other = join(root, "other"); await mkdir(other);
    const request = vi.fn(async () => topology([root, other]));
    const containerIdentity = vi.fn(async () => "apfs:disk3");
    const service = new DeviceResourceService({ request } as unknown as Pick<EngineSupervisor, "request">, root, { platform: "darwin", containerIdentity });
    const known = await service.topology({ storageDirectories: [other] });
    expect(new Set(known.devices.map((device) => device.memoryGroupId))).toEqual(new Set(["apfs:disk3"]));
    containerIdentity.mockImplementation(async (path?: string) => { if (path === other) throw new Error("fixture failure"); return "apfs:disk3"; });
    const unknown = await service.topology({ storageDirectories: [other] });
    expect(new Set(unknown.devices.map((device) => device.memoryGroupId))).toEqual(new Set(["darwin:storage-unverified"]));
    expect(unknown.warnings.join(" ")).toMatch(/conservatively counted once/);
  });

  it("refreshes capacity while preserving a measured rate, returning independent copies", async () => {
    const root = await directory();
    let calls = 0;
    const request = vi.fn(async () => { const value = topology([root], calls++ === 0); value.devices[0]!.availableBytes = 1234 + calls; return value; });
    const service = new DeviceResourceService({ request } as unknown as Pick<EngineSupervisor, "request">, root, { platform: "linux" });
    const initial = await service.topology({ measure: true }); initial.devices[0]!.measurement!.bytesPerSecond = 999;
    const current = await service.topology();
    expect(current.devices[0]!.availableBytes).toBe(1236);
    expect(current.devices[0]!.measurement?.bytesPerSecond).toBe(1000);
    expect(current.measurementRequested).toBe(false);
  });

  it("rejects renderer supplied capacities, malformed requests and relative paths", async () => {
    const root = await directory(); const request = vi.fn();
    const service = new DeviceResourceService({ request } as unknown as Pick<EngineSupervisor, "request">, root);
    await expect(service.topology({ capacityBytes: 123 } as never)).rejects.toThrow(/Invalid/);
    await expect(service.topology(null as never)).rejects.toThrow(/Invalid/);
    await expect(service.topology({ storageDirectories: ["relative"] })).rejects.toThrow(/absolute/);
    expect(request).not.toHaveBeenCalled();
  });
});
