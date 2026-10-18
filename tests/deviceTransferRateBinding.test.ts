import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it } from "vitest";
import { DeviceResourceService } from "../src/main/deviceResourceService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import type { DeviceResourceTopology } from "../src/shared/deviceResources";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
function profile(measured: boolean): DeviceResourceTopology {
  const rate = { bytesPerSecond: 1000, sampleBytes: 4096, runs: 3, medianNanoseconds: 4096000000, measuredAt: "2026-10-15T00:00:00Z", scope: "device-transfer" as const };
  return { format: "omni-device-resource-topology", version: 1, measuredAt: measured ? "2026-10-15T00:00:00Z" : "2026-10-15T00:01:00Z", platform: "fixture", architecture: "fixture", measurementRequested: measured, neuralModelConstructed: false, warnings: [],
    devices: [{ id: "ram:system", label: "Fixture RAM", kind: "ram", backend: "cpu", memoryGroupId: "ram:system", totalBytes: 4096, availableBytes: measured ? 2048 : 1024, allocatedBytes: 0, reservedBytes: 0, unifiedMemory: false, executionDevice: "cpu", nativeComputeSupported: true },
      ...["A", "B"].map((name, index) => ({ id: `cuda:${name}`, label: `Fixture GPU ${name}`, kind: "gpu" as const, backend: "cuda" as const, memoryGroupId: `cuda:${name}`, totalBytes: 4096, availableBytes: measured ? 2048 : 1536, allocatedBytes: 0, reservedBytes: 0, unifiedMemory: false, executionDevice: `cuda:${index}`, nativeComputeSupported: false, unavailableReason: "fixture-does-not-assert-native-CUDA-support" }))],
    links: [{ from: "cuda:A", to: "cuda:B", peerAccess: true }, { from: "cuda:B", to: "cuda:A", peerAccess: false }, { from: "ram:system", to: "cuda:A", peerAccess: null }].map(link => ({ ...link, ...(measured ? { measurement: { ...rate } } : {}) })) };
}

it("reuses only unchanged physical-route rates while fresh peer/backend/group changes invalidate their old evidence", async () => {
  for (const change of ["peer", "backend", "group"] as const) {
    const directory = await mkdtemp(join(tmpdir(), "omni-transfer-binding-")); roots.push(directory); let calls = 0;
    const request = async () => { const value = profile(calls++ === 0); if (calls === 2) {
      if (change === "peer") value.links[0]!.peerAccess = false;
      else if (change === "backend") { value.devices[2]!.backend = "xpu"; value.devices[2]!.executionDevice = "xpu:0"; }
      else value.devices[2]!.memoryGroupId = "physical:replacement-B";
    } return value; };
    const service = new DeviceResourceService({ request } as unknown as Pick<EngineSupervisor, "request">, directory, { platform: "linux" });
    const old = await service.topology({ measure: true }); expect(old.links[0]!.measurement?.bytesPerSecond).toBe(1000);
    const current = await service.topology(); expect(current.links[0]!.measurement).toBeUndefined(); expect(current.measurementRequested).toBe(false);
    expect(current.links[2]!.measurement?.bytesPerSecond).toBe(1000); expect(current.devices[0]!.availableBytes).toBe(1024); expect(current.devices[1]!.availableBytes).toBe(1536);
    expect(current.links[1]!.measurement?.bytesPerSecond).toBe(change === "peer" ? 1000 : undefined); expect(calls).toBe(2);
  }
});
