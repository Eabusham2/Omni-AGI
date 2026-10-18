import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it } from "vitest";
import { DeviceResourceService } from "../src/main/deviceResourceService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import type { DeviceResourceTopology } from "../src/shared/deviceResources";

it("retains device rates only for the same physical interface and never overwrites fresher evidence", async () => {
  const directory = await mkdtemp(join(tmpdir(), "omni-device-rate-"));
  try {
    const rate = { bytesPerSecond: 1000, sampleBytes: 4096, runs: 3, medianNanoseconds: 4096000000,
      measuredAt: "2026-10-15T00:00:00Z", scope: "memory-copy" as const };
    for (const change of ["same", "backend", "fresh"] as const) {
      let calls = 0;
      const request = async () => { const first = calls++ === 0;
        return { format: "omni-device-resource-topology", version: 1, measuredAt: rate.measuredAt, platform: "fixture", architecture: "fixture",
          devices: [{ id: "gpu:fixture", label: "Fixture GPU", kind: "gpu", backend: change === "backend" && !first ? "xpu" : "cuda", executionDevice: "cuda:0",
            memoryGroupId: "gpu:fixture", totalBytes: 4096, availableBytes: first ? 2048 : 1024, allocatedBytes: 0, reservedBytes: 0, unifiedMemory: false,
            nativeComputeSupported: false, ...(first ? { measurement: rate } : change === "fresh" ? { measurement: { ...rate, bytesPerSecond: 2000 } } : {}) }],
          links: [], measurementRequested: first, neuralModelConstructed: false, warnings: [] } satisfies DeviceResourceTopology;
      };
      const service = new DeviceResourceService({ request } as unknown as Pick<EngineSupervisor, "request">, directory, { platform: "linux" });
      await service.topology({ measure: true }); const current = await service.topology();
      expect(current.devices[0]!.availableBytes).toBe(1024);
      expect(current.devices[0]!.measurement?.bytesPerSecond).toBe(change === "same" ? 1000 : change === "fresh" ? 2000 : undefined);
    }
  } finally { await rm(directory, { recursive: true, force: true }); }
});
