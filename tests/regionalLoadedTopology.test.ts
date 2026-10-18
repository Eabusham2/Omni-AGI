import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it, vi } from "vitest";
import { DeviceResourceService } from "../src/main/deviceResourceService";
import type { EngineRequestContext, EngineSupervisor } from "../src/main/engineSupervisor";
import type { DeviceResourceTopology } from "../src/shared/deviceResources";

it("collects regional topology on the guarded loaded pipe, never general worker discovery", async () => {
  const directory = await mkdtemp(join(tmpdir(), "omni-regional-topology-"));
  try {
    const profile: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1, measuredAt: "2026-10-15T00:00:00Z", platform: "fixture", architecture: "fixture",
      devices: [], links: [], measurementRequested: false, neuralModelConstructed: false, warnings: [] };
    const guard = vi.fn(), request = vi.fn(() => { throw new Error("General discovery could Start a worker"); });
    const loaded = vi.fn(async (_method: string, _params: unknown, _timeout: number, _signal: AbortSignal, context: EngineRequestContext) => {
      await context.beforeDispatch?.(); return structuredClone(profile);
    });
    const engine = { request, requestLoadedRegionalExecution: loaded } as unknown as Pick<EngineSupervisor, "request" | "requestLoadedRegionalExecution">;
    const service = new DeviceResourceService(engine, directory, { platform: "linux" });
    const context = { requestId: "regional-topology", brainId: "fixture", owner: "system" as const, label: "Fixture", beforeDispatch: guard };
    await expect(service.topologyForLoadedOwner("fixture", "/fake-owner", new AbortController().signal, context)).resolves.toMatchObject({ platform: "fixture" });
    expect(loaded).toHaveBeenCalledOnce(); expect(loaded.mock.calls[0]![0]).toBe("hardware_topology");
    expect(loaded.mock.calls[0]![1]).toMatchObject({ brainId: "fixture", storagePath: "/fake-owner", measure: false });
    expect(guard).toHaveBeenCalledOnce(); expect(request).not.toHaveBeenCalled();
    await expect(service.topologyForLoadedOwner("foreign", "/fake-owner", new AbortController().signal, context)).rejects.toThrow("unavailable");
    expect(loaded).toHaveBeenCalledOnce();
  } finally { await rm(directory, { recursive: true, force: true }); }
});
