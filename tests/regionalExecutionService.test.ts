import { expect, it, vi } from "vitest";
import { RegionalExecutionService } from "../src/main/regionalExecutionService";
import type { EngineRequestContext } from "../src/main/engineSupervisor";
import type { DeviceResourceTopology, RegionalExecutionReview, RegionalExecutionStatus } from "../src/shared/deviceResources";

const topology: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1, measuredAt: "2026-10-15T00:00:00Z", platform: "fixture", architecture: "fixture",
  devices: [], links: [], measurementRequested: false, neuralModelConstructed: false, warnings: [] };
const status: RegionalExecutionStatus = { format: "omni-manual-regional-execution-status", version: 1, ownerId: "fixture", active: false, leaseValid: false, epoch: null, reason: "Fixture only",
  registeredRegions: 0, activeCalls: 0, routes: [], completedRegionDispatches: 0, actualActivationTransfers: 0, actualActivationTransferBytes: 0,
  trainingInapplicableCalls: 0, estimatedKnownTransferSeconds: 0, unknownTransferCosts: 0,
  trainingSupported: false, automaticDefault: false, contextGpuCapacityCredited: false, wholePathPerformanceMeasured: false, trace: [] };
function fixture() {
  let hash = "a".repeat(64), loaded = true;
  const dispatch = vi.fn(async (method: string, params: Record<string, unknown>, _signal: AbortSignal, context: EngineRequestContext) => {
    await context.beforeDispatch?.();
    if (method === "regional_execution_prepare") return { format: "omni-manual-regional-execution-review", version: 1, ownerId: "fixture",
      policySha256: "b".repeat(64), topologyIdentitySha256: "c".repeat(64), registrySha256: "d".repeat(64), topology: params.topology as DeviceResourceTopology,
      settings: { version: 1, enabled: true, mode: "manual", routes: [] }, start: params.start as number, total: 0, regions: [], status } satisfies RegionalExecutionReview;
    return { ...status, active: method === "regional_execution_activate", epoch: method === "regional_execution_activate" ? "fake-epoch" : null };
  });
  const observe = vi.fn(async (_owner: unknown, _signal: AbortSignal, context: EngineRequestContext) => { await context.beforeDispatch?.(); return structuredClone(topology); });
  const service = new RegionalExecutionService({ owner: async brainId => ({ brainId, storagePath: "/fake-loaded-owner", identitySha256: hash }),
    requireLoaded: () => { if (!loaded) throw new Error("Explicit Load required"); }, topology: observe,
    dispatch: dispatch as unknown as ConstructorParameters<typeof RegionalExecutionService>[0]["dispatch"] });
  return { service, dispatch, observe, change: () => { hash = "e".repeat(64); }, unload: () => { loaded = false; } };
}

it("uses main topology and one-use process-local review custody, never caller paths or copied hashes", async () => {
  const { service, dispatch, observe } = fixture();
  const prepared = await service.prepare({ brainId: "fixture", start: 0, count: 256 });
  prepared.topology.platform = "caller-mutated";
  await expect(service.activate({ brainId: "fixture", reviewId: prepared.reviewId })).resolves.toMatchObject({ active: true, epoch: "fake-epoch" });
  expect(observe).toHaveBeenCalledOnce();
  expect(dispatch.mock.calls[1]![1]).toMatchObject({ reviewed: { topology: { platform: "fixture" } } });
  await expect(service.activate({ brainId: "fixture", reviewId: prepared.reviewId })).rejects.toThrow("Prepare the current");
  await expect(service.prepare({ brainId: "fixture", topology } as never)).rejects.toThrow("main-owned");
  expect(dispatch).toHaveBeenCalledTimes(2);
});

it("invalidates reviews on settings changes, unload and host ownership retirement without neural reset", async () => {
  const first = fixture(), prepared = await first.service.prepare({ brainId: "fixture" }); first.change();
  await expect(first.service.activate({ brainId: "fixture", reviewId: prepared.reviewId })).rejects.toThrow("settings changed");
  expect(first.dispatch).toHaveBeenCalledOnce();
  const second = fixture(), old = await second.service.prepare({ brainId: "fixture" }); second.service.forget("fixture");
  await expect(second.service.activate({ brainId: "fixture", reviewId: old.reviewId })).rejects.toThrow("Prepare the current");
  second.unload(); await expect(second.service.status("fixture")).rejects.toThrow("Explicit Load");
  expect(second.dispatch).toHaveBeenCalledOnce();
});

it("rechecks authority in the acquired serial slot before activation dispatch", async () => {
  const { service, dispatch, change } = fixture(), prepared = await service.prepare({ brainId: "fixture" });
  dispatch.mockImplementationOnce(async (_method, _params, _signal, context) => { change(); await context.beforeDispatch?.(); throw new Error("Must not reach neural dispatch"); });
  await expect(service.activate({ brainId: "fixture", reviewId: prepared.reviewId })).rejects.toThrow("settings changed while waiting");
  await expect(service.activate({ brainId: "fixture", reviewId: prepared.reviewId })).rejects.toThrow("Prepare the current");
});
