import { describe, expect, it, vi } from "vitest";
import { BrainService, type RuntimeJobManager } from "../src/main/brainService";
import { ToolExecutor } from "../src/main/toolExecutor";

describe("main-owned portable destination admission", () => {
  it("blocks neural Load and creation without blocking settings inspection", async () => {
    const service = Object.create(BrainService.prototype) as BrainService;
    const load = vi.fn(), setAutoLoad = vi.fn(async () => ({ phase: "unloaded" }));
    Object.assign(service, { runtimeShutdownHolds: new Set(), instanceLifecycle: { load, setAutoLoad } });
    service.setRuntimeAdmissionGuard(() => { throw new Error("Review this portable destination first"); });
    expect(() => service.loadInstance("fixture")).toThrow("portable destination");
    await expect(service.create({} as never)).rejects.toThrow("portable destination");
    await expect(service.setInstanceAutoLoad("fixture", false)).resolves.toMatchObject({ phase: "unloaded" });
    expect(load).not.toHaveBeenCalled(); expect(setAutoLoad).toHaveBeenCalledOnce();
  });

  it("rechecks the destination gate after awaited permission before any host effect", async () => {
    let admitted = true;
    const service = { listToolPermissions: vi.fn(async () => {
      admitted = false; return [{ toolId: "brain.substrate", level: "full", revision: 1 }];
    }) } as unknown as BrainService;
    const executor = new ToolExecutor(service, {} as RuntimeJobManager, undefined, { execute: vi.fn() });
    const execute = vi.fn(async () => ({ state: "queued" })); executor.setSparseSubstrateService({ execute });
    vi.spyOn(executor as unknown as { audit(): Promise<void> }, "audit").mockResolvedValue(undefined);
    executor.setRuntimeAdmissionGuard(() => { if (!admitted) throw new Error("Destination admission changed"); });
    const result = await executor.execute({ brainId: "portable-fixture", toolId: "brain.substrate", action: "apply", arguments: {} });
    expect(result).toMatchObject({ state: "failed", error: "Destination admission changed" });
    expect(execute).not.toHaveBeenCalled();
    expect(executor.preferences()).toEqual({ approvalTimeoutSeconds: 30 });
  });
});
