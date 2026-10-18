/** Real supervisor scheduling, constructor-free stub transport; no worker/model. */
import { describe, expect, it, vi } from "vitest";
import { EngineSupervisor } from "../src/main/engineSupervisor";
const deferred = <T>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(yes => { resolve = yes; }); return { promise, resolve }; };
const context = (id: string) => ({ requestId: id, brainId: "teacher-brain", jobId: id, label: "Explicit local teacher fixture" });
function fixture(role: "neural" | "inspection" = "neural") {
  const engine = new EngineSupervisor({ appPath: process.cwd() }, role);
  const start = vi.spyOn(engine as unknown as { startCandidates(): Promise<boolean> }, "startCandidates").mockResolvedValue(true);
  const rpc = vi.spyOn(engine as unknown as { rawRequest(...args: unknown[]): Promise<unknown> }, "rawRequest").mockResolvedValue("native-result");
  const preempt = vi.spyOn(engine as unknown as { claimForegroundWorker(): Promise<void> }, "claimForegroundWorker").mockResolvedValue(undefined);
  const terminate = vi.spyOn(engine as unknown as { terminateChild(): Promise<void> }, "terminateChild").mockResolvedValue(undefined);
  return { engine, start, rpc, preempt, terminate };
}

describe("local teacher/native shared compute ordering", () => {
  it("host-before-native prevents later startup, foreground preemption and RPC until release", async () => {
    const f = fixture(), release = deferred<void>(), entered = deferred<void>();
    const host = f.engine.withHostCompute(context("host-first"), new AbortController().signal, async () => { entered.resolve(); await release.promise; return "teacher-result"; });
    await entered.promise;
    const native = f.engine.request("fixture_native"); await new Promise(resolve => setTimeout(resolve, 5));
    expect(f.start).not.toHaveBeenCalled(); expect(f.preempt).not.toHaveBeenCalled(); expect(f.rpc).not.toHaveBeenCalled();
    release.resolve(); expect(await host).toBe("teacher-result"); expect(await native).toBe("native-result");
  });
  it("an earlier native cold start and request complete before the host, without killing them", async () => {
    const f = fixture(), startup = deferred<boolean>(), finishNative = deferred<unknown>();
    f.start.mockImplementationOnce(() => startup.promise); f.rpc.mockImplementationOnce(() => finishNative.promise);
    const native = f.engine.request("earlier_native"); await vi.waitFor(() => expect(f.start).toHaveBeenCalled());
    const work = vi.fn(async () => "teacher"), host = f.engine.withHostCompute(context("host-after-start"), new AbortController().signal, work);
    expect(work).not.toHaveBeenCalled(); startup.resolve(true); await vi.waitFor(() => expect(f.rpc).toHaveBeenCalled()); expect(work).not.toHaveBeenCalled();
    finishNative.resolve("old-native-done"); expect(await native).toBe("old-native-done"); expect(await host).toBe("teacher"); expect(f.terminate).not.toHaveBeenCalled();
  });
  it("cancelled pending host detaches its waiter and leaves the earlier native owner intact", async () => {
    const f = fixture(), finish = deferred<unknown>(), cancel = new AbortController(); f.rpc.mockImplementationOnce(() => finish.promise);
    const native = f.engine.request("earlier_native"); await vi.waitFor(() => expect(f.rpc).toHaveBeenCalled());
    const work = vi.fn(async () => undefined), host = f.engine.withHostCompute(context("cancelled-host"), cancel.signal, work);
    cancel.abort(new Error("fixture stop pending teacher")); await expect(host).rejects.toThrow(/stop pending/);
    expect(work).not.toHaveBeenCalled(); expect(f.terminate).not.toHaveBeenCalled(); finish.resolve("done"); await expect(native).resolves.toBe("done");
    await expect(f.engine.request("later_native")).resolves.toBe("native-result");
  });
  it("a failed host releases scheduling and never enters native ingestion reentrantly", async () => {
    const f = fixture();
    await expect(f.engine.withHostCompute(context("failed-host"), new AbortController().signal, async () => {
      await expect(f.engine.request("ingest")).rejects.toThrow(/release it first/);
      await expect(f.engine.start()).rejects.toThrow(/Release/);
      await expect(f.engine.claimForeground()).rejects.toThrow(/Release/);
      throw new Error("fixture teacher failed");
    })).rejects.toThrow(/teacher failed/);
    expect(f.start).not.toHaveBeenCalled(); expect(f.rpc).not.toHaveBeenCalled();
    await expect(f.engine.request("after_failure")).resolves.toBe("native-result");
  });
  it("later inspection delegation cannot cold-start while the host owns compute", async () => {
    const f = fixture(), entered = deferred<void>(), release = deferred<void>(), inspection = fixture("inspection");
    const delegate = vi.spyOn(f.engine as unknown as { inspectionSupervisor(): EngineSupervisor }, "inspectionSupervisor").mockReturnValue(inspection.engine);
    const host = f.engine.withHostCompute(context("host-inspection"), new AbortController().signal, async () => { entered.resolve(); await release.promise; }); await entered.promise;
    const query = f.engine.request("query_cortex"); await new Promise(resolve => setTimeout(resolve, 5));
    expect(delegate).not.toHaveBeenCalled(); expect(inspection.start).not.toHaveBeenCalled();
    release.resolve(); await host; await query; expect(delegate).toHaveBeenCalled();
  });
  it("earlier inspection startup is included in the shared predecessor boundary", async () => {
    const f = fixture(), inspection = fixture("inspection"), startup = deferred<boolean>();
    Object.assign(f.engine, { inspectionWorker: inspection.engine }); inspection.start.mockImplementationOnce(() => startup.promise);
    const query = f.engine.request("query_cortex"); await vi.waitFor(() => expect(inspection.start).toHaveBeenCalled());
    const work = vi.fn(async () => undefined), host = f.engine.withHostCompute(context("after-inspection"), new AbortController().signal, work);
    expect(work).not.toHaveBeenCalled(); startup.resolve(true); await query; await host; expect(work).toHaveBeenCalledOnce();
  });
  it("direct startup/health-style callers wait without starting a model for the host", async () => {
    const f = fixture(), entered = deferred<void>(), release = deferred<void>();
    const host = f.engine.withHostCompute(context("host-direct-start"), new AbortController().signal, async () => { entered.resolve(); await release.promise; }); await entered.promise;
    const startup = f.engine.start(); await new Promise(resolve => setTimeout(resolve, 5)); expect(f.start).not.toHaveBeenCalled();
    release.resolve(); await host; await startup; expect(f.start).toHaveBeenCalledOnce(); expect(f.rpc).not.toHaveBeenCalled();
  });
  it("shutdown cancels and awaits the exact owned host before teardown", async () => {
    const f = fixture(), entered = deferred<void>();
    const host = f.engine.withHostCompute(context("shutdown-host"), new AbortController().signal, async signal => {
      entered.resolve(); await new Promise<void>((_resolve, reject) => signal.addEventListener("abort", () => reject(signal.reason), { once: true }));
    });
    const failed = expect(host).rejects.toThrow(/shutting down/); await entered.promise; await f.engine.stop(); await failed;
    expect(f.start).not.toHaveBeenCalled(); expect(f.terminate).toHaveBeenCalledOnce();
  });
});
