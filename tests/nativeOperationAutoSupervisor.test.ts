/** Fake pipe only; never starts a Python process or constructs a model. */
import { describe, expect, it, vi } from "vitest";
import { ENGINE_REQUEST_NO_DEADLINE, EngineSupervisor } from "../src/main/engineSupervisor";

function pipe() {
  const sendSignal = vi.fn(), host = new EngineSupervisor({ appPath: "/fixture", sendSignal });
  const state = host as unknown as Record<string, unknown>, terminate = vi.fn(async () => undefined);
  const methods: Array<{ id: string; method: string; params: Record<string, unknown> }> = [];
  const reply = (id: string, result: unknown) => (host as unknown as { consumeLine(line: string): void }).consumeLine(JSON.stringify({ jsonrpc: "2.0", id, result }));
  const child = { killed: false, exitCode: null, pid: 1234, stdin: { writable: true,
    write(line: string, callback: (error?: Error | null) => void) { const request = JSON.parse(line); methods.push(request); callback(null);
      if (request.method === "native_operation_auto_stop") queueMicrotask(() => {
        reply(request.id, { stopped: true }); reply(methods.find(item => item.method === "native_operation_auto_arm")!.id, { cancelled: true });
      }); return true; } } };
  Object.assign(state, { child, terminateChild: terminate });
  const raw = (method: string, params: Record<string, unknown>, signal: AbortSignal) =>
    (host as unknown as { rawRequest(method: string, params: Record<string, unknown>, timeout: number, signal: AbortSignal): Promise<unknown> })
      .rawRequest(method, params, ENGINE_REQUEST_NO_DEADLINE, signal);
  return { host, state, child, methods, reply, raw, sendSignal, terminate };
}
describe("loaded Auto cancellation does not replace the brain", () => {
  it("converts Arm abort to exact flags-only Stop and drains its ACK", async () => {
    const setup = pipe(), controller = new AbortController(), params = { brainId: "a", storagePath: "/owned/a", operationId: "main-operation" };
    const pending = setup.raw("native_operation_auto_arm", params, controller.signal); controller.abort();
    await expect(pending).resolves.toEqual({ cancelled: true });
    expect(setup.methods.map(value => value.method)).toEqual(["native_operation_auto_arm", "native_operation_auto_stop"]);
    expect(setup.methods[1]!.params).toEqual(params);
    expect(setup.sendSignal).not.toHaveBeenCalled(); expect(setup.terminate).not.toHaveBeenCalled(); expect(setup.state.child).toBe(setup.child);
  });
  it.each(["native_operation_auto_prepare", "hardware_topology"])("drains cancelled %s metadata without a process signal or replacement", async method => {
    const setup = pipe(), controller = new AbortController();
    const pending = setup.raw(method, { brainId: "a", storagePath: "/owned/a", measure: false }, controller.signal);
    controller.abort(); setup.reply(setup.methods[0]!.id, { discardedMetadata: true });
    await expect(pending).resolves.toEqual({ discardedMetadata: true });
    expect(setup.methods).toHaveLength(1); expect(setup.sendSignal).not.toHaveBeenCalled(); expect(setup.terminate).not.toHaveBeenCalled();
  });
  it("rejects a retired metadata intent after its existing serial owner's terminal ACK", async () => {
    const host = Object.create(EngineSupervisor.prototype) as EngineSupervisor, state = host as unknown as Record<string, unknown>;
    const child = { killed: false, exitCode: null, stdin: { writable: true } }, controller = new AbortController();
    const release = vi.fn(), transition = vi.fn();
    Object.assign(state, { child, workerRole: "neural", createRequestReservation: vi.fn((_method, _params, _priority, signal, context) =>
      ({ controller: { signal }, dispatched: false, beforeDispatch: context.beforeDispatch })),
      reserveRequestTurn: vi.fn(() => ({ ready: Promise.resolve(), release })), instanceDispatchHeld: vi.fn(() => false),
      transitionActivity: transition, finishRequestReservation: vi.fn(), rawRequest: vi.fn(async () => { controller.abort(new Error("Review retired")); return {}; }),
      start: vi.fn(() => { throw new Error("No implicit Start"); }) });
    await expect(host.requestLoadedNativeOperationAuto("native_operation_auto_prepare", { brainId: "a", storagePath: "/owned/a" }, controller.signal,
      { requestId: "review", brainId: "a", owner: "system", label: "metadata" })).rejects.toThrow(/retired/);
    expect(transition.mock.calls.at(-1)?.[1]).toBe("cancelled"); expect(release).toHaveBeenCalledOnce(); expect(state.start).not.toHaveBeenCalled();
  });
});
