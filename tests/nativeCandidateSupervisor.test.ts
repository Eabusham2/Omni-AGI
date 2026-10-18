import { describe, expect, it, vi } from "vitest";
import { EngineSupervisor } from "../src/main/engineSupervisor";

/** Constructor-free supervisor; fake pipe only, never a native brain/process. */
function fixture() {
  const host = Object.create(EngineSupervisor.prototype) as EngineSupervisor;
  const state = host as unknown as Record<string, unknown>;
  const child = { killed: false, exitCode: null, stdin: { writable: true } };
  const controller = new AbortController();
  const reservation = { controller, cancelled: false, dispatched: false, beforeDispatch: undefined as (() => void | Promise<void>) | undefined };
  let release!: () => void;
  const ready = new Promise<void>(resolve => { release = resolve; });
  const free = vi.fn();
  Object.assign(state, { workerRole: "neural", child,
    createRequestReservation: vi.fn((_method, _params, _priority, _signal, context) => { reservation.beforeDispatch = context.beforeDispatch; return reservation; }),
    reserveRequestTurn: vi.fn(() => ({ ready, release: free })),
    transitionActivity: vi.fn(), finishRequestReservation: vi.fn(), instanceDispatchHeld: vi.fn(() => false),
    rawRequest: vi.fn(async () => ({ committed: "fake-native-ack" })),
    start: vi.fn(() => { throw new Error("No implicit Start permitted"); }),
    claimForegroundWorker: vi.fn(() => { throw new Error("No preemptive replacement permitted"); }) });
  return { host, state, controller, child, release, free };
}
const context = { requestId: "owned-candidate", owner: "evolution" as const, label: "Fixture", brainId: "fixture" };

describe("loaded native candidate dispatch", () => {
  it("awaits the current async guard inside the serial slot and preserves a real late ACK", async () => {
    const { host, state, controller, release, free } = fixture();
    let validate!: () => void;
    const checked = new Promise<void>(resolve => { validate = resolve; });
    const beforeDispatch = vi.fn(() => checked);
    const work = host.requestLoadedNativeCandidate("evolution.propose", { brainId: "fixture" }, 120_000, controller.signal, { ...context, beforeDispatch });
    expect(beforeDispatch).not.toHaveBeenCalled(); release();
    await Promise.resolve(); expect(beforeDispatch).toHaveBeenCalledOnce();
    expect(state.rawRequest).not.toHaveBeenCalled(); validate();
    (state.rawRequest as ReturnType<typeof vi.fn>).mockImplementationOnce(async () => { controller.abort(); return { committed: "fake-native-ack" }; });
    await expect(work).resolves.toEqual({ committed: "fake-native-ack" });
    expect(state.start).not.toHaveBeenCalled(); expect(state.claimForegroundWorker).not.toHaveBeenCalled(); expect(free).toHaveBeenCalledOnce();
  });

  it("refuses a changed child or revoked guard without restarting or dispatching", async () => {
    const first = fixture();
    const changed = first.host.requestLoadedNativeCandidate("evolution.evaluate", { brainId: "fixture" }, 100, first.controller.signal, context);
    first.state.child = { ...first.child }; first.release();
    await expect(changed).rejects.toThrow("same explicitly loaded");
    expect(first.state.rawRequest).not.toHaveBeenCalled(); expect(first.state.start).not.toHaveBeenCalled();
    const second = fixture();
    const revoked = second.host.requestLoadedNativeCandidate("evolution.promote", { brainId: "fixture" }, 100, second.controller.signal,
      { ...context, beforeDispatch: async () => { throw new Error("Exact authority revoked"); } });
    second.release(); await expect(revoked).rejects.toThrow("Exact authority revoked");
    expect(second.state.rawRequest).not.toHaveBeenCalled(); expect(second.state.start).not.toHaveBeenCalled(); expect(second.free).toHaveBeenCalledOnce();
  });
});
