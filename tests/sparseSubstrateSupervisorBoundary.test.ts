import { describe, expect, it, vi } from "vitest";
import { EngineSupervisor, BackgroundRequestDeferredError } from "../src/main/engineSupervisor";

/** Supervisor shell only; no constructor, child process, worker or neural state. */
function shell() {
  const host = Object.create(EngineSupervisor.prototype) as EngineSupervisor;
  const state = host as unknown as Record<string, unknown>;
  const child = { killed: false, exitCode: null, stdin: { writable: true } };
  const controller = new AbortController();
  const reservation = { controller, cancelled: false, dispatched: false, beforeDispatch: undefined as (() => void) | undefined };
  Object.assign(state, { workerRole: "neural", child, requestReservations: new Set(), hostComputeReservations: new Map(),
    createRequestReservation: vi.fn((_method, _params, _priority, _signal, context) => { reservation.beforeDispatch = context.beforeDispatch; return reservation; }),
    reserveRequestTurn: vi.fn(() => ({ ready: Promise.resolve(), release: vi.fn() })),
    transitionActivity: vi.fn(), finishRequestReservation: vi.fn(), instanceDispatchHeld: vi.fn(() => false),
    rawRequest: vi.fn(async () => ({ committed: "fixture-only" })), start: vi.fn(() => { throw new Error("Unexpected worker start"); }) });
  return { host, state, child, controller, reservation };
}

describe("sparse substrate existing-owner supervisor boundaries", () => {
  it("routes all committed readers to inspection without starting or reserving a neural worker", async () => {
    const { host, state } = shell();
    const request = vi.fn(async () => ({ scope: "fixture" }));
    state.inspectionSupervisor = vi.fn(() => ({ request }));
    for (const method of ["sparse_substrate_inspect", "sparse_substrate_export", "sparse_substrate_validate_bundle", "sparse_substrate_prepare", "sparse_substrate_head"]) {
      await expect(host.request(method, { brainId: "fixture" })).resolves.toEqual({ scope: "fixture" });
    }
    expect(request).toHaveBeenCalledTimes(5);
    expect(state.start).not.toHaveBeenCalled(); expect(state.createRequestReservation).not.toHaveBeenCalled();
  });

  it("uses the existing serial writer and passes cancellation without any implicit load", async () => {
    const { host, state, controller, child } = shell(), guard = vi.fn();
    const params = { brainId: "fixture", storagePath: "/fixture", planPath: "/fixture/substrate-transfers/plans/one.sqlite3" };
    await expect(host.applySparseSubstrate(params, controller.signal, guard)).resolves.toEqual({ committed: "fixture-only" });
    expect(guard).toHaveBeenCalledOnce(); expect(state.start).not.toHaveBeenCalled();
    expect(state.rawRequest).toHaveBeenCalledWith("sparse_substrate_apply", params, 0, controller.signal);
    (state.requestReservations as Set<unknown>).add({ cancelled: false });
    await expect(host.applySparseSubstrate(params, controller.signal, guard)).rejects.toBeInstanceOf(BackgroundRequestDeferredError);
    (state.requestReservations as Set<unknown>).clear(); child.killed = true;
    await expect(host.applySparseSubstrate(params, controller.signal, guard)).rejects.toThrow("existing neural worker");
    expect(state.rawRequest).toHaveBeenCalledOnce(); expect(state.start).not.toHaveBeenCalled();
  });
});
