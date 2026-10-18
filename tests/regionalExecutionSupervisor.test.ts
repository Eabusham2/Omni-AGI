import { expect, it, vi } from "vitest";
import { EngineSupervisor } from "../src/main/engineSupervisor";

it("serializes regional observations on the same loaded pipe and refuses replacement or foreign ownership", async () => {
  const host = Object.create(EngineSupervisor.prototype) as EngineSupervisor;
  const state = host as unknown as Record<string, unknown>;
  const child = { killed: false, exitCode: null, stdin: { writable: true } };
  const abort = new AbortController();
  let acquire!: () => void; const ready = new Promise<void>(resolve => { acquire = resolve; });
  const guard = vi.fn(), release = vi.fn();
  Object.assign(state, { child, workerRole: "neural", createRequestReservation: vi.fn((_method, _params, _priority, _signal, context) =>
    ({ controller: abort, dispatched: false, beforeDispatch: context.beforeDispatch })), reserveRequestTurn: () => ({ ready, release }),
    instanceDispatchHeld: () => false, transitionActivity: vi.fn(), finishRequestReservation: vi.fn(), rawRequest: vi.fn(async () => ({ fake: true })),
    start: vi.fn(() => { throw new Error("No implicit start"); }) });
  const context = { requestId: "regional-review", brainId: "fixture", owner: "system" as const, label: "Fixture", beforeDispatch: guard };
  const run = host.requestLoadedRegionalExecution("regional_execution_status", { brainId: "fixture" }, 100, abort.signal, context);
  expect(guard).not.toHaveBeenCalled(); acquire(); await expect(run).resolves.toEqual({ fake: true });
  expect(guard).toHaveBeenCalledOnce(); expect(state.start).not.toHaveBeenCalled(); expect(release).toHaveBeenCalledOnce();
  await expect(host.requestLoadedRegionalExecution("regional_execution_activate", { brainId: "foreign" }, 100, abort.signal, context)).rejects.toThrow("exact main-owned");
  state.child = undefined;
  await expect(host.requestLoadedRegionalExecution("hardware_topology", { brainId: "fixture" }, 100, abort.signal, context)).rejects.toThrow("same explicitly loaded");
  expect(state.rawRequest).toHaveBeenCalledOnce(); expect(state.start).not.toHaveBeenCalled();
});
