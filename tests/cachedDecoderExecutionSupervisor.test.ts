/** Constructor-free loaded transport tests; no worker/model is started or loaded. */
import { describe, expect, it, vi } from "vitest";
import { ENGINE_REQUEST_NO_DEADLINE, EngineSupervisor } from "../src/main/engineSupervisor";

describe("loaded-only cached decoder supervisor boundary", () => {
  it("uses the existing loaded child for no-deadline prepare/arm and flags-only status/Stop", async () => {
    const host = Object.create(EngineSupervisor.prototype) as EngineSupervisor;
    const state = host as unknown as Record<string, unknown>;
    const child = { killed: false, exitCode: null, stdin: { writable: true } };
    const rawRequest = vi.fn(async () => ({ fakeHost: true }));
    const release = vi.fn(), finish = vi.fn(), transition = vi.fn();
    Object.assign(state, { child, workerRole: "neural", starting: undefined, terminating: undefined, stopping: false,
      createRequestReservation: vi.fn((_method, _params, _priority, signal, context) =>
        ({ controller: { signal }, dispatched: false, beforeDispatch: context.beforeDispatch })),
      reserveRequestTurn: vi.fn(() => ({ ready: Promise.resolve(), release })), instanceDispatchHeld: vi.fn(() => false),
      transitionActivity: transition, finishRequestReservation: finish, rawRequest,
      start: vi.fn(() => { throw new Error("No implicit start"); }) });
    const signal = new AbortController().signal;
    const context = { requestId: "prepare-request", brainId: "brain-a", owner: "system" as const, label: "Fake host" };
    await expect(host.requestLoadedCachedDecoderExecution("cached_decoder_execution_prepare",
      { brainId: "brain-a", storagePath: "/owned/a" }, signal, context)).resolves.toEqual({ fakeHost: true });
    expect(rawRequest).toHaveBeenLastCalledWith("cached_decoder_execution_prepare",
      { brainId: "brain-a", storagePath: "/owned/a" }, ENGINE_REQUEST_NO_DEADLINE, signal);
    const operationId = "11111111-1111-4111-8111-111111111111";
    await expect(host.requestLoadedCachedDecoderExecution("cached_decoder_execution_arm",
      { brainId: "brain-a", storagePath: "/owned/a", operationId }, signal,
      { ...context, requestId: operationId })).resolves.toEqual({ fakeHost: true });
    expect(rawRequest).toHaveBeenLastCalledWith("cached_decoder_execution_arm",
      { brainId: "brain-a", storagePath: "/owned/a", operationId }, ENGINE_REQUEST_NO_DEADLINE, signal);
    await expect(host.controlLoadedCachedDecoderExecution("cached_decoder_execution_status",
      { brainId: "brain-a", storagePath: "/owned/a" })).resolves.toEqual({ fakeHost: true });
    expect(rawRequest).toHaveBeenLastCalledWith("cached_decoder_execution_status",
      { brainId: "brain-a", storagePath: "/owned/a" }, 10_000, undefined, true);
    await expect(host.requestLoadedCachedDecoderExecution("cached_decoder_execution_arm",
      { brainId: "brain-a", operationId: "other" }, signal, context)).rejects.toThrow(/main-seeded/);
    expect(state.start).not.toHaveBeenCalled(); expect(release).toHaveBeenCalledTimes(2); expect(finish).toHaveBeenCalledTimes(2);
    state.child = undefined;
    await expect(host.controlLoadedCachedDecoderExecution("cached_decoder_execution_stop", { brainId: "brain-a" }))
      .rejects.toThrow(/same explicitly loaded/);
  });

  it("turns an in-flight arm abort into exact flags-only Stop without killing or replacing the warm worker", async () => {
    const sendSignal = vi.fn(), host = new EngineSupervisor({ appPath: "/fixture", sendSignal });
    const state = host as unknown as Record<string, unknown>;
    const methods: Array<{ id: string; method: string; params: Record<string, unknown> }> = [];
    const terminate = vi.fn(async () => undefined);
    const child = { killed: false, exitCode: null, pid: 1234, stdin: { writable: true,
      write: (line: string, callback: (error?: Error | null) => void) => {
        const request = JSON.parse(line) as { id: string; method: string; params: Record<string, unknown> };
        methods.push(request); callback(null);
        if (request.method === "cached_decoder_execution_stop") queueMicrotask(() => {
          (host as unknown as { consumeLine(line: string): void }).consumeLine(JSON.stringify({ jsonrpc: "2.0", id: request.id, result: { stopped: true } }));
          const arm = methods.find(item => item.method === "cached_decoder_execution_arm")!;
          (host as unknown as { consumeLine(line: string): void }).consumeLine(JSON.stringify({ jsonrpc: "2.0", id: arm.id, result: { cancelled: true } }));
        });
        return true;
      } } };
    Object.assign(state, { child, terminateChild: terminate });
    const controller = new AbortController(), operationId = "22222222-2222-4222-8222-222222222222";
    const request = (host as unknown as { rawRequest(method: string, params: Record<string, unknown>, timeout: number, signal: AbortSignal): Promise<unknown> })
      .rawRequest("cached_decoder_execution_arm", { brainId: "brain-a", storagePath: "/owned/a", operationId },
        ENGINE_REQUEST_NO_DEADLINE, controller.signal);
    await Promise.resolve(); controller.abort(new Error("Stop exact cached arm"));
    await expect(request).resolves.toEqual({ cancelled: true });
    expect(methods.map(item => item.method)).toEqual(["cached_decoder_execution_arm", "cached_decoder_execution_stop"]);
    expect(methods[1]!.params).toEqual({ brainId: "brain-a", storagePath: "/owned/a", operationId });
    expect(sendSignal).not.toHaveBeenCalled(); expect(terminate).not.toHaveBeenCalled(); expect(state.child).toBe(child);
  });
});
