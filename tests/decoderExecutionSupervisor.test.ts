import { describe, expect, it, vi } from "vitest";
import { ENGINE_REQUEST_NO_DEADLINE, EngineSupervisor } from "../src/main/engineSupervisor";

describe("loaded-only decoder supervisor boundary", () => {
  it("serializes mutations on the same child while status/Stop stay bytes-only controls", async () => {
    const host = Object.create(EngineSupervisor.prototype) as EngineSupervisor;
    const state = host as unknown as Record<string, unknown>;
    const child = { killed: false, exitCode: null, stdin: { writable: true } };
    const rawRequest = vi.fn(async () => ({ fixture: true }));
    const release = vi.fn(), finish = vi.fn(), transition = vi.fn();
    Object.assign(state, { child, workerRole: "neural", starting: undefined, terminating: undefined, stopping: false,
      createRequestReservation: vi.fn((_method, _params, _priority, signal, context) =>
        ({ controller: { signal }, dispatched: false, beforeDispatch: context.beforeDispatch })),
      reserveRequestTurn: vi.fn(() => ({ ready: Promise.resolve(), release })), instanceDispatchHeld: vi.fn(() => false),
      transitionActivity: transition, finishRequestReservation: finish, rawRequest,
      start: vi.fn(() => { throw new Error("No implicit start"); }) });
    const signal = new AbortController().signal;
    const captureContext = { requestId: "capture-request", brainId: "brain-a", owner: "system" as const, label: "Fixture" };
    await expect(host.requestLoadedDecoderExecution("decoder_execution_capture_next",
      { brainId: "brain-a", storagePath: "/owned/a" }, signal, captureContext)).resolves.toEqual({ fixture: true });
    expect(rawRequest).toHaveBeenLastCalledWith("decoder_execution_capture_next",
      { brainId: "brain-a", storagePath: "/owned/a" }, 30_000, signal);
    const operationId = "11111111-1111-4111-8111-111111111111";
    await expect(host.requestLoadedDecoderExecution("decoder_execution_calibrate",
      { brainId: "brain-a", storagePath: "/owned/a", operationId }, signal,
      { ...captureContext, requestId: operationId })).resolves.toEqual({ fixture: true });
    expect(rawRequest).toHaveBeenLastCalledWith("decoder_execution_calibrate",
      { brainId: "brain-a", storagePath: "/owned/a", operationId }, ENGINE_REQUEST_NO_DEADLINE, signal);
    await expect(host.controlLoadedDecoderExecution("decoder_execution_status",
      { brainId: "brain-a", storagePath: "/owned/a" })).resolves.toEqual({ fixture: true });
    expect(rawRequest).toHaveBeenLastCalledWith("decoder_execution_status",
      { brainId: "brain-a", storagePath: "/owned/a" }, 10_000, undefined, true);
    expect(state.start).not.toHaveBeenCalled(); expect(release).toHaveBeenCalledTimes(2); expect(finish).toHaveBeenCalledTimes(2);
    await expect(host.requestLoadedDecoderExecution("decoder_execution_apply", { brainId: "other" }, signal,
      captureContext)).rejects.toThrow(/exact main-owned/);
    state.child = undefined;
    await expect(host.controlLoadedDecoderExecution("decoder_execution_stop", { brainId: "brain-a" })).rejects.toThrow(/same explicitly loaded/);
  });

  it("uses exact decoder Stop on abort and never signals, kills or timeout-restarts the warm worker", async () => {
    const sendSignal = vi.fn(), host = new EngineSupervisor({ appPath: "/fixture", sendSignal });
    const state = host as unknown as Record<string, unknown>;
    const methods: Array<{ id: string; method: string; params: Record<string, unknown> }> = [];
    const terminate = vi.fn(async () => undefined);
    const child = { killed: false, exitCode: null, pid: 1234, stdin: { writable: true,
      write: (line: string, callback: (error?: Error | null) => void) => {
        const request = JSON.parse(line) as { id: string; method: string; params: Record<string, unknown> };
        methods.push(request); callback(null);
        if (request.method === "decoder_execution_stop") queueMicrotask(() => {
          (host as unknown as { consumeLine(line: string): void }).consumeLine(JSON.stringify({ jsonrpc: "2.0", id: request.id, result: { stopped: true } }));
          const calibration = methods.find(item => item.method === "decoder_execution_calibrate")!;
          (host as unknown as { consumeLine(line: string): void }).consumeLine(JSON.stringify({ jsonrpc: "2.0", id: calibration.id, result: { cancelled: true } }));
        });
        return true;
      } } };
    Object.assign(state, { child, terminateChild: terminate });
    const controller = new AbortController(), operationId = "22222222-2222-4222-8222-222222222222";
    const request = (host as unknown as { rawRequest(method: string, params: Record<string, unknown>, timeout: number, signal: AbortSignal): Promise<unknown> })
      .rawRequest("decoder_execution_calibrate", { brainId: "brain-a", storagePath: "/owned/a", operationId },
        ENGINE_REQUEST_NO_DEADLINE, controller.signal);
    await Promise.resolve(); controller.abort(new Error("Stop exact calibration"));
    await expect(request).resolves.toEqual({ cancelled: true });
    expect(methods.map(item => item.method)).toEqual(["decoder_execution_calibrate", "decoder_execution_stop"]);
    expect(methods[1]!.params).toEqual({ brainId: "brain-a", storagePath: "/owned/a", operationId });
    expect(sendSignal).not.toHaveBeenCalled(); expect(terminate).not.toHaveBeenCalled(); expect(state.child).toBe(child);
  });
});
