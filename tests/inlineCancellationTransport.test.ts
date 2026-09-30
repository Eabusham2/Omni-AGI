import { EventEmitter } from "node:events";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { describe, expect, it, vi } from "vitest";
import { ENGINE_REQUEST_NO_DEADLINE, EngineRequestError, EngineSupervisor } from "../src/main/engineSupervisor";

function fixture() {
  const child = Object.assign(new EventEmitter(), {
    pid: 7332, killed: false, exitCode: null, signalCode: null,
    kill: vi.fn(),
    stdin: { writable: true, write: vi.fn((_line: string, callback?: (error?: Error | null) => void) => {
      callback?.(null); return true;
    }) }
  });
  const engine = new EngineSupervisor({ appPath: process.cwd() });
  const internal = engine as unknown as {
    child: ChildProcessWithoutNullStreams; consumeLine(line: string): void;
  };
  internal.child = child as unknown as ChildProcessWithoutNullStreams;
  const reply = (index: number, result: unknown) => {
    const request = JSON.parse(String(child.stdin.write.mock.calls[index]?.[0])) as { id: string };
    internal.consumeLine(JSON.stringify({ jsonrpc: "2.0", id: request.id, result }));
  };
  return { child, engine, reply };
}

describe("exact inline cancellation transport", () => {
  it("yields a steered pre-dispatch request without writing chat, signalling or killing the warm worker", async () => {
    const { child, engine } = fixture();
    const chat = engine.request("chat", { streamId: "old" }, ENGINE_REQUEST_NO_DEADLINE, undefined,
      "foreground", { requestId: "old", owner: "chat", label: "Chat response", beforeDispatch: () => {
        throw new EngineRequestError("warm zero yield", -32801, { steered: true });
      } });
    await expect(chat).rejects.toMatchObject({ code: -32801 });
    expect(child.stdin.write).not.toHaveBeenCalled();
    expect(child.kill).not.toHaveBeenCalled();
    expect(engine.pid).toBe(child.pid);
  });
  it("sends true warm Steer directly to a busy generation without signalling or replacing its PID", async () => {
    const { child, engine, reply } = fixture();
    const chat = engine.request("chat", { streamId: "old" }, ENGINE_REQUEST_NO_DEADLINE);
    await vi.waitFor(() => expect(child.stdin.write).toHaveBeenCalledOnce());
    const steer = engine.steerChat("brain", "old", "new");
    expect(child.stdin.write).toHaveBeenCalledTimes(2);
    expect(JSON.parse(String(child.stdin.write.mock.calls[1]?.[0]))).toMatchObject({
      method: "steer_chat", params: { brainId: "brain", streamId: "old", successorTurnId: "new" }
    });
    reply(1, { requested: true, warm: true });
    await steer;
    expect(child.kill).not.toHaveBeenCalled();
    reply(0, { text: "actual prefix", steered: true });
    await expect(chat).resolves.toMatchObject({ text: "actual prefix", steered: true });
  });
  it("writes the owned control while the one neural request remains busy and never interrupts it", async () => {
    const { child, engine, reply } = fixture();
    const chat = engine.request("chat", { streamId: "turn" }, ENGINE_REQUEST_NO_DEADLINE);
    await vi.waitFor(() => expect(child.stdin.write).toHaveBeenCalledOnce());
    let chatSettled = false;
    void chat.then(() => { chatSettled = true; });
    const cancel = engine.cancelInlineGeneration("brain", "turn", "a".repeat(32));
    expect(child.stdin.write).toHaveBeenCalledTimes(2);
    const control = JSON.parse(String(child.stdin.write.mock.calls[1]?.[0]));
    expect(control).toMatchObject({ method: "cancel_inline_generation", params: {
      brainId: "brain", streamId: "turn", neuralActionId: "a".repeat(32)
    } });
    reply(1, { requested: true, acknowledged: false });
    await expect(cancel).resolves.toEqual({ requested: true, acknowledged: false });
    expect(chatSettled).toBe(false);
    expect(child.kill).not.toHaveBeenCalled();
    reply(0, { text: "exact saved reply" });
    await expect(chat).resolves.toEqual({ text: "exact saved reply" });
  });

  it("times out an unacknowledged control without killing or cancelling unrelated text", async () => {
    vi.useFakeTimers();
    try {
      const { child, engine, reply } = fixture();
      const chat = engine.request("chat", { streamId: "turn" }, ENGINE_REQUEST_NO_DEADLINE);
      await vi.advanceTimersByTimeAsync(0);
      const cancel = engine.cancelInlineGeneration("brain", "turn", "a".repeat(32));
      const error = cancel.catch((value: unknown) => value);
      await vi.advanceTimersByTimeAsync(10_001);
      await expect(error).resolves.toMatchObject({ message: expect.stringMatching(/timed out/) });
      expect(child.kill).not.toHaveBeenCalled();
      reply(0, { text: "reply survives the control timeout" });
      await expect(chat).resolves.toMatchObject({ text: "reply survives the control timeout" });
    } finally {
      vi.useRealTimers();
    }
  });
});
