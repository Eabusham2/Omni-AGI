import { EventEmitter } from "node:events";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EngineSupervisor } from "../src/main/engineSupervisor";

class WarmPipe extends EventEmitter {
  pid = 7127; killed = false; exitCode: number | null = null;
  stdout = Object.assign(new EventEmitter(), { setEncoding: vi.fn() });
  stderr = Object.assign(new EventEmitter(), { setEncoding: vi.fn() });
  kill = vi.fn(); frames: Array<Record<string, unknown>> = [];
  stdin = { writable: true, write: (line: string, callback?: (error?: Error | null) => void) => {
    this.frames.push(JSON.parse(line) as Record<string, unknown>); callback?.(null); return true;
  } };
}
const actionId = "a".repeat(32), params = { brainId: "brain", neuralActionId: actionId, modality: "image", conceptIds: ["fixture"] };
function setup() {
  const pipe = new WarmPipe(), engine = new EngineSupervisor({ appPath: process.cwd(), sendSignal: vi.fn() });
  const state = engine as unknown as { child: ChildProcessWithoutNullStreams; activeRequest: unknown; consumeLine(line: string): void; pending: Map<string, unknown> };
  const active = { method: "chat", brainId: "brain", turnId: "turn", controller: new AbortController(), cancelled: false };
  state.child = pipe as unknown as ChildProcessWithoutNullStreams; state.activeRequest = active;
  const ack = (frame: Record<string, unknown>, result: unknown) => state.consumeLine(JSON.stringify({ jsonrpc: "2.0", id: frame.id, result }));
  return { pipe, engine, state, active, ack };
}
afterEach(() => vi.useRealTimers());
describe("flags-only warm Ask artifact control", () => {
  it("deposits exact authorization beside a live chat without queueing a second neural call or changing its PID", async () => {
    const value = setup();
    const fields = { brainId: "brain", streamId: "turn", neuralActionId: actionId, executionId: "receipt", intentPath: "owned-intent", argumentsJson: "{}", argumentSha256: "b".repeat(64) };
    const pending = value.engine.authorizeInlineImagination(fields), frame = value.pipe.frames[0]!;
    expect(frame.method).toBe("authorize_inline_imagination"); expect(frame.params).toEqual(fields);
    value.ack(frame, { accepted: true, started: false, brainId: "brain", streamId: "turn", neuralActionId: actionId });
    await expect(pending).resolves.toMatchObject({ accepted: true, started: false });
    expect(value.state.activeRequest).toBe(value.active); expect(value.pipe.kill).not.toHaveBeenCalled(); expect(value.pipe.pid).toBe(7127);
  });
  it("waits through reader metadata while text generation keeps its serial ownership; only a ready artifact becomes claimable", async () => {
    vi.useFakeTimers(); const value = setup(), pending = value.engine.awaitInlineArtifact(params);
    const state = { exists: true, brainId: "brain", neuralActionId: actionId, streamId: "turn", started: true, ready: false, cancelled: false };
    value.ack(value.pipe.frames[0]!, state); await vi.advanceTimersByTimeAsync(101);
    expect(value.pipe.frames.map(frame => frame.method)).toEqual(["inline_generation_status", "inline_generation_status"]);
    expect(value.state.activeRequest).toBe(value.active);
    value.ack(value.pipe.frames[1]!, { ...state, ready: true }); await expect(pending).resolves.toBe(true);
    expect(value.pipe.kill).not.toHaveBeenCalled(); expect(value.state.pending.size).toBe(0);
  });
  it("cancels only the exact waiting artifact, never the live text response or worker", async () => {
    vi.useFakeTimers(); const value = setup(), abort = new AbortController(), pending = value.engine.awaitInlineArtifact(params, abort.signal);
    const rejected = expect(pending).rejects.toThrow("cancelled");
    value.ack(value.pipe.frames[0]!, { exists: true, brainId: "brain", neuralActionId: actionId, streamId: "turn", started: true, ready: false, cancelled: false });
    await vi.advanceTimersByTimeAsync(0); abort.abort(); await vi.advanceTimersByTimeAsync(0);
    const cancel = value.pipe.frames[1]!; expect(cancel.method).toBe("cancel_inline_generation");
    expect(cancel.params).toEqual({ brainId: "brain", streamId: "turn", neuralActionId: actionId });
    value.ack(cancel, { requested: true, acknowledged: true }); await rejected;
    expect(value.state.activeRequest).toBe(value.active); expect(value.active.controller.signal.aborted).toBe(false); expect(value.pipe.kill).not.toHaveBeenCalled();
  });
  it("rejects sibling control ACK/status while missing ownership remains an honest first-generation path", async () => {
    const value = setup(), authorization = value.engine.authorizeInlineImagination({ brainId: "brain", streamId: "turn", neuralActionId: actionId });
    value.ack(value.pipe.frames[0]!, { accepted: true, brainId: "other", streamId: "turn", neuralActionId: actionId });
    await expect(authorization).rejects.toThrow("different artifact");
    const waiting = value.engine.awaitInlineArtifact(params);
    value.ack(value.pipe.frames[1]!, { exists: true, brainId: "other", neuralActionId: actionId, streamId: "turn", ready: true, cancelled: false });
    await expect(waiting).rejects.toThrow("different owned");
    const absent = value.engine.awaitInlineArtifact(params); value.ack(value.pipe.frames[2]!, { exists: false });
    await expect(absent).resolves.toBe(false); expect(value.pipe.kill).not.toHaveBeenCalled();
  });
});
