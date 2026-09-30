import { EventEmitter } from "node:events";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { describe, expect, it, vi } from "vitest";
import { EngineSupervisor } from "../src/main/engineSupervisor";
import { createChatToolObservation } from "../src/main/chatToolObservation";
import type { ActionEvent } from "../src/shared/types";

class WarmControlPipe extends EventEmitter {
  readonly pid = 7347;
  killed = false;
  exitCode: number | null = null;
  readonly stdout = Object.assign(new EventEmitter(), { setEncoding: vi.fn() });
  readonly stderr = Object.assign(new EventEmitter(), { setEncoding: vi.fn() });
  readonly kill = vi.fn();
  frames: Array<Record<string, unknown>> = [];
  readonly stdin = { writable: true, write: (line: string, callback?: (error?: Error | null) => void) => {
    this.frames.push(JSON.parse(line) as Record<string, unknown>);
    callback?.(null); return true;
  } };
}
const time = "2026-09-30T12:00:00.000Z";
function source(): ActionEvent {
  return { id: "event", brainId: "brain", neuralActionId: "b".repeat(32),
    action: { kind: "tool", source: "brain", toolId: "web.fetch", action: "fetch", arguments: { url: "https://example.org" } },
    state: "complete", createdAt: time, updatedAt: time,
    execution: { id: "receipt", toolId: "web.fetch", action: "fetch", state: "complete", startedAt: time, finishedAt: time, output: "real observed text" } };
}
function setup() {
  const pipe = new WarmControlPipe();
  const engine = new EngineSupervisor({ appPath: process.cwd(), sendSignal: vi.fn() });
  const state = engine as unknown as {
    child: ChildProcessWithoutNullStreams; activeRequest: unknown;
    consumeLine(line: string): void; pending: Map<string, unknown>;
  };
  state.child = pipe as unknown as ChildProcessWithoutNullStreams;
  state.activeRequest = { method: "chat", brainId: "brain", turnId: "turn", cancelled: false, controller: new AbortController() };
  const event = source();
  const observation = createChatToolObservation("brain", "turn", event, event.execution!.output, () => true)!;
  return { engine, state, pipe, observation };
}

describe("live observation warm reader control", () => {
  it("writes the exact control immediately while a chat owns the neural queue and keeps the same PID", async () => {
    const value = setup();
    const promise = value.engine.observeChatAction(value.observation);
    const frame = value.pipe.frames[0]!;
    expect(frame.method).toBe("observe_chat_action");
    expect(frame.params).toEqual({ brainId: "brain", streamId: "turn", observation: value.observation });
    value.state.consumeLine(JSON.stringify({ jsonrpc: "2.0", id: frame.id,
      result: { brainId: "brain", turnId: "turn", observationId: value.observation.observationId, accepted: true } }));
    await expect(promise).resolves.toMatchObject({ accepted: true });
    expect(value.pipe.pid).toBe(7347);
    expect(value.pipe.kill).not.toHaveBeenCalled();
    expect(value.state.pending.size).toBe(0);
  });
  it("does not dispatch a sibling/retired owner", async () => {
    const value = setup();
    value.state.activeRequest = { method: "chat", brainId: "sibling", turnId: "other", cancelled: false, controller: new AbortController() };
    await expect(value.engine.observeChatAction(value.observation)).resolves.toMatchObject({ accepted: false, reason: "turn-not-active" });
    expect(value.pipe.frames).toHaveLength(0);
    expect(value.pipe.kill).not.toHaveBeenCalled();
  });
  it("rejects a sibling receipt without terminating the actual neural writer", async () => {
    const value = setup();
    const promise = value.engine.observeChatAction(value.observation);
    const frame = value.pipe.frames[0]!;
    value.state.consumeLine(JSON.stringify({ jsonrpc: "2.0", id: frame.id,
      result: { brainId: "brain", turnId: "sibling", observationId: value.observation.observationId, accepted: true } }));
    await expect(promise).rejects.toThrow("different turn/result");
    expect(value.pipe.kill).not.toHaveBeenCalled();
  });
  it("times out only the control receipt, never kills or replaces the warm worker", async () => {
    vi.useFakeTimers();
    try {
      const value = setup();
      const promise = value.engine.observeChatAction(value.observation);
      const failure = expect(promise).rejects.toThrow("timed out");
      await vi.advanceTimersByTimeAsync(10_001);
      await failure;
      expect(value.pipe.kill).not.toHaveBeenCalled();
      expect(value.state.child).toBe(value.pipe);
      expect(value.state.pending.size).toBe(0);
    } finally { vi.useRealTimers(); }
  });
});
