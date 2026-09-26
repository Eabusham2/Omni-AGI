import { EventEmitter } from "node:events";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { describe, expect, it, vi } from "vitest";
import { ChatActionController } from "../src/main/chatActionController";
import {
  ENGINE_REQUEST_NO_DEADLINE,
  EngineSupervisor
} from "../src/main/engineSupervisor";
import type { ChatResult } from "../src/shared/types";

class WarmWorker extends EventEmitter {
  readonly pid = 7331;
  killed = false;
  exitCode: number | null = null;
  signalCode: NodeJS.Signals | null = null;
  readonly stdout = Object.assign(new EventEmitter(), { setEncoding: vi.fn() });
  readonly stderr = Object.assign(new EventEmitter(), { setEncoding: vi.fn() });
  readonly stdin = {
    writable: true,
    write: vi.fn((
      _contents: string,
      callback?: (error?: Error | null) => void
    ): boolean => {
      callback?.(null);
      return true;
    })
  };
  readonly kill = vi.fn((_signal?: NodeJS.Signals): boolean => {
    this.killed = true;
    queueMicrotask(() => {
      if (this.exitCode !== null) return;
      this.exitCode = 0;
      this.emit("exit", 0, null);
      this.emit("close", 0, null);
    });
    return true;
  });
}

function chatResult(input: string, content: string): ChatResult {
  const createdAt = new Date().toISOString();
  return {
    brain: { messages: [] } as unknown as ChatResult["brain"],
    humanMessage: {
      id: `human-${input}`,
      role: "human",
      content: input,
      createdAt
    },
    brainMessage: {
      id: `brain-${content}`,
      role: "brain",
      content,
      createdAt
    },
    trace: { id: `trace-${content}` } as unknown as ChatResult["trace"]
  };
}

describe("warm Steer runtime handoff", () => {
  it("keeps the same warm PID and preserves the first reply before queued Steer", async () => {
    const worker = new WarmWorker();
    const sendSignal = vi.fn();
    const engine = new EngineSupervisor({
      appPath: process.cwd(),
      sendSignal
    });
    (
      engine as unknown as { child: ChildProcessWithoutNullStreams }
    ).child = worker as unknown as ChildProcessWithoutNullStreams;
    const service = {
      chat: vi.fn(async (
        brainId: string,
        input: string,
        signal?: AbortSignal,
        _onStream?: unknown,
        turnId?: string
      ): Promise<ChatResult> => {
        const response = await engine.request<{ text: string; cacheEpoch: string }>(
          "chat",
          { brainId, input, cacheEpoch: "native-cortex-1" },
          ENGINE_REQUEST_NO_DEADLINE,
          signal,
          "foreground",
          {
            requestId: turnId ?? input,
            owner: "chat",
            label: "Chat response",
            brainId,
            turnId
          }
        );
        expect(response.cacheEpoch).toBe("native-cortex-1");
        return chatResult(input, response.text);
      })
    };
    const controller = new ChatActionController(
      service,
      { execute: vi.fn(), cancel: vi.fn(() => 0) },
      { start: vi.fn() }
    );
    const consume = (
      engine as unknown as { consumeLine(line: string): void }
    ).consumeLine.bind(engine);

    try {
      const original = controller.send(
        "native-ground-up",
        "Start broadly.",
        undefined,
        "turn-original"
      );
      const originalResult = original.catch((error: unknown) => error);
      await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());
      const warmPid = engine.pid;

      const steered = controller.send(
        "native-ground-up",
        "Focus on the exact evidence.",
        undefined,
        "turn-steered",
        {
          kind: "steer",
          replacesTurnId: "turn-original",
          source: "human",
          createdAt: "2026-09-19T12:02:00.000Z"
        }
      );
      await new Promise<void>((resolve) => setImmediate(resolve));
      expect(worker.stdin.write).toHaveBeenCalledOnce();
      expect(engine.pid).toBe(warmPid);
      expect(worker.kill).not.toHaveBeenCalled();
      expect(sendSignal).not.toHaveBeenCalled();

      const first = JSON.parse(
        String(worker.stdin.write.mock.calls[0]?.[0])
      ) as { id: string };
      consume(JSON.stringify({
        jsonrpc: "2.0",
        id: first.id,
        result: { text: "original committed", cacheEpoch: "native-cortex-1" }
      }));
      await expect(originalResult).resolves.toMatchObject({
        brainMessage: { content: "original committed" }
      });

      await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledTimes(2));
      expect(engine.pid).toBe(warmPid);
      expect(worker.kill).not.toHaveBeenCalled();
      expect(sendSignal).not.toHaveBeenCalled();
      const second = JSON.parse(
        String(worker.stdin.write.mock.calls[1]?.[0])
      ) as { id: string };
      consume(JSON.stringify({
        jsonrpc: "2.0",
        id: second.id,
        result: { text: "steer committed", cacheEpoch: "native-cortex-1" }
      }));

      await expect(steered).resolves.toMatchObject({
        brainMessage: { content: "steer committed" },
        turnMetadata: { replacesTurnId: "turn-original" }
      });
      expect(engine.pid).toBe(warmPid);
      expect(worker.kill).not.toHaveBeenCalled();
    } finally {
      await engine.stop();
    }
  });
});
