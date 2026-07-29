import { EventEmitter } from "node:events";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { describe, expect, it, vi } from "vitest";
import {
  EngineSupervisor,
  packagedEnginePath
} from "../src/main/engineSupervisor";

class FakeWorker extends EventEmitter {
  readonly pid = 4242;
  killed = false;
  exitCode: number | null = null;
  readonly stdin = {
    writable: true,
    write: vi.fn(
      (
        _contents: string,
        callback?: (error?: Error | null) => void
      ): boolean => {
        callback?.(null);
        return true;
      }
    )
  };
  readonly kill = vi.fn((_signal?: NodeJS.Signals): boolean => {
    this.killed = true;
    queueMicrotask(() => {
      if (this.exitCode !== null) return;
      this.exitCode = 0;
      this.emit("exit", 0, null);
    });
    return true;
  });
}

function supervisorWith(worker: FakeWorker): EngineSupervisor {
  const supervisor = new EngineSupervisor({ appPath: process.cwd() });
  (
    supervisor as unknown as {
      child: ChildProcessWithoutNullStreams;
    }
  ).child = worker as unknown as ChildProcessWithoutNullStreams;
  return supervisor;
}

describe("EngineSupervisor interruption", () => {
  it("resolves the packaged worker executable for every desktop platform", () => {
    expect(packagedEnginePath("/resources", "win32")).toMatch(
      /engine-runtime[/\\]omni-engine\.exe$/
    );
    expect(packagedEnginePath("/resources", "darwin")).toMatch(
      /engine-runtime[/\\]omni-engine$/
    );
    expect(packagedEnginePath("/resources", "linux")).toMatch(
      /engine-runtime[/\\]omni-engine$/
    );
  });

  it("rejects an aborted request and terminates the serial worker", async () => {
    const worker = new FakeWorker();
    const supervisor = supervisorWith(worker);
    const controller = new AbortController();
    const request = supervisor.request("slow-method", {}, 60_000, controller.signal);
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());

    controller.abort();

    await expect(request).rejects.toThrow(/cancelled/i);
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(worker.kill).toHaveBeenCalled();
    await supervisor.stop();
  });

  it("terminates a worker whose request exceeded its deadline", async () => {
    const worker = new FakeWorker();
    const supervisor = supervisorWith(worker);

    await expect(supervisor.request("slow-method", {}, 20)).rejects.toThrow(
      /timed out/i
    );
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(worker.kill).toHaveBeenCalled();
    await supervisor.stop();
  });

  it("delivers only correlated, strictly ordered stream notifications", async () => {
    const worker = new FakeWorker();
    const supervisor = supervisorWith(worker);
    const received: number[] = [];
    const request = supervisor.requestStream<{ done: boolean }>(
      "chat",
      { brainId: "brain-stream" },
      (event) => received.push(event.sequence!),
      60_000,
      undefined,
      "turn-stream"
    );
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());
    const written = JSON.parse(
      String(worker.stdin.write.mock.calls[0]?.[0])
    ) as { id: string; params: { streamId: string } };
    expect(written.params.streamId).toBe("turn-stream");

    const consume = (
      supervisor as unknown as { consumeLine(line: string): void }
    ).consumeLine.bind(supervisor);
    const notification = (
      streamId: string,
      sequence: number
    ): void => consume(JSON.stringify({
      jsonrpc: "2.0",
      method: "event",
      params: {
        type: "chat-token",
        brainId: "brain-stream",
        streamId,
        sequence,
        data: { delta: String(sequence) }
      }
    }));
    notification("another-turn", 0);
    notification("turn-stream", 0);
    notification("turn-stream", 0);
    notification("turn-stream", 2);
    notification("turn-stream", 1);
    consume(JSON.stringify({
      jsonrpc: "2.0",
      id: written.id,
      result: { done: true }
    }));

    await expect(request).resolves.toEqual({ done: true });
    expect(received).toEqual([0, 2]);
    await supervisor.stop();
  });
});
