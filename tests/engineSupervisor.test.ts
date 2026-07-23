import { EventEmitter } from "node:events";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { describe, expect, it, vi } from "vitest";
import { EngineSupervisor } from "../src/main/engineSupervisor";

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
});
