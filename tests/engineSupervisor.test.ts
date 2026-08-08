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
  readonly stdout = Object.assign(new EventEmitter(), { setEncoding: vi.fn() });
  readonly stderr = Object.assign(new EventEmitter(), { setEncoding: vi.fn() });
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
      this.emit("close", 0, null);
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

  it("uses only request-correlated traceback data for an RPC failure", async () => {
    const worker = new FakeWorker();
    const supervisor = supervisorWith(worker);
    (
      supervisor as unknown as { recentStderr: string[] }
    ).recentStderr = ["stale warning from an earlier request"];
    const request = supervisor.request("chat", {}, 60_000);
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());
    const written = JSON.parse(String(worker.stdin.write.mock.calls[0]?.[0])) as {
      id: string;
    };
    (
      supervisor as unknown as { consumeLine(line: string): void }
    ).consumeLine(JSON.stringify({
      jsonrpc: "2.0",
      id: written.id,
      error: {
        code: -32000,
        message: "RuntimeError: exact failure",
        data: { traceback: "Traceback (most recent call last):\nexact sentinel" }
      }
    }));

    await expect(request).rejects.toThrow(/exact sentinel/);
    await expect(request).rejects.not.toThrow(/stale warning/);
    await supervisor.stop();
  });

  it("waits for an exited worker to close before starting a replacement", async () => {
    const worker = new FakeWorker();
    const supervisor = supervisorWith(worker);
    let resolveClosed!: () => void;
    const lifecycle = {
      child: worker as unknown as ChildProcessWithoutNullStreams,
      closed: new Promise<void>((resolve) => {
        resolveClosed = resolve;
      }),
      resolveClosed,
      stdoutBuffer: "",
      stdoutListener: vi.fn(),
      stderrListener: vi.fn(),
      stderr: ["complete old-worker diagnostic"],
      settled: false
    };
    (
      supervisor as unknown as { lifecycle: typeof lifecycle }
    ).lifecycle = lifecycle;
    const startCandidates = vi
      .spyOn(
        supervisor as unknown as { startCandidates(): Promise<boolean> },
        "startCandidates"
      )
      .mockResolvedValue(false);
    const oldRequest = supervisor.request("old-request", {}, 60_000);
    const oldRejection = oldRequest.catch((error: unknown) => error);
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());

    worker.exitCode = 9;
    worker.emit("exit", 9, null);
    const replacement = supervisor.start();
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(startCandidates).not.toHaveBeenCalled();

    (
      supervisor as unknown as {
        handleClose(
          state: typeof lifecycle,
          code: number | null,
          signal: NodeJS.Signals | null
        ): void;
      }
    ).handleClose(lifecycle, 9, null);

    await expect(oldRejection).resolves.toEqual(
      expect.objectContaining({
        message: expect.stringMatching(/complete old-worker diagnostic/)
      })
    );
    await expect(replacement).resolves.toBe(false);
    expect(startCandidates).toHaveBeenCalledOnce();
  });

  it("forces a missing close lifecycle to settle before replacement", async () => {
    vi.useFakeTimers();
    try {
      const worker = new FakeWorker();
      worker.kill.mockImplementation(() => {
        worker.killed = true;
        queueMicrotask(() => {
          worker.exitCode = 137;
          worker.emit("exit", 137, "SIGKILL");
        });
        return true;
      });
      const supervisor = supervisorWith(worker);
      let resolveClosed!: () => void;
      const lifecycle = {
        child: worker as unknown as ChildProcessWithoutNullStreams,
        closed: new Promise<void>((resolve) => {
          resolveClosed = resolve;
        }),
        resolveClosed,
        stdoutBuffer: "old partial JSON",
        stdoutListener: vi.fn(),
        stderrListener: vi.fn(),
        stderr: [],
        settled: false
      };
      (
        supervisor as unknown as { lifecycle: typeof lifecycle }
      ).lifecycle = lifecycle;
      const startCandidates = vi
        .spyOn(
          supervisor as unknown as { startCandidates(): Promise<boolean> },
          "startCandidates"
        )
        .mockResolvedValue(false);

      const termination = (
        supervisor as unknown as { terminateChild(): Promise<void> }
      ).terminateChild();
      const replacement = supervisor.start();
      await vi.advanceTimersByTimeAsync(3_100);

      await expect(termination).resolves.toBeUndefined();
      await expect(replacement).resolves.toBe(false);
      expect(lifecycle.settled).toBe(true);
      expect(startCandidates).toHaveBeenCalledOnce();
    } finally {
      vi.useRealTimers();
    }
  });
});
