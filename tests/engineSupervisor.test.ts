import { EventEmitter } from "node:events";
import {
  spawn,
  type ChildProcessWithoutNullStreams
} from "node:child_process";
import { describe, expect, it, vi } from "vitest";
import {
  BackgroundRequestDeferredError,
  ENGINE_REQUEST_NO_DEADLINE,
  EngineRequestError,
  EngineSupervisor,
  type EngineActivityTransition,
  packagedEnginePath
} from "../src/main/engineSupervisor";

const interruptSignal = process.platform === "win32" ? "SIGBREAK" : "SIGUSR1";

class FakeWorker extends EventEmitter {
  readonly pid = 4242;
  killed = false;
  exitCode: number | null = null;
  signalCode: NodeJS.Signals | null = null;
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

function supervisedSupervisorWith(worker: FakeWorker): EngineSupervisor {
  const supervisor = new EngineSupervisor({ appPath: process.cwd() });
  (
    supervisor as unknown as {
      superviseChild(child: ChildProcessWithoutNullStreams): unknown;
    }
  ).superviseChild(worker as unknown as ChildProcessWithoutNullStreams);
  return supervisor;
}

function inspectionSupervisorWith(worker: FakeWorker): EngineSupervisor {
  const supervisor = new EngineSupervisor(
    { appPath: process.cwd() },
    "inspection"
  );
  (
    supervisor as unknown as {
      child: ChildProcessWithoutNullStreams;
    }
  ).child = worker as unknown as ChildProcessWithoutNullStreams;
  return supervisor;
}

describe("EngineSupervisor interruption", () => {
  it("routes substrate inspection to a separate cancellable worker without blocking chat", async () => {
    const neuralWorker = new FakeWorker();
    const inspectionWorker = new FakeWorker();
    const supervisor = supervisorWith(neuralWorker);
    const inspection = inspectionSupervisorWith(inspectionWorker);
    (
      supervisor as unknown as {
        inspectionWorker: EngineSupervisor;
      }
    ).inspectionWorker = inspection;
    const controller = new AbortController();

    const query = supervisor.request(
      "query_substrate",
      { brainId: "large-brain" },
      ENGINE_REQUEST_NO_DEADLINE,
      controller.signal
    );
    await vi.waitFor(() => expect(inspectionWorker.stdin.write).toHaveBeenCalledOnce());
    expect(neuralWorker.stdin.write).not.toHaveBeenCalled();

    const chat = supervisor.request<{ text: string }>(
      "chat",
      { brainId: "large-brain" },
      ENGINE_REQUEST_NO_DEADLINE
    );
    await vi.waitFor(() => expect(neuralWorker.stdin.write).toHaveBeenCalledOnce());
    const chatRequest = JSON.parse(
      String(neuralWorker.stdin.write.mock.calls[0]?.[0])
    ) as { id: string };
    (
      supervisor as unknown as { consumeLine(line: string): void }
    ).consumeLine(JSON.stringify({
      jsonrpc: "2.0",
      id: chatRequest.id,
      result: { text: "foreground stayed responsive" }
    }));
    await expect(chat).resolves.toEqual({ text: "foreground stayed responsive" });

    controller.abort();
    await expect(query).rejects.toThrow(/cancelled/i);
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(inspectionWorker.kill).toHaveBeenCalled();
    expect(neuralWorker.kill).not.toHaveBeenCalled();
    await supervisor.stop();
  });

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
    expect(worker.kill).toHaveBeenCalledWith("SIGKILL");
    await supervisor.stop();
  });

  it("force-stops an actual busy child on pre-commit cancellation", async () => {
    const child = spawn(process.execPath, ["-e", "while (true) {}"], {
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
      shell: false
    });
    await new Promise<void>((resolveSpawn, rejectSpawn) => {
      child.once("spawn", resolveSpawn);
      child.once("error", rejectSpawn);
    });
    const supervisor = new EngineSupervisor({
      appPath: process.cwd(),
      sendSignal: () => {
        throw new Error("fixture has no cooperative signal handler");
      }
    });
    (
      supervisor as unknown as { child: ChildProcessWithoutNullStreams }
    ).child = child;
    const closed = new Promise<void>((resolveClose) => {
      child.once("close", () => resolveClose());
    });
    const controller = new AbortController();
    const request = supervisor.request(
      "chat",
      { streamId: "pre-commit-cancel" },
      ENGINE_REQUEST_NO_DEADLINE,
      controller.signal
    );
    try {
      await new Promise<void>((resolveTurn) => setImmediate(resolveTurn));
      controller.abort();
      await expect(request).rejects.toThrow(/cancelled/i);
      await Promise.race([
        closed,
        new Promise<never>((_resolve, rejectClose) => {
          setTimeout(
            () => rejectClose(new Error("busy child survived cancellation")),
            3_000
          );
        })
      ]);
      expect(child.exitCode !== null || child.signalCode !== null).toBe(true);
    } finally {
      if (child.exitCode === null && child.signalCode === null) {
        child.kill("SIGKILL");
      }
      await supervisor.stop();
    }
  });

  it("keeps restart health concise while retaining raw diagnostics internally", async () => {
    const supervisor = new EngineSupervisor({ appPath: process.cwd() });
    const raw = [
      "Worker exited with code null and signal SIGKILL.",
      "Worker stderr:",
      "Traceback (most recent call last):",
      "  File /private/engine/worker.py, line 42",
      "RuntimeError: private diagnostic"
    ].join("\n");
    const internal = supervisor as unknown as {
      lastError: string;
      recentStderr: string[];
      start(): Promise<boolean>;
    };
    internal.lastError = raw;
    internal.recentStderr = [raw];
    vi.spyOn(internal, "start").mockResolvedValue(false);

    await expect(supervisor.health()).resolves.toEqual({
      ready: false,
      worker: "unavailable",
      protocolVersion: 1,
      detail: "Neural engine restarting…"
    });
    expect(internal.lastError).toBe(raw);
    expect(internal.recentStderr).toEqual([raw]);
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

  it("allows a bounded large-brain load past five minutes but stops it at 30", async () => {
    vi.useFakeTimers();
    const longWorker = new FakeWorker();
    const expiredWorker = new FakeWorker();
    const longSupervisor = supervisorWith(longWorker);
    const expiredSupervisor = supervisorWith(expiredWorker);
    const largeLoadTimeoutMs = 30 * 60_000;
    try {
      const longLoad = longSupervisor.request<{ loaded: boolean }>(
        "load",
        {},
        largeLoadTimeoutMs
      );
      await vi.advanceTimersByTimeAsync(0);
      await vi.advanceTimersByTimeAsync(6 * 60_000);
      expect(longWorker.kill).not.toHaveBeenCalled();
      const longWritten = JSON.parse(
        String(longWorker.stdin.write.mock.calls[0]?.[0])
      ) as { id: string };
      (
        longSupervisor as unknown as { consumeLine(line: string): void }
      ).consumeLine(JSON.stringify({
        jsonrpc: "2.0",
        id: longWritten.id,
        result: { loaded: true }
      }));
      await expect(longLoad).resolves.toEqual({ loaded: true });

      const expiredLoad = expiredSupervisor.request(
        "load",
        {},
        largeLoadTimeoutMs
      );
      const expiredResult = expiredLoad.catch((error: unknown) => error);
      await vi.advanceTimersByTimeAsync(0);
      await vi.advanceTimersByTimeAsync(largeLoadTimeoutMs + 1);
      await expect(expiredResult).resolves.toEqual(
        expect.objectContaining({
          message: expect.stringMatching(/timed out/i)
        })
      );
      expect(expiredWorker.kill).toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
      await Promise.all([
        (
          longSupervisor as unknown as { terminateChild(): Promise<void> }
        ).terminateChild(),
        (
          expiredSupervisor as unknown as { terminateChild(): Promise<void> }
        ).terminateChild()
      ]);
    }
  });

  it("lets an explicit long job exceed 24 hours without a wall-clock timeout", async () => {
    vi.useFakeTimers();
    const worker = new FakeWorker();
    const supervisor = supervisorWith(worker);
    try {
      const request = supervisor.request<{ complete: boolean }>(
        "ingest",
        {},
        ENGINE_REQUEST_NO_DEADLINE
      );
      await vi.advanceTimersByTimeAsync(0);
      expect(worker.stdin.write).toHaveBeenCalledOnce();

      await vi.advanceTimersByTimeAsync(48 * 60 * 60 * 1_000);
      expect(worker.kill).not.toHaveBeenCalled();

      const written = JSON.parse(String(worker.stdin.write.mock.calls[0]?.[0])) as {
        id: string;
      };
      (
        supervisor as unknown as { consumeLine(line: string): void }
      ).consumeLine(JSON.stringify({
        jsonrpc: "2.0",
        id: written.id,
        result: { complete: true }
      }));
      await expect(request).resolves.toEqual({ complete: true });
    } finally {
      vi.useRealTimers();
      await (
        supervisor as unknown as { terminateChild(): Promise<void> }
      ).terminateChild();
    }
  });

  it("does not let a background deadline restart a healthy worker", async () => {
    vi.useFakeTimers();
    const worker = new FakeWorker();
    const supervisor = supervisorWith(worker);
    try {
      const request = supervisor.request<{ ran: boolean }>(
        "idle_cycle",
        {},
        20,
        undefined,
        "background"
      );
      await vi.advanceTimersByTimeAsync(0);
      expect(worker.stdin.write).toHaveBeenCalledOnce();

      await vi.advanceTimersByTimeAsync(24 * 60 * 60 * 1_000);
      expect(worker.kill).not.toHaveBeenCalled();

      const written = JSON.parse(String(worker.stdin.write.mock.calls[0]?.[0])) as {
        id: string;
      };
      (
        supervisor as unknown as { consumeLine(line: string): void }
      ).consumeLine(JSON.stringify({
        jsonrpc: "2.0",
        id: written.id,
        result: { ran: false }
      }));
      await expect(request).resolves.toEqual({ ran: false });
    } finally {
      vi.useRealTimers();
      await (
        supervisor as unknown as { terminateChild(): Promise<void> }
      ).terminateChild();
    }
  });

  it("still cancels an explicit no-deadline request and stops its worker", async () => {
    const worker = new FakeWorker();
    const supervisor = supervisorWith(worker);
    const controller = new AbortController();
    const request = supervisor.request(
      "ingest",
      {},
      ENGINE_REQUEST_NO_DEADLINE,
      controller.signal
    );
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());

    controller.abort();

    await expect(request).rejects.toThrow(/cancelled/i);
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(worker.kill).toHaveBeenCalledWith("SIGKILL");
    await supervisor.stop();
  });

  it("rejects a no-deadline request on worker exit and restarts after close", async () => {
    const worker = new FakeWorker();
    const replacement = new FakeWorker();
    const supervisor = supervisedSupervisorWith(worker);
    const internal = supervisor as unknown as {
      pending: Map<string, unknown>;
      startCandidates(): Promise<boolean>;
      superviseChild(child: ChildProcessWithoutNullStreams): unknown;
      terminateChild(): Promise<void>;
    };
    const startCandidates = vi
      .spyOn(internal, "startCandidates")
      .mockImplementation(async () => {
        internal.superviseChild(
          replacement as unknown as ChildProcessWithoutNullStreams
        );
        return true;
      });

    const chat = supervisor
      .request(
        "chat",
        { brainId: "crashed-recall" },
        ENGINE_REQUEST_NO_DEADLINE
      )
      .catch((error: unknown) => error);
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());

    worker.exitCode = 17;
    worker.emit("exit", 17, null);

    await expect(chat).resolves.toEqual(
      expect.objectContaining({
        message: expect.stringMatching(/exited with code 17/i)
      })
    );
    expect(internal.pending.size).toBe(0);

    const recovered = supervisor.request<{ ready: boolean }>(
      "chat",
      { brainId: "crashed-recall" },
      ENGINE_REQUEST_NO_DEADLINE
    );
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(startCandidates).not.toHaveBeenCalled();
    expect(replacement.stdin.write).not.toHaveBeenCalled();

    worker.emit("close", 17, null);
    await vi.waitFor(() => expect(startCandidates).toHaveBeenCalledOnce());
    await vi.waitFor(() => expect(replacement.stdin.write).toHaveBeenCalledOnce());
    const written = JSON.parse(
      String(replacement.stdin.write.mock.calls[0]?.[0])
    ) as { id: string };
    (
      supervisor as unknown as { consumeLine(line: string): void }
    ).consumeLine(JSON.stringify({
      jsonrpc: "2.0",
      id: written.id,
      result: { ready: true }
    }));

    await expect(recovered).resolves.toEqual({ ready: true });
    await internal.terminateChild();
  });

  it("keeps streamed chat alive past 30 minutes and still honors exact cancellation", async () => {
    vi.useFakeTimers();
    const worker = new FakeWorker();
    const cancelledWorker = new FakeWorker();
    const supervisor = supervisorWith(worker);
    const cancelledSupervisor = supervisorWith(cancelledWorker);
    try {
      const sequences: number[] = [];
      const chat = supervisor.requestStream<{ text: string }>(
        "chat",
        { brainId: "large-online-brain" },
        (event) => sequences.push(event.sequence!),
        ENGINE_REQUEST_NO_DEADLINE,
        undefined,
        "long-chat"
      );
      await vi.advanceTimersByTimeAsync(0);
      await vi.advanceTimersByTimeAsync(31 * 60_000);
      expect(worker.kill).not.toHaveBeenCalled();
      const written = JSON.parse(String(worker.stdin.write.mock.calls[0]?.[0])) as {
        id: string;
      };
      const consume = (
        supervisor as unknown as { consumeLine(line: string): void }
      ).consumeLine.bind(supervisor);
      consume(JSON.stringify({
        jsonrpc: "2.0",
        method: "event",
        params: {
          type: "chat-token",
          streamId: "long-chat",
          sequence: 0,
          data: { delta: "hi" }
        }
      }));
      consume(JSON.stringify({
        jsonrpc: "2.0",
        id: written.id,
        result: { text: "hi" }
      }));
      await expect(chat).resolves.toEqual({ text: "hi" });
      expect(sequences).toEqual([0]);

      const controller = new AbortController();
      const cancelledChat = cancelledSupervisor.requestStream(
        "chat",
        { brainId: "large-online-brain" },
        () => undefined,
        ENGINE_REQUEST_NO_DEADLINE,
        controller.signal,
        "cancelled-chat"
      );
      const cancelledResult = cancelledChat.catch((error: unknown) => error);
      await vi.advanceTimersByTimeAsync(0);
      controller.abort();
      await expect(cancelledResult).resolves.toEqual(
        expect.objectContaining({
          message: expect.stringMatching(/cancelled/i)
        })
      );
      expect(cancelledWorker.kill).toHaveBeenCalledWith("SIGKILL");
    } finally {
      vi.useRealTimers();
      await Promise.all([
        (
          supervisor as unknown as { terminateChild(): Promise<void> }
        ).terminateChild(),
        (
          cancelledSupervisor as unknown as { terminateChild(): Promise<void> }
        ).terminateChild()
      ]);
    }
  });

  it("starts queued deadlines only when the serial worker can execute them", async () => {
    const worker = new FakeWorker();
    const supervisor = supervisorWith(worker);
    const long = supervisor.request<{ name: string }>("ingest", {}, 60_000);
    const interactive = supervisor.request<{ name: string }>(
      "workspace",
      {},
      20
    );
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledTimes(1));

    // This is longer than the interactive request's own deadline, but that
    // request has not entered the Python execution loop yet.
    await new Promise<void>((resolve) => setTimeout(resolve, 35));
    expect(worker.stdin.write).toHaveBeenCalledTimes(1);
    expect(worker.kill).not.toHaveBeenCalled();

    const consume = (
      supervisor as unknown as { consumeLine(line: string): void }
    ).consumeLine.bind(supervisor);
    const first = JSON.parse(String(worker.stdin.write.mock.calls[0]?.[0])) as {
      id: string;
    };
    consume(JSON.stringify({
      jsonrpc: "2.0",
      id: first.id,
      result: { name: "ingest" }
    }));
    await expect(long).resolves.toEqual({ name: "ingest" });
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledTimes(2));

    const second = JSON.parse(String(worker.stdin.write.mock.calls[1]?.[0])) as {
      id: string;
    };
    consume(JSON.stringify({
      jsonrpc: "2.0",
      id: second.id,
      result: { name: "workspace" }
    }));
    await expect(interactive).resolves.toEqual({ name: "workspace" });
    expect(worker.kill).not.toHaveBeenCalled();
    await supervisor.stop();
  });

  it("does not queue optional background cognition behind foreground work", async () => {
    const worker = new FakeWorker();
    const supervisor = supervisorWith(worker);
    const foreground = supervisor.request<{ done: boolean }>(
      "create",
      {},
      60_000
    );
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());

    await expect(
      supervisor.request("idle_cycle", {}, 60_000, undefined, "background")
    ).rejects.toBeInstanceOf(BackgroundRequestDeferredError);
    expect(worker.stdin.write).toHaveBeenCalledOnce();

    const written = JSON.parse(String(worker.stdin.write.mock.calls[0]?.[0])) as {
      id: string;
    };
    (
      supervisor as unknown as { consumeLine(line: string): void }
    ).consumeLine(JSON.stringify({
      jsonrpc: "2.0",
      id: written.id,
      result: { done: true }
    }));
    await expect(foreground).resolves.toEqual({ done: true });
    await supervisor.stop();
  });

  it("preempts an active background cold load for a foreground Build", async () => {
    const worker = new FakeWorker();
    const replacement = new FakeWorker();
    const supervisor = supervisorWith(worker);
    vi.spyOn(
      supervisor as unknown as { startCandidates(): Promise<boolean> },
      "startCandidates"
    ).mockImplementation(async () => {
      (
        supervisor as unknown as {
          child: ChildProcessWithoutNullStreams;
        }
      ).child = replacement as unknown as ChildProcessWithoutNullStreams;
      return true;
    });

    const background = supervisor
      .request("idle_cycle", {}, 60_000, undefined, "background")
      .catch((error: unknown) => error);
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());

    const build = supervisor.request<{ built: boolean }>("create", {}, 60_000);
    await vi.waitFor(() => expect(worker.kill).toHaveBeenCalledOnce());
    await vi.waitFor(() => expect(replacement.stdin.write).toHaveBeenCalledOnce());
    expect(await background).toBeInstanceOf(BackgroundRequestDeferredError);

    const written = JSON.parse(
      String(replacement.stdin.write.mock.calls[0]?.[0])
    ) as { id: string };
    (
      supervisor as unknown as { consumeLine(line: string): void }
    ).consumeLine(JSON.stringify({
      jsonrpc: "2.0",
      id: written.id,
      result: { built: true }
    }));
    await expect(build).resolves.toEqual({ built: true });
    await supervisor.stop();
  });

  it("can claim foreground before a same-brain repository lock is available", async () => {
    const worker = new FakeWorker();
    const replacement = new FakeWorker();
    const supervisor = supervisorWith(worker);
    vi.spyOn(
      supervisor as unknown as { startCandidates(): Promise<boolean> },
      "startCandidates"
    ).mockImplementation(async () => {
      (
        supervisor as unknown as { child: ChildProcessWithoutNullStreams }
      ).child = replacement as unknown as ChildProcessWithoutNullStreams;
      return true;
    });
    const background = supervisor
      .request("idle_cycle", {}, 20, undefined, "background")
      .catch((error: unknown) => error);
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());

    await supervisor.claimForeground();

    expect(worker.kill).toHaveBeenCalledOnce();
    expect(await background).toBeInstanceOf(BackgroundRequestDeferredError);
    expect(replacement.stdin.write).not.toHaveBeenCalled();
    await (
      supervisor as unknown as { terminateChild(): Promise<void> }
    ).terminateChild();
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

    const error = await request.catch((value: unknown) => value);
    expect(error).toBeInstanceOf(EngineRequestError);
    expect(error).toMatchObject({
      code: -32000,
      data: {
        traceback: "Traceback (most recent call last):\nexact sentinel"
      }
    });
    expect((error as Error).message).toMatch(/exact sentinel/);
    expect((error as Error).message).not.toMatch(/stale warning/);
    await supervisor.stop();
  });

  it("waits for an exited worker to close before starting a replacement", async () => {
    const worker = new FakeWorker();
    const supervisor = supervisorWith(worker);
    const diagnostics: string[] = [];
    supervisor.on("diagnostic", (value) => diagnostics.push(String(value)));
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
    const oldRequest = supervisor.request(
      "old-request",
      {},
      ENGINE_REQUEST_NO_DEADLINE
    );
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
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toContain("complete old-worker diagnostic");
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

  it("escalates an in-flight graceful stop for an exact request cancel", async () => {
    const worker = new FakeWorker();
    const supervisor = supervisorWith(worker);
    const internal = supervisor as unknown as {
      terminateChild(force?: boolean): Promise<void>;
    };

    const graceful = internal.terminateChild();
    const forced = internal.terminateChild(true);

    expect(forced).toBe(graceful);
    expect(worker.kill.mock.calls).toEqual([[], ["SIGKILL"]]);
    await forced;
  });

  it("labels chat queued behind evolution and cancels only that queued turn", async () => {
    const worker = new FakeWorker();
    const supervisor = supervisorWith(worker);
    const evolutionTransitions: EngineActivityTransition[] = [];
    const chatTransitions: EngineActivityTransition[] = [];
    const evolution = supervisor.request<{ ready: boolean }>(
      "evolution.propose",
      { brainId: "brain-owned", jobId: "evolution-run" },
      ENGINE_REQUEST_NO_DEADLINE,
      undefined,
      "foreground",
      {
        requestId: "evolution-run",
        owner: "evolution",
        label: "Neural evolution",
        brainId: "brain-owned",
        jobId: "evolution-run",
        onTransition: (event) => evolutionTransitions.push(event)
      }
    );
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());

    const chat = supervisor.request(
      "load",
      { brainId: "brain-owned" },
      ENGINE_REQUEST_NO_DEADLINE,
      undefined,
      "foreground",
      {
        requestId: "chat-turn",
        owner: "chat",
        label: "Chat response",
        brainId: "brain-owned",
        turnId: "chat-turn",
        onTransition: (event) => chatTransitions.push(event)
      }
    );
    const chatResult = chat.catch((error: unknown) => error);
    await vi.waitFor(() => expect(chatTransitions).toContainEqual(
      expect.objectContaining({
        requestId: "chat-turn",
        state: "queued",
        queuePosition: 1,
        queuedBehind: expect.objectContaining({
          requestId: "evolution-run",
          owner: "evolution",
          label: "Neural evolution",
          jobId: "evolution-run"
        })
      })
    ));

    await expect(supervisor.cancelRequest("chat-turn")).resolves.toEqual({
      requestId: "chat-turn",
      acknowledged: true,
      phase: "queued",
      workerTerminationAcknowledged: false
    });
    await expect(chatResult).resolves.toEqual(
      expect.objectContaining({ message: expect.stringMatching(/cancelled/i) })
    );
    expect(worker.kill).not.toHaveBeenCalled();
    expect(worker.stdin.write).toHaveBeenCalledOnce();

    const request = JSON.parse(String(worker.stdin.write.mock.calls[0]?.[0])) as {
      id: string;
    };
    (
      supervisor as unknown as { consumeLine(line: string): void }
    ).consumeLine(JSON.stringify({
      jsonrpc: "2.0",
      id: request.id,
      result: { ready: true }
    }));
    await expect(evolution).resolves.toEqual({ ready: true });
    expect(evolutionTransitions).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ state: "running", requestId: "evolution-run" }),
        expect.objectContaining({ state: "complete", requestId: "evolution-run" })
      ])
    );
    await supervisor.stop();
  });

  it("acknowledges active cancellation only after the busy worker PID exits", async () => {
    const child = spawn(process.execPath, ["-e", "while (true) {}"], {
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
      shell: false
    });
    await new Promise<void>((resolveSpawn, rejectSpawn) => {
      child.once("spawn", resolveSpawn);
      child.once("error", rejectSpawn);
    });
    const pid = child.pid!;
    const supervisor = new EngineSupervisor({ appPath: process.cwd() });
    (
      supervisor as unknown as { child: ChildProcessWithoutNullStreams }
    ).child = child;
    const request = supervisor.request(
      "evolution.propose",
      { brainId: "brain-cpu", jobId: "evolution-cpu" },
      ENGINE_REQUEST_NO_DEADLINE,
      undefined,
      "foreground",
      {
        requestId: "evolution-cpu",
        owner: "evolution",
        label: "Neural evolution",
        brainId: "brain-cpu",
        jobId: "evolution-cpu"
      }
    );
    const result = request.catch((error: unknown) => error);
    try {
      await new Promise<void>((resolveTurn) => setImmediate(resolveTurn));
      await expect(supervisor.cancelRequest("evolution-cpu")).resolves.toEqual({
        requestId: "evolution-cpu",
        acknowledged: true,
        phase: "running",
        workerTerminationAcknowledged: true
      });
      await expect(result).resolves.toEqual(
        expect.objectContaining({ message: expect.stringMatching(/cancelled/i) })
      );
      expect(() => process.kill(pid, 0)).toThrow();
    } finally {
      if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
      await supervisor.stop();
    }
  });

  it("does not acknowledge cancellation when the worker never confirms exit", async () => {
    vi.useFakeTimers();
    try {
      const worker = new FakeWorker();
      worker.kill.mockImplementation(() => true);
      const supervisor = supervisorWith(worker);
      const transitions: EngineActivityTransition[] = [];
      const request = supervisor.request(
        "evolution.propose",
        { brainId: "brain-unacked", jobId: "unacked-run" },
        ENGINE_REQUEST_NO_DEADLINE,
        undefined,
        "foreground",
        {
          requestId: "unacked-run",
          owner: "evolution",
          label: "Neural evolution",
          onTransition: (event) => transitions.push(event)
        }
      );
      const result = request.catch((error: unknown) => error);
      await vi.advanceTimersByTimeAsync(0);
      const cancellation = supervisor.cancelRequest("unacked-run");
      await vi.advanceTimersByTimeAsync(1_300);

      await expect(cancellation).resolves.toEqual({
        requestId: "unacked-run",
        acknowledged: false,
        phase: "running",
        workerTerminationAcknowledged: false
      });
      await expect(result).resolves.toEqual(
        expect.objectContaining({ message: expect.stringMatching(/not acknowledged/i) })
      );
      expect(transitions.at(-1)).toMatchObject({
        state: "failed",
        cancellationPhase: "running",
        workerTerminationAcknowledged: false
      });
      await expect(supervisor.start()).rejects.toThrow(/will not be started/i);
    } finally {
      vi.useRealTimers();
    }
  });

  it("force-stops a stalled cold load after a short grace without cancelling queued work", async () => {
    vi.useFakeTimers();
    try {
      const coldWorker = new FakeWorker();
      const replacement = new FakeWorker();
      const sendSignal = vi.fn();
      const supervisor = new EngineSupervisor({ appPath: process.cwd(), sendSignal });
      const internal = supervisor as unknown as {
        child: ChildProcessWithoutNullStreams;
        startCandidates(): Promise<boolean>;
        consumeLine(line: string): void;
      };
      internal.child = coldWorker as unknown as ChildProcessWithoutNullStreams;
      vi.spyOn(internal, "startCandidates").mockImplementation(async () => {
        internal.child = replacement as unknown as ChildProcessWithoutNullStreams;
        return true;
      });
      const transitions: EngineActivityTransition[] = [];
      const load = supervisor.request(
        "load",
        { brainId: "brain-cold" },
        ENGINE_REQUEST_NO_DEADLINE,
        undefined,
        "foreground",
        {
          requestId: "cold-load",
          owner: "chat",
          label: "Chat response",
          brainId: "brain-cold",
          turnId: "cold-load",
          onTransition: (event) => transitions.push(event)
        }
      );
      const loadResult = load.catch((error: unknown) => error);
      await vi.advanceTimersByTimeAsync(0);
      expect(coldWorker.stdin.write).toHaveBeenCalledOnce();

      const queued = supervisor.request<{ ready: boolean }>(
        "health",
        {},
        ENGINE_REQUEST_NO_DEADLINE,
        undefined,
        "foreground",
        { requestId: "unrelated", owner: "system", label: "Health check" }
      );
      await vi.advanceTimersByTimeAsync(0);
      expect(replacement.stdin.write).not.toHaveBeenCalled();

      const cancellation = supervisor.cancelRequest("cold-load");
      await vi.advanceTimersByTimeAsync(4_999);
      expect(sendSignal).toHaveBeenCalledWith(4242, interruptSignal);
      expect(coldWorker.kill).not.toHaveBeenCalled();
      await vi.advanceTimersByTimeAsync(1);
      await expect(cancellation).resolves.toEqual({
        requestId: "cold-load",
        acknowledged: true,
        phase: "running",
        workerTerminationAcknowledged: true
      });
      await expect(loadResult).resolves.toEqual(
        expect.objectContaining({ message: expect.stringMatching(/cancelled/i) })
      );
      expect(coldWorker.kill).toHaveBeenCalledWith("SIGKILL");
      expect(transitions.at(-1)).toMatchObject({
        state: "cancelled",
        workerTerminationAcknowledged: true
      });

      await vi.advanceTimersByTimeAsync(0);
      expect(replacement.stdin.write).toHaveBeenCalledOnce();
      const request = JSON.parse(String(replacement.stdin.write.mock.calls[0]?.[0])) as {
        id: string;
      };
      internal.consumeLine(JSON.stringify({
        jsonrpc: "2.0",
        id: request.id,
        result: { ready: true }
      }));
      await expect(queued).resolves.toEqual({ ready: true });
      expect(replacement.kill).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it.each([false, true])(
    "force-stops a chat with no tokens or commit phase (action event: %s)",
    async (emitAction) => {
      vi.useFakeTimers();
      try {
        const worker = new FakeWorker();
        const sendSignal = vi.fn();
        const supervisor = new EngineSupervisor({ appPath: process.cwd(), sendSignal });
        const internal = supervisor as unknown as {
          child: ChildProcessWithoutNullStreams;
          consumeLine(line: string): void;
        };
        internal.child = worker as unknown as ChildProcessWithoutNullStreams;
        const chat = supervisor.requestStream(
          "chat",
          { brainId: "brain-cold-foundation" },
          vi.fn(),
          ENGINE_REQUEST_NO_DEADLINE,
          undefined,
          "turn-before-output",
          "foreground",
          {
            requestId: "turn-before-output",
            owner: "chat",
            label: "Chat response",
            brainId: "brain-cold-foundation",
            turnId: "turn-before-output"
          }
        );
        const chatResult = chat.catch((error: unknown) => error);
        await vi.advanceTimersByTimeAsync(0);
        expect(worker.stdin.write).toHaveBeenCalledOnce();

        const cancellation = supervisor.cancelRequest("turn-before-output");
        if (emitAction) {
          // Action proposals precede speech and do not commit fast experience.
          internal.consumeLine(JSON.stringify({
            jsonrpc: "2.0",
            method: "event",
            params: {
              type: "chat-action",
              brainId: "brain-cold-foundation",
              streamId: "turn-before-output",
              sequence: 0,
              data: { action: { kind: "ponder" } }
            }
          }));
        }
        await vi.advanceTimersByTimeAsync(4_999);
        expect(sendSignal).toHaveBeenCalledWith(4242, interruptSignal);
        expect(worker.kill).not.toHaveBeenCalled();
        await vi.advanceTimersByTimeAsync(1);
        await expect(cancellation).resolves.toEqual({
          requestId: "turn-before-output",
          acknowledged: true,
          phase: "running",
          workerTerminationAcknowledged: true
        });
        await expect(chatResult).resolves.toEqual(
          expect.objectContaining({ message: expect.stringMatching(/cancelled/i) })
        );
        expect(worker.kill).toHaveBeenCalledWith("SIGKILL");
      } finally {
        vi.useRealTimers();
      }
    }
  );

  it.each(["chat-token", "chat-phase"])(
    "keeps cooperative cancellation after correlated %s activity arrives",
    async (eventType) => {
      vi.useFakeTimers();
      try {
        const worker = new FakeWorker();
        const sendSignal = vi.fn();
        const supervisor = new EngineSupervisor({ appPath: process.cwd(), sendSignal });
        const internal = supervisor as unknown as {
          child: ChildProcessWithoutNullStreams;
          consumeLine(line: string): void;
        };
        internal.child = worker as unknown as ChildProcessWithoutNullStreams;
        const chat = supervisor.requestStream(
          "chat",
          { brainId: "brain-visible" },
          vi.fn(),
          ENGINE_REQUEST_NO_DEADLINE,
          undefined,
          "turn-visible",
          "foreground",
          {
            requestId: "turn-visible",
            owner: "chat",
            label: "Chat response",
            brainId: "brain-visible",
            turnId: "turn-visible"
          }
        );
        const chatResult = chat.catch((error: unknown) => error);
        await vi.advanceTimersByTimeAsync(0);
        const request = JSON.parse(String(worker.stdin.write.mock.calls[0]?.[0])) as {
          id: string;
        };

        const cancellation = supervisor.cancelRequest("turn-visible");
        internal.consumeLine(JSON.stringify({
          jsonrpc: "2.0",
          method: "event",
          params: {
            type: eventType,
            brainId: "brain-visible",
            streamId: "turn-visible",
            sequence: 0,
            data: eventType === "chat-token"
              ? { delta: "A" }
              : {
                  phase: "reply-complete-learning",
                  replyComplete: true,
                  turnCommitted: false,
                  learning: true,
                  saving: true
                }
          }
        }));
        await vi.advanceTimersByTimeAsync(5_000);
        expect(sendSignal).toHaveBeenCalledWith(4242, interruptSignal);
        expect(worker.kill).not.toHaveBeenCalled();

        internal.consumeLine(JSON.stringify({
          jsonrpc: "2.0",
          id: request.id,
          error: {
            code: -32800,
            message: "chat generation was cancelled",
            data: { cancelled: true, safeBoundary: true }
          }
        }));
        await expect(chatResult).resolves.toBeInstanceOf(EngineRequestError);
        await expect(cancellation).resolves.toEqual({
          requestId: "turn-visible",
          acknowledged: true,
          phase: "running",
          workerTerminationAcknowledged: false
        });
        expect(worker.kill).not.toHaveBeenCalled();
      } finally {
        vi.useRealTimers();
      }
    }
  );

  it("cooperatively cancels chat at a safe boundary and reuses the same warm worker", async () => {
    const worker = new FakeWorker();
    const sendSignal = vi.fn();
    const supervisor = new EngineSupervisor({
      appPath: process.cwd(),
      sendSignal
    });
    (
      supervisor as unknown as { child: ChildProcessWithoutNullStreams }
    ).child = worker as unknown as ChildProcessWithoutNullStreams;
    const chat = supervisor.request(
      "chat",
      { brainId: "warm-brain", streamId: "warm-turn" },
      ENGINE_REQUEST_NO_DEADLINE,
      undefined,
      "foreground",
      {
        requestId: "warm-turn",
        owner: "chat",
        label: "Chat response",
        brainId: "warm-brain",
        turnId: "warm-turn"
      }
    );
    const chatResult = chat.catch((error: unknown) => error);
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());
    const written = JSON.parse(
      String(worker.stdin.write.mock.calls[0]?.[0])
    ) as { id: string };

    const cancellation = supervisor.cancelRequest("warm-turn");
    await vi.waitFor(() => expect(sendSignal).toHaveBeenCalledWith(4242, interruptSignal));
    expect(worker.kill).not.toHaveBeenCalled();
    (
      supervisor as unknown as { consumeLine(line: string): void }
    ).consumeLine(JSON.stringify({
      jsonrpc: "2.0",
      id: written.id,
      error: {
        code: -32800,
        message: "chat generation was cancelled",
        data: { cancelled: true, safeBoundary: true }
      }
    }));

    await expect(chatResult).resolves.toBeInstanceOf(EngineRequestError);
    await expect(cancellation).resolves.toEqual({
      requestId: "warm-turn",
      acknowledged: true,
      phase: "running",
      workerTerminationAcknowledged: false
    });
    expect(supervisor.pid).toBe(4242);
    expect(worker.kill).not.toHaveBeenCalled();

    const health = supervisor.request<{ ready: boolean }>("health", {}, 5_000);
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledTimes(2));
    const next = JSON.parse(
      String(worker.stdin.write.mock.calls[1]?.[0])
    ) as { id: string };
    (
      supervisor as unknown as { consumeLine(line: string): void }
    ).consumeLine(JSON.stringify({
      jsonrpc: "2.0",
      id: next.id,
      result: { ready: true }
    }));
    await expect(health).resolves.toEqual({ ready: true });
    expect(supervisor.pid).toBe(4242);
    await supervisor.stop();
  });

  it("preempts post-turn consolidation cooperatively without replacing the warm worker", async () => {
    const worker = new FakeWorker();
    const sendSignal = vi.fn();
    const supervisor = new EngineSupervisor({
      appPath: process.cwd(),
      sendSignal
    });
    (
      supervisor as unknown as { child: ChildProcessWithoutNullStreams }
    ).child = worker as unknown as ChildProcessWithoutNullStreams;
    const replay = supervisor.request(
      "consolidate_chat_learning",
      { brainId: "warm-brain", jobId: "slow-turn" },
      ENGINE_REQUEST_NO_DEADLINE,
      undefined,
      "background"
    );
    const replayResult = replay.catch((error: unknown) => error);
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());
    const replayRequest = JSON.parse(
      String(worker.stdin.write.mock.calls[0]?.[0])
    ) as { id: string };

    const foreground = supervisor.request<{ loaded: boolean }>(
      "load",
      { brainId: "warm-brain" },
      60_000
    );
    await vi.waitFor(() => expect(sendSignal).toHaveBeenCalledWith(4242, interruptSignal));
    expect(worker.kill).not.toHaveBeenCalled();
    (
      supervisor as unknown as { consumeLine(line: string): void }
    ).consumeLine(JSON.stringify({
      jsonrpc: "2.0",
      id: replayRequest.id,
      error: {
        code: -32800,
        message: "background chat learning was cancelled",
        data: { cancelled: true, safeBoundary: true }
      }
    }));
    await expect(replayResult).resolves.toBeInstanceOf(
      BackgroundRequestDeferredError
    );
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledTimes(2));
    const loadRequest = JSON.parse(
      String(worker.stdin.write.mock.calls[1]?.[0])
    ) as { id: string };
    (
      supervisor as unknown as { consumeLine(line: string): void }
    ).consumeLine(JSON.stringify({
      jsonrpc: "2.0",
      id: loadRequest.id,
      result: { loaded: true }
    }));
    await expect(foreground).resolves.toEqual({ loaded: true });
    expect(supervisor.pid).toBe(4242);
    expect(worker.kill).not.toHaveBeenCalled();
    await supervisor.stop();
  });

  it("pauses only the selected brain's active background replay without waiting for its RPC", async () => {
    const worker = new FakeWorker();
    const sendSignal = vi.fn();
    const supervisor = new EngineSupervisor({ appPath: process.cwd(), sendSignal });
    (
      supervisor as unknown as { child: ChildProcessWithoutNullStreams }
    ).child = worker as unknown as ChildProcessWithoutNullStreams;
    const replay = supervisor.request(
      "consolidate_chat_learning",
      { brainId: "selected-brain" },
      ENGINE_REQUEST_NO_DEADLINE,
      undefined,
      "background"
    ).catch((error: unknown) => error);
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());
    expect(supervisor.cancelBackgroundRequest("other-brain", "consolidate_chat_learning"))
      .toBe(false);
    expect(supervisor.cancelBackgroundRequest("selected-brain", "consolidate_chat_learning"))
      .toBe(true);
    expect(sendSignal).toHaveBeenCalledWith(4242, interruptSignal);
    expect(worker.kill).not.toHaveBeenCalled();
    const request = JSON.parse(String(worker.stdin.write.mock.calls[0]?.[0])) as { id: string };
    (
      supervisor as unknown as { consumeLine(line: string): void }
    ).consumeLine(JSON.stringify({
      jsonrpc: "2.0",
      id: request.id,
      error: { code: -32800, message: "background chat learning was cancelled" }
    }));
    await expect(replay).resolves.toBeInstanceOf(BackgroundRequestDeferredError);
    expect(worker.kill).not.toHaveBeenCalled();
    await supervisor.stop();
  });

  it("restarts a clean worker for an unrelated request queued behind a cancelled owner", async () => {
    const worker = new FakeWorker();
    const replacement = new FakeWorker();
    const supervisor = supervisorWith(worker);
    vi.spyOn(
      supervisor as unknown as { startCandidates(): Promise<boolean> },
      "startCandidates"
    ).mockImplementation(async () => {
      (
        supervisor as unknown as { child: ChildProcessWithoutNullStreams }
      ).child = replacement as unknown as ChildProcessWithoutNullStreams;
      return true;
    });
    const evolution = supervisor.request(
      "evolution.propose",
      { brainId: "brain-restart", jobId: "restart-evolution" },
      ENGINE_REQUEST_NO_DEADLINE,
      undefined,
      "foreground",
      {
        requestId: "restart-evolution",
        owner: "evolution",
        label: "Neural evolution"
      }
    );
    const evolutionResult = evolution.catch((error: unknown) => error);
    await vi.waitFor(() => expect(worker.stdin.write).toHaveBeenCalledOnce());
    const training = supervisor.request<{ trained: boolean }>(
      "train",
      { brainId: "brain-restart", jobId: "unrelated-training" },
      ENGINE_REQUEST_NO_DEADLINE
    );

    await expect(supervisor.cancelRequest("restart-evolution")).resolves.toMatchObject({
      acknowledged: true,
      phase: "running",
      workerTerminationAcknowledged: true
    });
    await expect(evolutionResult).resolves.toEqual(
      expect.objectContaining({ message: expect.stringMatching(/cancelled/i) })
    );
    await vi.waitFor(() => expect(replacement.stdin.write).toHaveBeenCalledOnce());
    const replacementRequest = JSON.parse(
      String(replacement.stdin.write.mock.calls[0]?.[0])
    ) as { id: string };
    (
      supervisor as unknown as { consumeLine(line: string): void }
    ).consumeLine(JSON.stringify({
      jsonrpc: "2.0",
      id: replacementRequest.id,
      result: { trained: true }
    }));

    await expect(training).resolves.toEqual({ trained: true });
    expect(replacement.kill).not.toHaveBeenCalled();
    await supervisor.stop();
  });
});
