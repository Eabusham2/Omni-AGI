import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService, type RuntimeJobManager } from "../src/main/brainService";
import {
  BrainWriteCoordinator,
  withBrainWrite
} from "../src/main/brainWriteCoordinator";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { ToolExecutor } from "../src/main/toolExecutor";
import { DEFAULT_CONFIG, type ToolPermissionRecord } from "../src/shared/types";

function deferred<T = void>(): {
  promise: Promise<T>;
  resolve(value?: T): void;
  reject(error: unknown): void;
} {
  let resolve!: (value?: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = (value) => resolvePromise(value as T);
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

async function eventLoopTurn(): Promise<void> {
  await new Promise<void>((resolve) => setImmediate(resolve));
}

describe("per-brain writer coordination", () => {
  const temporaryRoots: string[] = [];

  afterEach(async () => {
    await Promise.all(
      temporaryRoots.splice(0).map((root) => rm(root, { recursive: true, force: true }))
    );
  });

  it("runs one brain FIFO and lets a successor continue after failure", async () => {
    const coordinator = new BrainWriteCoordinator();
    const firstStarted = deferred();
    const releaseFirst = deferred();
    const order: string[] = [];

    const first = coordinator.run("brain-a", async () => {
      order.push("first-start");
      firstStarted.resolve();
      await releaseFirst.promise;
      order.push("first-fail");
      throw new Error("expected writer failure");
    });
    await firstStarted.promise;
    const second = coordinator.run("brain-a", async () => {
      order.push("second");
      return "saved";
    });

    await eventLoopTurn();
    expect(order).toEqual(["first-start"]);
    releaseFirst.resolve();
    await expect(first).rejects.toThrow("expected writer failure");
    await expect(second).resolves.toBe("saved");
    expect(order).toEqual(["first-start", "first-fail", "second"]);
  });

  it("cancels a queued writer without running it or bypassing its predecessor", async () => {
    const coordinator = new BrainWriteCoordinator();
    const firstStarted = deferred();
    const releaseFirst = deferred();
    const abort = new AbortController();
    let cancelledRan = false;
    let successorRan = false;

    const first = coordinator.run("brain-a", async () => {
      firstStarted.resolve();
      await releaseFirst.promise;
    });
    await firstStarted.promise;
    const cancelled = coordinator.run(
      "brain-a",
      async () => {
        cancelledRan = true;
      },
      abort.signal
    );
    abort.abort(new Error("cancel queued writer"));
    await expect(cancelled).rejects.toThrow("cancel queued writer");

    const successor = coordinator.run("brain-a", async () => {
      successorRan = true;
    });
    await eventLoopTurn();
    expect(cancelledRan).toBe(false);
    expect(successorRan).toBe(false);

    releaseFirst.resolve();
    await Promise.all([first, successor]);
    expect(cancelledRan).toBe(false);
    expect(successorRan).toBe(true);
  });

  it("keeps an acquired writer locked until its abort-aware operation unwinds", async () => {
    const coordinator = new BrainWriteCoordinator();
    const abort = new AbortController();
    const firstStarted = deferred();
    const finishUnwind = deferred();
    let secondRan = false;

    const first = coordinator.run(
      "brain-a",
      async () => {
        firstStarted.resolve();
        await finishUnwind.promise;
        abort.signal.throwIfAborted();
      },
      abort.signal
    );
    await firstStarted.promise;
    const second = coordinator.run("brain-a", async () => {
      secondRan = true;
    });
    abort.abort(new Error("cancel active writer"));
    await eventLoopTurn();
    expect(secondRan).toBe(false);

    finishUnwind.resolve();
    await expect(first).rejects.toThrow("cancel active writer");
    await second;
    expect(secondRan).toBe(true);
  });

  it("sorts multi-brain acquisitions so opposite caller order cannot deadlock", async () => {
    const coordinator = new BrainWriteCoordinator();
    const firstStarted = deferred();
    const releaseFirst = deferred();
    const order: string[] = [];

    const first = coordinator.run(["brain-b", "brain-a"], async () => {
      order.push("first");
      firstStarted.resolve();
      await releaseFirst.promise;
    });
    await firstStarted.promise;
    const second = coordinator.run(["brain-a", "brain-b"], async () => {
      order.push("second");
    });
    await eventLoopTurn();
    expect(order).toEqual(["first"]);

    releaseFirst.resolve();
    await Promise.all([first, second]);
    expect(order).toEqual(["first", "second"]);
  });

  it("coordinates repository-direct writers through the same singleton queue", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-brain-writer-repository-"));
    temporaryRoots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Before" });
    const writerStarted = deferred();
    const releaseWriter = deferred();
    let updateFinished = false;

    const writer = withBrainWrite(repository, brain.id, async () => {
      writerStarted.resolve();
      await releaseWriter.promise;
    });
    await writerStarted.promise;
    const update = repository
      .updateConfig(brain.id, { ...brain.config, name: "After" })
      .then((value) => {
        updateFinished = true;
        return value;
      });
    await eventLoopTurn();
    expect(updateFinished).toBe(false);

    releaseWriter.resolve();
    await writer;
    await expect(update).resolves.toMatchObject({ name: "After" });
  });

  it("serializes concurrent BrainService chat commits without losing either turn", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-brain-writer-chat-"));
    temporaryRoots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "FIFO chat" });
    const firstStarted = deferred();
    const releaseFirst = deferred();
    let chatCalls = 0;
    const tryRequestStream = vi.fn(
      async (_method: string, params: Record<string, unknown>) => {
        chatCalls += 1;
        if (chatCalls === 1) {
          firstStarted.resolve();
          await releaseFirst.promise;
        }
        return {
          text: `answer:${String(params.input)}`,
          trace: { id: `trace-${chatCalls}` },
          metrics: {}
        };
      }
    );
    const service = new BrainService(
      repository,
      {
        tryRequest: vi.fn(async () => ({})),
        tryRequestStream
      } as unknown as EngineSupervisor
    );

    const first = service.chat(brain.id, "first turn");
    await firstStarted.promise;
    const second = service.chat(brain.id, "second turn");
    await eventLoopTurn();
    expect(tryRequestStream).toHaveBeenCalledTimes(1);

    releaseFirst.resolve();
    await Promise.all([first, second]);
    const saved = await repository.get(brain.id);
    expect(saved.messages.map((message) => message.content)).toEqual([
      "first turn",
      "answer:first turn",
      "second turn",
      "answer:second turn"
    ]);
  });

  it("preserves a blocked chat commit and the final Off tool permission", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-brain-writer-permission-"));
    temporaryRoots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Permission order" });
    const chatStarted = deferred();
    const releaseChat = deferred();
    const service = new BrainService(
      repository,
      {
        tryRequest: vi.fn(async () => ({})),
        tryRequestStream: vi.fn(async () => {
          chatStarted.resolve();
          await releaseChat.promise;
          return { text: "permission-safe response", trace: { id: "permission-trace" } };
        })
      } as unknown as EngineSupervisor
    );

    const chat = service.chat(brain.id, "keep this turn");
    await chatStarted.promise;
    let permissionFinished = false;
    const permission = service
      .setToolPermission(brain.id, "windows.files", "off")
      .then((records) => {
        permissionFinished = true;
        return records;
      });
    await eventLoopTurn();
    expect(permissionFinished).toBe(false);

    releaseChat.resolve();
    await chat;
    await expect(permission).resolves.toEqual(
      expect.arrayContaining([expect.objectContaining({ toolId: "windows.files", level: "off" })])
    );
    const saved = await repository.get(brain.id);
    expect(saved.messages.map((message) => message.content)).toEqual([
      "keep this turn",
      "permission-safe response"
    ]);
    expect(
      saved.toolPermissions?.find((record) => record.toolId === "windows.files")?.level
    ).toBe("off");
  });

  it("shares the repository queue with ToolExecutor audit commits", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-brain-writer-tool-"));
    temporaryRoots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "FIFO audit" });
    const target = join(root, "visible.txt");
    await writeFile(target, "visible result");
    const writerStarted = deferred();
    const releaseWriter = deferred();
    const originalGet = repository.get.bind(repository);
    const writer = withBrainWrite(repository, brain.id, async () => {
      const current = await originalGet(brain.id);
      writerStarted.resolve();
      await releaseWriter.promise;
      current.messages.push({
        id: "committed-pair",
        role: "human",
        content: "committed while tool ran",
        createdAt: new Date().toISOString(),
        status: "complete"
      });
      await repository.save(current);
    });
    await writerStarted.promise;

    const permissions: ToolPermissionRecord[] = [
      {
        toolId: "windows.files",
        label: "Windows Files",
        level: "full",
        updatedAt: new Date().toISOString()
      }
    ];
    const service = {
      repository,
      listToolPermissions: vi.fn(async () => permissions)
    } as unknown as BrainService;
    const executor = new ToolExecutor(service, {} as RuntimeJobManager);
    const publicGet = vi.spyOn(repository, "get");
    const execution = executor.execute({
      brainId: brain.id,
      toolId: "windows.files",
      action: "read",
      arguments: { path: target }
    });
    await eventLoopTurn();
    expect(publicGet).not.toHaveBeenCalled();

    releaseWriter.resolve();
    const [, result] = await Promise.all([writer, execution]);
    expect(result.state).toBe("complete");
    const saved = await originalGet(brain.id);
    expect(saved.messages.map((message) => message.content)).toContain(
      "committed while tool ran"
    );
    expect(saved.journal?.some((entry) => entry.summary === "windows.files.read: complete.")).toBe(
      true
    );
  });
});
