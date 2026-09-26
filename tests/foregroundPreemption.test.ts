import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import {
  BrainService,
  LARGE_FOREGROUND_LOAD_TIMEOUT_MS,
  STANDARD_FOREGROUND_LOAD_TIMEOUT_MS,
  foregroundLoadTimeoutMs
} from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { ENGINE_REQUEST_NO_DEADLINE } from "../src/main/engineSupervisor";
import { DEFAULT_CONFIG } from "../src/shared/types";

describe("same-brain foreground preemption", () => {
  it("claims the worker before chat waits on the repository write lock", async () => {
    const order: string[] = [];
    const repository = {
      get: vi.fn(async () => {
        order.push("brain-write");
        throw new Error("fixture stops after lock acquisition");
      }),
      brainDirectory: vi.fn(() => "/fixture/brain")
    } as unknown as BrainRepository;
    const engine = {
      claimForeground: vi.fn(async () => {
        order.push("foreground-claim");
      })
    } as unknown as EngineSupervisor;
    const service = new BrainService(repository, engine);

    await expect(service.chat("brain-a", "hello")).rejects.toThrow(
      /fixture stops after lock acquisition/
    );

    expect(order).toEqual(["foreground-claim", "brain-write"]);
    expect(engine.claimForeground).toHaveBeenCalledOnce();
  });

  it("gives a large persisted substrate a bounded 30-minute cold-load window", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-load-timeout-"));
    const engineDirectory = join(root, "engine");
    await mkdir(engineDirectory);
    try {
      await writeFile(
        join(engineDirectory, "brain.json"),
        JSON.stringify({
          substrate: { persistence: { shardCount: 9_456 } }
        }),
        "utf8"
      );
      await expect(foregroundLoadTimeoutMs(root)).resolves.toBe(
        LARGE_FOREGROUND_LOAD_TIMEOUT_MS
      );

      await writeFile(
        join(engineDirectory, "brain.json"),
        JSON.stringify({
          substrate: { persistence: { shardCount: 64 } }
        }),
        "utf8"
      );
      await expect(foregroundLoadTimeoutMs(root)).resolves.toBe(
        STANDARD_FOREGROUND_LOAD_TIMEOUT_MS
      );
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("runs the complete online-learning chat turn without a wall-clock deadline", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-chat-deadline-"));
    try {
      const repository = new BrainRepository(join(root, "brains"));
      await repository.initialize();
      const brain = await repository.create({
        ...DEFAULT_CONFIG,
        name: "Online chat deadline fixture"
      });
      const controller = new AbortController();
      const request = vi.fn(async () => ({}));
      const requestStream = vi.fn(async () => ({ text: "hi" }));
      const engine = {
        claimForeground: vi.fn(async () => undefined),
        request,
        requestStream
      } as unknown as EngineSupervisor;
      const service = new BrainService(repository, engine);

      await expect(
        service.chat(brain.id, "hello", controller.signal)
      ).resolves.toMatchObject({
        humanMessage: { content: "hello" },
        brainMessage: { content: "hi" }
      });
      expect(requestStream).toHaveBeenCalledWith(
        "chat",
        expect.objectContaining({
          brainId: brain.id,
          input: "hello",
          storagePath: repository.brainDirectory(brain.id)
        }),
        expect.any(Function),
        ENGINE_REQUEST_NO_DEADLINE,
        controller.signal,
        expect.any(String),
        "foreground",
        expect.objectContaining({
          owner: "chat",
          requestId: expect.any(String),
          turnId: expect.any(String)
        })
      );
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
