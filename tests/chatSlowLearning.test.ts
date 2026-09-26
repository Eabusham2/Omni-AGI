import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import {
  BrainService,
  chatSlowReplayDelayMs
} from "../src/main/brainService";
import {
  BackgroundRequestDeferredError,
  type EngineSupervisor
} from "../src/main/engineSupervisor";
import { DEFAULT_CONFIG } from "../src/shared/types";

const nextImmediate = (): Promise<void> =>
  new Promise((resolve) => setImmediate(resolve));

describe("background chat slow learning", () => {
  it("spaces high-priority replay sooner without a binary cutoff", () => {
    expect(chatSlowReplayDelayMs(0.9)).toBeLessThan(chatSlowReplayDelayMs(0.2));
    expect(chatSlowReplayDelayMs(0)).toBeGreaterThan(0);
    expect(chatSlowReplayDelayMs(1)).toBeGreaterThan(0);
  });

  it("does not cold-load an identity with no durable chat replay", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-no-slow-job-"));
    try {
      const repository = new BrainRepository(join(root, "brains"));
      await repository.initialize();
      const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Idle mind" });
      const request = vi.fn(async () => ({ processed: false, pending: 0 }));
      const service = new BrainService(
        repository,
        { request } as unknown as EngineSupervisor
      );

      service.resumePendingChatLearning(brain.id);
      await vi.waitFor(() => {
        expect(service.isChatParameterLearningActive(brain.id)).toBe(false);
      });
      expect(request).not.toHaveBeenCalled();
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("persists a fast pause without planning resources or waiting for a worker RPC", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-fast-learning-pause-"));
    try {
      const repository = new BrainRepository(join(root, "brains"));
      await repository.initialize();
      const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Paused mind" });
      const request = vi.fn(async () => ({}));
      const cancelBackgroundRequest = vi.fn(() => false);
      const service = new BrainService(
        repository,
        { request, cancelBackgroundRequest } as unknown as EngineSupervisor
      );
      const planner = vi.spyOn(service, "preflightStart");
      const paused = await service.setOnlineLearning(brain.id, false);
      expect(paused.config.onlineLearning).toBe(false);
      expect((await repository.get(brain.id)).config.onlineLearning).toBe(false);
      expect(cancelBackgroundRequest).toHaveBeenCalledWith(
        brain.id, "consolidate_chat_learning"
      );
      expect(request).not.toHaveBeenCalled();
      expect(planner).not.toHaveBeenCalled();

      // A fresh desktop service must honor the durable host switch rather
      // than eagerly restoring old pending jobs on launch.
      const restarted = new BrainService(
        repository,
        { request } as unknown as EngineSupervisor
      );
      restarted.resumePendingChatLearning(brain.id);
      await nextImmediate();
      expect(request).not.toHaveBeenCalled();
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("can suspend auto-replay for one QA launch without editing the saved brain", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-suspend-replay-launch-"));
    vi.stubEnv("OMNI_SUSPEND_BACKGROUND_LEARNING", "1");
    try {
      const repository = new BrainRepository(join(root, "brains"));
      await repository.initialize();
      const brain = await repository.create({ ...DEFAULT_CONFIG, name: "QA launch" });
      const request = vi.fn(async () => ({ processed: true, pending: 0 }));
      const service = new BrainService(
        repository,
        { request } as unknown as EngineSupervisor
      );
      service.resumePendingChatLearning(brain.id);
      await nextImmediate();
      expect(request).not.toHaveBeenCalled();
      expect((await repository.get(brain.id)).config.onlineLearning).toBe(true);
      await expect(service.setOnlineLearning(brain.id, true)).rejects.toThrow(
        "suspended for this app launch"
      );
    } finally {
      vi.unstubAllEnvs();
      await rm(root, { recursive: true, force: true });
    }
  });

  it("stops a running replay and retains its exact pending job for Resume", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-pause-running-replay-"));
    try {
      const repository = new BrainRepository(join(root, "brains"));
      await repository.initialize();
      const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Replay pause" });
      const engineDirectory = join(repository.brainDirectory(brain.id), "engine");
      await mkdir(engineDirectory, { recursive: true });
      const jobId = "b".repeat(64);
      await writeFile(
        join(engineDirectory, "brain.json"),
        JSON.stringify({ pending_chat_slow_learning: [{ jobId }] })
      );
      let rejectFirst!: (error: Error) => void;
      const firstReplay = new Promise<never>((_resolve, reject) => {
        rejectFirst = reject;
      });
      const request = vi.fn()
        .mockImplementationOnce(() => firstReplay)
        .mockResolvedValue({ processed: true, pending: 0, jobId });
      const cancelBackgroundRequest = vi.fn(() => {
        rejectFirst(new BackgroundRequestDeferredError("consolidate_chat_learning"));
        return true;
      });
      const service = new BrainService(
        repository,
        { request, cancelBackgroundRequest } as unknown as EngineSupervisor
      );
      vi.spyOn(service, "preflightStart").mockResolvedValue(undefined);
      service.resumePendingChatLearning(brain.id);
      await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(1));

      const paused = await service.setOnlineLearning(brain.id, false);
      expect(paused.config.onlineLearning).toBe(false);
      expect(cancelBackgroundRequest).toHaveBeenCalledOnce();
      await vi.waitFor(() => expect(service.isChatParameterLearningActive(brain.id)).toBe(false));
      expect(request).toHaveBeenCalledTimes(1);
      expect(JSON.parse(await readFile(
        join(engineDirectory, "brain.json"), "utf8"
      )).pending_chat_slow_learning).toEqual([{ jobId }]);

      await service.setOnlineLearning(brain.id, true);
      await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(2));
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("keeps a durable replay queued through a storage pause and resumes when space recovers", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-slow-storage-retry-"));
    try {
      const repository = new BrainRepository(join(root, "brains"));
      await repository.initialize();
      const brain = await repository.create({
        ...DEFAULT_CONFIG,
        name: "Storage retry fixture"
      });
      const engineDirectory = join(repository.brainDirectory(brain.id), "engine");
      await mkdir(engineDirectory, { recursive: true });
      await writeFile(
        join(engineDirectory, "brain.json"),
        JSON.stringify({ pending_chat_slow_learning: [{ jobId: "a".repeat(64) }] })
      );
      const request = vi.fn(async () => ({
        processed: true,
        pending: 0,
        jobId: "a".repeat(64),
        parameterChecksumBefore: "before",
        parameterChecksumAfter: "after"
      }));
      const service = new BrainService(
        repository,
        { request } as unknown as EngineSupervisor
      );
      const preflight = vi.spyOn(service, "preflightStart")
        .mockRejectedValueOnce(new Error("storage reserve reached"))
        .mockResolvedValue(undefined);
      const retry = service as unknown as {
        schedulePendingChatLearning(id: string, priority: number, jobId: string): void;
        chatSlowLearning: Map<string, Promise<void>>;
        chatSlowLearningTimers: Map<string, NodeJS.Timeout>;
      };
      retry.schedulePendingChatLearning(brain.id, 0.5, "a".repeat(64));
      service.resumePendingChatLearning(brain.id);
      await vi.waitFor(() => {
        expect(retry.chatSlowLearningTimers.has(brain.id)).toBe(true);
      });

      expect(request).not.toHaveBeenCalled();
      expect(retry.chatSlowLearning.has(brain.id)).toBe(false);
      expect(service.isChatParameterLearningActive(brain.id)).toBe(true);

      service.resumePendingChatLearning(brain.id);
      await vi.waitFor(() => {
        expect(preflight).toHaveBeenCalledTimes(2);
        expect(request).toHaveBeenCalledTimes(1);
      });
      expect(retry.chatSlowLearningTimers.has(brain.id)).toBe(false);
      expect(service.isChatParameterLearningActive(brain.id)).toBe(false);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("resolves the fast turn and starts the next chat while slow replay is pending", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-chat-slow-service-"));
    try {
      const repository = new BrainRepository(join(root, "brains"));
      await repository.initialize();
      const brain = await repository.create({
        ...DEFAULT_CONFIG,
        name: "Async slow chat fixture"
      });

      let releaseSlow!: () => void;
      const slowGate = new Promise<void>((resolve) => {
        releaseSlow = resolve;
      });
      let markSlowStarted!: () => void;
      const slowStarted = new Promise<void>((resolve) => {
        markSlowStarted = resolve;
      });
      let chatCount = 0;
      const request = vi.fn(async (method: string) => {
        if (method !== "consolidate_chat_learning") return {};
        markSlowStarted();
        await slowGate;
        return { processed: true, pending: 0 };
      });
      const requestStream = vi.fn(async () => {
        chatCount += 1;
        return {
          text: `reply-${chatCount}`,
          trace: {
            id: `trace-${chatCount}`,
            slow_learning_job: {
              jobId: `${chatCount}`.padStart(64, "a"),
              priority: 0.4
            }
          }
        };
      });
      const engine = {
        claimForeground: vi.fn(async () => undefined),
        request,
        requestStream
      } as unknown as EngineSupervisor;
      const service = new BrainService(repository, engine);

      await expect(service.chat(brain.id, "first turn")).resolves.toMatchObject({
        brainMessage: { content: "reply-1" }
      });
      service.resumePendingChatLearning(brain.id);
      await slowStarted;
      expect(service.isChatParameterLearningActive(brain.id)).toBe(true);

      await expect(service.chat(brain.id, "next turn")).resolves.toMatchObject({
        brainMessage: { content: "reply-2" }
      });
      expect(requestStream).toHaveBeenCalledTimes(2);
      expect(engine.claimForeground).toHaveBeenCalledTimes(2);
      expect(
        request.mock.calls.some(([method]) => method === "consolidate_chat_learning")
      ).toBe(true);

      releaseSlow();
      await nextImmediate();
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
