import { describe, expect, it, vi } from "vitest";
import { IdleCognitionScheduler } from "../src/main/idleCognitionScheduler";
import type {
  BrainDocument,
  BrainSummary,
  IdleCycleResult
} from "../src/shared/types";

function brain(id: string, enabled: boolean): BrainDocument {
  return {
    id,
    config: { idleCognition: enabled }
  } as unknown as BrainDocument;
}

function summary(id: string): BrainSummary {
  return { id } as BrainSummary;
}

describe("IdleCognitionScheduler", () => {
  it("delays the first startup cycle so recovery and Build can claim the worker", async () => {
    vi.useFakeTimers();
    try {
      const repository = {
        list: vi.fn().mockResolvedValue([summary("startup")]),
        get: vi.fn().mockResolvedValue(brain("startup", true))
      };
      const actions = {
        idle: vi.fn().mockResolvedValue({
          brainId: "startup",
          ran: false,
          reason: "cooldown",
          actions: []
        })
      };
      const scheduler = new IdleCognitionScheduler(repository, actions, {
        intervalMs: 1_000,
        startupDelayMs: 5_000
      });

      scheduler.start();
      await vi.advanceTimersByTimeAsync(4_999);
      expect(repository.list).not.toHaveBeenCalled();
      expect(actions.idle).not.toHaveBeenCalled();

      await vi.advanceTimersByTimeAsync(1);
      await vi.waitFor(() => expect(actions.idle).toHaveBeenCalledOnce());
      scheduler.stop();
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not enter prompt-free cognition while foreground initialization is busy", async () => {
    let initializationBusy = true;
    const repository = {
      list: vi.fn().mockResolvedValue([summary("existing")]),
      get: vi.fn().mockResolvedValue(brain("existing", true))
    };
    const cycle: IdleCycleResult = {
      brainId: "existing",
      ran: false,
      reason: "cooldown",
      actions: []
    };
    const actions = { idle: vi.fn().mockResolvedValue(cycle) };
    const scheduler = new IdleCognitionScheduler(repository, actions, {
      isInitializationBusy: () => initializationBusy
    });

    await expect(scheduler.tick()).resolves.toBeUndefined();
    expect(repository.list).not.toHaveBeenCalled();
    expect(actions.idle).not.toHaveBeenCalled();

    initializationBusy = false;
    await expect(scheduler.tick()).resolves.toBe(cycle);
    expect(actions.idle).toHaveBeenCalledOnce();
  });

  it("skips an identity whose persisted Active Mode is off", async () => {
    const documents = new Map([["legacy", brain("legacy", false)]]);
    const repository = {
      list: vi.fn().mockResolvedValue([summary("legacy")]),
      get: vi.fn(async (id: string) => documents.get(id)!)
    };
    const actions = { idle: vi.fn() };
    const scheduler = new IdleCognitionScheduler(repository, actions, {
      minimumIdleSeconds: 73
    });

    await expect(scheduler.tick()).resolves.toBeUndefined();
    expect(actions.idle).not.toHaveBeenCalled();
    expect(scheduler.status()).toMatchObject({
      phase: "paused-inactive",
      detail: "Active Mode is off for every mind."
    });
  });

  it("holds one defensive process lease even when legacy documents have multiple true flags", async () => {
    let clock = 0;
    const documents = new Map([
      ["newest", brain("newest", true)],
      ["older", brain("older", true)]
    ]);
    const repository = {
      list: vi.fn().mockResolvedValue([summary("newest"), summary("older")]),
      get: vi.fn(async (id: string) => documents.get(id)!)
    };
    const actions = {
      idle: vi.fn(async (brainId: string): Promise<IdleCycleResult> => ({
        brainId,
        ran: false,
        reason: "cooldown",
        actions: []
      }))
    };
    const scheduler = new IdleCognitionScheduler(repository, actions, {
      intervalMs: 1_000,
      now: () => clock
    });

    await scheduler.tick();
    clock = 2_000;
    await scheduler.tick();
    expect(actions.idle).toHaveBeenCalledTimes(2);
    expect(actions.idle).toHaveBeenNthCalledWith(1, "newest", 6);
    expect(actions.idle).toHaveBeenNthCalledWith(2, "newest", 6);
    expect(scheduler.status().activeModeBrainId).toBe("newest");
  });

  it("does not fall through to another identity while the active lease is not ready", async () => {
    const initializing = brain("initializing-owner", true);
    initializing.readiness = {
      state: "initializing",
      startedAt: "2026-09-12T00:00:00.000Z",
      attempt: 1
    };
    const documents = new Map([
      [initializing.id, initializing],
      ["legacy-second-owner", brain("legacy-second-owner", true)]
    ]);
    const repository = {
      list: vi.fn().mockResolvedValue([
        summary(initializing.id),
        summary("legacy-second-owner")
      ]),
      get: vi.fn(async (id: string) => documents.get(id)!)
    };
    const actions = { idle: vi.fn() };
    const scheduler = new IdleCognitionScheduler(repository, actions);

    await expect(scheduler.tick()).resolves.toBeUndefined();
    expect(actions.idle).not.toHaveBeenCalled();
    expect(scheduler.status()).toMatchObject({
      phase: "paused-initialization",
      activeModeBrainId: initializing.id
    });
  });

  it("does not overlap slow neural idle cycles", async () => {
    let release = (_value: IdleCycleResult): void => undefined;
    const pending = new Promise<IdleCycleResult>((resolve) => {
      release = resolve;
    });
    const repository = {
      list: vi.fn().mockResolvedValue([summary("active")]),
      get: vi.fn().mockResolvedValue(brain("active", true))
    };
    const actions = { idle: vi.fn().mockReturnValue(pending) };
    const scheduler = new IdleCognitionScheduler(repository, actions);

    const first = scheduler.tick();
    await vi.waitFor(() => expect(actions.idle).toHaveBeenCalledOnce());
    await expect(scheduler.tick()).resolves.toBeUndefined();
    release({ brainId: "active", ran: false, reason: "cooldown", actions: [] });
    await expect(first).resolves.toMatchObject({ reason: "cooldown" });
  });

  it("does not create repeated organic actions while approval or execution is pending", async () => {
    const repository = {
      list: vi.fn().mockResolvedValue([summary("active")]),
      get: vi.fn().mockResolvedValue(brain("active", true))
    };
    const actions = {
      idle: vi.fn(),
      isBusy: vi.fn().mockReturnValue(true)
    };
    const scheduler = new IdleCognitionScheduler(repository, actions);

    await expect(scheduler.tick()).resolves.toBeUndefined();
    expect(actions.isBusy).toHaveBeenCalledWith("active");
    expect(actions.idle).not.toHaveBeenCalled();
  });

  it("does not turn learning activity into a Ponder cycle", async () => {
    let clock = 1_000;
    const repository = {
      list: vi.fn().mockResolvedValue([summary("active")]),
      get: vi.fn().mockResolvedValue(brain("active", true))
    };
    const actions = { idle: vi.fn() };
    const isLearning = vi.fn().mockReturnValue(true);
    const scheduler = new IdleCognitionScheduler(repository, actions, {
      minimumIdleSeconds: 30,
      now: () => clock,
      isLearning
    });

    await expect(scheduler.tick()).resolves.toBeUndefined();
    expect(isLearning).toHaveBeenCalledWith("active");
    expect(actions.idle).not.toHaveBeenCalled();

    isLearning.mockReturnValue(false);
    clock += 29_999;
    await scheduler.tick();
    expect(actions.idle).not.toHaveBeenCalled();
    clock += 2;
    await scheduler.tick();
    expect(actions.idle).toHaveBeenCalledOnce();
  });

  it("keeps always-active cognition inside a wall-time duty budget", async () => {
    let clock = 0;
    const repository = {
      list: vi.fn().mockResolvedValue([summary("active")]),
      get: vi.fn().mockResolvedValue(brain("active", true))
    };
    const actions = {
      idle: vi.fn(async () => {
        clock += 100;
        return { brainId: "active", ran: true, actions: [] };
      })
    };
    const scheduler = new IdleCognitionScheduler(repository, actions, {
      intervalMs: 1_000,
      maxDutyCycle: 0.1,
      now: () => clock
    });

    await scheduler.tick();
    expect(actions.idle).toHaveBeenCalledOnce();

    // The 100 ms neural cycle is followed by at least the scheduler interval,
    // keeping the prompt-free loop bounded even when repeatedly ticked.
    clock = 999;
    await scheduler.tick();
    expect(actions.idle).toHaveBeenCalledOnce();

    clock = 1_100;
    await scheduler.tick();
    expect(actions.idle).toHaveBeenCalledTimes(2);
  });

  it("honors a longer neural cooldown returned by the worker", async () => {
    let clock = 0;
    const repository = {
      list: vi.fn().mockResolvedValue([summary("active")]),
      get: vi.fn().mockResolvedValue(brain("active", true))
    };
    const actions = {
      idle: vi.fn().mockResolvedValue({
        brainId: "active",
        ran: false,
        reason: "cooldown",
        retryAfterSeconds: 30,
        actions: []
      })
    };
    const scheduler = new IdleCognitionScheduler(repository, actions, {
      intervalMs: 1_000,
      now: () => clock
    });

    await scheduler.tick();
    clock = 29_999;
    await scheduler.tick();
    expect(actions.idle).toHaveBeenCalledOnce();
    clock = 30_001;
    await scheduler.tick();
    expect(actions.idle).toHaveBeenCalledTimes(2);
  });

  it("reserves the cancelled brain for an immediate foreground retry", async () => {
    let clock = 10_000;
    const repository = {
      list: vi.fn().mockResolvedValue([summary("cancelled")]),
      get: vi.fn().mockResolvedValue(brain("cancelled", true))
    };
    const actions = {
      idle: vi.fn().mockResolvedValue({
        brainId: "cancelled",
        ran: false,
        reason: "cooldown",
        actions: []
      })
    };
    const scheduler = new IdleCognitionScheduler(repository, actions, {
      intervalMs: 1_000,
      postCancelCooldownMs: 60_000,
      now: () => clock
    });

    scheduler.reserveForegroundAfterCancel("cancelled");
    await scheduler.tick();
    expect(actions.idle).not.toHaveBeenCalled();

    clock += 59_999;
    await scheduler.tick();
    expect(actions.idle).not.toHaveBeenCalled();

    clock += 2;
    await scheduler.tick();
    expect(actions.idle).toHaveBeenCalledOnce();
  });

  it("backs off exponentially when the neural worker is unavailable", async () => {
    let clock = 0;
    const repository = {
      list: vi.fn().mockResolvedValue([summary("active")]),
      get: vi.fn().mockResolvedValue(brain("active", true))
    };
    const actions = {
      idle: vi.fn().mockRejectedValue(new Error("worker unavailable"))
    };
    const onError = vi.fn();
    const scheduler = new IdleCognitionScheduler(repository, actions, {
      intervalMs: 1_000,
      now: () => clock,
      onError
    });

    await scheduler.tick();
    expect(actions.idle).toHaveBeenCalledOnce();
    expect(onError).toHaveBeenCalledOnce();

    clock = 999;
    await scheduler.tick();
    expect(actions.idle).toHaveBeenCalledOnce();
    clock = 1_001;
    await scheduler.tick();
    expect(actions.idle).toHaveBeenCalledTimes(2);

    // The second failure doubles the cooldown instead of retrying every tick.
    clock = 2_500;
    await scheduler.tick();
    expect(actions.idle).toHaveBeenCalledTimes(2);
    clock = 3_002;
    await scheduler.tick();
    expect(actions.idle).toHaveBeenCalledTimes(3);
  });

  it("publishes visible running state and cancels only the optional worker request", async () => {
    let finish = (_value: IdleCycleResult): void => undefined;
    const pending = new Promise<IdleCycleResult>((resolve) => {
      finish = resolve;
    });
    const repository = {
      list: vi.fn().mockResolvedValue([summary("resident")]),
      get: vi.fn().mockResolvedValue(brain("resident", true))
    };
    const preemptBackground = vi.fn().mockResolvedValue(undefined);
    const scheduler = new IdleCognitionScheduler(
      repository,
      { idle: vi.fn().mockReturnValue(pending) },
      { preemptBackground }
    );
    const statuses: string[] = [];
    scheduler.onStatus((status) => statuses.push(status.phase));

    const cycle = scheduler.tick();
    await vi.waitFor(() => expect(scheduler.status().phase).toBe("running"));
    expect(scheduler.status()).toMatchObject({
      activeBrainId: "resident",
      activeModeBrainId: "resident",
      cancellable: true,
      maxDutyCycle: 0.12
    });

    await expect(scheduler.cancelActive()).resolves.toBe(true);
    expect(preemptBackground).toHaveBeenCalledOnce();
    expect(scheduler.status().phase).toBe("cancelling");

    finish({
      brainId: "resident",
      ran: false,
      reason: "foreground-work",
      actions: []
    });
    await expect(cycle).resolves.toMatchObject({ reason: "foreground-work" });
    expect(scheduler.status()).toMatchObject({
      phase: "paused-foreground",
      cancellable: false
    });
    expect(statuses).toEqual(
      expect.arrayContaining(["stopped", "running", "cancelling", "paused-foreground"])
    );
  });

  it("exposes training and initialization preemption as operational state", async () => {
    let initializationBusy = true;
    let learning = false;
    const repository = {
      list: vi.fn().mockResolvedValue([summary("resident")]),
      get: vi.fn().mockResolvedValue(brain("resident", true))
    };
    const actions = { idle: vi.fn() };
    const scheduler = new IdleCognitionScheduler(repository, actions, {
      isInitializationBusy: () => initializationBusy,
      isLearning: () => learning
    });

    await scheduler.tick();
    expect(scheduler.status().phase).toBe("paused-initialization");
    initializationBusy = false;
    learning = true;
    await scheduler.tick();
    expect(scheduler.status()).toMatchObject({
      phase: "paused-training",
      cancellable: false
    });
    expect(actions.idle).not.toHaveBeenCalled();
  });
});
