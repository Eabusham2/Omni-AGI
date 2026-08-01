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
  it("runs one enabled brain per tick and passes the organic cooldown", async () => {
    const documents = new Map([
      ["disabled", brain("disabled", false)],
      ["active", brain("active", true)]
    ]);
    const repository = {
      list: vi.fn().mockResolvedValue([summary("disabled"), summary("active")]),
      get: vi.fn(async (id: string) => documents.get(id)!)
    };
    const cycle: IdleCycleResult = {
      brainId: "active",
      ran: true,
      actions: []
    };
    const actions = { idle: vi.fn().mockResolvedValue(cycle) };
    const scheduler = new IdleCognitionScheduler(repository, actions, {
      minimumIdleSeconds: 73
    });

    await expect(scheduler.tick()).resolves.toBe(cycle);
    expect(actions.idle).toHaveBeenCalledOnce();
    expect(actions.idle).toHaveBeenCalledWith("active", 73);
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
});
