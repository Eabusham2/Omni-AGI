import { EventEmitter } from "node:events";
import { describe, expect, it, vi } from "vitest";
import { BrainService, RuntimeJobManager } from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";

describe("crawl completion and policy recovery without a neural worker", () => {
  it.each([
    { stopped: true, coverage: { complete: false } },
    { stopped: true, coverage: { complete: true } },
    { stopped: false, coverage: { complete: false } },
    { stopped: false },
  ])("does not publish incomplete crawl output as complete: %j", async (output) => {
    const service = { crawlWeb: vi.fn(async () => output) } as unknown as BrainService;
    const manager = new RuntimeJobManager(service, new EventEmitter() as EngineSupervisor);
    const started = manager.startCrawl({ brainId: "fixture-brain", url: "https://example.com" });
    const terminal = await manager.wait(started.id);
    expect(terminal.state).toBe("failed");
    expect(terminal.progress).toBeLessThan(1);
    expect(terminal.output).toEqual(output);
    expect(terminal.error).toMatch(/coverage/i);
  });

  it("accepts a finished crawl only with explicit complete coverage", async () => {
    const service = {
      crawlWeb: vi.fn(async () => ({ stopped: false, coverage: { complete: true } })),
    } as unknown as BrainService;
    const manager = new RuntimeJobManager(service, new EventEmitter() as EngineSupervisor);
    const started = manager.startCrawl({ brainId: "fixture-brain", url: "https://example.com" });
    await expect(manager.wait(started.id)).resolves.toMatchObject({ state: "complete", progress: 1 });
  });

  it.each(["pretrain", "encode", "archive"] as const)(
    "resumes the exact persisted %s policy without converting it to encode",
    async (policy) => {
      const ingestManifest = vi.fn(async () => ({ paused: false, coverage: { complete: true } }));
      const service = {
        datasets: { progress: vi.fn(async () => ({ lastEntryReceipt: { policy } })) },
        ingestManifest,
      } as unknown as BrainService;
      const manager = new RuntimeJobManager(service, new EventEmitter() as EngineSupervisor);
      const started = await manager.resumeIngestion({ brainId: "fixture-brain", manifestId: "manifest-a" });
      await manager.wait(started.id);
      expect(ingestManifest.mock.calls[0]?.slice(0, 3)).toEqual(["fixture-brain", "manifest-a", policy]);
    },
  );

  it("does not guess a policy before the first committed entry", async () => {
    const ingestManifest = vi.fn();
    const service = {
      datasets: { progress: vi.fn(async () => ({})) },
      ingestManifest,
    } as unknown as BrainService;
    const manager = new RuntimeJobManager(service, new EventEmitter() as EngineSupervisor);
    await expect(manager.resumeIngestion({ brainId: "fixture-brain", manifestId: "manifest-a" }))
      .rejects.toThrow(/original ingestion policy/i);
    expect(ingestManifest).not.toHaveBeenCalled();
  });

  it("sends crawl cancellation to the exact worker request and waits for acknowledgement", async () => {
    let learnerSignal: AbortSignal | undefined;
    let learnerJobId: string | undefined;
    const crawlWeb = vi.fn(async (...args: unknown[]) => {
      learnerSignal = args[3] as AbortSignal;
      learnerJobId = args[4] as string;
      await new Promise<void>((resolve) => learnerSignal!.addEventListener("abort", () => resolve(), { once: true }));
      return { stopped: true, coverage: { complete: false } };
    });
    const service = {
      crawlWeb,
      repository: { brainDirectory: () => "/fixture/brain" },
    } as unknown as BrainService;
    const engine = new EventEmitter() as EngineSupervisor;
    const cancelRequest = vi.fn(async (requestId: string) => ({
      requestId, phase: "not-found" as const, acknowledged: true,
      workerTerminationAcknowledged: false,
    }));
    const request = vi.fn(async (_method: string, params: Record<string, unknown>) => ({
      jobId: params.jobId, cancelled: true, acknowledged: true,
    }));
    engine.cancelRequest = cancelRequest as EngineSupervisor["cancelRequest"];
    engine.request = request as EngineSupervisor["request"];
    const manager = new RuntimeJobManager(service, engine);
    const started = manager.startCrawl({ brainId: "fixture-brain", url: "https://example.com" });
    expect(learnerJobId).toBe(started.id);
    expect(learnerSignal?.aborted).toBe(false);
    const cancelled = await manager.cancel(started.id);
    expect(learnerSignal?.aborted).toBe(true);
    expect(cancelRequest).toHaveBeenCalledWith(started.id);
    expect(request.mock.calls[0]?.slice(0, 2)).toEqual([
      "cancel", expect.objectContaining({ jobId: started.id, kind: "crawl" }),
    ]);
    expect(cancelled.state).toBe("cancelled");
    expect(cancelled.progress).toBeLessThan(1);
  });
});
