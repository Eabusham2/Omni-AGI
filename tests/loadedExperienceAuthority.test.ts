import { describe, expect, it, vi } from "vitest";
import { BrainService } from "../src/main/brainService";
import { withBrainWrite } from "../src/main/brainWriteCoordinator";

const experience = { content: "Synthetic observed experiment outcome", name: "Fixture observation", sourceLabel: "fixture", license: "fixture" };
function fixture() {
  const brain = { id: "owner", trainingSources: [], journal: [], counters: {} };
  const repository = { brainDirectory: () => "/inert-owned-fixture", get: vi.fn(async () => brain), trainingSourceByContentHash: vi.fn(async () => undefined), save: vi.fn(async value => value) };
  const request = vi.fn(async () => ({ brainId: "owner", source: { learned_ideas: 1, parameter_update_steps: 1 } }));
  const service = Object.create(BrainService.prototype) as BrainService;
  Object.assign(service, { repository, engine: { request }, requireInstanceLoaded: vi.fn(), preflightStart: vi.fn(async () => { throw new Error("Unexpected implicit Load"); }) });
  return { service, repository, request };
}

describe("loaded-only integration authority (fake worker, no model)", () => {
  it("checks the live source after acquiring the existing writer, not just before queueing", async () => {
    const f = fixture(); let release!: () => void, entered!: () => void;
    const enteredPromise = new Promise<void>(resolve => { entered = resolve; });
    const held = withBrainWrite(f.repository as never, "owner", async () => { entered(); await new Promise<void>(resolve => { release = resolve; }); });
    await enteredPromise;
    const guard = vi.fn(async () => { throw new Error("Source revoked while queued"); });
    const pending = f.service.learnLoadedStructuredExperience("owner", experience, undefined, guard);
    expect(guard).not.toHaveBeenCalled(); release(); await held;
    await expect(pending).rejects.toThrow("Source revoked while queued"); expect(f.request).not.toHaveBeenCalled();
  });

  it("rechecks immediately before native admission without taking the preflight Load path", async () => {
    const f = fixture(), guard = vi.fn(async () => undefined);
    f.request.mockRejectedValueOnce(new Error("Synthetic RPC boundary"));
    await expect(f.service.learnLoadedStructuredExperience("owner", experience, undefined, guard)).rejects.toThrow("Synthetic RPC boundary");
    expect(guard).toHaveBeenCalledTimes(2); expect(f.service.preflightStart).not.toHaveBeenCalled(); expect(f.request).toHaveBeenCalledOnce();
  });

  it("retains the real returned ACK when Stop arrives after a synthetic native commit", async () => {
    const f = fixture(), stop = new AbortController();
    f.request.mockImplementationOnce(async () => { stop.abort(); return { brainId: "owner", source: { learned_ideas: 1, parameter_update_steps: 1 } }; });
    const result = await f.service.learnLoadedStructuredExperience("owner", experience, stop.signal, async () => undefined);
    expect(result.source.learnedParameterSteps).toBe(1); expect(f.repository.save).toHaveBeenCalledOnce();
    expect(f.service.preflightStart).not.toHaveBeenCalled(); expect(f.request).toHaveBeenCalledOnce();
  });
});
