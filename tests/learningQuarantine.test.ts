/** Real host task/SQLite orchestration with an inert crawler; no network or neural work. */
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LearningCoordinator, type LearningCoordinatorOptions } from "../src/main/learningCoordinator";
import { UserQuestionService } from "../src/main/userQuestions";
import { declaredLearningCapabilityPack } from "../src/main/learningCapabilityPack";
import catalog from "../tools/catalog.json";
import type { LearningTask, SpecificLearnRequest } from "../src/shared/learning";
import type { TrainingCoverage, WebCrawlRequest, WebCrawlResult } from "../src/shared/types";

function coverage(id: string, complete = true): TrainingCoverage {
  return { schemaVersion: 1, manifestId: id, discoveredFiles: 2, processedFiles: complete ? 2 : 1, rejectedFiles: 0,
    discoveredRecords: 2, processedRecords: complete ? 2 : 1, rejectedRecords: 0, discoveredBytes: 20, processedBytes: complete ? 20 : 10,
    shards: 1, modalityCounts: { text: 2 }, errors: [], complete, updatedAt: new Date().toISOString() };
}
function result(request: WebCrawlRequest, complete = true): WebCrawlResult {
  return { crawlId: request.crawlId!, startUrl: request.url, visited: complete ? 2 : 1, skipped: 0, results: [], resultCount: complete ? 2 : 1,
    resultsTruncated: true, resultLog: "existing-owned-crawl-receipts.jsonl", warnings: [], warningCount: 0, warningsTruncated: false,
    frontierRemaining: complete ? 0 : 1, stopped: !complete, coverage: coverage(request.crawlId!, complete) };
}
describe("optional chat web quarantine without another crawler backend", () => {
  let root: string, questions: UserQuestionService, coordinator: LearningCoordinator, options: LearningCoordinatorOptions;
  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "omni-learning-quarantine-")); await mkdir(join(root, "brain"));
    questions = new UserQuestionService({ brainDirectory: id => join(root, id) });
    options = { brainDirectory: id => join(root, id), requireLoaded: vi.fn(), authorize: vi.fn(async () => undefined), reserveWrite: vi.fn(async () => undefined), questions,
      manifest: async () => { throw new Error("No file grants in this web-only fixture"); }, prepare: vi.fn(async () => { throw new Error("No source preparation expected"); }),
      ingest: vi.fn(async () => { throw new Error("No neural ingestion expected in this fixture"); }),
      crawl: vi.fn(async (request, _control, policy) => { await policy.policy(); return result(request); }) };
    coordinator = new LearningCoordinator(options);
  });
  afterEach(async () => { await coordinator.close(); questions.close(); await rm(root, { recursive: true, force: true }); });
  const wait = (requestId: string, state: LearningTask["state"]) => {
    let unsubscribe = () => {};
    const promise = new Promise<LearningTask>((resolve, reject) => {
      unsubscribe = coordinator.subscribe(event => {
        if (event.task.brainId !== "brain" || event.task.requestId !== requestId) return;
        if (event.task.state === state) { unsubscribe(); resolve(event.task); }
        else if (["failed", "stopped"].includes(event.task.state)) { unsubscribe(); reject(new Error(event.task.error ?? event.task.label)); }
      });
    });
    return { promise, unsubscribe };
  };
  const run = async (request: SpecificLearnRequest) => {
    const terminal = wait(request.requestId!, "complete");
    try { await coordinator.start(request); return await terminal.promise; } finally { terminal.unsubscribe(); }
  };

  it("exposes optional boolean quarantine through the existing Ask-level declared chat schema", () => {
    const tool = catalog.tools.find(tool => tool.id === "data.learn")!;
    expect(tool.defaultGrant).toBe("ask"); expect(tool.actions.start!.input.quarantine).toBe("boolean?");
    const declared = declaredLearningCapabilityPack().schemas.find(schema => schema.id === "data.learn")!.actionInputSchemas.start!;
    expect(declared.additionalProperties).toBe(false); expect(declared.properties.quarantine).toEqual({ type: "boolean" });
    expect(declared.required).not.toContain("quarantine");
  });
  it("passes explicit quarantine to the existing crawl owner while retaining scope, full traversal and honest result labels", async () => {
    const completed = await run({ brainId: "brain", requestId: "collect-only", quarantine: true,
      inputs: [{ kind: "web", url: "https://docs.example.org/", followLinks: true }],
      constraints: { allowedSites: [{ domain: "docs.example.org" }] } });
    const [request, control, policy] = vi.mocked(options.crawl).mock.calls[0]!;
    expect(request).toMatchObject({ brainId: "brain", quarantine: true, resume: true, policy: "encode", respectRobots: true });
    expect(request.maxPages).toBeUndefined(); expect(request.maxDepth).toBeUndefined();
    expect(control.signal).toBeInstanceOf(AbortSignal); expect(typeof control.progress).toBe("function");
    expect((await policy.policy()).constraints.allowedSites).toEqual([{ domain: "docs.example.org", includeSubdomains: false }]);
    expect(completed).toMatchObject({ state: "complete", traversalComplete: true, sources: [{ state: "complete", quarantined: true,
      coverage: { processedRecords: 2, complete: true }, result: { resultLog: "existing-owned-crawl-receipts.jsonl" } }] });
    expect(completed.label).toContain("quarantined web sources requested no new learning");
    expect(options.ingest).not.toHaveBeenCalled(); expect(options.prepare).not.toHaveBeenCalled();
    expect(options.authorize).toHaveBeenCalledWith("brain", "specific-learn");
    await expect(coordinator.start({ brainId: "brain", requestId: "collect-only", quarantine: false,
      inputs: [{ kind: "web", url: "https://docs.example.org/", followLinks: true }], constraints: { allowedSites: [{ domain: "docs.example.org" }] } })).rejects.toThrow(/different contents/);
  });
  it("preserves the existing direct-learning default and permits an explicit per-web override", async () => {
    const ordinary = await run({ brainId: "brain", requestId: "ordinary", inputs: [{ kind: "web", url: "https://example.org/" }] });
    expect(vi.mocked(options.crawl).mock.calls[0]![0].quarantine).toBe(false);
    expect(ordinary.sources[0]!.quarantined).toBeUndefined();
    const mixed = await run({ brainId: "brain", requestId: "mixed", quarantine: true, inputs: [
      { kind: "web", url: "https://example.org/held" }, { kind: "web", url: "https://example.org/learn", quarantine: false }
    ] });
    expect(vi.mocked(options.crawl).mock.calls.slice(1).map(call => call[0].quarantine)).toEqual([true, false]);
    expect(mixed.sources.map(source => source.quarantined === true)).toEqual([true, false]);
  });
  it("retains the exact quarantine flag, crawl cursor and scope across pause, restart and resume", async () => {
    let entered!: () => void; const startedCrawl = new Promise<void>(resolve => { entered = resolve; });
    let calls = 0;
    options.crawl = vi.fn(async (request, control, policy) => {
      await policy.policy(); calls++;
      if (calls === 1) { entered(); await new Promise<void>(resolve => control.signal.addEventListener("abort", () => resolve(), { once: true })); return result(request, false); }
      return result(request);
    });
    const task = await coordinator.start({ brainId: "brain", requestId: "resume-held", inputs: [{ kind: "web", url: "https://example.org/", quarantine: true, followLinks: true }],
      constraints: { excludedSites: [{ domain: "ads.example.org" }] } });
    await startedCrawl; const paused = await coordinator.pause("brain", task.id);
    expect(paused.sources[0]).toMatchObject({ state: "paused", quarantined: true, coverage: { processedRecords: 1, complete: false } });
    await coordinator.close(); coordinator = new LearningCoordinator(options); await coordinator.recover("brain");
    const completed = wait("resume-held", "complete"); await coordinator.resume("brain", task.id); const final = await completed.promise;
    const requests = vi.mocked(options.crawl).mock.calls.map(call => call[0]);
    expect(requests.map(value => value.crawlId)).toEqual([paused.sources[0]!.crawlId, paused.sources[0]!.crawlId]);
    expect(requests.every(value => value.quarantine === true && value.resume === true)).toBe(true);
    expect(final.policy.sha256).toBe(paused.policy.sha256); expect(final.sources[0]!.coverage!.processedRecords).toBe(2);
  });
  it("rejects malformed flags and raw paths, and keeps current Off authorization before any crawl", async () => {
    for (const input of [
      { quarantine: "yes", inputs: [{ kind: "web", url: "https://example.org/" }] },
      { inputs: [{ kind: "web", url: "https://example.org/", quarantine: "yes" }] },
      { quarantine: true, inputs: [{ kind: "text", text: "not a web source" }] },
      { quarantine: true, inputs: [{ kind: "dataset", path: "/not-a-grant" }] }
    ]) await expect(coordinator.start({ brainId: "brain", ...input } as SpecificLearnRequest)).rejects.toThrow();
    options.authorize = vi.fn(async () => { throw new Error("Learning is Off"); });
    await expect(coordinator.start({ brainId: "brain", inputs: [{ kind: "web", url: "https://example.org/", quarantine: true }] })).rejects.toThrow(/Off/);
    expect(options.crawl).not.toHaveBeenCalled(); expect(await coordinator.list("brain")).toEqual([]);
  });
});
