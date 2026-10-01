import { randomUUID } from "node:crypto";
import { mkdtemp, mkdir, readFile, rm, unlink, link, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CompletedActionEvidenceStore, actionEvidenceFileHash, actionEvidenceJsonChunks,
  savedActionEvidenceFiles, rekeyActionEvidenceJobs, validateActionEvidenceReceipt } from "../src/main/completedActionEvidence";
import { BrainService } from "../src/main/brainService";
import type { ActionEvent } from "../src/shared/types";

vi.mock("node:os", async importActual => ({ ...await importActual<typeof import("node:os")>(), freemem: () => 8 * 1024 ** 3 }));

const time = "2026-09-30T12:00:00.000Z";
const roots: string[] = [];
afterEach(async () => { vi.useRealTimers(); await Promise.all(roots.splice(0).map(path => rm(path, { recursive: true, force: true }))); });
async function setup() {
  const root = await mkdtemp(join(tmpdir(), "omni-completed-evidence-")); roots.push(root);
  const engine = join(root, "brain", "engine"); await mkdir(engine, { recursive: true });
  const admit = vi.fn(async (_bytes: number, _directory: string, _writing?: boolean) => undefined);
  const store = new CompletedActionEvidenceStore(engine, "brain", admit);
  return { root, engine, store, admit };
}
function actual(output: unknown): ActionEvent {
  return { id: randomUUID(), brainId: "brain", neuralActionId: "a".repeat(32), state: "complete", createdAt: time, updatedAt: time,
    action: { kind: "tool", source: "brain", toolId: "web.fetch", action: "fetch", arguments: { url: "https://example.org" } },
    execution: { id: randomUUID(), toolId: "web.fetch", action: "fetch", state: "complete", dispatchStarted: true, startedAt: time, finishedAt: time, output } };
}
const route = (event: ActionEvent) => ({ eventId: event.id, utterance: "actual current human request", toolId: "web.fetch", action: "fetch", arguments: { url: "https://example.org" } });

describe("exact independent completed-outcome spool", () => {
  it("streams complete Unicode result bytes beyond the old display slice without one full-result stringify", async () => {
    const { store, admit } = await setup();
    const output = { body: `${"actual 文🧠\n".repeat(16_000)}EXACT-END`, controls: "\0\t\"\\", lone: "\uD800", scalar: 7 };
    const event = actual(output), job = await store.stage(event, output, route(event), "turn");
    const record = JSON.parse(await readFile(store.evidencePath(job.evidenceId), "utf8"));
    expect(record.output).toEqual(output);
    expect(record).toMatchObject({ executionId: event.execution!.id, actionEventId: event.id, chatTurnId: "turn", executionState: "complete" });
    expect(await actionEvidenceFileHash(store.evidencePath(job.evidenceId))).toBe(job.evidenceSha256);
    expect(Math.max(...admit.mock.calls.map(call => Number(call[0])))).toBeLessThan(300_000);
    const chunks = [...actionEvidenceJsonChunks(output)];
    expect(Math.max(...chunks.map(part => part.length))).toBeLessThan(66_000);
    expect(JSON.parse(chunks.join(""))).toEqual(output);
  });
  it("refuses implicit conversion, cyclic objects and accessors without executing them", () => {
    const getter = vi.fn(() => "not observed");
    const value = Object.defineProperty({}, "output", { enumerable: true, get: getter });
    expect(() => [...actionEvidenceJsonChunks(value)]).toThrow("accessors");
    expect(getter).not.toHaveBeenCalled();
    expect(() => [...actionEvidenceJsonChunks(new Date())]).toThrow("implicit");
    const cycle: Record<string, unknown> = {}; cycle.self = cycle;
    expect(() => [...actionEvidenceJsonChunks(cycle)]).toThrow("noncyclic");
  });
  it("retains actual failed dispatch outcome and error without a positive route label; denial is not an execution", async () => {
    const { store } = await setup(), event = actual({ exitCode: 7, stderr: "actual fixture failure" });
    event.state = "failed"; event.execution = { ...event.execution!, state: "failed", error: "nonzero actual exit" };
    const job = await store.stage(event, event.execution.output, route(event));
    expect(job.route).toBeUndefined(); expect(job.routeComplete).toBe(true);
    expect(JSON.parse(await readFile(store.evidencePath(job.evidenceId), "utf8"))).toMatchObject({ executionState: "failed", executionError: "nonzero actual exit", output: { exitCode: 7 } });
    event.execution = { ...event.execution, id: randomUUID(), dispatchStarted: undefined };
    await expect(store.stage(event, event.execution.output)).rejects.toThrow("Only an actual");
  });
  it("captures one immutable execution identity and reconstructs its pending queue after a manifest publication gap", async () => {
    const { store } = await setup(), event = actual({ observed: "exact" });
    event.action.arguments = { nested: { outputPresent: "not the outcome boundary" }, exact: 'string ,"outputPresent": false' };
    const first = await store.stage(event, event.execution!.output, route(event), "turn");
    expect(await store.stage(event, event.execution!.output, route(event), "turn")).toEqual(first);
    await unlink(store.jobPath(first.evidenceId));
    const recovered = await store.pending();
    expect(recovered).toHaveLength(1);
    expect(recovered[0]).toMatchObject({ evidenceId: first.evidenceId, evidenceSha256: first.evidenceSha256, route: route(event), state: "pending" });
    expect(await store.pendingCount()).toBe(1);
  });
  it("rekeys only queue ownership, leaves actual evidence unchanged and breaks any shared manifest link", async () => {
    const { root, engine, store } = await setup(), event = actual({ observed: "ancestral actual result" });
    const job = await store.stage(event, event.execution!.output);
    const originalBytes = await readFile(store.evidencePath(job.evidenceId));
    const fork = join(root, "fork", "engine"); await mkdir(join(fork, "action-result-learning", "jobs"), { recursive: true });
    await mkdir(join(fork, "action-result-learning", "evidence"));
    await link(store.jobPath(job.evidenceId), join(fork, "action-result-learning", "jobs", `${job.evidenceId}.json`));
    await link(store.evidencePath(job.evidenceId), join(fork, "action-result-learning", "evidence", `${job.evidenceId}.jsonl`));
    await rekeyActionEvidenceJobs(fork, "fork");
    expect((await store.read(job.evidenceId))!.brainId).toBe("brain");
    expect(JSON.parse(await readFile(join(fork, "action-result-learning", "jobs", `${job.evidenceId}.json`), "utf8")).brainId).toBe("fork");
    expect(await readFile(join(fork, "action-result-learning", "evidence", `${job.evidenceId}.jsonl`))).toEqual(originalBytes);
    expect((await savedActionEvidenceFiles(engine)).size).toBe(2);
  });
  it("detects changed saved outcome bytes and does not resurrect a deleted owner", async () => {
    const { root, engine, store } = await setup(), event = actual("actual");
    const job = await store.stage(event, "actual");
    await writeFile(store.evidencePath(job.evidenceId), "changed\n");
    await expect(savedActionEvidenceFiles(engine)).rejects.toThrow("binding");
    const absent = new CompletedActionEvidenceStore(join(root, "missing", "engine"), "brain", async () => undefined);
    await expect(absent.stage(event, "actual")).rejects.toThrow();
  });
  it("does not acknowledge a queued/noncommitted/sibling neural receipt as learned", async () => {
    const { store } = await setup(), event = actual({ actual: true }), job = await store.stage(event, event.execution!.output);
    const receipt = { brainId: "brain", evidenceId: job.evidenceId, executionId: job.executionId, evidenceSha256: job.evidenceSha256,
      committed: true as const, processed: true, duplicate: false };
    expect(validateActionEvidenceReceipt(receipt, job)).toEqual(receipt);
    expect(() => validateActionEvidenceReceipt({ ...receipt, brainId: "sibling" }, job)).toThrow("owner");
    expect(() => validateActionEvidenceReceipt({ ...receipt, committed: false }, job)).toThrow("committed");
  });
  it("pauses a refused physical write without acknowledging or publishing partial evidence", async () => {
    const { engine } = await setup(), event = actual({ actual: "keep visible, not truncated" });
    const store = new CompletedActionEvidenceStore(engine, "brain", async () => { throw new Error("typed physical allocation pause"); });
    await expect(store.stage(event, event.execution!.output)).rejects.toThrow("physical allocation");
    expect(await savedActionEvidenceFiles(engine)).toEqual(new Map());
  });
});

function methodService(store: CompletedActionEvidenceStore, request: ReturnType<typeof vi.fn>) {
  const value = Object.assign(Object.create(BrainService.prototype), {
    repository: { get: async () => ({ config: { onlineLearning: true } }), brainDirectory: () => store.enginePath.slice(0, -7) },
    engine: { request, cancelBackgroundRequest: vi.fn(() => false) },
    actionEvidenceStores: new Map([["brain", store]]), actionEvidenceDrains: new Map(), actionEvidenceRetries: new Map(),
    pausedChatLearning: new Set(), backgroundLearningSuspendedForLaunch: false,
    preflightStart: vi.fn(async () => undefined), learnToolRouteOutcome: vi.fn(async () => undefined),
    chatSlowLearningTimers: new Map(), resumePendingChatLearning: vi.fn()
  }) as BrainService;
  const privateState = value as unknown as { actionEvidenceDrains: Map<string, Promise<void>> };
  return { value, done: () => privateState.actionEvidenceDrains.get("brain") ?? Promise.resolve() };
}
describe("constructor-free durable host learning drain", () => {
  it("dormant replay enumeration cannot select or load a brain indirectly", () => {
    const get = vi.fn(), request = vi.fn();
    const service = Object.assign(Object.create(BrainService.prototype), {
      completedActionLearningOwner: "active", repository: { get }, engine: { request }
    }) as BrainService;
    service.resumePendingChatLearning("dormant-copy");
    expect(get).not.toHaveBeenCalled(); expect(request).not.toHaveBeenCalled();
  });
  it("retries a lost native ACK by the exact evidence identity without an interrupted signal or any effects replay", async () => {
    vi.useFakeTimers();
    const { store } = await setup(), event = actual({ actual: "completed once" });
    let committed = false;
    const request = vi.fn(async (_method: string, params: Record<string, unknown>, _deadline: number, signal?: AbortSignal, priority?: string) => {
      expect(signal).toBeUndefined(); expect(priority).toBe("background");
      if (!committed) { committed = true; throw new Error("native committed; transport ACK lost"); }
      return { brainId: "brain", evidenceId: params.evidenceId, executionId: params.executionId,
        evidenceSha256: params.evidenceSha256, committed: true, processed: false, duplicate: true };
    });
    const service = methodService(store, request);
    const job = await service.value.queueCompletedActionEvidence("brain", event, event.execution!.output);
    service.value.selectCompletedActionLearningOwner("brain"); await service.done();
    expect((await store.read(job.evidenceId))!.state).toBe("pending");
    await vi.advanceTimersByTimeAsync(15_001); await service.done();
    expect(request).toHaveBeenCalledTimes(2);
    expect(request.mock.calls.map(call => call[1].evidenceId)).toEqual([job.evidenceId, job.evidenceId]);
    expect((await store.read(job.evidenceId))!.state).toBe("complete");
  });
  it("keeps dormant identity queues read-only until explicit selection and never invokes the tool executor", async () => {
    const { store } = await setup(), event = actual({ actual: true });
    const request = vi.fn(async (_method: string, params: Record<string, unknown>) => ({ brainId: "brain", evidenceId: params.evidenceId,
      executionId: params.executionId, evidenceSha256: params.evidenceSha256, committed: true, processed: true, duplicate: false }));
    const service = methodService(store, request);
    await service.value.queueCompletedActionEvidence("brain", event, event.execution!.output);
    service.value.resumeCompletedActionLearning("brain"); await service.done();
    expect(request).not.toHaveBeenCalled(); expect(await store.pendingCount()).toBe(1);
    service.value.selectCompletedActionLearningOwner("brain"); await service.done();
    expect(request).toHaveBeenCalledOnce(); expect(await store.pendingCount()).toBe(0);
  });
});
