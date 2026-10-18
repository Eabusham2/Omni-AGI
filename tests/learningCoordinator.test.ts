import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LearningCoordinator, type LearningCoordinatorOptions } from "../src/main/learningCoordinator";
import { UserQuestionService } from "../src/main/userQuestions";
import { makeWebLearningPolicy, decideWebSource, publicationFromHtml } from "../src/main/webLearningPolicy";
import { admittedExperienceLearning } from "../src/main/admittedExperienceLearning";
import { factCheckNativeTask } from "../src/main/learningNativeTask";
import { LearningTaskStore } from "../src/main/learningTaskStore";
import type { DatasetManifest, TrainingCoverage, IngestResult } from "../src/shared/types";

const coverage = (id: string, processed = 3, complete = true): TrainingCoverage => ({ schemaVersion: 1, manifestId: id,
  discoveredFiles: 1, processedFiles: complete ? 1 : 0, rejectedFiles: 0, discoveredRecords: 3, processedRecords: processed,
  rejectedRecords: 0, discoveredBytes: 33, processedBytes: complete ? 33 : 11, shards: 1, modalityCounts: { text: 1 }, errors: [], complete, updatedAt: new Date().toISOString() });

describe("durable ordinary learning coordination without training a model", () => {
  let root: string, questions: UserQuestionService, coordinator: LearningCoordinator;
  let options: LearningCoordinatorOptions;
  let manifests: Map<string, DatasetManifest>;
  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "omni-learning-jobs-")); await mkdir(join(root, "brain"));
    questions = new UserQuestionService({ brainDirectory: id => join(root, id) });
    manifests = new Map();
    options = {
      brainDirectory: id => join(root, id), requireLoaded: vi.fn(), reserveWrite: vi.fn(async () => undefined), questions,
      manifest: async (_id, id) => { const row = manifests.get(id); if (!row) throw new Error("No selected manifest"); return row; },
      prepare: vi.fn(async (brainId, paths) => {
        const body = await readFile(paths[0]!, "utf8");
        const id = `manifest-${manifests.size}`;
        const row = { schemaVersion: 1, id, brainId, roots: paths, entryFile: "entries.ndjson", discoveredFiles: 1, discoveredBytes: Buffer.byteLength(body),
          manifestHash: createHash("sha256").update(body).digest("hex") } as DatasetManifest;
        manifests.set(id, row); return row;
      }),
      ingest: vi.fn(async (_id, manifestId) => ({ coverage: coverage(manifestId), paused: false })),
      crawl: vi.fn(async () => { throw new Error("No web fixture configured"); })
    };
    coordinator = new LearningCoordinator(options);
  });
  afterEach(async () => { await coordinator.close(); questions.close(); await rm(root, { recursive: true, force: true }); });
  const settle = async (id: string, state: string) => { await vi.waitFor(async () => expect((await coordinator.get("brain", id)).state).toBe(state)); return coordinator.get("brain", id); };

  it("returns queued work, learns the same persisted manifest once, and survives idempotent retry/restart", async () => {
    const request = { brainId: "brain", requestId: "reading-one", inputs: [{ kind: "text" as const, text: "A whole chapter 🌱.\nA second line." }] };
    const started = await coordinator.start(request); expect(started.state).toBe("queued"); expect(started.traversalComplete).toBe(false);
    const complete = await settle(started.id, "complete"); expect(complete.sources[0]?.coverage?.processedRecords).toBe(3);
    expect(options.prepare).toHaveBeenCalledTimes(1); expect(options.ingest).toHaveBeenCalledTimes(1);
    expect((options.ingest as ReturnType<typeof vi.fn>).mock.calls[0]?.[3].jobId.length).toBe(64);
    await coordinator.close(); coordinator = new LearningCoordinator(options);
    expect((await coordinator.start(request)).state).toBe("complete");
    expect(options.ingest).toHaveBeenCalledTimes(1);
    await expect(coordinator.start({ ...request, inputs: [{ kind: "text", text: "changed" }] })).rejects.toThrow(/different contents/);
  });

  it("resumes a paused record cursor without manufacturing another manifest or skipping its suffix", async () => {
    let calls = 0;
    options.ingest = vi.fn(async (_id, manifestId, _epochs, control) => {
      calls++;
      if (calls === 1) { await new Promise<void>(resolve => control.signal.addEventListener("abort", () => resolve(), { once: true })); return { coverage: coverage(manifestId, 1, false), paused: true }; }
      return { coverage: coverage(manifestId), paused: false };
    });
    const task = await coordinator.start({ brainId: "brain", inputs: [{ kind: "text", text: "three records" }] });
    await vi.waitFor(() => expect(options.ingest).toHaveBeenCalledTimes(1));
    const paused = await coordinator.pause("brain", task.id); expect(paused.sources[0]?.coverage?.processedRecords).toBe(1);
    const id = paused.sources[0]!.manifestId;
    await coordinator.close(); coordinator = new LearningCoordinator(options); await coordinator.recover("brain");
    await coordinator.resume("brain", task.id); const done = await settle(task.id, "complete");
    expect(done.sources[0]?.manifestId).toBe(id); expect(options.prepare).toHaveBeenCalledTimes(1);
    expect((options.ingest as ReturnType<typeof vi.fn>).mock.calls.map(call => call[1])).toEqual([id, id]);
  });

  it("keeps jobs serialized per brain and unload waits for the source boundary without aborting it", async () => {
    let release!: () => void, signal!: AbortSignal;
    options.ingest = vi.fn(async (_id, id, _epochs, control) => { signal = control.signal; await new Promise<void>(resolve => { release = resolve; }); return { coverage: coverage(id), paused: false }; });
    const first = await coordinator.start({ brainId: "brain", inputs: [{ kind: "text", text: "first" }] });
    await vi.waitFor(() => expect(options.ingest).toHaveBeenCalledTimes(1));
    const second = await coordinator.start({ brainId: "brain", inputs: [{ kind: "text", text: "second" }] });
    expect((await coordinator.get("brain", second.id)).state).toBe("queued");
    const unload = coordinator.pauseAtBoundary("brain"); expect(signal.aborted).toBe(false);
    release(); await unload;
    expect((await coordinator.get("brain", first.id)).state).toBe("paused");
    expect((await coordinator.get("brain", second.id)).state).toBe("paused"); expect(options.ingest).toHaveBeenCalledTimes(1);
  });

  it("rejects empty and path-forged learning instead of reporting a completed no-op", async () => {
    await expect(coordinator.start({ brainId: "brain", inputs: [] })).rejects.toThrow(/Select text/);
    await expect(coordinator.start({ brainId: "brain", inputs: [{ kind: "dataset", path: "/private/file" }] } as never)).rejects.toThrow(/raw file paths/);
    expect(options.ingest).not.toHaveBeenCalled();
  });
  it("honors current instance Off policy for explicit UI starts and resumes", async () => {
    options.authorize = vi.fn(async () => { throw new Error("Learning disabled"); });
    await expect(coordinator.start({ brainId: "brain", inputs: [{ kind: "text", text: "not authorized" }] })).rejects.toThrow(/disabled/);
    expect(options.prepare).not.toHaveBeenCalled(); expect(await coordinator.list("brain")).toEqual([]);
    options.authorize = undefined;
    const task = await coordinator.start({ brainId: "brain", inputs: [{ kind: "text", text: "authorized" }] }); await settle(task.id, "complete");
    options.authorize = vi.fn(async () => { throw new Error("Learning disabled"); });
    await expect(coordinator.resume("brain", task.id)).rejects.toThrow(/disabled/); expect(options.ingest).toHaveBeenCalledTimes(1);
  });

  it("unloads a parked native task without cancelling or awaiting its Load-dependent turn", async () => {
    let finish!: () => void, signal!: AbortSignal;
    options.followInstructions = vi.fn(async (_task, value) => { signal = value; await new Promise<void>(resolve => { finish = resolve; }); return { turnId: "retained-turn" }; });
    const task = await coordinator.start({ brainId: "brain", inputs: [{ kind: "text", text: "selected chapter" }], instructions: "Inspect the examples." });
    await vi.waitFor(() => expect(options.followInstructions).toHaveBeenCalledTimes(1));
    await coordinator.bindNativeTurn("brain", task.id, "retained-turn", undefined, "a".repeat(64));
    await coordinator.pauseAtBoundary("brain");
    expect(signal.aborted).toBe(false);
    expect(await coordinator.get("brain", task.id)).toMatchObject({ state: "paused", pausedForUnload: true, retainedNativeTurn: { turnId: "retained-turn" } });
    finish(); await new Promise(resolve => setTimeout(resolve, 10));
    expect((await coordinator.get("brain", task.id)).state).toBe("paused");
    await coordinator.resumeAfterLoad("brain"); await settle(task.id, "complete");
    expect(options.followInstructions).toHaveBeenCalledTimes(1); expect(signal.aborted).toBe(false);
    expect((await coordinator.get("brain", task.id)).retainedNativeTurn).toBeUndefined();
  });

  it("closes a durably suspended native task without converting Unload into Cancel", async () => {
    let finish!: () => void, signal!: AbortSignal;
    options.followInstructions = vi.fn(async (_task, value) => { signal = value; await new Promise<void>(resolve => { finish = resolve; }); return { turnId: "retained-turn" }; });
    const task = await coordinator.start({ brainId: "brain", inputs: [{ kind: "text", text: "chapter" }], instructions: "Inspect it." });
    await vi.waitFor(() => expect(options.followInstructions).toHaveBeenCalledTimes(1));
    await coordinator.pauseAtBoundary("brain"); await coordinator.close(); expect(signal.aborted).toBe(false);
    coordinator = new LearningCoordinator(options); await coordinator.recover("brain");
    expect(await coordinator.get("brain", task.id)).toMatchObject({ state: "paused", pausedForUnload: true });
    finish(); await new Promise(resolve => setTimeout(resolve, 10));
    expect((await coordinator.get("brain", task.id)).state).toBe("paused");
  });
  it("detaches a completed native reply parked at the unload gate without waiting forever", async () => {
    let finish!: () => void, signal!: AbortSignal;
    options.followInstructions = vi.fn(async (_task, value) => { signal = value; await new Promise<void>(resolve => { finish = resolve; }); return { turnId: "retained-turn" }; });
    const task = await coordinator.start({ brainId: "brain", inputs: [{ kind: "text", text: "chapter" }], instructions: "Inspect it." });
    await vi.waitFor(() => expect(options.followInstructions).toHaveBeenCalledTimes(1));
    await coordinator.bindNativeTurn("brain", task.id, "retained-turn", undefined, "a".repeat(64));
    await coordinator.pauseAtBoundary("brain"); finish(); await new Promise(resolve => setTimeout(resolve, 5));
    await coordinator.detachAndForget("brain"); expect(signal.aborted).toBe(false);
    expect(await coordinator.get("brain", task.id)).toMatchObject({ state: "paused", phase: "host-detached", retainedNativeTurn: { turnId: "retained-turn" } });
  });
  it("direct native Resume activates only its bound paused task and commits the real terminal receipt once", async () => {
    const input = "the exact previously admitted task", turnId = "root-inherited-turn", policy = makeWebLearningPolicy({}, "Read the chapter", 1);
    const store = await LearningTaskStore.open(join(root, "brain"), "brain");
    store.create({ format: "omni-learning-task", version: 1, id: "copied-task", brainId: "brain", requestId: "copied-request", requestSha256: "b".repeat(64), kind: "specific-learn", state: "paused",
      createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), revision: 1, policy, sources: [], epochs: 1, phase: "copied-instance", label: "Paused", pendingQuestions: [], traversalComplete: false,
      retainedNativeTurn: { turnId, inputSha256: createHash("sha256").update(input).digest("hex"), policySha256: policy.sha256 } }, []);
    store.bindNativeTurn("copied-task", turnId); store.close();
    const attachment = { brainId: "brain", input, publishedText: "", priorActions: [], hostAttached: false,
      frontier: { checkpointId: "checkpoint", turnId, inputSha256: createHash("sha256").update(input).digest("hex"), lastStreamSequence: 0,
        sampledTokens: 0, publishedTokens: 0, phase: "native-text-token-boundary" as const, exactSupportedResume: true as const } };
    const run = vi.fn(async () => {
      expect((await coordinator.get("brain", "copied-task")).state).toBe("running");
      await expect((await coordinator.executionContext("brain", turnId))!.policy()).resolves.toEqual(policy);
      return { humanMessage: { turnId, content: input, status: "complete" }, brainMessage: { turnId, status: "complete", traceId: "trace", createdAt: new Date().toISOString() }, trace: { id: "trace" } } as never;
    });
    await coordinator.resumeNativeContinuation("brain", attachment, run);
    expect(await coordinator.get("brain", "copied-task")).toMatchObject({ state: "complete", instructionReceipt: { turnId, policySha256: policy.sha256 } });
    await expect(coordinator.resumeNativeContinuation("brain", attachment, run)).rejects.toThrow(/different or already-active/); expect(run).toHaveBeenCalledTimes(1);
  });

  it("keeps independent claims moving, retains contradictory proposals, and learns only a user-accepted correction", async () => {
    options.crawl = vi.fn(async (request, _control, policy) => {
      const provenance = decideWebSource(request.url, new URL(request.url), "text/html", publicationFromHtml(""), await policy.policy());
      const result = { source: { id: `source-${request.url.endsWith("a") ? "a" : "b"}`, contentHash: createHash("sha256").update(request.url).digest("hex") } } as IngestResult;
      await policy.onSource?.(provenance, result);
      return { crawlId: request.crawlId!, startUrl: request.url, visited: 1, skipped: 0, results: [result], warnings: [], stopped: false, coverage: coverage(request.crawlId!) } as never;
    });
    options.assess = vi.fn(async (_workspace, claim) => ({ conclusion: "mixed" as const, explanation: "The source is inconclusive.", evidenceIds: [claim.evidence[0]!.id],
      ...(claim.text === "first claim" ? { question: { mode: "text" as const, prompt: "Which version?" } } : { proposedCorrection: "A source-scoped correction, not a universal fact." }) }));
    const started = await coordinator.startFactCheck({ brainId: "brain", claims: [{ text: "first claim", sources: ["https://example.org/a"] }, { text: "second claim", sources: ["https://example.org/b"] }] });
    await settle(started.task.id, "waiting"); const workspace = await coordinator.factCheck("brain", started.task.id);
    expect(workspace.claims.map(claim => claim.state)).toEqual(["waiting-answer", "reviewed"]); expect(options.assess).toHaveBeenCalledTimes(2);
    const second = workspace.claims[1]!;
    await coordinator.review({ brainId: "brain", taskId: workspace.task.id, claimId: second.id, requestId: "alternate-review", conclusion: "contradicted", explanation: "A different reading of this source.", evidenceIds: [second.evidence[0]!.id] });
    expect((await coordinator.factCheck("brain", workspace.task.id)).claims[1]!.findings).toHaveLength(2);
    const correction = await coordinator.acceptCorrection("brain", workspace.task.id, second.id, second.findings[0]!.id); await settle(correction.id, "complete");
    expect(options.ingest).toHaveBeenCalledTimes(1);
    const nativeInput = factCheckNativeTask(workspace, second);
    expect(JSON.parse(nativeInput.input)).toMatchObject({ kind: "user-requested-fact-check", claimId: second.id, capturedEvidence: second.evidence });
    expect(nativeInput.input).not.toContain("hiddenBehavioralPrompt");
  });

  it("persists explicit steering without replacing committed receipts", async () => {
    options.assess = vi.fn(async () => undefined);
    const { task } = await coordinator.startFactCheck({ brainId: "brain", claims: [{ text: "An unresolved statement" }] });
    await settle(task.id, "waiting");
    const changed = await coordinator.steer({ brainId: "brain", taskId: task.id, expectedRevision: 1, instructions: "Use only the named source.", constraints: { allowedSites: [{ domain: "Example.org" }] } });
    expect(changed.policy).toMatchObject({ revision: 2, instructions: "Use only the named source.", constraints: { allowedSites: [{ domain: "example.org", includeSubdomains: false }] } });
    await expect(coordinator.steer({ brainId: "brain", taskId: task.id, expectedRevision: 1, instructions: "stale" })).rejects.toThrow(/refresh/);
    await coordinator.close(); coordinator = new LearningCoordinator(options);
    expect((await coordinator.get("brain", task.id)).policy.sha256).toBe(changed.policy.sha256);
  });
});

describe("current-experience native replay binding", () => {
  const brainId = "brain", turnId = "controller-turn", inputSha256 = "a".repeat(64), humanMessageId = "human-message";
  const jobId = createHash("sha256").update(`${brainId}\0${turnId}\0${inputSha256}\0chat-slow-learning-v1`).digest("hex");
  const admission = { turnId, inputSha256, humanMessageId, committed: true };
  it("prioritizes the existing actual job and distinguishes retained activity from slow completion", () => {
    expect(admittedExperienceLearning({}, brainId, turnId)).toMatchObject({ state: "unresolved", admitted: false });
    expect(admittedExperienceLearning({ accepted_chat_inputs: [admission] }, brainId, turnId)).toMatchObject({ state: "retained", admitted: true });
    expect(admittedExperienceLearning({ accepted_chat_inputs: [admission], pending_chat_slow_learning: [{ jobId, turnId, inputSha256, humanMessageId,
      completedReplayPasses: 0, replayCursor: { phase: "text", character: 4, byte: 1 } }] }, brainId, turnId))
      .toMatchObject({ state: "queued", jobId, replayCursor: { phase: "text", character: 4, byte: 1 } });
    expect(admittedExperienceLearning({ accepted_chat_inputs: [admission], completed_chat_slow_learning: [jobId] }, brainId, turnId)).toMatchObject({ state: "complete", jobId });
    expect(admittedExperienceLearning({ accepted_chat_inputs: [admission], pending_chat_slow_learning: [{ jobId, turnId: "invented", inputSha256, humanMessageId }] }, brainId, turnId).state).toBe("retained");
  });
});
