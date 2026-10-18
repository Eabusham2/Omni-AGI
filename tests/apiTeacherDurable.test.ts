/** Fake provider + tiny SQLite custody only. No provider account, key, worker, model or training. */
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiTeacherTrainingService } from "../src/main/teacherTraining";
import { API_TEACHER_DATABASE, ApiTeacherStore, apiTeacherSha256,
  type ApiTeacherStoredItem, type ApiTeacherStoredJob } from "../src/main/apiTeacherStore";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
async function fixture(fetcher?: typeof fetch, admission: () => Promise<void> = async () => undefined) {
  const root = await mkdtemp(join(tmpdir(), "omni-api-teacher-")); roots.push(root);
  const key = "sk-private-teacher-fixture-1234567890";
  const secretValues = new Map<string, string>([["teacher:openai", key]]);
  const secrets = { get: vi.fn(async (id: string) => secretValues.get(id)),
    set: vi.fn(async (id: string, value: string) => { secretValues.set(id, value); }),
    remove: vi.fn(async (id: string) => { secretValues.delete(id); }), persistence: () => "encrypted" };
  const learn = vi.fn(async (brainId: string, experience: { content: string }) => ({ brain: { id: brainId }, source: {
    id: "source-1", name: "teacher", kind: "text", bytes: experience.content.length, learnedIdeas: 1,
    learnedConcepts: 1, learnedSynapses: 2, learnedParameterSteps: 1, parametersChanged: true,
    importedAt: new Date().toISOString(), rawTextRetained: false, contentHash: apiTeacherSha256(experience.content), policy: "pretrain"
  }, warnings: [] }));
  const admitWrite = vi.fn(admission);
  const service = new ApiTeacherTrainingService({ learnStructuredExperience: learn, preflightStart: vi.fn(async () => undefined) } as never,
    secrets as never, { brainRoot: async () => root, admitWrite }, fetcher ?? vi.fn(async () => new Response(JSON.stringify({ output_text: `safe answer ${key}` }),
      { status: 200, headers: { "content-type": "application/json" } })) as never);
  return { root, key, service, learn, admitWrite, secrets };
}
async function settled(service: ApiTeacherTrainingService, state: string) {
  await vi.waitFor(async () => { expect((await service.list("brain-a"))[0]?.state).toBe(state); }, { timeout: 2_000 });
  return (await service.list("brain-a"))[0]!;
}

describe("durable API teacher custody", () => {
  it("deduplicates a lost start ACK and redacts a known key before persistence and learning", async () => {
    const value = await fixture(), request = { brainId: "brain-a", requestId: "stable-submit-1", provider: "openai" as const,
      model: "model-a", prompts: [`teach without echoing ${value.key}`] };
    const first = await value.service.start(request); await settled(value.service, "complete");
    const retry = await value.service.start(request);
    expect(retry.id).toBe(first.id); expect(value.learn).toHaveBeenCalledOnce(); expect(value.admitWrite).toHaveBeenCalled();
    const stored = await readFile(join(value.root, API_TEACHER_DATABASE), "utf8");
    expect(stored).not.toContain(value.key); expect(value.learn.mock.calls[0]![1].content).not.toContain(value.key);
    const store = await ApiTeacherStore.open(value.root, "brain-a", async () => undefined, { create: false });
    try { const item = store!.items(first.id)[0]!; expect(item).toMatchObject({ state: "complete", rawResponseSha256: expect.stringMatching(/^[a-f0-9]{64}$/),
      learnedResponseSha256: expect.stringMatching(/^[a-f0-9]{64}$/), receipt: { brainId: "brain-a", deliveryId: item.deliveryId } }); }
    finally { store?.close(); }
  });

  it("never repeats an uncertain billed request until explicit review chooses retry", async () => {
    const failedFetch = vi.fn(async () => { throw new Error("connection lost after send"); });
    const first = await fixture(failedFetch as never), request = { brainId: "brain-a", requestId: "stable-submit-2",
      provider: "openai" as const, model: "model-a", prompts: ["one lesson"] };
    const job = await first.service.start(request); await settled(first.service, "cancelled");
    expect(failedFetch).toHaveBeenCalledOnce();
    expect(await first.service.details("brain-a", job.id)).toMatchObject({ state: "review-required", active: false,
      items: [{ index: 0, state: "uncertain-review", prompt: "one lesson" }] });
    const resumed = new ApiTeacherTrainingService({ learnStructuredExperience: first.learn,
      preflightStart: vi.fn(async () => undefined) } as never,
      { get: vi.fn(async (id: string) => id === "teacher:openai" ? first.key : undefined), set: vi.fn(), remove: vi.fn(), persistence: () => "encrypted" } as never,
      { brainRoot: async () => first.root, admitWrite: async () => undefined },
      vi.fn(async () => new Response(JSON.stringify({ output_text: "reviewed retry result" }), { status: 200 })) as never);
    await expect(resumed.resume("brain-a", job.id)).rejects.toThrow(/uncertain/);
    await resumed.resolveUncertain("brain-a", job.id, 0, "retry");
    await resumed.resume("brain-a", job.id); await settled(resumed, "complete");
    expect(failedFetch).toHaveBeenCalledOnce(); expect(first.learn).toHaveBeenCalledOnce();
  });

  it("requires explicit review before replaying an interrupted neural acknowledgement with the identical delivery payload", async () => {
    const value = await fixture(); let firstContent = "";
    value.learn.mockImplementationOnce(async (_brainId: string, experience: { content: string }) => {
      firstContent = experience.content; throw new Error("learning acknowledgement lost after possible commit");
    });
    const job = await value.service.start({ brainId: "brain-a", requestId: "stable-submit-3", provider: "openai",
      model: "model-a", prompts: ["stable learning delivery"] });
    await settled(value.service, "cancelled");
    await expect(value.service.resume("brain-a", job.id)).rejects.toThrow(/uncertain/);
    expect(await value.service.details("brain-a", job.id)).toMatchObject({ state: "review-required",
      items: [{ state: "learning-uncertain-review", experienceSha256: apiTeacherSha256(firstContent) }] });
    await value.service.resolveUncertain("brain-a", job.id, 0, "retry");
    await value.service.resume("brain-a", job.id); await settled(value.service, "complete");
    expect(value.learn).toHaveBeenCalledTimes(2);
    expect(value.learn.mock.calls[1]![1].content).toBe(firstContent);
    const store = await ApiTeacherStore.open(value.root, "brain-a", async () => undefined, { create: false });
    try { expect(store!.item(job.id, 0)).toMatchObject({ state: "complete", experienceSha256: apiTeacherSha256(firstContent),
      receipt: { contentHash: apiTeacherSha256(firstContent) } }); }
    finally { store?.close(); }
  });

  it("suspends and awaits an active provider attempt without replaying its uncertain effect", async () => {
    let dispatched!: () => void;
    const started = new Promise<void>(resolve => { dispatched = resolve; });
    const fetcher = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => new Promise<Response>((_resolve, reject) => {
      dispatched(); init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
    }));
    const value = await fixture(fetcher as never);
    const job = await value.service.start({ brainId: "brain-a", requestId: "stable-submit-4", provider: "openai",
      model: "model-a", prompts: ["suspend this attempt"] });
    await started; await value.service.suspend("brain-a");
    expect(fetcher).toHaveBeenCalledOnce();
    const store = await ApiTeacherStore.open(value.root, "brain-a", async () => undefined, { create: false });
    try {
      expect(store!.job(job.id).state).toBe("review-required");
      expect(store!.item(job.id, 0).state).toBe("uncertain-review");
    } finally { store?.close(); }
  });

  it("runs explicitly selected loopback Ollama without reading or forwarding an API key", async () => {
    const fetcher = vi.fn(async (_url: string | URL | Request, _init?: RequestInit) => new Response(JSON.stringify({
      choices: [{ message: { role: "assistant", content: "local teacher response" } }]
    }), { status: 200, headers: { "content-type": "application/json" } }));
    const value = await fixture(fetcher as never);
    await expect(value.service.start({ brainId: "brain-a", requestId: "stable-cloud-refusal", provider: "ollama",
      model: "remote:cloud", prompts: ["must remain local"] })).rejects.toThrow(/cloud model.*consent/i);
    expect(fetcher).not.toHaveBeenCalled();
    await value.service.start({ brainId: "brain-a", requestId: "stable-submit-5", provider: "ollama",
      model: "local-model", prompts: ["local lesson"] });
    await settled(value.service, "complete");
    const [url, init] = fetcher.mock.calls[0]!;
    expect(String(url)).toBe("http://127.0.0.1:11434/v1/chat/completions");
    expect((init?.headers as Record<string, string>).authorization).toBeUndefined();
    expect(value.learn).toHaveBeenCalledOnce();
  });

  it("refuses an unadmitted durable write before any provider dispatch", async () => {
    const fetcher = vi.fn(async () => new Response(JSON.stringify({ output_text: "must not arrive" }), { status: 200 }));
    const value = await fixture(fetcher as never, async () => { throw new Error("disk reserve denied"); });
    await expect(value.service.start({ brainId: "brain-a", requestId: "stable-submit-6", provider: "openai",
      model: "model-a", prompts: ["should not dispatch"] })).rejects.toThrow(/disk reserve denied/);
    expect(fetcher).not.toHaveBeenCalled(); expect(value.learn).not.toHaveBeenCalled();
  });

  it("recovers a crash-time neural delivery as review-required instead of replaying it", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-api-teacher-crash-")); roots.push(root);
    const store = await ApiTeacherStore.open(root, "brain-a", async () => undefined);
    if (!store) throw new Error("crash fixture store unavailable");
    const now = new Date().toISOString(), prompt = "crash boundary", response = "retained response",
      learnedResponseSha256 = apiTeacherSha256(response), experienceSha256 = apiTeacherSha256("stable delivery");
    const job: ApiTeacherStoredJob = { format: "omni-api-teacher-job", version: 1, id: "crash-job", brainId: "brain-a",
      originalBrainId: "brain-a", originalJobId: "crash-job", requestId: "crash-request", requestSha256: apiTeacherSha256("request"),
      provider: "openai", model: "model-a", maxOutputTokens: 32, promptCount: 1, learned: 0, failed: 0, skipped: 0,
      nextIndex: 0, state: "queued", revision: 1, createdAt: now, updatedAt: now };
    const item: ApiTeacherStoredItem = { format: "omni-api-teacher-item", version: 1, jobId: job.id, index: 0, prompt,
      promptSha256: apiTeacherSha256(prompt), deliveryId: "delivery-crash", state: "pending" };
    await store.createJob(job, [item]);
    await store.updateItem(job.id, 0, { state: "learning", attemptId: "attempt-crash", requestSha256: apiTeacherSha256("wire"),
      rawResponseSha256: apiTeacherSha256(response), learnedResponseSha256, response, experienceSha256 }, "learning-started");
    await store.updateJob(job.id, { state: "running" }, "running"); store.close();
    const recovered = await ApiTeacherStore.open(root, "brain-a", async () => undefined);
    try {
      await recovered!.recover();
      expect(recovered!.job(job.id).state).toBe("review-required");
      expect(recovered!.item(job.id, 0).state).toBe("learning-uncertain-review");
    } finally { recovered?.close(); }
  });

  it("retires and awaits an active provider request before replacing its device credential", async () => {
    let dispatched!: () => void;
    const started = new Promise<void>(resolve => { dispatched = resolve; });
    const fetcher = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => new Promise<Response>((_resolve, reject) => {
      dispatched(); init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
    }));
    const value = await fixture(fetcher as never);
    const job = await value.service.start({ brainId: "brain-a", requestId: "stable-submit-7", provider: "openai",
      model: "model-a", prompts: ["credential boundary"] });
    await started;
    await value.service.saveCredential({ provider: "openai", apiKey: "sk-replacement-fixture-1234567890" });
    expect(await value.service.details("brain-a", job.id)).toMatchObject({ state: "review-required", active: false,
      items: [{ state: "uncertain-review" }] });
    expect(value.learn).not.toHaveBeenCalled(); expect(fetcher).toHaveBeenCalledOnce();
    await value.service.close();
  });
});
