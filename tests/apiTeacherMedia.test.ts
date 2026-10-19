/** Tiny file/SQLite/fake-provider contracts only. No account, worker, model, decoder or training. */
import { mkdtemp, mkdir, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises";
import { on } from "node:events";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { apiTeacherMediaRequest, apiTeacherMediaParts } from "../src/main/apiTeacherMediaProviders";
import { apiTeacherMediaFiles, storeApiTeacherMediaLesson, verifyApiTeacherMediaFile, storeApiTeacherResponseEnvelope,
  readApiTeacherResponseEnvelope } from "../src/main/apiTeacherMedia";
import { ApiTeacherTrainingService } from "../src/main/teacherTraining";
import { ApiTeacherStore, apiTeacherSha256 } from "../src/main/apiTeacherStore";
import { captureApiTeacherWorkspace, rekeyApiTeacherWorkspace, validateApiTeacherWorkspace, mergeApiTeacherWorkspace } from "../src/main/apiTeacherPortability";
import { snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { validateApiTeacherMediaArchiveSources } from "../src/main/apiTeacherMediaArchive";
import { BrainService } from "../src/main/brainService";
import type { ApiTeacherMediaLesson, ApiTeacherMediaKind, ApiTeacherMediaCoverage } from "../src/shared/apiTeacherMedia";
import type { RuntimeJobEvent } from "../src/shared/types";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
async function temporary() { const root = await mkdtemp(join(tmpdir(), "omni-teacher-media-")); roots.push(root); return root; }
const bytes = {
  image: Buffer.from([137,80,78,71,13,10,26,10,0,0,0,0]),
  audio: Buffer.from("RIFF0000WAVEfmt fixture"),
  video: Buffer.from("0000ftypisomfixture")
};
const mime = { image: "image/png", audio: "audio/wav", video: "video/mp4" };
const body = () => ({ status: "completed", steps: [{ type: "model_output", content: [
  { type: "text", text: "A short aligned lesson." },
  ...(["image", "audio", "video"] as const).map(type => ({ type, data: bytes[type].toString("base64"), mime_type: mime[type] }))
] }] });
const input = { model: "user-selected-model", prompt: "A bird speaks, then moves.", maxOutputTokens: 128, credential: "fixture-device-key-keep-private" };
function coverage(lesson: ApiTeacherMediaLesson): ApiTeacherMediaCoverage {
  const byModality: ApiTeacherMediaCoverage["byModality"] = {};
  for (const part of lesson.parts) if (part.type !== "text") {
    const old = byModality[part.type]; byModality[part.type] = { records: (old?.records ?? 0) + 1,
      windows: (old?.windows ?? 0) + 1, processedUnits: (old?.processedUnits ?? 0) + 1, complete: true };
  }
  return { complete: true, manifestSha256: lesson.manifestSha256, byModality };
}
function learned(brainId: string, lesson: ApiTeacherMediaLesson) {
  return { brain: { id: brainId }, source: { id: "source-media", name: "teacher", kind: "json", bytes: 100,
    contentHash: lesson.manifestSha256, learnedIdeas: 1, learnedConcepts: 1, learnedSynapses: 2, learnedParameterSteps: 1,
    parametersChanged: true, importedAt: new Date().toISOString(), rawTextRetained: false, policy: "pretrain" },
    warnings: [], mediaCoverage: coverage(lesson) };
}
async function fixture() {
  const root = await temporary(), fetcher = vi.fn(async () => new Response(JSON.stringify(body()), { status: 200 }));
  const learn = vi.fn(async (brainId: string, experience: { lesson: ApiTeacherMediaLesson }) => learned(brainId, experience.lesson));
  const text = vi.fn(async () => { throw new Error("Media lesson must not take the text-only path"); });
  const admitMetadata = vi.fn(async (_id: string, _bytes: number, _purpose: string) => undefined);
  const service = new ApiTeacherTrainingService({ preflightStart: vi.fn(async () => undefined),
    learnStructuredExperience: text, learnTeacherMediaExperience: learn } as never,
  { get: async (id: string) => id === "teacher:gemini" ? input.credential : undefined, persistence: () => "encrypted" } as never,
  { brainRoot: async () => root, admitWrite: async () => undefined, admitMetadata }, fetcher as never);
  const request = { brainId: "brain-a", requestId: "stable-media-submit", provider: "gemini" as const, model: input.model,
    prompts: [input.prompt], responseModalities: ["text", "image", "audio", "video"] as Array<"text"|ApiTeacherMediaKind> };
  return { root, fetcher, learn, text, service, request, admitMetadata };
}
async function waitState(f: Awaited<ReturnType<typeof fixture>>, state: string) {
  await vi.waitFor(async () => expect((await f.service.details("brain-a", (await f.service.list("brain-a"))[0]!.id)).state).toBe(state), { timeout: 3000 });
  return f.service.details("brain-a", (await f.service.list("brain-a"))[0]!.id);
}

describe("typed API-teacher media custody", () => {
  it("uses documented provider output shapes and refuses unsupported modalities without dispatch", () => {
    expect(apiTeacherMediaRequest("openai", input, ["image"]).body.tools).toEqual([{ type: "image_generation", output_format: "png" }]);
    expect(apiTeacherMediaRequest("gemini", input, ["audio", "video"]).body.response_format).toEqual([{ type: "audio", mime_type: "audio/wav" }, { type: "video", delivery: "inline" }]);
    expect(apiTeacherMediaRequest("openrouter", input, ["image"]).url).toBe("https://openrouter.ai/api/v1/images");
    for (const provider of ["anthropic", "ollama", "openai", "openrouter"] as const)
      expect(() => apiTeacherMediaRequest(provider, input, ["video"])).toThrow(/no documented adapter/);
    expect(apiTeacherMediaParts("openai", { status: "completed", output: [{ type: "image_generation_call", status: "completed", result: bytes.image.toString("base64") }] }, ["image"])[0]?.type).toBe("image");
    expect(() => apiTeacherMediaParts("gemini", { status: "completed", steps: [{ type: "model_output", content: [{ type: "text", text: "video/mp4 of a bird" }] }] }, ["video"])).toThrow(/no actual video/);
    expect(() => apiTeacherMediaParts("gemini", { status: "completed", steps: [{ type: "model_output", content: [{ type: "video", uri: "https://host/video" }] }] }, ["video"])).toThrow(/inline typed bytes/);
  });
  it("persists ordered real image/audio/video bytes, typed hashes and a native media-pair manifest", async () => {
    const root = await temporary(), parts = apiTeacherMediaParts("gemini", body(), ["image", "audio", "video"]);
    const lesson = await storeApiTeacherMediaLesson(root, { provider: "gemini", sourceEndpoint: "https://generativelanguage.googleapis.com/v1beta/interactions",
      model: input.model, prompt: input.prompt, deliveryId: "delivery-fixture", responseModalities: ["image", "audio", "video"], parts });
    expect(lesson.parts.map(part => part.type)).toEqual(["text", "image", "audio", "video"]);
    for (const [path, expected] of apiTeacherMediaFiles(lesson, input.prompt)) await verifyApiTeacherMediaFile(root, path, expected);
    const manifest = await readFile(join(root, lesson.manifestPath), "utf8");
    const rows = manifest.trim().split("\n").map(row => JSON.parse(row));
    expect(rows.slice(2).map(row => row.kind)).toEqual(["image", "audio", "video"]);
    expect(rows[2]).toMatchObject({ format: "omni-teacher-media-pair-1", mimeType: "image/png", mediaSha256: apiTeacherSha256(bytes.image) });
    expect(rows[2].conditioningText).toContain(input.prompt); expect(rows[2].conditioningText).toContain("aligned lesson");
  });
  it("refuses known secret and signed/URI-bearing envelopes before durable publication", async () => {
    const root = await temporary();
    for (const text of [JSON.stringify({ text: input.credential }), JSON.stringify({ uri: "https://host/?token=signed" }),
      '{"text":"https:\\/\\/host/?X-Goog-Signature=secret"}'])
      await expect(storeApiTeacherResponseEnvelope(root, new Response(text), [input.credential], {})).rejects.toThrow(/credential|URL\/auth/);
    const envelope = await storeApiTeacherResponseEnvelope(root, new Response(JSON.stringify(body())), [input.credential], {});
    expect(await readApiTeacherResponseEnvelope(root, envelope, {})).toEqual(body());
  });
  it("drives the existing teacher job into the real media consumer seam and records actual coverage, not text labels", async () => {
    const f = await fixture(); const job = await f.service.start(f.request), details = await waitState(f, "complete");
    expect(f.fetcher).toHaveBeenCalledOnce(); expect(f.learn).toHaveBeenCalledOnce(); expect(f.text).not.toHaveBeenCalled();
    expect(details.items[0]).toMatchObject({ state: "complete", responseEnvelope: { sha256: expect.any(String) },
      mediaLesson: { parts: [{ type: "text" }, { type: "image" }, { type: "audio" }, { type: "video" }] },
      receipt: { mediaCoverage: { complete: true, byModality: { image: { records: 1 }, audio: { records: 1 }, video: { records: 1 } } } } });
    expect(details.items[0]!.receipt?.deliveryId).toBe(details.items[0]!.deliveryId);
    expect((await f.service.start(f.request)).id).toBe(job.id); expect(f.fetcher).toHaveBeenCalledOnce();
    await validateApiTeacherWorkspace(f.root, "brain-a"); await f.service.close();
  });
  it("pauses at selected parsing RAM after response custody and resumes without another billed request", async () => {
    const f = await fixture(); let deny = true;
    f.admitMetadata.mockImplementation(async (_id, _bytes, purpose) => { if (deny && purpose.includes("JSON/inline")) throw new Error("selected family RAM denied"); });
    const job = await f.service.start(f.request), paused = await waitState(f, "paused");
    expect(paused.items[0]?.state).toBe("response-staged"); expect(f.learn).not.toHaveBeenCalled();
    deny = false; await f.service.resume("brain-a", job.id); await waitState(f, "complete");
    expect(f.fetcher).toHaveBeenCalledOnce(); expect(f.learn).toHaveBeenCalledOnce(); await f.service.close();
  });
  it("requires explicit review of caption-only output and retry reparses custody without re-billing", async () => {
    const f = await fixture(); f.fetcher.mockResolvedValue(new Response(JSON.stringify({ status: "completed", steps: [{ type: "model_output", content: [{ type: "text", text: "This is a video" }] }] })));
    const job = await f.service.start(f.request), held = await waitState(f, "review-required");
    expect(held.items[0]?.state).toBe("response-review"); expect(f.learn).not.toHaveBeenCalled();
    await expect(f.service.resume("brain-a", job.id)).rejects.toThrow(/uncertain/);
    await f.service.resolveUncertain("brain-a", job.id, 0, "retry"); await f.service.resume("brain-a", job.id); await waitState(f, "review-required");
    expect(f.fetcher).toHaveBeenCalledOnce(); expect(f.learn).not.toHaveBeenCalled(); await f.service.close();
  });
  it("retains a genuine media-learning ACK when Stop arrives before host receipt publication", async () => {
    const f = await fixture(); let finish!: () => void, entered!: () => void;
    const started = new Promise<void>(resolve => { entered = resolve; });
    f.learn.mockImplementationOnce(async (id, experience) => { entered(); await new Promise<void>(resolve => { finish = resolve; }); return learned(id, experience.lesson); });
    const job = await f.service.start(f.request); await started;
    const stopped = f.service.stop("brain-a", job.id); finish(); await stopped;
    const details = await f.service.details("brain-a", job.id);
    expect(details.items[0]).toMatchObject({ state: "complete", receipt: { mediaCoverage: { complete: true } } });
    expect(f.learn).toHaveBeenCalledOnce(); expect(f.fetcher).toHaveBeenCalledOnce(); await f.service.close();
  });
  it("never treats a lost native media ACK as proof or silently repeats provider/learning work", async () => {
    const f = await fixture(); f.learn.mockRejectedValueOnce(new Error("ACK lost after possible neural commit"));
    const job = await f.service.start(f.request), held = await waitState(f, "review-required");
    expect(held.items[0]?.state).toBe("learning-uncertain-review");
    await expect(f.service.resume("brain-a", job.id)).rejects.toThrow(/uncertain/);
    await f.service.resolveUncertain("brain-a", job.id, 0, "retry"); await f.service.resume("brain-a", job.id); await waitState(f, "complete");
    expect(f.fetcher).toHaveBeenCalledOnce(); expect(f.learn.mock.calls[1]![1]).toEqual(f.learn.mock.calls[0]![1]); await f.service.close();
  });
  it("captures, copies, rekeys and merges complete media custody dormant without rewriting source IDs or bytes", async () => {
    const f = await fixture(), job = await f.service.start(f.request); const original = await waitState(f, "complete"); await f.service.close();
    const copied = await temporary(); await captureApiTeacherWorkspace(f.root, copied, "brain-a");
    await rekeyApiTeacherWorkspace(copied, "brain-a", "brain-b"); await validateApiTeacherWorkspace(copied, "brain-b");
    const store = await ApiTeacherStore.open(copied, "brain-b", async () => undefined, { create: false });
    try { expect(store!.jobs()[0]).toMatchObject({ state: "dormant", originalBrainId: "brain-a", originalJobId: job.id });
      expect(store!.items(store!.jobs()[0]!.id)[0]!.mediaLesson).toEqual(original.items[0]!.mediaLesson); } finally { store?.close(); }
    const merged = await temporary(); await mergeApiTeacherWorkspace(copied, merged, "brain-b", "parent-a", async () => undefined);
    await validateApiTeacherWorkspace(merged, "brain-b");
    for (const part of original.items[0]!.mediaLesson!.parts) if (part.type !== "text") expect(await readFile(join(merged, part.relativePath))).toEqual(bytes[part.type]);
  });
  it("validates exact portable inventory media closure before adoption and refuses a missing declared asset", async () => {
    const f = await fixture(), jobEvents = on(f.service, "event");
    try {
      // Subscribe before start so even immediate completion is retained. The
      // owner publishes this event only after committing its terminal state;
      // repeated SQLite reads and a second, shorter I/O deadline are unnecessary.
      const job = await f.service.start(f.request);
      for await (const [received] of jobEvents) {
        const current = (received as RuntimeJobEvent).job;
        if (current.brainId !== f.request.brainId || current.id !== job.id) continue;
        if (current.state === "complete") break;
        // Paused/review/stopped durable states project to runtime cancelled.
        if (current.state === "failed" || current.state === "cancelled")
          throw new Error(`Media fixture ended ${current.state}: ${current.error ?? current.label}`);
      }
      const committed = await f.service.details(f.request.brainId, job.id);
      expect(committed).toMatchObject({ brainId: f.request.brainId, jobId: job.id,
        state: "complete", learned: 1, failed: 0, items: [{ state: "complete" }] });
      expect(f.fetcher).toHaveBeenCalledOnce(); expect(f.learn).toHaveBeenCalledOnce();
    } finally { await jobEvents.return?.(); await f.service.close(); }
    const stage = await temporary(), workspace = await snapshotLearningWorkspace(f.root, join(stage, "workspace"), "brain-a");
    expect(workspace).toBeDefined();
    const entries = new Map([...workspace!.files].map(([name, path]) => [`learning-workspace/${name}`, { name, sourcePath: path }]));
    const validation = await temporary(); expect(await validateApiTeacherMediaArchiveSources(entries, validation)).toBe(true);
    const imageSha = apiTeacherSha256(bytes.image), damaged = new Map(entries); damaged.delete(`learning-workspace/blobs/${imageSha}.data`);
    await expect(validateApiTeacherMediaArchiveSources(damaged, await temporary())).rejects.toThrow(/missing.*blob/);
  });
  it("the actual BrainService seam dispatches typed file ingestion, rejects text-only ACKs and retains a late valid ACK", async () => {
    const root = await temporary(), lesson = await storeApiTeacherMediaLesson(root, { provider: "gemini", sourceEndpoint: "https://generativelanguage.googleapis.com/v1beta/interactions",
      model: input.model, prompt: input.prompt, deliveryId: "native-delivery", responseModalities: ["image"],
      parts: [{ type: "image", mimeType: "image/png", data: bytes.image.toString("base64") }] });
    const brain = { id: "brain-a", trainingSources: [], journal: [], counters: {} }, repository = { brainDirectory: () => root,
      get: vi.fn(async () => brain), save: vi.fn(async value => value), trainingSourceByContentHash: vi.fn(async () => undefined), admitMetadataTransfer: vi.fn(async () => undefined) };
    const stop = new AbortController(), request = vi.fn(async (_method: string, params: Record<string, unknown>) => {
      stop.abort(); return { transactionKey: params.transactionKey, source: { content_hash: lesson.manifestSha256, modality_trained: true,
        media_coverage: { complete: true, failedRecords: 0, byModality: { image: { records: 1, trainedRecords: 1, failedRecords: 0, windows: 1, processedUnits: 1, complete: true } } },
        coverage: { complete: true }, parameter_update_steps: 1 } };
    });
    const service = Object.create(BrainService.prototype) as BrainService;
    Object.assign(service, { repository, engine: { request }, preflightStart: vi.fn(async () => undefined) });
    const result = await service.learnTeacherMediaExperience("brain-a", { lesson, prompt: input.prompt }, stop.signal);
    expect(result.mediaCoverage.byModality.image?.records).toBe(1); expect(repository.save).toHaveBeenCalledOnce();
    expect(request.mock.calls[0]![1]).toMatchObject({ kind: "jsonl", path: join(root, lesson.manifestPath), contentHash: lesson.manifestSha256 });
    request.mockImplementationOnce(async () => ({ source: { kind: "text", coverage: { complete: true }, parameter_update_steps: 1 } }) as never);
    await expect(service.learnTeacherMediaExperience("brain-a", { lesson, prompt: input.prompt })).rejects.toThrow(/native ACK/);
  });
  it("finishes a received media lesson after restart without the removed provider key or another request", async () => {
    const f = await fixture(); let deny = true;
    f.admitMetadata.mockImplementation(async (_id, _bytes, purpose) => { if (deny && purpose.includes("JSON/inline")) throw new Error("parse RAM"); });
    const job = await f.service.start(f.request); await waitState(f, "paused"); await f.service.close(); deny = false;
    const resumed = new ApiTeacherTrainingService({ preflightStart: async () => undefined, learnTeacherMediaExperience: f.learn,
      learnStructuredExperience: f.text } as never, { get: async () => undefined, persistence: () => "none" } as never,
    { brainRoot: async () => f.root, admitWrite: async () => undefined }, f.fetcher as never);
    await resumed.resume("brain-a", job.id);
    await vi.waitFor(async () => expect((await resumed.details("brain-a", job.id)).state).toBe("complete"));
    expect(f.fetcher).toHaveBeenCalledOnce(); expect(f.learn).toHaveBeenCalledOnce(); await resumed.close();
  });
  it("refuses a linked publication root before creating anything outside its owned tree", async () => {
    const root = await temporary(), outside = await temporary(); await symlink(outside, join(root, "api-teachers"), process.platform === "win32" ? "junction" : "dir");
    await expect(storeApiTeacherResponseEnvelope(root, new Response(JSON.stringify(body())), [], {})).rejects.toThrow(/not owned/);
    expect(await readdir(outside)).toEqual([]);
  });
  it("refuses partial provider status and extra typed-block instructions without treating them as media", () => {
    expect(() => apiTeacherMediaParts("gemini", { ...body(), status: "in_progress" }, ["image"])).toThrow(/not a complete/);
    const fake = { status: "completed", steps: [{ type: "model_output", content: [{ type: "image", data: bytes.image.toString("base64"), mime_type: "image/png", execute: "ignored?" }] }] };
    expect(() => apiTeacherMediaParts("gemini", fake, ["image"])).toThrow(/inline typed bytes/);
  });
  it("refuses an escaped device key spanning response chunks before final envelope publication", async () => {
    const root = await temporary(), raw = '{"text":"\\u0066ixture-device-key-keep-private"}', encoded = Buffer.from(raw);
    const stream = new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(encoded.subarray(0,25)); controller.enqueue(encoded.subarray(25)); controller.close(); } });
    await expect(storeApiTeacherResponseEnvelope(root, new Response(stream), [input.credential], {})).rejects.toThrow(/encoded a device credential/);
    expect(await readdir(join(root, "api-teachers", "envelopes"))).toEqual([]);
  });
});
