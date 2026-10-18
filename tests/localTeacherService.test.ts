/** Synthetic teacher/native contracts around REAL SQLite/files/manifests.
 * No child/model/provider inference, trained state or quality claim. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { LocalTeacherSelection, LocalTeacherStartRequest } from "../src/shared/localTeacher";
import type { LearningTask } from "../src/shared/learning";
import { LocalTeacherService, type LocalTeacherServiceOptions } from "../src/main/localTeacherService";
import { LocalTeacherStore } from "../src/main/localTeacherStore";
import { DatasetManifestStore } from "../src/main/dataIngestion";
import { hashLocalTeacherFile } from "../src/main/localTeacherFiles";
import { localTeacherHash } from "../src/main/localTeacherFormat";
import { localTeacherDirectory, writeTeacherReceipt } from "../src/main/localTeacherArtifacts";
import { localTeacherRequiredFlags } from "../src/main/localTeacherRunner";
const fake = vi.hoisted(() => ({ calls: [] as string[], waitSecond: false, opens: 0, closes: 0 }));
vi.mock("../src/main/localTeacherServer", async original => {
  const actual = await original<typeof import("../src/main/localTeacherServer")>();
  return { ...actual, openLocalTeacherServer: async (options: import("../src/main/localTeacherServer").LocalTeacherServerOptions) => {
    fake.opens++; await options.requireCurrent();
    return { sessionId: options.sessionId, argvSha256: localTeacherHash("SYNTHETIC teacher argv, no process"),
      complete: async (prompt: string, attemptId: string, signal: AbortSignal) => {
        fake.calls.push(prompt);
        if (fake.waitSecond && fake.calls.length === 2) await new Promise((_resolve, reject) => signal.addEventListener("abort", () => reject(signal.reason), { once: true }));
        signal.throwIfAborted(); await options.requireCurrent();
        const relative = `local-teachers/attempts/${attemptId}`, directory = await localTeacherDirectory(options.root, relative), text = `SYNTHETIC response to exact ${prompt}`;
        await writeFile(join(directory, "content.bin"), text); await writeFile(join(directory, "response.sse"), "SYNTHETIC transport");
        return { content: { relativePath: `${relative}/content.bin`, sha256: localTeacherHash(text), bytes: Buffer.byteLength(text), coverage: "complete" as const },
          transport: { relativePath: `${relative}/response.sse`, sha256: localTeacherHash("SYNTHETIC transport"), bytes: Buffer.byteLength("SYNTHETIC transport"), coverage: "complete" as const },
          final: { stop: true }, elapsedNanoseconds: "1", requestSha256: localTeacherHash(actual.localTeacherCompletionBody(prompt, options.generation)) };
      }, close: async () => { fake.closes++; const relative = `local-teachers/attempts/${options.sessionId}`, directory = await localTeacherDirectory(options.root, relative);
        await writeFile(join(directory, "process.stdout.bin"), "SYNTHETIC no child"); await writeFile(join(directory, "process.stderr.bin"), "");
        return { exitCode: null, stdout: { relativePath: `${relative}/process.stdout.bin`, sha256: localTeacherHash("SYNTHETIC no child"), bytes: 18, coverage: "complete" as const },
          stderr: { relativePath: `${relative}/process.stderr.bin`, sha256: localTeacherHash(""), bytes: 0, coverage: "complete" as const }, elapsedNanoseconds: "1" }; }
    };
  } };
});
const roots: string[] = [], services: LocalTeacherService[] = [], brainId = "55555555-5555-4555-8555-555555555555";
beforeEach(() => { fake.calls = []; fake.opens = 0; fake.closes = 0; fake.waitSecond = false; });
afterEach(async () => { await Promise.all(services.splice(0).map(service => service.close())); await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
const generation = { contextTokens: 32, maxOutputTokens: 8, threads: 1, temperature: .7, seed: 1, timeoutMs: 0, maxOutputBytes: 4096 };
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-local-teacher-service-")), dependencies = await mkdtemp(join(tmpdir(), "omni-local-teacher-deps-")); roots.push(root, dependencies);
  await writeFile(join(dependencies, "llama-server"), "SYNTHETIC fixture dependency, NOT EXECUTED"); await writeFile(join(dependencies, "tiny.gguf"), "SYNTHETIC header, NO TENSORS/INFERENCE");
  const runtime = await hashLocalTeacherFile(join(dependencies, "llama-server"), new AbortController().signal, async () => undefined),
    model = await hashLocalTeacherFile(join(dependencies, "tiny.gguf"), new AbortController().signal, async () => undefined), teacherId = localTeacherHash("synthetic current teacher");
  const selection: LocalTeacherSelection = { format: "omni-local-gguf-teacher", version: 1, id: teacherId, brainId, originalBrainId: brainId, originalSelectionId: teacherId,
    createdAt: new Date().toISOString(), runner: "llama-server", currentHostAuthorized: false, license: { declaration: null, source: "not-declared", verified: false },
    runtime: { displayName: runtime.displayName, sha256: runtime.sha256, bytes: runtime.bytes, helpSha256: localTeacherHash("synthetic help"), rawOfflineSupported: true, missingFlags: [] },
    model: { displayName: model.displayName, sha256: model.sha256, bytes: model.bytes, header: { version: 3, tensorCount: "0", metadataCount: "0", metadataEnd: 24,
      architecture: "llama", geometry: { "llama.context_length": 128, "llama.embedding_length": 16, "llama.block_count": 2, "llama.attention.head_count": 2 }, scope: "header-metadata-only", tensorPayloadValidated: false } } };
  const store = (await LocalTeacherStore.open(root, brainId))!; store.putSelection({ selection, runtime, model, requestId: "synthetic-selection", requestSha256: localTeacherHash("synthetic selection") }); store.close();
  let loaded = true, writer = false, compute = false, revision = "current-policy", level: "off" | "ask" | "auto" = "ask"; const tasks: LearningTask[] = [], nativeInputs: string[][] = [];
  const manifests = new DatasetManifestStore(() => root);
  const options: LocalTeacherServiceOptions = {
    brainDirectory: async id => { if (id !== brainId) throw new Error("wrong brain"); return root; }, currentPermission: async () => ({ level, revision }), selectFiles: async () => null,
    isLoaded: async () => loaded, requireLoaded: () => { if (!loaded) throw new Error("explicit Load required"); },
    measurements: async () => ({ appFamilyResidentBytes: 1024 * 1024, selectedRamCeilingBytes: 256 * 1024 * 1024, physicalFreeBytes: 256 * 1024 * 1024, logicalCpuCount: 4 }),
    checkMetadata: async () => undefined, reserveWrite: async () => undefined,
    withBrainWrite: async (_id, work) => { writer = true; try { return await work(); } finally { writer = false; } },
    withHostCompute: async (_context, signal, work) => { compute = true; try { return await work(signal); } finally { compute = false; } },
    previewDataset: async (id, paths, signal) => { expect(writer).toBe(false); expect(compute).toBe(false); return manifests.create(id, paths, { signal }); },
    learning: { subscribe: () => () => undefined, list: async () => tasks, get: async (_id, taskId) => tasks.find(task => task.id === taskId)!,
      start: async request => { expect(writer).toBe(false); expect(compute).toBe(false); const manifestId = (request.inputs[0] as { manifestId: string }).manifestId,
        manifest = await manifests.manifest(brainId, manifestId), contents: string[] = [];
        for await (const entry of manifests.entries(brainId, manifestId)) contents.push(await readFile(entry.path, "utf8")); nativeInputs.push(contents);
        const epochs = request.epochs ?? 1, now = new Date().toISOString(), task: LearningTask = { format: "omni-learning-task", version: 1, id: localTeacherHash(`synthetic ${request.requestId}`), brainId, requestId: request.requestId!,
          requestSha256: localTeacherHash("synthetic native contract"), kind: "specific-learn", state: "complete", createdAt: now, updatedAt: now, revision: 1,
          policy: { revision: 1, sha256: localTeacherHash("empty instructions"), constraints: {}, instructions: "", changedAt: now, changedBy: "user" },
          sources: [{ id: "synthetic-source", kind: "dataset", label: "SYNTHETIC source traversal receipt", state: "complete", manifestId, coverage: {
            schemaVersion: 1, manifestId, requestedEpochs: epochs, completedEpochs: epochs, discoveredFiles: manifest.discoveredFiles * epochs, processedFiles: manifest.discoveredFiles * epochs, rejectedFiles: 0,
            discoveredRecords: manifest.discoveredFiles, processedRecords: manifest.discoveredFiles, rejectedRecords: 0, discoveredBytes: manifest.discoveredBytes, processedBytes: manifest.discoveredBytes,
            shards: manifest.discoveredFiles, modalityCounts: { jsonl: manifest.discoveredFiles }, errors: [], errorCount: 0, complete: true, updatedAt: now } }],
          phase: "synthetic-complete", label: "SYNTHETIC cursor, NOT neural learning", epochs: 1, pendingQuestions: [], traversalComplete: true }; tasks.push(task); return task; },
      pause: async (_id, taskId) => tasks.find(task => task.id === taskId)!, resume: async (_id, taskId) => tasks.find(task => task.id === taskId)!, stop: async (_id, taskId) => tasks.find(task => task.id === taskId)! }
  };
  const service = new LocalTeacherService(options); services.push(service);
  const privateGrants = (service as unknown as { grants: Map<string, unknown> }).grants;
  privateGrants.set(`${brainId}:${teacherId}`, { brainId, teacherId, runtimeSha256: runtime.sha256, modelSha256: model.sha256, permissionRevision: revision });
  const request: LocalTeacherStartRequest = { brainId, requestId: "synthetic-full-request", teacherId, prompts: ["first\nfull paragraph 🌷", "second full prompt"], generation };
  return { root, service, request, tasks, nativeInputs, setLoaded: (value: boolean) => { loaded = value; }, revoke: () => { revision = "revoked-policy"; level = "off"; service.permissionChanged(brainId); } };
}

describe("local teacher/source orchestrator", () => {
  it("accepts only every requested native epoch visit, not just one file pass", async () => {
    const f = await fixture(), job = await f.service.start({ ...f.request, epochs: 3 });
    await vi.waitFor(async () => expect((await f.service.get(brainId, job.id)).state).toBe("complete"));
    expect((await f.service.get(brainId, job.id)).coverage?.processedFiles).toBe(6); expect(fake.calls).toHaveLength(2);
  });
  it("selection executes only a tiny owned --help fixture, reports complete hashes and never loads a model", async () => {
    const f = await fixture(), selected = await mkdtemp(join(tmpdir(), "omni-teacher-select-")); roots.push(selected);
    const runtime = join(selected, "llama-server"), model = join(selected, "tiny.gguf");
    await writeFile(runtime, `#!${process.execPath}\nconsole.log(${JSON.stringify(localTeacherRequiredFlags("llama-server").join("\n") + "\nUNIX socket .sock")});\n`); await chmod(runtime, 0o700);
    const bytes = Buffer.alloc(24); bytes.write("GGUF"); bytes.writeUInt32LE(3, 4); await writeFile(model, bytes);
    const options = (f.service as unknown as { options: LocalTeacherServiceOptions }).options; options.selectFiles = async () => ({ runtimePath: runtime, modelPath: model });
    const requireLoaded = vi.spyOn(options, "requireLoaded"), events: string[] = [], unsubscribe = f.service.subscribeSelection(event => events.push(event.phase));
    try { const selection = await f.service.select({ brainId, requestId: "actual-owned-help-only", runner: "llama-server" });
      expect(selection?.runtime.rawOfflineSupported).toBe(true); expect(selection?.model.header.tensorPayloadValidated).toBe(false); expect(selection?.currentHostAuthorized).toBe(true);
      expect(events).toContain("hashing-model"); expect(events.at(-1)).toBe("complete"); expect(fake.opens).toBe(0); expect(requireLoaded).not.toHaveBeenCalled(); }
    finally { unsubscribe(); }
  });
  it("cancels actual selected-file hashing without starting a teacher job/model", async () => {
    const f = await fixture(), selected = await mkdtemp(join(tmpdir(), "omni-teacher-cancel-select-")); roots.push(selected);
    const runtime = join(selected, "llama-server"), model = join(selected, "owned-nonmodel.data"); await writeFile(runtime, "synthetic not executed"); await writeFile(model, Buffer.alloc(4 * 1024 * 1024));
    const options = (f.service as unknown as { options: LocalTeacherServiceOptions }).options; options.selectFiles = async () => ({ runtimePath: runtime, modelPath: model });
    const unsubscribe = f.service.subscribeSelection(event => { if (event.phase === "hashing-model") void f.service.cancelSelection(brainId, "cancel-owned-hash"); });
    try { await expect(f.service.select({ brainId, requestId: "cancel-owned-hash", runner: "llama-server" })).rejects.toThrow(/cancelled/);
      expect((await f.service.selectionProgress(brainId, "cancel-owned-hash"))?.phase).toBe("cancelled"); expect(fake.opens).toBe(0); expect(f.tasks).toHaveLength(0); }
    finally { unsubscribe(); }
  });
  it("publishes immutable receipt files atomically and never overwrites a conflicting concurrent source", async () => {
    const f = await fixture(), relative = "local-teachers/attempts/immutable-fixture/result.json"; await localTeacherDirectory(f.root, "local-teachers/attempts/immutable-fixture");
    const results = await Promise.allSettled([writeTeacherReceipt(f.root, relative, { source: "one" }), writeTeacherReceipt(f.root, relative, { source: "two" })]);
    expect(results.filter(row => row.status === "fulfilled")).toHaveLength(1); expect(results.filter(row => row.status === "rejected")).toHaveLength(1);
    expect(["one", "two"]).toContain(JSON.parse(await readFile(join(f.root, relative), "utf8")).source);
  });
  it("releases exact compute/writer before one exhaustive source manifest and normal learning handoff", async () => {
    const f = await fixture(), job = await f.service.start(f.request);
    await vi.waitFor(async () => expect((await f.service.get(brainId, job.id)).state).toBe("complete"));
    expect(fake.calls).toEqual(f.request.prompts); expect(fake.opens).toBe(1); expect(fake.closes).toBe(1); expect(f.tasks).toHaveLength(1);
    expect(f.nativeInputs[0]!.map(text => JSON.parse(text).messages[0].content)).toEqual(f.request.prompts);
    const complete = await f.service.get(brainId, job.id); expect(complete.generatedCount).toBe(2); expect(complete.learnedCount).toBe(2);
    expect((await f.service.items(brainId, job.id)).every(item => item.receipt?.completionProof === "http-terminal" && item.receipt.exitCode === null)).toBe(true);
  });
  it("a lost start ACK returns the same job and never repeats completed teacher/native work", async () => {
    const f = await fixture(), job = await f.service.start(f.request); await vi.waitFor(async () => expect((await f.service.get(brainId, job.id)).state).toBe("complete"));
    expect((await f.service.start(f.request)).id).toBe(job.id); expect(fake.calls).toHaveLength(2); expect(f.tasks).toHaveLength(1);
    await expect(f.service.start({ ...f.request, prompts: ["changed after ACK"] })).rejects.toThrow(/changed its complete/);
  });
  it("Pause seals completed prefix without learning and Resume generates only the unfinished source", async () => {
    const f = await fixture(); fake.waitSecond = true; const job = await f.service.start(f.request);
    await vi.waitFor(() => expect(fake.calls).toHaveLength(2)); const paused = await f.service.pause(brainId, job.id);
    expect(paused.state).toBe("paused"); expect(paused.generatedCount).toBe(1); expect(f.tasks).toHaveLength(0);
    fake.waitSecond = false; await f.service.resume(brainId, job.id); await vi.waitFor(async () => expect((await f.service.get(brainId, job.id)).state).toBe("complete"));
    expect(fake.calls).toEqual([f.request.prompts[0], f.request.prompts[1], f.request.prompts[1]]); expect(f.tasks).toHaveLength(1);
  });
  it("keeps completed sources waiting while unloaded and does not invoke native work or hidden Load", async () => {
    const f = await fixture(); f.setLoaded(false); const job = await f.service.start(f.request);
    await vi.waitFor(async () => expect((await f.service.get(brainId, job.id)).state).toBe("waiting-load")); expect(f.tasks).toHaveLength(0);
    f.setLoaded(true); await f.service.resumeAfterLoad(brainId); await vi.waitFor(async () => expect((await f.service.get(brainId, job.id)).state).toBe("complete"));
    expect(fake.calls).toHaveLength(2); expect(f.tasks).toHaveLength(1);
  });
  it("revoked current host policy stops generation and never seals/adopts an uncertain prefix", async () => {
    const f = await fixture(); fake.waitSecond = true; const job = await f.service.start(f.request); await vi.waitFor(() => expect(fake.calls).toHaveLength(2)); f.revoke();
    await vi.waitFor(async () => expect((await f.service.get(brainId, job.id)).state).toBe("failed"));
    expect((await f.service.get(brainId, job.id)).generatedCount).toBe(0); expect(f.tasks).toHaveLength(0);
    await expect(f.service.resume(brainId, job.id)).rejects.toThrow(/disabled/);
  });
  it("actual app-family overlap admission rejects before any synthetic teacher work", async () => {
    const f = await fixture(), options = (f.service as unknown as { options: LocalTeacherServiceOptions }).options;
    options.measurements = async () => ({ appFamilyResidentBytes: 256 * 1024 * 1024, selectedRamCeilingBytes: 256 * 1024 * 1024, physicalFreeBytes: 256 * 1024 * 1024, logicalCpuCount: 4 });
    await expect(f.service.start(f.request)).rejects.toThrow(/RAM envelope/); expect(fake.opens).toBe(0); expect(f.tasks).toHaveLength(0);
  });
});
