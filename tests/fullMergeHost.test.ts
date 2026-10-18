import { createHash } from "node:crypto";
import { copyFile, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { EventEmitter } from "node:events";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import { ConversationLedger } from "../src/main/conversationLedger";
import { ArtifactIndexStore } from "../src/main/artifactIndexStore";
import { UserQuestionService } from "../src/main/userQuestions";
import { savedUserQuestionFiles } from "../src/main/userQuestionPortability";
import { fullMergeFileHash, mergeCanonical } from "../src/main/fullMergeHost";
import { DEFAULT_CONFIG, type ActionEvent } from "../src/shared/types";
import type { FullMergeStartRequest } from "../src/shared/fullMerge";
import type { EngineSupervisor } from "../src/main/engineSupervisor";

const cleanup: string[] = [];
const runningServices: BrainService[] = [];
afterEach(async () => {
  // A timed-out test must acknowledge cancellation before deleting a staging
  // tree still owned by the asynchronous copy/verification operation.
  for (const service of runningServices.splice(0)) await service.pauseFullMerge("host-fixture").catch(() => undefined);
  for (const path of cleanup.splice(0)) await rm(path, { recursive: true, force: true, maxRetries: 5 });
});
const hash = (data: string | Buffer): string => createHash("sha256").update(data).digest("hex");
const timestamp = "2026-10-01T00:00:00.000Z";
const writeJson = async (path: string, value: unknown) => writeFile(path, JSON.stringify(value));

async function nativeFixture(root: string, id: string, name: string, parents?: Array<{ brainId: string }>) {
  const engine = join(root, "engine");
  await mkdir(join(engine, "packed-ternary"), { recursive: true });
  await writeFile(join(engine, "core.safetensors"), `tiny fixture ${id}`);
  await writeFile(join(engine, "plasticity.safetensors"), "tiny plastic fixture");
  const parameterChecksum = hash(id);
  const packed = { contentSha256: hash("packed"), metadata: { brainId: id, parameterChecksum } };
  await writeJson(join(engine, "packed-ternary", "manifest.json"), packed);
  const descriptor = parents ? { sha256: hash(JSON.stringify(parents)), parents: parents.map((parent) => ({ parentId: parent.brainId })) } : undefined;
  const metadata = {
    format: "omni-cortex-engine", release_format: "stable-1.0", brain_id: id, name,
    config: { origin_kind: "ground-up", name }, runtime_card: { origin_kind: "ground-up", pretrained: false, baseFrozen: false },
    substrate: { persistence: { contentSha256: hash("substrate") } }, mutable_state: { contentSha256: hash("mutable") },
    packed_ternary_manifest: { contentSha256: packed.contentSha256, parameterChecksum, baseFrozen: false },
    ...(descriptor ? { connected_regions: descriptor, connected_runtime_adoption: { sha256: hash("adoption") } } : {})
  };
  await writeJson(join(engine, "brain.json"), metadata);
  if (!descriptor) return { metadata, packed };
  const origin = join(engine, "origin");
  await mkdir(join(origin, "packed-ternary"), { recursive: true });
  for (const file of ["brain.json", "core.safetensors", "plasticity.safetensors", "packed-ternary/manifest.json"]) await copyFile(join(engine, file), join(origin, file));
  const body = { format: "omni-connected-native-origin", formatVersion: 1, brainId: id,
    descriptorSha256: descriptor.sha256, runtimeAdoptionSha256: hash("adoption"), parameterChecksum,
    metadataSha256: await fullMergeFileHash(join(origin, "brain.json")), coreSha256: await fullMergeFileHash(join(origin, "core.safetensors")),
    plasticitySha256: await fullMergeFileHash(join(origin, "plasticity.safetensors")),
    packedManifestSha256: await fullMergeFileHash(join(origin, "packed-ternary", "manifest.json")), initialCurriculumRerun: false };
  const marker = { ...body, sha256: hash(mergeCanonical(body)) };
  await writeJson(join(origin, "connected-origin.json"), marker);
  return { metadata, packed, marker };
}

/** Deliberately tiny transport fixture. Native tensor computation is tested in Python separately. */
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-host-merge-")); cleanup.push(root);
  const repository = new BrainRepository(root);
  const sharedBytes = Buffer.from("unsanitized private source bytes");
  const sharedBlobHash = await repository.storeBlob(sharedBytes);
  const parents = [];
  for (const name of ["Left", "Right"]) {
    let brain = await repository.create({ ...DEFAULT_CONFIG, name, idleCognition: false });
    brain.messages = [
      { id: "shared-message", role: "human", content: "Exact shared experience", createdAt: timestamp },
      { id: "conflicting-id", role: "brain", content: `${name} contradictory record`, createdAt: timestamp },
      { id: "ponder", role: "brain", content: `${name} actually emitted Ponder`, createdAt: timestamp }
    ];
    brain.toolPermissions = [{ toolId: "system.shell", level: "full", label: "Shell", updatedAt: timestamp }];
    brain.trainingSources = [{ id: "shared-source", name: "Private source", kind: "text", bytes: sharedBytes.length,
      learnedIdeas: 0, learnedConcepts: 0, learnedSynapses: 0, importedAt: timestamp, rawTextRetained: true,
      rawText: sharedBytes.toString(), blobHash: sharedBlobHash }];
    brain = await repository.save(brain);
    await nativeFixture(repository.brainDirectory(brain.id), brain.id, name);
    const questions = new UserQuestionService({
      brainDirectory: (brainId) => repository.brainDirectory(brainId),
      now: () => Date.parse(timestamp)
    });
    try {
      const question = await questions.request({
        brainId: brain.id,
        requestId: `full-merge-question-${name.toLocaleLowerCase()}`,
        prompt: `${name} retained task choice`,
        mode: "single",
        options: [
          { id: "keep", label: "Keep the reviewed evidence" },
          { id: "discard", label: "Discard the reviewed evidence" }
        ],
        allowOther: false,
        source: { kind: "host" }
      });
      await questions.respond({
        brainId: brain.id,
        questionId: question.id,
        response: { optionIds: ["keep"] }
      });
    } finally {
      questions.close();
    }
    const action: ActionEvent = { id: "old-action", brainId: brain.id, createdAt: timestamp, updatedAt: timestamp,
      state: "approval-required", action: { kind: "tool", source: "brain", toolId: "system.shell", action: "run", arguments: { command: "retained text only" } } };
    const ledger = await ConversationLedger.open(repository.brainDirectory(brain.id), brain.id);
    ledger.append([{ kind: "action", value: action }]); ledger.close();
    const directory = join(repository.brainDirectory(brain.id), "engine", "artifacts");
    const artifacts = await ArtifactIndexStore.open(directory, brain.id);
    const bytes = Buffer.from(`image ${name}`);
    await writeFile(join(directory, "picture.png"), bytes);
    artifacts.append([{ id: hash(name), sha256: hash(bytes), modality: "image", mimeType: "image/png", bytes: bytes.length,
      createdAt: timestamp, relativePath: "artifacts/picture.png" }]); artifacts.close();
    parents.push(brain);
  }
  const engine = new EventEmitter() as EventEmitter & { request: ReturnType<typeof vi.fn>; instanceStatus: ReturnType<typeof vi.fn>; claimForeground: ReturnType<typeof vi.fn>; cancelBackgroundRequest: ReturnType<typeof vi.fn> };
  engine.claimForeground = vi.fn(async () => undefined); engine.cancelBackgroundRequest = vi.fn(() => false);
  engine.instanceStatus = vi.fn(async (brainId: string) => ({ brainId, learnedRuntimeResident: false, minimumContextTokens: 0, reconnect: [] }));
  engine.request = vi.fn(async (method: string, params: Record<string, any>) => {
    if (method === "suspend_instance") return { brainId: params.brainId, learnedRuntimeResident: false, minimumContextTokens: 0, reconnect: [], unloaded: true };
    if (method === "checkpoint") {
      const path = join(params.storagePath, "engine");
      const metadata = JSON.parse(await readFile(join(path, "brain.json"), "utf8"));
      return { format: "omni-neural-checkpoint", formatVersion: 1, brainId: params.brainId, operationId: params.operationId,
        committed: true, createdAt: timestamp, snapshotCreated: false, parameterChecksum: metadata.packed_ternary_manifest.parameterChecksum,
        metadataSha256: await fullMergeFileHash(join(path, "brain.json")), substrateContentSha256: hash("substrate"),
        mutableStateContentSha256: hash("mutable"), packedManifestSha256: await fullMergeFileHash(join(path, "packed-ternary", "manifest.json")), packedContentSha256: hash("packed") };
    }
    if (method === "full_merge_assemble") {
      const result = await nativeFixture(params.storagePath, params.brainId, params.config.name, params.parents);
      return { format: "omni-full-native-assembly", formatVersion: 1, brainId: params.brainId, descriptorSha256: result.marker!.descriptorSha256, origin: result.marker };
    }
    if (method === "full_merge_verify") {
      const origin = JSON.parse(await readFile(join(params.storagePath, "engine", "origin", "connected-origin.json"), "utf8"));
      return { brainId: params.brainId, verified: true, descriptorSha256: origin.descriptorSha256 };
    }
    throw new Error(`Unexpected worker operation: ${method}`);
  });
  const service = new BrainService(repository, engine as unknown as EngineSupervisor);
  runningServices.push(service);
  const access = vi.fn(async () => undefined); service.configureFullMergeAccess(access);
  const request: FullMergeStartRequest = { operationId: "host-fixture", hardwareTier: "micro", parentIds: parents.map((parent) => parent.id).reverse(),
    name: "Merged", config: repository.prepareConfig({ ...DEFAULT_CONFIG, name: "Merged", idleCognition: false }),
    accessPolicy: { credentials: "none", mcp: "disabled", toolPermissions: [{ toolId: "brain.history", level: "auto" }] } };
  return { root, repository, engine, service, request, access, parents, sharedBlobHash, sharedBytes };
}

describe("production full merge host integration", () => {
  it("preserves opaque nested user Id fields in original completed tool output", async () => {
    const f = await fixture(), parent = f.parents[0]!;
    const output = { userId: "shared-message", nested: { actionId: "old-action", records: [{ fileId: "conflicting-id", id: "ponder" }] } };
    const action: ActionEvent = { id: "opaque-result-action", brainId: parent.id, createdAt: timestamp, updatedAt: timestamp, state: "complete",
      action: { kind: "tool", source: "brain", toolId: "system.files", action: "read", arguments: { path: "fixture.txt" } },
      execution: { id: "12345678-1234-4567-8123-123456789012", toolId: "system.files", action: "read", state: "complete", startedAt: timestamp, finishedAt: timestamp, output } };
    const source = await ConversationLedger.open(f.repository.brainDirectory(parent.id), parent.id); source.append([{ kind: "action", value: action }]); source.close();
    const result = await f.service.startFullMerge(f.request), ledger = await ConversationLedger.open(f.repository.brainDirectory(result.targetBrainId), result.targetBrainId);
    try { const retained = ledger.page().entries.find(entry => entry.action?.execution?.toolId === "system.files")?.action;
      expect(retained?.execution?.output).toEqual(output); expect(retained?.brainId).toBe(result.targetBrainId);
    } finally { ledger.close(); }
  }, 120_000);
  it("calls real host callbacks and publishes truthful origin, complete history and isolated authority", async () => {
    const f = await fixture();
    const before = await Promise.all(f.parents.map((parent) => fullMergeFileHash(join(f.root, parent.id, "brain.json"))));
    const result = await f.service.startFullMerge(f.request);
    expect(result.state).toBe("complete");
    const brain = await f.repository.get(result.targetBrainId);
    expect(brain.provenance).toMatchObject({ originKind: "merged-native", merge: { initialCurriculumRerun: false, parentIds: [...f.request.parentIds].sort() } });
    expect(brain.lineage.parentIds).toEqual([...f.request.parentIds].sort());
    expect(brain.toolPermissions!.find((permission) => permission.toolId === "system.shell")?.level).toBe("off");
    expect(f.access).toHaveBeenCalledWith(brain.id);
    expect(await readFile(join(f.root, brain.id, "merge", "shared-blobs", f.sharedBlobHash))).toEqual(f.sharedBytes);
    expect(f.engine.request.mock.calls.map((call) => call[0])).toEqual(["suspend_instance", "suspend_instance", "full_merge_assemble", "full_merge_verify"]);
    expect(f.engine.request.mock.calls.filter((call) => call[0] === "suspend_instance").map((call) => call[1].brainId)).toEqual([...f.request.parentIds].sort());
    const ledger = await ConversationLedger.open(f.repository.brainDirectory(brain.id), brain.id);
    const page = ledger.page(); ledger.close();
    expect(page.entries.filter((entry) => entry.message?.content === "Exact shared experience")).toHaveLength(1);
    expect(page.entries.filter((entry) => entry.message?.content.includes("contradictory"))).toHaveLength(2);
    expect(page.entries.filter((entry) => entry.message?.content.includes("Ponder"))).toHaveLength(2);
    expect(page.entries.filter((entry) => entry.action).every((entry) => entry.action!.state === "stopped")).toBe(true);
    expect(await readdir(f.repository.brainDirectory(brain.id))).toContain("questions");
    const retainedQuestions = await savedUserQuestionFiles(
      f.repository.brainDirectory(brain.id),
      brain.id
    );
    expect(
      [...retainedQuestions.keys()].filter((path) =>
        path.startsWith("questions/records/")
      )
    ).toHaveLength(2);
    const history = await f.service.fullMergeHistory(brain.id, 0, 200);
    expect(history.entries.filter((entry) => entry.kind === "message" && (entry.payload as any).content === "Exact shared experience")).toHaveLength(2);
    expect(history.entries.filter((entry) => entry.path.includes("questions/records"))).toHaveLength(2);
    const artifacts = await ArtifactIndexStore.open(join(f.root, brain.id, "engine", "artifacts"), brain.id);
    expect(artifacts.page().totalArtifacts).toBe(2);
    for (const artifact of artifacts.page().artifacts) expect(await fullMergeFileHash(join(f.root, brain.id, "engine", artifact.relativePath))).toBe(artifact.sha256);
    artifacts.close();
    expect(await Promise.all(f.parents.map((parent) => fullMergeFileHash(join(f.root, parent.id, "brain.json"))))).toEqual(before);
    const restarted = new BrainService(f.repository, f.engine as unknown as EngineSupervisor);
    expect((await restarted.listFullMerges())[0]?.targetBrainId).toBe(brain.id);
    expect((await restarted.resumeFullMerge(f.request.operationId)).targetBrainId).toBe(brain.id);
    expect(f.engine.request.mock.calls.filter((call) => call[0] === "full_merge_assemble")).toHaveLength(1);
    const nativePath = join(f.root, brain.id, "engine", "brain.json");
    const native = JSON.parse(await readFile(nativePath, "utf8"));
    const checkpoint = { transactionId: hash("original-transaction"), resolvedKind: "jsonl", policy: "encode", epoch: 3,
      contentHash: hash("source bytes"), sourceNameHash: hash("original.jsonl"), committedRecords: 7 };
    native.connected_pending_ingestions = [{ id: hash("pending"), status: "queued", parentId: f.parents[0]!.id, checkpoint }];
    await writeJson(nativePath, native);
    const normal = f.engine.request.getMockImplementation()!;
    f.engine.request.mockImplementation(async (method: string, params: Record<string, any>) => {
      if (method === "full_merge_activate_ingestion") return { id: hash("pending"), state: native.connected_pending_ingestions[0].status === "complete" ? "complete" : "active", checkpoint };
      if (method === "ingest") {
        expect(params).toMatchObject({ transactionKey: checkpoint.transactionId, idempotencyKey: checkpoint.transactionId,
          policy: "encode", epoch: 3, name: "original.jsonl", contentHash: checkpoint.contentHash, allowReplay: true });
        native.connected_pending_ingestions[0].status = "complete"; await writeJson(nativePath, native);
        return { brainId: brain.id, coverage: { complete: true, processedRecords: 9 } };
      }
      return normal(method, params);
    });
    const selection = { brainId: brain.id, pendingId: hash("pending"), sourceName: "original.jsonl" };
    expect((await f.service.fullMergePendingIngestions(brain.id))[0]?.committedRecords).toBe(7);
    await f.service.resumeFullMergeIngestion(selection, join(f.root, "selected.jsonl"), "resume-job");
    await f.service.resumeFullMergeIngestion(selection, join(f.root, "selected.jsonl"), "retry-job");
    expect(f.engine.request.mock.calls.filter((call) => call[0] === "ingest")).toHaveLength(1);
  }, 120_000);

  it("pauses and resumes an incomplete copy without re-checkpointing or changing the reviewed destination", async () => {
    const f = await fixture(); let paused = false;
    const unsubscribe = f.service.onFullMerge((progress) => {
      if (progress.filesCopied === 1 && !paused) { paused = true; void f.service.pauseFullMerge(progress.operationId); }
    });
    await expect(f.service.startFullMerge(f.request)).rejects.toThrow("paused"); unsubscribe();
    const state = await f.service.fullMergeStatus(f.request.operationId);
    expect(state.state).toBe("paused"); expect(await readdir(f.root)).not.toContain(state.targetBrainId);
    const result = await f.service.resumeFullMerge(f.request.operationId);
    expect(result.targetBrainId).toBe(state.targetBrainId);
    expect(f.engine.request.mock.calls.filter((call) => call[0] === "checkpoint")).toHaveLength(0);
    expect(f.engine.request.mock.calls.filter((call) => call[0] === "suspend_instance")).toHaveLength(2);
  }, 120_000);

  it("cancels only unpublished output with a durable receipt and rejects missing permission review", async () => {
    const f = await fixture(); let paused = false;
    const unsubscribe = f.service.onFullMerge((progress) => {
      if (progress.filesCopied === 1 && !paused) { paused = true; void f.service.pauseFullMerge(progress.operationId); }
    });
    await expect(f.service.startFullMerge(f.request)).rejects.toThrow("paused"); unsubscribe();
    const result = await f.service.cancelFullMerge(f.request.operationId);
    expect(result.state).toBe("cancelled");
    const restarted = new BrainService(f.repository, f.engine as unknown as EngineSupervisor);
    expect((await restarted.fullMergeStatus(f.request.operationId)).state).toBe("cancelled");
    for (const parent of f.parents) expect((await f.repository.get(parent.id)).id).toBe(parent.id);
    expect(() => f.service.startFullMerge({ ...f.request, accessPolicy: undefined } as unknown as FullMergeStartRequest)).toThrow("access policy");
    const beginning = f.service.startFullMerge({ ...f.request, operationId: "pause-before-validation" });
    const acknowledgement = f.service.pauseFullMerge("pause-before-validation");
    await expect(beginning).rejects.toThrow("paused");
    expect(await acknowledgement).toMatchObject({ state: "paused", filesCopied: 0 });
    await f.service.cancelFullMerge("pause-before-validation");
  }, 120_000);

  it("rejects forged or changed merged origin bytes during host loading", async () => {
    const f = await fixture(); const result = await f.service.startFullMerge(f.request);
    await writeFile(join(f.root, result.targetBrainId, "engine", "origin", "core.safetensors"), "changed original");
    await expect(f.repository.get(result.targetBrainId)).rejects.toThrow("origin bytes changed");
  }, 120_000);
});
