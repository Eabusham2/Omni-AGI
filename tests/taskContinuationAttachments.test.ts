/** Genuine small SQLite/file custody and synthetic controller boundaries; never loads a neural model. */
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { BrainRepository } from "../src/main/brainRepository";
import { TaskAttachmentService } from "../src/main/taskAttachmentService";
import { TaskAttachmentStore } from "../src/main/taskAttachmentStore";
import { TaskContinuationAttachments } from "../src/main/taskContinuationAttachments";
import { TaskContinuationScopeStore, type WarmTaskAttachmentTransfer } from "../src/main/taskContinuationScope";
import { mergeTaskAttachmentWorkspace, rekeyTaskAttachmentWorkspace, validateTaskAttachmentWorkspace } from "../src/main/taskAttachmentPortability";
import { adoptLearningWorkspace, snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { ChatActionController } from "../src/main/chatActionController";
import { EngineRequestError } from "../src/main/engineSupervisor";
import type { ChatResult } from "../src/shared/types";
import type { NeuralChatStreamEvent } from "../src/main/brainService";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import { ToolExecutor } from "../src/main/toolExecutor";
import { prepareNativeAudioChat } from "../src/main/nativeAudioChat";
import { pcmWave } from "../src/renderer/src/nativeAudioCapture";

const roots: string[] = [];
const sha = (value: string) => createHash("sha256").update(value).digest("hex");
const time = "2026-10-13T00:00:00.000Z";
afterEach(async () => { vi.restoreAllMocks(); for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });

async function fixture(brainId = "scope-owner") {
  const root = await mkdtemp(join(tmpdir(), "omni-task-continuation-")); roots.push(root);
  const repository = { get: async (id: string) => {
    if (id !== brainId) throw new Error("Wrong current owner"); return { id };
  }, brainDirectory: () => root } as unknown as BrainRepository;
  const base = new TaskAttachmentService(repository);
  let epoch = 3;
  const facade = new TaskContinuationAttachments(repository, base, async () => epoch);
  const stage = async (name: string, turnId?: string) => {
    const file = await base.stageBytes(brainId, { requestId: `source-${name}`, data: Buffer.from(`exact ${name} bytes`),
      displayName: `${name}.txt`, kind: "file", mimeType: "text/plain",
      source: { kind: "local-file", originalPath: `/private/${name}.txt`, selectedAt: time } });
    if (turnId) await base.bindToTurn(brainId, turnId, [file.id]);
    return file;
  };
  const transfer = (predecessorTurnId = "original", successorTurnId = "successor"): WarmTaskAttachmentTransfer => ({
    brainId, predecessorTurnId, successorTurnId, predecessorInputSha256: sha(predecessorTurnId),
    successorInputSha256: sha(successorTurnId), attentionEpoch: epoch });
  const scopes = () => new TaskContinuationScopeStore(TaskAttachmentStore.databasePath(root), brainId);
  return { root, brainId, repository, base, facade, stage, transfer, scopes, setEpoch: (value: number) => { epoch = value; } };
}

function response(turnId: string, text: string, steered = false): ChatResult {
  return { brain: { messages: [] } as unknown as ChatResult["brain"],
    humanMessage: { id: `human-${turnId}`, turnId, role: "human", content: text, createdAt: time },
    brainMessage: { id: `brain-${turnId}`, turnId, role: "brain", content: "bounded fixture", createdAt: time },
    trace: { id: `trace-${turnId}` } as ChatResult["trace"], ...(steered ? { generationEnd: "steered" as const } : {}) };
}

describe("trusted task attachment continuation scopes", () => {
  it("reads original bound files across a warm scope without rebinding, copying blobs, or changing source identity", async () => {
    const f = await fixture(), file = await f.stage("original", "original");
    const before = await f.base.resolveForTurn(f.brainId, "original", file.id);
    const blobs = await readdir(join(f.root, "attachments", "blobs"));
    await f.facade.transfer(f.transfer(), new AbortController().signal);
    expect(await f.facade.listForTurn(f.brainId, "successor")).toEqual([before]);
    expect(await f.facade.readForTask(f.brainId, "successor", file.id)).toMatchObject({ content: "exact original bytes",
      attachment: { id: file.id, turnId: "original", manifestSha256: before.manifestSha256, ingestionStarted: false, parametersChanged: false } });
    expect(await f.base.listForTurn(f.brainId, "successor")).toEqual([]);
    expect(await f.base.resolveForTurn(f.brainId, "original", file.id)).toEqual(before);
    expect(await readdir(join(f.root, "attachments", "blobs"))).toEqual(blobs);
    expect(await validateTaskAttachmentWorkspace(f.root, f.brainId)).toBe(true);
  });

  it("excludes unrelated history and unsent selections while retaining the successor's independently bound new file", async () => {
    const f = await fixture(), original = await f.stage("old", "original"), unrelated = await f.stage("history", "unrelated"),
      unsent = await f.stage("unsent"), own = await f.stage("new", "successor");
    await f.facade.transfer(f.transfer(), new AbortController().signal);
    const listed = await f.facade.execute(f.brainId, "successor", "list", {}, new AbortController().signal) as { attachments: Array<{ id: string }> };
    expect(listed.attachments.map(file => file.id).sort()).toEqual([original.id, own.id].sort());
    for (const file of [unrelated, unsent]) await expect(f.facade.readForTask(f.brainId, "successor", file.id)).rejects.toThrow(/outside the actual continued task/);
    await expect(f.facade.execute(f.brainId, undefined, "read", { attachmentId: original.id }, new AbortController().signal)).rejects.toThrow(/actual owning/);
    await expect(f.facade.execute(f.brainId, "successor", "read", { attachmentId: original.id, turnId: "unrelated" }, new AbortController().signal)).resolves.toMatchObject({ content: "exact old bytes" });
  });

  it("keeps transitive original bindings without duplicate files when the successor is genuinely steered again", async () => {
    const f = await fixture(), first = await f.stage("first", "original"), second = await f.stage("second", "successor");
    await f.facade.transfer(f.transfer(), new AbortController().signal);
    await f.facade.transfer(f.transfer("successor", "third"), new AbortController().signal);
    const files = await f.facade.listForTurn(f.brainId, "third");
    expect(files.map(file => [file.attachmentId, "turnId" in file ? file.turnId : undefined])).toEqual([[first.id, "original"], [second.id, "successor"]]);
    expect(f.scopes().snapshot("third")?.record.predecessorScopeSha256).toBe(f.scopes().snapshot("successor")?.scopeSha256);
    await expect(f.facade.readForTask(f.brainId, "third", first.id)).resolves.toMatchObject({ content: "exact first bytes" });
  });

  it("returns the same durable scope after lost ACK, rejecting changed inputs, reused successors, and cross-brain claims", async () => {
    const f = await fixture(); await f.stage("one", "original");
    await f.facade.transfer(f.transfer(), new AbortController().signal);
    const before = f.scopes().snapshot("successor");
    await f.facade.transfer(f.transfer(), new AbortController().signal);
    expect(f.scopes().snapshot("successor")).toEqual(before);
    await expect(f.facade.validateCurrentInput(f.brainId, "successor", sha("different"))).rejects.toThrow(/different immutable current input/);
    expect(() => f.scopes().transfer({ ...f.transfer(), successorInputSha256: sha("different") })).toThrow(/reused/);
    expect(() => f.scopes().transfer({ ...f.transfer(), brainId: "foreign" })).toThrow(/cross current owners/);
    expect(() => f.scopes().transfer({ ...f.transfer(), predecessorTurnId: undefined } as unknown as WarmTaskAttachmentTransfer)).toThrow(/identities/);
  });

  it("registers scope only after an actual warm yield, not before the predecessor's neural boundary", async () => {
    const f = await fixture(), file = await f.stage("controller", "original");
    let finish!: (value: ChatResult) => void;
    const pending = new Promise<ChatResult>(resolve => { finish = resolve; });
    let steered = false;
    const service = { currentChatAttentionEpoch: async () => 3,
      steerChat: vi.fn(async () => { steered = true; }),
      chat: vi.fn(async (_brain: string, input: string, _signal?: AbortSignal, _stream?: (event: NeuralChatStreamEvent) => void, turnId?: string) => {
        if (turnId === "original") return pending;
        expect(f.scopes().snapshot("successor")?.record.successorInputSha256).toBe(sha("new direction"));
        await expect(f.facade.readForTask(f.brainId, "successor", file.id)).resolves.toMatchObject({ content: "exact controller bytes" });
        return response(turnId!, input);
      }) };
    const controller = new ChatActionController(service, { execute: vi.fn(), cancel: vi.fn(() => 0) }, { start: vi.fn() });
    const transfer = vi.fn((request: WarmTaskAttachmentTransfer, signal: AbortSignal) => f.facade.transfer(request, signal));
    controller.setWarmTaskAttachmentTransfer(transfer);
    const old = controller.send(f.brainId, "  original\0 ", undefined, "original");
    await vi.waitFor(() => expect(service.chat).toHaveBeenCalledTimes(1));
    const next = controller.send(f.brainId, "  new direction\0 ", undefined, "successor", { kind: "steer", replacesTurnId: "original", source: "human", createdAt: time });
    await vi.waitFor(() => expect(steered).toBe(true));
    expect(transfer).not.toHaveBeenCalled();
    finish(response("original", "original", true));
    await next; await old;
    expect(transfer).toHaveBeenCalledOnce();
    expect(transfer.mock.calls[0]?.[0]).toMatchObject({ predecessorInputSha256: sha("original"), successorInputSha256: sha("new direction"), attentionEpoch: 3 });
  });

  it("does not turn a normal completed predecessor or a forged replacement ID into a scope", async () => {
    const f = await fixture(); await f.stage("completed", "original");
    let finish!: (value: ChatResult) => void;
    const pending = new Promise<ChatResult>(resolve => { finish = resolve; });
    const service = { currentChatAttentionEpoch: async () => 3,
      steerChat: vi.fn(async () => { finish(response("original", "original")); }),
      chat: vi.fn(async (_brain: string, input: string, _signal?: AbortSignal, _stream?: (event: NeuralChatStreamEvent) => void, turnId?: string) => turnId === "original" ? pending : response(turnId!, input)) };
    const controller = new ChatActionController(service, { execute: vi.fn(), cancel: vi.fn(() => 0) }, { start: vi.fn() });
    const transfer = vi.fn((request: WarmTaskAttachmentTransfer, signal: AbortSignal) => f.facade.transfer(request, signal));
    controller.setWarmTaskAttachmentTransfer(transfer);
    const old = controller.send(f.brainId, "original", undefined, "original");
    await vi.waitFor(() => expect(service.chat).toHaveBeenCalledOnce());
    await controller.send(f.brainId, "successor", undefined, "successor", { kind: "steer", replacesTurnId: "original", source: "human", createdAt: time });
    await old;
    await controller.send(f.brainId, "forged", undefined, "forged", { kind: "steer", replacesTurnId: "unrelated", source: "human", createdAt: time });
    expect(transfer).not.toHaveBeenCalled();
    expect(await f.facade.listForTurn(f.brainId, "successor")).toEqual([]);
  });

  it("carries already-bound original files through a genuine zero-token yield without granting unsent files", async () => {
    const f = await fixture(), file = await f.stage("early", "original"), unsent = await f.stage("not-sent");
    let yieldEarly!: (error: unknown) => void;
    const pending = new Promise<ChatResult>((_resolve, reject) => { yieldEarly = reject; });
    const service = { currentChatAttentionEpoch: async () => 3,
      steerChat: vi.fn(async () => { yieldEarly(new EngineRequestError("warm yield", -32801,
        { brainId: f.brainId, turnId: "original", steered: true, zeroTokenYield: true, safeBoundary: true, warm: true })); }),
      chat: vi.fn(async (_brain: string, input: string, _signal?: AbortSignal, _stream?: (event: NeuralChatStreamEvent) => void, turnId?: string) => turnId === "original" ? pending : response(turnId!, input)) };
    const controller = new ChatActionController(service, { execute: vi.fn(), cancel: vi.fn(() => 0) }, { start: vi.fn() });
    controller.setWarmTaskAttachmentTransfer((request, signal) => f.facade.transfer(request, signal));
    const old = controller.send(f.brainId, "original", undefined, "original").catch(error => error);
    await vi.waitFor(() => expect(service.chat).toHaveBeenCalledOnce());
    await controller.send(f.brainId, "successor", undefined, "successor", { kind: "steer", replacesTurnId: "original", source: "human", createdAt: time });
    await expect(old).resolves.toMatchObject({ code: -32801 });
    await expect(f.facade.readForTask(f.brainId, "successor", file.id)).resolves.toMatchObject({ content: "exact early bytes" });
    await expect(f.facade.readForTask(f.brainId, "successor", unsent.id)).rejects.toThrow(/outside/);
  });

  it("withholds bytes if Fresh or a durable scope revision changes while the delegated read awaits", async () => {
    const f = await fixture(), file = await f.stage("waiting", "original");
    await f.facade.transfer(f.transfer(), new AbortController().signal);
    const originalRead = f.base.readForTurn.bind(f.base);
    let entered!: () => void, release!: () => void;
    const started = new Promise<void>(resolve => { entered = resolve; }), waiting = new Promise<void>(resolve => { release = resolve; });
    vi.spyOn(f.base, "readForTurn").mockImplementationOnce(async (...args) => {
      const result = await originalRead(...args); entered(); await waiting; return result;
    });
    const reading = f.facade.readForTask(f.brainId, "successor", file.id);
    await started; f.setEpoch(4); release();
    await expect(reading).rejects.toThrow(/Fresh attention changed/);
    expect(f.scopes().snapshot("successor")?.state).toBe("retired");
    await expect(f.facade.readForTask(f.brainId, "successor", file.id)).rejects.toThrow(/Fresh attention retired/);
  });

  it("withholds delegated bytes after a same-epoch dormant/revision change, independently of original FD identity", async () => {
    const f = await fixture(), file = await f.stage("revoked", "original");
    await f.facade.transfer(f.transfer(), new AbortController().signal);
    const originalRead = f.base.readForTurn.bind(f.base);
    vi.spyOn(f.base, "readForTurn").mockImplementationOnce(async (...args) => {
      const result = await originalRead(...args), database = new DatabaseSync(TaskAttachmentStore.databasePath(f.root));
      try { database.exec("UPDATE task_continuation_scopes SET state='dormant',revision=revision+1"); } finally { database.close(); }
      return result;
    });
    await expect(f.facade.readForTask(f.brainId, "successor", file.id)).rejects.toThrow(/changed or was revoked/);
  });

  it("projects verified mixed owning-turn handles without rewriting their manifests or inventing a new binding", async () => {
    const f = await fixture(), original = await f.stage("projection-old", "original"), own = await f.stage("projection-own", "successor");
    await f.facade.transfer(f.transfer(), new AbortController().signal);
    const handles = await f.facade.handlesForTurn(f.brainId, "successor"), message = response("successor", "successor").humanMessage;
    expect(f.facade.projectMessage(message, handles)).toMatchObject({ turnId: "successor", taskAttachmentIds: [own.id, original.id] });
    expect(handles.find(file => file.id === original.id)?.turnId).toBe("original");
    expect(() => f.facade.projectMessage({ ...message, turnId: "different" }, handles)).toThrow(/provenance/);
    expect(() => f.facade.projectMessage(message, [...handles])).toThrow(/another or unbound turn/);
  });

  it("preserves immutable source records and raw bytes in export/adoption, dormant until an exact retained proof", async () => {
    const f = await fixture(), file = await f.stage("portable", "original");
    await f.facade.transfer(f.transfer(), new AbortController().signal);
    const sourceScope = f.scopes().snapshot("successor")!, originalBlob = await readFile(join(f.root, "attachments", "blobs", `${file.contentSha256}.data`));
    const archive = await snapshotLearningWorkspace(f.root, join(f.root, "portable-archive"), f.brainId);
    const destination = join(f.root, "portable-copy"); await mkdir(destination);
    await adoptLearningWorkspace(archive!, destination, "scope-copy");
    const copied = new TaskContinuationScopeStore(TaskAttachmentStore.databasePath(destination), "scope-copy");
    expect(copied.snapshot("successor")).toMatchObject({ state: "dormant", scopeSha256: sourceScope.scopeSha256, record: sourceScope.record });
    const repository = { get: async (id: string) => ({ id }), brainDirectory: () => destination } as unknown as BrainRepository;
    const facade = new TaskContinuationAttachments(repository, new TaskAttachmentService(repository), async () => 3);
    await expect(facade.readForTask("scope-copy", "successor", file.id)).rejects.toThrow(/exact authenticated saved execution/);
    const proof = { brainId: "scope-copy", turnId: "successor", inputSha256: sha("successor"), attentionEpoch: 3, checkpointId: "a".repeat(64) };
    await expect(facade.activateRetained({ ...proof, inputSha256: sha("different") })).rejects.toThrow(/does not prove/);
    expect(copied.snapshot("successor")?.state).toBe("dormant");
    await facade.activateRetained(proof);
    expect(copied.snapshot("successor")?.activation).toEqual(proof);
    await expect(facade.readForTask("scope-copy", "successor", file.id)).resolves.toMatchObject({ content: "exact portable bytes", attachment: { turnId: "original", brainId: "scope-copy" } });
    expect(await readFile(join(destination, "attachments", "blobs", `${file.contentSha256}.data`))).toEqual(originalBlob);
    expect(copied.snapshot("successor")?.record).toEqual(sourceScope.record);
    expect(await validateTaskAttachmentWorkspace(destination, "scope-copy")).toBe(true);
  });

  it("same-brain recovery also becomes dormant without changing source manifests, and records the actual resumed execution", async () => {
    const f = await fixture(), file = await f.stage("recovery", "original");
    await f.facade.transfer(f.transfer(), new AbortController().signal);
    const before = await f.base.resolveForTurn(f.brainId, "original", file.id), original = f.scopes().snapshot("successor")!.record;
    await rekeyTaskAttachmentWorkspace(f.root, f.brainId, f.brainId);
    expect(f.scopes().snapshot("successor")?.state).toBe("dormant");
    expect(await f.base.resolveForTurn(f.brainId, "original", file.id)).toEqual(before);
    const proof = { brainId: f.brainId, turnId: "successor", inputSha256: sha("successor"), attentionEpoch: 3, checkpointId: "b".repeat(64) };
    await f.facade.activateRetained(proof);
    expect(f.scopes().snapshot("successor")).toMatchObject({ record: original, activation: proof, state: "active" });
    await f.facade.activateRetained(proof);
    const database = new DatabaseSync(TaskAttachmentStore.databasePath(f.root), { readOnly: true });
    try { expect(database.prepare("SELECT COUNT(*) AS n FROM task_continuation_resumes").get()?.n).toBe(1); } finally { database.close(); }
  });

  it("full merge retains every distinct source binding as dormant custody and never blindly rebinds retired files", async () => {
    const f = await fixture(); await f.stage("merge-source", "original");
    await f.facade.transfer(f.transfer(), new AbortController().signal);
    const original = f.scopes().snapshot("successor")!;
    const archive = await snapshotLearningWorkspace(f.root, join(f.root, "merge-archive"), f.brainId);
    const stageA = join(f.root, "stage-a"), stageB = join(f.root, "stage-b"), target = join(f.root, "merged");
    for (const path of [stageA, stageB, target]) await mkdir(path);
    await adoptLearningWorkspace(archive!, stageA, "merged-owner"); await adoptLearningWorkspace(archive!, stageB, "merged-owner");
    await mergeTaskAttachmentWorkspace(stageA, target, "merged-owner", "parent-a", undefined, new Map([["successor", "root-turn-a"]]));
    await mergeTaskAttachmentWorkspace(stageB, target, "merged-owner", "parent-b", undefined, new Map([["successor", "root-turn-b"]]));
    const merged = new TaskContinuationScopeStore(TaskAttachmentStore.databasePath(target), "merged-owner");
    const a = merged.snapshot("root-turn-a")!, b = merged.snapshot("root-turn-b")!;
    expect(a).toMatchObject({ state: "dormant", record: original.record, scopeSha256: original.scopeSha256 });
    expect(b).toMatchObject({ state: "dormant", record: original.record, scopeSha256: original.scopeSha256 });
    expect(a.scopeId).not.toBe(b.scopeId); expect(a.aliases[0]!.attachmentId).not.toBe(b.aliases[0]!.attachmentId);
    expect(a.aliases[0]!.owningTurnId).toBeNull();
    expect(() => merged.activateRetained({ brainId: "merged-owner", turnId: "root-turn-a", inputSha256: sha("successor"), attentionEpoch: 3, checkpointId: "c".repeat(64) })).toThrow(/does not prove/);
    expect(await validateTaskAttachmentWorkspace(target, "merged-owner")).toBe(true);
  });

  it("rejects a tampered alias or immutable source instead of widening scope to same-content unrelated history", async () => {
    const f = await fixture(), original = await f.stage("target", "original"), unrelated = await f.stage("other", "unrelated");
    await f.facade.transfer(f.transfer(), new AbortController().signal);
    const database = new DatabaseSync(TaskAttachmentStore.databasePath(f.root));
    try { database.prepare("UPDATE task_continuation_files SET attachment_id=? WHERE attachment_id=?").run(unrelated.id, original.id); } finally { database.close(); }
    await expect(validateTaskAttachmentWorkspace(f.root, f.brainId)).rejects.toThrow(/original file binding|custody/);
    await expect(f.facade.readForTask(f.brainId, "successor", unrelated.id)).rejects.toThrow(/original file binding|custody/);
  });

  it("does not turn an inherited file scope into a ToolExecutor permission grant", async () => {
    const f = await fixture(), file = await f.stage("permission", "original");
    await f.facade.transfer(f.transfer(), new AbortController().signal);
    const service = { listToolPermissions: async () => [{ toolId: "task.attachments", level: "off", updatedAt: time, revision: 1 }] } as unknown as BrainService;
    const executor = new ToolExecutor(service, {} as RuntimeJobManager);
    executor.setTaskAttachmentService(f.facade);
    vi.spyOn(executor as unknown as { audit(...args: unknown[]): Promise<void> }, "audit").mockResolvedValue(undefined);
    const dispatch = vi.spyOn(f.facade, "execute");
    await expect(executor.execute({ brainId: f.brainId, toolId: "task.attachments", action: "read", arguments: { attachmentId: file.id } },
      undefined, "scope-permission", undefined, { actionEventId: "scope-action", chatTurnId: "successor" }))
      .resolves.toMatchObject({ state: "failed", error: expect.stringMatching(/disabled for the current brain/) });
    expect(dispatch).not.toHaveBeenCalled();
  });

  it("keeps a sealed native-audio input and its original capture identity unchanged while transferring file visibility", async () => {
    const f = await fixture(), wave = pcmWave([new Float32Array([0, .2, -.2, 0])], 16000);
    const lease = vi.fn(() => ({ permissionRevision: 7, maxBytes: 1024 }));
    const audio = await prepareNativeAudioChat(f.base, lease, f.brainId, "original", { format: "omni-native-audio-chat", version: 1,
      captureSessionId: "fixture-lease", utteranceId: "fixture-utterance", sequence: 0, mimeType: "audio/wav",
      dataBase64: Buffer.from(wave).toString("base64"), capturedAt: time });
    const originalIdentity = JSON.stringify(audio), leaseCalls = lease.mock.calls.length;
    let finish!: (value: ChatResult) => void;
    const pending = new Promise<ChatResult>(resolve => { finish = resolve; });
    const service = { currentChatAttentionEpoch: async () => 3,
      steerChat: vi.fn(async () => { finish(response("original", "", true)); }),
      chat: vi.fn(async (_brain: string, input: string, _signal?: AbortSignal, _stream?: (event: NeuralChatStreamEvent) => void, turnId?: string) => turnId === "original" ? pending : response(turnId!, input)) };
    const controller = new ChatActionController(service, { execute: vi.fn(), cancel: vi.fn(() => 0) }, { start: vi.fn() });
    controller.setWarmTaskAttachmentTransfer((request, signal) => f.facade.transfer(request, signal));
    const old = controller.send(f.brainId, "", undefined, "original", undefined, undefined, undefined, audio);
    await vi.waitFor(() => expect(service.chat).toHaveBeenCalledOnce());
    await controller.send(f.brainId, "successor", undefined, "successor", { kind: "steer", replacesTurnId: "original", source: "human", createdAt: time }); await old;
    expect(f.scopes().snapshot("successor")?.record.predecessorInputSha256).toBe(audio.inputSha256);
    expect(JSON.stringify(audio)).toBe(originalIdentity);
    expect(lease.mock.calls.length).toBe(leaseCalls); // File visibility is NOT a microphone regrant or audio admission replay.
    const read = await f.facade.readForTask(f.brainId, "successor", audio.attachment.attachmentId);
    expect(read.attachment.turnId).toBe("original"); expect(read.attachment.manifestSha256).toBe(audio.attachment.manifestSha256);
    expect(read.encoding).toBe("base64"); expect(read.content).toBe(Buffer.from(wave).toString("base64"));
  });
});
