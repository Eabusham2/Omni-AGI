/** Actual closed host files/SQLite with synthetic native-context callbacks. No brain/model/decoder is constructed. */
import { createHash } from "node:crypto";
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TaskAttachmentStore } from "../src/main/taskAttachmentStore";
import { TaskContinuationScopeStore } from "../src/main/taskContinuationScope";
import { snapshotLearningWorkspace, adoptLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { mergeTaskAttachmentWorkspace, rekeyTaskAttachmentWorkspace, validateTaskAttachmentWorkspace } from "../src/main/taskAttachmentPortability";
import { InheritedTaskFiles } from "../src/main/inheritedTaskFiles";
import { InheritedTaskFileReviewStore, hasInheritedTaskFileReviews } from "../src/main/inheritedTaskFileReviewStore";
import { inheritedTaskFileClosure, inheritedTaskFileHash, type InheritedTaskFileNativeContext } from "../src/main/inheritedTaskFileClosure";
import { TaskAttachmentReadVerifier } from "../src/main/taskAttachmentReadVerifier";
import type { ToolPermissionRecord } from "../src/shared/types";
import { BrainService } from "../src/main/brainService";
import { chatTurnKey } from "../src/shared/chatTurnKey";
import { ToolExecutor } from "../src/main/toolExecutor";
import type { RuntimeJobManager } from "../src/main/brainService";

const roots: string[] = [], time = "2026-10-13T00:00:00.000Z";
const sha = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");
afterEach(async () => { vi.restoreAllMocks(); for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });

async function fixture(warm = false, permissionLevel: ToolPermissionRecord["level"] = "ask") {
  const temporary = await mkdtemp(join(tmpdir(), "omni-inherited-file-review-")); roots.push(temporary);
  const sourceBrainId = "source-parent", brainId = "destination-owner";
  const source = join(temporary, "source"), root = join(temporary, brainId);
  await Promise.all([mkdir(join(source, "engine"), { recursive: true }), mkdir(join(root, "engine"), { recursive: true })]);
  const sourceStore = new TaskAttachmentStore(source, sourceBrainId);
  const stage = async (name: string, turnId?: string) => {
    const handle = await sourceStore.stageBytes(`source-${name}`, { data: Buffer.from(`original ${name} bytes`), displayName: `${name}.txt`, kind: "file", mimeType: "text/plain",
      source: { kind: "local-file", originalPath: `/private/original/${name}.txt`, selectedAt: time } });
    if (turnId) await sourceStore.bindToTurn(turnId, [handle.id]); return handle;
  };
  const file = await stage("task", "source-task"), history = await stage("history", "other-task"), unsent = await stage("unsent");
  let earlier: Awaited<ReturnType<typeof stage>> | undefined;
  if (warm) {
    earlier = await stage("earlier", "earlier-task");
    new TaskContinuationScopeStore(TaskAttachmentStore.databasePath(source), sourceBrainId).transfer({ brainId: sourceBrainId,
      predecessorTurnId: "earlier-task", successorTurnId: "source-task", predecessorInputSha256: sha("earlier-task"), successorInputSha256: sha("source-task"), attentionEpoch: 3 });
  }
  const sourceMetadata = Buffer.from(JSON.stringify({ brain_id: sourceBrainId })); await writeFile(join(source, "engine", "brain.json"), sourceMetadata);
  const archive = await snapshotLearningWorkspace(source, join(temporary, "sealed-workspace"), sourceBrainId);
  const stageRoot = join(temporary, "stage"); await mkdir(stageRoot);
  await adoptLearningWorkspace(archive!, stageRoot, brainId);
  await mergeTaskAttachmentWorkspace(stageRoot, root, brainId, sourceBrainId, undefined, new Map([["source-task", "root-task"]]));
  const parent = join(root, "merge", "parents", sourceBrainId); await mkdir(join(parent, "engine"), { recursive: true });
  await writeFile(join(parent, "engine", "brain.json"), sourceMetadata);
  await cp(archive!.root, join(parent, "portable-learning-workspace"), { recursive: true });
  const inventory = await readFile(join(parent, "portable-learning-workspace", "inventory.sqlite3"));
  const generationSha256 = sha("synthetic registered generation"), descriptorSha256 = sha("synthetic registered descriptor");
  await writeFile(join(root, "merge", "parents.jsonl"), [
    { format: "omni-full-merge-parent-index", formatVersion: 1, parents: [{ brainId: sourceBrainId, generationSha256 }] },
    { parentId: sourceBrainId, path: "engine/brain.json", sha256: sha(sourceMetadata), bytes: sourceMetadata.length },
    { parentId: sourceBrainId, path: "portable-learning-workspace/inventory.sqlite3", sha256: sha(inventory), bytes: inventory.length }
  ].map(value => JSON.stringify(value)).join("\n") + "\n");
  await writeFile(join(root, "engine", "brain.json"), JSON.stringify({ brain_id: brainId, connected_regions: { sha256: descriptorSha256,
    regions: [{ regionId: "registered-region", parentId: sourceBrainId, enginePath: `merge/parents/${sourceBrainId}/engine`,
      metadataSha256: sha(sourceMetadata), generationSha256 }] } }));
  const context: InheritedTaskFileNativeContext = { brainId, executionId: sha("synthetic native execution"), frontier: { checkpointId: "c".repeat(32), turnId: "root-task",
    inputSha256: sha("source-task"), lastStreamSequence: 7, sampledTokens: 3, publishedTokens: 2,
    phase: "native-text-token-boundary", exactSupportedResume: true }, inputSha256: sha("source-task"), sourceBrainId,
    sourceTurnId: "source-task", sourceAttentionEpoch: 3, rootAttentionEpoch: 0, descriptorSha256,
    regionPath: ["registered-region"], sourceManifestSha256: sha("synthetic source manifest"), sourceBindingId: sha("synthetic source binding") };
  let active = false, epoch = 0, permissionRevision = 1;
  const permissions = async () => [{ toolId: "task.attachments", label: "Task files", level: permissionLevel, updatedAt: time, revision: permissionRevision }] as ToolPermissionRecord[];
  const nativeContext = vi.fn(async () => ({ ...structuredClone(context), rootAttentionEpoch: epoch }));
  const verifier = new TaskAttachmentReadVerifier({ strongLocalChangeTimes: async () => true });
  const options = { root: () => root, context: nativeContext, permissions, rootEpoch: async () => epoch,
    assertActive: (value: InheritedTaskFileNativeContext) => {
      if (!active || value.brainId !== brainId || value.executionId !== context.executionId || value.inputSha256 !== context.inputSha256) throw new Error("No actual current inherited execution");
    }, verifier };
  const files = new InheritedTaskFiles(options);
  const alias = (id: string) => inheritedTaskFileHash({ parentBrainId: sourceBrainId, attachmentId: id });
  const approve = async () => {
    const review = await files.prepare({ brainId, executionId: context.executionId });
    if (!review) throw new Error("Fixture has no task files");
    const receipt = await files.approve({ brainId, reviewId: review.reviewId, reviewSha256: review.reviewSha256 });
    await files.enter({ ...structuredClone(context), rootAttentionEpoch: epoch }); active = true;
    return { review, receipt };
  };
  return { temporary, source, root, parent, brainId, sourceBrainId, sourceStore, file, earlier, history, unsent, files, context, options,
    nativeContext, verifier, alias, approve, setActive: (value: boolean) => { active = value; }, setEpoch: (value: number) => { epoch = value; },
    changePermission: () => { permissionRevision++; } };
}

describe("reviewed inherited task file adoption and retired custody reads", () => {
  it("dispatches Off-only saved-file exceptions through real ToolExecutor fences, not an argument token", async () => {
    const f = await fixture(false, "off"), execute = vi.fn(async (brainId, turnId, action, args, signal) => {
      if (action !== "read") throw new Error("Fixture supports the actual saved read only");
      return f.files.read(brainId, turnId, args.attachmentId, 0, 64, signal);
    });
    const service = { repository: { brainDirectory: () => f.root }, listToolPermissions: f.options.permissions,
      authorizeInlineImagination: vi.fn(async () => undefined), queueCompletedToolExecution: vi.fn(async () => undefined) } as unknown as BrainService;
    const tools = new ToolExecutor(service, {} as RuntimeJobManager, undefined, { execute: vi.fn() });
    vi.spyOn(tools as unknown as { audit(): Promise<void> }, "audit").mockResolvedValue(undefined);
    tools.setTaskAttachmentService({ execute, reviewedAccess: (...args) => f.files.reviewedAccess(...args) });
    const invocation = { brainId: f.brainId, toolId: "task.attachments", action: "read", arguments: { attachmentId: f.alias(f.file.id) } };
    const context = { actionEventId: "saved-file-read", chatTurnId: "root-task" };
    try {
      expect((await tools.execute(invocation, undefined, "before-review", undefined, context)).state).toBe("failed");
      const { review } = await f.approve();
      const result = await tools.execute(invocation, undefined, "after-review", undefined, context);
      expect(result.error).toBeUndefined();
      expect(result).toMatchObject({ state: "complete", output: { content: "original task bytes" } });
      expect(execute).toHaveBeenCalledTimes(1);
      expect((await tools.execute({ ...invocation, arguments: { ...invocation.arguments, chatTurnId: "root-task", reviewSha256: review.reviewSha256 } },
        undefined, "forged-arguments")).state).toBe("failed");
      expect((await tools.execute({ ...invocation, toolId: "system.files", action: "read", arguments: { path: "/private/original/task.txt" } },
        undefined, "host-path", undefined, context)).state).toBe("failed");
      await f.files.revoke({ brainId: f.brainId, reviewId: review.reviewId, reviewSha256: review.reviewSha256 });
      expect((await tools.execute(invocation, undefined, "after-revoke", undefined, context)).state).toBe("failed");
      expect(execute).toHaveBeenCalledTimes(1);
    } finally { tools.cancel(f.brainId); }
  });
  it("permits only explicitly reviewed saved copies while Off without opening original paths or other history", async () => {
    const f = await fixture(false, "off");
    expect(await f.files.reviewedAccess(f.brainId, "root-task", "read", { attachmentId: f.alias(f.file.id) })).toBeUndefined();
    const { review, receipt } = await f.approve();
    expect(review.permission.level).toBe("off"); expect(receipt.permissionsChanged).toBe(false);
    const access = await f.files.reviewedAccess(f.brainId, "root-task", "read", { attachmentId: f.alias(f.file.id) });
    expect(access?.revision).toMatch(/^[a-f0-9]{64}$/);
    await expect(f.files.read(f.brainId, "root-task", f.alias(f.file.id))).resolves.toMatchObject({ content: "original task bytes" });
    expect(await f.files.reviewedAccess(f.brainId, "root-task", "read", { attachmentId: f.alias(f.history.id) })).toBeUndefined();
    expect(await f.files.reviewedAccess(f.brainId, "root-task", "write", { path: "/private/original/task.txt" })).toBeUndefined();
    expect(await f.files.reviewedAccess(f.brainId, "different-task", "read", { attachmentId: f.alias(f.file.id) })).toBeUndefined();
    f.changePermission();
    await expect(f.files.reviewedAccess(f.brainId, "root-task", "read", { attachmentId: f.alias(f.file.id) })).rejects.toThrow(/current file permission/);
  });
  it("prepares without granting or writing review rows, disclosing only the real original task and no private path", async () => {
    const f = await fixture(), review = await f.files.prepare({ brainId: f.brainId, executionId: f.context.executionId });
    expect(review?.state).toBe("review-required"); expect(review?.files.map(file => file.attachmentId)).toEqual([f.alias(f.file.id)]);
    expect(JSON.stringify(review)).not.toContain("/private/original"); expect(review?.sourceAttentionEpoch).toBe(3); expect(review?.rootAttentionEpoch).toBe(0);
    const database = new DatabaseSync(TaskAttachmentStore.databasePath(f.root), { readOnly: true });
    try { expect(hasInheritedTaskFileReviews(database)).toBe(false); expect(database.prepare("SELECT COUNT(*) AS n FROM attachments WHERE state='bound'").get()?.n).toBe(0); } finally { database.close(); }
    await expect(f.files.enter(f.context)).rejects.toMatchObject({ code: "INHERITED_TASK_FILE_ADOPTION_REQUIRED" });
    await expect(f.files.read(f.brainId, "root-task", f.alias(f.file.id))).rejects.toThrow(/not disclosed and adopted/);
  });

  it("explicit approval reads exact retired bytes while original bindings/rows stay unchanged and no inline native RPC occurs", async () => {
    const f = await fixture(), before = await readFile(TaskAttachmentStore.databasePath(f.source));
    const { review } = await f.approve(), nativeCalls = f.nativeContext.mock.calls.length;
    const result = await f.files.read(f.brainId, "root-task", f.alias(f.file.id), 0, 10);
    expect(result).toMatchObject({ content: "original t", encoding: "utf8", attachment: { id: f.alias(f.file.id), state: "retired", ingestionStarted: false, parametersChanged: false },
      inheritedTask: { originalBinding: { brainId: f.sourceBrainId, turnId: "source-task", attachmentId: f.file.id }, custody: "retired", reviewSha256: review.reviewSha256 } });
    expect(result.attachment).not.toHaveProperty("turnId"); expect(f.nativeContext.mock.calls.length).toBe(nativeCalls);
    expect(await readFile(TaskAttachmentStore.databasePath(f.source))).toEqual(before);
    const db = new DatabaseSync(TaskAttachmentStore.databasePath(f.root), { readOnly: true });
    try { expect(db.prepare("SELECT state,turn_id FROM attachments WHERE id=?").get(f.alias(f.file.id))).toEqual({ state: "retired", turn_id: null }); } finally { db.close(); }
    await expect(f.files.read(f.brainId, "root-task", f.alias(f.history.id))).rejects.toThrow(/not disclosed/);
    await expect(f.files.read(f.brainId, "root-task", f.alias(f.unsent.id))).rejects.toThrow(/not disclosed/);
    expect(await validateTaskAttachmentWorkspace(f.root, f.brainId)).toBe(true);
  });

  it("combines only the source task's own and authentically inherited warm files from the closed original registry", async () => {
    const f = await fixture(true), { review } = await f.approve();
    expect(review.files.map(file => file.attachmentId).sort()).toEqual([f.alias(f.file.id), f.alias(f.earlier!.id)].sort());
    await expect(f.files.read(f.brainId, "root-task", f.alias(f.earlier!.id))).resolves.toMatchObject({ content: "original earlier bytes",
      inheritedTask: { originalBinding: { turnId: "earlier-task" } } });
    expect((await f.files.references(f.brainId, "root-task"))[0]).toMatchObject({ format: "omni-inherited-task-file-reference", custody: "retired" });
  });

  it("rejects changed native checkpoint, current policy revision, and review hash before adoption", async () => {
    const f = await fixture(), review = (await f.files.prepare({ brainId: f.brainId, executionId: f.context.executionId }))!;
    await expect(f.files.approve({ brainId: f.brainId, reviewId: review.reviewId, reviewSha256: "0".repeat(64) })).rejects.toThrow(/unavailable or changed/);
    f.context.frontier.checkpointId = "d".repeat(32);
    await expect(f.files.approve({ brainId: f.brainId, reviewId: review.reviewId, reviewSha256: review.reviewSha256 })).rejects.toThrow(/changed while review/);
    f.context.frontier.checkpointId = "c".repeat(32); f.changePermission();
    await expect(f.files.approve({ brainId: f.brainId, reviewId: review.reviewId, reviewSha256: review.reviewSha256 })).rejects.toThrow(/changed while review/);
  });

  it("recovers an exact lost approval ACK without duplicate receipts or restoring a changed grant", async () => {
    const f = await fixture(), review = (await f.files.prepare({ brainId: f.brainId, executionId: f.context.executionId }))!;
    const request = { brainId: f.brainId, reviewId: review.reviewId, reviewSha256: review.reviewSha256 };
    const first = await f.files.approve(request); expect(await f.files.approve(request)).toEqual(first);
    expect((await f.files.prepare({ brainId: f.brainId, executionId: f.context.executionId }))?.state).toBe("approved");
    const db = new DatabaseSync(TaskAttachmentStore.databasePath(f.root), { readOnly: true });
    try { expect(db.prepare("SELECT COUNT(*) AS n FROM inherited_task_file_reviews").get()?.n).toBe(1); } finally { db.close(); }
    f.changePermission(); await expect(f.files.approve(request)).rejects.toThrow(/changed while review/);
  });

  it("withholds bytes after approval revocation during an awaited exact-FD operation", async () => {
    const f = await fixture(), { review } = await f.approve(), real = f.verifier.withVerifiedFile.bind(f.verifier);
    vi.spyOn(f.verifier, "withVerifiedFile").mockImplementationOnce(async (...args) => {
      const result = await real(...args); await f.files.revoke({ brainId: f.brainId, reviewId: review.reviewId, reviewSha256: review.reviewSha256 }); return result;
    });
    await expect(f.files.read(f.brainId, "root-task", f.alias(f.file.id))).rejects.toThrow(/lease has ended/);
  });

  it("rejects stopped/unloaded task ownership, Fresh, and changed current permission without native writer re-entry", async () => {
    const f = await fixture(); await f.approve(); const calls = f.nativeContext.mock.calls.length;
    f.setActive(false); await expect(f.files.read(f.brainId, "root-task", f.alias(f.file.id))).rejects.toThrow(/actual current inherited/);
    f.setActive(true); f.setEpoch(1); await expect(f.files.read(f.brainId, "root-task", f.alias(f.file.id))).rejects.toThrow(/Fresh attention/);
    f.setEpoch(0); f.changePermission(); await expect(f.files.read(f.brainId, "root-task", f.alias(f.file.id))).rejects.toThrow(/current file permission/);
    expect(f.nativeContext.mock.calls.length).toBe(calls);
  });

  it("rejects a changed immutable original blob instead of silently trusting matching manifest metadata", async () => {
    const f = await fixture(), path = join(f.parent, "portable-learning-workspace", "blobs", `${f.file.contentSha256}.data`);
    await writeFile(path, Buffer.from("x".repeat(f.file.bytes)));
    await expect(f.files.prepare({ brainId: f.brainId, executionId: f.context.executionId })).rejects.toThrow(/integrity validation/);
  });

  it("preserves original accepted source turn and audio/text SHA during safe retired-handle projection", async () => {
    const f = await fixture(); await f.approve(); const handles = (await f.files.handles(f.brainId, "root-task"))!;
    const message = { id: "saved-original-human", role: "human" as const, turnId: "source-task", content: "source-task", createdAt: time,
      inputReceipt: { committed: true as const, turnId: "source-task", inputSha256: f.context.inputSha256, humanMessageId: "saved-original-human", afterimageId: "afterimage",
        createdAt: time, attentionEpoch: 3 } };
    const projected = f.files.projectMessage(message, handles)!;
    expect(projected.turnId).toBe("source-task"); expect(projected.inputReceipt?.inputSha256).toBe(message.inputReceipt.inputSha256);
    expect(projected.taskAttachmentIds).toEqual([f.alias(f.file.id)]); expect(handles[0]!.state).toBe("retired");
    expect(() => f.files.projectMessage({ ...message, turnId: "unrelated" }, handles)).toThrow(/original accepted input/);
  });

  it("copies/recovery retain approval audit as dormant without granting files or changing original review bytes", async () => {
    const f = await fixture(), { review, receipt } = await f.approve();
    const archive = await snapshotLearningWorkspace(f.root, join(f.temporary, "review-archive"), f.brainId), copyRoot = join(f.temporary, "review-copy"); await mkdir(copyRoot);
    await adoptLearningWorkspace(archive!, copyRoot, "copied-owner");
    const copied = new DatabaseSync(TaskAttachmentStore.databasePath(copyRoot), { readOnly: true });
    try { const row = copied.prepare("SELECT state,body_json,receipt_json FROM inherited_task_file_reviews").get()!;
      expect(row.state).toBe("dormant"); expect(JSON.parse(String(row.body_json)).reviewId).toBe(review.reviewId);
      expect(JSON.parse(String(row.receipt_json))).toEqual(receipt); } finally { copied.close(); }
    expect(await validateTaskAttachmentWorkspace(copyRoot, "copied-owner")).toBe(true);
    await rekeyTaskAttachmentWorkspace(f.root, f.brainId, f.brainId);
    await expect(f.files.read(f.brainId, "root-task", f.alias(f.file.id))).rejects.toThrow(/revoked or replaced/);
    expect(new InheritedTaskFileReviewStore(TaskAttachmentStore.databasePath(f.root), f.brainId).approved(review)).toBeUndefined();
  });

  it("preserves exact task files through nested merge source closures without widening to direct-parent history", async () => {
    const f = await fixture(); const { receipt } = await f.approve();
    const outer = join(f.temporary, "outer"), stage = join(f.temporary, "outer-stage"), outerBrainId = "outer-owner";
    await Promise.all([mkdir(join(outer, "engine"), { recursive: true }), mkdir(stage)]);
    const archive = await snapshotLearningWorkspace(f.root, join(f.temporary, "inner-archive"), f.brainId);
    await adoptLearningWorkspace(archive!, stage, outerBrainId);
    await mergeTaskAttachmentWorkspace(stage, outer, outerBrainId, f.brainId);
    const parent = join(outer, "merge", "parents", f.brainId); await mkdir(join(outer, "merge", "parents"), { recursive: true });
    await cp(f.root, parent, { recursive: true }); await cp(archive!.root, join(parent, "portable-learning-workspace"), { recursive: true });
    const metadata = await readFile(join(parent, "engine", "brain.json")), inventory = await readFile(join(parent, "portable-learning-workspace", "inventory.sqlite3"));
    const generationSha256 = sha("outer synthetic generation"), descriptorSha256 = sha("outer synthetic descriptor");
    await writeFile(join(outer, "merge", "parents.jsonl"), [
      { format: "omni-full-merge-parent-index", formatVersion: 1, parents: [{ brainId: f.brainId, generationSha256 }] },
      { parentId: f.brainId, path: "engine/brain.json", sha256: sha(metadata), bytes: metadata.length },
      { parentId: f.brainId, path: "portable-learning-workspace/inventory.sqlite3", sha256: sha(inventory), bytes: inventory.length }
    ].map(row => JSON.stringify(row)).join("\n") + "\n");
    await writeFile(join(outer, "engine", "brain.json"), JSON.stringify({ brain_id: outerBrainId, connected_regions: { sha256: descriptorSha256,
      regions: [{ regionId: "outer-region", parentId: f.brainId, enginePath: `merge/parents/${f.brainId}/engine`, metadataSha256: sha(metadata), generationSha256 }] } }));
    const context = { ...structuredClone(f.context), brainId: outerBrainId, descriptorSha256, regionPath: ["outer-region", ...f.context.regionPath],
      frontier: { ...f.context.frontier, turnId: "outer-task" } };
    const closure = await inheritedTaskFileClosure(outer, context);
    expect(closure.files.map(file => file.disclosure.attachmentId)).toEqual([inheritedTaskFileHash({ parentBrainId: f.brainId, attachmentId: f.alias(f.file.id) })]);
    expect(closure.files[0]!.disclosure.originalBinding).toMatchObject({ brainId: f.sourceBrainId, turnId: "source-task", attachmentId: f.file.id });
    const db = new DatabaseSync(TaskAttachmentStore.databasePath(outer), { readOnly: true });
    try { const row = db.prepare("SELECT state,receipt_json FROM inherited_task_file_reviews").get()!;
      expect(row.state).toBe("dormant"); expect(JSON.parse(String(row.receipt_json))).toEqual(receipt); } finally { db.close(); }
    expect(await validateTaskAttachmentWorkspace(outer, outerBrainId)).toBe(true);
  });

  it("returns safe historical UI handles without exposing file bytes or granting a task lease", async () => {
    const f = await fixture(); const handles = await f.files.historicalFileHandlesForUser(f.brainId, [f.alias(f.file.id)]);
    expect(handles).toEqual([expect.objectContaining({ id: f.alias(f.file.id), state: "retired", ingestionStarted: false, parametersChanged: false })]);
    expect(JSON.stringify(handles)).not.toMatch(/source_json|relativePath|\/private\/original|original task bytes/);
    await expect(f.files.read(f.brainId, "root-task", handles[0]!.id)).rejects.toThrow(/not disclosed and adopted/);
    await expect(f.files.historicalFileHandlesForUser(f.brainId, [handles[0]!.id, handles[0]!.id])).rejects.toThrow(/Invalid historical/);
  });

  it("the production main owner hook refuses foreign, stopped, detached and unloaded execution tuples without a worker RPC", async () => {
    const f = await fixture(), main = Object.create(BrainService.prototype) as BrainService;
    const owner = { inheritedExecutionId: f.context.executionId, inputSha256: f.context.inputSha256, inputSignal: new AbortController().signal };
    const tuples = new Map([[chatTurnKey(f.brainId, "root-task"), owner]]);
    Object.defineProperty(main, "liveChatTurns", { value: tuples });
    const loaded = vi.fn(); Object.defineProperty(main, "requireInstanceLoaded", { value: loaded });
    expect(() => main.assertActiveInheritedTaskFileContext(f.context)).not.toThrow();
    expect(loaded).toHaveBeenCalledWith(f.brainId, "root-task");
    expect(() => main.assertActiveInheritedTaskFileContext({ ...f.context, executionId: sha("foreign source") })).toThrow(/current actual inherited/);
    const cancelled = new AbortController(); cancelled.abort(); owner.inputSignal = cancelled.signal;
    expect(() => main.assertActiveInheritedTaskFileContext(f.context)).toThrow(/current actual inherited/);
    owner.inputSignal = new AbortController().signal; tuples.delete(chatTurnKey(f.brainId, "root-task"));
    expect(() => main.assertActiveInheritedTaskFileContext(f.context)).toThrow(/current actual inherited/);
    tuples.set(chatTurnKey(f.brainId, "root-task"), { ...owner, detached: new Error("replacement") } as typeof owner);
    expect(() => main.assertActiveInheritedTaskFileContext(f.context)).toThrow(/current actual inherited/);
    loaded.mockImplementation(() => { throw new Error("Runtime unloaded"); });
    expect(() => main.assertActiveInheritedTaskFileContext(f.context)).toThrow(/unloaded/);
  });
});
