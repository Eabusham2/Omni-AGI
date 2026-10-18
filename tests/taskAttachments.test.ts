import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService, type RuntimeJobManager } from "../src/main/brainService";
import { ChatActionController } from "../src/main/chatActionController";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import {
  adoptLearningWorkspace,
  snapshotLearningWorkspace
} from "../src/main/learningWorkspacePortability";
import {
  mergeTaskAttachmentWorkspace,
  validateTaskAttachmentWorkspace
} from "../src/main/taskAttachmentPortability";
import { TaskAttachmentService } from "../src/main/taskAttachmentService";
import { TaskAttachmentStore } from "../src/main/taskAttachmentStore";
import { projectTaskAttachmentIds } from "../src/main/taskAttachmentProjection";
import { ToolExecutor } from "../src/main/toolExecutor";
import { DEFAULT_CONFIG, type ChatResult } from "../src/shared/types";

describe("plain current-task attachments", () => {
  const roots: string[] = [];

  afterEach(async () => {
    await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
  });

  async function harness(name: string) {
    const root = await mkdtemp(join(tmpdir(), `omni-task-attachment-${name}-`));
    roots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name, idleCognition: false });
    return { root, repository, brain, attachments: new TaskAttachmentService(repository) };
  }

  it("stages exact bytes without learning, binds one immutable task set, and pages reads", async () => {
    const fixture = await harness("Attachment boundaries");
    const source = join(fixture.root, "private-note.txt");
    await writeFile(source, "first page\nsecond page\nthird page");
    const before = await fixture.repository.get(fixture.brain.id);
    const staged = await fixture.attachments.stagePaths(
      fixture.brain.id,
      "stage-note",
      [{ path: source }],
      "files"
    );
    expect(staged).toEqual([expect.objectContaining({
      brainId: fixture.brain.id,
      displayName: "private-note.txt",
      mimeType: "text/plain",
      state: "staged",
      ingestionStarted: false,
      parametersChanged: false
    })]);
    expect(JSON.stringify(staged)).not.toContain(source);
    const afterStage = await fixture.repository.get(fixture.brain.id);
    expect(afterStage.trainingSources).toEqual(before.trainingSources);
    expect(afterStage.counters).toEqual(before.counters);
    expect(await fixture.attachments.requestHandles(fixture.brain.id, "stage-note")).toEqual(staged);
    await fixture.repository.appendChatDeliveryReceipt(fixture.brain.id, {
      schemaVersion: 1,
      turnId: "queued-note-turn",
      content: "read the attached note",
      createdAt: new Date().toISOString(),
      state: "queued",
      attachmentIds: [staged[0]!.id]
    });
    expect((await fixture.repository.conversationPage(fixture.brain.id)).entries)
      .toEqual([expect.objectContaining({ message: expect.objectContaining({
        deliveryReceipt: expect.objectContaining({ attachmentIds: [staged[0]!.id] })
      }) })]);

    const turnId = "turn-note";
    const bound = await fixture.attachments.bindToTurn(fixture.brain.id, turnId, [staged[0]!.id]);
    expect(bound).toEqual([expect.objectContaining({
      brainId: fixture.brain.id,
      turnId,
      attachmentId: staged[0]!.id,
      relativePath: expect.stringMatching(/^attachments\/blobs\/[a-f0-9]{64}\.data$/)
    })]);
    await expect(fixture.attachments.bindToTurn(fixture.brain.id, turnId, [staged[0]!.id])).resolves.toEqual(bound);
    await expect(fixture.attachments.bindToTurn(fixture.brain.id, "other-turn", [staged[0]!.id]))
      .rejects.toThrow(/unavailable/i);
    await expect(fixture.attachments.remove(fixture.brain.id, staged[0]!.id))
      .rejects.toThrow(/cannot be deleted/i);

    const first = await fixture.attachments.readForTurn(fixture.brain.id, turnId, staged[0]!.id, 0, 11);
    expect(first).toMatchObject({ encoding: "utf8", content: "first page\n", offset: 0, returnedBytes: 11, nextOffset: 11 });
    const remainder = await fixture.attachments.readForTurn(
      fixture.brain.id, turnId, staged[0]!.id, first.nextOffset, 1024
    );
    expect(remainder.content).toBe("second page\nthird page");
    await expect(fixture.attachments.readForTurn(fixture.brain.id, "wrong-turn", staged[0]!.id))
      .rejects.toThrow(/exact chat task/i);
  });

  it("cancels a raw-byte stage and retires only an unsent handle", async () => {
    const fixture = await harness("Attachment cancellation");
    const cancelled = new AbortController();
    cancelled.abort(new DOMException("cancelled", "AbortError"));
    await expect(fixture.attachments.stageBytes(fixture.brain.id, {
      requestId: "cancelled-audio",
      data: new Uint8Array([1, 2, 3, 4]),
      displayName: "capture.wav",
      mimeType: "audio/wav",
      kind: "audio",
      source: {
        kind: "captured-audio",
        captureSessionId: "capture-session",
        utteranceId: "utterance-one",
        sequence: 0,
        capturedAt: "2026-10-11T00:00:00.000Z",
        selectedAt: "2026-10-11T00:00:00.000Z"
      }
    }, cancelled.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(await fixture.attachments.requestHandles(fixture.brain.id, "cancelled-audio")).toBeUndefined();

    const source = join(fixture.root, "unsent.json");
    await writeFile(source, "{\"still\":\"not learned\"}");
    const [handle] = await fixture.attachments.stagePaths(
      fixture.brain.id, "unsent-file", [{ path: source }], "files"
    );
    await expect(fixture.attachments.remove(fixture.brain.id, handle!.id)).resolves.toBe(true);
    expect(await fixture.attachments.list(fixture.brain.id)).toEqual([]);
    const database = new DatabaseSync(TaskAttachmentStore.databasePath(fixture.repository.brainDirectory(fixture.brain.id)), { readOnly: true });
    try {
      expect(database.prepare("SELECT state FROM attachments WHERE id=?").get(handle!.id)).toBeUndefined();
      expect(database.prepare("SELECT request_id FROM requests WHERE request_id='unsent-file'").get()).toBeUndefined();
    } finally { database.close(); }
    await expect(readFile(join(
      fixture.repository.brainDirectory(fixture.brain.id),
      "attachments", "blobs", `${handle!.contentSha256}.data`
    ))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("keeps exact private source custody through the ordinary portable workspace", async () => {
    const fixture = await harness("Attachment portability");
    const source = join(fixture.root, "unsanitized-private-name.md");
    await writeFile(source, "portable attachment source");
    const [boundSource] = await fixture.attachments.stagePaths(
      fixture.brain.id, "portable-bound", [{ path: source }], "files"
    );
    await fixture.attachments.bindToTurn(fixture.brain.id, "portable-turn", [boundSource!.id]);
    const stagedPath = join(fixture.root, "pending.txt");
    await writeFile(stagedPath, "pending selection");
    await fixture.attachments.stagePaths(fixture.brain.id, "portable-pending", [{ path: stagedPath }], "files");

    const archiveRoot = join(fixture.root, "workspace-archive");
    const archive = await snapshotLearningWorkspace(
      fixture.repository.brainDirectory(fixture.brain.id), archiveRoot, fixture.brain.id
    );
    expect(archive).toBeDefined();
    const targetId = "portable-copy";
    const targetRoot = join(fixture.root, "portable-copy-root");
    await mkdir(targetRoot, { recursive: true });
    await adoptLearningWorkspace(archive!, targetRoot, targetId);
    expect(await validateTaskAttachmentWorkspace(targetRoot, targetId)).toBe(true);
    const copied = new TaskAttachmentStore(targetRoot, targetId);
    const visible = await copied.list();
    expect(visible).toEqual([expect.objectContaining({
      id: boundSource!.id,
      brainId: targetId,
      state: "bound",
      turnId: "portable-turn"
    })]);
    await expect(copied.readForTurn("portable-turn", boundSource!.id)).resolves.toMatchObject({
      content: "portable attachment source"
    });
    const database = new DatabaseSync(TaskAttachmentStore.databasePath(targetRoot), { readOnly: true });
    try {
      const sources = [...database.prepare("SELECT source_json,state FROM attachments ORDER BY request_id").all()] as Array<{ source_json: string; state: string }>;
      expect(sources.some((row) => row.source_json.includes(source))).toBe(true);
      expect(sources.some((row) => row.state === "retired")).toBe(true);
    } finally { database.close(); }
  });

  it("allows only the owning chat action context to use the normal attachment read tool", async () => {
    const fixture = await harness("Attachment tool");
    const source = join(fixture.root, "tool-note.txt");
    await writeFile(source, "read through the visible task tool");
    const [attachment] = await fixture.attachments.stagePaths(
      fixture.brain.id, "tool-stage", [{ path: source }], "files"
    );
    await fixture.attachments.bindToTurn(fixture.brain.id, "tool-turn", [attachment!.id]);
    const brainService = new BrainService(fixture.repository, {
      request: vi.fn(async () => ({})),
      requestStream: vi.fn(async () => ({ text: "unused", trace: { id: "unused" } }))
    } as unknown as EngineSupervisor);
    const executor = new ToolExecutor(brainService, { generate: vi.fn() } as unknown as RuntimeJobManager);
    executor.setTaskAttachmentService(fixture.attachments);
    vi.spyOn(executor as unknown as { audit(...args: unknown[]): Promise<void> }, "audit").mockResolvedValue(undefined);
    const invocation = {
      brainId: fixture.brain.id,
      toolId: "task.attachments",
      action: "read",
      arguments: { attachmentId: attachment!.id }
    };
    await expect(executor.execute(invocation, undefined, "attachment-read", undefined, {
      actionEventId: "attachment-action",
      chatTurnId: "tool-turn"
    })).resolves.toMatchObject({ state: "complete", output: {
      content: "read through the visible task tool",
      attachment: { id: attachment!.id, ingestionStarted: false, parametersChanged: false }
    } });
    await expect(executor.execute(invocation, undefined, "attachment-read-wrong", undefined, {
      actionEventId: "attachment-action-wrong",
      chatTurnId: "wrong-turn"
    })).resolves.toMatchObject({ state: "failed", error: expect.stringMatching(/owning chat turn|exact chat task/i) });
  });

  it("retains full-merge parent attachments only as inert attributed history", async () => {
    const fixture = await harness("Attachment merge parent");
    const source = join(fixture.root, "parent-evidence.txt");
    await writeFile(source, "parent attachment evidence");
    const [attachment] = await fixture.attachments.stagePaths(
      fixture.brain.id, "merge-parent-stage", [{ path: source }], "files"
    );
    await fixture.attachments.bindToTurn(fixture.brain.id, "parent-turn", [attachment!.id]);
    const archive = await snapshotLearningWorkspace(
      fixture.repository.brainDirectory(fixture.brain.id),
      join(fixture.root, "merge-parent-archive"),
      fixture.brain.id
    );
    const owner = "merged-child", stage = join(fixture.root, "merge-stage"), target = join(fixture.root, "merge-target");
    await Promise.all([mkdir(stage, { recursive: true }), mkdir(target, { recursive: true })]);
    await adoptLearningWorkspace(archive!, stage, owner);
    await mergeTaskAttachmentWorkspace(stage, target, owner, fixture.brain.id);
    const merged = new TaskAttachmentStore(target, owner);
    expect(await merged.list()).toEqual([]);
    expect(await merged.list(true)).toEqual([expect.objectContaining({
      brainId: owner,
      state: "retired",
      contentSha256: attachment!.contentSha256
    })]);
    const database = new DatabaseSync(TaskAttachmentStore.databasePath(target), { readOnly: true });
    try {
      expect(String(database.prepare("SELECT source_json FROM attachments").get()?.source_json))
        .toContain(fixture.brain.id);
    } finally { database.close(); }
    await expect(fixture.attachments.readForTurn(fixture.brain.id, "parent-turn", attachment!.id))
      .resolves.toMatchObject({ content: "parent attachment evidence" });
  });

  it("binds renderer task metadata before the neural chat call", async () => {
    const now = new Date().toISOString();
    const result: ChatResult = {
      brain: { messages: [] } as unknown as ChatResult["brain"],
      humanMessage: { id: "human", role: "human", content: "read it", createdAt: now },
      brainMessage: { id: "brain", role: "brain", content: "ready", createdAt: now },
      trace: { id: "trace" } as ChatResult["trace"],
      proposedActions: []
    };
    const order: string[] = [];
    const service = {
      bindTaskAttachments: vi.fn(async () => { order.push("bind"); }),
      chat: vi.fn(async () => { order.push("chat"); return result; })
    };
    const controller = new ChatActionController(
      service,
      { execute: vi.fn(), cancel: vi.fn(() => 0) },
      { start: vi.fn() }
    );
    await controller.send("brain-one", "read it", undefined, "turn-one", {
      kind: "task",
      source: "human",
      createdAt: now,
      attachmentIds: ["attachment-one"]
    });
    expect(order).toEqual(["bind", "chat"]);
    expect(service.bindTaskAttachments).toHaveBeenCalledWith(
      "brain-one", "turn-one", ["attachment-one"]
    );
    const projected = projectTaskAttachmentIds({
      id: "human-one",
      role: "human",
      content: "read it",
      createdAt: now,
      inputReceipt: {
        committed: true,
        turnId: "turn-one",
        inputSha256: "a".repeat(64),
        humanMessageId: "human-one",
        afterimageId: "afterimage-one",
        createdAt: now,
        attentionEpoch: 0
      }
    }, [{
      format: "omni-chat-task-attachment",
      version: 1,
      id: "attachment-one",
      brainId: "brain-one",
      displayName: "note.txt",
      kind: "file",
      mimeType: "text/plain",
      contentSha256: "b".repeat(64),
      manifestSha256: "c".repeat(64),
      bytes: 4,
      state: "bound",
      turnId: "turn-one",
      createdAt: now,
      updatedAt: now,
      ingestionStarted: false,
      parametersChanged: false
    }]);
    expect(projected).toMatchObject({
      taskAttachmentIds: ["attachment-one"],
      inputReceipt: { taskAttachmentIds: ["attachment-one"] }
    });
    expect(JSON.stringify(projected)).not.toMatch(/relativePath|originalPath/);
  });
});
