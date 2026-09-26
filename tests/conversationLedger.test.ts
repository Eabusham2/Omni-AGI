import { randomUUID } from "node:crypto";
import { copyFile, mkdir, readFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BrainActivityLedger } from "../src/main/brainActivityLedger";
import { BrainRepository } from "../src/main/brainRepository";
import { ConversationLedger } from "../src/main/conversationLedger";
import { DEFAULT_CONFIG, type ActionEvent, type ChatMessage } from "../src/shared/types";

describe("append-only paged conversation ledger", () => {
  let root: string;
  let repository: BrainRepository;

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "omni-conversation-ledger-"));
    repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
  });

  afterEach(async () => {
    await rm(root, { recursive: true, force: true });
  });

  it("backfills every legacy message, keeps brain.json compact, and pages losslessly", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Long history" });
    const messages: ChatMessage[] = Array.from({ length: 2_000 }, (_, index) => ({
      id: randomUUID(),
      role: index % 2 ? "brain" : "human",
      content: `message-${index}-${"x".repeat(128)}`,
      createdAt: new Date(1_700_000_000_000 + index).toISOString(),
      runtime: "adaptive-core",
      status: "complete",
      attentionEpoch: Math.floor(index / 400)
    }));
    brain.messages = messages;
    const saved = await repository.save(brain);
    expect(saved.messages).toEqual([]);
    expect(saved.conversation).toMatchObject({
      messageCount: 2_000,
      totalEntries: 2_000,
      attentionEpoch: 4
    });
    const persistedJson = await readFile(
      join(repository.brainDirectory(brain.id), "brain.json"),
      "utf8"
    );
    expect(persistedJson).not.toContain("message-1999-");
    expect(Buffer.byteLength(persistedJson)).toBeLessThan(100_000);

    const latest = await repository.conversationPage(brain.id, undefined, 120);
    expect(latest.entries).toHaveLength(120);
    expect(latest.entries.at(-1)?.message?.content).toBe(messages.at(-1)?.content);
    expect(latest.hasOlder).toBe(true);
    const older = await repository.conversationPage(
      brain.id,
      latest.nextBeforeSequence,
      120
    );
    expect(older.entries).toHaveLength(120);
    expect(older.entries.at(-1)?.sequence).toBeLessThan(
      latest.entries[0]!.sequence
    );
  });

  it("persists terminal actions, attention epochs, and clone history without renumbering", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Action history" });
    const action: ActionEvent = {
      id: randomUUID(),
      brainId: brain.id,
      action: {
        kind: "imagine",
        source: "brain",
        toolId: "modality.imagine",
        action: "generate",
        arguments: { modality: "image" }
      },
      state: "complete",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      attentionEpoch: 7,
      preview: {
        revision: 4,
        mimeType: "image/png",
        mediaUrl: `omni-media://artifact/${"a".repeat(48)}/${"b".repeat(64)}`
      }
    };
    await repository.appendConversationActions(brain.id, [action]);
    const page = await repository.conversationPage(brain.id);
    expect(page.entries[0]?.action).toMatchObject({
      id: action.id,
      attentionEpoch: 7,
      state: "complete"
    });
    expect(JSON.stringify(page.entries[0])).not.toContain("omni-media://");

    const duplicate = await repository.duplicate(brain.id, "Action history copy");
    const duplicatePage = await repository.conversationPage(duplicate.id);
    expect(duplicatePage.entries[0]).toMatchObject({
      sequence: page.entries[0]?.sequence,
      attentionEpoch: 7,
      action: { brainId: duplicate.id }
    });
    expect(page.entries[0]?.action?.brainId).toBe(brain.id);
  });

  it("repairs historical clone action identity and recomputes the hash chain", async () => {
    const source = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Historical clone source"
    });
    const createdAt = "2026-09-10T10:00:00.000Z";
    const action: ActionEvent = {
      id: randomUUID(),
      brainId: source.id,
      action: {
        kind: "tool",
        source: "human",
        toolId: "system.files",
        action: "read",
        arguments: {
          auditSummary: "system.files.read: complete.",
          auditDetail: "fixture"
        }
      },
      state: "complete",
      createdAt,
      updatedAt: createdAt,
      attentionEpoch: 3
    };
    await repository.appendConversationActions(source.id, [action]);
    const sourceLedger = await ConversationLedger.open(
      repository.brainDirectory(source.id),
      source.id
    );
    sourceLedger.append([{
      kind: "message",
      value: {
        id: randomUUID(),
        role: "human",
        content: "A later row keeps the repaired previous-hash chain observable.",
        createdAt: "2026-09-10T10:00:01.000Z",
        attentionEpoch: 3
      }
    }]);
    const sourceSummary = sourceLedger.integrity();
    sourceLedger.close();

    const cloneId = randomUUID();
    const cloneDirectory = join(repository.root, cloneId);
    await mkdir(join(cloneDirectory, "conversation"), { recursive: true });
    const clonePath = join(cloneDirectory, "conversation", "ledger.sqlite3");
    await copyFile(
      join(repository.brainDirectory(source.id), "conversation", "ledger.sqlite3"),
      clonePath
    );
    // Reproduce the pre-fix clone format: only the meta identity changed.
    const legacy = new DatabaseSync(clonePath);
    legacy.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(cloneId);
    legacy.prepare("DELETE FROM meta WHERE key='actionPayloadBrainId'").run();
    legacy.close();

    const repaired = await ConversationLedger.open(cloneDirectory, cloneId);
    const repairedSummary = repaired.integrity();
    const repairedEntries = repaired.page().entries;
    expect(repairedEntries.map((entry) => entry.sequence)).toEqual([1, 2]);
    expect(repairedEntries[0]?.action?.brainId).toBe(cloneId);
    expect(repairedSummary.headSha256).not.toBe(sourceSummary.headSha256);
    expect(() => repaired.append([{
      kind: "action",
      value: { ...action, brainId: cloneId }
    }])).not.toThrow();
    expect(() => repaired.append([{
      kind: "action",
      value: {
        ...action,
        brainId: cloneId,
        action: {
          ...action.action,
          arguments: {
            auditSummary: "system.files.read: complete.",
            auditDetail: "conflicting fixture"
          }
        }
      }
    }])).toThrow(/idempotency conflict/i);
    repaired.close();

    const unchangedSource = await ConversationLedger.open(
      repository.brainDirectory(source.id),
      source.id
    );
    expect(unchangedSource.integrity()).toEqual(sourceSummary);
    expect(unchangedSource.page().entries[0]?.action?.brainId).toBe(source.id);
    unchangedSource.close();
  });

  it("refuses to rekey a historical clone whose old chain was tampered", async () => {
    const source = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Tampered historical clone source"
    });
    const createdAt = "2026-09-10T11:00:00.000Z";
    await repository.appendConversationActions(source.id, [{
      id: randomUUID(),
      brainId: source.id,
      action: {
        kind: "tool",
        source: "human",
        toolId: "system.files",
        action: "list",
        arguments: { auditSummary: "system.files.list: complete." }
      },
      state: "complete",
      createdAt,
      updatedAt: createdAt,
      attentionEpoch: 1
    }]);
    const cloneId = randomUUID();
    const cloneDirectory = join(repository.root, cloneId);
    await mkdir(join(cloneDirectory, "conversation"), { recursive: true });
    const clonePath = join(cloneDirectory, "conversation", "ledger.sqlite3");
    await copyFile(
      join(repository.brainDirectory(source.id), "conversation", "ledger.sqlite3"),
      clonePath
    );
    const tampered = new DatabaseSync(clonePath);
    tampered.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(cloneId);
    tampered.prepare("DELETE FROM meta WHERE key='actionPayloadBrainId'").run();
    tampered.prepare(
      "UPDATE entries SET payload_json=replace(payload_json,'complete','altered') WHERE kind='action'"
    ).run();
    tampered.close();

    await expect(
      ConversationLedger.open(cloneDirectory, cloneId)
    ).rejects.toThrow(/integrity verification failed/i);
  });

  it("fails closed when an append-only row is modified", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Tamper check" });
    const ledger = await ConversationLedger.open(
      repository.brainDirectory(brain.id),
      brain.id
    );
    ledger.append([{
      kind: "message",
      value: {
        id: randomUUID(),
        role: "human",
        content: "unaltered",
        createdAt: new Date().toISOString(),
        attentionEpoch: 2
      }
    }]);
    const path = ledger.path;
    ledger.close();
    const database = new DatabaseSync(path);
    database.prepare("UPDATE entries SET payload_json=? WHERE sequence=1").run(
      JSON.stringify({ id: "tampered", role: "human", content: "changed" })
    );
    database.close();
    await expect(repository.conversationPage(brain.id)).rejects.toThrow(
      /checksum failed/i
    );
  });

  it("backfills interleaved legacy messages and traces chronologically once", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Chronology" });
    const ledger = await ConversationLedger.open(
      repository.brainDirectory(brain.id),
      brain.id
    );
    const messages: ChatMessage[] = [
      {
        id: "message-early",
        role: "human",
        content: "early",
        createdAt: "2026-01-01T00:00:00.000Z"
      },
      {
        id: "message-late",
        role: "brain",
        content: "late",
        createdAt: "2026-01-01T00:00:02.000Z"
      }
    ];
    const traces = [{
      id: "trace-middle",
      createdAt: "2026-01-01T00:00:01.000Z",
      input: "middle",
      seed: 1,
      runtime: "adaptive-core" as const,
      activatedConcepts: [],
      recalledIdeas: [],
      driveScores: { novelty: 0, coherence: 0, curiosity: 0 },
      branches: 1,
      selectedBranch: 0,
      steps: [],
      note: "fixture"
    }];
    const first = ledger.backfill(messages, traces);
    const second = ledger.backfill(messages, traces);
    expect(second).toEqual(first);
    expect(ledger.recentEvidence().map((entry) => entry.kind)).toEqual([
      "message",
      "trace",
      "message"
    ]);
    ledger.close();
  });

  it("retries exactly once after activity projection failure without weakening semantics", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Projection retry" });
    const createdAt = new Date().toISOString();
    const journalId = randomUUID();
    brain.journal = [
      ...(brain.journal ?? []),
      {
        id: journalId,
        createdAt,
        kind: "tool",
        summary: "system.files.read: complete.",
        detail: JSON.stringify({ path: "/tmp/fixture.txt" })
      }
    ];
    brain.trainingSources.push({
      id: randomUUID(),
      name: "fixture.txt",
      kind: "text",
      bytes: 7,
      learnedIdeas: 1,
      learnedConcepts: 1,
      learnedSynapses: 1,
      importedAt: createdAt,
      rawTextRetained: false,
      contentHash: "a".repeat(64),
      policy: "pretrain"
    });
    const originalUpsert = BrainActivityLedger.prototype.upsertTrainingSources;
    const projection = vi
      .spyOn(BrainActivityLedger.prototype, "upsertTrainingSources")
      .mockImplementationOnce(() => {
        throw new Error("simulated old-schema projection failure");
      })
      .mockImplementation(function (this: BrainActivityLedger, values) {
        return originalUpsert.call(this, values);
      });
    try {
      await expect(repository.save(brain)).rejects.toThrow(/projection failure/);
      brain.conversation = {
        ...(brain.conversation!),
        attentionEpoch: 9
      };
      await expect(repository.save(brain)).resolves.toMatchObject({
        activity: { trainingSourceCount: 1 }
      });
    } finally {
      projection.mockRestore();
    }
    const actions = (await repository.conversationPage(brain.id)).entries
      .filter((entry) => entry.action?.id === journalId);
    expect(actions).toHaveLength(1);

    const ledger = await ConversationLedger.open(
      repository.brainDirectory(brain.id),
      brain.id
    );
    const stored = actions[0]!.action!;
    expect(() => ledger.append([{
      kind: "action",
      value: {
        ...stored,
        action: {
          ...stored.action,
          arguments: { path: "/tmp/changed.txt" }
        }
      }
    }])).toThrow(/idempotency conflict/i);
    ledger.close();
  });

  it("accepts only status/default projection drift for stable message and trace ids", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Semantic retry" });
    const ledger = await ConversationLedger.open(
      repository.brainDirectory(brain.id),
      brain.id
    );
    const message: ChatMessage & { turnId: string } = {
      id: randomUUID(),
      role: "brain",
      content: "immutable answer",
      traceId: "trace-semantic",
      turnId: "turn-semantic",
      createdAt: new Date().toISOString(),
      runtime: "adaptive-core",
      status: "complete"
    };
    ledger.append([{ kind: "message", value: message }]);
    expect(() => ledger.append([{
      kind: "message",
      value: { ...message, status: "error", attentionEpoch: 7 }
    }])).not.toThrow();
    expect(() => ledger.append([{
      kind: "message",
      value: { ...message, content: "changed answer" }
    }])).toThrow(/idempotency conflict/i);

    const trace = {
      id: "trace-semantic",
      createdAt: new Date().toISOString(),
      input: "immutable input",
      seed: 17,
      runtime: "adaptive-core" as const,
      activatedConcepts: [],
      recalledIdeas: [],
      driveScores: { novelty: 0, coherence: 0, curiosity: 0 },
      branches: 1,
      selectedBranch: 0,
      steps: [],
      note: "projection",
      turn_id: "turn-semantic",
      input_sha256: "b".repeat(64),
      parameter_checksum_after: "c".repeat(64)
    };
    ledger.append([{ kind: "trace", value: trace }]);
    expect(() => ledger.append([{
      kind: "trace",
      value: { ...trace, note: "default projection changed", attentionEpoch: 4 }
    }])).not.toThrow();
    const changedTrace = { ...trace, parameter_checksum_after: "d".repeat(64) };
    expect(() => ledger.append([{
      kind: "trace",
      value: changedTrace
    }])).toThrow(/idempotency conflict/i);
    ledger.close();
  });

  it("persists UI-only queue revisions without exposing them as neural evidence", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Delivery receipt history"
    });
    const createdAt = "2026-09-08T06:30:00.000Z";
    await repository.appendChatDeliveryReceipt(brain.id, {
      schemaVersion: 1,
      turnId: "queued-turn-one",
      content: "This text was queued but never sent to the neural worker.",
      createdAt,
      state: "queued"
    });
    await repository.appendChatDeliveryReceipt(brain.id, {
      schemaVersion: 1,
      turnId: "queued-turn-one",
      content: "This text was queued but never sent to the neural worker.",
      createdAt,
      state: "cancelled"
    });

    const page = await repository.conversationPage(brain.id);
    expect(page.entries.flatMap((entry) =>
      entry.message?.deliveryReceipt ? [entry.message.deliveryReceipt.state] : []
    )).toEqual(["queued", "cancelled"]);
    expect(await repository.recentConversationEvidence(brain.id)).toEqual([]);

    const restarted = new BrainRepository(join(root, "brains"));
    await restarted.initialize();
    const reopened = await restarted.conversationPage(brain.id);
    expect(reopened.entries.at(-1)?.message).toMatchObject({
      role: "human",
      content: "This text was queued but never sent to the neural worker.",
      deliveryReceipt: {
        presentationOnly: true,
        turnId: "queued-turn-one",
        state: "cancelled"
      }
    });
  });

  it("restores an in-flight human receipt after reopening without feeding it into neural evidence", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Pending delivery history"
    });
    await repository.appendChatDeliveryReceipt(brain.id, {
      schemaVersion: 1,
      turnId: "pending-turn-one",
      content: "This is visible before the reply commits.",
      createdAt: "2026-09-08T06:35:00.000Z",
      state: "pending"
    });

    const reopened = new BrainRepository(join(root, "brains"));
    await reopened.initialize();
    expect((await reopened.conversationPage(brain.id)).entries.at(-1)?.message)
      .toMatchObject({
        role: "human",
        content: "This is visible before the reply commits.",
        deliveryReceipt: {
          presentationOnly: true,
          turnId: "pending-turn-one",
          state: "pending"
        }
      });
    expect(await reopened.recentConversationEvidence(brain.id)).toEqual([]);
  });
});
