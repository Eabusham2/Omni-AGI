import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import { completedInheritedTurn } from "../src/main/inheritedChatReconciliation";
import { DEFAULT_CONFIG, type BrainDocument } from "../src/shared/types";

it("reconciles an inherited lost final ACK without another worker call or rewriting the original human owner", async () => {
  const root = await mkdtemp(join(tmpdir(), "omni-inherited-host-receipt-"));
  try {
    const repository = new BrainRepository(root), brain = await repository.create({ ...DEFAULT_CONFIG, idleCognition: false });
    const directory = repository.brainDirectory(brain.id), engine = join(directory, "engine"); await mkdir(engine);
    const turnId = "ancestral-current-turn", executionId = "e".repeat(64), input = "Hello";
    const inputSha256 = createHash("sha256").update(input).digest("hex"), after = "a".repeat(64);
    const human = { id: "adopted-human", role: "human", content: input, turn_id: "original-parent-turn", attention_epoch: 3, created_at: "2026-10-13T12:00:00.000Z" };
    const assistant = { id: "committed-assistant", role: "brain", content: "native completed output", turn_id: turnId, attention_epoch: 9, created_at: "2026-10-13T12:00:01.000Z" };
    const trace = { id: "committed-trace", turn_id: turnId, attention_epoch: 9, created_at: "2026-10-13T12:00:02.000Z", input_sha256: inputSha256,
      parameter_checksum_before: "d".repeat(64), parameter_checksum_after: after, steps: [] };
    const database = new DatabaseSync(join(engine, "conversation.sqlite3"));
    try {
      database.exec("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);CREATE TABLE entries(kind TEXT,entry_key TEXT PRIMARY KEY,payload_json TEXT,payload_sha256 TEXT)");
      database.prepare("INSERT INTO meta VALUES('brain_id',?)").run(brain.id);
      for (const [kind, value] of [["message", human], ["message", assistant], ["trace", trace]] as const) {
        const body = JSON.stringify(value); database.prepare("INSERT INTO entries VALUES(?,?,?,?)").run(kind, `${kind}:${value.id}`, body, createHash("sha256").update(body).digest("hex"));
      }
    } finally { database.close(); }
    const originalLedger = await readFile(join(engine, "conversation.sqlite3"));
    const receipt = { format: "omni-completed-chat-turn", formatVersion: 1, turnId, inputSha256, humanMessageId: human.id,
      brainMessageId: assistant.id, traceId: trace.id, inferenceCount: 1, parameterChecksumAfter: after, committedAt: "2026-10-13T12:00:03.000Z" };
    const entry = { id: executionId, rootTurnId: turnId, rootHumanMessageId: human.id, status: "complete",
      completion: { turnId, sha256: "f".repeat(64), resultPath: `engine/connected-executions/${executionId}/result.json` } };
    const metadata = { brain_id: brain.id, updated_at: "2026-10-13T12:00:03.000Z", counters: { inference_count: 1, plasticity_events: 0, consolidation_cycles: 0 },
      substrate: { persistence: { activeGeneration: "b".repeat(64) } }, mutable_state: { activeGeneration: "c".repeat(64) },
      completed_chat_turns: [receipt], connected_executions: { format: "omni-connected-execution-queue", formatVersion: 1, entries: [entry] } };
    await writeFile(join(engine, "brain.json"), JSON.stringify({ ...metadata, completed_chat_turns: [], connected_executions: { ...metadata.connected_executions, entries: [{ ...entry, status: "queued" }] } }));
    let current = brain;
    const request = vi.fn(() => { throw new Error("Reconciliation must not dispatch another worker request."); });
    const presentation = vi.fn(async () => ({ brainId: brain.id, input, publishedText: "native", inheritedExecutionId: executionId, priorActions: [], hostAttached: false,
      frontier: { turnId, inputSha256, checkpointId: "1".repeat(32), lastStreamSequence: 1, sampledTokens: 1, publishedTokens: 1, phase: "native-text-token-boundary", exactSupportedResume: true } }));
    const service = Object.create(BrainService.prototype) as BrainService;
    Object.assign(service, {
      repository: { brainDirectory: () => directory, get: async () => structuredClone(current), save: async (value: BrainDocument) => { current = structuredClone(value); return structuredClone(current); } },
      engine: { request, claimForeground: async () => undefined },
      instanceLifecycle: { isLoaded: () => true, canRun: () => false, status: async () => undefined, completed: vi.fn() },
      requireInstanceLoaded: () => undefined, waitUntilInstanceLoaded: async () => undefined,
      instanceInheritedResumePresentation: presentation,
      liveChatTurns: new Map(), suspendedTurnWaiters: new Map(), suspendedObservations: new Map(),
      chatUnlocked: async (...args: unknown[]) => {
        (args[6] as () => void)();
        await writeFile(join(engine, "brain.json"), JSON.stringify(metadata));
        throw new Error("One-use lost final RPC acknowledgement after committed native state");
      }
    });
    const completed = await service.resumeInheritedTurn(brain.id, executionId);
    expect(completed.humanMessage).toMatchObject({ id: human.id, turnId: human.turn_id, attentionEpoch: 3 });
    expect(completed.brainMessage).toMatchObject({ id: assistant.id, turnId, attentionEpoch: 9, content: assistant.content });
    expect(current.counters.inferenceCount).toBe(1); expect(request).not.toHaveBeenCalled();
    const calls = presentation.mock.calls.length;
    const retry = await service.resumeInheritedTurn(brain.id, executionId);
    expect(retry.brainMessage.id).toBe(assistant.id); expect(presentation).toHaveBeenCalledTimes(calls);
    expect(request).not.toHaveBeenCalled(); expect(current.counters.inferenceCount).toBe(1);
    expect(await readFile(join(engine, "conversation.sqlite3"))).toEqual(originalLedger);
    expect(completedInheritedTurn({ ...metadata, brain_id: "other" }, brain.id, executionId)).toBeUndefined();
    expect(() => completedInheritedTurn({ ...metadata, connected_executions: { ...metadata.connected_executions, entries: [{ ...entry, rootHumanMessageId: "different" }] } }, brain.id, executionId)).toThrow(/receipt/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
