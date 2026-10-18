import { createHash } from "node:crypto";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, expect, it, vi } from "vitest";
import { BrainService } from "../src/main/brainService";
import { chatTurnKey } from "../src/shared/chatTurnKey";

const roots: string[] = [], input = "A genuinely scheduled task", turnId = "scheduled-turn";
const hash = (value: string) => createHash("sha256").update(value).digest("hex");
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-native-turn-status-")); roots.push(root); await mkdir(join(root, "engine"));
  const metadata: Record<string, any> = { brain_id: "brain", completed_chat_turns: [], accepted_chat_inputs: [] };
  const database = new DatabaseSync(join(root, "engine/conversation.sqlite3"));
  database.exec("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE entries(entry_key TEXT PRIMARY KEY,kind TEXT,payload_json TEXT,payload_sha256 TEXT)");
  database.prepare("INSERT INTO meta VALUES('brain_id','brain')").run();
  const add = (kind: string, identifier: string, value: unknown) => { const body = JSON.stringify(value);
    database.prepare("INSERT OR REPLACE INTO entries VALUES(?,?,?,?)").run(`${kind}:${identifier}`, kind, body, hash(body)); };
  const persist = () => writeFile(join(root, "engine/brain.json"), JSON.stringify(metadata));
  let native: any = { brainId: "brain", learnedRuntimeResident: false, minimumContextTokens: 0, reconnect: [] };
  const engine = { instanceStatus: vi.fn(async () => native), request: vi.fn<(method: string, value?: unknown) => Promise<Record<string, unknown>>>(async () => { throw new Error("Neural writer/load must not run"); }) };
  const liveChatTurns = new Map(), service = Object.assign(Object.create(BrainService.prototype), {
    repository: { get: vi.fn(async () => ({ id: "brain" })), brainDirectory: () => root }, engine, liveChatTurns,
    runtimeShutdownHolds: new Set(), instanceLifecycle: { isLoaded: () => native.learnedRuntimeResident, canRun: () => native.learnedRuntimeResident }
  }) as BrainService;
  await persist(); return { metadata, database, add, persist, engine, service, liveChatTurns, setNative(value: any) { native = value; } };
}
it("queries exact absence, active ownership and conflicts without loading or sending", async () => {
  const f = await fixture(); try {
    expect(await f.service.queryNativeTurnStatus("brain", turnId, input)).toMatchObject({ status: "absent", loaded: false, admissionOpen: false });
    f.liveChatTurns.set(chatTurnKey("brain", turnId), { inputSha256: hash(input) });
    expect((await f.service.queryNativeTurnStatus("brain", turnId, input)).status).toBe("active");
    await expect(f.service.queryNativeTurnStatus("brain", turnId, "Changed input")).rejects.toThrow(/different input/);
    expect(f.engine.request).not.toHaveBeenCalled();
  } finally { f.database.close(); }
});
it("keeps an older admitted input ambiguous rather than replaying it when bounded metadata no longer lists it", async () => {
  const f = await fixture(); try {
    const createdAt = "2026-10-11T12:00:00Z", receipt = { committed: true, turnId, inputSha256: hash(input), humanMessageId: "human",
      afterimageId: "a".repeat(64), createdAt, attentionEpoch: 0 };
    f.add("message", "human", { id: "human", role: "human", content: input, turn_id: turnId, created_at: createdAt,
      attention_epoch: 0, input_accepted_before_reply: true });
    f.add("trace", hash(`brain\0${turnId}\0accepted-input-v1`), { inputAccepted: receipt });
    expect((await f.service.queryNativeTurnStatus("brain", turnId, input)).status).toBe("accepted-unfinished");
    await expect(f.service.queryNativeTurnStatus("brain", turnId, "Changed input")).rejects.toThrow();
  } finally { f.database.close(); }
});
it("preserves conservative ownership when a copied instance's old admission-trace ID still names the source", async () => {
  const f = await fixture(); try {
    f.add("message", "copied-human", { id: "copied-human", role: "human", content: input, turn_id: turnId });
    expect((await f.service.queryNativeTurnStatus("brain", turnId, input)).status).toBe("accepted-unfinished");
    await expect(f.service.queryNativeTurnStatus("brain", turnId, "Changed input")).rejects.toThrow(/previously saved/);
  } finally { f.database.close(); }
});
it("recognizes a ledger-bound completion without sending, learning or modifying its presentation", async () => {
  const f = await fixture(); try {
    const checksum = "b".repeat(64), committedAt = "2026-10-11T12:00:00Z";
    const receipt = { format: "omni-completed-chat-turn", formatVersion: 1, turnId, inputSha256: hash(input), humanMessageId: "human",
      brainMessageId: "reply", traceId: "trace", inferenceCount: 3, parameterChecksumAfter: checksum, committedAt };
    f.metadata.completed_chat_turns = [receipt]; await f.persist();
    f.add("message", "human", { id: "human", role: "human", content: input });
    f.add("message", "reply", { id: "reply", role: "brain", content: "Actual old reply" });
    f.add("trace", "trace", { id: "trace", turn_id: turnId, input_sha256: hash(input), parameter_checksum_after: checksum });
    expect(await f.service.queryNativeTurnStatus("brain", turnId, input)).toMatchObject({ status: "committed", receipt: { inferenceCount: 3, traceId: "trace" } });
    await expect(f.service.queryNativeTurnStatus("brain", turnId, "Changed input")).rejects.toThrow(/conflicts/);
    expect(f.engine.request).not.toHaveBeenCalled();
  } finally { f.database.close(); }
});
it("returns the authenticated retained attachment and rejects an unverifiable absence", async () => {
  const f = await fixture(); try {
    f.setNative({ brainId: "brain", learnedRuntimeResident: false, minimumContextTokens: 8, reconnect: [],
      resume: { checkpointId: "c".repeat(32), turnId, inputSha256: hash(input), lastStreamSequence: 2, sampledTokens: 2,
        publishedTokens: 1, phase: "native-text-token-boundary", exactSupportedResume: true }, presentation: { input, publishedText: "Already emitted", actions: [] } });
    expect(await f.service.queryNativeTurnStatus("brain", turnId, input)).toMatchObject({ status: "retained", attachment: { input, publishedText: "Already emitted", hostAttached: false } });
    delete f.metadata.accepted_chat_inputs; await f.persist();
    await expect(f.service.queryNativeTurnStatus("brain", turnId, input)).rejects.toThrow(/unverifiable/);
  } finally { f.database.close(); }
});
it("explicitly stops an unsupported cold saved frontier without loading or admitting its input", async () => {
  const f = await fixture(); try {
    const checkpointId = "d".repeat(32);
    f.setNative({ brainId: "brain", learnedRuntimeResident: false, minimumContextTokens: 8, reconnect: [],
      resumeCompatibility: { compatible: false, reason: "No supported exact runtime seal" },
      resume: { checkpointId, turnId, inputSha256: hash(input), exactSupportedResume: true } });
    const completed = vi.fn(), status = vi.fn(async () => ({ brainId: "brain", phase: "unloaded" }));
    Object.assign((f.service as any).instanceLifecycle, { completed, status });
    f.engine.request.mockImplementation(async (method: string, value: any) => {
      expect(method).toBe("stop_suspended_chat"); expect(value).not.toHaveProperty("input");
      return { stopped: true, brainId: "brain", turnId, checkpointId, inputReceiptRetained: true };
    });
    expect(await f.service.stopSavedTurn({ brainId: "brain", turnId, checkpointId })).toMatchObject({ phase: "unloaded" });
    expect(completed).toHaveBeenCalledExactlyOnceWith("brain"); expect(status).toHaveBeenCalledOnce();
    f.liveChatTurns.set(chatTurnKey("brain", turnId), { inputSha256: hash(input) });
    await expect(f.service.stopSavedTurn({ brainId: "brain", turnId, checkpointId })).rejects.toThrow(/active chat/);
    expect(f.engine.request).toHaveBeenCalledOnce();
  } finally { f.database.close(); }
});
