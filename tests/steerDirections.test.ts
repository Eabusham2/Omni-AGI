import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SteerDirectionStore } from "../src/main/steerDirectionStore";
import { SteerDirectionsService } from "../src/main/steerDirections";
import type { SteerDirectionEvent } from "../src/shared/steerDirections";
import { accepted, at, audio, managed, metadata, result, source } from "./fixtures/steerDirections";

describe("durable presentation-only steer directions (no model)", () => {
  let root: string;
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-directions-")); await managed(root, "owner"); });
  afterEach(async () => { await rm(root, { recursive: true, force: true }); });
  const service = (root: string, options: Partial<ConstructorParameters<typeof SteerDirectionsService>[0]> = {}) =>
    new SteerDirectionsService({ brainDirectory: () => root, ...options });

  it("keeps source content, input/reply receipts and applied host IDs immutable; dismiss is display-only CAS", async () => {
    const store = (await SteerDirectionStore.open(root, "owner", true))!;
    try {
      const initial = store.create(source()), received = accepted();
      const row = store.update(initial.id, view => ({ ...view, state: "received", inputReceipt: received.inputReceipt,
        acceptedAttachmentIds: [], humanMessageId: received.humanMessage.id, humanTurnId: "direction-one" }));
      expect(() => store.create({ ...source(), content: "A substituted direction" })).toThrow(/different immutable/);
      expect(() => store.update(row.id, view => ({ ...view, inputReceipt: { ...view.inputReceipt!, inputSha256: "f".repeat(64) } }))).toThrow(/immutable/);
      expect(() => store.update(row.id, view => ({ ...view, humanMessageId: "other-human" }))).toThrow(/immutable/);
      const hidden = store.dismiss(row.id, row.revision, true);
      expect(hidden.state).toBe("received"); expect(hidden.source).toEqual(source()); expect(hidden.inputReceipt).toEqual(received.inputReceipt);
      expect(store.list().total).toBe(0); expect(store.forTurn("direction-one")[0]?.dismissedAt).toBeDefined();
      expect(() => store.dismiss(row.id, row.revision, false)).toThrow(/changed/);
      const visible = store.dismiss(row.id, hidden.revision, false); expect(visible.dismissedAt).toBeUndefined(); store.validate();
    } finally { store.close(); }
  });

  it("paginates chronological IDs and exactly finds old dismissed turns without reading unrelated payloads", async () => {
    const store = (await SteerDirectionStore.open(root, "owner", true))!;
    try {
      const old = store.create(source("old-direction")); const hidden = store.dismiss(old.id, old.revision, true);
      for (let i = 0; i < 24; i++) store.create(source(`new-direction-${i}`));
      const first = store.list({ limit: 20, includeDismissed: true }), second = store.list({ limit: 20, includeDismissed: true, beforeSequence: first.nextBeforeSequence });
      expect([...first.entries, ...second.entries]).toHaveLength(25); expect(second.entries.at(-1)?.id).toBe(hidden.id);
      store.db.prepare("UPDATE directions SET source_json='broken unrelated row' WHERE id=?").run(first.entries[0]!.id);
      expect(store.forTurn("old-direction")[0]?.id).toBe(old.id);
      expect(JSON.stringify(store.db.prepare("EXPLAIN QUERY PLAN SELECT id FROM directions WHERE turn_id=? OR human_turn_id=?").all("old-direction", "old-direction"))).toContain("directions_human_turn");
    } finally { store.close(); }
  });

  it("refuses a tampered event chain, foreign owner, hidden trigger, and insufficient transfer RAM", async () => {
    const store = (await SteerDirectionStore.open(root, "owner", true))!, row = store.create(source()); store.close();
    await expect(SteerDirectionStore.open(root, "foreign")).rejects.toThrow(/another owner/);
    const bounded = (await SteerDirectionStore.open(root, "owner", false, () => 1))!;
    try { expect(() => bounded.get(row.id)).toThrow(/not truncated/); } finally { bounded.close(); }
    const database = new DatabaseSync(join(root, "directions", "workspace.sqlite3"));
    database.prepare("UPDATE events SET payload_json='{}' WHERE sequence=1").run(); database.close();
    const tampered = (await SteerDirectionStore.open(root, "owner"))!;
    try { expect(() => tampered.validate()).toThrow(/history changed/); } finally { tampered.close(); }
    const unsafe = new DatabaseSync(join(root, "directions", "workspace.sqlite3"));
    unsafe.exec("CREATE TRIGGER unexpected AFTER INSERT ON meta BEGIN SELECT 1; END;"); unsafe.close();
    await expect(SteerDirectionStore.open(root, "owner", true)).rejects.toThrow(/unsupported schema/);
  });

  it("records pending then actual accepted input, final saved reply and post-reply failure without undo", async () => {
    const events: SteerDirectionEvent[] = [], directions = service(root), unsubscribe = directions.subscribe(event => events.push(event));
    const tracker = (await directions.begin("owner", "direction-one", metadata(), "  Use the blue version\0  "))!;
    try {
      expect((await directions.list("owner")).entries[0]?.state).toBe("pending");
      const received = accepted("direction-one", "Use the blue version", ["inherited-original-file"]);
      tracker.received(received.inputReceipt, received.humanMessage);
      const row = (await directions.forTurns("owner", ["direction-one"]))[0]!;
      expect(row).toMatchObject({ state: "received", source: { content: "Use the blue version", submittedContent: "  Use the blue version\0  " }, acceptedAttachmentIds: ["inherited-original-file"] });
      tracker.completed(result(received.humanMessage)); tracker.terminal("failed");
      const complete = await directions.get("owner", row.id);
      expect(complete).toMatchObject({ state: "complete", failure: "post-reply-work-failed", humanMessageId: received.humanMessage.id, reply: { brainMessageId: "brain-direction-one" } });
      expect(events.map(event => event.state)).toEqual(["pending", "received", "complete", "complete"]);
      expect(JSON.stringify(events)).not.toContain("Use the blue version");
    } finally { tracker.close(); unsubscribe(); }
  });

  it("keeps original safe audio capture and accepted inherited attachment identities, never a transcript or lease", async () => {
    const directions = service(root), input = audio(), tracker = (await directions.begin("owner", "audio-direction", metadata(), "", input))!;
    try {
      const received = accepted("audio-direction", "", ["old-task-file", input.attachment.attachmentId], input);
      tracker.received(received.inputReceipt, received.humanMessage); tracker.completed(result(received.humanMessage));
      const row = (await directions.forTurns("owner", ["audio-direction"]))[0]!;
      expect(row.source.content).toBe(""); expect(row.source.audio?.inputSha256).toBe(input.inputSha256);
      expect(row.acceptedAttachmentIds).toEqual(["old-task-file", "original-audio"]);
      expect(row.inputReceipt?.nativeAudioInput?.transcriptUsed).toBe(false);
      expect(JSON.stringify(row)).not.toMatch(/relativePath|sessionId|permissionRevision|private-owned-lease|dataBase64/);
    } finally { tracker.close(); }
  });

  it("does not promote mismatched input or another saved reply; presentation failure cannot abort real computation", async () => {
    const failures = vi.fn(), directions = service(root, { onError: failures }), events: SteerDirectionEvent[] = [];
    directions.subscribe(event => events.push(event)); const tracker = (await directions.begin("owner", "direction-one", metadata(), "Use the blue version"))!;
    try {
      const wrong = accepted("direction-one", "wrong input"); tracker.received(wrong.inputReceipt, wrong.humanMessage);
      expect((await directions.list("owner")).entries[0]?.state).toBe("pending");
      const right = accepted(); tracker.received(right.inputReceipt, right.humanMessage);
      tracker.completed(result({ ...right.humanMessage, id: "different-human" }));
      expect((await directions.list("owner")).entries[0]?.state).toBe("received");
      expect(failures).toHaveBeenCalledTimes(2); expect(events.filter(event => event.type === "unavailable")).toHaveLength(2);
      tracker.terminal("stopped"); expect((await directions.list("owner")).entries[0]).toMatchObject({ state: "stopped", inputReceipt: right.inputReceipt });
    } finally { tracker.close(); }
  });

  it("marks interrupted pending records dormant without dispatch, and passively tracks an explicit same-turn continuation", async () => {
    const first = service(root), tracker = (await first.begin("owner", "direction-one", metadata(), "Use the blue version"))!; tracker.close();
    const restarted = service(root), page = await restarted.list("owner"), old = page.entries[0]!;
    expect(old).toMatchObject({ state: "pending", dormant: { reason: "process-interrupted" } }); expect(old.inputReceipt).toBeUndefined();
    await expect(restarted.begin("owner", "direction-one", metadata(), "Use the blue version")).rejects.toThrow(/historical/);
    const resumed = (await restarted.begin("owner", "direction-one", undefined, "Use the blue version", undefined, true))!;
    try { const actual = accepted(); resumed.received(actual.inputReceipt, actual.humanMessage); resumed.completed(result(actual.humanMessage)); }
    finally { resumed.close(); }
    expect(await restarted.get("owner", old.id)).toMatchObject({ state: "complete" });
    expect((await restarted.get("owner", old.id))?.dormant).toBeUndefined();
  });

  it("tracks no ordinary task, refuses duplicate active ownership, and rejects unsafe lookup/dismiss authority", async () => {
    const directions = service(root);
    expect(await directions.begin("owner", "task", { kind: "task", source: "human", createdAt: at, attachmentIds: [] }, "ordinary input")).toBeUndefined();
    const attempts = await Promise.allSettled([directions.begin("owner", "direction-one", metadata(), "Use the blue version"), directions.begin("owner", "direction-one", metadata(), "Use the blue version")]);
    expect(attempts.filter(value => value.status === "fulfilled")).toHaveLength(1);
    const tracker = attempts.find(value => value.status === "fulfilled"); if (tracker?.status !== "fulfilled") throw new Error("No tracker");
    expect(() => directions.forget("owner")).toThrow(/active/); tracker.value!.close(); directions.forget("owner");
    await expect(directions.forTurns("owner", Array.from({ length: 101 }, (_, i) => `turn-${i}`))).rejects.toThrow(/at most 100/);
    await expect(directions.forTurns("owner", ["duplicate", "duplicate"])).rejects.toThrow(/distinct/);
    await expect(directions.dismiss({ brainId: "owner", id: source().turnId, expectedRevision: 1, dismissed: true, content: "rewrite" } as never)).rejects.toThrow(/display request/);
  });

  it("requires every original audio capture field and preserves approved input attribution exactly", async () => {
    const failures = vi.fn(), directions = service(root, { onError: failures }), input = audio();
    const tracker = (await directions.begin("owner", "audio-direction", metadata(), "", input))!;
    try {
      const received = accepted("audio-direction", "", [input.attachment.attachmentId], input);
      const substituted = structuredClone(received); substituted.inputReceipt.nativeAudioInput!.utteranceId = "substituted-utterance";
      tracker.received(substituted.inputReceipt, substituted.humanMessage);
      expect((await directions.forTurns("owner", ["audio-direction"]))[0]?.state).toBe("pending");
      tracker.received(received.inputReceipt, received.humanMessage);
      expect((await directions.forTurns("owner", ["audio-direction"]))[0]?.state).toBe("received");
      expect(failures).toHaveBeenCalledTimes(1);
    } finally { tracker.close(); }
    const text = (await directions.begin("owner", "direction-one", metadata(), "Use the blue version"))!, received = accepted();
    received.inputReceipt.inputProvenance = { kind: "mcp-app", sourceBrainId: "owner", serverId: "server", serverLabel: "Reviewed source",
      endpoint: "https://example.test/mcp", resourceUri: "ui://source/view", artifactId: "artifact", captureSha256: "a".repeat(64),
      originalCallBrainId: "owner", toolId: "mcp.tool", reviewId: "review", observationId: "observation", streamId: "stream", ordinal: 1,
      recordSha256: "b".repeat(64), dataSha256: "c".repeat(64), nativeFrontierSha256: "d".repeat(64), inputSha256: received.inputReceipt.inputSha256 };
    try { text.received(received.inputReceipt, received.humanMessage);
      expect((await directions.forTurns("owner", ["direction-one"]))[0]?.inputReceipt?.inputProvenance).toEqual(received.inputReceipt.inputProvenance);
    } finally { text.close(); }
  });
});
