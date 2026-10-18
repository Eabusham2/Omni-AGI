import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { SteerDirectionStore } from "../src/main/steerDirectionStore";
import { SteerDirectionsService } from "../src/main/steerDirections";
import { mergeSteerDirectionWorkspace, mergedSteerDirectionProjection, validateSteerDirectionWorkspace } from "../src/main/steerDirectionPortability";
import { snapshotLearningWorkspace, adoptLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { ConversationLedger } from "../src/main/conversationLedger";
import { accepted, audio, managed, metadata, result, source, sha } from "./fixtures/steerDirections";

describe("complete unsanitized steering presentation custody (no model)", () => {
  let base: string, sourceRoot: string, target: string;
  beforeEach(async () => {
    base = await mkdtemp(join(tmpdir(), "omni-direction-custody-")); sourceRoot = join(base, "source"); target = join(base, "target");
    await managed(sourceRoot, "source"); await managed(target, "target");
  });
  afterEach(async () => { await rm(base, { recursive: true, force: true }); });

  it("snapshots WAL source/events and preserves exact raw text/audio bindings across import and another fork", async () => {
    const directions = new SteerDirectionsService({ brainDirectory: () => sourceRoot });
    const tracker = (await directions.begin("source", "direction-one", metadata(), "  Use the blue version\0  "))!;
    const received = accepted(); tracker.received(received.inputReceipt, received.humanMessage); tracker.completed(result(received.humanMessage)); tracker.close();
    const input = audio("audio-direction", "source"), audioTracker = (await directions.begin("source", "audio-direction", metadata(), "", input))!; audioTracker.close();
    const reader = (await SteerDirectionStore.open(sourceRoot, "source"))!, originalRows = reader.list({ includeDismissed: true });
    const rawEvents = reader.db.prepare("SELECT * FROM events ORDER BY sequence").all(); reader.close();
    const sourceBytes = await readFile(join(sourceRoot, "directions", "workspace.sqlite3"));
    const archive = (await snapshotLearningWorkspace(sourceRoot, join(base, "archive"), "source"))!;
    await adoptLearningWorkspace(archive, target, "target");
    const copied = (await SteerDirectionStore.open(target, "target"))!;
    try {
      copied.validate(); const entries = copied.list({ includeDismissed: true }).entries;
      expect(entries.map(row => row.source)).toEqual(originalRows.entries.map(row => row.source));
      expect(entries.find(row => row.turnId === "direction-one")?.inputReceipt).toEqual(received.inputReceipt);
      expect(entries.find(row => row.turnId === "audio-direction")).toMatchObject({ state: "pending", dormant: { reason: "copied-instance", fromBrainId: "source" } });
      expect(copied.db.prepare("SELECT * FROM events WHERE sequence<=? ORDER BY sequence").all(rawEvents.length)).toEqual(rawEvents);
      expect(entries.every(row => row.brainId === "target")).toBe(true);
    } finally { copied.close(); }
    expect(await readFile(join(sourceRoot, "directions", "workspace.sqlite3"))).toEqual(sourceBytes);
    const second = (await snapshotLearningWorkspace(target, join(base, "fork"), "target"))!, third = join(base, "third"); await managed(third, "third");
    await adoptLearningWorkspace(second, third, "third"); const forked = (await SteerDirectionStore.open(third, "third"))!;
    try { forked.validate(); expect(forked.list({ includeDismissed: true }).entries.map(row => row.source)).toEqual(originalRows.entries.map(row => row.source)); }
    finally { forked.close(); }
  });

  it("merges shared applied history once, preserves divergent outcomes, archives all events and maintains global order", async () => {
    const sourceStore = (await SteerDirectionStore.open(sourceRoot, "source", true))!;
    for (let i = 0; i < 25; i++) sourceStore.create(source(`direction-${i}`, "source", `Exact original ${i}`)); sourceStore.close();
    const archive = (await snapshotLearningWorkspace(sourceRoot, join(base, "archive"), "source"))!;
    await adoptLearningWorkspace(archive, target, "target");
    const staged = join(base, "stage"); await managed(staged, "target"); await adoptLearningWorkspace(archive, staged, "target");
    await mergeSteerDirectionWorkspace(staged, target, "target", "parent-one");
    const joined = (await SteerDirectionStore.open(target, "target"))!;
    try {
      joined.validate(); expect(joined.list({ includeDismissed: true, limit: 100 }).total).toBe(25);
      expect(joined.db.prepare("SELECT turn_id FROM directions ORDER BY sequence").all().map(row => row.turn_id)).toEqual(Array.from({ length: 25 }, (_, i) => `direction-${i}`));
    } finally { joined.close(); }
    // A later genuinely different applied outcome is retained as a branch,
    // while the immutable source request and archived event prefix stay exact.
    const changed = (await SteerDirectionStore.open(staged, "target", true))!, first = changed.forTurn("direction-0")[0]!;
    changed.update(first.id, view => ({ ...view, state: "failed", failure: "dispatch-failed" })); changed.close();
    await mergeSteerDirectionWorkspace(staged, target, "target", "parent-two");
    const final = (await SteerDirectionStore.open(target, "target"))!;
    try { final.validate(); expect(final.list({ includeDismissed: true, limit: 100 }).total).toBe(26);
      expect(final.forTurn("direction-0").map(row => row.state).sort()).toEqual(["failed", "pending"]);
    } finally { final.close(); }
    const names = await readdir(join(target, "directions", "archives")); expect(names).toHaveLength(2);
    for (const name of names) expect(sha(await readFile(join(target, "directions", "archives", name)))).toBe(name.slice(0, -8));
    await validateSteerDirectionWorkspace(target, "target");
    const later = (await snapshotLearningWorkspace(target, join(base, "joined-export"), "target"))!, destination = join(base, "joined-copy");
    await managed(destination, "copy"); await adoptLearningWorkspace(later, destination, "copy");
    expect(await readdir(join(destination, "directions", "archives"))).toEqual(names);
  });

  it("retains newer recovery outcomes as dormant history instead of erasing or replaying them", async () => {
    const selected = (await SteerDirectionStore.open(target, "target", true))!; selected.create(source("same-turn", "target")); selected.close();
    const current = (await SteerDirectionStore.open(sourceRoot, "target", true))!, row = current.create(source("same-turn", "target"));
    current.update(row.id, view => ({ ...view, state: "stopped", failure: "stopped" })); current.close();
    await mergeSteerDirectionWorkspace(sourceRoot, target, "target", "recovery-point-one", undefined, "recovered-history");
    const restored = (await SteerDirectionStore.open(target, "target"))!;
    try { restored.validate(); const records = restored.forTurn("same-turn"); expect(records).toHaveLength(2);
      expect(records.find(record => record.state === "stopped")?.dormant?.reason).toBe("recovered-history");
      expect(records.find(record => record.state === "pending")?.inputReceipt).toBeUndefined();
    } finally { restored.close(); }
  });

  it("projects only real current-owner human rows while retaining original native receipt and turn identities", async () => {
    const directions = new SteerDirectionsService({ brainDirectory: () => sourceRoot }), tracker = (await directions.begin("source", "direction-one", metadata(), "Use the blue version"))!;
    const received = accepted(); tracker.received(received.inputReceipt, received.humanMessage); tracker.completed(result(received.humanMessage)); tracker.close();
    const sourceStore = (await SteerDirectionStore.open(sourceRoot, "source"))!, row = sourceStore.list().entries[0]!; sourceStore.close();
    const ledger = await ConversationLedger.open(target, "target");
    ledger.append([{ kind: "message", value: { ...received.humanMessage, id: "current-human", turnId: "original-source-turn" } }]); ledger.close();
    await mkdir(join(target, "merge")); const history = new DatabaseSync(join(target, "merge", "host-history.sqlite3"));
    history.exec("CREATE TABLE identities(parent_id TEXT,source_id TEXT,destination_id TEXT,PRIMARY KEY(parent_id,source_id));");
    history.prepare("INSERT INTO identities VALUES(?,?,?)").run("parent", received.humanMessage.id, "current-human"); history.close();
    const projection = await mergedSteerDirectionProjection(target, "target", "parent", row, new Map([["direction-one", "ancestral-runtime-turn"]]));
    expect(projection).toEqual({ turnId: "ancestral-runtime-turn", predecessorTurnId: "original-task", humanMessageId: "current-human", humanTurnId: "original-source-turn" });
    const targetStore = (await SteerDirectionStore.open(target, "target", true))!; targetStore.adopt(row, projection, "parent", "merged-history");
    try { targetStore.validate(); const adopted = targetStore.forTurn("original-source-turn")[0]!;
      expect(adopted.turnId).toBe("ancestral-runtime-turn"); expect(adopted.inputReceipt).toEqual(row.inputReceipt); expect(adopted.source).toEqual(row.source);
    } finally { targetStore.close(); }
    const unsafe = new DatabaseSync(join(target, "conversation", "ledger.sqlite3")); unsafe.prepare("UPDATE entries SET previous_sha256=?").run("a".repeat(64)); unsafe.close();
    await expect(mergedSteerDirectionProjection(target, "target", "parent", row, new Map())).rejects.toThrow(/unavailable or changed/);
  });

  it("rejects changed archived bytes and unlisted private workspace payloads", async () => {
    const sourceStore = (await SteerDirectionStore.open(sourceRoot, "target", true))!; sourceStore.create(source()); sourceStore.close();
    await mergeSteerDirectionWorkspace(sourceRoot, target, "target", "parent");
    const [name] = await readdir(join(target, "directions", "archives"));
    await writeFile(join(target, "directions", "archives", name!), "changed archive");
    await expect(validateSteerDirectionWorkspace(target, "target")).rejects.toThrow(/archive changed/);
    await writeFile(join(sourceRoot, "directions", "secret.txt"), "unlisted private payload");
    await expect(snapshotLearningWorkspace(sourceRoot, join(base, "unsafe"), "target")).rejects.toThrow(/unowned payload/);
  });
});
