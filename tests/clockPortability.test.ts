/** Actual short SQLite/file portability; no native model or external effects. */
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { ClockService } from "../src/main/clockService";
import { ClockStore } from "../src/main/clockStore";
import { mergeClockWorkspace, rekeyClockWorkspace, validateClockWorkspace } from "../src/main/clockPortability";
import { adoptLearningWorkspace, cloneLearningWorkspace, snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";
import type { ScheduleRecord, ScheduledOccurrence, StopwatchRecord } from "../src/shared/clock";
const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function directory() { const root = await mkdtemp(join(tmpdir(), "omni-clock-copy-")); roots.push(root); return root; }
async function seed(root: string, brainId = "source-brain") {
  let wall = Date.parse("2026-10-11T13:00:00Z"), mono = 0;
  const service = new ClockService({ brainDirectory: async () => root, source: { wallMs: () => wall, monotonicMs: () => mono, sessionId: "clock-fixture" },
    authority: async () => ({ enabled: true, revision: "a".repeat(64) }), loaded: async () => false, busy: () => false, admissionOpen: () => true,
    turnStatus: async () => ({ state: "absent" }), wake: async () => { throw new Error("No native inference in portability fixture"); }, notify: () => undefined });
  const schedule = await service.execute(brainId, "studio.schedule", "create", { label: "Exact saved task", kind: "timer", delayMs: 1, delivery: "wake", task: "Review my retained document." }, "create-one") as ScheduleRecord;
  const watch = await service.execute(brainId, "studio.stopwatch", "start", { label: "Measured work" }, "watch-one") as StopwatchRecord;
  wall += 25; mono += 25; await service.execute(brainId, "studio.stopwatch", "lap", { stopwatchId: watch.id }, "lap-one");
  await service.tick(); await service.settle(brainId);
  return { service, schedule, watch };
}

describe("clock workspace ownership and portability", () => {
  it("snapshots every saved clock component and adopts dormant copies with exact input/receipt IDs", async () => {
    const source = await directory(), archiveRoot = await directory(), target = await directory(), state = await seed(source);
    const archive = await snapshotLearningWorkspace(source, archiveRoot, "source-brain");
    expect(archive).toBeDefined(); await adoptLearningWorkspace(archive!, target, "copied-brain");
    await validateClockWorkspace(target, "copied-brain");
    const original = await ClockStore.open(source, "source-brain", true), copy = await ClockStore.open(target, "copied-brain", true);
    try {
      const schedule = copy.get<ScheduleRecord>("schedule", state.schedule.id)!;
      expect(schedule.brainId).toBe("copied-brain"); expect(schedule.pendingPaused).toBe(true); expect(schedule.task).toBe(state.schedule.task);
      const occurrence = copy.records<ScheduledOccurrence>("occurrence").next().value!;
      const old = original.get<ScheduledOccurrence>("occurrence", occurrence.id)!;
      expect(occurrence.turnId).toBe(old.turnId); expect(occurrence.inputSha256).toBe(old.inputSha256); expect(occurrence.state).toBe("review-required"); expect(occurrence.copiedUnstarted).toBe(true);
      expect(copy.get<StopwatchRecord>("stopwatch", state.watch.id)?.elapsedMs).toBe(25);
      expect(copy.laps(state.watch.id).entries[0]?.lapMs).toBe(25);
      expect(copy.db.prepare("SELECT count(*) AS n FROM commands").get()?.n).toBe(0);
      expect(Number(copy.db.prepare("SELECT count(*) AS n FROM inherited_commands").get()?.n)).toBe(3);
      expect(original.get<StopwatchRecord>("stopwatch", state.watch.id)?.state).toBe("running");
    } finally { original.close(); copy.close(); }
  });
  it("uses the existing fork workspace hook without running or granting copied schedules", async () => {
    const source = await directory(), target = await directory(); await seed(source);
    await cloneLearningWorkspace(source, target, "source-brain", "fork-brain");
    const store = await ClockStore.open(target, "fork-brain", true);
    try { expect(store.records<ScheduledOccurrence>("occurrence").next().value?.state).toBe("review-required"); }
    finally { store.close(); }
  });
  it("retains all clock rows beyond the IPC page window, and rejects an extra unknown source file", async () => {
    const source = await directory(), target = await directory(); const state = await seed(source);
    const store = await ClockStore.open(source, "source-brain");
    try { for (let index = 0; index < 129; index++) store.put({ ...state.watch, id: `extra-${index}`, state: "paused", elapsedMs: 0, lapCount: 0 }, "stopwatch"); }
    finally { store.close(); }
    await cloneLearningWorkspace(source, target, "source-brain", "large-copy");
    const copied = await ClockStore.open(target, "large-copy", true);
    try { expect(copied.page("stopwatch", { limit: 50 }).total).toBe(130); expect([...copied.records("stopwatch")]).toHaveLength(130); }
    finally { copied.close(); }
    await writeFile(join(source, "clock", "unapproved.sh"), "not executable");
    await expect(snapshotLearningWorkspace(source, await directory(), "source-brain")).rejects.toThrow(/Unsafe portable/);
  });
  it("merges copied clocks idempotently, keeping parent state and original effects dormant", async () => {
    const source = await directory(), staged = await directory(), target = await directory(); const state = await seed(source);
    await cloneLearningWorkspace(source, staged, "source-brain", "merged-brain");
    await mergeClockWorkspace(staged, target, "merged-brain", "parent-one");
    await mergeClockWorkspace(staged, target, "merged-brain", "parent-one");
    const merged = await ClockStore.open(target, "merged-brain", true);
    try {
      expect(merged.page("schedule").total).toBe(1); expect(merged.page("occurrence").total).toBe(1); expect(merged.page("stopwatch").total).toBe(1);
      expect(merged.records<ScheduledOccurrence>("occurrence").next().value?.state).toBe("review-required");
      expect(merged.records<ScheduleRecord>("schedule").next().value?.task).toBe(state.schedule.task);
    } finally { merged.close(); }
  });
  it("rejects wrong-owner/corrupt learned-independent receipts without replacing the source", async () => {
    const source = await directory(); const state = await seed(source);
    await expect(rekeyClockWorkspace(source, "wrong-brain", "target")).rejects.toThrow(/owner/);
    const store = await ClockStore.open(source, "source-brain");
    try {
      expect(store.get<ScheduleRecord>("schedule", state.schedule.id)?.brainId).toBe("source-brain");
      const row = store.records<ScheduledOccurrence>("occurrence").next().value!;
      store.db.prepare("UPDATE records SET payload=? WHERE kind='occurrence' AND id=?").run(JSON.stringify({ ...row, input: "substituted" }), row.id);
    } finally { store.close(); }
    await expect(validateClockWorkspace(source, "source-brain")).rejects.toThrow(/differs/);
  });
  it("same-identity recovery retains request idempotency instead of archiving away its effect keys", async () => {
    const source = await directory(), target = await directory(); const state = await seed(source);
    await cloneLearningWorkspace(source, target, "source-brain", "source-brain");
    const store = await ClockStore.open(target, "source-brain");
    try {
      expect(Number(store.db.prepare("SELECT count(*) AS n FROM commands").get()?.n)).toBe(3);
      expect(store.get<ScheduleRecord>("schedule", state.schedule.id)?.pendingPaused).toBe(true);
      const command = store.db.prepare("SELECT result FROM commands WHERE id='create-one'").get();
      expect(JSON.parse(String(command?.result)).id).toBe(state.schedule.id);
    } finally { store.close(); }
  });
});
