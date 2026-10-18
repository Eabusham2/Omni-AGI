/** Host/SQLite fixtures only; no model, provider, real notification or timer waiting. */
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ClockService, type ClockServiceOptions, type ClockWakeStatus } from "../src/main/clockService";
import { hostClockSnapshot, type HostClockSource } from "../src/main/hostClock";
import { ClockStore } from "../src/main/clockStore";
import type { ScheduleRecord, ScheduledOccurrence, StopwatchRecord } from "../src/shared/clock";

const roots: string[] = [], services: ClockService[] = [];
afterEach(async () => { for (const service of services.splice(0)) service.stop(); for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-clock-fixture-")); roots.push(root);
  let wall = Date.parse("2026-10-11T12:00:00.000Z"), mono = 0, loaded = false, busy = false, enabled = true, revision = "a".repeat(64), admission = true;
  let session = "session-one";
  const source: HostClockSource = { wallMs: () => wall, monotonicMs: () => mono, get sessionId() { return session; } };
  const sent: ScheduledOccurrence[] = [], notified: ScheduledOccurrence[] = [], errors: Error[] = [], statuses = new Map<string, ClockWakeStatus>();
  const options: ClockServiceOptions = { brainDirectory: async () => root, source, authority: async () => ({ enabled, revision }), loaded: async () => loaded, busy: () => busy, admissionOpen: () => admission,
    turnStatus: async (_id, turn) => statuses.get(turn) ?? { state: "absent" },
    wake: async occurrence => { sent.push(occurrence); statuses.set(occurrence.turnId, { state: "committed" }); return { committed: true }; },
    notify: occurrence => { notified.push(occurrence); }, onError: error => errors.push(error) };
  const service = new ClockService(options); services.push(service);
  let request = 0;
  return { root, source, options, service, sent, notified, errors, statuses,
    advance(ms: number) { wall += ms; mono += ms; }, wall(ms: number) { wall += ms; },
    loaded(value: boolean) { loaded = value; }, busy(value: boolean) { busy = value; }, enabled(value: boolean) { enabled = value; },
    permission(value: string) { revision = value.repeat(64); }, admission(value: boolean) { admission = value; }, session(value: string) { session = value; },
    execute: (tool: string, action: string, args: Record<string, unknown>, id?: string) => service.execute("brain-one", tool, action, args, id ?? `request-${++request}`),
    async tick() { await service.tick(); await service.settle("brain-one"); expect(errors).toEqual([]); }
  };
}

describe("durable host clock and wake ownership", () => {
  it("reads actual injected UTC/local zone offsets without confusing wall and monotonic time", async () => {
    const f = await fixture(), snapshot = hostClockSnapshot(f.source, "Asia/Kolkata");
    expect(snapshot.utc).toBe("2026-10-11T12:00:00.000Z"); expect(snapshot.utcOffsetMinutes).toBe(330);
    expect(snapshot.monotonicMilliseconds).toBe(0);
    expect(() => hostClockSnapshot(f.source, "not/a-zone")).toThrow(/IANA/);
  });
  it("stopwatch laps use monotonic durations and restart excludes the unmeasured wall gap", async () => {
    const f = await fixture();
    const watch = await f.execute("studio.stopwatch", "start", { label: "Measured task" }) as StopwatchRecord;
    f.advance(2500); f.wall(-3_600_000);
    const lap = await f.execute("studio.stopwatch", "lap", { stopwatchId: watch.id }) as StopwatchRecord;
    expect(lap.elapsedMs).toBe(2500); expect((await f.service.laps("brain-one", watch.id)).entries[0]!.lapMs).toBe(2500);
    f.advance(6000); await f.tick();
    f.advance(20_000); f.session("session-two");
    const recovered = new ClockService(f.options); services.push(recovered); await recovered.recover(["brain-one"]);
    const saved = (await recovered.stopwatches("brain-one")).entries[0]!;
    expect(saved.state).toBe("paused"); expect(saved.elapsedMs).toBe(8500); expect(saved.pauseReason).toBe("restart"); expect(saved.excludedUnmeasuredGap).toBe(true);
    await recovered.execute("brain-one", "studio.stopwatch", "resume", { stopwatchId: watch.id }, "resume-one");
    f.advance(500); expect((await recovered.stopwatches("brain-one")).entries[0]!.elapsedMs).toBe(9000);
  });
  it("timer waits for monotonic duration then remains pending unloaded; loaded dispatch is exact and once", async () => {
    const f = await fixture();
    await f.execute("studio.schedule", "create", { label: "Actual reminder", kind: "timer", delivery: "wake", delayMs: 1000, task: "Check the selected document." });
    f.wall(86_400_000); await f.tick(); expect(f.notified).toHaveLength(0);
    f.advance(1000); await f.tick();
    expect(f.sent).toHaveLength(0); expect(f.notified).toHaveLength(1);
    expect((await f.service.occurrences("brain-one")).entries[0]!.waitingFor).toBe("load");
    f.loaded(true); await f.tick(); await f.tick();
    expect(f.sent).toHaveLength(1); expect(f.sent[0]!.input).toBe("Check the selected document.");
    expect((await f.service.occurrences("brain-one")).entries[0]!.state).toBe("committed");
  });
  it("coalesces missed fixed intervals without replaying a backlog and supports explicit skip", async () => {
    const f = await fixture();
    await f.execute("studio.schedule", "create", { label: "Coalesce", kind: "at", at: "2026-10-11T12:00:01Z", delivery: "notify", repeatMs: 1000 });
    f.advance(10_000); await f.tick();
    expect(f.notified).toHaveLength(1); expect(f.notified[0]!.missedIntervals).toBe(9);
    expect((await f.service.schedules("brain-one")).entries[0]!.nextIndex).toBe(10);
    await f.execute("studio.schedule", "create", { label: "Skip", kind: "at", at: "2026-10-11T12:00:01Z", delivery: "notify", repeatMs: 1000, missedPolicy: "skip" });
    await f.tick(); expect(f.notified).toHaveLength(1);
    expect((await f.service.occurrences("brain-one")).entries.some(row => row.state === "cancelled" && row.detail?.startsWith("Skipped 10"))).toBe(true);
  });
  it("pause/resume preserves timer remaining time; cancel retains receipts and prevents future work", async () => {
    const f = await fixture();
    const row = await f.execute("studio.schedule", "create", { label: "Pause me", kind: "timer", delayMs: 1000, delivery: "notify" }) as ScheduleRecord;
    f.advance(300); const paused = await f.execute("studio.schedule", "pause", { scheduleId: row.id }) as ScheduleRecord; expect(paused.remainingMs).toBe(700);
    f.advance(100_000); await f.tick(); expect(f.notified).toHaveLength(0);
    await f.execute("studio.schedule", "resume", { scheduleId: row.id }); f.advance(699); await f.tick(); expect(f.notified).toHaveLength(0);
    await f.execute("studio.schedule", "cancel", { scheduleId: row.id }); f.advance(100); await f.tick(); expect(f.notified).toHaveLength(0);
    expect((await f.service.schedules("brain-one")).entries[0]!.state).toBe("cancelled");
  });
  it("permission revisions cannot resurrect dormant scheduled authority without explicit resume", async () => {
    const f = await fixture(); f.loaded(true);
    const row = await f.execute("studio.schedule", "create", { label: "Bound scope", kind: "timer", delayMs: 1, delivery: "wake", task: "Inspect my files using permitted tools." }) as ScheduleRecord;
    f.permission("b"); f.advance(2); await f.tick(); expect(f.sent).toHaveLength(0);
    expect((await f.service.occurrences("brain-one")).entries[0]!.waitingFor).toBe("permission");
    await f.execute("studio.schedule", "resume", { scheduleId: row.id }); await f.tick(); expect(f.sent).toHaveLength(1);
    await f.tick(); expect(f.sent).toHaveLength(1);
  });
  it("busy and shutdown gates queue without steering, loading or granting permissions", async () => {
    const f = await fixture(); f.loaded(true); f.busy(true);
    await f.execute("studio.schedule", "create", { label: "Queue", kind: "timer", delayMs: 1, delivery: "wake", task: "Continue the saved work." });
    f.advance(2); await f.tick(); expect(f.sent).toHaveLength(0); expect((await f.service.occurrences("brain-one")).entries[0]!.waitingFor).toBe("busy");
    f.busy(false); f.admission(false); await f.tick(); expect(f.sent).toHaveLength(0); expect((await f.service.occurrences("brain-one")).entries[0]!.waitingFor).toBe("shutdown");
    f.admission(true); await f.tick(); expect(f.sent).toHaveLength(1);
  });
  it("restart consumes a committed receipt or reattaches an exact retained turn, never replays an ambiguous effect", async () => {
    const f = await fixture(); f.loaded(true);
    await f.execute("studio.schedule", "create", { label: "Crash case", kind: "timer", delayMs: 1, delivery: "wake", task: "Run the authorized task." });
    f.advance(2); f.busy(true); await f.tick();
    const occurrence = (await f.service.occurrences("brain-one")).entries[0]!;
    const store = await ClockStore.open(f.root, "brain-one"); store.put({ ...occurrence, state: "dispatching" }, "occurrence"); store.close();
    f.busy(false); await f.tick(); expect(f.sent).toHaveLength(0);
    expect((await f.service.occurrences("brain-one")).entries[0]!.state).toBe("review-required");
    const store2 = await ClockStore.open(f.root, "brain-one"); store2.put({ ...occurrence, state: "running" }, "occurrence"); store2.close();
    f.statuses.set(occurrence.turnId, { state: "committed", detail: "Exact original receipt" }); await f.tick();
    expect(f.sent).toHaveLength(0); expect((await f.service.occurrences("brain-one")).entries[0]!.state).toBe("committed");
  });
  it("typed events deduplicate source IDs and never evaluate event data as code or prompt instructions", async () => {
    const f = await fixture();
    await f.execute("studio.schedule", "create", { label: "Tracked event", kind: "event", eventKey: "source.changed", delivery: "notify" });
    const first = await f.execute("studio.schedule", "signal", { eventKey: "source.changed", eventId: "source-1", data: { text: "not executable" } }) as { occurrencesCreated: number };
    expect(first.occurrencesCreated).toBe(1);
    expect(await f.execute("studio.schedule", "signal", { eventKey: "source.changed", eventId: "source-1", data: { text: "not executable" } })).toMatchObject({ duplicate: true });
    await expect(f.execute("studio.schedule", "signal", { eventKey: "source.changed", eventId: "source-1", data: { text: "different" } })).rejects.toThrow(/reused/);
    await f.tick(); expect(f.notified).toHaveLength(1);
  });
  it("request idempotency and failed SQL transactions do not leak running stopwatch anchors", async () => {
    const f = await fixture();
    const first = await f.execute("studio.stopwatch", "start", { label: "One" }, "same-request") as StopwatchRecord;
    f.advance(100);
    expect(await f.execute("studio.stopwatch", "start", { label: "One" }, "same-request")).toEqual(first);
    expect((await f.service.stopwatches("brain-one")).entries[0]!.elapsedMs).toBe(100);
    await expect(f.execute("studio.stopwatch", "start", { label: "Different" }, "same-request")).rejects.toThrow(/reused/);
    const put = vi.spyOn(ClockStore.prototype, "put").mockImplementationOnce(() => { throw new Error("disk fixture failure"); });
    await expect(f.execute("studio.schedule", "create", { label: "Not stored", kind: "timer", delayMs: 10, delivery: "notify" })).rejects.toThrow(/disk fixture/); put.mockRestore();
    expect((await f.service.schedules("brain-one")).total).toBe(0);
    expect((f.service as unknown as { timers: Map<string, unknown> }).timers.size).toBe(0);
  });
  it("only an explicitly reviewed Load-when-due choice invokes normal Load; global Auto-load is untouched", async () => {
    const f = await fixture(); const load = vi.fn(async () => { f.loaded(true); }); f.options.load = load;
    await f.execute("studio.schedule", "create", { label: "Explicit online task", kind: "timer", delayMs: 10, delivery: "wake", task: "Run the selected task.", loadWhenDue: true });
    await f.tick(); expect(load).not.toHaveBeenCalled(); f.advance(10); await f.tick();
    expect(load).toHaveBeenCalledExactlyOnceWith("brain-one"); expect(f.sent).toHaveLength(1); expect(f.sent[0]?.loadWhenDue).toBe(true);
    await expect(f.execute("studio.schedule", "create", { label: "Invalid notification Load", kind: "timer", delayMs: 10, delivery: "notify", loadWhenDue: true })).rejects.toThrow(/wakeup-only/);
  });
  it("rechecks access around scheduled Load and leaves resource failures visibly pending with bounded retry", async () => {
    const f = await fixture(); const load = vi.fn(async () => { throw new Error("needs more resource headroom"); }); f.options.load = load;
    const schedule = await f.execute("studio.schedule", "create", { label: "Wait for resources", kind: "timer", delayMs: 1, delivery: "wake", task: "Continue my saved task.", loadWhenDue: true }) as ScheduleRecord;
    f.advance(2); await f.tick(); expect(load).toHaveBeenCalledTimes(1); expect(f.sent).toHaveLength(0);
    expect((await f.service.occurrences("brain-one")).entries[0]?.detail).toContain("normal resource admission");
    f.advance(1000); await f.tick(); expect(load).toHaveBeenCalledTimes(1);
    f.permission("b"); f.advance(10_000); await f.tick(); expect(load).toHaveBeenCalledTimes(1);
    await f.execute("studio.schedule", "resume", { scheduleId: schedule.id });
    f.options.load = vi.fn(async () => { f.loaded(true); f.permission("c"); }); await f.tick();
    expect(f.sent).toHaveLength(0); expect((await f.service.occurrences("brain-one")).entries[0]?.waitingFor).toBe("permission");
  });
  it("does not Load or resend a turn whose committed receipt already exists", async () => {
    const f = await fixture(); const load = vi.fn(async () => { f.loaded(true); }); f.options.load = load;
    await f.execute("studio.schedule", "create", { label: "Recover exact receipt", kind: "timer", delayMs: 1, delivery: "wake", task: "Previously run task.", loadWhenDue: true });
    f.advance(2); f.admission(false); await f.tick(); const occurrence = (await f.service.occurrences("brain-one")).entries[0]!;
    const store = await ClockStore.open(f.root, "brain-one"); store.put({ ...occurrence, state: "running", dispatchClaimedAt: "2026-10-11T12:00:00.001Z" }, "occurrence"); store.close();
    f.statuses.set(occurrence.turnId, { state: "committed" }); f.admission(true); await f.tick();
    expect(load).not.toHaveBeenCalled(); expect(f.sent).toHaveLength(0); expect((await f.service.occurrences("brain-one")).entries[0]?.state).toBe("committed");
  });
  it("captures future edits separately from a pending task's exact Load/input authorization", async () => {
    const f = await fixture(); const load = vi.fn(async () => { f.loaded(true); }); f.options.load = load;
    const schedule = await f.execute("studio.schedule", "create", { label: "Original scope", kind: "timer", delayMs: 1, delivery: "wake", task: "Original task." }) as ScheduleRecord;
    f.advance(2); await f.tick();
    await f.execute("studio.schedule", "edit", { scheduleId: schedule.id, expectedRevision: 1, task: "Different future task.", loadWhenDue: true });
    await f.tick(); expect(load).not.toHaveBeenCalled(); expect(f.sent).toHaveLength(0);
    const old = (await f.service.occurrences("brain-one")).entries[0]!;
    expect(old.input).toBe("Original task."); expect(old.loadWhenDue).toBe(false); expect(old.scheduleRevision).toBe(1);
  });
  it("records real native counts/durations and ignores invented or incomplete rate events", async () => {
    const f = await fixture();
    f.service.observeStream({ type: "chat-token", id: "event-one", brainId: "brain-one", turnId: "actual-turn", sequence: 0,
      createdAt: "2026-10-11T12:00:00.000Z", delta: "not stored here", outputTokens: 40, outputElapsedMs: 2000, tokensPerSecond: 9999 });
    f.service.observeStream({ type: "chat-token", id: "event-two", brainId: "brain-one", turnId: "incomplete-turn", sequence: 0,
      createdAt: "2026-10-11T12:00:00.000Z", delta: "no measured counters" });
    const page = await f.service.rates("brain-one"); expect(page.total).toBe(1); expect(page.entries[0]?.tokensPerSecond).toBe(20);
    expect(JSON.stringify(page)).not.toContain("not stored here");
  });
  it("pausing an absolute calendar instant does not silently shift it to a new date on resume", async () => {
    const f = await fixture(); const at = "2026-10-11T12:00:10.000Z";
    const schedule = await f.execute("studio.schedule", "create", { label: "Fixed date", kind: "at", at, delivery: "notify" }) as ScheduleRecord;
    f.advance(5000); await f.execute("studio.schedule", "pause", { scheduleId: schedule.id }); f.advance(60_000);
    const resumed = await f.execute("studio.schedule", "resume", { scheduleId: schedule.id }) as ScheduleRecord;
    expect(resumed.dueAt).toBe(at); await f.tick(); expect(f.notified).toHaveLength(1);
  });
  it("actual sleep reconciliation counts missing timer time without changing monotonic stopwatch readings", async () => {
    const f = await fixture();
    await f.execute("studio.schedule", "create", { label: "Sleep timer", kind: "timer", delayMs: 60000, delivery: "notify" });
    const watch = await f.execute("studio.stopwatch", "start", { label: "Monotonic awake measurement" }) as StopwatchRecord;
    f.advance(1000); f.service.systemSuspending(); f.wall(3600000);
    await f.service.systemResumed(); await f.service.settle("brain-one");
    expect(f.notified).toHaveLength(1); expect((await f.service.stopwatches("brain-one")).entries.find(row => row.id === watch.id)?.elapsedMs).toBe(1000);
    await f.service.systemResumed(); await f.tick(); expect(f.notified).toHaveLength(1);
  });
});
