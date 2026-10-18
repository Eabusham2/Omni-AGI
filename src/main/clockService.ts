import { randomUUID } from "node:crypto";
import { lstat } from "node:fs/promises";
import { join } from "node:path";
import type { ClockPage, ClockPageQuery, ClockWorkspaceEvent, ClockWorkspaceSnapshot, ScheduleCreateRequest,
  ScheduleEditRequest, ScheduleRecord, ScheduledOccurrence, StopwatchRecord, NativeInferenceRate } from "../shared/clock";
import type { ChatStreamEvent } from "../shared/types";
import { CLOCK_RELATIVE_PATH, ClockStore, clockHash, clockId, clockInstant, clockMilliseconds, clockObject, clockText } from "./clockStore";
import { hostClockSnapshot, realHostClock, validTimeZone, type HostClockSource } from "./hostClock";
import { validateScheduledAutomationReference } from "./automationFormat";

export interface ClockWakeStatus {
  state: "committed" | "retained" | "active" | "accepted-unfinished" | "absent";
  continuation?: unknown;
  detail?: string;
}
export interface ClockServiceOptions {
  brainDirectory(brainId: string): Promise<string>;
  source?: HostClockSource;
  authority(brainId: string, toolId?: string): Promise<{ enabled: boolean; revision: string }>;
  loaded(brainId: string): Promise<boolean>;
  load?(brainId: string): Promise<void>;
  canRun?(brainId: string, turnId: string): boolean;
  busy(brainId: string): boolean;
  admissionOpen(): boolean;
  turnStatus(brainId: string, turnId: string, input: string): Promise<ClockWakeStatus>;
  /** Same normal authorized controller; no alternate model, persona or auto permission grant. */
  wake(occurrence: ScheduledOccurrence, continuation?: unknown): Promise<{ committed: boolean; detail?: string }>;
  notify(occurrence: ScheduledOccurrence): Promise<void> | void;
  reserveWrite?(root: string, bytes: number): Promise<void>;
  onError?(error: Error, brainId: string): void;
}
interface Anchor { mono: number; base: number }
const TERMINAL = ["committed", "notified", "delegated", "cancelled"];

function known(input: Record<string, unknown>, allowed: readonly string[]): void {
  if (Object.keys(input).some(key => !allowed.includes(key))) throw new Error("Clock request contains unsupported fields.");
}
function safeDeadline(value: number): string {
  if (!Number.isSafeInteger(value) || !Number.isFinite(new Date(value).getTime())) throw new Error("Deadline exceeds supported calendar precision.");
  return new Date(value).toISOString();
}

/** A host event loop, not another Active-mode or neural scheduler. */
export class ClockService {
  private readonly source: HostClockSource;
  private readonly brains = new Set<string>();
  private readonly watches = new Map<string, Anchor>();
  private readonly timers = new Map<string, Anchor>();
  private readonly listeners = new Set<(event: ClockWorkspaceEvent) => void>();
  private readonly delivering = new Map<string, Promise<void>>();
  private readonly automationDelivering = new Map<string, Promise<void>>();
  private readonly ownerEpoch = new Map<string, number>();
  private readonly latestRates = new Map<string, { brainId: string; rate: NativeInferenceRate }>();
  private readonly nextLoadAttempt = new Map<string, number>();
  private timer?: NodeJS.Timeout;
  private ticking = false;
  private quitting = false;
  private lastWatchCheckpoint = 0;
  private sleepBoundary?: { wall: number; mono: number };
  private reconcilingSleep = false;
  private pendingAnchorChanges?: Array<() => void>;
  private automationDelegate?: (occurrence: ScheduledOccurrence) => Promise<{ accepted: true; runId: string }>;
  constructor(private readonly options: ClockServiceOptions) { this.source = options.source ?? realHostClock; }
  /** Typed, idempotent durable handoff. It must never submit a disguised wake prompt. */
  setAutomationDelegate(delegate: (occurrence: ScheduledOccurrence) => Promise<{ accepted: true; runId: string }>): void { this.automationDelegate = delegate; }
  private key(brainId: string, id: string): string { return `${brainId}:${id}`; }
  private anchor(map: Map<string, Anchor>, key: string, value?: Anchor): void {
    const apply = (): void => { if (value === undefined) map.delete(key); else map.set(key, value); };
    if (this.pendingAnchorChanges) this.pendingAnchorChanges.push(apply); else apply();
  }
  private commitAnchors<T>(operation: () => T): T {
    if (this.pendingAnchorChanges) throw new Error("Clock anchor transaction already active.");
    const changes: Array<() => void> = []; this.pendingAnchorChanges = changes;
    try { const result = operation(); for (const apply of changes) apply(); return result; }
    finally { this.pendingAnchorChanges = undefined; }
  }
  private iso(): string { return safeDeadline(Math.floor(this.source.wallMs())); }
  private async store(brainId: string): Promise<ClockStore> {
    const root = await this.options.brainDirectory(clockId(brainId));
    await this.options.reserveWrite?.(root, 64 * 1024);
    this.brains.add(brainId); return ClockStore.open(root, brainId);
  }
  subscribe(listener: (event: ClockWorkspaceEvent) => void): () => void { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; }
  private emit(brainId: string): void { const event: ClockWorkspaceEvent = { brainId, type: "changed", at: this.iso() }; for (const listener of this.listeners) listener(event); }
  private currentWatch(row: StopwatchRecord): StopwatchRecord {
    const anchor = this.watches.get(this.key(row.brainId, row.id));
    return anchor && row.state === "running" ? { ...row, elapsedMs: anchor.base + Math.max(0, this.source.monotonicMs() - anchor.mono) } : row;
  }
  private remaining(row: ScheduleRecord): number {
    const anchor = this.timers.get(this.key(row.brainId, row.id));
    return row.kind === "timer" && anchor ? Math.max(0, anchor.base - Math.max(0, this.source.monotonicMs() - anchor.mono))
      : Math.max(0, Date.parse(row.dueAt ?? this.iso()) - this.source.wallMs());
  }
  async recover(brainIds: readonly string[]): Promise<void> {
    for (const brainId of brainIds) {
      const root = await this.options.brainDirectory(clockId(brainId));
      if (!await lstat(join(root, CLOCK_RELATIVE_PATH)).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; })) continue;
      const store = await this.store(brainId);
      try {
        this.commitAnchors(() => store.transaction(() => {
          for (const row of store.records<StopwatchRecord>("stopwatch", ["running"])) {
            if (row.sessionId === this.source.sessionId && this.watches.has(this.key(brainId, row.id))) continue;
            // hrtime origins do not survive a process restart. Never add wall time to a stopwatch.
            store.put({ ...row, state: "paused", sessionId: this.source.sessionId, pauseReason: "restart", excludedUnmeasuredGap: true, revision: row.revision + 1 }, "stopwatch");
          }
          for (const row of store.records<ScheduleRecord>("schedule", ["active"])) if (row.kind === "timer")
            this.anchor(this.timers, this.key(brainId, row.id), { mono: this.source.monotonicMs(), base: Math.max(0, Date.parse(row.dueAt!) - this.source.wallMs()) });
        }));
      } finally { store.close(); }
    }
  }
  start(): void {
    if (this.timer) return; this.quitting = false;
    this.timer = setInterval(() => { void this.tick().catch(error => this.options.onError?.(error, "host-clock")); }, 1000);
    this.timer.unref();
  }
  stop(): void { this.quitting = true; if (this.timer) clearInterval(this.timer); this.timer = undefined; }
  systemSuspending(): void {
    this.sleepBoundary ??= { wall: this.source.wallMs(), mono: this.source.monotonicMs() };
  }
  /** Only an actual OS suspend/resume event reconciles sleep missing from hrtime. */
  async systemResumed(): Promise<void> {
    const boundary = this.sleepBoundary; this.sleepBoundary = undefined;
    if (!boundary || this.quitting) return;
    const missing = Math.max(0, this.source.wallMs() - boundary.wall - Math.max(0, this.source.monotonicMs() - boundary.mono));
    this.reconcilingSleep = true;
    try {
      if (missing > 0) for (const brainId of this.brains) {
        const store = await this.store(brainId);
        try { this.commitAnchors(() => store.transaction(() => {
          for (const row of store.records<ScheduleRecord>("schedule", ["active"])) {
            if (row.kind !== "timer") continue;
            const key = this.key(brainId, row.id), anchor = this.timers.get(key); if (!anchor) continue;
            const base = anchor.base - missing;
            store.put({ ...row, dueAt: safeDeadline(Math.floor(this.source.wallMs() + base - Math.max(0, this.source.monotonicMs() - anchor.mono))), updatedAt: this.iso() }, "schedule");
            this.anchor(this.timers, key, { ...anchor, base });
          }
        })); } finally { store.close(); }
      }
      this.nextLoadAttempt.clear();
    } finally { this.reconcilingSleep = false; }
    await this.tick();
  }
  async checkpointBeforeQuit(): Promise<void> {
    this.stop();
    for (const brainId of this.brains) {
      const store = await this.store(brainId);
      try { store.transaction(() => {
        for (const old of store.records<StopwatchRecord>("stopwatch", ["running"])) {
          const row = this.currentWatch(old); store.put({ ...row, state: "paused", pauseReason: "shutdown", recordedAt: this.iso(), revision: row.revision + 1 }, "stopwatch");
        }
        for (const row of store.records<ScheduleRecord>("schedule", ["active"])) if (row.kind === "timer")
          store.put({ ...row, dueAt: safeDeadline(Math.floor(this.source.wallMs() + this.remaining(row))), updatedAt: this.iso() }, "schedule");
      }); } finally { store.close(); }
    }
    this.watches.clear(); this.timers.clear();
  }
  async forget(brainId: string): Promise<void> {
    this.ownerEpoch.set(brainId, (this.ownerEpoch.get(brainId) ?? 0) + 1);
    this.brains.delete(brainId);
    this.nextLoadAttempt.delete(brainId);
    for (const key of [...this.watches.keys(), ...this.timers.keys()]) if (key.startsWith(`${brainId}:`)) { this.watches.delete(key); this.timers.delete(key); }
    for (const [key, row] of this.latestRates) if (row.brainId === brainId) this.latestRates.delete(key);
  }
  /** Metadata-only actual native stream; no generated words are stored here. */
  observeStream(event: ChatStreamEvent): void {
    if (event.type !== "chat-token" || !Number.isSafeInteger(event.outputTokens) || event.outputTokens! < 0 ||
        typeof event.outputElapsedMs !== "number" || !Number.isFinite(event.outputElapsedMs) || event.outputElapsedMs < 0) return;
    clockId(event.brainId); clockId(event.turnId); clockInstant(event.createdAt);
    const rate: NativeInferenceRate = { turnId: event.turnId, at: event.createdAt, outputTokens: event.outputTokens!, outputElapsedMs: event.outputElapsedMs,
      tokensPerSecond: event.outputElapsedMs > 0 ? event.outputTokens! * 1000 / event.outputElapsedMs : null,
      source: "native-output-stream", scope: "one-turn-cumulative" };
    this.latestRates.set(this.key(event.brainId, event.turnId), { brainId: event.brainId, rate }); this.brains.add(event.brainId);
  }
  async rates(brainId: string, query: ClockPageQuery = {}): Promise<ClockPage<NativeInferenceRate>> {
    const store = await this.store(brainId);
    try {
      const after = query.after === undefined ? "" : clockId(query.after), limit = clockMilliseconds(query.limit ?? 50, 1);
      if (limit > 200) throw new Error("Follow the rate-history cursor in batches of up to 200.");
      for (const [key, entry] of this.latestRates) if (entry.brainId === brainId) {
        store.db.prepare("INSERT INTO rates VALUES(?,?) ON CONFLICT(turn_id) DO UPDATE SET payload=excluded.payload").run(entry.rate.turnId, JSON.stringify(entry.rate)); this.latestRates.delete(key);
      }
      const rows = store.db.prepare("SELECT turn_id,payload FROM rates WHERE turn_id>? ORDER BY turn_id LIMIT ?").all(after, limit + 1);
      return { entries: rows.slice(0, limit).map(row => ({ ...JSON.parse(String(row.payload)) as NativeInferenceRate, recordId: String(row.turn_id) })), total: Number(store.db.prepare("SELECT count(*) AS n FROM rates").get()?.n ?? 0),
        ...(rows.length > limit ? { nextAfter: String(rows[limit - 1]!.turn_id) } : {}) };
    } finally { store.close(); }
  }
  async snapshot(brainId: string): Promise<ClockWorkspaceSnapshot> {
    const store = await this.store(brainId);
    try {
      const count = (kind: string, states?: string[]): number => Number(store.db.prepare(`SELECT count(*) AS n FROM records WHERE kind=?${states ? ` AND state IN (${states.map(() => "?").join(",")})` : ""}`).get(kind, ...(states ?? []))?.n ?? 0);
      return { brainId, clock: hostClockSnapshot(this.source), schedulerRunning: !!this.timer && !this.quitting,
        counts: { schedules: count("schedule"), stopwatches: count("stopwatch"), pending: count("occurrence", ["pending", "dispatching", "running"]), reviewRequired: count("occurrence", ["review-required"]) }, processResidentRequired: true };
    } finally { store.close(); }
  }
  now(timeZone?: string) { return hostClockSnapshot(this.source, timeZone); }
  async schedules(brainId: string, query?: ClockPageQuery): Promise<ClockPage<ScheduleRecord>> {
    const store = await this.store(brainId); try { const page = store.page<ScheduleRecord>("schedule", query);
      return { ...page, entries: page.entries.map(row => row.kind === "timer" && row.state === "active" ? { ...row, remainingMs: this.remaining(row) } : row) }; } finally { store.close(); }
  }
  async stopwatches(brainId: string, query?: ClockPageQuery): Promise<ClockPage<StopwatchRecord>> {
    const store = await this.store(brainId); try { const page = store.page<StopwatchRecord>("stopwatch", query); return { ...page, entries: page.entries.map(row => this.currentWatch(row)) }; } finally { store.close(); }
  }
  async occurrences(brainId: string, query?: ClockPageQuery): Promise<ClockPage<ScheduledOccurrence>> {
    const store = await this.store(brainId); try { return store.page("occurrence", query); } finally { store.close(); }
  }
  async laps(brainId: string, stopwatchId: string, after?: number) { const store = await this.store(brainId); try { return store.laps(stopwatchId, after); } finally { store.close(); } }

  /** Called only after ToolExecutor's permission and durable pre-effect gate. */
  async execute(brainId: string, toolId: string, action: string, argsValue: unknown, requestId: string, signal?: AbortSignal): Promise<unknown> {
    const args = clockObject(argsValue); signal?.throwIfAborted();
    if (toolId === "system.time") { known(args, ["timeZone"]); if (action !== "now") throw new Error("Unknown clock action."); return hostClockSnapshot(this.source, args.timeZone === undefined ? undefined : validTimeZone(args.timeZone)); }
    if (toolId === "studio.stopwatch" && action === "performance") { known(args, ["after", "limit"]); return { measuredAt: this.iso(), rates: await this.rates(brainId, args as ClockPageQuery), estimatesAreGuarantees: false }; }
    if (toolId === "studio.schedule" && action === "list") { known(args, ["after", "limit"]); return this.schedules(brainId, args as ClockPageQuery); }
    if (toolId === "studio.schedule" && action === "occurrences") { known(args, ["after", "limit"]); return this.occurrences(brainId, args as ClockPageQuery); }
    if (toolId === "studio.schedule" && action === "occurrence") { known(args, ["occurrenceId"]); const store = await this.store(brainId);
      try { const row = store.get<ScheduledOccurrence>("occurrence", clockId(args.occurrenceId)); if (!row) throw new Error("Saved wakeup not found."); return row; } finally { store.close(); } }
    if (toolId === "studio.stopwatch" && action === "list") { known(args, ["after", "limit"]); return this.stopwatches(brainId, args as ClockPageQuery); }
    if (toolId === "studio.stopwatch" && action === "laps") { known(args, ["stopwatchId", "after"]); return this.laps(brainId, clockId(args.stopwatchId), args.after === undefined ? undefined : clockMilliseconds(args.after)); }
    const authority = await this.options.authority(brainId, toolId), store = await this.store(brainId);
    try {
      signal?.throwIfAborted();
      if (!authority.enabled) throw new Error("Scheduled work is disabled for this instance.");
      if (toolId === "studio.schedule" && action === "resume-occurrence") {
        known(args, ["occurrenceId", "requestKey"]);
        const prior = store.get<ScheduledOccurrence>("occurrence", clockId(args.occurrenceId)); if (!prior) throw new Error("Saved wakeup not found.");
        if (TERMINAL.includes(prior.state)) return prior;
        if (prior.delivery === "automation" && prior.originBrainId && prior.originBrainId !== brainId) throw new Error("Copied historical action occurrences remain dormant. Review the action file and create a new trigger; prior effects cannot be replayed.");
        const status = prior.delivery === "automation" ? { state: "absent" as const } : await this.options.turnStatus(brainId, prior.turnId, prior.input); signal?.throwIfAborted();
        if (prior.delivery !== "automation" && !["retained", "active", "committed"].includes(status.state) && !(status.state === "absent" && prior.copiedUnstarted && !prior.dispatchClaimedAt))
          throw new Error("No exact supported continuation or proven unstarted copied task is available. The original effects will not be replayed.");
        const schedule = store.get<ScheduleRecord>("schedule", prior.scheduleId); if (!schedule) throw new Error("The schedule owner is missing.");
        if (schedule.state === "cancelled") throw new Error("Cancelled schedules do not resume pending work. Create an explicitly new schedule instead.");
        const result = store.command(args.requestKey === undefined ? requestId : clockId(args.requestKey), clockHash(JSON.stringify({ toolId, action, args })), () => {
          const current = store.get<ScheduledOccurrence>("occurrence", prior.id);
          if (!current || current.inputSha256 !== prior.inputSha256 || current.turnId !== prior.turnId) throw new Error("Saved wakeup changed during review.");
          if (TERMINAL.includes(current.state)) return current;
          store.put({ ...schedule, authorizationRevision: authority.revision, pendingPaused: false }, "schedule");
          const row: ScheduledOccurrence = { ...prior, state: status.state === "committed" ? "committed" : "pending", waitingFor: undefined, updatedAt: this.iso(),
            authorizationRevision: authority.revision,
            detail: status.state === "committed" ? "Original committed receipt retained; no replay." : "Reviewed exact saved task queued for normal continuation." };
          store.put(row, "occurrence"); return row;
        }); this.emit(brainId); if (prior.delivery === "automation") this.requestDelivery(brainId, true); return result;
      }
      const result = this.commitAnchors(() => store.command(args.requestKey === undefined ? requestId : clockId(args.requestKey), clockHash(JSON.stringify({ toolId, action, args })), () => {
        if (toolId === "studio.stopwatch") return this.stopwatchOperation(store, action, args);
        if (toolId === "studio.schedule") return this.scheduleOperation(store, action, args, authority.revision);
        throw new Error("Unknown clock tool.");
      }));
      this.emit(brainId); if (toolId === "studio.schedule" && action === "signal") this.requestDelivery(brainId, true); return result;
    } finally { store.close(); }
  }
  private stopwatchOperation(store: ClockStore, action: string, args: Record<string, unknown>): unknown {
    known(args, action === "start" ? ["label", "requestKey"] : ["stopwatchId", "label", "requestKey"]);
    const now = this.iso();
    if (action === "start") {
      const row: StopwatchRecord = { id: randomUUID(), brainId: store.brainId, label: clockText(args.label, "a stopwatch name"), revision: 1, state: "running", elapsedMs: 0, lapCount: 0, recordedAt: now, sessionId: this.source.sessionId };
      store.put(row, "stopwatch"); this.anchor(this.watches, this.key(row.brainId, row.id), { mono: this.source.monotonicMs(), base: 0 }); return row;
    }
    const old = store.get<StopwatchRecord>("stopwatch", clockId(args.stopwatchId)); if (!old) throw new Error("Stopwatch not found.");
    const row = this.currentWatch(old), key = this.key(row.brainId, row.id);
    if (action === "status") return row;
    if (!["lap", "pause", "resume", "reset"].includes(action)) throw new Error("Unknown stopwatch action.");
    const updated: StopwatchRecord = { ...row, revision: row.revision + 1, recordedAt: now, sessionId: this.source.sessionId };
    if (action === "lap") {
      if (row.state !== "running") throw new Error("Resume the stopwatch before recording a lap.");
      const last = store.db.prepare("SELECT elapsed_ms FROM laps WHERE watch_id=? ORDER BY number DESC LIMIT 1").get(row.id);
      updated.lapCount++;
      store.db.prepare("INSERT INTO laps VALUES(?,?,?,?,?)").run(row.id, updated.lapCount, row.elapsedMs, row.elapsedMs - Number(last?.elapsed_ms ?? 0), now);
      this.anchor(this.watches, key, { mono: this.source.monotonicMs(), base: row.elapsedMs });
    } else if (action === "pause") { updated.state = "paused"; updated.pauseReason = "user"; this.anchor(this.watches, key); }
    else if (action === "resume") { updated.state = "running"; delete updated.pauseReason; this.anchor(this.watches, key, { mono: this.source.monotonicMs(), base: row.elapsedMs }); }
    else { updated.state = "paused"; updated.elapsedMs = 0; updated.lapCount = 0; updated.pauseReason = "user"; delete updated.excludedUnmeasuredGap; this.anchor(this.watches, key); store.db.prepare("DELETE FROM laps WHERE watch_id=?").run(row.id); }
    store.put(updated, "stopwatch"); return updated;
  }
  private scheduleOperation(store: ClockStore, action: string, args: Record<string, unknown>, revision: string): unknown {
    const now = this.iso();
    if (action === "create") {
      known(args, ["label", "kind", "delivery", "automation", "task", "delayMs", "at", "timeZone", "repeatMs", "eventKey", "missedPolicy", "requestKey", "loadWhenDue"]);
      const input = args as unknown as ScheduleCreateRequest;
      if (!["timer", "at", "event"].includes(input.kind) || !["notify", "wake", "automation"].includes(input.delivery)) throw new Error("Choose a timer, calendar instant or event and notification, wakeup or action-file delivery.");
      const row: ScheduleRecord = { id: randomUUID(), brainId: store.brainId, label: clockText(input.label, "a schedule name"), revision: 1,
        kind: input.kind, state: "active", delivery: input.delivery, task: input.task === undefined ? "" : clockText(input.task, "task text", true),
        timeZone: validTimeZone(input.timeZone), loadWhenDue: input.loadWhenDue ?? false, missedPolicy: input.missedPolicy ?? "coalesce", nextIndex: 0, authorizationRevision: revision, createdAt: now, updatedAt: now,
        ...(input.repeatMs === undefined ? {} : { repeatMs: clockMilliseconds(input.repeatMs, 1) }) };
      if (input.automation !== undefined) row.automation = validateScheduledAutomationReference(input.automation);
      if (input.kind === "event") { if (input.repeatMs !== undefined || input.at !== undefined || input.delayMs !== undefined) throw new Error("Event schedules run once per unique signal, not an interval."); row.eventKey = clockText(input.eventKey, "an event key"); }
      else if (input.kind === "at") { if (input.delayMs !== undefined || input.eventKey !== undefined) throw new Error("A calendar schedule accepts at, not a delay or event key."); row.dueAt = clockInstant(input.at); }
      else { if (input.at !== undefined || input.eventKey !== undefined) throw new Error("A timer accepts a delay, not at or an event key."); const delay = clockMilliseconds(input.delayMs, 1); row.dueAt = safeDeadline(Math.floor(this.source.wallMs() + delay));
        this.anchor(this.timers, this.key(row.brainId, row.id), { mono: this.source.monotonicMs(), base: delay }); }
      store.put(row, "schedule"); return row;
    }
    if (action === "signal") {
      known(args, ["eventKey", "eventId", "data", "requestKey"]);
      const eventKey = clockText(args.eventKey, "an event key"), eventId = clockId(args.eventId), digest = clockHash(JSON.stringify(args.data ?? null));
      const prior = store.db.prepare("SELECT sha256 FROM events WHERE event_key=? AND event_id=?").get(eventKey, eventId);
      if (prior) { if (prior.sha256 !== digest) throw new Error("A source event ID was reused with different data."); return { duplicate: true, eventId, occurrencesCreated: 0 }; }
      store.db.prepare("INSERT INTO events VALUES(?,?,?)").run(eventKey, eventId, digest);
      let count = 0;
      for (const row of store.records<ScheduleRecord>("schedule", ["active"])) if (row.kind === "event" && row.eventKey === eventKey) {
        this.occurrence(store, row, now, 0, eventId, args.data); store.put({ ...row, nextIndex: row.nextIndex + 1, updatedAt: now }, "schedule"); count++;
      }
      return { duplicate: false, eventId, occurrencesCreated: count };
    }
    known(args, action === "edit" ? ["scheduleId", "expectedRevision", "label", "task", "delivery", "automation", "at", "delayMs", "timeZone", "repeatMs", "missedPolicy", "requestKey", "loadWhenDue"] : ["scheduleId", "requestKey"]);
    const old = store.get<ScheduleRecord>("schedule", clockId(args.scheduleId)); if (!old) throw new Error("Schedule not found.");
    if (action === "status") return old;
    if (!["edit", "pause", "resume", "cancel"].includes(action)) throw new Error("Unknown schedule action.");
    if (old.state === "cancelled" && action !== "edit") return old;
    const row: ScheduleRecord = { ...old, revision: old.revision + 1, updatedAt: now, authorizationRevision: revision };
    if (action === "pause") { row.pendingPaused = true; row.pauseReason = "user"; if (row.state !== "finished") {
      row.state = "paused"; if (row.kind === "timer") row.remainingMs = Math.ceil(this.remaining(old)); this.anchor(this.timers, this.key(row.brainId, row.id)); }
    }
    else if (action === "resume") { if (row.state !== "finished") row.state = "active"; delete row.pauseReason; delete row.pendingPaused; if (row.kind === "timer" && row.remainingMs !== undefined) {
      row.dueAt = safeDeadline(Math.floor(this.source.wallMs() + row.remainingMs)); if (row.kind === "timer") this.anchor(this.timers, this.key(row.brainId, row.id), { mono: this.source.monotonicMs(), base: row.remainingMs }); delete row.remainingMs;
    }
      for (const occurrence of store.records<ScheduledOccurrence>("occurrence", ["pending"])) if (occurrence.scheduleId === row.id)
        store.put({ ...occurrence, authorizationRevision: revision, updatedAt: now }, "occurrence");
    }
    else if (action === "cancel") {
      row.state = "cancelled"; this.anchor(this.timers, this.key(row.brainId, row.id));
      for (const occurrence of store.records<ScheduledOccurrence>("occurrence", ["pending", "review-required"])) if (occurrence.scheduleId === row.id) store.put({ ...occurrence, state: "cancelled", waitingFor: undefined, updatedAt: now, detail: "Future/pending wakeup cancelled; running work is controlled by the normal chat Stop." }, "occurrence");
    } else {
      const input = args as unknown as ScheduleEditRequest;
      if (clockMilliseconds(input.expectedRevision, 1) !== old.revision) throw new Error("This schedule changed. Refresh before editing.");
      if (input.label !== undefined) row.label = clockText(input.label, "a schedule name");
      if (input.task !== undefined) row.task = clockText(input.task, "task text", true);
      if (input.delivery !== undefined) row.delivery = input.delivery;
      if (input.automation !== undefined) row.automation = validateScheduledAutomationReference(input.automation);
      if (input.delivery !== undefined && input.delivery !== "automation") delete row.automation;
      if (input.loadWhenDue !== undefined) row.loadWhenDue = input.loadWhenDue;
      if (input.timeZone !== undefined) row.timeZone = validTimeZone(input.timeZone);
      if (input.missedPolicy !== undefined) row.missedPolicy = input.missedPolicy;
      if (input.repeatMs === 0) delete row.repeatMs; else if (input.repeatMs !== undefined) row.repeatMs = clockMilliseconds(input.repeatMs, 1);
      if (input.at !== undefined) { if (row.kind !== "at") throw new Error("Only a calendar schedule accepts at."); row.dueAt = clockInstant(input.at); if (row.state === "finished" || row.state === "cancelled") row.state = "active"; }
      if (input.delayMs !== undefined) { if (row.kind !== "timer") throw new Error("Only a timer accepts a delay."); const delay = clockMilliseconds(input.delayMs, 1); row.dueAt = safeDeadline(Math.floor(this.source.wallMs() + delay));
        if (row.state === "finished" || row.state === "cancelled") row.state = "active";
        if (row.state === "active") this.anchor(this.timers, this.key(row.brainId, row.id), { mono: this.source.monotonicMs(), base: delay }); else row.remainingMs = delay; }
      // Pending occurrences retain their exact authorized input; edits govern only future occurrences.
    }
    store.put(row, "schedule"); return row;
  }
  private occurrence(store: ClockStore, row: ScheduleRecord, dueAt: string, missedIntervals: number, eventId?: string, eventData?: unknown): ScheduledOccurrence {
    const id = clockHash(`${row.id}\0${row.nextIndex}`), occurrence: ScheduledOccurrence = { id, brainId: store.brainId, scheduleId: row.id, index: row.nextIndex,
      turnId: `scheduled-${id}`, input: row.task, inputSha256: clockHash(row.task), delivery: row.delivery, label: row.label, dueAt,
      authorizationRevision: row.authorizationRevision, scheduleRevision: row.revision, loadWhenDue: row.loadWhenDue,
      createdAt: this.iso(), updatedAt: this.iso(), missedIntervals, state: "pending", ...(row.automation ? { automation: structuredClone(row.automation) } : {}), ...(eventId ? { eventId } : {}), ...(eventData === undefined ? {} : { eventData }) };
    if (!store.get("occurrence", id)) store.put(occurrence, "occurrence"); return occurrence;
  }
  async tick(): Promise<void> {
    if (this.ticking || this.quitting || this.reconcilingSleep) return; this.ticking = true;
    try {
      const checkpointWatches = this.source.monotonicMs() - this.lastWatchCheckpoint >= 5000;
      for (const brainId of this.brains) {
        try {
          let changed = false;
          const store = await this.store(brainId);
          try {
            this.commitAnchors(() => store.transaction(() => {
              if (checkpointWatches) for (const old of store.records<StopwatchRecord>("stopwatch", ["running"])) {
                changed = true;
                const row = this.currentWatch(old); store.put({ ...row, recordedAt: this.iso() }, "stopwatch");
                this.anchor(this.watches, this.key(brainId, row.id), { mono: this.source.monotonicMs(), base: row.elapsedMs });
              }
              for (const row of store.records<ScheduleRecord>("schedule", ["active"])) {
                if (row.kind === "event" || this.remaining(row) > 0) continue;
                changed = true;
                const due = Date.parse(row.dueAt!), lateness = row.kind === "timer" ? Math.max(0, (this.source.monotonicMs() - (this.timers.get(this.key(brainId, row.id))?.mono ?? this.source.monotonicMs())) - (this.timers.get(this.key(brainId, row.id))?.base ?? 0)) : Math.max(0, this.source.wallMs() - due);
                const missed = row.repeatMs === undefined ? 0 : Math.floor(lateness / row.repeatMs);
                const occurrence = this.occurrence(store, row, row.dueAt!, missed);
                if (row.missedPolicy === "skip" && missed > 0) store.put({ ...occurrence, state: "cancelled", detail: `Skipped ${missed + 1} elapsed intervals by the saved missed-time policy.` }, "occurrence");
                if (row.repeatMs === undefined) { store.put({ ...row, state: "finished", nextIndex: row.nextIndex + 1, updatedAt: this.iso() }, "schedule"); this.anchor(this.timers, this.key(brainId, row.id)); }
                else {
                  const next = row.kind === "timer" ? this.source.wallMs() + row.repeatMs - lateness % row.repeatMs : due + (missed + 1) * row.repeatMs;
                  store.put({ ...row, nextIndex: row.nextIndex + missed + 1, dueAt: safeDeadline(Math.floor(next)), updatedAt: this.iso() }, "schedule");
                  if (row.kind === "timer") this.anchor(this.timers, this.key(brainId, row.id), { mono: this.source.monotonicMs(), base: row.repeatMs - lateness % row.repeatMs });
                }
              }
            }));
          } finally { store.close(); }
          if (checkpointWatches) await this.rates(brainId, { limit: 1 });
          this.requestDelivery(brainId, false); this.requestDelivery(brainId, true);
          if (changed) this.emit(brainId);
        } catch (error) { this.options.onError?.(error instanceof Error ? error : new Error(String(error)), brainId); }
      }
      if (checkpointWatches) this.lastWatchCheckpoint = this.source.monotonicMs();
    } finally { this.ticking = false; }
  }
  /** Tests and shutdown owners can await host delivery without waiting for a future tick. */
  async settle(brainId: string): Promise<void> { await Promise.all([this.delivering.get(brainId), this.automationDelivering.get(brainId)]); }
  private requestDelivery(brainId: string, automation: boolean): void {
    const owners = automation ? this.automationDelivering : this.delivering;
    if (owners.has(brainId) || this.quitting) return;
    const delivery = this.deliverNext(brainId, automation).catch(error => this.options.onError?.(error, brainId)).finally(() => { if (owners.get(brainId) === delivery) owners.delete(brainId); });
    owners.set(brainId, delivery);
  }
  private async updateOccurrence(brainId: string, id: string, update: Partial<ScheduledOccurrence>): Promise<ScheduledOccurrence> {
    const store = await this.store(brainId); try { const old = store.get<ScheduledOccurrence>("occurrence", id); if (!old) throw new Error("Wakeup receipt is missing.");
      if (TERMINAL.includes(old.state)) return old;
      const unchanged = { ...old, ...update, updatedAt: old.updatedAt };
      if (JSON.stringify(unchanged) === JSON.stringify(old)) return old;
      const row = { ...old, ...update, updatedAt: this.iso() }; store.put(row, "occurrence"); this.emit(brainId); return row; } finally { store.close(); }
  }
  private async deliverNext(brainId: string, automationOnly = false): Promise<void> {
    const epoch = this.ownerEpoch.get(brainId) ?? 0;
    const current = (): boolean => (this.ownerEpoch.get(brainId) ?? 0) === epoch && this.brains.has(brainId);
    const store = await this.store(brainId); let occurrence: ScheduledOccurrence | undefined, schedule: ScheduleRecord | undefined;
    try { const selected = store.db.prepare("SELECT payload FROM records WHERE kind='occurrence' AND state IN ('pending','dispatching','running') AND (json_extract(payload,'$.delivery')='automation')=? ORDER BY id LIMIT 1").get(automationOnly ? 1 : 0);
      occurrence = selected ? JSON.parse(String(selected.payload)) as ScheduledOccurrence : undefined;
      if (occurrence) schedule = store.get<ScheduleRecord>("schedule", occurrence.scheduleId); } finally { store.close(); }
    if (!occurrence || !schedule || TERMINAL.includes(occurrence.state)) return;
    if (this.quitting || !this.options.admissionOpen()) { await this.updateOccurrence(brainId, occurrence.id, { waitingFor: "shutdown", detail: "Pending until application exit/checkpointing finishes." }); return; }
    const authority = await this.options.authority(brainId);
    if (!current()) return;
    if (!authority.enabled || authority.revision !== occurrence.authorizationRevision) {
      await this.updateOccurrence(brainId, occurrence.id, { waitingFor: "permission", detail: "The schedule's saved permission changed. Explicitly resume it after reviewing current access." }); return;
    }
    if (schedule.pendingPaused) { await this.updateOccurrence(brainId, occurrence.id, { waitingFor: "review", detail: "Schedule paused; pending task remains saved." }); return; }
    if (schedule.state === "cancelled") { await this.updateOccurrence(brainId, occurrence.id, { state: "cancelled", waitingFor: undefined }); return; }
    if (occurrence.delivery === "automation") {
      if (!this.automationDelegate) { await this.updateOccurrence(brainId, occurrence.id, { waitingFor: "review", detail: "The independent action-file executor is unavailable. No native task or host effect was dispatched." }); return; }
      // If the handoff ACK is lost, exactly this occurrence is handed to an
      // idempotent run owner again. It recovers the same run, never a new effect.
      occurrence = await this.updateOccurrence(brainId, occurrence.id, { state: "dispatching", dispatchClaimedAt: occurrence.dispatchClaimedAt ?? this.iso(), waitingFor: undefined });
      if (!current() || this.quitting || !this.options.admissionOpen()) return;
      try {
        const receipt = await this.automationDelegate(occurrence);
        if (receipt.accepted !== true) throw new Error("No durable action-run handoff receipt was returned.");
        clockId(receipt.runId);
        if (current()) await this.updateOccurrence(brainId, occurrence.id, { state: "delegated", delegatedRunId: receipt.runId, waitingFor: undefined, detail: "Saved action run owns this occurrence. Inspect its individual effect receipts; this is not a native chat completion." });
      } catch (error) { if (current()) await this.updateOccurrence(brainId, occurrence.id, { state: "review-required", waitingFor: "review", detail: `Action-file handoff requires attention: ${error instanceof Error ? error.message : String(error)}. The same occurrence may be reviewed; no effect is blindly replayed.` }); }
      return;
    }
    if (!occurrence.notificationRequestedAt) {
      // Notification dispatch is at-most-once. A crash between this durable claim and the OS API may lose a banner, never duplicate a host effect.
      occurrence = await this.updateOccurrence(brainId, occurrence.id, { notificationRequestedAt: this.iso() });
      try { await this.options.notify(occurrence); }
      catch (error) {
        if (occurrence.delivery === "notify") {
          await this.updateOccurrence(brainId, occurrence.id, { state: "review-required", waitingFor: "review", detail: `Host notification was not dispatched: ${error instanceof Error ? error.message : String(error)}` }); return;
        }
      }
    }
    if (occurrence.delivery === "notify") { await this.updateOccurrence(brainId, occurrence.id, { state: "notified", waitingFor: undefined, detail: "Notification requested from the host; OS banner delivery is not guaranteed." }); return; }
    if (occurrence.state === "pending" && !occurrence.dispatchClaimedAt && !occurrence.loadWhenDue && !await this.options.loaded(brainId)) {
      await this.updateOccurrence(brainId, occurrence.id, { waitingFor: "load", detail: "Pending · Load this mind to run its saved task. No automatic Load was performed." }); return;
    }
    let status: ClockWakeStatus;
    try { status = await this.options.turnStatus(brainId, occurrence.turnId, occurrence.input); }
    catch (error) { if (current()) await this.updateOccurrence(brainId, occurrence.id, { state: "review-required", waitingFor: "review", detail: `Cannot establish the original native turn ownership: ${error instanceof Error ? error.message : String(error)}` }); return; }
    if (!current()) return;
    if (status.state === "committed") { await this.updateOccurrence(brainId, occurrence.id, { state: "committed", waitingFor: undefined, detail: status.detail ?? "Existing committed task receipt recovered; no resend." }); return; }
    if (status.state === "accepted-unfinished") { await this.updateOccurrence(brainId, occurrence.id, { state: "review-required", waitingFor: "review", detail: "The input was accepted before interruption, but its exact continuation is unavailable. It will not be replayed automatically." }); return; }
    if (status.state === "absent" && occurrence.state !== "pending") { await this.updateOccurrence(brainId, occurrence.id, { state: "review-required", waitingFor: "review", detail: "A prior dispatch was claimed but has no authoritative retained/completed turn. Review the original effects; this task will not be resent automatically." }); return; }
    if (status.state === "active") { await this.updateOccurrence(brainId, occurrence.id, { state: "running", waitingFor: "busy", detail: "The original scheduled turn is already attached; no second send." }); return; }
    if (!await this.options.loaded(brainId) && occurrence.loadWhenDue && this.options.load) {
      const loadAuthority = await this.options.authority(brainId), loadStore = await this.store(brainId);
      let currentSchedule: ScheduleRecord | undefined, currentOccurrence: ScheduledOccurrence | undefined;
      try { currentSchedule = loadStore.get<ScheduleRecord>("schedule", occurrence.scheduleId); currentOccurrence = loadStore.get<ScheduledOccurrence>("occurrence", occurrence.id); } finally { loadStore.close(); }
      if (!current()) return;
      if (!loadAuthority.enabled || loadAuthority.revision !== occurrence.authorizationRevision) { await this.updateOccurrence(brainId, occurrence.id, { waitingFor: "permission", detail: "Access changed before scheduled Load; review the saved task." }); return; }
      if (!currentSchedule || currentSchedule.state === "cancelled" || currentSchedule.pendingPaused || !currentOccurrence || TERMINAL.includes(currentOccurrence.state)) return;
      if ((this.nextLoadAttempt.get(brainId) ?? 0) > this.source.monotonicMs()) return;
      this.nextLoadAttempt.set(brainId, this.source.monotonicMs() + 10_000);
      try { await this.options.load(brainId); }
      catch (error) { if (current()) await this.updateOccurrence(brainId, occurrence.id, { waitingFor: "load", detail: `Explicit scheduled Load is waiting for normal resource admission: ${error instanceof Error ? error.message : String(error)}` }); return; }
      if (!current()) return;
    }
    if (!await this.options.loaded(brainId)) { await this.updateOccurrence(brainId, occurrence.id, { waitingFor: "load", detail: "Pending · Load this mind to run its saved task. No automatic Load was performed." }); return; }
    if (this.options.canRun && !this.options.canRun(brainId, occurrence.turnId)) { await this.updateOccurrence(brainId, occurrence.id, { waitingFor: "retained-turn", detail: "Continue the mind's other saved native turn before this queued task." }); return; }
    if (this.options.busy(brainId)) { await this.updateOccurrence(brainId, occurrence.id, { waitingFor: "busy", detail: "Queued behind current work; no steering or reset." }); return; }
    if (status.state === "retained" && !status.continuation) { await this.updateOccurrence(brainId, occurrence.id, { state: "review-required", waitingFor: "retained-turn", detail: "Saved native turn exists but its authenticated presentation is unavailable." }); return; }
    const finalAuthority = await this.options.authority(brainId), finalStore = await this.store(brainId);
    let finalSchedule: ScheduleRecord | undefined;
    try { finalSchedule = finalStore.get<ScheduleRecord>("schedule", occurrence.scheduleId); } finally { finalStore.close(); }
    if (!current()) return;
    if (!finalAuthority.enabled || finalAuthority.revision !== occurrence.authorizationRevision) { await this.updateOccurrence(brainId, occurrence.id, { waitingFor: "permission", detail: "Access changed before dispatch. Review and resume the saved task; nothing was sent." }); return; }
    if (!finalSchedule || finalSchedule.state === "cancelled" || finalSchedule.pendingPaused) return;
    if (this.quitting || !this.options.admissionOpen()) { await this.updateOccurrence(brainId, occurrence.id, { waitingFor: "shutdown", detail: "Pending at the application's final checkpoint boundary; no task was dispatched." }); return; }
    const claimed = await this.updateOccurrence(brainId, occurrence.id, { state: "dispatching", dispatchClaimedAt: occurrence.dispatchClaimedAt ?? this.iso(), waitingFor: undefined, detail: status.state === "retained" ? "Reattaching the exact saved turn." : "Submitting the saved task through the normal controller." });
    if (claimed.state !== "dispatching") return;
    if (!current()) return;
    try {
      const result = await this.options.wake(occurrence, status.continuation);
      if (!current()) return;
      await this.updateOccurrence(brainId, occurrence.id, { state: result.committed ? "committed" : "review-required", waitingFor: result.committed ? undefined : "review", detail: result.detail ?? (result.committed ? "Scheduled turn committed." : "No committed completion receipt; review before continuing.") });
    } catch (error) {
      if (!current()) return;
      // A rejection might follow a real external effect. Reconcile actual receipts on a later tick; never turn a transport error into a blind retry.
      await this.updateOccurrence(brainId, occurrence.id, { state: "running", detail: `Awaiting authoritative continuation/receipt: ${error instanceof Error ? error.message : String(error)}` });
    }
  }
  async flushRateHistory(): Promise<void> { for (const brainId of this.brains) await this.rates(brainId, { limit: 1 }); }
}
