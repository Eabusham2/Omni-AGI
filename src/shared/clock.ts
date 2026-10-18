import type { ScheduledAutomationReference } from "./automation";
/** Operational clocks and scheduled task data; never neural personality instructions. */
export interface HostClockSnapshot {
  epochMilliseconds: number;
  utc: string;
  local: string;
  timeZone: string;
  utcOffsetMinutes: number;
  monotonicMilliseconds: number;
  sessionId: string;
}

export interface StopwatchLap { number: number; elapsedMs: number; lapMs: number; recordedAt: string }
export interface StopwatchRecord {
  id: string; brainId: string; label: string; revision: number;
  state: "running" | "paused"; elapsedMs: number; lapCount: number;
  recordedAt: string; sessionId: string;
  pauseReason?: "user" | "restart" | "copied-instance" | "shutdown";
  /** After a hard restart, time since the last durable checkpoint is excluded, not guessed. */
  excludedUnmeasuredGap?: boolean;
}

export type ScheduleKind = "timer" | "at" | "event";
export interface ScheduleRecord {
  id: string; brainId: string; label: string; revision: number;
  kind: ScheduleKind; state: "active" | "paused" | "cancelled" | "finished";
  delivery: "notify" | "wake" | "automation"; task: string; timeZone: string;
  automation?: ScheduledAutomationReference;
  loadWhenDue: boolean;
  dueAt?: string; remainingMs?: number; repeatMs?: number; eventKey?: string;
  missedPolicy: "coalesce" | "skip"; nextIndex: number;
  /** Creating/editing/resuming the exact task authorizes only this saved scope. */
  authorizationRevision: string;
  createdAt: string; updatedAt: string;
  pauseReason?: "user" | "copied-instance";
  pendingPaused?: boolean;
  lineage?: Array<{ brainId: string; scheduleId: string }>;
}

export interface ScheduledOccurrence {
  id: string; brainId: string; scheduleId: string; index: number;
  turnId: string; input: string; inputSha256: string;
  delivery: "notify" | "wake" | "automation"; label: string;
  automation?: ScheduledAutomationReference;
  authorizationRevision: string; scheduleRevision: number; loadWhenDue: boolean;
  dueAt: string; createdAt: string; updatedAt: string; missedIntervals: number;
  state: "pending" | "dispatching" | "running" | "committed" | "notified" | "delegated" | "cancelled" | "review-required";
  delegatedRunId?: string;
  waitingFor?: "load" | "busy" | "permission" | "shutdown" | "retained-turn" | "review";
  notificationRequestedAt?: string;
  detail?: string; eventId?: string; eventData?: unknown;
  dispatchClaimedAt?: string;
  copiedUnstarted?: boolean;
  originBrainId?: string;
}

export interface ScheduleCreateRequest {
  label: string; kind: ScheduleKind; delivery: "notify" | "wake" | "automation";
  automation?: ScheduledAutomationReference;
  task?: string; delayMs?: number; at?: string; timeZone?: string;
  repeatMs?: number; eventKey?: string; missedPolicy?: "coalesce" | "skip";
  loadWhenDue?: boolean;
}
export interface ScheduleEditRequest {
  scheduleId: string; expectedRevision: number;
  label?: string; task?: string; delivery?: "notify" | "wake" | "automation";
  automation?: ScheduledAutomationReference;
  at?: string; delayMs?: number; timeZone?: string; repeatMs?: number;
  missedPolicy?: "coalesce" | "skip";
  loadWhenDue?: boolean;
}
export interface ClockPageQuery { after?: string; limit?: number }
export interface ClockPage<T> { entries: T[]; total: number; nextAfter?: string }
export interface ClockWorkspaceSnapshot {
  brainId: string; clock: HostClockSnapshot; schedulerRunning: boolean;
  counts: { schedules: number; stopwatches: number; pending: number; reviewRequired: number };
  /** The application must be running; this is not an OS boot/wake guarantee. */
  processResidentRequired: true;
}
export interface ClockWorkspaceEvent { brainId: string; type: "changed"; at: string }
export interface NativeInferenceRate {
  turnId: string; at: string; outputTokens: number; outputElapsedMs: number; tokensPerSecond: number | null;
  source: "native-output-stream"; scope: "one-turn-cumulative";
  recordId?: string;
}
export interface ClockWorkspaceApi {
  now(timeZone?: string): Promise<HostClockSnapshot>;
  snapshot(brainId: string): Promise<ClockWorkspaceSnapshot>;
  schedules(brainId: string, query?: ClockPageQuery): Promise<ClockPage<ScheduleRecord>>;
  stopwatches(brainId: string, query?: ClockPageQuery): Promise<ClockPage<StopwatchRecord>>;
  occurrences(brainId: string, query?: ClockPageQuery): Promise<ClockPage<ScheduledOccurrence>>;
  laps(brainId: string, stopwatchId: string, after?: number): Promise<{ entries: StopwatchLap[]; nextAfter?: number }>;
  rates(brainId: string, query?: ClockPageQuery): Promise<ClockPage<NativeInferenceRate>>;
  onEvent(listener: (event: ClockWorkspaceEvent) => void): () => void;
}
