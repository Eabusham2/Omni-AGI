import { useCallback, useEffect, useRef, useState } from "react";
import type { ClockPage, ClockWorkspaceSnapshot, NativeInferenceRate, ScheduleRecord, ScheduledOccurrence, StopwatchLap, StopwatchRecord } from "../../shared/clock";
import type { ToolInvocation, ToolExecutionResult } from "../../shared/types";
import "./clockWorkspace.css";

export interface ClockWorkspacePanelProps { brainId: string; onToast?(message: string): void; onActiveMode?(): void; onEditAutomationSchedule?(schedule: ScheduleRecord): void }
function clockStudio() { if (!window.omni?.clock) throw new Error("The desktop clock connection is unavailable."); return window.omni; }
export function clockDuration(ms: number): string {
  if (!Number.isFinite(ms)) return "—";
  const tenths = Math.max(0, Math.round(ms / 100));
  const hours = Math.floor(tenths / 36000), minutes = Math.floor(tenths % 36000 / 600);
  return `${hours ? `${hours}:` : ""}${hours ? String(minutes).padStart(2, "0") : minutes}:${(tenths % 600 / 10).toFixed(1).padStart(4, "0")}`;
}
function when(iso: string | undefined, zone?: string): string {
  if (!iso) return "On a matching event";
  try { return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "medium", timeZone: zone }).format(new Date(iso)); }
  catch { return iso; }
}
async function visibleClockPage<T>(read: (query?: { after?: string; limit?: number }) => Promise<ClockPage<T>>, visible: number): Promise<ClockPage<T>> {
  let page = await read({ limit: Math.min(200, Math.max(50, visible)) });
  while (page.nextAfter && page.entries.length < visible) { const next = await read({ after: page.nextAfter, limit: Math.min(200, visible - page.entries.length) }); page = { ...next, entries: [...page.entries, ...next.entries] }; }
  return page;
}

/** Available unloaded. Mutations still use the normal ToolExecutor policy/receipt path. */
export function ClockWorkspacePanel({ brainId, onToast, onActiveMode, onEditAutomationSchedule }: ClockWorkspacePanelProps) {
  const [snapshot, setSnapshot] = useState<ClockWorkspaceSnapshot>();
  const [schedules, setSchedules] = useState<ClockPage<ScheduleRecord>>({ entries: [], total: 0 });
  const [watches, setWatches] = useState<ClockPage<StopwatchRecord>>({ entries: [], total: 0 });
  const [occurrences, setOccurrences] = useState<ClockPage<ScheduledOccurrence>>({ entries: [], total: 0 });
  const [rates, setRates] = useState<ClockPage<NativeInferenceRate>>({ entries: [], total: 0 });
  const [laps, setLaps] = useState<Record<string, { entries: StopwatchLap[]; nextAfter?: number }>>({});
  const [sampledAt, setSampledAt] = useState(0), [clockSampledAt, setClockSampledAt] = useState(0), [clockTick, setClockTick] = useState(0);
  const [label, setLabel] = useState("Reminder"), [kind, setKind] = useState<"timer" | "at" | "event">("timer");
  const [delay, setDelay] = useState("10"), [at, setAt] = useState(""), [originalAtDraft, setOriginalAtDraft] = useState(""), [eventKey, setEventKey] = useState("source.changed");
  const [delivery, setDelivery] = useState<"notify" | "wake">("notify"), [task, setTask] = useState(""), [loadWhenDue, setLoadWhenDue] = useState(false);
  const [repeat, setRepeat] = useState(false), [repeatMinutes, setRepeatMinutes] = useState("60"), [missedPolicy, setMissedPolicy] = useState<"coalesce" | "skip">("coalesce");
  const [watchLabel, setWatchLabel] = useState("Task stopwatch"), [editing, setEditing] = useState<ScheduleRecord>();
  const [error, setError] = useState(""), [busy, setBusy] = useState(false);
  const [errorReveal, setErrorReveal] = useState(0), [actionStatus, setActionStatus] = useState("");
  const [approval, setApproval] = useState<{ invocation: ToolInvocation; result: ToolExecutionResult }>();
  const version = useRef(0);
  const approvalRegion = useRef<HTMLElement>(null), errorRegion = useRef<HTMLDivElement>(null);
  const owner = useRef<object | null>(null);
  const visible = useRef({ schedules: 50, watches: 50, occurrences: 50, rates: 50 });
  const userError = (failure: unknown): void => { setError(failure instanceof Error ? failure.message : String(failure)); setErrorReveal(value => value + 1); };
  const reveal = (element: HTMLElement | null): void => {
    if (!element) return; element.focus({ preventScroll: true });
    element.scrollIntoView({ block: "start", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  };
  useEffect(() => { if (approval) reveal(approvalRegion.current); }, [approval]);
  useEffect(() => { if (errorReveal) reveal(errorRegion.current); }, [errorReveal]);
  const refresh = useCallback(async () => {
    const current = ++version.current;
    try {
      const values = await Promise.all([clockStudio().clock.snapshot(brainId), visibleClockPage(query => clockStudio().clock.schedules(brainId, query), visible.current.schedules),
        visibleClockPage(query => clockStudio().clock.stopwatches(brainId, query), visible.current.watches), visibleClockPage(query => clockStudio().clock.occurrences(brainId, query), visible.current.occurrences),
        visibleClockPage(query => clockStudio().clock.rates(brainId, query), visible.current.rates)]);
      if (current !== version.current) return;
      setSnapshot(values[0]); setSchedules(values[1]); setWatches(values[2]); setOccurrences(values[3]); setRates(values[4]); setSampledAt(performance.now()); setClockSampledAt(performance.now());
    } catch (failure) { if (current === version.current) setError(failure instanceof Error ? failure.message : String(failure)); }
  }, [brainId]);
  useEffect(() => {
    const currentOwner = {}; owner.current = currentOwner;
    visible.current = { schedules: 50, watches: 50, occurrences: 50, rates: 50 };
    setSnapshot(undefined); setSchedules({ entries: [], total: 0 }); setWatches({ entries: [], total: 0 }); setOccurrences({ entries: [], total: 0 }); setRates({ entries: [], total: 0 }); setEditing(undefined); setApproval(undefined); setLaps({}); setError(""); setActionStatus(""); setBusy(false); void refresh();
    if (!window.omni?.clock) { setError("The desktop clock connection is unavailable."); return () => { version.current++; owner.current = null; }; }
    const unsubscribe = window.omni.clock.onEvent(event => { if (event.brainId === brainId) void refresh(); });
    const timer = setInterval(() => setClockTick(performance.now()), 100);
    const clockTimer = setInterval(() => { void clockStudio().clock.now().then(clock => {
      if (owner.current !== currentOwner) return; setSnapshot(current => current ? { ...current, clock } : current); setClockSampledAt(performance.now());
    }).catch(() => undefined); }, 1000);
    return () => { version.current++; if (owner.current === currentOwner) owner.current = null; unsubscribe(); clearInterval(timer); clearInterval(clockTimer); };
  }, [brainId, refresh]);
  useEffect(() => { visible.current.schedules = Math.max(50, schedules.entries.length); }, [schedules]);
  useEffect(() => { visible.current.watches = Math.max(50, watches.entries.length); }, [watches]);
  useEffect(() => { visible.current.occurrences = Math.max(50, occurrences.entries.length); }, [occurrences]);
  useEffect(() => { visible.current.rates = Math.max(50, rates.entries.length); }, [rates]);
  const run = async (toolId: string, action: string, args: Record<string, unknown>, approved?: typeof approval): Promise<void> => {
    if (busy) return; setBusy(true); setError(""); setActionStatus("Saving clock action…");
    const currentOwner = owner.current;
    const invocation = approved ? { ...approved.invocation, approvalToken: approved.result.approvalToken } :
      { brainId, toolId, action, arguments: { ...args, ...(["list", "status", "laps", "performance"].includes(action) || toolId === "system.time" ? {} : { requestKey: crypto.randomUUID() }) } };
    try {
      const result = await clockStudio().tool.execute(invocation);
      if (owner.current !== currentOwner) return;
      if (result.state === "approval-required") { setApproval({ invocation, result }); setActionStatus("Waiting for your approval."); return; }
      setApproval(undefined);
      if (result.state !== "complete") throw new Error(result.error ?? `The action is ${result.state}.`);
      if (action === "create" || action === "edit") setEditing(undefined);
      setActionStatus("Saved.");
      onToast?.(action === "cancel" ? "Future and pending wakeups cancelled; saved receipts retained." : "Clock action saved."); await refresh();
    } catch (failure) { if (owner.current === currentOwner) { userError(failure); setActionStatus("Action not completed."); } }
    finally { if (owner.current === currentOwner) setBusy(false); }
  };
  async function more<T>(read: () => Promise<ClockPage<T>>, update: (action: (current: ClockPage<T>) => ClockPage<T>) => void): Promise<void> {
    const currentOwner = owner.current;
    try { const page = await read(); if (owner.current !== currentOwner) return;
      update(current => { const key = (row: T): unknown => { const entry = row as { id?: string; recordId?: string; turnId?: string }; return entry.id ?? entry.recordId ?? entry.turnId; };
        const ids = new Set(current.entries.map(key)); return { ...page, entries: [...current.entries, ...page.entries.filter(row => !ids.has(key(row)))] }; });
    } catch (failure) { if (owner.current === currentOwner) userError(failure); }
  }
  async function loadLaps(id: string, after?: number): Promise<void> {
    const currentOwner = owner.current;
    try { const page = await clockStudio().clock.laps(brainId, id, after); if (owner.current !== currentOwner) return;
      setLaps(current => ({ ...current, [id]: { ...page, entries: after === undefined ? page.entries : [...(current[id]?.entries ?? []), ...page.entries.filter(row => !(current[id]?.entries ?? []).some(old => old.number === row.number))] } }));
    } catch (failure) { if (owner.current === currentOwner) userError(failure); }
  }
  const submit = (): void => {
    const interval = Math.round(Number(repeatMinutes) * 60_000), delayMs = Math.round(Number(delay) * 60_000);
    if (!label.trim() || kind === "timer" && (!Number.isSafeInteger(delayMs) || delayMs < 1) || repeat && (!Number.isSafeInteger(interval) || interval < 1)) { userError("Enter a name and a positive duration."); return; }
    let instant: string | undefined;
    if (kind === "at") { const parsed = new Date(at); if (!at || !Number.isFinite(parsed.getTime())) { userError("Choose a calendar date and time."); return; } instant = editing && at === originalAtDraft ? editing.dueAt : parsed.toISOString(); }
    const args: Record<string, unknown> = { label, task, delivery, timeZone: editing?.timeZone ?? snapshot?.clock.timeZone, missedPolicy, loadWhenDue: delivery === "wake" && loadWhenDue };
    if (editing) { args.scheduleId = editing.id; args.expectedRevision = editing.revision; args.repeatMs = repeat ? interval : 0; }
    else { args.kind = kind; if (repeat && kind !== "event") args.repeatMs = interval; }
    if (kind === "timer") args.delayMs = delayMs;
    if (kind === "at") args.at = instant;
    if (kind === "event" && !editing) args.eventKey = eventKey;
    void run("studio.schedule", editing ? "edit" : "create", args);
  };
  const edit = (row: ScheduleRecord): void => {
    if (row.delivery === "automation") {
      onEditAutomationSchedule?.(row);
      return;
    }
    setEditing(row); setLabel(row.label); setKind(row.kind); setDelivery(row.delivery); setTask(row.task); setEventKey(row.eventKey ?? "");
    setLoadWhenDue(row.loadWhenDue);
    setRepeat(!!row.repeatMs); setRepeatMinutes(String((row.repeatMs ?? 3600000) / 60000)); setMissedPolicy(row.missedPolicy); setDelay(String((row.remainingMs ?? 600000) / 60000));
    if (row.dueAt) { const date = new Date(row.dueAt), draft = new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 23); setAt(draft); setOriginalAtDraft(draft); }
  };
  const delta = Math.max(0, clockTick - sampledAt);
  const wallDelta = Math.max(0, clockTick - clockSampledAt);
  return <section className="clock-workspace" aria-label="Timers and scheduled tasks">
    <header className="clock-header"><div><h2>Time & schedules</h2>{snapshot ? <p><time dateTime={snapshot.clock.utc}>{when(new Date(snapshot.clock.epochMilliseconds + wallDelta).toISOString(), snapshot.clock.timeZone)}</time> · {snapshot.clock.timeZone}</p> : <p role="status">Reading host time…</p>}</div>
      <button type="button" onClick={() => void refresh()} title="Refresh the actual saved clock state">Refresh</button></header>
    {error ? <div className="clock-error" role="alert" tabIndex={-1} ref={errorRegion}><strong>Action needs attention</strong><details open><summary>Details</summary><p>{error}</p></details></div> : null}
    {approval ? <section className="clock-approval" role="region" tabIndex={-1} ref={approvalRegion} aria-label="Clock action approval"><h3>Approve this clock action?</h3><p>{approval.invocation.toolId} · {approval.invocation.action}</p>
      <details><summary>Exact saved arguments</summary><pre>{JSON.stringify(approval.invocation.arguments, null, 2)}</pre></details>
      <div className="clock-buttons"><button type="button" disabled={busy} onClick={() => void run("", "", {}, approval)}>Approve action</button><button type="button" disabled={busy} onClick={() => { setApproval(undefined); setActionStatus("Declined; no action ran."); }}>Decline</button></div></section> : null}
    <div className="clock-columns">
      <section className="clock-card"><h3>{editing ? `Edit ${editing.label}` : "New reminder or task"}</h3>
        {onActiveMode ? <div className="clock-reminder-choice"><span>Wait for a schedule, or keep working?</span><button type="button" onClick={onActiveMode}>Use Active mode</button></div> : null}
        <label>Name<input value={label} onChange={event => setLabel(event.target.value)} /></label>
        <div className="clock-form-pair"><label>When<select value={kind} disabled={!!editing} onChange={event => setKind(event.target.value as typeof kind)}><option value="timer">After a duration</option><option value="at">Calendar date & time</option><option value="event">Tracker event</option></select></label>
          <label>Then<select value={delivery} onChange={event => setDelivery(event.target.value as typeof delivery)}><option value="notify">Notify me</option><option value="wake">Run the saved task</option></select></label></div>
        {kind === "timer" ? <label>Minutes from now<input type="number" min="0.000017" step="any" value={delay} onChange={event => setDelay(event.target.value)} /></label> :
          kind === "at" ? <label>Date & time on this device<input type="datetime-local" step="0.001" value={at} onChange={event => setAt(event.target.value)} /></label> : <label>Event key<input value={eventKey} disabled={!!editing} onChange={event => setEventKey(event.target.value)} /></label>}
        <label>{delivery === "wake" ? "Saved task" : "Reminder text (optional)"}<textarea rows={3} value={task} onChange={event => setTask(event.target.value)} placeholder={delivery === "wake" ? "The exact task to send when due" : "What should the reminder say?"} /></label>
        {delivery === "wake" ? <label className="clock-check"><input type="checkbox" checked={loadWhenDue} onChange={event => setLoadWhenDue(event.target.checked)} />Load this instance when due (subject to available resources)</label> : null}
        {kind !== "event" ? <><label className="clock-check"><input type="checkbox" checked={repeat} onChange={event => setRepeat(event.target.checked)} />Repeat at a fixed interval</label>
          {repeat ? <div className="clock-form-pair"><label>Every (minutes)<input type="number" min="0.000017" step="any" value={repeatMinutes} onChange={event => setRepeatMinutes(event.target.value)} /></label><label>Missed intervals<select value={missedPolicy} onChange={event => setMissedPolicy(event.target.value as typeof missedPolicy)}><option value="coalesce">Run once when back</option><option value="skip">Skip missed intervals</option></select></label></div> : null}</> : null}
        <div className="clock-buttons"><button type="button" disabled={busy || delivery === "wake" && !task.trim()} onClick={submit}>{editing ? "Save changes" : "Save schedule"}</button>{editing ? <button type="button" onClick={() => setEditing(undefined)}>Close edit</button> : null}</div>
        {actionStatus ? <p className="clock-status" role="status" aria-live="polite">{actionStatus}</p> : null}
        <details className="clock-help"><summary>Timing & access</summary><p>The app must stay running to issue wakeups. Sleeping or closed apps catch up by the saved missed-time policy. Unloaded minds wait for manual Load unless this schedule explicitly enables Load when due; that Load still requires resources and current access. Busy minds queue. Repeats use elapsed intervals, not recurring local-time calendar rules.</p></details>
      </section>
      <section className="clock-card"><h3>Stopwatches</h3><div className="clock-watch-create"><label>Name<input value={watchLabel} onChange={event => setWatchLabel(event.target.value)} /></label><button type="button" disabled={busy || !watchLabel.trim()} onClick={() => void run("studio.stopwatch", "start", { label: watchLabel })}>Start</button></div>
        {watches.entries.length ? watches.entries.map(row => <article className="clock-record" key={row.id}><div className="clock-record-title"><strong>{row.label}</strong><output>{clockDuration(row.elapsedMs + (row.state === "running" ? delta : 0))}</output></div>
          <small>{row.state === "running" ? "Running · monotonic clock" : `Paused${row.pauseReason ? ` · ${row.pauseReason.replaceAll("-", " ")}` : ""}`}</small>
          {row.excludedUnmeasuredGap ? <small>Unmeasured restart time was excluded.</small> : null}
          <div className="clock-buttons"><button type="button" disabled={busy} onClick={() => void run("studio.stopwatch", row.state === "running" ? "pause" : "resume", { stopwatchId: row.id })}>{row.state === "running" ? "Pause" : "Resume"}</button>
            <button type="button" disabled={busy || row.state !== "running"} onClick={() => void run("studio.stopwatch", "lap", { stopwatchId: row.id })}>Lap</button>
            <button type="button" disabled={busy} onClick={() => { if (window.confirm("Reset this stopwatch and its lap measurements?")) void run("studio.stopwatch", "reset", { stopwatchId: row.id }); }}>Reset</button></div>
          <details onToggle={event => { if (event.currentTarget.open) void loadLaps(row.id); }}><summary>{row.lapCount} {row.lapCount === 1 ? "lap" : "laps"}</summary>
            <ol>{laps[row.id]?.entries.map(lap => <li key={lap.number}>Lap {lap.number}: {clockDuration(lap.lapMs)} · total {clockDuration(lap.elapsedMs)}</li>)}</ol>
            {laps[row.id]?.nextAfter ? <button type="button" onClick={() => void loadLaps(row.id, laps[row.id]?.nextAfter)}>More laps</button> : null}</details>
        </article>) : <p>No stopwatch started.</p>}
        {watches.nextAfter ? <button type="button" onClick={() => void more(() => clockStudio().clock.stopwatches(brainId, { after: watches.nextAfter }), setWatches)}>More stopwatches</button> : null}
      </section>
    </div>
    <section className="clock-card"><h3>Saved schedules <span className="clock-count">{schedules.total}</span></h3>
      <div className="clock-schedule-list">{schedules.entries.map(row => <article className="clock-record" key={row.id}><div className="clock-record-title"><strong>{row.label}</strong><small>{row.state === "finished" ? "Trigger elapsed" : row.state}{row.pendingPaused ? " · pending work paused" : ""}</small></div>
        <p>{row.kind === "timer" && row.state === "active" && row.remainingMs !== undefined ? `${clockDuration(Math.max(0, row.remainingMs - delta))} remaining` : row.kind === "event" ? `Event: ${row.eventKey}` : when(row.dueAt, row.timeZone)}{row.repeatMs ? ` · every ${clockDuration(row.repeatMs)}` : ""}{row.loadWhenDue ? " · explicit Load when due" : ""}</p>
        <details><summary>{row.delivery === "automation" ? "Action file reference" : row.delivery === "wake" ? "Saved task" : "Reminder"}</summary>{row.delivery === "automation" ? <pre>{JSON.stringify(row.automation, null, 2)}</pre> : <p className="clock-task-text">{row.task || row.label}</p>}</details>
        <div className="clock-buttons"><button type="button" disabled={row.delivery === "automation" && !onEditAutomationSchedule} title={row.delivery === "automation" ? "Edit in the action-file schedule controls" : undefined} onClick={() => edit(row)}>Edit</button>{row.state !== "cancelled" ? <><button type="button" disabled={busy} onClick={() => void run("studio.schedule", row.state === "paused" || row.pendingPaused ? "resume" : "pause", { scheduleId: row.id })}>{row.state === "paused" || row.pendingPaused ? "Resume" : "Pause"}</button>
          <button type="button" disabled={busy} onClick={() => void run("studio.schedule", "resume", { scheduleId: row.id })} title="Review current access and reauthorize this saved schedule">Review access</button>
          <button type="button" disabled={busy} onClick={() => void run("studio.schedule", "cancel", { scheduleId: row.id })}>Cancel future wakeups</button></> : null}</div>
      </article>)}</div>{!schedules.entries.length ? <p>No schedules saved.</p> : null}
      {schedules.nextAfter ? <button type="button" onClick={() => void more(() => clockStudio().clock.schedules(brainId, { after: schedules.nextAfter }), setSchedules)}>More schedules</button> : null}
    </section>
    <section className="clock-card"><h3>Wakeup activity</h3>{occurrences.entries.map(row => <article className="clock-record" key={row.id}><div className="clock-record-title"><strong>{row.label}</strong><small>{row.state.replaceAll("-", " ")}{row.waitingFor ? ` · ${row.waitingFor.replaceAll("-", " ")}` : ""}</small></div>
      <p>{row.detail ?? when(row.dueAt)}{row.missedIntervals ? ` · ${row.missedIntervals} missed intervals coalesced/skipped` : ""}</p>
      <details><summary>Saved task & receipt</summary>{row.delivery === "automation" ? <pre>{JSON.stringify(row.automation, null, 2)}</pre> : <p className="clock-task-text">{row.input || "Notification only"}</p>}<dl><dt>Due</dt><dd>{when(row.dueAt)}</dd><dt>Input hash</dt><dd><code>{row.inputSha256}</code></dd><dt>{row.delivery === "automation" ? "Action correlation" : "Turn"}</dt><dd><code>{row.turnId}</code></dd></dl></details>
      {row.state === "review-required" && row.delivery === "wake" ? <button type="button" disabled={busy} onClick={() => void run("studio.schedule", "resume-occurrence", { occurrenceId: row.id })}>Review & continue saved task</button> : null}</article>)}
      {!occurrences.entries.length ? <p>No due tasks yet.</p> : null}{occurrences.nextAfter ? <button type="button" onClick={() => void more(() => clockStudio().clock.occurrences(brainId, { after: occurrences.nextAfter }), setOccurrences)}>More activity</button> : null}
    </section>
    <details className="clock-card"><summary>Measured output speed</summary>{rates.entries.length ? rates.entries.map(rate => <p key={rate.recordId ?? rate.turnId}>{rate.tokensPerSecond === null ? "Rate not measured" : `${rate.tokensPerSecond.toFixed(2)} tokens/s`} · {rate.outputTokens.toLocaleString()} tokens · {clockDuration(rate.outputElapsedMs)} · {when(rate.at)}</p>) : <p>No native output measurements recorded yet.</p>}
      {rates.nextAfter ? <button type="button" onClick={() => void more(() => clockStudio().clock.rates(brainId, { after: rates.nextAfter }), setRates)}>More measured turns</button> : null}</details>
  </section>;
}
