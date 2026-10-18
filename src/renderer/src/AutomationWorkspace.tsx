import { useCallback, useEffect, useRef, useState } from "react";
import type { AutomationDefinition, AutomationPage, AutomationRun, AutomationStepReceipt, AutomationWorkspaceSnapshot } from "../../shared/automation";
import type { ScheduleRecord } from "../../shared/clock";
import type { ToolExecutionResult, ToolInvocation } from "../../shared/types";
import { actionFileInputs, actionScheduleDraft, actionScheduleEditRequest, actionScheduleRequest, NEW_ACTION_FILE, pendingActionStep } from "./automationPresentation";
import "./automationWorkspace.css";
import { ActionCanvasEditor } from "./ActionCanvasEditor";
import { actionCanvasProgram } from "./actionCanvas";

interface PendingAction { invocation: ToolInvocation; result: ToolExecutionResult; step?: AutomationStepReceipt }
const emptyPage = <T,>(): AutomationPage<T> => ({ entries: [], total: 0 });
function desktop() { if (!window.omni?.automation) throw new Error("The action-file connection is unavailable."); return window.omni; }
function failureText(value: unknown): string { return value instanceof Error ? value.message : String(value); }
const terminalRuns = new Set(["stopped", "complete"]);

/** Read-only inspection is native-independent; every mutation uses ToolExecutor. */
export function AutomationWorkspace({ brainId, scheduleToEdit, onScheduleEditFinished, onToast }: {
  brainId: string; scheduleToEdit?: ScheduleRecord; onScheduleEditFinished?(): void; onToast?(message: string): void;
}) {
  const [snapshot, setSnapshot] = useState<AutomationWorkspaceSnapshot>();
  const [definitions, setDefinitions] = useState<AutomationPage<AutomationDefinition>>(emptyPage);
  const [runs, setRuns] = useState<AutomationPage<AutomationRun>>(emptyPage);
  const [selectedId, setSelectedId] = useState("");
  const [editor, setEditor] = useState(() => JSON.stringify(NEW_ACTION_FILE, null, 2));
  const [editorView, setEditorView] = useState<"canvas" | "json">("canvas");
  const [canvasValid, setCanvasValid] = useState(true);
  const [canvasSession, setCanvasSession] = useState(0);
  const [editBase, setEditBase] = useState<AutomationDefinition>();
  const [inputs, setInputs] = useState("{}"), [allowTriggers, setAllowTriggers] = useState(false);
  const [steps, setSteps] = useState<Record<string, AutomationPage<AutomationStepReceipt>>>({});
  const [currentSteps, setCurrentSteps] = useState<Record<string, AutomationStepReceipt | null>>({});
  const [schedule, setSchedule] = useState(() => actionScheduleDraft());
  const [approval, setApproval] = useState<PendingAction>();
  const [error, setError] = useState(""), [status, setStatus] = useState(""), [busy, setBusy] = useState(false);
  const [reviewRunId, setReviewRunId] = useState(""), [reviewReason, setReviewReason] = useState("");
  const [reviewAcknowledged, setReviewAcknowledged] = useState(false);
  const owner = useRef<object | null>(null), readVersion = useRef(0), running = useRef(false);
  const approvalRegion = useRef<HTMLElement>(null), errorRegion = useRef<HTMLParagraphElement>(null);
  const reviewRegion = useRef<HTMLFieldSetElement>(null);
  const runVisible = useRef(50), definitionVisible = useRef(50), stepRead = useRef(new Map<string, number>());
  const openRuns = useRef(new Set<string>()), stepReader = useRef<(id: string) => Promise<void>>(async () => {});
  const editorRead = useRef(0), selectedRef = useRef(selectedId);
  selectedRef.current = selectedId;
  const selected = definitions.entries.find(row => row.id === selectedId);
  const reveal = (element: HTMLElement | null) => { element?.focus({ preventScroll: true }); element?.scrollIntoView({ block: "start", behavior: "auto" }); };
  useEffect(() => { if (approval) reveal(approvalRegion.current); }, [approval]);
  useEffect(() => { if (error) reveal(errorRegion.current); }, [error]);
  useEffect(() => { if (reviewRunId) reveal(reviewRegion.current); }, [reviewRunId]);

  const refresh = useCallback(async () => {
    const current = ++readVersion.current;
    const readPage = async <T,>(read: (query: { after?: string; limit: number }) => Promise<AutomationPage<T>>, visible: number) => {
      let page = await read({ limit: Math.min(200, visible) });
      while (page.nextAfter && page.entries.length < visible) {
        const next = await read({ after: page.nextAfter, limit: Math.min(200, visible - page.entries.length) });
        page = { ...next, entries: [...page.entries, ...next.entries] };
      }
      return page;
    };
    try {
      const api = desktop().automation;
      const values = await Promise.all([api.snapshot(brainId), readPage(query => api.definitions(brainId, query), definitionVisible.current), readPage(query => api.runs(brainId, query), runVisible.current)]);
      if (current !== readVersion.current) return;
      setSnapshot(values[0]); setDefinitions(values[1]); setRuns(values[2]);
    } catch (failure) { if (current === readVersion.current) setError(failureText(failure)); }
  }, [brainId]);
  useEffect(() => {
    const current = {}; owner.current = current; running.current = false;
    setSnapshot(undefined); setDefinitions(emptyPage()); setRuns(emptyPage()); setSteps({}); setCurrentSteps({}); setSelectedId(""); setApproval(undefined); setEditBase(undefined); setError(""); setStatus("");
    setEditor(JSON.stringify(NEW_ACTION_FILE, null, 2)); setCanvasValid(true);
    definitionVisible.current = 50; runVisible.current = 50; stepRead.current.clear(); openRuns.current.clear(); editorRead.current++;
    void refresh();
    const unsubscribe = window.omni?.automation?.onEvent(event => { if (event.brainId === brainId) {
      void refresh(); openRuns.current.forEach(id => { void stepReader.current(id); });
    } });
    return () => { if (owner.current === current) owner.current = null; readVersion.current++; unsubscribe?.(); };
  }, [brainId, refresh]);
  useEffect(() => { setAllowTriggers(selected?.activation?.allowTriggers ?? false); setApproval(undefined); editorRead.current++; }, [selected?.id, selected?.revisionSha256, selected?.revision]);
  useEffect(() => {
    if (!scheduleToEdit?.automation || scheduleToEdit.brainId !== brainId) return;
    setSelectedId(scheduleToEdit.automation.definitionId); setInputs(JSON.stringify(scheduleToEdit.automation.inputs, null, 2));
    setSchedule(actionScheduleDraft(scheduleToEdit)); setApproval(undefined);
  }, [brainId, scheduleToEdit]);

  const inspectSteps = async (runId: string, after?: string) => {
    const current = owner.current, version = (stepRead.current.get(runId) ?? 0) + 1; stepRead.current.set(runId, version);
    try {
      const [page, currentStep] = await Promise.all([desktop().automation.steps(brainId, runId, { after, limit: 50 }), desktop().automation.currentStep(brainId, runId)]);
      if (owner.current !== current || stepRead.current.get(runId) !== version) return;
      setCurrentSteps(old => ({ ...old, [runId]: currentStep }));
      setSteps(old => ({ ...old, [runId]: after ? { ...page, entries: [...(old[runId]?.entries ?? []), ...page.entries] } : page }));
    } catch (failure) { if (owner.current === current) setError(failureText(failure)); }
  };
  stepReader.current = inspectSteps;
  const perform = async (toolId: string, action: string, args: Record<string, unknown>, approved?: PendingAction) => {
    if (running.current) return;
    const current = owner.current; running.current = true; setBusy(true); setError(""); setStatus("Submitting action…");
    const invocation: ToolInvocation = approved ? { ...approved.invocation, approvalToken: approved.result.approvalToken } :
      { brainId, toolId, action, arguments: { ...args, requestKey: crypto.randomUUID() } };
    try {
      const result = await desktop().tool.execute(invocation);
      if (owner.current !== current) return;
      if (result.state === "approval-required") {
        const run = typeof args.runId === "string" ? runs.entries.find(row => row.id === args.runId) : undefined;
        setApproval({ invocation, result, step: approved?.step ?? (run ? currentSteps[run.id] ?? pendingActionStep(run, steps[run.id]?.entries ?? []) : undefined) });
        setStatus("Waiting for your review."); return;
      }
      setApproval(undefined);
      if (result.state !== "complete") throw new Error(result.error ?? `Action is ${result.state}; inspect its retained receipt.`);
      setStatus("Request accepted. Run status below records the actual outcome.");
      if (toolId === "studio.schedule" && action === "edit") onScheduleEditFinished?.();
      if (toolId === "studio.automation" && action === "save") { setEditBase(undefined); onToast?.("Action file saved; no steps ran."); }
      if (action === "review") { setReviewRunId(""); setReviewReason(""); setReviewAcknowledged(false); }
      await refresh();
      if (typeof args.runId === "string") await inspectSteps(args.runId);
    } catch (failure) { if (owner.current === current) { setError(failureText(failure)); setStatus("Request not confirmed complete."); } }
    finally { if (owner.current === current) { running.current = false; setBusy(false); } }
  };
  const act = (action: string, args: Record<string, unknown>) => void perform("studio.automation", action, args);
  const loadEditor = async (definition: AutomationDefinition) => {
    const current = owner.current, request = ++editorRead.current; setError("");
    try { const value = await desktop().automation.program(brainId, definition.id, definition.revisionSha256);
      if (owner.current !== current || request !== editorRead.current || selectedRef.current !== definition.id) return;
      setEditBase(value.definition); setEditor(JSON.stringify(value.program, null, 2)); setApproval(undefined); setCanvasValid(true); setCanvasSession(session => session + 1);
    } catch (failure) { if (owner.current === current) setError(failureText(failure)); }
  };
  const scheduleSelected = () => {
    try {
      if (!selected) throw new Error("Select the current action file first.");
      const request = actionScheduleRequest(schedule, { definitionId: selected.id, revisionSha256: selected.revisionSha256, inputs: actionFileInputs(inputs) });
      void perform("studio.schedule", scheduleToEdit ? "edit" : "create", scheduleToEdit ? actionScheduleEditRequest(scheduleToEdit, request) : { ...request });
    } catch (failure) { setError(failureText(failure)); }
  };
  const pendingRun = runs.entries.find(row => row.id === reviewRunId);
  const canvasProgram = actionCanvasProgram(editor);
  const approvingNested = approval?.invocation.toolId === "studio.automation" && approval.invocation.action === "approve";
  const reviewingEffects = approval?.invocation.toolId === "studio.automation" && approval.invocation.action === "review";
  return <section className="automation-workspace" aria-labelledby="automation-title">
    <header><div><h2 id="automation-title">Action files</h2><p>Reusable steps run independently of the brain. Each effect uses current tool access.</p></div><button type="button" disabled={busy} onClick={() => void refresh()}>Refresh</button></header>
    {snapshot ? <small>{snapshot.counts.definitions} {snapshot.counts.definitions === 1 ? "file" : "files"} · {snapshot.counts.activeRuns} active runs · {snapshot.counts.reviewRequired} need review. The app must remain running.</small> : <p role="status">Reading saved action files…</p>}
    {error ? <p className="automation-error" role="alert" ref={errorRegion} tabIndex={-1}>{error}</p> : null}
    {approval ? <section className="automation-approval" ref={approvalRegion} tabIndex={-1} aria-labelledby="automation-approval-title"><h3 id="automation-approval-title">Review this exact request</h3>
      <p>{approvingNested ? approval.step?.invocation?.toolId ?? approval.invocation.toolId : approval.invocation.toolId} · {approvingNested ? approval.step?.invocation?.action ?? approval.invocation.action : approval.invocation.action}</p>
      {reviewingEffects ? <><strong>{approval.invocation.arguments.resolution === "skip" ? "Skip the uncertain step; do not run it again." : "Authorize a new attempt after explicit Resume; it may repeat prior effects."}</strong><p>{String(approval.invocation.arguments.reason ?? "")}</p><p>The previous attempt is evidence, not a new tool request.</p></> : <pre>{JSON.stringify(approvingNested ? approval.step?.invocation?.arguments ?? approval.invocation.arguments : approval.invocation.arguments, null, 2)}</pre>}
      <details><summary>Exact control request & retained receipt</summary><pre>{JSON.stringify({ invocation: approval.invocation, step: approval.step }, null, 2)}</pre></details>
      {approval.result.error ? <p>{approval.result.error}</p> : null}
      <div className="automation-buttons"><button type="button" disabled={busy || !approval.result.approvalToken} onClick={() => void perform(approval.invocation.toolId, approval.invocation.action, approval.invocation.arguments, approval)}>Approve this request</button>
        <button type="button" disabled={busy} onClick={() => { setApproval(undefined); setStatus("Declined; this request was not approved."); }}>Decline</button></div></section> : null}
    {status ? <p role="status" aria-live="polite">{status}</p> : null}
    <label>Saved action file<select value={selectedId} disabled={busy} onChange={event => { setSelectedId(event.target.value); setApproval(undefined); }}><option value="">Select a file</option>{definitions.entries.map(row => <option key={row.id} value={row.id}>{row.name}</option>)}</select></label>
    {definitions.nextAfter ? <button type="button" onClick={() => { definitionVisible.current += 50; void refresh(); }}>More action files</button> : null}
    {selected ? <section className="automation-card"><h3>{selected.name}</h3><details><summary>Exact file & activation</summary><code>{selected.revisionSha256}</code><p>{selected.activation?.enabled ? selected.activation.allowTriggers ? "Enabled for timer/event triggers." : "Enabled without timer/event triggers." : "Not enabled for triggers."}</p></details>
      <label className="automation-check"><input type="checkbox" checked={allowTriggers} onChange={event => { setAllowTriggers(event.target.checked); setApproval(undefined); }} />Allow timer and tracker triggers for this file</label>
      <div className="automation-buttons"><button type="button" disabled={busy} onClick={() => act("activate", { definitionId: selected.id, expectedRevision: selected.revision, revisionSha256: selected.revisionSha256, allowTriggers })}>Review activation</button>
        <button type="button" disabled={busy || !selected.activation?.enabled} onClick={() => act("disable", { definitionId: selected.id, expectedRevision: selected.revision, revisionSha256: selected.revisionSha256 })}>Disable future runs</button>
        <button type="button" disabled={busy} onClick={() => void loadEditor(selected)}>Edit file</button></div>
      <label>Inputs (JSON object)<textarea rows={3} spellCheck={false} value={inputs} onChange={event => { setInputs(event.target.value); setApproval(undefined); }} /></label>
      <button type="button" disabled={busy} onClick={() => { try { act("run", { definitionId: selected.id, revisionSha256: selected.revisionSha256, inputs: actionFileInputs(inputs) }); } catch (failure) { setError(failureText(failure)); } }}>Run once</button>
      <details open={!!scheduleToEdit}><summary>{scheduleToEdit ? "Edit action-file schedule" : "Schedule this exact file"}</summary><p>Enable timer/event triggers first. This runs saved steps; it does not load the brain or turn the file into a prompt. Edits apply to future occurrences, not already-started work.</p>
        <label>Name<input value={schedule.label} onChange={event => { setSchedule({ ...schedule, label: event.target.value }); setApproval(undefined); }} /></label>
        <label>When<select value={schedule.kind} disabled={!!scheduleToEdit} onChange={event => { setSchedule({ ...schedule, kind: event.target.value as typeof schedule.kind }); setApproval(undefined); }}><option value="timer">After a duration</option><option value="at">Calendar date & time</option><option value="event">Tracker event</option></select></label>
        {schedule.kind === "timer" ? <label>Minutes from now<input type="number" step="any" value={schedule.minutes} onChange={event => { setSchedule({ ...schedule, minutes: event.target.value }); setApproval(undefined); }} /></label> : schedule.kind === "at" ? <label>Date & time on this device<input type="datetime-local" step="0.001" value={schedule.at} onChange={event => { setSchedule({ ...schedule, at: event.target.value }); setApproval(undefined); }} /></label> : <label>Event key<input disabled={!!scheduleToEdit} value={schedule.eventKey} onChange={event => { setSchedule({ ...schedule, eventKey: event.target.value }); setApproval(undefined); }} /></label>}
        {schedule.kind !== "event" ? <><label className="automation-check"><input type="checkbox" checked={schedule.repeat} onChange={event => { setSchedule({ ...schedule, repeat: event.target.checked }); setApproval(undefined); }} />Repeat at a fixed interval</label>{schedule.repeat ? <label>Repeat every (minutes)<input type="number" step="any" value={schedule.repeatMinutes} onChange={event => { setSchedule({ ...schedule, repeatMinutes: event.target.value }); setApproval(undefined); }} /></label> : null}<label>Missed intervals<select value={schedule.missedPolicy} onChange={event => { setSchedule({ ...schedule, missedPolicy: event.target.value as typeof schedule.missedPolicy }); setApproval(undefined); }}><option value="coalesce">Run once when back</option><option value="skip">Skip missed intervals</option></select></label></> : null}
        <div className="automation-buttons"><button type="button" disabled={busy || !selected.activation?.enabled || !selected.activation.allowTriggers} onClick={scheduleSelected}>{scheduleToEdit ? "Review schedule changes" : "Review new schedule"}</button>{scheduleToEdit ? <button type="button" onClick={() => { onScheduleEditFinished?.(); setApproval(undefined); }}>Close schedule edit</button> : null}</div>
      </details></section> : null}
    <details className="automation-card" open={!!editBase}><summary>{editBase ? "Edit action file" : "Create action file"}</summary>
      <div className="automation-buttons" role="group" aria-label="Action file editor"><button type="button" aria-pressed={editorView === "canvas"} onClick={() => setEditorView("canvas")}>Canvas</button><button type="button" disabled={!canvasValid} aria-pressed={editorView === "json"} onClick={() => { setEditorView("json"); setCanvasValid(true); }}>JSON</button></div>
      {editorView === "canvas" && canvasProgram ? <ActionCanvasEditor key={`${brainId}:${editBase?.id ?? "new"}:${canvasSession}`} program={canvasProgram} disabled={busy} onValidityChange={setCanvasValid} onChange={value => { setEditor(JSON.stringify(value, null, 2)); setApproval(undefined); }} /> : <><label>Action-file JSON<textarea rows={10} spellCheck={false} value={editor} onChange={event => { setEditor(event.target.value); setApproval(undefined); setCanvasValid(true); }} /></label>{editorView === "canvas" ? <p role="alert">Correct this JSON before opening its canvas. Nothing was replaced or run.</p> : null}</>}
      <div className="automation-buttons"><button type="button" disabled={busy || !canvasValid} onClick={() => { try { act("save", { program: JSON.parse(editor), ...(editBase ? { definitionId: editBase.id, expectedRevision: editBase.revision } : {}) }); } catch (failure) { setError(failureText(failure)); } }}>Save file</button>
        {editBase ? <button type="button" onClick={() => { setEditBase(undefined); setEditor(JSON.stringify(NEW_ACTION_FILE, null, 2)); setApproval(undefined); setCanvasValid(true); }}>New file instead</button> : null}</div></details>
    <section className="automation-card"><h3>Run activity</h3>{runs.entries.map(row => <article key={row.id} className="automation-run"><div className="automation-run-title"><strong>{definitions.entries.find(file => file.id === row.definitionId)?.name ?? "Saved action file"}</strong><span>{row.state.replaceAll("-", " ")}</span></div><p>{row.detail ?? `Next step ${row.nextStep + 1}`}</p>
      <details onToggle={event => { if (event.currentTarget.open) { openRuns.current.add(row.id); void inspectSteps(row.id); } else openRuns.current.delete(row.id); }}><summary>Steps & actual receipts</summary>{steps[row.id]?.entries.map(step => <details key={`${step.index}:${step.attempt}`}><summary>{step.stepId} · attempt {step.attempt} · {step.state.replaceAll("-", " ")}</summary><pre>{JSON.stringify(step, null, 2)}</pre></details>)}
        {steps[row.id]?.nextAfter ? <button type="button" onClick={() => void inspectSteps(row.id, steps[row.id]?.nextAfter)}>More receipts</button> : null}
        {row.state === "approval-required" ? (() => { const step = currentSteps[row.id]; return <><details><summary>Current pending invocation</summary><pre>{step ? JSON.stringify(step, null, 2) : "Reading the current retained step…"}</pre></details><button type="button" disabled={busy || !step?.outcome?.approvalToken || step.index !== row.nextStep || step.outcome.state !== "approval-required"} onClick={() => act("approve", { runId: row.id, approvalToken: step!.outcome!.approvalToken })}>Review pending tool request</button></>; })() : null}
      </details><div className="automation-buttons">{row.state === "paused" ? <button type="button" disabled={busy} onClick={() => act("resume", { runId: row.id })}>Resume</button> : !terminalRuns.has(row.state) && !["failed", "review-required", "stopping"].includes(row.state) ? <button type="button" disabled={busy} onClick={() => act("pause", { runId: row.id })}>Pause</button> : null}
        {!terminalRuns.has(row.state) ? <button type="button" disabled={busy} onClick={() => act("stop", { runId: row.id })}>Stop remaining steps</button> : null}
        {["review-required", "failed"].includes(row.state) ? <button type="button" disabled={busy} onClick={() => { setReviewRunId(row.id); setReviewReason(""); setReviewAcknowledged(false); setApproval(undefined); void inspectSteps(row.id); }}>Review possible effects</button> : null}</div>
      {reviewRunId === row.id && pendingRun ? <fieldset ref={reviewRegion} tabIndex={-1}><legend>Possible prior effects</legend><p>The interrupted step may already have run. Inspect its receipts before choosing a new attempt.</p><label className="automation-check"><input type="checkbox" checked={reviewAcknowledged} onChange={event => { setReviewAcknowledged(event.target.checked); setApproval(undefined); }} />I have checked for possible prior effects.</label><label>Reason<textarea rows={2} value={reviewReason} onChange={event => { setReviewReason(event.target.value); setApproval(undefined); }} /></label><div className="automation-buttons">{(["skip", "retry-confirmed"] as const).map(resolution => <button type="button" key={resolution} disabled={busy || !reviewAcknowledged || !reviewReason.trim()} onClick={() => act("review", { runId: row.id, resolution, acknowledgePossibleEffects: true, reason: reviewReason.trim() })}>{resolution === "skip" ? "Review skipping this step" : "Review a new attempt"}</button>)}<button type="button" onClick={() => { setReviewRunId(""); setApproval(undefined); }}>Close review</button></div></fieldset> : null}
    </article>)}{!runs.entries.length ? <p>No action runs yet.</p> : null}{runs.nextAfter ? <button type="button" onClick={() => { runVisible.current += 50; void refresh(); }}>More runs</button> : null}</section>
  </section>;
}
