import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { LocalTeacherApi, LocalTeacherGeneration, LocalTeacherItem, LocalTeacherJob, LocalTeacherPlan, LocalTeacherRunnerKind, LocalTeacherSelection, LocalTeacherSelectionProgress, LocalTeacherStartRequest } from "../../shared/localTeacher";
import { mergeTeacherJobs, splitTeacherLessons, teacherGenerationDraft, teacherGenerationFromDraft, teacherGenerationLabel, teacherJobActive, type TeacherGenerationDraft } from "./localTeacherPresentation";
import { conciseUiMessage } from "./uiPresentation";
import "./localTeacher.css";

export function LocalTeacherPanel({ brainId, api }: { brainId: string; api?: LocalTeacherApi }) {
  return api ? <TeacherOwner key={brainId} brainId={brainId} api={api} /> : null;
}

function TeacherOwner({ brainId, api }: { brainId: string; api: LocalTeacherApi }) {
  const [teachers, setTeachers] = useState<LocalTeacherSelection[]>([]), [jobs, setJobs] = useState<LocalTeacherJob[]>([]);
  const [teacherId, setTeacherId] = useState(""), [runner, setRunner] = useState<LocalTeacherRunnerKind>("llama-server"), [license, setLicense] = useState("");
  const [prompts, setPrompts] = useState(""), [plan, setPlan] = useState<LocalTeacherPlan>(), [generation, setGeneration] = useState<TeacherGenerationDraft>();
  const [busy, setBusy] = useState(false), [selecting, setSelecting] = useState(false), [stoppingSelection, setStoppingSelection] = useState(false), [progress, setProgress] = useState<LocalTeacherSelectionProgress>();
  const [loading, setLoading] = useState(true), [error, setError] = useState(""), [notice, setNotice] = useState("");
  const [selectedJob, setSelectedJob] = useState<string>(), [confirmStop, setConfirmStop] = useState<string>(), [items, setItems] = useState<LocalTeacherItem[]>([]), [nextItem, setNextItem] = useState<number>();
  const [itemsBusy, setItemsBusy] = useState(false), [itemsError, setItemsError] = useState(""), [retryStart, setRetryStart] = useState(false);
  const scope = useRef<object | null>(null), command = useRef(false), selectionRequest = useRef<string | undefined>(undefined), startRequest = useRef<LocalTeacherStartRequest | undefined>(undefined);
  const selectionStop = useRef<string | undefined>(undefined);
  const itemVersion = useRef(0), selectedRef = useRef(selectedJob), selectionVersion = useRef(0), pageBusy = useRef(false);
  selectedRef.current = selectedJob;
  const chosen = teachers.find(value => value.id === teacherId), job = jobs.find(value => value.id === selectedJob);
  useLayoutEffect(() => { const owner = {}; scope.current = owner; return () => { if (scope.current === owner) scope.current = null; }; }, [api, brainId]);
  const receive = (values: LocalTeacherJob[]) => setJobs(previous => mergeTeacherJobs(previous, values, brainId));
  useEffect(() => {
    const owner = scope.current; let alive = true;
    const current = () => alive && scope.current === owner;
    const readTeachers = async () => { const version = ++selectionVersion.current;
      try { const values = await api.selections(brainId); if (current() && version === selectionVersion.current) { setTeachers(values.filter(value => value.brainId === brainId)); setTeacherId(previous => previous || values.find(value => value.brainId === brainId)?.id || ""); } }
      catch (reason) { if (current()) setError(conciseUiMessage(reason, "Local teachers could not load.")); }
    };
    const unsubscribe = api.onEvent(event => { if (current() && event.brainId === brainId) receive([event.job]); });
    const unsubscribeProgress = api.onSelectionProgress(event => {
      if (!current() || event.brainId !== brainId) return;
      if (event.requestId === selectionRequest.current) setProgress(event);
      if (event.phase === "complete") void readTeachers();
    });
    void Promise.all([readTeachers(), api.jobs(brainId).then(values => { if (current()) { receive(values); setSelectedJob(previous => previous ?? values.find(value => value.brainId === brainId)?.id); } })])
      .catch(reason => { if (current()) setError(conciseUiMessage(reason, "Saved teacher jobs could not load.")); }).finally(() => { if (current()) setLoading(false); });
    return () => { alive = false; unsubscribe(); unsubscribeProgress(); };
  }, [api, brainId]);

  const run = async (work: (owner: object) => Promise<void>) => {
    const owner = scope.current; if (!owner || command.current || selectionRequest.current) return;
    command.current = true; setBusy(true); setError(""); setNotice("");
    try { await work(owner); }
    catch (reason) { if (scope.current === owner) setError(conciseUiMessage(reason, "Teacher control was not acknowledged.")); }
    finally { if (scope.current === owner) { command.current = false; setBusy(false); } }
  };
  const select = async (reapprove = false) => {
    const owner = scope.current; if (!owner || command.current || selectionRequest.current || retryStart) return;
    const requestId = crypto.randomUUID(); selectionRequest.current = requestId; selectionStop.current = undefined; setSelecting(true); setStoppingSelection(false); setProgress(undefined); setError(""); setNotice("");
    try {
      const value = await api.select({ brainId, requestId, runner: reapprove && chosen ? chosen.runner : runner, ...(license.trim() ? { licenseDeclaration: license.trim() } : {}), ...(reapprove && chosen ? { teacherId: chosen.id } : {}) });
      if (scope.current !== owner || value && value.brainId !== brainId) return;
      if (value) { selectionVersion.current++; setTeachers(previous => [value, ...previous.filter(item => item.id !== value.id)]); setTeacherId(value.id); setPlan(undefined); setGeneration(undefined); setNotice("Local teacher selected. Prepare its resource fit before starting lessons."); }
    } catch (reason) {
      const settled = selectionStop.current === requestId ? await api.selectionProgress(brainId, requestId).catch(() => null) : null;
      if (scope.current === owner && settled?.phase !== "cancelled") setError(conciseUiMessage(reason, "Teacher selection could not finish."));
    }
    finally { if (scope.current === owner) { if (selectionStop.current === requestId) setNotice("File preparation ended; no teacher job started."); selectionRequest.current = undefined; setSelecting(false); setStoppingSelection(false); } }
  };
  const stopSelection = async () => {
    const owner = scope.current, requestId = selectionRequest.current; if (!owner || !requestId || stoppingSelection) return;
    selectionStop.current = requestId; setStoppingSelection(true);
    try { const stopped = await api.cancelSelection(brainId, requestId); if (scope.current === owner) setNotice(stopped ? selectionRequest.current === requestId ? "Stopping this file selection…" : "File preparation ended; no teacher job started." : "File selection already settled."); }
    catch (reason) { if (scope.current === owner) setError(conciseUiMessage(reason, "Selection Stop was not acknowledged.")); }
    finally { if (scope.current === owner) setStoppingSelection(false); }
  };
  const prepare = () => void run(async owner => {
    if (!chosen) return;
    const id = chosen.id; const value = await api.plan({ brainId, teacherId: id, prompts: splitTeacherLessons(prompts) });
    if (scope.current !== owner || value.brainId !== brainId || value.teacherId !== id) return;
    setPlan(value); setGeneration(value.recommendation ? teacherGenerationDraft(value.recommendation) : undefined);
  });
  const start = () => void run(async owner => {
    if (!startRequest.current) {
      if (!chosen?.currentHostAuthorized || !plan?.recommendation || plan.teacherId !== chosen.id || !generation) throw new Error("Select and approve a local teacher, then prepare its current resource fit.");
      const lessons = splitTeacherLessons(prompts); if (!lessons.length) throw new Error("Add at least one complete lesson prompt.");
      startRequest.current = { brainId, requestId: crypto.randomUUID(), teacherId: chosen.id, prompts: lessons, generation: teacherGenerationFromDraft(generation) };
    }
    setRetryStart(true); const value = await api.start(startRequest.current);
    if (scope.current !== owner || value.brainId !== brainId) return;
    receive([value]); setSelectedJob(value.id); startRequest.current = undefined; setRetryStart(false); setNotice("Teacher job saved. Generated examples and native learning receipts are tracked separately.");
  });
  const control = (value: LocalTeacherJob, operation: "pause" | "resume" | "stop") => void run(async owner => {
    const result = await api[operation](brainId, value.id); if (scope.current === owner) { receive([result]); setConfirmStop(undefined); }
  });
  const loadItems = async (afterIndex?: number) => {
    const owner = scope.current, id = selectedRef.current; if (!owner || !id || pageBusy.current) return;
    pageBusy.current = true; const version = ++itemVersion.current; setItemsBusy(true); setItemsError("");
    try { const values = await api.items(brainId, id, afterIndex, 20);
      if (scope.current === owner && selectedRef.current === id && itemVersion.current === version) {
        setItems(previous => afterIndex === undefined ? values : [...previous, ...values.filter(value => !previous.some(item => item.index === value.index))]);
        setNextItem(values.length === 20 ? values.at(-1)!.index : undefined);
      }
    } catch (reason) { if (scope.current === owner && selectedRef.current === id) setItemsError(conciseUiMessage(reason, "Lesson receipts could not load.")); }
    finally { pageBusy.current = false; if (scope.current === owner && selectedRef.current === id && itemVersion.current === version) setItemsBusy(false); }
  };
  useEffect(() => { itemVersion.current++; setItems([]); setNextItem(undefined); setItemsError(""); setItemsBusy(false); }, [selectedJob]);
  const disabled = busy || selecting || retryStart;
  return <details className="surface local-teacher"><summary>Learn from a local GGUF teacher</summary>
    <div className="local-teacher__body">
    <p>The teacher produces training examples for this OmniCortex. Its weights do not replace the brain.</p>
    <div className="local-teacher__row"><label>Installed runtime<select value={runner} disabled={disabled} onChange={event => setRunner(event.target.value as LocalTeacherRunnerKind)}>
      <option value="llama-server">llama-server · structured output</option><option value="llama-completion">llama-completion · console output</option><option value="llama-cli">llama-cli · console output</option>
    </select></label><button className="button" type="button" disabled={disabled} onClick={() => void select()}>Choose runtime & GGUF files</button></div>
    <details><summary>License declaration · optional</summary><label>Training-use declaration<input value={license} disabled={disabled} onChange={event => setLicense(event.target.value)} /></label></details>
    {selecting ? <div role="status"><span>{progress ? `${progress.phase.replaceAll("-", " ")} · ${progress.fileName ?? "selected file"} · ${progress.bytes.toLocaleString()} of ${progress.totalBytes.toLocaleString()} bytes` : "Waiting for file selection…"}</span><button className="button" type="button" disabled={stoppingSelection} onClick={() => void stopSelection()}>{stoppingSelection ? "Stopping selection…" : "Stop file preparation"}</button></div> : null}
    <label>Selected local teacher<select value={teacherId} disabled={disabled} onChange={event => { setTeacherId(event.target.value); setPlan(undefined); setGeneration(undefined); }}><option value="">Choose a saved teacher</option>{teachers.map(value => <option key={value.id} value={value.id}>{value.model.header.name || value.model.displayName} · {value.runner}{value.currentHostAuthorized ? "" : " · needs file approval"}</option>)}</select></label>
    {chosen ? <><p>{chosen.model.displayName} · {(chosen.model.bytes / 1048576).toLocaleString(undefined, { maximumFractionDigits: 1 })} MiB · {chosen.license.declaration || "No declared license"} · license not verified</p>
      {!chosen.currentHostAuthorized ? <button className="button" type="button" disabled={disabled} onClick={() => void select(true)}>Reapprove selected teacher files</button> : null}
      {chosen.runner !== "llama-server" ? <p>Console output keeps runtime framing. It is attributed as teacher-console data, not cleaned conversational output.</p> : null}
      {!chosen.runtime.rawOfflineSupported ? <p role="alert">This runtime does not expose the required verified raw/offline flags: {chosen.runtime.missingFlags.join(", ")}.</p> : null}</> : null}
    <label>Lesson prompts<textarea rows={4} value={prompts} disabled={disabled} placeholder="Separate complete lessons with a line containing ---" onChange={event => { setPrompts(event.target.value); setPlan(undefined); setGeneration(undefined); }} /></label>
    <div className="local-teacher__row"><button className="button" type="button" disabled={disabled || !chosen?.currentHostAuthorized} onClick={prepare}>Prepare teacher fit</button>
      <button className="button button--primary" type="button" disabled={busy || selecting || (!retryStart && (!plan?.recommendation || !generation || !chosen?.currentHostAuthorized || !splitTeacherLessons(prompts).length))} onClick={start}>{busy && retryStart ? "Awaiting job acknowledgement…" : retryStart ? "Retry same teacher job" : "Start local lessons"}</button></div>
    {retryStart && !busy ? <p role="status">Job acknowledgement is unconfirmed. Retry uses the same request; it does not repeat saved lessons.</p> : null}
    {plan ? <div className="local-teacher__fit" role="status"><strong>{plan.fit === "estimated" ? "Estimated fit" : plan.fit === "blocked" ? "Does not fit current allocation" : "Fit unknown"}</strong>
      {plan.reservationBytes !== null ? <span>Recommended reservation estimate · {(plan.reservationBytes / 1048576).toLocaleString(undefined, { maximumFractionDigits: 1 })} MiB</span> : null}
      {plan.blockers.length ? <ul>{plan.blockers.map(value => <li key={value}>{value}</li>)}</ul> : null}
      <details><summary>Resource estimate and generation settings</summary><p>The estimate uses declared GGUF geometry and current app-family memory. Custom settings are rechecked when Start runs.</p>
        {generation ? <div className="local-teacher__fields">{(Object.keys(generation) as Array<keyof LocalTeacherGeneration>).map(key => <label key={key}>{teacherGenerationLabel(key)}<input type="number" value={generation[key]} disabled={disabled} step={key === "temperature" ? "any" : "1"} onChange={event => setGeneration(previous => previous ? { ...previous, [key]: event.target.value } : previous)} /></label>)}</div> : null}
        <ul>{plan.warnings.map(value => <li key={value}>{value}</li>)}</ul></details>
    </div> : null}
    {notice ? <p role="status">{notice}</p> : null}{error ? <p role="alert">{error}</p> : null}{loading ? <p role="status">Loading saved local teachers…</p> : null}
    <details><summary>Local teacher jobs ({jobs.length.toLocaleString()})</summary><ul className="local-teacher__jobs">{jobs.map(value => <li key={value.id}>
      <button className="button" type="button" aria-pressed={selectedJob === value.id} onClick={() => { setSelectedJob(value.id); setConfirmStop(undefined); }}><span>{new Date(value.createdAt).toLocaleString()}</span><small>{value.state.replaceAll("-", " ")} · {value.generatedCount.toLocaleString()}/{value.promptCount.toLocaleString()} generated · {value.learnedCount.toLocaleString()} learned-source receipts</small></button>
      {teacherJobActive(value) ? <><button className="button" type="button" disabled={busy || selecting} onClick={() => control(value, "pause")}>Pause lessons</button><button className="button" type="button" disabled={busy || selecting} onClick={() => setConfirmStop(value.id)}>Stop lessons</button></> : value.state === "dormant" ? <small>Copied history stays paused. Start a new approved job to use a teacher on this device.</small> : value.state !== "complete" ? <button className="button" type="button" disabled={busy || selecting} onClick={() => control(value, "resume")}>Resume lessons</button> : null}
    </li>)}</ul>
      {confirmStop ? <div role="group" aria-label="Confirm stopping local teacher"><p>Stop this teacher job? Generated data, learned state and cursors stay saved.</p><div className="local-teacher__row"><button className="button" type="button" disabled={busy || selecting} onClick={() => { const value = jobs.find(item => item.id === confirmStop); if (value) control(value, "stop"); }}>Stop this job</button><button className="button" type="button" disabled={busy} onClick={() => setConfirmStop(undefined)}>Keep running</button></div></div> : null}
      {job ? <section aria-label="Local teacher lesson receipts"><p>{job.phase.replaceAll("-", " ")}</p>{job.error ? <p role="alert">{conciseUiMessage(job.error)}</p> : null}
        <button className="button" type="button" disabled={itemsBusy} onClick={() => void loadItems()}>Load lesson receipts</button>
        {items.map(value => <details key={value.index}><summary>Lesson {value.index + 1} · {value.state}</summary><p className="local-teacher__prompt">{value.prompt}</p><p>{value.receipt?.outputChannel ?? "No generated result yet"}{value.receipt?.transportDecorationsPreserved ? " · console decorations retained" : ""}</p>{value.coverage ? <p>{value.coverage.processedRecords.toLocaleString()} processed records · {value.coverage.rejectedRecords.toLocaleString()} rejected</p> : null}{value.source ? <code>Source SHA-256: {value.source.sha256}</code> : null}</details>)}
        {nextItem !== undefined ? <button className="button" type="button" disabled={itemsBusy} onClick={() => void loadItems(nextItem)}>More lesson receipts</button> : null}{itemsError ? <p role="alert">{itemsError}</p> : null}
      </section> : null}
    </details>
    </div>
  </details>;
}
