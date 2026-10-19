import React, { useEffect, useId, useRef, useState } from "react";
import type { DatasetManifest, DatasetPreviewProgress, OmniApi } from "../../shared/types";
import type { FactCheckClaim, FactCheckFinding, FactCheckWorkspace, LearningApi, LearningInput, LearningTask } from "../../shared/learning";
import type { UserQuestionsApi } from "../../shared/userQuestions";
import { UserQuestionPanel } from "./UserQuestionPanel";
import { LocalTeacherPanel } from "./LocalTeacherPanel";
import { UniversalSourcePicker } from "./UniversalSourcePicker";
import type { LocalTeacherApi } from "../../shared/localTeacher";
import { drainLearningTaskEvents, learningConstraints, learningSourcePercent, learningTaskStatus, mergeLearningTasks } from "./learningPresentation";
import "./learningWorkspace.css";
import "./trainingEvolution.css";

interface Props {
  brainId: string;
  api?: LearningApi;
  questions?: UserQuestionsApi;
  data?: Pick<OmniApi["data"], "preview" | "cancelPreview" | "onPreviewProgress">;
  localTeacher?: LocalTeacherApi;
  embedded?: boolean;
  runtimeStatus?: React.ReactNode;
  renderImprovement?: (objective: string) => React.ReactNode;
}
export function LearningWorkspace(props: Props) { return <LearningWorkspaceInstance key={props.brainId} {...props} />; }
const messageOf = (error: unknown): string => error instanceof Error ? error.message : "The operation could not finish.";

function LearningWorkspaceInstance({ brainId, api, questions, data, localTeacher, embedded = false,
  runtimeStatus, renderImprovement }: Props) {
  const heading = useId();
  const [tasks, setTasks] = useState<LearningTask[]>([]);
  const [selectedId, setSelectedId] = useState<string>();
  const [workspace, setWorkspace] = useState<FactCheckWorkspace>();
  const [text, setText] = useState("");
  const [urls, setUrls] = useState("");
  const [instructions, setInstructions] = useState("");
  const [manifests, setManifests] = useState<DatasetManifest[]>([]);
  const [followLinks, setFollowLinks] = useState(true);
  const [externalLinks, setExternalLinks] = useState(false);
  const [robots, setRobots] = useState(true);
  const [allowed, setAllowed] = useState("");
  const [excluded, setExcluded] = useState("");
  const [from, setFrom] = useState("");
  const [through, setThrough] = useState("");
  const [domainRules, setDomainRules] = useState("");
  const [steering, setSteering] = useState("");
  const [changeScope, setChangeScope] = useState(false);
  const [scopeEdit, setScopeEdit] = useState("");
  const [busy, setBusy] = useState(false);
  const [choosing, setChoosing] = useState(false);
  const [preview, setPreview] = useState<DatasetPreviewProgress>();
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [confirmStop, setConfirmStop] = useState(false);
  const submitting = useRef(false);
  const submission = useRef<{ signature: string; requestId: string } | undefined>(undefined);
  const alive = useRef(true);
  const previewId = useRef<string | undefined>(undefined);
  const previewStopId = useRef<string | undefined>(undefined);
  const selectedTaskId = useRef(selectedId);
  const taskPanel = useRef<HTMLDivElement | null>(null);
  selectedTaskId.current = selectedId;
  const task = tasks.find(value => value.id === selectedId);
  const liveTask = useRef(task);
  liveTask.current = task;

  useEffect(() => { alive.current = true; return () => { alive.current = false; if (previewId.current) void data?.cancelPreview(previewId.current).catch(() => undefined); }; }, [data]);
  useEffect(() => {
    if (!api) return;
    let active = true;
    const duringScan = new Map<string, LearningTask>();
    const pending = new Map<string, LearningTask>();
    let flushTimer: ReturnType<typeof setTimeout> | undefined;
    let scanning = true;
    const flush = (): void => { flushTimer = undefined; const received = drainLearningTaskEvents(pending); if (active) setTasks(current => mergeLearningTasks(current, received)); };
    let unsubscribe = () => {};
    try { unsubscribe = api.onEvent(event => {
      if (!active || event.task.brainId !== brainId) return;
      if (scanning) duringScan.set(event.task.id, event.task);
      const prior = pending.get(event.task.id);
      if (!prior || event.task.revision > prior.revision) pending.set(event.task.id, event.task);
      if (["complete", "stopped", "paused", "failed"].includes(event.task.state)) { if (flushTimer) clearTimeout(flushTimer); flush(); }
      else flushTimer ??= setTimeout(flush, 125);
    }); } catch (failure) { setError(messageOf(failure)); }
    void api.list(brainId).then(saved => {
      if (!active) return;
      const merged = mergeLearningTasks(saved.filter(value => value.brainId === brainId), [...duringScan.values()]);
      setTasks(current => mergeLearningTasks(current, merged));
      setSelectedId(current => current ?? merged[0]?.id);
      scanning = false;
      duringScan.clear();
    }).catch(failure => { if (active) { scanning = false; setError(messageOf(failure)); } });
    return () => { active = false; unsubscribe(); if (flushTimer) clearTimeout(flushTimer); };
  }, [api, brainId]);
  useEffect(() => {
    if (!api || task?.kind !== "fact-check") { setWorkspace(undefined); return; }
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const taskId = task.id;
    const read = async (): Promise<void> => {
      try { const value = await api.factCheck(brainId, taskId); if (active) setWorkspace(value); }
      catch (failure) { if (active) setError(messageOf(failure)); }
      if (active && liveTask.current?.id === taskId && !["complete", "stopped", "failed", "paused"].includes(liveTask.current.state)) timer = setTimeout(() => void read(), 750);
    };
    void read();
    return () => { active = false; if (timer) clearTimeout(timer); };
  }, [api, brainId, task?.kind, task?.id, task?.state]);

  const receive = (value: LearningTask): void => {
    if (!alive.current || value.brainId !== brainId) return;
    setTasks(current => mergeLearningTasks(current, [value]));
  };
  const choose = async (selection: "files" | "folder"): Promise<void> => {
    if (!data || choosing) return;
    setChoosing(true); setError(""); setNotice("");
    const requestId = crypto.randomUUID(); previewId.current = requestId; setPreview(undefined);
    previewStopId.current = undefined;
    let prepared = false, failed = false;
    const unsubscribe = data.onPreviewProgress(value => { if (alive.current && value.brainId === brainId && value.requestId === requestId) setPreview(value); });
    try { const manifest = await data.preview({ brainId, selection, requestId, policy: "pretrain" }); prepared = Boolean(manifest); if (alive.current && manifest) setManifests(current => [...current.filter(value => value.id !== manifest.id), manifest]); }
    catch (failure) { failed = true; if (alive.current) setError(messageOf(failure)); }
    finally {
      unsubscribe(); if (previewId.current === requestId) previewId.current = undefined;
      if (alive.current) {
        setChoosing(false);
        if (previewStopId.current === requestId) setNotice(prepared ? "Source preparation finished before Stop took effect." : failed ? "Source preparation ended." : "Source preparation stopped.");
      }
    }
  };
  const start = async (event: React.FormEvent): Promise<void> => {
    event.preventDefault(); if (!api || submitting.current) return;
    submitting.current = true; setBusy(true); setError(""); setNotice("");
    try {
      const sourceUrls = urls.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
      const constraints = learningConstraints({ allowed, excluded, from, through, domainRules });
      const common = { brainId, instructions: instructions.trim(), constraints };
      const inputs: LearningInput[] = [ ...(text.trim() ? [{ kind: "text" as const, text }] : []),
        ...manifests.map(manifest => ({ kind: "dataset" as const, manifestId: manifest.id })),
        ...sourceUrls.map(url => ({ kind: "web" as const, url, followLinks, followExternalLinks: externalLinks, respectRobots: robots })) ];
      if (!inputs.length) throw new Error("Add text, selected files or a source URL.");
      const payload = { ...common, inputs };
      const signature = JSON.stringify(payload);
      if (submission.current?.signature !== signature) submission.current = { signature, requestId: crypto.randomUUID() };
      const requestId = submission.current.requestId;
      const saved = await api.start({ ...common, inputs, requestId });
      if (alive.current) {
        receive(saved); setSelectedId(saved.id); setNotice("");
        submission.current = undefined; setText(""); setUrls(""); setManifests([]);
        requestAnimationFrame(() => { if (alive.current) taskPanel.current?.scrollIntoView({ behavior: "smooth", block: "start" }); });
      }
    } catch (failure) { if (alive.current) setError(messageOf(failure)); }
    finally { submitting.current = false; if (alive.current) setBusy(false); }
  };
  const control = async (action: "pause" | "resume" | "stop" | "steer"): Promise<void> => {
    if (!api || !task || submitting.current) return;
    submitting.current = true; setBusy(true); setError("");
    try {
      let constraints;
      if (action === "steer" && changeScope) {
        try { constraints = JSON.parse(scopeEdit) as LearningTask["policy"]["constraints"]; }
        catch { throw new Error("The task's web scope must be a valid JSON object."); }
        if (!constraints || typeof constraints !== "object" || Array.isArray(constraints)) throw new Error("The task's web scope must be an object.");
      }
      const saved = action === "steer" ? await api.steer({ brainId, taskId: task.id, expectedRevision: task.revision,
        ...(steering.trim() ? { instructions: steering } : {}), ...(constraints ? { constraints } : {}) }) : await api[action](brainId, task.id);
      receive(saved);
      if (alive.current) { setConfirmStop(false); if (action === "steer") { setSteering(""); setChangeScope(false); } }
    } catch (failure) { if (alive.current) setError(messageOf(failure)); }
    finally { submitting.current = false; if (alive.current) setBusy(false); }
  };
  const percent = task ? learningSourcePercent(task) : undefined;
  const terminal = task && ["complete", "stopped", "failed"].includes(task.state);
  const hasSources = Boolean(text.trim() || urls.trim() || manifests.length);
  const taskQuestionIds = [...new Set([...(task?.pendingQuestions ?? []),
    ...(workspace?.task.id === task?.id ? workspace?.claims.flatMap(claim => claim.questionId ? [claim.questionId] : []) ?? [] : [])])];
  return <section className={`learning-workspace${embedded ? " learning-workspace--embedded" : ""}`}
    aria-labelledby={embedded ? undefined : heading} aria-label={embedded ? "Learning and improvement" : undefined}>
    {!embedded ? <header><h1 id={heading}>Learn</h1><p>Learn from text, files, folders and the web.</p></header> : null}
    {!api ? <p role="status">Learning is available in the desktop app.</p> : null}
    <div className={embedded ? "training-evolution-layout" : "learning-workspace__body"}>
    <div className="training-evolution-main">
    <form className="surface learning-workspace__form" onSubmit={event => void start(event)}>
      <label>Text to learn<textarea rows={5} value={text} onChange={event => setText(event.target.value)} /></label>
      <div className="learning-workspace__actions">
        <UniversalSourcePicker label="Add source" disabled={!data || choosing || busy}
          onSelect={selection => void choose(selection)} />
        {choosing ? <><span role="status">{preview ? `${preview.phase} · ${preview.discoveredFiles.toLocaleString()} files · ${preview.hashedBytes.toLocaleString()} bytes hashed` : "Preparing the full source manifest…"}</span>
          <button className="button button--secondary" type="button" onClick={() => { const id = previewId.current; if (id && data) {
            previewStopId.current = id;
            void data.cancelPreview(id).then(stopped => { if (alive.current && previewId.current === id) setNotice(stopped ? "Stopping source preparation…" : "No active source preparation was stopped."); }).catch(failure => { if (alive.current) setError(messageOf(failure)); });
          } }}>Stop preparation</button></> : null}
      </div><ul className="learning-workspace__selected">{manifests.map(manifest => <li key={manifest.id}>
        <span><strong>{manifest.roots.slice(0, 3).map(root => root.split(/[\\/]/).filter(Boolean).at(-1) ?? root).join(", ")}{manifest.roots.length > 3 ? ` · ${manifest.roots.length - 3} more` : ""}</strong>
          <small>{manifest.discoveredFiles.toLocaleString()} files · {manifest.discoveredBytes.toLocaleString()} bytes</small>
          <details><summary>Selected roots</summary><ul>{manifest.roots.map(root => <li key={root}>{root}</li>)}</ul></details>
        </span>
        <button className="button button--ghost" type="button" aria-label={`Remove selected source ${manifest.roots[0]?.split(/[\\/]/).filter(Boolean).at(-1) ?? "selection"}`} onClick={() => setManifests(current => current.filter(value => value.id !== manifest.id))}>Remove</button>
      </li>)}</ul>
      <label>Web sources · one URL per line<textarea rows={2} value={urls} onChange={event => setUrls(event.target.value)} /></label>
      <label>{renderImprovement ? "Instructions or improvement goal" : "Instructions · optional"}<textarea rows={2} value={instructions} onChange={event => setInstructions(event.target.value)} /></label>
      <details><summary>Web scope and dates</summary><div className="learning-workspace__fields">
        <label>Only these sites · optional<textarea rows={2} value={allowed} onChange={event => setAllowed(event.target.value)} placeholder="docs.example.com" /></label>
        <label>Exclude these sites<textarea rows={2} value={excluded} onChange={event => setExcluded(event.target.value)} /></label>
        <label>Earliest publication date<input type="date" value={from} onChange={event => setFrom(event.target.value)} /></label>
        <label>Latest publication date<input type="date" value={through} onChange={event => setThrough(event.target.value)} /></label>
        <label className="learning-workspace__wide">Per-site date overrides · JSON, optional<textarea rows={2} value={domainRules} onChange={event => setDomainRules(event.target.value)} placeholder={'[{"domain":"example.com","dates":{"from":"2025-01-01"}}]'} /></label>
      </div><div className="learning-workspace__checks">
        <label><input type="checkbox" checked={followLinks} onChange={event => setFollowLinks(event.target.checked)} /> Follow links until stopped</label>
        <label><input type="checkbox" checked={externalLinks} onChange={event => setExternalLinks(event.target.checked)} /> Follow external links</label>
        <label><input type="checkbox" checked={robots} onChange={event => setRobots(event.target.checked)} /> Respect robots.txt</label>
        {!robots ? <p role="alert">Check the site’s access terms. Request pacing and private-network protections still apply.</p> : null}
      </div></details>
      <button className="button button--primary" type="submit" disabled={!api || !hasSources || busy || choosing}>{busy ? "Submitting…" : "Learn"}</button>
      {notice ? <p role="status" aria-live="polite">{notice}</p> : null}
    </form>
    {localTeacher ? <details className="training-evolution-teacher"><summary>Learn with a local teacher</summary>
      <LocalTeacherPanel brainId={brainId} api={localTeacher} /></details> : null}
    {error ? <p className="learning-workspace__error" role="alert">{error}</p> : null}
    {tasks.length ? <div className="learning-workspace__results">
      <aside aria-label="Saved learning tasks"><h2>Tasks</h2>{tasks.length ? <ul>{tasks.map(value => <li key={value.id}>
        <button type="button" className={`button ${selectedId === value.id ? "button--primary" : "button--secondary"}`} aria-pressed={selectedId === value.id} onClick={() => { setSelectedId(value.id); setWorkspace(undefined); setConfirmStop(false); setChangeScope(false); setSteering(""); }}>
          <span>{value.label}</span><small>{learningTaskStatus(value)}</small>
        </button></li>)}</ul> : <p>No tasks yet.</p>}</aside>
      <div className="surface learning-workspace__task" ref={taskPanel}>
        {task ? <><h2>{task.label}</h2><p role="status" aria-live="polite">{learningTaskStatus(task)} · {task.phase}{percent ? ` · ${task.currentSource ? "current source " : ""}${percent}` : ""}</p>
          {task.currentSource && task.state === "running" ? <p>Remaining time: estimating · {task.phase}</p> : null}
          <div className="learning-workspace__actions">
            {["queued", "running", "waiting"].includes(task.state) ? <button className="button button--secondary" type="button" disabled={busy} onClick={() => void control("pause")}>Pause</button> : null}
            {["paused", "stopped", "failed"].includes(task.state) ? <button className="button button--secondary" type="button" disabled={busy} onClick={() => void control("resume")}>Resume</button> : null}
            {!terminal ? <button className="button button--secondary" type="button" disabled={busy} onClick={() => setConfirmStop(true)}>Stop</button> : null}
          </div>
          {confirmStop ? <div role="group" aria-label="Confirm stopping learning"><p>Stop this task? Committed learning and its cursor stay saved.</p>
            <div className="learning-workspace__actions"><button className="button button--secondary" type="button" disabled={busy} onClick={() => void control("stop")}>Stop task</button>
            <button className="button button--secondary" type="button" onClick={() => setConfirmStop(false)}>Keep running</button></div></div> : null}
          {task.error ? <p className="learning-workspace__error">{task.error}</p> : null}
          <details><summary>Source coverage and saved scope</summary><p>{task.sources.filter(value => value.state === "complete").length} of {task.sources.length} sources settled · scope revision {task.policy.revision}</p>
            <ul>{task.sources.map(source => <li key={source.id}><strong>{source.label}</strong> · {source.state}{source.quarantined ? " · quarantined, no new learning" : ""}
              {source.coverage ? <span> · {source.coverage.processedRecords.toLocaleString()} processed records · {source.coverage.rejectedRecords.toLocaleString()} rejected · {source.coverage.processedBytes.toLocaleString()} bytes</span> : null}
              {source.error ? <p className="learning-workspace__error">{source.error}</p> : null}</li>)}</ul>
            <pre>{JSON.stringify(task.policy.constraints, null, 2)}</pre>
          </details>
          {workspace?.task.id === task.id ? <div className="learning-workspace__claims">{workspace.claims.map(claim => <ClaimReview key={`${task.id}:${claim.id}`} claim={claim} workspace={workspace} api={api!} onUpdated={value => {
            if (alive.current && selectedTaskId.current === value.task.id) setWorkspace(value); receive(value.task);
          }} onTask={saved => {
            setNotice("Correction task saved in Learn.");
            void api!.factCheck(brainId, task.id).then(value => { if (alive.current && selectedTaskId.current === value.task.id) setWorkspace(value); }).catch(failure => { if (alive.current) setError(messageOf(failure)); });
          }} />)}</div> : null}
          <form className="learning-workspace__steer" onSubmit={event => { event.preventDefault(); void control("steer"); }}>
            <label>Steer this task<textarea rows={3} value={steering} onChange={event => setSteering(event.target.value)} /></label>
            <label className="learning-workspace__inline-check"><input type="checkbox" checked={changeScope} onChange={event => { setChangeScope(event.target.checked); if (event.target.checked) setScopeEdit(JSON.stringify(task.policy.constraints, null, 2)); }} /> Change this task’s web scope</label>
            {changeScope ? <label>Replacement web scope · JSON<textarea rows={4} value={scopeEdit} onChange={event => setScopeEdit(event.target.value)} /></label> : null}
            <button className="button button--secondary" type="submit" disabled={(!steering.trim() && !changeScope) || busy || Boolean(terminal)}>Send steering</button>
          </form>
          {questions && taskQuestionIds.length ? <UserQuestionPanel brainId={brainId} api={questions} questionIds={taskQuestionIds} /> : null}
        </> : <p>Select a task to see progress, evidence and questions.</p>}
      </div>
    </div> : <p className="training-evolution-empty">No tasks yet.</p>}
    </div>
    {runtimeStatus || renderImprovement ? <aside className="training-evolution-sidebar" aria-label="Progress and evolution">
      {runtimeStatus}{renderImprovement?.(instructions.trim())}
    </aside> : null}
    </div>
  </section>;
}

function ClaimReview({ claim, workspace, api, onUpdated, onTask }: {
  claim: FactCheckClaim; workspace: FactCheckWorkspace; api: LearningApi;
  onUpdated: (value: FactCheckWorkspace) => void; onTask: (value: LearningTask) => void;
}) {
  const [conclusion, setConclusion] = useState<FactCheckFinding["conclusion"]>("undetermined");
  const [explanation, setExplanation] = useState("");
  const [correction, setCorrection] = useState("");
  const [evidenceIds, setEvidenceIds] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const submitting = useRef(false);
  const reviewSubmission = useRef<{ signature: string; requestId: string } | undefined>(undefined);
  const review = async (event: React.FormEvent): Promise<void> => {
    event.preventDefault(); if (submitting.current) return;
    submitting.current = true; setBusy(true); setError("");
    try {
      const request = { brainId: workspace.task.brainId, taskId: workspace.task.id, claimId: claim.id,
        conclusion, explanation, evidenceIds, ...(correction.trim() ? { proposedCorrection: correction } : {}) };
      const signature = JSON.stringify(request);
      if (reviewSubmission.current?.signature !== signature) reviewSubmission.current = { signature, requestId: crypto.randomUUID() };
      onUpdated(await api.review({ ...request, requestId: reviewSubmission.current.requestId }));
      reviewSubmission.current = undefined; setExplanation(""); setCorrection(""); setEvidenceIds([]);
    }
    catch (failure) { setError(messageOf(failure)); }
    finally { submitting.current = false; setBusy(false); }
  };
  const learnCorrection = async (finding: FactCheckFinding): Promise<void> => {
    if (submitting.current) return;
    submitting.current = true; setBusy(true); setError("");
    try { onTask(await api.acceptCorrection(workspace.task.brainId, workspace.task.id, claim.id, finding.id)); }
    catch (failure) { setError(messageOf(failure)); }
    finally { submitting.current = false; setBusy(false); }
  };
  return <article className="learning-workspace__claim"><h3>{claim.text}</h3><p>{claim.state.replaceAll("-", " ")}</p>
    <ul>{claim.evidence.map(evidence => <li key={evidence.id}><span>{evidence.url}</span><small>{evidence.capture?.replaceAll("-", " ") ?? "source"} · {evidence.relation} · publication {evidence.publication?.status ?? "unknown"}</small></li>)}</ul>
    {!claim.evidence.length ? <p>No captured evidence yet.</p> : null}
    {claim.findings.map(finding => <details key={finding.id}><summary>{finding.conclusion} · {finding.source === "model" ? "model proposal" : "user review"}</summary>
      <p>{finding.explanation}</p>{finding.proposedCorrection ? <><p>{finding.proposedCorrection}</p><button className="button button--secondary" type="button" disabled={busy || Boolean(claim.correctionTaskId)} onClick={() => void learnCorrection(finding)}>{claim.correctionTaskId ? "Correction queued" : "Learn correction"}</button></> : null}</details>)}
    {claim.reviewError ? <p className="learning-workspace__error">{claim.reviewError}</p> : null}
    <details><summary>Add your review</summary><form onSubmit={event => void review(event)}>
      <label>Proposed conclusion<select value={conclusion} onChange={event => setConclusion(event.target.value as FactCheckFinding["conclusion"])}>
        <option value="undetermined">Undetermined</option><option value="supported">Supported by selected evidence</option><option value="contradicted">Contradicted by selected evidence</option><option value="mixed">Mixed evidence</option>
      </select></label>
      <fieldset><legend>Evidence for this review</legend>{claim.evidence.map(evidence => <label key={evidence.id}><input type="checkbox" checked={evidenceIds.includes(evidence.id)} onChange={event => setEvidenceIds(current => event.target.checked ? [...current, evidence.id] : current.filter(value => value !== evidence.id))} />{evidence.url}</label>)}</fieldset>
      <label>Explanation<textarea rows={2} value={explanation} onChange={event => setExplanation(event.target.value)} /></label>
      <label>Correction to learn · optional<textarea rows={2} value={correction} onChange={event => setCorrection(event.target.value)} /></label>
      <button className="button button--secondary" type="submit" disabled={busy || !explanation.trim() || (conclusion !== "undetermined" && !evidenceIds.length)}>Save review</button>
    </form></details>
    {error ? <p className="learning-workspace__error" role="alert">{error}</p> : null}
  </article>;
}
