import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ImaginationRecording } from "../../shared/imaginationCapture";
import type { ImaginationAssembly, ImaginationAssemblyApi, ImaginationAssemblyManifestSummary, ImaginationAssemblyPlan, ImaginationAssemblyPlanRequest, ImaginationPlanEvent } from "../../shared/imaginationAssembly";
import { EMPTY_REPLAY_OPTIONS, mergeReplayJobs, mergeReplayPlanProgress, replayIsActive, replayOptions, replayPlanStatus, replayRecordingId, replayState, type ReplayOptionsDraft } from "./imaginationReplayPresentation";
import { conciseUiMessage } from "./uiPresentation";

export function ImaginationReplayControls({ recording, api }: { recording: ImaginationRecording; api?: ImaginationAssemblyApi }) {
  return api ? <ReplayOwner key={`${recording.brainId}:${recording.id}`} recording={recording} api={api} /> : null;
}

function ReplayOwner({ recording, api }: { recording: ImaginationRecording; api: ImaginationAssemblyApi }) {
  const brainId = recording.brainId;
  const [jobs, setJobs] = useState<ImaginationAssembly[]>([]), [cursor, setCursor] = useState<string>();
  const [draft, setDraft] = useState<ReplayOptionsDraft>(EMPTY_REPLAY_OPTIONS), [plan, setPlan] = useState<{ value: ImaginationAssemblyPlan; revision: number }>();
  const [selectedId, setSelectedId] = useState<string>(), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const [loading, setLoading] = useState(false), [summary, setSummary] = useState<ImaginationAssemblyManifestSummary>();
  const [receiptError, setReceiptError] = useState("");
  const scope = useRef<object | null>(null), command = useRef(false), pageBusy = useRef(false);
  const pendingCreate = useRef<{ requestId: string; planId: string; revision: number } | undefined>(undefined);
  const [retryCreate, setRetryCreate] = useState(false);
  const pendingPlan = useRef<ImaginationAssemblyPlanRequest | undefined>(undefined), progressRef = useRef<ImaginationPlanEvent | undefined>(undefined);
  const [retryPlan, setRetryPlan] = useState(false), [preparing, setPreparing] = useState(false), [stoppingPlan, setStoppingPlan] = useState(false), [planNotice, setPlanNotice] = useState("");
  const [planProgress, setPlanProgress] = useState<ImaginationPlanEvent>();
  const recordingRef = useRef(recording); recordingRef.current = recording;
  const selectedRef = useRef(selectedId); selectedRef.current = selectedId;
  const receive = (values: ImaginationAssembly[]) => setJobs(previous => mergeReplayJobs(previous, values, brainId, recording.id));
  useLayoutEffect(() => { const owner = {}; scope.current = owner; return () => { if (scope.current === owner) scope.current = null; }; }, [api, brainId, recording.id]);
  useEffect(() => { setPlan(undefined); if (!pendingPlan.current) { progressRef.current = undefined; setPlanProgress(undefined); setPlanNotice(""); } }, [recording.revision]);

  useEffect(() => {
    const owner = scope.current; let alive = true;
    // Events read only the specific changed job; opening this panel never creates a replay.
    const inflight = new Set<string>(), dirty = new Set<string>();
    const current = () => alive && scope.current === owner;
    const read = async (id: string) => {
      if (!current()) return;
      if (inflight.has(id)) { dirty.add(id); return; }
      inflight.add(id);
      try { const value = await api.get(brainId, id); if (current()) receive([value]); }
      catch (reason) { if (current()) setError(conciseUiMessage(reason, "Replay status could not load.")); }
      finally { inflight.delete(id); if (current() && dirty.delete(id)) void read(id); }
    };
    const unsubscribe = api.onEvent(event => { if (event.brainId === brainId) void read(event.assemblyId); });
    const unsubscribePlan = api.onPlanEvent(event => {
      if (!current() || !pendingPlan.current) return;
      const next = mergeReplayPlanProgress(progressRef.current, event, brainId, pendingPlan.current.requestId);
      if (next === progressRef.current) return;
      progressRef.current = next; setPlanProgress(next);
    });
    setLoading(true);
    void api.list(brainId, { recordingId: recording.id, limit: 20 }).then(page => {
      if (!current()) return; receive(page.entries); setCursor(page.nextAfterId);
    }).catch(reason => { if (current()) setError(conciseUiMessage(reason, "Saved replays could not load.")); }).finally(() => { if (current()) setLoading(false); });
    return () => { alive = false; unsubscribe(); unsubscribePlan(); };
  }, [api, brainId, recording.id]);

  const run = async (work: (owner: object) => Promise<void>) => {
    const owner = scope.current; if (!owner || command.current) return;
    command.current = true; setBusy(true); setError("");
    try { await work(owner); }
    catch (reason) { if (scope.current === owner) setError(conciseUiMessage(reason, "Replay control was not acknowledged.")); }
    finally { if (scope.current === owner) { command.current = false; setBusy(false); } }
  };
  const prepare = () => void run(async owner => {
    if (!pendingPlan.current) {
      pendingPlan.current = { brainId, recordingId: recording.id, expectedRevision: recordingRef.current.revision, requestId: crypto.randomUUID(), options: replayOptions(draft) };
      progressRef.current = undefined; setPlanProgress(undefined);
    }
    const request = pendingPlan.current; setRetryPlan(true); setPreparing(true); setPlanNotice("");
    try {
      const value = await api.plan(request);
      if (scope.current !== owner) return;
      pendingPlan.current = undefined; setRetryPlan(false);
      if (recordingRef.current.revision !== request.expectedRevision) { setPlan(undefined); setPlanNotice("Recording changed. Prepare its current version before making a replay."); return; }
      setPlan({ value, revision: request.expectedRevision }); pendingCreate.current = undefined; setRetryCreate(false);
    } catch (reason) {
      if (scope.current !== owner) return;
      const state = progressRef.current?.state;
      if (state === "cancelled" || state === "failed" || state === "dormant") { pendingPlan.current = undefined; setRetryPlan(false); }
      if (state === "cancelled") setPlanNotice("Preparation stopped. Saved recordings and fragments are kept."); else throw reason;
    } finally { if (scope.current === owner) { setPreparing(false); setStoppingPlan(false); } }
  });
  const stopPreparation = async () => {
    const owner = scope.current, request = pendingPlan.current; if (!owner || !request || stoppingPlan) return;
    setStoppingPlan(true); setPlanNotice("");
    try {
      const result = await api.cancelPlan({ brainId, requestId: request.requestId });
      if (scope.current !== owner || result.brainId !== brainId || result.requestId !== request.requestId) return;
      if (pendingPlan.current && pendingPlan.current.requestId !== request.requestId) return;
      if (result.state === "cancelled" || result.state === "failed" || result.state === "dormant") {
        pendingPlan.current = undefined; setRetryPlan(false); setPlanNotice(result.state === "cancelled" ? "Preparation stopped. Saved recordings and fragments are kept." : "Preparation ended. Start a new preparation when ready.");
      } else setPlanNotice(result.state === "ready" ? pendingPlan.current ? "Preparation finished before Stop. Retry the same preparation to receive its plan." : "Preparation finished before Stop. Its ready plan is kept." : result.state === "absent" ? "Preparation ownership is unconfirmed. Retry the same request to reconcile it." : "Waiting for preparation to acknowledge Stop…");
    } catch (reason) { if (scope.current === owner) setError(conciseUiMessage(reason, "Preparation Stop was not acknowledged.")); }
    finally { if (scope.current === owner) setStoppingPlan(false); }
  };
  const make = () => void run(async owner => {
    const prepared = plan;
    if (!pendingCreate.current) {
      if (!prepared || prepared.revision !== recordingRef.current.revision) throw new Error("This recording changed. Prepare its current replay first.");
      pendingCreate.current = { requestId: crypto.randomUUID(), planId: prepared.value.id, revision: prepared.revision };
    }
    const request = pendingCreate.current;
    // A lost response retries the exact request, never a second codec job.
    setRetryCreate(true);
    const value = await api.assemble({ brainId, recordingId: recording.id, expectedRevision: request.revision, requestId: request.requestId, planId: request.planId });
    if (scope.current !== owner || value.brainId !== brainId || replayRecordingId(value) !== recording.id) return;
    receive([value]); setSelectedId(value.id); pendingCreate.current = undefined; setRetryCreate(false); setPlan(undefined);
  });
  const control = (value: ImaginationAssembly, operation: "cancel" | "resume") => void run(async owner => {
    const result = await api[operation]({ brainId, assemblyId: value.id, expectedRevision: value.revision });
    if (scope.current === owner) receive([result]);
  });
  const older = async () => {
    const owner = scope.current; if (!owner || pageBusy.current || !cursor) return;
    pageBusy.current = true; setLoading(true);
    try { const page = await api.list(brainId, { recordingId: recording.id, afterId: cursor, limit: 20 });
      if (scope.current === owner) { receive(page.entries); setCursor(page.nextAfterId); }
    } catch (reason) { if (scope.current === owner) setError(conciseUiMessage(reason, "Older replays could not load.")); }
    finally { if (scope.current === owner) { pageBusy.current = false; setLoading(false); } }
  };
  const receipt = () => void run(async owner => {
    if (!selectedId) return;
    const id = selectedId; setReceiptError("");
    try { const value = await api.manifest(brainId, id); if (scope.current === owner && selectedRef.current === id) setSummary(value); }
    catch (reason) { if (scope.current === owner && selectedRef.current === id) setReceiptError(conciseUiMessage(reason, "Replay receipt could not load.")); }
  });
  const edit = (key: keyof ReplayOptionsDraft, value: string) => { setDraft(previous => ({ ...previous, [key]: value })); setPlan(undefined); progressRef.current = undefined; setPlanProgress(undefined); setPlanNotice(""); };
  const selected = jobs.find(value => value.id === selectedId);
  const recordingActive = recording.state === "recording";
  return <section className="imagination-replay" aria-label="Recording replay">
    <strong>Replay</strong>
    <p className="imagination-capture__notice">Replays delivered changes on a presentation timeline. Original outputs stay saved separately.</p>
    <details><summary>Replay options</summary>
      <p className="imagination-capture__notice">Leave blank for Auto from recorded media. Overrides are checked, not silently reduced.</p>
      <div className="imagination-capture__row">{(["width", "height", "fps"] as const).map(key => <label key={key}>{key === "fps" ? "Frames per second" : key === "width" ? "Width" : "Height"}<input type="number" min="0" step={key === "fps" ? "any" : "1"} placeholder="Auto" value={draft[key]} disabled={busy || stoppingPlan || retryCreate || retryPlan} onChange={event => edit(key, event.target.value)} /></label>)}</div>
      <label>Historical ending hold · seconds<input type="number" min="0" step="0.001" placeholder="Measured end when available" value={draft.historicalTailSeconds} disabled={busy || stoppingPlan || retryCreate || retryPlan} onChange={event => edit("historicalTailSeconds", event.target.value)} /></label>
      <label>Retained source offsets · stream ID = seconds<textarea rows={2} value={draft.streamOffsets} disabled={busy || stoppingPlan || retryCreate || retryPlan} placeholder="Only for an explicitly known retained-source position" onChange={event => edit("streamOffsets", event.target.value)} /></label>
    </details>
    {recordingActive ? <p role="status">Stop recording to prepare a fixed replay.</p> : null}
    {plan ? <div role="status" className="imagination-replay__plan">
      <span>{(plan.value.durationMs / 1000).toLocaleString(undefined, { maximumFractionDigits: 3 })}s · {plan.value.sourceFragments.toLocaleString()} source fragments · {plan.value.totalSpans.toLocaleString()} spans</span>
      <span>{plan.value.outputKind === "audio-only" ? "Audio replay · WAV" : `${plan.value.options.width} × ${plan.value.options.height} · ${plan.value.options.fps.toLocaleString()} fps · MP4`} · estimated peak {(plan.value.estimatedPeakBytes / 1048576).toLocaleString(undefined, { maximumFractionDigits: 1 })} MiB</span>
      <span>{plan.value.timingIssueCount.toLocaleString()} timing issues · {plan.value.captureGapCount.toLocaleString()} capture gaps · up to {plan.value.quantizationMs.toLocaleString(undefined, { maximumFractionDigits: 3 })} ms presentation quantization</span>
    </div> : null}
    <div className="imagination-capture__row">
      <button type="button" className="button" disabled={busy || stoppingPlan || (recordingActive && !retryPlan) || retryCreate} onClick={prepare}>{preparing ? "Preparing replay…" : retryPlan ? "Retry same preparation" : "Prepare replay"}</button>
      {retryPlan ? <button type="button" className="button" disabled={stoppingPlan} onClick={() => void stopPreparation()}>{stoppingPlan ? "Awaiting preparation Stop…" : "Stop preparation"}</button> : null}
      {plan || retryCreate ? <button type="button" className="button button--primary" disabled={busy || stoppingPlan || (recordingActive && !retryCreate)} onClick={make}>{busy ? "Awaiting acknowledgement…" : retryCreate ? "Retry same replay request" : "Make replay"}</button> : null}
    </div>
    {planProgress && !planNotice ? <p role="status">{replayPlanStatus(planProgress)}</p> : null}
    {planNotice ? <p role="status">{planNotice}</p> : null}
    {retryCreate && !busy ? <p role="status">Acknowledgement is unconfirmed. Retry reconciles the same request without starting a duplicate.</p> : null}
    {error ? <p role="alert">{error}</p> : null}
    <ul className="imagination-capture__recordings">{jobs.map(value => <li key={value.id}>
      <button type="button" className="button" aria-pressed={selectedId === value.id} onClick={() => { setSelectedId(value.id); setSummary(undefined); setReceiptError(""); }}><span>{new Date(value.createdAt).toLocaleString()}</span><small>{replayState(value)}</small></button>
      {replayIsActive(value) ? <button type="button" className="button" disabled={busy} onClick={() => control(value, "cancel")}>Stop replay</button> : value.state === "dormant" ? <button type="button" className="button" disabled={busy} onClick={() => control(value, "resume")}>Resume replay</button> : null}
    </li>)}</ul>
    {cursor ? <button type="button" className="button" disabled={loading} onClick={() => void older()}>Older replays</button> : null}
    {loading ? <p role="status">Loading saved replays…</p> : null}
    {selected ? <div className="imagination-replay__result">
      {selected.state === "complete" && selected.available && selected.mediaUrl ? selected.mimeType === "audio/wav" ? <audio controls preload="none" src={selected.mediaUrl} aria-label="Saved imagination audio replay" /> : selected.mimeType === "video/mp4" ? <video controls preload="none" src={selected.mediaUrl} aria-label="Saved imagination video replay" /> : null : null}
      {selected.error ? <p role="alert">{conciseUiMessage(selected.error, "Replay could not finish.")}</p> : null}
      {selected.state === "complete" && !selected.available ? <p role="status">Replay file is unavailable. Original fragments are retained.</p> : null}
      <button type="button" className="button" disabled={busy} onClick={receipt}>Show replay receipt</button>
      {receiptError ? <p role="alert">{receiptError}</p> : null}
      {summary ? <p className="imagination-capture__notice">{summary.fragmentCount.toLocaleString()} fragments · {summary.trackCount.toLocaleString()} tracks · {summary.spanCount.toLocaleString()} spans · {summary.measuredEnd ? "Measured recording end" : "Explicit historical ending"}. Presentation timing, not native synchronization.<br /><span className="imagination-replay__hash">Manifest SHA-256: {summary.manifestSha256}</span></p> : null}
    </div> : null}
  </section>;
}
