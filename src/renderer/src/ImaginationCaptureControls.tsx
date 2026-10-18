import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ImaginationCaptureApi, ImaginationCapturedFragment, ImaginationCaptureStatus, ImaginationRecording } from "../../shared/imaginationCapture";
import { CAPTURE_WINDOW_PRESETS, captureRecordingState, captureTime, mergeCaptureFragments, mergeCaptureRecordings, parseCaptureSeconds } from "./imaginationCapturePresentation";
import "./imaginationCapture.css";
import { ImaginationReplayControls } from "./ImaginationReplayControls";

/** A new owner always gets a new local state; no renderer state is a capture grant. */
export function ImaginationCaptureControls({ brainId, api }: { brainId: string; api?: ImaginationCaptureApi }) {
  return api ? <CaptureOwner key={brainId} brainId={brainId} api={api} /> : <p className="imagination-capture__notice">Recording is available in the desktop app.</p>;
}

function CaptureOwner({ brainId, api }: { brainId: string; api: ImaginationCaptureApi }) {
  const [status, setStatus] = useState<ImaginationCaptureStatus>();
  const [records, setRecords] = useState<ImaginationRecording[]>([]);
  const [total, setTotal] = useState(0), [older, setOlder] = useState<number>();
  const [busy, setBusy] = useState(false), [loading, setLoading] = useState(true), [error, setError] = useState("");
  const [buffer, setBuffer] = useState("300"), [alwaysSave, setAlwaysSave] = useState(false);
  const [lookback, setLookback] = useState("0"), [pendingRecord, setPendingRecord] = useState<{ requestId: string; lookbackSeconds: number }>();
  const [selected, setSelected] = useState<string>();
  const [fragments, setFragments] = useState<ImaginationCapturedFragment[]>([]);
  const [fragmentTotal, setFragmentTotal] = useState(0), [nextFragment, setNextFragment] = useState<number>();
  const [fragmentBusy, setFragmentBusy] = useState(false), [fragmentError, setFragmentError] = useState("");
  const scope = useRef<object | null>(null), selectedRef = useRef<string | undefined>(undefined), readVersion = useRef(0);
  const settingsRevision = useRef<number | undefined>(undefined), refreshRef = useRef<() => void>(() => undefined), pendingRequest = useRef<typeof pendingRecord>(undefined);
  const commandPending = useRef(false), pagePending = useRef(false);
  const fragmentVersion = useRef(0);
  useLayoutEffect(() => { const owner = {}; scope.current = owner; return () => { if (scope.current === owner) scope.current = null; }; }, [api, brainId]);

  useEffect(() => {
    const owner = scope.current; let stopped = false, inflight = false, dirty = false;
    let scheduled: ReturnType<typeof setTimeout> | undefined;
    const current = () => !stopped && scope.current === owner;
    const refresh = async () => {
      if (!current()) return;
      if (commandPending.current) { dirty = true; return; }
      if (inflight) { dirty = true; return; }
      inflight = true; const version = ++readVersion.current;
      try {
        const [next, page] = await Promise.all([api.status(brainId), api.recordings(brainId, { limit: 20 })]);
        if (!current() || version !== readVersion.current || next.brainId !== brainId) return;
        if (commandPending.current) { dirty = true; return; }
        setStatus(next); setRecords(previous => mergeCaptureRecordings(previous, page.entries, brainId)); setTotal(page.total);
        setOlder(previous => previous === undefined ? page.nextBeforeSequence : previous);
        // Frame events never overwrite a user's unsaved settings edits.
        if (settingsRevision.current !== next.settings.revision) {
          settingsRevision.current = next.settings.revision; setBuffer(String(next.settings.bufferSeconds)); setAlwaysSave(next.settings.alwaysSave);
        }
        if (pendingRequest.current?.requestId === next.recording?.requestId) { pendingRequest.current = undefined; setPendingRecord(undefined); }
      } catch (reason) { if (current()) setError(reason instanceof Error ? reason.message : "Recording status could not load."); }
      finally { inflight = false; if (current()) { setLoading(false); if (dirty && !commandPending.current) { dirty = false; schedule(); } } }
    };
    const schedule = () => { if (!current() || scheduled) return; scheduled = setTimeout(() => { scheduled = undefined; void refresh(); }, 200); };
    refreshRef.current = schedule;
    const unsubscribe = api.onEvent(event => { if (event.brainId === brainId) schedule(); });
    void refresh();
    return () => { stopped = true; if (scheduled) clearTimeout(scheduled); unsubscribe(); if (refreshRef.current === schedule) refreshRef.current = () => undefined; };
  }, [api, brainId]);

  const run = async (work: () => Promise<void>) => {
    if (commandPending.current || !scope.current) return;
    const owner = scope.current; commandPending.current = true; readVersion.current++; setBusy(true); setError("");
    try { await work(); } catch (reason) { if (scope.current === owner) setError(reason instanceof Error ? reason.message : "Recording control was not acknowledged."); }
    finally { if (scope.current === owner) { commandPending.current = false; setBusy(false); refreshRef.current(); } }
  };
  const record = () => void run(async () => {
    const seconds = pendingRequest.current?.lookbackSeconds ?? parseCaptureSeconds(lookback, true);
    if (!status || seconds > status.settings.bufferSeconds) throw new Error("Lookback exceeds this rolling window; choose a shorter duration.");
    const request = pendingRequest.current ?? { requestId: crypto.randomUUID(), lookbackSeconds: seconds };
    pendingRequest.current = request; setPendingRecord(request); const owner = scope.current;
    const result = await api.record({ brainId, ...request });
    if (scope.current !== owner || result.brainId !== brainId) return;
    pendingRequest.current = undefined; setPendingRecord(undefined); setRecords(previous => mergeCaptureRecordings(previous, [result], brainId)); setSelected(result.id);
    setStatus(previous => previous ? { ...previous, recording: result, autoPaused: false } : previous);
  });
  const applySettings = () => void run(async () => {
    if (!status) return; const owner = scope.current;
    const next = await api.settings({ brainId, expectedRevision: status.settings.revision, bufferSeconds: parseCaptureSeconds(buffer), alwaysSave });
    if (scope.current === owner && next.brainId === brainId) setStatus(next);
  });
  const control = (recording: ImaginationRecording, operation: "stop" | "resume") => void run(async () => {
    const owner = scope.current;
    const next = await api[operation]({ brainId, recordingId: recording.id, expectedRevision: recording.controlRevision });
    if (scope.current === owner && next.brainId === brainId) {
      setRecords(previous => mergeCaptureRecordings(previous, [next], brainId));
      setStatus(previous => {
        if (!previous) return previous;
        if (next.state === "recording") return { ...previous, recording: next, autoPaused: false };
        if (previous.recording?.id !== next.id) return previous;
        const { recording: _recording, ...rest } = previous; return { ...rest, autoPaused: true };
      });
    }
  });
  const loadOlder = async () => {
    if (older === undefined || pagePending.current || !scope.current) return;
    const owner = scope.current; pagePending.current = true; setLoading(true);
    try { const page = await api.recordings(brainId, { beforeSequence: older, limit: 20 });
      if (scope.current === owner) { setRecords(previous => mergeCaptureRecordings(previous, page.entries, brainId)); setTotal(page.total); setOlder(page.nextBeforeSequence); }
    } catch (reason) { if (scope.current === owner) setError(reason instanceof Error ? reason.message : "Older recordings could not load."); }
    finally { if (scope.current === owner) { pagePending.current = false; setLoading(false); } }
  };

  const selectedRecord = records.find(recording => recording.id === selected);
  const loadFragments = useCallback(async (id: string, afterIndex?: number) => {
    const owner = scope.current; if (!owner) return; const version = ++fragmentVersion.current;
    setFragmentBusy(true); setFragmentError("");
    try { const page = await api.fragments(brainId, id, { ...(afterIndex === undefined ? {} : { afterIndex }), limit: 20 });
      if (scope.current !== owner || selectedRef.current !== id || version !== fragmentVersion.current) return;
      setFragments(previous => mergeCaptureFragments(afterIndex === undefined ? [] : previous, page.entries, id)); setFragmentTotal(page.total); setNextFragment(page.nextAfterIndex);
    } catch (reason) { if (scope.current === owner && selectedRef.current === id && version === fragmentVersion.current) setFragmentError(reason instanceof Error ? reason.message : "Saved fragments could not load."); }
    finally { if (scope.current === owner && selectedRef.current === id && version === fragmentVersion.current) setFragmentBusy(false); }
  }, [api, brainId]);
  useEffect(() => {
    selectedRef.current = selected; fragmentVersion.current++;
    setFragments([]); setFragmentTotal(0); setNextFragment(undefined); setFragmentError(""); setFragmentBusy(false);
    if (selected) void loadFragments(selected);
  }, [selected, loadFragments]);

  const active = status?.recording;
  const settingsChanged = status && (buffer !== String(status.settings.bufferSeconds) || alwaysSave !== status.settings.alwaysSave || status.settings.dormant || status.autoPaused);
  return <section className="imagination-capture" aria-label="Imagination recording">
    <div className="imagination-capture__heading"><strong>Imagination recording</strong>
      <span>{active ? "Recording" : status?.autoPaused ? "Recording stopped" : "Rolling window"}</span></div>
    {status ? <p className="imagination-capture__notice">{status.bufferedFragments.toLocaleString()} buffered fragments · {captureTime(status.availableLookbackSeconds)} available</p> : <p role="status">{loading ? "Loading recording controls…" : "Recording status unavailable"}</p>}
    <div className="imagination-capture__row">
      <label>Record from<select value={lookback === "0" || CAPTURE_WINDOW_PRESETS.includes(Number(lookback) as typeof CAPTURE_WINDOW_PRESETS[number]) ? lookback : "custom"} disabled={busy || Boolean(pendingRecord)} onChange={event => setLookback(event.target.value === "custom" ? "90" : event.target.value)}>
        <option value="0">Now</option>{CAPTURE_WINDOW_PRESETS.map(value => <option key={value} value={value}>{captureTime(value)} ago</option>)}<option value="custom">Custom lookback</option>
      </select></label>
      {active ? <button type="button" className="button" disabled={busy} onClick={() => control(active, "stop")}>Stop recording</button>
        : <button type="button" className="button button--primary" disabled={busy || !status} onClick={record}>{busy ? "Saving…" : pendingRecord ? "Retry recording request" : "Record"}</button>}
    </div>
    {lookback !== "0" ? <label>Lookback seconds<input aria-label="Record lookback seconds" inputMode="decimal" type="number" min="0" max="300" step="0.001" value={lookback} disabled={busy || Boolean(pendingRecord)} onChange={event => setLookback(event.target.value)} /></label> : null}
    {active ? <p role="status">{active.fragmentCount.toLocaleString()} saved fragments · {active.gapCount.toLocaleString()} gaps</p> : null}
    {pendingRecord ? <p className="imagination-capture__notice">Awaiting acknowledgement. Retry uses the same request, not a second recording.</p> : null}
    <details><summary>Recording options</summary>
      <div className="imagination-capture__row"><label>Rolling window<select value={CAPTURE_WINDOW_PRESETS.includes(Number(buffer) as typeof CAPTURE_WINDOW_PRESETS[number]) ? buffer : "custom"} disabled={busy} onChange={event => setBuffer(event.target.value === "custom" ? "90" : event.target.value)}>
        {CAPTURE_WINDOW_PRESETS.map(value => <option key={value} value={value}>{captureTime(value)}</option>)}<option value="custom">Custom</option>
      </select></label><label>Window seconds<input aria-label="Rolling window seconds" inputMode="decimal" type="number" min="0.001" max="300" step="0.001" value={buffer} disabled={busy} onChange={event => setBuffer(event.target.value)} /></label></div>
      <label className="imagination-capture__check"><input type="checkbox" checked={alwaysSave} disabled={busy || !status} onChange={event => setAlwaysSave(event.target.checked)} />Always save spontaneous imagination</label>
      <button type="button" className="button" disabled={busy || !settingsChanged} onClick={applySettings}>{status?.settings.dormant ? "Activate these recording options" : "Apply recording options"}</button>
    </details>
    {status?.settings.dormant ? <p className="imagination-capture__notice">Copied recording options are paused until you activate them.</p> : null}
    {error ? <p role="alert">{error}</p> : null}
    <details><summary>Saved recordings ({Math.max(total, records.length).toLocaleString()})</summary>
      {records.length ? <ul className="imagination-capture__recordings">{records.map(recording => <li key={recording.id}>
        <button type="button" className="button" aria-pressed={selected === recording.id} onClick={() => setSelected(recording.id)}>
          <span>{new Date(recording.startedAt).toLocaleString()}</span><small>{captureRecordingState(recording)} · {recording.fragmentCount.toLocaleString()} fragments · {recording.gapCount.toLocaleString()} gaps</small>
        </button>
        {recording.state === "dormant" || recording.state === "resource-paused" ? <button type="button" className="button" disabled={busy || Boolean(active)} onClick={() => control(recording, "resume")}>Resume recording</button> : null}
      </li>)}</ul> : <p>No saved recordings.</p>}
      {older !== undefined && records.length < total ? <button type="button" className="button" disabled={loading} onClick={() => void loadOlder()}>Older recordings</button> : null}
      {selectedRecord ? <section aria-label="Saved imagination fragments" className="imagination-capture__fragments">
        <ImaginationReplayControls recording={selectedRecord} api={api.assembly} />
        <strong>Saved fragments</strong><p className="imagination-capture__notice">{captureTime(selectedRecord.availableLookbackSeconds)} retained lookback · {fragmentTotal.toLocaleString()} fragments. Individual delivered outputs, not an assembled video.</p>
        <button type="button" className="button" disabled={fragmentBusy} onClick={() => void loadFragments(selectedRecord.id)}>Refresh fragments</button>
        {fragmentError ? <p role="alert">{fragmentError}</p> : null}
        {fragments.map(fragment => <figure key={fragment.id}>
          {fragment.available && fragment.mediaUrl ? fragment.mimeType.startsWith("image/") ? <img loading="lazy" src={fragment.mediaUrl} alt={`Saved imagination fragment ${fragment.index + 1}`} />
            : fragment.mimeType.startsWith("audio/") ? <audio controls preload="none" src={fragment.mediaUrl} aria-label={`Saved audio fragment ${fragment.index + 1}`} />
              : <video controls preload="none" src={fragment.mediaUrl} aria-label={`Saved video fragment ${fragment.index + 1}`} /> : <span>Fragment unavailable</span>}
          <figcaption>{fragment.index + 1} · {fragment.modality} · {captureTime(fragment.offsetMs / 1000)} · {fragment.kind === "preview-fragment" ? "Preview" : "Final output"}</figcaption>
        </figure>)}
        {nextFragment !== undefined ? <button type="button" className="button" disabled={fragmentBusy} onClick={() => void loadFragments(selectedRecord.id, nextFragment)}>More fragments</button> : null}
      </section> : null}
    </details>
  </section>;
}
