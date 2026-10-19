import React, { useEffect, useId, useRef, useState } from "react";
import type { InstanceLifecycleState } from "../../shared/instanceLifecycle";
import type { InstanceLifecycleApi, InstanceLoadFit } from "../../shared/instanceLifecycleApi";
import "./instanceLifecycle.css";

const labels: Record<InstanceLifecycleState["phase"], string> = { unloaded: "Unloaded", checking: "Checking resources", loading: "Loading",
  loaded: "Loaded", suspending: "Waiting to unload", suspended: "Paused · unloaded", resuming: "Resuming saved turn", error: "Needs attention" };

export function InstanceLifecycleControls({ brainId, name, api, onState, onEditSettings, reviewRequest = 0 }: {
  brainId: string; name: string; api?: InstanceLifecycleApi; onState(state: InstanceLifecycleState): void;
  onEditSettings(): void; reviewRequest?: number;
}) {
  const [state, setState] = useState<InstanceLifecycleState>();
  const [fit, setFit] = useState<InstanceLoadFit>();
  const [open, setOpen] = useState(false), [waiting, setWaiting] = useState(false), [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const dialog = useRef<HTMLDialogElement>(null), generation = useRef(0), stateCallback = useRef(onState);
  const latestState = useRef<InstanceLifecycleState | undefined>(undefined);
  const loadRef = useRef<() => Promise<void>>(async () => undefined);
  const title = useId(); stateCallback.current = onState;
  const accept = (next: InstanceLifecycleState) => {
    if (next.brainId !== brainId || latestState.current && next.revision < latestState.current.revision) return;
    latestState.current = next; setState(next); stateCallback.current(next);
  };
  loadRef.current = async () => {
    if (!api || busy) return;
    const currentGeneration = generation.current; setBusy(true); setError("");
    try {
      const next = await api.load({ brainId, ...(state?.resume ? { expectedCheckpointId: state.resume.checkpointId } : {}) });
      if (generation.current !== currentGeneration) return;
      accept(next);
      if (next.phase === "loaded") { setWaiting(false); setOpen(false); }
      else if (next.error?.code !== "RESOURCE_FIT") setWaiting(false);
    } catch (reason) { if (generation.current === currentGeneration) { setError(reason instanceof Error ? reason.message : "Load failed."); setWaiting(false); } }
    finally { if (generation.current === currentGeneration) setBusy(false); }
  };
  useEffect(() => {
    const currentGeneration = ++generation.current; let timer: number | undefined;
    latestState.current = undefined; setState(undefined); setFit(undefined); setError(""); setWaiting(false); setOpen(false);
    if (!api) return;
    const update = (next: InstanceLifecycleState) => {
      if (currentGeneration !== generation.current || next.brainId !== brainId) return;
      if (latestState.current && next.revision < latestState.current.revision) return;
      latestState.current = next; setState(next); stateCallback.current(next);
      if (next.resumeCompatibility?.compatible === false) setWaiting(false);
    };
    const unsubscribe = api.onLifecycle(update);
    const refresh = async (initial: boolean) => {
      try {
        const next = await api.lifecycle(brainId);
        if (currentGeneration !== generation.current) return;
        update(next);
        if (initial && next.phase !== "loaded") { setOpen(true); setWaiting(next.autoLoad && next.resumeCompatibility?.compatible !== false); }
      } catch (reason) { if (currentGeneration === generation.current) setError(reason instanceof Error ? reason.message : "Lifecycle status is unavailable."); }
      finally { if (currentGeneration === generation.current) timer = window.setTimeout(() => void refresh(false), 2500); }
    };
    void refresh(true);
    return () => { generation.current += 1; unsubscribe(); if (timer !== undefined) window.clearTimeout(timer); };
  }, [api, brainId]);
  useEffect(() => { if (reviewRequest) setOpen(true); }, [reviewRequest]);
  useEffect(() => {
    const element = dialog.current;
    if (open && element && !element.open) element.showModal();
    else if (!open && element?.open) element.close();
  }, [open]);
  useEffect(() => {
    if (!api || !open) return;
    let active = true, timer: number | undefined;
    const refresh = async () => {
      try {
        const next = await api.loadFit(brainId);
        if (!active) return;
        setFit(next); setError("");
        if (waiting && next.allowed) await loadRef.current();
      } catch (reason) { if (active) setError(reason instanceof Error ? reason.message : "Live fit check is unavailable."); }
      finally { if (active) timer = window.setTimeout(() => void refresh(), 2500); }
    };
    void refresh();
    return () => { active = false; if (timer !== undefined) window.clearTimeout(timer); };
  }, [api, brainId, open, waiting]);
  const transition = state && ["checking", "loading", "resuming"].includes(state.phase);
  const unload = async () => {
    if (!api || busy) return; setBusy(true); setError("");
    const currentGeneration = generation.current;
    try { const next = await api.unload({ brainId }); if (generation.current === currentGeneration) accept(next); }
    catch (reason) { if (generation.current === currentGeneration) setError(reason instanceof Error ? reason.message : "Unload failed."); }
    finally { if (generation.current === currentGeneration) setBusy(false); }
  };
  const close = () => { setWaiting(false); setOpen(false); };
  const stopSaved = async () => {
    if (!api || !state || busy) return;
    const inherited = state.inheritedExecutions?.entries.find(value => ["queued", "suspended"].includes(value.status));
    const turnId = state.resume?.turnId ?? inherited?.turnId;
    if (!turnId) return;
    const currentGeneration = generation.current; setBusy(true); setWaiting(false); setError("");
    try {
      const next = await api.stopSavedTurn({ brainId, turnId, ...(state.resume ? { checkpointId: state.resume.checkpointId } : { executionId: inherited!.id }) });
      if (generation.current === currentGeneration) accept(next);
    } catch (reason) { if (generation.current === currentGeneration) setError(reason instanceof Error ? reason.message : "Saved-turn Stop was not acknowledged."); }
    finally { if (generation.current === currentGeneration) setBusy(false); }
  };
  return <div className="instance-lifecycle-controls">
    <span role="status">{state ? state.phase === "unloaded" && state.learnedRuntimeResident ? "Unload incomplete · work still active" : labels[state.phase] : api ? "Checking instance…" : "Design preview"}</span>
    <button type="button" className="button" disabled={!api || busy || Boolean(transition)} onClick={() => state?.phase === "loaded" || state?.phase === "suspending" ? void unload() : setOpen(true)}>
      {state?.phase === "loaded" ? "Unload" : state?.phase === "suspending" ? "Retry unload" : "Load / review fit"}
    </button>
    {state?.waitingFor ? <small>{state.waitingFor.label}</small> : null}
    {error && !open ? <small role="alert">{error}</small> : null}
    <dialog ref={dialog} className="instance-load-dialog" aria-labelledby={title} onCancel={(event) => { event.preventDefault(); close(); }}>
      <div className="instance-load-dialog__heading"><h2 id={title}>Load {name}</h2><button type="button" className="button" aria-label="Close load review" onClick={close}>Close</button></div>
      <p>Load this brain to start chatting. Its saved state and your memory settings are checked first.</p>
      {state?.resume ? <p className="instance-load-receipt">Saved turn <code>{state.resume.turnId}</code> · {state.resume.publishedTokens.toLocaleString()} published tokens. {state.resumeCompatibility?.compatible === false ? "This runtime cannot continue that exact checkpoint." : <>Supported execution resumes from checkpoint <code>{state.resume.checkpointId}</code>; completed effects are not replayed.</>}</p> : null}
      {state?.resumeCompatibility?.compatible === false ? <p role="alert">{state.resumeCompatibility.reason} Your saved state is preserved. Use its original compatible runtime or explicitly stop the saved turn.</p> : null}
      {state?.inheritedExecutions?.pending ? <p className="instance-load-receipt">{state.inheritedExecutions.pending.toLocaleString()} saved {state.inheritedExecutions.pending === 1 ? "turn" : "turns"} from the merged instances. Load continues them in order before new neural work; completed actions are not replayed.</p> : null}
      <p role="status">{busy ? state?.waitingFor?.label ?? "Waiting for the brain to finish loading or unloading…" : waiting ? "Waiting for resources. This check updates while the review stays open." : fit?.allowed ? "Ready to load with these settings." : "Checking whether these settings fit your device."}</p>
      {fit ? <><p>Active execution needs at least {fit.minimumContextTokens.toLocaleString()} context tokens. Settings cannot reduce below that frontier.</p>
        {fit.blockers.length ? <ul>{fit.blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}</ul> : null}
        <small>Checked {new Date(fit.checkedAt).toLocaleTimeString()}. The native allocation check remains authoritative.</small></> : null}
      {state?.reconnect.length ? <section><h3>Reconnect explicitly after Load</h3><ul>{state.reconnect.map((item) => <li key={item.id}>{item.label}: {item.reason}{item.permissionRequired ? " Permission is required." : ""}</li>)}</ul></section> : null}
      {error || state?.error ? <p role="alert">{error || state?.error?.message}</p> : null}
      <label className="instance-load-auto"><input type="checkbox" checked={state?.autoLoad ?? false} disabled={!state || busy}
        onChange={(event) => { const currentGeneration = generation.current; if (api) void api.setAutoLoad({ brainId, enabled: event.target.checked })
          .then((next) => { if (generation.current === currentGeneration) accept(next); })
          .catch((reason: unknown) => { if (generation.current === currentGeneration) setError(reason instanceof Error ? reason.message : "Preference save failed."); }); }} /> Auto-load when opening this instance (off by default)</label>
      <div className="instance-load-dialog__actions">
        {state?.resumeCompatibility?.compatible === false ? <button type="button" className="button" disabled={busy} onClick={() => void stopSaved()}>Stop saved work</button> : null}
        <button type="button" className="button button--primary" disabled={!api || busy || Boolean(transition) || state?.resumeCompatibility?.compatible === false} onClick={() => { setWaiting(true); if (fit?.allowed) void loadRef.current(); }}>
          {state?.resume ? "Load and continue same turn" : state?.inheritedExecutions?.pending ? "Load and continue saved turns" : fit?.allowed ? "Load instance" : "Wait for fit and load"}
        </button>
        <button type="button" className="button" disabled={busy} onClick={() => { close(); onEditSettings(); }}>Edit resource settings</button>
        {waiting ? <button type="button" className="button" onClick={() => setWaiting(false)}>Pause fit waiting</button> : null}
      </div>
    </dialog>
  </div>;
}
