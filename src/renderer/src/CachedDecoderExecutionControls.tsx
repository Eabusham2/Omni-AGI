import { useEffect, useRef, useState } from "react";
import type { CachedDecoderExecutionApi, CachedDecoderExecutionPrepared, CachedDecoderExecutionStatus } from "../../shared/cachedDecoderExecution";
import "./decoderExecutionControls.css";

export function cachedDecoderAction(phase: CachedDecoderExecutionStatus["phase"] | undefined, reviewed: boolean) {
  if (phase === "awaiting-retained-frontier" || phase === "calibrating")
    return { kind: "wait", label: phase === "calibrating" ? "Comparing cached computation…" : "Waiting for a normal reply" } as const;
  if (phase === "prepared" && reviewed) return { kind: "arm", label: "Compare one step in the next reply" } as const;
  return { kind: "prepare", label: "Review devices for cached computation" } as const;
}
export function cachedDecoderStatusText(status: CachedDecoderExecutionStatus | undefined): string {
  if (!status || status.phase === "idle") return "No retained-context comparison requested.";
  if (status.phase === "prepared") return "Device review ready.";
  if (status.phase === "awaiting-retained-frontier") return "Ready to compare one cached step during a normal reply.";
  if (status.phase === "calibrating") return "Comparing the current cached computation.";
  if (status.phase === "cancelled") return "Comparison stopped; no future comparison is armed.";
  if (status.phase === "failed-fallback") return "Comparison was not adopted. Inspect details if needed.";
  if (status.phase === "unsupported") return "This loaded instance does not support this comparison.";
  return status.actualOperationConsumed && status.exactCacheAndSamplingMatched
    ? "One matching cached step completed." : "Comparison finished without adopting a different route.";
}
export function cachedDecoderDeviceLabel(device: string): string {
  return device === "cpu" ? "CPU" : device === "mps" ? "Metal" : device.startsWith("cuda:") ? `CUDA device ${device.slice(5)}` : device;
}

export function CachedDecoderExecutionControls(props: { api?: CachedDecoderExecutionApi; brainId: string; loaded: boolean;
  foregroundTurnActive: boolean; dirty: boolean; disabled?: boolean }) {
  return <CachedDecoderExecutionOwner key={props.brainId} {...props} />;
}
function CachedDecoderExecutionOwner({ api, brainId, loaded, foregroundTurnActive, dirty, disabled = false }: Parameters<typeof CachedDecoderExecutionControls>[0]) {
  const [open, setOpen] = useState(false), [status, setStatus] = useState<CachedDecoderExecutionStatus>();
  const [review, setReview] = useState<CachedDecoderExecutionPrepared>(), [busy, setBusy] = useState(false);
  const [error, setError] = useState(""), [cudaFallback, setCudaFallback] = useState(false);
  const owner = useRef<object | null>(null), pending = useRef<object | null>(null), stopping = useRef(false), revision = useRef(0);
  useEffect(() => {
    const scope = {}; owner.current = scope; pending.current = null; setStatus(undefined); setReview(undefined); setBusy(false); setError("");
    return () => { owner.current = null; revision.current++; };
  }, [api, brainId]);
  useEffect(() => {
    revision.current++; setReview(undefined);
    if (!loaded) setStatus(undefined);
  }, [loaded, dirty]);
  useEffect(() => {
    if (!api || !loaded || !open) return;
    const scope = owner.current; let retired = false, timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      const observedRevision = revision.current;
      try {
        const value = await api.status(brainId);
        if (!retired && owner.current === scope && revision.current === observedRevision) {
          if (value.ownerId !== brainId) throw new Error("The cached computation owner changed.");
          setStatus(value);
        }
      } catch (reason) {
        if (!retired && owner.current === scope && revision.current === observedRevision)
          setError(reason instanceof Error ? reason.message : "Cached-computation status is unavailable.");
      } finally { if (!retired && owner.current === scope) timer = setTimeout(() => { void poll(); }, 1000); }
    };
    void poll(); return () => { retired = true; if (timer) clearTimeout(timer); };
  }, [api, brainId, loaded, open]);
  const run = async (kind: "prepare" | "arm") => {
    if (!api || !loaded || disabled || dirty || foregroundTurnActive || pending.current || kind === "arm" && !review) return;
    const scope = owner.current, request = {}, requestRevision = ++revision.current;
    pending.current = request; setBusy(true); setError("");
    if (kind === "prepare") { setStatus(undefined); setReview(undefined); }
    try {
      const result = kind === "prepare"
        ? await api.prepare({ brainId, ...(cudaFallback ? { acknowledgeCudaCpuTileFallback: true } : {}) })
        : await api.arm({ brainId, reviewId: review!.reviewId });
      if (owner.current !== scope || pending.current !== request || revision.current !== requestRevision) return;
      const value = "status" in result ? result.status : result;
      if (value.ownerId !== brainId) throw new Error("The cached computation result belongs to another instance.");
      revision.current++; setStatus(value); setReview("reviewId" in result ? result : undefined);
    } catch (reason) {
      if (owner.current === scope && revision.current === requestRevision)
        setError(reason instanceof Error ? reason.message : "The cached comparison could not start.");
    } finally { if (pending.current === request) pending.current = null; if (owner.current === scope) setBusy(false); }
  };
  const stop = async () => {
    if (!api || !loaded || stopping.current) return;
    const scope = owner.current, stopRevision = ++revision.current; stopping.current = true; setError(""); setReview(undefined);
    try {
      const value = await api.stop({ brainId, ...(status?.operationId ? { operationId: status.operationId } : {}) });
      if (owner.current === scope && revision.current === stopRevision) {
        if (value.ownerId !== brainId) throw new Error("The Stop receipt belongs to another instance.");
        revision.current++; setStatus(value);
      }
    } catch (reason) {
      if (owner.current === scope && revision.current === stopRevision)
        setError(reason instanceof Error ? reason.message : "Stop has not been acknowledged.");
    } finally { stopping.current = false; }
  };
  const next = cachedDecoderAction(status?.phase, Boolean(review)), progress = status?.progress;
  const blocked = !api || !loaded || disabled || dirty || busy || foregroundTurnActive || next.kind === "wait";
  const stoppable = busy || status && ["prepared", "awaiting-retained-frontier", "calibrating"].includes(status.phase);
  return <details className="decoder-execution" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>Advanced · retained-context device comparison</summary>
    <p>Compare one real cached step without clearing working memory. This review applies only to that step, not a permanent whole-brain placement.</p>
    {!loaded ? <p>Load this instance before reviewing devices.</p> : null}
    {dirty ? <p>Apply resource settings before preparing a comparison.</p> : null}
    <label className="decoder-execution__choice"><input type="checkbox" checked={cudaFallback} disabled={busy || next.kind === "wait" || disabled}
      onChange={event => { setCudaFallback(event.target.checked); setReview(undefined); }} />Include CUDA fallback (CPU tiles and transfers)</label>
    <div className="decoder-execution__actions">
      <button type="button" className="button" disabled={blocked} onClick={() => { if (next.kind !== "wait") void run(next.kind); }}>{next.label}</button>
      <button type="button" className="button" disabled={!api || !loaded || !stoppable} onClick={() => void stop()}>Stop comparison</button>
    </div>
    <p role="status">{busy && !progress ? "Waiting for the owned execution boundary…" : cachedDecoderStatusText(status)}</p>
    {progress ? <p>{progress.stage}{progress.candidate > 0 && progress.candidates > 0 ? ` · device ${progress.candidate} of ${progress.candidates}` : ""}
      {progress.sample > 0 && progress.samplesPerCandidate > 0 ? ` · sample ${progress.sample} of ${progress.samplesPerCandidate}` : ""}</p> : null}
    {review ? <ul>{review.candidates.map(candidate => <li key={candidate.planSha256}>{cachedDecoderDeviceLabel(candidate.executionDevice)}
      {candidate.backendMode === "explicit-cuda-cpu-tile" ? " · CPU tiles and transfers" : ""}{candidate.supported ? "" : " · unavailable"}</li>)}</ul> : null}
    {status?.selectedCandidate && status.actualOperationConsumed && status.exactCacheAndSamplingMatched
      ? <p>Matched step: {cachedDecoderDeviceLabel(status.selectedCandidate.executionDevice)}</p> : null}
    {status ? <details><summary>Comparison details</summary><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{JSON.stringify(status, null, 2)}</pre></details> : null}
    {error ? <p role="alert">{error}</p> : null}
  </details>;
}
