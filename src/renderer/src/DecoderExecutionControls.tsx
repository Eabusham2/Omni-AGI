import { useEffect, useRef, useState } from "react";
import type { DecoderExecutionApi, DecoderExecutionPhase, DecoderExecutionPrepared, DecoderExecutionStatus } from "../../shared/decoderExecution";
import "./decoderExecutionControls.css";

export function decoderExecutionNextAction(phase: DecoderExecutionPhase | undefined, reviewed: boolean) {
  if (phase === "awaiting-normal-input") return { kind: "wait", label: "Waiting for your next turn" } as const;
  if (phase === "calibrating") return { kind: "wait", label: "Measuring…" } as const;
  if (phase === "captured" || phase === "prepared" && !reviewed) return { kind: "prepare", label: "Review device comparison" } as const;
  if (phase === "prepared" && reviewed) return { kind: "calibrate", label: "Measure devices" } as const;
  if (phase === "proposal") return { kind: "apply", label: "Use measured prefill route" } as const;
  return { kind: "capture", label: "Use next normal turn" } as const;
}
export function decoderMilliseconds(nanoseconds: string): string {
  if (!/^[0-9]{1,25}$/.test(nanoseconds)) return "unavailable";
  const value = BigInt(nanoseconds);
  return `${(value / 1_000_000n).toLocaleString()}.${String(value % 1_000_000n).padStart(6, "0").slice(0, 3)} ms`;
}

export function DecoderExecutionControls(props: { api?: DecoderExecutionApi; brainId: string; loaded: boolean;
  foregroundTurnActive: boolean; dirty: boolean; disabled?: boolean }) {
  return <DecoderExecutionOwner key={props.brainId} {...props} />;
}
function DecoderExecutionOwner({ api, brainId, loaded, foregroundTurnActive, dirty, disabled = false }: Parameters<typeof DecoderExecutionControls>[0]) {
  const [open, setOpen] = useState(false), [status, setStatus] = useState<DecoderExecutionStatus>();
  const [review, setReview] = useState<DecoderExecutionPrepared>(), [busy, setBusy] = useState(false);
  const [error, setError] = useState(""), [cudaFallback, setCudaFallback] = useState(false);
  const owner = useRef<object | null>(null), request = useRef<object | null>(null), stopping = useRef(false);
  const actionRevision = useRef(0);
  useEffect(() => { const scope = {}; owner.current = scope; request.current = null; setBusy(false); setStatus(undefined); setReview(undefined);
    return () => { owner.current = null; actionRevision.current++; }; }, [api, brainId]);
  useEffect(() => {
    if (!open || !loaded || !api) return;
    let closed = false; let timer: ReturnType<typeof setTimeout> | undefined;
    const scope = owner.current;
    const poll = async () => {
      const revision = actionRevision.current;
      try {
        const value = await api.status(brainId);
        if (!closed && owner.current === scope && revision === actionRevision.current) {
          if (value.ownerId !== brainId) throw new Error("The measured decoder owner changed.");
          setStatus(value);
        }
      } catch (reason) { if (!closed && owner.current === scope && revision === actionRevision.current) setError(reason instanceof Error ? reason.message : "Device status is unavailable."); }
      finally { if (!closed && owner.current === scope) timer = setTimeout(() => { void poll(); }, 1000); }
    };
    void poll(); return () => { closed = true; if (timer) clearTimeout(timer); };
  }, [open, loaded, api, brainId]);
  useEffect(() => { actionRevision.current++; if (!loaded || dirty) setReview(undefined); if (!loaded) setStatus(undefined); }, [loaded, dirty]);
  const run = async (action: () => Promise<DecoderExecutionStatus | DecoderExecutionPrepared | { status: DecoderExecutionStatus }>) => {
    if (!api || !loaded || request.current) return;
    const scope = owner.current, pending = {}, revision = ++actionRevision.current; request.current = pending; setBusy(true); setError("");
    try {
      const result = await action();
      if (owner.current !== scope || request.current !== pending || actionRevision.current !== revision) return;
      const value = "status" in result ? result.status : result;
      if (value.ownerId !== brainId) throw new Error("The decoder result belongs to another instance.");
      actionRevision.current++;
      setStatus(value);
      if ("reviewId" in result) setReview(result);
    } catch (reason) { if (owner.current === scope && actionRevision.current === revision) setError(reason instanceof Error ? reason.message : "Device measurement failed."); }
    finally { if (request.current === pending) request.current = null; if (owner.current === scope) setBusy(false); }
  };
  const stop = async () => {
    if (!api || !loaded || stopping.current) return;
    const scope = owner.current, revision = ++actionRevision.current; stopping.current = true; setError("");
    try {
      const value = await api.stop({ brainId, ...(status?.operationId ? { operationId: status.operationId } : {}) });
      if (owner.current === scope && actionRevision.current === revision) { actionRevision.current++; setStatus(value); setReview(undefined); }
    } catch (reason) { if (owner.current === scope && actionRevision.current === revision) setError(reason instanceof Error ? reason.message : "Stop has not been acknowledged."); }
    finally { stopping.current = false; }
  };
  const next = decoderExecutionNextAction(status?.phase, Boolean(review));
  const blocked = !api || !loaded || busy || disabled || dirty || foregroundTurnActive || next.kind === "wait";
  const primary = () => {
    if (blocked || !api) return;
    if (next.kind === "capture") { setReview(undefined); void run(() => api.captureNext({ brainId, ...(cudaFallback ? { acknowledgeCudaCpuTileFallback: true } : {}) })); }
    else if (next.kind === "prepare") void run(() => api.prepare({ brainId }));
    else if (next.kind === "calibrate" && review) void run(() => api.calibrate({ brainId, reviewId: review.reviewId }));
    else if (next.kind === "apply" && status?.proposal) void run(() => api.apply({ brainId,
      proposalId: status.proposal!.proposalId, proposalSha256: status.proposal!.proposalSha256 }));
  };
  const progress = status?.progress;
  return <details className="decoder-execution" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>Advanced · decoder device measurements</summary>
    <p>Compare the next normal turn’s isolated prefill operation on admitted devices. No training or generated quiz; this does not measure the whole brain, media, or cached continuation.</p>
    {!loaded ? <p>Load this instance before measuring.</p> : null}
    {dirty ? <p>Apply the resource settings before preparing a comparison.</p> : null}
    <label className="decoder-execution__choice"><input type="checkbox" checked={cudaFallback} disabled={busy || disabled}
      onChange={event => setCudaFallback(event.target.checked)} />Include CUDA fallback (packed CPU computation)</label>
    <div className="decoder-execution__actions">
      <button type="button" className="button" onClick={primary} disabled={blocked}>{next.label}</button>
      <button type="button" className="button" onClick={() => void run(() => api!.status(brainId))} disabled={!api || !loaded || busy}>Refresh status</button>
      <button type="button" className="button" onClick={() => void stop()} disabled={!api || !loaded || !status || ["idle", "cancelled", "failed", "unsupported"].includes(status.phase)}>Stop measurement</button>
    </div>
    <p role="status">{busy && !progress ? "Waiting for the owned execution boundary…" : status?.reason || "No measurement requested."}</p>
    {progress ? <p>{progress.stage}{progress.candidate > 0 && progress.candidates > 0 ? ` · device ${progress.candidate} of ${progress.candidates}` : ""}
      {progress.sample > 0 && progress.samplesPerCandidate > 0 ? ` · sample ${progress.sample} of ${progress.samplesPerCandidate}` : ""}</p> : null}
    {review ? <ul>{review.candidates.map(candidate => <li key={candidate.planSha256}>{candidate.executionDevice} · {candidate.backendMode}</li>)}</ul> : null}
    {status?.proposal ? <p title={`Baseline ${status.proposal.baselineMeanNanoseconds} ns; candidate ${status.proposal.candidateMeanNanoseconds} ns`}>
      Measured baseline {decoderMilliseconds(status.proposal.baselineMeanNanoseconds)} · {status.proposal.candidate.executionDevice} {decoderMilliseconds(status.proposal.candidateMeanNanoseconds)} · {status.proposal.minimumSamples} matched samples. Exact captured operation only.</p> : null}
    {error ? <p role="alert">{error}</p> : null}
  </details>;
}
