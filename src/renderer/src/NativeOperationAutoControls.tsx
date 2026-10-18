import { useEffect, useRef, useState } from "react";
import type { NativeOperationAutoApi, NativeOperationAutoPrepared, NativeOperationAutoStatus, NativeOperationAutoPlanPage, NativeOperationAutoWorkKind } from "../../shared/nativeOperationAuto";
import type { NativeOperationAutoCandidate } from "../../shared/nativeMixedRegions";
import { cachedDecoderDeviceLabel } from "./CachedDecoderExecutionControls";
import "./decoderExecutionControls.css";

const WORK_KIND_LABELS = {
  "native.perception.image-embedding": "Image perception",
  "native.perception.audio-embedding": "Audio perception",
  "native.perception.video-embedding": "Video perception",
  "native.decoder.cached-whole-step-sampling": "Decoder step",
  "native.connected-decoder.cached-whole-step-sampling": "Connected decoder step",
  "native.media.image-artifact": "Image artifact",
  "native.media.audio-artifact": "Audio artifact",
  "native.media.video-artifact": "Video artifact",
  "native.training.prepared-whole-wave": "Prepared training wave",
  "native.decoder.full-prefix-sampling": "Full decoder prefix",
  "native.connected-decoder.full-prefix-sampling": "Full connected prefix",
  "native.decoder.uncached-full-forward": "Full decoder forward",
  "native.connected-decoder.uncached-full-forward": "Full connected forward"
} satisfies Record<NativeOperationAutoWorkKind, string>;
export function nativeAutoWorkKindLabel(workKind: NativeOperationAutoWorkKind): string { return WORK_KIND_LABELS[workKind]; }
export function nativeAutoCandidateLabel(candidate: NativeOperationAutoCandidate): string {
  if (candidate.backendMode !== "native-mixed-sequential") return cachedDecoderDeviceLabel(candidate.executionDevice);
  return "Mixed sequential · " + [...new Set([candidate.executionDevice, ...candidate.regionSchedule.segments.map(row => row.candidate.executionDevice)])]
    .map(cachedDecoderDeviceLabel).join(" / ");
}
export const NATIVE_AUTO_REUSE_SCOPE_TEXT = "Qualification uses the exact gate for each registered operation. Perception, retained decoder-step and media routes may reuse a validated layout for new input. Each full decoder prefix is checked separately, with no cross-prefix reuse; each complete forward is checked separately, with no cross-input reuse; each prepared training wave is checked separately, with no cross-wave reuse.";
export const NATIVE_AUTO_MEASUREMENT_SCOPE_TEXT = "Only paths supported by the current native owner are included. Full training pipelines, distributed training, whole-brain and complete-reply placement are not covered. These comparisons do not establish a total speedup.";

export function nativeAutoStatusText(status?: NativeOperationAutoStatus): string {
  if (!status || status.phase === "idle") return "Automatic operation placement is off.";
  if (status.phase === "prepared") return "Device review ready; Auto is not enabled yet.";
  if (status.phase === "awaiting-operation") return "Waiting for a supported normal operation.";
  const prefill = status.workKind === "native.decoder.full-prefix-sampling" || status.workKind === "native.connected-decoder.full-prefix-sampling";
  const forward = status.workKind === "native.decoder.uncached-full-forward" || status.workKind === "native.connected-decoder.uncached-full-forward";
  if (status.phase === "qualifying" && forward) return "Comparing one complete decoder forward.";
  if (status.phase === "qualifying" && prefill) return "Comparing one actual full decoder prefix.";
  if (status.phase === "qualifying") return status.workKind === "native.training.prepared-whole-wave"
    ? "Checking and applying one prepared training wave." : "Comparing an actual supported operation.";
  if (status.phase === "active" && status.workKind === "native.training.prepared-whole-wave")
    return "Prepared training-wave route applied; future waves need a new check.";
  if (status.phase === "active" && prefill) return "Full-prefix route applied; another prefix needs a new check.";
  if (status.phase === "active" && forward) return "Complete forward returned; another input needs a new check.";
  if (status.phase === "active") return `${status.qualifiedPlanCount.toLocaleString()} qualified operation route${status.qualifiedPlanCount === 1 ? "" : "s"}.`;
  if (status.phase === "cancelled") return "Auto stopped; ordinary processing continues.";
  if (status.phase === "failed-fallback") return "This operation kept its ordinary route. Inspect details if needed.";
  return "This loaded instance does not support the requested operation route.";
}
export function nativeAutoPendingText(status: NativeOperationAutoStatus | undefined, busy: boolean): string {
  return busy ? status?.phase === "cancelled" ? "Auto stopped. The prior metadata review is settling; ordinary processing is unchanged."
    : "Waiting for the owned execution boundary…" : nativeAutoStatusText(status);
}
export function NativeOperationAutoControls(props: { api?: NativeOperationAutoApi; brainId: string; loaded: boolean;
  foregroundTurnActive: boolean; dirty: boolean; disabled?: boolean }) {
  return <NativeOperationAutoOwner key={props.brainId} {...props} />;
}
function NativeOperationAutoOwner({ api, brainId, loaded, foregroundTurnActive, dirty, disabled }: Parameters<typeof NativeOperationAutoControls>[0]) {
  const [open, setOpen] = useState(false), [status, setStatus] = useState<NativeOperationAutoStatus>();
  const [review, setReview] = useState<NativeOperationAutoPrepared>(), [page, setPage] = useState<NativeOperationAutoPlanPage>();
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [cuda, setCuda] = useState(false);
  const scope = useRef<object | null>(null), revision = useRef(0), pending = useRef<object | null>(null), stopping = useRef(false);
  useEffect(() => {
    const owner = {}; scope.current = owner; revision.current++; pending.current = null;
    setReview(undefined); setStatus(undefined); setPage(undefined); setBusy(false); setError("");
    return () => { scope.current = null; pending.current = null; revision.current++; };
  }, [api, brainId]);
  useEffect(() => { revision.current++; setReview(undefined); setPage(undefined); if (!loaded) setStatus(undefined); }, [loaded, dirty]);
  useEffect(() => {
    if (!api || !loaded || !open) return;
    const owner = scope.current; let retired = false, timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      const generation = revision.current;
      try {
        const value = await api.status(brainId);
        if (!retired && scope.current === owner && revision.current === generation) {
          if (value.ownerId !== brainId) throw new Error("Operation status belongs to another instance.");
          setStatus(value);
          setPage(previous => previous?.ownerSessionId === value.ownerSessionId ? previous : undefined);
        }
      } catch (reason) { if (!retired && scope.current === owner && revision.current === generation) setError(String(reason)); }
      finally { if (!retired && scope.current === owner) timer = setTimeout(() => { void poll(); }, 1000); }
    };
    void poll(); return () => { retired = true; if (timer) clearTimeout(timer); };
  }, [api, brainId, loaded, open]);
  const run = async (kind: "prepare" | "arm" | "plans", after?: string) => {
    if (!api || !loaded || pending.current || kind !== "plans" && (disabled || dirty || foregroundTurnActive) || kind === "arm" && !review) return;
    const owner = scope.current, generation = ++revision.current, flight = {}; pending.current = flight; setBusy(true); setError("");
    if (kind === "prepare") { setReview(undefined); setStatus(undefined); setPage(undefined); }
    try {
      const value = kind === "prepare" ? await api.prepare({ brainId, ...(cuda ? { acknowledgeCudaCpuTileFallback: true } : {}) })
        : kind === "arm" ? await api.arm({ brainId, reviewId: review!.reviewId }) : await api.plans({ brainId, limit: 16, ...(after ? { after } : {}) });
      if (scope.current !== owner || revision.current !== generation) return;
      if (value.ownerId !== brainId) throw new Error("Operation control returned another instance.");
      if ("plans" in value) {
        if (status && value.ownerSessionId !== status.ownerSessionId) throw new Error("The loaded operation process changed; refresh its status.");
        setPage(value);
      } else { setStatus("status" in value ? value.status : value); setReview("reviewId" in value ? value : undefined); }
    } catch (reason) { if (scope.current === owner && revision.current === generation) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (pending.current === flight) { pending.current = null; if (scope.current === owner) setBusy(false); } }
  };
  const stop = async () => {
    if (!api || !loaded || stopping.current) return;
    const owner = scope.current, generation = ++revision.current; stopping.current = true; setReview(undefined); setError("");
    try {
      const value = await api.stop({ brainId, ...(status?.operationId ? { operationId: status.operationId } : {}) });
      if (scope.current !== owner || revision.current !== generation) return;
      if (value.ownerId !== brainId) throw new Error("Stop belongs to another instance.");
      setStatus(value); setPage(undefined);
    } catch (reason) { if (scope.current === owner && revision.current === generation) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { stopping.current = false; }
  };
  const active = status && ["awaiting-operation", "qualifying", "active", "failed-fallback"].includes(status.phase);
  const blocked = !api || !loaded || disabled || dirty || busy || foregroundTurnActive || Boolean(active);
  return <details className="decoder-execution" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>Advanced · automatic operation placement</summary>
    <p>Compare complete supported paths during normal use. Route reuse depends on the operation; opening this panel does not run a benchmark.</p>
    {!loaded ? <p>Load this instance before reviewing devices.</p> : null}
    {dirty ? <p>Apply resource settings before reviewing devices.</p> : null}
    <label className="decoder-execution__choice"><input type="checkbox" checked={cuda} disabled={busy || Boolean(active) || disabled}
      onChange={event => { setCuda(event.target.checked); setReview(undefined); }} />Include CUDA fallback (CPU tiles and transfers)</label>
    <div className="decoder-execution__actions">
      <button type="button" className="button" disabled={blocked} onClick={() => void run(review && status?.phase === "prepared" ? "arm" : "prepare")}>
        {review && status?.phase === "prepared" ? "Enable Auto for future operations" : "Review devices"}</button>
      <button type="button" className="button" disabled={!api || !loaded || status?.phase === "cancelled" || !busy && !active && !review} onClick={() => void stop()}>Stop Auto</button>
      <button type="button" className="button" disabled={!api || !loaded || busy} onClick={() => void run("plans")}>Inspect qualified routes</button>
    </div>
    <p role="status">{nativeAutoPendingText(status, busy)}</p>
    {status?.progress ? <p>{status.progress.stage} · sample {status.progress.sample} of {status.progress.samplesPerCandidate}</p> : null}
    {review ? <ul>{review.candidates.map(candidate => <li key={candidate.planSha256}>{cachedDecoderDeviceLabel(candidate.executionDevice)}
      {candidate.backendMode === "explicit-cuda-cpu-tile" ? " · CPU tiles and transfers" : ""}{candidate.supported ? "" : " · unavailable"}</li>)}</ul> : null}
    {page ? <section aria-label="Qualified operation routes"><p>{page.plans.length.toLocaleString()} displayed of {page.total.toLocaleString()} qualified routes</p>
      <ul>{page.plans.map(plan => <li key={plan.id}>{nativeAutoWorkKindLabel(plan.workKind)} · {nativeAutoCandidateLabel(plan.candidate)} · {plan.qualifiedSamples} matched samples{plan.workKind === "native.training.prepared-whole-wave" ? " · this wave only" : plan.workKind === "native.decoder.full-prefix-sampling" || plan.workKind === "native.connected-decoder.full-prefix-sampling" ? " · this prefix only" : plan.workKind === "native.decoder.uncached-full-forward" || plan.workKind === "native.connected-decoder.uncached-full-forward" ? " · this input only" : ""}</li>)}</ul>
      {page.nextAfter ? <button type="button" className="button" disabled={busy} onClick={() => void run("plans", page.nextAfter)}>Next routes</button> : null}
    </section> : null}
    {status ? <details><summary>Measurement scope</summary><p>{NATIVE_AUTO_REUSE_SCOPE_TEXT}</p>
      <p>{NATIVE_AUTO_MEASUREMENT_SCOPE_TEXT}</p>
      <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{JSON.stringify(status, null, 2)}</pre></details> : null}
    {error ? <p role="alert">{error}</p> : null}
  </details>;
}
