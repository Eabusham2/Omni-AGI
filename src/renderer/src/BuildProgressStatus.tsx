import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { BrainStorageOperationEvent, BuildProgressEvent } from "../../shared/types";
import { buildProgressTimingPresentation, storageOperationTimingPresentation } from "./buildProgressPresentation";

function useMonotonicEventAge(event: object, active: boolean): number {
  const observedAt = useRef(performance.now());
  const [clock, setClock] = useState(() => observedAt.current);
  useLayoutEffect(() => { observedAt.current = performance.now(); setClock(observedAt.current); }, [event]);
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setClock(performance.now()), 250);
    return () => window.clearInterval(timer);
  }, [active, event]);
  return active ? Math.max(0, clock - observedAt.current) : 0;
}

/** Current host phase/timing only; no percentage-derived duration or capability claims. */
export function BuildProgressStatus({ progress }: { progress: BuildProgressEvent }) {
  const sinceUpdateMs = useMonotonicEventAge(progress, progress.phase !== "complete");
  const value = buildProgressTimingPresentation({ ...progress, sinceUpdateMs });
  return <>
    <span>{value.percent}%</span>
    <i className="build-readiness-overlay__meter" role="progressbar" aria-label="Core build progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={value.percent}>
      <b style={{ width: `${value.percent}%` }} />
    </i>
    <dl className="build-progress-timing" aria-live="off">
      <div><dt>Elapsed</dt><dd>{value.elapsedText}</dd></div>
      <div><dt>Remaining</dt><dd>{value.remainingText}</dd></div>
      {value.totalText ? <div><dt>{value.terminal ? "Total" : "Estimated total"}</dt><dd>{value.totalText}</dd></div> : null}
    </dl>
  </>;
}

export function StorageOperationTiming({ operation }: { operation: BrainStorageOperationEvent }) {
  const active = ["queued", "running", "cancelling"].includes(operation.state);
  const sinceUpdateMs = useMonotonicEventAge(operation, active);
  const value = storageOperationTimingPresentation({ ...operation, sinceUpdateMs });
  return <>
    <div className="storage-operation-timing"><dt>Elapsed</dt><dd aria-live="off">{value.elapsedText}</dd></div>
    <div className="storage-operation-timing"><dt>Remaining</dt><dd aria-live="off">{value.remainingText}</dd></div>
    {value.totalText ? <div className="storage-operation-timing"><dt>{value.terminal ? "Total" : "Estimated total"}</dt><dd aria-live="off">{value.totalText}</dd></div> : null}
  </>;
}
