import React from "react";
import type { CognitionProgress } from "../../shared/cognition";

export function CognitionProgressView({ progress }: { progress: CognitionProgress }) {
  const label = progress.phase === "ponder" ? "Ponder" : "Best Work";
  const state = progress.active ? "In progress" : progress.stopReason === "resource-pressure" ? "Paused for resources" :
    progress.stopReason === "steered" ? "Steered" : progress.stopReason === "cancelled" ? "Stopped" : "Finished";
  return <details className="cognition-progress">
    <summary>{label}<span>{state}</span></summary>
    <dl>
      {progress.passes !== undefined ? <div><dt>Recurrent passes</dt><dd>{progress.passes.toLocaleString()}</dd></div> : null}
      {progress.completedBranches !== undefined && progress.admittedBranches !== undefined ?
        <div><dt>Branches finished</dt><dd>{progress.completedBranches.toLocaleString()} of {progress.admittedBranches.toLocaleString()}</dd></div> : null}
      {progress.elapsedMs !== undefined ? <div><dt>Measured elapsed</dt><dd>{(progress.elapsedMs / 1000).toFixed(1)}s</dd></div> : null}
      {progress.delta !== undefined ? <div><dt>State change</dt><dd>{progress.delta.toPrecision(3)}</dd></div> : null}
      {progress.selectedAction ? <div><dt>Selected activity</dt><dd>{progress.selectedAction}</dd></div> : null}
      {progress.concurrency ? <div><dt>Execution</dt><dd>Interleaved on this brain</dd></div> : null}
      {!progress.active && progress.stopReason ? <div><dt>End reason</dt><dd>{progress.stopReason.replaceAll("-", " ")}</dd></div> : null}
    </dl>
  </details>;
}
