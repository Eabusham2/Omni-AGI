import type { SparseSubstrateJobView } from "../../shared/sparseSubstrateTransfer";
import type { ActionEvent } from "../../shared/types";

export function sparseJobLabel(state: SparseSubstrateJobView["state"]): string {
  return { queued: "Queued", "boundary-required": "Waiting for current work", applying: "Updating connections", complete: "Connections updated",
    "needs-review": "Review needed", cancelled: "Stopped", failed: "Needs attention", uncertain: "Outcome unconfirmed", dormant: "Saved history" }[state];
}
export function sparseJobCanStop(state: SparseSubstrateJobView["state"]): boolean { return ["queued", "boundary-required", "applying"].includes(state); }
export function sparseJobCanPreviewRollback(job: SparseSubstrateJobView): boolean { return job.state === "complete" && Boolean(job.receipt) && job.originalBrainId === job.brainId; }

/** Actual result identity or main-owned correlation, never generated operands. */
export function sparseJobForAction(event: ActionEvent, jobs: readonly SparseSubstrateJobView[]): string | undefined {
  if (event.action.toolId !== "brain.substrate") return undefined;
  const correlated = jobs.find(job => job.brainId === event.brainId && job.originalBrainId === event.brainId && job.actionEventId === event.id);
  if (correlated) return correlated.id;
  const output = event.execution?.output;
  if (!output || typeof output !== "object" || Array.isArray(output)) return undefined;
  const row = output as Record<string, unknown>;
  return row.format === "omni-sparse-substrate-job" && row.scope === "native-sparse-assemblies" && row.brainId === event.brainId &&
    typeof row.id === "string" && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(row.id) ? row.id : undefined;
}
