import { useState } from "react";
import type { ToolExecutionResult } from "../../shared/types";
import "./externalAdvice.css";

export function ActionReviewDetails({ execution, onDecline }: {
  execution?: ToolExecutionResult; onDecline?(instruction?: string): Promise<void>;
}) {
  const [instruction, setInstruction] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const review = execution?.actionReview;
  if (!review || review.kind !== "action-review" || review.formatVersion !== 1) return null;
  const decline = async (draft?: string) => {
    if (!onDecline || busy) return; setBusy(true); setError("");
    try { await onDecline(draft); } catch (reason) { setError(reason instanceof Error ? reason.message : "This action could not be declined."); }
    finally { setBusy(false); }
  };
  return <section className="action-review-details" aria-label="Separate action review">
    <strong>{review.verdict === "ask" ? "Reviewer asks for your decision" : "Reviewer found no reason to ask"}</strong>
    <p>{review.summary}</p>
    <small>{review.disposition} · {(review.elapsedMs / 1000).toFixed(1)} s · advisory, not permission</small>
    <details><summary>Exact proposed action and observed evidence</summary>
      <pre tabIndex={0}>{JSON.stringify({ proposed: review.proposedInvocation, resolved: review.resolvedInvocation, evidence: review.evidence }, null, 2)}</pre>
    </details>
    {execution?.state === "approval-required" && onDecline ? <>
      <button type="button" className="button button--ghost" disabled={busy} onClick={() => void decline()}>Decline this action</button>
      <details><summary>Give a different instruction</summary>
        <label>Instruction<textarea value={instruction} onChange={event => setInstruction(event.target.value)} rows={3} /></label>
        <button type="button" className="button" disabled={busy || !instruction.trim()} onClick={() => void decline(instruction.trim())}>Decline &amp; draft instruction</button>
      </details>
    </> : null}
    {error ? <p role="alert">{error}</p> : null}
  </section>;
}
