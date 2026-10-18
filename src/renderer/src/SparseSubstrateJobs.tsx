import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { SparseSubstrateJobView, SparseSubstratePlanView, SparseSubstrateUiApi } from "../../shared/sparseSubstrateTransfer";
import type { ActionEvent, OmniApi, ToolExecutionResult, ToolInvocation } from "../../shared/types";
import { ActionReviewDetails } from "./ActionReviewDetails";
import { SparseJobsController, emptySparseJobs, type SparseJobsSnapshot } from "./sparseSubstrateJobsController";
import { sparseJobCanPreviewRollback, sparseJobCanStop, sparseJobForAction, sparseJobLabel } from "./sparseSubstratePresentation";
import { Icon } from "./icons";
import "./sparseSubstrateJobs.css";

type Tools = Pick<OmniApi["tool"], "execute" | "declineApproval">;
interface Scope { brainId: string; api?: SparseSubstrateUiApi; tools?: Tools; snapshot: SparseJobsSnapshot; controller?: SparseJobsController }
const JobsContext = createContext<Scope | undefined>(undefined);
export function SparseSubstrateJobsProvider({ brainId, api = typeof window === "undefined" ? undefined : window.omni?.substrate,
  tools = typeof window === "undefined" ? undefined : window.omni?.tool, children }: { brainId: string; api?: SparseSubstrateUiApi; tools?: Tools; children: ReactNode }) {
  const [snapshot, setSnapshot] = useState(() => emptySparseJobs(brainId));
  const controller = useRef<SparseJobsController | undefined>(undefined);
  useEffect(() => {
    if (!api) return;
    const owned = new SparseJobsController(brainId, api, setSnapshot); controller.current = owned; owned.start();
    return () => { owned.stop(); if (controller.current === owned) controller.current = undefined; };
  }, [brainId, api]);
  const value = useMemo<Scope>(() => ({ brainId, api, tools, snapshot: snapshot.brainId === brainId ? snapshot : emptySparseJobs(brainId),
    controller: controller.current?.brainId === brainId ? controller.current : undefined }), [brainId, api, tools, snapshot]);
  return <JobsContext.Provider value={value}>{children}</JobsContext.Provider>;
}

export function useSparseSubstrateActionJob(event: ActionEvent): SparseSubstrateJobView | undefined {
  const scope = useContext(JobsContext), jobId = sparseJobForAction(event, scope?.snapshot.jobs ?? []);
  const controller = scope?.brainId === event.brainId ? scope.controller : undefined;
  useEffect(() => { if (jobId) void controller?.ensure(jobId); }, [jobId, controller]);
  return scope?.brainId === event.brainId ? scope.snapshot.jobs.find(job => job.id === jobId) : undefined;
}

export function SparseSubstrateJobsPanel() {
  const scope = useContext(JobsContext);
  if (!scope?.api) return null;
  const { snapshot, controller } = scope;
  return <section className="sparse-jobs" aria-label="Connection update history">
    <div className="sparse-jobs__heading"><h3>Connection updates</h3>
      <button type="button" className="button button--ghost" disabled={snapshot.loading || snapshot.paging} onClick={() => void controller?.refresh()}>Refresh</button></div>
    {snapshot.error ? <p role="alert">{snapshot.error}</p> : null}
    {snapshot.loading && !snapshot.jobs.length ? <p role="status">Loading saved updates…</p> : !snapshot.error && !snapshot.jobs.length ? <p>No connection updates yet.</p> : null}
    {snapshot.jobs.map(job => <SparseSubstrateJobCard key={`${job.brainId}/${job.id}`} job={job} />)}
    {snapshot.nextAfter ? <button type="button" className="button button--ghost" disabled={snapshot.loading || snapshot.paging} onClick={() => void controller?.older()}>
      {snapshot.paging ? "Loading more…" : "More saved updates"}</button> : null}
  </section>;
}

export function SparseSubstrateJobCard({ job }: { job: SparseSubstrateJobView }) {
  const scope = useContext(JobsContext);
  const [plan, setPlan] = useState<SparseSubstratePlanView>();
  const [loading, setLoading] = useState(false), [error, setError] = useState("");
  const [preview, setPreview] = useState<SparseSubstratePlanView>();
  const owner = useRef<object>({});
  useEffect(() => { const token = {}; owner.current = token; return () => { if (owner.current === token) owner.current = {}; }; }, [job.brainId, job.id]);
  const canUse = scope?.brainId === job.brainId && Boolean(scope.api);
  const readPlan = async () => {
    if (!canUse || loading || plan) return;
    const token = owner.current; setLoading(true); setError("");
    try { const next = await scope!.api!.plan(job.brainId, job.planId);
      if (owner.current !== token) return;
      if (next.brainId !== job.brainId || next.id !== job.planId) throw new Error("This plan belongs to another update.");
      setPlan(next);
    } catch (reason) { if (owner.current === token) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (owner.current === token) setLoading(false); }
  };
  const busy = scope?.snapshot.busyJobIds.includes(job.id) ?? false;
  const acceptPreview = async (result: ToolExecutionResult, operation: "import" | "rollback") => {
    const output = result.output as { id?: unknown; format?: unknown; scope?: unknown } | undefined;
    if (!output || output.format !== "omni-sparse-substrate-plan" || output.scope !== "native-sparse-assemblies" || typeof output.id !== "string") throw new Error("No verified plan was returned.");
    const token = owner.current, next = await scope!.api!.plan(job.brainId, output.id);
    if (owner.current !== token) return;
    if (next.brainId !== job.brainId || next.id !== output.id || next.operation !== operation) throw new Error("The returned plan does not match this review.");
    setPreview(next);
  };
  return <details className={`sparse-job sparse-job--${job.state}`} onToggle={event => { if (event.currentTarget.open && job.state === "needs-review") void readPlan(); }}>
    <summary><span><strong>{sparseJobLabel(job.state)}</strong><small>{job.receipt ? `${job.receipt.appliedRows.toLocaleString()} rows committed` : "Neural connection update"}</small></span>
      <span className="sparse-job__stamp"><time dateTime={job.updatedAt} title={new Date(job.updatedAt).toLocaleString()}>{new Date(job.updatedAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}</time><Icon name="chevron" size={13} /></span></summary>
    <div className="sparse-job__body">
      <p role={sparseJobCanStop(job.state) ? "status" : undefined}>{job.state === "queued" || job.state === "boundary-required" ? "Waiting for Load or the current reply to finish." :
        job.state === "applying" ? "The saved result will confirm when this finishes." : job.state === "uncertain" ? "The result is not confirmed. Refresh or inspect it before starting another update." :
        job.state === "dormant" ? "History is preserved; no copied permission or automatic work is active." : job.state === "needs-review" ? "Access or selected connections changed. Prepare a new preview before applying." :
        job.state === "cancelled" ? "Stopped. Previously committed work is preserved." : job.state === "complete" ? "The native saved-state receipt confirms this update." : "This update did not complete."}</p>
      {job.error ? <details className="sparse-job__error"><summary>What happened</summary><p>{job.error}</p></details> : null}
      {scope?.snapshot.jobErrors[job.id] || error ? <p role="alert">{scope?.snapshot.jobErrors[job.id] ?? error}</p> : null}
      <div className="sparse-job__actions">
        {canUse && sparseJobCanStop(job.state) ? <button type="button" className="button button--ghost" disabled={busy} onClick={() => void scope?.controller?.stopJob(job.id)}>{busy ? "Stopping…" : "Stop update"}</button> : null}
        {canUse ? <button type="button" className="button button--ghost" onClick={() => void scope?.controller?.refreshJob(job.id)}>Refresh status</button> : null}
      </div>
      <details onToggle={event => { if (event.currentTarget.open) void readPlan(); }}><summary>Connections and receipt details</summary>
        {loading ? <p role="status">Loading plan…</p> : null}
        {plan ? <SparseSubstratePlanDetails plan={plan} /> : null}
        <dl className="sparse-job__facts"><div><dt>Update ID</dt><dd>{job.id}</dd></div><div><dt>Plan ID</dt><dd>{job.planId}</dd></div>
          <div><dt>Plan checksum</dt><dd>{job.planSha256}</dd></div><div><dt>Revision</dt><dd>{job.revision}</dd></div>
          {job.receipt ? <><div><dt>Applied receipt</dt><dd>{job.receipt.receiptSha256}</dd></div><div><dt>Verified generation</dt><dd>{job.receipt.generationSha256}</dd></div></> : null}</dl>
      </details>
      {canUse && sparseJobCanPreviewRollback(job) ? <SparseToolOperation key={`inverse-${job.id}`} brainId={job.brainId} tools={scope?.tools}
        invocation={{ brainId: job.brainId, toolId: "brain.substrate", action: "prepare-rollback", arguments: { jobId: job.id } }} title="Preview rollback"
        onComplete={result => acceptPreview(result, "rollback")} /> : null}
      {canUse && job.state === "needs-review" && plan ? <SparseToolOperation key={`fresh-${plan.id}`} brainId={job.brainId} tools={scope?.tools} title="Prepare new preview"
        invocation={{ brainId: job.brainId, toolId: "brain.substrate", action: "prepare-import", arguments: { bundleId: plan.bundleId, mode: plan.mode } }}
        onComplete={result => acceptPreview(result, "import")} /> : null}
      {preview ? <section className="sparse-job__preview" aria-label="Connection update preview"><h4>{preview.operation === "rollback" ? "Rollback preview" : "New update preview"}</h4><SparseSubstratePlanDetails plan={preview} />
        {preview.conflicts ? <p role="alert">This preview has conflicts and cannot be applied.</p> : <SparseToolOperation key={`${preview.id}/${preview.sha256}`} brainId={job.brainId} tools={scope?.tools} title={preview.operation === "rollback" ? "Apply reviewed rollback" : "Apply reviewed update"}
          invocation={{ brainId: job.brainId, toolId: "brain.substrate", action: "apply", arguments: { planId: preview.id, reviewSha256: preview.sha256 } }} />}</section> : null}
    </div>
  </details>;
}

function SparseSubstratePlanDetails({ plan }: { plan: SparseSubstratePlanView }) {
  return <><p>{plan.operation === "rollback" ? "Rollback" : "Import"} · {plan.mode === "replace-exact" ? "Exact selected replacements" : "New connections"}</p>
    <dl className="sparse-job__facts"><div><dt>Rows reviewed</dt><dd>{plan.rows.toLocaleString()}</dd></div><div><dt>Added</dt><dd>{plan.additions.toLocaleString()}</dd></div>
      <div><dt>Replaced</dt><dd>{plan.replacements.toLocaleString()}</dd></div><div><dt>Unchanged</dt><dd>{plan.identical.toLocaleString()}</dd></div><div><dt>Conflicts</dt><dd>{plan.conflicts.toLocaleString()}</dd></div>
      {plan.unprovenRemovalRows ? <div><dt>Rows not safe to remove</dt><dd>{plan.unprovenRemovalRows.toLocaleString()}</dd></div> : null}</dl></>;
}

/** Only explicit user buttons invoke effects, always through existing ToolExecutor. */
function SparseToolOperation({ brainId, tools, invocation, title, onComplete }: { brainId: string; tools?: Tools; invocation: ToolInvocation; title: string; onComplete?(result: ToolExecutionResult): Promise<void> }) {
  const [result, setResult] = useState<ToolExecutionResult>();
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const generation = useRef<object>({}), currentToken = useRef<string | undefined>(undefined);
  const inFlight = useRef<object | undefined>(undefined);
  const invocationKey = JSON.stringify([brainId, invocation.toolId, invocation.action, invocation.arguments]);
  useEffect(() => { const owned = {}; generation.current = owned; return () => { if (generation.current === owned) generation.current = {}; const token = currentToken.current;
    if (token) void tools?.declineApproval(brainId, token).catch(() => undefined); currentToken.current = undefined; inFlight.current = undefined; }; }, [brainId, tools, invocationKey]);
  const execute = async () => {
    if (!tools || inFlight.current) return; const owned = generation.current, request = {}; inFlight.current = request; setBusy(true); setError("");
    try {
      const next = await tools.execute({ ...invocation, ...(currentToken.current ? { approvalToken: currentToken.current } : {}) });
      if (generation.current !== owned) { if (next.approvalToken) await tools.declineApproval(brainId, next.approvalToken); return; }
      if (next.toolId !== invocation.toolId || next.action !== invocation.action) {
        if (next.approvalToken) await tools.declineApproval(brainId, next.approvalToken);
        throw new Error("The returned action does not match this review.");
      }
      if (next.state === "approval-required" && !next.approvalToken) throw new Error("This action has no valid approval to review.");
      if (next.state === "complete" && invocation.action === "apply") {
        const output = next.output as { format?: unknown; scope?: unknown; brainId?: unknown; id?: unknown } | undefined;
        if (!output || output.format !== "omni-sparse-substrate-job" || output.scope !== "native-sparse-assemblies" || output.brainId !== brainId || typeof output.id !== "string") throw new Error("No verified update handle was returned.");
      }
      currentToken.current = next.state === "approval-required" ? next.approvalToken : undefined; setResult(next);
      if (next.state === "failed") throw new Error(next.error ?? "The action did not complete.");
      if (next.state === "complete") await onComplete?.(next);
    } catch (reason) { if (generation.current === owned) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (inFlight.current === request) inFlight.current = undefined; if (generation.current === owned) setBusy(false); }
  };
  const decline = async () => { const token = currentToken.current; if (!tools || !token || inFlight.current) return; const owned = generation.current, request = {}; inFlight.current = request; setBusy(true);
    try { await tools.declineApproval(brainId, token); if (generation.current === owned) { currentToken.current = undefined; setResult(undefined); } }
    catch (reason) { if (generation.current === owned) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (inFlight.current === request) inFlight.current = undefined; if (generation.current === owned) setBusy(false); } };
  return <section className="sparse-job__operation">
    {result?.state === "approval-required" ? <div aria-label="Exact connection action review"><strong>Review this action</strong><pre tabIndex={0}>{JSON.stringify({ toolId: invocation.toolId, action: invocation.action, arguments: invocation.arguments }, null, 2)}</pre><ActionReviewDetails execution={result} />
      <button type="button" className="button button--ghost" disabled={busy} onClick={() => void decline()}>Deny</button></div> : null}
    {result?.state === "complete" ? <p role="status">{invocation.action === "apply" ? "Update queued. Its live card will confirm the outcome." : "Preview prepared."}</p> :
      <button type="button" className="button" disabled={!tools || busy} onClick={() => void execute()}>{busy ? "Working…" : result?.state === "approval-required" ? "Approve exact action" : title}</button>}
    {error ? <p role="alert">{error}</p> : null}
    {error && result?.state === "complete" && onComplete ? <button type="button" className="button button--ghost" disabled={busy} onClick={() => {
      const owned = generation.current; setBusy(true); setError(""); void onComplete(result).catch(reason => {
        if (generation.current === owned) setError(reason instanceof Error ? reason.message : String(reason));
      }).finally(() => { if (generation.current === owned) setBusy(false); });
    }}>Open saved preview</button> : null}
  </section>;
}
