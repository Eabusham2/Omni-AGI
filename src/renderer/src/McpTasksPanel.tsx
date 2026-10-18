import { useEffect, useRef, useState } from "react";
import type { McpServerSummary, OmniApi, ToolExecutionResult, ToolInvocation } from "../../shared/types";
import type { McpTaskLocalPage, McpTasksApi } from "../../shared/mcpTasks";
import { ActionReviewDetails } from "./ActionReviewDetails";
import { McpTaskResultView } from "./McpTaskActionView";
import { McpTaskUiFlight, mcpTaskFromExecution, mcpTaskInvocation, mcpTaskRecordLabel, type McpTaskLocalEntry, type McpTaskManagementAction } from "./mcpTaskPresentation";
import "./mcpTasks.css";

type Tools = Pick<OmniApi["tool"], "execute" | "declineApproval">;
interface Owner { brainId: string; api?: McpTasksApi; tools?: Tools; servers?: McpServerSummary[]; gate: McpTaskUiFlight; stopping?: boolean }
function useOwner(brainId: string, api?: McpTasksApi, tools?: Tools, servers?: McpServerSummary[]) {
  const owner = useRef<Owner | undefined>(undefined);
  if (!owner.current || owner.current.brainId !== brainId || owner.current.api !== api || owner.current.tools !== tools || owner.current.servers !== servers) {
    owner.current?.gate.close(); owner.current = { brainId, api, tools, servers, gate: new McpTaskUiFlight() };
  }
  useEffect(() => { owner.current!.gate = new McpTaskUiFlight(); return () => { owner.current?.gate.close(); }; }, []);
  return owner as { current: Owner };
}
const message = (failure: unknown): string => failure instanceof Error ? failure.message : String(failure);

interface TaskRowState {
  owner: Owner;
  busy?: McpTaskManagementAction | "decline";
  stopping?: boolean;
  execution?: ToolExecutionResult;
  approval?: { invocation: ToolInvocation; token: string };
  note?: string;
  error?: string;
}

/** One exact source record. Controls still pass the original tool through main's ordinary permission/reviewer path. */
export function McpTaskRecordView({ brainId, entry, servers, api, tools, onUpdated }: {
  brainId: string; entry: McpTaskLocalEntry; servers: McpServerSummary[]; api?: McpTasksApi; tools?: Tools; onUpdated?(): void;
}) {
  const ownerRef = useOwner(brainId, api, tools, servers), owner = ownerRef.current;
  const [saved, setSaved] = useState<TaskRowState>(), [expanded, setExpanded] = useState(false);
  const state = saved?.owner === owner ? saved : undefined;
  const valid = (ticket: object) => ownerRef.current === owner && owner.gate.current(ticket);
  const allowed = (action: McpTaskManagementAction): boolean => { try { mcpTaskInvocation(brainId, entry, action, servers); return !!api && !!tools; } catch { return false; } };
  const dispatch = async (action: McpTaskManagementAction, approved?: TaskRowState["approval"]): Promise<void> => {
    if (!api || !tools || owner.stopping) return; const ticket = owner.gate.begin(); if (!ticket) return;
    setSaved({ owner, busy: action });
    try {
      const invocation = mcpTaskInvocation(brainId, entry, action, servers);
      if (approved && (approved.invocation.brainId !== invocation.brainId || approved.invocation.toolId !== invocation.toolId || approved.invocation.action !== invocation.action ||
          approved.invocation.arguments.taskRef !== invocation.arguments.taskRef)) throw new Error("This approval no longer belongs to the current task action.");
      const execution = await tools.execute({ ...invocation, ...(approved ? { approvalToken: approved.token } : {}) });
      if (!valid(ticket)) return;
      if (execution.toolId !== invocation.toolId || execution.action !== invocation.action) throw new Error("Task action returned a different tool or operation.");
      setSaved(current => ({ owner, execution, ...(current?.owner === owner && current.note ? { note: current.note } : {}), ...(owner.stopping ? { stopping: true } : {}),
        ...(execution.state === "approval-required" && execution.approvalToken ? { approval: { invocation, token: execution.approvalToken } } : {}),
        ...(execution.state === "failed" ? { error: execution.error ?? "The request failed; remote work was not assumed cancelled." } : {}) }));
      if (execution.state !== "approval-required") onUpdated?.();
    } catch (failure) { if (valid(ticket)) setSaved(current => ({ ...current, owner, busy: undefined, error: message(failure) })); }
    finally { if (valid(ticket)) { owner.gate.finish(ticket); setSaved(current => current?.owner === owner ? { ...current, busy: undefined } : current); } }
  };
  const stopWait = async (): Promise<void> => {
    if (!api || !state?.busy || state.busy === "decline" || owner.stopping) return;
    owner.stopping = true; setSaved(current => current?.owner === owner ? { ...current, stopping: true } : current);
    try {
      const stopped = await api.stopWait(brainId, entry.record.taskRef);
      if (ownerRef.current === owner) setSaved(current => ({ ...current, owner, note: stopped ? "Local wait stopped. Remote work was not cancelled or undone." : "That local wait had already finished; no remote cancellation was sent." }));
    } catch (failure) { if (ownerRef.current === owner) setSaved(current => ({ ...current, owner, error: message(failure) })); }
    finally { owner.stopping = false; if (ownerRef.current === owner) setSaved(current => current?.owner === owner ? { ...current, stopping: false } : current); }
  };
  const decline = async (): Promise<void> => {
    if (!tools || !state?.approval || owner.stopping) return; const ticket = owner.gate.begin(); if (!ticket) return;
    const approval = state.approval; setSaved({ ...state, busy: "decline", error: undefined });
    try {
      const declined = await tools.declineApproval(brainId, approval.token);
      if (valid(ticket)) setSaved({ owner, note: declined ? "This action was declined; no remote request was sent." : "That approval is no longer available; no request was sent." });
    } catch (failure) { if (valid(ticket)) setSaved(current => ({ ...current, owner, error: message(failure) })); }
    finally { if (valid(ticket)) { owner.gate.finish(ticket); setSaved(current => current?.owner === owner ? { ...current, busy: undefined } : current); } }
  };
  const result = mcpTaskFromExecution(state?.execution, brainId), blocked = !!state?.busy || !!state?.stopping || !!state?.approval;
  const source = entry.record.source, session = entry.management === "current-session" && source.brainId === brainId;
  return <article className="mcp-task-record" aria-label={`${source.remoteName} remote task`}>
    <div className="mcp-task-record__head"><span><strong>{source.remoteName}</strong><small>{source.serverId} · {mcpTaskRecordLabel(entry.record)}</small></span>
      <span>{session ? "Current session · permission checked per action" : "Saved history · no current account authority"}</span></div>
    {!session ? <p>{entry.reason ?? "This retained task was not rebound, resent or given a live grant."}</p> : null}
    <div className="mcp-task-actions">
      {([ ["task-status", "Refresh remote status"], ["task-result", "Read remote result"], ["task-cancel", "Cancel remote task"] ] as const).map(([action, label]) => allowed(action)
        ? <button type="button" className={`button button--${action === "task-cancel" ? "ghost" : "secondary"}`} disabled={blocked} key={action} onClick={() => void dispatch(action)}>{label}</button> : null)}
      {state?.busy && state.busy !== "decline" ? <button type="button" className="button button--ghost" disabled={!!state.stopping} onClick={() => void stopWait()}>{state.stopping ? "Stopping local wait…" : "Stop waiting"}</button> : null}
    </div>
    {state?.busy ? <p role="status">{state.busy === "task-result" ? "Waiting for the remote result…" : state.busy === "task-cancel" ? "Requesting remote cancellation; previous effects are not undone…" : state.busy === "decline" ? "Declining this exact action…" : "Reading remote status…"}</p> : null}
    {state?.approval ? <section aria-label="Remote task action approval"><p>{state.approval.invocation.action === "task-cancel" ? "Approve a cancellation request for this remote task?" : "Approve this exact remote task request?"}</p>
      <div className="mcp-task-actions"><button type="button" className="button button--primary" disabled={!!state.busy || !!state.stopping} onClick={() => void dispatch(state.approval!.invocation.action as McpTaskManagementAction, state.approval)}>Approve exact request</button>
        <button type="button" className="button button--ghost" disabled={!!state.busy || !!state.stopping} onClick={() => void decline()}>Decline request</button></div>
      {state.execution?.approvalExpiresAt ? <small>Approval expires {new Date(state.execution.approvalExpiresAt).toLocaleTimeString()}.</small> : null}
    </section> : null}
    <ActionReviewDetails execution={state?.execution} />
    {state?.note ? <p role="status">{state.note}</p> : null}
    {state?.error ? <p role="alert" className="mcp-task-error">{state.error}</p> : null}
    {result ? <McpTaskResultView value={result} /> : null}
    <details className="mcp-task-record__details" onToggle={event => setExpanded(event.currentTarget.open)}><summary>Source and saved custody</summary>
      {expanded ? <pre tabIndex={0}>{JSON.stringify(entry.record, null, 2)}</pre> : null}
    </details>
  </article>;
}

/** Explicit local metadata reads only. Render/open never connects to a remote service or starts work. */
export function McpTasksPanel({ brainId, api, tools, servers }: { brainId: string; api?: McpTasksApi; tools?: Tools; servers: McpServerSummary[] }) {
  const ownerRef = useOwner(brainId, api), owner = ownerRef.current;
  const [saved, setSaved] = useState<{ owner: Owner; page?: McpTaskLocalPage; busy?: boolean; error?: string; cursor?: string; previous: Array<string | undefined> }>();
  const state = saved?.owner === owner ? saved : undefined;
  const read = async (cursor?: string, previous: Array<string | undefined> = []): Promise<void> => {
    if (!api) return; const ticket = owner.gate.begin(); if (!ticket) return;
    setSaved(current => ({ ...(current?.owner === owner ? current : {}), owner, busy: true, error: undefined, cursor, previous }));
    const valid = () => ownerRef.current === owner && owner.gate.current(ticket);
    try {
      const page = await api.localPage(brainId, cursor, 25); if (!valid()) return;
      if (page.brainId !== brainId || page.format !== "omni-mcp-task-local-page" || page.networkRequestMade !== false) throw new Error("Task metadata did not return this instance's local-only page.");
      setSaved({ owner, page, cursor, previous });
    } catch (failure) { if (valid()) setSaved(current => ({ ...current, owner, cursor, previous, busy: false, error: message(failure) })); }
    finally { if (valid()) { owner.gate.finish(ticket); setSaved(current => current?.owner === owner ? { ...current, busy: false } : current); } }
  };
  if (!api) return null;
  return <details className="mcp-tasks" onToggle={event => { if (event.currentTarget.open && !state?.page && !state?.busy) void read(); }}>
    <summary>Remote task history</summary>
    <p>Saved receipts are local. Remote status, results and cancellation use the original tool and its current permissions.</p>
    <div className="mcp-task-actions"><button type="button" className="button button--secondary" disabled={!!state?.busy} onClick={() => void read()}>Refresh local history</button>
      {state?.previous.length ? <button type="button" className="button button--ghost" disabled={!!state.busy} onClick={() => void read(state.previous.at(-1), state.previous.slice(0, -1))}>Previous page</button> : null}
      {state?.page?.nextCursor ? <button type="button" className="button button--ghost" disabled={!!state.busy} onClick={() => void read(state.page?.nextCursor, [...state.previous, state.cursor])}>Next page</button> : null}
    </div>
    {state?.busy ? <p role="status">Reading local task metadata…</p> : null}
    {state?.error ? <p role="alert" className="mcp-task-error">{state.error}</p> : null}
    {state?.page?.records.map(entry => <McpTaskRecordView key={entry.custodyKey} brainId={brainId} entry={entry} servers={servers} api={api} tools={tools}
      onUpdated={() => { void read(state.cursor, state.previous); }} />)}
    {state?.page && !state.page.records.length ? <p>No saved task receipts on this page.</p> : null}
  </details>;
}
