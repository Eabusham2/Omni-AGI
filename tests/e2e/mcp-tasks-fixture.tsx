import React, { StrictMode, useCallback, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { McpTasksPanel } from "../../src/renderer/src/McpTasksPanel";
import { McpTaskActionView } from "../../src/renderer/src/McpTaskActionView";
import { mcpTaskActionLabel } from "../../src/renderer/src/mcpTaskPresentation";
import type { ActionEvent, OmniApi, ToolExecutionResult, ToolInvocation } from "../../src/shared/types";
import type { McpTaskLocalPage, McpTaskOutcome, McpTaskRecord, McpTasksApi } from "../../src/shared/mcpTasks";
import { mcpUiTaskPage, mcpUiTaskReceipt, mcpUiTaskRecord, mcpUiTaskServer } from "../fixtures/mcpTaskUiData";
import "../../src/renderer/src/styles.css";

type DelayedPage = { brainId: string; owner: number; value: McpTaskLocalPage; resolve(value: McpTaskLocalPage): void };
type DelayedResult = { brainId: string; owner: number; invocation: ToolInvocation; finish(result: ToolExecutionResult): void };
function Fixture() {
  const [brainId, setBrainId] = useState("task-ui-brain"), [apiOwner, setApiOwner] = useState(1), [compact, setCompact] = useState(false);
  const [ask, setAsk] = useState(true), [delayPage, setDelayPage] = useState(false), [failResult, setFailResult] = useState(false);
  const [counts, setCounts] = useState({ local: 0, dispatch: 0, stop: 0, decline: 0 }), [events, setEvents] = useState<string[]>([]);
  const askRef = useRef(ask), delayRef = useRef(delayPage), failRef = useRef(failResult); askRef.current = ask; delayRef.current = delayPage; failRef.current = failResult;
  const records = useRef(new Map<string, McpTaskRecord>()), pages = useRef<DelayedPage[]>([]), waits = useRef<DelayedResult[]>([]), serial = useRef(0);
  const log = useCallback((text: string) => setEvents(value => [...value.slice(-13), text]), []);
  const current = (id: string): McpTaskRecord => { let row = records.current.get(id); if (!row) { row = mcpUiTaskRecord(id); records.current.set(id, row); } return row; };
  const result = (invocation: ToolInvocation, body: Partial<ToolExecutionResult>): ToolExecutionResult => ({ id: `synthetic-read-${++serial.current}`, toolId: invocation.toolId, action: invocation.action,
    state: "complete", startedAt: "2026-10-17T00:00:00.000Z", finishedAt: new Date().toISOString(), ...body });
  const outcome = (invocation: ToolInvocation, retained = false): ToolExecutionResult => {
    const row = current(invocation.brainId), failed = failRef.current;
    const value: McpTaskOutcome = { format: "omni-mcp-task-outcome", version: 1, wireVersion: "mcp-2025-11-25", taskRef: row.taskRef, source: row.source,
      taskResultReceiptId: "d".repeat(64), payloadSha256: "e".repeat(64), kind: failed ? "rpc-error" : "result", retained,
      ...(failed ? { error: { code: -32001, message: "Synthetic remote failure; no real job or external effect exists." } } : { result: { content: [{ type: "text", text: "Synthetic final data, not a provider/model result." }] } }), untrustedTaskData: true };
    row.remote!.status = failed ? "failed" : "completed"; row.result = { receiptId: value.taskResultReceiptId, payloadSha256: value.payloadSha256, kind: value.kind, observedAt: new Date().toISOString() };
    log(`Synthetic ${retained ? "saved" : "new"} result read · ${invocation.brainId} · source receipt ${value.taskResultReceiptId.slice(0, 12)} · zero remote calls`);
    return result(invocation, { output: value });
  };
  const api = useMemo<McpTasksApi>(() => ({
    localPage: async (id, after) => {
      setCounts(value => ({ ...value, local: value.local + 1 })); log(`Local synthetic page ${after ?? "first"} · brain ${id} · API ${apiOwner}`);
      const row = structuredClone(current(id)), page = mcpUiTaskPage(row);
      if (after) {
        const inherited = { ...mcpUiTaskRecord("original-ancestor"), taskRef: "87654321-abcd-abcd-abcd-1234567890ab" };
        inherited.remote!.status = "completed"; page.records = [{ record: inherited, custodyKey: "c".repeat(64), management: "dormant", reason: "Synthetic inherited custody retains its original owner; no current session grant." }];
      } else {
        page.nextCursor = "synthetic-page-two";
        page.records.push({ record: { ...row, taskRef: "11111111-abcd-abcd-abcd-1234567890ab", creation: "creation-outcome-unknown", remote: undefined, result: undefined, lastOperation: undefined,
          reason: "Synthetic lost ACK retained without resend. " + "Long saved diagnostic text wraps without clipping or displacing controls. ".repeat(8) }, custodyKey: "b".repeat(64), management: "dormant", reason: "Lost ACK · no real current account proof and no automatic retry." });
      }
      if (delayRef.current) { delayRef.current = false; setDelayPage(false); return new Promise(resolve => { pages.current.push({ brainId: id, owner: apiOwner, value: page, resolve }); log("Holding synthetic local metadata reply for owner-fence test"); }); }
      return page;
    },
    stopWait: async (id, taskRef) => {
      setCounts(value => ({ ...value, stop: value.stop + 1 })); const index = waits.current.findIndex(wait => wait.brainId === id && wait.owner === apiOwner && wait.invocation.arguments.taskRef === taskRef);
      if (index < 0) { log("Synthetic Stop wait found no matching current wait; no remote cancellation"); return false; }
      const [pending] = waits.current.splice(index, 1), row = current(id); row.lastOperation = { id: "synthetic-result-wait", method: "tasks/result", state: "outcome-unknown", at: new Date().toISOString() };
      pending!.finish(result(pending!.invocation, { state: "failed", error: "Synthetic local wait stopped. Remote work was not cancelled.", mcpTaskReceipt: mcpUiTaskReceipt(structuredClone(row)) }));
      log("Synthetic local wait stopped · remote cancellations 0 · source lease remains current"); return true;
    }
  }), [brainId, apiOwner, log]);
  const tools = useMemo<Pick<OmniApi["tool"], "execute" | "declineApproval">>(() => ({
    execute: async invocation => {
      setCounts(value => ({ ...value, dispatch: value.dispatch + 1 })); log(`Synthetic existing tool.execute · ${invocation.action} · ${invocation.brainId} · ${invocation.approvalToken ? "exact approval" : "initial request"}`);
      if (askRef.current && !invocation.approvalToken) return result(invocation, { state: "approval-required", approvalToken: `synthetic-approval-${apiOwner}`, approvalExpiresAt: new Date(Date.now() + 60000).toISOString() });
      const row = current(invocation.brainId);
      if (invocation.action === "task-result") {
        if (row.result) return outcome(invocation, true);
        return new Promise(resolve => { waits.current.push({ brainId: invocation.brainId, owner: apiOwner, invocation, finish: resolve }); log("Synthetic result wait is held · Stop waiting or Release oldest result"); });
      }
      row.lastOperation = undefined; row.remote!.status = invocation.action === "task-cancel" ? "cancelled" : row.remote!.status;
      return result(invocation, { output: mcpUiTaskReceipt(structuredClone(row)) });
    },
    declineApproval: async (id, _token) => { setCounts(value => ({ ...value, decline: value.decline + 1 })); log(`Synthetic exact Ask declined · ${id} · no remote work`); return true; }
  }), [brainId, apiOwner, log]);
  const servers = useMemo(() => [mcpUiTaskServer(brainId)], [brainId, apiOwner]);
  const ack = mcpUiTaskReceipt(mcpUiTaskRecord(brainId)), action: ActionEvent = { id: "synthetic-chat-ack", brainId, action: { kind: "tool", source: "human", toolId: ack.record.source.toolId, action: "call-task", arguments: { arguments: { fixture: true } } },
    state: "complete", createdAt: ack.record.createdAt, updatedAt: ack.record.updatedAt, execution: { id: "synthetic-chat-ack-execution", toolId: ack.record.source.toolId, action: "call-task", state: "complete", startedAt: ack.record.createdAt, finishedAt: ack.record.updatedAt, output: ack } };
  return <main style={{ height: "100dvh", overflow: "auto", boxSizing: "border-box", padding: 16 }}><div style={{ width: "100%", maxWidth: compact ? 360 : 980, margin: "0 auto", minWidth: 0 }}>
    <header><p className="eyebrow">INERT · NO-BRAIN · NO-NETWORK</p><h1>Production MCP task controls</h1>
      <p>Actual React components with synthetic typed callbacks only. No preload, MCP server/process, provider, API key, model, filesystem grant, neural learning or remote effect.</p></header>
    <section aria-label="Synthetic fixture controls" className="mcp-task-actions">
      <button type="button" className="button" onClick={() => setBrainId(value => value === "task-ui-brain" ? "other-task-ui-brain" : "task-ui-brain")}>Change selected brain</button>
      <button type="button" className="button" onClick={() => setApiOwner(value => value + 1)}>Change API owner</button>
      <button type="button" className="button" aria-pressed={delayPage} onClick={() => setDelayPage(value => !value)}>Delay next local page</button>
      <button type="button" className="button" onClick={() => { const pending = pages.current.shift(); if (pending) { pending.resolve(pending.value); log(`Released old synthetic metadata · ${pending.brainId} · API ${pending.owner}`); } }}>Release delayed local page</button>
      <button type="button" className="button" aria-pressed={ask} onClick={() => setAsk(value => !value)}>Synthetic Ask {ask ? "on" : "off"}</button>
      <button type="button" className="button" aria-pressed={failResult} onClick={() => setFailResult(value => !value)}>Synthetic result {failResult ? "failure" : "success"}</button>
      <button type="button" className="button" onClick={() => { const pending = waits.current.shift(); if (pending) { pending.finish(outcome(pending.invocation)); log(`Released synthetic result · ${pending.brainId} · API ${pending.owner}`); } }}>Release oldest result</button>
      <button type="button" className="button" onClick={() => { records.current.delete(brainId); setApiOwner(value => value + 1); log("Reset only synthetic records/API; no saved instance exists"); }}>Reset synthetic task</button>
      <button type="button" className="button" aria-pressed={compact} onClick={() => setCompact(value => !value)}>{compact ? "Wide content" : "Compact 360px content"}</button>
      <button type="button" className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark"; }}>Switch appearance</button>
    </section>
    <p role="status">Selected {brainId} · API {apiOwner} · local pages {counts.local} · synthetic dispatch callbacks {counts.dispatch} · local Stop {counts.stop} · declines {counts.decline} · network 0 · neural 0</p>
    <McpTasksPanel brainId={brainId} api={api} tools={tools} servers={servers} />
    <section aria-label="Synthetic chat acceptance receipt"><h2>Chat action: {mcpTaskActionLabel(action)}</h2><McpTaskActionView event={action} /></section>
    <section aria-label="Synthetic callback log"><h2>Observed inert callbacks</h2><ol aria-live="polite">{events.map((entry, index) => <li key={`${index}-${entry}`}>{entry}</li>)}</ol></section>
  </div></main>;
}
createRoot(document.getElementById("root")!).render(<StrictMode><Fixture /></StrictMode>);
