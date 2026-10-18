import { useState } from "react";
import type { ActionEvent } from "../../shared/types";
import type { McpTaskOutcome, McpTaskReceipt } from "../../shared/mcpTasks";
import { mcpTaskFromAction, mcpTaskResultLabel } from "./mcpTaskPresentation";
import "./mcpTasks.css";

export function McpTaskResultView({ value }: { value: McpTaskReceipt | McpTaskOutcome }) {
  const [expanded, setExpanded] = useState(false), receipt = value.format === "omni-mcp-task", source = receipt ? value.record.source : value.source;
  const receiptDescription = !receipt ? "" : value.record.creation !== "accepted"
    ? "The committed attempt is retained; no remote task was automatically retried."
    : value.record.result ? "The final source result is saved. Read it to inspect the returned data."
      : value.record.remote?.status === "cancelled" ? "The server reports cancellation. Its final result has not been read."
        : value.record.remote?.status === "failed" ? "The server reports failure. Read its result for the returned details."
          : value.record.remote?.status === "completed" ? "The server reports completion. Its final result has not been read."
            : value.record.remote?.status === "input_required" ? "The server is waiting for input; the task has not completed."
              : "The server accepted this task. Acceptance is not its final result.";
  return <section className="mcp-task-result" aria-label="Remote MCP task receipt" data-task-pending={receipt ? "true" : undefined}>
    <strong>{mcpTaskResultLabel(value)}</strong>
    {receipt ? <p>{receiptDescription}
      {value.management === "dormant" ? " Saved history has no verified current account/session authority." : ""}</p>
      : <p>{value.retained ? "This read used the saved source result; the remote action did not run again." : "This is the received remote result, not a claim that its contents are correct."}</p>}
    <small>{source.remoteName} · {source.serverId}</small>
    {!receipt ? <><span className="mcp-task-result__receipt">Source result receipt <code title={value.taskResultReceiptId}>{value.taskResultReceiptId.slice(0, 12)}…</code></span>
      {value.kind === "rpc-error" ? <p role="status">{value.error?.message ?? "The server reported an underlying task failure."}</p> : null}</> : null}
    <details onToggle={event => setExpanded(event.currentTarget.open)}><summary>Inspect saved task receipt</summary>
      {expanded ? <pre tabIndex={0}>{JSON.stringify(value, null, 2)}</pre> : null}
    </details>
  </section>;
}

export function McpTaskActionView({ event }: { event: ActionEvent }) {
  const value = mcpTaskFromAction(event); return value ? <McpTaskResultView value={value} /> : null;
}
