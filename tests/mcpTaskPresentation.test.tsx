import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it, vi } from "vitest";
import type { ActionEvent, ToolExecutionResult } from "../src/shared/types";
import type { McpTaskOutcome, McpTasksApi } from "../src/shared/mcpTasks";
import { McpTasksPanel, McpTaskRecordView } from "../src/renderer/src/McpTasksPanel";
import { McpTaskActionView, McpTaskResultView } from "../src/renderer/src/McpTaskActionView";
import { McpTaskUiFlight, mcpTaskActionLabel, mcpTaskFromAction, mcpTaskInvocation, mcpTaskRecordLabel } from "../src/renderer/src/mcpTaskPresentation";
import { chatActionMcpAppViewId } from "../src/renderer/src/chatActionPresentation";
import { mcpUiTaskPage, mcpUiTaskReceipt, mcpUiTaskRecord, mcpUiTaskServer, MCP_UI_TASK_TOOL } from "./fixtures/mcpTaskUiData";

const execution = (output: unknown, overrides: Partial<ToolExecutionResult> = {}): ToolExecutionResult => ({ id: "synthetic-execution", toolId: MCP_UI_TASK_TOOL, action: "call-task", state: "complete", startedAt: "2026-10-17T00:00:00Z", finishedAt: "2026-10-17T00:00:01Z", output, ...overrides });
const event = (result: ToolExecutionResult): ActionEvent => ({ id: "synthetic-action", brainId: "task-ui-brain", action: { kind: "tool", source: "human", toolId: MCP_UI_TASK_TOOL, action: result.action, arguments: {} }, state: result.state === "complete" ? "complete" : "failed", createdAt: result.startedAt, updatedAt: result.finishedAt!, execution: result });

it("renders collapsed compact controls without reading local metadata, connecting, dispatching or polling on render", () => {
  const api = { localPage: vi.fn(), stopWait: vi.fn() } as unknown as McpTasksApi, tools = { execute: vi.fn(), declineApproval: vi.fn() };
  const html = renderToStaticMarkup(<McpTasksPanel brainId="task-ui-brain" api={api} tools={tools} servers={[mcpUiTaskServer()]} />);
  expect(html).toContain("Remote task history"); expect(html).not.toContain("<details open"); expect(api.localPage).not.toHaveBeenCalled(); expect(tools.execute).not.toHaveBeenCalled();
  expect(renderToStaticMarkup(<McpTasksPanel brainId="task-ui-brain" servers={[]} />)).toBe("");
});
it("labels accepted and unknown task attempts as pending/unknown rather than Done, without parsing prose", () => {
  const accepted = mcpUiTaskReceipt(), acceptedEvent = event(execution(accepted)); expect(mcpTaskActionLabel(acceptedEvent)).toBe("Remote work in progress");
  const html = renderToStaticMarkup(<McpTaskActionView event={acceptedEvent} />); expect(html).toContain("Acceptance is not its final result"); expect(html).not.toContain("<pre");
  const unknown = mcpUiTaskReceipt({ ...mcpUiTaskRecord(), creation: "creation-outcome-unknown", remote: undefined }, "dormant");
  const failed = event(execution(undefined, { state: "failed", mcpTaskReceipt: unknown, error: "arbitrary text with no state authority" }));
  expect(mcpTaskActionLabel(failed)).toBe("Creation outcome unknown"); expect(mcpTaskFromAction(failed)).toEqual(unknown);
  expect(mcpTaskFromAction(event(execution(undefined, { state: "failed", error: JSON.stringify(unknown) })))).toBeUndefined();
});
it("preserves source result receipt identity and truthful remote failures across separate per-read presentations", () => {
  const raw: McpTaskOutcome = { format: "omni-mcp-task-outcome", version: 1, wireVersion: "mcp-2025-11-25", source: mcpUiTaskRecord().source,
    taskRef: mcpUiTaskRecord().taskRef, taskResultReceiptId: "d".repeat(64), payloadSha256: "e".repeat(64), kind: "rpc-error", error: { code: -32001, message: "Synthetic remote failure" }, retained: false, untrustedTaskData: true };
  const first = event(execution(raw, { action: "task-result" })), next = event(execution({ ...raw, retained: true }, { id: "another-explicit-read", action: "task-result" }));
  expect(mcpTaskActionLabel(first)).toBe("Remote task failed"); expect(mcpTaskFromAction(next)).toMatchObject({ taskResultReceiptId: raw.taskResultReceiptId, retained: true });
  expect(renderToStaticMarkup(<McpTaskResultView value={{ ...raw, retained: true }} />)).toContain("remote action did not run again"); expect(next.execution?.id).not.toBe(first.execution?.id);
});
it("builds only original tool/ref management requests and leaves dormant/foreign/disabled records without remote controls", () => {
  const entry = mcpUiTaskPage().records[0]!, servers = [mcpUiTaskServer()];
  expect(mcpTaskInvocation("task-ui-brain", entry, "task-result", servers)).toEqual({ brainId: "task-ui-brain", toolId: MCP_UI_TASK_TOOL, action: "task-result", arguments: { taskRef: entry.record.taskRef } });
  const dormant = { ...entry, management: "dormant" as const };
  expect(() => mcpTaskInvocation("task-ui-brain", dormant, "task-status", servers)).toThrow(/not rebound/);
  expect(() => mcpTaskInvocation("other", entry, "task-status", servers)).toThrow(); expect(() => mcpTaskInvocation("task-ui-brain", entry, "task-cancel", [{ ...servers[0]!, effectiveEnabled: false }])).toThrow();
  const html = renderToStaticMarkup(<McpTaskRecordView brainId="task-ui-brain" entry={dormant} servers={servers} api={{ localPage: vi.fn(), stopWait: vi.fn() }} tools={{ execute: vi.fn(), declineApproval: vi.fn() }} />);
  expect(html).toContain("Saved history"); expect(html).not.toContain("Refresh remote status"); expect(html).not.toContain("Cancel remote task");
});
it("retires stale renderer/API flights without issuing any remote cancellation or accepting a late reply", async () => {
  const old = new McpTaskUiFlight(), first = old.begin()!; expect(old.begin()).toBeUndefined();
  let release!: () => void; const delayed = new Promise<void>(resolve => { release = resolve; }), seen: string[] = [];
  const reply = delayed.then(() => { if (old.current(first)) seen.push("old response"); }); old.close();
  const current = new McpTaskUiFlight(), second = current.begin()!; release(); await reply;
  expect(seen).toEqual([]); expect(old.finish(first)).toBe(false); expect(current.current(second)).toBe(true); expect(current.finish(second)).toBe(true);
});
it("keeps acknowledged task App presentation separate from immutable raw receipts and rejects other source bindings", () => {
  const source = mcpUiTaskRecord().source, outcome: McpTaskOutcome = { format: "omni-mcp-task-outcome", version: 1, wireVersion: "mcp-2025-11-25", source,
    taskRef: mcpUiTaskRecord().taskRef, taskResultReceiptId: "d".repeat(64), payloadSha256: "e".repeat(64), kind: "result", result: { content: [] }, retained: false,
    observedResult: { _meta: { "omni/mcpApp": { brainId: source.brainId, toolId: source.toolId, id: "saved-host-view", untrusted: true } } }, untrustedTaskData: true };
  const value = event(execution(outcome, { action: "task-result" })); expect(chatActionMcpAppViewId(value)).toBe("saved-host-view"); expect(outcome.result).toEqual({ content: [] });
  expect(mcpTaskFromAction({ ...value, brainId: "foreign" })).toBeUndefined(); expect(mcpTaskFromAction({ ...value, action: { ...value.action, toolId: "mcp.other" } })).toBeUndefined();
  expect(mcpTaskRecordLabel({ ...mcpUiTaskRecord(), remote: { ...mcpUiTaskRecord().remote!, status: "completed" } })).toContain("result not read");
});
