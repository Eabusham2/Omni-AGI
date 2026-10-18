import type { ActionEvent, McpServerSummary, ToolExecutionResult, ToolInvocation } from "../../shared/types";
import { MCP_TASK_WIRE_VERSION, type McpTaskLocalPage, type McpTaskOutcome, type McpTaskReceipt, type McpTaskRecord } from "../../shared/mcpTasks";

export type McpTaskLocalEntry = McpTaskLocalPage["records"][number];
export type McpTaskManagementAction = "task-status" | "task-result" | "task-cancel";
const object = (value: unknown): Record<string, unknown> | undefined => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
const ref = /^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i, sha = /^[a-f0-9]{64}$/;

/** Display-only validation. These values never authorize a remote action. */
export function mcpTaskValue(value: unknown, brainId: string, toolId: string): McpTaskReceipt | McpTaskOutcome | undefined {
  const root = object(value); if (!root || root.version !== 1 || root.wireVersion !== MCP_TASK_WIRE_VERSION || root.untrustedTaskData !== true) return undefined;
  if (root.format === "omni-mcp-task") {
    const record = object(root.record), source = object(record?.source);
    if (!record || !source || record.format !== "omni-mcp-task-record" || record.authority !== "historical-not-live" ||
        !ref.test(String(record.taskRef)) || root.operationComplete !== false || !["current-session", "dormant"].includes(String(root.management)) ||
        source.brainId !== brainId || source.toolId !== toolId) return undefined;
    return root as unknown as McpTaskReceipt;
  }
  const source = object(root.source);
  if (root.format !== "omni-mcp-task-outcome" || source?.brainId !== brainId || source.toolId !== toolId || !ref.test(String(root.taskRef)) ||
      !sha.test(String(root.taskResultReceiptId)) || !sha.test(String(root.payloadSha256)) || !["result", "rpc-error"].includes(String(root.kind)) || typeof root.retained !== "boolean") return undefined;
  return root as unknown as McpTaskOutcome;
}

export function mcpTaskFromExecution(execution: ToolExecutionResult | undefined, brainId: string): McpTaskReceipt | McpTaskOutcome | undefined {
  if (!execution?.toolId.startsWith("mcp.") || !["call-task", "task-status", "task-result", "task-cancel"].includes(execution.action)) return undefined;
  return mcpTaskValue(execution.mcpTaskReceipt ?? (execution.state === "complete" ? execution.output : undefined), brainId, execution.toolId);
}

export function mcpTaskFromAction(event: ActionEvent): McpTaskReceipt | McpTaskOutcome | undefined {
  if (event.action.kind !== "tool" || event.execution?.toolId !== event.action.toolId || event.execution?.action !== event.action.action) return undefined;
  return mcpTaskFromExecution(event.execution, event.brainId);
}

export function mcpTaskRecordLabel(record: McpTaskRecord): string {
  if (record.result) return record.result.kind === "rpc-error" ? "Remote task failed · result saved" : "Remote result saved";
  if (record.creation === "creation-outcome-unknown") return "Creation outcome unknown";
  if (record.lastOperation?.state === "outcome-unknown") return "Last request outcome unknown";
  if (record.creation === "prepared" || record.creation === "not-dispatched") return "Not dispatched";
  if (record.creation === "rejected-before-task") return "Task request rejected";
  if (record.creation === "dispatch-may-have-started") return "Request may have started";
  return ({ working: "Remote work in progress", input_required: "Remote task requests input", completed: "Remote work complete · result not read",
    failed: "Remote task failed · result not read", cancelled: "Remote task cancelled · previous effects not undone" } as const)[record.remote?.status ?? "working"];
}

export function mcpTaskResultLabel(value: McpTaskReceipt | McpTaskOutcome): string {
  if (value.format === "omni-mcp-task") return mcpTaskRecordLabel(value.record);
  if (value.kind === "rpc-error") return "Remote task failed";
  if (value.result?.isError === true) return "Remote tool reported an error";
  return value.retained ? "Saved remote result read" : "Remote result received";
}

export function mcpTaskActionLabel(event: ActionEvent): string | undefined {
  const value = mcpTaskFromAction(event); if (value) return mcpTaskResultLabel(value);
  if (event.state !== "running" || !event.action.toolId?.startsWith("mcp.")) return undefined;
  return event.action.action === "task-result" ? "Waiting for remote result" : event.action.action === "call-task" ? "Requesting remote task" : undefined;
}

/** Creates only the existing ordinary permissioned tool invocation; no task IDs become grants. */
export function mcpTaskInvocation(brainId: string, entry: McpTaskLocalEntry, action: McpTaskManagementAction, servers: McpServerSummary[]): ToolInvocation {
  const source = entry.record.source, server = servers.find(value => value.brainId === brainId && value.id === source.serverId), tool = server?.tools.find(value => value.id === source.toolId);
  if (entry.management !== "current-session" || source.brainId !== brainId || !entry.record.remote || !server?.connected || !server.effectiveEnabled || !tool?.actions.includes(action)) {
    throw new Error("This saved task has no verified current connection/action. History was not rebound or resent.");
  }
  if (action === "task-cancel" && ["completed", "failed", "cancelled"].includes(entry.record.remote.status)) throw new Error("This remote task is already terminal.");
  return { brainId, toolId: source.toolId, action, arguments: { taskRef: entry.record.taskRef } };
}

/** Renderer ownership only, not remote cancellation or an authorization token. */
export class McpTaskUiFlight {
  private ticket: object | undefined;
  private closed = false;
  begin(): object | undefined { if (this.closed || this.ticket) return undefined; return this.ticket = {}; }
  current(ticket: object): boolean { return !this.closed && this.ticket === ticket; }
  finish(ticket: object): boolean { if (!this.current(ticket)) return false; this.ticket = undefined; return true; }
  close(): void { this.closed = true; this.ticket = undefined; }
}
