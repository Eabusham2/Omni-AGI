import { createHash } from "node:crypto";
import { MCP_TASK_WIRE_VERSION, type McpRemoteTask, type McpTaskRecord, type McpTaskSource, type McpTaskStatus } from "../shared/mcpTasks";
import { validateMcpContentBlock } from "./mcpCapabilities";

export const MCP_TASK_REF = /^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i;
export const MCP_TASK_SHA = /^[a-f0-9]{64}$/;
export const mcpTaskSha = (value: string | Uint8Array): string => createHash("sha256").update(value).digest("hex");
export function mcpTaskObject(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`MCP ${label} must be an object.`); return value as Record<string, unknown>;
}
function text(value: unknown, label: string): string { if (typeof value !== "string" || !value.length) throw new Error(`Invalid MCP ${label}.`); return value; }
function time(value: unknown): string { const valueText = text(value, "task timestamp"); if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(valueText) || !Number.isFinite(Date.parse(valueText))) throw new Error("Invalid MCP task timestamp."); return valueText; }
function milliseconds(value: unknown, nullable = false): number | null {
  if (nullable && value === null) return null;
  if (!Number.isSafeInteger(value) || Number(value) < 0) throw new Error("MCP task duration must be nonnegative exact integer milliseconds."); return Number(value);
}
export function mcpTaskSupport(value: unknown): "required" | "optional" | "forbidden" {
  const execution = value === undefined ? {} : mcpTaskObject(value, "tool execution metadata");
  const support = execution.taskSupport ?? "forbidden";
  if (!["required", "optional", "forbidden"].includes(String(support))) throw new Error("MCP tool taskSupport is invalid."); return support as "required" | "optional" | "forbidden";
}
export function checkedMcpRemoteTask(value: unknown): McpRemoteTask {
  const task = mcpTaskObject(value, "task state"); text(task.taskId, "remote task id");
  if (!["working", "input_required", "completed", "failed", "cancelled"].includes(String(task.status))) throw new Error("MCP task status is invalid.");
  time(task.createdAt); time(task.lastUpdatedAt); milliseconds(task.ttl, true);
  if (Date.parse(String(task.lastUpdatedAt)) < Date.parse(String(task.createdAt))) throw new Error("MCP task update precedes creation.");
  if (task.statusMessage !== undefined && typeof task.statusMessage !== "string") throw new Error("MCP task statusMessage must be text.");
  if (task.pollInterval !== undefined) milliseconds(task.pollInterval);
  return structuredClone(task) as unknown as McpRemoteTask;
}
export function checkedMcpCreateTaskResult(value: unknown): McpRemoteTask {
  const result = mcpTaskObject(value, "CreateTaskResult"), task = checkedMcpRemoteTask(result.task);
  if (task.status !== "working") throw new Error("MCP 2025-11-25 tasks must begin working; no final result was invented from a malformed ACK.");
  if (result._meta !== undefined) mcpTaskObject(result._meta, "task result metadata");
  return task;
}
const TERMINAL = new Set<McpTaskStatus>(["completed", "failed", "cancelled"]);
export function isMcpTaskTerminal(status: McpTaskStatus): boolean { return TERMINAL.has(status); }
export function assertMcpTaskTransition(previous: McpRemoteTask | undefined, next: McpRemoteTask): void {
  if (!previous) return;
  if (previous.taskId !== next.taskId || previous.createdAt !== next.createdAt || Date.parse(next.lastUpdatedAt) < Date.parse(previous.lastUpdatedAt) ||
    isMcpTaskTerminal(previous.status) && previous.status !== next.status) throw new Error("MCP task identity, chronology or terminal state changed.");
}
export function relatedMcpTaskId(params: unknown): string | undefined {
  const root = mcpTaskObject(params, "task-related message"), metadata = root._meta === undefined ? {} : mcpTaskObject(root._meta, "task-related metadata");
  if (!Object.hasOwn(metadata, "io.modelcontextprotocol/related-task")) return undefined;
  return text(mcpTaskObject(metadata["io.modelcontextprotocol/related-task"], "related-task metadata").taskId, "related-task id");
}
export function checkedMcpTaskToolResult(value: unknown, taskId: string): Record<string, unknown> {
  const result = mcpTaskObject(value, "task tool result");
  if (relatedMcpTaskId(result) !== taskId) throw new Error("MCP task result did not bind its exact related-task id.");
  if (!Array.isArray(result.content)) throw new Error("MCP task tool result has no content array."); result.content.forEach(validateMcpContentBlock);
  if (result.isError !== undefined && typeof result.isError !== "boolean") throw new Error("MCP task tool isError must be boolean.");
  if (result.structuredContent !== undefined) mcpTaskObject(result.structuredContent, "task structured result"); return structuredClone(result);
}
export function checkedMcpTaskSource(value: unknown): McpTaskSource {
  const source = mcpTaskObject(value, "task source");
  for (const key of ["brainId", "serverId", "toolId", "remoteName", "endpoint"] as const) text(source[key], `task ${key}`);
  if (Object.keys(source).some(key => !["brainId", "serverId", "toolId", "remoteName", "endpoint", "protocolVersion", "configurationSha256", "schemaSha256"].includes(key))) throw new Error("MCP task source cannot contain session/grant or unknown fields.");
  if (source.protocolVersion !== "2025-11-25" || !MCP_TASK_SHA.test(String(source.configurationSha256)) || !MCP_TASK_SHA.test(String(source.schemaSha256))) throw new Error("MCP task source protocol/checksum is invalid.");
  return source as unknown as McpTaskSource;
}
export function checkedMcpTaskRecord(value: unknown): McpTaskRecord {
  const row = mcpTaskObject(value, "task custody record");
  if (row.format !== "omni-mcp-task-record" || row.version !== 1 || row.wireVersion !== MCP_TASK_WIRE_VERSION || !MCP_TASK_REF.test(String(row.taskRef)) || row.authority !== "historical-not-live" ||
    !["prepared", "dispatch-may-have-started", "creation-outcome-unknown", "accepted", "rejected-before-task", "not-dispatched"].includes(String(row.creation)) || !MCP_TASK_SHA.test(String(row.argumentsSha256))) throw new Error("Invalid inert MCP task custody record.");
  checkedMcpTaskSource(row.source); time(row.createdAt); time(row.updatedAt); text(row.requestId, "task request id");
  for (const key of ["actionEventId", "chatTurnId", "reason"] as const) if (row[key] !== undefined) text(row[key], key);
  if (row.remote !== undefined) checkedMcpRemoteTask(row.remote);
  if (row.createResult !== undefined) { const remote = checkedMcpCreateTaskResult(row.createResult); if (!row.remote || remote.taskId !== (row.remote as McpRemoteTask).taskId) throw new Error("MCP task creation ACK lost its actual receiver id."); }
  if (row.creation === "accepted" && row.remote === undefined) throw new Error("An accepted MCP task is missing its actual server state.");
  if (row.lastOperation !== undefined) {
    const operation = mcpTaskObject(row.lastOperation, "task operation"); text(operation.id, "task operation id"); time(operation.at);
    if (!["tools/call", "tasks/get", "tasks/result", "tasks/cancel"].includes(String(operation.method)) || !["prepared", "dispatch-may-have-started", "acknowledged", "outcome-unknown"].includes(String(operation.state))) throw new Error("Invalid MCP task operation custody.");
  }
  if (row.result !== undefined) {
    const result = mcpTaskObject(row.result, "task outcome custody");
    if (!MCP_TASK_SHA.test(String(result.receiptId)) || !MCP_TASK_SHA.test(String(result.payloadSha256)) || !["result", "rpc-error"].includes(String(result.kind)) || row.remote === undefined) throw new Error("Invalid MCP task outcome receipt."); time(result.observedAt);
  }
  // Persisted data cannot smuggle a serialized management/session grant.
  if (Object.keys(row).some(key => !["format", "version", "wireVersion", "taskRef", "source", "createdAt", "updatedAt", "requestId", "actionEventId", "chatTurnId", "argumentsSha256", "creation", "remote", "createResult", "lastOperation", "result", "reason", "authority"].includes(key))) throw new Error("MCP task custody contains unknown or live authority fields.");
  return structuredClone(row) as unknown as McpTaskRecord;
}
