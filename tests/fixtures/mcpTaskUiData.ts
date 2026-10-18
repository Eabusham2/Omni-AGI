import type { McpServerSummary } from "../../src/shared/types";
import { MCP_TASK_WIRE_VERSION, type McpTaskLocalPage, type McpTaskReceipt, type McpTaskRecord } from "../../src/shared/mcpTasks";

export const MCP_UI_TASK_REF = "12345678-abcd-abcd-abcd-1234567890ab";
export const MCP_UI_TASK_TOOL = "mcp.synthetic.original-tool";
export function mcpUiTaskRecord(brainId = "task-ui-brain"): McpTaskRecord {
  return { format: "omni-mcp-task-record", version: 1, wireVersion: MCP_TASK_WIRE_VERSION, taskRef: MCP_UI_TASK_REF,
    source: { brainId, serverId: "synthetic", toolId: MCP_UI_TASK_TOOL, remoteName: "Synthetic long-running tool", endpoint: "stdio:synthetic-no-process",
      protocolVersion: "2025-11-25", configurationSha256: "a".repeat(64), schemaSha256: "b".repeat(64) },
    createdAt: "2026-10-17T00:00:00.000Z", updatedAt: "2026-10-17T00:00:01.000Z", requestId: "original-request", actionEventId: "original-action", argumentsSha256: "c".repeat(64),
    creation: "accepted", remote: { taskId: "source-original-server-task", status: "working", createdAt: "2026-10-17T00:00:00.000Z", lastUpdatedAt: "2026-10-17T00:00:01.000Z", ttl: null, pollInterval: 0 },
    authority: "historical-not-live" };
}
export function mcpUiTaskServer(brainId = "task-ui-brain"): McpServerSummary {
  return { brainId, id: "synthetic", label: "Synthetic connection", transport: "stdio", endpoint: "synthetic-no-process", connected: true,
    tools: [{ id: MCP_UI_TASK_TOOL, remoteName: "Synthetic long-running tool", taskSupport: "optional", actions: ["call", "call-task", "task-status", "task-result", "task-cancel"], inputSchema: { type: "object" } }],
    credentialConfigured: false, registryId: "synthetic", scope: "instance", globallyEnabled: true, defaultEnabled: true, accessOverride: "on", effectiveEnabled: true, protocolVersion: "2025-11-25" };
}
export function mcpUiTaskReceipt(record = mcpUiTaskRecord(), management: McpTaskReceipt["management"] = "current-session"): McpTaskReceipt {
  return { format: "omni-mcp-task", version: 1, wireVersion: MCP_TASK_WIRE_VERSION, record, management, operationComplete: false, untrustedTaskData: true };
}
export function mcpUiTaskPage(record = mcpUiTaskRecord()): McpTaskLocalPage {
  return { format: "omni-mcp-task-local-page", version: 1, brainId: record.source.brainId,
    records: [{ record, custodyKey: "a".repeat(64), management: "current-session" }], networkRequestMade: false };
}
