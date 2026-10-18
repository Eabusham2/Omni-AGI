/** This adapter is only for the negotiated dated protocol, not the incompatible 2026 draft. */
export const MCP_TASK_WIRE_VERSION = "mcp-2025-11-25" as const;
export type McpTaskStatus = "working" | "input_required" | "completed" | "failed" | "cancelled";
export interface McpRemoteTask {
  taskId: string;
  status: McpTaskStatus;
  statusMessage?: string;
  createdAt: string;
  lastUpdatedAt: string;
  ttl: number | null;
  pollInterval?: number;
  [key: string]: unknown;
}
export interface McpTaskSource {
  brainId: string;
  serverId: string;
  toolId: string;
  remoteName: string;
  endpoint: string;
  protocolVersion: "2025-11-25";
  configurationSha256: string;
  schemaSha256: string;
}
export type McpTaskCreationState = "prepared" | "dispatch-may-have-started" | "creation-outcome-unknown" | "accepted" | "rejected-before-task" | "not-dispatched";
export interface McpTaskRecord {
  format: "omni-mcp-task-record";
  version: 1;
  wireVersion: typeof MCP_TASK_WIRE_VERSION;
  taskRef: string;
  source: McpTaskSource;
  createdAt: string;
  updatedAt: string;
  requestId: string;
  actionEventId?: string;
  chatTurnId?: string;
  argumentsSha256: string;
  creation: McpTaskCreationState;
  remote?: McpRemoteTask;
  createResult?: Record<string, unknown>;
  lastOperation?: { id: string; method: "tools/call" | "tasks/get" | "tasks/result" | "tasks/cancel"; state: "prepared" | "dispatch-may-have-started" | "acknowledged" | "outcome-unknown"; at: string };
  result?: { receiptId: string; payloadSha256: string; kind: "result" | "rpc-error"; observedAt: string };
  reason?: string;
  /** Historical receipt only. A live task/session/permission lease is never serialized. */
  authority: "historical-not-live";
}
export interface McpTaskReceipt {
  format: "omni-mcp-task";
  version: 1;
  wireVersion: typeof MCP_TASK_WIRE_VERSION;
  record: McpTaskRecord;
  operationComplete: false;
  management: "current-session" | "dormant";
  reason?: string;
  untrustedTaskData: true;
}
export interface McpTaskOutcome {
  format: "omni-mcp-task-outcome";
  version: 1;
  wireVersion: typeof MCP_TASK_WIRE_VERSION;
  taskRef: string;
  source: McpTaskSource;
  taskResultReceiptId: string;
  kind: "result" | "rpc-error";
  /** Exact decoded MCP payload; not a claim of raw HTTP wire preservation. */
  result?: Record<string, unknown>;
  error?: { code: number; message: string; data?: unknown };
  payloadSha256: string;
  retained: boolean;
  /** Separately normalized host presentation, not the immutable source payload. */
  observedResult?: Record<string, unknown>;
  untrustedTaskData: true;
}
export interface McpTaskLocalPage {
  format: "omni-mcp-task-local-page";
  version: 1;
  brainId: string;
  records: Array<{ record: McpTaskRecord; custodyKey: string; management: "current-session" | "dormant"; reason?: string }>;
  nextCursor?: string;
  networkRequestMade: false;
}
export interface McpTasksApi {
  localPage(brainId: string, after?: string, limit?: number): Promise<McpTaskLocalPage>;
  stopWait(brainId: string, taskRef: string): Promise<boolean>;
}
