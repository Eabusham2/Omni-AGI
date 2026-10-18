import type { McpRequestOrigin } from "./mcpInteractions";
import type { ToolPermissionLevel } from "./types";

export const MCP_LOG_LEVELS = ["debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"] as const;
export type McpLogLevel = typeof MCP_LOG_LEVELS[number];
export interface McpRoot { uri: string; name?: string; }

/** Main-derived connection identity. This object never authorizes a renderer. */
export interface McpHostConnection extends McpRequestOrigin {
  configurationSha256: string;
  /** Fresh main identity on each transport epoch, including remove/re-add. */
  connectionId: string;
  ready: boolean;
  enabled: boolean;
}
export interface McpRootDisclosureReview {
  roots: McpRoot[];
  selectionSha256: string;
  revision: number;
  policyRevision: string;
  disclosureOnly: true;
  filesystemSandbox: false;
}
export interface McpRootsStatus {
  format: "omni-mcp-roots-status";
  version: 1;
  brainId: string;
  serverId: string;
  endpoint: string;
  revision: number;
  roots: Array<McpRoot & { accessible: boolean; reason?: string }>;
  state: "empty" | "review-required" | "shared";
  connectionReady: boolean;
  sharedConnectionEpoch?: number;
  notification?: "sent" | "not-ready" | "failed";
  disclosureOnly: true;
  filesystemSandbox: false;
  permissionsPortable: false;
}
export interface McpRootCommand { brainId: string; serverId: string; requestId: string; }
/** Paths come from a trusted main picker, never these renderer DTOs. */
export interface McpRootsApi {
  status(brainId: string, serverId: string): Promise<McpRootsStatus>;
  select(request: McpRootCommand): Promise<McpRootsStatus>;
  review(request: McpRootCommand): Promise<McpRootsStatus>;
  clear(request: McpRootCommand): Promise<McpRootsStatus>;
}
export interface McpFileDisclosurePolicy { level: ToolPermissionLevel; revision: string; }

/** Operational debugging only: no native memory, hidden prompt or export grant. */
export interface McpDiagnosticRecord {
  format: "omni-mcp-diagnostic";
  version: 1;
  sequence: number;
  observedAt: string;
  origin: McpRequestOrigin;
  source: "server" | "app";
  viewId?: string;
  level: McpLogLevel;
  logger?: string;
  data: unknown;
  sourceSha256: string;
  redactionApplied: boolean;
  untrustedData: true;
  nativeLearning: false;
}
export interface McpDiagnosticPage {
  records: McpDiagnosticRecord[];
  nextSequence?: number;
  hasMore: boolean;
  retainedBytes: number;
  accepted: number;
  rejected: number;
  rateLimited: number;
  evicted: number;
  retention: "bounded-current-process-only";
}
export interface McpDiagnosticsApi {
  page(brainId: string, serverId: string, afterSequence?: number, limit?: number): Promise<McpDiagnosticPage>;
}

export type McpAppServerNotificationMethod = "notifications/tools/list_changed" | "notifications/resources/list_changed";
/** Token is main-authored for the exact current ready view, never remote data. */
export interface McpAppServerNotification {
  brainId: string;
  serverId: string;
  viewId: string;
  viewToken: string;
  connectionEpoch: number;
  sequence: number;
  method: McpAppServerNotificationMethod;
  params: Record<string, never>;
}
