export const MCP_APP_SCHEME = "omni-mcp-app";
export const MCP_APP_EXTENSION = "io.modelcontextprotocol/ui";
export const MCP_APP_MIME = "text/html;profile=mcp-app";
export const MCP_APP_PROTOCOL = "2026-01-26";
export const MCP_APP_DISPLAY_MODES = ["inline", "fullscreen"] as const;
export type McpAppHostDisplayMode = typeof MCP_APP_DISPLAY_MODES[number];
export type McpAppDisplayMode = McpAppHostDisplayMode | "pip";
export interface McpAppDisplayContext {
  displayMode: McpAppHostDisplayMode;
  availableDisplayModes: McpAppHostDisplayMode[];
}
export interface McpAppDisplayIdentity { brainId: string; viewId: string; viewToken: string; }
/** Main-authored command/state for an exact ephemeral view; never a device or tool grant. */
export interface McpAppDisplayEvent extends McpAppDisplayIdentity, McpAppDisplayContext {
  sequence: number;
  action: "apply" | "context" | "closed";
  operationId?: string;
  reason?: string;
}
export interface McpAppDisplayAcknowledgement extends McpAppDisplayIdentity {
  operationId: string;
  mode: McpAppHostDisplayMode;
}

export interface McpAppCsp {
  connectDomains: string[];
  resourceDomains: string[];
  frameDomains: string[];
  baseUriDomains: string[];
}
export interface McpAppDescriptor {
  id: string;
  brainId: string;
  serverId: string;
  serverLabel: string;
  resourceUri: string;
  toolId: string;
  remoteToolName: string;
  observedAt: string;
  /** Untrusted remote UI, not a Studio renderer or a native capability. */
  untrusted: true;
}
export interface McpAppLaunch {
  view: McpAppDescriptor;
  proxyUrl: string;
  proxyOrigin: string;
  viewToken: string;
  resourceUrl: string;
  resourceSha256: string;
  resourceHtml: string;
  csp: McpAppCsp;
  /** The original, completed call, not a new or replayed tool execution. */
  toolInput: Record<string, unknown>;
  toolResult: Record<string, unknown>;
  hostContext: McpAppDisplayContext & { theme: "light" | "dark" };
}
export interface McpAppRequest {
  brainId: string;
  viewId: string;
  viewToken: string;
  id: string | number;
  method: string;
  params?: Record<string, unknown>;
  /** Set by the trusted Studio relay only for a JSON-RPC message with no ID. */
  notification?: true;
}
export interface McpAppHostResult {
  result?: unknown;
  error?: { code: number; message: string };
}
export interface McpAppObservationEvent {
  brainId: string;
  artifactId: string;
  receipt?: import("./mcpAppTaskData").McpAppObservationReceipt;
  error?: string;
}
export interface McpAppsApi {
  describe(brainId: string, viewId: string): Promise<import("./mcpAppArtifacts").McpAppArtifactSummary | undefined>;
  open(brainId: string, viewId: string, selection?: { captureSha256: string; reviewedMapping: true }): Promise<McpAppLaunch>;
  request(request: McpAppRequest): Promise<McpAppHostResult>;
  close(brainId: string, viewId: string, viewToken: string): Promise<void>;
  /** Read-only projection. This listener cannot submit data or grant access. */
  onObservation?(listener: (event: McpAppObservationEvent) => void): () => void;
  /** Main-authored notifications for one exact ready view; no dispatch grant. */
  onServerNotification?(listener: (event: import("./mcpHostServices").McpAppServerNotification) => void): () => void;
  /** Exact-view presentation only. This channel never invokes a server action. */
  onDisplayMode?(listener: (event: McpAppDisplayEvent) => void): () => void;
  acknowledgeDisplayMode?(request: McpAppDisplayAcknowledgement): Promise<void>;
}
