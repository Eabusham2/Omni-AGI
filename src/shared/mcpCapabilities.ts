/** Client-owned RPC operations. These are not remote tools or persona prompts. */
export type McpClientOperation =
  | "resources/list"
  | "resources/templates/list"
  | "resources/read"
  | "resources/subscribe"
  | "resources/unsubscribe"
  | "prompts/list"
  | "prompts/get"
  | "completion/complete"
  | "logging/setLevel"
  | "tasks/list";

export interface McpNegotiatedCapabilities {
  tools?: { listChanged?: boolean };
  resources?: { subscribe?: boolean; listChanged?: boolean };
  prompts?: { listChanged?: boolean };
  completions?: Record<string, never>;
  logging?: Record<string, never>;
  tasks?: { list?: Record<string, never>; cancel?: Record<string, never>;
    requests?: { tools?: { call?: Record<string, never> } } };
}

export interface McpResource {
  uri: string;
  name: string;
  title?: string;
  description?: string;
  mimeType?: string;
  size?: number;
  icons?: Array<Record<string, unknown>>;
  annotations?: Record<string, unknown>;
  _meta?: Record<string, unknown>;
}

export interface McpResourceTemplate extends Omit<McpResource, "uri" | "size"> {
  uriTemplate: string;
}

export type McpResourceContents = {
  uri: string;
  mimeType?: string;
  _meta?: Record<string, unknown>;
} & ({ text: string; blob?: never } | { blob: string; text?: never });

export interface McpPrompt {
  name: string;
  title?: string;
  description?: string;
  arguments?: Array<{ name: string; description?: string; required?: boolean }>;
  icons?: Array<Record<string, unknown>>;
  _meta?: Record<string, unknown>;
}

export type McpContentBlock =
  | { type: "text"; text: string; annotations?: Record<string, unknown>; _meta?: Record<string, unknown> }
  | { type: "image" | "audio"; data: string; mimeType: string; annotations?: Record<string, unknown>; _meta?: Record<string, unknown> }
  | { type: "resource"; resource: McpResourceContents; annotations?: Record<string, unknown>; _meta?: Record<string, unknown> }
  | (McpResource & { type: "resource_link" });

export interface McpResourcePage {
  resources: McpResource[];
  nextCursor?: string;
  _meta?: Record<string, unknown>;
}
export interface McpResourceTemplatePage {
  resourceTemplates: McpResourceTemplate[];
  nextCursor?: string;
  _meta?: Record<string, unknown>;
}
export interface McpPromptPage {
  prompts: McpPrompt[];
  nextCursor?: string;
  _meta?: Record<string, unknown>;
}
export interface McpReadResourceResult {
  contents: McpResourceContents[];
  _meta?: Record<string, unknown>;
}
export interface McpGetPromptResult {
  description?: string;
  messages: Array<{ role: "user" | "assistant"; content: McpContentBlock }>;
  _meta?: Record<string, unknown>;
}
export interface McpCompleteRequest {
  ref: { type: "ref/prompt"; name: string } | { type: "ref/resource"; uri: string };
  argument: { name: string; value: string };
  context?: { arguments: Record<string, string> };
}
export interface McpCompleteResult {
  completion: { values: string[]; total?: number; hasMore?: boolean };
  _meta?: Record<string, unknown>;
}

export interface McpOperationResults {
  "resources/list": McpResourcePage;
  "resources/templates/list": McpResourceTemplatePage;
  "resources/read": McpReadResourceResult;
  "resources/subscribe": Record<string, unknown>;
  "resources/unsubscribe": Record<string, unknown>;
  "prompts/list": McpPromptPage;
  "prompts/get": McpGetPromptResult;
  "completion/complete": McpCompleteResult;
  "logging/setLevel": Record<string, unknown>;
  "tasks/list": { tasks: import("./mcpTasks").McpRemoteTask[]; nextCursor?: string; _meta?: Record<string, unknown> };
}

/** Preserve server messages as data; a role here never becomes a host role. */
export interface McpClientOperationResult<O extends McpClientOperation = McpClientOperation> {
  format: "omni-mcp-client-result";
  version: 1;
  operation: O;
  result: McpOperationResults[O];
  provenance: {
    brainId: string;
    serverId: string;
    transport: "http" | "stdio";
    endpoint: string;
    protocolVersion: string;
    observedAt: string;
    requestedUri?: string;
    requestedPrompt?: string;
  };
  untrustedTaskData: true;
  hiddenPromptUsed: false;
}

export interface McpResourceSubscription {
  uri: string;
  state: "active" | "connection-lost";
  updates: number;
  updatedAt?: string;
}

export interface McpCapabilityEvent {
  brainId: string;
  serverId: string;
  revision: number;
  observedAt: string;
  kind: "resources-list-changed" | "prompts-list-changed" | "resource-updated" | "subscriptions-lost"
    | "connection-ready" | "resource-subscribed" | "resource-unsubscribed" | "tools-list-changed";
  uri?: string;
}

export interface McpSuspensionReceipt {
  brainId: string;
  disconnectedServerIds: string[];
  lostResourceSubscriptions: Array<{ serverId: string; uri: string }>;
  reconnectRequired: string[];
}

export interface McpTransportObservation {
  httpMode?: "streamable-http" | "legacy-sse";
  eventStream: "off" | "connecting" | "listening" | "reconnecting" | "unsupported" | "error";
  lastSessionTermination?: "pending" | "complete" | "already-ended" | "unsupported" | "failed";
  error?: string;
}
