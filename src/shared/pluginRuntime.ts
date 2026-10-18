import type { ToolExecutionResult, ToolInvocation } from "./types";
import type { PluginRuntimeMediaKind, PluginRuntimeMediaOutputDeclaration } from "./plugins";
import type { HostActionOriginAuthority } from "./hostActionAuthority";

export const PLUGIN_RUNTIME_SCHEME = "omni-plugin-runtime" as const;
export const PLUGIN_RUNTIME_REQUEST_CHANNEL = "omni:plugin-runtime:request" as const;
export const PLUGIN_RUNTIME_DELIVERY_CHANNEL = "omni:plugin-runtime:delivery" as const;

export type PluginRuntimeContributionKind = "style" | "menu" | "button" | "panel";
export interface PluginRuntimeModuleDefinition {
  id: string;
  title: string;
  description: string;
  /** Immutable JavaScript body, not Node/main code or a repository setup script. */
  script: string;
  /** Host-published topics this module may subscribe to after explicit activation. */
  events: string[];
  /** Registration is generic, but restricted to the reviewed contribution kinds/slots. */
  ui: { kinds: PluginRuntimeContributionKind[]; slots: string[] };
  dynamicTools: boolean;
}

export interface PluginRuntimeContribution {
  id: string;
  kind: PluginRuntimeContributionKind;
  slot: string;
  title?: string;
  css?: string;
  html?: string;
  /** Name delivered back to this exact module; never an executable host command. */
  event?: string;
}
export interface PluginRuntimeMountedContribution extends PluginRuntimeContribution {
  brainId: string;
  pluginId: string;
  contentSha256: string;
  moduleId: string;
  sessionId: string;
  revision: number;
}

export interface PluginRuntimeToolDefinition {
  id: string;
  title: string;
  description: string;
  actions: Array<{ id: string; description: string; inputSchema: Record<string, unknown> }>;
}
export interface PluginRuntimeActivationRequest {
  brainId: string;
  pluginId: string;
  contentSha256: string;
  moduleId: string;
  expectedCatalogRevision: number;
  expectedInstanceRevision: number;
  requestId: string;
}
export interface PluginRuntimeSession {
  format: "omni-plugin-runtime-session";
  version: 1;
  brainId: string;
  pluginId: string;
  contentSha256: string;
  moduleId: string;
  sessionId: string;
  catalogRevision: number;
  instanceRevision: number;
  state: "starting" | "active" | "stopping" | "stopped" | "failed" | "dormant";
  startedAt: string;
  finishedAt?: string;
  error?: string;
  /** Absent means the ordinary explicitly activated feature runtime. */
  purpose?: "training-parser";
  parser?: {
    formatId: string;
    transactionKey: string;
    sourceSha256: string;
    sourceBytes: number;
    moduleSha256: string;
    inputMediaType?: string;
    /** Absent retains the original JSONL runtime-parser session contract. */
    output?: PluginRuntimeMediaOutputDeclaration;
  };
  /** Current-process diagnostics, never an imported grant. */
  processId?: number;
}
export interface PluginRuntimeResourceLease {
  maxMessageBytes: number;
  /** Must include module renderer and existing app family, not only JS heap. */
  check(processId?: number): Promise<void>;
  release(): Promise<void> | void;
}
export interface PluginRuntimeProcess {
  senderId: number;
  processId: number;
  send(message: PluginRuntimeDelivery): void;
  /** Factory must verify exclusive PID ownership before forcefully terminating. */
  terminate(): Promise<void>;
}
export interface PluginRuntimeLaunch {
  session: PluginRuntimeSession;
  url: string;
  /** Fresh private transport binding. Never portable or passed to Studio UI. */
  nonce: string;
  maxMessageBytes: number;
}
export interface PluginRuntimeProcessFactory {
  launch(input: PluginRuntimeLaunch, callbacks: {
    signal: AbortSignal;
    receive(senderId: number, frameUrl: string, message: unknown): Promise<PluginRuntimeRpcReply>;
    gone(reason: string): void;
  }): Promise<PluginRuntimeProcess>;
}

export interface PluginRuntimeRpcRequest {
  format: "omni-plugin-runtime-rpc";
  version: 1;
  sessionId: string;
  nonce: string;
  id: string;
  operation: "ready" | "call" | "state.get" | "state.read" | "state.put" | "register-tool" | "ui.put" | "ui.remove" | "tool-result" | "receipt.read" | "module-error" | "shutdown-ack" | "parser-ready" | "parser-result";
  arguments: Record<string, unknown>;
}
export interface PluginRuntimeRpcReply {
  format: "omni-plugin-runtime-reply";
  version: 1;
  id: string;
  state: "complete" | "failed" | "approval-required" | "uncertain" | "backpressure";
  value?: unknown;
  error?: string;
  receipt?: { id: string; bytes: number; sha256: string };
}
export interface PluginRuntimeDelivery {
  kind: "activate" | "deactivate" | "event" | "invoke-tool" | "rpc-reply" | "parser-step";
  id: string;
  topic?: string;
  data?: unknown;
  toolId?: string;
  action?: string;
  reply?: PluginRuntimeRpcReply;
  parserStep?: PluginRuntimeParserStep;
  /** Main-generated single-stop challenge; never portable or a public SDK method. */
  shutdown?: { format: "omni-plugin-runtime-shutdown"; version: 1; sessionId: string; nonce: string };
}

export interface PluginRuntimeTrainingParserActivationRequest {
  brainId: string;
  pluginId: string;
  contentSha256: string;
  moduleId: string;
  moduleSha256: string;
  formatId: string;
  transactionKey: string;
  sourceSha256: string;
  sourceBytes: number;
  inputMediaType?: string;
  /** Absent retains the original JSONL runtime-parser session contract. */
  output?: PluginRuntimeMediaOutputDeclaration;
  expectedCatalogRevision: number;
  expectedInstanceRevision: number;
  requestId: string;
}

export interface PluginRuntimeParserStep {
  id: string;
  operation: "push" | "drain";
  sequence: number;
  maxOutputBytes: number;
  inputBase64?: string;
  final?: boolean;
}

export type PluginRuntimeParserStepResult =
  | { state: "need-input" | "done" }
  | { state: "output"; outputBase64: string }
  | { state: "asset-start"; assetId: string; kind: PluginRuntimeMediaKind; mimeType: string }
  | { state: "asset-chunk"; assetId: string; offset: number; bytesBase64: string }
  | { state: "asset-end"; assetId: string; bytes: number; sha256: string };

export interface PluginRuntimeParserProgress {
  format: "omni-runtime-parser-progress";
  version: 1;
  brainId: string;
  pluginId: string;
  contentSha256: string;
  moduleId: string;
  formatId: string;
  transactionKey: string;
  state: "starting" | "streaming" | "waiting-for-approval" | "finalizing" | "complete" | "failed" | "cancelled";
  inputBytes: number;
  sourceBytes: number;
  outputBytes: number;
  outputRecords: number;
  outputAssets?: number;
  message: string;
  approval?: Pick<PluginRuntimePendingApproval, "sessionId" | "rpcId" | "reviewId" | "executionId"> & {
    toolId: string;
    action: string;
  };
}

export interface PluginRuntimeShutdownAck {
  callbacks: number;
  completed: number;
  failed: number;
  errors: Array<{ callback: number; message: string }>;
}
export interface PluginRuntimeShutdownDiagnostic {
  format: "omni-plugin-runtime-shutdown-diagnostic";
  version: 1;
  sessionId: string;
  requestId: string;
  state: "acknowledged" | "timeout" | "not-started" | "send-failed";
  deadlineMs: number;
  elapsedMs: number;
  /** Self-report from this owned bootstrap, not hostile-module attestation. */
  acknowledgement?: PluginRuntimeShutdownAck;
  error?: string;
}

export interface PluginRuntimePendingApproval {
  brainId: string;
  sessionId: string;
  rpcId: string;
  reviewId: string;
  pluginId: string;
  contentSha256: string;
  moduleId: string;
  invocation: ToolInvocation;
  executionId: string;
  /** Historical/presentation fields only; approval token never enters this type. */
  details?: unknown;
}
export interface PluginRuntimeReviewRequest {
  brainId: string;
  sessionId: string;
  rpcId: string;
  reviewId: string;
  decision: "approve" | "deny";
}
export interface PluginRuntimeEvent {
  brainId: string;
  kind: "session" | "contributions" | "tools" | "approval";
  session?: PluginRuntimeSession;
}

export interface PluginRuntimeSessionPage { sessions: PluginRuntimeSession[]; total: number; nextRowid?: number }
export interface PluginRuntimeHistoryPage { events: Array<{ sequence: number; kind: string; value?: unknown; sha256: string; receipt?: { bytes: number } }>; nextSequence?: number; total: number }
export interface PluginRuntimeStateSourcePage { sources: Array<{ sha256: string; owner: string; bytes: number }>; nextSha256?: string; total: number }
export interface PluginRuntimeStateConflictPage { conflicts: Array<{ pluginHash: string; moduleId: string; key: string }>; nextKey?: string; total: number }
export interface PluginRuntimeStateSelection {
  brainId: string; sourceSha256: string; pluginHash: string; moduleId: string; key: string; expectedRevision: number;
}
export interface PluginRuntimeStateValue { revision: number; value?: unknown; receipt?: { bytes: number; sha256: string } }

export interface PluginRuntimeApi {
  activate(request: PluginRuntimeActivationRequest): Promise<PluginRuntimeSession>;
  stop(brainId: string, sessionId: string): Promise<PluginRuntimeSession>;
  list(brainId: string, page?: { afterRowid?: number; maxBytes?: number }): Promise<PluginRuntimeSessionPage>;
  history(brainId: string, page?: { sourceSha256?: string; afterSequence?: number; maxBytes?: number }): Promise<PluginRuntimeHistoryPage>;
  readHistoryEvent(brainId: string, sequence: number, offset: number, length: number, sourceSha256?: string): Promise<{ text: string; bytes: number; sha256: string; nextOffset?: number }>;
  stateSources(brainId: string, page?: { afterSha256?: string; limit?: number }): Promise<PluginRuntimeStateSourcePage>;
  stateConflicts(brainId: string, page?: { afterKey?: string; limit?: number }): Promise<PluginRuntimeStateConflictPage>;
  readSourceState(brainId: string, sourceSha256: string, pluginHash: string, moduleId: string, key: string, maxBytes?: number): Promise<PluginRuntimeStateValue>;
  readSourceStateBytes(brainId: string, sourceSha256: string, pluginHash: string, moduleId: string, key: string, offset: number, length: number): Promise<{ text: string; bytes: number; sha256: string; nextOffset?: number }>;
  /** Human-reviewed selection only; a module cannot resolve its own inherited conflicts. */
  selectSourceState(request: PluginRuntimeStateSelection): Promise<void>;
  contributions(brainId: string): Promise<PluginRuntimeMountedContribution[]>;
  invokeContribution(brainId: string, sessionId: string, contributionId: string, eventId: string, data?: unknown): Promise<void>;
  approvals(brainId: string): Promise<PluginRuntimePendingApproval[]>;
  review(request: PluginRuntimeReviewRequest): Promise<PluginRuntimeRpcReply>;
  onEvent(listener: (event: PluginRuntimeEvent) => void): () => void;
}

export interface PluginRuntimeToolExecutor {
  execute(
    invocation: ToolInvocation,
    onProgress?: undefined,
    requestId?: string,
    onCompleted?: ((result: ToolExecutionResult) => Promise<void>) | undefined,
    context?: { actionEventId: string; originAuthority: HostActionOriginAuthority }
  ): Promise<ToolExecutionResult>;
  cancelAndWait(brainId: string, requestId?: string): Promise<number>;
  invalidateApproval(brainId: string, token: string): boolean;
}
export interface PluginRuntimeSchemaLearning {
  brainId: string;
  pluginId: string;
  moduleId: string;
  contentSha256: string;
  sourceSha256: string;
  text: string;
}
export interface PluginRuntimeSchemaReceipt { sourceSha256: string; receipt: unknown }
export interface PluginRuntimePublishedSchema {
  id: string;
  actions: string[];
  inputSchema?: Record<string, unknown>;
  pluginRuntimeSourceSha256: string;
}
export interface PluginRuntimePublishedTool {
  toolId: string;
  definition: PluginRuntimeToolDefinition;
  schema: PluginRuntimePublishedSchema;
  sourceSha256: string;
}
