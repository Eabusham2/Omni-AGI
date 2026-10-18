import { createHash, randomUUID } from "node:crypto";
import { McpTaskCustodyError } from "./mcpTaskFailure";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { stat } from "node:fs/promises";
import { isAbsolute } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import Ajv from "ajv";
import Ajv2019 from "ajv/dist/2019.js";
import Ajv2020 from "ajv/dist/2020.js";
import type { ValidateFunction } from "ajv";
import type {
  McpServerRegistrationRequest,
  McpServerSummary,
  McpToolSummary,
  McpRegistrySummary,
  McpRegistryUpdateRequest
} from "../shared/types";
import type { BrainService, ExternalToolSchema } from "./brainService";
import type { SecureSecretStore } from "./secureSecretStore";
import {
  McpRegistryStore, attachMcpInstance, mcpAccess, updateMcpAccess,
  type McpRegistryDocument, type McpRegistryServer
} from "./mcpRegistry";
import { readMcpRpcResponse, parseMcpRpcMessages, McpResumableStreamError, type McpRpcMessage } from "./mcpStream";
import { readMcpEvents, McpEventStreamIoError, type McpSseEvent } from "./mcpEventStream";
import { McpLegacySseSession } from "./mcpLegacySse";
import { McpRequestDeadline } from "./mcpRequestDeadline";
import { McpOAuthService } from "./mcpOAuth";
import type { McpCredentialUpdate, McpOAuthStart, McpOAuthStatus } from "../shared/mcpAuth";
import { mcpEnvironment, mcpEnvironmentKey } from "./mcpEnvironment";
import type { ToolExecutionContext } from "./learningExecutionContext";
import type { McpElicitationResult, McpRequestOrigin } from "../shared/mcpInteractions";
import { McpHostRequestError, mcpElicitation } from "./mcpInteractionProtocol";
import type { McpAppHost } from "./mcpApps";
import { mcpToolUi } from "./mcpAppMetadata";
import { MCP_APP_EXTENSION, MCP_APP_MIME } from "../shared/mcpApps";
import type { McpHostConnection, McpRoot } from "../shared/mcpHostServices";
import type {
  McpCapabilityEvent, McpClientOperation, McpClientOperationResult,
  McpNegotiatedCapabilities, McpResourceSubscription, McpSuspensionReceipt, McpTransportObservation
} from "../shared/mcpCapabilities";
import {
  mcpClientOperationTools, mcpOperationSupported, mcpUri,
  negotiatedMcpCapabilities, validateMcpOperationResult
} from "./mcpCapabilities";
import type { McpTaskStore } from "./mcpTaskStore";
import { MCP_TASK_WIRE_VERSION, type McpTaskRecord, type McpTaskReceipt, type McpTaskLocalPage, type McpTaskOutcome } from "../shared/mcpTasks";
import { checkedMcpCreateTaskResult, checkedMcpRemoteTask, checkedMcpTaskToolResult, isMcpTaskTerminal, mcpTaskObject, mcpTaskSha, mcpTaskSupport, relatedMcpTaskId, MCP_TASK_REF } from "./mcpTaskProtocol";

class McpAuthenticationRequiredError extends Error {
  constructor() { super("MCP server requires authentication. Sign in or configure a server credential."); }
}

type FetchLike = typeof fetch;

interface StoredMcpServer {
  brainId: string;
  id: string;
  label: string;
  transport: "http" | "stdio";
  url?: string;
  command?: string;
  args?: string[];
  tools: McpToolSummary[];
  capabilities?: McpNegotiatedCapabilities;
  namespaceId?: string;
}

interface JsonRpcResponse {
  jsonrpc?: string;
  id?: string | number | null;
  result?: unknown;
  error?: { code?: number; message?: string; data?: unknown };
}

class McpSessionExpiredError extends Error {
  constructor() { super("The MCP server expired this session."); }
}

interface McpRuntime {
  config: StoredMcpServer;
  summary: McpServerSummary;
  sessionId?: string;
  initialized: boolean;
  stdio?: StdioRpcSession;
  legacySse?: McpLegacySseSession;
  controller: AbortController;
  protocolVersion?: string;
  capabilities?: Record<string, unknown>;
  toolsDirty?: boolean;
  toolRevision?: number;
  schemasRegistered?: boolean;
  initialization?: Promise<void>;
  refresh?: Promise<McpServerSummary>;
  learning?: { hash: string; promise: Promise<void> };
  credentialsUpdating?: boolean;
  capabilityRevision?: number;
  subscriptions?: Map<string, McpResourceSubscription>;
  subscribing?: Map<string, { updates: number; updatedAt?: string }>;
  pagination?: Map<McpClientOperation, Map<string, string>>;
  completionNextAt?: number;
  sessionEpoch?: number;
  hostConnectionId?: string;
  sessionToken?: string;
  deliveredEventIds?: Set<string>;
  httpEvents?: { controller: AbortController; operation: Promise<void> };
  transportObservation?: McpTransportObservation;
  termination?: Promise<void>;
  terminationOwner?: object;
  activeCalls?: Map<string, McpActiveCall>;
  serverRequests?: Map<string, { controller: AbortController; parentId?: string; hash: string }>;
}

interface McpActiveCall {
  id: string;
  method: string;
  signal: AbortSignal;
  context?: McpRpcContext;
  deadline?: McpRequestDeadline;
}
export interface McpExecutionContext extends Pick<ToolExecutionContext, "chatTurnId" | "actionEventId" | "mcpAppViewId"> {
  requestId?: string;
  requireCurrentPermission?: () => Promise<void>;
}
interface McpRpcContext extends McpExecutionContext {
  beforeSend?: () => Promise<void>;
  task?: { taskRef: string; remoteTaskId?: string };
}
interface McpTaskLease {
  runtime: McpRuntime;
  connection: McpHostConnection;
  record: McpTaskRecord;
  tool: McpToolSummary;
  arguments: Record<string, unknown>;
  controller: AbortController;
  scope?: McpActiveCall;
  acknowledgement: Promise<void>;
  acknowledged(): void;
  rejectAcknowledgement(error: unknown): void;
  inputs: Set<Promise<unknown>>;
  busy?: boolean;
}
class McpRpcFailure extends Error {
  constructor(readonly method: string, readonly rpcError: { code: number; message: string; data?: unknown }) {
    super(`MCP ${method} failed: ${rpcError.message.slice(0, 1_000)}`);
  }
}
export interface McpHostInteractions {
  elicit?(origin: McpRequestOrigin, params: unknown, signal: AbortSignal): Promise<McpElicitationResult>;
  sample?(origin: McpRequestOrigin, params: unknown, signal: AbortSignal): Promise<Record<string, unknown>>;
  completeElicitation?(origin: McpRequestOrigin, elicitationId: string): Promise<void>;
}
export interface McpStandardHostServices {
  roots?(origin: McpRequestOrigin, signal: AbortSignal): Promise<{ roots: McpRoot[] }>;
  log?(origin: McpRequestOrigin, params: unknown, signal: AbortSignal): Promise<void>;
  connectionRetired?(brainId: string, serverId: string): void;
}

const MCP_PROTOCOL_VERSION = "2025-11-25";
const SUPPORTED_MCP_PROTOCOL_VERSIONS = new Set(["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]);
const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/;
const INTERNAL_TOOL_ID = /^[a-z][a-z0-9.-]{1,79}$/;
const MAX_INTERNAL_TOOL_ID_LENGTH = 80;
const MAX_MCP_MESSAGE_BYTES = 16 * 1024 * 1024;

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function runtimeKey(brainId: string, serverId: string): string {
  return `${brainId}:${serverId}`;
}

function safeId(value: unknown, label: string): string {
  if (typeof value !== "string" || !SAFE_ID.test(value)) {
    throw new Error(`Invalid MCP ${label}.`);
  }
  return value;
}

function safeLabel(value: unknown): string {
  if (typeof value !== "string") throw new Error("Invalid MCP server label.");
  const clean = value.replace(/[\0\r\n]/g, " ").trim().slice(0, 120);
  if (!clean) throw new Error("MCP server label is required.");
  return clean;
}

function record(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function compileToolInputValidator(schema: Record<string, unknown>): ValidateFunction<unknown> {
  // MCP 2025-06-18 requires an object-root inputSchema. Never turn a missing
  // or malformed schema into an unconstrained pass-through at call time.
  if (schema.type !== "object") {
    throw new Error("MCP tool input schema must have an object root.");
  }
  const dialect = schema.$schema;
  const options = {
    strict: false,
    validateFormats: false,
    coerceTypes: false,
    removeAdditional: false,
    useDefaults: false
  };
  let validator: Ajv | Ajv2019 | Ajv2020;
  if (dialect === undefined || dialect === "https://json-schema.org/draft/2020-12/schema" || dialect === "https://json-schema.org/draft/2020-12/schema#") {
    validator = new Ajv2020(options);
  } else if (dialect === "https://json-schema.org/draft/2019-09/schema" || dialect === "https://json-schema.org/draft/2019-09/schema#") {
    validator = new Ajv2019(options);
  } else if (dialect === "http://json-schema.org/draft-07/schema#" || dialect === "https://json-schema.org/draft-07/schema#" || dialect === "http://json-schema.org/draft-07/schema" || dialect === "https://json-schema.org/draft-07/schema") {
    validator = new Ajv(options);
  } else {
    throw new Error("MCP tool input schema uses an unsupported JSON Schema dialect.");
  }
  try {
    // Compile synchronously: remote $refs are never fetched from a tool server.
    return validator.compile(schema);
  } catch {
    throw new Error("MCP tool input schema is invalid or cannot be resolved locally.");
  }
}

function validateHttpUrl(value: unknown): string {
  if (typeof value !== "string" || value.length > 16_000) {
    throw new Error("MCP HTTP endpoint is invalid.");
  }
  const url = new URL(value);
  if (url.username || url.password || url.hash) {
    throw new Error("MCP endpoints cannot contain credentials or fragments.");
  }
  const loopback = ["localhost", "127.0.0.1", "::1", "[::1]"].includes(url.hostname);
  if (url.protocol !== "https:" && !(url.protocol === "http:" && loopback)) {
    throw new Error("MCP endpoints require HTTPS, except loopback HTTP servers.");
  }
  return url.toString();
}

function namespaceTool(brainId: string, serverId: string, remoteName: string): string {
  const brainSegment = sha256(brainId).slice(0, 8);
  const rawServerSlug = serverId
    .toLocaleLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "") || "server";
  // Keep already-compatible short server ids stable. When an accepted MCP
  // server id contains uppercase/underscores or is long, retain a readable
  // prefix and bind it to the exact id with a hash. This prevents two distinct
  // remote ids from collapsing onto the same internal namespace.
  const serverSegment = rawServerSlug === serverId && rawServerSlug.length <= 29
    ? rawServerSlug
    : `${rawServerSlug.slice(0, 20).replace(/-+$/g, "") || "server"}-${sha256(serverId).slice(0, 8)}`;
  const rawToolSlug = remoteName
    .toLocaleLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "") || "tool";
  const remoteHash = sha256(remoteName).slice(0, 8);
  const fixedLength = `mcp.${brainSegment}.${serverSegment}.-${remoteHash}`.length;
  const slugLength = Math.min(42, MAX_INTERNAL_TOOL_ID_LENGTH - fixedLength);
  const toolSlug = rawToolSlug.slice(0, slugLength).replace(/-+$/g, "") || "tool";
  const id = `mcp.${brainSegment}.${serverSegment}.${toolSlug}-${remoteHash}`;
  if (!INTERNAL_TOOL_ID.test(id)) {
    throw new Error("MCP tool could not be mapped to a valid internal tool id.");
  }
  return id;
}

function namespaceClientOperation(brainId: string, serverId: string, identity: string): string {
  if (!identity.startsWith("\0omni-client-operation:")) throw new Error("Invalid client-owned MCP operation identity.");
  const base = namespaceTool(brainId, serverId, identity);
  const separator = base.lastIndexOf(".");
  const prefix = `${base.slice(0, separator + 1)}client.`;
  const hashedSuffix = base.slice(-9);
  const slug = base.slice(separator + 1, -9).slice(0, MAX_INTERNAL_TOOL_ID_LENGTH - prefix.length - hashedSuffix.length).replace(/-+$/g, "") || "rpc";
  const id = `${prefix}${slug}${hashedSuffix}`;
  if (!INTERNAL_TOOL_ID.test(id)) throw new Error("MCP client operation could not be mapped to an internal id.");
  // Remote names always produce a single dot-free last segment. This additional
  // structural segment separates local RPC wrappers even if hash values collide.
  return id;
}

function encodeRpcMessage(value: Record<string, unknown>): string {
  const encoded = JSON.stringify(value);
  if (Buffer.byteLength(encoded, "utf8") > MAX_MCP_MESSAGE_BYTES) {
    throw new Error("MCP request exceeded 16 MiB.");
  }
  return encoded;
}

class StdioRpcSession {
  private child?: ChildProcessWithoutNullStreams;
  private starting?: Promise<void>;
  private stopped = false;
  private buffer = "";
  private stderr = "";
  private readonly pending = new Map<string, {
    resolve: (value: JsonRpcResponse) => void;
    reject: (error: Error) => void;
    deadline: McpRequestDeadline;
  }>();

  constructor(private readonly command: string, private readonly args: readonly string[],
    private readonly onMessage: (message: McpRpcMessage) => void | Promise<void>,
    private readonly onClosed: () => void,
    private readonly environment: () => Promise<Record<string, string>>) {}

  start(): Promise<void> {
    if (this.stopped) return Promise.reject(new Error("MCP stdio server stopped."));
    if (this.child && !this.child.killed && this.child.exitCode === null && this.child.signalCode === null) return Promise.resolve();
    if (this.starting) return this.starting;
    const operation = this.launch();
    this.starting = operation;
    void operation.finally(() => { if (this.starting === operation) this.starting = undefined; }).catch(() => undefined);
    return operation;
  }

  private async launch(): Promise<void> {
    const info = await stat(this.command);
    if (this.stopped) throw new Error("MCP stdio server stopped.");
    if (!info.isFile()) throw new Error("MCP stdio command is not an executable file.");
    const minimalEnvironment: NodeJS.ProcessEnv = {
      PATH: process.env.PATH,
      SystemRoot: process.env.SystemRoot,
      TEMP: process.env.TEMP,
      TMP: process.env.TMP,
      TMPDIR: process.env.TMPDIR,
      LANG: process.env.LANG
    };
    const explicitEnvironment = await this.environment();
    if (this.stopped) throw new Error("MCP stdio server stopped.");
    const child = spawn(this.command, [...this.args], {
      shell: false,
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...minimalEnvironment, ...explicitEnvironment }
    });
    this.child = child;
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => this.consume(chunk));
    child.stderr.on("data", (chunk: string) => {
      this.stderr = `${this.stderr}${chunk}`.slice(-8_000);
    });
    child.once("exit", () => {
      if (this.child !== child) return;
      this.child = undefined;
      this.failAll(new Error("MCP stdio server exited."));
      this.onClosed();
    });
    child.on("error", (error) => {
      this.failAll(error);
      this.onClosed();
    });
    await new Promise<void>((resolveStart, rejectStart) => {
      child.once("spawn", resolveStart);
      child.once("error", rejectStart);
    });
  }

  private consume(chunk: string): void {
    this.buffer += chunk;
    for (;;) {
      const end = this.buffer.indexOf("\n");
      if (end < 0) break;
      const line = this.buffer.slice(0, end).trim();
      this.buffer = this.buffer.slice(end + 1);
      if (!line) continue;
      if (Buffer.byteLength(line, "utf8") > MAX_MCP_MESSAGE_BYTES) {
        this.failAll(new Error("MCP stdio response exceeded 16 MiB.")); this.stop(); return;
      }
      let response: McpRpcMessage;
      try { response = JSON.parse(line) as McpRpcMessage; }
      catch { this.failAll(new Error("MCP stdio server returned invalid JSON.")); this.stop(); return; }
      if (!response || typeof response !== "object" || Array.isArray(response) || response.jsonrpc !== "2.0") {
        this.failAll(new Error("MCP stdio server returned invalid JSON-RPC.")); this.stop(); return;
      }
      if (typeof response.method === "string") {
        void Promise.resolve(this.onMessage(response)).catch((error: unknown) => {
          this.failAll(error instanceof Error ? error : new Error(String(error))); this.stop();
        });
        continue;
      }
      if (response.id === undefined || response.id === null) continue;
      const id = String(response.id);
      const pending = this.pending.get(id);
      if (!pending) continue;
      this.pending.delete(id);
      pending.deadline.close();
      pending.resolve(response);
    }
    if (Buffer.byteLength(this.buffer, "utf8") > MAX_MCP_MESSAGE_BYTES) {
      this.buffer = "";
      this.failAll(new Error("MCP stdio response exceeded 16 MiB."));
      this.stop();
    }
  }

  private failAll(error: Error): void {
    for (const pending of this.pending.values()) {
      pending.deadline.close();
      pending.reject(error);
    }
    this.pending.clear();
  }

  async request(method: string, params: Record<string, unknown>, signal?: AbortSignal,
    requestId = randomUUID(), suppliedDeadline?: McpRequestDeadline, cancelNotification = true): Promise<JsonRpcResponse> {
    await this.start();
    const deadline = suppliedDeadline ?? new McpRequestDeadline(120_000);
    signal = AbortSignal.any([signal ?? new AbortController().signal, deadline.signal]);
    signal?.throwIfAborted();
    const id = requestId;
    const child = this.child;
    if (!child || child.killed) throw new Error("MCP stdio server is unavailable.");
    const encoded = encodeRpcMessage({ jsonrpc: "2.0", id, method, params });
    return new Promise<JsonRpcResponse>((resolveRequest, rejectRequest) => {
      const abort = (): void => {
        const pending = this.pending.get(id);
        if (!pending) return;
        this.pending.delete(id);
        deadline.close();
        pending.reject(new Error(deadline.signal.aborted ? "MCP stdio request timed out." : "MCP request was cancelled."));
        if (cancelNotification && method !== "initialize" && this.child && !this.stopped) this.notify("notifications/cancelled", { requestId: id, reason: "The caller stopped this request." });
      };
      signal?.addEventListener("abort", abort, { once: true });
      this.pending.set(id, {
        resolve: (value) => {
          signal?.removeEventListener("abort", abort);
          resolveRequest(value);
        },
        reject: (error) => {
          signal?.removeEventListener("abort", abort);
          rejectRequest(error);
        },
        deadline
      });
      child.stdin.write(`${encoded}\n`, (error) => {
        if (!error) return;
        const pending = this.pending.get(id);
        if (!pending) return;
        this.pending.delete(id);
        deadline.close();
        pending.reject(error);
      });
    });
  }

  notify(method: string, params: Record<string, unknown> = {}): void {
    if (!this.child || this.child.killed) throw new Error("MCP stdio server is unavailable.");
    this.write({ jsonrpc: "2.0", method, params });
  }

  write(message: Record<string, unknown>): void {
    if (!this.child || this.child.killed || this.stopped) throw new Error("MCP stdio server is unavailable.");
    this.child.stdin.write(`${encodeRpcMessage(message)}\n`, (error) => {
      if (error) { this.failAll(error); this.stop(); }
    });
  }

  stop(): void {
    this.stopped = true;
    this.failAll(new Error("MCP stdio server stopped."));
    this.child?.kill();
    this.child = undefined;
    this.onClosed();
  }
}

export class McpClientService {
  private readonly runtimes = new Map<string, McpRuntime>();
  private readonly inputValidators = new Map<string, {
    schemaHash: string;
    validate: ValidateFunction<unknown>;
  }>();
  private readonly registryStore: McpRegistryStore;
  private document: McpRegistryDocument = { format: "omni-mcp-registry", version: 2, enabled: true, servers: [], instances: [] };
  private mutationQueue: Promise<void> = Promise.resolve();
  private readonly oauth: McpOAuthService;
  private readonly capabilityListeners = new Set<(event: McpCapabilityEvent) => void>();
  private sessionTransportsEnabled = false;
  private hostInteractions: McpHostInteractions = {};
  private standardHost: McpStandardHostServices = {};
  private readonly oauthAccountChanges = new Set<string>();
  private appHost?: McpAppHost;
  private taskStore?: McpTaskStore;
  private readonly taskLeases = new Map<string, McpTaskLease>();
  private readonly taskOperations = new Set<{ brainId: string; taskRef?: string; controller: AbortController; promise: Promise<unknown> }>();

  constructor(
    registryPath: string,
    private readonly secrets: SecureSecretStore,
    private readonly brain: Pick<BrainService,
      "registerExternalToolSchemas" |
      "unregisterExternalToolSchemas" |
      "learnStructuredExperience" |
      "listToolPermissions" |
      "setToolPermission"
    > & Partial<Pick<BrainService, "ensureToolPermission">>,
    private readonly fetcher: FetchLike = fetch,
    authOptions: Partial<ConstructorParameters<typeof McpOAuthService>[1]> = {}
  ) {
    this.registryStore = new McpRegistryStore(registryPath);
    this.oauth = new McpOAuthService(secrets, { ...authOptions, fetcher,
      openBrowser: authOptions.openBrowser ?? (async () => { throw new Error("Browser sign-in is unavailable on this host."); }),
      onChange: (id, status) => {
        if (status.state === "authorizing") this.oauthAccountChanges.add(id);
        else if (status.state === "authorized" && this.oauthAccountChanges.delete(id)) {
          // Interactive account replacement is not a same-account token refresh.
          // No old session/root/view authority may follow the new vault account.
          for (const runtime of this.runtimes.values()) if (runtime.config.id === id) {
            this.endHttpSession(runtime); runtime.controller.abort(new Error("The MCP account changed; reconnect and review disclosures explicitly."));
            runtime.legacySse?.stop(); runtime.legacySse = undefined; runtime.initialized = false; runtime.sessionId = undefined;
            runtime.protocolVersion = undefined; runtime.capabilities = undefined; runtime.pagination?.clear(); this.loseSubscriptions(runtime);
            if (this.isEnabled(runtime)) runtime.controller = new AbortController(); runtime.summary = this.summary(runtime.config, false);
          }
        } else if (["none", "error", "required"].includes(status.state)) this.oauthAccountChanges.delete(id);
        try { authOptions.onChange?.(id, status); } catch { /* status observers cannot change completed account publication */ }
      } });
  }

  private serialize<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.mutationQueue.then(operation, operation);
    this.mutationQueue = result.then(() => undefined, () => undefined);
    return result;
  }

  async initialize(): Promise<void> {
    this.document = await this.registryStore.load((stored, template) => this.validateStoredConfig(stored, template));
    this.syncRuntimes();
  }

  private definition(serverId: string): McpRegistryServer {
    const server = this.document.servers.find((candidate) => candidate.id === serverId);
    if (!server) throw new Error("MCP server was not found.");
    return server;
  }

  private toolsFor(brainId: string, server: McpRegistryServer): McpToolSummary[] {
    return server.tools.map((tool) => ({ ...structuredClone(tool),
      id: namespaceTool(brainId, Object.hasOwn(server.namespaceByBrain, brainId) ? server.namespaceByBrain[brainId]! : server.id, tool.remoteName) }));
  }

  private boundTools(config: StoredMcpServer): McpToolSummary[] {
    const runtime = this.runtimes.get(runtimeKey(config.brainId, config.id)), tasksReady = !!this.taskStore && runtime?.protocolVersion === "2025-11-25";
    const remote = config.tools.map(tool => {
      if (!tasksReady || !config.capabilities?.tasks?.requests?.tools?.call || tool.taskSupport === undefined || tool.taskSupport === "forbidden") return { ...tool,
        actions: tool.taskSupport === "required" ? [] : ["call"] };
      const reference = { type: "object", properties: { taskRef: { type: "string", pattern: MCP_TASK_REF.source } }, required: ["taskRef"], additionalProperties: false };
      const scopedInput = { ...tool.inputSchema, $id: typeof tool.inputSchema.$id === "string" ? tool.inputSchema.$id : `urn:omni:mcp-task-input:${sha256(JSON.stringify(tool.inputSchema))}` };
      const actionInputSchemas = { ...(tool.taskSupport === "optional" ? { call: tool.inputSchema } : {}),
        "call-task": { ...(tool.inputSchema.$schema ? { $schema: tool.inputSchema.$schema } : {}), type: "object", properties: { arguments: scopedInput, ttl: { type: "integer", minimum: 0 } }, required: ["arguments"], additionalProperties: false },
        "task-status": reference, "task-result": reference, ...(config.capabilities?.tasks?.cancel ? { "task-cancel": reference } : {}) };
      return { ...tool, actions: Object.keys(actionInputSchemas), actionInputSchemas };
    });
    return [...remote, ...mcpClientOperationTools(config.capabilities, identity =>
      namespaceClientOperation(config.brainId, config.namespaceId ?? config.id, identity)).filter(tool => tasksReady || tool.clientOperation !== "tasks/list")];
  }

  /** Read-only observations. Registering a listener grants no dispatch permission. */
  onCapabilityEvent(listener: (event: McpCapabilityEvent) => void): () => void {
    this.capabilityListeners.add(listener);
    return () => { this.capabilityListeners.delete(listener); };
  }

  private capabilityEvent(runtime: McpRuntime, kind: McpCapabilityEvent["kind"], uri?: string): void {
    const event: McpCapabilityEvent = { brainId: runtime.config.brainId, serverId: runtime.config.id,
      revision: runtime.capabilityRevision = (runtime.capabilityRevision ?? 0) + 1,
      observedAt: new Date().toISOString(), kind, ...(uri ? { uri } : {}) };
    for (const listener of this.capabilityListeners) {
      // A renderer/listener failure is not a reason to repeat a completed RPC.
      try { listener(structuredClone(event)); } catch { /* observer is isolated */ }
    }
  }

  private loseSubscriptions(runtime: McpRuntime, reason?: "session-initialized"): void {
    for (const [key, task] of this.taskLeases) if (task.runtime === runtime) { task.controller.abort(new Error("The MCP task source session retired; remote work may still be running.")); this.taskLeases.delete(key); }
    this.appHost?.invalidate(runtime.config.brainId, runtime.config.id, reason);
    runtime.sessionEpoch = (runtime.sessionEpoch ?? 0) + 1;
    runtime.hostConnectionId = randomUUID();
    try { this.standardHost.connectionRetired?.(runtime.config.brainId, runtime.config.id); } catch { /* remaining transport retirement must still complete */ }
    runtime.httpEvents?.controller.abort(); runtime.httpEvents = undefined;
    runtime.deliveredEventIds?.clear();
    runtime.transportObservation = { ...runtime.transportObservation, eventStream: "off" };
    runtime.subscribing?.clear();
    for (const request of runtime.serverRequests?.values() ?? []) request.controller.abort(new Error("The MCP connection ended."));
    runtime.serverRequests?.clear();
    let changed = false;
    for (const subscription of runtime.subscriptions?.values() ?? []) {
      if (subscription.state !== "active") continue;
      subscription.state = "connection-lost"; changed = true;
    }
    if (changed) this.capabilityEvent(runtime, "subscriptions-lost");
  }

  /** Unload cannot preserve an external live subscription; reconnect is explicit. */
  suspendBrain(brainId: string): McpSuspensionReceipt {
    safeId(brainId, "brain id");
    const receipt: McpSuspensionReceipt = { brainId, disconnectedServerIds: [], lostResourceSubscriptions: [], reconnectRequired: [] };
    for (const runtime of this.runtimes.values()) {
      if (runtime.config.brainId !== brainId) continue;
      if (runtime.initialized || runtime.stdio || runtime.sessionId) receipt.disconnectedServerIds.push(runtime.config.id);
      for (const subscription of runtime.subscriptions?.values() ?? []) {
        if (subscription.state !== "active") continue;
        receipt.lostResourceSubscriptions.push({ serverId: runtime.config.id, uri: subscription.uri });
        receipt.reconnectRequired.push(`MCP ${runtime.config.label}: request the resource subscription ${subscription.uri} again.`);
      }
      this.endHttpSession(runtime);
      runtime.controller.abort(new Error("The instance unloaded its MCP connections."));
      runtime.stdio?.stop(); runtime.stdio = undefined;
      runtime.legacySse?.stop(); runtime.legacySse = undefined;
      runtime.initialized = false; runtime.sessionId = undefined;
      runtime.protocolVersion = undefined; runtime.capabilities = undefined;
      runtime.pagination?.clear(); this.loseSubscriptions(runtime);
      runtime.summary = this.summary(runtime.config, false);
      if (this.isEnabled(runtime)) runtime.controller = new AbortController();
    }
    return receipt;
  }

  private observedCapabilities(runtime: McpRuntime): Pick<McpServerSummary, "capabilityRevision" | "resourceSubscriptions" | "transportObservation"> {
    return { capabilityRevision: runtime.capabilityRevision ?? 0,
      resourceSubscriptions: [...(runtime.subscriptions?.values() ?? [])].map(value => ({ ...value })),
      ...(runtime.transportObservation ? { transportObservation: { ...runtime.transportObservation } } : {}) };
  }

  /** Host composition enables optional session listeners; it grants no tool authority. */
  enableSessionTransports(): void {
    this.sessionTransportsEnabled = true;
    for (const runtime of this.runtimes.values()) if (runtime.initialized) this.startHttpEvents(runtime);
  }
  /** Installed only by main composition after real review/native adapters exist. */
  setHostInteractions(interactions: McpHostInteractions): void { this.hostInteractions = interactions; }
  /** Optional standards are installed only when their real host consumers exist. */
  setStandardHostServices(host: McpStandardHostServices): void { this.standardHost = host; }
  setAppHost(host: McpAppHost): void { this.appHost = host; }
  /** Installing a store grants no remote authority and starts no tasks or polling. */
  setTaskStore(store: McpTaskStore): void { this.taskStore = store; }
  async settleTaskWrites(brainId?: string): Promise<void> {
    for (;;) { const pending = [...this.taskOperations].filter(task => brainId === undefined || task.brainId === brainId); if (!pending.length) return; await Promise.allSettled(pending.map(task => task.promise)); }
  }
  stopTaskWait(brainId: string, taskRef: string): boolean {
    safeId(brainId, "brain id"); if (!MCP_TASK_REF.test(taskRef)) throw new Error("MCP task wait reference is invalid."); let stopped = false;
    for (const operation of this.taskOperations) if (operation.brainId === brainId && operation.taskRef === taskRef && !operation.controller.signal.aborted) {
      operation.controller.abort(new Error("The local MCP task wait stopped; remote work was not cancelled or undone.")); stopped = true;
    } return stopped;
  }
  async retireTaskAuthority(brainId: string, reason = "This instance retired its MCP task authority; remote work may still be running."): Promise<void> {
    for (const operation of this.taskOperations) if (operation.brainId === brainId) operation.controller.abort(new Error(reason));
    const records: McpTaskRecord[] = [];
    for (const [key, lease] of this.taskLeases) if (lease.record.source.brainId === brainId) { lease.controller.abort(new Error(reason)); records.push(lease.record); this.taskLeases.delete(key); }
    await this.settleTaskWrites(brainId);
    for (const record of records) if (this.taskStore) await this.taskStore.update(brainId, record.taskRef, { reason });
  }
  async tasksLocalPage(brainId: string, after?: string, limit?: number): Promise<McpTaskLocalPage> {
    if (!this.taskStore) throw new Error("Durable MCP task custody is unavailable.");
    const page = await this.taskStore.localPage(brainId, after, limit);
    return { format: "omni-mcp-task-local-page", version: 1, brainId, records: page.records.map(({ record, custodyPath }) => {
      const lease = this.taskLeases.get(`${brainId}:${record.taskRef}`), active = !!lease && !lease.controller.signal.aborted && !custodyPath.includes("/lineage/") && JSON.stringify(lease.record.source) === JSON.stringify(record.source);
      return { record, custodyKey: mcpTaskSha(custodyPath), management: active ? "current-session" as const : "dormant" as const,
        ...(!active ? { reason: "Retained task history has no verified current account/session grant. No task was resent or rebound." } : {}) };
    }), ...(page.nextCursor ? { nextCursor: page.nextCursor } : {}), networkRequestMade: false };
  }
  /** Metadata only. Never connects, refreshes credentials, starts or Loads. */
  hostConnection(brainId: string, serverId: string): McpHostConnection {
    const runtime = this.runtimes.get(runtimeKey(safeId(brainId, "brain id"), safeId(serverId, "server id")));
    if (!runtime) throw new Error("The MCP source connection is not configured for this instance.");
    runtime.hostConnectionId ??= randomUUID();
    return { ...this.interactionOrigin(runtime), configurationSha256: sha256(JSON.stringify({ transport: runtime.config.transport,
      url: runtime.config.url ?? null, command: runtime.config.command ?? null, args: runtime.config.args ?? [] })),
      connectionId: runtime.hostConnectionId, ready: runtime.initialized && !runtime.controller.signal.aborted && this.isEnabled(runtime), enabled: this.isEnabled(runtime) };
  }
  requireHostConnection(expected: McpHostConnection): void {
    const current = this.hostConnection(expected.brainId, expected.serverId);
    if (!current.ready || current.connectionId !== expected.connectionId || current.connectionEpoch !== expected.connectionEpoch ||
        current.configurationSha256 !== expected.configurationSha256 || current.endpoint !== expected.endpoint) throw new Error("The exact MCP session is no longer ready; explicitly reconnect and review its roots.");
    this.requireInteractionOrigin(expected);
  }
  /** No initialization/retry: an old selection never notifies a replacement peer. */
  async notifyRootsChanged(expected: McpHostConnection): Promise<boolean> {
    if (!this.standardHost.roots) return false;
    try { this.requireHostConnection(expected); } catch { return false; }
    const runtime = this.runtimes.get(runtimeKey(expected.brainId, expected.serverId))!;
    if (runtime.config.transport === "stdio") runtime.stdio!.notify("notifications/roots/list_changed", {});
    else await this.httpRpc(runtime, "notifications/roots/list_changed", {}, true, runtime.controller.signal);
    this.requireHostConnection(expected); return true;
  }
  serverCapabilities(origin: McpRequestOrigin): McpNegotiatedCapabilities {
    this.requireInteractionOrigin(origin);
    const runtime = this.runtimes.get(runtimeKey(origin.brainId, origin.serverId))!;
    if (!runtime.initialized) throw new Error("MCP capabilities require the existing initialized session.");
    return negotiatedMcpCapabilities(runtime.capabilities);
  }
  /** Private main redaction input only: reads snapshots, never OAuth refresh. */
  async knownOperationalSecrets(origin: McpRequestOrigin): Promise<string[]> {
    this.requireInteractionOrigin(origin);
    const runtime = this.runtimes.get(runtimeKey(origin.brainId, origin.serverId))!, definition = this.definition(origin.serverId), connectionId = runtime.hostConnectionId;
    const [bearer, encodedEnvironment, encodedOAuth] = await Promise.all([this.secrets.get(definition.credentialId),
      this.secrets.get(mcpEnvironmentKey(definition.id)), this.oauth.credentialSnapshot(definition.id)]);
    this.requireInteractionOrigin(origin);
    if (this.runtimes.get(runtimeKey(origin.brainId, origin.serverId)) !== runtime || runtime.hostConnectionId !== connectionId) throw new Error("MCP diagnostics lost their exact source while reading secret snapshots.");
    const environment = mcpEnvironment(encodedEnvironment ? JSON.parse(encodedEnvironment) : undefined), oauth = encodedOAuth ? record(JSON.parse(encodedOAuth)) : {};
    return [bearer, runtime.sessionId, runtime.sessionToken, ...Object.values(environment), oauth.accessToken, oauth.refreshToken, oauth.clientSecret]
      .filter((value): value is string => typeof value === "string" && value.length > 0);
  }
  currentAppOrigin(brainId: string, serverId: string): McpRequestOrigin {
    const runtime = this.runtimes.get(runtimeKey(safeId(brainId, "brain id"), safeId(serverId, "server id")));
    if (!runtime) throw new Error("Configure and permit this instance's source connection before reopening the saved App.");
    this.requireEnabled(runtime); return this.interactionOrigin(runtime);
  }
  appTool(origin: McpRequestOrigin, remoteName: string): McpToolSummary {
    this.requireInteractionOrigin(origin);
    const runtime = this.runtimes.get(runtimeKey(origin.brainId, origin.serverId))!;
    const tool = runtime.config.tools.find(tool => tool.remoteName === remoteName);
    if (!tool || !mcpToolUi(tool).visibility.includes("app")) throw new Error("This server tool is not exposed to Apps.");
    return structuredClone(tool);
  }
  appResourceTool(origin: McpRequestOrigin): McpToolSummary {
    this.requireInteractionOrigin(origin);
    const runtime = this.runtimes.get(runtimeKey(origin.brainId, origin.serverId))!;
    const tool = this.boundTools(runtime.config).find(tool => tool.clientOperation === "resources/read");
    if (!tool) throw new Error("The App server did not negotiate resource reads.");
    return structuredClone(tool);
  }
  requireInteractionOrigin(origin: McpRequestOrigin): void {
    const runtime = this.runtimes.get(runtimeKey(origin.brainId, origin.serverId));
    if (!runtime || (runtime.sessionEpoch ?? 0) !== origin.connectionEpoch || (runtime.config.url ?? runtime.config.command) !== origin.endpoint) throw new Error("The MCP request's connection no longer exists.");
    this.requireEnabled(runtime);
  }
  private interactionOrigin(runtime: McpRuntime, parent?: McpActiveCall): McpRequestOrigin {
    return { brainId: runtime.config.brainId, serverId: runtime.config.id, serverLabel: runtime.config.label,
      endpoint: runtime.config.url ?? runtime.config.command!, connectionEpoch: runtime.sessionEpoch ?? 0,
      ...(parent?.context?.chatTurnId ? { originatingTurnId: parent.context.chatTurnId } : {}),
      ...(parent?.context?.actionEventId ? { actionEventId: parent.context.actionEventId } : {}) };
  }
  private beginCall(runtime: McpRuntime, call: McpActiveCall): () => void {
    runtime.activeCalls ??= new Map(); runtime.activeCalls.set(call.id, call);
    return () => {
      runtime.activeCalls?.delete(call.id);
      for (const pending of runtime.serverRequests?.values() ?? []) if (pending.parentId === call.id) pending.controller.abort(new Error("The originating MCP request ended."));
    };
  }

  private acceptSseEvent(runtime: McpRuntime, event: McpSseEvent): boolean {
    return !event.id || !runtime.deliveredEventIds?.has(event.id);
  }
  private deliveredSseEvent(runtime: McpRuntime, event: McpSseEvent): void {
    if (event.id) { runtime.deliveredEventIds ??= new Set(); runtime.deliveredEventIds.add(event.id); }
  }

  private expireHttpSession(runtime: McpRuntime): void {
    runtime.initialized = false; runtime.sessionId = undefined; runtime.sessionToken = undefined;
    runtime.protocolVersion = undefined; runtime.toolsDirty = true;
    runtime.pagination?.clear(); this.loseSubscriptions(runtime);
    runtime.summary = this.summary(runtime.config, false, "The remote session expired; subscriptions require a new explicit request.");
  }

  private async waitForHttpRetry(milliseconds: number, signal: AbortSignal): Promise<void> {
    // Respect the server's retry delay without JavaScript timer overflow.
    let remaining = Math.max(50, milliseconds);
    while (remaining > 0) {
      const part = Math.min(remaining, 2_147_483_647);
      await delay(part, undefined, { signal }); remaining -= part;
    }
  }

  private lastEventHeader(value: string): string {
    if (/[\0\r\n]/.test(value)) throw new Error("MCP event ID cannot be used in a resume header.");
    // Last-Event-ID uses UTF-8 bytes. Headers requires a ByteString; Latin-1
    // here transports those bytes without parsing or changing the opaque ID.
    return Buffer.from(value, "utf8").toString("latin1");
  }

  private startHttpEvents(runtime: McpRuntime): void {
    if (!this.sessionTransportsEnabled || runtime.config.transport !== "http" || runtime.legacySse || !runtime.initialized || runtime.httpEvents || !this.isEnabled(runtime)) return;
    const controller = new AbortController();
    const epoch = runtime.sessionEpoch ?? 0;
    const signal = AbortSignal.any([controller.signal, runtime.controller.signal]);
    const operation = this.listenHttpEvents(runtime, epoch, signal).catch(error => {
      if (!signal.aborted) runtime.transportObservation = { ...runtime.transportObservation, eventStream: "error",
        error: (error instanceof Error ? error.message : String(error)).replace(/[\0\r\n]/g, " ").slice(0, 500) };
    });
    const owned = { controller, operation }; runtime.httpEvents = owned;
    void operation.finally(() => { if (runtime.httpEvents === owned) runtime.httpEvents = undefined; });
  }

  private async listenHttpEvents(runtime: McpRuntime, epoch: number, signal: AbortSignal): Promise<void> {
    let lastEventId: string | undefined;
    let retry = 1_000;
    while (!signal.aborted && runtime.initialized && (runtime.sessionEpoch ?? 0) === epoch) {
      this.requireEnabled(runtime);
      runtime.transportObservation = { ...runtime.transportObservation, eventStream: lastEventId ? "reconnecting" : "connecting" };
      const token = await this.tokenFor(runtime);
      let response: Response;
      try {
        response = await this.fetcher(runtime.config.url!, { method: "GET", redirect: "error", signal,
          headers: { accept: "text/event-stream", "mcp-protocol-version": runtime.protocolVersion!,
            ...(runtime.sessionId ? { "mcp-session-id": runtime.sessionId } : {}),
            ...(token ? { authorization: `Bearer ${token}` } : {}),
            ...(lastEventId ? { "last-event-id": this.lastEventHeader(lastEventId) } : {}) } });
      } catch (error) {
        signal.throwIfAborted();
        runtime.transportObservation = { ...runtime.transportObservation, eventStream: "reconnecting", error: "MCP notification connection interrupted; reconnecting without replaying requests." };
        await this.waitForHttpRetry(retry, signal); continue;
      }
      if (!response.ok) {
        await response.body?.cancel().catch(() => undefined);
        if (response.status === 405) { runtime.transportObservation = { ...runtime.transportObservation, eventStream: "unsupported" }; return; }
        if (response.status === 404 && runtime.sessionId) { this.expireHttpSession(runtime); return; }
        if (response.status === 401 || response.status === 403 && response.headers.has("www-authenticate")) {
          this.oauth.challenge(runtime.config.id, response.headers.get("www-authenticate")); throw new McpAuthenticationRequiredError();
        }
        if (response.status >= 500) { await this.waitForHttpRetry(retry, signal); continue; }
        throw new Error(`MCP notification listener returned HTTP ${response.status}.`);
      }
      runtime.transportObservation = { ...runtime.transportObservation, eventStream: "listening", error: undefined };
      try {
        for await (const event of readMcpEvents(response, MAX_MCP_MESSAGE_BYTES, signal)) {
          if ((runtime.sessionEpoch ?? 0) !== epoch) return;
          if (event.retry !== undefined) retry = event.retry;
          if (!this.acceptSseEvent(runtime, event)) continue;
          if (event.data.trim()) for (const message of parseMcpRpcMessages(event.data)) {
            if (!message.method) throw new Error("MCP background stream returned an unrelated RPC response.");
            await this.serverMessage(runtime, message, null);
          }
          if (event.id !== undefined) lastEventId = event.id;
          this.deliveredSseEvent(runtime, event);
        }
      } catch (error) {
        signal.throwIfAborted();
        if (!(error instanceof McpEventStreamIoError)) throw error;
      }
      if (!signal.aborted) await this.waitForHttpRetry(retry, signal);
    }
  }

  private endHttpSession(runtime: McpRuntime): void {
    if (!this.sessionTransportsEnabled || runtime.config.transport !== "http" || !runtime.sessionId) return;
    const sessionId = runtime.sessionId, token = runtime.sessionToken, version = runtime.protocolVersion ?? MCP_PROTOCOL_VERSION;
    runtime.sessionToken = undefined;
    runtime.transportObservation = { ...runtime.transportObservation, eventStream: "off", lastSessionTermination: "pending" };
    const owner = {}; runtime.terminationOwner = owner;
    const operation: Promise<void> = (async () => {
      try {
        const response = await this.fetcher(runtime.config.url!, { method: "DELETE", redirect: "error", signal: AbortSignal.timeout(5_000),
          headers: { "mcp-session-id": sessionId, "mcp-protocol-version": version, ...(token ? { authorization: `Bearer ${token}` } : {}) } });
        await response.body?.cancel().catch(() => undefined);
        const state = response.status === 405 ? "unsupported" : response.status === 404 ? "already-ended" : response.ok ? "complete" : "failed";
        if (runtime.terminationOwner === owner) runtime.transportObservation = { ...runtime.transportObservation, eventStream: runtime.transportObservation?.eventStream ?? "off", lastSessionTermination: state };
      } catch { if (runtime.terminationOwner === owner) runtime.transportObservation = { ...runtime.transportObservation, eventStream: runtime.transportObservation?.eventStream ?? "off", lastSessionTermination: "failed" }; }
    })();
    runtime.termination = operation;
  }

  private syncRuntimes(): void {
    const keep = new Set<string>();
    for (const instance of this.document.instances) {
      for (const server of this.document.servers) {
        if (server.onlyBrainId && server.onlyBrainId !== instance.brainId) continue;
        const key = runtimeKey(instance.brainId, server.id);
        keep.add(key);
        const boundRuntime = this.runtimes.get(key);
        const config: StoredMcpServer = { brainId: instance.brainId, id: server.id, label: server.label,
          transport: server.transport, url: server.url, command: server.command, args: server.args,
          namespaceId: Object.hasOwn(server.namespaceByBrain, instance.brainId) ? server.namespaceByBrain[instance.brainId] : server.id,
          tools: this.toolsFor(instance.brainId, server),
          // The installation cache helps an offline catalog, not another
          // instance's negotiation. Keep each established session authoritative.
          capabilities: boundRuntime?.initialized ? negotiatedMcpCapabilities(boundRuntime.capabilities) : server.capabilities };
        let runtime = boundRuntime;
        if (!runtime) {
          runtime = { config, summary: this.summary(config, false), initialized: false, controller: new AbortController() };
          this.runtimes.set(key, runtime);
        } else {
          this.oauth.revalidate(server.id);
          if (this.schemaHash(runtime.config) !== this.schemaHash(config)) {
            this.brain.unregisterExternalToolSchemas(this.boundTools(runtime.config).map((tool) => tool.id));
            runtime.schemasRegistered = false;
          }
          runtime.config = config;
        }
        const access = mcpAccess(this.document, instance.brainId, server);
        if (access.effectiveEnabled) {
          if (runtime.controller.signal.aborted) runtime.controller = new AbortController();
          if (!runtime.schemasRegistered && this.boundTools(config).length) {
            this.registerSchemas(this.boundTools(config));
            runtime.schemasRegistered = true;
          }
        } else {
          if (runtime.schemasRegistered) this.brain.unregisterExternalToolSchemas(this.boundTools(config).map((tool) => tool.id));
          runtime.schemasRegistered = false;
          this.endHttpSession(runtime);
          runtime.controller.abort(new Error(access.disabledReason));
          runtime.stdio?.stop();
          runtime.stdio = undefined;
          runtime.legacySse?.stop(); runtime.legacySse = undefined;
          runtime.initialized = false;
          runtime.sessionId = undefined;
          this.loseSubscriptions(runtime);
          runtime.pagination?.clear();
        }
        runtime.summary = { ...this.summary(config, runtime.initialized, runtime.summary.error),
          ...(runtime.protocolVersion ? { protocolVersion: runtime.protocolVersion } : {}),
          ...(runtime.capabilities ? { capabilities: structuredClone(runtime.capabilities) } : {}) };
      }
    }
    for (const [key, runtime] of this.runtimes) {
      if (keep.has(key)) continue;
      this.endHttpSession(runtime);
      runtime.controller.abort(new Error("MCP server was removed."));
      this.oauth.cancel(runtime.config.id);
      runtime.stdio?.stop();
      runtime.legacySse?.stop();
      this.loseSubscriptions(runtime);
      this.brain.unregisterExternalToolSchemas(this.boundTools(runtime.config).map((tool) => tool.id));
      this.runtimes.delete(key);
    }
    this.inputValidators.clear();
  }

  private async publish(document: McpRegistryDocument): Promise<void> {
    await this.registryStore.save(document);
    this.document = document;
    this.syncRuntimes();
  }

  /** Binds catalog tools without starting a neural worker or an MCP process. */
  async attachBrain(brainId: string, inheritFromBrainId?: string): Promise<void> {
    safeId(brainId, "brain id");
    await this.serialize(async () => {
      if (!this.document.instances.some((instance) => instance.brainId === brainId)) {
        const next = structuredClone(this.document);
        attachMcpInstance(next, brainId, inheritFromBrainId);
        await this.publish(next);
      }
    });
    for (const runtime of this.runtimes.values()) {
      if (runtime.config.brainId === brainId && this.isEnabled(runtime)) await this.ensurePermissions(brainId, this.boundTools(runtime.config));
    }
  }

  async configure(request: McpRegistryUpdateRequest): Promise<void> {
    await this.serialize(async () => {
      const next = structuredClone(this.document);
      updateMcpAccess(next, request);
      await this.publish(next);
    });
    for (const runtime of this.runtimes.values()) {
      if (this.isEnabled(runtime)) await this.ensurePermissions(runtime.config.brainId, this.boundTools(runtime.config));
    }
  }

  /** New merged identity never inherits last-instance defaults or credential access. */
  async initializeMergedIdentity(brainId: string): Promise<void> {
    safeId(brainId, "brain id");
    await this.serialize(async () => {
      const next = structuredClone(this.document);
      const isolated = { brainId, enabled: false,
        overrides: Object.fromEntries(next.servers.map((server) => [server.id, "off" as const])), learnedSchemas: {} };
      const index = next.instances.findIndex((entry) => entry.brainId === brainId);
      if (index >= 0) next.instances[index] = isolated;
      else next.instances.push(isolated);
      // Do not change lastConfiguredBrainId: assembling is not a catalog/default review.
      await this.publish(next);
    });
  }

  async registry(brainId: string): Promise<McpRegistrySummary> {
    const servers = await this.list(brainId);
    return { enabled: this.document.enabled,
      instanceEnabled: this.document.instances.find((instance) => instance.brainId === brainId)?.enabled ?? true, servers };
  }

  private isEnabled(runtime: McpRuntime): boolean {
    const server = this.document.servers.find((candidate) => candidate.id === runtime.config.id);
    return Boolean(server && mcpAccess(this.document, runtime.config.brainId, server).effectiveEnabled);
  }

  private requireEnabled(runtime: McpRuntime): void {
    if (this.runtimes.get(runtimeKey(runtime.config.brainId, runtime.config.id)) !== runtime) {
      throw new Error("MCP connection was replaced or removed.");
    }
    const server = this.definition(runtime.config.id);
    const access = mcpAccess(this.document, runtime.config.brainId, server);
    if (!access.effectiveEnabled) throw new Error(access.disabledReason);
    if (runtime.credentialsUpdating) throw new Error("This MCP connection is updating its credentials. Retry after it finishes.");
  }

  private async ensurePermissions(brainId: string, tools: readonly McpToolSummary[]): Promise<void> {
    if (!this.brain.ensureToolPermission) {
      // Compatibility for dependency-injected unit doubles only. Production
      // BrainService always exposes the atomic host-policy insertion below.
      const permissions = await this.brain.listToolPermissions(brainId);
      for (const tool of tools) {
        if (!permissions.some((permission) => permission.toolId === tool.id)) {
          await this.brain.setToolPermission(brainId, tool.id, "ask");
        }
      }
      return;
    }
    for (const tool of tools) {
      await this.brain.ensureToolPermission(
        brainId,
        tool.id,
        "ask",
        `mcp-ensure-${createHash("sha256").update(`${brainId}\0${tool.id}`).digest("hex")}`
      );
    }
  }

  private registerSchemas(tools: readonly McpToolSummary[]): void {
    this.brain.registerExternalToolSchemas(
      tools.filter(tool => tool.actions.length && mcpToolUi(tool).visibility.includes("model")).map((tool): ExternalToolSchema => ({
        id: tool.id,
        actions: tool.actions,
        inputSchema: tool.inputSchema,
        ...(tool.actionInputSchemas ? { actionInputSchemas: tool.actionInputSchemas } : {})
      }))
    );
  }

  private summary(config: StoredMcpServer, connected: boolean, error?: string): McpServerSummary {
    const definition = this.document.servers.find((server) => server.id === config.id);
    const access = definition ? mcpAccess(this.document, config.brainId, definition)
      : { effectiveEnabled: true, accessOverride: "inherit" as const };
    const instance = this.document.instances.find((entry) => entry.brainId === config.brainId);
    return {
      brainId: config.brainId,
      id: config.id,
      label: config.label,
      transport: config.transport,
      endpoint: config.transport === "http" ? config.url ?? "" : config.command ?? "",
      connected,
      tools: this.boundTools(config).map((tool) => ({ ...tool, inputSchema: { ...tool.inputSchema } })),
      credentialConfigured: false,
      registryId: config.id,
      scope: definition?.onlyBrainId ? "instance" : "all",
      ...(definition?.onlyBrainId ? { ownerBrainId: definition.onlyBrainId } : {}),
      globallyEnabled: this.document.enabled && (definition?.enabled ?? true),
      defaultEnabled: definition?.defaultEnabled ?? true,
      ...access,
      schemaLearning: definition && instance?.learnedSchemas[config.id] === this.schemaHash(config) ? "complete" : "pending",
      ...(definition?.environmentKeys ? { environmentKeys: [...definition.environmentKeys] } : {}),
      ...(error ? { error: error.replace(/[\r\n\0]+/g, " ").slice(0, 1_000) } : {})
    };
  }

  async list(brainId: string): Promise<McpServerSummary[]> {
    await this.attachBrain(brainId);
    const values = [...this.runtimes.values()].filter((runtime) => runtime.config.brainId === brainId);
    return Promise.all(values.map(async (runtime) => ({
      ...runtime.summary,
      ...this.observedCapabilities(runtime),
      credentialConfigured: await this.secrets.has(this.definition(runtime.config.id).credentialId) ||
        await this.secrets.has(mcpEnvironmentKey(runtime.config.id)) ||
        (await this.oauth.status(runtime.config.id)).state === "authorized",
      ...(runtime.config.transport === "http" ? { authentication: await this.oauth.status(runtime.config.id) } : {})
    })));
  }

  private validateRequest(request: McpServerRegistrationRequest): StoredMcpServer {
    const brainId = safeId(request?.brainId, "brain id");
    const id = safeId(request?.id, "server id");
    const label = safeLabel(request?.label);
    if (request.transport === "http") {
      return { brainId, id, label, transport: "http", url: validateHttpUrl(request.url), tools: [] };
    }
    if (request.transport !== "stdio") throw new Error("Invalid MCP transport.");
    if (typeof request.command !== "string" || !isAbsolute(request.command) || request.command.length > 32_000) {
      throw new Error("MCP stdio requires an absolute, already-installed executable path.");
    }
    const args = request.args ?? [];
    if (!Array.isArray(args) || args.some((value) => typeof value !== "string" || value.length > 32_000) || Buffer.byteLength(JSON.stringify(args)) > 1024 * 1024) {
      throw new Error("MCP stdio arguments are invalid.");
    }
    return { brainId, id, label, transport: "stdio", command: request.command, args: [...args], tools: [] };
  }

  private validateStoredConfig(value: unknown, registryTemplate = false): StoredMcpServer {
    const source = record(value);
    const base = this.validateRequest({
      brainId: source.brainId as string,
      id: source.id as string,
      label: source.label as string,
      transport: source.transport as "http" | "stdio",
      url: source.url as string | undefined,
      command: source.command as string | undefined,
      args: source.args as string[] | undefined
    });
    if (!Array.isArray(source.tools)) throw new Error("The MCP registry contains invalid tools.");
    const remoteNames = new Set<string>();
    const tools = source.tools.map((value): McpToolSummary => {
      const tool = record(value);
      const remoteName = typeof tool.remoteName === "string" ? tool.remoteName.trim() : "";
      if (!remoteName || remoteName.length > 256 || /[\0\r\n]/.test(remoteName) || remoteNames.has(remoteName)) {
        throw new Error("The MCP registry contains an invalid or duplicate tool name.");
      }
      remoteNames.add(remoteName);
      const expectedId = namespaceTool(base.brainId, base.id, remoteName);
      if (!registryTemplate && tool.id !== expectedId) throw new Error("The MCP registry contains a tampered tool id.");
      if (!Array.isArray(tool.actions) || tool.actions.length !== 1 || tool.actions[0] !== "call") {
        throw new Error("The MCP registry contains an invalid tool action.");
      }
      const inputSchema = record(tool.inputSchema);
      if (Buffer.byteLength(JSON.stringify(inputSchema), "utf8") > 4 * 1024 * 1024) {
        throw new Error("The MCP registry contains an oversized tool schema.");
      }
      return { id: expectedId, remoteName, actions: ["call"], inputSchema,
        ...this.toolMetadata(tool, true) };
    });
    return { ...base, tools };
  }

  async add(request: McpServerRegistrationRequest): Promise<McpServerSummary> {
    const runtime = await this.serialize(async () => {
      const config = this.validateRequest(request);
      if (request.scope !== undefined && !["all", "instance"].includes(request.scope)) throw new Error("Invalid MCP server scope.");
      if (request.defaultEnabled !== undefined && typeof request.defaultEnabled !== "boolean") throw new Error("Invalid MCP server default.");
      if (request.disabledBrainIds !== undefined && !Array.isArray(request.disabledBrainIds)) throw new Error("Invalid MCP excluded instances.");
      const disabledBrainIds = request.disabledBrainIds?.map((id) => safeId(id, "brain id")) ?? [];
      if (this.document.servers.some((server) => server.id === config.id)) throw new Error("An MCP server with this id already exists in the registry.");
      const credentialId = `mcp:registry:${config.id}`;
      if (request.environment !== undefined && config.transport !== "stdio") throw new Error("Only local stdio servers use environment credentials.");
      const environment = mcpEnvironment(request.environment);
      if (request.bearerToken?.trim()) await this.secrets.set(credentialId, request.bearerToken.trim());
      if (Object.keys(environment).length) {
        try { await this.secrets.set(mcpEnvironmentKey(config.id), JSON.stringify(environment)); }
        catch (error) { await this.secrets.remove(credentialId); throw error; }
      }
      const next = structuredClone(this.document);
      attachMcpInstance(next, config.brainId);
      next.lastConfiguredBrainId = config.brainId;
      const { brainId, ...definition } = config;
      next.servers.push({ ...definition, enabled: true, defaultEnabled: request.defaultEnabled ?? true,
        ...(request.scope === "instance" ? { onlyBrainId: brainId } : {}), credentialId, namespaceByBrain: {},
        ...(Object.keys(environment).length ? { environmentKeys: Object.keys(environment).sort() } : {}) });
      for (const id of disabledBrainIds) attachMcpInstance(next, id).overrides[config.id] = "off";
      try {
        await this.publish(next);
      } catch (error) {
        await this.secrets.remove(credentialId);
        await this.secrets.remove(mcpEnvironmentKey(config.id));
        throw error;
      }
      return this.runtimes.get(runtimeKey(brainId, config.id))!;
    });
    // Neither the remote handshake nor a neural write owns the registry mutex.
    // A master/instance Off change must be able to abort either immediately.
    if (!this.isEnabled(runtime)) return this.runtimeSummary(runtime);
    try { return await this.refreshRuntime(runtime); }
    catch (error) {
      // Authentication is an expected next step, not a failed registration.
      // Keep its declared endpoint and access switches for explicit user login.
      if (error instanceof McpAuthenticationRequiredError) {
        runtime.summary = this.summary(runtime.config, false, error.message);
        return this.runtimeSummary(runtime);
      }
      await this.serialize(async () => {
        if (this.runtimes.get(runtimeKey(runtime.config.brainId, runtime.config.id)) !== runtime) return;
        // A deliberate access revocation keeps the saved registration for later.
        if (!this.isEnabled(runtime)) return;
        const next = structuredClone(this.document);
        const server = next.servers.find((entry) => entry.id === runtime.config.id)!;
        next.servers = next.servers.filter((entry) => entry.id !== server.id);
        for (const instance of next.instances) {
          delete instance.overrides[server.id]; delete instance.learnedSchemas[server.id];
        }
        await this.publish(next);
        await this.secrets.remove(server.credentialId);
        await this.secrets.remove(mcpEnvironmentKey(server.id));
        await this.oauth.forget(server.id);
      });
      throw error;
    }
  }

  async remove(brainId: string, serverId: string): Promise<boolean> {
    safeId(brainId, "brain id"); safeId(serverId, "server id");
    return this.serialize(async () => {
      const server = this.document.servers.find((candidate) => candidate.id === serverId);
      if (!server) return false;
      if (server.onlyBrainId && server.onlyBrainId !== brainId) throw new Error("This MCP server belongs to another instance.");
      const next = structuredClone(this.document);
      next.servers = next.servers.filter((candidate) => candidate.id !== serverId);
      for (const instance of next.instances) {
        delete instance.overrides[serverId]; delete instance.learnedSchemas[serverId];
      }
      await this.publish(next);
      await this.secrets.remove(server.credentialId);
      await this.secrets.remove(mcpEnvironmentKey(server.id));
      await this.oauth.forget(server.id);
      return true;
    });
  }

  /** Explicit credential replacement never executes/retries a remote tool. */
  async updateCredentials(brainId: string, serverId: string, input: McpCredentialUpdate): Promise<McpServerSummary> {
    safeId(brainId, "brain id"); safeId(serverId, "server id");
    if (!input || typeof input !== "object" || Array.isArray(input) ||
      (input.bearerToken === undefined && input.environment === undefined)) throw new Error("A credential replacement is required.");
    if (input.bearerToken !== undefined && (typeof input.bearerToken !== "string" || input.bearerToken.length > 64_000 || /[\0\r\n]/.test(input.bearerToken))) throw new Error("Invalid MCP bearer credential.");
    const environment = input.environment === undefined ? undefined : mcpEnvironment(input.environment);
    await this.attachBrain(brainId);
    await this.serialize(async () => {
      const server = this.definition(serverId);
      if (server.onlyBrainId && server.onlyBrainId !== brainId) throw new Error("This MCP server belongs to another instance.");
      if ((server.transport === "http" && environment !== undefined) || (server.transport === "stdio" && input.bearerToken !== undefined)) throw new Error("Use the credential type for this server's transport.");
      const key = server.transport === "http" ? server.credentialId : mcpEnvironmentKey(serverId);
      const previous = await this.secrets.get(key);
      const previousOAuth = server.transport === "http" ? await this.oauth.credentialSnapshot(serverId) : undefined;
      const affected = [...this.runtimes.values()].filter(runtime => runtime.config.id === serverId);
      for (const runtime of affected) {
        this.endHttpSession(runtime);
        runtime.credentialsUpdating = true;
        runtime.controller.abort(new Error("MCP credentials are changing.")); runtime.stdio?.stop(); runtime.stdio = undefined;
        runtime.legacySse?.stop(); runtime.legacySse = undefined;
        runtime.initialized = false; runtime.sessionId = undefined; runtime.protocolVersion = undefined; runtime.capabilities = undefined;
        runtime.pagination?.clear(); this.loseSubscriptions(runtime);
      }
      try {
        if (server.transport === "http") await this.oauth.forget(serverId);
        const replacement = server.transport === "http" ? input.bearerToken!.trim()
          : Object.keys(environment!).length ? JSON.stringify(environment) : "";
        if (replacement) await this.secrets.set(key, replacement); else await this.secrets.remove(key);
        const next = structuredClone(this.document);
        const changed = next.servers.find(value => value.id === serverId)!;
        if (server.transport === "stdio") {
          if (Object.keys(environment!).length) changed.environmentKeys = Object.keys(environment!).sort();
          else delete changed.environmentKeys;
        }
        await this.registryStore.save(next);
        this.document = next;
      } catch (error) {
        try {
          if (previous !== undefined) await this.secrets.set(key, previous); else await this.secrets.remove(key);
          if (server.transport === "http") await this.oauth.restoreCredentialSnapshot(serverId, previousOAuth);
        } catch { throw new Error("Credential replacement failed and its original credentials could not be restored. Re-enter this server's credentials."); }
        throw error;
      } finally {
        for (const runtime of affected) { runtime.credentialsUpdating = false; runtime.summary = this.summary(runtime.config, false); }
        this.syncRuntimes();
      }
    });
    const runtime = this.runtimes.get(runtimeKey(brainId, serverId));
    if (!runtime) throw new Error("MCP server was removed while updating credentials.");
    return this.runtimeSummary(runtime);
  }

  private async httpRpc(
    runtime: McpRuntime,
    method: string,
    params: Record<string, unknown>,
    notification = false,
    signal?: AbortSignal,
    context?: McpRpcContext
  ): Promise<JsonRpcResponse | undefined> {
    const id = notification ? undefined : randomUUID();
    this.requireEnabled(runtime);
    const token = await this.tokenFor(runtime);
    this.requireEnabled(runtime);
    const body = encodeRpcMessage({ jsonrpc: "2.0", ...(id ? { id } : {}), method, params });
    const sessionId = runtime.sessionId;
    const sessionEpoch = runtime.sessionEpoch ?? 0;
    const endpoint = runtime.config.url!;
    const headers = {
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
        "mcp-protocol-version": runtime.protocolVersion ?? MCP_PROTOCOL_VERSION,
        ...(sessionId ? { "mcp-session-id": sessionId } : {}),
        ...(token ? { authorization: `Bearer ${token}` } : {})
    };
    const deadline = new McpRequestDeadline(120_000);
    const requestSignal = AbortSignal.any([runtime.controller.signal,
      signal ?? new AbortController().signal, deadline.signal]);
    const call: McpActiveCall | undefined = id ? { id, method, signal: requestSignal, deadline, ...(context ? { context } : {}) } : undefined;
    const finishCall = call ? this.beginCall(runtime, call) : () => undefined;
    if (runtime.legacySse) {
      try {
        if (notification) { await runtime.legacySse.notify(method, params, requestSignal); return undefined; }
        await context?.beforeSend?.(); requestSignal.throwIfAborted(); this.requireEnabled(runtime);
        return await runtime.legacySse.request(method, params, requestSignal, id, !context?.task);
      } finally { finishCall(); deadline.close(); }
    }
    let dispatched = false;
    const cancel = (): void => {
      if (!dispatched || !id || method === "initialize" || context?.task) return;
      // Closing a response stream is not a protocol cancellation. Send one
      // separate best-effort notification for the already-dispatched request;
      // never retry the effect or claim the remote process acknowledged Stop.
      void this.fetcher(endpoint, { method: "POST", headers, redirect: "error",
        body: encodeRpcMessage({ jsonrpc: "2.0", method: "notifications/cancelled",
          params: { requestId: id, reason: "The caller stopped this request." } }),
        signal: AbortSignal.timeout(5_000) })
        .then(async (response) => { await response.body?.cancel(); }).catch(() => undefined);
    };
    requestSignal.addEventListener("abort", cancel, { once: true });
    try {
      requestSignal.throwIfAborted();
      await context?.beforeSend?.(); requestSignal.throwIfAborted(); this.requireEnabled(runtime);
      dispatched = true;
      const response = await this.fetcher(endpoint, {
        method: "POST", headers, body, redirect: "error", signal: requestSignal
      });
      if (!response.ok) {
        await response.body?.cancel().catch(() => undefined);
        if (method === "initialize" && !sessionId && [400, 404, 405].includes(response.status)) {
          const legacy: McpLegacySseSession = new McpLegacySseSession({ endpoint, fetcher: this.fetcher,
            maximumBytes: MAX_MCP_MESSAGE_BYTES, signal: runtime.controller.signal,
            headers: async () => { const current = await this.tokenFor(runtime); return {
              "mcp-protocol-version": runtime.protocolVersion ?? MCP_PROTOCOL_VERSION,
              ...(current ? { authorization: `Bearer ${current}` } : {}) }; },
            onMessage: message => runtime.legacySse === legacy ? this.serverMessage(runtime, message) : undefined,
            onClosed: () => {
              if (runtime.legacySse !== legacy) return;
              runtime.legacySse = undefined; runtime.initialized = false;
              runtime.pagination?.clear(); this.loseSubscriptions(runtime);
              runtime.summary = this.summary(runtime.config, false, "The legacy MCP connection closed; pending requests were not repeated.");
            } });
          runtime.legacySse = legacy;
          runtime.transportObservation = { ...runtime.transportObservation, eventStream: "listening", httpMode: "legacy-sse" };
          return legacy.request(method, params, requestSignal, id);
        }
        if (response.status === 404 && sessionId && method !== "initialize") {
          this.expireHttpSession(runtime);
          throw new McpSessionExpiredError();
        }
        if (response.status === 401 || (response.status === 403 && response.headers.get("www-authenticate"))) {
          this.oauth.challenge(runtime.config.id, response.headers.get("www-authenticate"));
          throw new McpAuthenticationRequiredError();
        }
        throw new Error(`MCP server returned HTTP ${response.status}.`);
      }
      if (method === "initialize") {
        const initializedSessionId = response.headers.get("mcp-session-id");
        if (initializedSessionId && !/^[\x21-\x7e]+$/.test(initializedSessionId)) {
          await response.body?.cancel().catch(() => undefined);
          throw new Error("MCP server returned an invalid session id.");
        }
        runtime.sessionId = initializedSessionId ?? undefined;
        runtime.sessionToken = token;
        runtime.transportObservation = { ...runtime.transportObservation, eventStream: "off", httpMode: "streamable-http" };
      }
      if (notification || response.status === 202 || response.status === 204) {
        await response.body?.cancel().catch(() => undefined);
        return undefined;
      }
      let streamed = response;
      let resume: McpResumableStreamError | undefined;
      for (;;) {
        try {
          return await readMcpRpcResponse(streamed, id!, MAX_MCP_MESSAGE_BYTES,
            (message) => (runtime.sessionEpoch ?? 0) === sessionEpoch ? this.serverMessage(runtime, message, call) : undefined, requestSignal,
            { resumable: method !== "initialize", ...(resume ? { lastEventId: resume.lastEventId, retryMilliseconds: resume.retryMilliseconds } : {}),
              acceptEvent: event => this.acceptSseEvent(runtime, event), delivered: event => this.deliveredSseEvent(runtime, event) });
        } catch (error) {
          if (!(error instanceof McpResumableStreamError)) throw error;
          resume = error;
        }
        for (;;) {
          await this.waitForHttpRetry(resume.retryMilliseconds, requestSignal);
          this.requireEnabled(runtime);
          if ((runtime.sessionEpoch ?? 0) !== sessionEpoch) throw new Error("The MCP session changed before an earlier result arrived. Its outcome is unknown and was not repeated.");
          try {
            streamed = await this.fetcher(endpoint, { method: "GET", redirect: "error", signal: requestSignal,
              headers: { ...headers, accept: "text/event-stream", "last-event-id": this.lastEventHeader(resume.lastEventId) } });
          } catch { requestSignal.throwIfAborted(); continue; }
          if (streamed.ok) break;
          await streamed.body?.cancel().catch(() => undefined);
          if (streamed.status === 404 && sessionId) {
            this.expireHttpSession(runtime);
            throw new Error("The MCP session expired while awaiting an earlier result. Its outcome is unknown and was not repeated.");
          }
          if (streamed.status >= 500) continue;
          throw new Error(`MCP response resumption returned HTTP ${streamed.status}; the original request was not repeated.`);
        }
      }
    } finally {
      requestSignal.removeEventListener("abort", cancel);
      finishCall();
      deadline.close();
    }
  }

  private async serverMessage(runtime: McpRuntime, message: McpRpcMessage, originatingCall?: McpActiveCall | null): Promise<void> {
    const notification = message.id === undefined || message.id === null;
    if (notification && (runtime.controller.signal.aborted || !this.isEnabled(runtime) ||
      this.runtimes.get(runtimeKey(runtime.config.brainId, runtime.config.id)) !== runtime)) return;
    if (notification && message.method === "notifications/tools/list_changed") {
      runtime.toolsDirty = true;
      runtime.toolRevision = (runtime.toolRevision ?? 0) + 1;
    }
    const capabilities = negotiatedMcpCapabilities(runtime.capabilities ?? {});
    if (notification && runtime.protocolVersion === "2025-11-25" && this.taskStore && message.method === "notifications/tasks/status") {
      try {
        const task = checkedMcpRemoteTask(message.params), lease = [...this.taskLeases.values()].find(item => item.runtime === runtime && item.record.remote?.taskId === task.taskId && item.scope);
        // Notifications never create/adopt a task or revive idle/revoked work.
        // Explicit status polling remains authoritative and does not rely on them.
        if (lease?.scope) await this.trackTask(runtime.config.brainId, lease.scope.signal, async signal => {
          await this.taskCurrent(lease, lease.scope?.context, signal); lease.record = await this.taskStore!.update(runtime.config.brainId, lease.record.taskRef, { remote: task });
          await this.taskCurrent(lease, lease.scope?.context, signal);
        });
      } catch { /* malformed/stale unsolicited task data is not a new action */ }
      return;
    }
    if (notification && runtime.initialized && message.method === "notifications/tools/list_changed" && capabilities.tools?.listChanged) {
      this.capabilityEvent(runtime, "tools-list-changed");
      this.appHost?.serverNotification?.(this.interactionOrigin(runtime), "notifications/tools/list_changed");
    }
    if (notification && runtime.initialized && message.method === "notifications/message" && capabilities.logging && this.standardHost.log) {
      // Diagnostic failure cannot replay an RPC, become model input or kill its reader.
      await this.standardHost.log(this.interactionOrigin(runtime), message.params, runtime.controller.signal).catch(() => undefined);
    }
    if (notification && message.method === "notifications/cancelled") {
      const id = record(message.params).requestId;
      if (typeof id === "string" || typeof id === "number") runtime.serverRequests?.get(`${typeof id}:${id}`)?.controller.abort(new Error("The MCP server cancelled its request."));
    }
    if (notification && message.method === "notifications/elicitation/complete" && this.hostInteractions.completeElicitation) {
      const id = record(message.params).elicitationId;
      if (typeof id === "string") void this.hostInteractions.completeElicitation(this.interactionOrigin(runtime), id).catch(() => undefined);
    }
    if (notification && message.method === "notifications/resources/list_changed" && capabilities.resources?.listChanged) {
      runtime.pagination?.delete("resources/list"); runtime.pagination?.delete("resources/templates/list");
      this.capabilityEvent(runtime, "resources-list-changed");
      if (runtime.initialized) this.appHost?.serverNotification?.(this.interactionOrigin(runtime), "notifications/resources/list_changed");
    }
    if (notification && message.method === "notifications/prompts/list_changed" && capabilities.prompts?.listChanged) {
      runtime.pagination?.delete("prompts/list"); this.capabilityEvent(runtime, "prompts-list-changed");
    }
    if (notification && message.method === "notifications/resources/updated" && capabilities.resources?.subscribe) {
      let uri: string | undefined;
      try { uri = mcpUri(record(message.params).uri); } catch { /* ignore invalid notifications */ }
      const subscription = uri ? runtime.subscriptions?.get(uri) : undefined;
      if (subscription?.state === "active") {
        subscription.updates += 1; subscription.updatedAt = new Date().toISOString();
        this.capabilityEvent(runtime, "resource-updated", uri);
      } else {
        // An update may precede the matching subscribe response in the same
        // stream. Hold it until the subscription itself is acknowledged.
        const subscribing = uri ? runtime.subscribing?.get(uri) : undefined;
        if (subscribing) { subscribing.updates += 1; subscribing.updatedAt = new Date().toISOString(); }
      }
    }
    // Do not advertise request capabilities until a host handler is installed.
    // A server that nevertheless sends a request receives an explicit error.
    if (message.method && message.id !== undefined && message.id !== null) {
      this.requireEnabled(runtime);
      if (message.method === "elicitation/create" && this.hostInteractions.elicit || message.method === "sampling/createMessage" && this.hostInteractions.sample || message.method === "roots/list" && this.standardHost.roots) {
        const key = `${typeof message.id}:${message.id}`;
        runtime.serverRequests ??= new Map();
        const hash = sha256(JSON.stringify({ method: message.method, params: message.params }));
        const existing = runtime.serverRequests.get(key);
        if (existing) {
          if (existing.hash !== hash) await this.replyServer(runtime, { jsonrpc: "2.0", id: message.id, error: { code: -32600, message: "A pending server request ID was reused with different contents." } });
          return;
        }
        const candidates = originatingCall === undefined ? [...(runtime.activeCalls?.values() ?? [])].filter(call => call.method !== "initialize" && call.context) : [];
        let parent = originatingCall ?? (candidates.length === 1 ? candidates[0] : undefined);
        let relatedId: string | undefined;
        try { if (message.params && typeof message.params === "object") relatedId = relatedMcpTaskId(message.params); }
        catch (error) { await this.replyServer(runtime, { jsonrpc: "2.0", id: message.id, error: { code: -32602, message: error instanceof Error ? error.message : String(error) } }); return; }
        const taskCandidates = relatedId === undefined ? [] : [...this.taskLeases.values()].filter(lease => lease.runtime === runtime && lease.scope &&
          !lease.controller.signal.aborted && (lease.record.remote?.taskId === relatedId || lease.record.remote === undefined && lease.scope.method === "tools/call"));
        if (relatedId !== undefined) {
          // No arbitrary sole active RPC can lend its task/turn to this request.
          parent = taskCandidates.find(lease => lease.record.remote?.taskId === relatedId)?.scope;
          if (!taskCandidates.length) { await this.replyServer(runtime, { jsonrpc: "2.0", id: message.id, error: { code: -32000, message: "This related task has no currently authorized owned operation; explicitly open its result under current permissions." } }); return; }
        } else if (parent?.context?.task && message.method !== "roots/list") {
          await this.replyServer(runtime, { jsonrpc: "2.0", id: message.id, error: { code: -32602, message: "Task-related input omitted its exact related-task metadata." } }); return;
        }
        const controller = new AbortController();
        const signal = AbortSignal.any([runtime.controller.signal, controller.signal, parent?.signal ?? new AbortController().signal]);
        const epoch = runtime.sessionEpoch ?? 0;
        const pending = { controller, hash, ...(parent ? { parentId: parent.id } : {}) };
        runtime.serverRequests.set(key, pending);
        const releaseDeadline = message.method === "roots/list" ? undefined : parent?.deadline?.pause();
        // Do not block the HTTP/stdio reader while a human reviews: it must
        // continue receiving cancellation, other requests and the final reply.
        const input = (async () => {
          let releaseTaskScope: (() => void) | undefined, taskLease: McpTaskLease | undefined;
          try {
            if (!runtime.initialized) throw new McpHostRequestError(-32000, "MCP user interactions require an initialized connection.");
            if (relatedId !== undefined) {
              // The reader keeps running while a pre-ACK input waits. A received
              // ACK must prove the actual task id before any host review begins.
              await Promise.allSettled(taskCandidates.map(lease => lease.acknowledgement)); signal.throwIfAborted();
              const matches = taskCandidates.filter(lease => lease.record.remote?.taskId === relatedId && lease.scope && !lease.controller.signal.aborted);
              if (matches.length !== 1) throw new McpHostRequestError(-32000, "The related task id was not proved by one exact current creation/management scope.");
              taskLease = matches[0]!; parent = taskLease.scope;
              if (!parent || parent.signal.aborted) throw new McpHostRequestError(-32000, "The task input lost its current operation scope.");
              const sourceSignal = parent.signal, abortTask = (): void => controller.abort(sourceSignal.reason);
              sourceSignal.addEventListener("abort", abortTask, { once: true }); if (sourceSignal.aborted) abortTask(); releaseTaskScope = () => sourceSignal.removeEventListener("abort", abortTask);
              await this.taskCurrent(taskLease, parent.context, signal);
            }
            if (message.method === "roots/list" && message.params !== undefined && (!message.params || typeof message.params !== "object" || Array.isArray(message.params) ||
                Object.keys(message.params).some(key => key !== "_meta"))) throw new McpHostRequestError(-32602, "roots/list accepts no server-supplied filesystem paths or filters.");
            if (relatedId === undefined && message.method === "sampling/createMessage" && originatingCall === undefined && candidates.length > 1) throw new McpHostRequestError(-32000, "The sampling request has an ambiguous originating task; it was not assigned to another turn.");
            const origin = this.interactionOrigin(runtime, parent);
            const result = message.method === "elicitation/create"
              ? await this.hostInteractions.elicit!(origin, message.params, signal)
              : message.method === "sampling/createMessage" ? await this.hostInteractions.sample!(origin, message.params, signal)
                : await this.standardHost.roots!(origin, signal);
            signal.throwIfAborted(); this.requireEnabled(runtime);
            if (taskLease) await this.taskCurrent(taskLease, parent?.context, signal);
            const presented = relatedId === undefined ? result : { ...result, _meta: { ...record(record(result)._meta), "io.modelcontextprotocol/related-task": { taskId: relatedId } } };
            if ((runtime.sessionEpoch ?? 0) === epoch) await this.replyServer(runtime, { jsonrpc: "2.0", id: message.id, result: presented }, taskLease ? () => this.taskCurrent(taskLease!, parent?.context, signal) : undefined);
          } catch (error) {
            if (!signal.aborted && (runtime.sessionEpoch ?? 0) === epoch) await this.replyServer(runtime, { jsonrpc: "2.0", id: message.id,
              error: { code: error instanceof McpHostRequestError ? error.code : -32603, message: (error instanceof Error ? error.message : String(error)).slice(0, 1_000) } }).catch(() => undefined);
          } finally { releaseTaskScope?.(); releaseDeadline?.(); if (runtime.serverRequests?.get(key) === pending) runtime.serverRequests.delete(key); }
        })();
        for (const lease of taskCandidates) { lease.inputs.add(input); void input.finally(() => { lease.inputs.delete(input); }).catch(() => undefined); }
        void input.catch(() => undefined);
        return;
      }
      const reply = message.method === "ping"
        ? { jsonrpc: "2.0", id: message.id, result: {} }
        : { jsonrpc: "2.0", id: message.id, error: { code: -32601, message: "This client capability is not enabled." } };
      await this.replyServer(runtime, reply);
    }
  }

  private async replyServer(runtime: McpRuntime, reply: Record<string, unknown>, requireCurrent?: () => Promise<void>): Promise<void> {
    await requireCurrent?.();
    this.requireEnabled(runtime);
    if (runtime.config.transport === "stdio") { runtime.stdio?.write(reply); return; }
    if (runtime.legacySse) { await runtime.legacySse.reply(reply); return; }
    const token = await this.tokenFor(runtime);
    await requireCurrent?.();
    this.requireEnabled(runtime);
    const response = await this.fetcher(runtime.config.url!, { method: "POST", redirect: "error",
        headers: { "content-type": "application/json", accept: "application/json, text/event-stream",
          "mcp-protocol-version": runtime.protocolVersion ?? MCP_PROTOCOL_VERSION,
          ...(runtime.sessionId ? { "mcp-session-id": runtime.sessionId } : {}), ...(token ? { authorization: `Bearer ${token}` } : {}) },
        body: encodeRpcMessage(reply), signal: AbortSignal.any([runtime.controller.signal, AbortSignal.timeout(120_000)]) });
    await response.body?.cancel().catch(() => undefined);
    if (!response.ok) throw new Error(`MCP response delivery returned HTTP ${response.status}.`);
  }

  private async rpc(
    runtime: McpRuntime,
    method: string,
    params: Record<string, unknown>,
    signal?: AbortSignal,
    context?: McpRpcContext
  ): Promise<unknown> {
    this.requireEnabled(runtime);
    const combinedSignal = AbortSignal.any([runtime.controller.signal, signal ?? new AbortController().signal]);
    combinedSignal.throwIfAborted();
    let response: JsonRpcResponse | undefined;
    if (runtime.config.transport === "http") {
      response = await this.httpRpc(runtime, method, params, false, combinedSignal, context);
    } else {
      if (!runtime.stdio) {
        const session: StdioRpcSession = new StdioRpcSession(runtime.config.command!, runtime.config.args ?? [],
          (message): void | Promise<void> => runtime.stdio === session ? this.serverMessage(runtime, message) : undefined, () => {
            if (runtime.stdio !== session) return;
            runtime.stdio = undefined;
            runtime.initialized = false;
            runtime.pagination?.clear(); this.loseSubscriptions(runtime);
            runtime.summary = this.summary(runtime.config, false, "The local MCP connection closed.");
          }, async () => {
            this.requireEnabled(runtime);
            const encoded = await this.secrets.get(mcpEnvironmentKey(runtime.config.id));
            const environment = mcpEnvironment(encoded ? JSON.parse(encoded) : undefined);
            const expected = [...(this.definition(runtime.config.id).environmentKeys ?? [])].sort();
            if (JSON.stringify(Object.keys(environment).sort()) !== JSON.stringify(expected)) throw new Error("Local MCP environment credentials are missing or inconsistent. Configure them again.");
            this.requireEnabled(runtime); return environment;
          });
        runtime.stdio = session;
      }
      const stdio = runtime.stdio;
      if (!stdio) throw new Error("MCP stdio connection closed before dispatch.");
      const requestId = randomUUID();
      const deadline = new McpRequestDeadline(120_000);
      const requestSignal = AbortSignal.any([combinedSignal, deadline.signal]);
      const finish = this.beginCall(runtime, { id: requestId, method, signal: requestSignal, deadline, ...(context ? { context } : {}) });
      try { await context?.beforeSend?.(); requestSignal.throwIfAborted(); this.requireEnabled(runtime); response = await stdio.request(method, params, requestSignal, requestId, deadline, !context?.task); }
      finally { finish(); deadline.close(); }
    }
    if (response?.error) {
      if (!context?.task && response.error.code === -32042 && this.hostInteractions.elicit && runtime.initialized) {
        const elicitations = record(response.error.data).elicitations;
        if (!Array.isArray(elicitations) || !elicitations.length) throw new Error("MCP URL elicitation error omitted its required URL requests.");
        const requests = elicitations.map(value => mcpElicitation(value));
        const identities = new Set<string>();
        for (const request of requests) {
          if (request.mode !== "url" || identities.has(request.elicitationId)) throw new Error("MCP URL elicitation error contains invalid or duplicate URL requests.");
          identities.add(request.elicitationId);
        }
        const origin = this.interactionOrigin(runtime, { id: "url-error", method, signal: combinedSignal, context });
        for (const request of requests) {
          const reviewed = await this.hostInteractions.elicit(origin, request, combinedSignal);
          if (reviewed.action !== "accept") throw new Error("The required URL interaction was declined or cancelled. The original action was not retried.");
        }
        throw new Error("The server requires out-of-band URL interaction. Reviewed browser navigation is not confirmation of completion; the original action was not retried. Invoke it again explicitly when ready.");
      }
      throw new McpRpcFailure(method, { code: typeof response.error.code === "number" ? response.error.code : -32603,
        message: String(response.error.message ?? response.error.code ?? "unknown error"), ...(response.error.data !== undefined ? { data: response.error.data } : {}) });
    }
    if (!response || !Object.hasOwn(response, "result") || Object.hasOwn(response, "error")) {
      throw new Error(`MCP ${method} returned no valid response result.`);
    }
    return response.result;
  }

  private initializeRuntime(runtime: McpRuntime, signal?: AbortSignal): Promise<void> {
    this.requireEnabled(runtime);
    if (runtime.initialized) return Promise.resolve();
    if (runtime.initialization) return runtime.initialization;
    const operation = this.handshakeRuntime(runtime, AbortSignal.any([runtime.controller.signal,
      signal ?? new AbortController().signal]));
    runtime.initialization = operation;
    void operation.finally(() => {
      if (runtime.initialization === operation) runtime.initialization = undefined;
    }).catch(() => undefined);
    return operation;
  }

  private async handshakeRuntime(runtime: McpRuntime, signal: AbortSignal): Promise<void> {
    const result = record(await this.rpc(runtime, "initialize", {
      protocolVersion: MCP_PROTOCOL_VERSION,
      capabilities: { ...(this.hostInteractions.elicit ? { elicitation: { form: {}, url: {} } } : {}),
        ...(this.hostInteractions.sample ? { sampling: {} } : {}),
        ...(this.taskStore ? { tasks: {} } : {}),
        ...(this.standardHost.roots ? { roots: { listChanged: true } } : {}),
        ...(this.appHost ? { extensions: { [MCP_APP_EXTENSION]: { mimeTypes: [MCP_APP_MIME] } } } : {}) },
      clientInfo: { name: "Omni AGI Studio", version: "2.0.0" }
    }, signal));
    if (typeof result.protocolVersion !== "string" || !SUPPORTED_MCP_PROTOCOL_VERSIONS.has(result.protocolVersion)) {
      throw new Error("MCP server selected an unsupported protocol version.");
    }
    runtime.protocolVersion = result.protocolVersion;
    // Validate standard declarations before making any client-owned wrappers available.
    negotiatedMcpCapabilities(result.capabilities);
    runtime.capabilities = record(result.capabilities);
    runtime.pagination?.clear(); this.loseSubscriptions(runtime, "session-initialized");
    if (runtime.legacySse) runtime.transportObservation = { ...runtime.transportObservation, eventStream: "listening", httpMode: "legacy-sse" };
    if (runtime.config.transport === "http") {
      await this.httpRpc(runtime, "notifications/initialized", {}, true, signal);
    } else {
      runtime.stdio!.notify("notifications/initialized");
    }
    signal.throwIfAborted();
    this.requireEnabled(runtime);
    runtime.initialized = true;
    runtime.summary = { ...this.summary(runtime.config, true), protocolVersion: runtime.protocolVersion,
      capabilities: runtime.capabilities };
    this.capabilityEvent(runtime, "connection-ready");
    this.startHttpEvents(runtime);
  }

  private async listRemoteTools(runtime: McpRuntime, signal?: AbortSignal, sessionRetried = false): Promise<McpToolSummary[]> {
    await this.initializeRuntime(runtime, signal);
    const capabilities = negotiatedMcpCapabilities(runtime.capabilities);
    // Older compatible servers historically omitted all declarations. Retain
    // that tools-only path, but a resource/prompt-only server need not implement tools/list.
    if (!capabilities.tools && Object.keys(capabilities).length) return [];
    const found: McpToolSummary[] = [];
    const remoteNames = new Set<string>();
    const cursors = new Set<string>();
    let cursor: string | undefined;
    do {
      let response: unknown;
      try { response = await this.rpc(runtime, "tools/list", cursor ? { cursor } : {}, signal); }
      catch (error) {
        if (error instanceof McpSessionExpiredError && !sessionRetried) return this.listRemoteTools(runtime, signal, true);
        throw error;
      }
      const result = record(response);
      if (!Array.isArray(result.tools)) throw new Error("MCP tools/list returned no tool array.");
      for (const value of result.tools) {
        const tool = record(value);
        const remoteName = typeof tool.name === "string" ? tool.name.trim() : "";
        if (!remoteName || remoteName.length > 256 || /[\0\r\n]/.test(remoteName) || remoteNames.has(remoteName)) {
          throw new Error("MCP server returned an invalid tool name.");
        }
        remoteNames.add(remoteName);
        const inputSchema = record(tool.inputSchema);
        const encodedSchema = JSON.stringify(inputSchema);
        if (Buffer.byteLength(encodedSchema) > 4 * 1024 * 1024) {
          throw new Error("MCP tool input schema exceeded 4 MiB.");
        }
        found.push({
          id: namespaceTool(runtime.config.brainId, runtime.config.namespaceId ?? runtime.config.id, remoteName),
          remoteName,
          actions: ["call"],
          inputSchema,
          ...this.toolMetadata(tool)
        });
      }
      cursor = typeof result.nextCursor === "string" && result.nextCursor ? result.nextCursor : undefined;
      if (cursor && cursors.has(cursor)) throw new Error("MCP tools/list repeated a pagination cursor.");
      if (cursor) cursors.add(cursor);
    } while (cursor);
    return found;
  }

  async refresh(brainId: string, serverId: string): Promise<McpServerSummary> {
    await this.attachBrain(brainId);
    const key = runtimeKey(safeId(brainId, "brain id"), safeId(serverId, "server id"));
    const runtime = this.runtimes.get(key);
    if (!runtime) throw new Error("MCP server was not found.");
    this.requireEnabled(runtime);
    return this.refreshRuntime(runtime);
  }

  private async refreshRuntime(runtime: McpRuntime, learn = true): Promise<McpServerSummary> {
    if (!runtime.refresh) {
      const operation = this.discoverRuntime(runtime);
      runtime.refresh = operation;
      void operation.finally(() => {
        if (runtime.refresh === operation) runtime.refresh = undefined;
      }).catch(() => undefined);
    }
    await runtime.refresh;
    // Dispatch may run inside the neural chat writer. It must not wait on a
    // second writer to persist permissions or learn schemas before returning
    // that very tool result. New IDs remain Off until default Ask is persisted.
    // Pre-chat preparation commits pending changed schemas at the next boundary.
    if (learn) {
      for (const bound of this.runtimes.values()) {
        if (bound.config.id === runtime.config.id && this.isEnabled(bound)) await this.ensurePermissions(bound.config.brainId, this.boundTools(bound.config));
      }
      await this.learnSchemas(runtime, runtime.controller.signal);
    }
    return this.runtimeSummary(runtime);
  }

  private async discoverRuntime(runtime: McpRuntime): Promise<McpServerSummary> {
    this.requireEnabled(runtime);
    const { brainId, id: serverId } = runtime.config;
    const signal = runtime.controller.signal;
    const revision = runtime.toolRevision ?? 0;
    const tools = await this.listRemoteTools(runtime, signal);
    const capabilities = negotiatedMcpCapabilities(runtime.capabilities);
    await this.serialize(async () => {
      signal.throwIfAborted();
      this.requireEnabled(runtime);
      const next = structuredClone(this.document);
      const definition = next.servers.find((server) => server.id === serverId)!;
      definition.tools = tools; definition.capabilities = capabilities;
      await this.publish(next);
    });
    runtime.toolsDirty = revision !== (runtime.toolRevision ?? 0);
    return this.runtimeSummary(runtime);
  }

  private toolMetadata(tool: Record<string, unknown>, stored = false): Partial<McpToolSummary> {
    const taskDeclaration = stored ? tool.taskSupport === undefined ? undefined : { taskSupport: tool.taskSupport } : tool.execution;
    const taskSupport = mcpTaskSupport(taskDeclaration);
    const metadata: Partial<McpToolSummary> = {
      ...(typeof tool.description === "string" ? { description: tool.description } : {}),
      ...(typeof tool.title === "string" ? { title: tool.title } : {}),
      ...(tool.outputSchema ? { outputSchema: record(tool.outputSchema) } : {}),
      ...(tool.annotations ? { annotations: record(tool.annotations) } : {}),
      ...(tool._meta || tool.metadata ? { metadata: record(tool._meta ?? tool.metadata) } : {}),
      ...(taskDeclaration !== undefined ? { taskSupport } : {})
    };
    mcpToolUi(metadata);
    return metadata;
  }

  private schemaHash(config: StoredMcpServer): string {
    return sha256(JSON.stringify({ tools: this.boundTools(config), capabilities: config.capabilities ?? {} }));
  }

  private async runtimeSummary(runtime: McpRuntime): Promise<McpServerSummary> {
    return { ...runtime.summary, tools: structuredClone(runtime.summary.tools),
      ...this.observedCapabilities(runtime),
      credentialConfigured: await this.secrets.has(this.definition(runtime.config.id).credentialId) ||
        await this.secrets.has(mcpEnvironmentKey(runtime.config.id)) ||
        (await this.oauth.status(runtime.config.id)).state === "authorized",
      ...(runtime.config.transport === "http" ? { authentication: await this.oauth.status(runtime.config.id) } : {}) };
  }

  private async tokenFor(runtime: McpRuntime): Promise<string | undefined> {
    this.requireEnabled(runtime);
    const token = runtime.config.transport === "http" ? await this.oauth.accessToken(runtime.config.id, runtime.config.url!, () => this.requireEnabled(runtime), runtime.controller.signal) : undefined;
    return token ?? this.secrets.get(this.definition(runtime.config.id).credentialId);
  }

  private authRuntime(brainId: string, serverId: string): McpRuntime {
    const runtime = this.runtimes.get(runtimeKey(safeId(brainId, "brain id"), safeId(serverId, "server id")));
    if (!runtime || runtime.config.transport !== "http") throw new Error("HTTP MCP server was not found.");
    return runtime;
  }
  async authStatus(brainId: string, serverId: string): Promise<McpOAuthStatus> {
    await this.attachBrain(brainId); return this.oauth.status(this.authRuntime(brainId, serverId).config.id);
  }
  async authorize(brainId: string, serverId: string, input: McpOAuthStart): Promise<McpOAuthStatus> {
    await this.attachBrain(brainId);
    const runtime = this.authRuntime(brainId, serverId);
    this.requireEnabled(runtime);
    return this.oauth.begin(serverId, runtime.config.url!, input, () => this.requireEnabled(runtime));
  }
  cancelAuthorization(brainId: string, serverId: string): boolean {
    const runtime = this.authRuntime(brainId, serverId);
    return this.oauth.cancel(runtime.config.id);
  }
  async disconnectAuthorization(brainId: string, serverId: string): Promise<void> {
    const runtime = this.authRuntime(brainId, serverId);
    // Disconnect is permitted even when this connection is switched off.
    const revocation = this.oauth.disconnect(runtime.config.id, runtime.config.url!, () => undefined)
      .then(() => ({ ok: true as const }), (error: unknown) => ({ ok: false as const, error }));
    await this.secrets.remove(this.definition(serverId).credentialId);
    for (const bound of this.runtimes.values()) if (bound.config.id === serverId) {
      this.endHttpSession(bound);
      bound.controller.abort(new Error("The MCP account was disconnected."));
      bound.legacySse?.stop(); bound.legacySse = undefined;
      bound.controller = new AbortController();
      bound.initialized = false; bound.sessionId = undefined;
      bound.pagination?.clear(); this.loseSubscriptions(bound);
      bound.summary = this.summary(bound.config, false);
    }
    const result = await revocation;
    if (!result.ok) throw result.error;
  }

  /** Called at the active instance boundary; never silently injects descriptions into a prompt. */
  async learnCapabilities(brainId: string, signal?: AbortSignal): Promise<McpServerSummary[]> {
    signal?.throwIfAborted();
    await this.attachBrain(brainId);
    for (const runtime of this.runtimes.values()) {
      signal?.throwIfAborted();
      if (runtime.config.brainId !== brainId || !this.isEnabled(runtime)) continue;
      if (runtime.toolsDirty) await this.refreshRuntime(runtime, false);
      if (this.boundTools(runtime.config).length) await this.learnSchemas(runtime, signal);
    }
    signal?.throwIfAborted();
    return this.list(brainId);
  }

  private async learnSchemas(runtime: McpRuntime, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    this.requireEnabled(runtime);
    const hash = this.schemaHash(runtime.config);
    const { brainId, id: serverId } = runtime.config;
    if (this.document.instances.find((instance) => instance.brainId === brainId)?.learnedSchemas[serverId] === hash) return;
    // Concurrent refresh/chat callers share one experience transaction. A newly
    // discovered schema gets its own receipt after the prior transaction ends.
    if (runtime.learning) {
      const existing = runtime.learning;
      await existing.promise;
      signal?.throwIfAborted();
      if (existing.hash === hash || !this.isEnabled(runtime)) return;
      return this.learnSchemas(runtime, signal);
    }
    const operation = this.commitSchemas(runtime, hash,
      AbortSignal.any([runtime.controller.signal, signal ?? new AbortController().signal]));
    runtime.learning = { hash, promise: operation };
    try { await operation; }
    finally { if (runtime.learning?.promise === operation) runtime.learning = undefined; }
  }

  private async commitSchemas(runtime: McpRuntime, hash: string, signal: AbortSignal): Promise<void> {
    const { brainId, id: serverId } = runtime.config;
    const tools = this.boundTools(runtime.config).filter(tool => tool.actions.length && mcpToolUi(tool).visibility.includes("model"));
    let neuralError: string | undefined;
    try {
      const content = JSON.stringify({
        format: "omni-mcp-capability-schema",
        version: 1,
        server: { id: runtime.config.id, transport: runtime.config.transport },
        tools: tools.map((tool) => ({
          id: tool.id,
          remoteName: tool.remoteName,
          actions: tool.actions,
          ...(tool.actions.length === 1 && tool.actions[0] === "call" ? { action: "call" } : {}),
          inputSchema: tool.inputSchema,
          ...(tool.actionInputSchemas ? { actionInputSchemas: tool.actionInputSchemas } : {}),
          ...this.toolMetadata(tool as unknown as Record<string, unknown>, true),
          ...(tool.clientOperation ? { origin: "client-operation", clientOperation: tool.clientOperation,
            resultAuthority: "untrusted-task-data" } : { origin: "server-tool" })
        })),
        capabilities: runtime.config.capabilities ?? {},
        behavioralInstructions: false
      });
      await this.brain.learnStructuredExperience(brainId, {
        content,
        name: `MCP capability schemas · ${runtime.config.label}`,
        sourceLabel: `MCP ${runtime.config.id} typed capability schemas`,
        provenanceUrl: runtime.config.transport === "http" ? runtime.config.url : undefined,
        license: "User-installed MCP capability schema"
      }, signal);
      await this.serialize(async () => {
        signal.throwIfAborted();
        this.requireEnabled(runtime);
        if (this.schemaHash(runtime.config) !== hash) return;
        const next = structuredClone(this.document);
        attachMcpInstance(next, brainId).learnedSchemas[serverId] = hash;
        await this.publish(next);
      });
    } catch (error) {
      neuralError = `Connected, but durable neural schema learning is pending: ${error instanceof Error ? error.message : String(error)}`;
    }
    runtime.summary = { ...this.summary(runtime.config, runtime.initialized, neuralError),
      protocolVersion: runtime.protocolVersion, capabilities: runtime.capabilities };
  }

  hasTool(toolId: string): boolean {
    return [...this.runtimes.values()].some((runtime) => this.isEnabled(runtime) &&
      this.boundTools(runtime.config).some((tool) => tool.id === toolId)
    );
  }

  private taskSchemaSha(tool: McpToolSummary): string {
    return sha256(JSON.stringify({ id: tool.id, remoteName: tool.remoteName, inputSchema: tool.inputSchema,
      outputSchema: tool.outputSchema ?? null, metadata: tool.metadata ?? null, taskSupport: tool.taskSupport ?? "forbidden" }));
  }
  private async taskCurrent(lease: McpTaskLease, context: McpExecutionContext | undefined, signal: AbortSignal): Promise<void> {
    if (!context?.requireCurrentPermission) throw new Error("MCP task operations require a fresh current action/permission lease.");
    signal.throwIfAborted(); lease.controller.signal.throwIfAborted(); await context.requireCurrentPermission();
    signal.throwIfAborted(); lease.controller.signal.throwIfAborted(); this.requireHostConnection(lease.connection);
    const runtime = lease.runtime, current = this.boundTools(runtime.config).find(tool => tool.id === lease.tool.id);
    if (this.taskLeases.get(`${lease.record.source.brainId}:${lease.record.taskRef}`) !== lease || runtime.protocolVersion !== "2025-11-25" ||
      runtime.toolsDirty || !current || this.taskSchemaSha(current) !== lease.record.source.schemaSha256 || !negotiatedMcpCapabilities(runtime.capabilities).tasks?.requests?.tools?.call) throw new Error("The exact MCP task source/schema/session retired; retained history is dormant, not rebound.");
  }
  private trackTask<T>(brainId: string, signal: AbortSignal, operation: (signal: AbortSignal) => Promise<T>, taskRef?: string): Promise<T> {
    const controller = new AbortController(), promise = operation(AbortSignal.any([signal, controller.signal]));
    const ticket = { brainId, controller, promise, ...(taskRef ? { taskRef } : {}) }; this.taskOperations.add(ticket);
    return promise.finally(() => { this.taskOperations.delete(ticket); });
  }
  private taskReceipt(lease: McpTaskLease): McpTaskReceipt {
    return { format: "omni-mcp-task", version: 1, wireVersion: MCP_TASK_WIRE_VERSION, record: structuredClone(lease.record), operationComplete: false,
      management: lease.controller.signal.aborted ? "dormant" : "current-session", untrustedTaskData: true };
  }
  private async noTaskCredentialReflection(runtime: McpRuntime, value: unknown): Promise<void> {
    const encoded = JSON.stringify(value), secrets = await this.knownOperationalSecrets(this.interactionOrigin(runtime));
    if (secrets.some(secret => encoded.includes(secret) || encoded.includes(JSON.stringify(secret).slice(1, -1)))) throw new Error("The MCP task response reflected operational credentials; it was not retained or learned.");
  }
  private async startTask(runtime: McpRuntime, tool: McpToolSummary, args: Record<string, unknown>, signal: AbortSignal,
    callerBrainId: string | undefined, context?: McpExecutionContext): Promise<McpTaskReceipt> {
    if (!this.taskStore || callerBrainId !== runtime.config.brainId || !context?.requireCurrentPermission || runtime.protocolVersion !== "2025-11-25" ||
      !negotiatedMcpCapabilities(runtime.capabilities).tasks?.requests?.tools?.call || !["required", "optional"].includes(tool.taskSupport ?? "forbidden")) throw new Error("This MCP tool has no negotiated dated task consumer/current owner authority.");
    const store = this.taskStore, brainId = callerBrainId;
    return this.trackTask(brainId, signal, async ownedSignal => {
      await context.requireCurrentPermission!(); ownedSignal.throwIfAborted();
      const connection = this.hostConnection(brainId, runtime.config.id); this.requireHostConnection(connection);
      const argumentsValue = mcpTaskObject(args.arguments, "task tool arguments"), validate = compileToolInputValidator(tool.inputSchema);
      if (!validate(argumentsValue)) throw new Error("MCP task arguments failed the original remote tool input schema.");
      const reserved = await store.reserveCreation({ source: { brainId, serverId: runtime.config.id, toolId: tool.id, remoteName: tool.remoteName, endpoint: connection.endpoint,
        protocolVersion: "2025-11-25", configurationSha256: connection.configurationSha256, schemaSha256: this.taskSchemaSha(tool) },
        requestId: context.requestId ?? context.actionEventId ?? randomUUID(), ...(context.actionEventId ? { actionEventId: context.actionEventId } : {}),
        ...(context.chatTurnId ? { chatTurnId: context.chatTurnId } : {}), argumentsSha256: sha256(JSON.stringify(argumentsValue)) }); let row = reserved.record;
      const key = `${brainId}:${row.taskRef}`, existing = this.taskLeases.get(key);
      if (existing) { await this.taskCurrent(existing, context, ownedSignal);
        if (existing.record.creation === "accepted") return this.taskReceipt(existing);
        throw new Error("This exact task creation attempt is already in flight; the original tools/call was not resent."); }
      if (!reserved.created) return { format: "omni-mcp-task", version: 1, wireVersion: MCP_TASK_WIRE_VERSION, record: row,
        operationComplete: false, management: "dormant", reason: "The original task creation attempt is retained without a verified current lease; it was not resent or rebound.", untrustedTaskData: true };
      let acknowledged!: () => void, rejectAcknowledgement!: (error: unknown) => void;
      const acknowledgement = new Promise<void>((resolve, reject) => { acknowledged = resolve; rejectAcknowledgement = reject; }); void acknowledgement.catch(() => undefined);
      const lease: McpTaskLease = { runtime, connection, record: row, tool, arguments: structuredClone(argumentsValue), controller: new AbortController(), acknowledgement,
        acknowledged, rejectAcknowledgement, inputs: new Set() };
      this.taskLeases.set(key, lease);
      const operationId = randomUUID(); let dispatched = false, accepted = false;
      const currentSignal = AbortSignal.any([ownedSignal, lease.controller.signal]);
      const rpcContext: McpRpcContext = { ...context, task: { taskRef: row.taskRef }, beforeSend: async () => {
        await this.taskCurrent(lease, context, currentSignal);
        row = await store.update(brainId, row.taskRef, { creation: "dispatch-may-have-started", lastOperation: { id: operationId, method: "tools/call", state: "dispatch-may-have-started", at: new Date().toISOString() } }); lease.record = row;
        await this.taskCurrent(lease, context, currentSignal); dispatched = true;
      } };
      lease.scope = { id: `task-scope:${operationId}`, method: "tools/call", signal: currentSignal, context: rpcContext };
      try {
        const result = await this.rpc(runtime, "tools/call", { name: tool.remoteName, arguments: argumentsValue, task: args.ttl === undefined ? {} : { ttl: args.ttl } }, currentSignal, rpcContext);
        const task = checkedMcpCreateTaskResult(result); await this.noTaskCredentialReflection(runtime, result);
        if ([...this.taskLeases.values()].some(other => other !== lease && other.runtime === runtime && other.record.remote?.taskId === task.taskId)) throw new Error("The MCP server reused another owned task id; no task authority was reassigned.");
        row = await store.update(brainId, row.taskRef, { creation: "accepted", remote: task, createResult: mcpTaskObject(result, "task creation ACK"),
          lastOperation: { id: operationId, method: "tools/call", state: "acknowledged", at: new Date().toISOString() } }); lease.record = row; accepted = true;
        rpcContext.task!.remoteTaskId = task.taskId; acknowledged();
        while (lease.inputs.size) await Promise.allSettled([...lease.inputs]);
        await this.taskCurrent(lease, context, currentSignal); return this.taskReceipt(lease);
      } catch (error) {
        rejectAcknowledgement(error);
        let message = error instanceof Error ? error.message : String(error);
        if (error instanceof McpRpcFailure) try { await this.noTaskCredentialReflection(runtime, error.rpcError); } catch { message = "MCP task response carried private operational data and was withheld."; }
        const reason = `${message}${accepted ? " The actual accepted task is retained, but current management authority did not survive." : dispatched ? " The creation outcome may be unknown; the original tools/call was not resent." : " No task dispatch was confirmed."}`;
        row = await store.update(brainId, row.taskRef, { ...(!accepted ? { creation: error instanceof McpRpcFailure ? "rejected-before-task" as const : dispatched ? "creation-outcome-unknown" as const : "not-dispatched" as const } : {}), reason,
          lastOperation: { id: operationId, method: "tools/call", state: accepted || error instanceof McpRpcFailure ? "acknowledged" : dispatched ? "outcome-unknown" : "prepared", at: new Date().toISOString() } }); lease.record = row;
        lease.controller.abort(new Error(reason)); this.taskLeases.delete(key); throw new McpTaskCustodyError(reason, this.taskReceipt(lease));
      } finally { lease.scope = undefined; }
    });
  }
  private async executeTaskOperation(runtime: McpRuntime, tool: McpToolSummary, action: string, args: Record<string, unknown>, signal: AbortSignal,
    callerBrainId: string | undefined, context?: McpExecutionContext): Promise<McpTaskReceipt | McpTaskOutcome> {
    if (!this.taskStore || !context?.requireCurrentPermission || callerBrainId !== runtime.config.brainId || typeof args.taskRef !== "string" || !MCP_TASK_REF.test(args.taskRef) || Object.keys(args).some(key => key !== "taskRef")) throw new Error("MCP task management requires one exact local taskRef and its calling instance/current action authority.");
    const brainId = callerBrainId, key = `${brainId}:${args.taskRef}`, lease = this.taskLeases.get(key), store = this.taskStore;
    if (!lease || lease.runtime !== runtime || lease.tool.id !== tool.id || lease.record.source.brainId !== brainId || !lease.record.remote) throw new Error("This task has no verified current source/account lease. Retained/copied task history is dormant; no task was resent or silently rebound.");
    if (lease.scope || lease.busy) throw new Error("An operation for this exact task is already in flight; wait or Stop it rather than duplicating its request."); lease.busy = true;
    return this.trackTask(brainId, signal, async ownedSignal => {
      const currentSignal = AbortSignal.any([ownedSignal, lease.controller.signal]), method = action === "task-status" ? "tasks/get" : action === "task-result" ? "tasks/result" : "tasks/cancel";
      await this.taskCurrent(lease, context, currentSignal);
      if (method === "tasks/cancel" && (!negotiatedMcpCapabilities(runtime.capabilities).tasks?.cancel || isMcpTaskTerminal(lease.record.remote!.status))) throw new Error("This task is already terminal or the server did not negotiate task cancellation.");
      if (method === "tasks/result") { const retained = await store.retainedOutcome(brainId, lease.record.taskRef); if (retained) { await this.taskCurrent(lease, context, currentSignal); return retained; } }
      const id = randomUUID(); let dispatched = false;
      const rpcContext: McpRpcContext = { ...context, task: { taskRef: lease.record.taskRef, remoteTaskId: lease.record.remote!.taskId }, beforeSend: async () => {
        await this.taskCurrent(lease, context, currentSignal); lease.record = await store.update(brainId, lease.record.taskRef, { lastOperation: { id, method, state: "dispatch-may-have-started", at: new Date().toISOString() } });
        await this.taskCurrent(lease, context, currentSignal); dispatched = true;
      } };
      lease.scope = { id: `task-scope:${id}`, method, signal: currentSignal, context: rpcContext };
      try {
        if (method === "tasks/get" && lease.record.remote?.pollInterval) {
          const due = Date.parse(lease.record.updatedAt) + lease.record.remote.pollInterval;
          if (!Number.isSafeInteger(due)) throw new Error("MCP task polling deadline exceeds exact host timing.");
          while (due > Date.now()) { await delay(Math.min(due - Date.now(), 2147483647), undefined, { signal: currentSignal }); await this.taskCurrent(lease, context, currentSignal); }
        }
        await this.taskCurrent(lease, context, currentSignal);
        const result = await this.rpc(runtime, method, { taskId: lease.record.remote!.taskId }, currentSignal, rpcContext);
        await this.noTaskCredentialReflection(runtime, result);
        if (method === "tasks/result") {
          const raw = checkedMcpTaskToolResult(result, lease.record.remote!.taskId), outcome = await store.putOutcome(brainId, lease.record.taskRef, "result", raw);
          lease.record = await store.update(brainId, lease.record.taskRef, { lastOperation: { id, method, state: "acknowledged", at: new Date().toISOString() } });
          await this.taskCurrent(lease, context, currentSignal);
          const observedResult = await this.observeToolResult(runtime, lease.tool, lease.arguments, raw, currentSignal, context);
          await this.taskCurrent(lease, context, currentSignal); return { ...outcome, observedResult };
        }
        const task = checkedMcpRemoteTask(result);
        if (task.taskId !== lease.record.remote!.taskId || method === "tasks/cancel" && task.status !== "cancelled") throw new Error("The task response did not acknowledge the exact requested task/status.");
        lease.record = await store.update(brainId, lease.record.taskRef, { remote: task, lastOperation: { id, method, state: "acknowledged", at: new Date().toISOString() } });
        await this.taskCurrent(lease, context, currentSignal); return this.taskReceipt(lease);
      } catch (error) {
        if (method === "tasks/result" && error instanceof McpRpcFailure) {
          await this.noTaskCredentialReflection(runtime, error.rpcError);
          // Invalid/expired task ids are management failures, not evidence the
          // original tool failed. Confirm a real failed task before committing
          // the underlying operation's RPC error as its final result.
          let failed = lease.record.remote?.status === "failed";
          if (!failed) {
            try { await this.taskCurrent(lease, context, currentSignal);
              const state = checkedMcpRemoteTask(await this.rpc(runtime, "tasks/get", { taskId: lease.record.remote!.taskId }, currentSignal, { ...context, task: rpcContext.task, beforeSend: () => this.taskCurrent(lease, context, currentSignal) }));
              if (state.taskId !== lease.record.remote!.taskId) throw new Error("Task identity changed.");
              lease.record = await store.update(brainId, lease.record.taskRef, { remote: state }); failed = state.status === "failed";
            } catch { /* Unknown/expired management state does not invent a final outcome. */ }
          }
          if (failed) { const outcome = await store.putOutcome(brainId, lease.record.taskRef, "rpc-error", error.rpcError);
            lease.record = await store.update(brainId, lease.record.taskRef, { lastOperation: { id, method, state: "acknowledged", at: new Date().toISOString() } });
            await this.taskCurrent(lease, context, currentSignal); return outcome; }
        }
        let message = error instanceof Error ? error.message : String(error);
        if (error instanceof McpRpcFailure) try { await this.noTaskCredentialReflection(runtime, error.rpcError); } catch { message = "MCP task response carried private operational data and was withheld."; }
        const reason = `${message} The remote task's previous state is retained; no request was automatically resent.`;
        lease.record = await store.update(brainId, lease.record.taskRef, { reason, lastOperation: { id, method, state: error instanceof McpRpcFailure ? "acknowledged" : dispatched ? "outcome-unknown" : "prepared", at: new Date().toISOString() } }); throw new McpTaskCustodyError(reason, this.taskReceipt(lease));
      } finally { lease.scope = undefined; }
    }, lease.record.taskRef).finally(() => { lease.busy = false; lease.scope = undefined; });
  }
  private async observeToolResult(runtime: McpRuntime, tool: McpToolSummary, args: Record<string, unknown>, result: unknown, signal: AbortSignal,
    context?: McpExecutionContext): Promise<Record<string, unknown>> {
    const observed = { ...mcpTaskObject(result, "tool result") }, metadata = { ...record(observed._meta) };
    delete metadata["omni/mcpApp"];
    if (Object.keys(metadata).length) observed._meta = metadata; else delete observed._meta;
    try {
      const descriptor = await this.appHost?.capture({ origin: this.interactionOrigin(runtime, { id: "completed", method: "tools/call", signal, context }), tool, arguments: args, result: observed });
      if (descriptor) metadata["omni/mcpApp"] = descriptor;
    } catch (error) { metadata["omni/mcpAppUnavailable"] = (error instanceof Error ? error.message : String(error)).slice(0, 1000); }
    if (Object.keys(metadata).length) observed._meta = metadata; else delete observed._meta; return observed;
  }

  async execute(
    toolId: string,
    action: string,
    args: Record<string, unknown>,
    signal: AbortSignal,
    callerBrainId?: string,
    context?: McpExecutionContext
  ): Promise<unknown> {
    return this.executeCurrent(toolId, action, args, signal, callerBrainId, false, context);
  }

  private async executeOperation(runtime: McpRuntime, operation: McpClientOperation,
    args: Record<string, unknown>, signal: AbortSignal, context?: McpExecutionContext): Promise<McpClientOperationResult> {
    signal = AbortSignal.any([runtime.controller.signal, signal]);
    this.requireEnabled(runtime);
    const capabilities = negotiatedMcpCapabilities(runtime.capabilities);
    if (!mcpOperationSupported(capabilities, operation)) throw new Error(`MCP server did not negotiate ${operation}.`);
    if (operation === "tasks/list" && (!this.taskStore || runtime.protocolVersion !== "2025-11-25" || !context?.requireCurrentPermission)) throw new Error("MCP task listing requires its dated negotiated protocol, actual task consumer and current action authority.");
    if (operation.startsWith("resources/") && !operation.endsWith("/list")) mcpUri(args.uri);
    if (operation === "prompts/get" && (typeof args.name !== "string" || !args.name.length || /[\0\r\n]/.test(args.name))) throw new Error("MCP prompt name is invalid.");
    if (operation === "completion/complete") {
      const ref = record(args.ref);
      if (ref.type === "ref/resource") {
        if (!capabilities.resources) throw new Error("MCP resource completion requires negotiated resources.");
        mcpUri(ref.uri, "resource URI template");
      } else if (ref.type === "ref/prompt") {
        if (!capabilities.prompts) throw new Error("MCP prompt completion requires negotiated prompts.");
        if (typeof ref.name !== "string" || !ref.name.length || /[\0\r\n]/.test(ref.name)) throw new Error("MCP completion prompt name is invalid.");
      } else throw new Error("MCP completion reference type is unsupported.");
      // Backpressure concurrent interactive requests without changing suggestions
      // or retrying a failed request. UI clients can additionally debounce typing.
      const now = performance.now();
      const dispatchAt = Math.max(now, runtime.completionNextAt ?? now);
      runtime.completionNextAt = dispatchAt + 50;
      if (dispatchAt > now) await delay(dispatchAt - now, undefined, { signal });
    }
    const paginated = operation.endsWith("/list");
    let pages: Map<string, string> | undefined;
    const cursor = args.cursor as string | undefined;
    if (paginated) {
      runtime.pagination ??= new Map();
      pages = runtime.pagination.get(operation);
      if (!pages) { pages = new Map(); runtime.pagination.set(operation, pages); }
      // Cursors must originate from this operation in this connection. They are
      // opaque, not durable recovery tokens and not interchangeable between lists.
      if (cursor !== undefined && ![...pages.values()].includes(cursor)) throw new Error("MCP pagination cursor is not from this live listing. Restart without a cursor.");
    }
    signal.throwIfAborted(); this.requireEnabled(runtime);
    const requestEpoch = runtime.sessionEpoch ?? 0;
    const subscribing = operation === "resources/subscribe" ? { updates: 0, updatedAt: undefined as string | undefined } : undefined;
    if (subscribing) {
      runtime.subscribing ??= new Map();
      if (runtime.subscribing.has(args.uri as string)) throw new Error("MCP subscription to this URI is already being requested.");
      runtime.subscribing.set(args.uri as string, subscribing);
    }
    let result: McpClientOperationResult["result"];
    try {
      if (operation === "tasks/list") await context!.requireCurrentPermission!();
      result = validateMcpOperationResult(operation, await this.rpc(runtime, operation, args, signal,
        operation === "tasks/list" ? { ...context!, beforeSend: context!.requireCurrentPermission } : context));
      if (operation === "tasks/list") { await context!.requireCurrentPermission!(); await this.noTaskCredentialReflection(runtime, result); }
    }
    finally { if (subscribing && runtime.subscribing?.get(args.uri as string) === subscribing) runtime.subscribing.delete(args.uri as string); }
    signal.throwIfAborted(); this.requireEnabled(runtime);
    if (["resources/subscribe", "resources/unsubscribe"].includes(operation) && (runtime.sessionEpoch ?? 0) !== requestEpoch) {
      throw new Error("The MCP connection changed before subscription acknowledgement. Its external state is uncertain; request it again explicitly if needed.");
    }
    if (paginated) {
      const next = record(result).nextCursor as string | undefined;
      const key = cursor ?? "";
      // A concurrent list-change notification invalidates the old cursor chain.
      if (runtime.pagination?.get(operation) !== pages) throw new Error("The MCP listing changed while this page was read. Restart without a cursor.");
      if (next) {
        const visited = new Set<string>([key]);
        let candidate: string | undefined = next;
        while (candidate !== undefined) {
          if (visited.has(candidate)) throw new Error(`MCP ${operation} repeated a pagination cursor.`);
          visited.add(candidate); candidate = pages!.get(candidate);
        }
        pages!.set(key, next);
      } else pages!.delete(key);
    }
    if (operation === "resources/subscribe") {
      runtime.subscriptions ??= new Map();
      const previous = runtime.subscriptions.get(args.uri as string);
      runtime.subscriptions.set(args.uri as string, { uri: args.uri as string, state: "active",
        updates: (previous?.state === "active" ? previous.updates : 0) + (subscribing?.updates ?? 0),
        ...(subscribing?.updatedAt ? { updatedAt: subscribing.updatedAt } : previous?.updatedAt ? { updatedAt: previous.updatedAt } : {}) });
      this.capabilityEvent(runtime, "resource-subscribed", args.uri as string);
      if (subscribing?.updates) this.capabilityEvent(runtime, "resource-updated", args.uri as string);
    } else if (operation === "resources/unsubscribe") {
      runtime.subscriptions?.delete(args.uri as string);
      this.capabilityEvent(runtime, "resource-unsubscribed", args.uri as string);
    }
    return { format: "omni-mcp-client-result", version: 1, operation, result,
      provenance: { brainId: runtime.config.brainId, serverId: runtime.config.id,
        transport: runtime.config.transport, endpoint: runtime.config.transport === "http" ? runtime.config.url! : runtime.config.command!,
        protocolVersion: runtime.protocolVersion!, observedAt: new Date().toISOString(),
        ...(typeof args.uri === "string" ? { requestedUri: args.uri } : {}),
        ...(operation === "prompts/get" ? { requestedPrompt: args.name as string } : {}) },
      untrustedTaskData: true, hiddenPromptUsed: false };
  }

  private async executeCurrent(
    toolId: string, action: string, args: Record<string, unknown>, signal: AbortSignal,
    callerBrainId: string | undefined, sessionRetried: boolean,
    context?: McpExecutionContext
  ): Promise<unknown> {
    signal.throwIfAborted();
    if (!["call", "call-task", "task-status", "task-result", "task-cancel"].includes(action)) throw new Error("Unknown current MCP tool action.");
    const runtime = [...this.runtimes.values()].find((candidate) =>
      this.boundTools(candidate.config).some((tool) => tool.id === toolId)
    );
    const tool = runtime && this.boundTools(runtime.config).find((candidate) => candidate.id === toolId);
    if (!runtime || !tool) throw new Error("MCP tool is unavailable.");
    if (!tool.actions.includes(action)) throw new Error(tool.taskSupport === "required" ? "This MCP tool requires task-augmented execution; use its current call-task action." : "This MCP tool did not negotiate that task action.");
    if (tool.clientOperation && callerBrainId === undefined) throw new Error("MCP client operations require the calling instance identity.");
    if (callerBrainId !== undefined && callerBrainId !== runtime.config.brainId) throw new Error("MCP tool belongs to another instance.");
    this.requireEnabled(runtime);
    if (!tool.clientOperation) {
      if (context?.mcpAppViewId) {
        if (!this.appHost || callerBrainId === undefined) throw new Error("App calls require their trusted host and instance identity.");
        this.appHost.requireCall(context.mcpAppViewId, callerBrainId, runtime.config.id, tool.remoteName);
      } else if (!mcpToolUi(tool).visibility.includes("model")) throw new Error("This tool is App-only and is not exposed to the brain.");
    }
    if (action.startsWith("task-")) return this.executeTaskOperation(runtime, tool, action, args, signal, callerBrainId, context);
    if (runtime.toolsDirty) {
      await this.refreshRuntime(runtime, false);
      if (runtime.toolsDirty) throw new Error("MCP schemas changed again during discovery. Retry once the server's catalog stabilizes.");
      // Schema changes require a fresh validation pass on the current definition.
      return this.executeCurrent(toolId, action, args, signal, callerBrainId, sessionRetried, context);
    }
    const validationKey = action === "call" ? toolId : `${toolId}\0${action}`, inputSchema = tool.actionInputSchemas?.[action] ?? tool.inputSchema;
    let cached = this.inputValidators.get(validationKey);
    const schemaHash = sha256(JSON.stringify(inputSchema));
    if (!cached || cached.schemaHash !== schemaHash) {
      cached = { schemaHash, validate: compileToolInputValidator(inputSchema) };
      this.inputValidators.set(validationKey, cached);
    }
    if (!cached.validate(args)) {
      throw new Error(`MCP tool arguments failed input schema validation (${cached.validate.errors?.[0]?.keyword ?? "invalid"}).`);
    }
    await this.initializeRuntime(runtime, signal);
    this.requireEnabled(runtime);
    if (runtime.toolsDirty || JSON.stringify(negotiatedMcpCapabilities(runtime.capabilities)) !== JSON.stringify(runtime.config.capabilities ?? {})) {
      await this.refreshRuntime(runtime, false);
      if (runtime.toolsDirty) throw new Error("MCP schemas changed again during discovery. Retry once the server's catalog stabilizes.");
      return this.executeCurrent(toolId, action, args, signal, callerBrainId, sessionRetried, context);
    }
    try {
      // The descriptor was regenerated locally from a reserved hash identity;
      // a remote tool's clientOperation property is never consulted.
      if (tool.clientOperation) return await this.executeOperation(runtime, tool.clientOperation, args, signal, context);
      if (action === "call-task") return await this.startTask(runtime, tool, args, signal, callerBrainId, context);
      const result = await this.rpc(runtime, "tools/call", { name: tool.remoteName, arguments: args }, signal, context);
      if (!result || typeof result !== "object" || Array.isArray(result)) return result;
      const returned = result as Record<string, unknown>;
      if (returned.resultType === "task" || Object.hasOwn(returned, "task") && !Array.isArray(returned.content)) {
        throw new Error("The server returned an unsolicited task instead of this ordinary tool call's completed result. No task was adopted, rebound, cancelled or automatically resent; external effects may still be running.");
      }
      return this.observeToolResult(runtime, tool, args, result, signal, context);
    }
    catch (error) {
      // HTTP 404 for the established MCP session means the server rejected the
      // request before handling it. Reconnect once and revalidate fresh schemas.
      // Network/timeout/other failures are never retried as unknown effects.
      if (error instanceof McpSessionExpiredError && !sessionRetried && action === "call" && tool.clientOperation !== "tasks/list") {
        await this.refreshRuntime(runtime, false);
        if (tool.clientOperation && args.cursor !== undefined) throw new Error("The MCP session changed. Restart listing without its old pagination cursor.");
        return this.executeCurrent(toolId, action, args, signal, callerBrainId, true, context);
      }
      throw error;
    }
  }

  dispose(): void {
    for (const operation of this.taskOperations) operation.controller.abort(new Error("MCP host shut down; remote tasks may still be running."));
    for (const lease of this.taskLeases.values()) lease.controller.abort(new Error("MCP host shut down; task grants are not persistent.")); this.taskLeases.clear();
    this.oauth.dispose();
    for (const runtime of this.runtimes.values()) {
      this.endHttpSession(runtime);
      runtime.controller.abort(); runtime.stdio?.stop();
      runtime.legacySse?.stop();
      if (runtime.schemasRegistered) this.brain.unregisterExternalToolSchemas(this.boundTools(runtime.config).map((tool) => tool.id));
    }
    this.runtimes.clear();
    this.inputValidators.clear();
    this.capabilityListeners.clear();
  }
}
