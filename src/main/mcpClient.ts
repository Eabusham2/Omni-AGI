import { createHash, randomUUID } from "node:crypto";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { chmod, mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, isAbsolute } from "node:path";
import Ajv from "ajv";
import Ajv2019 from "ajv/dist/2019.js";
import Ajv2020 from "ajv/dist/2020.js";
import type { ValidateFunction } from "ajv";
import type {
  McpServerRegistrationRequest,
  McpServerSummary,
  McpToolSummary
} from "../shared/types";
import type { BrainService, ExternalToolSchema } from "./brainService";
import type { SecureSecretStore } from "./secureSecretStore";

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
}

interface McpDocument {
  format: "omni-mcp-registry";
  version: 1;
  servers: StoredMcpServer[];
}

interface JsonRpcResponse {
  jsonrpc?: string;
  id?: string | number | null;
  result?: unknown;
  error?: { code?: number; message?: string; data?: unknown };
}

interface McpRuntime {
  config: StoredMcpServer;
  summary: McpServerSummary;
  sessionId?: string;
  initialized: boolean;
  stdio?: StdioRpcSession;
}

const MCP_PROTOCOL_VERSION = "2025-06-18";
const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/;
const INTERNAL_TOOL_ID = /^[a-z][a-z0-9.-]{1,79}$/;
const MAX_INTERNAL_TOOL_ID_LENGTH = 80;
const MAX_MCP_MESSAGE_BYTES = 16 * 1024 * 1024;
const MAX_MCP_REGISTRY_BYTES = 64 * 1024 * 1024;

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function runtimeKey(brainId: string, serverId: string): string {
  return `${brainId}:${serverId}`;
}

function secretId(brainId: string, serverId: string): string {
  return `mcp:${brainId}:${serverId}`;
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
  const loopback = ["localhost", "127.0.0.1", "::1"].includes(url.hostname);
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

function parseJsonRpcPayload(text: string, expectedId?: string): JsonRpcResponse | undefined {
  const candidates: unknown[] = [];
  const trimmed = text.trim();
  if (!trimmed) return undefined;
  if (trimmed.startsWith("data:") || trimmed.includes("\n\ndata:")) {
    for (const event of trimmed.split(/\r?\n\r?\n/)) {
      const data = event
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!data) continue;
      try { candidates.push(JSON.parse(data)); } catch { /* Ignore non-JSON SSE events. */ }
    }
  } else {
    try { candidates.push(JSON.parse(trimmed)); } catch {
      throw new Error("MCP server returned invalid JSON-RPC.");
    }
  }
  for (const candidate of candidates) {
    const values = Array.isArray(candidate) ? candidate : [candidate];
    for (const value of values) {
      if (typeof value !== "object" || value === null || Array.isArray(value)) continue;
      const response = value as JsonRpcResponse;
      if (expectedId === undefined || String(response.id) === expectedId) return response;
    }
  }
  return undefined;
}

function encodeRpcMessage(value: Record<string, unknown>): string {
  const encoded = JSON.stringify(value);
  if (Buffer.byteLength(encoded, "utf8") > MAX_MCP_MESSAGE_BYTES) {
    throw new Error("MCP request exceeded 16 MiB.");
  }
  return encoded;
}

async function readBoundedResponse(response: Response): Promise<Uint8Array> {
  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > MAX_MCP_MESSAGE_BYTES) {
    throw new Error("MCP response exceeded 16 MiB.");
  }
  if (!response.body) return new Uint8Array();
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    if (!value) continue;
    total += value.byteLength;
    if (total > MAX_MCP_MESSAGE_BYTES) {
      await reader.cancel().catch(() => undefined);
      throw new Error("MCP response exceeded 16 MiB.");
    }
    chunks.push(value);
  }
  const output = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

class StdioRpcSession {
  private child?: ChildProcessWithoutNullStreams;
  private buffer = "";
  private stderr = "";
  private readonly pending = new Map<string, {
    resolve: (value: JsonRpcResponse) => void;
    reject: (error: Error) => void;
    timer: NodeJS.Timeout;
  }>();

  constructor(private readonly command: string, private readonly args: readonly string[]) {}

  async start(): Promise<void> {
    if (this.child && !this.child.killed) return;
    const info = await stat(this.command);
    if (!info.isFile()) throw new Error("MCP stdio command is not an executable file.");
    const minimalEnvironment: NodeJS.ProcessEnv = {
      PATH: process.env.PATH,
      SystemRoot: process.env.SystemRoot,
      TEMP: process.env.TEMP,
      TMP: process.env.TMP,
      TMPDIR: process.env.TMPDIR,
      LANG: process.env.LANG
    };
    const child = spawn(this.command, [...this.args], {
      shell: false,
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
      env: minimalEnvironment
    });
    this.child = child;
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => this.consume(chunk));
    child.stderr.on("data", (chunk: string) => {
      this.stderr = `${this.stderr}${chunk}`.slice(-8_000);
    });
    child.once("exit", () => this.failAll(new Error("MCP stdio server exited.")));
    await new Promise<void>((resolveStart, rejectStart) => {
      child.once("spawn", resolveStart);
      child.once("error", rejectStart);
    });
  }

  private consume(chunk: string): void {
    this.buffer += chunk;
    if (Buffer.byteLength(this.buffer, "utf8") > MAX_MCP_MESSAGE_BYTES) {
      this.buffer = "";
      this.failAll(new Error("MCP stdio response exceeded 16 MiB."));
      this.child?.kill();
      this.child = undefined;
      return;
    }
    for (;;) {
      const end = this.buffer.indexOf("\n");
      if (end < 0) break;
      const line = this.buffer.slice(0, end).trim();
      this.buffer = this.buffer.slice(end + 1);
      if (!line) continue;
      let response: JsonRpcResponse;
      try { response = JSON.parse(line) as JsonRpcResponse; } catch { continue; }
      if (response.id === undefined || response.id === null) continue;
      const id = String(response.id);
      const pending = this.pending.get(id);
      if (!pending) continue;
      this.pending.delete(id);
      clearTimeout(pending.timer);
      pending.resolve(response);
    }
  }

  private failAll(error: Error): void {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }

  async request(method: string, params: Record<string, unknown>, signal?: AbortSignal): Promise<JsonRpcResponse> {
    await this.start();
    signal?.throwIfAborted();
    const id = randomUUID();
    const child = this.child;
    if (!child || child.killed) throw new Error("MCP stdio server is unavailable.");
    const encoded = encodeRpcMessage({ jsonrpc: "2.0", id, method, params });
    return new Promise<JsonRpcResponse>((resolveRequest, rejectRequest) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        rejectRequest(new Error(`MCP stdio request timed out.${this.stderr ? ` ${this.stderr.slice(-500)}` : ""}`));
      }, 120_000);
      const abort = (): void => {
        const pending = this.pending.get(id);
        if (!pending) return;
        this.pending.delete(id);
        clearTimeout(timer);
        rejectRequest(new Error("MCP request was cancelled."));
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
        timer
      });
      child.stdin.write(`${encoded}\n`, (error) => {
        if (!error) return;
        const pending = this.pending.get(id);
        if (!pending) return;
        this.pending.delete(id);
        clearTimeout(timer);
        pending.reject(error);
      });
    });
  }

  notify(method: string, params: Record<string, unknown> = {}): void {
    if (!this.child || this.child.killed) throw new Error("MCP stdio server is unavailable.");
    this.child.stdin.write(`${encodeRpcMessage({ jsonrpc: "2.0", method, params })}\n`);
  }

  stop(): void {
    this.failAll(new Error("MCP stdio server stopped."));
    this.child?.kill();
    this.child = undefined;
  }
}

export class McpClientService {
  private readonly runtimes = new Map<string, McpRuntime>();
  private readonly inputValidators = new Map<string, {
    schema: Record<string, unknown>;
    validate: ValidateFunction<unknown>;
  }>();

  constructor(
    private readonly registryPath: string,
    private readonly secrets: SecureSecretStore,
    private readonly brain: Pick<BrainService,
      "registerExternalToolSchemas" |
      "unregisterExternalToolSchemas" |
      "learnStructuredExperience" |
      "listToolPermissions" |
      "setToolPermission"
    >,
    private readonly fetcher: FetchLike = fetch
  ) {}

  private async readDocument(): Promise<McpDocument> {
    try {
      const info = await stat(this.registryPath);
      if (!info.isFile() || info.size > MAX_MCP_REGISTRY_BYTES) {
        throw new Error("The MCP registry is invalid or exceeds 64 MiB.");
      }
      const parsed = JSON.parse(await readFile(this.registryPath, "utf8")) as McpDocument;
      if (parsed.format !== "omni-mcp-registry" || parsed.version !== 1 || !Array.isArray(parsed.servers)) {
        throw new Error("The MCP registry is invalid.");
      }
      return parsed;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return { format: "omni-mcp-registry", version: 1, servers: [] };
      }
      throw error;
    }
  }

  private async writeDocument(): Promise<void> {
    const document: McpDocument = {
      format: "omni-mcp-registry",
      version: 1,
      servers: [...this.runtimes.values()].map((runtime) => runtime.config)
    };
    await mkdir(dirname(this.registryPath), { recursive: true });
    const temporary = `${this.registryPath}.${randomUUID()}.tmp`;
    try {
      await writeFile(temporary, JSON.stringify(document, null, 2), { encoding: "utf8", mode: 0o600 });
      await chmod(temporary, 0o600).catch(() => undefined);
      await rename(temporary, this.registryPath);
    } finally {
      await rm(temporary, { force: true }).catch(() => undefined);
    }
  }

  async initialize(): Promise<void> {
    const document = await this.readDocument();
    const seen = new Set<string>();
    for (const stored of document.servers) {
      const config = this.validateStoredConfig(stored);
      const key = runtimeKey(config.brainId, config.id);
      if (seen.has(key)) throw new Error("The MCP registry contains a duplicate server id.");
      seen.add(key);
      const summary = this.summary(config, false);
      const runtime: McpRuntime = { config, summary, initialized: false };
      this.runtimes.set(key, runtime);
      this.registerSchemas(config.tools);
    }
  }

  private registerSchemas(tools: readonly McpToolSummary[]): void {
    this.brain.registerExternalToolSchemas(
      tools.map((tool): ExternalToolSchema => ({
        id: tool.id,
        actions: tool.actions,
        inputSchema: tool.inputSchema
      }))
    );
  }

  private summary(config: StoredMcpServer, connected: boolean, error?: string): McpServerSummary {
    return {
      brainId: config.brainId,
      id: config.id,
      label: config.label,
      transport: config.transport,
      endpoint: config.transport === "http" ? config.url ?? "" : config.command ?? "",
      connected,
      tools: config.tools.map((tool) => ({ ...tool, inputSchema: { ...tool.inputSchema } })),
      credentialConfigured: false,
      ...(error ? { error: error.replace(/[\r\n\0]+/g, " ").slice(0, 1_000) } : {})
    };
  }

  async list(brainId: string): Promise<McpServerSummary[]> {
    const values = [...this.runtimes.values()].filter((runtime) => runtime.config.brainId === brainId);
    return Promise.all(values.map(async (runtime) => ({
      ...runtime.summary,
      credentialConfigured: await this.secrets.has(secretId(brainId, runtime.config.id))
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

  private validateStoredConfig(value: unknown): StoredMcpServer {
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
      if (tool.id !== expectedId) throw new Error("The MCP registry contains a tampered tool id.");
      if (!Array.isArray(tool.actions) || tool.actions.length !== 1 || tool.actions[0] !== "call") {
        throw new Error("The MCP registry contains an invalid tool action.");
      }
      const inputSchema = record(tool.inputSchema);
      if (Buffer.byteLength(JSON.stringify(inputSchema), "utf8") > 4 * 1024 * 1024) {
        throw new Error("The MCP registry contains an oversized tool schema.");
      }
      return { id: expectedId, remoteName, actions: ["call"], inputSchema };
    });
    return { ...base, tools };
  }

  async add(request: McpServerRegistrationRequest): Promise<McpServerSummary> {
    const config = this.validateRequest(request);
    const key = runtimeKey(config.brainId, config.id);
    if (this.runtimes.has(key)) throw new Error("An MCP server with this id already exists for the brain.");
    if (request.bearerToken?.trim()) {
      await this.secrets.set(secretId(config.brainId, config.id), request.bearerToken.trim());
    }
    const runtime: McpRuntime = { config, summary: this.summary(config, false), initialized: false };
    this.runtimes.set(key, runtime);
    try {
      const summary = await this.refresh(config.brainId, config.id);
      return summary;
    } catch (error) {
      runtime.stdio?.stop();
      this.runtimes.delete(key);
      await this.secrets.remove(secretId(config.brainId, config.id));
      throw error;
    }
  }

  async remove(brainId: string, serverId: string): Promise<boolean> {
    const key = runtimeKey(safeId(brainId, "brain id"), safeId(serverId, "server id"));
    const runtime = this.runtimes.get(key);
    if (!runtime) return false;
    runtime.stdio?.stop();
    for (const tool of runtime.config.tools) this.inputValidators.delete(tool.id);
    this.brain.unregisterExternalToolSchemas(runtime.config.tools.map((tool) => tool.id));
    for (const tool of runtime.config.tools) {
      await this.brain.setToolPermission(brainId, tool.id, "off");
    }
    this.runtimes.delete(key);
    await this.secrets.remove(secretId(brainId, serverId));
    await this.writeDocument();
    return true;
  }

  private async httpRpc(
    runtime: McpRuntime,
    method: string,
    params: Record<string, unknown>,
    notification = false,
    signal?: AbortSignal
  ): Promise<JsonRpcResponse | undefined> {
    const id = notification ? undefined : randomUUID();
    const token = await this.secrets.get(secretId(runtime.config.brainId, runtime.config.id));
    const body = encodeRpcMessage({ jsonrpc: "2.0", ...(id ? { id } : {}), method, params });
    const response = await this.fetcher(runtime.config.url!, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
        "mcp-protocol-version": MCP_PROTOCOL_VERSION,
        ...(runtime.sessionId ? { "mcp-session-id": runtime.sessionId } : {}),
        ...(token ? { authorization: `Bearer ${token}` } : {})
      },
      body,
      signal: AbortSignal.any([signal ?? new AbortController().signal, AbortSignal.timeout(120_000)])
    });
    if (!response.ok) throw new Error(`MCP server returned HTTP ${response.status}.`);
    runtime.sessionId = response.headers.get("mcp-session-id") ?? runtime.sessionId;
    if (notification || response.status === 202 || response.status === 204) return undefined;
    const bytes = await readBoundedResponse(response);
    const parsed = parseJsonRpcPayload(new TextDecoder().decode(bytes), id);
    if (!parsed) throw new Error("MCP server returned no matching JSON-RPC response.");
    return parsed;
  }

  private async rpc(
    runtime: McpRuntime,
    method: string,
    params: Record<string, unknown>,
    signal?: AbortSignal
  ): Promise<unknown> {
    let response: JsonRpcResponse | undefined;
    if (runtime.config.transport === "http") {
      response = await this.httpRpc(runtime, method, params, false, signal);
    } else {
      runtime.stdio ??= new StdioRpcSession(runtime.config.command!, runtime.config.args ?? []);
      response = await runtime.stdio.request(method, params, signal);
    }
    if (response?.error) {
      throw new Error(`MCP ${method} failed: ${String(response.error.message ?? response.error.code ?? "unknown error").slice(0, 1_000)}`);
    }
    return response?.result;
  }

  private async initializeRuntime(runtime: McpRuntime, signal?: AbortSignal): Promise<void> {
    if (runtime.initialized) return;
    await this.rpc(runtime, "initialize", {
      protocolVersion: MCP_PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: "Omni AGI Studio", version: "1.1.1" }
    }, signal);
    if (runtime.config.transport === "http") {
      await this.httpRpc(runtime, "notifications/initialized", {}, true, signal);
    } else {
      runtime.stdio!.notify("notifications/initialized");
    }
    runtime.initialized = true;
  }

  private async listRemoteTools(runtime: McpRuntime, signal?: AbortSignal): Promise<McpToolSummary[]> {
    await this.initializeRuntime(runtime, signal);
    const found: McpToolSummary[] = [];
    const remoteNames = new Set<string>();
    const cursors = new Set<string>();
    let cursor: string | undefined;
    do {
      const result = record(await this.rpc(runtime, "tools/list", cursor ? { cursor } : {}, signal));
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
          id: namespaceTool(runtime.config.brainId, runtime.config.id, remoteName),
          remoteName,
          actions: ["call"],
          inputSchema
        });
      }
      cursor = typeof result.nextCursor === "string" && result.nextCursor ? result.nextCursor : undefined;
      if (cursor && cursors.has(cursor)) throw new Error("MCP tools/list repeated a pagination cursor.");
      if (cursor) cursors.add(cursor);
    } while (cursor);
    return found;
  }

  async refresh(brainId: string, serverId: string): Promise<McpServerSummary> {
    const key = runtimeKey(safeId(brainId, "brain id"), safeId(serverId, "server id"));
    const runtime = this.runtimes.get(key);
    if (!runtime) throw new Error("MCP server was not found.");
    runtime.initialized = false;
    runtime.sessionId = undefined;
    const oldIds = runtime.config.tools.map((tool) => tool.id);
    const tools = await this.listRemoteTools(runtime);
    for (const id of oldIds) this.inputValidators.delete(id);
    runtime.config.tools = tools;
    this.brain.unregisterExternalToolSchemas(oldIds.filter((id) => !tools.some((tool) => tool.id === id)));
    this.registerSchemas(tools);
    const permissions = await this.brain.listToolPermissions(brainId);
    for (const tool of tools) {
      if (!permissions.some((permission) => permission.toolId === tool.id)) {
        await this.brain.setToolPermission(brainId, tool.id, "ask");
      }
    }
    await this.writeDocument();
    let neuralError: string | undefined;
    try {
      const content = JSON.stringify({
        format: "omni-mcp-capability-schema",
        version: 1,
        server: { id: runtime.config.id, transport: runtime.config.transport },
        tools: tools.map((tool) => ({
          id: tool.id,
          remoteName: tool.remoteName,
          action: "call",
          inputSchema: tool.inputSchema
        })),
        behavioralInstructions: false
      });
      await this.brain.learnStructuredExperience(brainId, {
        content,
        name: `MCP capability schemas · ${runtime.config.label}`,
        sourceLabel: `MCP ${runtime.config.id} typed capability schemas`,
        provenanceUrl: runtime.config.transport === "http" ? runtime.config.url : undefined,
        license: "User-installed MCP capability schema"
      });
    } catch (error) {
      neuralError = `Connected, but durable neural schema learning is pending: ${error instanceof Error ? error.message : String(error)}`;
    }
    runtime.summary = this.summary(runtime.config, true, neuralError);
    runtime.summary.credentialConfigured = await this.secrets.has(secretId(brainId, serverId));
    return { ...runtime.summary, tools: runtime.summary.tools.map((tool) => ({ ...tool })) };
  }

  hasTool(toolId: string): boolean {
    return [...this.runtimes.values()].some((runtime) =>
      runtime.config.tools.some((tool) => tool.id === toolId)
    );
  }

  async execute(
    toolId: string,
    action: string,
    args: Record<string, unknown>,
    signal: AbortSignal
  ): Promise<unknown> {
    if (action !== "call") throw new Error("MCP tools use the call action.");
    const runtime = [...this.runtimes.values()].find((candidate) =>
      candidate.config.tools.some((tool) => tool.id === toolId)
    );
    const tool = runtime?.config.tools.find((candidate) => candidate.id === toolId);
    if (!runtime || !tool) throw new Error("MCP tool is unavailable.");
    let cached = this.inputValidators.get(toolId);
    if (!cached || cached.schema !== tool.inputSchema) {
      cached = { schema: tool.inputSchema, validate: compileToolInputValidator(tool.inputSchema) };
      this.inputValidators.set(toolId, cached);
    }
    if (!cached.validate(args)) {
      throw new Error(`MCP tool arguments failed input schema validation (${cached.validate.errors?.[0]?.keyword ?? "invalid"}).`);
    }
    await this.initializeRuntime(runtime, signal);
    return this.rpc(runtime, "tools/call", { name: tool.remoteName, arguments: args }, signal);
  }

  dispose(): void {
    for (const runtime of this.runtimes.values()) runtime.stdio?.stop();
    this.runtimes.clear();
    this.inputValidators.clear();
  }
}
