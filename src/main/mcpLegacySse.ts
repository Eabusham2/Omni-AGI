import { randomUUID } from "node:crypto";
import { readMcpEvents } from "./mcpEventStream";
import { parseMcpRpcMessages, type McpRpcMessage } from "./mcpStream";

interface LegacySseOptions {
  endpoint: string;
  fetcher: typeof fetch;
  headers(): Promise<Record<string, string>>;
  onMessage(message: McpRpcMessage): void | Promise<void>;
  onClosed(): void;
  signal: AbortSignal;
  maximumBytes: number;
}

/** Deprecated 2024 HTTP+SSE: responses arrive on one GET, not the POST body. */
export class McpLegacySseSession {
  private readonly controller = new AbortController();
  private connection?: Promise<void>;
  private messageEndpoint?: string;
  private stopped = false;
  private readonly pending = new Map<string, { resolve(value: McpRpcMessage): void; reject(error: Error): void }>();
  constructor(private readonly options: LegacySseOptions) {}

  private connect(): Promise<void> {
    if (this.stopped) return Promise.reject(new Error("MCP legacy SSE connection closed."));
    if (this.connection) return this.connection;
    let ready!: () => void;
    let failed!: (error: unknown) => void;
    const connection = new Promise<void>((resolve, reject) => { ready = resolve; failed = reject; });
    this.connection = connection;
    const signal = AbortSignal.any([this.controller.signal, this.options.signal]);
    void (async () => {
      try {
        const response = await this.options.fetcher(this.options.endpoint, { method: "GET", redirect: "error", signal,
          headers: { ...await this.options.headers(), accept: "text/event-stream" } });
        if (!response.ok) { await response.body?.cancel().catch(() => undefined); throw new Error(`MCP legacy SSE returned HTTP ${response.status}.`); }
        for await (const event of readMcpEvents(response, this.options.maximumBytes, signal)) {
          if (event.event === "endpoint") {
            if (this.messageEndpoint) throw new Error("MCP legacy SSE changed its message endpoint within a session.");
            if (/[\0\r\n]/.test(event.data)) throw new Error("MCP legacy SSE returned an invalid message endpoint.");
            const endpoint = new URL(event.data, this.options.endpoint);
            const original = new URL(this.options.endpoint);
            if (endpoint.origin !== original.origin || endpoint.username || endpoint.password || endpoint.hash) {
              throw new Error("The legacy MCP server requested a different-origin message endpoint. Configure that endpoint explicitly; credentials were not forwarded.");
            }
            this.messageEndpoint = endpoint.toString(); ready(); continue;
          }
          if (!event.data.trim()) continue;
          if (!this.messageEndpoint) throw new Error("MCP legacy SSE sent messages before its endpoint event.");
          for (const message of parseMcpRpcMessages(event.data)) {
            if (message.method) { await this.options.onMessage(message); continue; }
            if (message.id === undefined || message.id === null) continue;
            const pending = this.pending.get(String(message.id));
            if (!pending) continue;
            if (Object.hasOwn(message, "result") === Object.hasOwn(message, "error")) throw new Error("MCP legacy SSE returned an invalid RPC result.");
            this.pending.delete(String(message.id)); pending.resolve(message);
          }
        }
        throw new Error("MCP legacy SSE closed. Pending outcomes are unknown and were not repeated.");
      } catch (error) {
        const problem = error instanceof Error ? error : new Error(String(error));
        failed(problem); this.fail(problem); this.stopped = true; this.options.onClosed();
      }
    })();
    return connection;
  }

  private fail(error: Error): void { for (const pending of this.pending.values()) pending.reject(error); this.pending.clear(); }

  private async send(message: Record<string, unknown>, signal: AbortSignal): Promise<void> {
    await this.connect(); signal.throwIfAborted();
    const body = JSON.stringify(message);
    if (Buffer.byteLength(body) > this.options.maximumBytes) throw new Error("MCP request exceeded 16 MiB.");
    const response = await this.options.fetcher(this.messageEndpoint!, { method: "POST", redirect: "error",
      headers: { ...await this.options.headers(), "content-type": "application/json", accept: "application/json, text/event-stream" },
      body, signal });
    await response.body?.cancel().catch(() => undefined);
    if (!response.ok) throw new Error(`MCP legacy message delivery returned HTTP ${response.status}; its request was not repeated.`);
  }

  async request(method: string, params: Record<string, unknown>, signal: AbortSignal, requestId = randomUUID(), cancelNotification = true): Promise<McpRpcMessage> {
    await this.connectWithSignal(signal); signal.throwIfAborted();
    let resolve!: (value: McpRpcMessage) => void;
    let reject!: (error: Error) => void;
    const response = new Promise<McpRpcMessage>((yes, no) => { resolve = yes; reject = no; });
    // Attach the rejection handler before POST can fail, avoiding unhandled races.
    void response.catch(() => undefined);
    const abort = (): void => {
      const pending = this.pending.get(requestId); if (!pending) return;
      this.pending.delete(requestId); pending.reject(new Error("MCP request was cancelled."));
      if (cancelNotification && method !== "initialize" && !this.stopped) void this.notify("notifications/cancelled", { requestId,
        reason: "The caller stopped this request." }, AbortSignal.timeout(5_000)).catch(() => undefined);
    };
    signal.addEventListener("abort", abort, { once: true });
    this.pending.set(requestId, { resolve, reject });
    const posting = new AbortController();
    try {
      const delivery = this.send({ jsonrpc: "2.0", id: requestId, method, params }, AbortSignal.any([signal, posting.signal])).then(() => response);
      // A matching authenticated stream result is authoritative even if the
      // POST acknowledgement is delayed or its socket closes afterward.
      return await Promise.race([response, delivery]);
    } catch (error) {
      this.pending.delete(requestId); throw error;
    } finally { signal.removeEventListener("abort", abort); posting.abort(); }
  }

  private async connectWithSignal(signal: AbortSignal): Promise<void> {
    signal.throwIfAborted();
    let reject!: (error: unknown) => void;
    const aborted = new Promise<never>((_resolve, no) => { reject = no; });
    const abort = (): void => reject(signal.reason);
    signal.addEventListener("abort", abort, { once: true });
    try { await Promise.race([this.connect(), aborted]); }
    finally { signal.removeEventListener("abort", abort); }
  }

  notify(method: string, params: Record<string, unknown>, signal = this.controller.signal): Promise<void> {
    return this.send({ jsonrpc: "2.0", method, params }, signal);
  }
  reply(message: Record<string, unknown>, signal = this.controller.signal): Promise<void> { return this.send(message, signal); }
  stop(): void { if (this.stopped) return; this.stopped = true; this.controller.abort(); this.fail(new Error("MCP legacy SSE was closed.")); this.options.onClosed(); }
}
