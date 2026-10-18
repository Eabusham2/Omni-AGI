export interface McpRpcMessage {
  jsonrpc?: string;
  id?: string | number | null;
  method?: string;
  params?: Record<string, unknown>;
  result?: unknown;
  error?: { code?: number; message?: string; data?: unknown };
}

export interface McpRpcStreamOptions {
  resumable?: boolean;
  lastEventId?: string;
  retryMilliseconds?: number;
  /** Return false for an already-delivered event, without interpreting its ID. */
  acceptEvent?(event: McpSseEvent): boolean;
  delivered?(event: McpSseEvent): void;
}

export class McpResumableStreamError extends Error {
  constructor(readonly lastEventId: string, readonly retryMilliseconds: number, cause?: unknown) {
    super("MCP response stream disconnected before its result; resume the existing stream, not the request.", { cause });
  }
}

export function parseMcpRpcMessages(body: string): McpRpcMessage[] {
  let value: unknown;
  try { value = JSON.parse(body); } catch { throw new Error("MCP server returned invalid JSON-RPC."); }
  return (Array.isArray(value) ? value : [value]).map(item => {
    if (!item || typeof item !== "object" || Array.isArray(item)) throw new Error("MCP server returned invalid JSON-RPC.");
    const message = item as McpRpcMessage;
    if (message.jsonrpc !== "2.0") throw new Error("MCP response has an invalid JSON-RPC version.");
    return message;
  });
}

/** Return the matching SSE response without waiting for an intentionally open stream. */
export async function readMcpRpcResponse(
  response: Response,
  expectedId: string,
  maximumBytes: number,
  onMessage: (message: McpRpcMessage) => Promise<void> | void = () => undefined,
  signal?: AbortSignal,
  options: McpRpcStreamOptions = {}
): Promise<McpRpcMessage> {
  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > maximumBytes) {
    await response.body?.cancel().catch(() => undefined);
    throw new Error("MCP response exceeded 16 MiB.");
  }
  if (!response.body) throw new Error("MCP server returned no matching JSON-RPC response.");
  const eventStream = response.headers.get("content-type")?.toLowerCase().includes("text/event-stream") ?? false;
  const parse = async (body: string): Promise<McpRpcMessage | undefined> => {
    if (!body.trim()) return;
    let match: McpRpcMessage | undefined;
    for (const message of parseMcpRpcMessages(body)) {
      if (!message.method && String(message.id) === expectedId) {
        if (Object.hasOwn(message, "result") === Object.hasOwn(message, "error")) throw new Error("MCP server returned an invalid response result.");
        match = message;
      }
      else await onMessage(message);
    }
    return match;
  };
  if (eventStream) {
    let lastEventId = options.lastEventId;
    let retry = options.retryMilliseconds ?? 1_000;
    try {
      for await (const event of readMcpEvents(response, maximumBytes, signal)) {
        if (event.retry !== undefined) retry = event.retry;
        if (options.acceptEvent?.(event) === false) continue;
        const match = event.data.trim() ? await parse(event.data) : undefined;
        if (event.id !== undefined) lastEventId = event.id;
        options.delivered?.(event);
        if (match) return match;
      }
    } catch (error) {
      if (options.resumable && lastEventId && error instanceof McpEventStreamIoError) throw new McpResumableStreamError(lastEventId, retry, error);
      throw error;
    }
    if (options.resumable && lastEventId) throw new McpResumableStreamError(lastEventId, retry);
    throw new Error("MCP server returned no matching JSON-RPC response.");
  }
  const reader = response.body.getReader();
  const abort = (): void => { void reader.cancel(signal?.reason).catch(() => undefined); };
  signal?.addEventListener("abort", abort, { once: true });
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let bytes = 0;
  let buffer = "";
  try {
    for (;;) {
      signal?.throwIfAborted();
      const { value, done } = await reader.read();
      signal?.throwIfAborted();
      if (value) {
        bytes += value.byteLength;
        if (bytes > maximumBytes) throw new Error("MCP response exceeded 16 MiB.");
        buffer += decoder.decode(value, { stream: true });
      }
      if (done) buffer += decoder.decode();
      if (done) {
        const match = await parse(buffer);
        if (match) return match;
      }
      if (done) throw new Error("MCP server returned no matching JSON-RPC response.");
    }
  } finally {
    signal?.removeEventListener("abort", abort);
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
import { readMcpEvents, McpEventStreamIoError, type McpSseEvent } from "./mcpEventStream";
