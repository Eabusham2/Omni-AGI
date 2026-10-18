export interface McpSseEvent {
  data: string;
  event?: string;
  id?: string;
  retry?: number;
}

/** Only transport I/O is resumable; malformed/oversized data must fail closed. */
export class McpEventStreamIoError extends Error {
  constructor(cause: unknown) { super("MCP event stream disconnected.", { cause }); this.name = "McpEventStreamIoError"; }
}

function eventFrom(body: string): McpSseEvent | undefined {
  const event: McpSseEvent = { data: "" };
  const data: string[] = [];
  let recognized = false;
  for (const line of body.split(/\r\n|\n|\r/)) {
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "data") { data.push(value); recognized = true; }
    else if (field === "event") { event.event = value; recognized = true; }
    else if (field === "id" && !value.includes("\0")) { event.id = value; recognized = true; }
    else if (field === "retry" && /^\d+$/.test(value) && Number.isSafeInteger(Number(value))) { event.retry = Number(value); recognized = true; }
  }
  event.data = data.join("\n");
  return recognized ? event : undefined;
}

/** Bounded per-event buffering; a long-lived stream has no cumulative data cap. */
export async function* readMcpEvents(response: Response, maximumEventBytes: number, signal?: AbortSignal): AsyncGenerator<McpSseEvent> {
  if (!response.headers.get("content-type")?.toLowerCase().includes("text/event-stream")) {
    await response.body?.cancel().catch(() => undefined);
    throw new Error("MCP listener did not return text/event-stream.");
  }
  if (!response.body) throw new Error("MCP event stream has no body.");
  const reader = response.body.getReader();
  const abort = (): void => { void reader.cancel(signal?.reason).catch(() => undefined); };
  signal?.addEventListener("abort", abort, { once: true });
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let buffer = "";
  try {
    for (;;) {
      signal?.throwIfAborted();
      let next: ReadableStreamReadResult<Uint8Array>;
      try { next = await reader.read(); }
      catch (error) { signal?.throwIfAborted(); throw new McpEventStreamIoError(error); }
      signal?.throwIfAborted();
      if (next.value) buffer += decoder.decode(next.value, { stream: true });
      if (next.done) buffer += decoder.decode();
      for (;;) {
        const boundary = /\r\n\r\n|\n\n|\r\r/.exec(buffer);
        if (!boundary) break;
        const body = buffer.slice(0, boundary.index);
        buffer = buffer.slice(boundary.index + boundary[0].length);
        if (Buffer.byteLength(body, "utf8") > maximumEventBytes) throw new Error("MCP response exceeded 16 MiB.");
        const event = eventFrom(body); if (event) yield event;
      }
      if (Buffer.byteLength(buffer, "utf8") > maximumEventBytes) throw new Error("MCP response exceeded 16 MiB.");
      if (next.done) {
        if (buffer) { const event = eventFrom(buffer); if (event) yield event; }
        return;
      }
    }
  } finally {
    signal?.removeEventListener("abort", abort);
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
