import type { ExternalAdviceProvider } from "../shared/externalAdvice";
import { ollamaLoopbackEndpoint, ollamaLocalModelId } from "./apiTeacherProviders";

type FetchLike = typeof fetch;

export interface ExternalAdviceClientRequest {
  provider: ExternalAdviceProvider;
  model: string;
  endpoint?: string;
  text: string;
  maxOutputTokens: number;
  credential?: string;
}

export interface ExternalAdviceClientResult {
  text: string;
  providerRequestId?: string;
  finishReason?: string;
}

export type ExternalAdviceDeltaHandler = (delta: string) => void | Promise<void>;
export interface ExternalAdviceClientOptions { timeoutMs?: number }

const MAX_REQUEST_BYTES = 4 * 1024 * 1024;
const MAX_STREAM_BYTES = 16 * 1024 * 1024;
const MAX_EVENT_BYTES = 1024 * 1024;
const MAX_OUTPUT_BYTES = 4 * 1024 * 1024;
const MODEL = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$/;

const REMOTE_ENDPOINTS: Readonly<Record<Exclude<ExternalAdviceProvider, "openai-compatible-local" | "ollama">, string>> = {
  openai: "https://api.openai.com/v1/responses",
  anthropic: "https://api.anthropic.com/v1/messages",
  gemini: "https://generativelanguage.googleapis.com/v1beta",
  openrouter: "https://openrouter.ai/api/v1/chat/completions"
};

function provider(value: unknown): ExternalAdviceProvider {
  if (!["openai", "anthropic", "gemini", "openrouter", "ollama", "openai-compatible-local"].includes(String(value))) {
    throw new Error("Unsupported external-advice provider.");
  }
  return value as ExternalAdviceProvider;
}

function model(value: unknown): string {
  if (typeof value !== "string" || !MODEL.test(value)) throw new Error("Invalid external-advice model ID.");
  return value;
}

function cleanCredential(value: unknown, required: boolean): string | undefined {
  if (value === undefined && !required) return undefined;
  if (typeof value !== "string" || !value || value.length > 64_000 || /[\r\n\0]/.test(value)) {
    throw new Error("The external-advice credential is unavailable or invalid.");
  }
  return value;
}

function cleanError(value: unknown, credential?: string): string {
  let result = value instanceof Error ? value.message : String(value);
  if (credential) result = result.split(credential).join("[REDACTED]");
  return result
    .replace(/\bsk-[A-Za-z0-9_-]{12,}\b/g, "[REDACTED]")
    .replace(/\bAIza[A-Za-z0-9_-]{16,}\b/g, "[REDACTED]")
    .replace(/[\r\n\0]+/g, " ")
    .slice(0, 1_000);
}

export function externalAdviceLoopbackEndpoint(value: unknown): string {
  if (typeof value !== "string" || value.length > 2_048) throw new Error("Enter an explicit loopback OpenAI-compatible endpoint.");
  let url: URL;
  try { url = new URL(value); } catch { throw new Error("Enter a valid loopback OpenAI-compatible endpoint."); }
  const hostname = url.hostname.toLowerCase();
  if (!["localhost", "127.0.0.1", "::1", "[::1]"].includes(hostname) || !["http:", "https:"].includes(url.protocol) ||
      url.username || url.password || url.search || url.hash) {
    throw new Error("Local external advice is limited to an explicit loopback HTTP(S) endpoint without embedded credentials.");
  }
  const pathname = url.pathname.replace(/\/+$/, "");
  if (!pathname || pathname === "/") url.pathname = "/v1";
  else url.pathname = pathname;
  return url.toString().replace(/\/$/, "");
}

function endpointFor(request: ExternalAdviceClientRequest): string {
  if (request.provider !== "openai-compatible-local" && request.provider !== "ollama" && request.endpoint !== undefined) {
    throw new Error("Remote external-advice endpoints are fixed and cannot be overridden.");
  }
  if (request.provider === "gemini") {
    const name = request.model.startsWith("models/") ? request.model.slice("models/".length) : request.model;
    if (!name || name.includes("/")) throw new Error("Invalid Gemini model resource name.");
    return `${REMOTE_ENDPOINTS.gemini}/models/${encodeURIComponent(name)}:streamGenerateContent?alt=sse`;
  }
  if (request.provider === "openai-compatible-local") {
    return `${externalAdviceLoopbackEndpoint(request.endpoint)}/chat/completions`;
  }
  if (request.provider === "ollama") return `${ollamaLoopbackEndpoint(request.endpoint)}/chat/completions`;
  return REMOTE_ENDPOINTS[request.provider];
}

function requestBody(request: ExternalAdviceClientRequest): Record<string, unknown> {
  if (request.provider === "openai") {
    return { model: request.model, input: request.text, max_output_tokens: request.maxOutputTokens, stream: true, store: false };
  }
  if (request.provider === "anthropic") {
    return { model: request.model, max_tokens: request.maxOutputTokens, stream: true,
      messages: [{ role: "user", content: request.text }] };
  }
  if (request.provider === "gemini") {
    return { contents: [{ role: "user", parts: [{ text: request.text }] }],
      generationConfig: { maxOutputTokens: request.maxOutputTokens } };
  }
  return { model: request.model, messages: [{ role: "user", content: request.text }],
    ...(request.provider === "openrouter" ? { max_completion_tokens: request.maxOutputTokens, provider: { allow_fallbacks: false } } : { max_tokens: request.maxOutputTokens }), stream: true };
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} is invalid.`);
  return value as Record<string, unknown>;
}

function parseJson(value: string, label: string): Record<string, unknown> {
  try { return record(JSON.parse(value), label); }
  catch (error) { if (error instanceof Error && error.message.endsWith(" is invalid.")) throw error; throw new Error(`${label} is invalid JSON.`); }
}

async function readError(response: Response, credential?: string): Promise<string> {
  if (!response.body) return `HTTP ${response.status}`;
  const reader = response.body.getReader(); const chunks: Uint8Array[] = []; let total = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read(); if (done) break; if (!value) continue;
      total += value.byteLength;
      if (total > 64 * 1024) { await reader.cancel().catch(() => undefined); break; }
      chunks.push(value);
    }
  } catch { return `HTTP ${response.status}`; }
  const bytes = new Uint8Array(Math.min(total, 64 * 1024)); let offset = 0;
  for (const chunk of chunks) { const take = Math.min(chunk.byteLength, bytes.byteLength - offset); bytes.set(chunk.subarray(0, take), offset); offset += take; if (offset === bytes.byteLength) break; }
  let message = new TextDecoder().decode(bytes);
  try {
    const body = record(JSON.parse(message), "Provider error");
    const error = body.error && typeof body.error === "object" && !Array.isArray(body.error) ? body.error as Record<string, unknown> : body;
    message = String(error.message ?? body.message ?? `HTTP ${response.status}`);
  } catch { /* The bounded plain response is still useful after redaction. */ }
  return cleanError(message || `HTTP ${response.status}`, credential);
}

interface SseEvent { event?: string; data: string }

async function* sseEvents(body: ReadableStream<Uint8Array>, signal: AbortSignal): AsyncGenerator<SseEvent> {
  const reader = body.getReader(); const decoder = new TextDecoder();
  const abort = () => { void reader.cancel(signal.reason).catch(() => undefined); };
  signal.addEventListener("abort", abort, { once: true });
  let pending = "", eventName: string | undefined, data: string[] = [], rawBytes = 0, eventBytes = 0;
  const emit = (): SseEvent | undefined => {
    if (!data.length) { eventName = undefined; eventBytes = 0; return; }
    const value = { ...(eventName ? { event: eventName } : {}), data: data.join("\n") };
    eventName = undefined; data = []; eventBytes = 0; return value;
  };
  try {
    for (;;) {
      signal.throwIfAborted();
      const { value, done } = await reader.read();
      if (done) break;
      if (!value) continue;
      rawBytes += value.byteLength;
      if (rawBytes > MAX_STREAM_BYTES) throw new Error("External-advice stream exceeded 16 MiB.");
      pending += decoder.decode(value, { stream: true });
      for (;;) {
        const newline = pending.indexOf("\n"); if (newline < 0) break;
        let line = pending.slice(0, newline); pending = pending.slice(newline + 1);
        if (line.endsWith("\r")) line = line.slice(0, -1);
        eventBytes += Buffer.byteLength(line) + 1;
        if (eventBytes > MAX_EVENT_BYTES) throw new Error("External-advice stream event exceeded 1 MiB.");
        if (!line) { const value = emit(); if (value) yield value; continue; }
        if (line.startsWith(":")) continue;
        const colon = line.indexOf(":"), field = colon < 0 ? line : line.slice(0, colon);
        let fieldValue = colon < 0 ? "" : line.slice(colon + 1); if (fieldValue.startsWith(" ")) fieldValue = fieldValue.slice(1);
        if (field === "event") eventName = fieldValue;
        else if (field === "data") data.push(fieldValue);
      }
      if (Buffer.byteLength(pending) > MAX_EVENT_BYTES) throw new Error("External-advice stream event exceeded 1 MiB.");
    }
    pending += decoder.decode();
    if (pending) {
      if (pending.endsWith("\r")) pending = pending.slice(0, -1);
      eventBytes += Buffer.byteLength(pending);
      if (eventBytes > MAX_EVENT_BYTES) throw new Error("External-advice stream event exceeded 1 MiB.");
      if (pending.startsWith("data:")) data.push(pending.slice(5).replace(/^ /, ""));
      else if (pending.startsWith("event:")) eventName = pending.slice(6).replace(/^ /, "");
    }
    const final = emit(); if (final) yield final;
    signal.throwIfAborted();
  } finally { signal.removeEventListener("abort", abort); reader.releaseLock(); }
}

function toolContent(value: unknown): boolean {
  if (!value || typeof value !== "object") return false;
  const row = value as Record<string, unknown>;
  return row.type === "tool_use" || row.type === "server_tool_use" || row.type === "function_call" ||
    row.tool_calls != null && (!Array.isArray(row.tool_calls) || row.tool_calls.length > 0) || row.functionCall != null || row.function_call != null;
}

/** Direct text-only provider client. It has no tool executor, fallback router or credential store. */
export class ExternalAdviceClient {
  constructor(private readonly fetcher: FetchLike = fetch, private readonly options: ExternalAdviceClientOptions = {}) {}

  async stream(request: ExternalAdviceClientRequest, onDelta: ExternalAdviceDeltaHandler, signal: AbortSignal): Promise<ExternalAdviceClientResult> {
    const timeoutMs = this.options.timeoutMs ?? 120_000;
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 600_000) throw new Error("External-advice timeout must be between 1 ms and 10 minutes.");
    const timeout = new AbortController();
    const timer = setTimeout(() => timeout.abort(new Error(`External-advice request timed out after ${timeoutMs} ms.`)), timeoutMs);
    timer.unref?.();
    const combined = AbortSignal.any([signal, timeout.signal]);
    try { return await this.streamBounded(request, onDelta, combined); }
    finally { clearTimeout(timer); }
  }

  private async streamBounded(request: ExternalAdviceClientRequest, onDelta: ExternalAdviceDeltaHandler, signal: AbortSignal): Promise<ExternalAdviceClientResult> {
    const selectedProvider = provider(request?.provider), selectedModel = model(request?.model);
    if (selectedProvider === "ollama") ollamaLocalModelId(selectedModel);
    if (typeof request.text !== "string" || !request.text.trim() || request.text.includes("\0") || Buffer.byteLength(request.text) > MAX_REQUEST_BYTES) {
      throw new Error("External-advice request text is empty or exceeds 4 MiB.");
    }
    if (!Number.isSafeInteger(request.maxOutputTokens) || request.maxOutputTokens < 1 || request.maxOutputTokens > 65_536) {
      throw new Error("External-advice output tokens must be between 1 and 65,536.");
    }
    if (selectedProvider === "ollama" && request.credential) throw new Error("Local Ollama uses no API key; an unrelated credential cannot be forwarded.");
    const credential = cleanCredential(request.credential, selectedProvider !== "openai-compatible-local" && selectedProvider !== "ollama");
    const safe = { ...request, provider: selectedProvider, model: selectedModel, credential };
    const endpoint = endpointFor(safe), headers: Record<string, string> = {
      accept: "text/event-stream", "content-type": "application/json"
    };
    if (selectedProvider === "anthropic") { headers["x-api-key"] = credential!; headers["anthropic-version"] = "2023-06-01"; }
    else if (selectedProvider === "gemini") headers["x-goog-api-key"] = credential!;
    else if (credential) headers.authorization = `Bearer ${credential}`;
    if (selectedProvider === "openrouter") headers["x-openrouter-cache"] = "false";
    let response: Response;
    try {
      response = await this.fetcher(endpoint, { method: "POST", headers, body: JSON.stringify(requestBody(safe)), signal, redirect: "manual" });
    } catch (error) { throw new Error(`External-advice request failed: ${cleanError(error, credential)}`); }
    if (response.redirected || response.type === "opaqueredirect" || response.status >= 300 && response.status < 400) {
      throw new Error("External-advice request refused an authentication or endpoint redirect.");
    }
    if (response.url && response.url !== endpoint) throw new Error("External-advice response came from an unexpected endpoint.");
    if (!response.ok) throw new Error(`External-advice request failed: ${await readError(response, credential)}`);
    if (!response.body || !String(response.headers.get("content-type") ?? "").toLowerCase().includes("text/event-stream")) {
      throw new Error("External-advice provider did not return a streamed event response.");
    }
    let output = "", outputBytes = 0, complete = false, providerRequestId: string | undefined, finishReason: string | undefined;
    const append = async (delta: unknown): Promise<void> => {
      if (typeof delta !== "string" || !delta || delta.includes("\0")) return;
      const bytes = Buffer.byteLength(delta); outputBytes += bytes;
      if (bytes > MAX_EVENT_BYTES || outputBytes > MAX_OUTPUT_BYTES) throw new Error("External-advice text exceeded the 4 MiB output budget.");
      output += delta; await onDelta(delta);
    };
    try {
      for await (const event of sseEvents(response.body, signal)) {
        signal.throwIfAborted();
        if (event.data === "[DONE]") { complete = true; break; }
        const body = parseJson(event.data, "External-advice stream event");
        const kind = typeof body.type === "string" ? body.type : event.event;
        if (kind === "error" || kind === "response.error" || kind === "response.failed") {
          const remote = body.error && typeof body.error === "object" ? body.error as Record<string, unknown> : body;
          throw new Error(cleanError(remote.message ?? "The provider reported a stream error.", credential));
        }
        if (selectedProvider === "openai") {
          if (toolContent(body.item) || toolContent(body.output)) throw new Error("External-advice providers cannot request host tools.");
          if (kind === "response.output_text.delta") await append(body.delta);
          if (kind === "response.created" && typeof (body.response as Record<string, unknown> | undefined)?.id === "string") providerRequestId = String((body.response as Record<string, unknown>).id);
          if (kind === "response.completed") { complete = true; const responseRow = body.response as Record<string, unknown> | undefined;
            if (typeof responseRow?.id === "string") providerRequestId = responseRow.id; if (typeof responseRow?.status === "string") finishReason = responseRow.status; }
        } else if (selectedProvider === "anthropic") {
          if (kind === "content_block_start" && toolContent(body.content_block)) throw new Error("External-advice providers cannot request host tools.");
          if (kind === "content_block_delta") { const delta = record(body.delta, "Anthropic stream delta"); if (delta.type === "text_delta") await append(delta.text); }
          if (kind === "message_start") { const message = record(body.message, "Anthropic message start"); if (typeof message.id === "string") providerRequestId = message.id; }
          if (kind === "message_delta") { const delta = record(body.delta, "Anthropic message delta"); if (typeof delta.stop_reason === "string") finishReason = delta.stop_reason; }
          if (kind === "message_stop") complete = true;
        } else if (selectedProvider === "gemini") {
          if (body.error) { const error = record(body.error, "Gemini stream error"); throw new Error(cleanError(error.message ?? "Gemini reported a stream error.", credential)); }
          if (typeof body.responseId === "string") providerRequestId = body.responseId;
          if (!Array.isArray(body.candidates)) throw new Error("Gemini stream event is missing candidates.");
          for (const candidate of body.candidates) {
            const row = record(candidate, "Gemini candidate"), content = record(row.content, "Gemini candidate content");
            if (!Array.isArray(content.parts)) throw new Error("Gemini candidate parts are invalid.");
            for (const part of content.parts) { if (toolContent(part)) throw new Error("External-advice providers cannot request host tools."); await append(record(part, "Gemini candidate part").text); }
            if (typeof row.finishReason === "string") { finishReason = row.finishReason; complete = true; }
          }
        } else {
          if (typeof body.id === "string") providerRequestId = body.id;
          if (!Array.isArray(body.choices)) throw new Error("OpenAI-compatible stream event is missing choices.");
          for (const choice of body.choices) {
            const row = record(choice, "OpenAI-compatible choice"), delta = record(row.delta ?? {}, "OpenAI-compatible delta");
            if (toolContent(delta)) throw new Error("External-advice providers cannot request host tools.");
            await append(delta.content);
            if (typeof row.finish_reason === "string") { finishReason = row.finish_reason; complete = true; }
          }
        }
      }
    } catch (error) {
      if (signal.aborted) throw signal.reason instanceof Error ? signal.reason : new Error("External-advice request was cancelled.");
      throw new Error(`External-advice stream failed: ${cleanError(error, credential)}`);
    }
    if (!complete) throw new Error("External-advice stream ended without a completion event.");
    if (!output.trim()) throw new Error("External-advice provider returned no text.");
    return { text: output, ...(providerRequestId ? { providerRequestId } : {}), ...(finishReason ? { finishReason } : {}) };
  }
}
