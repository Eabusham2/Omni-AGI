import { describe, expect, it, vi } from "vitest";
import { ExternalAdviceClient } from "../src/main/externalAdviceClient";

const encoder = new TextEncoder();
function eventStream(events: string[], init: ResponseInit = {}): Response {
  return new Response(new ReadableStream<Uint8Array>({
    start(controller) { for (const event of events) controller.enqueue(encoder.encode(event)); controller.close(); }
  }), { status: 200, headers: { "content-type": "text/event-stream", ...(init.headers ?? {}) } });
}

describe("external advice provider streaming client", () => {
  it("streams OpenAI Responses text with store disabled and no tools", async () => {
    const fetcher = vi.fn<typeof fetch>(async (_url, _init) => eventStream([
      'event: response.created\ndata: {"type":"response.created","response":{"id":"resp-1"}}\n\n',
      'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hel"}\n\n',
      'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"lo"}\n\n',
      'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp-1","status":"completed"}}\n\n'
    ]));
    const deltas: string[] = [];
    const result = await new ExternalAdviceClient(fetcher as typeof fetch).stream({ provider: "openai", model: "manual-model",
      text: "visible request", maxOutputTokens: 321, credential: "synthetic-openai-secret" }, delta => { deltas.push(delta); }, new AbortController().signal);
    expect(result).toEqual({ text: "hello", providerRequestId: "resp-1", finishReason: "completed" });
    expect(deltas).toEqual(["hel", "lo"]);
    const [url, init] = fetcher.mock.calls[0]!;
    expect(url).toBe("https://api.openai.com/v1/responses");
    expect((init!.headers as Record<string, string>).authorization).toBe("Bearer synthetic-openai-secret");
    expect(JSON.parse(String(init!.body))).toEqual({ model: "manual-model", input: "visible request", max_output_tokens: 321, stream: true, store: false });
    expect(String(init!.body)).not.toContain("tools");
  });

  it("parses Anthropic text events while ignoring ping and refusing tool blocks", async () => {
    const fetcher = vi.fn<typeof fetch>(async () => eventStream([
      'event: message_start\ndata: {"type":"message_start","message":{"id":"msg-1"}}\n\n',
      'event: ping\ndata: {"type":"ping"}\n\n',
      'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"review"}}\n\n',
      'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n',
      'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    ]));
    const result = await new ExternalAdviceClient(fetcher as typeof fetch).stream({ provider: "anthropic", model: "manual-model",
      text: "proposal", maxOutputTokens: 40, credential: "synthetic-anthropic-secret" }, () => undefined, new AbortController().signal);
    expect(result).toEqual({ text: "review", providerRequestId: "msg-1", finishReason: "end_turn" });
    const [, init] = fetcher.mock.calls[0]!;
    expect(init!.headers).toMatchObject({ "x-api-key": "synthetic-anthropic-secret", "anthropic-version": "2023-06-01" });

    const toolClient = new ExternalAdviceClient((async () => eventStream([
      'event: content_block_start\ndata: {"type":"content_block_start","content_block":{"type":"tool_use","name":"shell"}}\n\n'
    ])) as typeof fetch);
    await expect(toolClient.stream({ provider: "anthropic", model: "manual-model", text: "no tools", maxOutputTokens: 40,
      credential: "synthetic-anthropic-secret" }, () => undefined, new AbortController().signal)).rejects.toThrow(/cannot request host tools/);
  });

  it("uses Gemini SSE text parts and an exact manual model resource", async () => {
    const fetcher = vi.fn<typeof fetch>(async () => eventStream([
      'data: {"responseId":"gem-1","candidates":[{"content":{"parts":[{"text":"one"}]}}]}\n\n',
      'data: {"responseId":"gem-1","candidates":[{"content":{"parts":[{"text":" two"}]},"finishReason":"STOP"}]}\n\n'
    ]));
    const result = await new ExternalAdviceClient(fetcher as typeof fetch).stream({ provider: "gemini", model: "models/user-selected",
      text: "visible", maxOutputTokens: 55, credential: "synthetic-gemini-secret" }, () => undefined, new AbortController().signal);
    expect(result).toEqual({ text: "one two", providerRequestId: "gem-1", finishReason: "STOP" });
    const [url, init] = fetcher.mock.calls[0]!;
    expect(url).toBe("https://generativelanguage.googleapis.com/v1beta/models/user-selected:streamGenerateContent?alt=sse");
    expect((init!.headers as Record<string, string>)["x-goog-api-key"]).toBe("synthetic-gemini-secret");
  });

  it("supports OpenRouter and explicit loopback OpenAI-compatible streams without fallback", async () => {
    const fetcher = vi.fn<typeof fetch>(async () => eventStream([
      'data: {"id":"chat-1","choices":[{"delta":{"content":"local text"},"finish_reason":"stop"}]}\n\n',
      "data: [DONE]\n\n"
    ]));
    const client = new ExternalAdviceClient(fetcher as typeof fetch);
    await expect(client.stream({ provider: "openrouter", model: "vendor/manual", text: "ask", maxOutputTokens: 20,
      credential: "synthetic-router-secret" }, () => undefined, new AbortController().signal)).resolves.toMatchObject({ text: "local text" });
    expect(fetcher.mock.calls[0]![0]).toBe("https://openrouter.ai/api/v1/chat/completions");
    expect(JSON.parse(String(fetcher.mock.calls[0]![1]!.body))).toMatchObject({ model: "vendor/manual", max_completion_tokens: 20,
      provider: { allow_fallbacks: false }, stream: true });
    expect((fetcher.mock.calls[0]![1]!.headers as Record<string, string>)["x-openrouter-cache"]).toBe("false");

    await expect(client.stream({ provider: "openai-compatible-local", model: "local-manual", endpoint: "http://127.0.0.1:8899/v1",
      text: "ask", maxOutputTokens: 20 }, () => undefined, new AbortController().signal)).resolves.toMatchObject({ text: "local text" });
    expect(fetcher.mock.calls[1]![0]).toBe("http://127.0.0.1:8899/v1/chat/completions");
    expect((fetcher.mock.calls[1]![1]!.headers as Record<string, string>).authorization).toBeUndefined();
    await expect(client.stream({ provider: "openai-compatible-local", model: "local-manual", endpoint: "http://example.com/v1",
      text: "ask", maxOutputTokens: 20 }, () => undefined, new AbortController().signal)).rejects.toThrow(/loopback/);
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("redacts provider errors, rejects redirects, and aborts a pending stream", async () => {
    const secret = "synthetic-sensitive-credential";
    const failed = new ExternalAdviceClient((async () => new Response(JSON.stringify({ error: { message: `bad ${secret}` } }), {
      status: 401, headers: { "content-type": "application/json" }
    })) as typeof fetch);
    const failure = failed.stream({ provider: "openai", model: "manual-model", text: "ask", maxOutputTokens: 10, credential: secret },
      () => undefined, new AbortController().signal);
    await expect(failure).rejects.toThrow("bad [REDACTED]"); await expect(failure).rejects.not.toThrow(secret);

    const redirected = new ExternalAdviceClient((async () => new Response(null, { status: 302, headers: { location: "https://example.com/login" } })) as typeof fetch);
    await expect(redirected.stream({ provider: "openai", model: "manual-model", text: "ask", maxOutputTokens: 10, credential: secret },
      () => undefined, new AbortController().signal)).rejects.toThrow(/redirect/);

    const controller = new AbortController();
    const pending = new ExternalAdviceClient((async (_url, init) => eventStream([
      'data: {"type":"response.output_text.delta","delta":"first"}\n\n'
    ])) as typeof fetch).stream({ provider: "openai", model: "manual-model", text: "ask", maxOutputTokens: 10, credential: secret },
      () => { controller.abort(new Error("cancelled fixture")); }, controller.signal);
    await expect(pending).rejects.toThrow(/cancelled fixture/);
  });

  it("refuses cumulative text beyond the explicit output-byte budget", async () => {
    const chunk = "x".repeat(900_000);
    const events = Array.from({ length: 5 }, () => `data: ${JSON.stringify({ id: "bounded", choices: [{ delta: { content: chunk }, finish_reason: null }] })}\n\n`);
    const client = new ExternalAdviceClient((async () => eventStream(events)) as typeof fetch);
    await expect(client.stream({ provider: "openrouter", model: "vendor/manual", text: "ask", maxOutputTokens: 65_536,
      credential: "synthetic-router-secret" }, () => undefined, new AbortController().signal)).rejects.toThrow(/4 MiB output budget/);
  });

  it("times out the actual pending stream while retaining already delivered deltas", async () => {
    const partial: string[] = [];
    const client = new ExternalAdviceClient((async () => new Response(new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(encoder.encode('data: {"type":"response.output_text.delta","delta":"partial"}\n\n')); }
    }), { status: 200, headers: { "content-type": "text/event-stream" } })) as typeof fetch, { timeoutMs: 20 });
    await expect(client.stream({ provider: "openai", model: "manual-model", text: "ask", maxOutputTokens: 10,
      credential: "synthetic-openai-secret" }, delta => { partial.push(delta); }, new AbortController().signal)).rejects.toThrow(/timed out after 20 ms/);
    expect(partial).toEqual(["partial"]);
  });
});
