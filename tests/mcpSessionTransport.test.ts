import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { McpClientService } from "../src/main/mcpClient";
import { SecureSecretStore } from "../src/main/secureSecretStore";
import { readMcpEvents } from "../src/main/mcpEventStream";
import type { McpCapabilityEvent } from "../src/shared/mcpCapabilities";

type Rpc = { id?: string; method: string; params: Record<string, unknown> };
const encode = (value: unknown) => new TextEncoder().encode(String(value));
const sse = (value: string) => new Response(value, { headers: { "content-type": "text/event-stream" } });

describe("MCP resumable HTTP sessions", () => {
  let root: string;
  const clients: McpClientService[] = [];
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-mcp-session-")); });
  afterEach(async () => { clients.splice(0).forEach(client => client.dispose()); await rm(root, { recursive: true, force: true }); });

  function fixture(background = false) {
    const permissions: Array<{ toolId: string; level: string }> = [];
    const brain = { registerExternalToolSchemas: vi.fn(), unregisterExternalToolSchemas: vi.fn(),
      learnStructuredExperience: vi.fn(async () => ({})), listToolPermissions: vi.fn(async () => permissions),
      setToolPermission: vi.fn(async (_id: string, toolId: string, level: string) => { permissions.push({ toolId, level }); return permissions; }) };
    const calls: Array<{ method: string; rpc?: Rpc; headers: Headers; body?: BodyInit | null; url: string }> = [];
    let legacyMode = false;
    let handler: (method: string, rpc: Rpc | undefined, init?: RequestInit, url?: string) => Response | Promise<Response> = (method, rpc) => {
      if (method === "GET") return new Response(null, { status: 405 });
      if (method === "DELETE") return new Response(null, { status: 204 });
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: rpc!.id, result: { content: [{ type: "text", text: "actual tool result" }] } }), { headers: { "content-type": "application/json" } });
    };
    const fetcher = vi.fn(async (_url: unknown, init?: RequestInit): Promise<Response> => {
      const method = init?.method ?? "GET";
      const rpc = init?.body ? JSON.parse(String(init.body)) as Rpc : undefined;
      const url = String(_url);
      calls.push({ method, rpc, headers: new Headers(init?.headers), body: init?.body, url });
      if (legacyMode) {
        if (url.includes("legacy-messages") || method === "GET") return handler(method, rpc, init, url);
        if (rpc?.method === "initialize") return new Response(null, { status: 405 });
      }
      if (rpc?.method === "notifications/initialized" || rpc?.method === "notifications/cancelled") return new Response(null, { status: 202 });
      const result = rpc?.method === "initialize" ? { protocolVersion: "2025-11-25", capabilities: { tools: { listChanged: true }, resources: { subscribe: true, listChanged: true } } }
        : rpc?.method === "tools/list" ? { tools: [{ name: "act", inputSchema: { type: "object" } }] } : undefined;
      if (result) return new Response(JSON.stringify({ jsonrpc: "2.0", id: rpc!.id, result }), { headers: { "content-type": "application/json", "mcp-session-id": "session-visible-ascii" } });
      return handler(method, rpc, init, url);
    });
    const secrets = new SecureSecretStore(join(root, "secrets.json"), { available: () => false, encrypt: value => Buffer.from(value), decrypt: value => value.toString() });
    const client = new McpClientService(join(root, "mcp.json"), secrets, brain as never, fetcher as typeof fetch);
    clients.push(client); if (background) client.enableSessionTransports();
    const add = async () => { await client.initialize(); return client.add({ brainId: "first", id: "remote", label: "Remote", transport: "http", url: "http://127.0.0.1:8199/mcp", bearerToken: "fixture-session-credential" }); };
    return { client, brain, calls, add, setHandler: (value: typeof handler) => { handler = value; }, rejectModernInitialize: () => { legacyMode = true; } };
  }

  it("resumes the existing POST stream with GET and a UTF-8 event ID without repeating its tool effect", async () => {
    const f = fixture(); const summary = await f.add();
    let requestId = "";
    f.setHandler((method, rpc, init) => {
      if (rpc?.method === "tools/call") { requestId = rpc.id!; return sse("id: start-☃\nretry: 85\ndata:\n\n"); }
      expect(method).toBe("GET");
      const eventId = new Headers(init?.headers).get("last-event-id")!;
      expect(Buffer.from(eventId, "latin1").toString("utf8")).toBe("start-☃");
      return sse(`id: final\ndata: ${JSON.stringify({ jsonrpc: "2.0", id: requestId, result: { content: [{ type: "text", text: "resumed original effect" }] } })}\n\n`);
    });
    const start = performance.now();
    const result = await f.client.execute(summary.tools[0]!.id, "call", {}, new AbortController().signal, "first");
    expect(result).toMatchObject({ content: [{ text: "resumed original effect" }] });
    expect(performance.now() - start).toBeGreaterThanOrEqual(80);
    expect(f.calls.filter(call => call.rpc?.method === "tools/call")).toHaveLength(1);
    expect(f.calls.filter(call => call.method === "GET")).toHaveLength(1);
  });

  it("reports lost-session outcome as unknown instead of reinitializing and replaying the earlier effect", async () => {
    const f = fixture(); const summary = await f.add();
    f.setHandler((method) => method === "POST" ? sse("id: before-result\nretry: 0\ndata:\n\n") : new Response(null, { status: 404 }));
    await expect(f.client.execute(summary.tools[0]!.id, "call", {}, new AbortController().signal, "first")).rejects.toThrow("outcome is unknown");
    expect(f.calls.filter(call => call.rpc?.method === "tools/call")).toHaveLength(1);
    expect(f.calls.filter(call => call.rpc?.method === "initialize")).toHaveLength(1);
    expect((await f.client.list("first"))[0]?.connected).toBe(false);
  });

  it("does not resume malformed JSON after an event ID or reinterpret cancellation as stream failure", async () => {
    const f = fixture(); const summary = await f.add();
    f.setHandler(() => sse("id: valid-cursor\ndata:\n\nid: invalid\ndata: broken JSON\n\n"));
    await expect(f.client.execute(summary.tools[0]!.id, "call", {}, new AbortController().signal, "first")).rejects.toThrow("invalid JSON-RPC");
    expect(f.calls.filter(call => call.method === "GET")).toHaveLength(0);
  });

  it("listens in the real client, resumes notifications once, deduplicates event IDs and terminates explicitly on Off", async () => {
    const f = fixture(true);
    const events: McpCapabilityEvent[] = []; f.client.onCapabilityEvent(event => events.push(event));
    let first!: ReadableStreamDefaultController<Uint8Array>;
    let second!: ReadableStreamDefaultController<Uint8Array>;
    let streams = 0, cancelled = 0;
    f.setHandler((method, _rpc, init) => {
      if (method === "DELETE") return new Response(null, { status: 204 });
      if (method !== "GET") return new Response(JSON.stringify({ jsonrpc: "2.0", id: _rpc!.id, result: {} }), { headers: { "content-type": "application/json" } });
      streams += 1;
      if (streams === 2) expect(new Headers(init?.headers).get("last-event-id")).toBe("notice-1");
      return new Response(new ReadableStream<Uint8Array>({ start(controller) { if (streams === 1) first = controller; else second = controller; }, cancel() { cancelled += 1; } }), { headers: { "content-type": "text/event-stream" } });
    });
    await f.add(); await vi.waitFor(() => expect(first).toBeDefined());
    const notice = { jsonrpc: "2.0", method: "notifications/resources/list_changed" };
    first.enqueue(encode(`id: notice-1\nretry: 20\ndata: ${JSON.stringify(notice)}\n\n`)); first.close();
    await vi.waitFor(() => expect(second).toBeDefined());
    second.enqueue(encode(`id: notice-1\ndata: ${JSON.stringify(notice)}\n\nid: notice-2\ndata: ${JSON.stringify(notice)}\n\n`));
    await vi.waitFor(() => expect(events.filter(event => event.kind === "resources-list-changed")).toHaveLength(2));
    await f.client.configure({ enabled: false });
    await vi.waitFor(() => expect(cancelled).toBe(1));
    await vi.waitFor(async () => expect((await f.client.list("first"))[0]?.transportObservation?.lastSessionTermination).toBe("complete"));
    const termination = f.calls.find(call => call.method === "DELETE")!;
    expect(termination.body).toBeUndefined();
    expect(termination.headers.get("mcp-session-id")).toBe("session-visible-ascii");
    expect(termination.headers.get("authorization")).toBe("Bearer fixture-session-credential");
    expect(await readFile(join(root, "mcp.json"), "utf8")).not.toContain("fixture-session-credential");
    expect(f.calls.filter(call => call.rpc?.method === "resources/subscribe")).toHaveLength(0);
  });

  it("reports optional GET/DELETE support honestly without blocking ordinary tools or replaying subscriptions", async () => {
    const f = fixture(true); const summary = await f.add();
    await vi.waitFor(async () => expect((await f.client.list("first"))[0]?.transportObservation?.eventStream).toBe("unsupported"));
    expect(await f.client.execute(summary.tools[0]!.id, "call", {}, new AbortController().signal, "first")).toMatchObject({ content: [{ text: "actual tool result" }] });
    f.setHandler((method) => new Response(null, { status: method === "DELETE" ? 405 : 200 }));
    f.client.suspendBrain("first");
    await vi.waitFor(async () => expect((await f.client.list("first"))[0]?.transportObservation?.lastSessionTermination).toBe("unsupported"));
    expect(f.calls.filter(call => call.method === "GET")).toHaveLength(1);
    expect(f.calls.filter(call => call.rpc?.method === "resources/subscribe")).toHaveLength(0);
  });

  it("falls back after a rejected initialize to real legacy endpoint events and routes subsequent RPC through its one SSE stream", async () => {
    const f = fixture(true); f.rejectModernInitialize();
    let stream!: ReadableStreamDefaultController<Uint8Array>;
    let cancelled = 0;
    f.setHandler((method, rpc, init, url) => {
      if (method === "GET") return new Response(new ReadableStream<Uint8Array>({ start(controller) {
        stream = controller; controller.enqueue(encode("event: endpoint\ndata: /legacy-messages?channel=local\n\n"));
      }, cancel() { cancelled += 1; } }), { headers: { "content-type": "text/event-stream" } });
      expect(url).toBe("http://127.0.0.1:8199/legacy-messages?channel=local");
      expect(new Headers(init?.headers).get("authorization")).toBe("Bearer fixture-session-credential");
      if (!rpc?.id) return new Response(null, { status: 202 });
      const result = rpc.method === "initialize" ? { protocolVersion: "2024-11-05", capabilities: { tools: {} } }
        : rpc.method === "tools/list" ? { tools: [{ name: "legacy_act", inputSchema: { type: "object" } }] }
          : { content: [{ type: "text", text: "actual legacy result" }] };
      stream.enqueue(encode(`event: message\ndata: ${JSON.stringify({ jsonrpc: "2.0", id: rpc.id, result })}\n\n`));
      return new Response(null, { status: 202 });
    });
    const summary = await f.add();
    expect(summary.connected).toBe(true); expect(summary.transportObservation?.httpMode).toBe("legacy-sse");
    expect(await f.client.execute(summary.tools[0]!.id, "call", {}, new AbortController().signal, "first")).toMatchObject({ content: [{ text: "actual legacy result" }] });
    expect(f.calls.filter(call => call.method === "GET")).toHaveLength(1);
    expect(f.calls.filter(call => call.rpc?.method === "tools/call")).toHaveLength(1);
    f.client.suspendBrain("first"); await vi.waitFor(() => expect(cancelled).toBe(1));
  });

  it("does not forward credentials to a different-origin legacy endpoint", async () => {
    const f = fixture(); f.rejectModernInitialize();
    f.setHandler(() => sse("event: endpoint\ndata: https://different.invalid/legacy-messages\n\n"));
    await expect(f.add()).rejects.toThrow("different-origin");
    expect(f.calls.some(call => call.url.startsWith("https://different.invalid"))).toBe(false);
    expect(f.calls.filter(call => call.rpc?.method === "tools/call")).toHaveLength(0);
  });
});

describe("bounded MCP SSE decoding", () => {
  it("handles split CRLF, opaque IDs and multiline data while bounding each frame rather than total stream bytes", async () => {
    const chunks = [": ping\r\nid: x\r", "\nretry: 12\r\ndata: one\r\ndata: two\r\n\r\n", "id: y\ndata: three\n\n"];
    const response = new Response(new ReadableStream<Uint8Array>({ start(controller) { chunks.forEach(chunk => controller.enqueue(encode(chunk))); controller.close(); } }), { headers: { "content-type": "text/event-stream" } });
    const events = [];
    for await (const event of readMcpEvents(response, 80)) events.push(event);
    expect(events).toEqual([{ id: "x", retry: 12, data: "one\ntwo" }, { id: "y", data: "three" }]);
    const many = [];
    for await (const event of readMcpEvents(sse("data: one\n\n".repeat(100)), 20)) many.push(event);
    expect(many).toHaveLength(100);
    await expect((async () => { for await (const _event of readMcpEvents(sse(`data: ${"x".repeat(100)}\n\n`), 20)) { /* consume */ } })()).rejects.toThrow("exceeded");
  });
});
