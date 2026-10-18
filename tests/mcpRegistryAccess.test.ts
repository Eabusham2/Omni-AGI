import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { McpClientService } from "../src/main/mcpClient";
import { SecureSecretStore } from "../src/main/secureSecretStore";
import { readMcpRpcResponse } from "../src/main/mcpStream";
import { ChatActionController } from "../src/main/chatActionController";
import type { ChatResult, ChatStreamEvent } from "../src/shared/types";
import type { ToolPermissionRecord } from "../src/shared/types";

describe("shared MCP registration and instance access", () => {
  let root: string;
  const services: McpClientService[] = [];
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-mcp-access-")); });
  afterEach(async () => { for (const service of services.splice(0)) service.dispose(); await rm(root, { recursive: true, force: true }); });

  function fixture(options: { pendingCalls?: boolean; protocolVersion?: string } = {}) {
    const permissions = new Map<string, ToolPermissionRecord[]>();
    const schemas = new Map<string, unknown>();
    const brain = {
      registerExternalToolSchemas: vi.fn((values: Array<{ id: string }>) => { for (const value of values) schemas.set(value.id, value); }),
      unregisterExternalToolSchemas: vi.fn((ids: string[]) => { for (const id of ids) schemas.delete(id); }),
      learnStructuredExperience: vi.fn(async () => ({})),
      listToolPermissions: vi.fn(async (brainId: string) => permissions.get(brainId) ?? []),
      setToolPermission: vi.fn(async (brainId: string, toolId: string, level: ToolPermissionRecord["level"]) => {
        const values = permissions.get(brainId) ?? [];
        const entry = values.find((value) => value.toolId === toolId);
        if (entry) entry.level = level;
        else values.push({ toolId, level, label: toolId, updatedAt: new Date().toISOString() });
        permissions.set(brainId, values);
        return values;
      })
    };
    const remoteCalls: Array<{ method: string; params: Record<string, unknown>; headers: Headers }> = [];
    const fetcher = vi.fn(async (_url: unknown, init?: RequestInit): Promise<Response> => {
      const request = JSON.parse(String(init?.body));
      remoteCalls.push({ method: request.method, params: request.params ?? {}, headers: new Headers(init?.headers) });
      if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
      if (request.method === "tools/call" && options.pendingCalls) return new Promise<Response>((_resolve, reject) => {
        const abort = (): void => reject(new Error("network aborted"));
        if (init?.signal?.aborted) abort();
        else init?.signal?.addEventListener("abort", abort, { once: true });
      });
      const result = request.method === "initialize"
        ? { protocolVersion: options.protocolVersion ?? "2025-11-25", capabilities: { tools: { listChanged: true } }, serverInfo: { name: "fixture", version: "1" } }
        : request.method === "tools/list"
          ? { tools: [{ name: "inspect_fixture", description: "Inspect a project using a typed path.",
            inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
            outputSchema: { type: "object" }, annotations: { readOnlyHint: true }, _meta: { ui: { resourceUri: "ui://fixture/panel" } } }] }
          : { content: [{ type: "text", text: "actual evidence" }] };
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: request.id, result }), { headers: { "content-type": "application/json", "mcp-session-id": "fixture-session" } });
    });
    const secrets = new SecureSecretStore(join(root, "secrets.json"), {
      available: () => false, encrypt: (value) => Buffer.from(value), decrypt: (value) => value.toString()
    });
    const create = (file = "mcp.json"): McpClientService => {
      const service = new McpClientService(join(root, file), secrets, brain as never, fetcher as typeof fetch);
      services.push(service);
      return service;
    };
    return { create, brain, schemas, permissions, remoteCalls, fetcher, secrets };
  }

  const registration = { brainId: "first", id: "coding", label: "Coding", transport: "http" as const, url: "http://127.0.0.1:8197/mcp" };

  it("isolates a merged identity from parent defaults and credentials even after its master is enabled", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    await service.add({ ...registration, bearerToken: "private-token" });
    await service.configure({ brainId: "first", registryId: "coding", accessOverride: "on" });
    await service.initializeMergedIdentity("merged");
    expect((await service.registry("merged")).instanceEnabled).toBe(false);
    expect((await service.list("merged"))[0]).toMatchObject({ accessOverride: "off", effectiveEnabled: false });
    await service.configure({ brainId: "merged", enabled: true });
    expect((await service.list("merged"))[0]).toMatchObject({ accessOverride: "off", effectiveEnabled: false });
    expect((await service.list("first"))[0]?.effectiveEnabled).toBe(true);
  });

  it("adds once for all instances, learns full typed descriptions, and inherits the last instance selection", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    const first = await service.add({ ...registration, bearerToken: "private-token", disabledBrainIds: ["excluded"] });
    expect(first).toMatchObject({ registryId: "coding", scope: "all", effectiveEnabled: true, schemaLearning: "complete", protocolVersion: "2025-11-25" });
    expect(first.tools[0]).toMatchObject({ description: "Inspect a project using a typed path.", metadata: { ui: { resourceUri: "ui://fixture/panel" } } });
    expect(JSON.stringify(f.brain.learnStructuredExperience.mock.calls)).toContain("Inspect a project");
    expect(JSON.stringify(f.brain.learnStructuredExperience.mock.calls)).not.toContain("private-token");
    const second = (await service.list("second"))[0]!;
    expect(second.effectiveEnabled).toBe(true);
    expect(second.tools[0]!.id).not.toBe(first.tools[0]!.id);
    expect((await service.list("excluded"))[0]!.effectiveEnabled).toBe(false);
    expect(f.brain.learnStructuredExperience).toHaveBeenCalledTimes(1);
    await service.learnCapabilities("second"); await service.learnCapabilities("second");
    expect(f.brain.learnStructuredExperience).toHaveBeenCalledTimes(2);
    await service.configure({ brainId: "second", registryId: "coding", accessOverride: "off" });
    expect((await service.list("third"))[0]).toMatchObject({ effectiveEnabled: false, accessOverride: "off" });
    await service.configure({ brainId: "third", registryId: "coding", accessOverride: "inherit" });
    expect((await service.list("third"))[0]!.effectiveEnabled).toBe(true);
    expect(await readFile(join(root, "mcp.json"), "utf8")).not.toContain("private-token");
  });

  it("enforces master switches at schema discovery and final dispatch even for remembered tool ids", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    const first = await service.add(registration);
    const toolId = first.tools[0]!.id;
    await expect(service.execute(toolId, "call", { path: "/one" }, new AbortController().signal, "different")).rejects.toThrow("another instance");
    await service.configure({ enabled: false });
    expect(service.hasTool(toolId)).toBe(false); expect(f.schemas.has(toolId)).toBe(false);
    const count = f.remoteCalls.filter((call) => call.method === "tools/call").length;
    await expect(service.execute(toolId, "call", { path: "/one" }, new AbortController().signal, "first")).rejects.toThrow("installation");
    await expect(service.refresh("first", "coding")).rejects.toThrow("installation");
    expect(f.remoteCalls.filter((call) => call.method === "tools/call")).toHaveLength(count);
    await service.configure({ brainId: "first", registryId: "coding", accessOverride: "on" });
    expect((await service.registry("first")).servers[0]?.effectiveEnabled).toBe(false);
    await service.configure({ enabled: true });
    await service.configure({ registryId: "coding", enabled: false });
    await expect(service.execute(toolId, "call", { path: "/one" }, new AbortController().signal)).rejects.toThrow("all instances");
    await service.configure({ registryId: "coding", enabled: true });
    await service.configure({ brainId: "first", enabled: false });
    expect(service.hasTool(toolId)).toBe(false);
    await service.configure({ brainId: "first", enabled: true });
    await expect(service.execute(toolId, "call", { path: "/one" }, new AbortController().signal, "first")).resolves.toMatchObject({ content: [{ text: "actual evidence" }] });
  });

  it("aborts in-flight transport calls on disable and persists concurrent preference edits", async () => {
    const f = fixture({ pendingCalls: true }); const service = f.create(); await service.initialize();
    const first = await service.add(registration);
    const result = service.execute(first.tools[0]!.id, "call", { path: "/one" }, new AbortController().signal).catch((error) => error);
    await vi.waitFor(() => expect(f.remoteCalls.some((call) => call.method === "tools/call")).toBe(true));
    await Promise.all([service.configure({ enabled: false }), service.configure({ brainId: "first", registryId: "coding", accessOverride: "off" })]);
    expect(await result).toMatchObject({ message: "network aborted" });
    expect(f.remoteCalls.filter((call) => call.method === "notifications/cancelled")).toHaveLength(1);
    service.dispose();
    const restored = f.create(); await restored.initialize();
    expect(await restored.registry("first")).toMatchObject({ enabled: false, servers: [{ effectiveEnabled: false, accessOverride: "off" }] });
  });

  it("keeps instance-only servers private and removes a shared registration from every instance", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    expect(await service.add({ ...registration, scope: "instance" })).toMatchObject({ scope: "instance", ownerBrainId: "first" });
    expect(await service.list("other")).toEqual([]);
    await expect(service.configure({ brainId: "other", registryId: "coding", accessOverride: "on" })).rejects.toThrow("another instance");
    await service.add({ ...registration, id: "shared" });
    const other = (await service.list("other"))[0]!;
    expect(other.id).toBe("shared");
    expect(await service.remove("first", "shared")).toBe(true);
    expect(await service.list("other")).toEqual([]);
    expect(service.hasTool(other.tools[0]!.id)).toBe(false);
  });

  it("migrates local registry ids without sharing credentials or changing existing tool identities", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    const first = await service.add(registration);
    const second = (await service.list("second"))[0]!;
    service.dispose();
    await writeFile(join(root, "legacy.json"), JSON.stringify({ format: "omni-mcp-registry", version: 1,
      servers: [first, second].map((server) => ({ brainId: server.brainId, id: "coding", label: server.label,
        transport: "http", url: registration.url, tools: server.tools })) }));
    const restored = f.create("legacy.json"); await restored.initialize();
    const firstOwned = (await restored.list("first")).find((server) => server.effectiveEnabled)!;
    const secondOwned = (await restored.list("second")).find((server) => server.effectiveEnabled)!;
    expect(firstOwned.tools[0]!.id).toBe(first.tools[0]!.id);
    expect(secondOwned.tools[0]!.id).toBe(second.tools[0]!.id);
    expect(firstOwned.registryId).not.toBe(secondOwned.registryId);
    expect((await restored.list("new")).every((server) => !server.effectiveEnabled)).toBe(true);
    restored.dispose();
    const restarted = f.create("legacy.json"); await restarted.initialize();
    expect((await restarted.list("second")).find((server) => server.effectiveEnabled)?.tools[0]!.id).toBe(second.tools[0]!.id);
  });

  it("negotiates older supported protocols and rejects unsupported versions without orphan registrations", async () => {
    const f = fixture({ protocolVersion: "2025-06-18" }); const service = f.create(); await service.initialize();
    expect((await service.add(registration)).protocolVersion).toBe("2025-06-18");
    expect(f.remoteCalls.find((call) => call.method === "tools/list")?.headers.get("mcp-protocol-version")).toBe("2025-06-18");
    const bad = fixture({ protocolVersion: "2099-01-01" }); const unsupported = bad.create("bad.json"); await unsupported.initialize();
    await expect(unsupported.add(registration)).rejects.toThrow("unsupported protocol");
    expect((await unsupported.registry("first")).servers).toEqual([]);
  });

  it("reconnects an explicitly expired HTTP session once, rediscovering and revalidating tools", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    const server = await service.add(registration);
    f.fetcher.mockImplementationOnce(async () => new Response(null, { status: 404 }));
    await expect(service.execute(server.tools[0]!.id, "call", { path: "/after-expiry" }, new AbortController().signal, "first"))
      .resolves.toMatchObject({ content: [{ text: "actual evidence" }] });
    const handshakes = f.remoteCalls.filter((call) => call.method === "initialize");
    expect(handshakes).toHaveLength(2);
    expect(handshakes[1]!.headers.has("mcp-session-id")).toBe(false);
    expect(f.remoteCalls.filter((call) => call.method === "tools/list")).toHaveLength(2);
    expect(f.remoteCalls.filter((call) => call.method === "tools/call")).toHaveLength(1);
  });

  it("does not repeat an effect when the HTTP connection fails after dispatch", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    const server = await service.add(registration);
    f.fetcher.mockRejectedValueOnce(new Error("Connection closed after possible remote effect"));
    const before = f.fetcher.mock.calls.length;
    await expect(service.execute(server.tools[0]!.id, "call", { path: "/unknown" }, new AbortController().signal, "first"))
      .rejects.toThrow("possible remote effect");
    expect(f.fetcher.mock.calls.length - before).toBe(1);
  });

  it("rejects invalid session headers and releases the response stream", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    const cancelled = vi.fn();
    f.fetcher.mockImplementationOnce(async () => new Response(new ReadableStream({ cancel: cancelled }),
      { headers: { "mcp-session-id": "invalid session" } }));
    await expect(service.add(registration)).rejects.toThrow("invalid session id");
    expect(cancelled).toHaveBeenCalledOnce();
    expect((await service.registry("first")).servers).toEqual([]);
  });

  it("does not let an old full tool grant bypass instance or global revocation", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    const first = await service.add(registration);
    const toolId = first.tools[0]!.id;
    await f.brain.setToolPermission("first", toolId, "full");
    await service.configure({ brainId: "first", registryId: "coding", accessOverride: "off" });
    expect(f.permissions.get("first")![0]!.level).toBe("full");
    expect(service.hasTool(toolId)).toBe(false);
    await expect(service.execute(toolId, "call", { path: "/one" }, new AbortController().signal, "first")).rejects.toThrow("switched off");
    expect(f.remoteCalls.some((call) => call.method === "tools/call")).toBe(false);
    await service.configure({ brainId: "first", registryId: "coding", accessOverride: "on" });
    expect(f.permissions.get("first")![0]!.level).toBe("full");
  });

  it("rejects empty target ids instead of changing the installation switch", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    await service.add(registration);
    await expect(service.configure({ brainId: "", enabled: false })).rejects.toThrow("brain id");
    await expect(service.configure({ registryId: "", enabled: false })).rejects.toThrow("server id");
    expect((await service.registry("first")).enabled).toBe(true);
  });

  it("handles object-property names as ordinary brain/server ids", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    const server = await service.add({ ...registration, brainId: "constructor", id: "constructor" });
    expect(server.effectiveEnabled).toBe(true);
    await service.configure({ brainId: "constructor", registryId: "constructor", accessOverride: "off" });
    service.dispose();
    const restored = f.create(); await restored.initialize();
    expect((await restored.registry("constructor")).servers[0]).toMatchObject({ effectiveEnabled: false, accessOverride: "off" });
  });

  it("persists Off while neural schema learning is pending, aborting that write without a success receipt", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    let learningSignal: AbortSignal | undefined;
    f.brain.learnStructuredExperience.mockImplementationOnce((...args: unknown[]) => new Promise((_resolve, reject) => {
      learningSignal = args[2] as AbortSignal;
      learningSignal.addEventListener("abort", () => reject(learningSignal!.reason), { once: true });
    }));
    const adding = service.add(registration);
    await vi.waitFor(() => expect(learningSignal).toBeDefined());
    await service.configure({ enabled: false });
    expect(learningSignal!.aborted).toBe(true);
    expect(await adding).toMatchObject({ effectiveEnabled: false, schemaLearning: "pending" });
    const saved = JSON.parse(await readFile(join(root, "mcp.json"), "utf8"));
    expect(saved.instances[0].learnedSchemas).toEqual({});
    await service.configure({ enabled: true });
    await service.learnCapabilities("first");
    expect((await service.registry("first")).servers[0]!.schemaLearning).toBe("complete");
  });

  it("does not hold registry changes behind discovery or publish results from a disabled connection", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    const first = await service.add(registration);
    let response!: (value: Response) => void;
    f.fetcher.mockImplementationOnce(async () => new Promise<Response>((resolve) => { response = resolve; }));
    const refreshing = service.refresh("first", "coding").catch((error: unknown) => error);
    await vi.waitFor(() => expect(response).toBeDefined());
    await service.configure({ registryId: "coding", enabled: false });
    response(new Response(JSON.stringify({ jsonrpc: "2.0", id: "late", result: { tools: [] } }), { headers: { "content-type": "application/json" } }));
    expect(await refreshing).toBeInstanceOf(Error);
    expect((await service.registry("first")).servers[0]!.tools).toEqual(first.tools);
    expect(service.hasTool(first.tools[0]!.id)).toBe(false);
  });

  it("deduplicates concurrent active-instance schema learning and leaves a failed receipt pending", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    await service.add(registration);
    await service.attachBrain("second");
    let release!: () => void;
    f.brain.learnStructuredExperience.mockImplementationOnce(async () => { await new Promise<void>((resolve) => { release = resolve; }); return {}; });
    const first = service.learnCapabilities("second");
    const second = service.learnCapabilities("second");
    await vi.waitFor(() => expect(release).toBeDefined());
    release(); await Promise.all([first, second]);
    expect(f.brain.learnStructuredExperience).toHaveBeenCalledTimes(2);
    await service.attachBrain("third");
    f.brain.learnStructuredExperience.mockRejectedValueOnce(new Error("Fixture neural worker unavailable"));
    const pending = await service.learnCapabilities("third");
    expect(pending[0]).toMatchObject({ schemaLearning: "pending", error: expect.stringContaining("worker unavailable") });
    expect((await service.learnCapabilities("third"))[0]!.schemaLearning).toBe("complete");
  });

  it("prepares inherited schemas before the actual chat writer and reports pending learning without quizzing", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    await service.add(registration);
    await service.attachBrain("second");
    const chat = vi.fn(async () => {
      expect(f.brain.learnStructuredExperience).toHaveBeenCalledTimes(2);
      return { brain: { messages: [] }, humanMessage: { id: "human", role: "human", content: "Hello" },
        brainMessage: { id: "reply", role: "brain", content: "Hello" }, trace: { id: "trace" } } as unknown as ChatResult;
    });
    f.brain.learnStructuredExperience.mockRejectedValueOnce(new Error("Retry neural write later"));
    const controller = new ChatActionController({ chat } as never, {} as never, {} as never);
    controller.setCapabilityPreparation(async (brainId, signal) => (await service.learnCapabilities(brainId, signal))
      .filter((server) => server.effectiveEnabled && server.schemaLearning === "pending").map((server) => server.error!));
    const streams: ChatStreamEvent[] = [];
    controller.on("stream", (event: ChatStreamEvent) => streams.push(event));
    const result = await controller.send("second", "Hello");
    expect(chat).toHaveBeenCalledOnce();
    expect(result.capabilityWarnings?.[0]).toContain("Retry neural write later");
    expect(streams.some((event) => event.type === "chat-state" && event.notices?.some((notice) => notice.includes("Retry neural write later")))).toBe(true);
  });

  it("handles stdio server requests and schema notifications before dispatching stale arguments", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    const program = [
      'const send = (message) => process.stdout.write(JSON.stringify({jsonrpc:"2.0", ...message}) + "\\n");',
      'let phase = 0, ping = false, refusedSampling = false;',
      'require("node:readline").createInterface({input:process.stdin}).on("line", (line) => {',
      'const request = JSON.parse(line);',
      'if (!request.method) { if (request.id === "server-ping") ping = !!request.result; if (request.id === "server-sampling") refusedSampling = request.error?.code === -32601; return; }',
      'if (request.method === "initialize") { send({id:"server-ping", method:"ping"}); send({id:"server-sampling",method:"sampling/createMessage",params:{}}); send({id:request.id,result:{protocolVersion:"2025-11-25",capabilities:{tools:{listChanged:true}},serverInfo:{name:"fixture",version:"1"}}}); }',
      'else if (request.method === "tools/list") send({id:request.id,result:{tools:[{name:"inspect",inputSchema:{type:"object",properties:{value:{type:phase ? "number" : "string"}},required:["value"]}}]}});',
      'else if (request.method === "tools/call") { if (!phase) { phase = 1; send({method:"notifications/tools/list_changed"}); } send({id:request.id,result:{content:[{type:"text",text:JSON.stringify({ping,refusedSampling,value:request.params.arguments.value})}]}}); }',
      '});'
    ].join("\n");
    const server = await service.add({ ...registration, transport: "stdio", command: process.execPath, args: ["-e", program], url: undefined });
    const toolId = server.tools[0]!.id;
    const result = await service.execute(toolId, "call", { value: "before" }, new AbortController().signal, "first") as {content:Array<{text:string}>};
    expect(JSON.parse(result.content[0]!.text)).toEqual({ ping: true, refusedSampling: true, value: "before" });
    await expect(service.execute(toolId, "call", { value: "stale" }, new AbortController().signal, "first")).rejects.toThrow("schema validation");
    await expect(service.execute(toolId, "call", { value: 3 }, new AbortController().signal, "first")).resolves.toMatchObject({ content: [{text: expect.stringContaining('"value":3')}] });
    // Discovery cannot await a second neural writer from an inline tool call.
    expect(f.brain.learnStructuredExperience).toHaveBeenCalledTimes(1);
    expect((await service.list("first"))[0]?.schemaLearning).toBe("pending");
    await service.learnCapabilities("first");
    expect(f.brain.learnStructuredExperience).toHaveBeenCalledTimes(2);
    expect((await service.list("first"))[0]?.schemaLearning).toBe("complete");
  });

  it("sends stdio cancellation and reinitializes a closed local server once for concurrent callers", async () => {
    const f = fixture(); const service = f.create(); await service.initialize();
    const program = [
      'const send = (message) => process.stdout.write(JSON.stringify({jsonrpc:"2.0",...message}) + "\\n");',
      'let initialized = 0, cancellations = 0;',
      'require("node:readline").createInterface({input:process.stdin}).on("line", (line) => { const request=JSON.parse(line);',
      'if (request.method === "notifications/cancelled") { cancellations++; return; }',
      'if (request.method === "initialize") { initialized++; send({id:request.id,result:{protocolVersion:"2025-11-25",capabilities:{tools:{}},serverInfo:{name:"fixture",version:"1"}}}); }',
      'else if (request.method === "tools/list") send({id:request.id,result:{tools:[{name:"inspect",inputSchema:{type:"object",properties:{wait:{type:"boolean"},exit:{type:"boolean"}}}}]}});',
      'else if (request.method === "tools/call") { if (request.params.arguments.wait) return; send({id:request.id,result:{content:[{type:"text",text:JSON.stringify({initialized,cancellations})}]}}); if(request.params.arguments.exit) setTimeout(()=>process.exit(0),10); }',
      '});'
    ].join("\n");
    const server = await service.add({ ...registration, transport: "stdio", command: process.execPath, args: ["-e", program], url: undefined });
    const toolId = server.tools[0]!.id;
    const stop = new AbortController();
    const waiting = service.execute(toolId, "call", { wait: true }, stop.signal, "first");
    await new Promise<void>((resolve) => setTimeout(resolve, 30));
    stop.abort();
    await expect(waiting).rejects.toThrow("cancelled");
    const result = await service.execute(toolId, "call", { exit: true }, new AbortController().signal, "first") as {content:Array<{text:string}>};
    expect(JSON.parse(result.content[0]!.text)).toEqual({ initialized: 1, cancellations: 1 });
    await vi.waitFor(async () => expect((await service.list("first"))[0]!.connected).toBe(false));
    const results = await Promise.all([1, 2].map(() => service.execute(toolId, "call", {}, new AbortController().signal, "first"))) as Array<{content:Array<{text:string}>}>;
    expect(results.map((value) => JSON.parse(value.content[0]!.text).initialized)).toEqual([1, 1]);
  });
});

describe("MCP streamed HTTP responses", () => {
  it("returns a split SSE response while the stream is still open, delivering notifications", async () => {
    const cancelled = vi.fn(); const incoming = vi.fn();
    const encoder = new TextEncoder();
    const response = new Response(new ReadableStream({ start(controller) {
      controller.enqueue(encoder.encode(': keepalive\r\ndata: {"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\r\n\r\n'));
      controller.enqueue(encoder.encode('event: message\ndata: {"jsonrpc":"2.0","id":"req",'));
      controller.enqueue(encoder.encode('"result":{"tools":[]}}\n\n'));
    }, cancel: cancelled }), { headers: { "content-type": "text/event-stream" } });
    expect(await readMcpRpcResponse(response, "req", 16_000, incoming)).toMatchObject({ result: { tools: [] } });
    expect(incoming).toHaveBeenCalledWith({ jsonrpc: "2.0", method: "notifications/tools/list_changed" });
    expect(cancelled).toHaveBeenCalledOnce();
  });

  it("rejects malformed JSON-RPC and oversized streams and cancels their body", async () => {
    await expect(readMcpRpcResponse(new Response('{"id":"req","result":{}}'), "req", 100)).rejects.toThrow("JSON-RPC version");
    const cancelled = vi.fn();
    const response = new Response(new ReadableStream({ start(controller) { controller.enqueue(new Uint8Array(101)); }, cancel: cancelled }));
    await expect(readMcpRpcResponse(response, "req", 100)).rejects.toThrow("exceeded");
    expect(cancelled).toHaveBeenCalledOnce();
  });

  it("cancels an open response reader when access is revoked", async () => {
    const cancelled = vi.fn(); const controller = new AbortController();
    const response = new Response(new ReadableStream({ cancel: cancelled }), { headers: { "content-type": "text/event-stream" } });
    const waiting = readMcpRpcResponse(response, "req", 100, undefined, controller.signal);
    controller.abort(new Error("Access revoked"));
    await expect(waiting).rejects.toThrow("Access revoked");
    expect(cancelled).toHaveBeenCalledOnce();
  });

  it("rejects responses without a result/error and releases a declared oversized body", async () => {
    await expect(readMcpRpcResponse(new Response('{"jsonrpc":"2.0","id":"req"}'), "req", 100)).rejects.toThrow("response result");
    const cancelled = vi.fn();
    const response = new Response(new ReadableStream({ cancel: cancelled }), { headers: { "content-length": "101" } });
    await expect(readMcpRpcResponse(response, "req", 100)).rejects.toThrow("exceeded");
    expect(cancelled).toHaveBeenCalledOnce();
  });
});
