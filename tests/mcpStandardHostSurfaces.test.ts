import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { mkdir, mkdtemp, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { McpClientService } from "../src/main/mcpClient";
import { McpRootsService } from "../src/main/mcpRoots";
import { McpDiagnosticService } from "../src/main/mcpDiagnostics";
import { SecureSecretStore } from "../src/main/secureSecretStore";
import { ToolExecutor } from "../src/main/toolExecutor";
import type { McpOAuthStatus } from "../src/shared/mcpAuth";
import type { ToolPermissionRecord } from "../src/shared/types";

let root: string;
const closes: Array<() => void | Promise<void>> = [];
beforeEach(async () => { root = await realpath(await mkdtemp(join(tmpdir(), "omni-mcp-standard-host-"))); });
afterEach(async () => { for (const close of closes.splice(0)) await close(); await rm(root, { recursive: true, force: true }); });

async function fixture(installed = true, rootParams?: unknown) {
  await mkdir(join(root, "first"), { recursive: true });
  const permissions: ToolPermissionRecord[] = [], calls: Array<Record<string, any>> = [];
  const brain = { repository: { brainDirectory: (id: string) => join(root, id), get: vi.fn(async () => ({ journal: [], traces: [] })), save: vi.fn(async value => value) },
    authorizeInlineImagination: vi.fn(async () => undefined), queueCompletedToolExecution: vi.fn(async () => undefined),
    registerExternalToolSchemas: vi.fn(), unregisterExternalToolSchemas: vi.fn(), learnStructuredExperience: vi.fn(async () => ({})),
    listToolPermissions: vi.fn(async () => permissions), setToolPermission: vi.fn(async (_id: string, toolId: string, level: ToolPermissionRecord["level"]) => {
      const previous = permissions.find(value => value.toolId === toolId); if (previous) { previous.level = level; previous.updatedAt = new Date().toISOString(); }
      else permissions.push({ toolId, label: toolId, level, updatedAt: new Date().toISOString() }); return permissions;
    }) };
  let stream: ReadableStreamDefaultController<Uint8Array> | undefined, toolId: string | undefined;
  const send = (value: unknown) => stream!.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(value)}\n\n`));
  const fetcher = vi.fn(async (_url: unknown, init?: RequestInit): Promise<Response> => {
    const request = JSON.parse(String(init?.body)); calls.push(request);
    if (request.method?.startsWith("notifications/")) return new Response(null, { status: 202 });
    if (request.id === "server-roots" && !request.method) { send({ jsonrpc: "2.0", id: toolId, result: { content: [{ type: "text", text: "New host methods completed" }] } }); stream!.close(); stream = undefined; return new Response(null, { status: 202 }); }
    if (request.method === "tools/call") {
      toolId = request.id;
      return new Response(new ReadableStream<Uint8Array>({ start(controller) { stream = controller;
        send({ jsonrpc: "2.0", method: "notifications/message", params: { level: "info", logger: "fixture", data: "fixture-bearer-secret diagnostic" } });
        send({ jsonrpc: "2.0", method: "notifications/tools/list_changed" }); send({ jsonrpc: "2.0", method: "notifications/resources/list_changed" });
        send({ jsonrpc: "2.0", id: "server-roots", method: "roots/list", ...(rootParams === undefined ? {} : { params: rootParams }) });
      } }), { headers: { "content-type": "text/event-stream" } });
    }
    const result = request.method === "initialize" ? { protocolVersion: "2025-11-25", capabilities: { tools: { listChanged: true }, resources: { listChanged: true }, logging: {} } }
      : request.method === "tools/list" ? { tools: [{ name: "fixture_action", inputSchema: { type: "object" } }] } : {};
    return Response.json({ jsonrpc: "2.0", id: request.id, result });
  }) as typeof fetch;
  const secrets = new SecureSecretStore(join(root, "private-secret.json"), { available: () => false, encrypt: value => Buffer.from(value), decrypt: value => value.toString() });
  const client = new McpClientService(join(root, "registry.json"), secrets, brain as never, fetcher); closes.push(() => client.dispose());
  const notifications = vi.fn(), invalidated = vi.fn();
  client.setAppHost({ capture: async () => undefined, requireCall: () => undefined, invalidate: invalidated, serverNotification: notifications });
  const diagnostic = new McpDiagnosticService({ requireAccess: origin => client.requireInteractionOrigin(origin), knownSecrets: origin => client.knownOperationalSecrets(origin), admit: async () => undefined }); closes.push(() => diagnostic.close());
  const roots = new McpRootsService({ directory: join(root, "private-roots"), connection: (brainId, serverId) => client.hostConnection(brainId, serverId),
    requireConnection: connection => client.requireHostConnection(connection), filePolicy: async () => ({ level: "ask", revision: "current-file-policy" }), approve: async () => true,
    admit: async () => undefined, checkDisk: async () => undefined, notify: connection => client.notifyRootsChanged(connection) }); closes.push(() => roots.close());
  const list = vi.fn((origin: Parameters<McpRootsService["list"]>[0], signal: AbortSignal) => roots.list(origin, signal));
  if (installed) client.setStandardHostServices({ roots: list, log: async (origin, params, signal) => { await diagnostic.receive(origin, params, signal); } });
  await client.initialize();
  const server = await client.add({ brainId: "first", id: "fixture", label: "Fixture", transport: "http", url: "http://127.0.0.1:48123/mcp", bearerToken: "fixture-bearer-secret" });
  const selected = join(root, "selected-project"); await mkdir(selected); if (installed) await roots.select("first", "fixture", [selected], new AbortController().signal);
  const tools = new ToolExecutor(brain as never, {} as never, undefined, undefined, client);
  return { brain, client, roots, diagnostic, server, calls, tools, list, notifications, invalidated, secrets };
}

it("routes actual server roots requests and operational logs, and forwards only new list-change events without invoking tools", async () => {
  const f = await fixture(), tool = f.server.tools.find(value => !value.clientOperation)!;
  const result = await f.client.execute(tool.id, "call", {}, new AbortController().signal, "first");
  expect(result).toMatchObject({ content: [{ text: "New host methods completed" }] });
  expect(f.calls.find(value => value.method === "initialize")?.params.capabilities.roots).toEqual({ listChanged: true });
  expect(f.calls.find(value => value.id === "server-roots" && !value.method)?.result.roots).toHaveLength(1);
  expect(f.list).toHaveBeenCalledOnce(); expect(f.diagnostic.page("first", "fixture").records[0]?.data).toBe("[redacted] diagnostic");
  expect(f.notifications.mock.calls.map(value => value[1])).toEqual(["notifications/tools/list_changed", "notifications/resources/list_changed"]);
  expect(f.notifications.mock.calls[0]?.[0]).toMatchObject({ brainId: "first", serverId: "fixture", endpoint: "http://127.0.0.1:48123/mcp", connectionEpoch: 1 });
  expect(f.calls.filter(value => value.method === "tools/call")).toHaveLength(1);
  expect(f.calls.filter(value => value.method === "tools/list")).toHaveLength(1); // notification itself never refreshes
});

it("exposes logging/setLevel only as a negotiated reserved operation under current ToolExecutor permissions", async () => {
  const f = await fixture(), tool = f.server.tools.find(value => value.clientOperation === "logging/setLevel")!;
  expect(tool.id).toContain(".client."); expect(tool.inputSchema).toMatchObject({ properties: { level: { enum: ["debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"] } } });
  await f.brain.setToolPermission("first", tool.id, "off");
  expect(await f.tools.execute({ brainId: "first", toolId: tool.id, action: "call", arguments: { level: "warning" } })).toMatchObject({ state: "failed" });
  expect(f.calls.some(value => value.method === "logging/setLevel")).toBe(false);
  await f.brain.setToolPermission("first", tool.id, "full");
  const result = await f.tools.execute({ brainId: "first", toolId: tool.id, action: "call", arguments: { level: "warning" } });
  expect(result.error).toBeUndefined(); expect(result).toMatchObject({ state: "complete", output: { operation: "logging/setLevel", result: {} } });
  await expect(f.client.execute(tool.id, "call", { level: "trace" }, new AbortController().signal, "first")).rejects.toThrow("arguments");
  expect(f.calls.filter(value => value.method === "logging/setLevel")).toHaveLength(1);
});

it("does not advertise absent roots, and rejects server-supplied roots filters rather than selecting their paths", async () => {
  const absent = await fixture(false), tool = absent.server.tools.find(value => !value.clientOperation)!;
  await absent.client.execute(tool.id, "call", {}, new AbortController().signal, "first");
  expect(absent.calls.find(value => value.method === "initialize")?.params.capabilities.roots).toBeUndefined();
  expect(absent.calls.find(value => value.id === "server-roots" && !value.method)?.error.code).toBe(-32601);
  // A separate device root avoids sharing registration state with the first fixture.
  const firstRoot = root; root = join(firstRoot, "second-device"); await mkdir(root);
  try {
    const invalid = await fixture(true, { path: "/server-proposed-private-directory" }), selected = invalid.server.tools.find(value => !value.clientOperation)!;
    await invalid.client.execute(selected.id, "call", {}, new AbortController().signal, "first");
    expect(invalid.calls.find(value => value.id === "server-roots" && !value.method)?.error.code).toBe(-32602); expect(invalid.list).not.toHaveBeenCalled();
  } finally { root = firstRoot; }
});

it("retires disclosure identity for credential/account replacement and rejects stale secret-await completion without reconnect", async () => {
  const f = await fixture(), initial = f.client.hostConnection("first", "fixture"), before = f.calls.length;
  await f.client.updateCredentials("first", "fixture", { bearerToken: "replacement-fixture-secret" });
  const replaced = f.client.hostConnection("first", "fixture"); expect(replaced.connectionId).not.toBe(initial.connectionId); expect(replaced.ready).toBe(false);
  expect(await f.client.notifyRootsChanged(initial)).toBe(false); expect(await f.roots.status("first", "fixture")).toMatchObject({ state: "review-required" });
  expect(f.calls.slice(before).some(value => value.method === "initialize" || value.method === "tools/call")).toBe(false);
  await f.client.refresh("first", "fixture"); await f.roots.review("first", "fixture", new AbortController().signal);
  // Calls the real installed status callback with synthetic account publication;
  // this does not contact an OAuth provider or prove a real login.
  const oauth = Reflect.get(f.client, "oauth") as { options: { onChange(id: string, status: McpOAuthStatus): void } };
  const current = f.client.hostConnection("first", "fixture"); oauth.options.onChange("fixture", { state: "authorized" });
  expect(f.client.hostConnection("first", "fixture").connectionId).toBe(current.connectionId); // ordinary token refresh
  oauth.options.onChange("fixture", { state: "authorizing", flowId: "explicit-synthetic-account-flow" }); oauth.options.onChange("fixture", { state: "authorized" });
  expect(f.client.hostConnection("first", "fixture")).toMatchObject({ ready: false }); expect(f.client.hostConnection("first", "fixture").connectionId).not.toBe(current.connectionId);
  const origin = f.client.hostConnection("first", "fixture"); let release!: (value: string) => void;
  const original = f.secrets.get.bind(f.secrets);
  vi.spyOn(f.secrets, "get").mockImplementation(id => id === "mcp:registry:fixture" ? new Promise<string>(resolve => { release = resolve; }) : original(id));
  const pending = f.client.knownOperationalSecrets(origin).catch(error => error); await vi.waitFor(() => expect(release).toBeTypeOf("function"));
  f.client.suspendBrain("first"); release("must-never-be-published"); expect(await pending).toBeInstanceOf(Error);
});
