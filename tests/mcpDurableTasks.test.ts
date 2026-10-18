/** Real host client/store with synthetic HTTP/SSE; no provider/model/native learning. */
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import { McpClientService } from "../src/main/mcpClient";
import { McpTaskStore } from "../src/main/mcpTaskStore";
import { McpTaskCustodyError } from "../src/main/mcpTaskFailure";
import { SecureSecretStore } from "../src/main/secureSecretStore";
import type { McpRemoteTask, McpTaskOutcome, McpTaskReceipt } from "../src/shared/mcpTasks";

type Rpc = { jsonrpc: "2.0"; id?: string; method?: string; params?: Record<string, any>; result?: unknown; error?: unknown };
const roots: string[] = [], clients: McpClientService[] = [];
afterEach(async () => { for (const client of clients.splice(0)) { client.dispose(); await client.settleTaskWrites(); } for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
const task = (status: McpRemoteTask["status"] = "working"): McpRemoteTask => ({ taskId: "server-generated-task-one", status, createdAt: "2026-10-17T00:00:00.000Z", lastUpdatedAt: "2026-10-17T00:00:01.000Z", ttl: null, pollInterval: 0 });
const rawResult = () => ({ content: [{ type: "text", text: "Actual synthetic remote outcome" }], structuredContent: { from: "fixture" }, _meta: { "io.modelcontextprotocol/related-task": { taskId: task().taskId }, "omni/mcpApp": { forged: true } } });
async function fixture(options: { tasks?: boolean; protocolVersion?: string } = {}) {
  const root = await mkdtemp(join(tmpdir(), "omni-mcp-durable-task-")); roots.push(root); await mkdir(join(root, "brain"));
  const calls: Rpc[] = [], schemas = vi.fn(), learn = vi.fn(async () => ({})), permissions: Array<{ toolId: string; level: string }> = [];
  const brain = { registerExternalToolSchemas: schemas, unregisterExternalToolSchemas: vi.fn(), learnStructuredExperience: learn,
    listToolPermissions: async () => permissions, setToolPermission: async (_id: string, toolId: string, level: string) => { permissions.push({ toolId, level }); return permissions; } };
  const capabilities = { tools: {}, ...(options.tasks === false ? {} : { tasks: { requests: { tools: { call: {} } }, list: {}, cancel: {} } }) };
  const tools = ["required", "optional", "forbidden"].map(support => ({ name: support, execution: { taskSupport: support }, inputSchema: { type: "object", $defs: { text: { type: "string" } }, properties: { input: { $ref: "#/$defs/text" } }, required: ["input"], additionalProperties: false } }));
  let handler: (rpc: Rpc, init?: RequestInit) => unknown | Promise<unknown> = rpc => rpc.method === "tools/call" ? { task: task() }
    : rpc.method === "tasks/get" ? task("completed") : rpc.method === "tasks/result" ? rawResult() : rpc.method === "tasks/cancel" ? task("cancelled")
      : rpc.method === "tasks/list" ? { tasks: [task(), { ...task(), taskId: "foreign-account-task" }], nextCursor: "opaque-second-page" } : {};
  const fetcher = vi.fn(async (_url: unknown, init?: RequestInit): Promise<Response> => {
    if (init?.method === "DELETE") return new Response(null, { status: 204 });
    const rpc = JSON.parse(String(init?.body)) as Rpc; calls.push(rpc);
    if (!rpc.method || rpc.method.startsWith("notifications/")) { await handler(rpc, init); return new Response(null, { status: 202 }); }
    const result = rpc.method === "initialize" ? { protocolVersion: options.protocolVersion ?? "2025-11-25", capabilities } : rpc.method === "tools/list" ? { tools } : await handler(rpc, init);
    if (result instanceof Response) return result;
    return new Response(JSON.stringify({ jsonrpc: "2.0", id: rpc.id, result }), { headers: { "content-type": "application/json", "mcp-session-id": "session-main-only-secret" } });
  });
  const secrets = new SecureSecretStore(join(root, "secrets.json"), { available: () => false, encrypt: value => Buffer.from(value), decrypt: value => value.toString() });
  const client = new McpClientService(join(root, "mcp.json"), secrets, brain as never, fetcher as typeof fetch), store = new McpTaskStore({ brainDirectory: async id => join(root, id), admit: async () => undefined, checkDisk: async () => undefined });
  clients.push(client); client.setTaskStore(store); await client.initialize();
  const summary = await client.add({ brainId: "brain", id: "remote", label: "Synthetic task server", transport: "http", url: "http://127.0.0.1:48999/mcp" });
  let permitted = true;
  const authority = { requestId: "current-request", actionEventId: "current-action", chatTurnId: "current-turn", requireCurrentPermission: async () => { if (!permitted) throw new Error("Fresh exact task action permission revoked"); } };
  const invoke = (name: string, action: string, args: Record<string, unknown>, signal = new AbortController().signal) => client.execute(summary.tools.find(tool => tool.remoteName === name)!.id, action, args, signal, "brain", authority);
  return { root, client, store, summary, calls, schemas, learn, authority, invoke, revoke() { permitted = false; }, setHandler(value: typeof handler) { handler = value; }, fetcher };
}

it("negotiates only dated outbound tasks and preserves each original tool id/grant/action schema with required/optional/forbidden semantics", async () => {
  const f = await fixture(), required = f.summary.tools.find(tool => tool.remoteName === "required")!, optional = f.summary.tools.find(tool => tool.remoteName === "optional")!, forbidden = f.summary.tools.find(tool => tool.remoteName === "forbidden")!;
  expect(required.actions).not.toContain("call"); expect(required.actions).toContain("call-task"); expect(optional.actions).toContain("call"); expect(forbidden.actions).toEqual(["call"]);
  const registered = f.schemas.mock.calls.flatMap(([value]) => value as any[]).find(value => value.id === required.id); expect(registered.actionInputSchemas["call-task"]).toBeDefined();
  expect(f.calls.find(call => call.method === "initialize")?.params?.capabilities.tasks).toEqual({});
  await expect(f.invoke("required", "call", { input: "x" })).rejects.toThrow(/requires task/); await expect(f.invoke("forbidden", "call-task", { arguments: { input: "x" } })).rejects.toThrow(/did not negotiate/);
  const receipt = await f.invoke("optional", "call-task", { arguments: { input: "Exact remote argument" }, ttl: 60000 }) as McpTaskReceipt;
  expect(receipt).toMatchObject({ operationComplete: false, record: { creation: "accepted", remote: task(), authority: "historical-not-live" } });
  expect(f.calls.find(call => call.method === "tools/call")?.params).toEqual({ name: "optional", arguments: { input: "Exact remote argument" }, task: { ttl: 60000 } });
});
it("does not fake required-task support when the receiver capability or actual dated protocol is absent", async () => {
  const noTasks = await fixture({ tasks: false }); await expect(noTasks.invoke("required", "call", { input: "x" })).rejects.toThrow(/requires task/);
  expect(noTasks.calls.some(call => call.method === "tools/call")).toBe(false);
  const old = await fixture({ protocolVersion: "2025-06-18" }); await expect(old.invoke("optional", "call-task", { arguments: { input: "x" } })).rejects.toThrow(/did not negotiate/);
  expect(old.calls.some(call => call.method === "tools/call")).toBe(false);
});
it("does not present unsolicited dated or draft task ACKs as completed synchronous results or adopt their task authority", async () => {
  const f = await fixture(), capture = vi.fn(); f.client.setAppHost({ capture, requireCall() {}, invalidate() {} } as never);
  await expect(f.invoke("optional", "call", { input: "Ordinary synchronous request" })).rejects.toThrow(/unsolicited task/);
  f.setHandler(rpc => rpc.method === "tools/call" ? { resultType: "task", task: { ...task(), status: "completed" }, result: rawResult() } : {});
  await expect(f.invoke("forbidden", "call", { input: "Another ordinary request" })).rejects.toThrow(/unsolicited task/);
  expect((await f.client.tasksLocalPage("brain")).records).toEqual([]); expect(capture).not.toHaveBeenCalled();
  expect(f.calls.filter(call => call.method === "tools/call")).toHaveLength(2);
  expect(f.calls.some(call => ["tasks/get", "tasks/result", "tasks/cancel"].includes(call.method ?? ""))).toBe(false);
});
it("commits one actual final payload, normalizes App presentation separately, and returns one stable retained receipt without refetching execution", async () => {
  const f = await fixture(), capture = vi.fn(async (input: any) => { expect(input.arguments).toEqual({ input: "Original args" }); expect(input.result._meta["omni/mcpApp"]).toBeUndefined(); return { capturedByHost: true }; });
  f.client.setAppHost({ capture, requireCall() {}, invalidate() {} } as never);
  const start = await f.invoke("optional", "call-task", { arguments: { input: "Original args" } }) as McpTaskReceipt;
  const args = { taskRef: start.record.taskRef }; await f.invoke("optional", "task-status", args);
  const first = await f.invoke("optional", "task-result", args) as McpTaskOutcome, second = await f.invoke("optional", "task-result", args) as McpTaskOutcome;
  expect(first.result).toEqual(rawResult()); expect(first.observedResult?._meta).toMatchObject({ "omni/mcpApp": { capturedByHost: true } });
  expect(second).toMatchObject({ retained: true, taskResultReceiptId: first.taskResultReceiptId, result: rawResult() });
  expect(capture).toHaveBeenCalledOnce(); expect(f.calls.filter(call => call.method === "tools/call")).toHaveLength(1); expect(f.calls.filter(call => call.method === "tasks/result")).toHaveLength(1);
});
it("retains a lost creation ACK as unknown without reinitializing or resending the original tool call", async () => {
  const f = await fixture(); f.setHandler(rpc => { if (rpc.method === "tools/call") throw new TypeError("Synthetic lost ACK"); return {}; });
  await expect(f.invoke("optional", "call-task", { arguments: { input: "x" } })).rejects.toThrow(/unknown/);
  const page = await f.client.tasksLocalPage("brain"); expect(page.records[0]).toMatchObject({ management: "dormant", record: { creation: "creation-outcome-unknown" } });
  expect(f.calls.filter(call => call.method === "tools/call")).toHaveLength(1); expect(f.calls.filter(call => call.method === "initialize")).toHaveLength(1);
});
it("publishes an actual lost-ACK custody failure as a typed copy of its committed store row, not inferred error text", async () => {
  const f = await fixture(); f.setHandler(rpc => { if (rpc.method === "tools/call") throw new TypeError("Synthetic transport interruption"); return {}; });
  let failure: unknown; try { await f.invoke("optional", "call-task", { arguments: { input: "Typed custody boundary" } }); } catch (error) { failure = error; }
  expect(failure).toBeInstanceOf(McpTaskCustodyError); const receipt = (failure as McpTaskCustodyError).receipt;
  expect(receipt).toMatchObject({ format: "omni-mcp-task", operationComplete: false, management: "dormant", record: { creation: "creation-outcome-unknown" } });
  expect(receipt.record).toEqual(await f.store.get("brain", receipt.record.taskRef)); expect(receipt.record.remote).toBeUndefined();
});
it("keeps a genuinely received creation ACK but retires authority if current permission is revoked before delivery", async () => {
  const f = await fixture(); f.setHandler(rpc => { if (rpc.method === "tools/call") { f.revoke(); return { task: task() }; } return {}; });
  await expect(f.invoke("optional", "call-task", { arguments: { input: "x" } })).rejects.toThrow(/actual accepted task is retained/);
  const page = await f.client.tasksLocalPage("brain"); expect(page.records[0]).toMatchObject({ management: "dormant", record: { creation: "accepted", remote: task() } });
  expect(f.calls.filter(call => call.method === "tools/call")).toHaveLength(1);
});
it("never adopts foreign task ids or rebinds a historical task to a replacement account/session", async () => {
  const f = await fixture(), start = await f.invoke("optional", "call-task", { arguments: { input: "x" } }) as McpTaskReceipt;
  await expect(f.invoke("required", "task-status", { taskRef: start.record.taskRef })).rejects.toThrow(/current source\/account/);
  await f.client.updateCredentials("brain", "remote", { bearerToken: "replacement-task-fixture-key" });
  const before = f.calls.length; await expect(f.invoke("optional", "task-status", { taskRef: start.record.taskRef })).rejects.toThrow(/did not negotiate/);
  expect(f.calls).toHaveLength(before); expect((await f.client.tasksLocalPage("brain")).records[0]?.management).toBe("dormant");
});
it("binds task-related elicitation to the exact active task/result operation and echoes task metadata after independent host review", async () => {
  const f = await fixture(), elicit = vi.fn(async (_origin: unknown, _params: unknown, _signal: AbortSignal) => ({ action: "accept" as const, content: { answer: "reviewed" } })); f.client.setHostInteractions({ elicit });
  const start = await f.invoke("optional", "call-task", { arguments: { input: "x" } }) as McpTaskReceipt;
  f.setHandler((rpc, init) => {
    if (rpc.method === "tasks/result") return new Response(new ReadableStream<Uint8Array>({ start(controller) {
      const bytes = (value: unknown) => new TextEncoder().encode(`data: ${JSON.stringify(value)}\n\n`);
      controller.enqueue(bytes({ jsonrpc: "2.0", id: "server-input", method: "elicitation/create", params: { mode: "form", message: "Review this fixture", requestedSchema: { type: "object", properties: { answer: { type: "string" } } }, _meta: { "io.modelcontextprotocol/related-task": { taskId: task().taskId } } } }));
      const listener = setInterval(() => { if (f.calls.some(call => call.id === "server-input" && call.result)) { clearInterval(listener); controller.enqueue(bytes({ jsonrpc: "2.0", id: rpc.id, result: rawResult() })); controller.close(); } }, 5);
      init?.signal?.addEventListener("abort", () => { clearInterval(listener); controller.error(init.signal?.reason); }, { once: true });
    } }), { headers: { "content-type": "text/event-stream" } }); return {};
  });
  const result = await f.invoke("optional", "task-result", { taskRef: start.record.taskRef }) as McpTaskOutcome;
  expect(result.result).toEqual(rawResult()); expect(elicit).toHaveBeenCalledOnce(); expect(elicit.mock.calls[0]?.[0]).toMatchObject({ brainId: "brain", serverId: "remote", actionEventId: "current-action", originatingTurnId: "current-turn" });
  expect(f.calls.find(call => call.id === "server-input" && call.result)?.result).toMatchObject({ _meta: { "io.modelcontextprotocol/related-task": { taskId: task().taskId } } });
});
it("does not turn expired/missing result lookup errors into a failed original tool result", async () => {
  const f = await fixture(), start = await f.invoke("optional", "call-task", { arguments: { input: "x" } }) as McpTaskReceipt;
  f.setHandler(rpc => new Response(JSON.stringify({ jsonrpc: "2.0", id: rpc.id, error: { code: -32602, message: "Task expired" } }), { headers: { "content-type": "application/json" } }));
  await expect(f.invoke("optional", "task-result", { taskRef: start.record.taskRef })).rejects.toThrow(/expired/);
  expect((await f.store.get("brain", start.record.taskRef)).result).toBeUndefined(); expect(f.calls.filter(call => call.method === "tools/call")).toHaveLength(1);
});
it("returns account-scoped task listing as data without adopting foreign IDs, and validates terminal cancellation without deletion or undo claims", async () => {
  const f = await fixture(), list = f.summary.tools.find(tool => tool.clientOperation === "tasks/list")!;
  const page = await f.client.execute(list.id, "call", {}, new AbortController().signal, "brain", f.authority) as any;
  expect(page.result.tasks).toHaveLength(2); expect((await f.client.tasksLocalPage("brain")).records).toHaveLength(0);
  const start = await f.invoke("optional", "call-task", { arguments: { input: "x" } }) as McpTaskReceipt, args = { taskRef: start.record.taskRef };
  expect(await f.invoke("optional", "task-cancel", args)).toMatchObject({ record: { remote: { status: "cancelled" } } });
  await expect(f.invoke("optional", "task-cancel", args)).rejects.toThrow(/already terminal/); expect(f.calls.filter(call => call.method === "tasks/cancel")).toHaveLength(1);
});
it("does not resend the same accepted, unknown or parallel logical creation intent", async () => {
  const f = await fixture(), start = await f.invoke("optional", "call-task", { arguments: { input: "x" } }) as McpTaskReceipt;
  expect(await f.invoke("optional", "call-task", { arguments: { input: "x" } })).toMatchObject({ record: { taskRef: start.record.taskRef } });
  expect(f.calls.filter(call => call.method === "tools/call")).toHaveLength(1);
  const unknown = await fixture(); unknown.setHandler(rpc => { if (rpc.method === "tools/call") throw new Error("Synthetic lost acknowledgement"); return {}; });
  await expect(unknown.invoke("optional", "call-task", { arguments: { input: "x" } })).rejects.toThrow(/unknown/);
  expect(await unknown.invoke("optional", "call-task", { arguments: { input: "x" } })).toMatchObject({ management: "dormant", record: { creation: "creation-outcome-unknown" } });
  expect(unknown.calls.filter(call => call.method === "tools/call")).toHaveLength(1);
  const parallel = await fixture(); const outcomes = await Promise.allSettled([parallel.invoke("optional", "call-task", { arguments: { input: "x" } }), parallel.invoke("optional", "call-task", { arguments: { input: "x" } })]);
  expect(outcomes.some(outcome => outcome.status === "fulfilled")).toBe(true); expect(parallel.calls.filter(call => call.method === "tools/call")).toHaveLength(1);
});
it("rejects foreign related-task input instead of assigning it to the only active RPC", async () => {
  const f = await fixture(), elicit = vi.fn(async () => ({ action: "accept" as const })); f.client.setHostInteractions({ elicit });
  const start = await f.invoke("optional", "call-task", { arguments: { input: "x" } }) as McpTaskReceipt;
  f.setHandler(rpc => rpc.method === "tasks/result" ? new Response([
    { jsonrpc: "2.0", id: "foreign-input", method: "elicitation/create", params: { mode: "form", message: "Foreign request", requestedSchema: { type: "object", properties: {} }, _meta: { "io.modelcontextprotocol/related-task": { taskId: "not-this-task" } } } },
    { jsonrpc: "2.0", id: rpc.id, result: rawResult() }
  ].map(message => `data: ${JSON.stringify(message)}\n\n`).join(""), { headers: { "content-type": "text/event-stream" } }) : {});
  await f.invoke("optional", "task-result", { taskRef: start.record.taskRef }); expect(elicit).not.toHaveBeenCalled();
  expect(f.calls.find(call => call.id === "foreign-input" && call.error)?.error).toMatchObject({ code: -32000 });
});
it("Stop and retirement drain durable unknown custody without sending a remote task cancellation or claiming undo", async () => {
  const f = await fixture(), start = await f.invoke("optional", "call-task", { arguments: { input: "x" } }) as McpTaskReceipt;
  let began!: () => void; const started = new Promise<void>(resolve => { began = resolve; });
  f.setHandler((rpc, init) => { if (rpc.method === "tasks/result") return new Promise<Response>((_resolve, reject) => { began(); const abort = () => reject(init?.signal?.reason); init?.signal?.addEventListener("abort", abort, { once: true }); if (init?.signal?.aborted) abort(); }); return {}; });
  const pending = f.invoke("optional", "task-result", { taskRef: start.record.taskRef }), rejected = expect(pending).rejects.toThrow(/retired|retire|previous state/); await started;
  await f.client.retireTaskAuthority("brain"); await rejected; await f.client.settleTaskWrites("brain");
  const row = (await f.client.tasksLocalPage("brain")).records[0]!;
  expect(row.management).toBe("dormant"); expect(row.record.lastOperation?.state).toBe("outcome-unknown"); expect(row.record.remote?.status).toBe("working"); expect(f.calls.some(call => call.method === "tasks/cancel")).toBe(false);
});
it("stops one local task wait without retiring its current lease or sending cancellation, then permits a fresh status read", async () => {
  const f = await fixture(), start = await f.invoke("optional", "call-task", { arguments: { input: "x" } }) as McpTaskReceipt;
  let began!: () => void; const started = new Promise<void>(resolve => { began = resolve; });
  f.setHandler((rpc, init) => {
    if (rpc.method === "tasks/result") return new Promise<Response>((_resolve, reject) => { began(); init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true }); });
    if (rpc.method === "tasks/get") return task("completed"); return {};
  });
  const pending = f.invoke("optional", "task-result", { taskRef: start.record.taskRef }), rejected = expect(pending).rejects.toThrow(/local MCP task wait stopped/); await started;
  expect(f.client.stopTaskWait("other", start.record.taskRef)).toBe(false); expect(f.client.stopTaskWait("brain", start.record.taskRef)).toBe(true); await rejected;
  await f.client.settleTaskWrites("brain"); expect((await f.client.tasksLocalPage("brain")).records[0]?.management).toBe("current-session");
  expect(await f.invoke("optional", "task-status", { taskRef: start.record.taskRef })).toMatchObject({ record: { remote: { status: "completed" } } });
  expect(f.calls.some(call => call.method === "tasks/cancel" || call.method === "notifications/cancelled")).toBe(false);
});
it("holds pre-ACK related input until the actual creation ACK proves its receiver-generated task id", async () => {
  const f = await fixture(), elicit = vi.fn(async () => ({ action: "accept" as const })); f.client.setHostInteractions({ elicit });
  f.setHandler(rpc => rpc.method === "tools/call" ? new Response([
    { jsonrpc: "2.0", id: "early-input", method: "elicitation/create", params: { mode: "form", message: "Early input", requestedSchema: { type: "object", properties: {} }, _meta: { "io.modelcontextprotocol/related-task": { taskId: task().taskId } } } },
    { jsonrpc: "2.0", id: rpc.id, result: { task: task() } }
  ].map(message => `data: ${JSON.stringify(message)}\n\n`).join(""), { headers: { "content-type": "text/event-stream" } }) : {});
  const start = await f.invoke("optional", "call-task", { arguments: { input: "x" } }) as McpTaskReceipt;
  expect(start.record.creation).toBe("accepted"); expect(elicit).toHaveBeenCalledOnce(); expect(f.calls.find(call => call.id === "early-input" && call.result)?.result).toMatchObject({ _meta: { "io.modelcontextprotocol/related-task": { taskId: task().taskId } } });
});
it("retains a real underlying RPC failure only after its actual failed task state is confirmed", async () => {
  const f = await fixture(), start = await f.invoke("optional", "call-task", { arguments: { input: "x" } }) as McpTaskReceipt;
  f.setHandler(rpc => rpc.method === "tasks/get" ? task("failed") : new Response(JSON.stringify({ jsonrpc: "2.0", id: rpc.id, error: { code: -32001, message: "Actual synthetic job failure", data: { fixture: true } } }), { headers: { "content-type": "application/json" } }));
  await f.invoke("optional", "task-status", { taskRef: start.record.taskRef });
  const outcome = await f.invoke("optional", "task-result", { taskRef: start.record.taskRef }) as McpTaskOutcome;
  expect(outcome).toMatchObject({ kind: "rpc-error", error: { code: -32001, message: "Actual synthetic job failure", data: { fixture: true } } }); expect((await f.store.get("brain", start.record.taskRef)).result?.receiptId).toBe(outcome.taskResultReceiptId);
});
