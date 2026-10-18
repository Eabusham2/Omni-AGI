import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { McpClientService } from "../src/main/mcpClient";
import { McpInteractionService } from "../src/main/mcpInteractions";
import { SecureSecretStore } from "../src/main/secureSecretStore";
import { ToolExecutor } from "../src/main/toolExecutor";
import type { ApprovedNativeMcpSamplingInput, NativeMcpSamplingResult } from "../src/shared/nativeMcpSampling";
import type { ToolPermissionRecord } from "../src/shared/types";

describe("MCP server-initiated reviewed requests", () => {
  let root: string;
  const cleanup: Array<() => void> = [];
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-mcp-server-requests-")); await mkdir(join(root, "first")); });
  afterEach(async () => { cleanup.splice(0).forEach(close => close()); await rm(root, { recursive: true, force: true }); });
  async function fixture(method: "elicitation/create" | "sampling/createMessage", urlRequired = false) {
    const permissions: ToolPermissionRecord[] = [];
    const brain = { repository: { brainDirectory: (id: string) => join(root, id), get: vi.fn(async () => ({ journal: [], traces: [] })), save: vi.fn(async value => value) },
      authorizeInlineImagination: vi.fn(async () => undefined), queueCompletedToolExecution: vi.fn(async () => undefined),
      registerExternalToolSchemas: vi.fn(), unregisterExternalToolSchemas: vi.fn(), learnStructuredExperience: vi.fn(async () => ({})),
      listToolPermissions: vi.fn(async () => permissions), setToolPermission: vi.fn(async (_id: string, toolId: string, level: ToolPermissionRecord["level"]) => {
        const prior = permissions.find(value => value.toolId === toolId); if (prior) prior.level = level;
        else permissions.push({ toolId, label: toolId, level, updatedAt: new Date().toISOString() }); return permissions;
      }) };
    let stream!: ReadableStreamDefaultController<Uint8Array>;
    let outerId = "";
    const calls: Array<Record<string, unknown>> = [];
    const send = (value: unknown) => stream.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(value)}\n\n`));
    const fetcher = vi.fn(async (_url: unknown, init?: RequestInit): Promise<Response> => {
      const request = JSON.parse(String(init?.body)); calls.push(request);
      if (request.method === "notifications/initialized" || request.method === "notifications/cancelled") return new Response(null, { status: 202 });
      if (request.id === "server-review" && !request.method) {
        send({ jsonrpc: "2.0", id: outerId, result: { content: [{ type: "text", text: "Server received the explicit review" }] } });
        stream.close(); return new Response(null, { status: 202 });
      }
      if (request.method === "tools/call") {
        if (urlRequired) return Response.json({ jsonrpc: "2.0", id: request.id, error: { code: -32042, message: "Public fixture connection needed", data: { elicitations: [
          { mode: "url", elicitationId: "fixture-required-url", message: "Review this full destination", url: "https://example.invalid/authorize?scope=public" }
        ] } } });
        outerId = request.id;
        const params = method === "elicitation/create" ? { mode: "form", message: "Choose a public nickname", requestedSchema: { type: "object", properties: { nickname: { type: "string" } }, required: ["nickname"] } }
          : { messages: [{ role: "user", content: { type: "text", text: "Approved task from the actual server" } }], maxTokens: 4, systemPrompt: "Unapproved persona", originatingTurnId: "server-forged-turn" };
        return new Response(new ReadableStream<Uint8Array>({ start(controller) { stream = controller;
          send({ jsonrpc: "2.0", id: "server-review", method, params });
        } }), { headers: { "content-type": "text/event-stream" } });
      }
      const result = request.method === "initialize" ? { protocolVersion: "2025-11-25", capabilities: { tools: {} } }
        : { tools: [{ name: "request_review", inputSchema: { type: "object" } }] };
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: request.id, result }), { headers: { "content-type": "application/json", "mcp-session-id": "fixture-live-session" } });
    });
    const secrets = new SecureSecretStore(join(root, "secrets.json"), { available: () => false, encrypt: value => Buffer.from(value), decrypt: value => value.toString() });
    const client = new McpClientService(join(root, "mcp.json"), secrets, brain as never, fetcher as typeof fetch); cleanup.push(() => client.dispose());
    const sample = vi.fn(async (input: ApprovedNativeMcpSamplingInput): Promise<NativeMcpSamplingResult> => ({ role: "assistant", content: { type: "text", text: "Same-brain private output" }, model: `OmniCortex/${input.brainId}`,
      stopReason: "endTurn", brainId: input.brainId, requestId: input.requestId, inputSha256: input.inputSha256, privateWorkspace: true, parametersMutated: false }));
    const reviews = new McpInteractionService({ directory: join(root, "private-reviews"), requireAccess: origin => client.requireInteractionOrigin(origin), admit: async () => undefined,
      openBrowser: async () => undefined, sampleNative: sample }); cleanup.push(() => reviews.close()); await reviews.initialize();
    client.setHostInteractions({ elicit: (origin, params, signal) => reviews.elicit(origin, params, signal), sample: (origin, params, signal) => reviews.sample(origin, params, signal) });
    await client.initialize(); const server = await client.add({ brainId: "first", id: "fixture", label: "Fixture", transport: "http", url: "http://127.0.0.1:8196/mcp" });
    await brain.setToolPermission("first", server.tools[0]!.id, "full");
    const tools = new ToolExecutor(brain as never, {} as never, undefined, undefined, client);
    return { client, reviews, tools, server, calls, sample, send, finish: () => { send({ jsonrpc: "2.0", id: outerId, result: { content: [] } }); stream.close(); } };
  }

  it("keeps the response reader live while a real form waits and only responds after reviewed submission", async () => {
    const f = await fixture("elicitation/create");
    const pending = f.tools.execute({ brainId: "first", toolId: f.server.tools[0]!.id, action: "call", arguments: {} }, undefined, "actual-request", undefined,
      { actionEventId: "actual-action", chatTurnId: "actual-turn" });
    await vi.waitFor(async () => expect(await f.reviews.list("first")).toHaveLength(1));
    const review = (await f.reviews.list("first"))[0]!;
    expect(review.origin).toMatchObject({ originatingTurnId: "actual-turn", actionEventId: "actual-action", brainId: "first", serverId: "fixture" });
    expect(f.calls.some(call => call.id === "server-review" && !call.method)).toBe(false);
    await f.reviews.respond({ brainId: "first", id: review.id, action: "accept", values: { nickname: "Reviewed" } });
    expect((await pending).state).toBe("complete");
    expect(f.calls.find(call => call.id === "server-review" && !call.method)).toMatchObject({ result: { action: "accept", content: { nickname: "Reviewed" } } });
  });

  it("binds sampling to internal controller context, never schema arguments or server-supplied turn IDs", async () => {
    const f = await fixture("sampling/createMessage");
    const pending = f.tools.execute({ brainId: "first", toolId: f.server.tools[0]!.id, action: "call", arguments: { chatTurnId: "argument-forged-turn" } }, undefined, "actual-request", undefined,
      { actionEventId: "actual-action", chatTurnId: "actual-turn" });
    await vi.waitFor(async () => expect((await f.reviews.list("first"))[0]?.kind).toBe("sampling-input"));
    const input = (await f.reviews.list("first"))[0]!;
    await f.reviews.respond({ brainId: "first", id: input.id, action: "accept" });
    await vi.waitFor(async () => expect((await f.reviews.list("first"))[0]?.kind).toBe("sampling-output"));
    expect(f.sample.mock.calls[0]?.[0]).toMatchObject({ originatingTurnId: "actual-turn" });
    expect(JSON.stringify(f.sample.mock.calls)).not.toContain("forged-turn");
    expect(JSON.stringify(f.sample.mock.calls)).not.toContain("Unapproved persona");
    expect(f.calls.some(call => call.id === "server-review" && !call.method)).toBe(false);
    const output = (await f.reviews.list("first"))[0]!;
    await f.reviews.respond({ brainId: "first", id: output.id, action: "accept" });
    expect((await pending).state).toBe("complete");
    expect(f.calls.find(call => call.id === "server-review" && !call.method)).toMatchObject({ result: { model: "OmniCortex/first", content: { text: "Same-brain private output" } } });
    const initialized = f.calls.find(call => call.method === "initialize")!;
    expect(initialized.params).toMatchObject({ capabilities: { elicitation: { form: {}, url: {} }, sampling: {} } });
    expect(JSON.stringify(initialized.params)).not.toContain('"context"'); expect(JSON.stringify(initialized.params)).not.toContain('"sampling":{"tools"');
  });

  it("receives server cancellation during review without inventing an answer or stopping the stream reader", async () => {
    const f = await fixture("elicitation/create");
    const pending = f.client.execute(f.server.tools[0]!.id, "call", {}, new AbortController().signal, "first", { actionEventId: "actual-action", chatTurnId: "actual-turn" });
    await vi.waitFor(async () => expect(await f.reviews.list("first")).toHaveLength(1));
    f.send({ jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId: "server-review" } });
    await vi.waitFor(async () => expect(await f.reviews.list("first")).toEqual([]));
    expect(f.calls.some(call => call.id === "server-review" && !call.method)).toBe(false);
    f.finish(); await pending;
  });
  it("surfaces URL_ELICITATION_REQUIRED for reviewed navigation without retrying the original tool", async () => {
    const f = await fixture("elicitation/create", true);
    const pending = f.client.execute(f.server.tools[0]!.id, "call", {}, new AbortController().signal, "first", { actionEventId: "actual-action", chatTurnId: "actual-turn" }).catch(error => error);
    await vi.waitFor(async () => expect((await f.reviews.list("first"))[0]?.kind).toBe("url"));
    const review = (await f.reviews.list("first"))[0]!;
    expect(review).toMatchObject({ origin: { originatingTurnId: "actual-turn" }, request: { url: "https://example.invalid/authorize?scope=public" } });
    await f.reviews.respond({ brainId: "first", id: review.id, action: "accept" });
    expect((await pending as Error).message).toContain("original action was not retried");
    expect(f.calls.filter(call => call.method === "tools/call")).toHaveLength(1);
  });
});
