import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatActionController } from "../src/main/chatActionController";
import { McpClientService } from "../src/main/mcpClient";
import { SecureSecretStore, type SecretProtector } from "../src/main/secureSecretStore";
import { ApiTeacherTrainingService } from "../src/main/teacherTraining";
import { ToolExecutor } from "../src/main/toolExecutor";
import type { RuntimeJobEvent, ToolPermissionRecord } from "../src/shared/types";

const protector: SecretProtector = {
  available: () => true,
  encrypt: (value) => Buffer.from([...value].reverse().join(""), "utf8"),
  decrypt: (value) => [...value.toString("utf8")].reverse().join("")
};

describe("API teacher and MCP neural integrations", () => {
  let root: string;
  let secrets: SecureSecretStore;

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "omni-integrations-"));
    secrets = new SecureSecretStore(join(root, "secrets.json"), protector);
  });

  afterEach(async () => {
    await rm(root, { recursive: true, force: true });
  });

  it("serializes concurrent encrypted credential updates without losing records", async () => {
    await Promise.all([
      secrets.set("teacher:openai", "openai-key"),
      secrets.set("teacher:anthropic", "anthropic-key"),
      secrets.set("mcp:brain-1:coding", "mcp-token")
    ]);
    await expect(secrets.get("teacher:openai")).resolves.toBe("openai-key");
    await expect(secrets.get("teacher:anthropic")).resolves.toBe("anthropic-key");
    await expect(secrets.get("mcp:brain-1:coding")).resolves.toBe("mcp-token");
    const persisted = await readFile(join(root, "secrets.json"), "utf8");
    expect(persisted).not.toContain("openai-key");
    expect(persisted).not.toContain("anthropic-key");
    expect(persisted).not.toContain("mcp-token");
  });

  it("encrypts API credentials outside the brain and learns OpenAI output without a system prompt", async () => {
    const learned: string[] = [];
    const brain = {
      preflightStart: vi.fn(async () => undefined),
      learnStructuredExperience: vi.fn(async (_brainId: string, value: { content: string }) => {
        learned.push(value.content);
        return {};
      })
    };
    const fetcher = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      expect(body).toMatchObject({ model: "gpt-test", input: "teach this", store: false });
      expect(body).not.toHaveProperty("system");
      expect(new Headers(init?.headers).get("authorization")).toBe("Bearer sk-private-teacher-key");
      return new Response(JSON.stringify({ output_text: "a learned answer" }), {
        status: 200,
        headers: { "content-type": "application/json" }
      });
    });
    const service = new ApiTeacherTrainingService(
      brain as never,
      secrets,
      fetcher as typeof fetch
    );
    const statuses = await service.saveCredential({
      provider: "openai",
      apiKey: "sk-private-teacher-key"
    });
    expect(statuses.find((value) => value.provider === "openai")).toMatchObject({
      configured: true,
      persistence: "encrypted",
      keyHint: "••••-key"
    });

    const terminal = new Promise<RuntimeJobEvent>((resolve) => {
      service.on("event", (event: RuntimeJobEvent) => {
        if (["complete", "failed", "cancelled"].includes(event.job.state)) resolve(event);
      });
    });
    service.start({
      brainId: "brain-1",
      provider: "openai",
      model: "gpt-test",
      prompts: ["teach this"],
      maxOutputTokens: 128
    });
    expect((await terminal).job.state).toBe("complete");
    expect(learned).toHaveLength(1);
    expect(learned[0]).toContain("a learned answer");
    expect(learned[0]).toContain('"hiddenBehavioralPrompt":false');
    expect(learned[0]).toContain('"rlhf":false');
    expect(learned[0]).not.toContain("sk-private-teacher-key");
    expect(await readFile(join(root, "secrets.json"), "utf8")).not.toContain(
      "sk-private-teacher-key"
    );
  });

  it.each([
    {
      provider: "anthropic" as const,
      model: "claude-test",
      response: { content: [{ type: "text", text: "Claude fixture answer" }] },
      header: "x-api-key",
      expected: "Claude fixture answer"
    },
    {
      provider: "gemini" as const,
      model: "gemini-test",
      response: { output_text: "Gemini fixture answer" },
      header: "x-goog-api-key",
      expected: "Gemini fixture answer"
    }
  ])("learns $provider responses through its current official request shape", async ({ provider, model, response, header, expected }) => {
    const learned: string[] = [];
    const fetcher = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      expect(body.model).toBe(model);
      expect(body).not.toHaveProperty("system");
      expect(new Headers(init?.headers).get(header)).toBe("provider-secret-key");
      if (provider === "anthropic") expect(body.messages).toEqual([{ role: "user", content: "provider prompt" }]);
      else expect(body).toMatchObject({ input: "provider prompt", store: false });
      return new Response(JSON.stringify(response), { status: 200 });
    });
    const service = new ApiTeacherTrainingService({
      preflightStart: vi.fn(async () => undefined),
      learnStructuredExperience: vi.fn(async (_brainId: string, value: { content: string }) => {
        learned.push(value.content);
        return {};
      })
    } as never, secrets, fetcher as typeof fetch);
    await service.saveCredential({ provider, apiKey: "provider-secret-key" });
    const terminal = new Promise<RuntimeJobEvent>((resolve) => {
      service.on("event", (event: RuntimeJobEvent) => {
        if (["complete", "failed", "cancelled"].includes(event.job.state)) resolve(event);
      });
    });
    service.start({ brainId: "brain-1", provider, model, prompts: ["provider prompt"] });
    expect((await terminal).job.state).toBe("complete");
    expect(learned[0]).toContain(expected);
    expect(learned[0]).not.toContain("provider-secret-key");
  });

  it("redacts a credential even when a remote provider echoes it in an error", async () => {
    const service = new ApiTeacherTrainingService({
      preflightStart: vi.fn(async () => undefined),
      learnStructuredExperience: vi.fn()
    } as never, secrets, vi.fn(async () => new Response(JSON.stringify({
      error: { message: "credential provider-secret-key was rejected" }
    }), { status: 401 })) as typeof fetch);
    await service.saveCredential({ provider: "openai", apiKey: "provider-secret-key" });
    const events: RuntimeJobEvent[] = [];
    const terminal = new Promise<RuntimeJobEvent>((resolve) => {
      service.on("event", (event: RuntimeJobEvent) => {
        events.push(event);
        if (["complete", "failed", "cancelled"].includes(event.job.state)) resolve(event);
      });
    });
    service.start({
      brainId: "brain-1",
      provider: "openai",
      model: "gpt-test",
      prompts: ["explicit question"]
    });
    const event = await terminal;
    expect(event.job.state).toBe("failed");
    expect(JSON.stringify(events)).toContain("[REDACTED]");
    expect(JSON.stringify(events)).not.toContain("provider-secret-key");
  });

  it("discovers MCP tools, learns typed schemas, and routes calls through the dynamic bridge", async () => {
    const permissions: ToolPermissionRecord[] = [];
    const registered: Array<{ id: string; actions: readonly string[] }> = [];
    const learned: string[] = [];
    const auditDocument = { journal: [], traces: [] };
    const repository = {
      get: vi.fn(async () => auditDocument),
      save: vi.fn(async (value: unknown) => value)
    };
    const brain = {
      repository,
      registerExternalToolSchemas: vi.fn((schemas: typeof registered) => registered.push(...schemas)),
      unregisterExternalToolSchemas: vi.fn(),
      learnStructuredExperience: vi.fn(async (_brainId: string, value: { content: string }) => {
        learned.push(value.content);
        return {};
      }),
      listToolPermissions: vi.fn(async () => permissions.map((value) => ({ ...value }))),
      setToolPermission: vi.fn(async (_brainId: string, toolId: string, level: ToolPermissionRecord["level"]) => {
        const existing = permissions.find((value) => value.toolId === toolId);
        if (existing) existing.level = level;
        else permissions.push({ toolId, label: toolId, level, updatedAt: new Date(0).toISOString() });
        return permissions;
      })
    };
    const calls: string[] = [];
    const remoteCalls: Array<Record<string, unknown>> = [];
    let malformedSchema = false;
    const fetcher = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const request = JSON.parse(String(init?.body)) as {
        id?: string;
        method: string;
        params?: Record<string, unknown>;
      };
      calls.push(request.method);
      if (request.method === "tools/call") remoteCalls.push(request.params ?? {});
      const headers = { "content-type": "application/json", "mcp-session-id": "session-1" };
      if (request.method === "notifications/initialized") return new Response(null, { status: 202, headers });
      const result = request.method === "initialize"
        ? { protocolVersion: "2025-06-18", capabilities: {}, serverInfo: { name: "fixture", version: "1" } }
        : request.method === "tools/list"
          ? { tools: [{ name: "read_project", inputSchema: malformedSchema ? {} : {
            type: "object",
            properties: { path: { $ref: "#/$defs/projectPath" } },
            required: ["path"],
            additionalProperties: false,
            $defs: { projectPath: { type: "string" } }
          } }] }
          : request.method === "tools/call"
            ? { content: [{ type: "text", text: "visible MCP evidence" }] }
            : {};
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: request.id, result }), { status: 200, headers });
    });
    const service = new McpClientService(
      join(root, "mcp.json"),
      secrets,
      brain as never,
      fetcher as typeof fetch
    );
    await service.initialize();
    const summary = await service.add({
      brainId: "brain-1",
      id: "coding",
      label: "Coding tools",
      transport: "http",
      url: "http://127.0.0.1:3100/mcp",
      bearerToken: "mcp-private-token"
    });

    expect(summary.connected).toBe(true);
    expect(summary.tools).toHaveLength(1);
    expect(summary.tools[0]?.id).toMatch(/^mcp\./);
    expect(registered.at(-1)).toMatchObject({ actions: ["call"] });
    expect(permissions.at(-1)?.level).toBe("ask");
    expect(learned[0]).toContain("read_project");
    expect(learned[0]).toContain("inputSchema");
    expect(learned[0]).not.toContain("mcp-private-token");

    const executor = new ToolExecutor(
      brain as never,
      {} as never,
      undefined,
      undefined,
      service
    );
    const invocation = {
      brainId: "brain-1",
      toolId: summary.tools[0]!.id,
      action: "call",
      arguments: { path: "/project" }
    };
    const approval = await executor.execute(invocation);
    expect(approval.state).toBe("approval-required");
    const executed = await executor.execute({
      ...invocation,
      approvalToken: approval.approvalToken
    });
    expect(executed).toMatchObject({
      state: "complete",
      output: { content: [{ type: "text", text: "visible MCP evidence" }] }
    });
    expect(repository.save).toHaveBeenCalledOnce();

    // The worker emits MCP actions with schema arguments only. The chat path
    // must bind approval/audit to that same payload and send it unchanged.
    const now = new Date().toISOString();
    const chatController = new ChatActionController({
      chat: vi.fn(async () => ({
        brain: { messages: [] },
        humanMessage: { id: "human", role: "human", content: "Read the project", createdAt: now },
        brainMessage: { id: "brain", role: "brain", content: "I can read it.", createdAt: now },
        trace: { id: "trace" },
        proposedActions: [{
          kind: "tool",
          source: "brain",
          toolId: summary.tools[0]!.id,
          action: "call",
          arguments: { path: "/chat-project" }
        }]
      })),
      learnStructuredExperience: vi.fn(async () => ({})),
      recordConversationActions: vi.fn(async () => undefined)
    } as never, executor, { start: vi.fn() } as never);
    const chat = await chatController.send("brain-1", "Read the project");
    const pending = chat.actionEvents?.[0];
    expect(pending?.state).toBe("approval-required");
    const remoteCallsBeforeApproval = remoteCalls.length;
    const approvedChat = await chatController.approveAction({
      brainId: "brain-1",
      actionEventId: pending!.id,
      approvalToken: pending!.execution!.approvalToken!
    });
    expect(approvedChat.actionEvent.state).toBe("complete");
    expect(remoteCalls).toHaveLength(remoteCallsBeforeApproval + 1);
    expect(remoteCalls.at(-1)).toEqual({
      name: "read_project",
      arguments: { path: "/chat-project" }
    });
    expect(repository.save).toHaveBeenCalledTimes(2);

    const callsBeforeInvalidArguments = calls.filter((method) => method === "tools/call").length;
    await expect(service.execute(
      summary.tools[0]!.id, "call", {}, new AbortController().signal
    )).rejects.toThrow(/input schema validation \(required\)/);
    await expect(service.execute(
      summary.tools[0]!.id, "call", { path: 12 }, new AbortController().signal
    )).rejects.toThrow(/input schema validation \(type\)/);
    await expect(service.execute(
      summary.tools[0]!.id, "call", { path: "/project", unexpected: true }, new AbortController().signal
    )).rejects.toThrow(/input schema validation \(additionalProperties\)/);
    expect(calls.filter((method) => method === "tools/call")).toHaveLength(callsBeforeInvalidArguments);

    const result = await service.execute(
      summary.tools[0]!.id,
      "call",
      { path: "/project" },
      new AbortController().signal
    );
    expect(result).toEqual({ content: [{ type: "text", text: "visible MCP evidence" }] });
    expect(calls).toContain("tools/call");
    malformedSchema = true;
    await service.refresh("brain-1", "coding");
    const callsBeforeMalformedSchema = calls.filter((method) => method === "tools/call").length;
    await expect(service.execute(
      summary.tools[0]!.id, "call", { path: "/project" }, new AbortController().signal
    )).rejects.toThrow("MCP tool input schema must have an object root.");
    expect(calls.filter((method) => method === "tools/call")).toHaveLength(callsBeforeMalformedSchema);
    expect(await readFile(join(root, "mcp.json"), "utf8")).not.toContain("mcp-private-token");
    service.dispose();
  });

  it("maps MCP names to strict internal ids while calling the exact remote tool name", async () => {
    const permissions: ToolPermissionRecord[] = [];
    const registered: string[] = [];
    const toolCalls: Array<Record<string, unknown>> = [];
    const brain = {
      registerExternalToolSchemas: vi.fn((schemas: Array<{ id: string }>) => {
        for (const schema of schemas) {
          // Match BrainService's fail-closed internal identifier contract. MCP
          // names are protocol data and must be encoded before reaching it.
          expect(schema.id).toMatch(/^[a-z][a-z0-9.-]{1,79}$/);
          expect(schema.id.length).toBeLessThanOrEqual(80);
          registered.push(schema.id);
        }
      }),
      unregisterExternalToolSchemas: vi.fn(),
      learnStructuredExperience: vi.fn(async () => ({})),
      listToolPermissions: vi.fn(async () => permissions.map((value) => ({ ...value }))),
      setToolPermission: vi.fn(async (_brainId: string, toolId: string, level: ToolPermissionRecord["level"]) => {
        permissions.push({ toolId, label: toolId, level, updatedAt: new Date(0).toISOString() });
        return permissions;
      })
    };
    const fetcher = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const request = JSON.parse(String(init?.body)) as {
        id?: string;
        method: string;
        params?: Record<string, unknown>;
      };
      const headers = { "content-type": "application/json", "mcp-session-id": "fixture-session" };
      if (request.method === "notifications/initialized") return new Response(null, { status: 202, headers });
      if (request.method === "tools/call") toolCalls.push(request.params ?? {});
      const result = request.method === "initialize"
        ? { protocolVersion: "2025-06-18", capabilities: { tools: {} }, serverInfo: { name: "fixture", version: "1" } }
        : request.method === "tools/list"
          ? { tools: [{ name: "inspect_fixture", inputSchema: { $schema: "http://json-schema.org/draft-07/schema#", type: "object", properties: { value: { type: "string" } } } }] }
          : request.method === "tools/call"
            ? { content: [{ type: "text", text: "MCP-EVIDENCE:HELIO" }] }
            : {};
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: request.id, result }), { status: 200, headers });
    });
    const registry = join(root, "standards-compatible-mcp.json");
    const service = new McpClientService(registry, secrets, brain as never, fetcher as typeof fetch);
    await service.initialize();
    const summary = await service.add({
      brainId: "brain-1",
      id: "acceptance-fixture",
      label: "Acceptance fixture",
      transport: "http",
      url: "http://127.0.0.1:41917/mcp"
    });

    expect(summary.tools).toHaveLength(1);
    expect(summary.tools[0]).toMatchObject({
      id: expect.stringMatching(/^mcp\.[a-f0-9]{8}\.acceptance-fixture\.inspect-fixture-[a-f0-9]{8}$/),
      remoteName: "inspect_fixture",
      actions: ["call"]
    });
    expect(registered).toEqual([summary.tools[0]!.id]);
    await expect(service.execute(
      summary.tools[0]!.id,
      "call",
      { value: "HELIO" },
      new AbortController().signal
    )).resolves.toEqual({ content: [{ type: "text", text: "MCP-EVIDENCE:HELIO" }] });
    expect(toolCalls).toEqual([{ name: "inspect_fixture", arguments: { value: "HELIO" } }]);
    service.dispose();

    // Persisted ids must validate and re-register identically on restart.
    const restarted = new McpClientService(registry, secrets, brain as never, fetcher as typeof fetch);
    await restarted.initialize();
    expect((await restarted.list("brain-1"))[0]?.tools[0]?.id).toBe(summary.tools[0]!.id);
    restarted.dispose();
  });

  it("connects an installed local stdio MCP server without a shell", async () => {
    const fixture = join(root, "stdio-mcp.cjs");
    await writeFile(fixture, `
      const readline = require("node:readline");
      const lines = readline.createInterface({ input: process.stdin });
      lines.on("line", (line) => {
        const request = JSON.parse(line);
        if (request.id == null) return;
        let result = {};
        if (request.method === "initialize") {
          result = { protocolVersion: "2025-06-18", capabilities: { tools: {} }, serverInfo: { name: "stdio-fixture", version: "1" } };
        } else if (request.method === "tools/list") {
          result = { tools: [{ name: "inspect_workspace", inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } }] };
        } else if (request.method === "tools/call") {
          result = { content: [{ type: "text", text: "stdio:" + request.params.arguments.path }] };
        }
        process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: request.id, result }) + "\\n");
      });
    `, "utf8");
    const permissions: ToolPermissionRecord[] = [];
    const learned: string[] = [];
    const brain = {
      registerExternalToolSchemas: vi.fn(),
      unregisterExternalToolSchemas: vi.fn(),
      learnStructuredExperience: vi.fn(async (_brainId: string, value: { content: string }) => {
        learned.push(value.content);
        return {};
      }),
      listToolPermissions: vi.fn(async () => permissions),
      setToolPermission: vi.fn(async (_brainId: string, toolId: string, level: ToolPermissionRecord["level"]) => {
        permissions.push({ toolId, label: toolId, level, updatedAt: new Date(0).toISOString() });
        return permissions;
      })
    };
    const service = new McpClientService(join(root, "stdio-registry.json"), secrets, brain as never);
    await service.initialize();
    const summary = await service.add({
      brainId: "brain-stdio",
      id: "local-coding",
      label: "Local coding",
      transport: "stdio",
      command: process.execPath,
      args: [fixture]
    });
    expect(summary).toMatchObject({ connected: true, transport: "stdio", endpoint: process.execPath });
    expect(summary.tools[0]?.remoteName).toBe("inspect_workspace");
    expect(permissions[0]?.level).toBe("ask");
    expect(learned[0]).toContain("inspect_workspace");
    await expect(service.execute(
      summary.tools[0]!.id,
      "call",
      { path: root },
      new AbortController().signal
    )).resolves.toEqual({ content: [{ type: "text", text: `stdio:${root}` }] });
    service.dispose();
  });

  it("fails closed when a persisted MCP registry is tampered", async () => {
    const registry = join(root, "tampered-mcp.json");
    await writeFile(registry, JSON.stringify({
      format: "omni-mcp-registry",
      version: 1,
      servers: [{
        brainId: "../other-brain",
        id: "coding",
        label: "Tampered",
        transport: "http",
        url: "https://example.com/mcp",
        tools: []
      }]
    }), "utf8");
    const brain = {
      registerExternalToolSchemas: vi.fn(),
      unregisterExternalToolSchemas: vi.fn(),
      learnStructuredExperience: vi.fn(),
      listToolPermissions: vi.fn(),
      setToolPermission: vi.fn()
    };
    const service = new McpClientService(registry, secrets, brain as never);
    await expect(service.initialize()).rejects.toThrow("Invalid MCP brain id");
    expect(brain.registerExternalToolSchemas).not.toHaveBeenCalled();
  });
});
