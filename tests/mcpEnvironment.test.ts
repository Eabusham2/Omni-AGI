import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { McpClientService } from "../src/main/mcpClient";
import { mcpEnvironment } from "../src/main/mcpEnvironment";
import { SecureSecretStore } from "../src/main/secureSecretStore";
import { McpRegistryStore } from "../src/main/mcpRegistry";

describe("explicit MCP stdio environment", () => {
  const roots: string[] = [];
  const clients: McpClientService[] = [];
  afterEach(async () => { clients.splice(0).forEach(client => client.dispose()); for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
  it("validates names, NULs, type and encrypted-record budget without replacing values", () => {
    expect(mcpEnvironment({ TOKEN: " Unicode: value \n", EMPTY: "" })).toEqual({ TOKEN: " Unicode: value \n", EMPTY: "" });
    for (const value of [null, [], { "invalid-name": "x" }, { A: 1 }, { A: "a\0b" }, { A: "x".repeat(32001) }, { A: "x".repeat(31000), B: "x".repeat(31000) }]) expect(() => mcpEnvironment(value)).toThrow();
  });
  it("passes only explicit credentials to a real local process and retains them outside registry/schema state", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-local-mcp-env-")); roots.push(root);
    const permissions: Array<{ toolId: string; level: string }> = [];
    const brain = { registerExternalToolSchemas: vi.fn(), unregisterExternalToolSchemas: vi.fn(), learnStructuredExperience: vi.fn(async () => ({})),
      listToolPermissions: vi.fn(async () => permissions), setToolPermission: vi.fn(async (_brain: string, toolId: string, level: string) => { permissions.push({ toolId, level }); }) };
    const secrets = new SecureSecretStore(join(root, "secrets.json"), { available: () => false, encrypt: value => Buffer.from(value), decrypt: value => value.toString() });
    const client = new McpClientService(join(root, "registry.json"), secrets, brain as never); clients.push(client);
    await client.initialize();
    const previous = process.env.OMNI_PARENT_FIXTURE_SECRET; process.env.OMNI_PARENT_FIXTURE_SECRET = "do-not-inherit";
    try {
      const server = await client.add({ brainId: "first", id: "local", label: "Local fixture", transport: "stdio", command: process.execPath,
        args: [resolve("tests/fixtures/mcp-env-server.cjs")], environment: { OMNI_MCP_FIXTURE_KEY: "test-only-value" } });
      expect(server.environmentKeys).toEqual(["OMNI_MCP_FIXTURE_KEY"]); expect(server.credentialConfigured).toBe(true);
      const result = await client.execute(server.tools[0]!.id, "call", {}, new AbortController().signal, "first") as { content: Array<{ text: string }> };
      expect(JSON.parse(result.content[0]!.text)).toEqual({ explicitCredential: true, parentSecretAbsent: true });
      expect(await readFile(join(root, "registry.json"), "utf8")).not.toContain("test-only-value");
      expect(JSON.stringify(brain.learnStructuredExperience.mock.calls)).not.toContain("test-only-value");
      await client.remove("first", "local"); expect(await secrets.has("mcp:environment:local")).toBe(false);
    } finally { if (previous === undefined) delete process.env.OMNI_PARENT_FIXTURE_SECRET; else process.env.OMNI_PARENT_FIXTURE_SECRET = previous; }
  });
  it("does not accept stdio credentials on an HTTP server or orphan them on registration failure", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-local-mcp-refusal-")); roots.push(root);
    const brain = { registerExternalToolSchemas: vi.fn(), unregisterExternalToolSchemas: vi.fn(), learnStructuredExperience: vi.fn(), listToolPermissions: vi.fn(async () => []), setToolPermission: vi.fn() };
    const secrets = new SecureSecretStore(join(root, "secrets.json"), { available: () => false, encrypt: value => Buffer.from(value), decrypt: value => value.toString() });
    const client = new McpClientService(join(root, "registry.json"), secrets, brain as never); clients.push(client); await client.initialize();
    await expect(client.add({ brainId: "first", id: "http", label: "HTTP", transport: "http", url: "https://mcp.example/mcp", environment: { KEY: "private" } })).rejects.toThrow(/Only local/);
    await expect(client.add({ brainId: "first", id: "invalid", label: "Invalid", transport: "stdio", command: join(root, "missing-executable"), environment: { KEY: "private" } })).rejects.toThrow();
    expect(await secrets.has("mcp:environment:invalid")).toBe(false); expect(await client.list("first")).toEqual([]);
  });
  it("replaces existing credentials without new tool permissions, automatic calls, or deleting registration", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-mcp-replace-")); roots.push(root);
    const permissions: Array<{ toolId: string; level: string }> = [];
    const brain = { registerExternalToolSchemas: vi.fn(), unregisterExternalToolSchemas: vi.fn(), learnStructuredExperience: vi.fn(async () => ({})),
      listToolPermissions: vi.fn(async () => permissions), setToolPermission: vi.fn(async (_brain: string, toolId: string, level: string) => { permissions.push({ toolId, level }); }) };
    const secrets = new SecureSecretStore(join(root, "secrets.json"), { available: () => false, encrypt: value => Buffer.from(value), decrypt: value => value.toString() });
    const client = new McpClientService(join(root, "registry.json"), secrets, brain as never); clients.push(client); await client.initialize();
    const registered = await client.add({ brainId: "first", id: "local", label: "Local fixture", transport: "stdio", command: process.execPath,
      args: [resolve("tests/fixtures/mcp-env-server.cjs")], environment: { OMNI_MCP_FIXTURE_KEY: "test-only-value" } });
    const changes = brain.learnStructuredExperience.mock.calls.length;
    const updated = await client.updateCredentials("first", "local", { environment: { OMNI_MCP_FIXTURE_KEY: "changed-test-value" } });
    expect(updated).toMatchObject({ id: "local", connected: false, credentialConfigured: true, environmentKeys: ["OMNI_MCP_FIXTURE_KEY"] });
    expect(updated.tools[0]!.id).toBe(registered.tools[0]!.id); expect(permissions).toHaveLength(1);
    expect(brain.learnStructuredExperience.mock.calls).toHaveLength(changes);
    await client.refresh("first", "local");
    const result = await client.execute(registered.tools[0]!.id, "call", {}, new AbortController().signal, "first") as { content: Array<{ text: string }> };
    expect(JSON.parse(result.content[0]!.text).explicitCredential).toBe(false);
    expect(await readFile(join(root, "registry.json"), "utf8")).not.toContain("changed-test-value");
    await client.configure({ enabled: false });
    const cleared = await client.updateCredentials("first", "local", { environment: {} });
    expect(cleared.effectiveEnabled).toBe(false); expect(cleared.credentialConfigured).toBe(false);
    expect(cleared.environmentKeys).toBeUndefined();
    await expect(client.execute(registered.tools[0]!.id, "call", {}, new AbortController().signal, "first")).rejects.toThrow(/switched off/);
  });
  it("restores the previous environment if atomic registry publication fails", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-mcp-replace-rollback-")); roots.push(root);
    const brain = { registerExternalToolSchemas: vi.fn(), unregisterExternalToolSchemas: vi.fn(), learnStructuredExperience: vi.fn(async () => ({})), listToolPermissions: vi.fn(async () => []), setToolPermission: vi.fn() };
    const secrets = new SecureSecretStore(join(root, "secrets.json"), { available: () => false, encrypt: value => Buffer.from(value), decrypt: value => value.toString() });
    const client = new McpClientService(join(root, "registry.json"), secrets, brain as never); clients.push(client); await client.initialize();
    await client.add({ brainId: "first", id: "local", label: "Local fixture", transport: "stdio", command: process.execPath,
      args: [resolve("tests/fixtures/mcp-env-server.cjs")], environment: { OMNI_MCP_FIXTURE_KEY: "test-only-value" } });
    const before = await readFile(join(root, "registry.json"), "utf8");
    const publish = vi.spyOn(McpRegistryStore.prototype, "save").mockRejectedValueOnce(new Error("fixture publication failure"));
    try { await expect(client.updateCredentials("first", "local", { environment: { OTHER: "replacement" } })).rejects.toThrow(/publication failure/); }
    finally { publish.mockRestore(); }
    expect(await readFile(join(root, "registry.json"), "utf8")).toBe(before);
    expect(JSON.parse((await secrets.get("mcp:environment:local"))!)).toEqual({ OMNI_MCP_FIXTURE_KEY: "test-only-value" });
    const server = await client.refresh("first", "local");
    const result = await client.execute(server.tools[0]!.id, "call", {}, new AbortController().signal, "first") as { content: Array<{ text: string }> };
    expect(JSON.parse(result.content[0]!.text).explicitCredential).toBe(true);
  });
});
