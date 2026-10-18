import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { expect, it, vi } from "vitest";
import { McpClientService } from "../src/main/mcpClient";
import { McpTaskStore } from "../src/main/mcpTaskStore";
import { SecureSecretStore } from "../src/main/secureSecretStore";
import type { McpTaskReceipt } from "../src/shared/mcpTasks";

it("stops only a real owned stdio task wait without killing/restarting its process or sending a remote cancellation", async () => {
  const root = await mkdtemp(join(tmpdir(), "omni-owned-stdio-task-")); await mkdir(join(root, "brain"));
  const permissions: unknown[] = [], brain = { registerExternalToolSchemas() {}, unregisterExternalToolSchemas() {}, learnStructuredExperience: async () => ({}),
    listToolPermissions: async () => permissions, setToolPermission: async (_id: string, toolId: string, level: string) => { permissions.push({ toolId, level }); return permissions; } };
  const client = new McpClientService(join(root, "mcp.json"), new SecureSecretStore(join(root, "secrets.json"), { available: () => false, encrypt: value => Buffer.from(value), decrypt: value => value.toString() }), brain as never);
  client.setTaskStore(new McpTaskStore({ brainDirectory: async id => join(root, id), admit: async () => undefined, checkDisk: async () => undefined }));
  try {
    await client.initialize(); const server = await client.add({ brainId: "brain", id: "local", label: "Owned stdio task fixture", transport: "stdio", command: process.execPath, args: [resolve("tests/fixtures/mcp-task-server.cjs")] });
    const tool = server.tools.find(value => value.remoteName === "waiter")!, context = { requestId: "original-fixture-request", actionEventId: "fixture-action", requireCurrentPermission: async () => undefined };
    const start = await client.execute(tool.id, "call-task", { arguments: {} }, new AbortController().signal, "brain", context) as McpTaskReceipt;
    const pending = client.execute(tool.id, "task-result", { taskRef: start.record.taskRef }, new AbortController().signal, "brain", context), rejected = expect(pending).rejects.toThrow(/local MCP task wait stopped|cancelled/);
    await vi.waitFor(async () => { const page = await client.tasksLocalPage("brain"); expect(page.records[0]!.record.lastOperation?.method).toBe("tasks/result"); expect(page.records[0]!.record.lastOperation?.state).toBe("dispatch-may-have-started"); });
    expect(client.stopTaskWait("brain", start.record.taskRef)).toBe(true); await rejected; await client.settleTaskWrites("brain");
    const after = await client.execute(tool.id, "task-status", { taskRef: start.record.taskRef }, new AbortController().signal, "brain", context) as McpTaskReceipt;
    expect(after.record.remote?.fixtureProcessId).toBe(start.record.remote?.fixtureProcessId); expect(after.record.remote?.fixtureCancelNotifications).toBe(0); expect(after.record.remote?.status).toBe("working");
  } finally { client.dispose(); await client.settleTaskWrites(); await rm(root, { recursive: true, force: true }); }
});
