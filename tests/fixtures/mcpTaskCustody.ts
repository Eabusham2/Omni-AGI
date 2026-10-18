import { McpTaskStore } from "../../src/main/mcpTaskStore";
import type { McpRemoteTask } from "../../src/shared/mcpTasks";
/** Local JSON custody fixture only: never a client, remote request or grant. */
export async function completedMcpTaskFixture(root: string, brainId: string, marker = "exact private source task") {
  const store = new McpTaskStore({ brainDirectory: async () => root, admit: async () => undefined, checkDisk: async () => undefined });
  const prepared = await store.prepare({ source: { brainId, serverId: "fixture-server", toolId: "mcp.fixture-original", remoteName: "fixture-original",
    endpoint: "https://fixture.invalid/mcp", protocolVersion: "2025-11-25", configurationSha256: "a".repeat(64), schemaSha256: "b".repeat(64) },
    requestId: "source-original-request", argumentsSha256: "c".repeat(64) });
  const task: McpRemoteTask = { taskId: `remote-${brainId}`, status: "working", createdAt: "2026-10-17T00:00:00.000Z",
    lastUpdatedAt: "2026-10-17T00:00:01.000Z", ttl: null, pollInterval: 0 };
  await store.update(brainId, prepared.taskRef, { creation: "accepted", remote: task, createResult: { task } });
  await store.update(brainId, prepared.taskRef, { remote: { ...task, status: "completed" } });
  const payload = { content: [{ type: "text", text: marker }], _meta: { "io.modelcontextprotocol/related-task": { taskId: task.taskId } } };
  const outcome = await store.putOutcome(brainId, prepared.taskRef, "result", payload);
  return { store, row: await store.get(brainId, prepared.taskRef), outcome, payload };
}
