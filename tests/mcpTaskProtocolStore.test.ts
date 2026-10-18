import { mkdtemp, mkdir, cp, readFile, realpath, rm, symlink, unlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import { afterEach, expect, it } from "vitest";
import { McpTaskStore, mcpTaskReadJson } from "../src/main/mcpTaskStore";
import { captureMcpTaskCustody, mcpTaskCustodyFiles, mergeMcpTaskCustody, safeMcpTaskCustodyPath } from "../src/main/mcpTaskCustody";
import { assertMcpTaskTransition, checkedMcpCreateTaskResult, checkedMcpRemoteTask, checkedMcpTaskToolResult, mcpTaskSupport } from "../src/main/mcpTaskProtocol";
import type { McpRemoteTask, McpTaskRecord } from "../src/shared/mcpTasks";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
const task = (status: McpRemoteTask["status"] = "working"): McpRemoteTask => ({ taskId: "actual-receiver-task-id", status, createdAt: "2026-10-17T00:00:00.000Z", lastUpdatedAt: "2026-10-17T00:00:01.000Z", ttl: null, pollInterval: 0 });
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-mcp-task-")); roots.push(root); for (const id of ["brain-a", "brain-b", "brain-c", "stage"]) await mkdir(join(root, id));
  const store = new McpTaskStore({ brainDirectory: async id => join(root, id), admit: async () => undefined, checkDisk: async () => undefined });
  const prepared = await store.prepare({ source: { brainId: "brain-a", serverId: "server", toolId: "mcp.original.tool", remoteName: "original", endpoint: "https://fixture.invalid/mcp", protocolVersion: "2025-11-25", configurationSha256: "a".repeat(64), schemaSha256: "b".repeat(64) },
    requestId: "source-original-request", actionEventId: "source-original-action", chatTurnId: "source-original-turn", argumentsSha256: "c".repeat(64) });
  return { root, store, prepared, engine: join(root, "brain-a", "engine"), recordPath: join(root, "brain-a", "engine", "operational-mcp-tasks", `${prepared.taskRef}.json`) };
}
async function accept(f: Awaited<ReturnType<typeof fixture>>): Promise<McpTaskRecord> { return f.store.update("brain-a", f.prepared.taskRef, { creation: "accepted", remote: task(), createResult: { task: task(), _meta: { "io.modelcontextprotocol/model-immediate-response": "Untrusted provider metadata retained, not a host instruction" } } }); }
it("validates only the dated task state machine, original related-task result and exact required/optional/forbidden semantics", () => {
  expect(mcpTaskSupport(undefined)).toBe("forbidden"); expect(mcpTaskSupport({ taskSupport: "required" })).toBe("required"); expect(() => mcpTaskSupport({ taskSupport: "maybe" })).toThrow();
  expect(checkedMcpCreateTaskResult({ task: task() })).toEqual(task()); expect(() => checkedMcpCreateTaskResult({ task: task("completed") })).toThrow(/begin working/);
  expect(() => checkedMcpRemoteTask({ ...task(), ttl: -1 })).toThrow(/milliseconds/); expect(() => assertMcpTaskTransition(task("completed"), task("working"))).toThrow(/terminal/);
  expect(checkedMcpTaskToolResult({ content: [{ type: "text", text: "Actual fixture result" }], _meta: { "io.modelcontextprotocol/related-task": { taskId: task().taskId } } }, task().taskId).content).toHaveLength(1);
  expect(() => checkedMcpTaskToolResult({ content: [], _meta: { "io.modelcontextprotocol/related-task": { taskId: "foreign" } } }, task().taskId)).toThrow(/related-task/);
  expect(() => checkedMcpRemoteTask({ resultType: "task", ...task(), ttl: undefined, ttlMs: 1000 })).toThrow();
});
it("persists a pre-dispatch and lost-ACK attempt without treating it as a failed/cancelled tool or a replay grant", async () => {
  const f = await fixture(); await f.store.update("brain-a", f.prepared.taskRef, { creation: "dispatch-may-have-started" });
  const row = await f.store.update("brain-a", f.prepared.taskRef, { creation: "creation-outcome-unknown", reason: "Synthetic lost ACK; never resent" });
  expect(row.remote).toBeUndefined(); expect(row.authority).toBe("historical-not-live"); expect((await f.store.get("brain-a", row.taskRef)).creation).toBe("creation-outcome-unknown");
});
it("retains the original source/ACK/result with a stable final receipt for repeated delivery, not repeated execution", async () => {
  const f = await fixture(), row = await accept(f), raw = { content: [{ type: "text", text: "Exact source-original outcome" }], _meta: { "io.modelcontextprotocol/related-task": { taskId: task().taskId } } };
  const first = await f.store.putOutcome("brain-a", row.taskRef, "result", raw), second = await f.store.retainedOutcome("brain-a", row.taskRef);
  expect(second).toMatchObject({ retained: true, taskResultReceiptId: first.taskResultReceiptId, result: raw }); expect(second?.source).toEqual(f.prepared.source);
  await expect(f.store.putOutcome("brain-a", row.taskRef, "result", { ...raw, content: [] })).rejects.toThrow(/conflicting/);
  expect((await f.store.get("brain-a", row.taskRef)).createResult?._meta).toEqual(row.createResult?._meta);
});
it("stages a coherent exact snapshot and lets later source commits proceed without changing staged bytes", async () => {
  const f = await fixture(); await accept(f); const stage = join(f.root, "stage", "engine"), staged = await captureMcpTaskCustody(f.engine, stage);
  const local = `operational-mcp-tasks/${f.prepared.taskRef}.json`, before = await readFile(staged.get(local)!);
  await f.store.update("brain-a", f.prepared.taskRef, { remote: task("completed") });
  expect(await readFile(staged.get(local)!)).toEqual(before); expect(JSON.parse(before.toString()).remote.status).toBe("working"); expect((await f.store.get("brain-a", f.prepared.taskRef)).remote?.status).toBe("completed");
});
it("flattens repeated shared-ancestor merges without recursively duplicating exact earlier history", async () => {
  const f = await fixture(); await accept(f); const b = join(f.root, "brain-b", "engine"), c = join(f.root, "brain-c", "engine"); await mkdir(b); await mkdir(c);
  await mergeMcpTaskCustody(f.engine, b); await mergeMcpTaskCustody(b, c); const first = await mcpTaskCustodyFiles(c);
  await mergeMcpTaskCustody(f.engine, c); await mergeMcpTaskCustody(b, c); const again = await mcpTaskCustodyFiles(c);
  expect([...again.keys()].sort()).toEqual([...first.keys()].sort()); expect([...again.keys()].every(path => (path.match(/\/lineage\//g) ?? []).length === 1)).toBe(true);
  const rows = await f.store.localPage("brain-c"); expect(rows.records).toHaveLength(1); expect(rows.records[0]!.record.source.brainId).toBe("brain-a");
});
it("preserves original ids on a copied instance but refuses source-owner mutation rather than regranting it", async () => {
  const f = await fixture(); await accept(f); await cp(f.engine, join(f.root, "brain-b", "engine"), { recursive: true });
  const row = (await f.store.localPage("brain-b")).records[0]!.record; expect(row.source.brainId).toBe("brain-a"); expect(row.taskRef).toBe(f.prepared.taskRef);
  await expect(f.store.update("brain-b", row.taskRef, { remote: task("completed") })).rejects.toThrow(/dormant/);
});
it.skipIf(process.platform === "win32")("accepts normal OS root aliases but rejects a final symlink", async () => {
  const f = await fixture(); await accept(f); const canonical = await realpath(join(f.root, "brain-a"));
  expect((await mcpTaskReadJson(join(f.root, "brain-a"), `engine/operational-mcp-tasks/${f.prepared.taskRef}.json`)).bytes).toEqual((await mcpTaskReadJson(canonical, `engine/operational-mcp-tasks/${f.prepared.taskRef}.json`)).bytes);
  const saved = await readFile(f.recordPath), target = join(f.root, "external.json"); await writeFile(target, saved); await unlink(f.recordPath); await symlink(target, f.recordPath);
  await expect(f.store.get("brain-a", f.prepared.taskRef)).rejects.toThrow(/escaped|regular/);
});
it("refuses a tampered final payload rather than delivering it as another successful result", async () => {
  const f = await fixture(), row = await accept(f), raw = { content: [], _meta: { "io.modelcontextprotocol/related-task": { taskId: task().taskId } } };
  const outcome = await f.store.putOutcome("brain-a", row.taskRef, "result", raw), path = join(f.engine, "operational-mcp-tasks", "payloads", `${outcome.payloadSha256}.json`);
  await writeFile(path, JSON.stringify({ content: [{ type: "text", text: "Tampered" }] }));
  await expect(f.store.retainedOutcome("brain-a", row.taskRef)).rejects.toThrow(/changed/); await expect(mcpTaskCustodyFiles(f.engine)).rejects.toThrow(/binding|address/);
});
it("keeps snapshot roots disjoint, admits accumulated descriptors and recognizes only the inert task namespace", async () => {
  const f = await fixture(), original = await readFile(f.recordPath);
  await expect(captureMcpTaskCustody(f.engine, f.engine)).rejects.toThrow(/disjoint/);
  await expect(captureMcpTaskCustody(f.engine, join(f.engine, "nested"))).rejects.toThrow(/disjoint/); expect(await readFile(f.recordPath)).toEqual(original);
  await expect(mcpTaskCustodyFiles(f.engine, { admit: async (_bytes, purpose) => { if (purpose.includes("descriptors")) throw new Error("Selected metadata RAM cap"); } })).rejects.toThrow(/metadata RAM cap/);
  expect(safeMcpTaskCustodyPath(`operational-mcp-tasks/${f.prepared.taskRef}.json`)).toBe(true);
  expect(safeMcpTaskCustodyPath(`operational-mcp-tasks/lineage/${"a".repeat(64)}/history/${"b".repeat(64)}.json`)).toBe(true);
  for (const path of ["operational-mcp-tasks/../secret.json", "operational-mcp-tasks/program.js", "operational-mcp-tasks/payloads/unknown.json", "operational-mcp-tasks/lineage/nope/file.json"]) expect(safeMcpTaskCustodyPath(path)).toBe(false);
});
it("preserves recognized interrupted publication bytes without parsing or promoting them into a task head or grant", async () => {
  const f = await fixture(), local = `operational-mcp-tasks/${f.prepared.taskRef}.json.${randomUUID()}.next`, raw = Buffer.from('{"unfinished":');
  await writeFile(join(f.engine, local), raw); const files = await mcpTaskCustodyFiles(f.engine); expect(files.has(local)).toBe(true); expect(safeMcpTaskCustodyPath(local)).toBe(true);
  const staged = await captureMcpTaskCustody(f.engine, join(f.root, "stage", "engine")); expect(await readFile(staged.get(local)!)).toEqual(raw);
  expect((await f.store.localPage("brain-a")).records).toHaveLength(1); expect((await f.store.get("brain-a", f.prepared.taskRef)).creation).toBe("prepared");
});
