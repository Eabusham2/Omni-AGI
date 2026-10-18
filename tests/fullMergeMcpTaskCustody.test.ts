import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it } from "vitest";
import { FullBrainMergeCoordinator, type FullMergeHooks } from "../src/main/fullBrainMerge";
import { adoptFullMergeMcpTasks } from "../src/main/fullMergeMcpTasks";
import { mcpTaskCustodyFiles } from "../src/main/mcpTaskCustody";
import { McpTaskStore } from "../src/main/mcpTaskStore";
import { validateEmbeddedMcpTaskSources } from "../src/main/mcpTaskArchive";
import { completedMcpTaskFixture } from "./fixtures/mcpTaskCustody";
import { DEFAULT_CONFIG } from "../src/shared/types";

const roots: string[] = [], sha = (value: Uint8Array | string) => createHash("sha256").update(value).digest("hex");
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-full-merge-mcp-custody-")); roots.push(root);
  const sources = [];
  for (const [path, id, marker] of [["a", "a", "current-A"], ["b", "b", "current-B"], ["a/engine/origin/source-setup", "old-a", "immutable-origin-A"],
    ["a/merge/parents/ancestor", "ancestor", "shared-ancestor"], ["a/snapshots/old-point", "a", "historical-selected-task"]] as const) {
    const directory = join(root, ...path.split("/")); await mkdir(join(directory, "engine"), { recursive: true });
    await writeFile(join(directory, "engine", "brain.json"), JSON.stringify({ brain_id: id }));
    const task = await completedMcpTaskFixture(directory, id, marker); sources.push({ root: directory, id, marker, task });
  }
  // Install immutable-origin fixture bytes in their actual saved namespace;
  // the temporary source setup is not itself an extra historical source.
  const origin = sources[2]!, originTarget = join(root, "a", "engine", "origin");
  await import("node:fs/promises").then(fs => fs.cp(join(origin.root, "engine", "operational-mcp-tasks"), join(originTarget, "operational-mcp-tasks"), { recursive: true }));
  await rm(origin.root, { recursive: true, force: true });
  const calls: string[] = [];
  const hooks: FullMergeHooks = {
    withParentsFrozen: async (_ids, operation) => operation(),
    sealParent: async id => ({ brainId: id, name: id, ancestry: [], generationSha256: sha(await readFile(join(root, id, "engine", "brain.json"))) }),
    validateSettings: async () => undefined, reserve: async (_bytes, purpose) => { calls.push(purpose); },
    admit: async (_bytes, purpose) => { calls.push(purpose); },
    assemble: async input => {
      await mkdir(join(input.targetDirectory, "engine", "origin"), { recursive: true });
      await adoptFullMergeMcpTasks(input);
      await writeFile(join(input.targetDirectory, "brain.json"), JSON.stringify({ id: input.targetBrainId }));
      await writeFile(join(input.targetDirectory, "engine", "brain.json"), JSON.stringify({ brain_id: input.targetBrainId }));
    },
    verify: async input => { await mcpTaskCustodyFiles(join(input.targetDirectory, "engine")); }
  };
  return { root, sources, hooks, calls };
}

it("stages all actual parent/current/origin/ancestor/recovery task scopes and merges exact dormant rows", async () => {
  const f = await fixture(), before = await readFile(join(f.root, "a", "engine", "operational-mcp-tasks", `${f.sources[0]!.task.row.taskRef}.json`));
  let changed = false;
  const reserve = f.hooks.reserve;
  f.hooks.reserve = async (bytes, purpose) => {
    await reserve(bytes, purpose);
    if (!changed && purpose === "complete parent state blob") {
      changed = true;
      await f.sources[0]!.task.store.update("a", f.sources[0]!.task.row.taskRef, { reason: "source advanced after coherent task capture" });
    }
  };
  const coordinator = new FullBrainMergeCoordinator(f.root, f.hooks);
  const done = await coordinator.start({ operationId: "task-custody-merge", parentIds: ["a", "b"], name: "Joined", config: { ...DEFAULT_CONFIG, name: "Joined" } });
  expect(done.state).toBe("complete"); expect(changed).toBe(true);
  const target = join(f.root, done.targetBrainId), store = new McpTaskStore({ brainDirectory: async () => target });
  const page = await store.localPage(done.targetBrainId);
  expect(page.records).toHaveLength(5); expect(page.records.every(row => row.custodyPath.includes("/lineage/") && row.record.authority === "historical-not-live")).toBe(true);
  expect(page.records.map(row => row.record.source.brainId).sort()).toEqual(["a", "a", "ancestor", "b", "old-a"]);
  // Current and historical source versions may legitimately share one stable
  // taskRef. Select this exact outcome version, never collapse history by ID.
  const selected = page.records.find(row => row.record.taskRef === f.sources[0]!.task.row.taskRef &&
    row.record.result?.payloadSha256 === f.sources[0]!.task.outcome.payloadSha256);
  expect(selected?.record.reason).toBeUndefined();
  expect(selected?.record).toEqual(JSON.parse(before.toString("utf8")));
  const files = await mcpTaskCustodyFiles(join(target, "engine"));
  expect([...files.keys()].every(path => (path.match(/\/lineage\//g) ?? []).length === 1)).toBe(true);
  const originFiles = await mcpTaskCustodyFiles(join(target, "engine", "origin"));
  expect([...originFiles.keys()].sort()).toEqual([...files.keys()].sort());
  for (const [local, path] of files) expect(await readFile(originFiles.get(local)!)).toEqual(await readFile(path));
  expect(JSON.parse(await readFile(join(f.root, "a", "engine", "origin", "operational-mcp-tasks", `${f.sources[2]!.task.row.taskRef}.json`), "utf8"))).toEqual(f.sources[2]!.task.row);
  expect(JSON.parse(await readFile(join(f.root, "b", "engine", "operational-mcp-tasks", `${f.sources[1]!.task.row.taskRef}.json`), "utf8"))).toEqual(f.sources[1]!.task.row);
  await expect(stat(join(f.root, ".full-merges", "task-custody-merge", ".mcp-task-snapshots"))).rejects.toMatchObject({ code: "ENOENT" });
  await expect(adoptFullMergeMcpTasks({ targetBrainId: done.targetBrainId, targetDirectory: target,
    parents: [{ brainId: "a", name: "a", ancestry: [], generationSha256: "a".repeat(64) }],
    request: { operationId: "not-an-active-assembly", parentIds: ["a", "b"], name: "Joined", config: { ...DEFAULT_CONFIG, name: "Joined" } },
    files: function* () { yield { parentId: "a", path: `engine/operational-mcp-tasks/${f.sources[0]!.task.row.taskRef}.json`, sha256: "a".repeat(64), bytes: 1, mode: 0o600 }; },
    blobPath: () => "unused", materializeParent: async () => f.sources[0]!.root, reserve: async () => undefined,
    checkpoint: async () => undefined })).rejects.toThrow(/unpublished coordinator assembly/);
  expect(f.calls).toContain("coherent complete parent MCP task snapshot"); expect(f.calls).toContain("MCP task custody JSON");
});

it("validates task closure inside opaque connected-parent sources without a neural materialization", async () => {
  const f = await fixture(), source = f.sources[0]!, files = await mcpTaskCustodyFiles(join(source.root, "engine"));
  const stage = await mkdtemp(join(f.root, ".embedded-tasks-"));
  expect(await validateEmbeddedMcpTaskSources([...files].map(([local, sourcePath]) => [`merge/parents/a/engine/${local}`, { name: local, sourcePath }]), stage)).toBe(true);
  const copied = await mcpTaskCustodyFiles(join(stage, "merge", "parents", "a", "engine"));
  expect([...copied.keys()].sort()).toEqual([...files.keys()].sort());
  const badStage = await mkdtemp(join(f.root, ".embedded-missing-"));
  await expect(validateEmbeddedMcpTaskSources([...files].filter(([local]) => !local.includes("/payloads/")).map(([local, sourcePath]) => [`merge/parents/a/engine/${local}`, { name: local, sourcePath }]), badStage)).rejects.toThrow();
});
