import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { EnvironmentStore } from "../src/main/environmentStore";
import { mergeEnvironmentWorkspace, rekeyEnvironmentWorkspace, validateEnvironmentWorkspace } from "../src/main/environmentPortability";
import { ENVIRONMENT_OPERATION_FORMAT, ENVIRONMENT_RECORD_FORMAT, type EnvironmentOperationReceipt, type ManagedEnvironmentRecord } from "../src/shared/environments";

const roots: string[] = [], sourceId = "33333333-3333-4333-8333-333333333333", targetId = "44444444-4444-4444-8444-444444444444";
afterEach(async () => { await Promise.all(roots.splice(0).map(path => rm(path, { recursive: true, force: true }))); });
async function prepared(brainId: string): Promise<{ root: string; environmentId: string }> {
  const root = await mkdtemp(join(tmpdir(), "omni-environment-portability-")); roots.push(root); const store = await EnvironmentStore.open(root, brainId), createdAt = new Date().toISOString();
  const environmentId = "c".repeat(32), operationId = "d".repeat(32), directory = store.instanceDirectory(environmentId);
  await mkdir(join(directory, "workspace", "nested"), { recursive: true }); await writeFile(join(directory, "workspace", "nested", "evidence.txt"), "exact bytes"); await writeFile(join(directory, "sandbox.sb"), "source path profile");
  const environment: ManagedEnvironmentRecord = { format: ENVIRONMENT_RECORD_FORMAT, version: 1, id: environmentId, brainId, kind: "sandbox", name: "Portable",
    adapter: "darwin-sandbox-exec", state: "loaded", revision: 3, network: "none", createdAt, updatedAt: createdAt, externalId: undefined, lastOperationId: operationId };
  const operation: EnvironmentOperationReceipt = { format: ENVIRONMENT_OPERATION_FORMAT, version: 1, id: operationId, brainId, environmentId, requestId: "portable-request",
    action: "run", state: "complete", argumentsSha256: "e".repeat(64), dispatchStarted: true, createdAt, updatedAt: createdAt,
    process: { exitCode: 0, stdout: "done", stderr: "", truncated: false } };
  store.transaction(() => { store.putEnvironment(environment); store.putOperation(operation); store.beginRequest(operation.requestId, operation.argumentsSha256, operation.id); store.finishRequest(operation.requestId, { environment, operation }); }); store.close();
  await validateEnvironmentWorkspace(root, brainId); return { root, environmentId };
}

describe("managed environment portability", () => {
  it("makes copied state dormant while retaining exact files/receipts without adapter identity or requests", async () => {
    const source = await prepared(sourceId); await rekeyEnvironmentWorkspace(source.root, sourceId, targetId);
    await validateEnvironmentWorkspace(source.root, targetId); const store = await EnvironmentStore.open(source.root, targetId, true);
    const environment = store.getEnvironment(source.environmentId)!;
    expect(environment).toMatchObject({ brainId: targetId, state: "dormant", inheritedFrom: { brainId: sourceId, environmentId: source.environmentId } });
    expect(environment.externalId).toBeUndefined();
    expect(store.request("portable-request", "e".repeat(64))).toBeUndefined(); expect(store.operations(source.environmentId)[0]).toMatchObject({ brainId: targetId, process: { stdout: "done" } }); const revision = environment.revision; store.close();
    await rekeyEnvironmentWorkspace(source.root, sourceId, targetId); const reopened = await EnvironmentStore.open(source.root, targetId, true); expect(reopened.getEnvironment(source.environmentId)?.revision).toBe(revision); reopened.close();
    await expect(import("node:fs/promises").then(fs => fs.readFile(join(source.root, "environments", "instances", source.environmentId, "workspace", "nested", "evidence.txt"), "utf8"))).resolves.toBe("exact bytes");
    await expect(import("node:fs/promises").then(fs => fs.lstat(join(source.root, "environments", "instances", source.environmentId, "sandbox.sb")))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("merges a deterministic dormant private copy without a live parent process or grant", async () => {
    const source = await prepared(sourceId), targetRoot = await mkdtemp(join(tmpdir(), "omni-environment-merge-")); roots.push(targetRoot);
    const aliases = await mergeEnvironmentWorkspace(source.root, targetRoot, sourceId, targetId, sourceId);
    const localId = aliases.get(source.environmentId)!; await validateEnvironmentWorkspace(targetRoot, targetId);
    const store = await EnvironmentStore.open(targetRoot, targetId, true), environment = store.getEnvironment(localId)!;
    expect(environment).toMatchObject({ id: localId, brainId: targetId, state: "dormant", inheritedFrom: { brainId: sourceId, environmentId: source.environmentId } });
    expect(store.operations(localId)[0]).toMatchObject({ brainId: targetId, environmentId: localId, process: { stdout: "done" } }); store.close();
  });

  it("rejects a native filename whose backslashes could become portable traversal", async context => {
    if (process.platform === "win32") { context.skip(); return; }
    const source = await prepared(sourceId), ambiguous = join(source.root, "environments", "instances", source.environmentId, "workspace", "..\\..\\..\\..\\escape");
    await writeFile(ambiguous, "must stay contained");
    await expect(validateEnvironmentWorkspace(source.root, sourceId)).rejects.toThrow(/nonportable path component/i);
  });

  it("does not publish an environment row or target directory when staged merge copying fails", async () => {
    const source = await prepared(sourceId), targetRoot = await mkdtemp(join(tmpdir(), "omni-environment-merge-failure-")); roots.push(targetRoot);
    await expect(mergeEnvironmentWorkspace(source.root, targetRoot, sourceId, targetId, sourceId, async () => { throw new Error("fixture disk refusal"); })).rejects.toThrow(/fixture disk refusal/i);
    await validateEnvironmentWorkspace(targetRoot, targetId); const target = await EnvironmentStore.open(targetRoot, targetId, true);
    expect(target.page()).toMatchObject({ total: 0, entries: [] }); target.close();
  });

  it("rejects orphan environment files when the ownership database is absent", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-environment-orphan-files-")); roots.push(root);
    const workspace = join(root, "environments", "instances", "a".repeat(32), "workspace"); await mkdir(workspace, { recursive: true }); await writeFile(join(workspace, "orphan.txt"), "unowned");
    await expect(validateEnvironmentWorkspace(root, sourceId)).rejects.toThrow(/without their ownership database/i);
  });
});
