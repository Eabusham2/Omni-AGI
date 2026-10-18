import { chmod, mkdtemp, mkdir, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { EnvironmentStore } from "../src/main/environmentStore";
import { ENVIRONMENT_OPERATION_FORMAT, ENVIRONMENT_RECORD_FORMAT, type EnvironmentOperationReceipt, type ManagedEnvironmentRecord } from "../src/shared/environments";

const roots: string[] = [];
const brainId = "11111111-1111-4111-8111-111111111111", environmentId = "a".repeat(32), operationId = "b".repeat(32);
async function root(): Promise<string> { const path = await mkdtemp(join(tmpdir(), "omni-environment-store-")); roots.push(path); await mkdir(path, { recursive: true }); return path; }
afterEach(async () => { const { rm } = await import("node:fs/promises"); await Promise.all(roots.splice(0).map(path => rm(path, { recursive: true, force: true }))); });

function records(now = new Date().toISOString()): { environment: ManagedEnvironmentRecord; operation: EnvironmentOperationReceipt } {
  return {
    environment: { format: ENVIRONMENT_RECORD_FORMAT, version: 1, id: environmentId, brainId, kind: "sandbox", name: "Fixture", adapter: "darwin-sandbox-exec",
      state: "busy", revision: 2, network: "none", createdAt: now, updatedAt: now, lastOperationId: operationId },
    operation: { format: ENVIRONMENT_OPERATION_FORMAT, version: 1, id: operationId, brainId, environmentId, requestId: "fixture-request",
      action: "run", state: "running", argumentsSha256: "c".repeat(64), dispatchStarted: true, createdAt: now, updatedAt: now }
  };
}

describe("managed environment store", () => {
  it("keeps exact checksummed ownership, request idempotence, paging and interrupted recovery", async () => {
    const path = await root(), store = await EnvironmentStore.open(path, brainId), { environment, operation } = records();
    await mkdir(store.instanceDirectory(environment.id), { recursive: true });
    store.transaction(() => { store.putEnvironment(environment); store.putOperation(operation); store.beginRequest(operation.requestId, operation.argumentsSha256, operation.id); });
    expect(store.page()).toMatchObject({ total: 1, entries: [{ id: environmentId, state: "busy" }] });
    expect(store.request(operation.requestId, operation.argumentsSha256)).toEqual({ operationId });
    expect(() => store.request(operation.requestId, "d".repeat(64))).toThrow(/different operands/i);
    expect(store.recoverInterrupted()).toBe(1);
    expect(store.getOperation(operationId)?.state).toBe("interrupted");
    expect(store.getEnvironment(environmentId)?.state).toBe("review-required");
    const result = { environment: store.getEnvironment(environmentId)!, operation: store.getOperation(operationId)! };
    store.finishRequest(operation.requestId, result);
    expect(store.request(operation.requestId, operation.argumentsSha256)?.result).toEqual(result);
    store.close();
  });

  it("rejects SQL index and lost-ACK result bodies that differ from their checksummed ledger payload", async () => {
    const path = await root(), store = await EnvironmentStore.open(path, brainId), { environment, operation } = records(); await mkdir(store.instanceDirectory(environment.id), { recursive: true });
    const complete = { ...operation, state: "complete" as const, updatedAt: new Date(Date.parse(operation.updatedAt) + 1000).toISOString() };
    store.transaction(() => { store.putEnvironment({ ...environment, state: "loaded" }); store.putOperation(complete); store.beginRequest(complete.requestId, complete.argumentsSha256, complete.id);
      store.finishRequest(complete.requestId, { environment: { ...environment, state: "loaded" }, operation: complete }); });
    const result = store.request(complete.requestId, complete.argumentsSha256)!.result!;
    store.db.prepare("UPDATE requests SET result=? WHERE id=?").run(JSON.stringify({ ...result, operation: { ...result.operation, state: "failed", error: "forged" } }), complete.requestId);
    expect(() => store.request(complete.requestId, complete.argumentsSha256)).toThrow(/differs from the retained ledger/i);
    store.db.prepare("UPDATE environments SET kind='vm' WHERE id=?").run(environment.id);
    expect(() => store.getEnvironment(environment.id)).toThrow(/SQL index differs/i); store.close();
  });

  it("repairs app-owned environment directories to private mode before opening state", async context => {
    if (process.platform === "win32") { context.skip(); return; }
    const path = await root(), base = join(path, "environments"), instances = join(base, "instances"); await mkdir(instances, { recursive: true }); await chmod(base, 0o755); await chmod(instances, 0o755);
    const store = await EnvironmentStore.open(path, brainId); store.close();
    expect((await stat(base)).mode & 0o077).toBe(0); expect((await stat(instances)).mode & 0o077).toBe(0);
  });
});
