/** Actual small SQLite settings/effect journal; native calls explicitly mocked. */
import { lstat, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ProcessMemoryService } from "../src/main/processMemoryService";
import { ProcessMemoryStore } from "../src/main/processMemoryStore";
import { ProcessMemoryOperationError, uncertainMemoryReceipt } from "../src/main/processMemoryBridge";
import { memoryAddress, memoryReceipt, processMemoryCanonical, processMemoryHash } from "../src/main/processMemoryFormat";
import { PROCESS_MEMORY_FORMAT, type ProcessMemoryNativeRequest, type ProcessMemoryReceipt, type ProcessMemoryTarget } from "../src/shared/processMemory";

const roots: string[] = [];
afterEach(async () => { vi.restoreAllMocks(); for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
const identity = { platform: "linux" as const, pid: 123456, creation: "fixture-incarnation", executable: "/fixture/owned", executableIdentity: "1:2" };
const bytes = Buffer.from("abcd");
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-memory-fixture-")); roots.push(root);
  const state = { permission: { level: "full" as "full" | "off", revision: "fixture-permission-1" }, beforeReady: undefined as (() => Promise<void>) | undefined,
    afterCommit: undefined as (() => Promise<void>) | undefined, readState: "complete" as "complete" | "partial", actualWrites: 0 };
  const checkResources = vi.fn(async () => undefined);
  const run = vi.fn(async (request: ProcessMemoryNativeRequest, context: Parameters<ConstructorParameters<typeof ProcessMemoryService>[0]["run"]>[1]) => {
    await state.beforeReady?.();
    await context.beforeCommit({ format: PROCESS_MEMORY_FORMAT, type: "ready", operationId: request.operationId,
      requestSha256: processMemoryHash(processMemoryCanonical(request)), nonce: "a".repeat(48), identity }, 234567);
    if (request.operation === "write") state.actualWrites++;
    await state.afterCommit?.();
    return { format: PROCESS_MEMORY_FORMAT, operationId: request.operationId, operation: request.operation,
      requestSha256: processMemoryHash(processMemoryCanonical(request)), identity, state: request.operation === "read" ? state.readState : "complete",
      requestedBytes: request.length ?? 0, transferredBytes: request.length ?? 0, writeMayHaveOccurred: request.operation === "write",
      retrySafe: request.operation !== "write", compareAtomic: false,
      ...(request.operation === "read" ? { dataBase64: bytes.toString("base64"), dataSha256: processMemoryHash(bytes) } : {}),
      ...(request.operation === "write" ? { beforeSha256: processMemoryHash(bytes), afterSha256: processMemoryHash(Buffer.from(request.dataBase64!, "base64")) } : {}) } as ProcessMemoryReceipt;
  });
  const dependencies = { installationDirectory: root, platform: "linux" as const, assertBrainOwner: vi.fn(async () => undefined),
    currentPermission: async () => state.permission, protectedPids: () => [345678], checkResources, reserveWrite: vi.fn(async () => undefined), run };
  const service = new ProcessMemoryService(dependencies), signal = new AbortController().signal;
  const enable = async (brainId = "brain") => { const prefs = await service.preferences(brainId); return service.update({ brainId, expectedRevision: prefs.revision, enabled: true }); };
  const attach = async (access: "read" | "read-write" = "read-write") => (await service.execute("brain", "attach", { pid: identity.pid, access }, "attach", signal) as { target: ProcessMemoryTarget }).target;
  return { root, dependencies, state, run, service, signal, enable, attach, checkResources };
}

describe("opt-in OS process-memory authority", () => {
  it("actual lifecycle closure between preparation and commit does not implicitly reload", async () => {
    const f = await fixture(); await f.enable(); let loaded = true;
    const requireActiveOwner = vi.fn(async () => { if (!loaded) throw new Error("fixture explicitly unloaded"); });
    Object.assign(f.dependencies, { requireActiveOwner });
    f.state.beforeReady = async () => { loaded = false; };
    await expect(f.attach()).rejects.toThrow(/unloaded/); expect(f.state.actualWrites).toBe(0);
    expect(requireActiveOwner).toHaveBeenCalled();
  });
  it("a failed terminal journal save preserves the real write outcome and prevents replay", async () => {
    const f = await fixture(); await f.enable(); const target = await f.attach();
    const save = vi.spyOn(ProcessMemoryStore.prototype, "terminal").mockImplementationOnce(() => { throw new Error("fixture receipt disk-full"); });
    const args = { targetId: target.id, address: "0x1000", dataBase64: "V1hZWg==", expectedBeforeSha256: processMemoryHash(bytes) };
    try { await f.service.execute("brain", "write", args, "save-failed", f.signal); throw new Error("expected rejection"); }
    catch (error) { expect((error as ProcessMemoryOperationError).receipt).toMatchObject({ state: "complete", transferredBytes: 4, writeMayHaveOccurred: true, retrySafe: false }); }
    save.mockRestore();
    await expect(f.service.execute("brain", "write", args, "save-failed", f.signal)).rejects.toThrow(/not repeated/);
    expect(f.state.actualWrites).toBe(1);
  });
  it("inspection of default settings creates no host-authority file", async () => {
    const f = await fixture(); await f.service.preferences("brain");
    await expect(lstat(join(f.root, "process-memory.sqlite3"))).rejects.toMatchObject({ code: "ENOENT" });
  });
  it("disabling immediately revokes a lease even when persistence is refused for low disk", async () => {
    const f = await fixture(); await f.enable(); const target = await f.attach();
    f.dependencies.reserveWrite.mockRejectedValueOnce(new Error("fixture low disk"));
    await expect(f.service.update({ brainId: "brain", expectedRevision: 1, enabled: false })).rejects.toThrow(/low disk/);
    expect((await f.service.preferences("brain")).enabled).toBe(false);
    await expect(f.service.execute("brain", "read", { targetId: target.id, address: "0x1000", length: 4 }, "revoked-disk", f.signal)).rejects.toThrow(/off/);
  });
  it("is off by default and attachment cannot turn it on", async () => {
    const f = await fixture(); expect((await f.service.preferences("brain")).enabled).toBe(false);
    await expect(f.attach()).rejects.toThrow(/off/); expect(f.run).not.toHaveBeenCalled();
    expect((await f.service.preferences("copy")).enabled).toBe(false);
  });
  it("uses CAS settings and refuses stale updates", async () => {
    const f = await fixture(); await f.enable();
    await expect(f.service.update({ brainId: "brain", expectedRevision: 0, enabled: false })).rejects.toThrow(/changed/);
    expect((await f.service.preferences("brain")).enabled).toBe(true);
  });
  it("authorizes only the exact brain/session/OS creation lease and bounded addresses", async () => {
    const f = await fixture(); await f.enable(); const target = await f.attach();
    const receipt = await f.service.execute("brain", "read", { targetId: target.id, address: "0x1000", length: 4 }, "read", f.signal) as ProcessMemoryReceipt;
    expect(receipt.dataSha256).toBe(processMemoryHash(bytes));
    expect(f.run.mock.calls[1]![0].expectedIdentity).toEqual(identity);
    await f.enable("copy"); await expect(f.service.execute("copy", "read", { targetId: target.id, address: "0x1000", length: 4 }, "other", f.signal)).rejects.toThrow(/lease/);
    await expect(f.service.execute("brain", "read", { targetId: target.id, address: 4096, length: 4 }, "rounded", f.signal)).rejects.toThrow(/hexadecimal/);
  });
  it("never silently falls back from sandbox or targets Omni authority", async () => {
    const f = await fixture(); await f.enable();
    await expect(f.service.execute("brain", "attach", { pid: identity.pid, access: "read", environment: "sandbox" }, "sandbox", f.signal)).rejects.toThrow(/fallback/);
    for (const pid of [process.pid, 345678]) await expect(f.service.execute("brain", "attach", { pid, access: "read" }, `protected-${pid}`, f.signal)).rejects.toThrow(/protected/);
    expect(f.run).not.toHaveBeenCalled();
  });
  it("rechecks actual settings after prepared helper and before sending any commit", async () => {
    const f = await fixture(); await f.enable(); f.state.beforeReady = async () => { await f.service.update({ brainId: "brain", expectedRevision: 1, enabled: false }); };
    await expect(f.attach()).rejects.toThrow(/DMA|preference|authority|aborted/i);
    expect(f.state.actualWrites).toBe(0);
  });
  it("invalidates leases on permission revision and Unload; a new service cannot reuse one", async () => {
    const f = await fixture(); await f.enable(); const target = await f.attach();
    const action = (service: ProcessMemoryService, id: string) => service.execute("brain", "read", { targetId: target.id, address: "0x1000", length: 4 }, id, f.signal);
    f.state.permission = { level: "full", revision: "changed" }; await expect(action(f.service, "changed")).rejects.toThrow(/lease/);
    f.state.permission = { level: "full", revision: "fixture-permission-1" }; await f.service.forget("brain");
    await expect(action(f.service, "unloaded")).rejects.toThrow(/lease/);
    await expect(action(new ProcessMemoryService(f.dependencies), "restart")).rejects.toThrow(/lease/);
  });
  it("does not write through a read-only lease", async () => {
    const f = await fixture(); await f.enable(); const target = await f.attach("read");
    await expect(f.service.execute("brain", "write", { targetId: target.id, address: "0x1000", dataBase64: "V1hZWg==", expectedBeforeSha256: processMemoryHash(bytes) }, "write", f.signal)).rejects.toThrow(/read-only/);
    expect(f.state.actualWrites).toBe(0);
  });
  it("persists the exact write result and never repeats that request or changed operands", async () => {
    const f = await fixture(); await f.enable(); const target = await f.attach();
    const args = { targetId: target.id, address: "0x1000", dataBase64: "V1hZWg==", expectedBeforeSha256: processMemoryHash(bytes) };
    expect((await f.service.execute("brain", "write", args, "write", f.signal) as ProcessMemoryReceipt).state).toBe("complete");
    await expect(f.service.execute("brain", "write", args, "write", f.signal)).rejects.toThrow(/not repeated/);
    await expect(f.service.execute("brain", "write", { ...args, dataBase64: "MTIzNA==" }, "write", f.signal)).rejects.toThrow(/reused/);
    expect(f.state.actualWrites).toBe(1);
  });
  it("retains post-dispatch uncertainty and does not auto retry lost write ACKs", async () => {
    const f = await fixture(); await f.enable(); const target = await f.attach();
    f.state.afterCommit = async () => { throw new Error("fixture lost ACK after bounded write"); };
    const args = { targetId: target.id, address: "0x1000", dataBase64: "V1hZWg==", expectedBeforeSha256: processMemoryHash(bytes) };
    let error: ProcessMemoryOperationError | undefined;
    try { await f.service.execute("brain", "write", args, "lost", f.signal); } catch (caught) { error = caught as ProcessMemoryOperationError; }
    expect(error?.receipt).toMatchObject({ state: "write-uncertain", transferredBytes: null, writeMayHaveOccurred: true, retrySafe: false });
    await expect(f.service.execute("brain", "write", args, "lost", f.signal)).rejects.toThrow(/not repeated/);
    expect(f.state.actualWrites).toBe(1);
  });
  it("post-read revocation does not disclose bytes, but preserves real write receipts", async () => {
    const f = await fixture(); await f.enable(); const target = await f.attach();
    f.state.afterCommit = async () => { f.state.permission = { level: "off", revision: "revoked" }; };
    try { await f.service.execute("brain", "read", { targetId: target.id, address: "0x1000", length: 4 }, "revoked", f.signal); throw new Error("expected rejection"); }
    catch (error) { expect((error as ProcessMemoryOperationError).receipt.dataBase64).toBeUndefined(); expect((error as ProcessMemoryOperationError).receipt.dataSha256).toBe(processMemoryHash(bytes)); }
  });
  it("rejects forged successful count/identity/byte-hash receipts", () => {
    const request: ProcessMemoryNativeRequest = { format: PROCESS_MEMORY_FORMAT, operationId: "read", operation: "read", pid: identity.pid,
      access: "read", expectedIdentity: identity, address: "0x1000", length: 4 };
    const receipt = { ...uncertainMemoryReceipt(request, false, "fixture"), state: "complete", transferredBytes: 4, dataBase64: "YWJjZA==", dataSha256: processMemoryHash(bytes) };
    expect(memoryReceipt(receipt, request).dataSha256).toBe(processMemoryHash(bytes));
    for (const patch of [{ transferredBytes: 3 }, { dataSha256: "0".repeat(64) }, { identity: { ...identity, creation: "reused" } }]) expect(() => memoryReceipt({ ...receipt, ...patch }, request)).toThrow();
    expect(() => memoryAddress("0x0001")).toThrow();
  });
});
