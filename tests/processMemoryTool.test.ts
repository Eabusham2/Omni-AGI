/** Actual ToolExecutor policy/receipt path; OS calls are explicit fixture adapters. */
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ToolExecutor } from "../src/main/toolExecutor";
import { ProcessMemoryOperationError, uncertainMemoryReceipt } from "../src/main/processMemoryBridge";
import { declaredProcessMemoryCapabilityPack } from "../src/main/processMemoryCapabilityPack";
import { DEFAULT_TOOL_PERMISSIONS } from "../src/main/brainRepository";
import { PROCESS_MEMORY_FORMAT } from "../src/shared/processMemory";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import type { ToolInvocation, ToolPermissionRecord } from "../src/shared/types";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function fixture(level: ToolPermissionRecord["level"]) {
  const root = await mkdtemp(join(tmpdir(), "omni-memory-tool-")); roots.push(root); const directory = join(root, "brain-one"); await mkdir(directory);
  let permission: ToolPermissionRecord = { toolId: "system.memory", label: "DMA", level, updatedAt: "2026-10-13T12:00:00Z" };
  const listeners: Array<(event: { brainId: string; toolId: string }) => void> = [];
  const service = { repository: { brainDirectory: () => directory }, listToolPermissions: async () => [permission],
    authorizeInlineImagination: async () => undefined, onToolPermissionChanged: (listener: typeof listeners[number]) => { listeners.push(listener); return () => undefined; } } as unknown as BrainService;
  const tools = new ToolExecutor(service, {} as RuntimeJobManager);
  vi.spyOn(tools as unknown as { audit(...args: unknown[]): Promise<void> }, "audit").mockResolvedValue(undefined);
  const invocation: ToolInvocation = { brainId: "brain-one", toolId: "system.memory", action: "read", arguments: { targetId: "fixture-lease", address: "0x1000", length: 4 } };
  const complete = vi.fn(async () => undefined);
  return { tools, invocation, complete,
    setPermission(level: ToolPermissionRecord["level"]) { permission = { ...permission, level, revision: (permission.revision ?? 0) + 1 }; listeners.forEach(listener => listener({ brainId: "brain-one", toolId: "system.memory" })); },
    execute: (value = invocation, requestId?: string) => tools.execute(value, undefined, requestId, complete) };
}

describe("DMA native tool authority wiring", () => {
  it("is default Off and disabled calls never reach the memory adapter", async () => {
    expect(DEFAULT_TOOL_PERMISSIONS.find(row => row.toolId === "system.memory")?.level).toBe("off");
    const f = await fixture("off"), adapter = { execute: vi.fn(async () => ({})) }; f.tools.setProcessMemoryService(adapter);
    expect((await f.execute()).state).toBe("failed"); expect(adapter.execute).not.toHaveBeenCalled();
  });
  it("reviews every attaching/regions/read/write operation under Auto and binds exact operands", async () => {
    const f = await fixture("auto"), adapter = { execute: vi.fn(async () => ({ syntheticReceipt: true })) }; f.tools.setProcessMemoryService(adapter);
    for (const action of ["attach", "regions", "read", "write"]) {
      const result = await f.execute({ ...f.invocation, action }); expect(result.state).toBe("approval-required");
    }
    expect(adapter.execute).not.toHaveBeenCalled();
    const first = await f.execute(), changed = { ...f.invocation, arguments: { ...f.invocation.arguments, address: "0x1004" }, approvalToken: first.approvalToken };
    const second = await f.execute(changed); expect(second.state).toBe("approval-required");
    expect((await f.execute({ ...changed, approvalToken: second.approvalToken })).state).toBe("complete");
    expect(adapter.execute).toHaveBeenCalledTimes(1);
  });
  it("preserves a typed uncertain write even when revocation aborts dispatch", async () => {
    const f = await fixture("full"); let entered = false;
    const receipt = uncertainMemoryReceipt({ format: PROCESS_MEMORY_FORMAT, operation: "write", operationId: "memory-fixture", pid: 123456,
      access: "read-write", address: "0x1000", length: 4, dataBase64: "V1hZWg==", expectedBeforeSha256: "a".repeat(64) }, true, "fixture result lost after dispatch");
    f.tools.setProcessMemoryService({ execute: async (_brain, _action, _args, _requestId, signal) => {
      entered = true; await new Promise<void>((_resolve, reject) => signal.addEventListener("abort", () => reject(new ProcessMemoryOperationError("Access revoked after attempted write", receipt)), { once: true })); return {};
    } });
    const pending = f.execute({ ...f.invocation, action: "write" }, "memory-fixture"); await vi.waitFor(() => expect(entered).toBe(true)); f.setPermission("off");
    const result = await pending; expect(result.state).toBe("failed"); expect(result.dispatchStarted).toBe(true);
    expect(result.output).toBe(receipt); expect(f.complete).toHaveBeenCalledWith(result);
  });
  it("declares truthful capability teaching with no executed target or hidden grant", () => {
    const pack = declaredProcessMemoryCapabilityPack();
    expect(pack.executedHostActions).toBe(false); expect(pack.rewardModel).toBe(false);
    expect(pack.schemas[0]?.grant).toBe("off");
    expect(pack.records.map(row => row.action)).toEqual(["capabilities", "attach", "regions", "read", "write", "detach"]);
    expect(pack.records.some(row => /not atomic/.test(row.utterance))).toBe(true);
  });
});
