import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ToolExecutor } from "../src/main/toolExecutor";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import type { ToolInvocation, ToolPermissionRecord } from "../src/shared/types";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(path => rm(path, { recursive: true, force: true }))); });
async function fixture(level: ToolPermissionRecord["level"] = "auto") {
  const root = await mkdtemp(join(tmpdir(), "omni-environment-tool-")); roots.push(root);
  const brainDirectory = join(root, "brain-one"); await mkdir(brainDirectory);
  let permission: ToolPermissionRecord = { toolId: "runtime.sandbox", label: "Managed sandbox", level, updatedAt: "2026-10-13T12:00:00.000Z" };
  const listeners: Array<(event: { brainId: string; toolId: string }) => void> = [];
  const service = { repository: { brainDirectory: () => brainDirectory }, listToolPermissions: async () => [permission], authorizeInlineImagination: async () => undefined,
    onToolPermissionChanged: (listener: typeof listeners[number]) => { listeners.push(listener); return () => undefined; } } as unknown as BrainService;
  const tools = new ToolExecutor(service, {} as RuntimeJobManager); vi.spyOn(tools as any, "audit").mockResolvedValue(undefined);
  const adapter = { execute: vi.fn(async () => ({ actual: true })) }; tools.setEnvironmentService(adapter);
  const invocation: ToolInvocation = { brainId: "brain-one", toolId: "runtime.sandbox", action: "create", arguments: { name: "fixture" } };
  return { tools, adapter, invocation, set(level: ToolPermissionRecord["level"]) { permission = { ...permission, level, updatedAt: new Date(Date.parse(permission.updatedAt) + 1000).toISOString() }; listeners.forEach(listener => listener({ brainId: "brain-one", toolId: "runtime.sandbox" })); } };
}

describe("managed environment ToolExecutor boundary", () => {
  it("keeps mutation risky under Auto and passes only the exact current turn after one-use review", async () => {
    const f = await fixture(), complete = vi.fn(async () => undefined);
    const pending = await f.tools.execute(f.invocation, undefined, "environment-request", complete, { chatTurnId: "turn-one", actionEventId: "action-one" });
    expect(pending.state).toBe("approval-required"); expect(f.adapter.execute).not.toHaveBeenCalled();
    const result = await f.tools.execute({ ...f.invocation, approvalToken: pending.approvalToken }, undefined, "environment-request", complete, { chatTurnId: "turn-one", actionEventId: "action-one" });
    expect(result, JSON.stringify(result)).toMatchObject({ state: "complete", output: { actual: true } });
    expect(f.adapter.execute).toHaveBeenCalledWith("brain-one", "runtime.sandbox", "create", { name: "fixture" }, "environment-request", expect.any(AbortSignal), { chatTurnId: "turn-one" });
  });

  it("denies Off before adapter dispatch and permits read-only capability inspection under Auto", async () => {
    const f = await fixture("off");
    expect((await f.tools.execute(f.invocation)).state).toBe("failed"); expect(f.adapter.execute).not.toHaveBeenCalled();
    f.set("auto"); const capabilities = await f.tools.execute({ ...f.invocation, action: "capabilities", arguments: {} }, undefined, "capabilities-request");
    expect(capabilities.state, JSON.stringify(capabilities)).toBe("complete"); expect(f.adapter.execute).toHaveBeenCalledTimes(1);
  });
});
