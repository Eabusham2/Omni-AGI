import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Ajv from "ajv";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ToolExecutor } from "../src/main/toolExecutor";
import { ClipboardOperationError } from "../src/main/clipboardService";
import { declaredClipboardCapabilityPack } from "../src/main/clipboardCapabilityPack";
import { DEFAULT_TOOL_PERMISSIONS } from "../src/main/brainRepository";
import { CLIPBOARD_FORMAT, type ClipboardMutationReceipt } from "../src/shared/clipboard";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import type { ToolInvocation, ToolPermissionRecord } from "../src/shared/types";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
async function fixture(instance: ToolPermissionRecord["level"], host: ToolPermissionRecord["level"]) {
  const root = await mkdtemp(join(tmpdir(), "omni-clipboard-tool-")); roots.push(root); const directory = join(root, "brain-one"); await mkdir(directory);
  const permissions = new Map<string, ToolPermissionRecord>([["studio.clipboard", { toolId: "studio.clipboard", label: "Instance clipboard", level: instance, updatedAt: new Date(0).toISOString() }],
    ["system.clipboard", { toolId: "system.clipboard", label: "Host clipboard", level: host, updatedAt: new Date(0).toISOString() }]]), listeners: Array<(event: { brainId: string; toolId: string }) => void> = [];
  const service = { repository: { brainDirectory: () => directory }, listToolPermissions: async () => [...permissions.values()], authorizeInlineImagination: async () => undefined,
    onToolPermissionChanged: (listener: typeof listeners[number]) => { listeners.push(listener); return () => undefined; } } as unknown as BrainService;
  const tools = new ToolExecutor(service, {} as RuntimeJobManager); vi.spyOn(tools as any, "audit").mockResolvedValue(undefined); const complete = vi.fn(async () => undefined);
  return { tools, complete, set(toolId: "studio.clipboard" | "system.clipboard", level: ToolPermissionRecord["level"]) { const row = permissions.get(toolId)!; permissions.set(toolId, { ...row, level, revision: (row.revision ?? 0) + 1 }); listeners.forEach(listener => listener({ brainId: "brain-one", toolId })); },
    execute: (invocation: ToolInvocation, requestId?: string) => tools.execute(invocation, undefined, requestId, complete) };
}
const invocation = (toolId: "studio.clipboard" | "system.clipboard", action = "read"): ToolInvocation => ({ brainId: "brain-one", toolId, action, arguments: { scope: toolId === "studio.clipboard" ? "instance" : "host" } });

describe("clipboard ToolExecutor authority and declared material", () => {
  it("defaults private instance clipboard Auto and host clipboard Off", async () => {
    expect(DEFAULT_TOOL_PERMISSIONS.find(row => row.toolId === "studio.clipboard")?.level).toBe("auto");
    expect(DEFAULT_TOOL_PERMISSIONS.find(row => row.toolId === "system.clipboard")?.level).toBe("off");
    const f = await fixture("auto", "off"), adapter = { execute: vi.fn(async () => ({ actual: true })) }; f.tools.setClipboardService(adapter);
    expect((await f.execute(invocation("studio.clipboard"))).state).toBe("complete");
    expect((await f.execute(invocation("system.clipboard"))).state).toBe("failed"); expect(adapter.execute).toHaveBeenCalledTimes(1);
  });
  it("reviews host content/effects under Auto and binds the exact scope operands", async () => {
    const f = await fixture("auto", "auto"), adapter = { execute: vi.fn(async () => ({ actual: true })) }; f.tools.setClipboardService(adapter);
    const first = await f.execute(invocation("system.clipboard")); expect(first.state).toBe("approval-required");
    const changed = { ...invocation("system.clipboard"), arguments: { scope: "instance" }, approvalToken: first.approvalToken };
    expect((await f.execute(changed)).state).toBe("approval-required"); expect(adapter.execute).not.toHaveBeenCalled();
  });
  it("preserves an uncertain host mutation receipt across permission revocation", async () => {
    const f = await fixture("auto", "full"); let entered = false;
    const receipt: ClipboardMutationReceipt = { format: CLIPBOARD_FORMAT, version: 1, action: "write", requestId: "clipboard-effect", requestSha256: "a".repeat(64), scope: "host",
      state: "write-uncertain", contentType: "text/plain", contentBytes: 1, contentSha256: "b".repeat(64), writeMayHaveOccurred: true, retrySafe: false,
      provenance: { kind: "host-os-clipboard", capturedAt: new Date().toISOString() }, error: "fixture unknown effect" };
    f.tools.setClipboardService({ execute: async (_brain, _tool, _action, _args, _request, signal) => { entered = true;
      await new Promise<void>((_resolve, reject) => signal.addEventListener("abort", () => reject(new ClipboardOperationError("revoked", receipt)), { once: true })); } });
    const pending = f.execute({ ...invocation("system.clipboard", "write"), arguments: { scope: "host", contentType: "text/plain", text: "x" } }, "clipboard-effect");
    await vi.waitFor(() => expect(entered).toBe(true)); f.set("system.clipboard", "off"); const result = await pending;
    expect(result).toMatchObject({ state: "failed", dispatchStarted: true, output: receipt }); expect(f.complete).toHaveBeenCalledWith(result);
  });
  it("declares schema-valid ordinary clipboard actions without fabricated host outcomes", () => {
    const pack = declaredClipboardCapabilityPack(), ajv = new Ajv({ strict: false }); expect(pack).toEqual(declaredClipboardCapabilityPack());
    expect(pack).toMatchObject({ id: "clipboard-tools-1", executedHostActions: false, rewardModel: false, preferenceLabels: false });
    expect(pack.schemas.map(schema => schema.id)).toEqual(["studio.clipboard", "system.clipboard"]);
    for (const record of pack.records) { const schema = pack.schemas.find(item => item.id === record.toolId)?.actionInputSchemas[record.action!]; expect(schema).toBeDefined(); expect(ajv.validate(schema!, record.arguments), JSON.stringify(ajv.errors)).toBe(true); }
  });
});
