/** Real ToolExecutor policy/cancellation and one tiny safe installed native fixture. */
import { mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir, totalmem } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ToolExecutor, runProcess } from "../src/main/toolExecutor";
import { executeAssembly, AssemblyExecutionError, type AssemblyReceipt } from "../src/main/assemblyRuntime";
import { automationResidentBytes, requireAutomationHeadroom } from "../src/main/automationResources";
import { requireDiskWrite } from "../src/main/diskSpace";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import type { ToolInvocation, ToolPermissionRecord } from "../src/shared/types";
const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function fixture(level: ToolPermissionRecord["level"] = "ask") {
  const root = await mkdtemp(join(tmpdir(), "omni-assembly-tool-")); roots.push(root); const directory = join(root, "brain-one"); await mkdir(directory);
  let permission: ToolPermissionRecord = { toolId: "code.assembly", label: "Assembly", level, updatedAt: "2026-10-11T12:00:00Z" };
  const listeners: Array<(event: { brainId: string; toolId: string }) => void> = [], nativeGet = vi.fn(() => { throw new Error("Assembly must not Load a brain"); });
  const service = { repository: { brainDirectory: () => directory }, get: nativeGet, listToolPermissions: async () => [permission], authorizeInlineImagination: async () => undefined,
    onToolPermissionChanged: (listener: typeof listeners[number]) => { listeners.push(listener); return () => undefined; } } as unknown as BrainService;
  const tools = new ToolExecutor(service, {} as RuntimeJobManager); vi.spyOn(tools as unknown as { audit(...args: unknown[]): Promise<void> }, "audit").mockResolvedValue(undefined);
  const invocation: ToolInvocation = { brainId: "brain-one", toolId: "code.assembly", action: "compile", arguments: { source: ".text\n" } };
  const complete = vi.fn(async () => undefined);
  return { root, directory, tools, invocation, complete, nativeGet, setPermission(level: ToolPermissionRecord["level"]) { permission = { ...permission, level, updatedAt: new Date(Date.parse(permission.updatedAt) + 1000).toISOString() }; listeners.forEach(listener => listener({ brainId: "brain-one", toolId: "code.assembly" })); },
    execute: (value = invocation, requestId?: string) => tools.execute(value, undefined, requestId, complete) };
}
describe("assembly policy and actual native process boundary", () => {
  it("keeps compile/run risky under Auto and binds one-use Ask approval to exact source", async () => {
    const f = await fixture("auto"), adapter = { execute: vi.fn(async () => ({ state: "compiled", fixtureOnly: true })) }; f.tools.setAssemblyService(adapter);
    const prompt = await f.execute(); expect(prompt.state).toBe("approval-required"); expect(adapter.execute).not.toHaveBeenCalled();
    const changed = { ...f.invocation, arguments: { source: ".text\n// different exact source" }, approvalToken: prompt.approvalToken };
    const renewed = await f.execute(changed); expect(renewed.state).toBe("approval-required"); expect(adapter.execute).not.toHaveBeenCalled();
    expect((await f.execute({ ...changed, approvalToken: renewed.approvalToken })).state).toBe("complete"); expect(adapter.execute).toHaveBeenCalledTimes(1); expect(f.nativeGet).not.toHaveBeenCalled();
    const run = await f.execute({ ...f.invocation, action: "run", arguments: { artifactId: "a".repeat(32), artifactSha256: "b".repeat(64) } }); expect(run.state).toBe("approval-required");
  });
  it("denies Off and stale approvals before adapter dispatch", async () => {
    const f = await fixture(), adapter = { execute: vi.fn(async () => ({})) }; f.tools.setAssemblyService(adapter);
    const pending = await f.execute(); f.setPermission("off"); expect((await f.execute({ ...f.invocation, approvalToken: pending.approvalToken })).state).toBe("failed"); expect(adapter.execute).not.toHaveBeenCalled();
  });
  it("retains typed partial native failure output when permission changes during dispatch", async () => {
    const f = await fixture("full"); let entered = false;
    const receipt = { state: "cancelled", receiptPath: join(f.directory, "fixture-cancelled.json"), partialFiles: [{ name: "source.s", bytes: 6, sha256: "a".repeat(64) }] } as AssemblyReceipt;
    f.tools.setAssemblyService({ execute: async (_brain, _action, _args, _request, signal) => { entered = true; await new Promise<void>((_resolve, reject) => signal.addEventListener("abort", () => reject(new AssemblyExecutionError("Current access revoked", receipt)), { once: true })); return {}; } });
    const pending = f.execute(f.invocation, "assembly-owned-request"); await vi.waitFor(() => expect(entered).toBe(true)); f.setPermission("off");
    const result = await pending; expect(result.state).toBe("failed"); expect(result.dispatchStarted).toBe(true); expect(result.output).toBe(receipt); expect(result.error).toContain("Retained receipt"); expect(f.complete).toHaveBeenCalledWith(result);
  });
  it("compiles and runs one harmless printed-text fixture with an actually installed native compiler", async context => {
    if (process.platform !== "darwin" || process.arch !== "arm64") { context.skip(); return; }
    const f = await fixture();
    f.tools.setAssemblyService({ execute: (_brain, action, args, _request, signal) => executeAssembly({ root: f.directory, run: runProcess,
      checkResources: async (bytes, _purpose, children = []) => requireAutomationHeadroom(bytes, { residentBytes: await automationResidentBytes([process.pid, ...children]), ceilingBytes: Math.floor(totalmem() * .3) }),
      reserveWrite: async bytes => { await requireDiskWrite(f.directory, { operationWriteBytes: bytes }); }
    }, action, args, signal) });
    const inspect = await f.execute({ ...f.invocation, action: "inspect", arguments: {} });
    const approvedInspect = inspect.state === "approval-required" ? await f.execute({ ...f.invocation, action: "inspect", arguments: {}, approvalToken: inspect.approvalToken }) : inspect;
    if (!(approvedInspect.output as any)?.toolchains?.some((row: any) => row.id === "clang" && row.available && !row.probeError)) { context.skip(); return; }
    const source = ".section __TEXT,__text,regular,pure_instructions\n.globl _main\n.p2align 2\n_main:\n stp x29, x30, [sp, #-16]!\n mov x29, sp\n adrp x0, message@PAGE\n add x0, x0, message@PAGEOFF\n bl _puts\n mov w0, #0\n ldp x29, x30, [sp], #16\n ret\n.section __TEXT,__cstring,cstring_literals\nmessage:\n .asciz \"Omni safe assembly fixture\"\n";
    const invocation = { ...f.invocation, arguments: { source, syntax: "gas", kind: "executable", toolchain: "clang", timeoutMs: 10000 } };
    const request = await f.execute(invocation), compiledResult = await f.execute({ ...invocation, approvalToken: request.approvalToken });
    expect(compiledResult.state).toBe("complete"); const compiled = compiledResult.output as AssemblyReceipt; expect(compiled.state, JSON.stringify(compiled)).toBe("compiled");
    const run: ToolInvocation = { ...f.invocation, action: "run", arguments: { artifactId: compiled.artifactId, artifactSha256: compiled.artifact!.sha256, arguments: [], timeoutMs: 10000 } };
    const runPrompt = await f.execute(run); expect(runPrompt.state).toBe("approval-required");
    const result = await f.execute({ ...run, approvalToken: runPrompt.approvalToken }); expect(result.state).toBe("complete"); const receipt = result.output as AssemblyReceipt;
    expect(receipt.state).toBe("exited"); expect(receipt.processes[0]?.exitCode).toBe(0); expect(receipt.processes[0]?.stdout).toBe("Omni safe assembly fixture\n");
    expect(JSON.parse(await readFile(receipt.receiptPath, "utf8"))).toMatchObject({ state: "exited", outputHashScope: "captured-bounded-output" }); expect(f.nativeGet).not.toHaveBeenCalled();
  }, 30_000);
});
