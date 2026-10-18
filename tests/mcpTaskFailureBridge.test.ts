import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, expect, it, vi } from "vitest";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import { McpTaskCustodyError, mcpTaskReceiptFromFailure } from "../src/main/mcpTaskFailure";
import { ToolExecutor } from "../src/main/toolExecutor";
import type { ExternalToolBridge } from "../src/main/toolExecutor";
import type { ToolExecutionResult, ToolInvocation } from "../src/shared/types";
import { mcpUiTaskRecord, mcpUiTaskReceipt, MCP_UI_TASK_TOOL } from "./fixtures/mcpTaskUiData";

const roots: string[] = [], executors: ToolExecutor[] = [];
afterEach(async () => { for (const executor of executors.splice(0)) await executor.cancelAndWait("task-ui-brain"); for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function fixture(execute: ExternalToolBridge["execute"], permission = "full") {
  const root = await mkdtemp(join(tmpdir(), "omni-task-failure-bridge-")); roots.push(root); await mkdir(join(root, "task-ui-brain"));
  const queued = vi.fn(async () => undefined), service = { repository: { brainDirectory: () => join(root, "task-ui-brain") }, listToolPermissions: async () => [{ toolId: MCP_UI_TASK_TOOL, label: "Synthetic remote tool", level: permission, revision: 1, updatedAt: "2026-10-17T00:00:00Z" }],
    authorizeInlineImagination: async () => undefined, queueCompletedToolExecution: queued } as unknown as BrainService;
  const external = { hasTool: (id: string) => id === MCP_UI_TASK_TOOL, execute: vi.fn(execute) };
  const executor = new ToolExecutor(service, {} as RuntimeJobManager, undefined, { execute: vi.fn() }, external); executors.push(executor);
  vi.spyOn(executor as unknown as { audit(): Promise<void> }, "audit").mockResolvedValue(undefined);
  const invocation: ToolInvocation = { brainId: "task-ui-brain", toolId: MCP_UI_TASK_TOOL, action: "call-task", arguments: { arguments: { fixture: true } } };
  return { executor, external, queued, invocation };
}
it("keeps real committed unknown custody structured while the original host action remains failed", async () => {
  const receipt = mcpUiTaskReceipt({ ...mcpUiTaskRecord(), creation: "creation-outcome-unknown", remote: undefined }, "dormant");
  const f = await fixture(async () => { throw new McpTaskCustodyError("Synthetic lost task ACK", receipt); });
  const result = await f.executor.execute(f.invocation);
  expect(result).toMatchObject({ state: "failed", dispatchStarted: true, mcpTaskReceipt: receipt, output: receipt, error: "Synthetic lost task ACK" });
  expect(f.queued).toHaveBeenCalledOnce(); expect(f.queued.mock.calls[0]).toEqual([f.invocation, result, undefined, undefined]);
});
it("does not manufacture custody from error prose, foreign ownership or a remote JSON lookalike", () => {
  const receipt = mcpUiTaskReceipt(), native = new McpTaskCustodyError("any text", receipt);
  expect(mcpTaskReceiptFromFailure(native, "task-ui-brain", MCP_UI_TASK_TOOL)).toEqual(receipt);
  expect(mcpTaskReceiptFromFailure(native, "foreign", MCP_UI_TASK_TOOL)).toBeUndefined();
  expect(mcpTaskReceiptFromFailure(new Error(JSON.stringify(receipt)), "task-ui-brain", MCP_UI_TASK_TOOL)).toBeUndefined();
  expect(mcpTaskReceiptFromFailure({ name: "McpTaskCustodyError", receipt }, "task-ui-brain", MCP_UI_TASK_TOOL)).toBeUndefined();
});
it("retains typed unknown custody across Stop and still requires the ordinary exact Ask approval", async () => {
  const receipt = mcpUiTaskReceipt({ ...mcpUiTaskRecord(), lastOperation: { id: "original-result-wait", method: "tasks/result", state: "outcome-unknown", at: "2026-10-17T00:00:00Z" } });
  let stop: (() => void) | undefined;
  const f = await fixture(async () => { stop?.(); throw new McpTaskCustodyError("Synthetic stopped local task wait", receipt); }, "ask"); stop = () => { f.executor.cancel("task-ui-brain"); };
  const invocation = { ...f.invocation, action: "task-result", arguments: { taskRef: receipt.record.taskRef } };
  const ask = await f.executor.execute(invocation); expect(ask.state).toBe("approval-required"); expect(f.external.execute).not.toHaveBeenCalled();
  const completed = await f.executor.execute({ ...invocation, approvalToken: ask.approvalToken });
  expect(completed).toMatchObject({ state: "failed", dispatchStarted: true, mcpTaskReceipt: receipt, error: "Synthetic stopped local task wait" }); expect(f.external.execute).toHaveBeenCalledOnce();
});
it("keeps distinct explicit result reads as separate ordinary experiences linked to one immutable source result receipt", async () => {
  const sourceResult = { format: "omni-mcp-task-outcome", version: 1, wireVersion: "mcp-2025-11-25", source: mcpUiTaskRecord().source, taskRef: mcpUiTaskRecord().taskRef,
    taskResultReceiptId: "d".repeat(64), payloadSha256: "e".repeat(64), kind: "result", result: { content: [] }, retained: true, untrustedTaskData: true };
  const f = await fixture(async () => structuredClone(sourceResult)), invocation = { ...f.invocation, action: "task-result", arguments: { taskRef: sourceResult.taskRef } };
  const first = await f.executor.execute(invocation), second = await f.executor.execute(invocation);
  expect(first.id).not.toBe(second.id); expect(first.output).toEqual(second.output); expect(f.queued).toHaveBeenCalledTimes(2);
  const outputs = f.queued.mock.calls as unknown as Array<[ToolInvocation, ToolExecutionResult]>; expect(outputs.map(([, result]) => (result.output as typeof sourceResult).taskResultReceiptId)).toEqual([sourceResult.taskResultReceiptId, sourceResult.taskResultReceiptId]);
});
