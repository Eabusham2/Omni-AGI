import { mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ToolExecutor } from "../src/main/toolExecutor";
import { ChatActionController } from "../src/main/chatActionController";
import type { BrainService, NeuralChatStreamEvent } from "../src/main/brainService";
import type { BrainStorageOperationHooks } from "../src/main/brainStorageOperations";
import type { ChatResult, StructuredAction } from "../src/shared/types";

const roots: string[] = [], time = "2026-09-30T12:00:00.000Z";
afterEach(async () => { await Promise.all(roots.splice(0).map(path => rm(path, { recursive: true, force: true }))); });
const reply = (brain: string, objective: string, turnId: string): ChatResult => ({
  brain: { id: brain, messages: [], concepts: {}, synapses: {} } as unknown as ChatResult["brain"],
  humanMessage: { id: `human-${brain}`, turnId, role: "human", content: objective, createdAt: time },
  brainMessage: { id: `reply-${brain}`, turnId, role: "brain", content: "host fixture reply", createdAt: time },
  trace: { id: `trace-${brain}` } as ChatResult["trace"] });
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-agent-runner-")); roots.push(root); const owner = join(root, "brain"); await mkdir(owner);
  let count = 0;
  const service = { repository: { brainDirectory: () => owner,
    fork: vi.fn(async (_parent: string, _name: string, operation?: BrainStorageOperationHooks) => {
      operation?.signal.throwIfAborted(); return { id: `fork-${++count}` };
    }) }, preflightStart: vi.fn(async () => undefined), chat: vi.fn(),
    authorizeInlineImagination: vi.fn(async () => undefined), queueCompletedToolExecution: vi.fn(async () => undefined) };
  const executor = Object.assign(Object.create(ToolExecutor.prototype), { service: service as unknown as BrainService,
    activeExecutions: new Map(), approvals: new Map(), permission: vi.fn(async () => ({ level: "ask", revision: "fixture-permission" })),
    audit: vi.fn(async () => undefined) }) as ToolExecutor;
  const direct = executor as unknown as { agent(brain: string, action: string, args: Record<string, unknown>, signal: AbortSignal): Promise<Record<string, unknown>> };
  return { root, service, executor, direct };
}

describe("same-brain trusted fork task runner", () => {
  it("executes ordinary permissioned child actions and keeps their exact receipts rather than dropping proposed actions", async () => {
    const value = await fixture();
    const toolAction: StructuredAction = { kind: "tool", source: "brain", toolId: "web.fetch", action: "fetch", actionId: "a".repeat(32), arguments: { url: "https://example.org" } };
    const childTools = { execute: vi.fn(async () => ({ id: randomUUID(), toolId: "web.fetch", action: "fetch", state: "complete" as const,
      startedAt: time, finishedAt: time, dispatchStarted: true as const, output: { actual: "stub owned tool output" } })), cancel: vi.fn(() => 0) };
    const childService = { chat: vi.fn(async (brain: string, objective: string, _signal?: AbortSignal,
      stream?: (event: NeuralChatStreamEvent) => void, turnId?: string) => {
      stream?.({ type: "chat-action", sequence: 0, actionId: toolAction.actionId, action: toolAction });
      return { ...reply(brain, objective, turnId!), proposedActions: [toolAction] };
    }) };
    const controller = new ChatActionController(childService, childTools, { start: vi.fn() });
    value.executor.setAgentChatRunner((brain, objective, signal, turnId) => controller.send(brain, objective, signal, turnId));
    const result = await value.direct.agent("brain", "start", { objective: "actual unwrapped task" }, new AbortController().signal);
    expect(childTools.execute).toHaveBeenCalledOnce(); expect(value.service.chat).not.toHaveBeenCalled();
    expect(childService.chat.mock.calls[0]?.[1]).toBe("actual unwrapped task");
    const actual = (result.results as Array<Record<string, unknown>>)[0]!;
    expect(actual).toMatchObject({ forkId: "fork-1", humanMessageId: "human-fork-1", brainMessageId: "reply-fork-1",
      actionEvents: [expect.objectContaining({ state: "complete", execution: expect.objectContaining({ dispatchStarted: true, output: { actual: "stub owned tool output" } }) })] });
    expect(actual).not.toHaveProperty("concepts"); expect(actual).not.toHaveProperty("synapses"); // no fabricated zero from obsolete mirrors
    expect(result).toMatchObject({ executionMode: "serial", mergePolicy: "ideas-evidence-replay-only" });
  });
  it("honors explicit5workers without silent4clipping and passes complete>20k objective with no96-token caller ceiling", async () => {
    const value = await fixture(), objective = `START:${"actual task ".repeat(2500)}:EXACT-END`;
    const runner = vi.fn(async (brain: string, text: string, _signal: AbortSignal, turnId: string) => reply(brain, text, turnId));
    value.executor.setAgentChatRunner(runner);
    const result = await value.direct.agent("brain", "start", { objective, workers: 5 }, new AbortController().signal);
    expect(value.service.repository.fork).toHaveBeenCalledTimes(5); expect(runner).toHaveBeenCalledTimes(5);
    expect(runner.mock.calls.every(call => call[1] === objective && call.length === 4)).toBe(true);
    expect(value.service.preflightStart.mock.calls).toHaveLength(5); expect(result.executionMode).toBe("serial");
    const source = await readFile(new URL("../src/main/toolExecutor.ts", import.meta.url), "utf8");
    expect(source).not.toContain("SUBAGENT_RESPONSE_TOKENS"); expect(source).not.toContain("Math.min(4, requestedWorkers)");
  });
  it("rejects invalid explicit worker counts and a missing trusted runner before any fork or fallback chat", async () => {
    const value = await fixture();
    await expect(value.direct.agent("brain", "start", { objective: "actual" }, new AbortController().signal)).rejects.toThrow("Trusted agent action runner");
    value.executor.setAgentChatRunner(vi.fn());
    for (const workers of [0, -1, 1.5, Infinity, "5", Number.MAX_SAFE_INTEGER + 1]) {
      await expect(value.direct.agent("brain", "start", { objective: "actual", workers }, new AbortController().signal)).rejects.toThrow("positive safe integer");
    }
    expect(value.service.repository.fork).not.toHaveBeenCalled(); expect(value.service.chat).not.toHaveBeenCalled();
  });
  it("parent Ask permission/consumed exact token still gates forks and does not automatically merge any child", async () => {
    const value = await fixture(); value.executor.setAgentChatRunner(async (brain, objective, _signal, turnId) => reply(brain, objective, turnId));
    const invocation = { brainId: "brain", toolId: "agent.fork", action: "start", arguments: { objective: "actual requested task" } };
    const asked = await value.executor.execute(invocation, undefined, "parent-turn");
    expect(asked.state).toBe("approval-required"); expect(value.service.repository.fork).not.toHaveBeenCalled();
    const approved = await value.executor.execute({ ...invocation, approvalToken: asked.approvalToken }, undefined, "parent-turn");
    expect(approved.state).toBe("complete"); expect(value.service.repository.fork).toHaveBeenCalledOnce();
    const replay = await value.executor.execute({ ...invocation, approvalToken: asked.approvalToken }, undefined, "parent-turn");
    expect(replay.state).toBe("approval-required"); expect(value.service.repository.fork).toHaveBeenCalledOnce();
  });
  it("preserves cancellation in child runner and COW storage hooks without starting subsequent tasks", async () => {
    const value = await fixture(), abort = new AbortController();
    const runner = vi.fn(async (brain: string, objective: string, signal: AbortSignal, turnId: string) => {
      expect(signal).toBe(abort.signal); abort.abort(); return reply(brain, objective, turnId);
    }); value.executor.setAgentChatRunner(runner);
    await expect(value.direct.agent("brain", "start", { objective: "actual task", workers: 2 }, abort.signal)).rejects.toThrow();
    expect(runner).toHaveBeenCalledOnce();
    expect(value.service.repository.fork.mock.calls.every(call => call[2]?.signal === abort.signal)).toBe(true);
    const index = await readFile(new URL("../src/main/index.ts", import.meta.url), "utf8");
    expect(index).toContain("tools.setAgentChatRunner"); expect(index).toContain("actions.cancel(brainId, turnId); tools.cancel(brainId, turnId)");
    const repository = await readFile(new URL("../src/main/brainRepository.ts", import.meta.url), "utf8");
    expect(repository).toContain('this.copyOnWriteClone(id, name, "fork", operation), operation?.signal');
  });
});
