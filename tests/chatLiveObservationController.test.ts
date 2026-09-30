import { describe, expect, it, vi } from "vitest";
import { ChatActionController } from "../src/main/chatActionController";
import { createChatToolObservation, type ChatToolObservationReceipt } from "../src/main/chatToolObservation";
import type { NeuralChatStreamEvent, StructuredNeuralExperience } from "../src/main/brainService";
import type { TemporarySteeringContext } from "../src/main/temporarySteeringContext";
import type { ActionEvent, ChatResult, ChatStreamEvent, StructuredAction, ToolExecutionResult } from "../src/shared/types";

const time = "2026-09-30T12:00:00.000Z";
const action: StructuredAction = { kind: "tool", source: "brain", toolId: "web.search", action: "search",
  actionId: "a".repeat(32), arguments: { query: "actual request" } };
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
const result = (): ChatResult => ({ brain: { messages: [] } as unknown as ChatResult["brain"],
  humanMessage: { id: "human-turn", turnId: "turn", role: "human", content: "actual current human input", createdAt: time },
  brainMessage: { id: "brain-turn", turnId: "turn", role: "brain", content: "actual fixture completion", createdAt: time },
  trace: { id: "trace" } as ChatResult["trace"], proposedActions: [action] });
const execution = (output: unknown): ToolExecutionResult => ({ id: "execution", toolId: "web.search", action: "search",
  state: "complete", startedAt: time, finishedAt: time, output });

function setup(observer?: (brain: string, turn: string, event: ActionEvent, output: unknown) => Promise<ChatToolObservationReceipt | undefined>, approvalSeconds?: number) {
  const neural = deferred<ChatResult>(), tool = deferred<ToolExecutionResult>(), offered = deferred<void>();
  let stream!: (event: NeuralChatStreamEvent) => void;
  const observe = vi.fn(async (brain: string, turn: string, event: ActionEvent, output: unknown) => {
    offered.resolve();
    if (observer) return observer(brain, turn, event, output);
    const observation = createChatToolObservation(brain, turn, event, output, () => true)!;
    return { brainId: brain, turnId: turn, observationId: observation.observationId, accepted: true };
  });
  const service = { chat: vi.fn((_brain: string, _input: string, _signal?: AbortSignal, listener?: (event: NeuralChatStreamEvent) => void,
    _turnId?: string, _budget?: number, _temporary?: TemporarySteeringContext, _toolObservationWaitMs?: number) => {
    stream = listener!; return neural.promise;
  }), observeChatActionResult: observe, learnStructuredExperience: vi.fn(async (_brain: string, _experience: StructuredNeuralExperience) => ({})), recordConversationActions: vi.fn(async () => undefined) };
  const tools = { execute: vi.fn(() => tool.promise), cancel: vi.fn(() => 0),
    ...(approvalSeconds === undefined ? {} : { preferences: () => ({ approvalTimeoutSeconds: approvalSeconds }) }) };
  const controller = new ChatActionController(service, tools, { start: vi.fn() });
  const reply = controller.send("brain", "actual current human input", undefined, "turn");
  stream({ type: "chat-action", sequence: 0, actionId: action.actionId!, action });
  return { controller, service, tools, neural, tool, offered, reply, stream };
}

describe("same-turn result observation and durable fallback", () => {
  it("forwards the real customized runtime approval preference privately, never a hardcoded30second wait", async () => {
    const value = setup(undefined, 73);
    expect(value.service.chat.mock.calls[0]?.[7]).toBe(73_000);
    value.tool.resolve(execution({ actual: true }));
    value.neural.resolve(result());
    await value.reply;
  });
  it("offers the real result before neural completion, learns after commit, and never reexecutes final replay", async () => {
    const value = setup();
    value.tool.resolve(execution({ actual: "observed data" }));
    await value.offered.promise;
    expect(value.service.observeChatActionResult).toHaveBeenCalledTimes(1);
    expect(value.service.chat).toHaveBeenCalledTimes(1);
    expect(value.service.learnStructuredExperience).not.toHaveBeenCalled();
    value.neural.resolve(result());
    await value.reply;
    expect(value.tools.execute).toHaveBeenCalledTimes(1);
    expect(value.service.observeChatActionResult).toHaveBeenCalledTimes(1);
    expect(value.service.learnStructuredExperience).toHaveBeenCalledTimes(1);
  });
  it("does not offer completed-output late results but keeps full >48k output for durable learning", async () => {
    const value = setup();
    value.stream({ type: "chat-phase", sequence: 1, phase: "reply-complete-learning", replyComplete: true, turnCommitted: false, learning: true, saving: true });
    const output = `START:${"observed ".repeat(10_000)}:EXACT-END`;
    value.tool.resolve(execution(output));
    value.neural.resolve(result());
    await value.reply;
    expect(value.service.observeChatActionResult).not.toHaveBeenCalled();
    const experience = value.service.learnStructuredExperience.mock.calls[0]?.[1];
    expect(experience).toMatchObject({ content: expect.stringContaining(":EXACT-END") });
    expect(value.tools.execute).toHaveBeenCalledTimes(1);
  });
  it("observation refusal/failure does not fail a completed tool or lose postcommit learning", async () => {
    const value = setup(async () => { throw new Error("typed observation admission pause"); });
    value.tool.resolve(execution({ actual: true }));
    await value.offered.promise;
    value.neural.resolve(result());
    const completed = await value.reply;
    expect(completed.actionEvents?.[0]?.state).toBe("complete");
    expect(completed.brainMessage.content).toBe("actual fixture completion");
    expect(value.service.learnStructuredExperience).toHaveBeenCalledTimes(1);
    expect(value.tools.execute).toHaveBeenCalledTimes(1);
  });
  it("cannot offer failed effects or manufacture observations from them", async () => {
    const value = setup();
    value.tool.resolve({ ...execution(undefined), state: "failed", error: "real tool failure" });
    value.neural.resolve(result());
    const completed = await value.reply;
    expect(completed.actionEvents?.[0]?.state).toBe("failed");
    expect(value.service.observeChatActionResult).not.toHaveBeenCalled();
    expect(value.service.learnStructuredExperience).not.toHaveBeenCalled();
  });
  it("observes an exact Ask-approved result during decoding and does not learn the stale approval-required outcome twice", async () => {
    const value = setup();
    const approval = deferred<ActionEvent>();
    value.controller.on("stream", (event: ChatStreamEvent) => {
      if (event.type === "chat-action" && event.actionEvent.state === "approval-required") approval.resolve(event.actionEvent);
    });
    value.tool.resolve({ ...execution(undefined), id: "approval-intent", state: "approval-required",
      approvalToken: "exact-token", approvalExpiresAt: new Date(Date.now() + 60_000).toISOString() });
    const event = await approval.promise;
    const actual = { actual: "approved observed result" };
    value.tools.execute.mockResolvedValueOnce(execution(actual));
    value.service.learnStructuredExperience.mockImplementation(async () => { await value.neural.promise; return {}; });
    const approved = value.controller.approveAction({ brainId: "brain", actionEventId: event.id, approvalToken: "exact-token" });
    await value.offered.promise;
    expect(value.service.observeChatActionResult).toHaveBeenCalledTimes(1);
    expect(value.service.observeChatActionResult.mock.calls[0]?.[3]).toEqual(actual);
    value.neural.resolve(result());
    await Promise.all([value.reply, approved]);
    expect(value.tools.execute).toHaveBeenCalledTimes(2); // intent gate, then one authorized effect
    expect(value.service.learnStructuredExperience).toHaveBeenCalledTimes(1);
    expect(value.service.learnStructuredExperience.mock.calls[0]?.[1].content).toContain("approved observed result");
  });
});
