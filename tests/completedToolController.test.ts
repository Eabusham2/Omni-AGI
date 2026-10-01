import { mkdtemp, mkdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatActionController } from "../src/main/chatActionController";
import { ToolExecutor } from "../src/main/toolExecutor";
import { CompletedActionEvidenceStore } from "../src/main/completedActionEvidence";
import type { BrainService, NeuralChatStreamEvent } from "../src/main/brainService";
import type { ActionEvent, ChatResult, ChatStreamEvent, StructuredAction, ToolExecutionResult, ToolInvocation } from "../src/shared/types";

vi.mock("node:os", async importActual => ({ ...await importActual<typeof import("node:os")>(), freemem: () => 8 * 1024 ** 3 }));

const time = "2026-09-30T12:00:00.000Z", turnId = "turn-fixture";
const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(path => rm(path, { recursive: true, force: true }))); });
function deferred<T>() { let resolve!: (value: T) => void, reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }
const completion = (): ChatResult => ({ brain: { messages: [] } as unknown as ChatResult["brain"],
  humanMessage: { id: "human", role: "human", turnId, content: "actual human fixture input", createdAt: time },
  brainMessage: { id: "reply", role: "brain", turnId, content: "stub committed prefix", createdAt: time }, trace: { id: "trace" } as ChatResult["trace"] });
const web: StructuredAction = { kind: "tool", source: "brain", actionId: "a".repeat(32), toolId: "web.fetch", action: "fetch", arguments: { url: "https://example.org" } };
const imagination: StructuredAction = { kind: "imagine", source: "brain", actionId: "b".repeat(32), toolId: "modality.imagine", action: "generate", arguments: { modality: "image", conceptIds: ["fixture"] } };

async function harness(level: "off" | "ask" | "auto" = "auto") {
  const root = await mkdtemp(join(tmpdir(), "omni-completed-controller-")); roots.push(root);
  const owner = join(root, "brain"); await mkdir(join(owner, "engine"), { recursive: true });
  const store = new CompletedActionEvidenceStore(join(owner, "engine"), "brain", async () => undefined);
  const neural = deferred<ChatResult>(), retained = deferred<ActionEvent>(), offered = deferred<void>();
  let stream!: (event: NeuralChatStreamEvent) => void;
  const service = { repository: { brainDirectory: () => owner },
    chat: vi.fn((_brain: string, _text: string, signal?: AbortSignal, listener?: (value: NeuralChatStreamEvent) => void) => {
      stream = listener!; signal?.addEventListener("abort", () => neural.reject(new Error("actual fixture Stop")), { once: true }); return neural.promise;
    }),
    queueCompletedActionEvidence: vi.fn(async (_brain: string, event: ActionEvent, output: unknown, route?: Parameters<CompletedActionEvidenceStore["stage"]>[2], turn?: string) => {
      const job = await store.stage(event, output, route, turn); retained.resolve(event); return job;
    }),
    queueCompletedToolExecution: vi.fn(async (invocation: ToolInvocation, result: ToolExecutionResult) => { await store.stageTool(invocation, result); }),
    authorizeInlineImagination: vi.fn(async () => undefined), observeChatActionResult: vi.fn(async () => { offered.resolve(); return undefined; }),
    learnStructuredExperience: vi.fn(async () => ({})), recordConversationActions: vi.fn(async () => undefined)
  };
  // Host method fixture only. No Electron app, runtime process, brain, model,
  // shell, network request or native forward is constructed or executed.
  const executor = Object.assign(Object.create(ToolExecutor.prototype), {
    service: service as unknown as BrainService, activeExecutions: new Map(), approvals: new Map(),
    permission: vi.fn(async () => ({ level, revision: "fixture-permission" })),
    dispatch: vi.fn(async () => ({ body: "actual stub tool output" })),
    audit: vi.fn(async () => undefined)
  }) as ToolExecutor;
  const privateExecutor = executor as unknown as { dispatch: ReturnType<typeof vi.fn>; audit: ReturnType<typeof vi.fn> };
  const controller = new ChatActionController(service, executor, { start: vi.fn() });
  const values: ChatStreamEvent[] = []; controller.on("stream", (event: ChatStreamEvent) => values.push(event));
  const reply = controller.send("brain", "actual human fixture input", undefined, turnId);
  return { owner, store, service, controller, executor, privateExecutor, neural, retained, offered, reply, values,
    emit: (action: StructuredAction) => stream({ type: "chat-action", sequence: 0, actionId: action.actionId!, action }) };
}

describe("performed outcome survives text/audit lifecycle", () => {
  it("retains success before a rejected post-effect audit and never reexecutes a final action replay", async () => {
    const value = await harness(); value.privateExecutor.audit.mockRejectedValue(new Error("fixture audit disk failure"));
    value.emit(web); const event = await value.retained.promise;
    expect(await value.store.pendingCount()).toBe(1);
    value.neural.resolve({ ...completion(), proposedActions: [web] });
    const result = await value.reply;
    expect(value.privateExecutor.dispatch).toHaveBeenCalledOnce();
    expect(value.service.queueCompletedActionEvidence).toHaveBeenCalledOnce();
    expect(event.execution?.state).toBe("complete"); expect(result.actionEvents?.[0]?.state).toBe("complete");
    expect(event.execution?.output).toEqual({ body: "actual stub tool output" });
    expect(value.service.learnStructuredExperience).not.toHaveBeenCalled();
  });
  it("ordinary transport failure after an effect keeps its exact queued outcome and does not start pending native work", async () => {
    const value = await harness(), rejected = expect(value.reply).rejects.toThrow("fixture transport failure");
    value.emit(web); const event = await value.retained.promise;
    const pending: StructuredAction = { kind: "ponder", source: "brain", actionId: "c".repeat(32), arguments: {} };
    value.emit(pending);
    value.neural.reject(new Error("fixture transport failure")); await rejected;
    await value.controller.cancelAndWait("brain", turnId);
    expect(event.execution?.state).toBe("complete"); expect(await value.store.pendingCount()).toBe(1);
    expect(value.privateExecutor.dispatch).toHaveBeenCalledOnce();
    expect(value.service.queueCompletedActionEvidence).toHaveBeenCalledOnce();
  });
  it("Stop after dispatch completion cannot erase the independent outcome or pass its aborted turn signal into learning", async () => {
    const value = await harness(), rejected = expect(value.reply).rejects.toThrow("fixture Stop");
    value.emit(web); const event = await value.retained.promise;
    value.controller.cancel("brain", turnId); await rejected; await value.controller.cancelAndWait("brain", turnId);
    expect(event.execution?.state).toBe("complete"); expect(await value.store.pendingCount()).toBe(1);
    expect(value.service.queueCompletedActionEvidence.mock.calls.every(call => call.length <= 5)).toBe(true);
    expect(value.service.learnStructuredExperience).not.toHaveBeenCalled(); expect(value.privateExecutor.dispatch).toHaveBeenCalledOnce();
  });
  it("learns genuine failed attempts generically, not as positive tool-route correction", async () => {
    const value = await harness(); value.privateExecutor.dispatch.mockRejectedValue(new Error("actual stub dispatch rejected"));
    value.emit(web); const event = await value.retained.promise;
    await value.offered.promise;
    value.neural.resolve(completion()); await value.reply;
    expect(event.execution).toMatchObject({ state: "failed", dispatchStarted: true, error: "actual stub dispatch rejected" });
    expect(value.service.observeChatActionResult).toHaveBeenCalledOnce();
    const job = (await value.store.pending())[0]!;
    expect(job.route).toBeUndefined();
    expect(JSON.parse(await readFile(value.store.evidencePath(job.evidenceId), "utf8"))).toMatchObject({ executionState: "failed", executionError: "actual stub dispatch rejected", outputPresent: false });
  });
  it("never retains a permission denial as an actually attempted effect", async () => {
    const value = await harness("off"); value.emit(web); value.neural.resolve(completion());
    const result = await value.reply;
    expect(result.actionEvents?.[0]?.state).toBe("failed"); expect(value.privateExecutor.dispatch).not.toHaveBeenCalled();
    expect(value.service.queueCompletedActionEvidence).not.toHaveBeenCalled(); expect(await value.store.pendingCount()).toBe(0);
  });
});

describe("Ask imagination starts through the owned warm authorization bridge", () => {
  it("streams approval during writing; consumed Ask token produces exact durable intent/control before dispatch, not after text completion", async () => {
    const value = await harness("ask"), artifact = deferred<unknown>(), approval = deferred<ActionEvent>();
    value.privateExecutor.dispatch.mockImplementation(() => artifact.promise);
    value.controller.on("stream", (event: ChatStreamEvent) => {
      if (event.type === "chat-action" && event.actionEvent.state === "approval-required") approval.resolve(event.actionEvent);
    });
    value.emit(imagination); const proposed = await approval.promise;
    expect(value.values.some(event => event.type === "chat-reply-committed")).toBe(false);
    expect(value.privateExecutor.dispatch).not.toHaveBeenCalled(); expect(value.service.authorizeInlineImagination).not.toHaveBeenCalled();
    const authorized = deferred<void>();
    value.service.authorizeInlineImagination.mockImplementation(async (...raw: unknown[]) => {
      const invocation = raw[0] as ToolInvocation, intentPath = raw[3] as string;
      const intent = JSON.parse(await readFile(intentPath, "utf8"));
      expect(intent).toMatchObject({ state: "authorized-before-side-effects", permission: "ask", chatTurnId: turnId, neuralActionId: imagination.actionId });
      expect(invocation.arguments).toMatchObject({ chatTurnId: turnId, neuralActionId: imagination.actionId });
      authorized.resolve();
    });
    const approved = value.controller.approveAction({ brainId: "brain", actionEventId: proposed.id, approvalToken: proposed.execution!.approvalToken! });
    await authorized.promise;
    expect(value.values.some(event => event.type === "chat-reply-committed")).toBe(false);
    value.neural.resolve(completion()); const text = await value.reply;
    expect(text.brainMessage.content).toBe("stub committed prefix");
    artifact.resolve({ path: "stub-owned-artifact", generatedDuringChat: true });
    expect((await approved).actionEvent.execution?.state).toBe("complete");
    expect(value.privateExecutor.dispatch).toHaveBeenCalledOnce();
    await expect(value.controller.approveAction({ brainId: "brain", actionEventId: proposed.id, approvalToken: proposed.execution!.approvalToken! })).rejects.toThrow("unavailable");
    expect(value.privateExecutor.dispatch).toHaveBeenCalledOnce();
  });
});
