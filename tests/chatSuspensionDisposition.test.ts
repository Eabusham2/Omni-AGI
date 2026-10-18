import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatActionController, type ActionChatService } from "../src/main/chatActionController";
import { BrainService } from "../src/main/brainService";
import { ConversationLedger } from "../src/main/conversationLedger";
import { InstanceContinuationDetachedError } from "../src/main/instanceContinuationDetached";
import { chatHostSuspensionForDetach, persistChatHostSuspension } from "../src/main/chatSuspension";
import { chatHostSuspensionStatus } from "../src/shared/chatSuspension";
import { advanceChatGenerationPhase, retainUncommittedChatOutput, settleUncommittedChatOutput } from "../src/renderer/src/chatGenerationLifecycle";
import { chatMessageDeliveryState, mergeChatMessagesForPresentation, settleOptimisticChatTurn } from "../src/renderer/src/chatTurnPresentation";
import type { ChatMessage, ChatStreamEvent } from "../src/shared/types";
import type { InstanceResumeFrontier } from "../src/shared/instanceLifecycle";
import { accepted, at, managed, sha } from "./fixtures/steerDirections";

const input = "Keep my original task", turnId = "suspended-turn";
const frontier = (): InstanceResumeFrontier => ({ checkpointId: "c".repeat(32), turnId, inputSha256: sha(input),
  lastStreamSequence: 4, sampledTokens: 3, publishedTokens: 3, phase: "native-text-token-boundary", exactSupportedResume: true });

describe("acknowledged detach is not failure or Stop (new host checks only)", () => {
  let root: string;
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-chat-suspension-")); await managed(root, "owner"); });
  afterEach(async () => { await rm(root, { recursive: true, force: true }); });

  it("distinguishes qualified native suspension from unadmitted host detach and rejects foreign or changed proof", () => {
    const receipt = accepted(turnId, input).inputReceipt, expected = { brainId: "owner", turnId, inputSha256: sha(input), inputReceipt: receipt };
    const suspended = chatHostSuspensionForDetach(new InstanceContinuationDetachedError("owner", turnId, frontier()), expected)!;
    expect(suspended).toMatchObject({ state: "suspended", acknowledgedUnload: true, frontier: frontier(), inputReceipt: receipt });
    expect(chatHostSuspensionStatus(suspended)).toContain("Input saved");
    const detached = chatHostSuspensionForDetach(new InstanceContinuationDetachedError("owner", turnId), { ...expected, inputReceipt: undefined })!;
    expect(detached.state).toBe("detached"); expect(detached.frontier).toBeUndefined(); expect(detached.inputReceipt).toBeUndefined();
    expect(chatHostSuspensionStatus(detached)).toContain("not confirmed received");
    expect(chatHostSuspensionForDetach({ code: "INSTANCE_CONTINUATION_DETACHED", brainId: "owner", turnId }, expected)).toBeUndefined();
    expect(chatHostSuspensionForDetach(new InstanceContinuationDetachedError("foreign", turnId, frontier()), expected)).toBeUndefined();
    expect(() => chatHostSuspensionForDetach(new InstanceContinuationDetachedError("owner", turnId, { ...frontier(), inputSha256: "f".repeat(64) }), expected)).toThrow(/differs/);
  });

  it("emits suspended with real accepted input and published prefix, without failed/cancelled/stopped or duplicate proposal dispatch", async () => {
    const actual = accepted(turnId, input), error = new InstanceContinuationDetachedError("owner", turnId, frontier()), events: ChatStreamEvent[] = [];
    let originalSignal!: AbortSignal;
    const service: ActionChatService = { chat: vi.fn(async (...args) => {
      originalSignal = args[2]!; args[3]?.({ type: "chat-input-accepted", sequence: 0, ...actual });
      args[3]?.({ type: "chat-token", sequence: 1, delta: "Preserved prefix", outputTokens: 3 });
      args[3]?.({ type: "chat-action", sequence: 2, action: { kind: "ponder", source: "brain", arguments: { topic: "pending idea" } } }); throw error;
    }), recordConversationActions: vi.fn(async () => undefined) };
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) }, actions = new ChatActionController(service, tools, { start: vi.fn() });
    const recorder = vi.fn(async (receipt, source) => persistChatHostSuspension(root, receipt, source)); actions.setChatSuspensionRecorder(recorder);
    actions.on("stream", event => events.push(event));
    await expect(actions.send("owner", input, undefined, turnId)).rejects.toBe(error);
    const terminal = events.at(-1)!;
    expect(terminal).toMatchObject({ type: "chat-state", state: "suspended", suspension: { inputReceipt: actual.inputReceipt, frontier: frontier() } });
    expect(events.filter(event => event.type === "chat-state" && ["failed", "cancelled", "stopped"].includes(event.state))).toHaveLength(0);
    expect(events.find(event => event.type === "chat-token")).toMatchObject({ delta: "Preserved prefix" });
    expect(events.some(event => event.type === "chat-reply-committed")).toBe(false);
    expect(originalSignal.aborted).toBe(false); expect(tools.execute).not.toHaveBeenCalled(); expect(tools.cancel).not.toHaveBeenCalled();
    expect(recorder).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => expect(actions.isBusy("owner")).toBe(false));
    const recorded = service.recordConversationActions as ReturnType<typeof vi.fn>;
    expect(recorded.mock.calls.at(-1)?.[1]?.[0]).toMatchObject({ state: "proposed" });
  });

  it("records an unadmitted host wait as detached without acceptance and does not promote a display-write failure to native failure", async () => {
    const service: ActionChatService = { chat: vi.fn(async () => { throw new InstanceContinuationDetachedError("owner", turnId); }) };
    const actions = new ChatActionController(service, { execute: vi.fn(), cancel: vi.fn(() => 0) }, { start: vi.fn() }), events: ChatStreamEvent[] = [];
    actions.setChatSuspensionRecorder(async () => { throw new Error("Synthetic disk refusal"); }); actions.on("stream", event => events.push(event));
    await expect(actions.send("owner", input, undefined, turnId)).rejects.toMatchObject({ code: "INSTANCE_CONTINUATION_DETACHED" });
    expect(events.at(-1)).toMatchObject({ type: "chat-state", state: "detached", suspension: { state: "detached" }, notices: [expect.stringContaining("display receipt")] });
    const last = events.at(-1); if (last?.type !== "chat-state") throw new Error("No disposition");
    expect(last.suspension?.inputReceipt).toBeUndefined(); expect(last.error).toBeUndefined(); expect(last.cancellation).toBeUndefined();
  });

  it("stores only display history, keeps neural files untouched, and allows another distinct suspend receipt without rewriting the first", async () => {
    const receipt = chatHostSuspensionForDetach(new InstanceContinuationDetachedError("owner", turnId, frontier()),
      { brainId: "owner", turnId, inputSha256: sha(input), inputReceipt: accepted(turnId, input).inputReceipt })!;
    const before = await readFile(join(root, "brain.json")), source = { content: input, createdAt: at, attachmentIds: ["original-file"] };
    await persistChatHostSuspension(root, receipt, source); await persistChatHostSuspension(root, receipt, source);
    await persistChatHostSuspension(root, { ...receipt, createdAt: "2026-10-13T15:00:01Z", frontier: { ...frontier(), sampledTokens: 5, publishedTokens: 5 } }, source);
    expect(await readFile(join(root, "brain.json"))).toEqual(before);
    const ledger = await ConversationLedger.open(root, "owner");
    try { ledger.integrity(); const page = ledger.page(); expect(page.entries).toHaveLength(2);
      const messages = page.entries.flatMap(row => row.message ? [row.message] : []); expect(messages.every(message => message.role === "human" && message.deliveryReceipt?.presentationOnly)).toBe(true);
      expect(mergeChatMessagesForPresentation(messages, [])).toHaveLength(1);
      expect(ledger.search("original task", undefined, 20, false).entries).toHaveLength(0);
    } finally { ledger.close(); }
  });

  it("settles generation controls while preserving exact partial text and optimistic identity without failure wording", () => {
    const suspension = chatHostSuspensionForDetach(new InstanceContinuationDetachedError("owner", turnId, frontier()), { brainId: "owner", turnId, inputSha256: sha(input) })!;
    const event: ChatStreamEvent = { id: "suspension-event", brainId: "owner", turnId, sequence: 5, createdAt: at, type: "chat-state", state: "suspended", suspension };
    expect(advanceChatGenerationPhase("responding", event)).toBe("settled");
    const outputs = settleUncommittedChatOutput(retainUncommittedChatOutput([], turnId, "Preserved prefix", at), turnId, "suspended");
    expect(outputs[0]).toMatchObject({ state: "suspended", message: { content: "Preserved prefix", turnId } });
    const optimistic = settleOptimisticChatTurn([{ id: `pending-${turnId}`, role: "human", content: input, createdAt: at }], turnId, "suspended");
    expect(chatMessageDeliveryState(optimistic[0]!)).toBe("suspended"); expect(optimistic[0]?.content).toBe(input);
    expect(chatHostSuspensionStatus(suspension)).not.toMatch(/failed|cancelled|stopped/i);
  });

  it("binds the actual no-load native frontier when detaching and leaves queued host waits unqualified", async () => {
    const detached: InstanceContinuationDetachedError[] = [], native = { brainId: "owner", learnedRuntimeResident: false, resume: frontier() };
    const pending = { brainId: "owner", turnId: "not-admitted", inputSha256: sha("Queued input"), chatDispatched: false, settled: Promise.resolve(), detach: (error: InstanceContinuationDetachedError) => detached.push(error) };
    const owner = { inspectNativeInstance: vi.fn(async () => native), liveChatTurns: new Map([["first", { ...pending, turnId, inputSha256: sha(input) }], ["second", pending]]),
      instanceDetachBoundaries: new Set(), engine: { inheritedExecutionPresentation: vi.fn() } } as unknown as BrainService;
    await BrainService.prototype.detachInstanceHostContinuations.call(owner, "owner");
    expect(detached).toHaveLength(2); expect(detached[0]?.frontier).toEqual(frontier()); expect(detached[1]?.frontier).toBeUndefined();
    expect((owner as unknown as { engine: { inheritedExecutionPresentation: ReturnType<typeof vi.fn> } }).engine.inheritedExecutionPresentation).not.toHaveBeenCalled();
    native.learnedRuntimeResident = true; await expect(BrainService.prototype.detachInstanceHostContinuations.call(owner, "owner")).rejects.toThrow(/unload acknowledgement/);
  });

  it("retains a genuinely started completed effect once and never dispatches or learns it again at detach", async () => {
    let detach!: () => void; const held = new Promise<never>((_resolve, reject) => { detach = () => reject(new InstanceContinuationDetachedError("owner", turnId, frontier())); });
    const retained = vi.fn(async () => ({ evidenceId: "evidence", state: "pending" } as never)), learn = vi.fn();
    const service: ActionChatService = { chat: vi.fn(async (...args) => {
      args[3]?.({ type: "chat-action", sequence: 0, actionId: "a".repeat(32), action: { kind: "tool", source: "brain", toolId: "web.fetch", action: "fetch", arguments: { url: "https://example.test/fixture" } } });
      return held;
    }), queueCompletedActionEvidence: retained, learnStructuredExperience: learn, recordConversationActions: vi.fn(async () => undefined) };
    const tools = { execute: vi.fn(async () => ({ id: "actual-synthetic-effect", toolId: "web.fetch", action: "fetch", state: "complete" as const,
      dispatchStarted: true as const, startedAt: at, finishedAt: at, output: { observed: "Retained mock outcome; no network used" } })), cancel: vi.fn(() => 0) };
    const actions = new ChatActionController(service, tools, { start: vi.fn() }), pending = actions.send("owner", input, undefined, turnId).catch(error => error);
    await vi.waitFor(() => expect(retained).toHaveBeenCalledTimes(1)); detach();
    expect(await pending).toMatchObject({ code: "INSTANCE_CONTINUATION_DETACHED" });
    await vi.waitFor(() => expect(actions.isBusy("owner")).toBe(false));
    expect(tools.execute).toHaveBeenCalledTimes(1); expect(retained).toHaveBeenCalledTimes(1); expect(learn).not.toHaveBeenCalled(); expect(tools.cancel).not.toHaveBeenCalled();
  });

  it("lets a later certified suspension replace a stale failure receipt without hiding a genuinely later failure", () => {
    const receipt = chatHostSuspensionForDetach(new InstanceContinuationDetachedError("owner", turnId, frontier()), { brainId: "owner", turnId, inputSha256: sha(input) })!;
    const message = (state: "failed" | "suspended", updatedAt: string): ChatMessage => ({ id: `delivery-${state}-${updatedAt}`, role: "human", turnId, content: input, createdAt: at,
      deliveryReceipt: { schemaVersion: 1, presentationOnly: true, turnId, state, updatedAt, ...(state === "suspended" ? { suspension: receipt } : {}) } });
    const oldFailure = message("failed", "2026-10-13T10:00:00Z"), suspended = message("suspended", "2026-10-13T10:00:01Z"), laterFailure = message("failed", "2026-10-13T10:00:02Z");
    expect(mergeChatMessagesForPresentation([oldFailure, suspended], [])[0]?.deliveryReceipt?.state).toBe("suspended");
    expect(mergeChatMessagesForPresentation([oldFailure, suspended, laterFailure], [])[0]?.id).toBe(laterFailure.id);
  });

  it("projects only declared safe cursor fields rather than copying private native metadata into the stream", () => {
    const privateFrontier = { ...frontier(), enginePath: "/private/native-owner", grantToken: "not-public", nested: { private: true } };
    const receipt = chatHostSuspensionForDetach(new InstanceContinuationDetachedError("owner", turnId, privateFrontier), { brainId: "owner", turnId, inputSha256: sha(input) })!;
    expect(receipt.frontier).toEqual(frontier()); expect(JSON.stringify(receipt)).not.toMatch(/enginePath|grantToken|native-owner|not-public|nested/);
  });
});
