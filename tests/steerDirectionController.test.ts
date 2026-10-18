import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatActionController, type ActionChatService } from "../src/main/chatActionController";
import { SteerDirectionsService } from "../src/main/steerDirections";
import { EngineRequestError } from "../src/main/engineSupervisor";
import { InstanceContinuationDetachedError } from "../src/main/instanceContinuationDetached";
import type { ChatResult, ChatStreamEvent } from "../src/shared/types";
import { accepted, at, managed, metadata, result } from "./fixtures/steerDirections";

describe("controller-owned durable steering receipts (synthetic native boundary only)", () => {
  let root: string, directions: SteerDirectionsService;
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-direction-controller-")); await managed(root, "owner");
    directions = new SteerDirectionsService({ brainDirectory: () => root }); });
  afterEach(async () => { await rm(root, { recursive: true, force: true }); });
  function controller(service: ActionChatService) {
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) }, controller = new ChatActionController(service, tools, { start: vi.fn() });
    controller.setSteerDirections(directions); return { controller, tools };
  }

  it("persists real accepted/final receipts and republishes MAIN-normalized inherited attachment IDs", async () => {
    const actual = accepted("direction-one", "Use the blue version", ["source-task-file", "new-file"]);
    const service: ActionChatService = { chat: vi.fn(async (...args) => {
      expect(args[1]).toBe("Use the blue version"); args[3]?.({ type: "chat-input-accepted", sequence: 0, ...actual }); return result(actual.humanMessage);
    }) };
    const { controller: actions, tools } = controller(service), events: ChatStreamEvent[] = [];
    actions.on("stream", event => events.push(event));
    await actions.send("owner", "  Use the blue version\0  ", undefined, "direction-one", metadata());
    const received = events.find(event => event.type === "chat-input-accepted");
    expect(received).toMatchObject({ inputReceipt: { taskAttachmentIds: ["source-task-file", "new-file"] }, humanMessage: { taskAttachmentIds: ["source-task-file", "new-file"] } });
    expect((await directions.forTurns("owner", ["direction-one"]))[0]).toMatchObject({ state: "complete", acceptedAttachmentIds: ["source-task-file", "new-file"],
      humanMessageId: actual.humanMessage.id, reply: { brainMessageId: "brain-direction-one" } });
    expect(tools.execute).not.toHaveBeenCalled(); expect(tools.cancel).not.toHaveBeenCalled();
  });

  it("stores a new direction only after genuine predecessor yield without rewriting original task or partial reply", async () => {
    let release!: (value: ChatResult) => void; const held = new Promise<ChatResult>(resolve => { release = resolve; });
    const original = accepted("original-task", "Keep the original task"), firstResult = result(original.humanMessage, "steered"); firstResult.brainMessage.content = "Published original prefix";
    const service: ActionChatService = { currentChatAttentionEpoch: vi.fn(async () => 3), steerChat: vi.fn(async () => { release(firstResult); }),
      chat: vi.fn(async (...args) => args[4] === "original-task" ? held : result(accepted(args[4]!, args[1]).humanMessage)) };
    const { controller: actions } = controller(service), transfer = vi.fn(async () => undefined); actions.setWarmTaskAttachmentTransfer(transfer);
    const first = actions.send("owner", original.humanMessage.content, undefined, "original-task", { kind: "task", source: "human", createdAt: at, attachmentIds: [] });
    await vi.waitFor(() => expect(service.chat).toHaveBeenCalledTimes(1));
    const next = actions.send("owner", "Use the blue version", undefined, "direction-one", metadata());
    expect((await first).brainMessage.content).toBe("Published original prefix"); await next;
    expect(service.steerChat).toHaveBeenCalledWith("owner", "original-task", "direction-one");
    expect(transfer).toHaveBeenCalledTimes(1); expect(original.humanMessage.content).toBe("Keep the original task");
    expect((await directions.list("owner")).entries.map(row => row.turnId)).toEqual(["direction-one"]);
  });

  it("Stop during asynchronous direction persistence cannot later dispatch a native turn", async () => {
    let release!: () => void; const held = new Promise<void>(resolve => { release = resolve; });
    const reserveWrite = vi.fn(async () => held); directions = new SteerDirectionsService({ brainDirectory: () => root, reserveWrite });
    const service: ActionChatService = { chat: vi.fn() }, { controller: actions } = controller(service);
    const pending = actions.send("owner", "Use the blue version", undefined, "direction-one", metadata()), failed = pending.catch(error => error);
    await vi.waitFor(() => expect(reserveWrite).toHaveBeenCalledTimes(1));
    const stopped = actions.cancelAndWait("owner", "direction-one");
    expect(service.chat).not.toHaveBeenCalled();
    release(); await stopped;
    expect(await failed).toMatchObject({ name: "AbortError" }); expect(service.chat).not.toHaveBeenCalled();
    expect((await directions.list("owner")).entries[0]).toMatchObject({ state: "stopped", failure: "stopped" });
  });

  it("keeps an acknowledged detached native frontier received/dormant, never falsely stopped or failed", async () => {
    const actual = accepted(), service: ActionChatService = { chat: vi.fn(async (...args) => {
      args[3]?.({ type: "chat-input-accepted", sequence: 0, ...actual }); throw new InstanceContinuationDetachedError("owner", "direction-one");
    }) }, { controller: actions } = controller(service);
    await expect(actions.send("owner", "Use the blue version", undefined, "direction-one", metadata())).rejects.toMatchObject({ code: "INSTANCE_CONTINUATION_DETACHED" });
    const row = (await directions.list("owner")).entries[0]!;
    expect(row).toMatchObject({ state: "received", dormant: { reason: "saved-continuation" }, inputReceipt: actual.inputReceipt }); expect(row.failure).toBeUndefined();
    expect(() => directions.forget("owner")).not.toThrow();
  });

  it("records actual early warm yield without manufacturing acceptance or a completed reply", async () => {
    const service: ActionChatService = { chat: vi.fn(async () => { throw new EngineRequestError("actual synthetic boundary", -32801,
      { brainId: "owner", turnId: "direction-one", steered: true, zeroTokenYield: true, safeBoundary: true, warm: true }); }) };
    const { controller: actions } = controller(service);
    await expect(actions.send("owner", "Use the blue version", undefined, "direction-one", metadata())).rejects.toMatchObject({ code: -32801 });
    const row = (await directions.list("owner")).entries[0]!; expect(row.state).toBe("steered"); expect(row.inputReceipt).toBeUndefined(); expect(row.reply).toBeUndefined();
  });
});
