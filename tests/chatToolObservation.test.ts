import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { ChatObservationResourcePause, createChatToolObservation, observationReceipt } from "../src/main/chatToolObservation";
import type { ActionEvent } from "../src/shared/types";

const time = "2026-09-30T12:00:00.000Z";
export function completeEvent(output: unknown): ActionEvent {
  return { id: "event", brainId: "brain", neuralActionId: "a".repeat(32),
    action: { kind: "tool", source: "brain", toolId: "web.search", action: "search", arguments: { query: "actual request" } },
    state: "complete", createdAt: time, updatedAt: time,
    execution: { id: "execution", toolId: "web.search", action: "search", state: "complete", startedAt: time, finishedAt: time, output } };
}

describe("exact typed live tool observation", () => {
  it("preserves actual structured Unicode output and binds the completed action/receipt", () => {
    const output = { z: [true, null, 1.25], text: "actual observed data العربية 中文", a: "quoted \"data\"" };
    const observation = createChatToolObservation("brain", "turn", completeEvent(output), output, () => true)!;
    expect(JSON.parse(observation.payloadJson)).toEqual({ outputPresent: true, output });
    expect(observation.payloadJson).toBe('{"output":{"a":"quoted \\\"data\\\"","text":"actual observed data العربية 中文","z":[true,null,1.25]},"outputPresent":true}');
    expect(observation.payloadSha256).toBe(createHash("sha256").update(observation.payloadJson, "utf8").digest("hex"));
    expect(observation.observationId).toBe(createHash("sha256").update([observation.format, observation.brainId,
      observation.turnId, observation.neuralActionId, observation.actionEventId, observation.executionId,
      observation.toolId, observation.action, observation.completedAt, observation.payloadSha256].join("\0"), "utf8").digest("hex"));
    expect(observation).not.toHaveProperty("humanMessage");
    expect(observation).not.toHaveProperty("prompt");
  });
  it("represents absent output truthfully, not as invented prose or a null observation", () => {
    const observation = createChatToolObservation("brain", "turn", completeEvent(undefined), undefined, () => true)!;
    expect(JSON.parse(observation.payloadJson)).toEqual({ outputPresent: false });
  });
  it("does not slice a large result and refuses before serialization when allocation is not admitted", () => {
    const output = `START:${"real-data ".repeat(10_000)}:END`;
    const source = completeEvent(output);
    const observation = createChatToolObservation("brain", "turn", source, output, () => true)!;
    expect(JSON.parse(observation.payloadJson).output).toBe(output);
    let admittedBytes = 0;
    expect(() => createChatToolObservation("brain", "turn", source, output, (bytes) => { admittedBytes = bytes; return false; }))
      .toThrow(ChatObservationResourcePause);
    expect(admittedBytes).toBeGreaterThan(output.length);
    expect(source.execution?.output).toBe(output);
    expect(source.state).toBe("complete");
  });
  it("does not observe approval-required/failed/cancelled/unidentified effects", () => {
    for (const patch of [{ state: "approval-required" }, { state: "failed" }, { cancellationRequested: true }, { neuralActionId: undefined }]) {
      const output = { actual: true };
      expect(createChatToolObservation("brain", "turn", { ...completeEvent(output), ...patch } as ActionEvent, output, () => true)).toBeUndefined();
    }
  });
  it("rejects changed result ownership and unsupported implicit JSON coercion", () => {
    const event = completeEvent({ actual: true });
    expect(() => createChatToolObservation("brain", "turn", event, { invented: true }, () => true)).toThrow("execution binding");
    for (const output of [NaN, { value: undefined }, new Date(time)]) {
      expect(() => createChatToolObservation("brain", "turn", completeEvent(output), output, () => true)).toThrow();
    }
    let getterCalled = false;
    const output = Object.defineProperty({}, "unsafe", { enumerable: true, get: () => { getterCalled = true; return "not data"; } });
    expect(() => createChatToolObservation("brain", "turn", completeEvent(output), output, () => true)).toThrow("accessors");
    expect(getterCalled).toBe(false);
  });
  it("accepts exact legacy Windows alias execution binding without changing the issued tool identity", () => {
    const output = "actual file";
    const event = completeEvent(output);
    event.action.toolId = "windows.files";
    event.action.action = "read";
    event.execution!.toolId = "system.files";
    event.execution!.action = "read";
    expect(createChatToolObservation("brain", "turn", event, output, () => true)?.toolId).toBe("windows.files");
  });
  it("validates a queued receipt without pretending it was consumed and rejects a sibling receipt", () => {
    const observation = createChatToolObservation("brain", "turn", completeEvent("actual"), "actual", () => true)!;
    const receipt = { brainId: "brain", turnId: "turn", observationId: observation.observationId, accepted: true };
    expect(observationReceipt(receipt, observation)).toEqual(receipt);
    expect(receipt).not.toHaveProperty("consumed");
    expect(() => observationReceipt({ ...receipt, turnId: "sibling" }, observation)).toThrow("different turn/result");
  });
});
