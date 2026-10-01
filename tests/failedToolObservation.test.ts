import { describe, expect, it } from "vitest";
import { createChatToolObservation } from "../src/main/chatToolObservation";
import type { ActionEvent } from "../src/shared/types";

const time = "2026-09-30T12:00:00.000Z";
function failed(output: unknown): ActionEvent {
  return { id: "event", brainId: "brain", state: "failed", neuralActionId: "a".repeat(32), createdAt: time, updatedAt: time,
    action: { kind: "tool", source: "brain", toolId: "system.shell", action: "run", arguments: { script: "actual fixture attempt" } },
    execution: { id: "receipt", toolId: "system.shell", action: "run", state: "failed", dispatchStarted: true,
      startedAt: time, finishedAt: time, output, error: "actual fixture nonzero exit" } };
}
describe("typed same-turn actual failed attempt", () => {
  it("offers real error/status/output without a success label, prose instruction, or shortened result", () => {
    const output = { exitCode: 7, stderr: `${"actual observed failure\n".repeat(4000)}EXACT-END` }, event = failed(output);
    const result = createChatToolObservation("brain", "turn", event, output, () => true)!;
    expect(JSON.parse(result.payloadJson)).toEqual({ executionState: "failed", dispatchStarted: true,
      executionError: "actual fixture nonzero exit", outputPresent: true, output });
    expect(result.payloadJson).toContain("EXACT-END"); expect(result.payloadJson).not.toContain('"role"');
    expect(result.payloadJson).not.toContain('"outcome":"success"');
  });
  it("keeps successful wire payload exactly compatible", () => {
    const event = failed({ actual: true }); event.state = "complete"; event.execution!.state = "complete";
    const result = createChatToolObservation("brain", "turn", event, event.execution!.output, () => true)!;
    expect(result.payloadJson).toBe('{"output":{"actual":true},"outputPresent":true}');
  });
  it("permission denial/no dispatch, unfinished receipt, and explicit cancellation remain ineligible", () => {
    const event = failed(undefined);
    event.execution!.dispatchStarted = undefined;
    expect(createChatToolObservation("brain", "turn", event, undefined, () => true)).toBeUndefined();
    event.execution!.dispatchStarted = true; event.execution!.finishedAt = undefined;
    expect(createChatToolObservation("brain", "turn", event, undefined, () => true)).toBeUndefined();
    event.execution!.finishedAt = time; event.cancellationRequested = true;
    expect(createChatToolObservation("brain", "turn", event, undefined, () => true)).toBeUndefined();
  });
  it("retains actual no-output failure status/error, while a different output reference cannot masquerade as the receipt", () => {
    const event = failed(undefined), result = createChatToolObservation("brain", "turn", event, undefined, () => true)!;
    expect(JSON.parse(result.payloadJson)).toEqual({ executionState: "failed", dispatchStarted: true,
      executionError: "actual fixture nonzero exit", outputPresent: false });
    expect(() => createChatToolObservation("brain", "turn", event, "invented", () => true)).toThrow("binding");
  });
});
