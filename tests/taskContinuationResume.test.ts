import { describe, expect, it } from "vitest";
import type { InstanceResumeFrontier } from "../src/shared/instanceLifecycle";
import type { NativeInheritedPresentation, NativeInstanceStatus } from "../src/shared/instanceLifecycleApi";
import { verifiedTaskContinuationResume } from "../src/main/taskContinuationResume";

const frontier: InstanceResumeFrontier = { checkpointId: "c".repeat(64), turnId: "continued", inputSha256: "a".repeat(64),
  lastStreamSequence: 17, sampledTokens: 5, publishedTokens: 5, phase: "native-text-token-boundary", exactSupportedResume: true };
const expected = { brainId: "current-owner", frontier, input: "exact original input", inputSha256: frontier.inputSha256 };
const observed = (): NativeInstanceStatus => ({ brainId: expected.brainId, learnedRuntimeResident: false, minimumContextTokens: 256, reconnect: [],
  resume: { ...frontier }, presentation: { input: expected.input, publishedText: "emitted ", actions: [],
    inputReceipt: { committed: true, turnId: frontier.turnId, inputSha256: frontier.inputSha256, attentionEpoch: 3 } } });

describe("main-only exact native task scope resume evidence", () => {
  it("binds actual saved checkpoint, current owner, exact input, and receipt epoch without a Load grant", () => {
    expect(verifiedTaskContinuationResume(expected, observed())).toEqual({ brainId: expected.brainId, turnId: frontier.turnId,
      checkpointId: frontier.checkpointId, inputSha256: frontier.inputSha256, attentionEpoch: 3 });
  });
  it("refuses foreign owners, fabricated/changed receipts and altered cursor counters", () => {
    for (const mutation of [
      (value: NativeInstanceStatus) => { value.brainId = "other-owner"; },
      (value: NativeInstanceStatus) => { value.resume!.publishedTokens++; },
      (value: NativeInstanceStatus) => { value.presentation!.input = "different"; },
      (value: NativeInstanceStatus) => { value.presentation!.inputReceipt = { committed: false, turnId: frontier.turnId, inputSha256: frontier.inputSha256, attentionEpoch: 3 }; },
      (value: NativeInstanceStatus) => { value.presentation!.inputReceipt = { committed: true, turnId: "foreign-turn", inputSha256: frontier.inputSha256, attentionEpoch: 3 }; }
    ]) { const value = observed(); mutation(value); expect(() => verifiedTaskContinuationResume(expected, value)).toThrow(/cannot authenticate/); }
  });
  it("retains the original accepted audio SHA rather than hashing empty text or rewriting source identity", () => {
    const value = observed(); value.presentation!.input = "";
    expect(verifiedTaskContinuationResume({ ...expected, input: "" }, value).inputSha256).toBe(frontier.inputSha256);
    expect(value.presentation!.inputReceipt).toEqual({ committed: true, turnId: frontier.turnId, inputSha256: frontier.inputSha256, attentionEpoch: 3 });
  });
  it("binds inherited execution lineage and original accepted turn separately from the destination turn", () => {
    const value: NativeInheritedPresentation = { brainId: expected.brainId, executionId: "d".repeat(64), parentId: "source-parent",
      originalTurnId: "original-source-turn", rootHumanMessageId: "root-human", frontier,
      presentation: { input: expected.input, publishedText: "emitted ", actions: [], inputReceipt: {
        committed: true, turnId: "original-source-turn", inputSha256: frontier.inputSha256, attentionEpoch: 3 } } };
    expect(verifiedTaskContinuationResume({ ...expected, inheritedExecutionId: value.executionId }, value).inherited)
      .toEqual({ executionId: value.executionId, sourceBrainId: value.parentId, originalTurnId: value.originalTurnId });
    expect(() => verifiedTaskContinuationResume(expected, value)).toThrow(/cannot authenticate/);
    expect(() => verifiedTaskContinuationResume({ ...expected, inheritedExecutionId: "e".repeat(64) }, value)).toThrow(/cannot authenticate/);
  });
});
