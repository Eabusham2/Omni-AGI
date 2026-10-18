import { describe, expect, it } from "vitest";
import { pausedOutputLabel, savedChatSuspension } from "../src/renderer/src/chatSuspensionPresentation";
import { chatDirectionStatus } from "../src/renderer/src/chatDirectionPresentation";
import type { ChatHostSuspensionReceipt } from "../src/shared/chatSuspension";
import type { ChatMessage } from "../src/shared/types";

const suspension: ChatHostSuspensionReceipt = { format: "omni-chat-host-suspension", version: 1,
  presentationOnly: true, brainId: "brain", turnId: "turn", state: "suspended", inputSha256: "a".repeat(64), acknowledgedUnload: true,
  createdAt: "2026-10-13T12:00:00Z", frontier: { checkpointId: "checkpoint", turnId: "turn", inputSha256: "a".repeat(64),
    sampledTokens: 2, publishedTokens: 1, lastStreamSequence: 3, phase: "native-text-token-boundary", exactSupportedResume: true } };
const saved: ChatMessage = { id: "saved", turnId: "turn", role: "human", content: "Original request", createdAt: suspension.createdAt,
  deliveryReceipt: { schemaVersion: 1, presentationOnly: true, turnId: "turn", state: "suspended", updatedAt: suspension.createdAt, suspension } };

describe("saved execution is not a cancelled or failed response", () => {
  it("labels qualified suspension and unqualified detachment separately", () => {
    expect(pausedOutputLabel("suspended")).toBe("Reply paused · checkpoint retained");
    expect(pausedOutputLabel("detached")).toBe("Host detached · partial output retained");
    expect(pausedOutputLabel("failed")).toBeUndefined();
  });
  it("does not call a saved same-instance direction a copied one", () => {
    expect(chatDirectionStatus({ id: "direction", content: "Continue", state: "received", inputReceived: true,
      dormant: true, dormantReason: "saved-continuation", dismissed: false })).toBe("Saved turn · paused");
  });
  it("recovers only a main-certified latest suspension when stream delivery was lost", () => {
    expect(savedChatSuspension([saved], "brain", "turn")).toEqual(suspension);
    expect(savedChatSuspension([{ ...saved, role: "brain" }], "brain", "turn")).toBeUndefined();
    expect(savedChatSuspension([{ ...saved, deliveryReceipt: undefined }], "brain", "turn")).toBeUndefined();
  });
  it("does not resurrect a saved cursor after later Stop or a real reply", () => {
    const stopped: ChatMessage = { ...saved, id: "stopped", deliveryReceipt: { schemaVersion: 1, presentationOnly: true,
      turnId: "turn", state: "stopped", updatedAt: "2026-10-13T12:00:01Z" } };
    expect(savedChatSuspension([saved, stopped], "brain", "turn")).toBeUndefined();
    expect(savedChatSuspension([saved, { ...saved, id: "reply", role: "brain", deliveryReceipt: undefined }], "brain", "turn")).toBeUndefined();
  });
  it("rejects foreign or mismatched qualified frontier evidence", () => {
    expect(savedChatSuspension([saved], "other-brain", "turn")).toBeUndefined();
    expect(savedChatSuspension([saved], "brain", "other-turn")).toBeUndefined();
    const mismatch = { ...saved, deliveryReceipt: { ...saved.deliveryReceipt!, suspension: { ...suspension,
      frontier: { ...suspension.frontier!, inputSha256: "b".repeat(64) } } } };
    expect(savedChatSuspension([mismatch], "brain", "turn")).toBeUndefined();
  });
});
