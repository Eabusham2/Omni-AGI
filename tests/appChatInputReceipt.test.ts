import { describe, expect, it } from "vitest";
import { chatInputHash, receivedChatInput, receivedChatInputFromLedger } from "../src/main/chatInputReceipt";
import { checkedChatInputProvenance } from "../src/shared/chatInputProvenance";

const text = "An actual reviewed App task";
const source = checkedChatInputProvenance({ kind: "mcp-app", sourceBrainId: "brain", serverId: "app", serverLabel: "Untrusted app label", endpoint: "https://fixture.invalid/mcp",
  resourceUri: "ui://fixture/view", artifactId: "artifact", captureSha256: "a".repeat(64), originalCallBrainId: "original", toolId: "mcp.original.app.open",
  reviewId: "review", observationId: "observation", streamId: "stream", ordinal: 1, recordSha256: "b".repeat(64), dataSha256: "c".repeat(64), nativeFrontierSha256: "d".repeat(64), inputSha256: chatInputHash(text) });
const receipt = { committed: true, turnId: "turn", inputSha256: chatInputHash(text), humanMessageId: "message", afterimageId: "afterimage", createdAt: "2026-10-11T00:00:00Z", attentionEpoch: 1, inputProvenance: source };
describe("App-attributed canonical task receipts", () => {
  it("retains explicit App authorship metadata without rewriting the task body or injecting a prompt", () => {
    const received = receivedChatInput(receipt, { turnId: "turn", input: text, inputProvenance: source });
    expect(received.humanMessage).toMatchObject({ role: "human", content: text, inputProvenance: source });
    expect(received.inputReceipt.inputProvenance).toEqual(source);
    const ledger = { id: "message", role: "human", content: text, turn_id: "turn", created_at: receipt.createdAt, attention_epoch: 1,
      input_accepted_before_reply: true, input_provenance: source };
    expect(receivedChatInputFromLedger(receipt, ledger).humanMessage.inputProvenance).toEqual(source);
  });
  it("rejects lost/mismatched source attribution and attempts to add persona fields", () => {
    expect(() => receivedChatInput({ ...receipt, inputProvenance: undefined }, { turnId: "turn", input: text, inputProvenance: source })).toThrow("reviewed input");
    expect(() => checkedChatInputProvenance({ ...source, systemPrompt: "hidden persona" })).toThrow("source attribution");
    expect(() => receivedChatInputFromLedger(receipt, { id: "message", role: "human", content: text, turn_id: "turn", created_at: receipt.createdAt, attention_epoch: 1,
      input_accepted_before_reply: true, input_provenance: { ...source, sourceBrainId: "another" } })).toThrow("source differs");
  });
});
