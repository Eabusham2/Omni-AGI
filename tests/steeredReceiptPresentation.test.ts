import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { validateWorkerChatPresentation } from "../src/main/brainService";

function fixture(disposition: "steered" | "native-stop" = "steered") {
  const input = "actual original direction";
  const inputSha256 = createHash("sha256").update(input).digest("hex");
  const response = " actual emitted prefix ";
  const turnId = "original";
  const checksum = "a".repeat(64);
  const value = {
    text: response, steered: disposition === "steered", nativeStopped: disposition === "native-stop",
    turnCommitted: true, idempotentCompletion: false,
    humanMessage: { id: "human-original", role: "human", content: input, turn_id: turnId,
      generation_end: disposition, created_at: "2026-09-29T12:00:00.000Z", attention_epoch: 1 },
    message: { id: "brain-original", role: "brain", content: response, turn_id: turnId,
      generation_end: disposition, created_at: "2026-09-29T12:00:01.000Z", attention_epoch: 1 },
    trace: { id: "trace-original", turn_id: turnId, input_sha256: inputSha256,
      generation_stop_reason: disposition === "steered" ? "steered" : "native-action-stop",
      parameter_checksum_after: checksum, created_at: "2026-09-29T12:00:02.000Z", attention_epoch: 1 },
    turnReceipt: { format: "omni-completed-chat-turn", formatVersion: 1, turnId, inputSha256,
      humanMessageId: "human-original", brainMessageId: "brain-original", traceId: "trace-original",
      inferenceCount: 1, parameterChecksumAfter: checksum,
      committedAt: "2026-09-29T12:00:03.000Z", generationEnd: disposition }
  };
  return { value, expected: { input, inputSha256, response, turnId } };
}

describe("receipt-bound interrupted output", () => {
  it.each(["steered", "native-stop"] as const)("preserves the exact %s prefix and distinguishes a saved partial pair", (disposition) => {
    const { value, expected } = fixture(disposition);
    const result = validateWorkerChatPresentation(value, expected);
    expect(result?.brainMessage).toMatchObject({ content: " actual emitted prefix ", generationEnd: disposition });
    expect(result?.humanMessage).toMatchObject({ content: expected.input, generationEnd: disposition });
  });

  it("rejects a steering label that conflicts with the saved receipt/trace", () => {
    const { value, expected } = fixture();
    value.message.generation_end = "native-stop";
    expect(() => validateWorkerChatPresentation(value, expected)).toThrow(/disposition/);
  });
});
