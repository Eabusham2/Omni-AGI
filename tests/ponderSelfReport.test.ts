import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import { normalizeChatEngineEvent, type NeuralChatStreamEvent } from "../src/main/brainService";
import { ChatActionController } from "../src/main/chatActionController";
import { normalizePonderSelfReportDelta, validatePonderSelfReport, type PonderSelfReportDelta } from "../src/shared/ponderSelfReport";
import type { ActionEvent, ChatStreamEvent } from "../src/shared/types";

const sha = (value: string): string => createHash("sha256").update(value).digest("hex");
function frame(overrides: Partial<PonderSelfReportDelta> = {}): PonderSelfReportDelta {
  return { format: "omni-ponder-self-report-delta", version: 1, reportId: "b".repeat(64), actionId: "a".repeat(32),
    phase: "pre-speech", chunkIndex: 0, byteOffset: 0, delta: "tentative model report", publicationComplete: true,
    totalBytes: Buffer.byteLength("tentative model report"), textSha256: sha("tentative model report"), modelEmitted: true,
    selfReportNotChainOfThought: true, hiddenPromptUsed: false, inferenceCompletedBeforePublication: true, ...overrides };
}

describe("typed Ponder self-report boundary", () => {
  it("accepts exact bounded publication frames and rejects extra or inconsistent fields", () => {
    expect(normalizePonderSelfReportDelta(frame())).toEqual(frame());
    expect(normalizePonderSelfReportDelta({ ...frame(), hiddenInstruction: "reason aloud" })).toBeUndefined();
    expect(normalizePonderSelfReportDelta({ ...frame(), publicationComplete: false })).toBeUndefined();
    expect(normalizeChatEngineEvent({ type: "chat-ponder-self-report", brainId: "brain-a", actionId: "a".repeat(32),
      sequence: 3, data: frame() }, "brain-a")).toEqual({ type: "chat-ponder-self-report", sequence: 3, report: frame() });
  });

  it("distinguishes complete hash-bound reports from honestly partial custody", () => {
    const complete = { format: "omni-ponder-self-report", version: 1, id: "b".repeat(64), actionId: "a".repeat(32),
      phase: "pre-speech", text: "visible", state: "complete", emittedBytes: 7, textSha256: sha("visible"), modelEmitted: true,
      selfReportNotChainOfThought: true, hiddenPromptUsed: false, inferenceCompletedBeforePublication: true } as const;
    expect(validatePonderSelfReport(complete, sha)).toEqual(complete);
    const partial = { ...complete, state: "partial" as const, text: "vis", emittedBytes: 3, textSha256: undefined };
    expect(validatePonderSelfReport(partial, sha)).toEqual(expect.objectContaining({ state: "partial", text: "vis" }));
    expect(() => validatePonderSelfReport({ ...complete, textSha256: "0".repeat(64) }, sha)).toThrow(/checksum/);
  });

  it("retains only actually published text as partial action custody when the neural turn stops", async () => {
    const action = { kind: "ponder" as const, source: "brain" as const, actionId: "a".repeat(32), arguments: {
      handledInTurn: true, ponderSelfReport: { format: "omni-ponder-self-report", version: 1, id: "b".repeat(64),
        actionId: "a".repeat(32), phase: "pre-speech", state: "publishing", modelEmitted: true,
        selfReportNotChainOfThought: true, hiddenPromptUsed: false, inferenceCompletedBeforePublication: true }
    } };
    const partialFrame = frame({ delta: "visible prefix", publicationComplete: false, totalBytes: undefined, textSha256: undefined });
    const recorded: ActionEvent[][] = [], streamed: ChatStreamEvent[] = [];
    const service = {
      chat: vi.fn(async (_brain: string, _input: string, _signal: AbortSignal | undefined,
        onStream?: (event: NeuralChatStreamEvent) => void) => {
        onStream?.({ type: "chat-action", sequence: 0, actionId: action.actionId, action });
        onStream?.({ type: "chat-ponder-self-report", sequence: 1, report: partialFrame });
        throw new Error("fixture neural stop");
      }),
      recordConversationActions: vi.fn(async (_brain: string, events: ActionEvent[]) => { recorded.push(structuredClone(events)); })
    };
    const controller = new ChatActionController(service, { execute: vi.fn(), cancel: vi.fn(() => 0) }, { start: vi.fn() });
    controller.on("stream", event => { streamed.push(event); });
    await expect(controller.send("brain-a", "question", undefined, "turn-a")).rejects.toThrow(/fixture neural stop/);
    await vi.waitFor(() => expect(recorded.length).toBeGreaterThan(0));
    const report = recorded.at(-1)?.[0]?.action.arguments.ponderSelfReport as Record<string, unknown>;
    expect(report).toMatchObject({ state: "partial", text: "visible prefix", emittedBytes: 14 });
    expect(report).not.toHaveProperty("textSha256");
    expect(streamed.some(event => event.type === "chat-ponder-self-report")).toBe(true);
  });
});
