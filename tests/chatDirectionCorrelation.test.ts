import { describe, expect, it } from "vitest";
import { chatDirectionRevisionText, chatDirectionStripItem, isSteerDirectionMessage, mergeChatDirections } from "../src/renderer/src/chatDirectionPresentation";
import type { SteerDirectionRecord } from "../src/shared/steerDirections";
import type { ChatMessage } from "../src/shared/types";
const record: SteerDirectionRecord = { format: "omni-steer-direction", version: 1, id: "direction", brainId: "brain",
  sequence: 1, turnId: "successor", predecessorTurnId: "original-task", bindingSha256: "a".repeat(64), state: "pending", revision: 1,
  createdAt: "2026-10-13T12:00:00Z", updatedAt: "2026-10-13T12:00:00Z",
  source: { brainId: "brain", turnId: "successor", predecessorTurnId: "original-task", createdAt: "2026-10-13T12:00:00Z",
    content: "Keep working, use the second source", attachmentIds: [], inputKind: "text" } };
const human: ChatMessage = { id: "pending-successor", turnId: "successor", role: "human", content: record.source.content, createdAt: record.source.createdAt };
describe("only exact successor directions move above the composer", () => {
  it("keeps the original task and unrelated equal text in the normal timeline", () => {
    expect(isSteerDirectionMessage(human, [record])).toBe(true);
    expect(isSteerDirectionMessage({ ...human, id: "steered-original-task", turnId: "original-task" }, [record])).toBe(false);
    expect(isSteerDirectionMessage({ ...human, id: "other", turnId: "other" }, [record])).toBe(false);
    expect(isSteerDirectionMessage({ ...human, createdAt: "2026-10-13T13:00:00Z" }, [record])).toBe(false);
    expect(isSteerDirectionMessage({ ...human, role: "brain" }, [record])).toBe(false);
  });
  it("correlates the actual current human receipt even when the strip was dismissed", () => {
    const saved = { ...record, humanMessageId: "actual-human", dismissedAt: "2026-10-13T12:01:00Z" };
    expect(isSteerDirectionMessage({ ...human, id: "actual-human" }, [saved])).toBe(true);
    expect(isSteerDirectionMessage({ ...human, id: "different-human" }, [saved])).toBe(false);
  });
  it("keeps newest revisions and rejects silently replaced immutable bindings", () => {
    const newer = { ...record, revision: 3, state: "received" as const };
    expect(mergeChatDirections([newer], [record])).toEqual([newer]);
    expect(() => mergeChatDirections([record], [{ ...record, bindingSha256: "b".repeat(64), revision: 4 }])).toThrow(/immutable/);
  });
  it("uses an audio label only for display without manufacturing a transcript or changing the source", () => {
    const audio = { ...record, source: { ...record.source, content: "", inputKind: "native-audio" as const } };
    expect(chatDirectionStripItem(audio).content).toBe("Voice direction");
    expect(audio.source.content).toBe(""); expect(chatDirectionStripItem(audio).inputReceived).toBe(false);
  });
  it("uses only the exact saved human projection after a merge without rekeying the native source", () => {
    const merged = { ...record, turnId: "new-execution", humanTurnId: "saved-human-turn", humanMessageId: "saved-human" };
    expect(isSteerDirectionMessage({ ...human, id: "saved-human", turnId: "saved-human-turn" }, [merged])).toBe(true);
    expect(isSteerDirectionMessage({ ...human, id: "saved-human", turnId: "new-execution" }, [merged])).toBe(false);
    expect(merged.source.turnId).toBe("successor");
  });
  it("never converts an audio display label into revision text", () => {
    const audio = { ...record, source: { ...record.source, content: "", inputKind: "native-audio" as const } };
    const item = chatDirectionStripItem(audio);
    expect(item.content).toBe("Voice direction");
    expect(chatDirectionRevisionText(item)).toBe("");
    expect(chatDirectionRevisionText(chatDirectionStripItem(record))).toBe(record.source.content);
  });
});
