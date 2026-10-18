import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PonderPresentationLedger, reconcilePonderPresentation } from "../src/renderer/src/ponderSelfReportPresentation";
import { PonderSelfReportView } from "../src/renderer/src/PonderSelfReportView";
import type { ActionEvent, ChatMessage } from "../src/shared/types";
import type { PonderSelfReportDelta } from "../src/shared/ponderSelfReport";
const actionId = "a".repeat(32), reportId = "b".repeat(64);
const hostActionId = "host-action-event", action = { id: hostActionId, neuralActionId: actionId, brainId: "a", action: { kind: "ponder", arguments: {} } } as ActionEvent;
const frame = (delta: string, extra: Partial<PonderSelfReportDelta> = {}): PonderSelfReportDelta => ({ format: "omni-ponder-self-report-delta", version: 1,
  reportId, actionId, phase: "pre-speech", chunkIndex: 0, byteOffset: 0, delta, publicationComplete: false, modelEmitted: true,
  selfReportNotChainOfThought: true, hiddenPromptUsed: false, inferenceCompletedBeforePublication: true, ...extra });
describe("actual Ponder publication presentation", () => {
  it("requires a visible same-turn Ponder action and contiguous UTF-8 byte coverage", () => {
    const ledger = new PonderPresentationLedger("a");
    expect(() => ledger.receive("turn", frame("🙂"))).toThrow(/matching/); ledger.recognize(action, "turn");
    ledger.receive("turn", frame("🙂")); expect(ledger.receive("turn", frame("🙂"))).toBe(false);
    expect(() => ledger.receive("turn", frame("x", { chunkIndex: 1, byteOffset: 2 }))).toThrow(/contiguous/);
    ledger.receive("turn", frame(" observed", { chunkIndex: 1, byteOffset: 4, publicationComplete: true,
      totalBytes: 13, textSha256: "c".repeat(64) }));
    expect(ledger.snapshot()[0]?.report).toMatchObject({ text: "🙂 observed", emittedBytes: 13, state: "complete" });
  });
  it("keeps an interrupted prefix partial and rejects late text without inventing a final checksum", () => {
    const ledger = new PonderPresentationLedger("a"); ledger.recognize(action, "turn"); ledger.receive("turn", frame("a prefix"));
    expect(ledger.finish("turn")).toBe(true); expect(ledger.finish("turn")).toBe(false);
    expect(ledger.snapshot()[0]).toMatchObject({ live: false, report: { text: "a prefix", state: "partial" } });
    expect(ledger.snapshot()[0]?.report).not.toHaveProperty("textSha256");
    expect(() => ledger.receive("turn", frame(" later", { chunkIndex: 1, byteOffset: 8 }))).toThrow(/matching/);
  });
  it("reconciles committed publication by identity without repeating the action or discarding a longer accepted prefix", () => {
    const ledger = new PonderPresentationLedger("a"); ledger.recognize(action, "turn"); ledger.receive("turn", frame("actual text"));
    const report = ledger.snapshot()[0]!.report;
    const message = { id: "reply", role: "brain", turnId: "turn", ponderSelfReports: [{ ...report, state: "complete", textSha256: "c".repeat(64) }] } as ChatMessage;
    const result = reconcilePonderPresentation(ledger.snapshot(), [message]);
    expect(result.byAction.get(hostActionId)).toHaveLength(1); expect(result.byAction.get(hostActionId)?.[0]?.live).toBe(false);
    expect(reconcilePonderPresentation(ledger.snapshot(), [{ ...message, ponderSelfReports: [{ ...report, text: "actual", emittedBytes: 6 }] }]).byAction.get(hostActionId)?.[0]?.report.text).toBe("actual text");
  });
  it("restores a saved partial action report without a reply or old live ledger", () => {
    const ledger = new PonderPresentationLedger("a"); ledger.recognize(action, "turn"); ledger.receive("turn", frame("saved prefix"));
    const report = ledger.snapshot()[0]!.report, saved = { ...action, action: { ...action.action, arguments: { ponderSelfReport: report } } };
    const restored = reconcilePonderPresentation([], [], [saved]);
    expect(restored.byAction.get(hostActionId)?.[0]).toMatchObject({ live: false, report: { text: "saved prefix", state: "partial" } });
    expect(restored.byAction.has(actionId)).toBe(false); expect(restored.error).toBeUndefined();
  });
  it("does not turn a publication announcement into prose or a spurious invalid-report warning", () => {
    const value = reconcilePonderPresentation([], [], [{ ...action, action: { ...action.action,
      arguments: { ponderSelfReport: { format: "omni-ponder-self-report", version: 1, state: "publishing" } } } }]);
    expect(value.byAction.size).toBe(0); expect(value.error).toBeUndefined();
  });
  it("retires only exact durably saved settled prefixes, not active or uncommitted display state", () => {
    const ledger = new PonderPresentationLedger("a"); ledger.recognize(action, "turn"); ledger.receive("turn", frame("saved prefix"));
    const report = ledger.snapshot()[0]!.report, saved = { ...action, action: { ...action.action, arguments: { ponderSelfReport: report } } };
    expect(ledger.retirePersisted([], [saved])).toBe(false); ledger.finish("turn");
    expect(ledger.retirePersisted([], [{ ...saved, action: { ...saved.action, arguments: { ponderSelfReport: { ...report, text: "wrong prefix" } } } }])).toBe(false);
    expect(ledger.retirePersisted([], [saved])).toBe(true); expect(ledger.snapshot()).toEqual([]);
    expect(reconcilePonderPresentation([], [], [saved]).byAction.get(hostActionId)?.[0]?.report.text).toBe("saved prefix");
  });
  it("does not fast-forward readable display merely because the final action publication arrived", () => {
    const ledger = new PonderPresentationLedger("a"); ledger.recognize(action, "turn"); ledger.receive("turn", frame("actual text",
      { publicationComplete: true, totalBytes: 11, textSha256: "c".repeat(64) }));
    const report = ledger.snapshot()[0]!.report, published = { ...action, action: { ...action.action, arguments: { ponderSelfReport: report } } };
    expect(reconcilePonderPresentation(ledger.snapshot(), [], [published]).byAction.get(hostActionId)?.[0]?.live).toBe(true);
    ledger.finish("turn");
    expect(reconcilePonderPresentation(ledger.snapshot(), [], [published]).byAction.get(hostActionId)?.[0]?.live).toBe(false);
  });
  it("renders only provided text with readable-speed off and flushes the published prefix on hidden/terminal display", () => {
    const ledger = new PonderPresentationLedger("a"); ledger.recognize(action, "turn"); ledger.receive("turn", frame("Only actual text."));
    const row = ledger.snapshot()[0]!;
    const normal = renderToStaticMarkup(<PonderSelfReportView row={row} readable={false} onReadableChange={() => undefined} />);
    expect(normal).toContain('aria-pressed="false"'); expect(normal).toContain("Only actual text."); expect(normal).toContain("published excerpt");
    const paced = renderToStaticMarkup(<PonderSelfReportView row={row} readable onReadableChange={() => undefined} />);
    expect(paced).not.toContain("Only actual text.");
    expect(renderToStaticMarkup(<PonderSelfReportView row={row} readable flush onReadableChange={() => undefined} />)).toContain("Only actual text.");
  });
});
