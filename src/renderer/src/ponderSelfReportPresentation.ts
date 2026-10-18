import { normalizePonderSelfReportDelta, validatePonderSelfReport, type PonderSelfReport } from "../../shared/ponderSelfReport";
import type { ActionEvent, ChatMessage } from "../../shared/types";

export interface PonderPresentation { report: PonderSelfReport; actionEventId?: string; turnId?: string; live: boolean }
interface PendingReport extends PonderPresentation { nextChunk: number }

/** Presentation custody only. No parser of ordinary reply text or neural activity. */
export class PonderPresentationLedger {
  private actions = new Map<string, { turnId: string; eventId: string }>();
  private reports = new Map<string, PendingReport>();
  constructor(private readonly brainId: string) {}
  recognize(event: ActionEvent, turnId: string): void {
    if (event.brainId === this.brainId && event.action.kind === "ponder" && event.neuralActionId)
      this.actions.set(event.neuralActionId, { turnId, eventId: event.id });
  }
  receive(turnId: string, value: unknown): boolean {
    const delta = normalizePonderSelfReportDelta(value);
    const action = delta && this.actions.get(delta.actionId);
    if (!delta || !action || action.turnId !== turnId) throw new Error("Ponder text has no matching visible action.");
    const previous = this.reports.get(delta.reportId);
    if (previous && (previous.turnId !== turnId || previous.report.actionId !== delta.actionId || previous.report.phase !== delta.phase))
      throw new Error("Ponder text crossed its action or turn.");
    if (previous && delta.chunkIndex < previous.nextChunk) return false; // Duplicate delivery, never duplicate text.
    if (previous && (!previous.live || previous.report.state === "complete") || delta.chunkIndex !== (previous?.nextChunk ?? 0) || delta.byteOffset !== (previous?.report.emittedBytes ?? 0))
      throw new Error("Ponder text frames arrived outside their contiguous publication.");
    const text = (previous?.report.text ?? "") + delta.delta, emittedBytes = new TextEncoder().encode(text).byteLength;
    if (delta.publicationComplete && emittedBytes !== delta.totalBytes) throw new Error("Ponder text byte coverage differs from its completion receipt.");
    const report: PonderSelfReport = { format: "omni-ponder-self-report", version: 1, id: delta.reportId, actionId: delta.actionId,
      phase: delta.phase, text, emittedBytes, state: delta.publicationComplete ? "complete" : "partial",
      ...(delta.publicationComplete ? { textSha256: delta.textSha256 } : {}), modelEmitted: true, selfReportNotChainOfThought: true,
      hiddenPromptUsed: false, inferenceCompletedBeforePublication: true };
    this.reports.set(report.id, { report, actionEventId: action.eventId, turnId, live: true, nextChunk: delta.chunkIndex + 1 }); return true;
  }
  finish(turnId: string): boolean {
    let changed = false;
    for (const row of this.reports.values()) if (row.turnId === turnId && row.live) { row.live = false; changed = true; }
    for (const [id, action] of this.actions) if (action.turnId === turnId) this.actions.delete(id);
    return changed; // An interrupted prefix remains partial; no invented final checksum.
  }
  retirePersisted(messages: readonly ChatMessage[], actions: readonly ActionEvent[]): boolean {
    let changed = false;
    const retire = (value: unknown) => {
      try {
        const saved = validatePonderSelfReport(value), current = this.reports.get(saved.id);
        if (current && !current.live && current.report.actionId === saved.actionId && current.report.phase === saved.phase &&
            saved.emittedBytes >= current.report.emittedBytes && saved.text.startsWith(current.report.text) &&
            (current.report.state !== "complete" || saved.state === "complete" && current.report.textSha256 === saved.textSha256)) {
          this.reports.delete(saved.id); changed = true;
        }
      } catch { /* Invalid durable data is surfaced by reconciliation, not used to discard an accepted prefix. */ }
    };
    for (const message of messages) for (const report of message.ponderSelfReports ?? []) retire(report);
    for (const action of actions) if (action.action.kind === "ponder") retire(action.action.arguments.ponderSelfReport);
    return changed;
  }
  snapshot(): PonderPresentation[] { return [...this.reports.values()].map(({ report, actionEventId, turnId, live }) => ({ report, actionEventId, turnId, live })); }
}

export function reconcilePonderPresentation(live: readonly PonderPresentation[], messages: readonly ChatMessage[], actions: readonly ActionEvent[] = []): {
  byAction: Map<string, PonderPresentation[]>; error?: string;
} {
  const rows = new Map(live.map(row => [row.report.id, row])); let error: string | undefined;
  const eventIds = new Map(actions.filter(event => event.action.kind === "ponder" && event.neuralActionId).map(event => [event.neuralActionId!, event.id]));
  const merge = (value: unknown, turnId?: string, expectedNeuralId?: string, committedMessage = false) => {
    try {
      const report = validatePonderSelfReport(value), previous = rows.get(report.id);
      if (expectedNeuralId !== undefined && report.actionId !== expectedNeuralId) throw new Error("Ponder action/report identity differs.");
      if (previous && (previous.report.actionId !== report.actionId || previous.report.phase !== report.phase ||
          !previous.report.text.startsWith(report.text) && !report.text.startsWith(previous.report.text))) {
        error = "Saved Ponder text differs from the live publication; the saved version is shown.";
      } else if (previous && report.state === "partial" && previous.report.emittedBytes > report.emittedBytes) return;
      rows.set(report.id, { report, actionEventId: eventIds.get(report.actionId) ?? previous?.actionEventId, turnId: turnId ?? previous?.turnId,
        live: committedMessage ? false : previous?.live ?? false });
    } catch { error = "A Ponder report could not be displayed because its saved publication is invalid."; }
  };
  for (const event of actions) if (event.action.kind === "ponder" && event.action.arguments.ponderSelfReport !== undefined) {
    const report = event.action.arguments.ponderSelfReport;
    // The native action can announce a forthcoming publication without any
    // text. That announcement is neither a report nor an invalid stored prefix.
    if (report && typeof report === "object" && !Array.isArray(report) &&
        (report as Record<string, unknown>).format === "omni-ponder-self-report" && (report as Record<string, unknown>).state === "publishing") continue;
    merge(report, undefined, event.neuralActionId);
  }
  for (const message of messages) for (const value of message.ponderSelfReports ?? []) merge(value, message.turnId, undefined, true);
  const byAction = new Map<string, PonderPresentation[]>();
  for (const row of rows.values()) { const key = row.actionEventId ?? row.report.actionId, action = byAction.get(key) ?? []; action.push(row); byAction.set(key, action); }
  return { byAction, ...(error ? { error } : {}) };
}
