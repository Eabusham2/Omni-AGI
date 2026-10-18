import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { ImaginationCaptureControls } from "../../src/renderer/src/ImaginationCaptureControls";
import type { ImaginationCaptureApi, ImaginationCaptureEvent, ImaginationCaptureStatus, ImaginationRecording } from "../../src/shared/imaginationCapture";
import "../../src/renderer/src/styles.css";

const statuses = new Map<string, ImaginationCaptureStatus>();
const recordings = new Map<string, ImaginationRecording[]>();
const listeners = new Set<(event: ImaginationCaptureEvent) => void>();
const calls: string[] = [];
let loseNextAcknowledgement = false, statusUnavailable = false;
function status(id: string) { let value = statuses.get(id); if (!value) {
  value = { brainId: id, settings: { version: 1, revision: 1, bufferSeconds: 300, alwaysSave: false, dormant: id === "copy" }, bufferedFragments: 3, bufferedBytes: 300, availableLookbackSeconds: 12, autoPaused: false, unclassifiedChatIntent: true };
  statuses.set(id, value); recordings.set(id, []);
} return value; }
function emit(brainId: string) { for (const listener of listeners) listener({ brainId, type: "changed", at: new Date().toISOString() }); }
const api: ImaginationCaptureApi = {
  async status(id) { if (statusUnavailable) throw new Error("Synthetic lost acknowledgement; retry the same request."); return structuredClone(status(id)); },
  async settings(request) { const value = status(request.brainId); if (value.settings.revision !== request.expectedRevision) throw new Error("Synthetic stale settings");
    value.settings = { ...value.settings, revision: value.settings.revision + 1, bufferSeconds: request.bufferSeconds, alwaysSave: request.alwaysSave, dormant: false }; value.autoPaused = false;
    calls.push(`${request.brainId}:settings:${request.bufferSeconds}:${request.alwaysSave}`); emit(request.brainId); return structuredClone(value); },
  async record(request) { const value = status(request.brainId), rows = recordings.get(request.brainId)!;
    const prior = rows.find(row => row.requestId === request.requestId); if (prior) { statusUnavailable = false; calls.push(`${request.brainId}:retry-same-request`); return structuredClone(prior); }
    if (value.recording) throw new Error("Synthetic recording already active");
    const recording: ImaginationRecording = { format: "omni-imagination-recording", version: 1, id: `recording-${rows.length + 1}`, brainId: request.brainId, sequence: rows.length + 1, requestId: request.requestId, mode: "manual", state: "recording", revision: 1, controlRevision: 1, startedAt: "2026-10-13T12:00:00Z", requestedLookbackSeconds: request.lookbackSeconds, availableLookbackSeconds: Math.min(12, request.lookbackSeconds), fragmentCount: request.lookbackSeconds ? 3 : 0, bytes: request.lookbackSeconds ? 300 : 0, gapCount: 0, representation: "verified-delivered-fragment-timeline", label: "Delivered fragments — not a completed video" };
    value.recording = recording; rows.push(recording); calls.push(`${request.brainId}:record:${request.lookbackSeconds}`);
    if (loseNextAcknowledgement) { loseNextAcknowledgement = false; statusUnavailable = true; throw new Error("Synthetic recording committed but acknowledgement lost."); }
    emit(request.brainId); return structuredClone(recording); },
  async stop(request) { const value = status(request.brainId), record = value.recording; if (!record || record.controlRevision !== request.expectedRevision) throw new Error("Synthetic stale Stop");
    record.state = "complete"; record.revision++; record.controlRevision = record.revision; delete value.recording; value.autoPaused = true; calls.push(`${request.brainId}:stop`); emit(request.brainId); return structuredClone(record); },
  async cancel() { throw new Error("No cancel operation in this controls check"); },
  async resume(request) { const value = status(request.brainId), record = recordings.get(request.brainId)!.find(row => row.id === request.recordingId)!;
    record.state = "recording"; record.revision++; record.controlRevision = record.revision; value.recording = record; emit(request.brainId); return structuredClone(record); },
  async recordings(id) { status(id); return { entries: structuredClone(recordings.get(id)!).reverse(), total: recordings.get(id)!.length }; },
  async fragments(id, recordingId) { const record = recordings.get(id)!.find(row => row.id === recordingId)!;
    return { entries: Array.from({ length: record.fragmentCount }, (_, index) => ({ id: `fragment-${index}`, recordingId, index, streamId: "synthetic-stream", observedAt: "2026-10-13T12:00:00Z", offsetMs: index * 1000, origin: { kind: "spontaneous" as const, authority: "main-organic-action" as const, actionEventId: "synthetic-event" }, kind: "preview-fragment" as const, modality: "image" as const, mimeType: "image/png", sha256: "a".repeat(64), bytes: 100, available: false })), total: record.fragmentCount }; },
  onEvent(listener) { listeners.add(listener); return () => { listeners.delete(listener); }; }
};
function Fixture() {
  const [brainId, setBrainId] = useState("original"), [narrow, setNarrow] = useState(false), [dark, setDark] = useState(false), [receipt, setReceipt] = useState("");
  return <main style={{ padding: 16, maxWidth: 960, margin: "0 auto", height: "100vh", overflow: "auto", boxSizing: "border-box" }}>
    <h1>Existing imagination recording controls</h1><p>Actual React controls, synthetic host receipts only. No brain, model, file, decoder, recording bytes or permissions run.</p>
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}><button className="button" onClick={() => setNarrow(value => !value)}>{narrow ? "Full width" : "320px surface"}</button>
      <button className="button" onClick={() => { setDark(value => !value); document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; }}>Switch theme</button>
      <button className="button" onClick={() => setBrainId(value => value === "original" ? "copy" : "original")}>{brainId === "original" ? "Open dormant copy" : "Open original"}</button>
      <button className="button" onClick={() => { loseNextAcknowledgement = true; setReceipt("Next synthetic recording will commit with a lost acknowledgement."); }}>Lose next recording acknowledgement</button>
      <button className="button" onClick={() => setReceipt(calls.join("\n") || "No operations")}>Inspect synthetic calls</button></div>
    <section className="surface" style={{ marginTop: 14, padding: 14, maxWidth: "100%", width: narrow ? 320 : "100%", boxSizing: "border-box" }}>
      <ImaginationCaptureControls brainId={brainId} api={api} />
    </section><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{receipt}</pre>
  </main>;
}
document.documentElement.dataset.colorScheme = "light"; document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
