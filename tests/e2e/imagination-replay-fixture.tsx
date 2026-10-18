import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import type { ImaginationRecording } from "../../src/shared/imaginationCapture";
import type { ImaginationAssembly, ImaginationAssemblyApi, ImaginationAssemblyEvent, ImaginationAssemblyManifestSummary, ImaginationAssemblyPlan, ImaginationAssemblyPlanRequest, ImaginationPlanEvent } from "../../src/shared/imaginationAssembly";
import { ImaginationReplayControls } from "../../src/renderer/src/ImaginationReplayControls";
import "../../src/renderer/src/styles.css";
import "../../src/renderer/src/imaginationCapture.css";

const listeners = new Set<(event: ImaginationAssemblyEvent) => void>(), jobs = new Map<string, ImaginationAssembly>(), requests = new Map<string, string>();
const receipts: unknown[] = [];
let audioOnly = false, loseAck = false, lost = false, n = 0, updateReceipt = () => {};
let holdPlan = false, heldPlan: { request: ImaginationAssemblyPlanRequest; value: ImaginationAssemblyPlan; resolve: (value: ImaginationAssemblyPlan) => void; reject: (reason: Error) => void } | undefined;
const planListeners = new Set<(value: ImaginationPlanEvent) => void>();
const rec = (brainId: string): ImaginationRecording => ({ format: "omni-imagination-recording", version: 1, id: `recording-${brainId}`, brainId, sequence: 1, requestId: "fixture-capture", mode: "manual", state: "complete", revision: 2, controlRevision: 2, startedAt: "2026-10-13T12:00:00Z", endedAt: "2026-10-13T12:00:02Z", timelineEndMs: 2000, requestedLookbackSeconds: 0, availableLookbackSeconds: 0, fragmentCount: 3, bytes: 200, gapCount: 1, representation: "verified-delivered-fragment-timeline", label: "Delivered fragments — not a completed video" });
const base = (brainId: string, id: string): ImaginationAssembly => ({ format: "omni-imagination-assembly", version: 1, id, brainId, recordingId: rec(brainId).id, recordingRevision: 2, requestId: "inert-copy", revision: 1, state: "dormant", createdAt: "2026-10-13T11:00:00Z", updatedAt: "2026-10-13T11:00:00Z", completedSpans: 0, totalSpans: 3, durationMs: 2000, manifestSha256: "a".repeat(64), label: "Delivered timeline replay — presentation timing, not native synchronization" });
jobs.set("copied-a", base("a", "copied-a"));
function publish(value: ImaginationAssembly) { jobs.set(value.id, value); listeners.forEach(listener => listener({ brainId: value.brainId, assemblyId: value.id, state: value.state, revision: value.revision })); updateReceipt(); }
function silentWav() {
  const buffer = new Uint8Array(44 + 16000), view = new DataView(buffer.buffer);
  for (const [offset, text] of [[0, "RIFF"], [8, "WAVE"], [12, "fmt "], [36, "data"]] as const) for (let i = 0; i < text.length; i++) buffer[offset + i] = text.charCodeAt(i);
  view.setUint32(4, buffer.length - 8, true); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, 8000, true); view.setUint32(28, 16000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); view.setUint32(40, 16000, true);
  return `data:audio/wav;base64,${btoa(String.fromCharCode(...buffer))}`;
}
const api: ImaginationAssemblyApi = {
  async plan(request) { receipts.push({ operation: "plan", request }); updateReceipt(); const value: ImaginationAssemblyPlan = { id: "fixed-fixture-plan", outputKind: audioOnly ? "audio-only" : "audiovisual", options: { width: request.options?.width ?? 960, height: request.options?.height ?? 540, fps: request.options?.fps ?? 12 }, durationMs: 2000, sourceFragments: 3, totalSpans: 3, estimatedPeakBytes: 10485760, automatic: !request.options, timingIssueCount: 1, quantizationMs: audioOnly ? .0625 : 41.667, captureGapCount: 1 };
    if (!holdPlan) return value;
    return new Promise((resolve, reject) => { heldPlan = { request, value, resolve, reject }; planListeners.forEach(listener => listener({ brainId: request.brainId, requestId: request.requestId, sequence: 1, state: "runtime", runtime: { state: "downloading", target: "synthetic", message: "Fixture progress only", completedBytes: 10485760, totalBytes: 20971520 } })); });
  },
  async cancelPlan(request) { receipts.push({ operation: "stop-plan", request }); updateReceipt();
    if (heldPlan?.request.brainId === request.brainId && heldPlan.request.requestId === request.requestId) { const held = heldPlan; heldPlan = undefined; planListeners.forEach(listener => listener({ ...request, sequence: 2, state: "cancelled" })); held.reject(new Error("Synthetic preparation cancelled")); return { ...request, state: "cancelled" }; }
    return { ...request, state: "absent" };
  },
  onPlanEvent(listener) { planListeners.add(listener); return () => { planListeners.delete(listener); }; },
  async assemble(request) {
    receipts.push({ operation: "assemble", request });
    let id = requests.get(request.requestId);
    if (!id) { id = `synthetic-${++n}`; requests.set(request.requestId, id); publish({ ...base(request.brainId, id), requestId: request.requestId, state: "encoding", revision: 2, createdAt: `2026-10-13T12:00:0${n}Z`, currentSpan: 0 }); }
    updateReceipt(); if (loseAck && !lost) { lost = true; throw new Error("Synthetic lost acknowledgement; the same request is safe to reconcile."); }
    return jobs.get(id)!;
  },
  async get(brainId, id) { const value = jobs.get(id); if (!value || value.brainId !== brainId) throw new Error("Unknown fixture replay"); return value; },
  async list(brainId, query) { receipts.push({ operation: "list", brainId, query }); updateReceipt(); return { entries: [...jobs.values()].filter(value => value.brainId === brainId && (!query?.recordingId || value.recordingId === query.recordingId)), nextAfterId: query?.afterId ? undefined : "older-fixture-page" }; },
  async manifest(brainId, id) {
    receipts.push({ operation: "manifest", id }); updateReceipt(); const recording = rec(brainId);
    return { format: "omni-delivered-timeline-composition", version: 1, policy: "delivered-timeline-v1", outputKind: "audio-only", recording, options: { width: 1, height: 1, fps: 1 }, durationMs: 2000, measuredEnd: true, captureGapCount: 1, fragmentCount: 3, trackCount: 2, spanCount: 3, manifestSha256: "a".repeat(64), presentation: { layout: "fixed-stream-grid", missingVisual: "neutral-black", exhaustedVisual: "last-verified-frame-hold", missingAudio: "declared-silence", independentAudio: "delivery-timed-equal-weight-mix", quantization: "nearest-output-sample", maximumQuantizationMs: .0625, nativeSynchronizationClaimed: false, supersededInSameFrameCount: 0 } } satisfies ImaginationAssemblyManifestSummary;
  },
  async manifestPage() { return { entries: [], total: 0 }; },
  async cancel(request) { receipts.push({ operation: "stop", request }); const value = { ...jobs.get(request.assemblyId)!, state: "cancelled" as const, revision: request.expectedRevision + 1 }; publish(value); return value; },
  async resume(request) { receipts.push({ operation: "resume", request }); const value = { ...jobs.get(request.assemblyId)!, state: "encoding" as const, revision: request.expectedRevision + 1 }; publish(value); return value; },
  onEvent(listener) { listeners.add(listener); return () => { listeners.delete(listener); }; }
};
function Fixture() {
  const [brainId, setBrainId] = useState("a"), [dark, setDark] = useState(false), [, redraw] = useState(0);
  updateReceipt = () => redraw(value => value + 1);
  return <main style={{ maxWidth: 860, height: "100vh", overflow: "auto", boxSizing: "border-box", padding: 16, margin: "0 auto" }}>
    <h1>Recording replay control check</h1><p>Actual replay controls with synthetic receipts and a silent WAV. No model, codec, device, provider, or real recording runs.</p>
    <div className="imagination-capture__row">
      <button className="button" onClick={() => { setDark(value => !value); document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; }}>Switch theme</button>
      <button className="button" onClick={() => setBrainId(value => value === "a" ? "b" : "a")}>Switch synthetic instance</button>
      <button className="button" onClick={() => { for (const value of jobs.values()) if (value.brainId === brainId && value.state === "encoding") publish({ ...value, state: "complete", revision: value.revision + 1, completedSpans: 3, mimeType: "audio/wav", available: true, mediaUrl: silentWav(), bytes: 16044 }); }}>Complete synthetic replay</button>
    </div>
    <div className="imagination-capture__row" style={{ marginBlock: 12 }}><label><input type="checkbox" onChange={event => { audioOnly = event.target.checked; }} />Synthetic audio-only plan</label><label><input type="checkbox" onChange={event => { loseAck = event.target.checked; lost = false; }} />Lose first synthetic acknowledgement</label><label><input type="checkbox" onChange={event => { holdPlan = event.target.checked; }} />Hold synthetic preparation</label></div>
    <section className="surface imagination-capture" style={{ padding: 14, marginBlock: 12 }}><ImaginationReplayControls recording={rec(brainId)} api={api} /></section>
    <details><summary>Synthetic receipts</summary><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", fontSize: 12 }}>{JSON.stringify({ brainId, uniqueJobs: n, receipts }, null, 2)}</pre></details>
  </main>;
}
document.documentElement.dataset.colorScheme = "light"; document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
