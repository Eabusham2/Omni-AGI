import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { LocalTeacherPanel } from "../../src/renderer/src/LocalTeacherPanel";
import type { LocalTeacherApi, LocalTeacherEvent, LocalTeacherGeneration, LocalTeacherJob, LocalTeacherSelection, LocalTeacherSelectionProgress } from "../../src/shared/localTeacher";
import "../../src/renderer/src/styles.css";
const events = new Set<(value: LocalTeacherEvent) => void>(), progressEvents = new Set<(value: LocalTeacherSelectionProgress) => void>(), selections = new Map<string, LocalTeacherSelection>(), jobs = new Map<string, LocalTeacherJob>();
const receipts: unknown[] = []; let redraw = () => {}, fit: "estimated" | "unknown" = "estimated", loseAck = true, lost = false, uniqueJobs = 0;
let pendingSelection: { brainId: string; requestId: string; resolve: (value: LocalTeacherSelection | null) => void; runner: LocalTeacherSelection["runner"] } | undefined;
const gen: LocalTeacherGeneration = { contextTokens: 4096, maxOutputTokens: 512, threads: 3, temperature: .7, seed: 0, timeoutMs: 0, maxOutputBytes: 65536 };
function teacher(brainId: string, runner: LocalTeacherSelection["runner"]): LocalTeacherSelection { return { format: "omni-local-gguf-teacher", version: 1, id: `teacher-${brainId}`, brainId, originalBrainId: brainId, originalSelectionId: `teacher-${brainId}`, createdAt: "2026-10-13T00:00:00Z", runner, runtime: { displayName: runner, bytes: 100, sha256: "a".repeat(64), helpSha256: "b".repeat(64), rawOfflineSupported: true, missingFlags: [] }, model: { displayName: "synthetic-teacher.gguf", bytes: 104857600, sha256: "c".repeat(64), header: { version: 3, tensorCount: "1", metadataCount: "1", metadataEnd: 64, name: "Synthetic teacher", architecture: "llama", geometry: {}, scope: "header-metadata-only", tensorPayloadValidated: false } }, license: { declaration: "Fixture data only", source: "user-declared", verified: false }, currentHostAuthorized: true }; }
function publish(value: LocalTeacherJob) { jobs.set(value.id, value); events.forEach(listener => listener({ brainId: value.brainId, job: value })); redraw(); return value; }
const api: LocalTeacherApi = {
  select(request) { receipts.push({ operation: "select", request }); redraw(); return new Promise(resolve => { pendingSelection = { ...request, resolve }; progressEvents.forEach(listener => listener({ brainId: request.brainId, requestId: request.requestId, phase: "hashing-model", fileName: "synthetic-teacher.gguf", bytes: 31457280, totalBytes: 104857600 })); }); },
  async cancelSelection(brainId, requestId) { receipts.push({ operation: "cancel-selection", brainId, requestId }); if (pendingSelection?.brainId === brainId && pendingSelection.requestId === requestId) { pendingSelection.resolve(null); pendingSelection = undefined; redraw(); return true; } return false; },
  async selectionProgress() { return null; }, onSelectionProgress(listener) { progressEvents.add(listener); return () => { progressEvents.delete(listener); }; },
  async selections(brainId) { return [...selections.values()].filter(value => value.brainId === brainId); },
  async plan(request) { receipts.push({ operation: "plan", request }); redraw(); return { brainId: request.brainId, teacherId: request.teacherId, fit, recommendation: fit === "estimated" ? gen : null, reservationBytes: fit === "estimated" ? 268435456 : null, measured: { appFamilyResidentBytes: 100000000, selectedRamCeilingBytes: 1000000000, physicalFreeBytes: 900000000, logicalCpuCount: 4 }, declared: { modelBytes: 104857600, kvBytesPerToken: fit === "estimated" ? 100 : null, contextLimit: fit === "estimated" ? 4096 : null, tensorPayloadValidated: false }, blockers: fit === "unknown" ? ["Synthetic missing KV geometry; no teacher was loaded."] : [], warnings: ["Synthetic estimate only; no OS, runtime or model is contacted.", "A source receipt is not proof of comprehension."] }; },
  async start(request) { receipts.push({ operation: "start", request }); let value = [...jobs.values()].find(job => job.requestId === request.requestId);
    if (!value) value = publish({ format: "omni-local-teacher-job", version: 1, id: `job-${++uniqueJobs}`, brainId: request.brainId, originalBrainId: request.brainId, originalJobId: `job-${uniqueJobs}`, requestId: request.requestId, requestSha256: "d".repeat(64), teacherId: request.teacherId, generation: request.generation, epochs: request.epochs ?? 1, state: "queued", revision: 1, createdAt: "2026-10-13T12:00:00Z", updatedAt: "2026-10-13T12:00:00Z", promptCount: request.prompts.length, generatedCount: 0, learnedCount: 0, phase: "waiting-for-native-compute-boundary", proof: "attributed-generation-and-native-source-receipts" });
    redraw(); if (loseAck && !lost) { lost = true; throw new Error("Synthetic job response lost. The saved request can be reconciled."); } return value;
  },
  async jobs(brainId) { return [...jobs.values()].filter(value => value.brainId === brainId); }, async get(brainId, jobId) { const value = jobs.get(jobId); if (!value || value.brainId !== brainId) throw new Error("Unknown synthetic job"); return value; },
  async items(_brainId, jobId) { receipts.push({ operation: "items", jobId }); redraw(); return [{ jobId, index: 0, prompt: "One synthetic multiline lesson.\nNo model ran.", inputSha256: "e".repeat(64), state: "generated", source: { relativePath: "synthetic.jsonl", sha256: "f".repeat(64), bytes: 100 } }]; },
  async pause(brainId, jobId) { receipts.push({ operation: "pause", brainId, jobId }); const value = jobs.get(jobId)!; return publish({ ...value, state: "paused", revision: value.revision + 1, phase: "safe-boundary" }); },
  async resume(brainId, jobId) { receipts.push({ operation: "resume", brainId, jobId }); const value = jobs.get(jobId)!; return publish({ ...value, state: "running", revision: value.revision + 1, phase: "synthetic-running" }); },
  async stop(brainId, jobId) { receipts.push({ operation: "stop", brainId, jobId }); const value = jobs.get(jobId)!; return publish({ ...value, state: "stopped", revision: value.revision + 1, phase: "safe-boundary" }); },
  onEvent(listener) { events.add(listener); return () => { events.delete(listener); }; }
};
function Fixture() {
  const [brainId, setBrainId] = useState("a"), [dark, setDark] = useState(false), [, update] = useState(0); redraw = () => update(value => value + 1);
  return <main style={{ maxWidth: 900, height: "100vh", overflow: "auto", boxSizing: "border-box", padding: 16, margin: "0 auto" }}><h1>Local teacher controls check</h1><p>Actual Learn-page controls with synthetic main replies. No files, process, model, training, provider or key is used.</p>
    <div className="local-teacher__row"><button className="button" onClick={() => { setDark(value => !value); document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; }}>Switch theme</button><button className="button" onClick={() => setBrainId(value => value === "a" ? "b" : "a")}>Switch synthetic instance</button>
      <button className="button" onClick={() => { if (!pendingSelection) return; const pending = pendingSelection, value = teacher(pending.brainId, pending.runner); selections.set(value.id, value); pendingSelection = undefined; progressEvents.forEach(listener => listener({ brainId: pending.brainId, requestId: pending.requestId, phase: "complete", bytes: 104857600, totalBytes: 104857600 })); pending.resolve(value); }}>Finish synthetic file preparation</button>
      <button className="button" onClick={() => { fit = fit === "estimated" ? "unknown" : "estimated"; redraw(); }}>Toggle synthetic fit ({fit})</button>
    </div>
    <div style={{ marginBlock: 16 }}><LocalTeacherPanel brainId={brainId} api={api} /></div>
    <details><summary>Synthetic receipts</summary><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", fontSize: 12 }}>{JSON.stringify({ brainId, uniqueJobs, receipts }, null, 2)}</pre></details>
  </main>;
}
document.documentElement.dataset.colorScheme = "light"; document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
