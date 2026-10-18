import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { ApiTeacherPanel } from "../../src/renderer/src/ApiTeacherPanel";
import type { ApiTeacherProvider, ApiTeacherTrainingReport, OmniApi, RuntimeJob, RuntimeJobEvent } from "../../src/shared/types";
import "../../src/renderer/src/styles.css";

const events = new Set<(event: RuntimeJobEvent) => void>(), jobs = new Map<string, RuntimeJob>(), receipts: string[] = [];
let redraw = () => {}, lostAck = false, ordinal = 0;
const baseReport: ApiTeacherTrainingReport = { provider: "ollama", model: "synthetic-local", requested: 2, learned: 0, failed: 0, skipped: 0,
  requestHashes: [], responseHashes: [], rawCredentialsStoredInBrain: false, hiddenBehavioralPromptUsed: false,
  durableState: "running", nextIndex: 0, uncertainItemIndexes: [], resumable: false };
function receipt(value: string) { receipts.push(value); redraw(); }
function publish(value: RuntimeJob) { jobs.set(value.id, value); events.forEach(listener => listener({ job: value })); redraw(); return value; }
const api: OmniApi["teacher"] = {
  async status() { return ["openai", "anthropic", "gemini", "openrouter", "ollama"].map(id => ({ provider: id as ApiTeacherProvider,
    configured: id === "ollama", required: id !== "ollama", persistence: "none" as const })); },
  async saveCredential() { throw new Error("Credentials are not accepted by this synthetic fixture."); },
  async removeCredential() { throw new Error("No credentials exist in this fixture."); },
  async discoverModels(request) { receipt(`Synthetic metadata only: ${request.provider} ${request.requestId}`);
    return { format: "omni-api-teacher-model-discovery", version: 1, provider: request.provider, requestId: request.requestId,
      sourceEndpoint: request.provider === "ollama" ? "http://127.0.0.1:11434/api/tags" : "https://openrouter.ai/api/v1/models/user",
      discoveredAt: new Date().toISOString(), pageCount: 1, complete: true, generationCompatibilityVerified: false,
      models: [{ provider: request.provider, id: "synthetic-local" }],
      excludedModelCount: 1, excludedModels: [{ id: "synthetic:cloud", reason: "Separate cloud consent is required." }] }; },
  async cancelModelDiscovery(request) { receipt(`Synthetic discovery Stop: ${request.provider}`); return { ...request, cancelled: true }; },
  async start(request) {
    receipt(`Synthetic start: ${request.provider} · ${request.requestId} · no key/model/network`);
    let value = [...jobs.values()].find(job => (job as RuntimeJob & { requestId?: string }).requestId === request.requestId);
    if (!value) {
      const now = new Date().toISOString(); value = { id: `synthetic-job-${++ordinal}`, brainId: request.brainId, kind: "teacher", state: "running", progress: 0,
        label: "Synthetic teacher running", createdAt: now, updatedAt: now,
        output: { ...baseReport, provider: request.provider, model: request.model, requested: request.prompts.length }, requestId: request.requestId } as RuntimeJob;
      jobs.set(value.id, value);
    }
    if (!lostAck) { lostAck = true; throw new Error("Synthetic lost start acknowledgement. Retry keeps the same request ID."); }
    return publish(value);
  },
  async list(id) { return [...jobs.values()].filter(value => value.brainId === id); },
  async cancel(id) { return api.stop({ brainId: jobs.get(id)!.brainId, jobId: id }); },
  async pause(request) { const value = jobs.get(request.jobId)!; receipt("Synthetic Pause acknowledged"); return publish({ ...value,
    state: "cancelled", label: "Synthetic teacher paused", updatedAt: new Date().toISOString(), output: { ...value.output as ApiTeacherTrainingReport, durableState: "paused", resumable: true } }); },
  async resume(request) { const value = jobs.get(request.jobId)!; receipt("Synthetic Resume acknowledged"); return publish({ ...value,
    state: "running", label: "Synthetic teacher running", updatedAt: new Date().toISOString(), output: { ...value.output as ApiTeacherTrainingReport, durableState: "running", resumable: false } }); },
  async stop(request) { const value = jobs.get(request.jobId)!; receipt("Synthetic Stop acknowledged"); return publish({ ...value,
    state: "cancelled", label: "Synthetic teacher stopped", updatedAt: new Date().toISOString(), output: { ...value.output as ApiTeacherTrainingReport, durableState: "stopped", resumable: false } }); },
  async details() { throw new Error("No uncertain provider effect exists in this fixture."); },
  async resolveUncertain() { throw new Error("No uncertain provider effect exists in this fixture."); },
  onEvent(listener) { events.add(listener); return () => { events.delete(listener); }; }
};
function Fixture() {
  const [brainId, setBrainId] = useState("fixture-a"), [dark, setDark] = useState(false), [compact, setCompact] = useState(false), [notice, setNotice] = useState(""), [, update] = useState(0);
  redraw = () => update(value => value + 1);
  return <main style={{ boxSizing: "border-box", height: "100vh", maxWidth: compact ? 380 : 980, overflow: "auto", padding: 20, margin: "auto" }}>
    <h1>Optional provider controls</h1><p>Actual teacher card with simulated replies only. No key, provider, brain, model, training or network call.</p>
    <div className="integration-actions"><button className="button" onClick={() => { document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; setDark(!dark); }}>Switch theme</button>
      <button className="button" onClick={() => setCompact(!compact)}>Compact panel</button><button className="button" onClick={() => setBrainId(brainId === "fixture-a" ? "fixture-b" : "fixture-a")}>Switch instance</button></div>
    <ApiTeacherPanel brainId={brainId} api={api} onNotice={setNotice} />
    {notice ? <p role="status">{notice}</p> : null}
    <details open><summary>Synthetic receipts · {ordinal} unique job(s)</summary><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{receipts.join("\n")}</pre></details>
  </main>;
}
document.documentElement.dataset.colorScheme = "light";
createRoot(document.getElementById("root")!).render(<Fixture />);
