import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { ProcessMemorySettings } from "../../src/renderer/src/ProcessMemorySettings";
import type { ProcessMemoryApi, ProcessMemoryPreferences } from "../../src/shared/processMemory";
import "../../src/renderer/src/styles.css";

const values = new Map<string, ProcessMemoryPreferences>(), calls: string[] = [];
const preferences = (brainId: string): ProcessMemoryPreferences => {
  let value = values.get(brainId); if (!value) { value = { brainId, enabled: false, revision: 1, updatedAt: null, authority: "current-installation-only", capabilities: { platform: "synthetic", adapter: "mach-task-port", implemented: true, targetAccess: "not-probed", maxTransferBytes: 1048576, maxRegionPage: 200, environment: "host", physicalRamAccess: false, elevationPerformed: false, protectionChanges: false, detail: "Synthetic host capability only. No process has been inspected or modified." } }; values.set(brainId, value); } return value;
};
const api: ProcessMemoryApi = { async preferences(id) { return structuredClone(preferences(id)); }, async update(request) {
  const current = preferences(request.brainId); if (current.revision !== request.expectedRevision) throw new Error("Synthetic stale grant rejected");
  const next = { ...current, enabled: request.enabled, revision: current.revision + 1, updatedAt: new Date().toISOString() }; values.set(request.brainId, next);
  calls.push(`${request.brainId}:${request.expectedRevision}:${request.enabled}`); return structuredClone(next);
} };
function Fixture() {
  const [brainId, setBrainId] = useState("original"), [dark, setDark] = useState(false), [receipt, setReceipt] = useState("");
  return <main style={{ padding: 14, maxWidth: 760, margin: "0 auto", height: "100vh", overflow: "auto", boxSizing: "border-box" }}>
    <h1>Existing Access page control</h1><p>Actual setting component, synthetic preference API. No process access, real grant, OS probe or model.</p>
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}><button className="button" onClick={() => setBrainId(value => value === "original" ? "copy" : "original")}>Switch fixture owner</button>
      <button className="button" onClick={() => { setDark(value => !value); document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; }}>Switch theme</button>
      <button className="button" onClick={() => setReceipt(calls.join("\n") || "No setting updates")}>Inspect synthetic calls</button></div>
    <section className="surface" style={{ marginTop: 14 }}><ProcessMemorySettings brainId={brainId} api={api} /></section>
    <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{receipt}</pre>
  </main>;
}
document.documentElement.dataset.colorScheme = "light"; document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
