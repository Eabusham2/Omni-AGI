import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { InstanceLifecycleControls } from "../../src/renderer/src/InstanceLifecycleControls";
import type { InstanceLifecycleState } from "../../src/shared/instanceLifecycle";
import type { InstanceLifecycleApi } from "../../src/shared/instanceLifecycleApi";
import "../../src/renderer/src/styles.css";

let state: InstanceLifecycleState = { brainId: "fixture", phase: "unloaded", autoLoad: false, revision: 0,
  updatedAt: new Date().toISOString(), learnedRuntimeResident: false, pausedJobIds: [], reconnect: [] };
const listeners = new Set<(value: InstanceLifecycleState) => void>();
const calls: string[] = []; let allowed = false, release: (() => void) | undefined;
const update = (patch: Partial<InstanceLifecycleState>) => { state = { ...state, ...patch, revision: state.revision + 1, updatedAt: new Date().toISOString() }; for (const listener of listeners) listener(state); return state; };
Object.assign(window, { __lifecycleFixture: { calls, fit: (value: boolean) => { allowed = value; }, release: () => release?.(),
  unsupported: () => { allowed = true; return update({ phase: "suspended", autoLoad: false, learnedRuntimeResident: false,
    resume: { checkpointId: "c".repeat(32), turnId: "older-saved-turn", inputSha256: "a".repeat(64), lastStreamSequence: 2,
      sampledTokens: 2, publishedTokens: 1, phase: "native-text-token-boundary", exactSupportedResume: true },
    resumeCompatibility: { compatible: false, reason: "This saved frontier predates the exact runtime compatibility seal; exact restore is unsupported." } }); },
  reset: () => update({ phase: "unloaded", learnedRuntimeResident: false, resume: undefined, reconnect: [], waitingFor: undefined }) } });
const api: InstanceLifecycleApi = {
  lifecycle: async () => state,
  loadFit: async () => { calls.push("fit"); return { brainId: "fixture", checkedAt: new Date().toISOString(), allowed,
    minimumContextTokens: 256, blockers: allowed ? [] : ["Current free RAM is below the selected envelope."] }; },
  load: async () => { calls.push("load"); return update({ phase: "loaded", learnedRuntimeResident: true, waitingFor: undefined }); },
  unload: async () => {
    calls.push("unload-request"); update({ phase: "suspending", waitingFor: { kind: "native-boundary", label: "Waiting for the exact token boundary" } });
    await new Promise<void>((resolve) => { release = resolve; }); calls.push("unload-ack");
    return update({ phase: "suspended", learnedRuntimeResident: false, waitingFor: undefined,
      resume: { checkpointId: "checkpoint-fixture", turnId: "same-turn-fixture", inputSha256: "a".repeat(64), lastStreamSequence: 9,
        sampledTokens: 7, publishedTokens: 5, phase: "native-text-token-boundary", exactSupportedResume: true },
      reconnect: [{ id: "camera-fixture", kind: "camera", label: "Camera capture", reason: "The external device handle must be reopened explicitly.", permissionRequired: true }] });
  },
  setAutoLoad: async ({ enabled }) => { calls.push(`auto:${enabled}`); return update({ autoLoad: enabled }); },
  onLifecycle: (listener) => { listeners.add(listener); return () => { listeners.delete(listener); }; },
  resumePresentation: async () => null,
  resumeTurn: async () => { throw new Error("This renderer fixture never runs a neural turn."); },
  stopSavedTurn: async () => { calls.push("stop-saved-turn"); return update({ phase: "unloaded", resume: undefined, resumeCompatibility: undefined }); },
  inheritedResumePresentation: async () => { throw new Error("This renderer fixture never reads a native region."); },
  resumeInheritedTurn: async () => { throw new Error("This renderer fixture never runs a neural region."); }
};
function Fixture() {
  const [latest, setLatest] = useState<InstanceLifecycleState>(), [settings, setSettings] = useState(false);
  const [mount, setMount] = useState(0);
  return <main style={{ maxWidth: 800, padding: 24, margin: "0 auto" }}>
    <h1>Explicit lifecycle UI fixture</h1><p>Synthetic lifecycle receipts only. No model is built, loaded or trained.</p>
    <button type="button" className="button" onClick={() => setMount((value) => value + 1)}>Reopen fixture instance</button>
    <InstanceLifecycleControls key={mount} brainId="fixture" name="Fixture mind" api={api} onState={setLatest} onEditSettings={() => { calls.push("edit-settings"); setSettings(true); }} />
    {settings ? <section><h2>Offline settings</h2><p>Opening settings paused the live fit waiter.</p></section> : null}
    <p aria-label="Lifecycle receipt">{latest?.phase ?? "unknown"} · resident={String(latest?.learnedRuntimeResident ?? false)}</p>
  </main>;
}
document.documentElement.dataset.colorScheme = "dark";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
