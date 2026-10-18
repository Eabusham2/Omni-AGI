import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { AutomationWorkspace } from "../../src/renderer/src/AutomationWorkspace";
import { PluginManager } from "../../src/renderer/src/PluginManager";
import { BestWorkSettings, type BestWorkPreferences } from "../../src/renderer/src/BestWorkSettings";
import { CallLayout } from "../../src/renderer/src/CallLayout";
import { NEW_ACTION_FILE } from "../../src/renderer/src/automationPresentation";
import type { AutomationDefinition, AutomationRun, AutomationStepReceipt, AutomationWorkspaceApi, AutomationWorkspaceEvent } from "../../src/shared/automation";
import type { ScheduleRecord } from "../../src/shared/clock";
import type { OmniPluginPackage, PluginCatalogEntry, PluginEvent, PluginInstanceAccess, PluginsApi } from "../../src/shared/plugins";
import type { ToolExecutionResult, ToolInvocation } from "../../src/shared/types";
import "../../src/renderer/src/styles.css";

// Synthetic UI edges only: no AutomationService, model, permissions, files,
// schedules, subprocesses, installation or external effects are created.
const brainId = "controls-ui-fixture", hash = "a".repeat(64), now = () => new Date().toISOString();
const definitions: AutomationDefinition[] = [{ id: "file", brainId, revision: 2, revisionSha256: hash,
  name: "Inspect the selected source, then notify me", createdAt: now(), updatedAt: now(),
  activation: { enabled: true, allowTriggers: true, ownerBrainId: brainId, ownerEpoch: "fixture", catalogRevision: 2, policyRevision: "fixture-only", revisionSha256: hash, changedAt: now() } }];
const run = (id: string, state: AutomationRun["state"], nextStep: number): AutomationRun => ({ id, brainId, definitionId: "file", revisionSha256: hash,
  inputSha256: hash, inputs: {}, state, nextStep, createdAt: now(), updatedAt: now(), activation: definitions[0].activation!,
  detail: state === "review-required" ? "Fixture: an interrupted request has no confirmed outcome. It will not be replayed." : "Synthetic control state; no step executes." });
const runs = [run("approval-run", "approval-required", 75), run("review-run", "review-required", 80), run("paused-run", "paused", 1)];
const currentStep = (id: string): AutomationStepReceipt => ({ runId: id, stepId: "inspect-selected-source", index: runs.find(row => row.id === id)!.nextStep, attempt: 2,
  requestId: `fixture-${id}`, state: id === "approval-run" ? "approval-required" : "unknown", startedAt: now(), updatedAt: now(),
  invocation: { brainId, toolId: "files.read", action: "read", arguments: { path: "synthetic-source.txt", exactUserText: "Fixture only; no real file exists." } },
  ...(id === "approval-run" ? { outcome: { id: "fixture-tool", toolId: "files.read", action: "read", state: "approval-required", approvalToken: "fixture-step-token", startedAt: now() } } : {}) });
const automationListeners = new Set<(value: AutomationWorkspaceEvent) => void>();
const receipts: unknown[] = [];
const changed = () => automationListeners.forEach(listener => listener({ brainId, type: "changed", at: now() }));
const automation: AutomationWorkspaceApi = {
  snapshot: async () => ({ brainId, catalogRevision: 2, counts: { definitions: definitions.length, activeDefinitions: 1, runs: runs.length, activeRuns: runs.filter(row => row.state !== "complete").length, reviewRequired: runs.filter(row => row.state === "review-required").length, approvals: 1 }, hostExecution: true, processResidentRequired: true }),
  definitions: async () => ({ entries: structuredClone(definitions), total: definitions.length }),
  program: async () => ({ definition: structuredClone(definitions[0]), program: structuredClone(NEW_ACTION_FILE) }),
  runs: async () => ({ entries: structuredClone(runs), total: runs.length }),
  steps: async (_brain, id) => ({ entries: [{ ...currentStep(id), index: 0, stepId: "earlier-receipt", state: "complete" }], total: 80, nextAfter: undefined }),
  currentStep: async (_brain, id) => structuredClone(currentStep(id)),
  onEvent: listener => { automationListeners.add(listener); return () => { automationListeners.delete(listener); }; }
};
let updateReceipts = () => {};
async function execute(invocation: ToolInvocation): Promise<ToolExecutionResult> {
  const result: ToolExecutionResult = { id: crypto.randomUUID(), toolId: invocation.toolId, action: invocation.action, state: "complete", startedAt: now(), finishedAt: now() };
  if (!invocation.approvalToken) return { ...result, state: "approval-required", approvalToken: "fixture-outer-review" };
  receipts.push({ invocation, fixtureOnly: true, actualHostEffects: false });
  const row = runs.find(value => value.id === invocation.arguments.runId);
  if (row) {
    row.state = invocation.action === "stop" ? "stopped" : invocation.action === "pause" || invocation.action === "review" ? "paused" : invocation.action === "approve" ? "complete" : "queued";
    row.detail = "Fixture controller accepted the request; no real effects ran.";
  }
  changed(); updateReceipts(); return result;
}

const packageValue: OmniPluginPackage = { format: "omni-plugin-package", formatVersion: 1, manifest: {
  id: "fixture.notes", version: "1.0.0", name: "Source notes", description: "Synthetic plugin controls and ordinary learning status.", author: "UI fixture", license: "Test only", documentation: "Synthetic docs.",
  capabilities: [{ toolId: "files.read", actions: ["read"], reason: "Read the exact selected file." }], tools: [], panels: [], themes: [],
  trainingFormats: [{ id: "notes", title: "Notes stream", description: "Use the existing text parser.", extensions: [".notes"], mediaTypes: [], parser: { kind: "builtin-stream", format: "text" } }]
} };
let entry: PluginCatalogEntry = { pluginId: "fixture.notes", name: "Source notes", description: packageValue.manifest.description, enabled: true, activeVersion: "1.0.0", activeContentSha256: hash, revision: 1, updatedAt: now(), versions: [{ version: "1.0.0", contentSha256: hash, installedAt: now(), sourceLabel: "Synthetic fixture" }] };
let access: PluginInstanceAccess = { brainId, pluginId: entry.pluginId, contentSha256: hash, enabled: false, capabilities: [{ toolId: "files.read", actions: ["read"], enabled: false }], toolPermissions: [], revision: 1, updatedAt: now(),
  semanticLearning: { brainId, pluginId: entry.pluginId, contentSha256: hash, experienceSha256: "e".repeat(64), state: "pending", attempt: 1, updatedAt: now() } };
const pluginListeners = new Set<(event: PluginEvent) => void>();
const pluginEvent = (type: PluginEvent["type"]) => pluginListeners.forEach(listener => listener({ type, brainId, pluginId: entry.pluginId, contentSha256: hash, catalogRevision: entry.revision, instanceRevision: access.revision, semanticState: access.semanticLearning?.state, at: now() }));
const pluginResult = () => ({ entry: structuredClone(entry), package: structuredClone(packageValue), contentSha256: hash, installed: false });
const plugins: PluginsApi = {
  list: async () => [structuredClone(entry)], package: async () => structuredClone(packageValue), instance: async () => structuredClone(access),
  installFile: async () => null, installPackage: async () => pluginResult(), edit: async () => pluginResult(), activate: async () => structuredClone(entry), exportFile: async () => null,
  setEnabled: async (_id, enabled, expectedHash, revision) => { if (expectedHash !== hash || revision !== entry.revision) throw new Error("Fixture stale catalog"); entry = { ...entry, enabled, revision: entry.revision + 1 }; pluginEvent("catalog"); return structuredClone(entry); },
  setInstanceAccess: async value => { access = { ...access, enabled: value.enabled, capabilities: value.capabilities, toolPermissions: value.toolPermissions, revision: access.revision + 1 }; pluginEvent("instance"); return structuredClone(access); },
  onEvent: listener => { pluginListeners.add(listener); return () => { pluginListeners.delete(listener); }; }
};
window.omni = { automation, plugins, tool: { execute } } as unknown as typeof window.omni;
const editSchedule: ScheduleRecord = { id: "action-timer", brainId, label: "Source check timer", revision: 9, kind: "timer", state: "paused", delivery: "automation", task: "", automation: { definitionId: "file", revisionSha256: hash, inputs: {} }, timeZone: "America/Chicago", loadWhenDue: false, remainingMs: 600000, repeatMs: 3600000, missedPolicy: "coalesce", nextIndex: 0, authorizationRevision: "fixture", createdAt: now(), updatedAt: now() };
function Fixture() {
  const [narrow, setNarrow] = useState(false), [view, setView] = useState<"actions" | "plugins" | "best-work" | "call-layout">("actions");
  const [editing, setEditing] = useState<ScheduleRecord>(), [notice, setNotice] = useState("");
  const [foreground, setForeground] = useState(true);
  const [callOpen, setCallOpen] = useState(true);
  const [workPreferences, setWorkPreferences] = useState<BestWorkPreferences>({ bestWorkEnabled: false, bestWorkExplorationBranches: 8 });
  const [, update] = useState(0); updateReceipts = () => update(value => value + 1);
  return <main style={{ maxWidth: 1050, padding: 16, margin: "0 auto", height: "100vh", overflowY: "auto", boxSizing: "border-box" }}>
    <h1>Action files & plugins · UI check</h1><p>Synthetic controls only. No model, permission, schedule, plugin installation or external action runs.</p>
    <p><a href="/tests/e2e/voice-call-fixture.html">Voice and task attachment controls</a></p>
    <div className="automation-buttons"><button className="button" onClick={() => setView("actions")}>Action controls</button><button className="button" onClick={() => setView("plugins")}>Plugin controls</button><button className="button" onClick={() => setView("best-work")}>Best Work controls</button><button className="button" onClick={() => setView("call-layout")}>Call layout</button><button className="button" onClick={() => setNarrow(value => !value)}>{narrow ? "Full surface" : "320px surface"}</button><button className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark"; }}>Switch appearance</button>
      {view === "actions" ? <button className="button" onClick={() => setEditing(structuredClone(editSchedule))}>Edit existing action schedule</button> : view === "plugins" ? <><button className="button" onClick={() => { access = { ...access, semanticLearning: { ...access.semanticLearning!, state: "complete", trainingSourceId: "fixture-receipt" } }; pluginEvent("semantic-learning"); }}>Complete synthetic learning</button><button className="button" onClick={() => { access = { ...access, revision: access.revision + 1, enabled: false }; pluginEvent("instance"); }}>Simulate external access change</button></> : view === "best-work" ? <button className="button" onClick={() => setForeground(value => !value)}>{foreground ? "End synthetic foreground turn" : "Start synthetic foreground turn"}</button> : null}
    </div><p role="status">{notice}</p><div style={{ width: narrow ? 320 : "100%", maxWidth: "100%" }}>{view === "actions" ? <AutomationWorkspace brainId={brainId} scheduleToEdit={editing} onScheduleEditFinished={() => setEditing(undefined)} onToast={setNotice} /> : view === "plugins" ? <PluginManager brainId={brainId} api={plugins} onNotice={setNotice} /> : view === "best-work" ? <BestWorkSettings preferences={workPreferences} applyBlockedReason={foreground ? "Apply after the current reply; editing remains available." : undefined} onSave={async value => { receipts.push({ bestWorkPreferences: value, fixtureOnly: true, modelExecution: false }); setWorkPreferences(value); updateReceipts(); }} /> : <section className="conversation" style={{ height: 550 }}><div className="chat-call-toolbar"><button onClick={() => setCallOpen(value => !value)} aria-expanded={callOpen}>Call controls</button></div><CallLayout open={callOpen} controls={<section className="chat-call-panel" hidden={!callOpen} aria-label="Synthetic call controls"><h2>Call</h2><p>No camera, microphone or screen is shared in this fixture.</p><button className="button" disabled>Start unavailable in layout check</button><details><summary>Source settings</summary><p>The actual call controller is not mounted here.</p></details></section>}><div className="conversation__date"><time>Layout check only</time></div><div className="message-stream-wrap"><div className="message-stream" role="log"><p>Conversation remains readable beside the call controls at desktop width.</p><p>Compact layouts stack the controls without covering the message input.</p></div></div><div className="composer-wrap"><label>Message input<textarea rows={2} aria-label="Synthetic message input" placeholder="Layout check; no message sends" /></label></div></CallLayout></section>}</div>
    <details><summary>Synthetic request receipts</summary><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{JSON.stringify(receipts, null, 2)}</pre></details>
  </main>;
}
document.documentElement.dataset.colorScheme = "dark"; document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
