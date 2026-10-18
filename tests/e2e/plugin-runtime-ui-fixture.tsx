import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { PluginRuntimeControls } from "../../src/renderer/src/PluginRuntimeControls";
import { PluginRuntimeApprovalQueue } from "../../src/renderer/src/PluginRuntimeApprovals";
import { SparseSubstrateUiFixture } from "./sparse-substrate-ui-fixture";
import { PluginRuntimeContributionsProvider, PluginRuntimeSlot } from "../../src/renderer/src/PluginRuntimeContributions";
import type { OmniPluginPackage, PluginCatalogEntry, PluginInstanceAccess, PluginsApi } from "../../src/shared/plugins";
import type { PluginRuntimeApi, PluginRuntimeEvent, PluginRuntimeMountedContribution, PluginRuntimePendingApproval, PluginRuntimeSession } from "../../src/shared/pluginRuntime";
import "../../src/renderer/src/styles.css";
import "../../src/renderer/src/pluginManager.css";

// Existing components, inert synthetic transport: no module JS, brain, key or host action runs.
const hash = "a".repeat(64), brainId = "plugin-ui-fixture";
const packageValue: OmniPluginPackage = { format: "omni-plugin-package", formatVersion: 1, manifest: {
  id: "fixture.module", version: "1.0.0", name: "Fixture feature", description: "Inert UI transport", author: "Fixture", license: "MIT", documentation: "UI only",
  capabilities: [{ toolId: "system.files", actions: ["read"], reason: "Synthetic review only" }], tools: [], panels: [], themes: [], trainingFormats: [],
  runtimeModules: [{ id: "helper", title: "Useful feature", description: "Synthetic module controls · no source executes here", script: "omniRuntime.onActivate(() => {});",
    events: ["brain.lifecycle", "chat.action"], ui: { kinds: ["button", "panel", "style"], slots: ["chat.toolbar", "tools"] }, dynamicTools: true }]
} };
const entry: PluginCatalogEntry = { pluginId: "fixture.module", name: "Fixture feature", description: "Inert UI transport", enabled: true,
  activeVersion: "1.0.0", activeContentSha256: hash, revision: 1, updatedAt: new Date().toISOString(),
  versions: [{ version: "1.0.0", contentSha256: hash, installedAt: new Date().toISOString(), sourceLabel: "synthetic UI" }] };
const access = { brainId, pluginId: entry.pluginId, contentSha256: hash, enabled: true, revision: 1 } as PluginInstanceAccess;
const pluginsApi = { list: async () => [entry], instance: async () => access, package: async () => packageValue, onEvent: () => () => {} } as unknown as PluginsApi;
const listeners = new Set<(event: PluginRuntimeEvent) => void>();
let sessions: PluginRuntimeSession[] = [], approvals: PluginRuntimePendingApproval[] = [], additions: PluginRuntimeMountedContribution[] = [], count = 0;
const publish = () => { for (const listener of listeners) listener({ brainId, kind: "contributions" }); };
const runtimeApi: PluginRuntimeApi = {
  activate: async request => {
    const session: PluginRuntimeSession = { format: "omni-plugin-runtime-session", version: 1, brainId, pluginId: entry.pluginId,
      moduleId: "helper", contentSha256: hash, sessionId: crypto.randomUUID(), catalogRevision: 1, instanceRevision: 1, state: "active", startedAt: new Date().toISOString() };
    sessions = [session];
    const common = { brainId, pluginId: entry.pluginId, contentSha256: hash, moduleId: "helper", sessionId: session.sessionId, revision: 1 };
    additions = [{ ...common, id: "counter", kind: "button", slot: "chat.toolbar", title: "Increment fixture", event: "increment" },
      { ...common, id: "panel", kind: "panel", slot: "tools", title: "Inert result panel", html: "<h2>Fixture result</h2><p>No module source or host action executed.</p>" }];
    approvals = [{ brainId, sessionId: session.sessionId, rpcId: request.requestId, reviewId: "review-fixture", pluginId: entry.pluginId,
      contentSha256: hash, moduleId: "helper", executionId: "execution-fixture", invocation: { brainId, toolId: "system.files", action: "read", arguments: { path: "/synthetic/readme.txt" } } }];
    publish(); return session;
  },
  stop: async (_id, sessionId) => { sessions = sessions.map(item => ({ ...item, state: "stopped" as const, finishedAt: new Date().toISOString() })); additions = []; approvals = []; publish(); return sessions.find(item => item.sessionId === sessionId)!; },
  list: async () => ({ sessions, total: sessions.length }),
  history: async () => ({ events: [{ sequence: 1, kind: "fixture", sha256: hash, value: { synthetic: true } }], total: 1 }),
  readHistoryEvent: async () => ({ text: "{\"synthetic\":true}", bytes: 18, sha256: hash }),
  stateSources: async () => ({ sources: [{ sha256: "b".repeat(64), owner: "parent-fixture", bytes: 4096 }], total: 1 }),
  stateConflicts: async () => ({ conflicts: [], total: 0 }),
  readSourceState: async () => ({ revision: 0, value: { synthetic: true } }),
  readSourceStateBytes: async () => ({ text: "{\"synthetic\":true}", bytes: 18, sha256: hash }),
  selectSourceState: async () => {},
  contributions: async () => structuredClone(additions),
  invokeContribution: async () => { count++; additions = additions.map(item => item.id === "counter" ? { ...item, title: `Fixture count ${count}`, revision: item.revision + 1 } : item); publish(); },
  approvals: async () => structuredClone(approvals),
  review: async request => { approvals = approvals.filter(item => item.reviewId !== request.reviewId); publish(); return { format: "omni-plugin-runtime-reply", version: 1, id: request.rpcId, state: request.decision === "approve" ? "complete" : "failed", value: { synthetic: true } }; },
  onEvent: listener => { listeners.add(listener); return () => { listeners.delete(listener); }; }
};
function Fixture() {
  const [narrow, setNarrow] = useState(false), [notice, setNotice] = useState(""), [training, setTraining] = useState(false), [sparse, setSparse] = useState(false);
  return <main style={{ width: narrow ? 360 : "100%", maxWidth: 1060, height: "100vh", overflowY: "auto", margin: "0 auto", padding: 16, boxSizing: "border-box" }}>
    <h1>Existing plugin controls · synthetic UI check</h1><p>No model, module source, external request or real host action runs.</p>
    <div className="automation-buttons"><button className="button" onClick={() => setNarrow(value => !value)}>Toggle 360px surface</button>
      <button className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "light" ? "dark" : "light"; }}>Switch appearance</button>
      <button className="button" onClick={() => setSparse(value => !value)}>Show connection updates</button>
      <button className="button" onClick={() => void (async () => {
        await runtimeApi.activate({ brainId, pluginId: entry.pluginId, contentSha256: hash, moduleId: "helper", expectedCatalogRevision: 1, expectedInstanceRevision: 1, requestId: "synthetic-training-request" });
        approvals.push({ ...approvals[0]!, rpcId: "synthetic-second-request", reviewId: "synthetic-second-review", executionId: "synthetic-second-execution",
          invocation: { brainId, toolId: "system.files", action: "read", arguments: { path: "/synthetic/long-training-source-path/with-complete-readable-operands/document.dat" } } });
        setTraining(true); publish();
      })()}>Show synthetic training review</button></div>
    {sparse ? <SparseSubstrateUiFixture /> : training ? <section className="build-readiness-overlay" style={{ position: "relative", boxSizing: "border-box", width: "100%", marginTop: 16 }}>
      <h2>Training action review</h2><p>No training is running; two synthetic pending actions only.</p>
      <PluginRuntimeApprovalQueue brainId={brainId} api={runtimeApi} />
    </section> : <>
    <PluginRuntimeContributionsProvider brainId={brainId} api={runtimeApi} onNotice={setNotice}>
      <PluginRuntimeSlot slot="chat.toolbar" />
      <div className="plugin-manager"><PluginRuntimeControls brainId={brainId} entry={entry} packageValue={packageValue} pluginsApi={pluginsApi} runtimeApi={runtimeApi} onNotice={setNotice} /></div>
      <PluginRuntimeSlot slot="tools" showProviderError />
    </PluginRuntimeContributionsProvider>
    </>}
    <p role="status">{notice}</p>
  </main>;
}
document.documentElement.dataset.colorScheme = "light";
createRoot(document.getElementById("root")!).render(<Fixture />);
