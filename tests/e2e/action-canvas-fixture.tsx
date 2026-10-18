import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { AutomationWorkspace } from "../../src/renderer/src/AutomationWorkspace";
import type { AutomationWorkspaceApi } from "../../src/shared/automation";
import type { ToolInvocation } from "../../src/shared/types";
import "../../src/renderer/src/styles.css";

// UI-only synthetic transport. No model, host effects, real save or grants.
const receipts: ToolInvocation[] = []; let refreshReceipts = () => {};
const empty = async () => ({ entries: [], total: 0 });
const automation: AutomationWorkspaceApi = {
  snapshot: async () => ({ brainId: "canvas-fixture", catalogRevision: 0, counts: { definitions: 0, activeDefinitions: 0, runs: 0, activeRuns: 0, reviewRequired: 0, approvals: 0 }, hostExecution: true, processResidentRequired: true }),
  definitions: empty, runs: empty, steps: empty, currentStep: async () => null,
  program: async () => { throw new Error("There are no stored fixture files."); }, onEvent: () => () => {},
};
window.omni = { automation, tool: { execute: async (invocation: ToolInvocation) => {
  receipts.push(structuredClone(invocation)); refreshReceipts();
  return { id: "fixture-receipt", toolId: invocation.toolId, action: invocation.action, state: "complete", startedAt: new Date().toISOString() };
} } } as unknown as NonNullable<typeof window.omni>;
function Fixture() {
  const [, refresh] = useState(0); refreshReceipts = () => refresh(value => value + 1);
  const [narrow, setNarrow] = useState(false);
  return <main style={{ maxWidth: narrow ? 360 : 1050, margin: "0 auto", padding: 16, height: "100vh", overflow: "auto", boxSizing: "border-box" }}>
    <h1>Action canvas controls</h1><p>Synthetic transport only. No AI, real tool execution or saved instance.</p>
    <div className="automation-buttons"><button className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "light" ? "dark" : "light"; }}>Switch appearance</button><button className="button" onClick={() => setNarrow(value => !value)}>Toggle narrow surface</button></div>
    <AutomationWorkspace brainId="canvas-fixture" />
    <details><summary>Synthetic save receipts · {receipts.length}</summary><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{JSON.stringify(receipts, null, 2)}</pre></details>
  </main>;
}
document.documentElement.dataset.colorScheme = "light";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<React.StrictMode><Fixture /></React.StrictMode>);
