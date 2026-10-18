import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { McpInteractionPanel } from "../../src/renderer/src/McpInteractionPanel";
import type { McpInteractionEvent, McpInteractionRecord, McpInteractionsApi } from "../../src/shared/mcpInteractions";
import "../../src/renderer/src/styles.css";

const listeners = new Set<(event: McpInteractionEvent) => void>();
const now = new Date().toISOString();
const origin = { brainId: "fixture", serverId: "synthetic-app", serverLabel: "Synthetic connected app", endpoint: "https://fixture.invalid/mcp", connectionEpoch: 1 };
const base = { format: "omni-mcp-interaction" as const, version: 1 as const, origin, createdAt: now, updatedAt: now, status: "pending" as const };
const records: McpInteractionRecord[] = [
  { ...base, id: "form", kind: "form", request: { mode: "form", message: "Choose formats for this fixture", requestedSchema: { type: "object", required: ["formats"], properties: {
    formats: { type: "array", title: "Output formats", minItems: 2, maxItems: 3, items: { anyOf: [{ const: "markdown", title: "Markdown" }, { const: "pdf", title: "PDF" }, { const: "html", title: "HTML" }] } },
    title: { type: "string", title: "Optional document title" }
  } } } },
  { ...base, id: "url", kind: "url", request: { mode: "url", message: "Review this fixture destination", url: "https://xn--bcher-kva.example/authorize?scope=documents%3Aread&request=synthetic", elicitationId: "fixture-only" } },
  { ...base, id: "input", kind: "sampling-input", systemPromptOmitted: "Synthetic server instruction; this will not be injected.",
    request: { brainId: "fixture", requestId: "private-fixture", messages: [{ role: "user", content: { type: "text", text: "Summarize the public fixture notes." } }], maxTokens: 128 } },
  { ...base, id: "output", kind: "sampling-output", request: { brainId: "fixture", requestId: "private-fixture", role: "assistant", content: { type: "text", text: "Synthetic output only; no model generated this fixture." },
    model: "OmniCortex/fixture", stopReason: "endTurn", inputSha256: "fixture-only", privateWorkspace: true, parametersMutated: false } },
  { ...base, id: "other-brain", origin: { ...origin, brainId: "other" }, kind: "url", request: { mode: "url", message: "Unrelated instance must not appear", url: "https://fixture.invalid/other", elicitationId: "other" } }
];
const receipts: unknown[] = [];
let refreshReceipts = () => {};
const api: McpInteractionsApi = {
  list: async () => structuredClone(records),
  onEvent: listener => { listeners.add(listener); return () => { listeners.delete(listener); }; },
  respond: async response => {
    const record = records.find(value => value.id === response.id)!;
    if (response.action === "accept" && record.kind === "form" && (!Array.isArray(response.values?.formats) || response.values.formats.length < 2)) throw new Error("Fixture validation: select at least two output formats.");
    if (response.action === "accept" && record.kind === "sampling-input" && (!Number.isSafeInteger(response.sampling?.maxTokens) || response.sampling!.maxTokens < 1)) throw new Error("Fixture validation: enter a positive whole token budget.");
    record.status = response.action === "accept" ? "accepted" : response.action === "decline" ? "declined" : "cancelled";
    receipts.push({ id: record.id, kind: record.kind, action: response.action, valueKeys: Object.keys(response.values ?? {}),
      ...(response.sampling ? { reviewedMessages: response.sampling.messages, maxTokens: response.sampling.maxTokens } : {}),
      ...(response.editedText !== undefined ? { editedText: response.editedText } : {}), remoteEffects: false });
    refreshReceipts(); listeners.forEach(listener => listener({ record: structuredClone(record) }));
    return structuredClone(record);
  }
};
function Fixture() {
  const [narrow, setNarrow] = useState(false);
  const [, update] = useState(0); refreshReceipts = () => update(value => value + 1);
  return <main style={{ maxWidth: 960, margin: "0 auto", padding: 20, height: "100vh", overflowY: "auto", boxSizing: "border-box" }}>
    <h1>Connected-app reviews fixture</h1><p>Synthetic UI transport only. No neural inference, browser navigation, credential or network request occurs here.</p>
    <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBlock: 16 }}>
      <button className="button" onClick={() => setNarrow(value => !value)}>{narrow ? "Full surface" : "320px surface"}</button>
      <button className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark"; }}>Switch appearance</button>
    </div>
    <div style={{ width: narrow ? 320 : "100%", maxWidth: "100%" }}><McpInteractionPanel brainId="fixture" api={api} /></div>
    <details><summary>Fixture review receipts</summary><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{JSON.stringify(receipts, null, 2)}</pre></details>
  </main>;
}
document.documentElement.dataset.colorScheme = "dark";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
