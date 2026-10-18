import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { ExternalAdvicePanel } from "../../src/renderer/src/ExternalAdvicePanel";
import { ActionReviewDetails } from "../../src/renderer/src/ActionReviewDetails";
import type { ExternalAdviceApi, ExternalAdviceEvent, ExternalAdviceRequest, ExternalAdviceSettings } from "../../src/shared/externalAdvice";
import type { ToolExecutionResult } from "../../src/shared/types";
import "../../src/renderer/src/styles.css";

// Inert UI transport: no provider, real credential store, brain, tool or grant.
const configuration = { enabled: false, provider: "openai" as const, model: "fixture-model", maxOutputTokens: 2048,
  context: { includeTask: true, includeActivity: true, maxCharacters: 16000 } };
let settings: ExternalAdviceSettings = { format: "omni-external-advice-settings", version: 1, brainId: "advice-fixture", revision: 1,
  updatedAt: new Date().toISOString(), assistant: configuration, reviewer: configuration,
  deviceAuthority: { assistant: false, reviewer: false }, deviceAuthorityRequired: true, credentialsStoredInBrain: false };
const requests: ExternalAdviceRequest[] = [], listeners = new Set<(event: ExternalAdviceEvent) => void>();
const keys = ["openai", "anthropic", "gemini", "openrouter", "openai-compatible-local"].map(provider => ({ provider: provider as typeof configuration.provider, configured: false, required: provider !== "openai-compatible-local", persistence: "none" as const, revision: 0 }));
const hash = "a".repeat(64);
let preview: Awaited<ReturnType<ExternalAdviceApi["preview"]>>;
const api: ExternalAdviceApi = {
  settings: async () => structuredClone(settings),
  configure: async (_id, input) => { settings = { ...settings, revision: settings.revision + 1, [input.channel]: structuredClone(input.configuration),
    deviceAuthority: { ...settings.deviceAuthority!, [input.channel]: input.configuration.enabled } }; return structuredClone(settings); },
  credentialStatus: async () => keys,
  saveCredential: async input => { const key = keys.find(item => item.provider === input.provider)!; key.configured = true; key.revision++; return keys; },
  removeCredential: async provider => { keys.find(item => item.provider === provider)!.configured = false; return keys; },
  preview: async (_id, input) => { preview = { format: "omni-external-advice-preview", version: 1, brainId: "advice-fixture", channel: input.channel,
    provider: settings.assistant.provider, model: settings.assistant.model, prompt: input.prompt, requestText: `${input.prompt}\nSelected synthetic task: explain this recorded work.`,
    context: { format: "omni-external-advice-selected-context", version: 1, brainId: "advice-fixture", channel: input.channel, selection: configuration.context,
      items: [{ id: "fixture-task", kind: "task", label: "Synthetic recorded task", text: "explain this recorded work", contentSha256: hash }], includedCharacters: 26, omittedItems: 42, truncated: true },
    settingsRevision: settings.revision, authorityRevision: 1, authorityOwnerEpoch: "fixture-device", contextSha256: hash, sha256: hash,
    maxOutputTokens: 2048, advisoryOnly: true, automaticLearning: false }; return preview; },
  start: async (_id, input) => { const at = new Date().toISOString();
    const item: ExternalAdviceRequest = { ...preview, format: "omni-external-advice-request", id: input.requestId, state: "complete", response: "Synthetic assistant reply. OmniCortex was not run.",
      createdAt: at, updatedAt: at, requestSha256: hash, previewSha256: hash, hostActionAuthority: false, credentialsStoredInBrain: false };
    requests.push(item); for (const listener of listeners) listener({ brainId: item.brainId, request: item }); return item; },
  cancel: async (brainId, requestId) => ({ brainId, requestId, cancelled: false }),
  history: async () => ({ entries: structuredClone(requests), total: requests.length, activeRequestIds: [] }),
  onEvent: listener => { listeners.add(listener); return () => { listeners.delete(listener); }; }
};
const execution: ToolExecutionResult = { id: "review-fixture", toolId: "system.shell", action: "run", state: "approval-required", approvalToken: "synthetic-only",
  startedAt: new Date().toISOString(), actionReview: { kind: "action-review", formatVersion: 1, reviewId: "fixture-review", brainId: "advice-fixture", requestId: "fixture-request",
    configurationRevision: "1", ownerRevision: "fixture:1", reviewerId: "synthetic-adviser", permissionRevision: "1", proposalSha256: hash,
    proposedInvocation: { brainId: "advice-fixture", toolId: "system.shell", action: "run", arguments: { command: "synthetic proposed command — never executed" } },
    verdict: "ask", summary: "Review this proposed action before allowing it. This is synthetic advice, not a safety guarantee.", evidence: [{ sourceId: "action", quote: "synthetic proposed command" }],
    disposition: "reviewed", startedAt: new Date().toISOString(), finishedAt: new Date().toISOString(), elapsedMs: 1030, observedContext: [], requestBytes: 200, responseBytes: 150 } };
function Fixture() {
  const [open, setOpen] = useState(true), [narrow, setNarrow] = useState(false), [denial, setDenial] = useState("");
  return <main style={{ minHeight: "100vh", position: "relative", maxWidth: narrow ? 380 : 1100, margin: "0 auto", padding: 16, boxSizing: "border-box" }}>
    <h1>Existing Omni components · synthetic UI check</h1><p>No brain, external request, real key or host action.</p>
    <div className="automation-buttons"><button className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "light" ? "dark" : "light"; }}>Switch appearance</button>
      <button className="button" onClick={() => setNarrow(value => !value)}>Toggle narrow surface</button><button className="button" onClick={() => setOpen(true)}>Open assistant</button>
      <button className="button" onClick={async () => {
        const selected = await api.preview("advice-fixture", { channel: "assistant", prompt: "Interrupted synthetic case" });
        const at = new Date().toISOString();
        const orphan: ExternalAdviceRequest = { ...selected, format: "omni-external-advice-request", version: 1, id: "orphan-ui-fixture", state: "streaming", createdAt: at, updatedAt: at,
          response: "Only this prefix reached a saved checkpoint.", requestSha256: hash, previewSha256: hash, hostActionAuthority: false, credentialsStoredInBrain: false };
        requests.push(orphan);
        for (const listener of listeners) listener({ brainId: orphan.brainId, request: orphan, active: false, persistence: { receivedCharacters: orphan.response.length, persistedCharacters: orphan.response.length } });
      }}>Show interrupted-prefix case</button></div>
    <div style={{ marginTop: 40, maxWidth: 540 }}><ActionReviewDetails execution={execution} onDecline={async instruction => { setDenial(instruction ? `Declined; draft: ${instruction}` : "Declined exact synthetic action; nothing dispatched."); }} /><p role="status">{denial}</p></div>
    <ExternalAdvicePanel brainId="advice-fixture" api={api} hidden={!open} onClose={() => setOpen(false)} />
  </main>;
}
document.documentElement.dataset.colorScheme = "light"; document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<React.StrictMode><Fixture /></React.StrictMode>);
