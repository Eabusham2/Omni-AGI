import React, { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import { ApiTeacherModelPicker } from "../../src/renderer/src/ApiTeacherModelPicker";
import type { ApiTeacherModelDiscoveryRequest, ApiTeacherModelDiscoveryResult, ApiTeacherProvider, OmniApi } from "../../src/shared/types";
import "../../src/renderer/src/styles.css";
const receipts: unknown[] = [], held = new Map<string, { request: ApiTeacherModelDiscoveryRequest; resolve: (value: ApiTeacherModelDiscoveryResult) => void; reject: (reason: Error) => void }>();
let hold = false, redraw = () => {};
const result = (request: ApiTeacherModelDiscoveryRequest): ApiTeacherModelDiscoveryResult => ({ format: "omni-api-teacher-model-discovery", version: 1, ...request, sourceEndpoint: "https://example.invalid/fixture-only", discoveredAt: "2026-10-13T00:00:00Z", pageCount: 1, complete: true, generationCompatibilityVerified: false, models: [{ provider: request.provider, id: "synthetic-code-model" }, { provider: request.provider, id: "synthetic-general-model" }] });
const api: Pick<OmniApi["teacher"], "discoverModels" | "cancelModelDiscovery"> = {
  async discoverModels(request) { receipts.push({ operation: "discover", request }); redraw(); if (!hold) return result(request); return new Promise((resolve, reject) => { held.set(request.requestId, { request, resolve, reject }); }); },
  async cancelModelDiscovery(request) { receipts.push({ operation: "cancel", request }); const prior = held.get(request.requestId); if (prior) { held.delete(request.requestId); prior.reject(new Error("Synthetic discovery cancelled; manual model unchanged.")); } redraw(); return { ...request, cancelled: Boolean(prior) }; }
};
function Fixture() {
  const [provider, setProvider] = useState<ApiTeacherProvider>("openai"), [model, setModel] = useState("my-manual-model"), [configured, setConfigured] = useState(false), [epoch, setEpoch] = useState(0), [dark, setDark] = useState(false), [, update] = useState(0);
  redraw = () => update(value => value + 1);
  return <main style={{ maxWidth: 860, height: "100vh", overflow: "auto", padding: 16, margin: "0 auto", boxSizing: "border-box" }}><h1>Provider model picker check</h1><p>Actual picker with synthetic main replies and Strict Mode. No real key, provider request, model, training or permission runs.</p>
    <div className="integration-actions"><button className="button" onClick={() => { setDark(value => !value); document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; }}>Switch theme</button><button className="button" onClick={() => setEpoch(value => value + 1)}>Synthetic credential changed</button><button className="button" onClick={() => { for (const entry of held.values()) entry.resolve(result(entry.request)); held.clear(); }}>Finish synthetic discovery</button></div>
    <div className="integration-actions" style={{ marginBlock: 12 }}><label><input type="checkbox" checked={configured} onChange={event => setConfigured(event.target.checked)} />Synthetic key configured</label><label><input type="checkbox" onChange={event => { hold = event.target.checked; }} />Hold synthetic discovery</label></div>
    <section className="surface integration-card" style={{ marginBlock: 16 }}><div className="integration-fields integration-fields--three"><label><span>Provider</span><select value={provider} onChange={event => setProvider(event.target.value as ApiTeacherProvider)}><option value="openai">OpenAI</option><option value="anthropic">Claude</option><option value="gemini">Gemini</option></select></label>
      <ApiTeacherModelPicker provider={provider} model={model} onChange={setModel} configured={configured} credentialEpoch={epoch} api={api} /><label><span>Key storage</span><input readOnly value="No real key in this fixture" /></label></div></section>
    <p role="status">Manual model: {model}</p><details><summary>Synthetic receipts</summary><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", fontSize: 12 }}>{JSON.stringify(receipts, null, 2)}</pre></details>
  </main>;
}
document.documentElement.dataset.colorScheme = "light"; document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<StrictMode><Fixture /></StrictMode>);
