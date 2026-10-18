import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { McpAppView } from "../../src/renderer/src/McpAppView";
import { PluginStaticPanel } from "../../src/renderer/src/PluginManager";
import "../../src/renderer/src/styles.css";

declare global { interface Window { mcpAppFixture: { descriptor(): Promise<{ id: string }>; trace(): Promise<unknown[]>; pluginPanel(): Promise<{ title: string; pluginId: string; contentSha256: string; panelId: string }>;
  setAppAccess(enabled: boolean): Promise<void>; onAppAccess(listener: (enabled: boolean) => void): () => void;
  acknowledgement(mode: "normal" | "hold" | "reject" | "retire-after-fullscreen"): Promise<void>; releaseAcknowledgements(): Promise<void> } } }
function Fixture() {
  const [id, setId] = useState<string>();
  const [trace, setTrace] = useState<unknown[]>([]);
  const [error, setError] = useState("");
  const [access, setAccess] = useState(true);
  const [acknowledgement, setAcknowledgement] = useState<"normal" | "hold" | "reject" | "retire-after-fullscreen">("normal");
  const [panel, setPanel] = useState<Awaited<ReturnType<Window["mcpAppFixture"]["pluginPanel"]>>>();
  useEffect(() => { let active = true; void window.mcpAppFixture.descriptor().then(value => { if (active) setId(value.id); }).catch(failure => { if (active) setError(String(failure)); }); return () => { active = false; }; }, []);
  useEffect(() => window.mcpAppFixture.onAppAccess(setAccess), []);
  return <main style={{ maxWidth: 960, margin: "0 auto", padding: 20, height: "100vh", overflowY: "auto", boxSizing: "border-box" }}>
    <h1>Embedded connected-app check</h1><p>Real App host, custom protocol and sandbox. Synthetic completed call/resource; no brain, account, external network or filesystem action.</p>
    <button className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark"; }}>Switch appearance</button>
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, marginBlock: 10 }}>
      <label>Fixture display acknowledgement <select value={acknowledgement} onChange={event => {
        const mode = event.currentTarget.value as typeof acknowledgement;
        void window.mcpAppFixture.acknowledgement(mode).then(() => setAcknowledgement(mode)).catch(failure => setError(String(failure)));
      }}><option value="normal">Normal</option><option value="hold">Hold until released</option><option value="reject">Refuse acknowledgement</option><option value="retire-after-fullscreen">Retire access 3s after fullscreen</option></select></label>
      <button className="button" onClick={() => void window.mcpAppFixture.releaseAcknowledgements().catch(failure => setError(String(failure)))}>Release held acknowledgements</button>
      <button className="button" onClick={() => void window.mcpAppFixture.setAppAccess(!access).then(() => setAccess(!access)).catch(failure => setError(String(failure)))}>
        {access ? "Retire fixture App access" : "Restore fixture App access"}</button>
    </div>
    {id ? <McpAppView brainId="fixture" viewId={id} /> : <p role="status">Preparing the isolated host…</p>}
    <button className="button" onClick={() => void window.mcpAppFixture.pluginPanel().then(setPanel).catch(failure => setError(String(failure)))}>Show static plugin panel</button>
    {panel ? <PluginStaticPanel {...panel} /> : null}
    {error ? <p role="alert">{error}</p> : null}
    <button className="button" onClick={() => void window.mcpAppFixture.trace().then(setTrace)}>Read actual host trace</button>
    <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{JSON.stringify(trace, null, 2)}</pre>
  </main>;
}
document.documentElement.dataset.colorScheme = "dark";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
