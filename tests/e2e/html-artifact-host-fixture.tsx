import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { HtmlArtifactView } from "../../src/renderer/src/HtmlArtifactView";
import type { ActionEvent } from "../../src/shared/types";
import "../../src/renderer/src/styles.css";

declare global {
  interface Window {
    htmlArtifactFixture: {
      descriptors(): Promise<{ enabled: ActionEvent; dormant: ActionEvent }>;
      trace(): Promise<Array<Record<string, unknown>>>;
    };
  }
}

function Fixture() {
  const [descriptors, setDescriptors] = useState<{ enabled: ActionEvent; dormant: ActionEvent }>();
  const [trace, setTrace] = useState<Array<Record<string, unknown>>>([]);
  const [error, setError] = useState("");
  useEffect(() => { let active = true;
    void window.htmlArtifactFixture.descriptors().then(value => { if (active) setDescriptors(value); })
      .catch(failure => { if (active) setError(String(failure)); });
    return () => { active = false; };
  }, []);
  return <main style={{ maxWidth: 1020, margin: "0 auto", padding: 20, height: "100vh", overflowY: "auto", boxSizing: "border-box" }}>
    <h1>Saved HTML artifact host check</h1>
    <p>Real artifact store, SQLite custody, service, custom protocol, CSP, opaque iframe and renderer component. The only host action is a fixture-only readonly callback that always requires visible Ask review. No brain, model, provider, real file, credential or external effect is used.</p>
    <button className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark"; }}>Switch appearance</button>
    {descriptors ? <div style={{ display: "grid", gap: 20, marginTop: 18 }}>
      <section><h2>Current-owner saved artifact</h2><p>Increment stays inside the sandbox. Host inspection must show an exact Studio-owned review card.</p><HtmlArtifactView event={descriptors.enabled} /></section>
      <section><h2>Copied artifact starts dormant</h2><p>The exact presentation opens without its script or bridge until explicit current-owner activation.</p><HtmlArtifactView event={descriptors.dormant} /></section>
    </div> : <p role="status">Creating the task-owned SQLite fixture…</p>}
    {error ? <p role="alert">{error}</p> : null}
    <button className="button" style={{ marginTop: 18 }} onClick={() => void window.htmlArtifactFixture.trace().then(setTrace).catch(failure => setError(String(failure)))}>Read actual host trace</button>
    <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{JSON.stringify(trace, null, 2)}</pre>
  </main>;
}

document.documentElement.dataset.colorScheme = "dark";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
