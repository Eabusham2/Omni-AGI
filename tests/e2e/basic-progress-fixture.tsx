import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { BuildProgressStatus, StorageOperationTiming } from "../../src/renderer/src/BuildProgressStatus";
import { ResponseOutputCounter } from "../../src/renderer/src/ResponseOutputCounter";
import { PluginThemeChooser, PluginThemeHost } from "../../src/renderer/src/PluginThemes";
import type { OmniPluginPackage, PluginCatalogEntry, PluginsApi, PluginEvent } from "../../src/shared/plugins";
import type { BrainStorageOperationEvent, BuildProgressEvent } from "../../src/shared/types";
import "../../src/renderer/src/styles.css";
import "../../src/renderer/src/pluginManager.css";

// Actual production components; synthetic catalog/progress only. No neural or host execution.
const hash = "a".repeat(64);
const packageValue: OmniPluginPackage = { format: "omni-plugin-package", formatVersion: 1, manifest: {
  id: "fixture.theme", version: "1.0.0", name: "Fixture colors", description: "Inert UI theme fixture",
  author: "Fixture", license: "MIT", documentation: "UI test only", capabilities: [], tools: [], panels: [], trainingFormats: [],
  themes: [{ id: "sea", title: "Sea glass", tokens: { "--accent": "#087f8c", "--text-2": "#185362" } }]
} };
let catalog: PluginCatalogEntry = { pluginId: "fixture.theme", name: "Fixture colors", description: "Inert UI theme fixture",
  enabled: true, activeVersion: "1.0.0", activeContentSha256: hash, revision: 1, updatedAt: new Date().toISOString(),
  versions: [{ version: "1.0.0", contentSha256: hash, installedAt: new Date().toISOString(), sourceLabel: "synthetic UI" }] };
const listeners = new Set<(event: PluginEvent) => void>();
const api = { list: async () => [structuredClone(catalog)], package: async () => structuredClone(packageValue),
  onEvent: (listener: (event: PluginEvent) => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; }
} as PluginsApi;
const baseBuild: BuildProgressEvent = { sequence: 1, phase: "readiness", progress: 0.999, label: "Saving initial learning", elapsedMs: 30_000 };
const baseStorage = { state: "running", phase: "finalizing", elapsedMs: 60_000 } as BrainStorageOperationEvent;
function Fixture() {
  const [progress, setProgress] = useState(baseBuild), [storage, setStorage] = useState(baseStorage);
  const [narrow, setNarrow] = useState(false), [enabled, setEnabled] = useState(true);
  return <main className="control-fixture" style={{ width: narrow ? 360 : "100%", maxWidth: 1000, height: "100vh", overflowY: "auto", margin: "0 auto", padding: 16, boxSizing: "border-box" }}>
    <PluginThemeHost api={api} />
    <h1>Existing Omni controls · synthetic UI check</h1><p>No brain, training, provider request or host action.</p>
    <div className="automation-buttons">
      <button className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "light" ? "dark" : "light"; }}>Switch appearance</button>
      <button className="button" onClick={() => setNarrow(value => !value)}>Toggle 360px surface</button>
      <button className="button" onClick={() => setProgress({ ...baseBuild, sequence: progress.sequence + 1, etaMs: 15_000 })}>Supply measured ETA</button>
      <button className="button" onClick={() => setProgress({ ...baseBuild, sequence: progress.sequence + 1, phase: "complete", progress: 1, elapsedMs: 45_000, label: "Saved" })}>Durable completion</button>
      <button className="button" onClick={() => setStorage({ ...storage, state: storage.state === "paused" ? "running" : "paused" })}>Pause/resume storage timing</button>
      <button className="button" onClick={() => { catalog = { ...catalog, enabled: !catalog.enabled, revision: catalog.revision + 1 }; setEnabled(catalog.enabled);
        for (const listener of listeners) listener({ type: "catalog", pluginId: catalog.pluginId } as PluginEvent); }}>Toggle catalog enabled</button>
    </div>
    <section className="loading-overlay build-readiness-overlay" style={{ position: "relative", inset: "auto", minHeight: 220, margin: "16px 0", width: "100%", boxSizing: "border-box" }}>
      <strong>{progress.label}</strong><BuildProgressStatus progress={progress} />
    </section>
    <section className="storage-operation-card" style={{ position: "relative", inset: "auto", width: "100%", margin: "16px 0", boxSizing: "border-box" }}>
      <strong>Storage operation</strong><dl><StorageOperationTiming operation={storage} /></dl>
    </section>
    <div className="message__meta"><strong>Responding</strong><ResponseOutputCounter presentation={{ phase: "streaming", outputTokens: 12345678, tokensPerSecond: 12.34,
      label: "12.346M tokens · 12.3 tok/s", ariaLabel: "Live response output: 12,345,678 output tokens used. Average delivered speed since first output: 12.34 tokens per second." }} /></div>
    <div className="plugin-manager" style={{ marginTop: 24 }}><PluginThemeChooser entry={{ ...catalog, enabled }} packageValue={packageValue} /></div>
  </main>;
}
document.documentElement.dataset.colorScheme = "light";
document.documentElement.dataset.layout = "studio";
createRoot(document.getElementById("root")!).render(<Fixture />);
