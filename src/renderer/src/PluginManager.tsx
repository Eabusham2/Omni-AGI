import { useEffect, useMemo, useRef, useState } from "react";
import {
  pluginToolId,
  type OmniPluginPackage,
  type PluginCatalogEntry,
  type PluginInstanceAccess,
  type PluginsApi
} from "../../shared/plugins";
import type { ToolPermissionLevel } from "../../shared/types";
import { pluginPanelUrl } from "../../shared/pluginPanels";
import { refreshedPluginAccess } from "./pluginAccessPresentation";
import { PluginInteractivePanel } from "./PluginInteractivePanel";
import { PluginPanelHistory } from "./PluginPanelHistory";
import { PluginThemeChooser } from "./PluginThemes";
import { PluginRuntimeControls } from "./PluginRuntimeControls";
import { PluginModuleAuthoring } from "./PluginModuleAuthoring";
import { newPluginAuthoringPackage } from "./pluginModuleDraft";
import "./pluginManager.css";

function currentApi(): PluginsApi | undefined {
  return typeof window === "undefined" ? undefined : window.omni?.plugins;
}

/** Validated static markup is also isolated in an opaque, no-capability frame. */
export function PluginStaticPanel({ title, pluginId, contentSha256, panelId }: {
  title: string;
  pluginId: string;
  contentSha256: string;
  panelId: string;
}) {
  return <iframe className="plugin-static-panel" title={`${title} · static plugin panel`}
    sandbox="" src={pluginPanelUrl({ pluginId, contentSha256, panelId })} referrerPolicy="no-referrer"
    allow="camera 'none'; microphone 'none'; geolocation 'none'; clipboard-read 'none'; clipboard-write 'none'; display-capture 'none'" />;
}

function defaultAccess(packageValue: OmniPluginPackage, brainId: string, hash: string): PluginInstanceAccess {
  return {
    brainId,
    pluginId: packageValue.manifest.id,
    contentSha256: hash,
    enabled: false,
    capabilities: packageValue.manifest.capabilities.map((capability) => ({ ...capability, enabled: false })),
    toolPermissions: packageValue.manifest.tools.map((tool) => ({
      toolId: pluginToolId(packageValue.manifest.id, tool.id),
      level: "off" as const
    })),
    revision: 0,
    updatedAt: new Date(0).toISOString()
  };
}

export function PluginManager({ brainId, api = currentApi(), onNotice, onCapabilitiesChanged }: {
  brainId?: string;
  api?: PluginsApi;
  onNotice?: (message: string) => void;
  onCapabilitiesChanged?: (pluginId: string) => void;
}) {
  const [entries, setEntries] = useState<PluginCatalogEntry[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [loadedPackage, setPackageValue] = useState<OmniPluginPackage>();
  const [loadedAccess, setAccess] = useState<PluginInstanceAccess>();
  const [loadedScope, setLoadedScope] = useState("");
  const [editor, setEditor] = useState(() => JSON.stringify(newPluginAuthoringPackage(), null, 2));
  const [editBase, setEditBase] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const selected = useMemo(() => entries.find((entry) => entry.pluginId === selectedId), [entries, selectedId]);
  const scope = JSON.stringify([brainId ?? null, selected?.pluginId ?? null, selected?.activeContentSha256 ?? null]);
  const scopeRef = useRef(scope), brainOwner = useRef(brainId), mounted = useRef(true), running = useRef(false);
  const catalogRead = useRef(0), accessRead = useRef(0), accessDirty = useRef(false);
  const accessDraft = useRef<PluginInstanceAccess | undefined>(undefined);
  const readyScope = useRef(""), refreshWhileLoading = useRef(false);
  const changedHandler = useRef(onCapabilitiesChanged);
  scopeRef.current = scope; brainOwner.current = brainId;
  changedHandler.current = onCapabilitiesChanged;
  const packageValue = loadedScope === scope ? loadedPackage : undefined;
  const access = selected && loadedAccess && loadedScope === scope && loadedAccess.brainId === brainId &&
    loadedAccess.pluginId === selected.pluginId && loadedAccess.contentSha256 === selected.activeContentSha256
    ? loadedAccess : undefined;
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const updateAccess = (value: PluginInstanceAccess | undefined) => { accessDraft.current = value; setAccess(value); };

  const refresh = async (preferred = selectedId): Promise<void> => {
    if (!api) return;
    const request = ++catalogRead.current;
    const next = await api.list();
    if (!mounted.current || brainOwner.current !== brainId || request !== catalogRead.current) return;
    setEntries(next);
    const id = next.some((entry) => entry.pluginId === preferred) ? preferred : next[0]?.pluginId ?? "";
    setSelectedId(id);
  };

  useEffect(() => { void refresh().catch((failure) => setError(String(failure))); }, [api]);
  useEffect(() => {
    if (!api || !selected) { setPackageValue(undefined); updateAccess(undefined); setLoadedScope(""); readyScope.current = ""; return; }
    let current = true;
    const requestedScope = scope;
    accessRead.current++;
    accessDirty.current = false;
    readyScope.current = ""; refreshWhileLoading.current = false;
    setPackageValue(undefined); updateAccess(undefined); setLoadedScope("");
    void api.package(selected.pluginId, selected.activeContentSha256).then(async (loaded) => {
      if (!current || scopeRef.current !== requestedScope) return;
      setPackageValue(loaded);
      if (brainId) {
        let saved: PluginInstanceAccess | null;
        do {
          refreshWhileLoading.current = false;
          saved = await api.instance(brainId, selected.pluginId);
          if (!current || scopeRef.current !== requestedScope) return;
        } while (refreshWhileLoading.current);
        if (!current || scopeRef.current !== requestedScope) return;
        updateAccess(saved?.contentSha256 === selected.activeContentSha256 ? saved : defaultAccess(loaded, brainId, selected.activeContentSha256));
      } else updateAccess(undefined);
      readyScope.current = requestedScope;
      setLoadedScope(requestedScope);
    }).catch((failure) => { if (current) setError(failure instanceof Error ? failure.message : String(failure)); });
    return () => { current = false; };
  }, [api, brainId, selected?.pluginId, selected?.activeContentSha256, scope]);

  useEffect(() => {
    if (!api) return;
    const requestedScope = scope;
    let current = true;
    const unsubscribe = api.onEvent(event => {
      if (!current) return;
      if (event.type === "catalog") {
        changedHandler.current?.(event.pluginId);
        void refresh(selectedId).catch(failure => { if (current) setError(String(failure)); });
        return;
      }
      if (!brainId || !selected || event.brainId !== brainId || event.pluginId !== selected.pluginId ||
          event.contentSha256 !== selected.activeContentSha256) return;
      if (readyScope.current !== requestedScope) { refreshWhileLoading.current = true; return; }
      const request = ++accessRead.current;
      if (event.type === "instance") changedHandler.current?.(event.pluginId);
      void api.instance(brainId, selected.pluginId).then(saved => {
        if (!current || scopeRef.current !== requestedScope || request !== accessRead.current ||
            saved?.brainId !== brainId || saved.pluginId !== selected.pluginId ||
            saved.contentSha256 !== selected.activeContentSha256) return;
        const next = refreshedPluginAccess(accessDraft.current, saved, accessDirty.current);
        if (next.conflict) setError("Plugin access changed elsewhere. Review the current settings before saving again.");
        accessDirty.current = next.dirty;
        updateAccess(next.access);
      }).catch(failure => { if (current && scopeRef.current === requestedScope) setError(String(failure)); });
    });
    return () => { current = false; unsubscribe(); };
  }, [api, brainId, selected?.pluginId, selected?.activeContentSha256, scope, selectedId]);

  if (!api) return <section className="plugin-manager"><p>Plugin management is unavailable in this renderer.</p></section>;
  const run = async (operation: () => Promise<void>): Promise<void> => {
    if (running.current) return;
    running.current = true;
    setBusy(true); setError("");
    try { await operation(); }
    catch (failure) { if (mounted.current && brainOwner.current === brainId) setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { running.current = false; if (mounted.current) setBusy(false); }
  };
  const parseEditor = (): unknown => JSON.parse(editor) as unknown;
  const beginEdit = () => {
    if (!packageValue || !selected) return;
    setEditor(JSON.stringify(packageValue, null, 2));
    setEditBase(selected.activeContentSha256);
  };

  return <section className="plugin-manager" aria-labelledby="plugin-manager-heading">
    <header><div><h2 id="plugin-manager-heading">Plugins and custom tools</h2>
      <p>Packages are immutable and declarative. Tool calls still pass both plugin and nested host permissions.</p></div>
      <button type="button" className="button" disabled={busy} onClick={() => void run(async () => {
        const installed = await api.installFile();
        if (installed) { if (installed.installed) onCapabilitiesChanged?.(installed.entry.pluginId); await refresh(installed.entry.pluginId); onNotice?.(`Installed ${installed.entry.name}.`); }
      })}>Import package</button></header>
    <div className="plugin-manager__layout">
      <nav aria-label="Installed plugins">
        {entries.length ? entries.map((entry) => <button type="button" key={entry.pluginId}
          className={entry.pluginId === selectedId ? "is-selected" : ""}
          disabled={busy} onClick={() => setSelectedId(entry.pluginId)}><strong>{entry.name}</strong><span>{entry.activeVersion} · {entry.enabled ? "on" : "off"}</span></button>)
          : <p>No plugin packages are installed.</p>}
      </nav>
      <div className="plugin-manager__detail">
        {selected && packageValue ? <>
          <div className="plugin-manager__title"><div><h3>{selected.name}</h3><code>{selected.activeContentSha256}</code></div>
            <label className="plugin-manager__check"><input type="checkbox" checked={selected.enabled} disabled={busy}
              onChange={(event) => void run(async () => { await api.setEnabled(selected.pluginId, event.currentTarget.checked, selected.activeContentSha256, selected.revision); onCapabilitiesChanged?.(selected.pluginId); await refresh(selected.pluginId); })} />Enable globally</label></div>
          <p>{selected.description}</p>
          <div className="plugin-manager__actions">
            <button type="button" className="button" disabled={busy} onClick={beginEdit}>Edit as new version</button>
            <button type="button" className="button" disabled={busy} onClick={() => void run(async () => {
              const path = await api.exportFile(selected.pluginId, selected.activeContentSha256);
              if (path) onNotice?.(`Exported ${path}`);
            })}>Export</button>
          </div>
          <PluginThemeChooser entry={selected} packageValue={packageValue} onNotice={onNotice} />
          {brainId && packageValue.manifest.runtimeModules?.length ? <PluginRuntimeControls brainId={brainId}
            entry={selected} packageValue={packageValue} pluginsApi={api} onNotice={onNotice} /> : null}
          <details><summary>Versions</summary><div className="plugin-manager__versions">{selected.versions.map((version) =>
            <button type="button" className="button" key={version.contentSha256}
              disabled={busy || version.contentSha256 === selected.activeContentSha256}
              onClick={() => void run(async () => { await api.activate(selected.pluginId, version.contentSha256); onCapabilitiesChanged?.(selected.pluginId); await refresh(selected.pluginId); })}>
              {version.version}{version.contentSha256 === selected.activeContentSha256 ? " · active" : " · activate"}
            </button>)}</div></details>
          {brainId && access ? <fieldset><legend>Access for this instance</legend>
            {access.semanticLearning ? <div role="status" className="plugin-manager__learning">
              <strong>Capability learning · {access.semanticLearning.state}</strong>
              <p>{access.semanticLearning.state === "complete" ? "The declared documentation and schemas have a committed learning receipt." : access.semanticLearning.state === "running" ? "Learning through the existing loaded brain." : "Queued for an already-loaded brain and a safe learning boundary."}</p>
              {access.semanticLearning.error ? <details><summary>Learning needs attention</summary><p>{access.semanticLearning.error}</p></details> : null}
            </div> : null}
            <label className="plugin-manager__check"><input type="checkbox" checked={access.enabled}
              onChange={(event) => { accessDirty.current = true; updateAccess({ ...access, enabled: event.currentTarget.checked }); }} />Enable this exact version</label>
            <p>Nested capabilities never inherit more access than their ordinary Tools setting.</p>
            {access.capabilities.map((capability, index) => <label className="plugin-manager__check" key={capability.toolId}>
              <input type="checkbox" checked={capability.enabled}
                onChange={(event) => { accessDirty.current = true; updateAccess({ ...access, capabilities: access.capabilities.map((item, itemIndex) =>
                  itemIndex === index ? { ...item, enabled: event.currentTarget.checked } : item) }); }} />
              {capability.toolId} · {capability.actions.join(", ")}
            </label>)}
            {access.toolPermissions.map((permission, index) => <label key={permission.toolId}>{permission.toolId}
              <select value={permission.level} onChange={(event) => { accessDirty.current = true; updateAccess({ ...access,
                toolPermissions: access.toolPermissions.map((item, itemIndex) => itemIndex === index
                  ? { ...item, level: event.currentTarget.value as ToolPermissionLevel } : item) }); }}>
                <option value="off">Off</option><option value="ask">Ask every run</option>
                <option value="auto">Auto within nested policy</option><option value="full">Full within nested policy</option>
              </select></label>)}
            <button type="button" className="button button--primary" disabled={busy} onClick={() => void run(async () => {
              const requestedScope = scope;
              const saved = await api.setInstanceAccess({ ...access, requestId: globalThis.crypto.randomUUID() });
              onCapabilitiesChanged?.(saved.pluginId);
              if (mounted.current && scopeRef.current === requestedScope) { accessDirty.current = false; updateAccess(saved); onNotice?.("Plugin access saved."); }
            })}>Save instance access</button>
          </fieldset> : null}
          {packageValue.manifest.panels.map((panel) => <details key={panel.id}><summary>{panel.title}</summary>
            <PluginStaticPanel title={panel.title} pluginId={packageValue.manifest.id}
              contentSha256={selected.activeContentSha256} panelId={panel.id} /></details>)}
          {brainId ? packageValue.manifest.interactivePanels?.map((panel) => <details key={panel.id}>
            <summary>{panel.title} · interactive</summary>
            <PluginInteractivePanel brainId={brainId} pluginId={packageValue.manifest.id}
              contentSha256={selected.activeContentSha256} panelId={panel.id} title={panel.title} api={api} />
          </details>) : null}
          {brainId ? <PluginPanelHistory key={brainId} brainId={brainId} api={api} /> : null}
          {packageValue.manifest.trainingFormats.length ? <details><summary>Training-format declarations</summary>
            <p>Declaration-only formats stay inert. Built-in streams use Studio parsers. Immutable WASM and dedicated Runtime JS streams require this exact package and parser capability; Runtime JS host actions also use your tool permissions.</p>
            {packageValue.manifest.trainingFormats.map((format) => <p key={format.id}><strong>{format.title}</strong> · {format.extensions.join(", ") || format.mediaTypes.join(", ")}</p>)}</details> : null}
        </> : <p>Select or import a plugin package.</p>}
      </div>
    </div>
    <details className="plugin-manager__maker" open={!entries.length || Boolean(editBase)}><summary>{editBase ? "Edit package as a new immutable version" : "Plugin maker"}</summary>
      <PluginModuleAuthoring value={editor} onChange={setEditor} disabled={busy} />
      <div className="plugin-manager__actions"><button type="button" className="button button--primary" disabled={busy}
        onClick={() => void run(async () => {
          const result = editBase && selectedId
            ? await api.edit(selectedId, editBase, parseEditor())
            : await api.installPackage(parseEditor(), "Plugin maker");
          if (result.installed) onCapabilitiesChanged?.(result.entry.pluginId);
          setEditBase(""); await refresh(result.entry.pluginId); onNotice?.(`Saved ${result.entry.name} ${result.package.manifest.version}.`);
        })}>{editBase ? "Save new version" : "Add package"}</button>
        {editBase ? <button type="button" className="button" onClick={() => setEditBase("")}>Cancel edit</button> : null}</div>
    </details>
    {error ? <p className="plugin-manager__error" role="alert">{error}</p> : null}
  </section>;
}
