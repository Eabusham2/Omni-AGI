import { useMemo, useState } from "react";
import type { OmniPluginManifest, OmniPluginPackage } from "../../shared/plugins";
import type { PluginRuntimeModuleDefinition } from "../../shared/pluginRuntime";
import {
  PLUGIN_MODULE_STARTERS, addPluginModuleStarter, pluginStyleModuleCss, readPluginAuthoringPackage,
  removePluginRuntimeModule, replacePluginRuntimeModule, replacePluginStyleModuleCss
} from "./pluginModuleDraft";
import "./pluginModuleAuthoring.css";

const splitNames = (value: string) => [...new Set(value.split(",").map(name => name.trim()).filter(Boolean))];
const CONTRIBUTION_KINDS = ["style", "menu", "button", "panel"] as const;

/** Edits package text only. Saving/review/activation stay with the existing host workflow. */
export function PluginModuleAuthoring({ value, onChange, disabled = false }: {
  value: string; onChange(value: string): void; disabled?: boolean;
}) {
  const [selectedId, setSelectedId] = useState("");
  const parsed = useMemo(() => {
    try { return { packageValue: readPluginAuthoringPackage(value), error: "" }; }
    catch (failure) { return { packageValue: undefined, error: failure instanceof Error ? failure.message : "Check the package JSON." }; }
  }, [value]);
  const packageValue = parsed.packageValue, modules = packageValue?.manifest.runtimeModules ?? [];
  const selected = modules.find(module => module.id === selectedId) ?? modules[0];
  const css = selected ? pluginStyleModuleCss(selected.script) : undefined;
  const publish = (next: OmniPluginPackage) => onChange(JSON.stringify(next, null, 2));
  const metadata = (key: "name" | "id" | "version" | "description" | "documentation", next: string) => {
    if (packageValue) publish({ ...packageValue, manifest: { ...packageValue.manifest, [key]: next } });
  };
  const edit = (next: Partial<PluginRuntimeModuleDefinition>) => {
    if (packageValue && selected) publish(replacePluginRuntimeModule(packageValue, selected.id, { ...selected, ...next }));
  };

  return <div className="plugin-module-authoring">
    {packageValue ? <>
      <div className="plugin-module-authoring__identity">
        <label>Package name<input value={packageValue.manifest.name} disabled={disabled} onChange={event => metadata("name", event.currentTarget.value)} /></label>
        <label>Package ID<input value={packageValue.manifest.id} disabled={disabled} onChange={event => metadata("id", event.currentTarget.value)} spellCheck={false} /></label>
        <label>Version<input value={packageValue.manifest.version} disabled={disabled} onChange={event => metadata("version", event.currentTarget.value)} spellCheck={false} /></label>
      </div>
      <div className="plugin-module-authoring__starters" role="group" aria-label="Add feature module">
        {PLUGIN_MODULE_STARTERS.map(starter => <button type="button" className="button" key={starter.id} title={starter.description} disabled={disabled}
          onClick={() => { const next = addPluginModuleStarter(packageValue, starter.id); publish(next.packageValue); setSelectedId(next.moduleId); }}>+ {starter.title}</button>)}
      </div>
      {modules.length > 1 ? <div className="plugin-module-authoring__tabs" role="group" aria-label="Edit module">
        {modules.map(module => <button type="button" className="button" aria-pressed={module.id === selected?.id} key={module.id}
          onClick={() => setSelectedId(module.id)}>{module.title}</button>)}
      </div> : null}
      {selected ? <section className="plugin-module-authoring__module" aria-label="Feature module editor">
        <div className="plugin-module-authoring__heading"><label>Module name<input value={selected.title} disabled={disabled}
          onChange={event => edit({ title: event.currentTarget.value })} /></label>
          <button type="button" className="button" disabled={disabled} onClick={() => publish(removePluginRuntimeModule(packageValue, selected.id))}>Remove from draft</button></div>
        {css !== undefined ? <label>App CSS<textarea aria-label="Feature stylesheet" spellCheck={false} value={css} disabled={disabled}
          onChange={event => edit({ script: replacePluginStyleModuleCss(selected.script, event.currentTarget.value) })} /></label> : null}
        {css === undefined ? <label>JavaScript<textarea aria-label="Feature JavaScript" spellCheck={false} value={selected.script} disabled={disabled}
          onChange={event => edit({ script: event.currentTarget.value })} /></label>
          : <details><summary>JavaScript</summary><label className="plugin-module-authoring__code">Module source<textarea aria-label="Feature JavaScript" spellCheck={false} value={selected.script} disabled={disabled}
            onChange={event => edit({ script: event.currentTarget.value })} /></label></details>}
        <details><summary>Events, placements & tools</summary>
          <label>Description<input value={selected.description} disabled={disabled} onChange={event => edit({ description: event.currentTarget.value })} /></label>
          <label>Host events<input value={selected.events.join(", ")} placeholder="brain.lifecycle, chat.action" disabled={disabled}
            onChange={event => edit({ events: splitNames(event.currentTarget.value) })} /></label>
          <label>UI placements<input value={selected.ui.slots.join(", ")} placeholder="chat.toolbar, chat.before, chat.after, tools" disabled={disabled}
            onChange={event => edit({ ui: { ...selected.ui, slots: splitNames(event.currentTarget.value) } })} /></label>
          <div className="plugin-module-authoring__kinds" role="group" aria-label="Allowed UI additions">{CONTRIBUTION_KINDS.map(kind =>
            <label key={kind}><input type="checkbox" checked={selected.ui.kinds.includes(kind)} disabled={disabled} onChange={event =>
              edit({ ui: { ...selected.ui, kinds: event.currentTarget.checked ? [...selected.ui.kinds, kind] : selected.ui.kinds.filter(item => item !== kind) } })} />{kind}</label>)}</div>
          <label className="plugin-module-authoring__check"><input type="checkbox" checked={selected.dynamicTools} disabled={disabled}
            onChange={event => edit({ dynamicTools: event.currentTarget.checked })} />Register custom tools</label>
        </details>
      </section> : <p>Add a feature, stylesheet, tool or training format above.</p>}
      <PluginAuthoringFormats formats={packageValue.manifest.trainingFormats} />
      <details><summary>Package description & documentation</summary>
        <label>Description<input value={packageValue.manifest.description} disabled={disabled} onChange={event => metadata("description", event.currentTarget.value)} /></label>
        <label>Documentation<textarea value={packageValue.manifest.documentation} disabled={disabled} onChange={event => metadata("documentation", event.currentTarget.value)} /></label>
      </details>
    </> : <p role="alert">{parsed.error}</p>}
    <details open={packageValue ? undefined : true}><summary>Complete package JSON</summary>
      <textarea aria-label="Plugin package JSON" value={value} onChange={event => onChange(event.currentTarget.value)} spellCheck={false} disabled={disabled} />
    </details>
    <details><summary>Module API & permission review</summary>
      <p>Saving creates a package; code runs only after you explicitly activate its reviewed version. Imported code and tools do not grant themselves access.</p>
      <dl><dt>Lifecycle & events</dt><dd><code>globalThis.omniRuntime.onActivate(fn)</code>, <code>onDeactivate(fn)</code> and <code>on(topic, fn)</code>. Host topics are <code>brain.lifecycle</code> and <code>chat.action</code>; UI callbacks use <code>ui:event</code>.</dd>
        <dt>UI & tools</dt><dd><code>ui.put</code>, <code>ui.remove</code>, <code>registerTool</code>. Styles affect the app; panels are isolated HTML. Supported placements: <code>chat.toolbar</code>, <code>chat.before</code>, <code>chat.after</code>, <code>tools</code>.</dd>
        <dt>Host actions</dt><dd><code>call(toolId, action, arguments)</code> requires a declared, enabled package capability and the tool's current permission. OS restrictions still apply.</dd>
        <dt>State & training</dt><dd><code>state.get/read/put</code>, <code>receipts.read</code> and <code>registerParser</code>. Parser declarations and capabilities are preserved in the complete package JSON.</dd></dl>
    </details>
  </div>;
}

function PluginAuthoringFormats({ formats }: { formats: OmniPluginManifest["trainingFormats"] }) {
  if (!formats.length) return null;
  return <details><summary>Training formats · {formats.length}</summary><ul>{formats.map(format => <li key={format.id}>
    {format.title} · {format.extensions.map(extension => `.${extension}`).join(", ") || format.mediaTypes.join(", ")}
  </li>)}</ul><p>Edit extensions, native-media output or parser bindings in Complete package JSON. Learning uses the exact reviewed parser and its existing approval flow.</p></details>;
}
