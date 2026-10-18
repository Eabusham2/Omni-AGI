import { useState } from "react";
import { MCP_PRESETS, mcpPresetDraft, type McpPreset, type McpPresetDraft } from "../../shared/mcpPresets";
import "./maps.css";

/** Selection prepares a form only. It cannot install, authenticate or connect. */
export function McpPresetPicker({ onChoose, disabled = false }: { onChoose(draft: McpPresetDraft, preset: McpPreset): void; disabled?: boolean }) {
  const [selected, setSelected] = useState("");
  const preset = MCP_PRESETS.find(value => value.id === selected);
  return <details className="integration-preset-picker"><summary>Reviewed connection templates</summary>
    <label>Service<select value={selected} disabled={disabled} onChange={event => setSelected(event.currentTarget.value)}>
      <option value="">Choose a service…</option>{MCP_PRESETS.map(value => <option key={value.id} value={value.id}>{value.label}</option>)}</select></label>
    {preset ? <div className="integration-preset-picker__review">
      <p>{preset.scope}</p><p>{preset.authentication}</p>
      {preset.url ? <p className="integration-preset-picker__endpoint">{preset.url}</p> : <p>Installed package: <code>{preset.packageName}</code>. Select its executable path separately.</p>}
      <ul>{preset.prerequisites.map(value => <li key={value}>{value}</li>)}</ul>
      <details><summary>Access and setup notes</summary><ul>{preset.cautions.map(value => <li key={value}>{value}</li>)}</ul>
        <p>{preset.sources.map((value, index) => <span key={value.url}>{index ? " · " : ""}<a href={value.url} target="_blank" rel="noopener noreferrer">{value.title}</a></span>)}</p></details>
      <p>Tools are discovered only when you explicitly connect. This prepares the form without signing in, installing, granting access or calling a tool.</p>
      <button type="button" className="button button--primary" disabled={disabled} onClick={() => onChoose(mcpPresetDraft(preset.id), preset)}>Use reviewed draft</button>
    </div> : null}
  </details>;
}
