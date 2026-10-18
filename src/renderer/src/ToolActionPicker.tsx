import type { ToolDescription } from "../../shared/toolDescriptions";
import { useEffect, useState } from "react";

export function defaultToolAction(description?: ToolDescription): string {
  if (!description) return "";
  for (const candidate of ["capabilities", "inspect", "connections", "list", "status"]) {
    const schema = description.actionInputSchemas?.[candidate] ?? description.inputSchema;
    if (description.actions.includes(candidate) && (!Array.isArray(schema?.required) || schema.required.length === 0)) return candidate;
  }
  return description.actions[0] ?? "";
}
export function ToolActionPicker({ description, action, disabled, onChange }: {
  description?: ToolDescription; action: string; disabled: boolean; onChange(action: string): void;
}) {
  const [fieldsOpen, setFieldsOpen] = useState(false);
  useEffect(() => { setFieldsOpen(false); }, [action, description?.id]);
  const schema = description?.actionInputSchemas?.[action] ?? description?.inputSchema;
  return <>
    <label><span>Action</span>{description?.actions.length ? <select aria-label="Tool action" disabled={disabled} value={action} onChange={event => onChange(event.target.value)}>
      {!description.actions.includes(action) ? <option value={action}>{action || "Choose an action"}</option> : null}
      {description.actions.map(value => <option key={value} value={value}>{value}</option>)}
    </select> : <input aria-label="Tool action" value={action} disabled={disabled} onChange={event => onChange(event.target.value)} />}</label>
    {schema ? <details open={fieldsOpen} onToggle={event => setFieldsOpen(event.currentTarget.open)}><summary>Declared action fields</summary>{fieldsOpen ? <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", maxHeight: 220, overflow: "auto" }}>{JSON.stringify(schema, null, 2)}</pre> : null}</details> : null}
    {description?.source === "unregistered" ? <p role="status">This saved tool is not currently registered.</p> : null}
  </>;
}
