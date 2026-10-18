import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { ToolActionPicker, defaultToolAction } from "../../src/renderer/src/ToolActionPicker";
import type { ToolDescription } from "../../src/shared/toolDescriptions";
import "../../src/renderer/src/styles.css";
const description: ToolDescription = { id: "runtime.vm", label: "Virtual machine", actions: ["create", "capabilities", "status"], source: "builtin", actionInputSchemas: {
  capabilities: { type: "object", properties: {}, required: [], additionalProperties: false },
  status: { type: "object", properties: { environmentId: { type: "string" } }, required: ["environmentId"], additionalProperties: false },
  create: { type: "object", properties: { name: { type: "string" }, imageAttachmentId: { type: "string" } }, required: ["name", "imageAttachmentId"], additionalProperties: false }
} };
function Fixture() {
  const [action, setAction] = useState(defaultToolAction(description)), [text, setText] = useState("{}"), [calls, setCalls] = useState(0), [dark, setDark] = useState(false);
  return <main style={{ maxWidth: 900, height: "100vh", overflow: "auto", padding: 18, margin: "0 auto", boxSizing: "border-box" }}>
    <h1>Existing tool console</h1><p>Actual action picker, synthetic registration and button receipt. No tool, VM, permission change, model or external action runs.</p>
    <button className="button" onClick={() => { setDark(value => !value); document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; }}>Switch theme</button>
    <aside className="surface tool-console" style={{ width: 320, maxWidth: "100%", boxSizing: "border-box", marginTop: 16 }}>
      <div className="tool-console__head"><div><small>TEST PROTOCOL</small><strong>Virtual machine</strong></div></div>
      <ToolActionPicker description={description} action={action} disabled={false} onChange={setAction} />
      <label><span>Arguments · JSON</span><textarea rows={4} value={text} onChange={event => setText(event.target.value)} /></label>
      <button type="button" className="button button--primary" disabled={!action.trim()} onClick={() => setCalls(value => value + 1)}>Run synthetic test</button>
      <p role="status">Synthetic clicks: {calls}. Selecting or inspecting an action never invokes it.</p>
    </aside>
  </main>;
}
document.documentElement.dataset.colorScheme = "light"; document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
