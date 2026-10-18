import React, { useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ToastNotice } from "../../src/renderer/src/ToastNotice";
import { conciseUiMessage } from "../../src/renderer/src/uiPresentation";
import "../../src/renderer/src/styles.css";
const text = "The selected task could not continue because the temporary resource reservation changed. Your committed work is kept. Review the current allocation before trying again. This is a deliberately long synthetic notification with an unbroken reference abcdef0123456789abcdef0123456789abcdef0123456789; no real task runs.";
function Fixture() {
  const [notice, setNotice] = useState<{ id: number; text: string } | null>(null), [dark, setDark] = useState(false), next = useRef(0);
  return <main style={{ maxWidth: 860, boxSizing: "border-box", padding: 20, margin: "0 auto" }}><h1>Notification reading check</h1><p>Actual notification component. No model, task, process, key, permission or provider runs.</p>
    <button className="button" onClick={() => { setDark(value => !value); document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; }}>Switch theme</button>
    <button className="button" onClick={() => setNotice({ id: ++next.current, text: conciseUiMessage(text) })}>Show long synthetic notice</button>
    <p role="status">Notice owner: {notice?.id ?? "none"}</p>
    {notice ? <ToastNotice key={notice.id} text={notice.text} onDismiss={() => setNotice(current => current?.id === notice.id ? null : current)} /> : null}
  </main>;
}
document.documentElement.dataset.colorScheme = "light"; document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
