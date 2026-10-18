import React, { useRef, useState } from "react";
import type { McpServerSummary } from "../../shared/types";
import "./mcpAuthentication.css";

export function McpCredentialPanel({ brainId, server, onSaved }: {
  brainId: string;
  server: McpServerSummary;
  onSaved: (summary: McpServerSummary) => void;
}) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const [message, setMessage] = useState("");
  const submitting = useRef(false);
  const local = server.transport === "stdio";
  const available = typeof window !== "undefined" && Boolean(window.omni?.mcp?.updateCredentials);
  const save = async (clear: boolean): Promise<void> => {
    const api = window.omni?.mcp;
    if (!api?.updateCredentials || submitting.current) return;
    submitting.current = true; setBusy(true); setMessage("");
    try {
      let environment: Record<string, string> = {};
      if (local && !clear) {
        try {
          const parsed = JSON.parse(value) as unknown;
          if (!parsed || typeof parsed !== "object" || Array.isArray(parsed) || Object.values(parsed).some(item => typeof item !== "string")) throw new Error();
          environment = parsed as Record<string, string>;
        } catch { throw new Error("Enter a JSON object whose variable names have text values."); }
      }
      const updated = await api.updateCredentials(brainId, server.id, local ? { environment } : { bearerToken: clear ? "" : value });
      setValue(""); setConfirmClear(false); setMessage(clear ? "Credentials cleared." : "Credentials saved. Refresh to reconnect.");
      onSaved(updated);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Credentials could not be saved."); }
    finally { submitting.current = false; setBusy(false); }
  };
  return <details className="mcp-authentication mcp-credential-panel">
    <summary>Replace credentials</summary>
    <form onSubmit={event => { event.preventDefault(); void save(false); }}>
      <div className="mcp-authentication__fields">
        {local ? <label>Replacement environment · JSON<textarea rows={3} autoComplete="off" spellCheck={false} value={value}
          onChange={event => setValue(event.target.value)} disabled={busy} placeholder={'{"SERVICE_API_KEY":"…"}'} /></label>
          : <label>Replacement bearer token<input type="password" autoComplete="off" value={value} onChange={event => setValue(event.target.value)} disabled={busy} /></label>}
      </div>
      {local && server.environmentKeys?.length ? <p className="mcp-authentication__message">Configured variables: {server.environmentKeys.join(", ")}</p> : null}
      <p className="mcp-authentication__message">{local ? "Replaces all variables. Stored values are not shown." : "Replaces the current account login."} The connection stops; completed actions are not repeated.</p>
      <div className="mcp-authentication__actions">
        <button type="submit" className="button button--secondary" disabled={busy || !available || !value.trim()}>{busy ? "Saving…" : "Save credentials"}</button>
        <button type="button" className="button button--secondary" disabled={busy || !available} onClick={() => setConfirmClear(true)}>Clear credentials</button>
      </div>
      {confirmClear ? <div className="mcp-authentication__actions" role="group" aria-label="Confirm credential removal">
        <span>Clear this server’s stored credentials?</span>
        <button type="button" className="button button--secondary" disabled={busy} onClick={() => void save(true)}>Clear now</button>
        <button type="button" className="button button--secondary" disabled={busy} onClick={() => setConfirmClear(false)}>Keep credentials</button>
      </div> : null}
      {message ? <p className="mcp-authentication__message" role="status" aria-live="polite">{message}</p> : null}
    </form>
  </details>;
}
