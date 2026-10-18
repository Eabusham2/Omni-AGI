import React, { useEffect, useRef, useState } from "react";
import type { McpServerSummary } from "../../shared/types";
import type { McpOAuthStatus } from "../../shared/mcpAuth";
import "./mcpAuthentication.css";

const LABELS: Record<McpOAuthStatus["state"], string> = {
  none: "Not signed in", required: "Sign-in needed", authorizing: "Waiting for browser sign-in",
  authorized: "Signed in", error: "Sign-in needs attention"
};

export function McpAuthenticationPanel({ brainId, server, onChanged }: {
  brainId: string;
  server: McpServerSummary;
  onChanged?: () => void;
}) {
  const [status, setStatus] = useState<McpOAuthStatus>(server.authentication ?? { state: "none" });
  const [clientId, setClientId] = useState("");
  const [secret, setSecret] = useState("");
  const [metadataUrl, setMetadataUrl] = useState("");
  const [port, setPort] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const generation = useRef(0);
  const submitting = useRef(false);
  const notify = useRef(onChanged);
  notify.current = onChanged;
  const validPort = !port || (Number.isInteger(Number(port)) && Number(port) >= 1024 && Number(port) <= 65535);
  const available = typeof window !== "undefined" && Boolean(window.omni?.mcp?.authorize);

  useEffect(() => {
    let alive = true;
    const revision = generation.current;
    if (window.omni?.mcp?.authStatus) void window.omni.mcp.authStatus(brainId, server.id)
      .then(value => { if (alive && revision === generation.current) setStatus(value); })
      .catch(error => { if (alive && revision === generation.current) setMessage(error instanceof Error ? error.message : "Sign-in status is unavailable."); });
    return () => { alive = false; generation.current++; };
  }, [brainId, server.id]);

  useEffect(() => {
    const api = window.omni?.mcp;
    if (status.state !== "authorizing" || !api?.authStatus) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async (): Promise<void> => {
      const revision = generation.current;
      try {
        const next = await api.authStatus(brainId, server.id);
        if (disposed || revision !== generation.current) return;
        setStatus(next);
        if (next.state === "authorized") {
          setMessage("Signed in. Refresh the server to discover its available tools."); notify.current?.();
        }
        if (next.state === "authorizing") timer = setTimeout(() => void poll(), 1000);
      } catch (error) {
        if (!disposed && revision === generation.current) {
          setMessage(error instanceof Error ? error.message : "Sign-in status is unavailable.");
          timer = setTimeout(() => void poll(), 3000);
        }
      }
    };
    timer = setTimeout(() => void poll(), 1000);
    return () => { disposed = true; clearTimeout(timer); };
  }, [brainId, server.id, status.state]);

  const run = async (operation: "start" | "stop" | "disconnect"): Promise<void> => {
    if (submitting.current || (operation === "start" && !validPort)) return;
    const api = window.omni?.mcp;
    if (!api) { setMessage("Account sign-in is available in the desktop app."); return; }
    submitting.current = true; generation.current++; setBusy(true); setMessage("");
    const revision = generation.current;
    try {
      if (operation === "start") {
        const next = await api.authorize(brainId, server.id, {
          ...(clientId.trim() ? { clientId: clientId.trim() } : {}), ...(secret ? { clientSecret: secret } : {}),
          ...(metadataUrl.trim() ? { clientMetadataUrl: metadataUrl.trim() } : {}), ...(port ? { redirectPort: Number(port) } : {})
        });
        if (revision === generation.current) setStatus(next);
      } else {
        if (operation === "stop") await api.cancelAuthorization(brainId, server.id);
        else await api.disconnectAuthorization(brainId, server.id);
        const next = await api.authStatus(brainId, server.id);
        if (revision === generation.current) { setStatus(next); notify.current?.(); }
      }
    } catch (error) {
      if (revision === generation.current) setMessage(error instanceof Error ? error.message : "Sign-in could not finish.");
    } finally {
      if (revision === generation.current) { setSecret(""); setBusy(false); }
      submitting.current = false;
    }
  };
  if (server.transport !== "http") return null;
  return <section className="mcp-authentication" aria-label={`${server.label} account sign-in`}>
    <div className="mcp-authentication__status" role="status" aria-live="polite">
      <span>{LABELS[status.state]}{status.persistence === "session" ? " · this session only" : ""}</span>
      {status.issuer ? <small>{status.issuer}</small> : null}
    </div>
    <details>
      <summary>Client configuration · optional</summary>
      <div className="mcp-authentication__fields">
        <label>Registered client ID<input value={clientId} autoComplete="off" onChange={event => setClientId(event.target.value)} disabled={busy || status.state === "authorizing"} /></label>
        <label>Client secret<input type="password" value={secret} autoComplete="off" onChange={event => setSecret(event.target.value)} disabled={busy || status.state === "authorizing"} /></label>
        <label>Hosted client metadata<input type="url" value={metadataUrl} placeholder="https://your-app.example/client.json" onChange={event => setMetadataUrl(event.target.value)} disabled={busy || status.state === "authorizing"} /></label>
        <label>Registered loopback port<input type="number" inputMode="numeric" min={1024} max={65535} value={port} placeholder="Automatic" aria-invalid={!validPort} onChange={event => setPort(event.target.value)} disabled={busy || status.state === "authorizing"} /></label>
      </div>
      {!validPort ? <p className="mcp-authentication__message" role="alert">Use a port from 1024 to 65535, or leave it empty for an automatic port.</p> : null}
    </details>
    {status.scopes?.length ? <p className="mcp-authentication__scopes">Requested access: {status.scopes.join(", ")}</p> : null}
    <div className="mcp-authentication__actions">
      {status.state === "authorizing" ? <button className="button button--secondary" type="button" disabled={busy || !available} onClick={() => void run("stop")}>Stop sign-in</button>
        : <button className="button button--secondary" type="button" disabled={busy || !validPort || !server.effectiveEnabled || !available} onClick={() => void run("start")}>{busy ? "Opening…" : status.state === "authorized" ? "Sign in again" : "Sign in"}</button>}
      {status.state === "authorized" ? <button className="button button--secondary" type="button" disabled={busy || !available} onClick={() => void run("disconnect")}>Disconnect account</button> : null}
    </div>
    {message || status.error ? <p className="mcp-authentication__message" role="status" aria-live="polite">{message || status.error}</p> : null}
  </section>;
}
