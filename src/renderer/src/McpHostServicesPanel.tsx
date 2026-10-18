import { useEffect, useRef, useState } from "react";
import type { McpDiagnosticPage, McpDiagnosticsApi, McpRootsApi, McpRootsStatus } from "../../shared/mcpHostServices";
import "./mcpAuthentication.css";

export function McpHostServicesPanel(props: { brainId: string; serverId: string; connected: boolean; enabled: boolean;
  roots?: McpRootsApi; diagnostics?: McpDiagnosticsApi }) {
  return <McpHostServicesOwner key={`${props.brainId}:${props.serverId}`} {...props} />;
}
function McpHostServicesOwner({ brainId, serverId, connected, enabled, roots, diagnostics }: Parameters<typeof McpHostServicesPanel>[0]) {
  const [status, setStatus] = useState<McpRootsStatus>(), [logs, setLogs] = useState<McpDiagnosticPage>();
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const owner = useRef<object | null>(null), pending = useRef<object | null>(null), revision = useRef(0);
  useEffect(() => { const scope = {}; owner.current = scope; pending.current = null; revision.current++; setBusy(false); setStatus(undefined); setLogs(undefined); setError("");
    return () => { owner.current = null; pending.current = null; revision.current++; }; }, [brainId, serverId, roots, diagnostics]);
  useEffect(() => { revision.current++; setStatus(undefined); }, [connected, enabled]);
  const run = async (kind: "status" | "select" | "review" | "clear" | "logs", after?: number) => {
    if (pending.current || kind === "logs" && !diagnostics || kind !== "logs" && !roots) return;
    const scope = owner.current, generation = ++revision.current, flight = {}; pending.current = flight; setBusy(true); setError("");
    try {
      if (kind === "logs") {
        const result = await diagnostics!.page(brainId, serverId, after, 50);
        if (owner.current === scope && revision.current === generation) setLogs(result);
      } else {
        const result = kind === "status" ? await roots!.status(brainId, serverId)
          : await roots![kind]({ brainId, serverId, requestId: crypto.randomUUID() });
        if (owner.current !== scope || revision.current !== generation) return;
        if (result.brainId !== brainId || result.serverId !== serverId) throw new Error("Filesystem disclosure returned another source connection.");
        setStatus(result);
      }
    } catch (reason) { if (owner.current === scope && revision.current === generation) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (pending.current === flight) { pending.current = null; if (owner.current === scope) setBusy(false); } }
  };
  return <details className="mcp-authentication" onToggle={event => { if (event.currentTarget.open && !status && !busy) void run("status"); }}>
    <summary>Filesystem disclosures and diagnostics</summary>
    <p className="mcp-authentication__message">Choose which folder names and URIs this connection may see. This does not grant file access or sandbox a local server.</p>
    <div className="mcp-authentication__actions">
      <button type="button" className="button button--secondary" disabled={busy || !roots || !enabled} onClick={() => void run("select")}>Choose folders…</button>
      <button type="button" className="button button--secondary" disabled={busy || !roots || !enabled || !connected || !status?.roots.length} onClick={() => void run("review")}>Review disclosure</button>
      <button type="button" className="button button--secondary" disabled={busy || !roots || !status?.roots.length} onClick={() => void run("clear")}>Clear selection</button>
      <button type="button" className="button button--secondary" disabled={busy || !roots} onClick={() => void run("status")}>Refresh status</button>
    </div>
    {status ? <><p className="mcp-authentication__message" role="status">{status.state === "shared" ? "Shared with the current connection" : status.state === "review-required" ? "Selected; fresh disclosure review required" : "No filesystem roots shared"}
      {status.notification === "failed" ? " · change notification failed" : status.notification === "not-ready" ? " · source not ready for notification" : ""}</p>
      {status.roots.length ? <ul>{status.roots.map(root => <li key={root.uri} style={{ overflowWrap: "anywhere" }}>{root.name ?? root.uri}<small style={{ display: "block" }}>{root.uri}</small>
        {!root.accessible ? <small>{root.reason ?? "Currently inaccessible"}</small> : null}</li>)}</ul> : null}</> : null}
    <div className="mcp-authentication__actions"><button type="button" className="button button--secondary" disabled={busy || !diagnostics || !enabled} onClick={() => void run("logs")}>Inspect current-session logs</button></div>
    {logs ? <section aria-label="MCP operational diagnostics"><p className="mcp-authentication__message">{logs.accepted.toLocaleString()} accepted · {logs.rejected.toLocaleString()} rejected · {logs.rateLimited.toLocaleString()} rate-limited · {logs.evicted.toLocaleString()} expired from the bounded current-process log.</p>
      {logs.records.length ? logs.records.map(record => <details key={record.sequence}><summary>{record.level} · {record.logger ?? record.source} · {new Date(record.observedAt).toLocaleTimeString()}</summary>
        <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{JSON.stringify(record.data, null, 2)}</pre></details>) : <p>No retained diagnostics on this page.</p>}
      {logs.hasMore && logs.nextSequence !== undefined ? <button type="button" className="button button--secondary" disabled={busy} onClick={() => void run("logs", logs.nextSequence)}>Next log page</button> : null}
    </section> : null}
    {error ? <p role="alert" className="mcp-authentication__message">{error}</p> : null}
  </details>;
}
