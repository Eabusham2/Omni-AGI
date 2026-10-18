import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { McpAuthenticationPanel } from "../../src/renderer/src/McpAuthenticationPanel";
import { McpCredentialPanel } from "../../src/renderer/src/McpCredentialPanel";
import type { McpOAuthStatus } from "../../src/shared/mcpAuth";
import type { McpServerSummary } from "../../src/shared/types";
import "../../src/renderer/src/styles.css";

let status: McpOAuthStatus = { state: "required", scopes: ["project:read", "mail:read"] };
const credentialReceipts: unknown[] = [];
let publishReceipts = () => {};
Object.assign(window, { omni: { mcp: {
  authStatus: async () => structuredClone(status),
  authorize: async () => { status = { state: "authorizing", issuer: "https://accounts.example/provider", scopes: ["project:read", "mail:read"] }; return structuredClone(status); },
  cancelAuthorization: async () => { status = { state: "required", error: "Sign-in stopped." }; return true; },
  disconnectAuthorization: async () => { status = { state: "none" }; },
  updateCredentials: async (_brainId: string, id: string, input: { environment?: Record<string, string>; bearerToken?: string }) => {
    credentialReceipts.push({ id, operation: input.environment ? "replace-environment" : "replace-bearer", keys: Object.keys(input.environment ?? {}), valuePresent: Boolean(input.bearerToken), remoteEffects: false });
    publishReceipts(); status = { state: "none" };
    return { ...server, id, transport: input.environment ? "stdio" : "http", connected: false,
      credentialConfigured: Boolean(input.bearerToken || Object.keys(input.environment ?? {}).length),
      environmentKeys: Object.keys(input.environment ?? {}), authentication: status };
  }
} } });
const server: McpServerSummary = { brainId: "fixture", id: "project", registryId: "project", label: "Project tools", transport: "http", endpoint: "https://mcp.example/mcp",
  connected: false, tools: [], credentialConfigured: false, scope: "all", globallyEnabled: true, defaultEnabled: true, accessOverride: "inherit", effectiveEnabled: true };

function Fixture() {
  const [revision, setRevision] = useState(0);
  const [theme, setTheme] = useState("dark");
  const [width, setWidth] = useState("100%");
  const [remote, setRemote] = useState(server);
  const [local, setLocal] = useState<McpServerSummary>({ ...server, id: "local", label: "Local fixture tools", transport: "stdio", endpoint: "synthetic-installed-command", environmentKeys: ["SERVICE_API_KEY"] });
  const [, refreshReceipts] = useState(0); publishReceipts = () => refreshReceipts(value => value + 1);
  const changeTheme = (value: string): void => { document.documentElement.dataset.colorScheme = value; setTheme(value); };
  return <main style={{ padding: 24, maxWidth: 920, margin: "0 auto", height: "100vh", overflowY: "auto", boxSizing: "border-box" }}>
    <h1>MCP account controls</h1>
    <p>Synthetic UI fixture. No account sign-in, neural model, or remote request occurs here.</p>
    <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 24 }}>
      <label>Appearance<select value={theme} onChange={event => changeTheme(event.target.value)}><option value="dark">Dark</option><option value="light">Light</option></select></label>
      <label>Surface width<select value={width} onChange={event => setWidth(event.target.value)}><option value="100%">Full</option><option value="320px">Narrow 320</option></select></label>
      <button className="button secondary" type="button" onClick={() => { status = { state: "authorized", issuer: "https://accounts.example/provider", persistence: "session", scopes: ["project:read"] }; setRevision(value => value + 1); }}>Complete fixture login</button>
      <a className="button" href="./mcp-review-fixture.html">Connected-app reviews fixture</a>
    </div>
    <div style={{ width, maxWidth: "100%" }} className="integration-card">
      <h2>Project tools</h2>
      <McpAuthenticationPanel key={revision} brainId="fixture" server={{ ...server, authentication: status }} />
      <McpCredentialPanel brainId="fixture" server={remote} onSaved={updated => { setRemote(updated); setRevision(value => value + 1); }} />
    </div>
    <div style={{ width, maxWidth: "100%", marginTop: 20 }} className="integration-card">
      <h2>Local fixture tools</h2><McpCredentialPanel brainId="fixture" server={local} onSaved={setLocal} />
    </div>
    <details><summary>Credential fixture receipts · no secret values</summary><pre>{JSON.stringify(credentialReceipts, null, 2)}</pre></details>
  </main>;
}
document.documentElement.dataset.colorScheme = "dark";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
