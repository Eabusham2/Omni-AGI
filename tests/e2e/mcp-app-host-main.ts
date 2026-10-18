import { app, BrowserWindow, ipcMain, protocol, session } from "electron";
import { resolve } from "node:path";
import { McpAppService } from "../../src/main/mcpApps";
import { allowMcpAppFrameNavigation, MCP_APP_SCHEME_PRIVILEGES, mcpAppFrameOwner, mcpAppRequestHeaders, mcpAppResponseHeaders } from "../../src/main/mcpAppProxy";
import { requireStudioIpcSender } from "../../src/main/studioIpcTrust";
import { MCP_APP_MIME, MCP_APP_PROTOCOL, MCP_APP_SCHEME, type McpAppDisplayAcknowledgement } from "../../src/shared/mcpApps";
import { PluginService } from "../../src/main/pluginService";
import type { PluginCatalog } from "../../src/main/pluginCatalog";
import { allowPluginPanelFrameNavigation, allowPluginPanelRequest, isPluginPanelUrl, pluginPanelFrameIdentity, pluginPanelRequestHeaders, pluginPanelResponseHeaders, PLUGIN_PANEL_SCHEME_PRIVILEGES } from "../../src/main/pluginPanelProtocol";
import { PLUGIN_PANEL_SCHEME } from "../../src/shared/pluginPanels";
import type { McpToolSummary } from "../../src/shared/types";

const rendererOrigin = new URL(process.env.OMNI_MCP_APP_UI_ORIGIN ?? "http://127.0.0.1:5192");
if (rendererOrigin.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(rendererOrigin.hostname) || rendererOrigin.username || rendererOrigin.password || rendererOrigin.search || rendererOrigin.hash || rendererOrigin.pathname !== "/") throw new Error("Fixture renderer must use a local development origin.");
const rendererUrl = new URL("/tests/e2e/mcp-app-host-fixture.html", rendererOrigin).href;
app.setPath("userData", resolve(process.env.OMNI_MCP_APP_UI_DATA ?? ".runtime/mcp-app-host-ui-profile"));
protocol.registerSchemesAsPrivileged([{ scheme: MCP_APP_SCHEME, privileges: MCP_APP_SCHEME_PRIVILEGES }, { scheme: PLUGIN_PANEL_SCHEME, privileges: PLUGIN_PANEL_SCHEME_PRIVILEGES }]);
const trace: unknown[] = [];
// The catalog edge is synthetic; real PluginService protocol/frame enforcement
// runs below. No package is installed in the user's actual catalog.
const panel = { title: "Static package panel", pluginId: "fixture.panel", contentSha256: "a".repeat(64), panelId: "about" };
const pluginPanels = new PluginService({ package: async (id: string, sha: string) => {
  if (id !== panel.pluginId || sha !== panel.contentSha256) throw new Error("Unknown fixture package");
  return { manifest: { panels: [{ id: panel.panelId, title: panel.title, html: "<section><h2>Static package panel</h2><p>Delivered through the actual dedicated protocol in an opaque, script-disabled frame.</p></section>", css: "body { margin: 16px; font: 16px/1.5 system-ui; color: #16202a; background: #f6f8fa; }" }] } };
} } as unknown as PluginCatalog, {} as ConstructorParameters<typeof PluginService>[1]);
const uri = "ui://fixture/real-sandbox.html";
const origin = { brainId: "fixture", serverId: "fixture-server", serverLabel: "Isolated fixture App", endpoint: "https://fixture.invalid/mcp", connectionEpoch: 1 };
const tool: McpToolSummary = { id: "mcp.fixture.editor.hash", remoteName: "fixture_action", actions: ["call"], description: "Synthetic fixture action", inputSchema: { type: "object" },
  metadata: { ui: { resourceUri: uri, visibility: ["app", "model"] } } };
const resourceTool: McpToolSummary = { ...tool, id: "mcp.fixture.client.resources", remoteName: "resource_wrapper" };
const html = `<!doctype html><html><head><meta charset="UTF-8"><style>body{font:16px system-ui;margin:16px;color:#16202a;background:#f6f8fa}button{padding:10px 14px;border:1px solid #758596;border-radius:8px;background:#fff;color:#16202a}p{overflow-wrap:anywhere}pre{white-space:pre-wrap}</style></head><body>
<h2>Isolated connected App</h2><p id="status">Waiting for actual handshake</p><p id="boundary"></p><p id="network"></p><p id="document"></p><p id="display">Display mode: inline</p>
<button id="fullscreen" disabled>App requests fullscreen</button> <button id="inline" disabled>App requests inline</button> <button id="call" disabled>Run synthetic tool once</button><pre id="result"></pre><pre id="display-result"></pre>
<script>let initialized=false,displaySequence=0;const send=message=>parent.postMessage({jsonrpc:"2.0",...message},"*");
document.getElementById("document").textContent="Page instance: "+Math.random().toString(36).slice(2);
document.getElementById("boundary").textContent="Node globals: "+typeof require+" / "+typeof process+"; Studio bridge: "+typeof window.omni;
addEventListener("message",event=>{if(event.source!==parent||event.data?.jsonrpc!=="2.0")return;const message=event.data;
 if(message.id===1){if(message.error){document.getElementById("status").textContent=JSON.stringify(message.error);return;}initialized=true;send({method:"ui/notifications/initialized"});document.getElementById("status").textContent="Actual App handshake complete";document.getElementById("call").disabled=false;
  const modes=message.result?.hostContext?.availableDisplayModes||[];document.getElementById("fullscreen").disabled=!modes.includes("fullscreen");document.getElementById("inline").disabled=!modes.includes("inline");}
 if(message.id==="fixture-action-1")document.getElementById("result").textContent=JSON.stringify(message.result||message.error,null,2);
 if(typeof message.id==="string"&&message.id.startsWith("fixture-display-"))document.getElementById("display-result").textContent="Actual display response: "+JSON.stringify(message.result||message.error);
 if(message.method==="ui/notifications/host-context-changed"&&message.params?.displayMode)document.getElementById("display").textContent="Display mode: "+message.params.displayMode;
});
document.getElementById("call").onclick=()=>{if(initialized)send({id:"fixture-action-1",method:"tools/call",params:{name:"fixture_action",arguments:{fixture:true}}});};
for(const mode of ["fullscreen","inline"])document.getElementById(mode).onclick=()=>{if(initialized)send({id:"fixture-display-"+(++displaySequence),method:"ui/request-display-mode",params:{mode}});};
fetch("https://blocked.invalid/fixture").then(()=>{document.getElementById("network").textContent="Unexpected network success";}).catch(()=>{document.getElementById("network").textContent="Undeclared network blocked";});
send({id:1,method:"ui/initialize",params:{protocolVersion:${JSON.stringify(MCP_APP_PROTOCOL)},appInfo:{name:"isolated-fixture",version:"1"},appCapabilities:{availableDisplayModes:["inline","fullscreen"]}}});
</script></body></html>`;
let calls = 0;
let appAccess = true, acknowledgementPolicy: "normal" | "hold" | "reject" | "retire-after-fullscreen" = "normal";
let retirementTimer: ReturnType<typeof setTimeout> | undefined;
const heldAcknowledgements: Array<{ input: McpAppDisplayAcknowledgement; resolve(): void; reject(error: Error): void }> = [];
const apps = new McpAppService({
  requireAccess: value => { if (!appAccess || value.brainId !== "fixture" || value.serverId !== origin.serverId) throw new Error("Fixture App access is retired or belongs to another owner"); },
  tool: () => tool, resourceTool: () => resourceTool,
  execute: async invocation => {
    trace.push({ operation: invocation.toolId === resourceTool.id ? "resource-read" : "synthetic-tool", arguments: invocation.arguments });
    const output = invocation.toolId === resourceTool.id
      ? { format: "omni-mcp-client-result", operation: "resources/read", untrustedTaskData: true, provenance: { requestedUri: uri }, result: { contents: [{ uri, mimeType: MCP_APP_MIME, text: html }] } }
      : { content: [{ type: "text", text: `Actual synthetic host dispatch count: ${++calls}` }] };
    return { id: "fixture-receipt", toolId: invocation.toolId, action: invocation.action, state: "complete", startedAt: new Date().toISOString(), output };
  },
  approveTool: async () => false, approveNetwork: async () => false, approveLink: async () => false,
  openBrowser: async () => { throw new Error("Fixture never opens external URLs"); }, admit: async () => undefined, theme: () => "dark"
});
app.whenReady().then(async () => {
  const descriptor = await apps.capture({ origin, tool, arguments: { title: "Synthetic completed call" }, result: { content: [{ type: "text", text: "Synthetic completed call result" }] } });
  protocol.handle(MCP_APP_SCHEME, request => apps.document(request.url));
  protocol.handle(PLUGIN_PANEL_SCHEME, request => pluginPanels.panelDocument(request.url));
  session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  const requests = new Map<number, string>();
  const panelRequests = new Set<number>();
  session.defaultSession.webRequest.onBeforeRequest((details, callback) => {
    if (isPluginPanelUrl(details.url) || pluginPanelFrameIdentity(details.frame)) {
      panelRequests.add(details.id);
      const allowed = allowPluginPanelRequest(details);
      trace.push({ operation: allowed ? "static-panel-document" : "static-panel-request-denied", url: details.url });
      callback({ cancel: !allowed }); return;
    }
    let owner = requests.get(details.id) ?? mcpAppFrameOwner(details.frame);
    if (!owner) { try { const url = new URL(details.url); if (url.protocol === `${MCP_APP_SCHEME}:`) owner = url.hostname; } catch {} }
    if (owner) { requests.set(details.id, owner); const allowed = apps.allowNetwork(owner, details.url, details.resourceType); if (!allowed) trace.push({ operation: "network-denied", url: details.url }); callback({ cancel: !allowed }); }
    else callback({});
  });
  session.defaultSession.webRequest.onBeforeSendHeaders((details, callback) => callback({ requestHeaders: panelRequests.has(details.id) ? pluginPanelRequestHeaders(details.requestHeaders) : requests.has(details.id) ? mcpAppRequestHeaders(details.requestHeaders) : details.requestHeaders }));
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => callback({ responseHeaders: panelRequests.has(details.id) ? pluginPanelResponseHeaders(details.responseHeaders ?? {}) : requests.has(details.id) ? mcpAppResponseHeaders(details.responseHeaders ?? {}) : details.url === rendererUrl ? {
    ...details.responseHeaders, "Content-Security-Policy": [`default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-src ${MCP_APP_SCHEME}: ${PLUGIN_PANEL_SCHEME}:; connect-src 'self' ${rendererOrigin.origin.replace(/^http:/, "ws:")}; object-src 'none'; base-uri 'none'`]
  } : details.responseHeaders }));
  session.defaultSession.webRequest.onCompleted(details => { requests.delete(details.id); panelRequests.delete(details.id); });
  session.defaultSession.webRequest.onErrorOccurred(details => { requests.delete(details.id); panelRequests.delete(details.id); });
  const window = new BrowserWindow({ width: 1100, height: 900, title: "Embedded App — isolated host", webPreferences: {
    preload: resolve("tests/e2e/mcp-app-host-preload.cjs"), sandbox: true, contextIsolation: true, nodeIntegration: false, nodeIntegrationInSubFrames: false
  } });
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-frame-navigate", event => { if (!event.isMainFrame && !allowPluginPanelFrameNavigation(event, rendererUrl) && !allowMcpAppFrameNavigation(event, (owner, url) => apps.allowNetwork(owner, url, "subFrame"))) event.preventDefault(); });
  window.webContents.on("will-navigate", event => event.preventDefault());
  const trusted = (event: Electron.IpcMainInvokeEvent): void => requireStudioIpcSender({ senderId: event.sender.id, trustedSenderId: window.webContents.id, senderFrame: event.senderFrame,
    mainFrame: event.sender.mainFrame, frameUrl: event.senderFrame?.url, rendererUrl });
  ipcMain.handle("fixture:descriptor", event => { trusted(event); return descriptor; });
  ipcMain.handle("fixture:plugin-panel", event => { trusted(event); return panel; });
  ipcMain.handle("fixture:trace", event => { trusted(event); return trace; });
  ipcMain.handle("fixture:app-open", (event, brainId, id) => { trusted(event); return apps.open(brainId, id); });
  ipcMain.handle("fixture:app-request", (event, input) => { trusted(event); trace.push({ operation: "reviewed-app-rpc", method: input.method }); return apps.request(input); });
  ipcMain.handle("fixture:app-close", (event, brainId, id, token) => { trusted(event); apps.close(brainId, id, token); return true; });
  const stopDisplay = apps.subscribeDisplayModes(event => {
    trace.push({ operation: "app-display-event", action: event.action, mode: event.displayMode, sequence: event.sequence, ...(event.reason ? { reason: event.reason } : {}) });
    if (!window.isDestroyed()) window.webContents.send("fixture:app-display-mode", event);
  });
  ipcMain.handle("fixture:app-display-ack", (event, input: McpAppDisplayAcknowledgement) => {
    trusted(event); trace.push({ operation: "app-display-ack", mode: input.mode, policy: acknowledgementPolicy });
    if (acknowledgementPolicy === "reject") throw new Error("Synthetic renderer acknowledgement was refused.");
    if (acknowledgementPolicy === "hold") return new Promise<void>((resolve, reject) => heldAcknowledgements.push({ input, resolve, reject }));
    apps.acknowledgeDisplayMode(input);
    if (acknowledgementPolicy === "retire-after-fullscreen" && input.mode === "fullscreen") {
      clearTimeout(retirementTimer);
      retirementTimer = setTimeout(() => { appAccess = false; apps.invalidate("fixture", origin.serverId); if (!window.isDestroyed()) window.webContents.send("fixture:app-access-changed", false);
        trace.push({ operation: "fixture-app-access", enabled: false, source: "explicit-retirement-fixture" }); }, 3000);
    }
  });
  ipcMain.handle("fixture:display-ack-policy", (event, mode: string) => {
    trusted(event); if (!["normal", "hold", "reject", "retire-after-fullscreen"].includes(mode)) throw new Error("Unknown fixture acknowledgement policy."); acknowledgementPolicy = mode as typeof acknowledgementPolicy;
  });
  ipcMain.handle("fixture:display-ack-release", event => {
    trusted(event);
    for (const held of heldAcknowledgements.splice(0)) {
      try { apps.acknowledgeDisplayMode(held.input); held.resolve(); trace.push({ operation: "held-display-ack", state: "accepted" }); }
      catch (error) { held.reject(error instanceof Error ? error : new Error(String(error))); trace.push({ operation: "held-display-ack", state: "expired" }); }
    }
  });
  ipcMain.handle("fixture:app-access", (event, enabled: boolean) => {
    trusted(event); if (typeof enabled !== "boolean") throw new Error("Fixture access requires an explicit boolean.");
    clearTimeout(retirementTimer); appAccess = enabled; if (!enabled) apps.invalidate("fixture", origin.serverId); window.webContents.send("fixture:app-access-changed", enabled); trace.push({ operation: "fixture-app-access", enabled });
  });
  window.on("closed", () => { clearTimeout(retirementTimer); stopDisplay(); for (const held of heldAcknowledgements.splice(0)) held.reject(new Error("Fixture window closed.")); });
  await window.loadURL(rendererUrl);
});
app.on("window-all-closed", () => { apps.dispose(); app.quit(); });
