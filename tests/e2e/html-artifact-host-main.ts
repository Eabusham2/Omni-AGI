import { randomBytes } from "node:crypto";
import { cp, mkdtemp, mkdir, rm } from "node:fs/promises";
import { readFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { app, BrowserWindow, ipcMain, protocol, session } from "electron";
import { HtmlArtifactService, HTML_ARTIFACT_SCHEME_PRIVILEGES, allowHtmlArtifactNavigation,
  allowHtmlArtifactRequest, htmlArtifactFrameOwner, htmlArtifactRequestHeaders,
  htmlArtifactResponseHeaders, isHtmlArtifactUrl, type HtmlArtifactToolExecutor } from "../../src/main/htmlArtifactService";
import { rekeyHtmlArtifactWorkspace } from "../../src/main/htmlArtifactPortability";
import { HtmlArtifactStore } from "../../src/main/htmlArtifactStore";
import { requireStudioIpcSender } from "../../src/main/studioIpcTrust";
import { HTML_ARTIFACT_SCHEME, type HtmlArtifactResult } from "../../src/shared/htmlArtifacts";
import type { ActionEvent, ToolExecutionResult } from "../../src/shared/types";

let rendererUrl = "";
app.setPath("userData", resolve(".runtime/html-artifact-host-ui-profile"));
protocol.registerSchemesAsPrivileged([{ scheme: HTML_ARTIFACT_SCHEME, privileges: HTML_ARTIFACT_SCHEME_PRIVILEGES }]);

const trace: Array<Record<string, unknown>> = [];
const approvals = new Map<string, string>();
let fixtureRoot: string | undefined;
let artifacts: HtmlArtifactService | undefined;
let rendererServer: Server | undefined;

async function startRendererServer(): Promise<string> {
  const [html, javascript, css] = await Promise.all([
    readFile(resolve("tests/e2e/html-artifact-host-fixture.html")),
    readFile(resolve(".runtime/html-artifact-host-ui/renderer-check.js")),
    readFile(resolve(".runtime/html-artifact-host-ui/renderer-check.css"))
  ]);
  let expectedHost = "";
  const routes = new Map<string, { body: Buffer; type: string }>([
    ["/index.html", { body: html, type: "text/html; charset=utf-8" }],
    ["/renderer.js", { body: javascript, type: "text/javascript; charset=utf-8" }],
    ["/renderer.css", { body: css, type: "text/css; charset=utf-8" }]
  ]);
  rendererServer = createServer((request, response) => {
    let path = "";
    try {
      const parsed = new URL(request.url ?? "", "http://127.0.0.1");
      if (!parsed.search && !parsed.hash) path = parsed.pathname;
    } catch { /* handled as a missing route */ }
    const resource = request.method === "GET" && request.headers.host === expectedHost ? routes.get(path) : undefined;
    if (!resource) {
      response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'" });
      response.end("Not found"); return;
    }
    const headers: Record<string, string | number> = { "Content-Type": resource.type,
      "Content-Length": resource.body.length, "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
      "Cross-Origin-Resource-Policy": "same-origin", "Referrer-Policy": "no-referrer" };
    if (path === "/index.html") headers["Content-Security-Policy"] =
      `default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-src ${HTML_ARTIFACT_SCHEME}:; connect-src 'none'; object-src 'none'; base-uri 'none'`;
    response.writeHead(200, headers); response.end(resource.body);
  });
  await new Promise<void>((resolveReady, reject) => {
    rendererServer!.once("error", reject);
    rendererServer!.listen(0, "127.0.0.1", () => { rendererServer!.off("error", reject); resolveReady(); });
  });
  const address = rendererServer.address();
  if (!address || typeof address === "string") throw new Error("Fixture renderer server has no loopback port.");
  expectedHost = `127.0.0.1:${address.port}`;
  return `http://${expectedHost}/index.html`;
}

const executor: HtmlArtifactToolExecutor = {
  async execute(invocation, _progress, requestId = "missing-request"): Promise<ToolExecutionResult> {
    const now = new Date().toISOString();
    if (invocation.toolId !== "fixture.readonly" || invocation.action !== "inspect") {
      return { id: requestId, toolId: invocation.toolId, action: invocation.action,
        state: "failed", startedAt: now, finishedAt: now, error: "Fixture permits only its synthetic readonly inspection." };
    }
    if (!invocation.approvalToken) {
      const token = randomBytes(32).toString("hex"); approvals.set(requestId, token);
      trace.push({ operation: "synthetic-readonly-ask", requestId, arguments: invocation.arguments });
      return { id: requestId, toolId: invocation.toolId, action: invocation.action,
        state: "approval-required", startedAt: now, approvalToken: token };
    }
    if (approvals.get(requestId) !== invocation.approvalToken) {
      return { id: requestId, toolId: invocation.toolId, action: invocation.action,
        state: "failed", startedAt: now, finishedAt: now, error: "Synthetic approval identity changed." };
    }
    approvals.delete(requestId);
    trace.push({ operation: "synthetic-readonly-complete", requestId, arguments: invocation.arguments });
    return { id: requestId, toolId: invocation.toolId, action: invocation.action,
      state: "complete", dispatchStarted: true, startedAt: now, finishedAt: new Date().toISOString(),
      output: { syntheticFixture: true, readonly: true, observedCounter: invocation.arguments.value } };
  },
  async cancelAndWait(_brainId, requestId): Promise<number> {
    const removed = requestId && approvals.delete(requestId) ? 1 : 0;
    trace.push({ operation: "artifact-view-cancel", requestId, removed }); return removed;
  },
  invalidateApproval(_brainId, token): boolean {
    for (const [requestId, candidate] of approvals) if (candidate === token) {
      approvals.delete(requestId); trace.push({ operation: "synthetic-approval-invalidated", requestId }); return true;
    }
    return false;
  }
};

const content = {
  title: "Local counter and reviewed host inspection",
  html: `<main><h2>Opaque saved HTML artifact</h2><p id="boundary">Starting isolated script…</p>
    <p id="network">Checking undeclared network…</p><canvas id="counter" width="520" height="120"></canvas>
    <div><button id="increment">Increment locally</button><button id="inspect">Request reviewed host inspection</button></div>
    <pre id="result">No host action requested.</pre></main>`,
  css: `body{margin:0;padding:16px;font:15px/1.45 system-ui;color:#e8edf4;background:#131820}main{display:grid;gap:12px}h2,p,pre{margin:0}canvas{box-sizing:border-box;width:100%;max-width:520px;border:1px solid #66758a;border-radius:8px;background:#0b0f15}div{display:flex;gap:8px;flex-wrap:wrap}button{padding:9px 12px;color:#e8edf4;background:#273447;border:1px solid #71829a;border-radius:7px}pre{white-space:pre-wrap;overflow-wrap:anywhere}`,
  script: `let count=0;const canvas=document.getElementById('counter'),context=canvas.getContext('2d'),result=document.getElementById('result');
const draw=()=>{context.clearRect(0,0,canvas.width,canvas.height);context.fillStyle='#18212d';context.fillRect(0,0,canvas.width,canvas.height);context.fillStyle='#76d6ff';context.fillRect(18,40,Math.min(480,count*42),40);context.fillStyle='#f5f8fc';context.font='20px system-ui';context.fillText('Local counter: '+count,18,28);};draw();
document.getElementById('boundary').textContent='Node globals: '+typeof require+' / '+typeof process+'; Studio bridge: '+typeof window.omni+'; document: '+location.href;
document.getElementById('increment').onclick=()=>{count++;draw();result.textContent='Local-only update '+count+'; no host action.';};
document.getElementById('inspect').onclick=async()=>{result.textContent='Waiting for Studio-owned Ask review…';try{const value=await omniArtifact.call('fixture.readonly','inspect',{value:count,note:'Synthetic readonly fixture'});result.textContent='Actual reviewed fixture receipt: '+JSON.stringify(value,null,2);}catch(error){result.textContent='Host request ended: '+String(error);}};
fetch('https://blocked.invalid/html-artifact-fixture').then(()=>{document.getElementById('network').textContent='Unexpected network success';}).catch(()=>{document.getElementById('network').textContent='Undeclared network blocked by the artifact host';});`,
  tools: [{ toolId: "fixture.readonly", actions: ["inspect"] }],
  height: 430
};

function event(brainId: string, result: HtmlArtifactResult): ActionEvent {
  const now = new Date().toISOString();
  return { id: `fixture-${brainId}`, brainId, state: "complete", createdAt: now, updatedAt: now,
    action: { kind: "tool", source: "brain", toolId: "studio.artifact", action: "create", arguments: {} },
    execution: { id: `fixture-execution-${brainId}`, toolId: "studio.artifact", action: "create",
      state: "complete", startedAt: now, finishedAt: now, output: result } };
}

app.whenReady().then(async () => {
  rendererUrl = await startRendererServer();
  fixtureRoot = await mkdtemp(join(tmpdir(), "omni-html-artifact-host-"));
  const sourceBrainId = "fixture-source", copiedBrainId = "fixture-copy";
  const sourceRoot = join(fixtureRoot, sourceBrainId), copiedRoot = join(fixtureRoot, copiedBrainId);
  const created = await new HtmlArtifactStore(sourceRoot, sourceBrainId).create("fixture-artifact-action", content);
  await mkdir(copiedRoot, { recursive: true });
  await cp(join(sourceRoot, "html-artifacts"), join(copiedRoot, "html-artifacts"), { recursive: true });
  await rekeyHtmlArtifactWorkspace(copiedRoot, sourceBrainId, copiedBrainId);
  const roots = new Map([[sourceBrainId, sourceRoot], [copiedBrainId, copiedRoot]]);
  artifacts = new HtmlArtifactService({ brainDirectory: (brainId: string) => {
    const root = roots.get(brainId); if (!root) throw new Error("Unknown fixture artifact owner."); return root;
  } }, executor);
  const descriptors = { enabled: event(sourceBrainId, created), dormant: event(copiedBrainId, created) };

  protocol.handle(HTML_ARTIFACT_SCHEME, async request => {
    const response = await artifacts!.document(request.url);
    trace.push({ operation: "artifact-document", url: request.url, status: response.status,
      csp: response.headers.get("content-security-policy") });
    return response;
  });
  session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  session.defaultSession.webRequest.onBeforeRequest((details, callback) => {
    if (isHtmlArtifactUrl(details.url) || htmlArtifactFrameOwner(details.frame)) {
      const allowed = allowHtmlArtifactRequest(details);
      if (!allowed) trace.push({ operation: "artifact-request-denied", url: details.url, resourceType: details.resourceType });
      callback({ cancel: !allowed }); return;
    }
    callback({});
  });
  session.defaultSession.webRequest.onBeforeSendHeaders((details, callback) => {
    if (isHtmlArtifactUrl(details.url) || htmlArtifactFrameOwner(details.frame)) {
      callback({ requestHeaders: htmlArtifactRequestHeaders(details.requestHeaders) }); return;
    }
    callback({ requestHeaders: details.requestHeaders });
  });
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    if (isHtmlArtifactUrl(details.url) || htmlArtifactFrameOwner(details.frame)) {
      callback({ responseHeaders: htmlArtifactResponseHeaders(details.responseHeaders ?? {}) }); return;
    }
    if (details.url === rendererUrl) {
      callback({ responseHeaders: { ...details.responseHeaders,
        "Content-Security-Policy": [`default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-src ${HTML_ARTIFACT_SCHEME}:; connect-src 'none'; object-src 'none'; base-uri 'none'`] } }); return;
    }
    callback({ responseHeaders: details.responseHeaders });
  });

  const window = new BrowserWindow({ width: 1120, height: 900, title: "Saved HTML artifact — bounded host fixture",
    webPreferences: { preload: resolve("tests/e2e/html-artifact-host-preload.cjs"), sandbox: true,
      contextIsolation: true, nodeIntegration: false, nodeIntegrationInSubFrames: false } });
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-frame-navigate", event => {
    if (!event.isMainFrame && !allowHtmlArtifactNavigation(event, rendererUrl)) event.preventDefault();
  });
  window.webContents.on("will-navigate", event => event.preventDefault());
  const trusted = (ipcEvent: Electron.IpcMainInvokeEvent): void => requireStudioIpcSender({
    senderId: ipcEvent.sender.id, trustedSenderId: window.webContents.id, senderFrame: ipcEvent.senderFrame,
    mainFrame: ipcEvent.sender.mainFrame, frameUrl: ipcEvent.senderFrame?.url, rendererUrl
  });
  ipcMain.handle("fixture:artifact-descriptors", ipcEvent => { trusted(ipcEvent); return descriptors; });
  ipcMain.handle("fixture:artifact-trace", ipcEvent => { trusted(ipcEvent); return trace; });
  ipcMain.handle("fixture:artifact-open", async (ipcEvent, request) => {
    trusted(ipcEvent); const launch = await artifacts!.open(request);
    trace.push({ operation: "artifact-open", brainId: request.brainId, viewId: launch.viewId,
      origin: new URL(launch.url).origin, scriptEnabled: launch.scriptEnabled }); return launch;
  });
  ipcMain.handle("fixture:artifact-activate", async (ipcEvent, request) => {
    trusted(ipcEvent); const result = await artifacts!.activate(request);
    trace.push({ operation: "artifact-activate", brainId: request.brainId, revision: result.revision }); return result;
  });
  ipcMain.handle("fixture:artifact-request", async (ipcEvent, request) => {
    trusted(ipcEvent); return artifacts!.request(request);
  });
  ipcMain.handle("fixture:artifact-review", async (ipcEvent, request) => {
    trusted(ipcEvent); trace.push({ operation: "host-review", requestId: request.requestId, decision: request.decision });
    return artifacts!.review(request);
  });
  ipcMain.handle("fixture:artifact-close", async (ipcEvent, brainId, viewId, nonce) => {
    trusted(ipcEvent); await artifacts!.close(brainId, viewId, nonce);
    trace.push({ operation: "artifact-close", brainId, viewId });
  });
  await window.loadURL(rendererUrl);
}).catch(error => { console.error(error); app.quit(); });

app.on("window-all-closed", () => {
  void artifacts?.dispose().catch(() => undefined).finally(async () => {
    if (fixtureRoot) await rm(fixtureRoot, { recursive: true, force: true }).catch(() => undefined);
    if (rendererServer?.listening) await new Promise<void>(resolveClosed => rendererServer!.close(() => resolveClosed()));
    app.quit();
  });
});
