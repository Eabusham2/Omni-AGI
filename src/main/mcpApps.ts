import { createHash, randomBytes, randomUUID } from "node:crypto";
import type { McpAppCsp, McpAppDescriptor, McpAppHostResult, McpAppLaunch, McpAppRequest, McpAppObservationEvent, McpAppDisplayEvent, McpAppDisplayAcknowledgement } from "../shared/mcpApps";
import { MCP_APP_MIME, MCP_APP_PROTOCOL, MCP_APP_SCHEME } from "../shared/mcpApps";
import type { McpRequestOrigin } from "../shared/mcpInteractions";
import type { McpToolSummary, ToolExecutionResult, ToolInvocation } from "../shared/types";
import { mcpAppCsp, mcpAppResource, mcpToolUi } from "./mcpAppMetadata";
import { mcpAppProxyDocument, mcpAppResourceCsp } from "./mcpAppProxy";
import { McpHostRequestError, mcpElicitation } from "./mcpInteractionProtocol";
import type { McpAppArtifactStore } from "./mcpAppArtifacts";
import type { McpAppArtifactSummary } from "../shared/mcpAppArtifacts";
import type { McpAppDataReview, McpAppObservationReceipt } from "../shared/mcpAppTaskData";
import { mcpAppTaskData } from "./mcpAppTaskData";
import type { McpAppServerNotification, McpAppServerNotificationMethod } from "../shared/mcpHostServices";
import type { McpNegotiatedCapabilities } from "../shared/mcpCapabilities";
import { McpAppDisplayModeHost, type McpAppDisplayLease } from "./mcpAppDisplayModes";

interface AppView {
  handleId: string;
  descriptor: McpAppDescriptor;
  origin: McpRequestOrigin;
  input: Record<string, unknown>;
  result: Record<string, unknown>;
  tool: McpToolSummary;
  controller: AbortController;
  token?: string;
  html?: string;
  csp?: McpAppCsp;
  phase: "available" | "loading" | "initialized" | "ready" | "closed";
  opening?: Promise<McpAppLaunch>;
  requests: Map<string, { hash: string; promise: Promise<McpAppHostResult>; controller: AbortController }>;
  captureSha256?: string;
  serverNotifications?: { tools: boolean; resources: boolean };
  display?: McpAppDisplayLease;
}
export interface McpCompletedAppCall {
  origin: McpRequestOrigin;
  tool: McpToolSummary;
  arguments: Record<string, unknown>;
  result: Record<string, unknown>;
}
export interface McpAppHost {
  capture(call: McpCompletedAppCall): Promise<McpAppDescriptor | undefined>;
  requireCall(viewId: string, brainId: string, serverId: string, remoteToolName: string): void;
  invalidate(brainId: string, serverId: string, reason?: "session-initialized"): void;
  serverNotification?(origin: McpRequestOrigin, method: McpAppServerNotificationMethod): void;
}
export interface McpAppServiceOptions {
  requireAccess(origin: McpRequestOrigin): void;
  tool(origin: McpRequestOrigin, remoteName: string): McpToolSummary;
  resourceTool(origin: McpRequestOrigin): McpToolSummary;
  execute(invocation: ToolInvocation, requestId: string, viewId: string, signal: AbortSignal): Promise<ToolExecutionResult>;
  /** Actual reviewed permission, never a timed or fabricated approval. */
  approveTool(origin: McpRequestOrigin, invocation: ToolInvocation, signal: AbortSignal): Promise<boolean>;
  approveNetwork(origin: McpRequestOrigin, csp: McpAppCsp, signal: AbortSignal): Promise<boolean>;
  approveLink(origin: McpRequestOrigin, url: string, signal: AbortSignal): Promise<boolean>;
  openBrowser(url: string): Promise<void>;
  admit(bytes: number): Promise<void>;
  theme(): "light" | "dark";
  artifacts?(brainId: string): McpAppArtifactStore;
  /** Current installed/access-checked identity only; no network or implicit Load. */
  currentOrigin?(brainId: string, serverId: string): McpRequestOrigin;
  /** Explicit reviewed pending data only; never claims neural admission. */
  observe?(origin: McpRequestOrigin, request: McpAppDataReview, signal: AbortSignal): Promise<McpAppObservationReceipt>;
  log?(origin: McpRequestOrigin, params: Record<string, unknown>, signal: AbortSignal,
    context: { viewId: string; privateSecrets: readonly string[] }): Promise<void> | void;
  serverCapabilities?(origin: McpRequestOrigin): McpNegotiatedCapabilities;
}
const object = (value: unknown): Record<string, unknown> => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const sha = (value: string): string => createHash("sha256").update(value).digest("hex");

/** An untrusted view may propose actions; it never owns a dispatch bridge. */
export class McpAppService implements McpAppHost {
  private readonly views = new Map<string, AppView>();
  private readonly handles = new Map<string, AppView>();
  private readonly preparing = new Map<string, AppView>();
  private readonly opening = new Map<string, Promise<McpAppLaunch>>();
  private readonly observationListeners = new Set<(event: McpAppObservationEvent) => void>();
  private readonly serverNotificationListeners = new Set<(event: McpAppServerNotification) => void>();
  private serverNotificationSequence = 0;
  private readonly displayModes = new McpAppDisplayModeHost();
  constructor(private readonly options: McpAppServiceOptions) {}
  subscribeObservations(listener: (event: McpAppObservationEvent) => void): () => void { this.observationListeners.add(listener); return () => { this.observationListeners.delete(listener); }; }
  publishObservation(event: McpAppObservationEvent): void { for (const listener of this.observationListeners) { try { listener(structuredClone(event)); } catch { /* read-only observers grant no authority */ } } }
  subscribeServerNotifications(listener: (event: McpAppServerNotification) => void): () => void {
    this.serverNotificationListeners.add(listener); return () => { this.serverNotificationListeners.delete(listener); };
  }
  subscribeDisplayModes(listener: (event: McpAppDisplayEvent) => void): () => void { return this.displayModes.subscribe(listener); }
  acknowledgeDisplayMode(input: McpAppDisplayAcknowledgement): void { this.displayModes.acknowledge(input); }
  serverNotification(origin: McpRequestOrigin, method: McpAppServerNotificationMethod): void {
    if (!this.serverNotificationListeners.size || !["notifications/tools/list_changed", "notifications/resources/list_changed"].includes(method)) return;
    for (const view of this.views.values()) {
      if (view.phase !== "ready" || !view.token || view.controller.signal.aborted || view.origin.brainId !== origin.brainId || view.origin.serverId !== origin.serverId ||
          view.origin.connectionEpoch !== origin.connectionEpoch || view.origin.endpoint !== origin.endpoint ||
          !(method === "notifications/tools/list_changed" ? view.serverNotifications?.tools : view.serverNotifications?.resources)) continue;
      try { this.options.requireAccess(view.origin); } catch { continue; }
      const event: McpAppServerNotification = { brainId: view.descriptor.brainId, serverId: view.descriptor.serverId, viewId: view.descriptor.id, viewToken: view.token,
        connectionEpoch: view.origin.connectionEpoch, sequence: ++this.serverNotificationSequence, method, params: {} };
      for (const listener of this.serverNotificationListeners) {
        try { if (view.phase !== "ready" || view.token !== event.viewToken) break; this.options.requireAccess(view.origin); listener(structuredClone(event)); }
        catch { /* a readonly relay failure never invokes or replays a remote effect */ }
      }
    }
  }
  async capture(call: McpCompletedAppCall): Promise<McpAppDescriptor | undefined> {
    const ui = mcpToolUi(call.tool); if (!ui.resourceUri) return undefined;
    await this.options.admit(Buffer.byteLength(JSON.stringify({ arguments: call.arguments, result: call.result })) * 3);
    this.options.requireAccess(call.origin);
    const descriptor: McpAppDescriptor = { id: randomUUID(), brainId: call.origin.brainId, serverId: call.origin.serverId,
      serverLabel: call.origin.serverLabel, resourceUri: ui.resourceUri, toolId: call.tool.id, remoteToolName: call.tool.remoteName,
      observedAt: new Date().toISOString(), untrusted: true };
    const saved = await this.options.artifacts?.(call.origin.brainId).capture(descriptor.id, call);
    this.options.requireAccess(call.origin);
    this.views.set(`${descriptor.brainId}:${descriptor.id}`, { handleId: randomUUID(), descriptor, origin: structuredClone(call.origin), input: structuredClone(call.arguments), result: structuredClone(call.result), tool: structuredClone(call.tool),
      controller: new AbortController(), phase: "available", requests: new Map(), ...(saved ? { captureSha256: saved.sha256 } : {}) });
    return structuredClone(descriptor);
  }
  private view(brainId: string, viewId: string, token?: string): AppView {
    const view = this.views.get(`${brainId}:${viewId}`);
    if (!view || view.descriptor.brainId !== brainId || view.phase === "closed" || token !== undefined && token !== view.token) throw new Error("This connected-app view is not available for that instance.");
    this.options.requireAccess(view.origin); view.controller.signal.throwIfAborted(); return view;
  }
  requireCall(viewId: string, brainId: string, serverId: string, remoteToolName: string): void {
    const view = this.view(brainId, viewId);
    if (view.descriptor.serverId !== serverId || view.phase !== "ready" && view.phase !== "loading" ||
      !mcpToolUi(this.options.tool(view.origin, remoteToolName)).visibility.includes("app")) throw new Error("The view cannot call this server tool.");
  }
  private async permitted(view: AppView, tool: McpToolSummary, args: Record<string, unknown>, requestId: string, signal: AbortSignal): Promise<unknown> {
    const invocation: ToolInvocation = { brainId: view.descriptor.brainId, toolId: tool.id, action: "call", arguments: args };
    let outcome = await this.options.execute(invocation, requestId, view.descriptor.id, signal);
    if (outcome.state === "approval-required") {
      if (!outcome.approvalToken || !await this.options.approveTool(view.origin, invocation, signal)) throw new McpHostRequestError(-32000, "The user did not approve this app action.");
      signal.throwIfAborted(); this.options.requireAccess(view.origin);
      outcome = await this.options.execute({ ...invocation, approvalToken: outcome.approvalToken }, requestId, view.descriptor.id, signal);
    }
    if (outcome.state !== "complete") throw new McpHostRequestError(-32000, outcome.error ?? "The app action did not complete; it was not repeated.");
    return outcome.output;
  }
  async describe(brainId: string, viewId: string): Promise<McpAppArtifactSummary | undefined> {
    return this.options.artifacts?.(brainId).summary(viewId);
  }
  async open(brainId: string, viewId: string, selection?: { captureSha256: string; reviewedMapping: true }): Promise<McpAppLaunch> {
    const key = `${brainId}:${viewId}`, pending = this.opening.get(key); if (pending) return pending;
    const operation = this.prepareOpen(brainId, viewId, selection); this.opening.set(key, operation);
    void operation.finally(() => { if (this.opening.get(key) === operation) this.opening.delete(key); }).catch(() => undefined);
    return operation;
  }
  private async prepareOpen(brainId: string, viewId: string, selection?: { captureSha256: string; reviewedMapping: true }): Promise<McpAppLaunch> {
    if (selection && (Object.keys(selection).sort().join(",") !== "captureSha256,reviewedMapping" || !/^[a-f0-9]{64}$/.test(selection.captureSha256) || selection.reviewedMapping !== true)) throw new Error("Invalid reviewed App source mapping.");
    const key = `${brainId}:${viewId}`;
    let view = this.views.get(key);
    // Immutable artifacts are portable, a live handle is not. A repeated Open
    // cannot replace an active handle or leave its old network grant alive.
    if (view && view.phase !== "closed" && view.phase !== "available") throw new Error("This view has already been opened. Close it explicitly instead of replaying its effects.");
    const artifacts = this.options.artifacts?.(brainId), summary = await artifacts?.summary(viewId);
    if (summary && artifacts) {
      if (summary.needsMappingReview && (!selection || selection.reviewedMapping !== true)) throw new Error("Saved App branches require an explicit source mapping review before reopening. Every prior artifact remains preserved.");
      const chosen = selection?.captureSha256 ?? summary.selectedCapture;
      if (!chosen || !summary.captures.some(capture => capture.sha256 === chosen)) throw new Error("The selected App source is not an owned capture.");
      if (selection) await artifacts.select(viewId, chosen);
      const saved = await artifacts.load(viewId, chosen), source = saved.capture.source;
      const origin = this.options.currentOrigin?.(brainId, source.serverId);
      if (!origin) throw new Error("Reopening saved Apps requires an explicit currently permitted connection; no session was restored.");
      if (origin.endpoint !== source.endpoint) throw new Error("The installed connection differs from the saved App source. Review/reconfigure that source mapping before connecting; no request was sent.");
      view = { handleId: randomUUID(), descriptor: { id: viewId, brainId, serverId: origin.serverId, serverLabel: origin.serverLabel, resourceUri: saved.capture.resourceUri,
          toolId: saved.capture.tool.id, remoteToolName: saved.capture.tool.remoteName, observedAt: saved.capture.observedAt, untrusted: true },
        origin, input: saved.input, result: saved.result, tool: saved.capture.tool, captureSha256: saved.sha256,
        controller: new AbortController(), phase: "available", requests: new Map() };
    }
    if (!view || view.descriptor.brainId !== brainId) throw new Error("This connected-app view is not available for that instance.");
    this.options.requireAccess(view.origin);
    if (view.phase === "closed") { view.controller = new AbortController(); view.requests.clear(); view.handleId = randomUUID(); view.phase = "available"; }
    if (view.phase !== "available") throw new Error("This view has already been opened. Close it explicitly instead of replaying its effects.");
    // A reconnect can advance the session epoch. This is not a live handle
    // until a permitted resource read and fresh policy checks finish.
    this.views.delete(key); this.handles.delete(view.handleId); this.preparing.set(key, view); view.phase = "loading";
    try { return await this.openView(view); }
    catch (error) { view.phase = "closed"; view.controller.abort(); this.views.set(key, view); throw error; }
    finally { this.preparing.delete(key); }
  }
  private async openView(view: AppView): Promise<McpAppLaunch> {
    const resource = await this.permitted(view, this.options.resourceTool(view.origin), { uri: view.descriptor.resourceUri }, randomUUID(), view.controller.signal);
    const parsed = mcpAppResource(resource, view.descriptor.resourceUri), csp = mcpAppCsp(parsed.metadata);
    if (this.options.currentOrigin) view.origin = this.options.currentOrigin(view.descriptor.brainId, view.descriptor.serverId);
    await this.options.admit(Buffer.byteLength(parsed.html) * 3);
    if (view.captureSha256) await this.options.artifacts?.(view.descriptor.brainId).resource(view.descriptor.id, view.captureSha256, {
      brainId: view.origin.brainId, serverId: view.origin.serverId, serverLabel: view.origin.serverLabel, endpoint: view.origin.endpoint
    }, view.descriptor.resourceUri, parsed.html, parsed.metadata, csp);
    if (Object.values(csp).some(values => values.length) && !await this.options.approveNetwork(view.origin, csp, view.controller.signal)) throw new Error("The app's declared network origins were not approved; no UI was loaded.");
    view.controller.signal.throwIfAborted(); this.options.requireAccess(view.origin);
    view.html = parsed.html; view.csp = csp; view.token = randomBytes(32).toString("hex");
    this.views.set(`${view.descriptor.brainId}:${view.descriptor.id}`, view); this.handles.set(view.handleId, view);
    const identity = { brainId: view.descriptor.brainId, viewId: view.descriptor.id, viewToken: view.token };
    view.display = this.displayModes.create(identity, () => {
      if (this.view(identity.brainId, identity.viewId, identity.viewToken) !== view || view.phase !== "ready")
        throw new Error("This app is not the current ready view.");
    });
    return { view: structuredClone(view.descriptor), proxyUrl: `${MCP_APP_SCHEME}://${view.handleId}/proxy.html`,
      proxyOrigin: `${MCP_APP_SCHEME}://${view.handleId}`, resourceUrl: `${MCP_APP_SCHEME}://${view.handleId}/resource.html`,
      viewToken: view.token, resourceSha256: sha(view.html), csp: structuredClone(csp), toolInput: structuredClone(view.input), toolResult: structuredClone(view.result),
      resourceHtml: view.html,
      hostContext: { theme: this.options.theme(), ...this.displayModes.context(view.display) } };
  }
  document(url: string): Response {
    try {
      const target = new URL(url), view = this.handles.get(target.hostname);
      if (target.protocol !== `${MCP_APP_SCHEME}:` || target.username || target.password || target.search || target.hash || !view || !view.token || !view.html || !view.csp || view.phase === "closed") throw new Error("Expired app view.");
      this.options.requireAccess(view.origin);
      const proxy = target.pathname === "/proxy.html" ? mcpAppProxyDocument(view.handleId, view.token) : undefined;
      if (!proxy && target.pathname !== "/resource.html") throw new Error("Unknown app resource.");
      return new Response(proxy?.html ?? view.html, { headers: { "Content-Type": "text/html; charset=utf-8", "Content-Security-Policy": proxy?.csp ?? mcpAppResourceCsp(view.csp),
        "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Permissions-Policy": "camera=(), microphone=(), geolocation=(), clipboard-write=(), usb=(), serial=(), hid=()" } });
    } catch { return new Response("This connected-app view expired or lost access. Reopen through an explicit permitted action.", { status: 410, headers: { "Content-Type": "text/plain", "Content-Security-Policy": "default-src 'none'", "Cache-Control": "no-store" } }); }
  }
  allowNetwork(handleId: string, url: string, resourceType: string): boolean {
    const view = this.handles.get(handleId);
    if (!view || !view.csp || !view.token || view.phase === "closed") return false;
    try {
      this.options.requireAccess(view.origin);
      const target = new URL(url);
      if (target.username || target.password) return false;
      if (target.protocol === `${MCP_APP_SCHEME}:`) return target.hostname === handleId && ["/proxy.html", "/resource.html"].includes(target.pathname) && !target.search && !target.hash;
      if (["data:", "blob:"].includes(target.protocol)) return !["subFrame", "mainFrame"].includes(resourceType);
      if (target.username || target.password) return false;
      const allowed = ["xhr", "webSocket", "ping"].includes(resourceType) ? view.csp.connectDomains : resourceType === "subFrame" ? view.csp.frameDomains : view.csp.resourceDomains;
      return allowed.includes(target.origin);
    } catch { return false; }
  }
  async request(input: McpAppRequest): Promise<McpAppHostResult> {
    let view: AppView;
    try {
      if (!input || typeof input.viewToken !== "string" || !/^[a-f0-9]{64}$/.test(input.viewToken)) throw new Error("Invalid connected-app view token.");
      view = this.view(input.brainId, input.viewId, input.viewToken);
    }
    catch (error) { return { error: { code: -32000, message: error instanceof Error ? error.message : String(error) } }; }
    if (typeof input.id !== "string" && typeof input.id !== "number" || typeof input.id === "number" && !Number.isSafeInteger(input.id) || typeof input.method !== "string" || input.method.includes("\0")) return { error: { code: -32600, message: "Invalid app request." } };
    if (input.notification !== undefined && (input.notification !== true || !/^(?:ui\/)?notifications\//.test(input.method))) return { error: { code: -32600, message: "Invalid app notification discriminator." } };
    const key = `${input.notification ? "notification" : "rpc"}:${typeof input.id}:${input.id}`, hash = sha(JSON.stringify({ method: input.method, params: input.params })), prior = view.requests.get(key);
    if (prior) return prior.hash === hash ? prior.promise : { error: { code: -32600, message: "An app request ID was reused with different content." } };
    const controller = new AbortController(), signal = AbortSignal.any([view.controller.signal, controller.signal]);
    const promise = this.options.admit(Buffer.byteLength(JSON.stringify(input)) * 3).then(() =>
      this.dispatch(view, input.method, object(input.params), signal, input.notification === true)).then(result => ({ result }), error => ({ error: { code: error instanceof McpHostRequestError ? error.code : -32603, message: (error instanceof Error ? error.message : String(error)).slice(0, 1000) } }));
    view.requests.set(key, { hash, promise, controller }); return promise;
  }
  private async dispatch(view: AppView, method: string, params: Record<string, unknown>, signal: AbortSignal, notification: boolean): Promise<unknown> {
    signal.throwIfAborted(); this.options.requireAccess(view.origin);
    if (method === "ui/initialize") {
      if (view.phase !== "loading" || params.protocolVersion !== MCP_APP_PROTOCOL || typeof object(params.appInfo).name !== "string" || !params.appCapabilities || typeof params.appCapabilities !== "object" || Array.isArray(params.appCapabilities) || !view.display) throw new McpHostRequestError(-32602, "Invalid or duplicate MCP App initialization.");
      this.displayModes.configure(view.display, object(params.appCapabilities));
      const capabilities = this.options.serverCapabilities?.(view.origin);
      view.serverNotifications = { tools: Boolean(this.serverNotificationListeners.size && capabilities?.tools?.listChanged),
        resources: Boolean(this.serverNotificationListeners.size && capabilities?.resources?.listChanged) };
      view.phase = "initialized";
      return { protocolVersion: MCP_APP_PROTOCOL, hostInfo: { name: "Omni AGI Studio", version: "2.0.0" },
        hostCapabilities: { openLinks: {}, serverTools: { ...(view.serverNotifications.tools ? { listChanged: true } : {}) },
          serverResources: { ...(view.serverNotifications.resources ? { listChanged: true } : {}) }, ...(this.options.log ? { logging: {} } : {}), sandbox: { csp: view.csp, permissions: {} } },
        hostContext: { theme: this.options.theme(), ...this.displayModes.context(view.display),
          toolInfo: { tool: { name: view.tool.remoteName, inputSchema: view.tool.inputSchema, ...(view.tool.description ? { description: view.tool.description } : {}),
            ...(view.tool.title ? { title: view.tool.title } : {}), ...(view.tool.metadata ? { _meta: view.tool.metadata } : {}) } } } };
    }
    if (method === "ui/notifications/initialized") {
      if (view.phase !== "initialized") throw new McpHostRequestError(-32000, "The app did not finish its initialization handshake.");
      view.phase = "ready"; return {};
    }
    if (view.phase !== "ready") throw new McpHostRequestError(-32000, "The app must finish initialization before requesting effects.");
    if (method === "ping") return {};
    if (method === "notifications/cancelled") {
      const request = view.requests.get(`rpc:${typeof params.requestId}:${params.requestId}`); request?.controller.abort(); return {};
    }
    if (method === "tools/call") {
      if (typeof params.name !== "string" || params.arguments !== undefined && (!params.arguments || typeof params.arguments !== "object" || Array.isArray(params.arguments))) throw new McpHostRequestError(-32602, "Invalid app tool call.");
      const tool = this.options.tool(view.origin, params.name); this.requireCall(view.descriptor.id, view.descriptor.brainId, view.descriptor.serverId, tool.remoteName);
      return this.permitted(view, tool, object(params.arguments), randomUUID(), signal);
    }
    if (method === "resources/read") {
      if (typeof params.uri !== "string") throw new McpHostRequestError(-32602, "Invalid app resource URI.");
      const result = object(await this.permitted(view, this.options.resourceTool(view.origin), { uri: params.uri }, randomUUID(), signal));
      return result.result;
    }
    if (method === "ui/open-link") {
      const request = mcpElicitation({ mode: "url", message: "Open a link from this connected app", elicitationId: randomUUID(), url: params.url });
      if (request.mode !== "url" || !await this.options.approveLink(view.origin, request.url, signal)) throw new McpHostRequestError(-32000, "Opening this app link was declined.");
      signal.throwIfAborted(); this.options.requireAccess(view.origin); await this.options.openBrowser(request.url); return {};
    }
    if (method === "ui/request-display-mode") {
      if (notification || Object.keys(params).some(key => !["mode", "_meta"].includes(key)) || !view.display)
        throw new McpHostRequestError(-32602, "Invalid app display-mode request.");
      return { mode: await this.displayModes.request(view.display, params.mode, signal) };
    }
    if (method === "ui/notifications/size-changed") return {};
    if (method === "notifications/message" && this.options.log) {
      if (!notification) throw new McpHostRequestError(-32600, "App logging uses a notification, not a host-action request.");
      await this.options.log(view.origin, params, signal, { viewId: view.descriptor.id, privateSecrets: view.token ? [view.token] : [] });
      signal.throwIfAborted(); this.options.requireAccess(view.origin); return {};
    }
    if ((method === "ui/message" || method === "ui/update-model-context") && this.options.observe) {
      if (!view.captureSha256) throw new McpHostRequestError(-32000, "App task data requires its exact durable completed-call provenance.");
      const data = mcpAppTaskData(method, params);
      const receipt = await this.options.observe(view.origin, { message: method === "ui/message" ? "Add this App-proposed task to this instance?" : "Allow this App's current task data?",
        view: structuredClone(view.descriptor), captureSha256: view.captureSha256, dataSha256: sha(JSON.stringify(data)), data, untrustedTaskData: true, hiddenPromptUsed: false }, signal);
      signal.throwIfAborted(); this.options.requireAccess(view.origin);
      this.publishObservation({ brainId: view.descriptor.brainId, artifactId: view.descriptor.id, receipt });
      return { _meta: { "omni/appObservation": receipt } };
    }
    throw new McpHostRequestError(-32601, "This App host method is not enabled; no hidden context or external action was applied.");
  }
  close(brainId: string, viewId: string, token: string): void {
    const view = this.views.get(`${brainId}:${viewId}`);
    if (!view || typeof token !== "string" || !/^[a-f0-9]{64}$/.test(token) || view.descriptor.brainId !== brainId || view.token !== token) throw new Error("Invalid connected-app close identity.");
    view.phase = "closed"; view.controller.abort(new Error("The app view closed; effects were not replayed."));
    if (view.display) this.displayModes.retire(view.display, "The app view closed.");
    this.handles.delete(view.handleId);
    view.html = undefined; view.csp = undefined; view.requests.clear();
  }
  invalidate(brainId: string, serverId: string, reason?: "session-initialized"): void {
    const targets = reason === "session-initialized" ? [...this.views.values()] : [...this.views.values(), ...this.preparing.values()];
    for (const view of targets) if (view.descriptor.brainId === brainId && view.descriptor.serverId === serverId) {
      view.phase = "closed"; view.controller.abort(new Error("This MCP connection ended. Explicit reopening is required.")); view.html = undefined; view.csp = undefined; view.requests.clear();
      if (view.display) this.displayModes.retire(view.display, "This MCP connection ended. Explicit reopening is required.");
      this.handles.delete(view.handleId);
    }
  }
  dispose(): void { for (const view of [...this.views.values(), ...this.preparing.values()]) view.controller.abort(); this.displayModes.dispose(); this.views.clear(); this.handles.clear(); this.preparing.clear(); this.opening.clear(); this.observationListeners.clear(); this.serverNotificationListeners.clear(); }
}
