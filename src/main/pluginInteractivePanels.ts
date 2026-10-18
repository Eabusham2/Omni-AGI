import { createHash, randomBytes, randomUUID } from "node:crypto";
import type {
  OmniPluginPackage,
  PluginApprovalReview,
  PluginCatalogEntry,
  PluginInstanceAccess,
  PluginInteractivePanelDefinition
} from "../shared/plugins";
import { pluginToolId } from "../shared/plugins";
import {
  PLUGIN_INTERACTIVE_MESSAGE_FORMAT,
  PLUGIN_INTERACTIVE_MESSAGE_VERSION,
  PLUGIN_INTERACTIVE_PANEL_SCHEME,
  parsePluginInteractivePanelUrl,
  pluginInteractivePanelUrl,
  type PluginInteractivePanelLaunch,
  type PluginInteractiveReviewRequest,
  type PluginInteractiveToolReceipt,
  type PluginInteractiveToolRequest
} from "../shared/pluginInteractivePanels";
import type { ToolExecutionResult, ToolInvocation } from "../shared/types";
import type { PluginService } from "./pluginService";
import { PluginPanelReceiptStore, type RetainedPanelRequest } from "./pluginPanelReceiptStore";
import { openPanelReceiptSource } from "./pluginPanelReceiptPortability";

export const PLUGIN_INTERACTIVE_PANEL_SCHEME_PRIVILEGES = {
  standard: true,
  secure: true,
  supportFetchAPI: false,
  corsEnabled: false
};

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const NONCE = /^[a-f0-9]{64}$/;
const SHA256 = /^[a-f0-9]{64}$/;
const LOCAL_ID = /^[a-z][a-z0-9-]{0,63}$/;
const ACTION_ID = /^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$/;
const MAX_REQUEST_BYTES = 1024 * 1024;
const MAX_RESULT_BYTES = 2 * 1024 * 1024;

interface PanelAuthority {
  entry: PluginCatalogEntry;
  access: PluginInstanceAccess;
  packageValue: OmniPluginPackage;
  panel: PluginInteractivePanelDefinition;
}

interface PanelRequestRecord {
  hash: string;
  invocation: ToolInvocation;
  hostRequestId: string;
  initial: Promise<PluginInteractiveToolReceipt>;
  receipt?: PluginInteractiveToolReceipt;
  approvalToken?: string;
  reviewId?: string;
  reviewDecision?: "approve" | "deny";
  reviewPromise?: Promise<PluginInteractiveToolReceipt>;
  release(): void | Promise<void>;
}

interface PanelView {
  launch: PluginInteractivePanelLaunch;
  panel: PluginInteractivePanelDefinition;
  controller: AbortController;
  requests: Map<string, PanelRequestRecord>;
  hostRequestIds: Set<string>;
  closed: boolean;
  store: PluginPanelReceiptStore;
  invalidating?: Promise<void>;
}

export interface PluginPanelReceiptOptions {
  brainRoot(brainId: string): string | Promise<string>;
  checkDisk(brainId: string, bytes: number): Promise<void>;
  admitTransfer(brainId: string, bytes: number): Promise<{ release(): void | Promise<void> }>;
}

function jsonFootprint(value: unknown, depth = 0, ancestors = new Set<object>()): number {
  if (depth > 64) throw new Error("Actual panel result exceeds supported structured depth.");
  if (value === undefined) return 4;
  if (value === null) return 4;
  if (typeof value === "string") return Buffer.byteLength(value) * 6 + 2;
  if (typeof value === "number") { if (!Number.isFinite(value)) throw new Error("Actual panel result is not finite JSON."); return 32; }
  if (typeof value === "boolean") return 5;
  if (!value || typeof value !== "object" || ancestors.has(value)) throw new Error("Actual panel result is not acyclic JSON.");
  ancestors.add(value); let bytes = 2;
  try { if (Array.isArray(value)) for (const item of value) bytes += jsonFootprint(item, depth + 1, ancestors) + 1;
    else for (const key in value) if (Object.hasOwn(value, key) && (value as Record<string, unknown>)[key] !== undefined) bytes += Buffer.byteLength(key) * 6 + 4 + jsonFootprint((value as Record<string, unknown>)[key], depth + 1, ancestors);
  } finally { ancestors.delete(value); }
  if (!Number.isSafeInteger(bytes)) throw new Error("Actual panel result exceeds safe resource accounting."); return bytes;
}

export interface PluginPanelToolExecutor {
  execute(
    invocation: ToolInvocation,
    onProgress?: undefined,
    requestId?: string
  ): Promise<ToolExecutionResult>;
  cancelAndWait(brainId: string, requestId?: string): Promise<number>;
  invalidateApproval(brainId: string, token: string): boolean;
}

function digest(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function exactKeys(value: Record<string, unknown>, keys: readonly string[], label: string): void {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new Error(`${label} contains unsupported fields.`);
  }
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object.`);
  return value as Record<string, unknown>;
}

function stable(value: unknown): string {
  const encode = (item: unknown, depth = 0): unknown => {
    if (depth > 64) throw new Error("Interactive plugin data is nested too deeply.");
    if (item === null || typeof item === "string" || typeof item === "boolean") return item;
    if (typeof item === "number") {
      if (!Number.isFinite(item)) throw new Error("Interactive plugin data contains a non-finite number.");
      return item;
    }
    if (Array.isArray(item)) return item.map((nested) => encode(nested, depth + 1));
    const source = record(item, "Interactive plugin data");
    const output: Record<string, unknown> = {};
    for (const key of Object.keys(source).sort()) {
      if (["__proto__", "prototype", "constructor"].includes(key) || key.includes("\0")) {
        throw new Error("Interactive plugin data contains an unsafe key.");
      }
      output[key] = encode(source[key], depth + 1);
    }
    return output;
  };
  return JSON.stringify(encode(value));
}

function boundedJson(value: unknown, maximum: number, label: string): unknown {
  if (value === undefined) return undefined;
  const json = stable(value);
  if (Buffer.byteLength(json) > maximum) throw new Error(`${label} exceeds the bounded panel transport size.`);
  return JSON.parse(json) as unknown;
}

function escapedText(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function scriptLiteral(value: unknown): string {
  return JSON.stringify(value).replaceAll("<", "\\u003c").replaceAll(">", "\\u003e")
    .replaceAll("\u2028", "\\u2028").replaceAll("\u2029", "\\u2029");
}

function panelScript(view: PanelView): string {
  const context = {
    format: "omni-plugin-panel-context",
    version: 1,
    viewId: view.launch.viewId,
    pluginId: view.launch.pluginId,
    contentSha256: view.launch.contentSha256,
    panelId: view.launch.panelId
  };
  const bootstrap = `"use strict";(()=>{const format=${scriptLiteral(PLUGIN_INTERACTIVE_MESSAGE_FORMAT)},version=${PLUGIN_INTERACTIVE_MESSAGE_VERSION},viewId=${scriptLiteral(view.launch.viewId)},nonce=${scriptLiteral(view.launch.channelNonce)},context=Object.freeze(${scriptLiteral(context)}),pending=new Map(),queued=[];let hostReady=false;
const validId=value=>typeof value==="string"&&/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(value);
const post=message=>{if(hostReady)parent.postMessage(message,"*");else queued.push(message);};
const api=Object.freeze({context,call(toolId,action,args={}){if(typeof toolId!=="string"||typeof action!=="string"||!args||typeof args!=="object"||Array.isArray(args))return Promise.reject(new Error("Invalid plugin tool request."));const requestId=crypto.randomUUID();return new Promise((resolve,reject)=>{pending.set(requestId,{resolve,reject});post({format,version,viewId,channelNonce:nonce,requestId,toolId,action,arguments:args});});}});
Object.defineProperty(window,"omniPlugin",{value:api,writable:false,configurable:false});
addEventListener("message",event=>{const data=event.data;if(event.source!==parent||!data||typeof data!=="object"||data.version!==1||data.viewId!==viewId||data.channelNonce!==nonce)return;
if(data.format==="omni-plugin-panel-host"&&data.kind==="ready"){if(hostReady)return;hostReady=true;for(const message of queued.splice(0))parent.postMessage(message,"*");return;}
if(data.format!=="omni-plugin-panel-response"||!validId(data.requestId))return;const waiter=pending.get(data.requestId);if(!waiter)return;if(typeof data.error==="string"){pending.delete(data.requestId);waiter.reject(new Error(data.error));return;}const receipt=data.receipt;if(!receipt||receipt.viewId!==viewId||receipt.requestId!==data.requestId)return;if(receipt.state==="approval-required"){dispatchEvent(new CustomEvent("omni-plugin-review-required",{detail:Object.freeze({requestId:data.requestId})}));return;}pending.delete(data.requestId);if(receipt.state==="complete")waiter.resolve(receipt.output);else waiter.reject(new Error(typeof receipt.error==="string"?receipt.error:"The host action did not complete."));});
parent.postMessage({format:"omni-plugin-panel-ready",version,viewId,channelNonce:nonce},"*");})();`;
  return `${bootstrap}\n(()=>{"use strict";\n${view.panel.script}\n})();`;
}

function interactivePluginPanelDocument(view: PanelView): { html: string; csp: string } {
  const script = panelScript(view);
  // Package validation refuses a closing script element. Assert once more at
  // the last source-embedding boundary so neither HTML nor script text can
  // change the one CSP-authorized program.
  if (/<\/script\b/i.test(script)) throw new Error("Interactive panel script cannot be embedded safely.");
  const hash = createHash("sha256").update(script).digest("base64");
  const csp = [
    "default-src 'none'",
    `script-src 'sha256-${hash}'`,
    "style-src 'unsafe-inline'",
    "connect-src 'none'",
    "img-src 'none'",
    "media-src 'none'",
    "font-src 'none'",
    "frame-src 'none'",
    "child-src 'none'",
    "worker-src 'none'",
    "object-src 'none'",
    "form-action 'none'",
    "navigate-to 'none'",
    "base-uri 'none'",
    "sandbox allow-scripts"
  ].join("; ");
  const css = (view.panel.css ?? "").replaceAll("<", "\\3c ");
  return {
    html: `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="${csp}"><meta name="referrer" content="no-referrer"><title>${escapedText(view.panel.title)}</title><style>${css}</style></head><body>${view.panel.html}<script>${script}</script></body></html>`,
    csp
  };
}

function responseHeaders(csp = "default-src 'none'"): Record<string, string> {
  return {
    "Content-Type": "text/html; charset=utf-8",
    "Content-Security-Policy": csp,
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), clipboard-read=(), clipboard-write=(), display-capture=(), usb=(), serial=(), hid=()"
  };
}

function approvalReview(
  result: ToolExecutionResult,
  invocation: ToolInvocation,
  expectedPluginId: string,
  expectedContentSha256: string
): PluginApprovalReview {
  const output = record(result.output, "Plugin approval review") as Partial<PluginApprovalReview>;
  if (result.state !== "approval-required" || !result.approvalToken ||
      output.kind !== "plugin-nested-operation" || output.pluginId !== expectedPluginId ||
      output.contentSha256 !== expectedContentSha256 ||
      output.outer?.toolId !== invocation.toolId || output.outer.action !== invocation.action ||
      !output.resolved || typeof output.resolved.toolId !== "string" ||
      typeof output.resolved.action !== "string" || !output.resolved.arguments ||
      typeof output.resolved.arguments !== "object" || Array.isArray(output.resolved.arguments) ||
      !Array.isArray(output.permissionSources)) {
    throw new Error("ToolExecutor returned an invalid interactive plugin approval review.");
  }
  return boundedJson(output, MAX_RESULT_BYTES, "Plugin approval review") as PluginApprovalReview;
}

/**
 * Fresh per-open authority for a locally installed interactive plugin panel.
 * No handle, nonce, approval or in-flight request is portable.
 */
export class PluginInteractivePanelService {
  private readonly views = new Map<string, PanelView>();
  private readonly closing = new Map<string, { brainId: string; promise: Promise<void> }>();
  private readonly unsubscribe: () => void;

  constructor(
    private readonly plugins: PluginService,
    private readonly tools: PluginPanelToolExecutor,
    private readonly custody: PluginPanelReceiptOptions
  ) {
    this.unsubscribe = plugins.onEvent((event) => {
      if (event.type === "semantic-learning") return;
      for (const view of [...this.views.values()]) {
        if (view.launch.pluginId !== event.pluginId ||
            event.type === "instance" && view.launch.brainId !== event.brainId) continue;
        void this.invalidate(view, "Plugin access changed; reopen this interactive panel after review.").catch(() => undefined);
      }
    });
  }

  private async authority(
    brainId: string,
    pluginId: string,
    contentSha256: string,
    panelId: string
  ): Promise<PanelAuthority> {
    if (!SAFE_ID.test(brainId) || !LOCAL_ID.test(panelId) || !SHA256.test(contentSha256)) {
      throw new Error("Invalid interactive plugin panel identity.");
    }
    const entry = (await this.plugins.list()).find((candidate) => candidate.pluginId === pluginId);
    if (!entry?.enabled || entry.activeContentSha256 !== contentSha256) {
      throw new Error("This interactive plugin version is not globally enabled.");
    }
    const [packageValue, access] = await Promise.all([
      this.plugins.package(pluginId, contentSha256),
      this.plugins.instance(brainId, pluginId)
    ]);
    if (!access?.enabled || access.contentSha256 !== contentSha256) {
      throw new Error("This interactive plugin version is not enabled for this brain.");
    }
    const panel = packageValue.manifest.interactivePanels?.find((candidate) => candidate.id === panelId);
    if (!panel) throw new Error("This immutable plugin has no such interactive panel.");
    return { entry, access, packageValue, panel };
  }

  private async requireCurrent(view: PanelView): Promise<PanelAuthority> {
    if (view.closed || view.controller.signal.aborted || this.views.get(view.launch.viewId) !== view) {
      throw new Error("This interactive plugin panel handle is closed.");
    }
    const authority = await this.authority(
      view.launch.brainId,
      view.launch.pluginId,
      view.launch.contentSha256,
      view.launch.panelId
    );
    // Closing/revoking can win while catalog and policy reads are pending.
    // Recheck the exact in-memory owner after the await before returning any
    // authority derived from that earlier view.
    if (view.closed || view.controller.signal.aborted || this.views.get(view.launch.viewId) !== view) {
      throw new Error("This interactive plugin panel handle is closed.");
    }
    if (authority.entry.revision !== view.launch.catalogRevision ||
        authority.access.revision !== view.launch.instanceRevision) {
      throw new Error("Plugin access changed; reopen the interactive panel after review.");
    }
    return authority;
  }

  async open(
    brainId: string,
    pluginId: string,
    contentSha256: string,
    panelId: string
  ): Promise<PluginInteractivePanelLaunch> {
    const authority = await this.authority(brainId, pluginId, contentSha256, panelId);
    const viewId = randomBytes(16).toString("hex");
    const launch: PluginInteractivePanelLaunch = {
      format: "omni-plugin-panel-launch",
      version: 1,
      viewId,
      channelNonce: randomBytes(32).toString("hex"),
      brainId,
      pluginId: authority.entry.pluginId,
      contentSha256,
      catalogRevision: authority.entry.revision,
      instanceRevision: authority.access.revision,
      panelId,
      title: authority.panel.title,
      url: pluginInteractivePanelUrl(viewId)
    };
    const store = await PluginPanelReceiptStore.open(await this.custody.brainRoot(brainId), brainId, { checkDisk: bytes => this.custody.checkDisk(brainId, bytes) }, true);
    if (!store) throw new Error("Interactive panel replay custody is unavailable.");
    try { await store.view(launch); } catch (error) { store.close(); throw error; }
    this.views.set(viewId, {
      launch,
      panel: structuredClone(authority.panel),
      controller: new AbortController(),
      requests: new Map(),
      hostRequestIds: new Set(),
      closed: false,
      store
    });
    return structuredClone(launch);
  }

  async document(url: string): Promise<Response> {
    const viewId = parsePluginInteractivePanelUrl(url);
    const view = viewId ? this.views.get(viewId) : undefined;
    if (!view) return new Response("Interactive plugin panel expired.", { status: 410, headers: responseHeaders() });
    try {
      await this.requireCurrent(view);
      const document = interactivePluginPanelDocument(view);
      return new Response(document.html, { status: 200, headers: responseHeaders(document.csp) });
    } catch {
      await this.invalidate(view, "Interactive plugin panel access expired.");
      return new Response("Interactive plugin panel expired.", { status: 410, headers: responseHeaders() });
    }
  }

  private requireRequest(input: PluginInteractiveToolRequest): { view: PanelView; argumentsJson: string } {
    const source = record(input, "Interactive plugin request");
    exactKeys(source, ["format", "version", "viewId", "channelNonce", "requestId", "toolId", "action", "arguments"], "Interactive plugin request");
    if (input.format !== PLUGIN_INTERACTIVE_MESSAGE_FORMAT ||
        input.version !== PLUGIN_INTERACTIVE_MESSAGE_VERSION ||
        !SAFE_ID.test(input.requestId) || !LOCAL_ID.test(input.toolId) || !ACTION_ID.test(input.action) ||
        !NONCE.test(input.channelNonce)) throw new Error("Invalid interactive plugin request.");
    const view = this.views.get(input.viewId);
    if (!view || view.launch.channelNonce !== input.channelNonce || view.closed) {
      throw new Error("Interactive plugin request used an expired or foreign handle.");
    }
    const argumentsJson = stable(record(input.arguments, "Interactive plugin arguments"));
    if (Buffer.byteLength(argumentsJson) > MAX_REQUEST_BYTES) {
      throw new Error("Interactive plugin request exceeds the bounded host bridge size.");
    }
    return { view, argumentsJson };
  }

  async request(input: PluginInteractiveToolRequest): Promise<PluginInteractiveToolReceipt> {
    const { view, argumentsJson } = this.requireRequest(input);
    const lease = await this.custody.admitTransfer(view.launch.brainId, Buffer.byteLength(argumentsJson) + MAX_RESULT_BYTES + 65536); let adopted = false;
    try { return await this.requestAdmitted(input, view, argumentsJson, () => { adopted = true; }, lease.release); }
    finally { if (!adopted) await lease.release(); }
  }

  private retainedReceipt(view: PanelView, requestId: string, retained: RetainedPanelRequest): PluginInteractiveToolReceipt {
    if (retained.receipt) return retained.receipt;
    return { format: "omni-plugin-panel-receipt", version: 1, viewId: view.launch.viewId, requestId,
      executionId: retained.hostRequestId, toolId: retained.invocation.toolId, action: retained.invocation.action, state: "uncertain", startedAt: retained.createdAt,
      error: "This exact request was durably prepared/dispatched without a terminal acknowledgement. It will not repeat an attempted effect." };
  }

  private async requestAdmitted(input: PluginInteractiveToolRequest, view: PanelView, argumentsJson: string, adopt: () => void, release: () => void | Promise<void>): Promise<PluginInteractiveToolReceipt> {
    const authority = await this.requireCurrent(view);
    const hash = digest(stable({ toolId: input.toolId, action: input.action, arguments: JSON.parse(argumentsJson) }));
    const prior = view.requests.get(input.requestId);
    if (prior) {
      if (prior.hash !== hash) throw new Error("Interactive plugin request id was replayed with different operands.");
      return structuredClone(await (prior.reviewPromise ?? Promise.resolve(prior.receipt).then((value) => value ?? prior.initial)));
    }
    const retained = view.store.get(view.launch.viewId, input.requestId);
    if (retained) { if (retained.hash !== hash) throw new Error("Interactive plugin request id was replayed with different operands."); return structuredClone(this.retainedReceipt(view, input.requestId, retained)); }
    // Authority reads yield. Another request with this same child identity may
    // have reserved the slot while this one was waiting. Recheck and reserve
    // synchronously so exact retries share one operation and changed operands
    // never dispatch under a reused id.
    const concurrent = view.requests.get(input.requestId);
    if (concurrent) {
      if (concurrent.hash !== hash) {
        throw new Error("Interactive plugin request id was replayed with different operands.");
      }
      return structuredClone(await (concurrent.reviewPromise ??
        Promise.resolve(concurrent.receipt).then((value) => value ?? concurrent.initial)));
    }
    const allowed = authority.panel.tools.find((selection) => selection.toolId === input.toolId);
    if (!allowed?.actions.includes(input.action)) {
      throw new Error("This panel did not declare that plugin tool action.");
    }
    const invocation: ToolInvocation = {
      brainId: view.launch.brainId,
      toolId: pluginToolId(view.launch.pluginId, input.toolId),
      action: input.action,
      arguments: JSON.parse(argumentsJson) as Record<string, unknown>
    };
    const hostRequestId = `plugin-panel-${digest(`${view.launch.viewId}\0${input.requestId}`).slice(0, 64)}`;
    let resolveInitial!: (value: PluginInteractiveToolReceipt) => void;
    let rejectInitial!: (reason: unknown) => void;
    const initial = new Promise<PluginInteractiveToolReceipt>((resolve, reject) => {
      resolveInitial = resolve;
      rejectInitial = reject;
    });
    const request: PanelRequestRecord = { hash, invocation, hostRequestId, initial, release };
    adopt();
    view.requests.set(input.requestId, request);
    view.hostRequestIds.add(hostRequestId);
    void this.executeInitial(view, input.requestId, request).then(resolveInitial, rejectInitial).finally(() => this.releaseTerminal(view, input.requestId, request)).catch(() => undefined);
    return structuredClone(await initial);
  }

  private async executeInitial(
    view: PanelView,
    requestId: string,
    request: PanelRequestRecord
  ): Promise<PluginInteractiveToolReceipt> {
    await this.requireCurrent(view);
    const prior = await view.store.prepare(view.launch.viewId, requestId, { hash: request.hash, invocation: request.invocation, hostRequestId: request.hostRequestId }, view.controller.signal);
    if (prior) return this.retainedReceipt(view, requestId, prior);
    await view.store.status(view.launch.viewId, requestId, "dispatched", undefined, undefined, undefined, view.controller.signal); await this.requireCurrent(view);
    const result = await this.executeForView(view, request.invocation, request.hostRequestId);
    await this.recordActual(view, requestId, result);
    const receipt = this.receipt(view, requestId, request, result);
    try { await view.store.status(view.launch.viewId, requestId, receipt.state, receipt, request.reviewId); }
    catch (error) { if (request.approvalToken) { this.tools.invalidateApproval(view.launch.brainId, request.approvalToken); request.approvalToken = undefined; } throw error; }
    request.receipt = receipt;
    return receipt;
  }

  private async recordActual(view: PanelView, requestId: string, result: ToolExecutionResult): Promise<void> {
    const { approvalToken: _privateGrant, ...history } = result; let lease: { release(): void | Promise<void> } | undefined;
    try { lease = await this.custody.admitTransfer(view.launch.brainId, jsonFootprint(history)); await view.store.record(view.launch.viewId, requestId, "actual-host-result", history); }
    catch (error) { if (result.approvalToken) this.tools.invalidateApproval(view.launch.brainId, result.approvalToken); throw error; }
    finally { await lease?.release(); }
  }
  private async releaseTerminal(view: PanelView, id: string, request: PanelRequestRecord): Promise<void> {
    if (request.approvalToken && !view.closed) return;
    if (view.requests.get(id) === request) { view.requests.delete(id); view.hostRequestIds.delete(request.hostRequestId); await request.release(); }
  }

  private async executeForView(
    view: PanelView,
    invocation: ToolInvocation,
    hostRequestId: string
  ): Promise<ToolExecutionResult> {
    const abort = (): void => { void this.tools.cancelAndWait(view.launch.brainId, hostRequestId); };
    view.controller.signal.throwIfAborted();
    view.controller.signal.addEventListener("abort", abort, { once: true });
    try {
      // ToolExecutor registers its active request synchronously before its
      // first await. Repeat cancellation after registration if close/revoke
      // won the tiny check-to-call race.
      const operation = this.tools.execute(invocation, undefined, hostRequestId);
      if (view.controller.signal.aborted) abort();
      return await operation;
    } finally {
      view.controller.signal.removeEventListener("abort", abort);
    }
  }

  private receipt(
    view: PanelView,
    requestId: string,
    request: PanelRequestRecord,
    result: ToolExecutionResult
  ): PluginInteractiveToolReceipt {
    const base: PluginInteractiveToolReceipt = {
      format: "omni-plugin-panel-receipt",
      version: 1,
      viewId: view.launch.viewId,
      requestId,
      executionId: result.id,
      toolId: request.invocation.toolId,
      action: request.invocation.action,
      state: result.state,
      startedAt: result.startedAt,
      ...(result.finishedAt ? { finishedAt: result.finishedAt } : {}),
      ...(result.dispatchStarted ? { dispatchStarted: true as const } : {}),
      ...(result.error ? { error: result.error.slice(0, 4_000) } : {})
    };
    if (result.state === "approval-required") {
      if (view.closed || view.controller.signal.aborted) { if (result.approvalToken) this.tools.invalidateApproval(view.launch.brainId, result.approvalToken); return { ...base, state: "failed", error: "The panel closed before review; no late approval authority was retained." }; }
      const review = approvalReview(result, request.invocation, view.launch.pluginId, view.launch.contentSha256);
      request.approvalToken = result.approvalToken!;
      request.reviewId = randomUUID();
      return { ...base, review, reviewId: request.reviewId };
    }
    try {
      const output = boundedJson(result.output, MAX_RESULT_BYTES, "Interactive plugin tool result");
      return output === undefined ? base : { ...base, output };
    } catch (error) {
      // The ordinary ToolExecutor receipt already owns the actual outcome.
      // A panel transport failure must not relabel or repeat that effect.
      return {
        ...base,
        error: `${base.error ? `${base.error}; ` : ""}The actual ${result.state} tool receipt was retained, but its output is unavailable in this bounded panel: ${error instanceof Error ? error.message : String(error)}`
      };
    }
  }

  async review(input: PluginInteractiveReviewRequest): Promise<PluginInteractiveToolReceipt> {
    const source = record(input, "Interactive plugin review");
    const view = this.views.get(String(source.viewId)); if (!view || view.launch.channelNonce !== source.channelNonce) throw new Error("Interactive plugin review used a foreign or expired handle.");
    const lease = await this.custody.admitTransfer(view.launch.brainId, MAX_RESULT_BYTES + 65536);
    try { return await this.reviewAdmitted(input); } finally { await lease.release(); }
  }

  private async reviewAdmitted(input: PluginInteractiveReviewRequest): Promise<PluginInteractiveToolReceipt> {
    const source = record(input, "Interactive plugin review");
    exactKeys(source, ["viewId", "channelNonce", "requestId", "reviewId", "decision"], "Interactive plugin review");
    if (!SAFE_ID.test(input.requestId) || !SAFE_ID.test(input.reviewId) || !NONCE.test(input.channelNonce) ||
        !["approve", "deny"].includes(input.decision)) throw new Error("Invalid interactive plugin review.");
    const view = this.views.get(input.viewId);
    const request = view?.requests.get(input.requestId);
    if (!view || view.closed || view.launch.channelNonce !== input.channelNonce) throw new Error("Interactive plugin review is stale or belongs to another request.");
    await this.requireCurrent(view);
    if (!request) { const retained = view.store.get(input.viewId, input.requestId); if (retained?.reviewId === input.reviewId && retained.decision === input.decision && retained.receipt) return structuredClone(retained.receipt); throw new Error("Interactive plugin review is stale or already resolved differently."); }
    if (
        request.reviewId !== input.reviewId) {
      throw new Error("Interactive plugin review is stale or belongs to another request.");
    }
    if (request.reviewDecision) {
      if (request.reviewDecision !== input.decision) {
        throw new Error("Interactive plugin review was already resolved with another decision.");
      }
      if (request.reviewPromise) return structuredClone(await request.reviewPromise);
    }
    if (!request.approvalToken || request.receipt?.state !== "approval-required") {
      throw new Error("Interactive plugin review is stale or belongs to another request.");
    }
    request.reviewDecision = input.decision;
    request.reviewPromise = (async () => {
      await this.requireCurrent(view);
      if (input.decision === "deny") {
        this.tools.invalidateApproval(view.launch.brainId, request.approvalToken!);
        const denied: PluginInteractiveToolReceipt = {
          ...request.receipt!,
          state: "denied",
          finishedAt: new Date().toISOString(),
          error: "The user declined this interactive plugin action.",
          review: undefined,
          reviewId: undefined
        };
        request.approvalToken = undefined;
        await view.store.status(input.viewId, input.requestId, "denied", denied, input.reviewId, input.decision);
        request.receipt = denied;
        return denied;
      }
      await view.store.status(input.viewId, input.requestId, "review-dispatched", undefined, input.reviewId, input.decision, view.controller.signal); await this.requireCurrent(view);
      const result = await this.executeForView(
        view,
        { ...request.invocation, approvalToken: request.approvalToken },
        request.hostRequestId
      );
      request.approvalToken = undefined;
      await this.recordActual(view, input.requestId, result);
      const receipt = this.receipt(view, input.requestId, request, result);
      try { await view.store.status(input.viewId, input.requestId, receipt.state, receipt, request.reviewId ?? input.reviewId, receipt.state === "approval-required" ? undefined : input.decision); }
      catch (error) { if (request.approvalToken) { this.tools.invalidateApproval(view.launch.brainId, request.approvalToken); request.approvalToken = undefined; } throw error; }
      // A changed authority can require a fresh review. It is a new exact
      // review identity; the child still never receives the approval token.
      request.receipt = receipt;
      return receipt;
    })();
    let resolved: PluginInteractiveToolReceipt; try { resolved = await request.reviewPromise; } finally { await this.releaseTerminal(view, input.requestId, request); }
    if (resolved.state === "approval-required" && request.reviewId !== input.reviewId) {
      // ToolExecutor deliberately minted a new exact authority review (for
      // example after a policy revision). The old decision cannot approve it.
      request.reviewDecision = undefined;
      request.reviewPromise = undefined;
    }
    return structuredClone(resolved);
  }

  async close(brainId: string, viewId: string, channelNonce: string): Promise<void> {
    if (!SAFE_ID.test(brainId) || !NONCE.test(channelNonce)) throw new Error("Invalid interactive plugin close identity.");
    const view = this.views.get(viewId);
    if (!view || view.launch.brainId !== brainId || view.launch.channelNonce !== channelNonce) {
      throw new Error("Interactive plugin close used a foreign or expired handle.");
    }
    await this.invalidate(view, "Interactive plugin panel closed.");
  }

  private async invalidate(view: PanelView, reason: string): Promise<void> {
    if (view.invalidating) return view.invalidating;
    const operation = this.invalidateOwned(view, reason), record = { brainId: view.launch.brainId, promise: operation };
    this.closing.set(view.launch.viewId, record); view.invalidating = operation.finally(() => { if (this.closing.get(view.launch.viewId) === record) this.closing.delete(view.launch.viewId); }); return view.invalidating;
  }
  private async invalidateOwned(view: PanelView, reason: string): Promise<void> {
    view.closed = true;
    this.views.delete(view.launch.viewId);
    view.controller.abort(new Error(reason));
    for (const request of view.requests.values()) {
      if (request.approvalToken) { this.tools.invalidateApproval(view.launch.brainId, request.approvalToken); request.approvalToken = undefined; }
    }
    await Promise.all([...view.hostRequestIds].map((requestId) =>
      this.tools.cancelAndWait(view.launch.brainId, requestId).catch(() => 0)
    ));
    await Promise.allSettled([...view.requests.values()].flatMap(request => [request.initial, ...(request.reviewPromise ? [request.reviewPromise] : [])]));
    for (const [id, request] of view.requests) await this.releaseTerminal(view, id, request);
    try { await view.store.retire(view.launch.viewId, reason); } finally { view.store.close(); }
  }

  async closeOwner(brainId: string): Promise<void> { const active = [...this.views.values()].filter(view => view.launch.brainId === brainId).map(view => this.invalidate(view, "Instance unloaded or restored.")); await Promise.all([...active, ...[...this.closing.values()].filter(value => value.brainId === brainId).map(value => value.promise)]); }

  async history(brainId: string, page: { sourceSha256?: string; afterSequence?: number; maxBytes?: number } = {}): Promise<import("../shared/pluginInteractivePanels").PluginPanelHistoryPage> {
    const maximum = page.maxBytes ?? 262144, lease = await this.custody.admitTransfer(brainId, maximum), root = await this.custody.brainRoot(brainId);
    let store: PluginPanelReceiptStore | undefined, verify: (() => Promise<void>) | undefined;
    try { if (page.sourceSha256) ({ store, verify } = await openPanelReceiptSource(root, brainId, page.sourceSha256)); else store = await PluginPanelReceiptStore.open(root, brainId);
      return store?.history(page.afterSequence, maximum) ?? { records: [], total: 0 };
    } finally { try { await verify?.(); } finally { store?.close(); await lease.release(); } }
  }
  async historyBytes(brainId: string, sequence: number, offset: number, length: number, sourceSha256?: string): Promise<{ text: string; bytes: number; sha256: string; nextOffset?: number }> {
    const lease = await this.custody.admitTransfer(brainId, length), root = await this.custody.brainRoot(brainId); let store: PluginPanelReceiptStore | undefined, verify: (() => Promise<void>) | undefined;
    try { if (sourceSha256) ({ store, verify } = await openPanelReceiptSource(root, brainId, sourceSha256)); else store = await PluginPanelReceiptStore.open(root, brainId); if (!store) throw new Error("Panel history is unavailable."); return store.bytes(sequence, offset, length); }
    finally { try { await verify?.(); } finally { store?.close(); await lease.release(); } }
  }
  async historySources(brainId: string, page: { afterSha256?: string; limit?: number } = {}): Promise<import("../shared/pluginInteractivePanels").PluginPanelHistorySourcePage> {
    const after = page.afterSha256 ?? "", limit = page.limit ?? 64; if (after && !SHA256.test(after) || !Number.isSafeInteger(limit) || limit < 1 || limit > 1024) throw new Error("Invalid bounded panel source page.");
    const lease = await this.custody.admitTransfer(brainId, limit * 256); let store: PluginPanelReceiptStore | undefined;
    try { store = await PluginPanelReceiptStore.open(await this.custody.brainRoot(brainId), brainId); if (!store) return { sources: [], total: 0 };
      const sources = store.db.prepare("SELECT sha256,owner,bytes FROM archives WHERE sha256>? ORDER BY sha256 LIMIT ?").all(after, limit).map(row => ({ sha256: String(row.sha256), owner: String(row.owner), bytes: Number(row.bytes) })), last = sources.at(-1)?.sha256 ?? after;
      return { sources, total: Number(store.db.prepare("SELECT COUNT(*) AS n FROM archives").get()?.n), ...(store.db.prepare("SELECT 1 FROM archives WHERE sha256>? LIMIT 1").get(last) ? { nextSha256: last } : {}) };
    } finally { store?.close(); await lease.release(); }
  }

  async dispose(): Promise<void> {
    this.unsubscribe();
    const active = [...this.views.values()].map((view) => this.invalidate(view, "Plugin host stopped.")); await Promise.all([...active, ...[...this.closing.values()].map(value => value.promise)]);
  }
}

export interface PluginInteractivePanelFrame {
  url: string;
  parent?: PluginInteractivePanelFrame | null;
}

export function pluginInteractivePanelFrameOwner(
  frame?: PluginInteractivePanelFrame | null
): string | undefined {
  const visited = new Set<PluginInteractivePanelFrame>();
  while (frame && !visited.has(frame)) {
    const current = frame;
    visited.add(current);
    const owner = parsePluginInteractivePanelUrl(current.url);
    if (owner) return owner;
    try { frame = current.parent; } catch { return undefined; }
  }
  return undefined;
}

export function allowPluginInteractivePanelNavigation(input: {
  url: string;
  frame?: PluginInteractivePanelFrame | null;
  initiator?: PluginInteractivePanelFrame | null;
  isMainFrame: boolean;
}, trustedStudioUrl: string): boolean {
  if (input.isMainFrame) return false;
  const target = parsePluginInteractivePanelUrl(input.url);
  if (!target) return false;
  const current = pluginInteractivePanelFrameOwner(input.frame);
  if (pluginInteractivePanelFrameOwner(input.initiator)) return false;
  if (current) {
    return current === target && input.frame?.parent?.url === trustedStudioUrl &&
      input.initiator?.url === trustedStudioUrl;
  }
  return input.frame?.parent?.url === trustedStudioUrl || input.initiator?.url === trustedStudioUrl;
}

export function allowPluginInteractivePanelRequest(input: {
  url: string;
  resourceType: string;
  frame?: PluginInteractivePanelFrame | null;
}): boolean {
  const target = parsePluginInteractivePanelUrl(input.url);
  const owner = pluginInteractivePanelFrameOwner(input.frame);
  if (target) return input.resourceType === "subFrame" && (!owner ||
    owner === target && !pluginInteractivePanelFrameOwner(input.frame?.parent));
  return !owner;
}

export function pluginInteractivePanelRequestHeaders(headers: Record<string, string>): Record<string, string> {
  return Object.fromEntries(Object.entries(headers).filter(([key]) =>
    !["cookie", "authorization", "proxy-authorization", "referer", "origin"].includes(key.toLowerCase())
  ));
}

export function pluginInteractivePanelResponseHeaders(
  headers: Record<string, string[]>,
  csp?: string
): Record<string, string[]> {
  const declaredCsp = csp ?? Object.entries(headers)
    .find(([key]) => key.toLowerCase() === "content-security-policy")?.[1]?.[0];
  const clean = Object.fromEntries(Object.entries(headers).filter(([key]) =>
    ["content-type", "content-length", "cache-control"].includes(key.toLowerCase())
  ));
  return {
    ...clean,
    "Content-Security-Policy": [declaredCsp ?? "default-src 'none'"],
    "X-Content-Type-Options": ["nosniff"],
    "Referrer-Policy": ["no-referrer"],
    "Permissions-Policy": ["camera=(), microphone=(), geolocation=(), clipboard-read=(), clipboard-write=(), display-capture=(), usb=(), serial=(), hid=()"]
  };
}

export function isPluginInteractivePanelUrl(value: string): boolean {
  try { return new URL(value).protocol === `${PLUGIN_INTERACTIVE_PANEL_SCHEME}:`; }
  catch { return false; }
}
