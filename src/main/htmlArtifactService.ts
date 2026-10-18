import { createHash, randomBytes, randomUUID } from "node:crypto";
import type { BrainRepository } from "./brainRepository";
import { requireDiskWrite } from "./diskSpace";
import {
  HTML_ARTIFACT_MESSAGE_FORMAT,
  HTML_ARTIFACT_SCHEME,
  type HtmlArtifactActivationRequest,
  type HtmlArtifactApprovalReview,
  type HtmlArtifactContent,
  type HtmlArtifactLaunch,
  type HtmlArtifactOpenRequest,
  type HtmlArtifactProjection,
  type HtmlArtifactReviewRequest,
  type HtmlArtifactToolReceipt,
  type HtmlArtifactToolRequest
} from "../shared/htmlArtifacts";
import type { ToolExecutionResult, ToolInvocation } from "../shared/types";
import { canonicalHtmlArtifactValue, normalizeHtmlArtifactContent } from "./htmlArtifactFormat";
import { HtmlArtifactStore, type PersistedHtmlArtifactRow } from "./htmlArtifactStore";

export const HTML_ARTIFACT_SCHEME_PRIVILEGES = {
  standard: true,
  secure: true,
  supportFetchAPI: false,
  corsEnabled: false
};

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const NONCE = /^[a-f0-9]{64}$/;
const HANDLE = /^[a-f0-9]{32}$/;
const TOOL_ID = /^[a-z][a-z0-9.-]{1,255}$/;
const ACTION_ID = /^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$/;
const MAX_REQUEST_BYTES = 1024 * 1024;
const MAX_RESULT_BYTES = 2 * 1024 * 1024;
const MAX_REQUESTS_PER_VIEW = 256;

export interface HtmlArtifactToolExecutor {
  execute(invocation: ToolInvocation, onProgress?: undefined, requestId?: string): Promise<ToolExecutionResult>;
  cancelAndWait(brainId: string, requestId?: string): Promise<number>;
  invalidateApproval(brainId: string, token: string): boolean;
}

interface RequestRecord {
  hash: string;
  invocation: ToolInvocation;
  hostRequestId: string;
  initial: Promise<HtmlArtifactToolReceipt>;
  receipt?: HtmlArtifactToolReceipt;
  approvalToken?: string;
  reviewId?: string;
  reviewDecision?: "approve" | "deny";
  reviewPromise?: Promise<HtmlArtifactToolReceipt>;
}

interface ArtifactView {
  launch: HtmlArtifactLaunch;
  row: PersistedHtmlArtifactRow;
  content: HtmlArtifactContent;
  controller: AbortController;
  requests: Map<string, RequestRecord>;
  hostRequestIds: Set<string>;
  closed: boolean;
}

function digest(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object.`);
  return value as Record<string, unknown>;
}

function exactKeys(value: Record<string, unknown>, keys: readonly string[], label: string): void {
  const actual = Object.keys(value).sort(), expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new Error(`${label} contains unsupported fields.`);
  }
}

function stable(value: unknown, depth = 0): string {
  const normalize = (item: unknown, level: number): unknown => {
    if (level > 64) throw new Error("HTML artifact tool data is nested too deeply.");
    if (item === null || typeof item === "string" || typeof item === "boolean") return item;
    if (typeof item === "number") {
      if (!Number.isFinite(item)) throw new Error("HTML artifact tool data contains a non-finite number.");
      return item;
    }
    if (Array.isArray(item)) return item.map((child) => normalize(child, level + 1));
    const row = object(item, "HTML artifact tool data"), result: Record<string, unknown> = {};
    for (const key of Object.keys(row).sort()) {
      if (["__proto__", "prototype", "constructor"].includes(key) || key.includes("\0")) {
        throw new Error("HTML artifact tool data contains an unsafe key.");
      }
      result[key] = normalize(row[key], level + 1);
    }
    return result;
  };
  return JSON.stringify(normalize(value, depth));
}

function bounded(value: unknown, maximum: number): unknown {
  if (value === undefined) return undefined;
  const json = stable(value);
  if (Buffer.byteLength(json) > maximum) throw new Error("HTML artifact tool output exceeds the bounded view transport.");
  return JSON.parse(json) as unknown;
}

function artifactUrl(viewId: string): string {
  if (!HANDLE.test(viewId)) throw new Error("Invalid HTML artifact view handle.");
  return `${HTML_ARTIFACT_SCHEME}://${viewId}/index.html`;
}

export function parseHtmlArtifactUrl(value: string): string | null {
  try {
    const url = new URL(value);
    if (url.protocol !== `${HTML_ARTIFACT_SCHEME}:` || !HANDLE.test(url.hostname) ||
        url.pathname !== "/index.html" || url.username || url.password || url.port || url.search || url.hash ||
        artifactUrl(url.hostname) !== url.href) return null;
    return url.hostname;
  } catch { return null; }
}

function escapeText(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function literal(value: unknown): string {
  return JSON.stringify(value).replaceAll("<", "\\u003c").replaceAll(">", "\\u003e")
    .replaceAll("\u2028", "\\u2028").replaceAll("\u2029", "\\u2029");
}

function runtimeScript(view: ArtifactView): string {
  const bootstrap = `"use strict";(()=>{const format=${literal(HTML_ARTIFACT_MESSAGE_FORMAT)},viewId=${literal(view.launch.viewId)},nonce=${literal(view.launch.channelNonce)},pending=new Map(),queued=[];let ready=false;
const valid=value=>typeof value==="string"&&/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(value),post=value=>{if(ready)parent.postMessage(value,"*");else queued.push(value);};
const api=Object.freeze({call(toolId,action,args={}){if(typeof toolId!=="string"||typeof action!=="string"||!args||typeof args!=="object"||Array.isArray(args))return Promise.reject(new Error("Invalid artifact tool request."));const requestId=crypto.randomUUID();return new Promise((resolve,reject)=>{pending.set(requestId,{resolve,reject});post({format,version:1,viewId,channelNonce:nonce,requestId,toolId,action,arguments:args});});}});Object.defineProperty(window,"omniArtifact",{value:api,writable:false,configurable:false});
addEventListener("message",event=>{const data=event.data;if(event.source!==parent||!data||typeof data!=="object"||data.version!==1||data.viewId!==viewId||data.channelNonce!==nonce)return;if(data.format==="omni-html-artifact-host"&&data.kind==="ready"){if(ready)return;ready=true;for(const value of queued.splice(0))parent.postMessage(value,"*");return;}if(data.format!=="omni-html-artifact-response"||!valid(data.requestId))return;const waiter=pending.get(data.requestId);if(!waiter)return;if(typeof data.error==="string"){pending.delete(data.requestId);waiter.reject(new Error(data.error));return;}const receipt=data.receipt;if(!receipt||receipt.viewId!==viewId||receipt.requestId!==data.requestId)return;if(receipt.state==="approval-required"){dispatchEvent(new CustomEvent("omni-artifact-review-required",{detail:Object.freeze({requestId:data.requestId})}));return;}pending.delete(data.requestId);if(receipt.state==="complete")waiter.resolve(receipt.output);else waiter.reject(new Error(typeof receipt.error==="string"?receipt.error:"The artifact host action did not complete."));});parent.postMessage({format:"omni-html-artifact-ready",version:1,viewId,channelNonce:nonce},"*");})();`;
  return `${bootstrap}\n(()=>{"use strict";\n${view.content.script ?? ""}\n})();`;
}

function documentFor(view: ArtifactView): { html: string; csp: string } {
  const css = (view.content.css ?? "").replaceAll("<", "\\3c ");
  if (!view.launch.scriptEnabled) {
    const csp = "default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; connect-src 'none'; img-src 'none'; media-src 'none'; font-src 'none'; frame-src 'none'; worker-src 'none'; object-src 'none'; form-action 'none'; navigate-to 'none'; base-uri 'none'; sandbox";
    return { csp, html: `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="${csp}"><meta name="referrer" content="no-referrer"><title>${escapeText(view.content.title)}</title><style>${css}</style></head><body>${view.content.html}</body></html>` };
  }
  const script = runtimeScript(view);
  if (/<\/script\b/i.test(script)) throw new Error("HTML artifact script cannot be embedded safely.");
  const hash = createHash("sha256").update(script).digest("base64");
  const csp = `default-src 'none'; script-src 'sha256-${hash}'; style-src 'unsafe-inline'; connect-src 'none'; img-src 'none'; media-src 'none'; font-src 'none'; frame-src 'none'; worker-src 'none'; object-src 'none'; form-action 'none'; navigate-to 'none'; base-uri 'none'; sandbox allow-scripts`;
  return { csp, html: `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="${csp}"><meta name="referrer" content="no-referrer"><title>${escapeText(view.content.title)}</title><style>${css}</style></head><body>${view.content.html}<script>${script}</script></body></html>` };
}

function responseHeaders(csp = "default-src 'none'"): Record<string, string> {
  return { "Content-Type": "text/html; charset=utf-8", "Content-Security-Policy": csp,
    "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), clipboard-read=(), clipboard-write=(), display-capture=(), usb=(), serial=(), hid=()" };
}

function validatedOpenRequest(value: HtmlArtifactOpenRequest): HtmlArtifactOpenRequest {
  const row = object(value, "HTML artifact open request");
  exactKeys(row, ["brainId", "originalBrainId", "artifactId", "sourceActionId", "contentSha256"], "HTML artifact open request");
  return structuredClone(value);
}

export class HtmlArtifactService {
  private readonly views = new Map<string, ArtifactView>();

  constructor(
    private readonly repository: Pick<BrainRepository, "brainDirectory">,
    private readonly tools: HtmlArtifactToolExecutor
  ) {}

  private store(brainId: string): HtmlArtifactStore {
    return new HtmlArtifactStore(this.repository.brainDirectory(brainId), brainId);
  }

  async execute(
    brainId: string,
    action: string,
    args: Record<string, unknown>,
    sourceActionId: string,
    signal: AbortSignal
  ): Promise<unknown> {
    if (action !== "create") throw new Error("Unknown HTML artifact action.");
    signal.throwIfAborted();
    const content = normalizeHtmlArtifactContent(args);
    await requireDiskWrite(this.repository.brainDirectory(brainId), {
      operationWriteBytes: Buffer.byteLength(canonicalHtmlArtifactValue(content)) + 128 * 1024
    });
    signal.throwIfAborted();
    return this.store(brainId).create(sourceActionId, args);
  }

  private async requireCurrent(view: ArtifactView): Promise<PersistedHtmlArtifactRow> {
    if (view.closed || view.controller.signal.aborted || this.views.get(view.launch.viewId) !== view) {
      throw new Error("This HTML artifact view is closed.");
    }
    const row = await this.store(view.launch.artifact.brainId).find(view.launch.artifact);
    if (view.closed || view.controller.signal.aborted || this.views.get(view.launch.viewId) !== view) {
      throw new Error("This HTML artifact view is closed.");
    }
    if (row.revision !== view.launch.artifact.revision || row.operational_state !== view.launch.artifact.operationalState) {
      throw new Error("HTML artifact authority changed; reopen its exact current revision.");
    }
    return row;
  }

  async open(request: HtmlArtifactOpenRequest): Promise<HtmlArtifactLaunch> {
    request = validatedOpenRequest(request);
    const store = this.store(request.brainId), row = await store.find(request), content = await store.content(row);
    const viewId = randomBytes(16).toString("hex"), artifact = await store.describe(request);
    const launch: HtmlArtifactLaunch = { format: "omni-chat-html-artifact-launch", version: 1,
      viewId, channelNonce: randomBytes(32).toString("hex"), artifact,
      url: artifactUrl(viewId), height: content.height,
      scriptEnabled: artifact.operationalState === "enabled" && Boolean(content.script) };
    this.views.set(viewId, { launch, row, content, controller: new AbortController(), requests: new Map(),
      hostRequestIds: new Set(), closed: false });
    return structuredClone(launch);
  }

  async activate(request: HtmlArtifactActivationRequest): Promise<HtmlArtifactProjection> {
    const source = object(request, "HTML artifact activation");
    exactKeys(source, ["brainId", "originalBrainId", "artifactId", "sourceActionId", "contentSha256", "expectedRevision", "acknowledged"], "HTML artifact activation");
    if (request.acknowledged !== true) throw new Error("Dormant HTML artifact interaction requires explicit review.");
    const base = validatedOpenRequest({ brainId: request.brainId, originalBrainId: request.originalBrainId,
      artifactId: request.artifactId, sourceActionId: request.sourceActionId, contentSha256: request.contentSha256 });
    const result = await this.store(request.brainId).activate({ ...base, expectedRevision: request.expectedRevision });
    for (const view of [...this.views.values()]) if (view.launch.artifact.brainId === request.brainId &&
      view.launch.artifact.originalBrainId === request.originalBrainId && view.launch.artifact.artifactId === request.artifactId) {
      await this.invalidate(view, "HTML artifact revision changed after explicit review.");
    }
    return result;
  }

  async document(url: string): Promise<Response> {
    const viewId = parseHtmlArtifactUrl(url), view = viewId ? this.views.get(viewId) : undefined;
    if (!view) return new Response("HTML artifact view expired.", { status: 410, headers: responseHeaders() });
    try {
      await this.requireCurrent(view);
      const document = documentFor(view);
      return new Response(document.html, { status: 200, headers: responseHeaders(document.csp) });
    } catch {
      await this.invalidate(view, "HTML artifact view expired.");
      return new Response("HTML artifact view expired.", { status: 410, headers: responseHeaders() });
    }
  }

  private input(request: HtmlArtifactToolRequest): { view: ArtifactView; argumentsJson: string; hash: string } {
    const row = object(request, "HTML artifact tool request");
    exactKeys(row, ["format", "version", "viewId", "channelNonce", "requestId", "toolId", "action", "arguments"], "HTML artifact tool request");
    if (request.format !== HTML_ARTIFACT_MESSAGE_FORMAT || request.version !== 1 || !HANDLE.test(request.viewId) ||
        !NONCE.test(request.channelNonce) || !SAFE_ID.test(request.requestId) || !TOOL_ID.test(request.toolId) ||
        !ACTION_ID.test(request.action)) throw new Error("Invalid HTML artifact tool request.");
    const view = this.views.get(request.viewId);
    if (!view || view.closed || view.launch.channelNonce !== request.channelNonce || !view.launch.scriptEnabled) {
      throw new Error("HTML artifact tool request used an expired, dormant, or foreign view.");
    }
    const argumentsJson = stable(object(request.arguments, "HTML artifact tool arguments"));
    if (Buffer.byteLength(argumentsJson) > MAX_REQUEST_BYTES) throw new Error("HTML artifact tool request is too large.");
    return { view, argumentsJson, hash: digest(stable({ toolId: request.toolId, action: request.action,
      arguments: JSON.parse(argumentsJson) })) };
  }

  private retained(record: RequestRecord): Promise<HtmlArtifactToolReceipt> {
    return record.reviewPromise ?? Promise.resolve(record.receipt).then((value) => value ?? record.initial);
  }

  async request(input: HtmlArtifactToolRequest): Promise<HtmlArtifactToolReceipt> {
    const { view, argumentsJson, hash } = this.input(input);
    const prior = view.requests.get(input.requestId);
    if (prior) {
      if (prior.hash !== hash) throw new Error("HTML artifact request id was replayed with different operands.");
      return structuredClone(await this.retained(prior));
    }
    const row = await this.requireCurrent(view);
    const concurrent = view.requests.get(input.requestId);
    if (concurrent) {
      if (concurrent.hash !== hash) throw new Error("HTML artifact request id was replayed with different operands.");
      return structuredClone(await this.retained(concurrent));
    }
    if (view.requests.size >= MAX_REQUESTS_PER_VIEW) throw new Error("Close and reopen this artifact before starting more host requests.");
    if (row.operational_state !== "enabled") throw new Error("This saved artifact's operational bridge is dormant.");
    const capability = view.content.tools.find((item) => item.toolId === input.toolId);
    if (!capability?.actions.includes(input.action)) throw new Error("This artifact did not declare that host tool action.");
    const invocation: ToolInvocation = { brainId: view.launch.artifact.brainId, toolId: input.toolId,
      action: input.action, arguments: JSON.parse(argumentsJson) as Record<string, unknown> };
    const hostRequestId = `html-artifact-${digest(`${view.launch.viewId}\0${input.requestId}`).slice(0, 64)}`;
    let resolve!: (value: HtmlArtifactToolReceipt) => void, reject!: (reason: unknown) => void;
    const initial = new Promise<HtmlArtifactToolReceipt>((yes, no) => { resolve = yes; reject = no; });
    const request: RequestRecord = { hash, invocation, hostRequestId, initial };
    view.requests.set(input.requestId, request); view.hostRequestIds.add(hostRequestId);
    void this.executeInitial(view, input.requestId, request).then(resolve, reject);
    return structuredClone(await initial);
  }

  private async executeForView(view: ArtifactView, invocation: ToolInvocation, hostRequestId: string): Promise<ToolExecutionResult> {
    const abort = (): void => { void this.tools.cancelAndWait(view.launch.artifact.brainId, hostRequestId); };
    view.controller.signal.throwIfAborted(); view.controller.signal.addEventListener("abort", abort, { once: true });
    try {
      const operation = this.tools.execute(invocation, undefined, hostRequestId);
      if (view.controller.signal.aborted) abort();
      return await operation;
    } finally { view.controller.signal.removeEventListener("abort", abort); }
  }

  private async executeInitial(view: ArtifactView, requestId: string, request: RequestRecord): Promise<HtmlArtifactToolReceipt> {
    await this.requireCurrent(view);
    const result = await this.executeForView(view, request.invocation, request.hostRequestId);
    const receipt = this.receipt(view, requestId, request, result); request.receipt = receipt; return receipt;
  }

  private receipt(view: ArtifactView, requestId: string, request: RequestRecord, result: ToolExecutionResult): HtmlArtifactToolReceipt {
    const base: HtmlArtifactToolReceipt = { format: "omni-chat-html-artifact-receipt", version: 1,
      viewId: view.launch.viewId, requestId, executionId: result.id, toolId: request.invocation.toolId,
      action: request.invocation.action, state: result.state, startedAt: result.startedAt,
      ...(result.finishedAt ? { finishedAt: result.finishedAt } : {}),
      ...(result.dispatchStarted ? { dispatchStarted: true as const } : {}),
      ...(result.error ? { error: result.error.slice(0, 4_000) } : {}) };
    if (result.state === "approval-required") {
      if (!result.approvalToken) throw new Error("HTML artifact approval has no host authority token.");
      const review: HtmlArtifactApprovalReview = { kind: "html-artifact-operation",
        artifactId: view.launch.artifact.artifactId, contentSha256: view.launch.artifact.contentSha256,
        resolved: { toolId: request.invocation.toolId, action: request.invocation.action,
          arguments: structuredClone(request.invocation.arguments) } };
      request.approvalToken = result.approvalToken; request.reviewId = randomUUID();
      return { ...base, review, reviewId: request.reviewId };
    }
    try {
      const output = bounded(result.output, MAX_RESULT_BYTES);
      return output === undefined ? base : { ...base, output };
    } catch (error) {
      return { ...base, error: `${base.error ? `${base.error}; ` : ""}The actual ${result.state} tool receipt was retained, but its output is unavailable in this bounded artifact view: ${error instanceof Error ? error.message : String(error)}` };
    }
  }

  async review(input: HtmlArtifactReviewRequest): Promise<HtmlArtifactToolReceipt> {
    const row = object(input, "HTML artifact review");
    exactKeys(row, ["viewId", "channelNonce", "requestId", "reviewId", "decision"], "HTML artifact review");
    if (!HANDLE.test(input.viewId) || !NONCE.test(input.channelNonce) || !SAFE_ID.test(input.requestId) ||
        !SAFE_ID.test(input.reviewId) || !["approve", "deny"].includes(input.decision)) throw new Error("Invalid HTML artifact review.");
    const view = this.views.get(input.viewId), request = view?.requests.get(input.requestId);
    if (!view || view.closed || view.launch.channelNonce !== input.channelNonce || !request || request.reviewId !== input.reviewId) {
      throw new Error("HTML artifact review is stale or belongs to another request.");
    }
    if (request.reviewDecision) {
      if (request.reviewDecision !== input.decision) throw new Error("HTML artifact review already resolved another way.");
      if (request.reviewPromise) return structuredClone(await request.reviewPromise);
    }
    if (!request.approvalToken || request.receipt?.state !== "approval-required") throw new Error("HTML artifact review is stale.");
    request.reviewDecision = input.decision;
    request.reviewPromise = (async () => {
      await this.requireCurrent(view);
      if (input.decision === "deny") {
        this.tools.invalidateApproval(view.launch.artifact.brainId, request.approvalToken!);
        const denied: HtmlArtifactToolReceipt = { ...request.receipt!, state: "denied",
          finishedAt: new Date().toISOString(), error: "The user declined this HTML artifact action.",
          review: undefined, reviewId: undefined };
        request.approvalToken = undefined; request.receipt = denied; return denied;
      }
      const result = await this.executeForView(view, { ...request.invocation, approvalToken: request.approvalToken }, request.hostRequestId);
      request.approvalToken = undefined;
      const receipt = this.receipt(view, input.requestId, request, result); request.receipt = receipt; return receipt;
    })();
    const resolved = await request.reviewPromise;
    if (resolved.state === "approval-required" && request.reviewId !== input.reviewId) {
      request.reviewDecision = undefined; request.reviewPromise = undefined;
    }
    return structuredClone(resolved);
  }

  async close(brainId: string, viewId: string, channelNonce: string): Promise<void> {
    if (!SAFE_ID.test(brainId) || !HANDLE.test(viewId) || !NONCE.test(channelNonce)) throw new Error("Invalid HTML artifact close identity.");
    const view = this.views.get(viewId);
    // Cleanup can race activation, revocation or application shutdown. An
    // absent view owns no authority to close; never recreate it or log a fake
    // action failure. Live handles still require exact current ownership.
    if (!view) return;
    if (view.launch.artifact.brainId !== brainId || view.launch.channelNonce !== channelNonce) {
      throw new Error("HTML artifact close used a foreign live handle.");
    }
    await this.invalidate(view, "HTML artifact view closed.");
  }

  private async invalidate(view: ArtifactView, reason: string): Promise<void> {
    if (view.closed) return;
    view.closed = true; this.views.delete(view.launch.viewId); view.controller.abort(new Error(reason));
    for (const request of view.requests.values()) if (request.approvalToken) {
      this.tools.invalidateApproval(view.launch.artifact.brainId, request.approvalToken);
    }
    await Promise.all([...view.hostRequestIds].map((id) =>
      this.tools.cancelAndWait(view.launch.artifact.brainId, id).catch(() => 0)));
  }

  async dispose(): Promise<void> {
    await Promise.all([...this.views.values()].map((view) => this.invalidate(view, "HTML artifact host stopped.")));
  }
}

export interface HtmlArtifactFrame { url: string; parent?: HtmlArtifactFrame | null }
export function htmlArtifactFrameOwner(frame?: HtmlArtifactFrame | null): string | undefined {
  const visited = new Set<HtmlArtifactFrame>();
  while (frame && !visited.has(frame)) {
    const current = frame; visited.add(current); const owner = parseHtmlArtifactUrl(current.url);
    if (owner) return owner;
    try { frame = current.parent; } catch { return undefined; }
  }
  return undefined;
}
export function allowHtmlArtifactNavigation(input: { url: string; frame?: HtmlArtifactFrame | null; initiator?: HtmlArtifactFrame | null; isMainFrame: boolean }, studio: string): boolean {
  if (input.isMainFrame) return false;
  const target = parseHtmlArtifactUrl(input.url); if (!target || htmlArtifactFrameOwner(input.initiator)) return false;
  const current = htmlArtifactFrameOwner(input.frame);
  if (current) return current === target && input.frame?.parent?.url === studio && input.initiator?.url === studio;
  return input.frame?.parent?.url === studio || input.initiator?.url === studio;
}
export function allowHtmlArtifactRequest(input: { url: string; resourceType: string; frame?: HtmlArtifactFrame | null }): boolean {
  const target = parseHtmlArtifactUrl(input.url), owner = htmlArtifactFrameOwner(input.frame);
  if (target) return input.resourceType === "subFrame" && (!owner || owner === target && !htmlArtifactFrameOwner(input.frame?.parent));
  return !owner;
}
export function htmlArtifactRequestHeaders(headers: Record<string, string>): Record<string, string> {
  return Object.fromEntries(Object.entries(headers).filter(([key]) => !["cookie", "authorization", "proxy-authorization", "referer", "origin"].includes(key.toLowerCase())));
}
export function htmlArtifactResponseHeaders(headers: Record<string, string[]>): Record<string, string[]> {
  const csp = Object.entries(headers).find(([key]) => key.toLowerCase() === "content-security-policy")?.[1]?.[0] ?? "default-src 'none'";
  const clean = Object.fromEntries(Object.entries(headers).filter(([key]) => ["content-type", "content-length", "cache-control"].includes(key.toLowerCase())));
  return { ...clean, "Content-Security-Policy": [csp], "X-Content-Type-Options": ["nosniff"],
    "Referrer-Policy": ["no-referrer"], "Permissions-Policy": ["camera=(), microphone=(), geolocation=(), clipboard-read=(), clipboard-write=(), display-capture=(), usb=(), serial=(), hid=()"] };
}
export function isHtmlArtifactUrl(value: string): boolean {
  try { return new URL(value).protocol === `${HTML_ARTIFACT_SCHEME}:`; } catch { return false; }
}
