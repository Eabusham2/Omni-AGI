import type { ActionEvent } from "../../shared/types";
import {
  HTML_ARTIFACT_MESSAGE_FORMAT,
  HTML_ARTIFACT_RESULT_FORMAT,
  type HtmlArtifactLaunch,
  type HtmlArtifactOpenRequest,
  type HtmlArtifactToolReceipt,
  type HtmlArtifactToolRequest
} from "../../shared/htmlArtifacts";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const HANDLE = /^[a-f0-9]{32}$/;
const NONCE = /^[a-f0-9]{64}$/;
const SHA256 = /^[a-f0-9]{64}$/;
const TOOL_ID = /^[a-z][a-z0-9.-]{1,255}$/;
const ACTION_ID = /^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$/;
const MAX_REQUEST_BYTES = 1024 * 1024;

function object(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
}

export function chatHtmlArtifactRequest(event: ActionEvent): HtmlArtifactOpenRequest | null {
  if (event.state !== "complete" || event.execution?.state !== "complete" ||
      event.action.toolId !== "studio.artifact" || event.action.action !== "create" ||
      event.execution.toolId !== event.action.toolId || event.execution.action !== event.action.action) return null;
  const output = object(event.execution.output), artifact = object(output?.artifact);
  if (output?.format !== HTML_ARTIFACT_RESULT_FORMAT || output.version !== 1 ||
      artifact?.format !== "omni-chat-html-artifact-projection" || artifact.version !== 1 ||
      artifact.savedArtifact !== true || artifact.untrusted !== true ||
      typeof artifact.originalBrainId !== "string" || !SAFE_ID.test(artifact.originalBrainId) ||
      typeof artifact.artifactId !== "string" || !SAFE_ID.test(artifact.artifactId) ||
      typeof artifact.sourceActionId !== "string" || !SAFE_ID.test(artifact.sourceActionId) ||
      typeof artifact.contentSha256 !== "string" || !SHA256.test(artifact.contentSha256)) return null;
  return { brainId: event.brainId, originalBrainId: artifact.originalBrainId,
    artifactId: artifact.artifactId, sourceActionId: artifact.sourceActionId,
    contentSha256: artifact.contentSha256 };
}

export function htmlArtifactIdentity(request: HtmlArtifactOpenRequest): string {
  return [request.brainId, request.originalBrainId, request.artifactId,
    request.sourceActionId, request.contentSha256].join("\0");
}

export function isCurrentHtmlArtifactOperation(input: {
  mounted: boolean;
  currentIdentity: string;
  expectedIdentity: string;
  currentOpenEpoch: number;
  expectedOpenEpoch: number;
  currentViewId?: string;
  expectedViewId?: string;
}): boolean {
  return input.mounted && input.currentIdentity === input.expectedIdentity &&
    input.currentOpenEpoch === input.expectedOpenEpoch &&
    (input.expectedViewId === undefined || input.currentViewId === input.expectedViewId);
}

function stable(value: unknown, depth = 0): unknown {
  if (depth > 64) throw new Error("HTML artifact request is nested too deeply.");
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("HTML artifact request contains a non-finite number.");
    return value;
  }
  if (Array.isArray(value)) return value.map((item) => stable(item, depth + 1));
  const row = object(value); if (!row) throw new Error("HTML artifact request is not structured data.");
  return Object.fromEntries(Object.keys(row).sort().map((key) => {
    if (["__proto__", "prototype", "constructor"].includes(key) || key.includes("\0")) throw new Error("HTML artifact request contains an unsafe key.");
    return [key, stable(row[key], depth + 1)];
  }));
}

export function htmlArtifactRequestFromMessage(
  event: Pick<MessageEvent, "source" | "origin" | "data">,
  expectedSource: MessageEventSource | null,
  launch: HtmlArtifactLaunch
): { request: HtmlArtifactToolRequest; operandsJson: string } | null {
  const row = object(event.data);
  if (!expectedSource || event.source !== expectedSource || event.origin !== "null" || !row ||
      Object.keys(row).sort().join(",") !== "action,arguments,channelNonce,format,requestId,toolId,version,viewId" ||
      row.format !== HTML_ARTIFACT_MESSAGE_FORMAT || row.version !== 1 || row.viewId !== launch.viewId ||
      row.channelNonce !== launch.channelNonce || typeof row.requestId !== "string" || !SAFE_ID.test(row.requestId) ||
      typeof row.toolId !== "string" || !TOOL_ID.test(row.toolId) || typeof row.action !== "string" ||
      !ACTION_ID.test(row.action) || !object(row.arguments)) return null;
  let operandsJson: string;
  try { operandsJson = JSON.stringify(stable({ toolId: row.toolId, action: row.action, arguments: row.arguments })); }
  catch { return null; }
  if (new TextEncoder().encode(operandsJson).byteLength > MAX_REQUEST_BYTES) return null;
  return { request: { format: HTML_ARTIFACT_MESSAGE_FORMAT, version: 1, viewId: launch.viewId,
    channelNonce: launch.channelNonce, requestId: row.requestId, toolId: row.toolId,
    action: row.action, arguments: structuredClone(row.arguments as Record<string, unknown>) }, operandsJson };
}

export function htmlArtifactHostReady(launch: HtmlArtifactLaunch): Record<string, unknown> {
  return { format: "omni-html-artifact-host", version: 1, viewId: launch.viewId,
    channelNonce: launch.channelNonce, kind: "ready" };
}

export function htmlArtifactResponse(launch: HtmlArtifactLaunch, receipt: HtmlArtifactToolReceipt): Record<string, unknown> {
  const { review: _review, reviewId: _reviewId, ...child } = receipt;
  return { format: "omni-html-artifact-response", version: 1, viewId: launch.viewId,
    channelNonce: launch.channelNonce, requestId: receipt.requestId, receipt: child };
}

export function htmlArtifactError(launch: HtmlArtifactLaunch, requestId: string, error: unknown): Record<string, unknown> {
  return { format: "omni-html-artifact-response", version: 1, viewId: launch.viewId,
    channelNonce: launch.channelNonce, requestId,
    error: (error instanceof Error ? error.message : String(error)).slice(0, 1_000) };
}

export function htmlArtifactLaunchIdentity(value: Pick<HtmlArtifactLaunch, "viewId" | "channelNonce">): string {
  return HANDLE.test(value.viewId) && NONCE.test(value.channelNonce) ? `${value.viewId}:${value.channelNonce}` : "";
}

export function nextHtmlArtifactReviews(current: readonly HtmlArtifactToolReceipt[], receipt: HtmlArtifactToolReceipt): HtmlArtifactToolReceipt[] {
  const next = new Map(current.map((item) => [item.requestId, item]));
  if (receipt.state === "approval-required" && receipt.review && receipt.reviewId) next.set(receipt.requestId, receipt);
  else next.delete(receipt.requestId);
  return [...next.values()];
}
