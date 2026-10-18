import {
  PLUGIN_INTERACTIVE_MESSAGE_FORMAT,
  PLUGIN_INTERACTIVE_MESSAGE_VERSION,
  type PluginInteractivePanelLaunch,
  type PluginInteractiveToolReceipt,
  type PluginInteractiveToolRequest
} from "../../shared/pluginInteractivePanels";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const LOCAL_ID = /^[a-z][a-z0-9-]{0,63}$/;
const ACTION_ID = /^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$/;
const MAX_REQUEST_BYTES = 1024 * 1024;

export function interactivePanelIdentity(value: {
  brainId: string;
  pluginId: string;
  contentSha256: string;
  panelId: string;
}): string {
  return JSON.stringify([value.brainId, value.pluginId, value.contentSha256, value.panelId]);
}

export function isCurrentInteractivePanelOperation(input: {
  mounted: boolean;
  currentIdentity: string;
  expectedIdentity: string;
  currentViewId?: string;
  expectedViewId?: string;
}): boolean {
  return input.mounted && input.currentIdentity === input.expectedIdentity &&
    (input.expectedViewId === undefined || input.currentViewId === input.expectedViewId);
}

export function nextPendingInteractivePanelReviews(
  current: readonly PluginInteractiveToolReceipt[],
  receipt: PluginInteractiveToolReceipt
): PluginInteractiveToolReceipt[] {
  const next = new Map(current.map((item) => [item.requestId, item]));
  if (receipt.state === "approval-required" && receipt.review && receipt.reviewId) {
    next.set(receipt.requestId, receipt);
  } else {
    next.delete(receipt.requestId);
  }
  return [...next.values()];
}

function plainRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function exactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const keys = [...expected].sort();
  return actual.length === keys.length && actual.every((key, index) => key === keys[index]);
}

function stable(value: unknown, depth = 0): unknown {
  if (depth > 64) throw new Error("Panel request is nested too deeply.");
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("Panel request contains a non-finite number.");
    return value;
  }
  if (Array.isArray(value)) return value.map((item) => stable(item, depth + 1));
  if (!plainRecord(value)) throw new Error("Panel request is not structured data.");
  return Object.fromEntries(Object.keys(value).sort().map((key) => {
    if (["__proto__", "prototype", "constructor"].includes(key) || key.includes("\0")) {
      throw new Error("Panel request contains an unsafe key.");
    }
    return [key, stable(value[key], depth + 1)];
  }));
}

export interface ValidatedInteractivePanelRequest {
  request: PluginInteractiveToolRequest;
  operandsJson: string;
}

/** Renderer defense in depth; main repeats identity, replay and permission checks. */
export function interactivePanelRequestFromMessage(
  event: Pick<MessageEvent, "source" | "origin" | "data">,
  expectedSource: MessageEventSource | null,
  launch: PluginInteractivePanelLaunch
): ValidatedInteractivePanelRequest | null {
  if (!expectedSource || event.source !== expectedSource || event.origin !== "null" || !plainRecord(event.data) ||
      !exactKeys(event.data, ["format", "version", "viewId", "channelNonce", "requestId", "toolId", "action", "arguments"])) return null;
  const value = event.data;
  if (value.format !== PLUGIN_INTERACTIVE_MESSAGE_FORMAT ||
      value.version !== PLUGIN_INTERACTIVE_MESSAGE_VERSION ||
      value.viewId !== launch.viewId || value.channelNonce !== launch.channelNonce ||
      typeof value.requestId !== "string" || !SAFE_ID.test(value.requestId) ||
      typeof value.toolId !== "string" || !LOCAL_ID.test(value.toolId) ||
      typeof value.action !== "string" || !ACTION_ID.test(value.action) ||
      !plainRecord(value.arguments)) return null;
  let operandsJson: string;
  try {
    operandsJson = JSON.stringify(stable({
      toolId: value.toolId,
      action: value.action,
      arguments: value.arguments
    }));
  } catch {
    return null;
  }
  if (new TextEncoder().encode(operandsJson).byteLength > MAX_REQUEST_BYTES) return null;
  return {
    request: {
      format: PLUGIN_INTERACTIVE_MESSAGE_FORMAT,
      version: PLUGIN_INTERACTIVE_MESSAGE_VERSION,
      viewId: launch.viewId,
      channelNonce: launch.channelNonce,
      requestId: value.requestId,
      toolId: value.toolId,
      action: value.action,
      arguments: structuredClone(value.arguments)
    },
    operandsJson
  };
}

export function interactivePanelHostReady(launch: PluginInteractivePanelLaunch): Record<string, unknown> {
  return {
    format: "omni-plugin-panel-host",
    version: 1,
    viewId: launch.viewId,
    channelNonce: launch.channelNonce,
    kind: "ready"
  };
}

/** Approval identities and nested review details stay in trusted Studio UI. */
export function interactivePanelResponse(
  launch: PluginInteractivePanelLaunch,
  receipt: PluginInteractiveToolReceipt
): Record<string, unknown> {
  const { review: _review, reviewId: _reviewId, ...childReceipt } = receipt;
  return {
    format: "omni-plugin-panel-response",
    version: 1,
    viewId: launch.viewId,
    channelNonce: launch.channelNonce,
    requestId: receipt.requestId,
    receipt: childReceipt
  };
}

export function interactivePanelErrorResponse(
  launch: PluginInteractivePanelLaunch,
  requestId: string,
  error: unknown
): Record<string, unknown> {
  return {
    format: "omni-plugin-panel-response",
    version: 1,
    viewId: launch.viewId,
    channelNonce: launch.channelNonce,
    requestId,
    error: (error instanceof Error ? error.message : String(error)).slice(0, 1_000)
  };
}
