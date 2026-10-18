import type { PluginApprovalReview } from "./plugins";

export const PLUGIN_INTERACTIVE_PANEL_SCHEME = "omni-plugin-app" as const;
export const PLUGIN_INTERACTIVE_MESSAGE_FORMAT = "omni-plugin-panel-message" as const;
export const PLUGIN_INTERACTIVE_MESSAGE_VERSION = 1 as const;

const HANDLE = /^[a-f0-9]{32}$/;

export interface PluginInteractivePanelLaunch {
  format: "omni-plugin-panel-launch";
  version: 1;
  viewId: string;
  channelNonce: string;
  brainId: string;
  pluginId: string;
  contentSha256: string;
  catalogRevision: number;
  instanceRevision: number;
  panelId: string;
  title: string;
  url: string;
}

export interface PluginInteractiveToolRequest {
  format: typeof PLUGIN_INTERACTIVE_MESSAGE_FORMAT;
  version: typeof PLUGIN_INTERACTIVE_MESSAGE_VERSION;
  viewId: string;
  channelNonce: string;
  requestId: string;
  toolId: string;
  action: string;
  arguments: Record<string, unknown>;
}

export interface PluginInteractiveToolReceipt {
  format: "omni-plugin-panel-receipt";
  version: 1;
  viewId: string;
  requestId: string;
  executionId: string;
  toolId: string;
  action: string;
  state: "approval-required" | "complete" | "failed" | "denied" | "uncertain";
  startedAt: string;
  finishedAt?: string;
  dispatchStarted?: true;
  output?: unknown;
  error?: string;
  /** Host-owned review projection only. No approval token is exposed. */
  review?: PluginApprovalReview;
  reviewId?: string;
}

export interface PluginInteractiveReviewRequest {
  viewId: string;
  channelNonce: string;
  requestId: string;
  reviewId: string;
  decision: "approve" | "deny";
}

export interface PluginPanelHistoryPage {
  records: Array<{ sequence: number; viewId: string; requestId?: string; kind: string; sha256: string; value?: unknown; bytes?: number }>;
  total: number;
  nextSequence?: number;
}
export interface PluginPanelHistorySourcePage { sources: Array<{ sha256: string; owner: string; bytes: number }>; total: number; nextSha256?: string }

export function pluginInteractivePanelUrl(viewId: string): string {
  if (!HANDLE.test(viewId)) throw new Error("Invalid interactive plugin panel handle.");
  return `${PLUGIN_INTERACTIVE_PANEL_SCHEME}://${viewId}/index.html`;
}

export function parsePluginInteractivePanelUrl(value: string): string | null {
  try {
    const url = new URL(value);
    if (url.protocol !== `${PLUGIN_INTERACTIVE_PANEL_SCHEME}:` ||
        !HANDLE.test(url.hostname) || url.pathname !== "/index.html" ||
        url.username || url.password || url.port || url.search || url.hash ||
        pluginInteractivePanelUrl(url.hostname) !== url.href) return null;
    return url.hostname;
  } catch {
    return null;
  }
}
