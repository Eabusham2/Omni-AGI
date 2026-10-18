import {
  PLUGIN_PANEL_CSP,
  PLUGIN_PANEL_SCHEME,
  parsePluginPanelUrl,
  type PluginPanelIdentity
} from "../shared/pluginPanels";

export const PLUGIN_PANEL_SCHEME_PRIVILEGES = {
  standard: true,
  secure: true,
  supportFetchAPI: false,
  corsEnabled: false
};

export interface PluginPanelFrame {
  url: string;
  parent?: PluginPanelFrame | null;
}

export function pluginPanelFrameIdentity(frame?: PluginPanelFrame | null): PluginPanelIdentity | undefined {
  const visited = new Set<PluginPanelFrame>();
  while (frame && !visited.has(frame)) {
    const current = frame;
    visited.add(current);
    const identity = parsePluginPanelUrl(current.url);
    if (identity) return identity;
    try { frame = current.parent; } catch { return undefined; }
  }
  return undefined;
}

export function allowPluginPanelFrameNavigation(input: {
  url: string;
  frame?: PluginPanelFrame | null;
  initiator?: PluginPanelFrame | null;
  isMainFrame: boolean;
}, trustedStudioUrl: string): boolean {
  if (input.isMainFrame) return false;
  const target = parsePluginPanelUrl(input.url);
  if (!target) return false;
  // Once a plugin panel is loaded, neither it nor any descendant may navigate
  // itself, a sibling, the top frame, or another panel—even to the same URL.
  const current = pluginPanelFrameIdentity(input.frame);
  const initiatorOwner = pluginPanelFrameIdentity(input.initiator);
  if (initiatorOwner) return false;
  const parentUrl = input.frame?.parent?.url;
  const initiatorUrl = input.initiator?.url;
  if (current) {
    return initiatorUrl === trustedStudioUrl &&
      current.contentSha256 === target.contentSha256 &&
      current.pluginId === target.pluginId && current.panelId === target.panelId;
  }
  return parentUrl === trustedStudioUrl || initiatorUrl === trustedStudioUrl;
}

export function allowPluginPanelRequest(input: {
  url: string;
  resourceType: string;
  frame?: PluginPanelFrame | null;
}): boolean {
  const target = parsePluginPanelUrl(input.url);
  const owner = pluginPanelFrameIdentity(input.frame);
  if (target) {
    // Only the canonical top document load is possible. Descendant resource
    // requests and cross-panel loads are denied independently of inner CSP.
    return input.resourceType === "subFrame" && (!owner ||
      owner.contentSha256 === target.contentSha256 && owner.pluginId === target.pluginId && owner.panelId === target.panelId);
  }
  return !owner;
}

export function pluginPanelRequestHeaders(headers: Record<string, string>): Record<string, string> {
  return Object.fromEntries(Object.entries(headers).filter(([key]) =>
    !["cookie", "authorization", "proxy-authorization", "referer", "origin"].includes(key.toLowerCase())
  ));
}

export function pluginPanelResponseHeaders(headers: Record<string, string[]>): Record<string, string[]> {
  const clean = Object.fromEntries(Object.entries(headers).filter(([key]) =>
    ["content-type", "content-length", "cache-control"].includes(key.toLowerCase())
  ));
  return {
    ...clean,
    "Content-Security-Policy": [PLUGIN_PANEL_CSP],
    "X-Content-Type-Options": ["nosniff"],
    "Referrer-Policy": ["no-referrer"],
    "Permissions-Policy": ["camera=(), microphone=(), geolocation=(), clipboard-read=(), clipboard-write=(), display-capture=(), usb=(), serial=(), hid=()"]
  };
}

export function isPluginPanelUrl(value: string): boolean {
  try { return new URL(value).protocol === `${PLUGIN_PANEL_SCHEME}:`; }
  catch { return false; }
}
