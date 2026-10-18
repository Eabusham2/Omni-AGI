export const PLUGIN_PANEL_SCHEME = "omni-plugin-panel";
export const PLUGIN_PANEL_CSP = [
  "default-src 'none'",
  "style-src 'unsafe-inline'",
  "script-src 'none'",
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
  "sandbox"
].join("; ");

const SHA256 = /^[a-f0-9]{64}$/;
const PLUGIN_ID = /^[a-z0-9](?:[a-z0-9.-]{0,126}[a-z0-9])?$/;
const PANEL_ID = /^[a-z][a-z0-9-]{0,63}$/;

export interface PluginPanelIdentity {
  contentSha256: string;
  pluginId: string;
  panelId: string;
}

export function pluginPanelUrl(identity: PluginPanelIdentity): string {
  if (!SHA256.test(identity.contentSha256) || !PLUGIN_ID.test(identity.pluginId) ||
    !PANEL_ID.test(identity.panelId)) throw new Error("Invalid static plugin panel identity.");
  return `${PLUGIN_PANEL_SCHEME}://${identity.contentSha256}/${encodeURIComponent(identity.pluginId)}/${encodeURIComponent(identity.panelId)}`;
}

export function parsePluginPanelUrl(value: string): PluginPanelIdentity | null {
  try {
    const url = new URL(value);
    if (url.protocol !== `${PLUGIN_PANEL_SCHEME}:` || url.username || url.password || url.port ||
      url.search || url.hash) return null;
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length !== 2) return null;
    const identity = {
      contentSha256: url.hostname,
      pluginId: decodeURIComponent(parts[0]!),
      panelId: decodeURIComponent(parts[1]!)
    };
    if (!SHA256.test(identity.contentSha256) || !PLUGIN_ID.test(identity.pluginId) ||
      !PANEL_ID.test(identity.panelId) || pluginPanelUrl(identity) !== url.href) return null;
    return identity;
  } catch {
    return null;
  }
}

function escapeMarkupText(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

export function staticPluginPanelDocument(title: string, html: string, css: string): string {
  const style = css.replaceAll("<", "\\3c ");
  return `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="${PLUGIN_PANEL_CSP}"><meta name="referrer" content="no-referrer"><title>${escapeMarkupText(title)}</title><style>${style}</style></head><body>${html}</body></html>`;
}

export function pluginPanelSecurityHeaders(): Record<string, string> {
  return {
    "Content-Type": "text/html; charset=utf-8",
    "Content-Security-Policy": PLUGIN_PANEL_CSP,
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), clipboard-read=(), clipboard-write=(), display-capture=(), usb=(), serial=(), hid=()"
  };
}
