import { createHash } from "node:crypto";
import { MCP_APP_SCHEME, type McpAppCsp } from "../shared/mcpApps";

export const MCP_APP_SCHEME_PRIVILEGES = { standard: true, secure: true, supportFetchAPI: false, corsEnabled: false };
const escape = (value: string): string => JSON.stringify(value).replace(/</g, "\\u003c").replace(/>/g, "\\u003e");

/** Trusted, capability-free relay. The inner app never sees a Studio preload. */
export function mcpAppProxyDocument(viewId: string, token: string): { html: string; csp: string } {
  const resourceUrl = `${MCP_APP_SCHEME}://${viewId}/resource.html`;
  const script = `"use strict";const token=${escape(token)},resourceUrl=${escape(resourceUrl)};let app,hostOrigin,loaded=false;
const send=message=>parent.postMessage({omniMcpApp:token,message},hostOrigin||"*");
addEventListener("message",event=>{const data=event.data;
 if(event.source===parent){if(!data||data.omniMcpApp!==token||!data.message)return;
  if(hostOrigin!==undefined&&event.origin!==hostOrigin)return;hostOrigin=event.origin;
  const message=data.message;if(message.jsonrpc!=="2.0")return;
  if(message.method==="ui/notifications/sandbox-proxy-ping"){if(!loaded)send({jsonrpc:"2.0",method:"ui/notifications/sandbox-proxy-ready"});return;}
  if(message.method==="ui/notifications/sandbox-resource-ready"){
   if(loaded||message.params?.resourceUrl!==resourceUrl||typeof message.params?.html!=="string")return;
   loaded=true;app=document.createElement("iframe");app.title="Connected app";app.sandbox="allow-scripts";
   app.referrerPolicy="no-referrer";app.allow="camera 'none'; microphone 'none'; geolocation 'none'; clipboard-write 'none'";
   app.src=resourceUrl;document.body.appendChild(app);return;
  }
  if(message.method?.startsWith("ui/notifications/sandbox-"))return;
  app?.contentWindow?.postMessage(message,"*");return;
 }
 if(!app||event.source!==app.contentWindow||event.origin!=="null"||!data||data.jsonrpc!=="2.0")return;
 if(typeof data.method==="string"&&data.method.startsWith("ui/notifications/sandbox-"))return;
 send(data);
});send({jsonrpc:"2.0",method:"ui/notifications/sandbox-proxy-ready"});`;
  const hash = createHash("sha256").update(script).digest("base64");
  return { html: `<!doctype html><html><head><meta charset="utf-8"><title>Connected app sandbox</title><style>html,body,iframe{margin:0;width:100%;height:100%;border:0;display:block}body{overflow:hidden}</style></head><body><script>${script}</script></body></html>`,
    csp: `default-src 'none'; script-src 'sha256-${hash}'; style-src 'unsafe-inline'; frame-src ${MCP_APP_SCHEME}://${viewId}; object-src 'none'; base-uri 'none'; form-action 'none'` };
}
export function mcpAppResourceCsp(csp: McpAppCsp): string {
  const origins = (values: string[]): string => values.length ? values.join(" ") : "'none'";
  return [`default-src 'none'`, `script-src 'unsafe-inline' ${csp.resourceDomains.join(" ")}`,
    `style-src 'unsafe-inline' ${csp.resourceDomains.join(" ")}`, `img-src data: blob: ${csp.resourceDomains.join(" ")}`,
    `media-src data: blob: ${csp.resourceDomains.join(" ")}`, `font-src data: ${csp.resourceDomains.join(" ")}`,
    `connect-src ${origins(csp.connectDomains)}`, `frame-src ${origins(csp.frameDomains)}`, `base-uri ${origins(csp.baseUriDomains)}`,
    `object-src 'none'`, `form-action 'none'`, `sandbox allow-scripts`].join("; ");
}

export interface McpAppFrame { url: string; parent?: McpAppFrame | null; }
/** Includes descendants even when the sandboxed child's browser origin is opaque. */
export function mcpAppFrameOwner(frame?: McpAppFrame | null): string | undefined {
  const visited = new Set<McpAppFrame>();
  while (frame && !visited.has(frame)) {
    const current = frame; visited.add(current);
    try { const url = new URL(current.url); if (url.protocol === `${MCP_APP_SCHEME}:`) return url.hostname; frame = current.parent; }
    catch { try { frame = current.parent; } catch { return undefined; } }
  }
  return undefined;
}

export function mcpAppRequestHeaders(headers: Record<string, string>): Record<string, string> {
  return Object.fromEntries(Object.entries(headers).filter(([key]) => !["cookie", "authorization", "proxy-authorization", "referer"].includes(key.toLowerCase())));
}
export function mcpAppResponseHeaders(headers: Record<string, string[]>): Record<string, string[]> {
  return Object.fromEntries(Object.entries(headers).filter(([key]) => !["set-cookie", "set-cookie2", "clear-site-data"].includes(key.toLowerCase())));
}

/** Subframes have no authority to navigate themselves or the trusted relay. */
export function allowMcpAppFrameNavigation(input: { url: string; frame?: McpAppFrame | null; initiator?: McpAppFrame | null; isMainFrame: boolean }, allowNested?: (owner: string, url: string) => boolean): boolean {
  if (input.isMainFrame) return false;
  const initiator = mcpAppFrameOwner(input.initiator), current = mcpAppFrameOwner(input.frame);
  let target: URL; try { target = new URL(input.url); } catch { return false; }
  if (initiator || current) {
    // Only the trusted proxy may perform the one inner resource load. Its
    // opaque descendant cannot navigate itself back to that privileged proxy.
    if (initiator && input.initiator?.url === `${MCP_APP_SCHEME}://${initiator}/proxy.html` &&
      target.href === `${MCP_APP_SCHEME}://${initiator}/resource.html` && (!current || current === initiator)) return true;
    const owner = current ?? initiator!;
    return Boolean(input.frame?.parent && input.frame.parent.url !== `${MCP_APP_SCHEME}://${owner}/proxy.html` &&
      target.protocol !== `${MCP_APP_SCHEME}:` && allowNested?.(owner, target.href));
  }
  return target.protocol === `${MCP_APP_SCHEME}:` && target.pathname === "/proxy.html";
}
