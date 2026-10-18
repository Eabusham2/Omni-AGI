import { MCP_APP_MIME, type McpAppCsp } from "../shared/mcpApps";
import type { McpToolSummary } from "../shared/types";

const object = (value: unknown): Record<string, unknown> => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
export function mcpToolUi(tool: Pick<McpToolSummary, "metadata">): { resourceUri?: string; visibility: Array<"model" | "app"> } {
  const ui = object(tool.metadata?.ui);
  if (tool.metadata?.ui !== undefined && (!tool.metadata.ui || typeof tool.metadata.ui !== "object" || Array.isArray(tool.metadata.ui))) throw new Error("Invalid MCP App tool UI metadata.");
  if (ui.resourceUri !== undefined) {
    if (typeof ui.resourceUri !== "string" || /[\0\r\n]/.test(ui.resourceUri)) throw new Error("Invalid MCP App resource URI.");
    let uri: URL; try { uri = new URL(ui.resourceUri); } catch { throw new Error("Invalid MCP App resource URI."); }
    if (uri.protocol !== "ui:" || uri.username || uri.password) throw new Error("MCP App resource URI must use ui:// without credentials.");
  }
  if (ui.visibility !== undefined && (!Array.isArray(ui.visibility) || !ui.visibility.length ||
    ui.visibility.some(value => value !== "model" && value !== "app") || new Set(ui.visibility).size !== ui.visibility.length)) throw new Error("Invalid MCP App tool visibility.");
  return { ...(ui.resourceUri ? { resourceUri: ui.resourceUri as string } : {}), visibility: (ui.visibility ?? ["model", "app"]) as Array<"model" | "app"> };
}
export function mcpAppCsp(value: unknown): McpAppCsp {
  const csp = object(object(value).ui).csp ?? {};
  if (!csp || typeof csp !== "object" || Array.isArray(csp)) throw new Error("Invalid MCP App CSP declaration.");
  const result: McpAppCsp = { connectDomains: [], resourceDomains: [], frameDomains: [], baseUriDomains: [] };
  for (const key of Object.keys(result) as Array<keyof McpAppCsp>) {
    const values = object(csp)[key] ?? [];
    if (!Array.isArray(values) || values.some(value => typeof value !== "string")) throw new Error(`Invalid MCP App ${key}.`);
    result[key] = [...new Set(values.map(value => {
      let url: URL; try { url = new URL(value); } catch { throw new Error("MCP App domains require exact HTTPS origins; wildcard grants are not automatic."); }
      const local = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
      const secure = url.protocol === "https:" || key === "connectDomains" && url.protocol === "wss:";
      const development = local && (url.protocol === "http:" || key === "connectDomains" && url.protocol === "ws:");
      if (url.username || url.password || url.origin === "null" || url.origin !== value ||
        !secure && !development || url.hostname.includes("*")) throw new Error("MCP App domains require exact HTTPS origins (WSS for connections), except loopback development.");
      return url.origin;
    }))];
  }
  return result;
}
export function mcpAppResource(value: unknown, requestedUri: string): { html: string; metadata: Record<string, unknown>; mimeType: string } {
  const envelope = object(value);
  if (envelope.format !== "omni-mcp-client-result" || envelope.operation !== "resources/read" ||
    envelope.untrustedTaskData !== true || object(envelope.provenance).requestedUri !== requestedUri) throw new Error("MCP App resource lacks the permitted read provenance.");
  const contents = object(envelope.result).contents;
  if (!Array.isArray(contents)) throw new Error("MCP App resources/read returned no content.");
  const matches = contents.map(object).filter(item => item.uri === requestedUri && item.mimeType === MCP_APP_MIME);
  if (matches.length !== 1) throw new Error("MCP App requires one exact ui:// HTML resource with the negotiated profile MIME.");
  const item = matches[0]!;
  let html: string;
  if (typeof item.text === "string") html = item.text;
  else if (typeof item.blob === "string" && item.blob.length % 4 === 0 && /^[A-Za-z0-9+/]*={0,2}$/.test(item.blob)) {
    const bytes = Buffer.from(item.blob, "base64");
    if (bytes.toString("base64") !== item.blob) throw new Error("MCP App HTML blob is not canonical base64.");
    html = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } else throw new Error("MCP App HTML resource is neither text nor a valid UTF-8 blob.");
  if (html.includes("\0")) throw new Error("MCP App HTML contains invalid NUL text.");
  return { html, metadata: { ...object(object(envelope.result)._meta), ...object(item._meta) }, mimeType: MCP_APP_MIME };
}
