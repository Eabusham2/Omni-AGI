import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PluginCatalog } from "../src/main/pluginCatalog";
import { PluginService } from "../src/main/pluginService";
import {
  allowPluginPanelFrameNavigation,
  allowPluginPanelRequest,
  pluginPanelRequestHeaders,
  pluginPanelResponseHeaders
} from "../src/main/pluginPanelProtocol";
import {
  OMNI_PLUGIN_FORMAT,
  OMNI_PLUGIN_FORMAT_VERSION,
  type OmniPluginPackage
} from "../src/shared/plugins";
import {
  PLUGIN_PANEL_CSP,
  parsePluginPanelUrl,
  pluginPanelUrl,
  staticPluginPanelDocument
} from "../src/shared/pluginPanels";

function packageValue(): OmniPluginPackage {
  return { format: OMNI_PLUGIN_FORMAT, formatVersion: OMNI_PLUGIN_FORMAT_VERSION, manifest: {
    id: "example.panel", version: "1.0.0", name: "Panel", description: "Static panel fixture",
    author: "Fixture", license: "MIT", documentation: "# Panel", capabilities: [], tools: [],
    panels: [{ id: "status", title: "Status </title>", html: "<section><strong>Verified panel</strong></section>",
      css: "section::after { content: '</style>'; }" }], themes: [], trainingFormats: []
  } };
}

describe("dedicated static plugin panel protocol", () => {
  const roots: string[] = [];
  afterEach(async () => { await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))); });

  it("uses canonical hash/package/panel URLs only", () => {
    const identity = { contentSha256: "a".repeat(64), pluginId: "example.panel", panelId: "status" };
    const url = pluginPanelUrl(identity);
    expect(parsePluginPanelUrl(url)).toEqual(identity);
    expect(parsePluginPanelUrl(`${url}?x=1`)).toBeNull();
    expect(parsePluginPanelUrl(`${url}#x`)).toBeNull();
    expect(parsePluginPanelUrl(`omni-plugin-panel://user:pass@${"a".repeat(64)}/example.panel/status`)).toBeNull();
    expect(parsePluginPanelUrl(`omni-plugin-panel://${"a".repeat(64)}/example.panel/status/extra`)).toBeNull();
  });

  it("serves only registered immutable content with strict response and document CSP", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-plugin-panel-")); roots.push(root);
    const catalog = new PluginCatalog(join(root, "plugins"));
    const brain = { registerExternalToolSchemas: vi.fn(), unregisterExternalToolSchemas: vi.fn(),
      listToolPermissions: vi.fn(async () => []), setToolPermission: vi.fn(async () => []) };
    const service = new PluginService(catalog, brain);
    await service.initialize();
    const installed = await service.installPackage(packageValue());
    const url = pluginPanelUrl({ contentSha256: installed.contentSha256, pluginId: "example.panel", panelId: "status" });
    const response = await service.panelDocument(url);
    expect(response.status).toBe(200);
    expect(response.headers.get("content-security-policy")).toBe(PLUGIN_PANEL_CSP);
    expect(response.headers.get("permissions-policy")).toContain("camera=()");
    const html = await response.text();
    expect(html).toContain("Verified panel");
    expect(html).toContain("&lt;/title&gt;");
    expect(html).not.toContain("</style>'; }<");
    expect((await service.panelDocument(pluginPanelUrl({ contentSha256: "b".repeat(64),
      pluginId: "example.panel", panelId: "status" }))).status).toBe(404);
    expect((await service.panelDocument(pluginPanelUrl({ contentSha256: installed.contentSha256,
      pluginId: "example.panel", panelId: "missing" }))).status).toBe(404);
  });

  it("allows one trusted Studio frame load and denies plugin navigation and resource requests", () => {
    const studio = "file:///app/renderer/index.html";
    const panel = pluginPanelUrl({ contentSha256: "a".repeat(64), pluginId: "example.panel", panelId: "status" });
    const parent = { url: studio };
    expect(allowPluginPanelFrameNavigation({ url: panel, isMainFrame: false,
      frame: { url: "about:blank", parent }, initiator: parent }, studio)).toBe(true);
    expect(allowPluginPanelFrameNavigation({ url: panel, isMainFrame: false,
      frame: { url: panel, parent }, initiator: parent }, studio)).toBe(true);
    expect(allowPluginPanelFrameNavigation({ url: panel, isMainFrame: false,
      frame: { url: panel, parent }, initiator: { url: panel, parent } }, studio)).toBe(false);
    expect(allowPluginPanelFrameNavigation({ url: "https://example.test", isMainFrame: false,
      frame: { url: panel, parent }, initiator: { url: panel, parent } }, studio)).toBe(false);
    expect(allowPluginPanelFrameNavigation({ url: panel, isMainFrame: true, frame: parent }, studio)).toBe(false);

    expect(allowPluginPanelRequest({ url: panel, resourceType: "subFrame", frame: { url: "about:blank", parent } })).toBe(true);
    expect(allowPluginPanelRequest({ url: "https://example.test/leak", resourceType: "image",
      frame: { url: panel, parent } })).toBe(false);
    expect(allowPluginPanelRequest({ url: "data:image/png;base64,AAAA", resourceType: "image",
      frame: { url: panel, parent } })).toBe(false);
    expect(allowPluginPanelRequest({ url: pluginPanelUrl({ contentSha256: "b".repeat(64), pluginId: "example.panel", panelId: "status" }),
      resourceType: "subFrame", frame: { url: panel, parent } })).toBe(false);
  });

  it("strips ambient authority and replaces security headers exactly", () => {
    expect(pluginPanelRequestHeaders({ Cookie: "secret", Authorization: "Bearer secret", Referer: "https://studio",
      Origin: "file://studio", Accept: "text/html" }))
      .toEqual({ Accept: "text/html" });
    const headers = pluginPanelResponseHeaders({
      "content-type": ["text/html"],
      "content-security-policy": ["default-src *"],
      "set-cookie": ["session=secret"],
      "x-untrusted": ["value"]
    });
    expect(headers["Content-Security-Policy"]).toEqual([PLUGIN_PANEL_CSP]);
    expect(headers["set-cookie"]).toBeUndefined();
    expect(headers["x-untrusted"]).toBeUndefined();
  });

  it("keeps Studio CSP scoped to the dedicated scheme rather than about or blob", async () => {
    const source = await readFile(join(process.cwd(), "src/main/index.ts"), "utf8");
    expect(source).toContain("frame-src ${MCP_APP_SCHEME}: ${PLUGIN_PANEL_SCHEME}:");
    expect(source).not.toMatch(/frame-src[^\n]*(?:about:|blob:)/);
    expect(source).toContain("details.webContentsId !== trustedStudioWebContentsId");
    const document = staticPluginPanelDocument("safe", "<p>body</p>", "p{color:red}");
    expect(document).toContain(PLUGIN_PANEL_CSP);
  });
});
