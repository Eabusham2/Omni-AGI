import React from "react";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { PluginManager, PluginStaticPanel } from "../src/renderer/src/PluginManager";
import { PLUGIN_PANEL_CSP, pluginPanelUrl, staticPluginPanelDocument } from "../src/shared/pluginPanels";
import type { PluginsApi } from "../src/shared/plugins";

describe("plugin manager presentation boundary", () => {
  it("offers the maker/import surface without exposing a direct executor", () => {
    const api = {
      list: vi.fn(async () => []), package: vi.fn(), installFile: vi.fn(), installPackage: vi.fn(),
      edit: vi.fn(), activate: vi.fn(), setEnabled: vi.fn(), exportFile: vi.fn(),
      instance: vi.fn(), setInstanceAccess: vi.fn()
    } as unknown as PluginsApi;
    const html = renderToStaticMarkup(<PluginManager brainId="brain" api={api} />);
    expect(html).toContain("Plugins and custom tools");
    expect(html).toContain("Import package");
    expect(html).toContain("Plugin package JSON");
    expect("execute" in api).toBe(false);
  });

  it("keeps static plugin markup in an opaque iframe without privileges", () => {
    const source = readFileSync(resolve(process.cwd(), "src/renderer/src/PluginManager.tsx"), "utf8");
    expect(source).toContain('sandbox=""');
    expect(source).toContain("pluginPanelUrl");
    expect(source).not.toContain("srcDoc=");
    expect(source).not.toContain("dangerouslySetInnerHTML");
    expect(source).not.toContain("allow-scripts");
    expect(source).not.toContain("allow-same-origin");
    expect(source).not.toMatch(/\beval\s*\(/);
    const document = staticPluginPanelDocument("</title><script>bad()</script>", "<svg/onload=bad()>", "</style><script>bad()</script>");
    expect(PLUGIN_PANEL_CSP).toContain("default-src 'none'");
    expect(document).toContain("default-src 'none'");
    expect(document).toContain("&lt;/title&gt;&lt;script&gt;bad()&lt;/script&gt;");
    expect(document).not.toContain("</style><script>");
    const url = pluginPanelUrl({ pluginId: "example.notes", contentSha256: "a".repeat(64), panelId: "about" });
    const frame = renderToStaticMarkup(<PluginStaticPanel title="Static" pluginId="example.notes"
      contentSha256={"a".repeat(64)} panelId="about" />);
    expect(frame).toContain("sandbox=\"\"");
    expect(frame).toContain(url);
    expect(frame).toContain("referrerPolicy=\"no-referrer\"");
    expect(frame).not.toContain("allow-scripts");
  });
});
