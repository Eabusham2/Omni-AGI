import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { OmniPluginPackage, PluginCatalogEntry, PluginsApi } from "../src/shared/plugins";
import {
  PluginThemeChooser,
  applyPluginThemeTokens,
  createPluginThemeController,
  loadPluginThemeSelection,
  parsePluginThemeSelection,
  resolvePluginTheme,
  savePluginThemeSelection,
  type PluginThemeSelection,
  type PluginThemeStorage
} from "../src/renderer/src/PluginThemes";

const HASH = "a".repeat(64);
const packageValue: OmniPluginPackage = {
  format: "omni-plugin-package",
  formatVersion: 1,
  manifest: {
    id: "example.theme",
    version: "1.0.0",
    name: "Example theme",
    description: "Theme fixture",
    author: "Fixture",
    license: "MIT",
    documentation: "Fixture",
    capabilities: [], tools: [], panels: [], interactivePanels: [],
    themes: [
      { id: "violet", title: "Violet", tokens: { "--accent": "#8b5cf6", "--panel": "#111827" } },
      { id: "aqua", title: "Aqua", tokens: { "--accent": "#0891b2" } }
    ],
    trainingFormats: []
  }
};
const entry: PluginCatalogEntry = {
  pluginId: "example.theme",
  name: "Example theme",
  description: "Theme fixture",
  enabled: true,
  activeVersion: "1.0.0",
  activeContentSha256: HASH,
  revision: 1,
  updatedAt: "2026-10-13T00:00:00.000Z",
  versions: [{ version: "1.0.0", contentSha256: HASH, installedAt: "2026-10-13T00:00:00.000Z", sourceLabel: "fixture" }]
};

class MemoryStorage implements PluginThemeStorage {
  readonly values = new Map<string, string>();
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
  removeItem(key: string) { this.values.delete(key); }
}

class FakeStyle {
  readonly values = new Map<string, { value: string; priority: string }>();
  getPropertyValue(name: string) { return this.values.get(name)?.value ?? ""; }
  getPropertyPriority(name: string) { return this.values.get(name)?.priority ?? ""; }
  setProperty(name: string, value: string, priority = "") { this.values.set(name, { value, priority }); }
  removeProperty(name: string) { const value = this.getPropertyValue(name); this.values.delete(name); return value; }
}

function target(): HTMLElement {
  return { style: new FakeStyle(), dataset: {} } as unknown as HTMLElement;
}

function selection(themeId = "violet"): PluginThemeSelection {
  return { schemaVersion: 1, pluginId: "example.theme", contentSha256: HASH, themeId };
}

describe("plugin Studio themes", () => {
  it("persists only an exact immutable package selection", () => {
    const storage = new MemoryStorage();
    savePluginThemeSelection(storage, selection());
    expect(loadPluginThemeSelection(storage)).toEqual(selection());
    expect(parsePluginThemeSelection({ ...selection(), contentSha256: "short" })).toBeNull();
    expect(parsePluginThemeSelection({ ...selection(), approvalToken: "not-allowed" })).toBeNull();
  });

  it("requires the enabled active catalog hash and rejects remote-resource token values", async () => {
    const api = { list: vi.fn(async () => [entry]), package: vi.fn(async () => packageValue) };
    await expect(resolvePluginTheme(api, selection())).resolves.toMatchObject({ theme: { id: "violet" } });
    await expect(resolvePluginTheme({ ...api, list: vi.fn(async () => [{ ...entry, enabled: false }]) }, selection()))
      .rejects.toThrow("disabled");
    await expect(resolvePluginTheme({ ...api, package: vi.fn(async () => ({ ...packageValue, manifest: { ...packageValue.manifest,
      themes: [{ id: "violet", title: "Unsafe", tokens: { "--canvas": "image-set(https://example.invalid/a.png)" } }] } })) }, selection()))
      .rejects.toThrow("unsafe CSS token");
  });

  it("restores prior inline values, preserves later owners, and ignores stale asynchronous packages", async () => {
    const element = target();
    element.style.setProperty("--accent", "#base");
    const resolved = { selection: selection(), entry, theme: packageValue.manifest.themes[0]! };
    const cleanup = applyPluginThemeTokens(element, resolved);
    expect(element.style.getPropertyValue("--accent")).toBe("#8b5cf6");
    element.style.setProperty("--accent", "#external");
    cleanup();
    expect(element.style.getPropertyValue("--accent")).toBe("#external");
    expect(element.style.getPropertyValue("--panel")).toBe("");

    let releaseFirst!: (value: OmniPluginPackage) => void;
    let calls = 0;
    let enabled = true;
    const api = {
      list: vi.fn(async () => [{ ...entry, enabled }]),
      package: vi.fn(async () => {
        calls++;
        if (calls === 1) return new Promise<OmniPluginPackage>(resolve => { releaseFirst = resolve; });
        return packageValue;
      })
    };
    const controller = createPluginThemeController({ api, target: element });
    const stale = controller.reconcile(selection("violet"));
    await Promise.resolve(); await Promise.resolve();
    const current = controller.reconcile(selection("aqua"));
    await current;
    releaseFirst(packageValue);
    await stale;
    expect(element.style.getPropertyValue("--accent")).toBe("#0891b2");
    enabled = false;
    await controller.reconcile(selection("aqua"));
    expect(element.style.getPropertyValue("--accent")).toBe("#external");
    controller.dispose();
  });

  it("renders explicit accessible Apply controls without executing package code", () => {
    const html = renderToStaticMarkup(<PluginThemeChooser entry={entry} packageValue={packageValue} />);
    expect(html).toContain("Studio themes");
    expect(html).toContain("Applying a theme never runs package code");
    expect(html).toContain("Violet · Apply");
    expect(html).toContain("aria-pressed=\"false\"");
  });
});
