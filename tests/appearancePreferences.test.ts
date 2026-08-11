import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  APPEARANCE_STORAGE_KEY,
  APPEARANCE_PACKS,
  DEFAULT_APPEARANCE,
  activeAppearancePack,
  loadAppearancePreferences,
  parseAppearancePreferences,
  preferencesForPack,
  resolveColorScheme,
  saveAppearancePreferences,
  type StorageLike
} from "../src/renderer/src/appearance";

class MemoryStorage implements StorageLike {
  readonly values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

const root = resolve(import.meta.dirname, "..");
const read = (path: string): string => readFileSync(resolve(root, path), "utf8");

describe("versioned appearance preferences", () => {
  it("defaults safely when storage is missing, corrupt, or from another schema", () => {
    const storage = new MemoryStorage();
    expect(loadAppearancePreferences(storage)).toEqual(DEFAULT_APPEARANCE);

    storage.values.set(APPEARANCE_STORAGE_KEY, "not-json");
    expect(loadAppearancePreferences(storage)).toEqual(DEFAULT_APPEARANCE);

    expect(
      parseAppearancePreferences({
        schemaVersion: 0,
        mode: "dark",
        palette: "spectrum",
        layout: "expressive"
      })
    ).toEqual(DEFAULT_APPEARANCE);
  });

  it("persists a validated v1 document and repairs invalid individual choices", () => {
    const storage = new MemoryStorage();
    const chosen = {
      schemaVersion: 1 as const,
      mode: "dark" as const,
      palette: "aqua" as const,
      layout: "glass" as const
    };
    saveAppearancePreferences(storage, chosen);
    expect(loadAppearancePreferences(storage)).toEqual(chosen);

    expect(
      parseAppearancePreferences({
        schemaVersion: 1,
        mode: "sepia",
        palette: "aqua",
        layout: "floating"
      })
    ).toEqual({ ...DEFAULT_APPEARANCE, palette: "aqua" });
  });

  it("tracks the OS in auto mode and pins explicit light or dark mode", () => {
    expect(resolveColorScheme("system", true)).toBe("dark");
    expect(resolveColorScheme("system", false)).toBe("light");
    expect(resolveColorScheme("light", true)).toBe("light");
    expect(resolveColorScheme("dark", false)).toBe("dark");
  });

  it("keeps mode independent while packs and individual palette/layout choices change", () => {
    const dark = { ...DEFAULT_APPEARANCE, mode: "dark" as const };
    const colorful = preferencesForPack(dark, "colorful");
    expect(colorful).toEqual({
      schemaVersion: 1,
      mode: "dark",
      palette: "spectrum",
      layout: "expressive"
    });
    expect(activeAppearancePack(colorful)).toBe("colorful");
    expect(activeAppearancePack({ ...colorful, layout: "classic" })).toBe("custom");
  });

  it("applies every advertised pack as its complete palette and layout pair", () => {
    for (const pack of APPEARANCE_PACKS) {
      const applied = preferencesForPack(
        { ...DEFAULT_APPEARANCE, mode: "light" },
        pack.id
      );
      expect(applied.mode).toBe("light");
      expect(applied.palette).toBe(pack.palette);
      expect(applied.layout).toBe(pack.layout);
      expect(activeAppearancePack(applied)).toBe(pack.id);
    }
  });
});

describe("appearance integration structure", () => {
  it("exposes a narrow typed native bridge and validates it in the main process", () => {
    const types = read("src/shared/types.ts");
    const preload = read("src/preload/index.ts");
    const ipc = read("src/main/ipc.ts");
    expect(types).toContain("setAppearance(request: NativeAppearanceRequest)");
    expect(preload).toContain("setAppearance: (request) => invoke(IPC.window.setAppearance, request)");
    expect(ipc).toContain("requireNativeAppearance");
    expect(ipc).toContain("nativeTheme.themeSource = request.mode");
    expect(ipc).toContain("window.setBackgroundColor(backgroundColor)");
  });

  it("provides semantic schemes, layouts, and accessibility fallbacks", () => {
    const css = read("src/renderer/src/styles.css");
    for (const selector of [
      ':root[data-color-scheme="light"]',
      ':root[data-palette="spectrum"]',
      ':root[data-layout="classic"]',
      ':root[data-layout="expressive"]',
      ':root[data-layout="glass"]',
      "@media (prefers-reduced-transparency: reduce)",
      "@media (prefers-reduced-motion: reduce)",
      "@media (prefers-contrast: more)",
      "@media (forced-colors: active)"
    ]) {
      expect(css).toContain(selector);
    }
    for (const token of [
      "--canvas:",
      "--panel:",
      "--line:",
      "--ink:",
      "--accent:",
      "--focus-ring:"
    ]) {
      expect(css).toContain(token);
    }
  });

  it("keeps every appearance choice reachable through labeled controls", () => {
    const app = read("src/renderer/src/App.tsx");
    expect(app).toContain('aria-label="Appearance settings"');
    expect(app).toContain('role="dialog"');
    expect(app).toContain("APPEARANCE_MODES.map");
    expect(app).toContain("APPEARANCE_PACKS.map");
    expect(app).toContain("APPEARANCE_PALETTES.map");
    expect(app).toContain("APPEARANCE_LAYOUTS.map");
    expect(app).toContain("aria-pressed={preferences.mode === mode}");
  });
});
