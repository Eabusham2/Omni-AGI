import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { DEFAULT_APPEARANCE, type StorageLike } from "../src/renderer/src/appearance";
import {
  APPEARANCE_WORKSPACE_STORAGE_KEY,
  DEFAULT_APPEARANCE_WORKSPACE,
  WORKSPACE_ARRANGEMENTS,
  createAppearanceProfile,
  loadAppearanceWorkspace,
  parseAppearanceWorkspace,
  randomizedAppearance,
  removeAppearanceProfile,
  saveAppearanceWorkspace,
  upsertAppearanceProfile
} from "../src/renderer/src/appearanceWorkspace";

class MemoryStorage implements StorageLike {
  readonly values = new Map<string, string>();
  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

describe("appearance workspace preferences", () => {
  it("loads a quiet side-by-side default and repairs malformed storage", () => {
    const storage = new MemoryStorage();
    expect(loadAppearanceWorkspace(storage)).toEqual(DEFAULT_APPEARANCE_WORKSPACE);
    storage.values.set(APPEARANCE_WORKSPACE_STORAGE_KEY, "not json");
    expect(loadAppearanceWorkspace(storage)).toEqual(DEFAULT_APPEARANCE_WORKSPACE);
    expect(
      parseAppearanceWorkspace({
        schemaVersion: 1,
        arrangement: "floating",
        notificationsEnabled: "yes",
        profiles: [{ id: "unsafe id", name: "Bad", arrangement: "split" }]
      })
    ).toEqual(DEFAULT_APPEARANCE_WORKSPACE);
  });

  it("persists layout, notification opt-in, and validated appearance profiles", () => {
    const storage = new MemoryStorage();
    const profile = createAppearanceProfile(
      "Writing",
      { ...DEFAULT_APPEARANCE, palette: "aqua", layout: "glass" },
      "focus",
      "profile-writing",
      "2026-08-12T12:00:00.000Z"
    );
    const chosen = upsertAppearanceProfile(
      { ...DEFAULT_APPEARANCE_WORKSPACE, arrangement: "focus", notificationsEnabled: true },
      profile
    );
    saveAppearanceWorkspace(storage, chosen);
    expect(loadAppearanceWorkspace(storage)).toEqual(chosen);

    const updated = upsertAppearanceProfile(chosen, {
      ...profile,
      name: "Writing updated",
      arrangement: "stacked",
      updatedAt: "2026-08-12T12:30:00.000Z"
    });
    expect(updated.profiles).toHaveLength(1);
    expect(updated.profiles[0]).toMatchObject({
      name: "Writing updated",
      arrangement: "stacked",
      createdAt: profile.createdAt
    });
    expect(removeAppearanceProfile(updated, profile.id).profiles).toEqual([]);
  });

  it("randomizes palette, surface style, and structural layout without changing OS color choice", () => {
    const next = randomizedAppearance(DEFAULT_APPEARANCE, "split", () => 0);
    expect(next.appearance.mode).toBe("system");
    expect(
      next.appearance.palette !== DEFAULT_APPEARANCE.palette ||
      next.appearance.layout !== DEFAULT_APPEARANCE.layout ||
      next.arrangement !== "split"
    ).toBe(true);
    expect(WORKSPACE_ARRANGEMENTS).toContain(next.arrangement);
  });

  it("exposes full-width structural layouts, profile controls, notifications, and reliable rail labels", () => {
    const root = resolve(import.meta.dirname, "..");
    const app = readFileSync(resolve(root, "src/renderer/src/App.tsx"), "utf8");
    const css = readFileSync(resolve(root, "src/renderer/src/styles.css"), "utf8");
    expect(app).toContain("appearance-layout-gallery");
    expect(app).toContain("WORKSPACE_ARRANGEMENTS.map");
    expect(app).toContain("New appearance profile name");
    expect(app).toContain('role="switch"');
    expect(app).toContain('className="rail-label"');
    for (const arrangement of WORKSPACE_ARRANGEMENTS) {
      expect(css).toContain(`[data-workspace-arrangement="${arrangement}"]`);
      expect(css).toContain(`[data-arrangement-preview="${arrangement}"]`);
    }
    expect(css).toContain(".workspace-rail button:focus-visible > .rail-label");
  });
});
