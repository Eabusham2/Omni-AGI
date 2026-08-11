import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const root = resolve(import.meta.dirname, "..");
const read = (path: string): string => readFileSync(resolve(root, path), "utf8");

describe("responsive application shell", () => {
  it("allows the native window to reach the tested compact viewport floor", () => {
    const main = read("src/main/index.ts");
    expect(main).toMatch(/minWidth:\s*360/);
    expect(main).toMatch(/minHeight:\s*480/);
    expect(main).not.toMatch(/minWidth:\s*1040/);
    expect(main).not.toMatch(/minHeight:\s*700/);
  });

  it("keeps safe areas, dynamic viewport height, and compact inspector semantics", () => {
    const css = read("src/renderer/src/styles.css");
    const app = read("src/renderer/src/App.tsx");
    expect(css).toContain("height: 100dvh");
    expect(css).toContain("env(safe-area-inset-top, 0px)");
    expect(css).toContain("env(safe-area-inset-bottom, 0px)");
    expect(css).toContain(".workspace-body--inspector-open .cortex-panel");
    expect(css).toContain("@media (prefers-reduced-motion: reduce)");
    expect(app).toContain('aria-controls="chat-cortex-panel"');
    expect(app).toContain("aria-expanded={compactInspectorOpen}");
    expect(app).toContain('aria-label="Cortex and runtime inspector"');
  });

  it("exercises every supported reference viewport in rendered Electron", () => {
    const e2e = read("tests/e2e/responsive.spec.ts");
    for (const viewport of [
      "width: 390, height: 844",
      "width: 768, height: 1_024",
      "width: 1_024, height: 600",
      "width: 1_366, height: 768",
      "width: 1_480, height: 940",
      "width: 2_560, height: 1_080"
    ]) {
      expect(e2e).toContain(viewport);
    }
    expect(e2e).toContain("scrollWidth - document.documentElement.clientWidth");
    expect(e2e).toContain('getByLabel("Workspace")');
    expect(e2e).toContain('getByLabel("Open cortex inspector")');
  });
});
