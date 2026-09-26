import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const repository = resolve(process.cwd());
const read = (path: string): string =>
  readFileSync(resolve(repository, path), "utf8");

describe("native platform chrome", () => {
  it("integrates macOS traffic lights without removing the Windows overlay", () => {
    const main = read("src/main/index.ts");

    expect(main).toContain('process.platform === "darwin"');
    expect(main).toContain('titleBarStyle: "hiddenInset" as const');
    expect(main).toContain("trafficLightPosition: { x: 14, y: 15 }");
    expect(main).toContain('process.platform === "win32"');
    expect(main).toContain('backgroundMaterial: "mica" as const');
    expect(main).toContain("titleBarOverlay:");
  });

  it("reserves renderer chrome only where native controls overlay it", () => {
    const app = read("src/renderer/src/App.tsx");
    const css = read("src/renderer/src/styles.css");

    expect(app).toContain("document.documentElement.dataset.platform");
    expect(css).toContain(':root[data-platform="macos"] .titlebar__drag');
    expect(css).toContain(':root[data-platform="linux"] .titlebar__drag');
    expect(css).toContain("padding-left: 84px");
  });
});
