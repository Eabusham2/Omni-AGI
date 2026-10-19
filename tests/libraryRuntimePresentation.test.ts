import { expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { libraryRuntimePresentation } from "../src/renderer/src/libraryRuntimePresentation";

it("labels the unconnected interface honestly rather than claiming compute is ready", () => {
  expect(libraryRuntimePresentation(true)).toEqual({ title: "No brain connected", detail: "Preview only. Neural work is unavailable." });
  expect(libraryRuntimePresentation(true).badge).toBeUndefined();
});

it("uses the theme's destructive ink and solid surface in the Home Delete button", () => {
  const css = readFileSync(new URL("../src/renderer/src/styles.css", import.meta.url), "utf8");
  const declaration = css.match(/\.button--danger\s*\{([^}]+)\}/)?.[1];
  expect(declaration).toContain("color: var(--danger)");
  expect(declaration).toContain("var(--surface-solid)");
  expect(declaration).not.toMatch(/#[a-f0-9]{3,8}\b/i);
});

it("does not infer an online worker, hardware measurement or loaded model from the desktop bridge", () => {
  const notice = libraryRuntimePresentation(false);
  expect(notice).toEqual({ title: "Local workspace", detail: "Instances are loaded and checked separately." });
  expect(notice.badge).toBeUndefined();
  expect(JSON.stringify(notice)).not.toMatch(/online|ready|scaling enabled/i);
});
