/** Inert server rendering only: no browser, host picker, worker or learning. */
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { UniversalSourcePicker } from "../src/renderer/src/UniversalSourcePicker";

it("renders the universal attachment choices without selecting or learning", () => {
  const html = renderToStaticMarkup(createElement(UniversalSourcePicker, {
    label: "Attach", onSelect: () => { throw new Error("Static rendering must not select or learn."); }
  }));
  expect(html).toContain('aria-label="Attach"');
  expect(html).toContain("Files &amp; media");
  expect(html).toContain("Folder");
  expect(html.match(/<button/g)).toHaveLength(3);
  expect(html).not.toContain("Images</button>");
});

it("keeps popup choices full-width rows above composer icon-button rules", () => {
  const css = readFileSync(new URL("../src/renderer/src/universalSourcePicker.css", import.meta.url), "utf8");
  const rule = css.match(/\.composer__bottom \.universal-source-picker__menu > button\s*\{([^}]+)\}/)?.[1];
  expect(rule).toBeDefined();
  expect(rule).toContain("display: flex;");
  expect(rule).toContain("width: 100%;");
  expect(rule).toContain("min-width: 0;");
  expect(rule).toContain("height: auto;");
  expect(rule).toContain("justify-content: flex-start;");
});
