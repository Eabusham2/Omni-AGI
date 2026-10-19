/** Inert server rendering only: no browser, host picker, worker or learning. */
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
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
