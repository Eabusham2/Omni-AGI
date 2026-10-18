import React from "react";
import { readFile } from "node:fs/promises";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { BrainMapSubstrateGraph } from "../src/renderer/src/BrainMapSubstrateGraph";

it("places graph explanation below the SVG and keeps measured spikes visibly distinct", async () => {
  const html = renderToStaticMarkup(<BrainMapSubstrateGraph nodes={[
    { id: "channel", label: "Router channel", region: "router", importance: 0, members: ["channel"],
      activation: 0, activationObserved: false, firingEvent: { source: "recorded-spike-event", observedAt: "2026-10-15T19:00:00Z", count: 1 } }
  ]} edges={[]} selectedId="" onSelect={() => {}} />);
  expect(html.indexOf('class="brain-map-graph__scope"')).toBeGreaterThan(html.indexOf("</svg>"));
  expect(html).toMatch(/data-firing="observed"[\s\S]*?<circle r="[^"]+" fill="var\(--graph-node-ink\)" opacity="1"/);
  const css = await readFile("src/renderer/src/brainMapSubstrateGraph.css", "utf8");
  const scope = css.match(/\.brain-map-graph__scope\{([^}]+)\}/)?.[1];
  expect(scope).toContain("position:static"); expect(scope).toContain("font-size:12px");
  expect(scope).not.toMatch(/(?:bottom|top):/); expect(css).not.toContain("font-size:10px");
});
