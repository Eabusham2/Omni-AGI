import React from "react";
import { readFile } from "node:fs/promises";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { BrainMapSubstrateGraph } from "../src/renderer/src/BrainMapSubstrateGraph";

it("keeps map context accessible without a fixed footer and distinguishes measured spikes", async () => {
  const html = renderToStaticMarkup(<BrainMapSubstrateGraph nodes={[
    { id: "channel", label: "Router channel", region: "router", importance: 0, members: ["channel"],
      activation: 0, activationObserved: false, firingEvent: { source: "recorded-spike-event", observedAt: "2026-10-15T19:00:00Z", count: 1 } }
  ]} edges={[]} selectedId="" onSelect={() => {}} />);
  const description = html.match(/<desc>([\s\S]*?)<\/desc>/)?.[1];
  expect(description).toContain("1 inspected nodes and 0 visible recorded pathways");
  expect(description).toContain("Positions are a visualization layout, not anatomy");
  expect(description).toContain("Activation is not a recorded spike");
  expect(html).not.toContain('class="brain-map-graph__scope"');
  expect(html).toMatch(/data-firing="observed"[\s\S]*?<circle r="[^"]+" fill="var\(--graph-node-ink\)" opacity="1"/);
  expect(html).toContain("Recorded spikes: 1");
  const css = await readFile("src/renderer/src/brainMapSubstrateGraph.css", "utf8");
  expect(css).not.toContain("brain-map-graph__scope");
  expect(css).not.toContain("font-size:10px");
});

it("retains real missing-activity state and 3D rotation instructions in the graph description", () => {
  const html = renderToStaticMarkup(<BrainMapSubstrateGraph nodes={[
    { id: "quiet", label: "Unobserved region", region: "fixture", importance: 0, members: ["quiet"],
      activation: 0, activationObserved: false }
  ]} edges={[]} selectedId="" onSelect={() => {}} initialDimension="3d" />);
  expect(html).toContain('data-dimension="3d"');
  expect(html).toContain('data-activation="unobserved" data-firing="unobserved"');
  expect(html).toContain("Measured activation: unobserved");
  expect(html).toContain("Spike events: unobserved");
  expect(html).toContain("Drag or use arrow keys to rotate.</desc>");
  expect(html).toContain('aria-label="Rotate brain graph left"');
  expect(html).not.toContain("brain-map-graph__scope");
});
