import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { BrainMapSubstrateGraph } from "../src/renderer/src/BrainMapSubstrateGraph";

it("offers exact connection selection without inventing a firing event", () => {
  const html = renderToStaticMarkup(<BrainMapSubstrateGraph
    nodes={["a", "b"].map(id => ({ id, label: id, activation: 0, activationObserved: false, importance: 0, members: [id] }))}
    edges={[{ id: "real-repeat", sourceId: "a", targetId: "b", effectiveWeight: -1, connectionStrength: 3,
      signedStrength: -3, stability: 0.2, pathways: 1, isRepeatedContact: true, parallelOrdinal: 2 }]}
    selectedId="" selectedEdgeId="real-repeat" onSelect={() => {}} onSelectEdge={() => {}} />);
  const contact = html.match(/<path\b[^>]*data-contact-id="real-repeat"[^>]*>/)?.[0];
  expect(contact).toContain('role="button"'); expect(contact).toContain('tabindex="0"');
  expect(contact).toContain('aria-pressed="true"'); expect(contact).toContain('data-repeated="true"');
  expect(contact).toContain('data-firing="unobserved"'); expect(contact).toContain("Ternary sign: -1");
});
