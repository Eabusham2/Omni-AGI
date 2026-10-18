import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { WholeBrainMap, wholeBrainDisplayGraph } from "../src/renderer/src/WholeBrainMap";
import { graphEdgeTitle } from "../src/renderer/src/brainMapGraphPresentation";
import type { WholeBrainGraphProjection } from "../src/shared/wholeBrainInspection";

const graph: WholeBrainGraphProjection = { brainId: "fixture", snapshot: null, aliases: [], inspectionBindings: [],
  boundaries: [], unavailable: [], semanticAxisMapping: false, tokenNeuronEquivalence: false,
  nodes: [{ id: "whole", recordId: "whole", kind: "region", label: "Whole brain", scopeId: "root", source: "committed", regionIds: [], firingObserved: false,
    data: { kind: "region", id: "whole", parentRegionId: null, label: "Whole brain", regionKind: "whole-brain", lineage: { brainId: "fixture" }, evidence: "actual-brain-owner" } },
  { id: "owner", recordId: "owner", kind: "registered-owner", label: "Shared native owner", scopeId: "root", source: "committed", regionIds: ["whole"], firingObserved: false,
    data: { kind: "owner", id: "owner", canonicalAlias: "decoder.shared", className: "InertFixture", packedFields: [], evidence: "actual-registered-object" } }],
  edges: [{ id: "membership", recordId: "membership", kind: "registered-membership", sourceId: "whole", targetId: "owner", scopeId: "root", learnedConnection: false, firingObserved: false }] };

it("keeps whole-brain identities while labeling structural edges as structure, never zero synapses", () => {
  const view = wholeBrainDisplayGraph(graph);
  expect(view.nodes.map(n => n.id)).toEqual(["whole", "owner"]);
  expect(view.edges[0]).toMatchObject({ id: "membership", sourceId: "whole", targetId: "owner", structuralRelationship: "registered-membership" });
  expect(graphEdgeTitle(view.edges[0]!)).toContain("not an observed synaptic weight");
  expect(graphEdgeTitle(view.edges[0]!)).not.toContain("Ternary sign");
  expect(view.nodes.every(node => node.activationObserved === false && node.firingEvent === undefined)).toBe(true);
});
it("retains actual ternary signs/strength and does not make up unknown stability or firing", () => {
  const view = wholeBrainDisplayGraph({ ...graph, edges: [{ id: "contact", recordId: "actual-contact", kind: "tensor-contact", sourceId: "whole", targetId: "owner",
    scopeId: "root", learnedConnection: true, firingObserved: false, effectiveWeight: -1, strength: 4 }] });
  expect(view.edges[0]).toMatchObject({ id: "contact", effectiveWeight: -1, connectionStrength: 4, effectiveContribution: -4 });
  expect(view.edges[0]?.structuralRelationship).toBeUndefined();
  expect(graphEdgeTitle(view.edges[0]!)).toContain("Stability: unreported");
  expect(view.edges[0]?.firingEvent).toBeUndefined();
});
it("does not substitute a router/demo map or Load a brain when whole inspection is unavailable", () => {
  const html = renderToStaticMarkup(<WholeBrainMap brainId="fixture" onBack={() => {}} />);
  expect(html).toContain("Whole-brain inspection is unavailable"); expect(html).toContain("Brain Map");
  expect(html).not.toContain("Brain map domain"); expect(html).not.toContain("No token or semantic-neuron mapping");
});
it("preserves actual repeated-contact and plasticity metadata in the unified canvas", () => {
  const view = wholeBrainDisplayGraph({ ...graph, edges: [{ id: "repeat", recordId: "repeat-record", kind: "substrate-synapse",
    sourceId: "layout:whole", targetId: "layout:owner", scopeId: "root", learnedConnection: true, firingObserved: false, effectiveWeight: 1, strength: 2,
    data: { id: "repeat-record", sourceId: "whole", targetId: "owner", kind: "recurrent", effectiveWeight: 1,
      eligibility: 3, plasticity: 0.25, stability: 0.75, uses: 9, connectionStrength: 2,
      isRepeatedContact: true, parallelOrdinal: 2, parallelRootId: "first-record" } }] });
  expect(view.edges[0]).toMatchObject({ isRepeatedContact: true, parallelOrdinal: 2, parallelRootId: "first-record",
    eligibility: 3, plasticity: 0.25, stability: 0.75, uses: 9, connectionStrength: 2,
    nativeEndpoints: { sourceId: "whole", targetId: "owner" } });
});
