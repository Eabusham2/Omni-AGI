import { describe, expect, it } from "vitest";
import { projectBrainMapContact, uniqueRegionLayoutNodes } from "../src/renderer/src/brainMapGraphProjection";
import type { SubstrateSynapse } from "../src/shared/types";
const contact: SubstrateSynapse = { id: "actual-contact", sourceId: "neuron-a", targetId: "neuron-b", kind: "learned",
  effectiveWeight: -1, connectionStrength: 7, effectiveContribution: -7, isRepeatedContact: true,
  parallelOrdinal: 3, parallelRootId: "contact-root", eligibility: 0.25, plasticity: 0.4, stability: 0.9, uses: 8 };
describe("actual synapse projection into existing Brain Map", () => {
  it("preserves every actual weighted/plastic contact field and original endpoints", () => {
    const edge = projectBrainMapContact(contact, new Map([["neuron-a", "assembly"], ["neuron-b", "assembly"]]));
    expect(edge).toMatchObject({ id: contact.id, sourceId: "assembly", targetId: "assembly", nativeEndpoints: { sourceId: "neuron-a", targetId: "neuron-b" },
      connectionStrength: 7, effectiveContribution: -7, effectiveWeight: -1, parallelOrdinal: 3, parallelRootId: "contact-root",
      isRepeatedContact: true, eligibility: 0.25, plasticity: 0.4, stability: 0.9, uses: 8, pathways: 1, aggregate: false });
  });
  it("does not invent off-page endpoints or missing strength/firing data", () => {
    expect(projectBrainMapContact(contact, new Map([["neuron-a", "node-a"]]))).toBeNull();
    const minimal = { id: "basic", sourceId: "a", targetId: "b", effectiveWeight: 1 as const, stability: 0, plasticity: 0, uses: 0 };
    const edge = projectBrainMapContact(minimal, new Map([["a", "a"], ["b", "b"]]))!;
    expect(edge.connectionStrength).toBeUndefined(); expect(edge.effectiveContribution).toBeUndefined(); expect(edge.firingEvent).toBeUndefined();
  });
  it("does not attach regional paths to the last arbitrary activation band", () => {
    const nodes = ["band-a", "band-b"].map(id => ({ id, label: id, region: "language", activation: 0, importance: 0, members: [id] }));
    expect(uniqueRegionLayoutNodes(nodes, []).has("language")).toBe(false);
    const region = { ...nodes[0]!, id: "region-language" };
    expect(uniqueRegionLayoutNodes([...nodes, region], [{ id: region.id, kind: "region" }]).get("language")).toBe(region.id);
  });
});
