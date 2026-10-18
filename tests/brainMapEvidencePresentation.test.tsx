import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { BrainMapSubstrateGraph } from "../src/renderer/src/BrainMapSubstrateGraph";
import { graphContactPaths, graphEdgeTitle, graphPositions, observedActivation, recordedFiring,
  type BrainMapGraphEdge, type BrainMapGraphNode } from "../src/renderer/src/brainMapGraphPresentation";
import { cortexActivityViewport, cortexInventoryLabel, cortexRecordLabel, cortexRequestKey } from "../src/renderer/src/cortexPresentation";
import type { CortexActivity, CortexModule, CortexPage } from "../src/shared/cortexInspection";

const node = (id: string, extra: Partial<BrainMapGraphNode> = {}): BrainMapGraphNode => ({
  id, label: id, activation: 0.9, importance: 0.5, members: [id], ...extra
});
const edge = (id: string, extra: Partial<BrainMapGraphEdge> = {}): BrainMapGraphEdge => ({
  id, sourceId: "a", targetId: "b", effectiveWeight: 1, signedStrength: 7, connectionStrength: 7,
  stability: 0.3, plasticity: 0.125, eligibility: 0.75, effectiveContribution: 7, pathways: 1, ...extra
});
const module: CortexModule = {
  id: "test._packed_forward_weight", module: "not-an-embedding-name", field: "weight", role: "core", group: "test",
  rows: 32, columns: 90, packedColumns: 23, packedBytes: 736, logicalShape: [32, 90], logicalParameters: 2880,
  layout: "row-packed", shapeEvidence: "native-connection-state-shape", activationObserved: false, activation: null,
  storageKind: "base-projection", activityOwner: "actual-owner"
};
const page = (extra: Partial<CortexPage> = {}): CortexPage => ({
  brainId: "b", revision: "native-generation", source: "committed-packed-native-cortex", entity: "elements",
  total: 90, offset: 12, returned: 4, hasMore: true, records: [], selectedModule: module, groups: ["test"], moduleCount: 1,
  packedBytes: 736, logicalParameters: 2880, logicalInventoryComplete: true,
  activity: { observed: false, reason: "committed" }, integrity: { generationManifestVerified: true, selectedTernaryCodesValidated: true, wholeRolePayloadScanned: false },
  relationships: { kind: "computational-dependencies-not-semantic-explanation", ideaInputs: [], tokenInputs: [] }, ...extra
});
const descriptor = (axis: NonNullable<CortexActivity["outputAxis"]>, size: number): CortexActivity => ({
  available: true, observed: false, observation: null, outputAxis: axis, outputSize: size
});

describe("new Brain Map evidence/data contracts (no model or device run)", () => {
  it("retains every passed node and independent repeated/self contact in 2D and3D", () => {
    const nodes = Array.from({ length: 173 }, (_, index) => node(index === 0 ? "a" : index === 1 ? "b" : `n${index}`));
    const edges = [edge("original"), edge("repeat1", { isRepeatedContact: true, parallelOrdinal: 1 }),
      edge("repeat2", { isRepeatedContact: true, parallelOrdinal: 2 }), edge("self", { targetId: "a" })];
    for (const dimension of ["2d", "3d"] as const) {
      const positions = graphPositions(nodes, dimension);
      expect(positions.size).toBe(173);
      const paths = graphContactPaths(edges, positions);
      expect(paths).toHaveLength(4);
      expect(new Set(paths.map((value) => value.path)).size).toBe(4);
      const html = renderToStaticMarkup(<BrainMapSubstrateGraph nodes={nodes} edges={edges} selectedId="" onSelect={() => undefined} initialDimension={dimension} />);
      expect((html.match(/data-node-id=/g) ?? []).length).toBe(173);
      expect((html.match(/data-contact-id=/g) ?? []).length).toBe(4);
      expect(html).toContain(`data-dimension="${dimension}"`);
      expect(html).toContain("layout coordinates are not anatomy");
    }
  });

  it("never equates confidence, strength or unobserved activation with spikes", () => {
    expect(observedActivation(node("a"))).toBeNull();
    expect(observedActivation(node("a", { activationObserved: true }))).toBe(0.9);
    expect(observedActivation(node("a", { activationObserved: true, activityOnly: true }))).toBeNull();
    expect(recordedFiring(undefined)).toBe(false);
    expect(recordedFiring({ source: "recorded-spike-event", observedAt: "bad", count: 1 })).toBe(false);
    const html = renderToStaticMarkup(<BrainMapSubstrateGraph nodes={[node("a"), node("b")]} edges={[edge("strong", { connectionStrength: 16 })]} selectedId="" onSelect={() => undefined} />);
    expect(html).not.toContain('data-firing="observed"');
    expect(html).not.toContain('data-activation="observed"');
    expect(html).toContain("Path firing: unobserved");
    const actual = { source: "recorded-spike-event" as const, observedAt: "2026-10-13T12:00:00Z", count: 2 };
    expect(recordedFiring(actual)).toBe(true);
  });

  it("preserves exact weighted/plasticity/repeat/source inspection values", () => {
    const title = graphEdgeTitle(edge("repeat", { effectiveWeight: -1, connectionStrength: 16, effectiveContribution: -16,
      isRepeatedContact: true, parallelOrdinal: 2049, parallelRootId: "original" }));
    expect(title).toContain("a → b");
    for (const text of ["Ternary sign: -1", "Strength: 16", "Effective contribution: -16", "Plasticity: 0.125", "Stability: 0.3", "Eligibility: 0.75", "ordinal: 2049", "Root: original"]) expect(title).toContain(text);
    expect(graphContactPaths([edge("off-page", { targetId: "missing" })], graphPositions([node("a")], "3d"))).toEqual([]);
    const aggregate = graphEdgeTitle(edge("region", { aggregate: true, sourceId: "assembly", targetId: "assembly",
      nativeEndpoints: { sourceId: "actual-neuron-a", targetId: "actual-neuron-b" }, aggregateSignCounts: { negative: 4, zero: 2, positive: 9 } }));
    expect(aggregate).toContain("actual-neuron-a → actual-neuron-b");
    expect(aggregate).toContain("aliases are not new neural connections");
    expect(aggregate).not.toContain("Ternary sign:");
    expect(aggregate).toContain("-1=4, 0=2, +1=9");
  });

  it("makes stable3D layout independent of node activity and exposes orbit rotation", () => {
    const original = graphPositions([node("a"), node("b")], "3d");
    expect(graphPositions([node("b", { activation: 0 }), node("a", { activation: 0.1 })], "3d").get("a")).toEqual(original.get("a"));
    expect(graphPositions([node("a"), node("b")], "3d", 0.2).get("a")).not.toEqual(original.get("a"));
    const html = renderToStaticMarkup(<BrainMapSubstrateGraph nodes={[node("a")]} edges={[]} selectedId="" onSelect={() => undefined} initialDimension="3d" />);
    expect(html).toContain("Rotate brain graph left");
    expect(html).toContain("Reset brain graph rotation");
  });

  it("uses actual forward axis descriptors for embedding, ordinary, bias and repeat viewports", () => {
    expect(cortexActivityViewport(module, { entity: "elements", row: 9 }, page(), descriptor("embedding-column", 90), true))
      .toEqual({ module: "actual-owner", enabled: true, start: 12, count: 4 });
    expect(cortexActivityViewport({ ...module, module: "scale_delta" }, { entity: "elements", row: 9 }, page(), descriptor("output-row", 32), true)?.start).toBe(9);
    expect(cortexActivityViewport({ ...module, field: "bias" }, { entity: "links" }, page(), descriptor("output-row", 32), true)?.start).toBe(12);
    const repeated = { ...module, storageKind: "repeated-contact-page" as const };
    const repeatedPage = page({ records: [{ row: 7, column: 40, value: 1, activationObserved: false, activation: null, isRepeatedContact: true, ordinal: 9 }] });
    expect(cortexActivityViewport(repeated, { entity: "elements" }, repeatedPage, descriptor("output-row", 32), true)?.start).toBe(7);
    expect(cortexActivityViewport(repeated, { entity: "elements" }, repeatedPage, descriptor("embedding-column", 90), true)?.start).toBe(40);
    expect(cortexActivityViewport(module, { entity: "elements", row: 9 }, page(), descriptor("output-channel", 4), true)?.start).toBe(0);
    expect(cortexActivityViewport(module, {}, page(), { available: false, observed: false, observation: null }, true)).toBeNull();
  });

  it("separates known base inventory from allocated repeated slots and preserves role/bias labels", () => {
    const value = page({ logicalInventoryComplete: false, inventory: { baseLogicalParameters: 2880, baseLogicalInventoryComplete: true,
      repeatedContactPages: 2, allocatedRepeatedContactSlots: 2048, repeatedLiveContacts: null } });
    expect(cortexInventoryLabel(value)).toBe("2,880 base parameter slots · live repeated contacts paged separately");
    expect(cortexRecordLabel({ ...module, role: "plasticity", field: "bias" })).toContain("bias · plasticity");
    expect(cortexRecordLabel({ sourceAxis: "constant-one", sourceIndex: 0, targetAxis: "output-row", targetIndex: 19,
      moduleId: "bias", relationship: "packed-bias-addition", semanticExplanationVerified: false, value: -1, strength: 5, resistance: 12 }))
      .toBe("constant 1 → output 19: -1 × 5 · resistance 12");
  });

  it("invalidates old owner/query/generation UI result keys without reusing their selection", () => {
    const key = cortexRequestKey("b", "old", { entity: "elements", moduleId: "one", row: 3 });
    expect(cortexRequestKey("c", "old", { entity: "elements", moduleId: "one", row: 3 })).not.toBe(key);
    expect(cortexRequestKey("b", "new", { entity: "elements", moduleId: "one", row: 3 })).not.toBe(key);
    expect(cortexRequestKey("b", "old", { entity: "elements", moduleId: "two", row: 3 })).not.toBe(key);
  });
});
