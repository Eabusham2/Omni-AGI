import { describe, expect, it } from "vitest";
import { graphLabelVisibility, graphViewportMetrics, type BrainMapGraphNode, type GraphPoint } from "../src/renderer/src/brainMapGraphPresentation";

const selected: BrainMapGraphNode = { id: "selected", label: "Observed activity", activation: 0.8, activationObserved: true, importance: 0.7, members: ["selected"] };

describe("new confirmed desktop label-collision contract", () => {
  it("retains selected text and suppresses only the overlapping optional cluster text/count", () => {
    const other: BrainMapGraphNode = { ...selected, id: "cluster", members: ["cluster"], label: "Overlapping cluster", clusterCount: 100000 };
    const positions = new Map<string, GraphPoint>([["selected", { x: 500, y: 250, depth: 0, scale: 1 }], ["cluster", { x: 500, y: 285, depth: 0, scale: 1 }]]);
    const visible = graphLabelVisibility([selected, other], positions, "selected", true, graphViewportMetrics(960, 440));
    expect(visible.get("selected")!.label).toBe(true);
    expect(visible.get("cluster")!.count).toBe(false);
    expect(visible.size).toBe(2);
    // Independent text can remain if it does not intersect the selected name.
    expect(positions.size).toBe(2);
  });

  it("keeps all labels in accessible records while hiding narrow optional text", () => {
    const nodes = [selected, { ...selected, id: "other", members: ["other"] }];
    const positions = new Map<string, GraphPoint>(nodes.map((node, index) => [node.id, { x: 300 + index * 300, y: 250, depth: 0, scale: 1 }]));
    const visible = graphLabelVisibility(nodes, positions, "selected", false, graphViewportMetrics(320, 440));
    expect(visible.get("selected")!.label).toBe(true);
    expect(visible.get("other")!.label).toBe(false);
    expect(visible.size).toBe(nodes.length);
  });
});
