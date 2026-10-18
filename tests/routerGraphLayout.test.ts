import { expect, it } from "vitest";
import { graphPositions, type BrainMapGraphNode } from "../src/renderer/src/brainMapGraphPresentation";

const nodes: BrainMapGraphNode[] = Array.from({ length: 64 }, (_, channel) => ({
  id: `original:root.router.population:${channel}`, label: `Router channel ${channel}`,
  activation: 0, activationObserved: false, importance: 0, members: [`original:root.router.population:${channel}`]
}));
it("spreads similar actual channel IDs through 3D instead of collapsing their suffixes", () => {
  const points = [...graphPositions(nodes, "3d").values()];
  expect(Math.max(...points.map(p => p.x)) - Math.min(...points.map(p => p.x))).toBeGreaterThan(300);
  expect(Math.max(...points.map(p => p.y)) - Math.min(...points.map(p => p.y))).toBeGreaterThan(200);
  expect(points.every(p => Number.isFinite(p.depth) && p.x > 0 && p.x < 1000 && p.y > 0 && p.y < 650)).toBe(true);
});
it("keeps channel coordinates stable across contact pages, input order and activity values", () => {
  const full = graphPositions(nodes, "3d", 0.2, -0.1);
  const selected = nodes.slice(8, 16).reverse().map(node => ({ ...node, activation: 100, importance: 1 }));
  const page = graphPositions(selected, "3d", 0.2, -0.1);
  for (const node of selected) expect(page.get(node.id)).toEqual(full.get(node.id));
});
