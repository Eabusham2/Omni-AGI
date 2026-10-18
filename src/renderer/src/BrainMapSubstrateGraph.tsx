import { useEffect, useId, useMemo, useRef, useState } from "react";
import { graphContactPaths, graphEdgeTitle, graphLabelOffset, graphLabelVisibility, graphNodeRadius, graphNodeTitle, graphPositions, graphViewportMetrics, observedActivation, recordedFiring, recordedTransmission,
  type BrainMapDimension, type BrainMapGraphEdge, type BrainMapGraphNode } from "./brainMapGraphPresentation";
import "./brainMapSubstrateGraph.css";

export type { BrainMapGraphEdge, BrainMapGraphNode } from "./brainMapGraphPresentation";

/** Draw only the current real paged records; never synthesize missing links/activity. */
export function BrainMapSubstrateGraph({ nodes, edges, selectedId, onSelect, selectedEdgeId, onSelectEdge, clustered = false, initialDimension = "2d" }: {
  nodes: readonly BrainMapGraphNode[]; edges: readonly BrainMapGraphEdge[]; selectedId: string;
  onSelect: (node: BrainMapGraphNode) => void; clustered?: boolean; initialDimension?: BrainMapDimension;
  selectedEdgeId?: string; onSelectEdge?: (edge: BrainMapGraphEdge) => void;
}) {
  const [dimension, setDimension] = useState<BrainMapDimension>(initialDimension);
  const [rotation, setRotation] = useState({ yaw: 0, pitch: 0 });
  const [viewport, setViewport] = useState({ width: 1000, height: 650 });
  const svgViewport = useRef<SVGSVGElement>(null);
  const dragging = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(null);
  useEffect(() => {
    const svg = svgViewport.current; if (!svg) return;
    const measure = () => { const bounds = svg.getBoundingClientRect(); setViewport((value) => value.width === bounds.width && value.height === bounds.height ? value : { width: bounds.width, height: bounds.height }); };
    measure();
    const observer = new ResizeObserver(measure); observer.observe(svg);
    return () => observer.disconnect();
  }, []);
  const metrics = graphViewportMetrics(viewport.width, viewport.height);
  const markerId = useId().replaceAll(":", "");
  const positions = useMemo(() => graphPositions(nodes, dimension, rotation.yaw, rotation.pitch), [nodes, dimension, rotation]);
  const contacts = useMemo(() => graphContactPaths(edges, positions), [edges, positions]);
  const labels = useMemo(() => graphLabelVisibility(nodes, positions, selectedId, clustered, metrics),
    [nodes, positions, selectedId, clustered, metrics.scale, metrics.narrow]);
  const orderedNodes = useMemo(() => dimension === "3d" ? [...nodes].sort((a, b) => positions.get(a.id)!.depth - positions.get(b.id)!.depth) : nodes,
    [nodes, positions, dimension]);
  const rotate = (yaw: number, pitch: number) => setRotation((value) => ({ yaw: value.yaw + yaw, pitch: Math.max(-Math.PI / 2, Math.min(Math.PI / 2, value.pitch + pitch)) }));
  return <div className="brain-map-graph" data-dimension={dimension}>
    <div className="brain-map-graph__dimensions segmented" role="group" aria-label="Brain graph dimensions">
      <button aria-pressed={dimension === "2d"} className={dimension === "2d" ? "is-active" : ""} onClick={() => setDimension("2d")}>2D</button>
      <button aria-pressed={dimension === "3d"} className={dimension === "3d" ? "is-active" : ""} onClick={() => setDimension("3d")}>3D</button>
      {dimension === "3d" ? <><button aria-label="Rotate brain graph left" title="Rotate left" onClick={() => rotate(-0.2, 0)}>↶</button>
        <button aria-label="Rotate brain graph right" title="Rotate right" onClick={() => rotate(0.2, 0)}>↷</button>
        <button aria-label="Reset brain graph rotation" title="Reset rotation" onClick={() => setRotation({ yaw: 0, pitch: 0 })}>Reset</button></> : null}
    </div>
    <svg ref={svgViewport} viewBox="0 0 1000 650" preserveAspectRatio="xMidYMid meet" role="group" aria-label={`${dimension.toUpperCase()} paged neural substrate; layout coordinates are not anatomy`} tabIndex={dimension === "3d" ? 0 : undefined}
      onKeyDown={(event) => {
        if (event.target !== event.currentTarget || dimension !== "3d") return;
        const delta = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] }[event.key];
        if (delta) { event.preventDefault(); rotate(delta[0]!, delta[1]!); }
      }}
      onPointerDown={(event) => {
        if (dimension !== "3d" || (event.target as Element).closest("[data-node-id]")) return;
        dragging.current = { x: event.clientX, y: event.clientY, ...rotation }; event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        const initial = dragging.current; if (!initial) return;
        setRotation({ yaw: initial.yaw + (event.clientX - initial.x) * 0.008,
          pitch: Math.max(-Math.PI / 2, Math.min(Math.PI / 2, initial.pitch + (event.clientY - initial.y) * 0.008)) });
      }}
      onPointerUp={() => { dragging.current = null; }} onPointerCancel={() => { dragging.current = null; }}>
      <desc>{nodes.length} inspected nodes and {contacts.length} visible recorded pathways. {edges.length - contacts.length} pathways have off-page or invalid endpoints. No spike is inferred from connection strength or confidence.</desc>
      <defs><marker id={markerId} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" /></marker></defs>
      {contacts.map(({ edge, path }) => {
        const firing = recordedFiring(edge.firingEvent);
        const transmission = recordedTransmission(edge.transmissionEvent);
        const strength = edge.connectionStrength !== undefined && Number.isFinite(edge.connectionStrength) ? Math.abs(edge.connectionStrength) : 1;
        return <path key={edge.id} data-contact-id={edge.id} data-repeated={edge.isRepeatedContact === true ? "true" : undefined}
          role={onSelectEdge ? "button" : undefined} tabIndex={onSelectEdge ? 0 : undefined}
          aria-label={onSelectEdge ? graphEdgeTitle(edge) : undefined} aria-pressed={onSelectEdge ? selectedEdgeId === edge.id : undefined}
          onClick={onSelectEdge ? () => onSelectEdge(edge) : undefined}
          onKeyDown={onSelectEdge ? event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelectEdge(edge); } } : undefined}
          data-firing={firing ? "observed" : "unobserved"} data-transmission={transmission ? edge.transmissionEvent!.classification : "unobserved"}
          data-relationship={edge.aggregate ? "actual-contact-aggregate" : edge.structuralRelationship ? "registered-structure" : "learned-contact"}
          className={`graph-edge ${edge.aggregate ? "graph-edge--aggregate" : edge.structuralRelationship ? "graph-edge--structural" : edge.effectiveWeight < 0 ? "graph-edge--negative" : edge.effectiveWeight === 0 ? "graph-edge--neutral" : "graph-edge--positive"}${transmission ? ` graph-edge--transmission-${edge.transmissionEvent!.classification}` : ""}`}
          d={path} fill="none" strokeOpacity={selectedEdgeId === edge.id || firing || transmission ? 0.95 : 0.35} strokeWidth={(selectedEdgeId === edge.id ? 1.75 : 0.75) + Math.log2(strength + 1) * 0.55}
          strokeDasharray={edge.aggregate ? undefined : edge.effectiveWeight < 0 ? "5 5" : edge.effectiveWeight === 0 ? "2 7" : undefined} markerEnd={`url(#${markerId})`}>
          <title>{graphEdgeTitle(edge)}</title></path>;
      })}
      {orderedNodes.map((node) => {
        const position = positions.get(node.id)!;
        const selected = node.id === selectedId || node.members.includes(selectedId);
        const activation = observedActivation(node);
        const firing = recordedFiring(node.firingEvent);
        const radius = graphNodeRadius(node, position, clustered, metrics);
        const label = node.label.length > 20 ? `${node.label.slice(0, 19)}…` : node.label;
        return <g key={node.id} data-node-id={node.id} data-activation={activation === null ? "unobserved" : "observed"} data-firing={firing ? "observed" : "unobserved"}
          className="graph-node" role="button" tabIndex={0} aria-pressed={selected} aria-label={graphNodeTitle(node)} transform={`translate(${position.x} ${position.y})`}
          onClick={() => onSelect(node)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect(node); } }}>
          <title>{graphNodeTitle(node)}</title><circle className="graph-node__hit-area" r={Math.max(metrics.minimumHitRadius, radius + 5)} />
          <circle r={radius + (selected ? 9 : 3)} fill="var(--graph-node-stroke)" opacity={selected ? 0.18 : 0.065} />
          <circle r={radius} fill={firing ? "var(--graph-node-ink)" : "var(--graph-node-stroke)"}
            opacity={firing ? 1 : activation === null ? 0.45 : 0.45 + Math.max(0, Math.min(1, activation)) * 0.45}
            stroke={selected ? "var(--graph-node-stroke-selected)" : "var(--graph-node-stroke)"} strokeWidth={selected ? 2.2 : 1} />
          {labels.get(node.id)?.count ? <text y="4" textAnchor="middle" fill="var(--graph-node-ink)" fontSize={metrics.labelFont} fontWeight="700">{node.clusterCount!.toLocaleString()}</text>
            : <circle r={Math.max(2.5, radius * 0.22)} fill="var(--graph-node-ink)" opacity=".9" />}
          {labels.get(node.id)?.label ? <text x={graphLabelOffset(position.x, label.length, metrics)} y={radius + metrics.labelFont * 1.4} textAnchor="middle" fill="var(--graph-label)" fontSize={metrics.labelFont}>
            {label}</text> : null}
        </g>;
      })}
    </svg>
    <small className="brain-map-graph__scope">Current paged records · {dimension === "3d" ? "drag or use arrow keys to rotate · " : ""}layout, not anatomy · activation is not a recorded spike</small>
  </div>;
}
