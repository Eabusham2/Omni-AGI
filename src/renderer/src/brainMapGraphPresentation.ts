/** Paged neural inspection data. Coordinates below are layout, not anatomy. */
export interface BrainMapGraphNode {
  id: string; label: string; region?: string; activation: number; activationObserved?: boolean;
  importance: number; members: string[]; clusterCount?: number; activityOnly?: boolean; lastActivatedAt?: string;
  aggregateUnit?: "neurons" | "logical parameters";
  /** Only a real spike-event source may supply this; activation is not a spike. */
  firingEvent?: { source: "recorded-spike-event"; observedAt: string; count: number };
}

export interface BrainMapGraphEdge {
  id: string; sourceId: string; targetId: string; effectiveWeight: -1 | 0 | 1;
  connectionStrength?: number; effectiveContribution?: number; signedStrength: number;
  isRepeatedContact?: boolean; parallelOrdinal?: number; parallelRootId?: string;
  stability: number; plasticity?: number; eligibility?: number; uses?: number; pathways: number;
  /** False/absent for ordinary stored synapses; do not infer firing from sign. */
  firingEvent?: { source: "recorded-spike-event"; observedAt: string; count: number };
  /** Actual packed contact operands, separate from postsynaptic firing. */
  transmissionEvent?: { source: "recorded-contact-transmission"; observedAt: string;
    count: number; contribution: number; classification: "inhibitory" | "excitatory" | "zero" };
  /** An aggregate may not masquerade as a single recorded contact. */
  aggregate?: boolean;
  /** Original endpoints are preserved when layout nodes represent assemblies/clusters. */
  nativeEndpoints?: { sourceId: string; targetId: string };
  /** Registered structure is not a learned zero-valued synapse. */
  structuralRelationship?: string;
  aggregateSignCounts?: { negative: number; zero: number; positive: number };
}

export interface GraphPoint { x: number; y: number; depth: number; scale: number }
export type BrainMapDimension = "2d" | "3d";

export function graphViewportMetrics(width: number, height: number) {
  const scale = Math.max(0.01, Math.min(Number.isFinite(width) && width > 0 ? width / 1000 : 1,
    Number.isFinite(height) && height > 0 ? height / 650 : 1));
  return { scale, narrow: width < 520, labelFont: 12 / scale, minimumHitRadius: 22 / scale,
    minimumNodeRadius: 5 / scale, labelPadding: 8 / scale };
}

export function graphLabelOffset(x: number, labelLength: number, metrics: ReturnType<typeof graphViewportMetrics>): number {
  const halfWidth = Math.min(500 - metrics.labelPadding, labelLength * metrics.labelFont * 0.3);
  return Math.max(halfWidth + metrics.labelPadding, Math.min(1000 - halfWidth - metrics.labelPadding, x)) - x;
}

export function graphNodeRadius(node: BrainMapGraphNode, point: GraphPoint, clustered: boolean, metrics: ReturnType<typeof graphViewportMetrics>): number {
  const importance = Number.isFinite(node.importance) ? Math.max(0, Math.min(1, node.importance)) : 0;
  return Math.max(metrics.minimumNodeRadius, (10 + importance * 10 + (clustered ? Math.min(15, Math.log2((node.clusterCount ?? 1) + 1) * 4) : 0)) * point.scale);
}

/** Suppress optional text only, in screen-space-equivalent SVG units. Every
 * actual node/contact and its accessible title remains available. */
export function graphLabelVisibility(nodes: readonly BrainMapGraphNode[], positions: ReadonlyMap<string, GraphPoint>, selectedId: string,
  clustered: boolean, metrics: ReturnType<typeof graphViewportMetrics>): Map<string, { label: boolean; count: boolean }> {
  const result = new Map(nodes.map((node) => [node.id, { label: false, count: false }]));
  const occupied: Array<{ left: number; right: number; top: number; bottom: number }> = [];
  const overlaps = (box: typeof occupied[number]) => occupied.some((other) => box.left < other.right && box.right > other.left && box.top < other.bottom && box.bottom > other.top);
  const labelBox = (node: BrainMapGraphNode) => {
    const point = positions.get(node.id)!;
    const length = Math.min(20, node.label.length);
    const center = point.x + graphLabelOffset(point.x, length, metrics);
    const radius = graphNodeRadius(node, point, clustered, metrics);
    const half = length * metrics.labelFont * 0.325 + metrics.labelPadding / 2;
    const baseline = point.y + radius + metrics.labelFont * 1.4;
    return { left: center - half, right: center + half, top: baseline - metrics.labelFont, bottom: baseline + metrics.labelFont * 0.3 };
  };
  const selected = (node: BrainMapGraphNode) => node.id === selectedId || node.members.includes(selectedId);
  // Selected text is never removed in favor of another label or count.
  for (const node of nodes) if (selected(node) && positions.has(node.id)) {
    result.get(node.id)!.label = true; occupied.push(labelBox(node));
  }
  const optional = [...nodes].filter((node) => positions.has(node.id)).sort((a, b) => Number(selected(b)) - Number(selected(a)) || b.importance - a.importance || a.id.localeCompare(b.id));
  for (const node of optional) {
    const value = result.get(node.id)!;
    if (clustered && (node.clusterCount ?? 0) > 1 && !metrics.narrow) {
      const point = positions.get(node.id)!;
      const half = node.clusterCount!.toLocaleString().length * metrics.labelFont * 0.325 + metrics.labelPadding / 2;
      const box = { left: point.x - half, right: point.x + half, top: point.y + 4 - metrics.labelFont, bottom: point.y + 4 + metrics.labelFont * 0.3 };
      if (!overlaps(box)) { value.count = true; occupied.push(box); }
    }
    if (!selected(node) && !metrics.narrow && (nodes.length <= 90 || clustered)) {
      const box = labelBox(node);
      if (!overlaps(box)) { value.label = true; occupied.push(box); }
    }
  }
  return result;
}

function hash(value: string, seed = 2_166_136_261): number {
  let result = seed;
  for (let index = 0; index < value.length; index++) result = Math.imul(result ^ value.charCodeAt(index), 16_777_619);
  // Similar channel IDs often differ only in their final digit. Avalanche the
  // display hash so those real, distinct nodes do not collapse into one point.
  // This affects coordinates only, never connectivity or measured activity.
  result ^= result >>> 16;
  result = Math.imul(result, 0x85ebca6b);
  result ^= result >>> 13;
  result = Math.imul(result, 0xc2b2ae35);
  result ^= result >>> 16;
  return result >>> 0;
}

export function graphPositions(nodes: readonly BrainMapGraphNode[], dimension: BrainMapDimension, yaw = 0, pitch = 0): Map<string, GraphPoint> {
  const positions = new Map<string, GraphPoint>();
  const total = Math.max(1, nodes.length);
  const cosineYaw = Math.cos(yaw), sineYaw = Math.sin(yaw), cosinePitch = Math.cos(pitch), sinePitch = Math.sin(pitch);
  nodes.forEach((node, index) => {
    if (dimension === "2d") {
      const angle = index * 2.399963229728653;
      const radial = Math.sqrt((index + 0.65) / total);
      positions.set(node.id, { x: 500 + Math.cos(angle) * radial * 420, y: 325 + Math.sin(angle) * radial * 265, depth: 0, scale: 1 });
      return;
    }
    // Stable layout across pages and rerenders, independent of firing, strength
    // or semantic claims. No node/cardinality ceiling is applied here.
    const longitude = hash(node.id) / 0x1_0000_0000 * Math.PI * 2;
    const vertical = hash(node.id, 2_246_822_519) / 0x1_0000_0000 * 2 - 1;
    const radius = 0.3 + hash(node.id, 3_266_489_917) / 0x1_0000_0000 * 0.7;
    const ring = Math.sqrt(1 - vertical * vertical) * radius;
    const x = Math.cos(longitude) * ring, y = vertical * radius, z = Math.sin(longitude) * ring;
    const rotatedX = x * cosineYaw + z * sineYaw;
    const yawZ = -x * sineYaw + z * cosineYaw;
    const rotatedY = y * cosinePitch - yawZ * sinePitch;
    const depth = y * sinePitch + yawZ * cosinePitch;
    const scale = 1 / (1.8 - depth * 0.5);
    positions.set(node.id, { x: 500 + rotatedX * 620 * scale, y: 325 + rotatedY * 400 * scale, depth, scale: scale * 1.8 });
  });
  return positions;
}

export function recordedFiring(event: BrainMapGraphNode["firingEvent"]): boolean {
  return event?.source === "recorded-spike-event" && Number.isSafeInteger(event.count) && event.count > 0 && Number.isFinite(Date.parse(event.observedAt));
}

export function recordedTransmission(event: BrainMapGraphEdge["transmissionEvent"]): boolean {
  return event?.source === "recorded-contact-transmission" && Number.isSafeInteger(event.count) && event.count > 0 &&
    Number.isFinite(event.contribution) && Number.isFinite(Date.parse(event.observedAt)) &&
    event.classification === (event.contribution < 0 ? "inhibitory" : event.contribution > 0 ? "excitatory" : "zero");
}

export function observedActivation(node: BrainMapGraphNode): number | null {
  return node.activationObserved === true && !node.activityOnly && Number.isFinite(node.activation) ? node.activation : null;
}

function numeric(value: number | undefined): string { return value === undefined || !Number.isFinite(value) ? "unreported" : String(value); }

export function graphNodeTitle(node: BrainMapGraphNode): string {
  const activation = observedActivation(node);
  return `${node.label}\nID: ${node.id}\nRegion: ${node.region ?? "unreported"}` +
    (node.clusterCount === undefined ? "" : `\n${node.clusterCount} ${node.aggregateUnit ?? "represented neurons"} in this inspected group`) +
    `\nMeasured activation: ${activation === null ? "unobserved" : activation}` +
    (node.lastActivatedAt ? `\nLast recorded activation: ${node.lastActivatedAt}` : "") +
    (recordedFiring(node.firingEvent) ? `\nRecorded spikes: ${node.firingEvent!.count} at ${node.firingEvent!.observedAt}` : "\nSpike events: unobserved") +
    "\nPosition is a visualization layout, not biological location.";
}

export function graphEdgeTitle(edge: BrainMapGraphEdge): string {
  if (edge.structuralRelationship) return `${edge.structuralRelationship}\n${edge.sourceId} → ${edge.targetId}\nRecorded source structure, not an observed synaptic weight or firing event.`;
  if (edge.aggregate) return `${edge.sourceId} → ${edge.targetId}\nActual aggregate pathway: ${edge.id}\n${edge.pathways} stored contacts` +
    (edge.aggregateSignCounts ? `\nSource sign counts: -1=${edge.aggregateSignCounts.negative}, 0=${edge.aggregateSignCounts.zero}, +1=${edge.aggregateSignCounts.positive}` : "") +
    "\nNo representative ternary sign, strength, transmission or spike event is inferred.";
  const sign = edge.effectiveWeight === 1 ? "+1" : String(edge.effectiveWeight);
  const native = edge.nativeEndpoints ?? edge;
  const alias = native.sourceId !== edge.sourceId || native.targetId !== edge.targetId;
  return `${native.sourceId} → ${native.targetId}\n${edge.aggregate ? "Aggregate pathway" : "Contact"}: ${edge.id}` +
    (alias ? `\nLayout endpoints: ${edge.sourceId} → ${edge.targetId}; these aliases are not new neural connections.` : "") +
    (edge.aggregate ? `\nNet display classification: ${sign}; not a learned ternary sign.` : `\nTernary sign: ${sign}`) +
    (edge.aggregateSignCounts ? `\nSource sign counts: -1=${edge.aggregateSignCounts.negative}, 0=${edge.aggregateSignCounts.zero}, +1=${edge.aggregateSignCounts.positive}` : "") +
    `\nStrength: ${numeric(edge.connectionStrength)}` +
    `\nEffective contribution: ${numeric(edge.effectiveContribution)}` +
    `\nPlasticity: ${numeric(edge.plasticity)}\nStability: ${numeric(edge.stability)}\nEligibility: ${numeric(edge.eligibility)}` +
    (edge.isRepeatedContact ? `\nRepeated contact ordinal: ${numeric(edge.parallelOrdinal)}${edge.parallelRootId ? `\nRoot: ${edge.parallelRootId}` : ""}` : "") +
    (edge.aggregate ? `\n${edge.pathways} source pathways; aggregate values are not an individual learned weight.` : "") +
    (recordedTransmission(edge.transmissionEvent) ? `\nRecorded contact transmissions: ${edge.transmissionEvent!.count}; last contribution ${edge.transmissionEvent!.contribution} (${edge.transmissionEvent!.classification}) at ${edge.transmissionEvent!.observedAt}` : "\nContact transmission: unobserved") +
    (recordedFiring(edge.firingEvent) ? `\nRecorded spikes: ${edge.firingEvent!.count} at ${edge.firingEvent!.observedAt}` : "\nPostsynaptic/path firing: unobserved");
}

export function graphContactPaths(edges: readonly BrainMapGraphEdge[], positions: ReadonlyMap<string, GraphPoint>): Array<{ edge: BrainMapGraphEdge; path: string }> {
  const groups = new Map<string, BrainMapGraphEdge[]>();
  for (const edge of edges) {
    if (!positions.has(edge.sourceId) || !positions.has(edge.targetId) || ![-1, 0, 1].includes(edge.effectiveWeight)) continue;
    const key = JSON.stringify([edge.sourceId, edge.targetId]);
    const values = groups.get(key) ?? []; values.push(edge); groups.set(key, values);
  }
  const paths: Array<{ edge: BrainMapGraphEdge; path: string }> = [];
  for (const group of groups.values()) {
    group.sort((a, b) => a.id.localeCompare(b.id));
    group.forEach((edge, index) => {
      const from = positions.get(edge.sourceId)!, to = positions.get(edge.targetId)!;
      const offset = (index - (group.length - 1) / 2) * 12;
      if (edge.sourceId === edge.targetId) {
        const radius = 24 + index * 8;
        paths.push({ edge, path: `M ${from.x} ${from.y - 8} C ${from.x + radius} ${from.y - radius * 2} ${from.x - radius} ${from.y - radius * 2} ${from.x} ${from.y - 8}` });
      } else {
        const distance = Math.hypot(to.x - from.x, to.y - from.y) || 1;
        const centerX = (from.x + to.x) / 2 - (to.y - from.y) / distance * offset;
        const centerY = (from.y + to.y) / 2 + (to.x - from.x) / distance * offset;
        paths.push({ edge, path: `M ${from.x} ${from.y} Q ${centerX} ${centerY} ${to.x} ${to.y}` });
      }
    });
  }
  return paths;
}
