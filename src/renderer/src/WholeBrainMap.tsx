import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { WholeBrainGraphNode, WholeBrainGraphProjection, WholeBrainInspectionApi, WholeBrainInspectionPage, WholeBrainInspectionQuery } from "../../shared/wholeBrainInspection";
import { projectWholeBrainInspection } from "../../shared/wholeBrainProjection";
import { BrainMapSubstrateGraph, type BrainMapGraphEdge, type BrainMapGraphNode } from "./BrainMapSubstrateGraph";
import { routerActivityGraph } from "./RouterActivityBrainMap";
import { checkedRouterViewport, retainRouterObservation } from "./RouterSpikeView";
import type { RouterSpikeActivity, RouterSpikeQuery } from "../../shared/routerActivity";
import { hydrateWholeBrainViewport, initialWholeBrainQueries, retainWholeBrainWindow, sameWholeBrainSnapshot, wholeBrainWindowPages, type WholeBrainViewport } from "../../shared/wholeBrainViewport";
import "./wholeBrainMap.css";

/** Only display coordinates are shared. Actual source identities stay intact. */
export function wholeBrainDisplayGraph(graph: WholeBrainGraphProjection, activity?: RouterSpikeActivity) {
  const live = activity?.available ? routerActivityGraph(activity) : undefined;
  const liveNodes = new Map(live?.nodes.map(node => [node.id, node]) ?? []);
  const liveEdges = new Map(live?.edges.map(edge => [edge.id, edge]) ?? []);
  const nodes: BrainMapGraphNode[] = graph.nodes.map(node => ({ id: node.id, label: node.label,
    region: node.regionIds.join(", "), activation: 0, activationObserved: false, importance: 0, members: [node.id],
    ...(node.kind === "substrate-cluster" && "count" in node.data ? { clusterCount: node.data.count, aggregateUnit: "neurons" as const,
      activation: node.data.meanActivation, activationObserved: true } : {}),
    ...(node.kind === "cortical-storage" && "logicalParameters" in node.data && node.data.logicalParameters !== null
      ? { clusterCount: node.data.logicalParameters, aggregateUnit: "logical parameters" as const } : {}),
    ...(node.source === "independent-live-router" && liveNodes.get(node.recordId)?.firingEvent
      ? { firingEvent: liveNodes.get(node.recordId)!.firingEvent, lastActivatedAt: liveNodes.get(node.recordId)!.lastActivatedAt } : {}) }));
  const edges: BrainMapGraphEdge[] = graph.edges.map(edge => {
    const data = edge.data, observed = edge.kind === "router-contact" ? liveEdges.get(edge.recordId)?.transmissionEvent : undefined;
    const aggregate = edge.kind === "substrate-pathway-aggregate" && data && "effectiveWeights" in data ? data : undefined;
    return { id: edge.id, sourceId: edge.sourceId, targetId: edge.targetId,
    effectiveWeight: edge.effectiveWeight ?? 0, connectionStrength: edge.strength,
    signedStrength: (edge.effectiveWeight ?? 0) * (edge.strength ?? 1),
    ...(edge.effectiveWeight !== undefined && edge.strength !== undefined ? { effectiveContribution: edge.effectiveWeight * edge.strength } : {}),
    stability: data && "stability" in data ? data.stability : Number.NaN, pathways: aggregate?.count ?? 1,
    ...(aggregate ? { aggregate: true, aggregateSignCounts: aggregate.effectiveWeights } : {}),
    ...(data && "isRepeatedContact" in data ? { isRepeatedContact: data.isRepeatedContact } : {}),
    ...(data && "parallelOrdinal" in data ? { parallelOrdinal: data.parallelOrdinal } : data && "ordinal" in data ? { parallelOrdinal: data.ordinal } : {}),
    ...(data && "parallelRootId" in data ? { parallelRootId: data.parallelRootId } : {}),
    ...(data && "eligibility" in data ? { eligibility: data.eligibility } : {}),
    ...(data && "plasticity" in data ? { plasticity: data.plasticity } : {}),
    ...(data && "uses" in data ? { uses: data.uses } : {}),
    ...(observed ? { transmissionEvent: observed } : {}),
    // A registered bridge/containment with unselected weights must never be
    // passed off as a learned zero connection or a recorded spike.
    ...(!aggregate && (!edge.learnedConnection || edge.effectiveWeight === undefined) ? { structuralRelationship: edge.kind } : {}),
    nativeEndpoints: { sourceId: data && "sourceId" in data ? data.sourceId : edge.sourceId,
      targetId: data && "targetId" in data ? data.targetId : edge.targetId } }; });
  return { nodes, edges };
}
/** Opening a real node changes only the viewport; it never creates a connection. */
export function wholeBrainMapFocusQuery(node: WholeBrainGraphNode): WholeBrainInspectionQuery | undefined {
  const regionId = node.scopeId === "root" ? undefined : node.scopeId;
  if (node.kind === "region") return { entity: "catalog", catalogFacet: "members", regionId: node.recordId, pageSize: 128 };
  if (node.kind === "registered-owner") return { entity: "catalog", catalogFacet: "owner", ownerId: node.recordId, pageSize: 128 };
  if (node.kind === "cortical-storage") return { entity: "cortex", regionId, pageSize: 64, cortex: { entity: "links", moduleId: node.recordId, row: 0 } };
  if (node.kind === "tensor-axis" && "moduleId" in node.data && "axis" in node.data)
    return { entity: "cortex", regionId, pageSize: 64, cortex: { entity: "links", moduleId: node.data.moduleId,
      row: node.data.axis === "tensor-row" || node.data.axis === "output-row" ? node.data.index : 0 } };
  if (node.kind === "substrate-cluster" && "effectiveWeights" in node.data) {
    const cluster = node.data;
    if (cluster.kind === "activation-band" && !cluster.activationBand) return undefined;
    return { entity: "substrate", regionId, pageSize: 64, substrate: cluster.kind === "pathway"
      ? { entity: "synapses", zoom: 1, sourceRegion: cluster.sourceRegion, targetRegion: cluster.targetRegion }
      : { entity: "neurons", zoom: 1, region: cluster.region,
        ...(cluster.kind === "activation-band" ? { activationBand: cluster.activationBand } : {}) } };
  }
  if (node.kind === "substrate-assembly") return { entity: "substrate", regionId, pageSize: 64,
    substrate: { entity: "assemblies", zoom: 1, membersOf: node.recordId } };
  if (node.kind === "substrate-neuron") return { entity: "substrate", regionId, pageSize: 64,
    substrate: { entity: "synapses", zoom: 1, connectedTo: node.recordId } };
  return undefined;
}
export function wholeBrainMapNextQuery(page: WholeBrainInspectionPage | undefined, query: WholeBrainInspectionQuery): WholeBrainInspectionQuery | undefined {
  if (!page) return undefined;
  if (page.hasMore && page.nextCursor) return { ...query, cursor: page.nextCursor };
  const module = page.cortex?.selectedModule, row = query.cortex?.row ?? 0;
  if (page.cortex?.entity === "links" && module && row + 1 < module.rows)
    return { ...query, cursor: undefined, cortex: { entity: "links", moduleId: module.id, row: row + 1 } };
  return undefined;
}
const initialMapWindowQueries = () => ({ parameters: initialWholeBrainQueries[1]!, aggregates: initialWholeBrainQueries[2]!, ideas: initialWholeBrainQueries[3]! });

export function WholeBrainMap(props: { brainId: string; api?: WholeBrainInspectionApi;
  observeActivity?(brainId: string, query: RouterSpikeQuery): Promise<RouterSpikeActivity>; onBack(): void }) {
  return <WholeBrainMapOwner key={props.brainId} {...props} />;
}
function WholeBrainMapOwner({ brainId, api, observeActivity, onBack }: Parameters<typeof WholeBrainMap>[0]) {
  const [activity, setActivity] = useState<RouterSpikeActivity>();
  const [overview, setOverview] = useState<WholeBrainInspectionPage>();
  const [page, setPage] = useState<WholeBrainInspectionPage>();
  const [windows, setWindows] = useState<WholeBrainViewport>({});
  const [anchors, setAnchors] = useState<{ region?: WholeBrainInspectionPage; owner?: WholeBrainInspectionPage }>({});
  const [selected, setSelected] = useState(""), [selectedEdgeId, setSelectedEdgeId] = useState("");
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const [previousQueries, setPreviousQueries] = useState<WholeBrainInspectionQuery[]>([]);
  const owner = useRef<object | null>(null), sequence = useRef(0);
  const overviewRef = useRef<WholeBrainInspectionPage | undefined>(undefined);
  const rootWindows = useRef<WholeBrainViewport>({});
  const windowQueries = useRef<Partial<Record<keyof WholeBrainViewport, WholeBrainInspectionQuery>>>({});
  const query = useRef<WholeBrainInspectionQuery>({ entity: "catalog", catalogFacet: "overview", pageSize: 128 });
  const busyRef = useRef(false);

  const read = useCallback(async (next: WholeBrainInspectionQuery, refresh = false, previous = false) => {
    if (!api || !owner.current || busyRef.current) return;
    const scope = owner.current, ticket = ++sequence.current;
    busyRef.current = true; setBusy(true); setError("");
    try {
      if (refresh) {
        const fresh = await hydrateWholeBrainViewport(api, brainId, () => scope === owner.current && sequence.current === ticket);
        projectWholeBrainInspection({ pages: [fresh.overview, ...wholeBrainWindowPages(fresh.windows)] });
        if (scope !== owner.current || sequence.current !== ticket) return;
        overviewRef.current = fresh.overview; rootWindows.current = fresh.windows; query.current = fresh.query;
        windowQueries.current = initialMapWindowQueries();
        setOverview(fresh.overview); setWindows(fresh.windows); setPage(fresh.active); setAnchors({});
        setSelected(""); setSelectedEdgeId(""); setPreviousQueries([]);
        return;
      }
      const value = await api.query(brainId, next);
      if (scope !== owner.current || sequence.current !== ticket) return;
      if (value.brainId !== brainId || value.entity !== next.entity) throw new Error("The map response does not match this brain.");
      const anchor = overviewRef.current;
      if (anchor && !sameWholeBrainSnapshot(anchor, value))
        throw new Error("The brain changed while viewing the map. Refresh to read its current connections.");
      if (next.entity === "catalog" && next.catalogFacet === "members") setAnchors({ region: value });
      if (next.entity === "catalog" && next.catalogFacet === "owner") setAnchors(current => ({ ...current, owner: value }));
      for (const key of Object.keys(retainWholeBrainWindow({}, value)) as Array<keyof WholeBrainViewport>) windowQueries.current[key] = next;
      const before = query.current;
      setPreviousQueries(values => previous ? values.slice(0, -1) : [...values, before]);
      query.current = next; setPage(value); setWindows(current => retainWholeBrainWindow(current, value));
    } catch (reason) {
      if (scope === owner.current && ticket === sequence.current)
        setError(reason instanceof Error ? reason.message : "Brain Map is unavailable.");
    } finally {
      if (scope === owner.current && ticket === sequence.current) { busyRef.current = false; setBusy(false); }
    }
  }, [api, brainId]);

  useEffect(() => {
    owner.current = {}; overviewRef.current = undefined; rootWindows.current = {}; windowQueries.current = {}; busyRef.current = false;
    setOverview(undefined); setPage(undefined); setWindows({}); setAnchors({});
    setSelected(""); setSelectedEdgeId(""); setPreviousQueries([]); setBusy(false); setError("");
    query.current = { entity: "catalog", catalogFacet: "overview", pageSize: 128 };
    if (api) void read(query.current, true);
    return () => { owner.current = null; sequence.current++; busyRef.current = false; };
  }, [api, brainId, read]);

  useEffect(() => {
    setActivity(undefined);
    if (!observeActivity || !globalThis.crypto?.randomUUID) return;
    // The existing observer reads an already-loaded owner. It cannot Load a brain.
    const capture = { id: crypto.randomUUID(), epoch: undefined as string | undefined, afterSequence: 0 };
    let current = true, timer: ReturnType<typeof setTimeout> | undefined;
    let transmissionEpoch: string | undefined, afterTransmissionSequence = 0;
    const stop = async () => {
      if (capture.epoch) await observeActivity(brainId, { enabled: false, captureId: capture.id,
        epoch: capture.epoch, afterSequence: capture.afterSequence });
    };
    const poll = async () => {
      try {
        const value = checkedRouterViewport(await observeActivity(brainId, { enabled: true, captureId: capture.id,
          start: 0, count: 64, pageSize: 32, contactOffset: 0, contactPageSize: 128,
          ...(capture.epoch ? { epoch: capture.epoch, afterSequence: capture.afterSequence } : {}),
          ...(transmissionEpoch ? { transmissionEpoch, afterTransmissionSequence } : {}), transmissionPageSize: 128 }), brainId, capture);
        if (value.epoch) capture.epoch = value.epoch;
        if (!current) { await stop(); return; }
        setActivity(previous => retainRouterObservation(previous, value));
        if (!value.available || !value.enabled || !value.epoch) return;
        capture.afterSequence = value.nextSequence;
        transmissionEpoch = value.transmissionPage?.epoch ?? undefined;
        afterTransmissionSequence = value.transmissionPage?.nextSequence ?? 0;
        timer = setTimeout(() => { void poll(); }, 400);
      } catch {
        // Missing measurements are not replaced with simulated firing.
        if (current) setActivity(undefined);
        void stop().catch(() => undefined);
      }
    };
    void poll();
    return () => { current = false; if (timer) clearTimeout(timer); void stop().catch(() => undefined); };
  }, [brainId, observeActivity]);

  const projected = useMemo(() => {
    if (!overview) return { graph: undefined, error: "" };
    const pages = [...new Set([overview, ...Object.values(anchors), page, ...wholeBrainWindowPages(windows)]
      .filter((value): value is WholeBrainInspectionPage => Boolean(value)))];
    try { return { graph: projectWholeBrainInspection({ pages, router: activity }), error: "" }; }
    catch (reason) { return { graph: undefined, error: reason instanceof Error ? reason.message : "The map sources changed." }; }
  }, [overview, anchors, page, windows, activity]);
  const display = useMemo(() => projected.graph ? wholeBrainDisplayGraph(projected.graph, activity) : undefined, [projected.graph, activity]);
  const focus = (id: string) => {
    setSelected(id); setSelectedEdgeId("");
    const node = projected.graph?.nodes.find(value => value.id === id);
    const next = node && wholeBrainMapFocusQuery(node);
    if (next && !busyRef.current) void read(next);
  };
  const wholeBrain = () => {
    if (!overview || busyRef.current) return;
    setPage(rootWindows.current.aggregates); setWindows(rootWindows.current); setAnchors({});
    setSelected(""); setSelectedEdgeId(""); setPreviousQueries([]); setError("");
    windowQueries.current = initialMapWindowQueries(); query.current = initialWholeBrainQueries[2]!;
  };
  const selectedLabel = projected.graph?.nodes.find(value => value.id === selected)?.label;
  const previous = previousQueries.at(-1);
  const nextArea = wholeBrainMapNextQuery(page, query.current) ?? (Object.entries(windows) as Array<[keyof WholeBrainViewport, WholeBrainInspectionPage]>)
    .map(([key, value]) => value === page || !windowQueries.current[key] ? undefined : wholeBrainMapNextQuery(value, windowQueries.current[key]!)).find(Boolean);
  const status = !api ? "Brain Map is unavailable in this UI-only preview."
    : busy ? "Reading the brain…" : error || projected.error;

  return <section className="whole-brain-map" aria-label="Whole brain map">
    <header className="whole-brain-map__header">
      <button type="button" className="button" onClick={onBack}>Back to chat</button>
      <h1>Brain Map</h1>
      {previousQueries.length ? <button type="button" className="button" disabled={busy} onClick={wholeBrain}>Whole brain</button> : null}
      <button type="button" className="button whole-brain-map__refresh" title="Refresh brain map" aria-label="Refresh brain map"
        disabled={!api || busy} onClick={() => void read({ entity: "catalog", catalogFacet: "overview", pageSize: 128 }, true)}>↻</button>
    </header>
    <div className="whole-brain-map__canvas" tabIndex={0} aria-label="Connected brain map"
      onKeyDown={event => {
        if (event.target !== event.currentTarget) return;
        if (event.key === "Escape") { event.preventDefault(); wholeBrain(); }
        if (event.key === "PageDown" && nextArea) { event.preventDefault(); void read(nextArea); }
        if (event.key === "PageUp" && previous) { event.preventDefault(); void read(previous, false, true); }
      }}>
      {display ? <BrainMapSubstrateGraph nodes={display.nodes} edges={display.edges} selectedId={selected}
        selectedEdgeId={selectedEdgeId} onSelectEdge={edge => {
          setSelectedEdgeId(edge.id); setSelected("");
          const actual = projected.graph?.edges.find(value => value.id === edge.id);
          if (actual?.kind === "substrate-pathway-aggregate" && actual.data && "effectiveWeights" in actual.data)
            void read({ entity: "substrate", regionId: actual.scopeId === "root" ? undefined : actual.scopeId, pageSize: 64,
              substrate: { entity: "synapses", zoom: 1, sourceRegion: actual.data.sourceRegion, targetRegion: actual.data.targetRegion } });
        }} onSelect={node => focus(node.id)} clustered /> : null}
      {!display && !status ? <p className="whole-brain-map__empty" role="status">No neural records are available yet.</p> : null}
      {status ? <p className="whole-brain-map__status" role={error || projected.error ? "alert" : "status"}>{status}</p> : null}
    </div>
    {selectedLabel || previous || nextArea ? <div className="whole-brain-map__navigation" aria-label="Map navigation">
      {selectedLabel ? <span>{selectedLabel}</span> : null}
      {previous ? <button type="button" className="button" disabled={busy} onClick={() => void read(previous, false, true)} aria-label="Previous map area">←</button> : null}
      {nextArea ? <button type="button" className="button" disabled={busy}
        onClick={() => void read(nextArea)} aria-label="Next map area">→</button> : null}
    </div> : null}
  </section>;
}
