import { useEffect, useMemo, useRef, useState } from "react";
import type { WholeBrainGraphProjection, WholeBrainInspectionApi, WholeBrainInspectionPage, WholeBrainInspectionQuery } from "../../shared/wholeBrainInspection";
import type { CortexModule } from "../../shared/cortexInspection";
import { projectWholeBrainInspection } from "../../shared/wholeBrainProjection";
import { BrainMapSubstrateGraph, type BrainMapGraphEdge, type BrainMapGraphNode } from "./BrainMapSubstrateGraph";
import { routerActivityGraph } from "./RouterActivityBrainMap";
import { RouterSpikeView } from "./RouterSpikeView";
import type { RouterSpikeActivity, RouterSpikeQuery } from "../../shared/routerActivity";
import { hydrateWholeBrainViewport, retainWholeBrainWindow, sameWholeBrainSnapshot, wholeBrainWindowPages, type WholeBrainViewport } from "../../shared/wholeBrainViewport";
import type { SubstrateCluster } from "../../shared/types";
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
export function WholeBrainMap(props: { brainId: string; api?: WholeBrainInspectionApi;
  observeActivity?(brainId: string, query: RouterSpikeQuery): Promise<RouterSpikeActivity>; onBack(): void }) {
  return <WholeBrainMapOwner key={props.brainId} {...props} />;
}
function WholeBrainMapOwner({ brainId, api, observeActivity, onBack }: Parameters<typeof WholeBrainMap>[0]) {
  const [activity, setActivity] = useState<RouterSpikeActivity>();
  const [overview, setOverview] = useState<WholeBrainInspectionPage>(), [page, setPage] = useState<WholeBrainInspectionPage>();
  const [regionContext, setRegionContext] = useState<WholeBrainInspectionPage>(), [ownerContext, setOwnerContext] = useState<WholeBrainInspectionPage>();
  const [contextRegionId, setContextRegionId] = useState<string>();
  const [windows, setWindows] = useState<WholeBrainViewport>({});
  const [layers, setLayers] = useState({ parameters: true, ideas: true, connections: true });
  const [query, setQuery] = useState<WholeBrainInspectionQuery>({ entity: "catalog", pageSize: 128 });
  const [selected, setSelected] = useState(""), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const [selectedEdgeId, setSelectedEdgeId] = useState("");
  const [row, setRow] = useState("0");
  const [previousQueries, setPreviousQueries] = useState<WholeBrainInspectionQuery[]>([]);
  const owner = useRef<object | null>(null), sequence = useRef(0), overviewRef = useRef<WholeBrainInspectionPage | undefined>(undefined);
  const read = async (next: WholeBrainInspectionQuery, refresh = false, previous = false) => {
    if (!api || !owner.current) return;
    const scope = owner.current, ticket = ++sequence.current; setBusy(true); setError("");
    try {
      if (refresh) {
        const fresh = await hydrateWholeBrainViewport(api, brainId, () => scope === owner.current && sequence.current === ticket);
        projectWholeBrainInspection({ pages: [fresh.overview, ...wholeBrainWindowPages(fresh.windows)] });
        overviewRef.current = fresh.overview; setOverview(fresh.overview); setWindows(fresh.windows); setPage(fresh.active); setQuery(fresh.query);
        setSelected(""); setSelectedEdgeId(""); setRegionContext(undefined); setOwnerContext(undefined); setContextRegionId(undefined); setPreviousQueries([]);
        return;
      }
      const value = await api.query(brainId, next);
      if (scope !== owner.current || sequence.current !== ticket) return;
      if (value.brainId !== brainId) throw new Error("The map response belongs to another instance.");
      const anchor = overviewRef.current;
      if (anchor && !sameWholeBrainSnapshot(anchor, value))
        throw new Error("The brain changed while browsing. Refresh the whole map to use one consistent snapshot.");
      if (refresh || !anchor) { overviewRef.current = value; setOverview(value); setSelected(""); setSelectedEdgeId(""); setRegionContext(undefined); setOwnerContext(undefined); setContextRegionId(undefined); }
      if (next.entity === "catalog" && next.catalogFacet === "members") { setRegionContext(value); setContextRegionId(next.regionId); setOwnerContext(undefined); }
      if (next.entity === "catalog" && next.catalogFacet === "owner") setOwnerContext(value);
      if (refresh) setPreviousQueries([]);
      else if (previous) setPreviousQueries(values => values.slice(0, -1));
      else setPreviousQueries(values => [...values, query]);
      setPage(value); setQuery(next);
      setWindows(current => retainWholeBrainWindow(current, value));
    } catch (reason) { if (scope === owner.current && ticket === sequence.current) setError(reason instanceof Error ? reason.message : "Brain Map is unavailable."); }
    finally { if (scope === owner.current && ticket === sequence.current) setBusy(false); }
  };
  useEffect(() => {
    const scope = {}; owner.current = scope; overviewRef.current = undefined;
    setOverview(undefined); setPage(undefined); setWindows({}); setRegionContext(undefined); setOwnerContext(undefined); setContextRegionId(undefined); setSelected(""); setSelectedEdgeId(""); setPreviousQueries([]); setBusy(false); setError("");
    if (api) void read({ entity: "catalog", catalogFacet: "overview", pageSize: 128 }, true);
    return () => { owner.current = null; sequence.current++; };
  }, [brainId, api]);
  const projected = useMemo(() => {
    if (!overview) return { graph: undefined, error: "" };
    // Retain the actual selected region/owner anchors during weight drill-down.
    // Only current inspection pages live in RAM; traversal is not a model cap.
    const pages = [...new Set([overview, regionContext, ownerContext, page, ...Object.values(windows)].filter((value): value is WholeBrainInspectionPage => Boolean(value)))];
    try { return { graph: projectWholeBrainInspection({ pages, router: activity }), error: "" }; }
    catch (reason) { return { graph: undefined, error: reason instanceof Error ? reason.message : "Inconsistent map sources." }; }
  }, [overview, regionContext, ownerContext, page, windows, activity]);
  const graph = projected.graph;
  const display = useMemo(() => {
    if (!graph) return undefined;
    const nodes = graph.nodes.filter(node => (layers.parameters || !["registered-owner", "cortical-storage", "tensor-axis"].includes(node.kind)) &&
      (layers.ideas || !node.kind.startsWith("substrate-")));
    const ids = new Set(nodes.map(node => node.id));
    return wholeBrainDisplayGraph({ ...graph, nodes, edges: graph.edges.filter(edge => ids.has(edge.sourceId) && ids.has(edge.targetId) &&
      (layers.connections || !edge.learnedConnection && edge.kind !== "substrate-pathway-aggregate")) }, activity);
  }, [graph, activity, layers]);
  const selectedNode = graph?.nodes.find(node => node.id === selected);
  const selectedEdge = graph?.edges.find(edge => edge.id === selectedEdgeId);
  const selectedRegion = selectedNode?.kind === "region" ? selectedNode.recordId : query.regionId ?? contextRegionId;
  const bindings = graph?.inspectionBindings.filter(record => record.ownerId === selectedNode?.recordId) ?? [];
  const catalogNodes = (page?.records ?? []).filter(record => record.kind === "region" || record.kind === "owner");
  const modules = (page?.cortex?.records ?? []).filter((record): record is CortexModule => "module" in record && "field" in record);
  const parameterInventory = (windows.parameters?.cortex?.records ?? []).filter((record): record is CortexModule => "module" in record && "field" in record);
  const aggregate = selectedNode && "effectiveWeights" in selectedNode.data ? selectedNode.data
    : selectedEdge?.kind === "substrate-pathway-aggregate" && selectedEdge.data && "effectiveWeights" in selectedEdge.data ? selectedEdge.data : undefined;
  const sparseRegion = aggregate?.region ?? query.substrate?.region;
  const sparseScope = windows.aggregates?.scope.nativeRegionPath.length ? windows.aggregates.scope.id : undefined;
  const zoom = windows.aggregates?.substrate?.zoom ?? 0;
  const inspectAggregate = (value: SubstrateCluster, scopeId: string) => {
    if (value.kind === "activation-band" && !value.activationBand) { setError("This source omitted its exact activation-group field. Refresh the map."); return; }
    void read({ entity: "substrate", regionId: scopeId === "root" ? undefined : scopeId, pageSize: 64,
    substrate: value.kind === "pathway" ? { entity: "synapses", zoom: 1, sourceRegion: value.sourceRegion, targetRegion: value.targetRegion }
      : { entity: "neurons", zoom: 1, region: value.region,
        ...(value.kind === "activation-band" ? { activationBand: value.activationBand } : {}) } }); };
  const changeFacet = (entity: "cortex" | "substrate") => void read({ entity, pageSize: 128,
    ...(selectedRegion ? { regionId: selectedRegion } : {}),
    ...(entity === "cortex" ? { cortex: { entity: "modules" } } : { substrate: { entity: "overview", zoom: 0 } }) });
  const openModule = (moduleId: string) => { setRow("0"); void read({ entity: "cortex", regionId: selectedRegion,
    pageSize: 64, cortex: { entity: "links", moduleId, row: 0 } }); };
  const inspectRow = () => {
    const module = page?.cortex?.selectedModule, value = Number(row);
    if (!module || !/^\d+$/.test(row) || !Number.isSafeInteger(value) || value < 0 || value >= module.rows) {
      setError("Choose a row inside the selected parameter matrix."); return;
    }
    void read({ entity: "cortex", regionId: query.regionId, pageSize: 64, cortex: { entity: "links", moduleId: module.id, row: value } });
  };
  return <section className="whole-brain-map" aria-label="Whole brain map">
    <header className="whole-brain-map__header"><button type="button" className="button" onClick={onBack}>Back to chat</button>
      <h1>Brain Map</h1><button type="button" className="button" disabled={!api || busy}
        onClick={() => void read({ entity: "catalog", catalogFacet: "overview", pageSize: 128 }, true)}>Refresh whole brain</button></header>
    <nav className="whole-brain-map__navigation" aria-label="Brain map location">
      <button type="button" className="button" disabled={!overview || busy} onClick={() => { setPage(overview); setRegionContext(undefined); setOwnerContext(undefined); setContextRegionId(undefined); setQuery({ entity: "catalog", catalogFacet: "overview", pageSize: 128 }); setPreviousQueries([]); setSelected(""); setSelectedEdgeId(""); setError(""); }}>Whole brain</button>
      <span>{page?.cortex?.entity === "modules" ? "Joint cortical inventory" : page?.scope.nativeRegionPath.length ? "Selected connected region" : query.regionId || contextRegionId ? "Selected region" : "All regions"}</span>
    </nav>
    <fieldset className="whole-brain-map__layers"><legend>Show together</legend>{(["parameters", "ideas", "connections"] as const).map(layer =>
      <label key={layer}><input type="checkbox" checked={layers[layer]} onChange={event => setLayers(current => ({ ...current, [layer]: event.target.checked }))} />
        {layer === "parameters" ? "Parameters and owners" : layer === "ideas" ? "Ideas and neurons" : "Connections"}</label>)}</fieldset>
    <div className="whole-brain-map__navigation" role="group" aria-label="Brain map source zoom">
      <span>Source detail</span>{([{ label: "Regions", value: 0 }, { label: "Activation groups", value: .5 }, { label: "Exact neurons", value: 1 }] as const).map(level =>
        <button type="button" className="button" key={level.value} disabled={!api || busy} aria-pressed={page?.entity === "substrate" && page.substrate?.zoom === level.value}
          onClick={() => void read({ entity: "substrate", regionId: sparseScope ?? selectedRegion, pageSize: 64,
            substrate: { entity: level.value === 1 ? "neurons" : "overview", zoom: level.value, ...(sparseRegion ? { region: sparseRegion } : {}) } })}>{level.label}</button>)}
      <span>{zoom < .35 ? "Region totals" : "Activation groups"} retained alongside parameters and ideas</span>
    </div>
    {!api ? <p role="status">Whole-brain inspection is unavailable; no substitute map or model was created.</p> : null}
    {busy ? <p role="status">Reading the selected neural records…</p> : null}
    {error || projected.error ? <p role="alert">{error || projected.error}</p> : null}
    {display ? <BrainMapSubstrateGraph nodes={display.nodes} edges={display.edges} selectedId={selected}
      selectedEdgeId={selectedEdgeId} onSelectEdge={edge => { setSelectedEdgeId(edge.id); setSelected(""); }}
      onSelect={node => { setSelected(node.id); setSelectedEdgeId(""); }} clustered /> : null}
    {graph ? <><p className="whole-brain-map__evidence">One brain · {(display?.nodes.length ?? 0).toLocaleString()} displayed records · {(display?.edges.length ?? 0).toLocaleString()} visible recorded relationships</p>
      <p>{windows.parameters?.cortex?.logicalInventoryComplete ? windows.parameters.cortex.logicalParameters.toLocaleString() : "Unverified total"} logical parameters · {windows.aggregates?.substrate?.totals.neurons.toLocaleString() ?? "—"} sparse neurons · {windows.ideas?.substrate?.totals.assemblies.toLocaleString() ?? "—"} idea assemblies. Counts describe different source kinds.</p>
      <details><summary>Map legend</summary><p>Grey dashed links show source structure. Aggregate pathways show exact contact counts and sign distributions, not representative weights. Individual contacts retain their own signs and strengths. Positions are a layout, not anatomy.</p></details></> : null}
    <div className="whole-brain-map__details">
      <aside aria-label="Selected brain structure"><h2>{selectedNode?.label ?? (selectedEdge ? "Selected connection" : "Inspect the brain")}</h2>
        {selectedEdge ? <><p>{selectedEdge.kind.replaceAll("-", " ")}</p>
          {selectedEdge.learnedConnection && selectedEdge.effectiveWeight !== undefined ? <p>Ternary sign: {selectedEdge.effectiveWeight > 0 ? "+1" : selectedEdge.effectiveWeight} · Strength: {selectedEdge.strength ?? "unreported"}</p>
            : <p>Registered relationship; not a measured synaptic weight.</p>}
          <details><summary>Connection values and provenance</summary><pre>{JSON.stringify(selectedEdge, null, 2)}</pre></details></> : null}
        {selectedNode ? <p>{selectedNode.kind.replaceAll("-", " ")}</p> : <p>Select a region, neural owner or neuron to inspect its connections.</p>}
        {aggregate ? <><p>{aggregate.count.toLocaleString()} {aggregate.kind === "pathway" ? "actual contacts" : "neurons in this aggregate"}
          {aggregate.kind === "pathway" ? ` · −1: ${aggregate.effectiveWeights.negative.toLocaleString()} · 0: ${aggregate.effectiveWeights.zero.toLocaleString()} · +1: ${aggregate.effectiveWeights.positive.toLocaleString()}` : ` · ${aggregate.activeCount.toLocaleString()} active; not a spike count`}</p>
          <button type="button" className="button" disabled={busy} onClick={() => inspectAggregate(aggregate, selectedNode?.scopeId ?? selectedEdge?.scopeId ?? "root")}>{aggregate.kind === "pathway" ? "Inspect exact pathway contacts" : aggregate.kind === "activation-band" ? "Inspect exact activation group" : "Inspect this region’s neurons"}</button></> : null}
        <div className="whole-brain-map__navigation">
          <button type="button" className="button" disabled={!api || busy} onClick={() => changeFacet("cortex")}>Browse parameter groups</button>
          <button type="button" className="button" disabled={!api || busy} onClick={() => changeFacet("substrate")}>Browse neuron groups</button>
          {selectedNode?.kind === "region" ? <button type="button" className="button" disabled={busy}
            onClick={() => void read({ entity: "catalog", catalogFacet: "members", regionId: selectedNode.recordId, pageSize: 128 })}>Open region</button> : null}
          {selectedNode?.kind === "registered-owner" ? <button type="button" className="button" disabled={busy}
            onClick={() => void read({ entity: "catalog", catalogFacet: "owner", ownerId: selectedNode.recordId, pageSize: 128 })}>Inspect selected owner</button> : null}
        </div>
        {bindings.map(binding => <button type="button" className="button" key={binding.id} disabled={busy}
          onClick={() => openModule(binding.moduleId)}>Inspect {binding.field} connections</button>)}
        {selectedNode?.kind === "cortical-storage" ? <button type="button" className="button" disabled={busy}
          onClick={() => openModule(selectedNode.recordId)}>Inspect these parameter connections</button> : null}
        {selectedNode?.kind === "substrate-assembly" ? <button type="button" className="button" disabled={busy}
          onClick={() => void read({ entity: "substrate", regionId: selectedNode.scopeId === "root" ? undefined : selectedNode.scopeId, pageSize: 64,
            substrate: { entity: "assemblies", zoom: 1, membersOf: selectedNode.recordId } })}>Members of this idea</button> : null}
        {page?.substrate || windows.ideas ? <div className="whole-brain-map__navigation">{(["neurons", "assemblies", "synapses"] as const).map(entity =>
          <button type="button" className="button" disabled={busy} key={entity}
            onClick={() => void read({ entity: "substrate", regionId: selectedNode?.kind.startsWith("substrate-") ? (selectedNode.scopeId === "root" ? undefined : selectedNode.scopeId) : query.regionId, pageSize: 128,
              substrate: { entity, zoom: 1, ...(entity === "synapses" && selectedNode && ["substrate-neuron", "substrate-assembly"].includes(selectedNode.kind)
                ? { connectedTo: selectedNode.recordId } : {}) } })}>{entity === "assemblies" ? "Idea assemblies" : entity === "synapses" ? "Synapses" : "Neurons"}</button>)}</div> : null}
        {selectedNode ? <details><summary>Exact source identity</summary><code>{selectedNode.recordId}</code><pre>{JSON.stringify(selectedNode.data, null, 2)}</pre></details> : null}
      </aside>
      <section aria-label="Paged brain records"><h2>Selected records</h2>
        <details><summary>Parameter groups on this viewport</summary><ul className="whole-brain-map__record-list">{parameterInventory.map(module =>
          <li key={module.id}><button type="button" className="button" disabled={busy} onClick={() => openModule(module.id)}>{module.module} · {module.field}
            {module.logicalParameters !== null ? ` · ${module.logicalParameters.toLocaleString()} logical parameters` : " · logical count unavailable"}</button></li>)}</ul></details>
        {page?.substrate?.clusters.length ? <ul className="whole-brain-map__record-list">{page.substrate.clusters.map(cluster => <li key={cluster.id}>
          <button type="button" className="button" disabled={busy} onClick={() => inspectAggregate(cluster, page.scope.id)}>{cluster.label} · {cluster.count.toLocaleString()} {cluster.kind === "pathway" ? "contacts" : "neurons"}</button></li>)}</ul> : null}
        {page?.substrate?.membershipPage ? <p>Exact idea memberships {(page.substrate.membershipPage.offset + (page.substrate.membershipPage.returned ? 1 : 0)).toLocaleString()}–{(page.substrate.membershipPage.offset + page.substrate.membershipPage.returned).toLocaleString()} of {page.substrate.membershipPage.total.toLocaleString()}. Original array ordinals are retained.</p> : null}
        {page?.cortex?.entity === "modules" ? <p>Joint cortical inventory. Select an owner’s exact connections for a region-specific weight view.</p> : null}
        {catalogNodes.length ? <ul className="whole-brain-map__record-list">{catalogNodes.map(record => <li key={record.id}><button type="button" className="button" aria-pressed={selected === record.id}
          title={record.kind === "owner" ? record.canonicalAlias : record.label} onClick={() => { setSelected(record.id); setSelectedEdgeId(""); }}>
            {record.kind === "owner" ? record.canonicalAlias : record.label}</button></li>)}</ul> : null}
        {modules.length ? <ul>{modules.map(module => <li key={module.id}><button type="button" className="button" disabled={busy}
          onClick={() => openModule(module.id)}>{module.module} · {module.field}</button></li>)}</ul> : null}
        {page?.cortex?.selectedModule ? <><p>{page.cortex.selectedModule.rows.toLocaleString()} rows · {page.cortex.selectedModule.columns?.toLocaleString() ?? "unknown"} columns</p>
          <div className="whole-brain-map__navigation"><label>Parameter row<input className="input" inputMode="numeric" aria-label="Parameter row" value={row} onChange={event => setRow(event.target.value)} /></label>
            <button type="button" className="button" disabled={busy} onClick={inspectRow}>View row</button></div></> : null}
        {graph?.edges.length ? <details><summary>Connections on this view</summary><ul className="whole-brain-map__record-list">{graph.edges.map((edge, index) => <li key={edge.id}>
          <button type="button" className="button" aria-pressed={edge.id === selectedEdgeId}
            onClick={() => { setSelectedEdgeId(edge.id); setSelected(""); }}>{edge.kind.replaceAll("-", " ")} {index + 1}</button></li>)}</ul></details> : null}
        <details><summary>Record values and provenance</summary><pre>{JSON.stringify(page?.cortex?.records ?? page?.substrate ?? page?.records ?? [], null, 2)}</pre></details>
        <div className="whole-brain-map__navigation">
          {previousQueries.length ? <button type="button" className="button" disabled={busy}
            onClick={() => void read(previousQueries.at(-1)!, false, true)}>Previous records</button> : null}
          {page?.hasMore && page.nextCursor ? <button type="button" className="button" disabled={busy}
            onClick={() => void read({ ...query, cursor: page.nextCursor! })}>Next records</button> : null}
          {page?.entity === "catalog" ? <button type="button" className="button" disabled={busy}
            onClick={() => void read({ entity: "catalog", catalogFacet: "all", pageSize: 128 })}>Browse all records</button> : null}
        </div>
      </section>
    </div>
    <details><summary>Activity inspection</summary><RouterSpikeView brainId={brainId} observe={observeActivity} showMap={false} onObservation={setActivity} />
      <p>Recorded activity stays in this graph. Live and committed identities are not joined without a verified source mapping.</p></details>
    {graph?.unavailable.length ? <details><summary>Unmapped or unavailable evidence</summary><ul>{graph.unavailable.map((reason, index) => <li key={`${index}:${reason}`}>{reason}</li>)}</ul></details> : null}
    {page ? <details><summary>Snapshot details</summary><p>{page.catalog.records.toLocaleString()} catalog records · {page.catalog.shards.toLocaleString()} shards</p><pre>{JSON.stringify(page.snapshot, null, 2)}</pre></details> : null}
  </section>;
}
