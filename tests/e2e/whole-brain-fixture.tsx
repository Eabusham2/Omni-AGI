import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { WholeBrainMap } from "../../src/renderer/src/WholeBrainMap";
import type { WholeBrainCatalogRecord, WholeBrainInspectionApi, WholeBrainInspectionPage, WholeBrainInspectionQuery, WholeBrainRegion } from "../../src/shared/wholeBrainInspection";
import type { CortexLink, CortexModule, CortexPage } from "../../src/shared/cortexInspection";
import type { SubstrateAssembly, SubstrateCluster, SubstrateNeuron, SubstratePage, SubstrateSynapse } from "../../src/shared/types";
import "../../src/renderer/src/styles.css";

const BRAIN = "synthetic-whole-brain";
const ROOT: WholeBrainRegion = { kind: "region", id: "fixture-whole-root", parentRegionId: null, label: "One whole brain · synthetic",
  regionKind: "whole-brain", lineage: { brainId: BRAIN }, evidence: "actual-brain-owner" };
const REGION_A: WholeBrainRegion = { kind: "region", id: "fixture-region-a", parentRegionId: ROOT.id, label: "Region A · shared pathways",
  regionKind: "connected-parent", lineage: { regionId: "fixture-A", parentId: "synthetic-parent-a", generationSha256: "c".repeat(64) },
  nativeRegionPath: ["fixture-A"], evidence: "actual-NativeRegion-identity" };
const REGION_B: WholeBrainRegion = { ...REGION_A, id: "fixture-region-b", label: "Region B · shared pathways",
  lineage: { regionId: "fixture-B", parentId: "synthetic-parent-b", generationSha256: "d".repeat(64) }, nativeRegionPath: ["fixture-B"] };
const OWNER: WholeBrainCatalogRecord = { kind: "owner", id: "fixture-shared-owner", canonicalAlias: "decoder.shared_projection",
  className: "SyntheticPackedOwner", packedFields: [{ field: "_packed_forward_weight", shape: [2, 1] }], evidence: "actual-registered-object" };
const SECOND_OWNER: WholeBrainCatalogRecord = { ...OWNER, id: "fixture-second-owner", canonicalAlias: "router.output_projection",
  packedFields: [{ field: "_packed_forward_weight", shape: [1, 1] }] };
const MODULE: CortexModule = { id: "decoder.shared_projection._packed_forward_weight", module: "decoder.shared_projection", role: "core", field: "weight",
  group: "synthetic-inventory", rows: 2, columns: 4, packedColumns: 1, packedBytes: 2, logicalShape: [2, 4], logicalParameters: 8, layout: "row-packed",
  shapeEvidence: "native-connection-state-shape", activationObserved: false, activation: null, connectionStateBytes: 8, structuralBytes: 0, activityOwner: "decoder.shared_projection" };
const SECOND_MODULE: CortexModule = { ...MODULE, id: "router.output_projection._packed_forward_weight", module: "router.output_projection", role: "plasticity",
  rows: 1, columns: 2, packedBytes: 1, logicalShape: [1, 2], logicalParameters: 2, connectionStateBytes: 2, activityOwner: "router.output_projection" };
const CATALOG: WholeBrainCatalogRecord[] = [ROOT, REGION_A, REGION_B, OWNER, SECOND_OWNER,
  { kind: "alias", id: "fixture-alias-a", alias: "decoder.shared_projection", ownerId: OWNER.id, evidence: "same-actual-registered-object" },
  { kind: "alias", id: "fixture-alias-b", alias: "modalities.shared_projection", ownerId: OWNER.id, evidence: "same-actual-registered-object" },
  { kind: "membership", id: "fixture-member-a", regionId: REGION_A.id, ownerId: OWNER.id, evidence: "actual-registered-containment-not-semantic-neuron-overlap" },
  { kind: "membership", id: "fixture-member-b", regionId: REGION_B.id, ownerId: OWNER.id, evidence: "actual-registered-containment-not-semantic-neuron-overlap" },
  { kind: "membership", id: "fixture-member-second", regionId: REGION_B.id, ownerId: SECOND_OWNER.id, evidence: "actual-registered-containment-not-semantic-neuron-overlap" },
  { kind: "inspection", id: "fixture-inspection-shared", ownerId: OWNER.id, role: "core", moduleId: MODULE.id, tensorKey: MODULE.id, field: "weight",
    evidence: "actual-save-role-and-registered-buffer-key" },
  { kind: "inspection", id: "fixture-inspection-second", ownerId: SECOND_OWNER.id, role: "plasticity", moduleId: SECOND_MODULE.id, tensorKey: SECOND_MODULE.id, field: "weight",
    evidence: "actual-save-role-and-registered-buffer-key" },
  { kind: "bridge", id: "fixture-exchange-bridge", sourceRegionId: REGION_A.id, targetRegionId: REGION_B.id, ownerId: OWNER.id, exchange: "workspace_exchange",
    evidence: "actual-RegionExchange-registration", learnedNonzeroObserved: false, firingObserved: false }];

const neuron = (id: string, label: string, region = "semantic"): SubstrateNeuron => ({ id, label, region, activation: 0,
  importance: 0.4, uncertainty: 0.2, exposures: 2, aliases: [] });
const SHARED_NEURON = neuron("fixture-sparse-shared", "Shared sparse member · synthetic");
const EXTRA_NEURON = neuron("fixture-sparse-extra", "Second exact member · synthetic");
const ASSEMBLIES: SubstrateAssembly[] = ["a", "b"].map(name => ({ id: "fixture-assembly-" + name, label: "Idea assembly " + name.toUpperCase() + " · synthetic",
  region: "assembly", neuronIds: [SHARED_NEURON.id], childAssemblyIds: [], neuronCount: 1, childAssemblyCount: 0, relationshipsPaged: false,
  kind: "knowledge", source: "synthetic-UI-fixture", confidence: 0.7, activation: 0, activationObserved: false, importance: 0.4, rehearsals: 2, retainsSourceText: false }));
ASSEMBLIES[0]!.neuronIds.push(EXTRA_NEURON.id);
ASSEMBLIES[0]!.neuronCount = 2;
ASSEMBLIES[0]!.childAssemblyIds = [ASSEMBLIES[1]!.id];
ASSEMBLIES[0]!.childAssemblyCount = 1;
const NEURONS = [SHARED_NEURON, EXTRA_NEURON, ...ASSEMBLIES.map(row => neuron(row.id, row.label + " backing row", "assembly"))];
const SYNAPSES: SubstrateSynapse[] = [{ id: "fixture-sparse-contact-a", sourceId: ASSEMBLIES[0]!.id, targetId: SHARED_NEURON.id, kind: "contains",
  effectiveWeight: 1, connectionStrength: 4, effectiveContribution: 4, eligibility: 0, plasticity: 0.3, stability: 0.4, uses: 6 },
  { id: "fixture-sparse-contact-b", sourceId: SHARED_NEURON.id, targetId: ASSEMBLIES[1]!.id, kind: "participates",
    effectiveWeight: -1, connectionStrength: 2, effectiveContribution: -2, eligibility: 0, plasticity: 0.5, stability: 0.1, uses: 3 }];

function fixturePage(generation: number, query: WholeBrainInspectionQuery): WholeBrainInspectionPage {
  const code = generation.toString(16).padStart(64, "0"), entity = query.entity ?? "catalog";
  const region = [ROOT, REGION_A, REGION_B].find(row => row.id === query.regionId);
  if (query.regionId && !region) throw new Error("Synthetic query selected an unknown region");
  const connected = entity === "substrate" && region?.regionKind === "connected-parent" ? region : undefined;
  const page: WholeBrainInspectionPage = { format: "omni-whole-brain-inspection", version: 1, brainId: BRAIN,
    snapshot: { brainId: BRAIN, metadataSha256: code, coreGeneration: code, substrateGeneration: code,
      attentionOverlaySha256: "e".repeat(64), connectedDescriptorSha256: "f".repeat(64), connectedGraphStatesSha256: "c".repeat(64), catalogSha256: code },
    entity, scope: { id: connected?.id ?? "root", sourceBrainId: connected?.lineage.parentId ?? BRAIN,
      nativeRegionPath: entity === "catalog" || connected ? region?.nativeRegionPath ?? [] : [] },
    catalog: { available: true, reason: "Explicit synthetic typed source fixture", records: CATALOG.length, shards: 2 },
    records: [], anchors: region ? [region] : [], sparseAliases: [], hasMore: false, nextCursor: null, cortex: null, substrate: null,
    evidence: { topology: "registered-ownership-and-explicit-regional-bridges", semanticAxisMapping: false, tokenNeuronEquivalence: false,
      liveActivityJoined: false, snapshotConsistent: true, unavailable: ["Synthetic fixture only: no native model, learned activity or semantic channel map was measured"] } };
  if (entity === "catalog") {
    const facet = query.catalogFacet ?? (query.ownerId ? "owner" : query.regionId ? "members" : "overview");
    if (facet === "overview") page.records = CATALOG.filter(row => row.kind === "region" || row.kind === "bridge");
    else if (facet === "owner") {
      const selected = CATALOG.find(row => row.kind === "owner" && row.id === query.ownerId);
      if (!selected) throw new Error("Synthetic query selected an unknown owner");
      page.records = CATALOG.filter(row => row.id === selected.id || "ownerId" in row && row.ownerId === selected.id);
      page.anchors.push(selected);
    } else if (facet === "members" && region && region.id !== ROOT.id) {
      const members = new Set(CATALOG.filter(row => row.kind === "membership" && row.regionId === region.id).map(row => row.kind === "membership" ? row.ownerId : ""));
      page.records = CATALOG.filter(row => row.id === region.id || row.kind === "owner" && members.has(row.id) || "ownerId" in row && members.has(row.ownerId));
    } else page.records = CATALOG;
    const actual = new Set(page.records.map(row => row.id));
    for (const row of page.records) for (const field of ["parentRegionId", "regionId", "ownerId", "sourceRegionId", "targetRegionId"] as const) {
      const reference = field in row ? row[field as keyof typeof row] : undefined;
      if (typeof reference === "string" && !actual.has(reference)) {
        const anchor = CATALOG.find(item => item.id === reference && (item.kind === "owner" || item.kind === "region"));
        if (anchor) { page.anchors.push(anchor); actual.add(anchor.id); }
      }
    }
  } else if (entity === "cortex") {
    const selected = query.cortex?.moduleId === SECOND_MODULE.id ? SECOND_MODULE : MODULE;
    const tail = Boolean(query.cursor), row = query.cortex?.row ?? 0;
    if (query.cursor && query.cursor !== `synthetic-${query.cortex?.entity ?? "modules"}-second-${generation}`) throw new Error("Synthetic cursor belongs to a previous snapshot");
    let records: CortexPage["records"];
    if ((query.cortex?.entity ?? "modules") === "modules") records = [tail ? SECOND_MODULE : MODULE];
    else {
      if (row >= selected.rows) throw new Error("Synthetic parameter row is outside the exact matrix");
      const columns = selected.columns ?? 0, first = tail ? Math.min(2, columns) : 0, end = tail ? columns : Math.min(2, columns);
      const links: CortexLink[] = Array.from({ length: end - first }, (_, index) => ({ moduleId: selected.id, sourceAxis: "tensor-column",
        sourceIndex: first + index, targetAxis: "tensor-row", targetIndex: row, relationship: "packed-weight-coordinate", semanticExplanationVerified: false,
        value: (first + index + row) % 2 ? -1 : 1, strength: 2 + first + index, resistance: 3, effectiveContribution: ((first + index + row) % 2 ? -1 : 1) * (2 + first + index),
        isRepeatedContact: false }));
      if (tail && columns > 2) links.push({ ...links[0]!, value: 1, strength: 3, resistance: 5, effectiveContribution: 3, isRepeatedContact: true, ordinal: 1 });
      records = links;
    }
    const modules = (query.cortex?.entity ?? "modules") === "modules", more = !tail && (modules || (selected.columns ?? 0) > 2);
    page.hasMore = more; page.nextCursor = more ? `synthetic-${modules ? "modules" : "links"}-second-${generation}` : null;
    page.cortex = { brainId: BRAIN, revision: code, source: "committed-packed-native-cortex", entity: modules ? "modules" : "links", records,
      selectedModule: modules ? null : selected, total: modules ? 2 : selected.columns ?? 0, offset: tail ? modules ? 1 : 2 : 0, returned: records.length,
      hasMore: more, nextCursor: page.nextCursor, groups: ["synthetic-inventory"], moduleCount: 2, packedBytes: 3, logicalParameters: 10,
      logicalInventoryComplete: true, connectionStateBytes: 10, structuralBytes: 0,
      activity: { observed: false, reason: "Synthetic committed DTO; no forward or firing occurred" },
      integrity: { generationManifestVerified: true, selectedTernaryCodesValidated: !modules, wholeRolePayloadScanned: false },
      relationships: { kind: "computational-dependencies-not-semantic-explanation", ideaInputs: [], tokenInputs: [] } };
    if (page.cortex.selectedModule) {
      const selectedMatrix = page.cortex.selectedModule;
      const binding = CATALOG.find((record): record is Extract<WholeBrainCatalogRecord, { kind: "inspection" }> => record.kind === "inspection" &&
        record.moduleId === selectedMatrix.id && record.tensorKey === selectedMatrix.id && record.role === selectedMatrix.role);
      if (binding) {
        page.anchors.push(binding);
        const owner = CATALOG.find(record => record.kind === "owner" && record.id === binding.ownerId);
        if (owner) page.anchors.push(owner);
        for (const membership of CATALOG) if (membership.kind === "membership" && membership.ownerId === binding.ownerId) {
          page.anchors.push(membership);
          const region = CATALOG.find(record => record.kind === "region" && record.id === membership.regionId);
          if (region) page.anchors.push(region);
        }
        page.anchors.push(ROOT);
      }
    }
  } else {
    const selected = query.substrate?.entity ?? "neurons", zoom = query.substrate?.zoom ?? 0, detailed = zoom >= 0.66;
    const regions: SubstrateCluster[] = ["semantic", "assembly"].map(region => ({ id: "region:" + region, label: region + " · synthetic aggregate", kind: "region", region,
      count: NEURONS.filter(row => row.region === region).length, activeCount: 0, meanActivation: 0, maxActivation: 0, effectiveWeights: { negative: 0, zero: 0, positive: 0 } }));
    const pathways: SubstrateCluster[] = ["assembly", "semantic"].map((source, index) => ({ id: "pathway:" + source + ">" + (index ? "assembly" : "semantic"), label: source + " → " + (index ? "assembly" : "semantic") + " · synthetic aggregate",
      kind: "pathway", sourceRegion: source, targetRegion: index ? "assembly" : "semantic", count: 1, activeCount: 0, meanActivation: 0, maxActivation: 0,
      effectiveWeights: { negative: index ? 1 : 0, zero: 0, positive: index ? 0 : 1 } }));
    const groups = zoom < .35 ? regions : regions.map(row => ({ ...row, id: row.id + ":quiet", kind: "activation-band" as const, activationBand: "quiet" as const, label: row.label + " · quiet" }));
    const matchesRegion = (id: string, region: string | undefined) => !region || NEURONS.find(row => row.id === id)?.region === region;
    const contacts = SYNAPSES.filter(row => (!query.substrate?.connectedTo || row.sourceId === query.substrate.connectedTo || row.targetId === query.substrate.connectedTo) &&
      matchesRegion(row.sourceId, query.substrate?.sourceRegion) && matchesRegion(row.targetId, query.substrate?.targetRegion));
    const sparse: SubstratePage = { brainId: page.scope.sourceBrainId, queriedAt: "2026-10-15T00:00:00.000Z", revision: code,
      entity: selected, zoom, totals: { neurons: NEURONS.length, assemblies: ASSEMBLIES.length, synapses: SYNAPSES.length },
      matched: selected === "assemblies" ? ASSEMBLIES.length : selected === "synapses" ? SYNAPSES.length : NEURONS.length,
      hasMore: false, clusters: detailed ? [] : [...groups, ...pathways].filter(row => !query.substrate?.region || row.region === query.substrate.region || row.sourceRegion === query.substrate.region || row.targetRegion === query.substrate.region),
      clusterAnchors: detailed ? [] : regions,
      neurons: detailed ? NEURONS.filter(row => selected !== "neurons" || matchesRegion(row.id, query.substrate?.region) && (!query.substrate?.activationBand || query.substrate.activationBand === "quiet")) : [],
      assemblies: detailed && selected === "assemblies" ? ASSEMBLIES : [],
      synapses: detailed && selected === "synapses" ? contacts : [],
      ...(query.substrate?.sourceRegion && query.substrate.targetRegion ? { pathwayFilter: { sourceRegion: query.substrate.sourceRegion, targetRegion: query.substrate.targetRegion } } : {}),
      ...(query.substrate?.activationBand ? { activationBand: query.substrate.activationBand } : {}) };
    if (detailed && query.substrate?.membersOf) {
      const parent = ASSEMBLIES.find(row => row.id === query.substrate?.membersOf);
      if (!parent) throw new Error("Synthetic member query selected an absent assembly");
      const all = [...parent.neuronIds.map((memberId, ordinal) => ({ assemblyId: parent.id, memberKind: "neuron" as const, memberId, ordinal })),
        ...parent.childAssemblyIds.map((memberId, ordinal) => ({ assemblyId: parent.id, memberKind: "assembly" as const, memberId, ordinal }))];
      const prefix = `synthetic-members-${parent.id}-${generation}-`, offset = query.cursor?.startsWith(prefix) ? Number(query.cursor.slice(prefix.length)) : 0;
      if (query.cursor && (!query.cursor.startsWith(prefix) || !Number.isSafeInteger(offset) || offset < 0 || offset >= all.length)) throw new Error("Synthetic exact-member cursor differs");
      const members = all.slice(offset, offset + 1); // Explicit one-record viewport exercises actual ordinal Next/Previous controls.
      sparse.membershipPage = { assemblyId: parent.id, total: all.length, offset, returned: members.length, members };
      sparse.assemblies = [{ ...parent, neuronIds: members.filter(row => row.memberKind === "neuron").map(row => row.memberId),
        childAssemblyIds: members.filter(row => row.memberKind === "assembly").map(row => row.memberId), relationshipsPaged: members.length < all.length },
        ...ASSEMBLIES.filter(row => members.some(member => member.memberKind === "assembly" && member.memberId === row.id)).map(row => ({ ...row, neuronIds: [], childAssemblyIds: [], relationshipsPaged: true }))];
      sparse.matched = all.length; sparse.offset = offset; sparse.returned = members.length;
      sparse.hasMore = offset + members.length < all.length; page.hasMore = sparse.hasMore;
      page.nextCursor = sparse.hasMore ? prefix + (offset + members.length) : null;
      if (page.nextCursor) sparse.nextCursor = page.nextCursor;
    } else if (detailed && selected === "synapses") sparse.matched = contacts.length;
    page.substrate = sparse;
    if (detailed && selected === "assemblies") page.sparseAliases = sparse.assemblies.map(row => ({ assemblyId: row.id, neuronId: row.id, evidence: "actual-assembly_neuron_id" }));
  }
  return structuredClone(page);
}

function Fixture() {
  const generation = useRef(1), failNextOwner = useRef(false), sequence = useRef(0);
  const [revision, setRevision] = useState(1), [dark, setDark] = useState(false), [compact, setCompact] = useState(false);
  const [receipts, setReceipts] = useState<string[]>(["Synthetic typed source adapter ready · no native runtime"]);
  const log = (text: string) => setReceipts(current => [...current.slice(-19), text]);
  const api = useMemo<WholeBrainInspectionApi>(() => ({ query: async (brainId, query = {}) => {
    const request = ++sequence.current;
    try {
      if (brainId !== BRAIN) throw new Error("Synthetic request belongs to another owner");
      const result = fixturePage(generation.current, query);
      if (failNextOwner.current) { failNextOwner.current = false; result.brainId = "synthetic-wrong-owner"; result.snapshot.brainId = result.brainId; }
      log(JSON.stringify({ synthetic: true, request, query, responseOwner: result.brainId, generation: generation.current,
        catalogRows: result.records.length, anchors: result.anchors.length, parameterRows: result.cortex?.records.length ?? 0,
        sparseNeurons: result.substrate?.neurons.length ?? 0, responseZoom: result.substrate?.zoom, aggregateRows: result.substrate?.clusters.length,
        exactMembers: result.substrate?.membershipPage, pathwayFilter: result.substrate?.pathwayFilter, nextPage: result.hasMore }));
      return result;
    } catch (error) { log(JSON.stringify({ synthetic: true, request, query, rejected: error instanceof Error ? error.message : String(error) })); throw error; }
  } }), []);
  return <main style={{ width: "100%", maxWidth: compact ? 360 : 1180, margin: "0 auto", height: "100dvh", overflow: "auto", padding: 12, boxSizing: "border-box" }}>
    <header style={{ padding: 12 }}><p className="eyebrow">Explicit synthetic UI fixture</p><h1>Production whole-brain view</h1>
      <p>Inert typed JSON only. No brain, native API/preload, model, Load, training, firing, filesystem, provider or learned-quality claim.</p>
      <p>Parameters, source aggregates and ideas remain together. Use source detail Regions/Activation groups/Exact neurons, or select an actual aggregate pathway and inspect its exact contacts.
        Idea A has three exact ordinal memberships across three synthetic one-record pages. Parameter and row views also have second pages.</p></header>
    <section aria-label="Synthetic whole-brain fixture controls" style={{ display: "flex", flexWrap: "wrap", gap: 8, padding: 12 }}>
      <button className="button" type="button" onClick={() => { generation.current++; setRevision(generation.current); log("Synthetic snapshot advanced; next browsing query must refuse mixed generations until Refresh whole brain."); }}>Advance synthetic snapshot</button>
      <button className="button" type="button" onClick={() => { failNextOwner.current = true; log("Next synthetic response will have the wrong owner; production view must refuse it."); }}>Wrong owner on next query</button>
      <button className="button" type="button" onClick={() => { failNextOwner.current = false; log("Synthetic fault cleared; Refresh whole brain to recover."); }}>Clear synthetic fault</button>
      <button className="button" type="button" aria-pressed={dark} onClick={() => { setDark(value => !value); document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; }}>Switch theme</button>
      <button className="button" type="button" aria-pressed={compact} onClick={() => setCompact(value => !value)}>Compact content · 360px</button>
      <p role="status">Synthetic snapshot {revision} · no runtime effects</p>
    </section>
    <WholeBrainMap brainId={BRAIN} api={api} onBack={() => log("Synthetic Back callback · no navigation or runtime effect")} />
    <section aria-labelledby="whole-brain-fixture-receipts" style={{ padding: 12 }}><h2 id="whole-brain-fixture-receipts">Synthetic query receipts</h2>
      <ol aria-live="polite">{receipts.map((value, index) => <li key={index}><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{value}</pre></li>)}</ol>
      <p><strong>Neural work:</strong> 0 · <strong>Provider calls:</strong> 0 · <strong>Native writes:</strong> 0</p>
    </section>
  </main>;
}

document.documentElement.dataset.colorScheme = "light";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
