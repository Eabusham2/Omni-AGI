/** New aggregate/member/combined-renderer-window contracts. No worker/model/Load. */
import { expect, it } from "vitest";
import { checkedWholeBrainInspectionPage, checkedWholeBrainInspectionQuery } from "../src/main/wholeBrainInspectionProtocol";
import { projectWholeBrainInspection } from "../src/shared/wholeBrainProjection";
import { hydrateWholeBrainViewport, retainWholeBrainWindow } from "../src/shared/wholeBrainViewport";
import { wholeBrainDisplayGraph } from "../src/renderer/src/WholeBrainMap";
import { graphEdgeTitle, graphNodeTitle } from "../src/renderer/src/brainMapGraphPresentation";
import type { WholeBrainInspectionPage, WholeBrainInspectionQuery } from "../src/shared/wholeBrainInspection";
import type { SubstratePage, SubstrateCluster, SubstrateAssembly } from "../src/shared/types";
import type { CortexModule, CortexPage } from "../src/shared/cortexInspection";

const brainId = "combined-fixture", hash = "a".repeat(64);
function page(extra: Partial<WholeBrainInspectionPage> = {}): WholeBrainInspectionPage {
  return { format: "omni-whole-brain-inspection", version: 1, brainId, snapshot: { brainId, metadataSha256: hash, coreGeneration: hash,
    substrateGeneration: hash, attentionOverlaySha256: hash, connectedDescriptorSha256: null, connectedGraphStatesSha256: null, catalogSha256: hash },
    entity: "catalog", scope: { id: "root", sourceBrainId: brainId, nativeRegionPath: [] }, catalog: { available: true, reason: "Synthetic constructor-free source", records: 1, shards: 1 },
    records: [{ kind: "region", id: "actual-root", label: "Whole fixture", parentRegionId: null, regionKind: "whole-brain", lineage: { brainId }, evidence: "actual-brain-owner" }],
    anchors: [], hasMore: false, nextCursor: null, cortex: null, substrate: null, evidence: { topology: "registered-ownership-and-explicit-regional-bridges",
      semanticAxisMapping: false, tokenNeuronEquivalence: false, liveActivityJoined: false, snapshotConsistent: true, unavailable: [] }, ...extra };
}
const cluster = (region: string, count: number): SubstrateCluster => ({ id: "region:" + region, label: region, kind: "region", region,
  count, activeCount: 0, meanActivation: 0, maxActivation: 0, effectiveWeights: { negative: 0, zero: 0, positive: 0 } });
const pathway: SubstrateCluster = { id: "pathway:input>ideas", label: "input → ideas", kind: "pathway", sourceRegion: "input", targetRegion: "ideas",
  count: 20_003, activeCount: 0, meanActivation: 0, maxActivation: 0, effectiveWeights: { negative: 10_000, zero: 3, positive: 10_000 } };
const assembly: SubstrateAssembly = { id: "idea", label: "Actual fixture idea", region: "assembly", neuronIds: [], childAssemblyIds: [], neuronCount: 5000,
  childAssemblyCount: 0, relationshipsPaged: true, kind: "knowledge", source: "fixture", confidence: .5, importance: .5, rehearsals: 1, retainsSourceText: false };
function sparse(extra: Partial<SubstratePage> = {}): SubstratePage { return { brainId, queriedAt: "2026-10-15T00:00:00Z", revision: hash,
  entity: "overview", zoom: 0, totals: { neurons: 5001, assemblies: 1, synapses: 20_003 }, matched: 1, hasMore: false,
  clusters: [pathway], clusterAnchors: [cluster("input", 1), cluster("ideas", 5000)], neurons: [], assemblies: [], synapses: [], ...extra }; }
const module: CortexModule = { id: "actual._packed_forward_weight", module: "actual", role: "core", field: "weight", group: "actual-group", rows: 2, columns: 4,
  packedColumns: 1, packedBytes: 2, logicalShape: [2, 4], logicalParameters: 8, layout: "row-packed", shapeEvidence: "native-connection-state-shape", activation: null, activationObserved: false };
const cortex: CortexPage = { brainId, revision: hash, source: "committed-packed-native-cortex", entity: "modules", records: [module], selectedModule: null,
  total: 1, offset: 0, returned: 1, hasMore: false, groups: ["actual-group"], moduleCount: 1, packedBytes: 2, logicalParameters: 8, logicalInventoryComplete: true,
  activity: { observed: false, reason: "No firing source" }, integrity: { generationManifestVerified: true, selectedTernaryCodesValidated: false, wholeRolePayloadScanned: false },
  relationships: { kind: "computational-dependencies-not-semantic-explanation", ideaInputs: [], tokenInputs: [] } };

it("draws actual aggregate pathways with verified off-page region anchors and no representative contact sign", () => {
  const source = page({ entity: "substrate", records: [], substrate: sparse() });
  const checked = checkedWholeBrainInspectionPage(source, brainId, { entity: "substrate", substrate: { entity: "overview", zoom: 0 } });
  const graph = projectWholeBrainInspection({ pages: [page(), checked] });
  const aggregate = graph.edges.find(edge => edge.kind === "substrate-pathway-aggregate")!;
  const known = new Set(graph.nodes.map(node => node.id));
  expect(known.has(aggregate.sourceId) && known.has(aggregate.targetId)).toBe(true);
  expect(aggregate.effectiveWeight).toBeUndefined(); expect(aggregate.learnedConnection).toBe(false);
  const display = wholeBrainDisplayGraph(graph), contact = display.edges.find(edge => edge.id === aggregate.id)!;
  expect(contact.aggregateSignCounts).toEqual(pathway.effectiveWeights);
  expect(graphEdgeTitle(contact)).toContain("20003 stored contacts");
  expect(graphEdgeTitle(contact)).not.toContain("Net display classification");
  expect(contact.structuralRelationship).toBeUndefined();
  expect(display.nodes.find(node => node.id === aggregate.targetId)?.clusterCount).toBe(5000);
  const bad = sparse({ clusters: [{ ...pathway, count: 99 }] });
  expect(() => checkedWholeBrainInspectionPage(page({ entity: "substrate", records: [], substrate: bad }), brainId, { entity: "substrate" })).toThrow("sign counts");
  const unavailable = projectWholeBrainInspection({ pages: [page({ substrate: sparse({ clusterAnchors: [] }) })] });
  expect(unavailable.edges.some(edge => edge.kind === "substrate-pathway-aggregate")).toBe(false);
  expect(unavailable.unavailable[0]).toContain("no endpoint was fabricated");
});

it("keeps actual assembly identity while merging distinct ordinal member windows, never replaces a large list with fake paging", () => {
  const memberPage = (ordinal: number) => page({ entity: "substrate", records: [], substrate: sparse({ entity: "assemblies", zoom: 1, clusters: [], clusterAnchors: [], matched: 5000,
    assemblies: [{ ...assembly, neuronIds: ["member-" + ordinal] }], neurons: [{ id: "member-" + ordinal, label: "member", region: "ideas", activation: 0, importance: .5, uncertainty: .2, exposures: 1, aliases: [] }],
    membershipPage: { assemblyId: "idea", total: 5000, offset: ordinal, returned: 1, members: [{ assemblyId: "idea", memberKind: "neuron", memberId: "member-" + ordinal, ordinal }] } }) });
  const first = checkedWholeBrainInspectionPage(memberPage(0), brainId, { entity: "substrate", substrate: { entity: "assemblies", membersOf: "idea", zoom: 1 } });
  const tail = checkedWholeBrainInspectionPage(memberPage(4999), brainId, { entity: "substrate", substrate: { entity: "assemblies", membersOf: "idea", zoom: 1 } });
  const graph = projectWholeBrainInspection({ pages: [first, tail] });
  expect(graph.nodes.filter(node => node.kind === "substrate-assembly")).toHaveLength(1);
  expect(graph.edges.filter(edge => edge.kind === "assembly-membership").map(edge => edge.recordId)).toEqual(["idea:neurons:0:member-0", "idea:neurons:4999:member-4999"]);
  expect(() => checkedWholeBrainInspectionPage(first, brainId, { entity: "substrate", substrate: { membersOf: "other" } })).toThrow("member source");
  expect(checkedWholeBrainInspectionQuery({ entity: "substrate", substrate: { entity: "synapses", zoom: 1, sourceRegion: "input", targetRegion: "ideas" } }).substrate?.sourceRegion).toBe("input");
});

it("sequentially hydrates one fenced combined viewport and retains parameter inventory when sparse detail changes", async () => {
  let active = 0, maximum = 0;
  const requests: WholeBrainInspectionQuery[] = [];
  const refreshed = await hydrateWholeBrainViewport({ query: async (_, query = {}) => {
    active++; maximum = Math.max(maximum, active); requests.push(query); await Promise.resolve(); active--;
    return query.entity === "cortex" ? page({ entity: "cortex", records: [], cortex }) : query.entity === "substrate"
      ? page({ entity: "substrate", records: [], substrate: sparse(query.substrate?.entity === "assemblies" ? { entity: "assemblies", zoom: 1, clusters: [], clusterAnchors: [], assemblies: [assembly] } : {}) }) : page();
  } }, brainId, () => true);
  expect(maximum).toBe(1); expect(requests).toHaveLength(4);
  const next = retainWholeBrainWindow(refreshed.windows, page({ entity: "substrate", records: [], substrate: sparse({ entity: "neurons", zoom: 1, clusters: [], clusterAnchors: [] }) }));
  expect(next.parameters).toBe(refreshed.windows.parameters); expect(next.ideas).toBe(refreshed.windows.ideas);
  const graph = projectWholeBrainInspection({ pages: [refreshed.overview, ...Object.values(next)] });
  expect(graph.nodes.some(node => node.kind === "cortical-storage")).toBe(true);
  expect(graph.nodes.some(node => node.kind === "substrate-assembly")).toBe(true);
  expect(graph.edges.some(edge => edge.kind === "substrate-pathway-aggregate")).toBe(true);
  expect(graphNodeTitle(wholeBrainDisplayGraph(graph).nodes.find(node => node.id.includes("cortex-storage"))!)).toContain("8 logical parameters");
});

it("refuses mixed generations or owners before publishing any replacement viewport", async () => {
  let calls = 0;
  await expect(hydrateWholeBrainViewport({ query: async () => { const result = page(); if (++calls === 2) result.snapshot.coreGeneration = "b".repeat(64); return result; } }, brainId, () => true)).rejects.toThrow("previous consistent map was kept");
  expect(calls).toBe(2);
  await expect(hydrateWholeBrainViewport({ query: async () => page({ brainId: "foreign" }) }, brainId, () => true)).rejects.toThrow("brain changed");
});

it("requires exact current activation-band and ordered-pathway receipts, not name-derived membership", () => {
  const neurons = [{ id: "input-neuron", label: "input", region: "input", activation: 0, importance: .5, uncertainty: .2, exposures: 1, aliases: [] },
    { id: "idea-neuron", label: "idea", region: "ideas", activation: 0, importance: .5, uncertainty: .2, exposures: 1, aliases: [] }];
  const source = sparse({ entity: "synapses", zoom: 1, clusters: [], clusterAnchors: [], neurons,
    synapses: [{ id: "actual-directed-contact", sourceId: neurons[0]!.id, targetId: neurons[1]!.id, kind: "association", effectiveWeight: -1, eligibility: 0, plasticity: .2, stability: .3, uses: 1 }],
    pathwayFilter: { sourceRegion: "input", targetRegion: "ideas" } });
  const query: WholeBrainInspectionQuery = { entity: "substrate", substrate: { entity: "synapses", zoom: 1, sourceRegion: "input", targetRegion: "ideas" } };
  expect(checkedWholeBrainInspectionPage(page({ entity: "substrate", records: [], substrate: source }), brainId, query).substrate?.synapses).toHaveLength(1);
  expect(() => checkedWholeBrainInspectionPage(page({ entity: "substrate", records: [], substrate: { ...source, neurons: [{ ...neurons[0]!, region: "wrong" }, neurons[1]!] } }), brainId, query)).toThrow("different actual endpoint region");
  const band = sparse({ entity: "neurons", zoom: 1, clusters: [], clusterAnchors: [], neurons, activationBand: "quiet" });
  expect(checkedWholeBrainInspectionPage(page({ entity: "substrate", records: [], substrate: band }), brainId, { entity: "substrate", substrate: { activationBand: "quiet" } }).substrate?.neurons).toHaveLength(2);
  expect(() => checkedWholeBrainInspectionPage(page({ entity: "substrate", records: [], substrate: { ...band, neurons: [{ ...neurons[0]!, activation: .9 }] } }), brainId,
    { entity: "substrate", substrate: { activationBand: "quiet" } })).toThrow("activation-band source");
});
