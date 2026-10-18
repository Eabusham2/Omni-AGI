/** Pure protocol/projector fixtures only. No worker, brain, model, media or hardware runs. */
import { describe, expect, it } from "vitest";
import { checkedWholeBrainInspectionPage, checkedWholeBrainInspectionQuery } from "../src/main/wholeBrainInspectionProtocol";
import { projectWholeBrainInspection } from "../src/shared/wholeBrainProjection";
import type { WholeBrainCatalogRecord, WholeBrainInspectionPage } from "../src/shared/wholeBrainInspection";
import type { CortexModule, CortexPage } from "../src/shared/cortexInspection";
import type { SubstratePage, SubstrateSynapse } from "../src/shared/types";
import type { RouterSpikeActivity } from "../src/shared/routerActivity";

const brainId = "whole-brain-fixture", hash = "a".repeat(64);
const snapshot = { brainId, metadataSha256: hash, coreGeneration: hash, substrateGeneration: hash, attentionOverlaySha256: hash,
  connectedDescriptorSha256: null, connectedGraphStatesSha256: null, catalogSha256: hash };
const root: WholeBrainCatalogRecord = { kind: "region", id: "region-root", parentRegionId: null, regionKind: "whole-brain",
  label: "Whole brain", lineage: { brainId }, evidence: "actual-brain-owner" };
const owner: WholeBrainCatalogRecord = { kind: "owner", id: "owner-shared", canonicalAlias: "actual.shared", className: "NativePackedOwner",
  packedFields: [{ field: "_packed_forward_weight", shape: [1, 1] }], evidence: "actual-registered-object" };
const role = (id: string): WholeBrainCatalogRecord => ({ kind: "region", id, parentRegionId: root.id, label: id,
  regionKind: "registered-role", lineage: { registeredRoot: id }, evidence: "actual-trainable-role-reference" });
function page(records: WholeBrainCatalogRecord[] = [], extra: Partial<WholeBrainInspectionPage> = {}): WholeBrainInspectionPage {
  return { format: "omni-whole-brain-inspection", version: 1, brainId, snapshot, entity: "catalog", scope: { id: "root", sourceBrainId: brainId, nativeRegionPath: [] },
    catalog: { available: true, reason: "Actual tiny catalog fixture", records: records.length, shards: 1 }, records, anchors: [], hasMore: false,
    nextCursor: null, cortex: null, substrate: null, evidence: { topology: "registered-ownership-and-explicit-regional-bridges",
      semanticAxisMapping: false, tokenNeuronEquivalence: false, liveActivityJoined: false, snapshotConsistent: true,
      unavailable: ["No semantic-axis/channel mapping exists in this fixture"] }, ...extra };
}
const module: CortexModule = { id: "actual.saved._packed_forward_weight", module: "actual.saved", role: "core", field: "weight", group: "technical",
  rows: 1, columns: 1, packedColumns: 1, packedBytes: 1, logicalShape: [1, 1], logicalParameters: 1, layout: "row-packed",
  shapeEvidence: "native-connection-state-shape", activationObserved: false, activation: null };
function cortex(records: CortexPage["records"], entity: CortexPage["entity"] = "modules"): CortexPage {
  return { brainId, revision: hash, source: "committed-packed-native-cortex", entity, records, selectedModule: entity === "modules" ? null : module,
    total: records.length, offset: 0, returned: records.length, hasMore: false, groups: ["technical"], moduleCount: 1, packedBytes: 1,
    logicalParameters: 1, logicalInventoryComplete: true, activity: { observed: false, reason: "Committed weights have no live firing" },
    integrity: { generationManifestVerified: true, selectedTernaryCodesValidated: entity === "links", wholeRolePayloadScanned: false },
    relationships: { kind: "computational-dependencies-not-semantic-explanation", ideaInputs: [], tokenInputs: [] } };
}
const synapse: SubstrateSynapse = { id: "true-contact", sourceId: "real-neuron", targetId: "assembly-a", kind: "associates", effectiveWeight: 1,
  connectionStrength: 2, eligibility: 0, plasticity: 0.2, stability: 0.1, uses: 3 };
function substrate(): SubstratePage { return { brainId, queriedAt: "2026-10-15T00:00:00Z", revision: hash, entity: "synapses", zoom: 1,
  totals: { neurons: 1, assemblies: 2, synapses: 1 }, matched: 1, hasMore: false, clusters: [],
  neurons: [{ id: "real-neuron", label: "Actual fixture neuron", region: "semantic", activation: 0.1, importance: 0.5, uncertainty: 0.1, exposures: 1, aliases: [] }],
  assemblies: ["assembly-a", "assembly-b"].map(id => ({ id, label: id, region: "assembly", neuronIds: ["real-neuron"], childAssemblyIds: [], kind: "knowledge",
    source: "", confidence: 0.6, importance: 0.4, rehearsals: 1, retainsSourceText: false })), synapses: [{ ...synapse }] }; }

describe("one whole-brain exact source graph", () => {
  it("adds only verified sparse source-region membership and preserves the actual assembly/backing-neuron alias as distinct typed records", () => {
    const connected: WholeBrainCatalogRecord = { kind: "region", id: "actual-parent-region", parentRegionId: root.id,
      label: "Actual parent", regionKind: "connected-parent", lineage: { regionId: "parent", parentId: "source-parent", generationSha256: hash },
      nativeRegionPath: ["parent"], evidence: "actual-NativeRegion-identity" };
    const source = substrate(); source.brainId = "source-parent";
    source.neurons.push({ ...source.neurons[0]!, id: "assembly-a", label: "Actual shared assembly row", region: "assembly" });
    const regional = page([], { entity: "substrate", scope: { id: connected.id, sourceBrainId: "source-parent", nativeRegionPath: ["parent"] },
      anchors: [connected], substrate: source, sparseAliases: [{ assemblyId: "assembly-a", neuronId: "assembly-a", evidence: "actual-assembly_neuron_id" }] });
    const checked = checkedWholeBrainInspectionPage(regional, brainId, { entity: "substrate", regionId: connected.id });
    expect(checked.sparseAliases).toEqual(regional.sparseAliases);
    expect(() => checkedWholeBrainInspectionPage({ ...regional, sparseAliases: [{ assemblyId: "missing", neuronId: "missing", evidence: "actual-assembly_neuron_id" }] },
      brainId, { entity: "substrate", regionId: connected.id })).toThrow("shared assembly/neuron alias");
    const graph = projectWholeBrainInspection({ pages: [page([root]), regional] });
    expect(graph.nodes.filter(row => row.recordId === "assembly-a").map(row => row.kind).sort()).toEqual(["substrate-assembly", "substrate-neuron"]);
    expect(graph.nodes.filter(row => row.scopeId === connected.id).every(row => row.regionIds.includes(connected.id))).toBe(true);
    expect(graph.edges.find(row => row.kind === "assembly-neuron-alias")).toMatchObject({ learnedConnection: false, firingObserved: false });
    const known = new Set(graph.nodes.map(row => row.id));
    expect(graph.edges.filter(row => row.kind === "substrate-synapse").every(row => known.has(row.sourceId) && known.has(row.targetId))).toBe(true);
    const generic = projectWholeBrainInspection({ pages: [page([root, role("decoder-role")]), page([], { entity: "substrate", anchors: [role("decoder-role")], substrate: substrate() })] });
    expect(generic.nodes.filter(row => row.kind.startsWith("substrate-")).every(row => row.regionIds.includes(root.id) && !row.regionIds.includes("decoder-role"))).toBe(true);
    const parameterPage = page([], { entity: "cortex", cortex: cortex([{ moduleId: module.id, sourceAxis: "tensor-column", sourceIndex: 0,
      targetAxis: "tensor-row", targetIndex: 0, relationship: "packed-weight-coordinate", semanticExplanationVerified: false, value: 1 }], "links") });
    const parameters = projectWholeBrainInspection({ pages: [parameterPage] });
    expect(parameters.edges.filter(row => row.kind === "tensor-axis-ownership")).toHaveLength(2);
    expect(parameters.edges.filter(row => row.kind === "tensor-axis-ownership").every(row => !row.learnedConnection &&
      row.sourceId === parameters.nodes.find(node => node.kind === "cortical-storage")?.id)).toBe(true);
    expect(() => checkedWholeBrainInspectionPage(parameterPage, brainId, { entity: "cortex", cortex: { entity: "rows" } })).toThrow("source facet");
    expect(() => checkedWholeBrainInspectionPage(parameterPage, brainId, { entity: "cortex", cortex: { entity: "links", moduleId: "wrong-matrix" } })).toThrow("selected matrix");
    expect(() => checkedWholeBrainInspectionPage(page([], { entity: "substrate", substrate: substrate() }), brainId,
      { entity: "substrate", substrate: { entity: "assemblies" } })).toThrow("Sparse source facet");
  });
  it("retains one shared owner with two real memberships without calling structural lines learned synapses", () => {
    const records: WholeBrainCatalogRecord[] = [root, role("region-a"), role("region-b"), owner,
      ...["a", "b"].map((name): WholeBrainCatalogRecord => ({ kind: "membership", id: "membership-" + name,
        regionId: "region-" + name, ownerId: owner.id, evidence: "actual-registered-containment-not-semantic-neuron-overlap" }))];
    const projected = projectWholeBrainInspection({ pages: [checkedWholeBrainInspectionPage(page(records), brainId, { entity: "catalog" })] });
    expect(projected.nodes.filter(row => row.kind === "registered-owner")).toHaveLength(1);
    expect(projected.nodes.find(row => row.id === owner.id)?.regionIds).toEqual(["region-a", "region-b"]);
    expect(projected.edges.every(row => !row.learnedConnection && !row.firingObserved)).toBe(true);
    expect(projected.semanticAxisMapping).toBe(false); expect(projected.tokenNeuronEquivalence).toBe(false);
  });

  it("opens exact saved field identity only through registry bindings and preserves actual ternary matrix axes", () => {
    const binding: WholeBrainCatalogRecord = { kind: "inspection", id: "inspection-exact", ownerId: owner.id, role: "core", moduleId: module.id,
      tensorKey: module.id, field: "weight", evidence: "actual-save-role-and-registered-buffer-key" };
    const contacts = cortex([{ sourceAxis: "tensor-column", sourceIndex: 0, targetAxis: "tensor-row", targetIndex: 0,
      moduleId: module.id, relationship: "packed-weight-coordinate", semanticExplanationVerified: false, value: -1, strength: 4 }], "links");
    const projected = projectWholeBrainInspection({ pages: [page([root, owner, binding]), checkedWholeBrainInspectionPage(page([], { entity: "cortex", cortex: contacts }), brainId, { entity: "cortex" })] });
    expect(projected.inspectionBindings[0]?.moduleId).toBe(module.id);
    expect(projected.edges.find(row => row.kind === "packed-storage-ownership")?.sourceId).toBe(owner.id);
    expect(projected.edges.find(row => row.kind === "tensor-contact")).toMatchObject({ learnedConnection: true, effectiveWeight: -1, strength: 4, firingObserved: false });
    expect(projected.nodes.filter(row => row.kind === "tensor-axis")).toHaveLength(2);
    expect(projected.nodes.some(row => row.kind === "substrate-neuron")).toBe(false);
  });

  it("keeps true neuron/assembly kinds, real overlapping assembly membership and learned sparse contact IDs", () => {
    const projected = projectWholeBrainInspection({ pages: [checkedWholeBrainInspectionPage(page([], { entity: "substrate", substrate: substrate() }), brainId, { entity: "substrate" })] });
    expect(projected.nodes.filter(row => row.kind === "substrate-neuron")).toHaveLength(1);
    expect(projected.nodes.filter(row => row.kind === "substrate-assembly")).toHaveLength(2);
    expect(projected.edges.filter(row => row.kind === "assembly-membership").map(row => row.targetId)).toEqual([
      "substrate:root:real-neuron", "substrate:root:real-neuron"
    ]);
    expect(projected.edges.find(row => row.recordId === synapse.id)?.data).toEqual(synapse);
    expect(projected.nodes.every(row => !row.firingObserved)).toBe(true);
  });

  it("rejects incompatible owner/generation envelopes and fabricated semantic/activity claims", () => {
    expect(() => checkedWholeBrainInspectionPage(page([], { brainId: "foreign" }), brainId, { entity: "catalog" })).toThrow("identity");
    expect(() => projectWholeBrainInspection({ pages: [page(), page([], { snapshot: { ...snapshot, coreGeneration: "b".repeat(64) } })] })).toThrow("snapshots differ");
    expect(() => checkedWholeBrainInspectionPage({ ...page(), evidence: { ...page().evidence, tokenNeuronEquivalence: true } }, brainId, { entity: "catalog" })).toThrow("unsupported observation");
  });

  it("rejects contradictory same-ID catalog rows, nodes and edges even under one unchanged snapshot", () => {
    expect(() => projectWholeBrainInspection({ pages: [page([root]), page([{ ...root, label: "Contradictory same ID" }])] })).toThrow("record identity conflicts");
    const first = substrate(), changedNode = substrate(); changedNode.neurons[0]!.activation = 0.8;
    expect(() => projectWholeBrainInspection({ pages: [page([], { substrate: first }), page([], { substrate: changedNode })] })).toThrow("node identity conflicts");
    const changedEdge = substrate(); changedEdge.synapses[0]!.connectionStrength = 8;
    expect(() => projectWholeBrainInspection({ pages: [page([], { substrate: first }), page([], { substrate: changedEdge })] })).toThrow("edge identity conflicts");
  });

  it("keeps observed live LIF firing in an independently fenced capture, never joins it to semantic/cortical neurons", () => {
    const router: RouterSpikeActivity = { brainId, source: "actual-lif-hard-spikes", owner: "root.router.population", channelIdentity: "brainId:root.router.population:channelIndex",
      available: true, enabled: true, observed: true, captureId: "owned-capture", epoch: "e".repeat(32), totalChannels: 2, channelStart: 0,
      admittedCount: 2, requestedCount: 2, ringBytes: 64, frameCapacity: 2, lastSequence: 1, firstAvailableSequence: 1, nextSequence: 1,
      droppedFrames: 0, missedFrames: 0, hasMore: false, frames: [{ sequence: 1, observedAtUnix: 1, monotonicNs: "1000000000", channelStart: 0, spikes: [1, 0] }],
      semanticNeuronMapping: false, synapseFiringObserved: false };
    const projected = projectWholeBrainInspection({ pages: [page([root])], router });
    expect(projected.nodes.filter(row => row.source === "independent-live-router").map(row => row.firingObserved)).toEqual([true, false]);
    expect(projected.boundaries).toEqual(expect.arrayContaining([expect.objectContaining({ source: "independent-live-router", joinable: false, captureId: "owned-capture" })]));
    expect(projected.edges.filter(row => row.learnedConnection)).toEqual([]);
    expect(() => projectWholeBrainInspection({ pages: [page()], router: { ...router, brainId: "other" } })).toThrow("another brain");
  });

  it("accepts typed indexed facets but never renderer paths, topology or a forged delegated cursor", () => {
    expect(checkedWholeBrainInspectionQuery({ entity: "catalog", catalogFacet: "members", regionId: "actual-region", pageSize: 64 })).toMatchObject({ catalogFacet: "members" });
    expect(checkedWholeBrainInspectionQuery({ entity: "cortex", cortex: { entity: "modules", search: "" } })).toMatchObject({ cortex: { search: "" } });
    expect(() => checkedWholeBrainInspectionQuery({ storagePath: "/untrusted" })).toThrow("no source paths");
    expect(() => checkedWholeBrainInspectionQuery({ cortex: { cursor: "foreign" } })).toThrow("delegated source cursor");
    expect(() => checkedWholeBrainInspectionQuery({ pageSize: 0 })).toThrow("1–256");
  });
});
