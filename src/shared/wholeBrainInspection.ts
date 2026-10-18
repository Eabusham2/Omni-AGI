import type { CortexLink, CortexModule, CortexPage, CortexQuery } from "./cortexInspection";
import type { SubstrateAssembly, SubstrateAssemblyMember, SubstrateCluster, SubstrateNeuron, SubstratePage, SubstrateQuery, SubstrateSynapse } from "./types";
import type { RouterContactRecord, RouterSpikeActivity } from "./routerActivity";

export interface WholeBrainSnapshotFence {
  brainId: string;
  metadataSha256: string;
  coreGeneration: string;
  substrateGeneration: string;
  attentionOverlaySha256: string;
  connectedDescriptorSha256: string | null;
  connectedGraphStatesSha256: string | null;
  catalogSha256: string | null;
}
export interface WholeBrainRegion {
  kind: "region"; id: string; parentRegionId: string | null; label: string;
  regionKind: "whole-brain" | "registered-role" | "connected-parent";
  lineage: { brainId?: string; registeredRoot?: string; regionId?: string; parentId?: string; generationSha256?: string };
  nativeRegionPath?: string[];
  evidence: "actual-brain-owner" | "actual-trainable-role-reference" | "actual-NativeRegion-identity";
}
export interface WholeBrainRegisteredOwner {
  kind: "owner"; id: string; canonicalAlias: string; className: string;
  packedFields: Array<{ field: string; shape: number[] }>;
  evidence: "actual-registered-object";
}
export interface WholeBrainAlias {
  kind: "alias"; id: string; alias: string; ownerId: string; evidence: "same-actual-registered-object";
}
export interface WholeBrainMembership {
  kind: "membership"; id: string; regionId: string; ownerId: string;
  evidence: "actual-registered-containment-not-semantic-neuron-overlap";
}
export interface WholeBrainBridge {
  kind: "bridge"; id: string; sourceRegionId: string; targetRegionId: string; ownerId: string;
  exchange: "workspace_exchange" | "cortical_exchange";
  evidence: "actual-RegionExchange-registration"; learnedNonzeroObserved: false; firingObserved: false;
}
export interface WholeBrainOwnerInspection {
  kind: "inspection"; id: string; ownerId: string; role: "core" | "plasticity";
  moduleId: string; tensorKey: string; field: "weight" | "bias";
  evidence: "actual-save-role-and-registered-buffer-key";
}
export type WholeBrainCatalogRecord = WholeBrainRegion | WholeBrainRegisteredOwner | WholeBrainAlias | WholeBrainMembership | WholeBrainBridge | WholeBrainOwnerInspection;
export interface WholeBrainAssemblyNeuronAlias {
  assemblyId: string; neuronId: string; evidence: "actual-assembly_neuron_id";
}

/** Source facets page ONE graph; they are not separate brain/map identities. */
export interface WholeBrainInspectionQuery {
  entity?: "catalog" | "cortex" | "substrate";
  cursor?: string; pageSize?: number;
  catalogFacet?: "overview" | "all" | "members" | "owner";
  ownerId?: string;
  /** Exact catalog region ID. Omit for the current root graph. */
  regionId?: string;
  cortex?: CortexQuery;
  substrate?: SubstrateQuery;
}
export interface WholeBrainInspectionPage {
  format: "omni-whole-brain-inspection"; version: 1; brainId: string;
  snapshot: WholeBrainSnapshotFence;
  entity: "catalog" | "cortex" | "substrate";
  scope: { id: string; sourceBrainId: string; nativeRegionPath: string[] };
  catalog: { available: boolean; reason: string; records: number; shards: number };
  records: WholeBrainCatalogRecord[];
  /** Real indexed endpoint records, not placeholders manufactured from IDs. */
  anchors: WholeBrainCatalogRecord[];
  sparseAliases?: WholeBrainAssemblyNeuronAlias[];
  hasMore: boolean; nextCursor: string | null;
  cortex: CortexPage | null;
  substrate: SubstratePage | null;
  evidence: {
    topology: "registered-ownership-and-explicit-regional-bridges";
    semanticAxisMapping: false; tokenNeuronEquivalence: false;
    liveActivityJoined: false; snapshotConsistent: true;
    unavailable: string[];
  };
}
export interface WholeBrainInspectionApi {
  query(brainId: string, query?: WholeBrainInspectionQuery): Promise<WholeBrainInspectionPage>;
}

export type WholeBrainGraphNodeKind = "region" | "registered-owner" | "cortical-storage" | "tensor-axis" | "substrate-neuron" | "substrate-assembly" | "substrate-cluster" | "substrate-pathway-summary" | "router-channel";
export interface WholeBrainGraphNode {
  id: string; recordId: string; kind: WholeBrainGraphNodeKind; label: string; scopeId: string;
  source: "committed" | "independent-live-router";
  regionIds: string[];
  /** True only for an actual observed event, never inferred from a stored sign. */
  firingObserved: boolean;
  data: WholeBrainRegion | WholeBrainRegisteredOwner | CortexModule | SubstrateNeuron | SubstrateAssembly | SubstrateCluster |
    Omit<SubstrateAssembly, "neuronIds" | "childAssemblyIds" | "relationshipsPaged"> |
    { moduleId: string; axis: "tensor-column" | "tensor-row" | "constant-one" | "output-row"; index: number } |
    { channel: number; captureId: string; epoch: string; owner: "root.router.population" };
}
export type WholeBrainGraphEdgeKind = "registered-containment" | "registered-membership" | "packed-storage-ownership" | "regional-trainable-bridge" |
  "tensor-contact" | "tensor-axis-ownership" | "substrate-synapse" | "substrate-pathway-aggregate" | "aggregate-band-membership" | "assembly-membership" | "assembly-neuron-alias" | "source-region-membership" | "router-contact";
export interface WholeBrainGraphEdge {
  id: string; recordId: string; kind: WholeBrainGraphEdgeKind; sourceId: string; targetId: string; scopeId: string;
  learnedConnection: boolean; firingObserved: false;
  effectiveWeight?: -1 | 0 | 1; strength?: number; resistance?: number;
  data?: WholeBrainBridge | WholeBrainOwnerInspection | WholeBrainAssemblyNeuronAlias | SubstrateCluster | SubstrateAssemblyMember | CortexLink | SubstrateSynapse | RouterContactRecord;
}
export interface WholeBrainGraphProjection {
  brainId: string; snapshot: WholeBrainSnapshotFence | null;
  nodes: WholeBrainGraphNode[]; edges: WholeBrainGraphEdge[];
  aliases: WholeBrainAlias[];
  inspectionBindings: WholeBrainOwnerInspection[];
  boundaries: Array<{ source: "committed" | "independent-live-router"; label: string; joinable: false; captureId?: string; epoch?: string }>;
  unavailable: string[];
  /** Membership is actual containment/aliasing, not inferred semantic neuron overlap. */
  semanticAxisMapping: false; tokenNeuronEquivalence: false;
}
export interface WholeBrainProjectionInput { pages: readonly WholeBrainInspectionPage[]; router?: RouterSpikeActivity }
