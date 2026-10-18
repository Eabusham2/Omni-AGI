import type { CortexLink, CortexModule } from "./cortexInspection";
import type { WholeBrainGraphEdge, WholeBrainGraphNode, WholeBrainGraphProjection, WholeBrainProjectionInput, WholeBrainSnapshotFence } from "./wholeBrainInspection";

const canonicalValue = (value: unknown): unknown => Array.isArray(value) ? value.map(canonicalValue) : value && typeof value === "object"
  ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([name, item]) => [name, canonicalValue(item)])) : value;
const canonical = (value: unknown): string => JSON.stringify(canonicalValue(value));
const key = (...parts: Array<string | number>): string => parts.map(value => encodeURIComponent(String(value))).join(":");

/** One graph with exact typed records. Incompatible snapshots are never merged. */
export function projectWholeBrainInspection({ pages, router }: WholeBrainProjectionInput): WholeBrainGraphProjection {
  const brainId = pages[0]?.brainId ?? router?.brainId ?? "";
  let snapshot: WholeBrainSnapshotFence | null = null;
  const nodes = new Map<string, WholeBrainGraphNode>(), edges = new Map<string, WholeBrainGraphEdge>();
  const result: WholeBrainGraphProjection = { brainId, snapshot, nodes: [], edges: [], aliases: [], inspectionBindings: [],
    boundaries: [], unavailable: [], semanticAxisMapping: false, tokenNeuronEquivalence: false };
  const recordIds = new Map<string, string>();
  const node = (value: WholeBrainGraphNode) => { const prior = nodes.get(value.id);
    if (prior && canonical(prior) !== canonical(value)) throw new Error("Whole-brain node identity conflicts under the same snapshot");
    nodes.set(value.id, value); };
  const edge = (value: WholeBrainGraphEdge) => { const prior = edges.get(value.id);
    if (prior && canonical(prior) !== canonical(value)) throw new Error("Whole-brain edge identity conflicts under the same snapshot");
    edges.set(value.id, value); };
  for (const page of pages) {
    if (page.brainId !== brainId || page.snapshot.brainId !== brainId) throw new Error("Whole-brain pages belong to different owners");
    if (snapshot && canonical(snapshot) !== canonical(page.snapshot)) throw new Error("Whole-brain snapshots differ; refresh instead of joining them");
    snapshot = page.snapshot;
    for (const item of [...page.records, ...page.anchors]) {
      const encoded = canonical(item), prior = recordIds.get(item.id);
      if (prior) { if (prior !== encoded) throw new Error("Whole-brain catalog record identity conflicts under the same snapshot"); continue; }
      recordIds.set(item.id, encoded);
      if (item.kind === "alias") result.aliases.push(item);
      else if (item.kind === "inspection") result.inspectionBindings.push(item);
      else if (item.kind === "region") {
        node({ id: item.id, recordId: item.id, kind: "region", label: item.label, scopeId: "root", source: "committed", regionIds: [], firingObserved: false, data: item });
        if (item.parentRegionId) edge({ id: key("containment", item.id), recordId: item.id, kind: "registered-containment",
          sourceId: item.parentRegionId, targetId: item.id, scopeId: "root", learnedConnection: false, firingObserved: false });
      } else if (item.kind === "owner") node({ id: item.id, recordId: item.id, kind: "registered-owner", label: item.canonicalAlias,
        scopeId: "root", source: "committed", regionIds: [], firingObserved: false, data: item });
      else if (item.kind === "membership") edge({ id: item.id, recordId: item.id, kind: "registered-membership", sourceId: item.regionId,
        targetId: item.ownerId, scopeId: "root", learnedConnection: false, firingObserved: false });
      else edge({ id: item.id, recordId: item.id, kind: "regional-trainable-bridge", sourceId: item.sourceRegionId,
        targetId: item.targetRegionId, scopeId: "root", learnedConnection: false, firingObserved: false, data: item });
    }
    result.unavailable.push(...page.evidence.unavailable);
  }
  const tensorNode = (scope: string, moduleId: string, axis: CortexLink["sourceAxis"] | CortexLink["targetAxis"], index: number): string => {
    const id = key("cortex-axis", scope, moduleId, axis, index);
    node({ id, recordId: key(moduleId, axis, index), kind: "tensor-axis", label: `${axis} ${index}`, scopeId: scope,
      source: "committed", regionIds: [], firingObserved: false, data: { moduleId, axis, index } }); return id;
  };
  const neuronIdentities = new Set(pages.flatMap(page => page.substrate?.neurons.map(row => key(page.scope.id, row.id)) ?? []));
  const assemblyIdentities = new Set(pages.flatMap(page => page.substrate?.assemblies.map(row => key(page.scope.id, row.id)) ?? []));
  const sparseEndpoint = (scope: string, id: string): string => neuronIdentities.has(key(scope, id)) ? key("substrate", scope, id)
    : assemblyIdentities.has(key(scope, id)) ? key("substrate-assembly", scope, id) : key("substrate", scope, id);
  for (const page of pages) {
    const scope = page.scope.id;
    const cortical = page.cortex;
    const storage = (item: CortexModule) => {
      const id = key("cortex-storage", scope, item.id);
      const rootOwner = [...nodes.values()].find(value => value.kind === "region" && "regionKind" in value.data &&
        value.data.regionKind === "whole-brain" && value.data.lineage.brainId === page.brainId);
      node({ id, recordId: item.id, kind: "cortical-storage", label: item.id, scopeId: scope, source: "committed",
        regionIds: rootOwner ? [rootOwner.id] : [], firingObserved: false, data: item });
      if (rootOwner) edge({ id: key("source-region-membership", rootOwner.id, id), recordId: item.id, kind: "source-region-membership",
        sourceId: rootOwner.id, targetId: id, scopeId: scope, learnedConnection: false, firingObserved: false });
      for (const binding of result.inspectionBindings) if (binding.moduleId === item.id && binding.role === item.role)
        edge({ id: key("packed-storage-ownership", binding.id), recordId: binding.id, kind: "packed-storage-ownership",
          sourceId: binding.ownerId, targetId: id, scopeId: scope, learnedConnection: false, firingObserved: false, data: binding });
    };
    if (cortical?.entity === "modules") for (const item of cortical.records as CortexModule[]) storage(item);
    else if (cortical?.selectedModule) storage(cortical.selectedModule);
    if (cortical?.entity === "links") for (const item of cortical.records as CortexLink[]) {
      if (cortical.selectedModule?.id !== item.moduleId) throw new Error("Returned tensor coordinates differ from their actual selected storage owner");
      const sourceId = tensorNode(scope, item.moduleId, item.sourceAxis, item.sourceIndex);
      const targetId = tensorNode(scope, item.moduleId, item.targetAxis, item.targetIndex);
      for (const axisId of [sourceId, targetId]) edge({ id: key("tensor-axis-ownership", scope, item.moduleId, axisId), recordId: axisId,
        kind: "tensor-axis-ownership", sourceId: key("cortex-storage", scope, item.moduleId), targetId: axisId, scopeId: scope,
        learnedConnection: false, firingObserved: false });
      const recordId = key(item.moduleId, item.sourceAxis, item.sourceIndex, item.targetAxis, item.targetIndex,
        item.isRepeatedContact ? "repeated" : "base", item.ordinal ?? 0);
      edge({ id: key("tensor-contact", scope, recordId), recordId, kind: "tensor-contact", sourceId, targetId, scopeId: scope,
        learnedConnection: true, firingObserved: false, ...(item.value !== undefined ? { effectiveWeight: item.value } : {}),
        ...(item.strength !== undefined ? { strength: item.strength } : {}), ...(item.resistance !== undefined ? { resistance: item.resistance } : {}), data: item });
    }
    const substrate = page.substrate;
    if (substrate) {
      const actualRegion = page.scope.nativeRegionPath.length ? nodes.get(page.scope.id) : [...nodes.values()].find(value => value.kind === "region" &&
        value.data && "regionKind" in value.data && value.data.regionKind === "whole-brain" && value.data.lineage.brainId === page.brainId);
      const ownershipRegion = actualRegion?.data && "regionKind" in actualRegion.data &&
        (page.scope.nativeRegionPath.length ? actualRegion.data.regionKind === "connected-parent" && actualRegion.data.lineage.parentId === page.scope.sourceBrainId &&
          canonical(actualRegion.data.nativeRegionPath ?? []) === canonical(page.scope.nativeRegionPath)
          : page.scope.id === "root" && page.scope.sourceBrainId === page.brainId) ? actualRegion.id : undefined;
      if (page.scope.nativeRegionPath.length && !ownershipRegion) throw new Error("Sparse source region has no exact actual owner anchor");
      const sparseNode = (value: WholeBrainGraphNode) => {
        node(value);
        if (ownershipRegion) edge({ id: key("source-region-membership", ownershipRegion, value.id), recordId: value.recordId,
          kind: "source-region-membership", sourceId: ownershipRegion, targetId: value.id, scopeId: scope, learnedConnection: false, firingObserved: false });
      };
      const regionIds = ownershipRegion ? [ownershipRegion] : [];
      const aggregates = [...(substrate.clusterAnchors ?? []), ...substrate.clusters];
      for (const item of aggregates) if (item.kind !== "pathway") sparseNode({ id: key("substrate-cluster", scope, item.id), recordId: item.id,
        kind: "substrate-cluster", label: item.label, scopeId: scope, source: "committed", regionIds, firingObserved: false, data: item });
      const regionCluster = (region: string | undefined) => aggregates.find(item => item.kind === "region" && item.region === region);
      for (const item of aggregates) {
        if (item.kind === "pathway") {
          const source = regionCluster(item.sourceRegion), target = regionCluster(item.targetRegion);
          if (source && target) edge({ id: key("substrate-pathway-aggregate", scope, item.id), recordId: item.id, kind: "substrate-pathway-aggregate",
            sourceId: key("substrate-cluster", scope, source.id), targetId: key("substrate-cluster", scope, target.id), scopeId: scope,
            learnedConnection: false, firingObserved: false, data: item });
          else {
            sparseNode({ id: key("substrate-pathway-summary", scope, item.id), recordId: item.id, kind: "substrate-pathway-summary",
              label: item.label, scopeId: scope, source: "committed", regionIds, firingObserved: false, data: item });
            result.unavailable.push(`Actual aggregate pathway ${item.id} has an unavailable region endpoint; no endpoint was fabricated`);
          }
        } else if (item.kind === "activation-band") {
          const region = regionCluster(item.region);
          if (region) edge({ id: key("aggregate-band-membership", scope, item.id), recordId: item.id, kind: "aggregate-band-membership",
            sourceId: key("substrate-cluster", scope, region.id), targetId: key("substrate-cluster", scope, item.id), scopeId: scope,
            learnedConnection: false, firingObserved: false });
        }
      }
      for (const item of substrate.neurons) sparseNode({ id: key("substrate", scope, item.id), recordId: item.id, kind: "substrate-neuron",
        label: item.label, scopeId: scope, source: "committed", regionIds, firingObserved: false, data: item });
      for (const item of substrate.assemblies) {
        const id = key("substrate-assembly", scope, item.id);
        // Member windows can differ while the immutable assembly is identical.
        // Keep identity metadata on its node and exact ordinal relationships on
        // edges, rather than declaring two partial arrays contradictory state.
        const { neuronIds, childAssemblyIds, relationshipsPaged, ...identity } = item;
        sparseNode({ id, recordId: item.id, kind: "substrate-assembly", label: item.label, scopeId: scope, source: "committed", regionIds, firingObserved: false,
          data: { ...identity, neuronCount: item.neuronCount ?? neuronIds.length, childAssemblyCount: item.childAssemblyCount ?? childAssemblyIds.length } });
        const exactPage = substrate.membershipPage?.assemblyId === item.id ? substrate.membershipPage : undefined;
        for (const [memberKind, members] of [["neurons", item.neuronIds], ["assemblies", item.childAssemblyIds]] as const)
          for (const [ordinal, member] of members.entries()) if (!exactPage) edge({ id: key("assembly-membership", scope, item.id, memberKind, ordinal, member),
            recordId: key(item.id, memberKind, ordinal, member), kind: "assembly-membership", sourceId: id,
            targetId: key(memberKind === "neurons" ? "substrate" : "substrate-assembly", scope, member), scopeId: scope,
            learnedConnection: false, firingObserved: false });
        for (const member of exactPage?.members ?? []) edge({ id: key("assembly-membership", scope, item.id, member.memberKind === "neuron" ? "neurons" : "assemblies", member.ordinal, member.memberId),
          recordId: key(item.id, member.memberKind === "neuron" ? "neurons" : "assemblies", member.ordinal, member.memberId), kind: "assembly-membership", sourceId: id,
          targetId: key(member.memberKind === "neuron" ? "substrate" : "substrate-assembly", scope, member.memberId), scopeId: scope, learnedConnection: false, firingObserved: false });
        if (relationshipsPaged) result.unavailable.push("Additional exact assembly relationships are available through the selected assembly’s Members pages");
      }
      for (const item of substrate.synapses) edge({ id: key("substrate-synapse", scope, item.id), recordId: item.id, kind: "substrate-synapse",
        sourceId: sparseEndpoint(scope, item.sourceId), targetId: sparseEndpoint(scope, item.targetId), scopeId: scope,
        learnedConnection: true, firingObserved: false, effectiveWeight: item.effectiveWeight,
        ...(item.connectionStrength !== undefined ? { strength: item.connectionStrength } : {}), data: item });
      for (const alias of page.sparseAliases ?? []) {
        if (!neuronIdentities.has(key(scope, alias.neuronId)) || !assemblyIdentities.has(key(scope, alias.assemblyId)))
          throw new Error("Shared assembly/neuron alias has no actual typed endpoint records");
        edge({ id: key("assembly-neuron-alias", scope, alias.assemblyId, alias.neuronId), recordId: key(alias.assemblyId, alias.neuronId),
          kind: "assembly-neuron-alias", sourceId: key("substrate-assembly", scope, alias.assemblyId), targetId: key("substrate", scope, alias.neuronId),
          scopeId: scope, learnedConnection: false, firingObserved: false, data: alias });
      }
    }
  }
  for (const item of edges.values()) if (item.kind === "registered-membership") {
    const value = nodes.get(item.targetId); if (value && !value.regionIds.includes(item.sourceId)) value.regionIds.push(item.sourceId);
  }
  if (snapshot) result.boundaries.push({ source: "committed", label: "One committed brain snapshot · ownership is not a semantic neuron map", joinable: false });
  if (router) {
    if (router.brainId !== brainId) throw new Error("Live router observation belongs to another brain");
    if (router.available && router.epoch) {
      const scope = key("live-router", router.captureId, router.epoch);
      const channels = new Set<number>(), fired = new Set<number>();
      for (const frame of router.frames) for (let index = 0; index < frame.spikes.length; index++) {
        const channel = frame.channelStart + index; channels.add(channel); if (frame.spikes[index] === 1) fired.add(channel);
      }
      for (const contact of router.contactSnapshot?.contacts ?? []) { channels.add(contact.sourceChannel); channels.add(contact.targetChannel); }
      for (const channel of channels) node({ id: key(scope, channel), recordId: `${brainId}:root.router.population:${channel}`,
        kind: "router-channel", label: `LIF channel ${channel}`, scopeId: scope, source: "independent-live-router", regionIds: [],
        firingObserved: fired.has(channel), data: { channel, captureId: router.captureId, epoch: router.epoch, owner: "root.router.population" } });
      for (const item of router.contactSnapshot?.contacts ?? []) edge({ id: key(scope, item.id), recordId: item.id, kind: "router-contact",
        sourceId: key(scope, item.sourceChannel), targetId: key(scope, item.targetChannel), scopeId: scope, learnedConnection: true,
        firingObserved: false, effectiveWeight: item.effectiveWeight, strength: item.connectionStrength, resistance: item.connectionResistance, data: item });
      result.boundaries.push({ source: "independent-live-router", label: "Live router capture · not joined to committed cortex/substrate axes",
        joinable: false, captureId: router.captureId, epoch: router.epoch });
    } else result.unavailable.push(router.reason ?? "Live router is unavailable; no firing was invented");
  }
  result.snapshot = snapshot; result.nodes = [...nodes.values()]; result.edges = [...edges.values()];
  result.unavailable = [...new Set(result.unavailable)];
  return result;
}
