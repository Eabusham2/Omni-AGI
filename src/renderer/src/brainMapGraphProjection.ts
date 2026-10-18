import type { SubstrateCluster, SubstrateSynapse } from "../../shared/types";
import type { BrainMapGraphEdge, BrainMapGraphNode } from "./brainMapGraphPresentation";

type Contact = Pick<SubstrateSynapse, "id" | "sourceId" | "targetId" | "effectiveWeight" | "stability" | "plasticity" | "uses"> &
  Partial<Pick<SubstrateSynapse, "connectionStrength" | "effectiveContribution" | "isRepeatedContact" | "parallelOrdinal" | "parallelRootId" | "eligibility">>;

/** One actual contact remains one contact, even within a grouped layout node. */
export function projectBrainMapContact(contact: Contact, aliases: ReadonlyMap<string, string>): BrainMapGraphEdge | null {
  const sourceId = aliases.get(contact.sourceId), targetId = aliases.get(contact.targetId);
  if (!sourceId || !targetId) return null;
  return { id: contact.id, sourceId, targetId, nativeEndpoints: { sourceId: contact.sourceId, targetId: contact.targetId },
    effectiveWeight: contact.effectiveWeight, connectionStrength: contact.connectionStrength,
    effectiveContribution: contact.effectiveContribution, isRepeatedContact: contact.isRepeatedContact,
    parallelOrdinal: contact.parallelOrdinal, parallelRootId: contact.parallelRootId,
    signedStrength: contact.effectiveContribution ?? contact.effectiveWeight,
    stability: contact.stability, plasticity: contact.plasticity, eligibility: contact.eligibility,
    uses: contact.uses, pathways: 1, aggregate: false };
}

/** Region-level paths cannot identify which of several activation bands owns an endpoint. */
export function uniqueRegionLayoutNodes(nodes: readonly BrainMapGraphNode[], clusters: readonly Pick<SubstrateCluster, "id" | "kind">[]): Map<string, string> {
  const regionIds = new Set(clusters.filter(cluster => cluster.kind === "region").map(cluster => cluster.id));
  const groups = new Map<string, BrainMapGraphNode[]>();
  for (const node of nodes) if (node.region) { const values = groups.get(node.region) ?? []; values.push(node); groups.set(node.region, values); }
  const result = new Map<string, string>();
  for (const [region, candidates] of groups) {
    const exact = candidates.filter(node => regionIds.has(node.id));
    if (exact.length === 1) result.set(region, exact[0]!.id);
    else if (!exact.length && candidates.length === 1) result.set(region, candidates[0]!.id);
  }
  return result;
}
