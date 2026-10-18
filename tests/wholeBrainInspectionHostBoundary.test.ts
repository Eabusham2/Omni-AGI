/** Host-boundary fixtures only: no worker, brain, model, Load or training. */
import { describe, expect, it } from "vitest";
import { checkedWholeBrainInspectionPage, checkedWholeBrainInspectionQuery } from "../src/main/wholeBrainInspectionProtocol";
import type { WholeBrainInspectionPage } from "../src/shared/wholeBrainInspection";

const brainId = "whole-host-owner", parentId = "actual-parent-owner", hash = "a".repeat(64);
const base = (): WholeBrainInspectionPage => ({
  format: "omni-whole-brain-inspection", version: 1, brainId, entity: "catalog",
  snapshot: { brainId, metadataSha256: hash, coreGeneration: hash, substrateGeneration: hash,
    attentionOverlaySha256: hash, connectedDescriptorSha256: null, connectedGraphStatesSha256: null, catalogSha256: hash },
  scope: { id: "root", sourceBrainId: brainId, nativeRegionPath: [] },
  catalog: { available: true, reason: "actual registered owner catalog", records: 1, shards: 1 },
  records: [], anchors: [], hasMore: false, nextCursor: null, cortex: null, substrate: null,
  evidence: { topology: "registered-ownership-and-explicit-regional-bridges", semanticAxisMapping: false,
    tokenNeuronEquivalence: false, liveActivityJoined: false, snapshotConsistent: true, unavailable: [] }
});
const connectedRegion = { kind: "region" as const, id: "connected-region", parentRegionId: "root-region",
  label: "Connected parent", regionKind: "connected-parent" as const,
  lineage: { regionId: "native-r0", parentId, generationSha256: hash }, nativeRegionPath: ["native-r0"],
  evidence: "actual-NativeRegion-identity" as const };

describe("whole-brain main-process source binding", () => {
  it("requires the exact requested facet and one matching delegated payload", () => {
    const mixed = { ...base(), cortex: {} };
    expect(() => checkedWholeBrainInspectionPage(mixed, brainId, { entity: "catalog" }))
      .toThrow(/payload does not match/);
    expect(() => checkedWholeBrainInspectionPage(base(), brainId, { entity: "cortex" }))
      .toThrow(/page identity/);
    expect(() => checkedWholeBrainInspectionQuery({ entity: "substrate", cortex: { entity: "modules" } }))
      .toThrow(/must match/);
  });

  it("rejects an unscoped page that substitutes another owner or region", () => {
    const foreign = base(); foreign.scope = { id: "root", sourceBrainId: "foreign-owner", nativeRegionPath: [] };
    expect(() => checkedWholeBrainInspectionPage(foreign, brainId, { entity: "catalog" }))
      .toThrow(/root scope/);
    const claimed = base(); claimed.scope.nativeRegionPath = ["unrequested-region"];
    expect(() => checkedWholeBrainInspectionPage(claimed, brainId, { entity: "catalog" }))
      .toThrow(/root scope|foreign region/);
  });

  it("binds a connected substrate page to its exact selected catalog anchor and parent owner", () => {
    const response = base();
    response.entity = "substrate";
    response.scope = { id: connectedRegion.id, sourceBrainId: parentId, nativeRegionPath: [...connectedRegion.nativeRegionPath] };
    response.anchors = [connectedRegion];
    response.substrate = { brainId: parentId, queriedAt: "2026-10-15T00:00:00Z", revision: hash, entity: "neurons", zoom: 1,
      totals: { neurons: 0, assemblies: 0, synapses: 0 }, matched: 0, hasMore: false,
      clusters: [], neurons: [], assemblies: [], synapses: [] };
    expect(checkedWholeBrainInspectionPage(response, brainId, { entity: "substrate", regionId: connectedRegion.id }))
      .toMatchObject({ scope: { id: connectedRegion.id, sourceBrainId: parentId } });
    const noAnchor = { ...response, anchors: [] };
    expect(() => checkedWholeBrainInspectionPage(noAnchor, brainId, { entity: "substrate", regionId: connectedRegion.id }))
      .toThrow(/selected region anchor/);
    const foreignPayload = { ...response, substrate: { ...response.substrate!, brainId: "foreign-parent" } };
    expect(() => checkedWholeBrainInspectionPage(foreignPayload, brainId, { entity: "substrate", regionId: connectedRegion.id }))
      .toThrow(/source owner differs/);
  });
});
