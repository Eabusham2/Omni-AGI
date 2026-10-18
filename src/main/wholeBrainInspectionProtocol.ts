import type { CortexPage } from "../shared/cortexInspection";
import type { SubstratePage } from "../shared/types";
import type { WholeBrainCatalogRecord, WholeBrainInspectionPage, WholeBrainInspectionQuery, WholeBrainSnapshotFence } from "../shared/wholeBrainInspection";

type Row = Record<string, unknown>;
const SHA = /^[a-f0-9]{64}$/;
function object(value: unknown): Row { if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid whole-brain source object"); return value as Row; }
function text(value: unknown): asserts value is string { if (typeof value !== "string" || !value || value.includes("\0")) throw new Error("Invalid whole-brain source identity"); }
function label(value: unknown): void { if (typeof value !== "string" || value.includes("\0")) throw new Error("Invalid whole-brain source text"); }
function count(value: unknown): asserts value is number { if (!Number.isSafeInteger(value) || (value as number) < 0) throw new Error("Invalid whole-brain source count"); }
function finite(value: unknown): void { if (typeof value !== "number" || !Number.isFinite(value)) throw new Error("Invalid whole-brain source measurement"); }
function array(value: unknown): asserts value is unknown[] { if (!Array.isArray(value)) throw new Error("Invalid whole-brain source array"); }
function strings(value: unknown): void { array(value); value.forEach(text); }
function sha(value: unknown, nullable = false): void { if (nullable && value === null) return; if (typeof value !== "string" || SHA.exec(value)?.[0] !== value) throw new Error("Invalid whole-brain generation hash"); }
function isFalse(value: unknown): void { if (value !== false) throw new Error("Whole-brain source invents unsupported observation"); }
function sign(value: unknown): void { if (value !== -1 && value !== 0 && value !== 1) throw new Error("Invalid actual ternary sign"); }
function bool(value: unknown): void { if (typeof value !== "boolean") throw new Error("Invalid whole-brain source flag"); }
function fields(row: Row, names: readonly string[], check: (value: unknown) => void): void { for (const name of names) check(row[name]); }
function optional(row: Row, names: readonly string[], check: (value: unknown) => void): void { for (const name of names) if (row[name] !== undefined) check(row[name]); }

export function checkedWholeBrainInspectionQuery(value: unknown = {}): WholeBrainInspectionQuery {
  const row = object(value), allowed = ["entity", "cursor", "pageSize", "regionId", "ownerId", "catalogFacet", "cortex", "substrate"];
  if (Object.keys(row).some(key => !allowed.includes(key))) throw new Error("Whole-brain query accepts no source paths or topology authority");
  if (row.entity !== undefined && !["catalog", "cortex", "substrate"].includes(String(row.entity))) throw new Error("Invalid whole-brain facet");
  optional(row, ["cursor", "regionId", "ownerId"], text);
  if (row.catalogFacet !== undefined && !["overview", "all", "members", "owner"].includes(String(row.catalogFacet))) throw new Error("Invalid whole-brain catalog filter");
  if (row.pageSize !== undefined) { count(row.pageSize); if (row.pageSize < 1 || row.pageSize > 256) throw new Error("Whole-brain pages use 1–256 records; all records remain pageable"); }
  for (const name of ["cortex", "substrate"]) if (row[name] !== undefined) {
    const query = object(row[name]), keys = name === "cortex" ? ["entity", "moduleId", "group", "search", "row", "offset"] : ["entity", "connectedTo", "region", "search", "zoom", "membersOf", "sourceRegion", "targetRegion", "activationBand"];
    if (Object.keys(query).some(key => !keys.includes(key))) throw new Error("Whole-brain paging owns the delegated source cursor and range");
    optional(query, ["entity", "moduleId", "group", "search", "connectedTo", "region", "membersOf", "sourceRegion", "targetRegion"], label);
    optional(query, ["row", "offset"], count); optional(query, ["zoom"], finite);
    if (query.activationBand !== undefined && !["quiet", "medium", "high"].includes(String(query.activationBand))) throw new Error("Invalid exact activation-band filter");
  }
  const entity = row.entity ?? "catalog";
  if ((entity !== "catalog" && (row.catalogFacet !== undefined || row.ownerId !== undefined)) ||
      (entity !== "cortex" && row.cortex !== undefined) || (entity !== "substrate" && row.substrate !== undefined)) {
    throw new Error("Whole-brain facet controls must match the requested source entity");
  }
  return structuredClone(row) as WholeBrainInspectionQuery;
}

function catalogRecord(value: unknown): WholeBrainCatalogRecord {
  const row = object(value); fields(row, ["id", "kind", "evidence"], text);
  switch (row.kind) {
    case "region": {
      fields(row, ["label", "regionKind"], text); if (row.parentRegionId !== null) text(row.parentRegionId);
      if (!["whole-brain", "registered-role", "connected-parent"].includes(String(row.regionKind))) throw new Error("Invalid whole-brain region kind");
      const lineage = object(row.lineage); optional(lineage, ["brainId", "registeredRoot", "regionId", "parentId"], text); optional(lineage, ["generationSha256"], sha);
      if (row.nativeRegionPath !== undefined) strings(row.nativeRegionPath);
      if (!["actual-brain-owner", "actual-trainable-role-reference", "actual-NativeRegion-identity"].includes(String(row.evidence))) throw new Error("Invalid whole-brain region evidence"); break;
    }
    case "owner":
      fields(row, ["canonicalAlias", "className"], text); array(row.packedFields);
      for (const value of row.packedFields) { const item = object(value); text(item.field); array(item.shape); item.shape.forEach(count); }
      if (row.evidence !== "actual-registered-object") throw new Error("Invalid registered-owner evidence"); break;
    case "alias":
      fields(row, ["alias", "ownerId"], text); if (row.evidence !== "same-actual-registered-object") throw new Error("Invalid shared alias evidence"); break;
    case "membership":
      fields(row, ["regionId", "ownerId"], text); if (row.evidence !== "actual-registered-containment-not-semantic-neuron-overlap") throw new Error("Invalid membership evidence"); break;
    case "bridge":
      fields(row, ["sourceRegionId", "targetRegionId", "ownerId"], text);
      if (!["workspace_exchange", "cortical_exchange"].includes(String(row.exchange)) || row.evidence !== "actual-RegionExchange-registration") throw new Error("Invalid registered bridge");
      isFalse(row.learnedNonzeroObserved); isFalse(row.firingObserved); break;
    case "inspection":
      fields(row, ["ownerId", "moduleId", "tensorKey"], text);
      if (!["core", "plasticity"].includes(String(row.role)) || !["weight", "bias"].includes(String(row.field)) || row.moduleId !== row.tensorKey || row.evidence !== "actual-save-role-and-registered-buffer-key") throw new Error("Invalid exact owner inspection binding"); break;
    default: throw new Error("Unknown whole-brain source record kind");
  }
  return structuredClone(row) as unknown as WholeBrainCatalogRecord;
}

function corticalModule(value: unknown): void {
  const row = object(value); fields(row, ["id", "module", "group", "layout", "shapeEvidence"], text);
  fields(row, ["rows", "packedColumns", "packedBytes"], count);
  if (row.columns !== null) count(row.columns); if (row.logicalParameters !== null) count(row.logicalParameters);
  if (row.logicalShape !== null) { array(row.logicalShape); row.logicalShape.forEach(count); }
  if (!["core", "plasticity"].includes(String(row.role)) || !["weight", "bias"].includes(String(row.field))) throw new Error("Invalid cortical owner role");
  isFalse(row.activationObserved); if (row.activation !== null) throw new Error("Committed cortical module has no live activation");
  optional(row, ["connectionStateBytes", "structuralBytes"], count); optional(row, ["activityOwner"], text);
}

function cortex(value: unknown, brainId: string): CortexPage {
  const row = object(value); if (row.brainId !== brainId || row.source !== "committed-packed-native-cortex") throw new Error("Cortical source owner differs");
  sha(row.revision); fields(row, ["total", "offset", "returned", "moduleCount", "packedBytes", "logicalParameters"], count); bool(row.hasMore);
  bool(row.logicalInventoryComplete); strings(row.groups); array(row.records);
  if (row.selectedModule !== null) corticalModule(row.selectedModule);
  const activity = object(row.activity); isFalse(activity.observed); text(activity.reason);
  for (const value of row.records) {
    const item = object(value);
    if (row.entity === "modules") corticalModule(item);
    else if (row.entity === "rows") { count(item.row); if (item.columns !== null) count(item.columns); isFalse(item.activationObserved); }
    else if (row.entity === "elements") { fields(item, ["row", "column"], count); sign(item.value); isFalse(item.activationObserved); }
    else if (row.entity === "links") {
      text(item.moduleId); fields(item, ["sourceIndex", "targetIndex"], count);
      if (!["tensor-column", "constant-one"].includes(String(item.sourceAxis)) || !["tensor-row", "output-row"].includes(String(item.targetAxis))) throw new Error("Invalid exact tensor axis");
      text(item.relationship); isFalse(item.semanticExplanationVerified); optional(item, ["value"], sign);
    } else if (row.entity === "boundaries") { fields(item, ["position", "tokenId", "embeddingRow", "languageHeadRow"], count); text(item.kind); isFalse(item.firingObserved); }
    else throw new Error("Invalid cortical facet");
    optional(item, ["strength", "resistance", "effectiveContribution"], finite); optional(item, ["isRepeatedContact"], bool); optional(item, ["ordinal", "slot"], count);
  }
  if (row.returned !== row.records.length) throw new Error("Cortical returned count differs");
  return structuredClone(row) as unknown as CortexPage;
}

function substrate(value: unknown, brainId: string): SubstratePage {
  const row = object(value); if (row.brainId !== brainId) throw new Error("Substrate source owner differs");
  sha(row.revision); text(row.queriedAt); finite(row.zoom); count(row.matched); bool(row.hasMore);
  const totals = object(row.totals); fields(totals, ["neurons", "assemblies", "synapses"], count);
  for (const kind of ["neurons", "assemblies", "synapses", "clusters", "clusterAnchors"] as const) {
    const values = row[kind] ?? (kind === "clusterAnchors" ? [] : undefined);
    array(values); for (const value of values) {
      const item = object(value); text(item.id);
      if (kind === "neurons") { fields(item, ["label", "region"], label); fields(item, ["activation", "importance", "uncertainty"], finite); count(item.exposures); strings(item.aliases); }
      else if (kind === "assemblies") { label(item.label); if (item.region !== "assembly") throw new Error("Invalid assembly identity");
        strings(item.neuronIds); strings(item.childAssemblyIds); text(item.kind); label(item.source); fields(item, ["confidence", "importance"], finite); count(item.rehearsals); bool(item.retainsSourceText); optional(item, ["relationshipsPaged", "activationObserved"], bool); optional(item, ["neuronCount", "childAssemblyCount"], count); }
      else if (kind === "synapses") { fields(item, ["sourceId", "targetId", "kind"], text); sign(item.effectiveWeight); fields(item, ["eligibility", "plasticity", "stability"], finite); count(item.uses); optional(item, ["connectionStrength", "effectiveContribution"], finite); }
      else { text(item.label); if (!["region", "activation-band", "pathway"].includes(String(item.kind))) throw new Error("Invalid aggregate kind");
        fields(item, ["count", "activeCount"], count); fields(item, ["meanActivation", "maxActivation"], finite); const weights = object(item.effectiveWeights); fields(weights, ["negative", "zero", "positive"], count);
        if ((item.activeCount as number) > (item.count as number)) throw new Error("Aggregate activity exceeds actual count");
        if (item.kind === "pathway") { fields(item, ["sourceRegion", "targetRegion"], text);
          if ((weights.negative as number) + (weights.zero as number) + (weights.positive as number) !== item.count) throw new Error("Aggregate sign counts differ from contact count");
        } else text(item.region);
        if (item.kind === "activation-band" && !["quiet", "medium", "high"].includes(String(item.activationBand))) throw new Error("Aggregate activation group omitted its actual band");
        if (kind === "clusterAnchors" && item.kind !== "region") throw new Error("Aggregate endpoint anchor must be an actual region total"); }
    }
  }
  if (row.membershipPage !== undefined) {
    const membership = object(row.membershipPage); text(membership.assemblyId);
    fields(membership, ["total", "offset", "returned"], count); array(membership.members);
    if (membership.returned !== membership.members.length || (membership.offset as number) + (membership.returned as number) > (membership.total as number) || row.matched !== membership.total) throw new Error("Assembly member coverage differs");
    const known = new Set<string>(); let previousKind = "", previousOrdinal = -1;
    for (const value of membership.members) { const member = object(value); fields(member, ["assemblyId", "memberId"], text); count(member.ordinal);
      if (member.assemblyId !== membership.assemblyId || !["neuron", "assembly"].includes(String(member.memberKind))) throw new Error("Assembly member identity differs");
      const memberKey = `${member.memberKind}:${member.ordinal}`;
      if (known.has(memberKey) || previousKind === "assembly" && member.memberKind === "neuron" || previousKind === member.memberKind && (member.ordinal as number) <= previousOrdinal) throw new Error("Assembly member ordinals are not the actual ordered page");
      known.add(memberKey); previousKind = member.memberKind as string; previousOrdinal = member.ordinal;
      const parent = (row.assemblies as Array<Row>).find(item => item.id === membership.assemblyId);
      if (!parent || !(parent[member.memberKind === "neuron" ? "neuronIds" : "childAssemblyIds"] as unknown[]).includes(member.memberId)) throw new Error("Assembly member page has no actual parent relationship");
      const neuronCount = parent.neuronCount, childCount = parent.childAssemblyCount;
      count(neuronCount); count(childCount);
      if (neuronCount + childCount !== membership.total || member.ordinal >= (member.memberKind === "neuron" ? neuronCount : childCount)) throw new Error("Assembly member ordinal exceeds its actual source array");
    }
  }
  if (row.pathwayFilter !== undefined) fields(object(row.pathwayFilter), ["sourceRegion", "targetRegion"], text);
  if (row.activationBand !== undefined && !["quiet", "medium", "high"].includes(String(row.activationBand))) throw new Error("Invalid source activation band");
  return structuredClone(row) as unknown as SubstratePage;
}

function samePath(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

export function checkedWholeBrainInspectionPage(
  value: unknown,
  brainId: string,
  requestedQuery: WholeBrainInspectionQuery
): WholeBrainInspectionPage {
  const row = object(value);
  const requestedEntity = requestedQuery.entity ?? "catalog";
  if (row.format !== "omni-whole-brain-inspection" || row.version !== 1 || row.brainId !== brainId || row.entity !== requestedEntity) throw new Error("Invalid whole-brain page identity");
  const snapshot = object(row.snapshot); if (snapshot.brainId !== brainId) throw new Error("Whole-brain snapshot owner differs");
  fields(snapshot, ["metadataSha256", "coreGeneration", "substrateGeneration", "attentionOverlaySha256"], sha);
  fields(snapshot, ["connectedDescriptorSha256", "connectedGraphStatesSha256", "catalogSha256"], value => sha(value, true));
  const scope = object(row.scope); fields(scope, ["id", "sourceBrainId"], text); strings(scope.nativeRegionPath);
  const catalog = object(row.catalog); bool(catalog.available); text(catalog.reason); fields(catalog, ["records", "shards"], count);
  array(row.records); const records = row.records.map(catalogRecord);
  array(row.anchors); const anchors = row.anchors.map(catalogRecord);
  if (requestedEntity !== "catalog" && records.length) throw new Error("Delegated whole-brain facets cannot inject catalog records");
  const hasCortex = row.cortex !== null, hasSubstrate = row.substrate !== null;
  if (hasCortex !== (requestedEntity === "cortex") || hasSubstrate !== (requestedEntity === "substrate")) {
    throw new Error("Whole-brain response payload does not match its requested facet");
  }
  const selectedRegion = requestedQuery.regionId === undefined ? undefined : [...records, ...anchors]
    .find((record): record is Extract<WholeBrainCatalogRecord, { kind: "region" }> =>
      record.kind === "region" && record.id === requestedQuery.regionId);
  if (requestedQuery.regionId !== undefined && !selectedRegion) {
    throw new Error("Whole-brain response omitted its exact selected region anchor");
  }
  if (requestedQuery.ownerId !== undefined && ![...records, ...anchors].some(record =>
    record.kind === "owner" && record.id === requestedQuery.ownerId)) {
    throw new Error("Whole-brain response omitted its exact selected owner anchor");
  }
  const regionPath = selectedRegion?.nativeRegionPath ?? [];
  if (requestedEntity === "substrate" && regionPath.length) {
    if (scope.id !== selectedRegion!.id || selectedRegion!.regionKind !== "connected-parent" ||
        selectedRegion!.lineage.parentId !== scope.sourceBrainId || !samePath(scope.nativeRegionPath as string[], regionPath)) {
      throw new Error("Connected substrate scope differs from its actual selected region lineage");
    }
  } else if (scope.id !== "root" || scope.sourceBrainId !== brainId ||
      !samePath(scope.nativeRegionPath as string[], requestedEntity === "catalog" ? regionPath : [])) {
    throw new Error("Whole-brain root scope differs from its requested owner and region");
  }
  if (requestedQuery.regionId === undefined && (scope.id !== "root" || scope.sourceBrainId !== brainId || (scope.nativeRegionPath as string[]).length)) {
    throw new Error("Unscoped whole-brain response claims a foreign region");
  }
  bool(row.hasMore); if (row.nextCursor !== null) text(row.nextCursor);
  if (row.hasMore !== (row.nextCursor !== null)) throw new Error("Whole-brain cursor/coverage disagree");
  const evidence = object(row.evidence);
  if (evidence.topology !== "registered-ownership-and-explicit-regional-bridges" || evidence.snapshotConsistent !== true) throw new Error("Whole-brain topology evidence differs");
  fields(evidence, ["semanticAxisMapping", "tokenNeuronEquivalence", "liveActivityJoined"], isFalse); strings(evidence.unavailable);
  const sparseAliases = row.sparseAliases ?? []; array(sparseAliases);
  const checkedCortex = row.cortex === null ? null : cortex(row.cortex, scope.sourceBrainId as string);
  const checkedSubstrate = row.substrate === null ? null : substrate(row.substrate, scope.sourceBrainId as string);
  if (requestedQuery.cortex?.entity !== undefined && checkedCortex?.entity !== requestedQuery.cortex.entity) throw new Error("Cortical source facet differs from the explicit request");
  if (requestedQuery.cortex?.moduleId !== undefined && checkedCortex?.selectedModule?.id !== requestedQuery.cortex.moduleId) throw new Error("Cortical selected matrix differs from the explicit request");
  if (requestedQuery.substrate?.entity !== undefined && checkedSubstrate?.entity !== requestedQuery.substrate.entity) throw new Error("Sparse source facet differs from the explicit request");
  if (requestedQuery.substrate?.membersOf !== undefined && checkedSubstrate?.membershipPage?.assemblyId !== requestedQuery.substrate.membersOf) throw new Error("Assembly member source differs from the explicit request");
  if (requestedQuery.substrate?.sourceRegion !== undefined || requestedQuery.substrate?.targetRegion !== undefined) {
    const filter = checkedSubstrate?.pathwayFilter;
    if (!checkedSubstrate || !filter || filter.sourceRegion !== requestedQuery.substrate.sourceRegion || filter.targetRegion !== requestedQuery.substrate.targetRegion) throw new Error("Exact ordered pathway filter differs from the explicit request");
    for (const contact of checkedSubstrate.synapses) {
      if (checkedSubstrate.neurons.find(neuron => neuron.id === contact.sourceId)?.region !== filter.sourceRegion ||
          checkedSubstrate.neurons.find(neuron => neuron.id === contact.targetId)?.region !== filter.targetRegion) throw new Error("Exact pathway contact has a different actual endpoint region");
    }
  }
  if (requestedQuery.substrate?.activationBand !== undefined) {
    const band = requestedQuery.substrate.activationBand;
    if (checkedSubstrate?.activationBand !== band || checkedSubstrate.neurons.some(neuron => (neuron.activation >= .66 ? "high" : neuron.activation >= .2 ? "medium" : "quiet") !== band)) throw new Error("Exact activation-band source differs from the explicit request");
  }
  for (const value of sparseAliases) { const alias = object(value); fields(alias, ["assemblyId", "neuronId"], text);
    if (alias.evidence !== "actual-assembly_neuron_id" || alias.assemblyId !== alias.neuronId ||
        !checkedSubstrate?.neurons.some(neuron => neuron.id === alias.neuronId) ||
        !checkedSubstrate.assemblies.some(assembly => assembly.id === alias.assemblyId)) throw new Error("Invalid native shared assembly/neuron alias"); }
  return { format: "omni-whole-brain-inspection", version: 1, brainId, snapshot: structuredClone(snapshot) as unknown as WholeBrainSnapshotFence,
    entity: row.entity as WholeBrainInspectionPage["entity"], scope: structuredClone(scope) as unknown as WholeBrainInspectionPage["scope"],
    catalog: structuredClone(catalog) as unknown as WholeBrainInspectionPage["catalog"], records, anchors,
    sparseAliases: structuredClone(sparseAliases) as WholeBrainInspectionPage["sparseAliases"], hasMore: row.hasMore as boolean, nextCursor: row.nextCursor as string | null,
    cortex: checkedCortex,
    substrate: checkedSubstrate,
    evidence: structuredClone(evidence) as unknown as WholeBrainInspectionPage["evidence"] };
}
