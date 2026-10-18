import { createHash } from "node:crypto";
import type { DeclaredCapabilityPack } from "./learningCapabilityPack";

/** Declared examples train semantics, not fabricated provider outcomes. */
export function declaredMapCapabilityPack(): DeclaredCapabilityPack {
  const schemas: DeclaredCapabilityPack["schemas"] = [
    { id: "maps.search", actions: ["connections", "search"], grant: "ask", actionInputSchemas: {
      connections: { type: "object", properties: {}, required: [], additionalProperties: false },
      search: { type: "object", properties: { connectionId: { type: "string" }, expectedRevision: { type: "number" }, query: { type: "string" },
        limit: { type: "number" }, language: { type: "string" }, bbox: { type: "array", items: { type: "number" } }, refresh: { type: "boolean" } },
        required: ["connectionId", "expectedRevision", "query"], additionalProperties: false } } },
    { id: "maps.display", actions: ["show"], grant: "ask", actionInputSchemas: { show: { type: "object", properties: {
      searchResultId: { type: "string" }, geoJson: { type: "object" }, attribution: { type: "string" }, attributionUrl: { type: "string" } }, required: [], additionalProperties: false } } }
  ];
  const records: DeclaredCapabilityPack["records"] = [
    { utterance: "Which private map sources are configured, and what are their current revisions?", kind: "tool", toolId: "maps.search", action: "connections", arguments: {} },
    { utterance: "Search the selected private source for Berlin using its reviewed current revision.", kind: "tool", toolId: "maps.search", action: "search",
      arguments: { connectionId: "selected-private-source", expectedRevision: 1, query: "Berlin" } },
    { utterance: "Display the selected genuine map lookup on the offline coordinate canvas; do not invent a street basemap.", kind: "tool", toolId: "maps.display", action: "show",
      arguments: { searchResultId: "selected-owner-bound-map-result" } },
    { utterance: "Show these explicitly supplied example coordinates with their data attribution.", kind: "tool", toolId: "maps.display", action: "show",
      arguments: { geoJson: { type: "FeatureCollection", features: [{ type: "Feature", properties: { label: "Supplied example point" }, geometry: { type: "Point", coordinates: [0, 0] } }] }, attribution: "User-supplied example coordinates" } }
  ];
  const body = { format: "omni-declared-capability-pack", version: 1, id: "private-photon-offline-coordinates-1",
    sourceKind: "project-authored-declared-examples", executedHostActions: false, preferenceLabels: false, rewardModel: false,
    license: "PolyForm-Noncommercial-1.0.0-or-commercial-license", provenanceUrl: "https://github.com/Eabusham2/Omni-AGI", schemas, records };
  const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object"
    ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => [key, canonical(child)])) : value;
  return { ...body, sha256: createHash("sha256").update(JSON.stringify(canonical(body))).digest("hex") };
}
