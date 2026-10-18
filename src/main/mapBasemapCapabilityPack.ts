import { createHash } from "node:crypto";
import type { DeclaredCapabilityPack } from "./learningCapabilityPack";

/** Additive capability teaching, not provider outcomes, a runtime persona or hidden prompt. */
export function declaredMapBasemapCapabilityPack(): DeclaredCapabilityPack {
  const basemapSchema = { type: "object", properties: { profileId: { type: "string" }, expectedRevision: { type: "integer" },
    bounds: { type: "array", items: { type: "number" }, minItems: 4, maxItems: 4 }, width: { type: "integer", minimum: 1 }, height: { type: "integer", minimum: 1 }, zoom: { type: "integer", minimum: 0 } },
    required: ["profileId", "expectedRevision", "bounds", "width", "height"], additionalProperties: false };
  const schemas: DeclaredCapabilityPack["schemas"] = [{ id: "maps.display", actions: ["basemaps", "basemap"], grant: "ask", actionInputSchemas: {
    basemaps: { type: "object", properties: {}, required: [], additionalProperties: false },
    basemap: basemapSchema
  } }];
  const records: DeclaredCapabilityPack["records"] = [
    { utterance: "Which optional basemaps have actually been configured, and what are their current revisions?", kind: "tool", toolId: "maps.display", action: "basemaps", arguments: {} },
    { utterance: "Request the selected authorized basemap for this supplied area and viewport using its exact current revision.", kind: "tool", toolId: "maps.display", action: "basemap",
      arguments: { profileId: "selected-reviewed-profile", expectedRevision: 1, bounds: [13.3, 52.4, 13.5, 52.6], width: 640, height: 400 } }
  ];
  const body = { format: "omni-declared-capability-pack", version: 1, id: "reviewed-retained-raster-maps-1",
    sourceKind: "project-authored-declared-examples", executedHostActions: false, preferenceLabels: false, rewardModel: false,
    license: "PolyForm-Noncommercial-1.0.0-or-commercial-license", provenanceUrl: "https://github.com/Eabusham2/Omni-AGI", schemas, records };
  const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object"
    ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => [key, canonical(child)])) : value;
  return { ...body, sha256: createHash("sha256").update(JSON.stringify(canonical(body))).digest("hex") };
}
