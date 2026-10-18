import catalog from "../../tools/catalog.json";
import type { DeclaredCapabilityPack } from "./learningCapabilityPack";
import { webImageHash } from "./webImageReferences";

/** Project-authored schema examples only; no image/network/model is executed. */
export function declaredWebImageCapabilityPack(): DeclaredCapabilityPack {
  const schemas = catalog.tools.filter(tool => ["web.search", "web.fetch"].includes(tool.id)).map(tool => ({
    id: tool.id, actions: Object.keys(tool.actions), grant: "ask" as const,
    actionInputSchemas: Object.fromEntries(Object.entries(tool.actions).map(([action, definition]) => {
      const input = (definition as { input: Record<string, string> }).input;
      return [action, { type: "object", additionalProperties: false,
        properties: Object.fromEntries(Object.entries(input).map(([key, type]) => [key, { type: type.replace(/\?$/, "") }])),
        required: Object.entries(input).filter(([, type]) => !type.endsWith("?")).map(([key]) => key) }];
    }))
  }));
  const records: DeclaredCapabilityPack["records"] = [
    { utterance: "Search for image references about a lighthouse. Metadata is not seen pixels; the provider may return no image references.",
      kind: "tool", toolId: "web.search", action: "search", arguments: { query: "lighthouse", images: true } },
    { utterance: "Fetch this public page to discover its attributable image references without automatically downloading their image bytes.",
      kind: "tool", toolId: "web.fetch", action: "fetch", arguments: { url: "https://example.com/gallery" } },
    { utterance: "Inspect the explicitly selected reference from this current task through this same brain's image region. This declared example ID is illustrative, not a real returned reference or a successful observation.",
      kind: "tool", toolId: "web.fetch", action: "inspect-image", arguments: { referenceId: "0".repeat(64) } }
  ];
  const body = { format: "omni-declared-capability-pack", version: 1, id: "web-image-tools-1",
    sourceKind: "project-authored-declared-examples", executedHostActions: false, preferenceLabels: false, rewardModel: false,
    license: "PolyForm-Noncommercial-1.0.0-or-commercial-license", provenanceUrl: "https://github.com/Eabusham2/Omni-AGI", schemas, records };
  return { ...body, sha256: webImageHash(body) } as DeclaredCapabilityPack;
}
