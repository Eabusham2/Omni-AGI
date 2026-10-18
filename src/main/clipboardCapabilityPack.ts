import catalog from "../../tools/catalog.json";
import type { DeclaredCapabilityPack } from "./learningCapabilityPack";
import { clipboardCanonical, clipboardHash } from "./clipboardFormat";

function descriptor(value: string): { type: string; enum?: string[]; items?: { type: string } } {
  const raw = value.replace(/\?$/, ""), array = raw.endsWith("[]"), item = array ? raw.slice(0, -2) : raw;
  const primitive = ["string", "number", "boolean", "object", "integer"].includes(item) ? { type: item }
    : { type: "string", enum: item.split("|") };
  if (array) return { type: "array", items: { type: primitive.type } };
  return primitive;
}
/** Ordinary declarations only. No host or instance clipboard is read or changed. */
export function declaredClipboardCapabilityPack(): DeclaredCapabilityPack {
  const schemas = catalog.tools.filter(tool => ["studio.clipboard", "system.clipboard"].includes(tool.id)).map(tool => ({ id: tool.id,
    actions: Object.keys(tool.actions), grant: "ask" as const,
    actionInputSchemas: Object.fromEntries(Object.entries(tool.actions).map(([action, value]) => { const input = (value as { input: Record<string, string> }).input;
      return [action, { type: "object", additionalProperties: false, properties: Object.fromEntries(Object.entries(input).map(([key, type]) => [key, descriptor(type)])),
        required: Object.entries(input).filter(([, type]) => !type.endsWith("?")).map(([key]) => key) }]; })) }));
  const records: DeclaredCapabilityPack["records"] = [
    { utterance: "Inspect this brain's private clipboard and retained version provenance without reading the computer clipboard.", kind: "tool", toolId: "studio.clipboard", action: "capabilities", arguments: { scope: "instance" } },
    { utterance: "Read a bounded page from this brain's current private clipboard. Continue only with the exact returned cursor.", kind: "tool", toolId: "studio.clipboard", action: "read", arguments: { scope: "instance", maxBytes: 65536 } },
    { utterance: "Copy this exact text into the brain's portable private clipboard without changing the operating-system clipboard.", kind: "tool", toolId: "studio.clipboard", action: "write", arguments: { scope: "instance", contentType: "text/plain", text: "Exact private clipboard text." } },
    { utterance: "Clear the brain's current private clipboard by creating an empty retained version; do not clear the computer clipboard.", kind: "tool", toolId: "studio.clipboard", action: "clear", arguments: { scope: "instance" } },
    { utterance: "Inspect whether a trusted, text-only current-user OS clipboard adapter is available without reading clipboard content.", kind: "tool", toolId: "system.clipboard", action: "capabilities", arguments: { scope: "host" } },
    { utterance: "Only under the current explicit host clipboard policy, read one monitored bounded text page. Other formats remain unsupported.", kind: "tool", toolId: "system.clipboard", action: "read", arguments: { scope: "host", maxBytes: 65536 } },
    { utterance: "Only under the current explicit host clipboard policy, replace the OS clipboard with this exact text and retain an effect receipt that is never replayed after lost acknowledgement.", kind: "tool", toolId: "system.clipboard", action: "write", arguments: { scope: "host", contentType: "text/plain", text: "Exact host clipboard text." } },
    { utterance: "Only under the current explicit host clipboard policy, clear the OS text clipboard once; an uncertain dispatched effect must never be retried automatically.", kind: "tool", toolId: "system.clipboard", action: "clear", arguments: { scope: "host" } }
  ];
  const body = { format: "omni-declared-capability-pack", version: 1, id: "clipboard-tools-1", sourceKind: "project-authored-declared-examples",
    executedHostActions: false, preferenceLabels: false, rewardModel: false, license: "PolyForm-Noncommercial-1.0.0-or-commercial-license",
    provenanceUrl: "https://github.com/Eabusham2/Omni-AGI", schemas, records };
  return { ...body, sha256: clipboardHash(clipboardCanonical(body)) } as DeclaredCapabilityPack;
}
