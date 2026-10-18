import catalog from "../../tools/catalog.json";
import type { DeclaredCapabilityPack } from "./learningCapabilityPack";
import { automationCanonical, automationHash } from "./automationFormat";
/** Separate ordinary capability declarations; no code is executed by these examples. */
export function declaredAssemblyCapabilityPack(): DeclaredCapabilityPack {
  const schemas = catalog.tools.filter(tool => tool.id === "code.assembly").map(tool => ({ id: tool.id, actions: Object.keys(tool.actions), grant: "ask",
    actionInputSchemas: Object.fromEntries(Object.entries(tool.actions).map(([action, definition]) => {
      const input = (definition as { input: Record<string, string> }).input;
      return [action, { type: "object", additionalProperties: false,
        properties: Object.fromEntries(Object.entries(input).map(([key, value]) => { const raw = value.replace(/\?$/, ""), array = raw.endsWith("[]"), descriptor = array ? raw.slice(0, -2) : raw;
          const schema = descriptor.includes("|") ? { type: "string", enum: descriptor.split("|") } : { type: descriptor }; return [key, array ? { type: "array", items: schema } : schema]; })),
        required: Object.entries(input).filter(([, value]) => !value.endsWith("?")).map(([key]) => key) }];
    })) }));
  const records: DeclaredCapabilityPack["records"] = [
    { utterance: "Inspect genuinely installed native assembler/compiler commands and the current OS architecture; do not install anything.", kind: "tool", toolId: "code.assembly", action: "inspect", arguments: {} },
    { utterance: "Inspect this selected saved assembly artifact, its original receipt and current bytes without executing it.", kind: "tool", toolId: "code.assembly", action: "inspect", arguments: { artifactId: "a".repeat(32) } },
    { utterance: "Compile this exact assembly source as an object in a new managed directory; preserve failed compiler diagnostics and partial output.", kind: "tool", toolId: "code.assembly", action: "compile", arguments: { source: ".text\n", syntax: "gas", kind: "object", toolchain: "auto", timeoutMs: 60000 } },
    { utterance: "Separately request native execution of the exact compiled executable hash with current access. This is not sandbox, admin or raw RAM authority.", kind: "tool", toolId: "code.assembly", action: "run", arguments: { artifactId: "a".repeat(32), artifactSha256: "b".repeat(64), arguments: [], timeoutMs: 60000 } }
  ];
  const body = { format: "omni-declared-capability-pack", version: 1, id: "native-assembly-tools-1", sourceKind: "project-authored-declared-examples", executedHostActions: false, preferenceLabels: false, rewardModel: false,
    license: "PolyForm-Noncommercial-1.0.0-or-commercial-license", provenanceUrl: "https://github.com/Eabusham2/Omni-AGI", schemas, records };
  return { ...body, sha256: automationHash(automationCanonical(body)) };
}
