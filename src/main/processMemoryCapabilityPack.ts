import catalog from "../../tools/catalog.json";
import type { DeclaredCapabilityPack } from "./learningCapabilityPack";
import { processMemoryCanonical, processMemoryHash } from "./processMemoryFormat";

/** Declared semantics only: examples perform no inspection, writes or rewards. */
export function declaredProcessMemoryCapabilityPack(): DeclaredCapabilityPack {
  const schemas = catalog.tools.filter(tool => tool.id === "system.memory").map(tool => ({ id: tool.id,
    actions: Object.keys(tool.actions), grant: "off", actionInputSchemas: Object.fromEntries(Object.entries(tool.actions).map(([action, definition]) => {
      const input = (definition as { input: Record<string, string> }).input;
      return [action, { type: "object", additionalProperties: false,
        properties: Object.fromEntries(Object.entries(input).map(([key, value]) => {
          const name = value.replace(/\?$/, "");
          return [key, name.includes("|") ? { type: "string", enum: name.split("|") } : { type: name }];
        })), required: Object.entries(input).filter(([, value]) => !value.endsWith("?")).map(([key]) => key) }];
    })) }));
  const records: DeclaredCapabilityPack["records"] = [
    { utterance: "Inspect current DMA capabilities/settings without opening a process. This is advanced OS-permitted virtual-process memory, not unrestricted physical RAM; protected applications may be unavailable, and powerful hosts/models do not bypass those protections.", kind: "tool", toolId: "system.memory", action: "capabilities", arguments: {} },
    { utterance: "Only after the user enables DMA and current tool access allows it, attach the explicitly selected permitted process for inspection. This illustrative PID is not a discovered target; never scan processes or change privileges.", kind: "tool", toolId: "system.memory", action: "attach", arguments: { pid: 42000, access: "read", environment: "host" } },
    { utterance: "Read a bounded page of regions from the current exact process-creation lease. A host lease expires on Unload, access changes, restart, copies, and imported instances.", kind: "tool", toolId: "system.memory", action: "regions", arguments: { targetId: "example-current-host-lease", limit: 50 } },
    { utterance: "Read only the chosen exact address range under current access; hexadecimal addresses avoid floating-point rounding. No whole-process memory dump or unrelated-process access is implied.", kind: "tool", toolId: "system.memory", action: "read", arguments: { targetId: "example-current-host-lease", address: "0x1000", length: 16 } },
    { utterance: "A separately reviewed write requires a read-write lease and the exact observed preimage hash. The running target can change concurrently: this is not atomic compare-and-swap. Preserve partial/uncertain effects and never automatically repeat a lost write receipt.", kind: "tool", toolId: "system.memory", action: "write", arguments: { targetId: "example-current-host-write-lease", address: "0x1000", dataBase64: "AA==", expectedBeforeSha256: processMemoryHash(Buffer.from([0])) } },
    { utterance: "Release this current host process-memory lease without terminating the selected application or undoing committed writes.", kind: "tool", toolId: "system.memory", action: "detach", arguments: { targetId: "example-current-host-lease" } }
  ];
  const body = { format: "omni-declared-capability-pack", version: 1, id: "os-process-memory-tools-1",
    sourceKind: "project-authored-declared-examples", executedHostActions: false, preferenceLabels: false, rewardModel: false,
    license: "PolyForm-Noncommercial-1.0.0-or-commercial-license", provenanceUrl: "https://github.com/Eabusham2/Omni-AGI", schemas, records };
  return { ...body, sha256: processMemoryHash(processMemoryCanonical(body)) };
}
