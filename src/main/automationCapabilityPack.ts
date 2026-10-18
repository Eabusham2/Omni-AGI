import catalog from "../../tools/catalog.json";
import type { DeclaredCapabilityPack } from "./learningCapabilityPack";
import { automationCanonical, automationHash } from "./automationFormat";

/** Declared tool trajectories; placeholders are not evidence of a real host action. */
export function declaredAutomationCapabilityPack(): DeclaredCapabilityPack {
  const schemas = catalog.tools.filter(tool => tool.id === "studio.automation").map(tool => ({ id: tool.id, actions: Object.keys(tool.actions), grant: "ask",
    actionInputSchemas: Object.fromEntries(Object.entries(tool.actions).map(([action, definition]) => {
      const input = (definition as { input: Record<string, string> }).input;
      return [action, { type: "object", additionalProperties: false, properties: Object.fromEntries(Object.entries(input).map(([key, value]) => {
        const descriptor = value.replace(/\?$/, ""); return [key, descriptor.includes("|") ? { type: "string", enum: descriptor.split("|") } : { type: descriptor }];
      })), required: Object.entries(input).filter(([, value]) => !value.endsWith("?")).map(([key]) => key) }];
    })) }));
  const program = { format: "omni-host-action-file", version: 1, name: "Read host time after a short wait", steps: [
    { id: "wait", kind: "wait", delayMs: 1000 }, { id: "clock", kind: "tool", toolId: "system.time", action: "now", arguments: {} }
  ] };
  const records: DeclaredCapabilityPack["records"] = [
    { utterance: "Save this inert reusable action file; do not execute it as a side effect of saving.", kind: "tool", toolId: "studio.automation", action: "save", arguments: { program } },
    { utterance: "Edit the exact selected action revision; discard its old trigger activation until reviewed again.", kind: "tool", toolId: "studio.automation", action: "save", arguments: { definitionId: "selected-action", expectedRevision: 1, program } },
    { utterance: "Activate this exact immutable action file for explicit timer and event handoffs using current nested tool access.", kind: "tool", toolId: "studio.automation", action: "activate", arguments: { definitionId: "selected-action", expectedRevision: 1, revisionSha256: "a".repeat(64), allowTriggers: true } },
    { utterance: "Run the selected action file independently of token generation, not through a hidden task prompt.", kind: "tool", toolId: "studio.automation", action: "run", arguments: { definitionId: "selected-action", revisionSha256: "a".repeat(64), inputs: {} } },
    { utterance: "Stop the selected host sequence without undoing its completed effects or deleting receipts.", kind: "tool", toolId: "studio.automation", action: "stop", arguments: { runId: "selected-run" } },
    { utterance: "Ask the human to review a possibly started effect before an explicitly new attempt. The tool itself cannot grant approval.", kind: "tool", toolId: "studio.automation", action: "review", arguments: { runId: "selected-run", resolution: "retry-confirmed", acknowledgePossibleEffects: true, reason: "The operator must inspect prior effects and explicitly approve this proposed new attempt." } },
    { utterance: "Propose approval of the exact paused step. Its nested token is not human authority; a separate one-use human approval is mandatory.", kind: "tool", toolId: "studio.automation", action: "approve", arguments: { runId: "selected-run", approvalToken: "selected-pending-step-token" } }
  ];
  for (const schema of schemas) for (const action of schema.actions) if (!records.some(row => row.action === action)) {
    const args = ["pause", "resume", "steps", "current-step"].includes(action) ? { runId: "selected-run" }
      : action === "export" ? { definitionId: "selected-action" }
      : action === "disable" ? { definitionId: "selected-action", expectedRevision: 1, revisionSha256: "a".repeat(64) } : {};
    records.push({ utterance: `Use ${schema.id} ${action} to inspect or control the selected saved record under current access.`, kind: "tool", toolId: schema.id, action, arguments: args });
  }
  const body = { format: "omni-declared-capability-pack", version: 1, id: "independent-host-action-files-1", sourceKind: "project-authored-declared-examples", executedHostActions: false, preferenceLabels: false, rewardModel: false,
    license: "PolyForm-Noncommercial-1.0.0-or-commercial-license", provenanceUrl: "https://github.com/Eabusham2/Omni-AGI", schemas, records };
  return { ...body, sha256: automationHash(automationCanonical(body)) };
}
