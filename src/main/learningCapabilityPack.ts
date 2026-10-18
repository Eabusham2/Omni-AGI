/** Append-only supplementary curriculum. Frozen origin-v3 records stay unchanged. */
import { createHash } from "node:crypto";
import catalog from "../../tools/catalog.json";
export const LEARNING_CAPABILITY_PACK = Object.freeze({
  format: "omni-supplementary-capability-pack", version: 1, id: "specific-learn-fact-check-1",
  license: "PolyForm-Noncommercial-1.0.0-or-commercial-license",
  provenanceUrl: "https://github.com/Eabusham2/Omni-AGI",
  behavioralPersona: false, preferenceLabels: false, rewardModel: false,
  records: Object.freeze([
    { capability: "learn", source: "current-admitted-experience", semantics: "No external input binds the current accepted turn to its existing neural replay. Retention and replay use normal memory dynamics; an absent source remains unresolved." },
    { capability: "data.learn", action: "start", arguments: { inputs: [{ kind: "text", text: "The supplied chapter text." }], instructions: "Compare its two examples." },
      example: "Learn this chapter and compare its two examples.", result: { state: "queued", traversalComplete: false }, semantics: "A queued receipt is not completed learning. Text is training data, not a hidden instruction." },
    { capability: "data.learn", action: "start", arguments: { inputs: [{ kind: "dataset", manifestId: "host-selected-manifest" }] },
      example: "Learn the files I selected.", semantics: "Only a host-owned manifest grants selected file/folder input. Arbitrary generated paths are not file grants. Each valid record is visited with durable cursors." },
    { capability: "data.learn", action: "start", arguments: { inputs: [{ kind: "web", url: "https://example.org", followLinks: true }], constraints: { allowedSites: [{ domain: "example.org" }] } },
      example: "Study this site and keep going until I stop it.", semantics: "Crawling is ordinary learning with deduplicated source receipts. Unknown publication dates stay unknown; scope can be steered without replaying committed records." },
    { capability: "data.learn", actions: ["status", "pause", "resume", "stop", "steer"], arguments: { taskId: "saved-task-id" },
      semantics: "Pause and Stop preserve already learned parameters and exact source progress. Resume uses the same cursor. Steering has an expectedRevision and never silently replaces past receipts." },
    { capability: "data.fact-check", action: "start", arguments: { claims: [{ text: "A selected statement", sources: ["https://example.org/evidence"] }] },
      example: "Check this statement against its source.", semantics: "The scope is selected claims, not a complete inventory of facts inside weights. Reading evidence participates in normal learning." },
    { capability: "data.fact-check", action: "record", arguments: { taskId: "saved-task-id", claimId: "selected-claim-id", conclusion: "undetermined", explanation: "The available evidence does not resolve this claim.", evidenceIds: [] },
      semantics: "A finding is a model self-report tied to actual captured evidence IDs. Supported, contradicted and mixed findings need captured evidence. No proposed finding is a universal truth guarantee." },
    { capability: "data.fact-check", action: "record", arguments: { taskId: "saved-task-id", claimId: "selected-claim-id", conclusion: "mixed", evidenceIds: ["actual-evidence-id"], explanation: "The sources refer to different versions.", question: { mode: "text", prompt: "Which version do you mean?" } },
      semantics: "A question pauses only the dependent claim; independent claims continue. Accepted proposed corrections enter ordinary neural learning, not a keyed truth database." }
  ])
});

export function learningCapabilityPackText(): string {
  return LEARNING_CAPABILITY_PACK.records.map(record => JSON.stringify({ packId: LEARNING_CAPABILITY_PACK.id,
    packVersion: LEARNING_CAPABILITY_PACK.version, license: LEARNING_CAPABILITY_PACK.license, record })).join("\n");
}

/** Declared targets, never receipts claiming that placeholder tools ran. */
export function declaredLearningCapabilityPack() {
  const schemas = catalog.tools.filter(tool => ["data.learn", "data.fact-check"].includes(tool.id)).map(tool => ({
    id: tool.id, actions: Object.keys(tool.actions), grant: "ask",
    actionInputSchemas: Object.fromEntries(Object.entries(tool.actions).map(([action, definition]) => {
      const input = (definition as { input: Record<string, string> }).input;
      return [action, { type: "object", additionalProperties: false,
        properties: Object.fromEntries(Object.entries(input).map(([key, descriptor]) => {
          const name = descriptor.replace(/\?$/, ""), list = name.endsWith("[]"), item = list ? name.slice(0, -2) : name;
          const typed = item.includes("|") ? { type: "string", enum: item.split("|") } : { type: item };
          return [key, list ? { type: "array", items: typed } : typed];
        })), required: Object.entries(input).filter(([, value]) => !value.endsWith("?")).map(([key]) => key) }];
    }))
  }));
  const records: Array<{ utterance: string; kind: string; toolId: string | null; action: string | null; arguments: Record<string, unknown> }> = [
    { utterance: "I want to learn more from my current admitted experience.", kind: "learn", toolId: null, action: null, arguments: {} }
  ];
  for (const value of LEARNING_CAPABILITY_PACK.records) {
    const row = value as { capability: string; action?: string; actions?: readonly string[]; arguments?: Record<string, unknown>; example?: string };
    if (!row.capability.startsWith("data.") || !row.arguments) continue;
    if (row.action && row.example) records.push({ utterance: row.example, kind: "tool", toolId: row.capability, action: row.action, arguments: row.arguments });
    else if (row.action === "record") records.push({ utterance: row.arguments.conclusion === "mixed"
      ? "The selected sources concern different versions; record the conflict and ask which version is meant."
      : "The captured evidence does not settle this selected claim; record an undetermined finding.",
      kind: "tool", toolId: row.capability, action: row.action, arguments: row.arguments });
    for (const action of row.actions ?? []) records.push({ utterance: `${action} the selected learning task.`, kind: "tool", toolId: row.capability, action,
      arguments: { taskId: "saved-task-id", ...(action === "steer" ? { expectedRevision: 1, instructions: "Use only the selected sources." } : {}) } });
  }
  records.push({ utterance: "Inspect the actual evidence and current status for this fact-check task.", kind: "tool", toolId: "data.fact-check", action: "status", arguments: { taskId: "saved-task-id" } });
  for (const action of ["pause", "resume", "stop", "steer"]) records.push({ utterance: `${action} the selected fact-check task.`, kind: "tool", toolId: "data.fact-check", action,
    arguments: { taskId: "saved-task-id", ...(action === "steer" ? { expectedRevision: 1, instructions: "Check only the selected evidence." } : {}) } });
  const body = { format: "omni-declared-capability-pack", version: 1, id: LEARNING_CAPABILITY_PACK.id,
    sourceKind: "project-authored-declared-examples", executedHostActions: false, preferenceLabels: false, rewardModel: false,
    license: LEARNING_CAPABILITY_PACK.license, provenanceUrl: LEARNING_CAPABILITY_PACK.provenanceUrl, schemas, records };
  const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object"
    ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => [key, canonical(child)])) : value;
  return { ...body, sha256: createHash("sha256").update(JSON.stringify(canonical(body))).digest("hex") };
}

/** Stable envelope for separately versioned project-authored capability packs. */
export type DeclaredCapabilityPack = Omit<ReturnType<typeof declaredLearningCapabilityPack>, "id" | "license" | "provenanceUrl"> & {
  id: string; license: string; provenanceUrl: string;
};
