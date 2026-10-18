import catalog from "../../tools/catalog.json";
import type { DeclaredCapabilityPack } from "./learningCapabilityPack";
import { localTeacherCanonical, localTeacherHash } from "./localTeacherFormat";
/** Declared usage examples, never executed teacher calls or competence labels. */
export function declaredLocalTeacherCapabilityPack(): DeclaredCapabilityPack {
  const tool = catalog.tools.find(row => row.id === "data.distill-local")!;
  const schemas = [{ id: tool.id, actions: Object.keys(tool.actions), grant: "ask" as const,
    actionInputSchemas: Object.fromEntries(Object.entries(tool.actions).map(([action, value]) => {
      const input = (value as { input: Record<string, string> }).input;
      return [action, { type: "object", additionalProperties: false, properties: Object.fromEntries(Object.entries(input).map(([key, kind]) =>
        [key, kind.replace(/\?$/, "").endsWith("[]") ? { type: "array", items: { type: "string" } } : { type: kind.replace(/\?$/, "") }])),
      required: Object.entries(input).filter(([, kind]) => !kind.endsWith("?")).map(([key]) => key) }];
    })) }];
  const records: DeclaredCapabilityPack["records"] = [
    { utterance: "List explicitly selected current local GGUF teachers, without selecting paths or downloading anything.", kind: "tool", toolId: tool.id, action: "selections", arguments: {} },
    { utterance: "Ask for measured host-resource recommendations for this selected local teacher before starting. Unknown fit requires user action, not guessed defaults.", kind: "tool", toolId: tool.id, action: "plan", arguments: { teacherId: "selected-teacher" } },
    { utterance: "Start an attributable external teacher job for these explicitly selected lesson prompts, using the real current plan's generation values. This declared sample envelope is illustrative, not a fit receipt or a runtime invocation.",
      kind: "tool", toolId: tool.id, action: "start", arguments: { teacherId: "selected-teacher", prompts: ["Compare these two complete code examples and explain the tests."],
        generation: { contextTokens: 512, maxOutputTokens: 128, threads: 1, temperature: .7, seed: 0, timeoutMs: 0, maxOutputBytes: 65536 }, epochs: 1 } },
    { utterance: "Inspect this attributable teacher job. Generated sources and native committed traversal are separate from teacher truth or competence.", kind: "tool", toolId: tool.id, action: "status", arguments: { jobId: "selected-job" } },
    { utterance: "Pause this local teacher or native source cursor at its acknowledged boundary, preserving completed source receipts.", kind: "tool", toolId: tool.id, action: "pause", arguments: { jobId: "selected-job" } },
    { utterance: "Resume this same job only after exact local dependencies/current policy and resource fit are verified. Never replay copied dormant jobs.", kind: "tool", toolId: tool.id, action: "resume", arguments: { jobId: "selected-job" } },
    { utterance: "Stop this job without deleting committed learned state, source data, or already completed teacher responses.", kind: "tool", toolId: tool.id, action: "stop", arguments: { jobId: "selected-job" } }
  ];
  const body = { format: "omni-declared-capability-pack", version: 1, id: "local-gguf-teacher-1", sourceKind: "project-authored-declared-examples",
    executedHostActions: false, preferenceLabels: false, rewardModel: false, license: "PolyForm-Noncommercial-1.0.0-or-commercial-license",
    provenanceUrl: "https://github.com/Eabusham2/Omni-AGI", schemas, records };
  return { ...body, sha256: localTeacherHash(localTeacherCanonical(body)) } as DeclaredCapabilityPack;
}
