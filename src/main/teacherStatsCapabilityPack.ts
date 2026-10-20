import { createHash } from "node:crypto";
import catalog from "../../tools/catalog.json";
import type { DeclaredCapabilityPack } from "./learningCapabilityPack";

/** Append-only declared lessons; no provider calls, host readings or invented outcomes. */
export function declaredTeacherStatsCapabilityPack(): DeclaredCapabilityPack {
  const schemas = catalog.tools.filter(tool => ["data.distill-api", "system.stats"].includes(tool.id)).map(tool => ({
    id: tool.id, actions: Object.keys(tool.actions), grant: "ask",
    actionInputSchemas: Object.fromEntries(Object.entries(tool.actions).map(([action, value]) => {
      const input = (value as { input: Record<string, string> }).input;
      return [action, { type: "object", additionalProperties: false,
        properties: Object.fromEntries(Object.entries(input).map(([name, descriptor]) => {
          const base = descriptor.replace(/\?$/, ""), list = base.endsWith("[]"), item = list ? base.slice(0, -2) : base;
          const schema = item.includes("|") ? { type: "string", enum: item.split("|") } : { type: item };
          return [name, list ? { type: "array", items: schema } : schema];
        })), required: Object.entries(input).filter(([, kind]) => !kind.endsWith("?")).map(([name]) => name) }];
    }))
  }));
  const records: DeclaredCapabilityPack["records"] = [
    { utterance: "Read current actual system resources and sensor availability before planning this work; do not invent temperatures or benchmark a model.", kind: "tool", toolId: "system.stats", action: "snapshot", arguments: {} },
    { utterance: "Which API teachers are securely configured on this device? Do not expose or ask a tool for a credential.", kind: "tool", toolId: "data.distill-api", action: "providers", arguments: {} },
    { utterance: "Discover account-visible models for the explicitly selected teacher provider without choosing one or claiming generation compatibility.", kind: "tool", toolId: "data.distill-api", action: "discover-models", arguments: { provider: "openrouter" } },
    { utterance: "Use the explicitly selected configured provider/model to generate these attributable lessons and learn through the existing native pipeline. This declared model name is a placeholder, not a discovered model or a dispatched request.", kind: "tool", toolId: "data.distill-api", action: "start", arguments: { provider: "openai", model: "selected-model", prompts: ["Compare the supplied code examples and their tests."], responseModalities: ["text"] } },
    { utterance: "List this brain's API teacher jobs; queued or generated responses are not proof of learned parameters.", kind: "tool", toolId: "data.distill-api", action: "list", arguments: {} },
    { utterance: "Inspect the selected teacher job's actual per-item learning receipts and uncertain acknowledgements.", kind: "tool", toolId: "data.distill-api", action: "status", arguments: { jobId: "selected-job" } },
    { utterance: "Pause the selected API teacher job while preserving committed provider responses and learned state.", kind: "tool", toolId: "data.distill-api", action: "pause", arguments: { jobId: "selected-job" } },
    { utterance: "Resume the same teacher job only through fresh current permission, resource and credential checks; do not repeat uncertain or copied dormant work.", kind: "tool", toolId: "data.distill-api", action: "resume", arguments: { jobId: "selected-job" } },
    { utterance: "Stop the selected teacher job without deleting completed lessons or claiming remote cancellation is proven.", kind: "tool", toolId: "data.distill-api", action: "stop", arguments: { jobId: "selected-job" } },
    { utterance: "Request exact human approval to skip this uncertain teacher item; an unknown acknowledgement never authorizes a resend. Resolve does not resume execution.", kind: "tool", toolId: "data.distill-api", action: "resolve-uncertain", arguments: { jobId: "selected-job", itemIndex: 0, decision: "skip" } }
  ];
  const body = { format: "omni-declared-capability-pack", version: 1, id: "api-teacher-system-stats-1",
    sourceKind: "project-authored-declared-examples", executedHostActions: false, preferenceLabels: false, rewardModel: false,
    license: "PolyForm-Noncommercial-1.0.0-or-commercial-license", provenanceUrl: "https://github.com/Eabusham2/Omni-AGI", schemas, records };
  const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object"
    ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => [key, canonical(child)])) : value;
  return { ...body, sha256: createHash("sha256").update(JSON.stringify(canonical(body))).digest("hex") };
}
