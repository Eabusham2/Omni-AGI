import { createHash } from "node:crypto";
import catalog from "../../tools/catalog.json";
import type { DeclaredCapabilityPack } from "./learningCapabilityPack";

/** New versioned declared examples, never edits to frozen origin-v3 or fake tool outcomes. */
export function declaredClockCapabilityPack(): DeclaredCapabilityPack {
  const schemas = catalog.tools.filter(tool => ["system.time", "studio.stopwatch", "studio.schedule", "studio.question"].includes(tool.id)).map(tool => ({
    id: tool.id, actions: Object.keys(tool.actions), grant: "ask",
    actionInputSchemas: Object.fromEntries(Object.entries(tool.actions).map(([action, definition]) => {
      const input = (definition as { input: Record<string, string> }).input;
      return [action, { type: "object", additionalProperties: false,
        properties: Object.fromEntries(Object.entries(input).map(([key, descriptor]) => {
          const base = descriptor.replace(/\?$/, ""), list = base.endsWith("[]"), item = list ? base.slice(0, -2) : base;
          const schema = item.includes("|") ? { type: "string", enum: item.split("|") } : { type: item };
          return [key, list ? { type: "array", items: schema } : schema];
        })), required: Object.entries(input).filter(([, value]) => !value.endsWith("?")).map(([key]) => key) }];
    }))
  }));
  const records: DeclaredCapabilityPack["records"] = [
    { utterance: "What date and time is it here, including my time zone?", kind: "tool", toolId: "system.time", action: "now", arguments: {} },
    { utterance: "Measure this task with a stopwatch.", kind: "tool", toolId: "studio.stopwatch", action: "start", arguments: { label: "Current task" } },
    { utterance: "Record a lap on the selected stopwatch.", kind: "tool", toolId: "studio.stopwatch", action: "lap", arguments: { stopwatchId: "selected-stopwatch-id" } },
    { utterance: "Inspect actual previous output speed; do not guess a future completion time.", kind: "tool", toolId: "studio.stopwatch", action: "performance", arguments: {} },
    { utterance: "Notify me in ten minutes.", kind: "tool", toolId: "studio.schedule", action: "create", arguments: { label: "Ten-minute reminder", kind: "timer", delayMs: 600000, delivery: "notify" } },
    { utterance: "In an hour, continue checking this report with my existing permissions.", kind: "tool", toolId: "studio.schedule", action: "create", arguments: { label: "Continue report", kind: "timer", delayMs: 3600000, delivery: "wake", task: "Continue checking the report." } },
    { utterance: "In an hour, load this instance if resources permit and run the saved report task.", kind: "tool", toolId: "studio.schedule", action: "create", arguments: { label: "Explicit scheduled Load", kind: "timer", delayMs: 3600000, delivery: "wake", task: "Continue checking the report.", loadWhenDue: true } },
    { utterance: "Schedule this saved task at the explicit calendar instant.", kind: "tool", toolId: "studio.schedule", action: "create", arguments: { label: "Calendar task", kind: "at", at: "2030-01-02T09:00:00+00:00", timeZone: "UTC", delivery: "wake", task: "Review the selected work." } },
    { utterance: "Run the saved review every hour, coalescing missed occurrences rather than replaying a backlog.", kind: "tool", toolId: "studio.schedule", action: "create", arguments: { label: "Hourly review", kind: "timer", delayMs: 3600000, repeatMs: 3600000, missedPolicy: "coalesce", delivery: "wake", task: "Review the selected work using current access." } },
    { utterance: "Use the source-changed tracker event to notify me once per unique event.", kind: "tool", toolId: "studio.schedule", action: "create", arguments: { label: "Source tracker", kind: "event", eventKey: "source.changed", delivery: "notify" } },
    { utterance: "Hand this timer to the explicitly enabled action file without loading the brain or submitting a chat prompt.", kind: "tool", toolId: "studio.schedule", action: "create", arguments: { label: "Selected host action", kind: "timer", delayMs: 60000, delivery: "automation", loadWhenDue: false,
      automation: { definitionId: "selected-action-id", revisionSha256: "a".repeat(64), inputs: {} } } },
    { utterance: "Pause the selected schedule without deleting its saved task or receipts.", kind: "tool", toolId: "studio.schedule", action: "pause", arguments: { scheduleId: "selected-schedule-id" } },
    { utterance: "Resume the selected schedule after reviewing its current permissions.", kind: "tool", toolId: "studio.schedule", action: "resume", arguments: { scheduleId: "selected-schedule-id" } },
    { utterance: "Cancel future reminders; do not undo the task's completed effects.", kind: "tool", toolId: "studio.schedule", action: "cancel", arguments: { scheduleId: "selected-schedule-id" } },
    { utterance: "For this reminder, ask whether I want ongoing Active mode or a scheduled wakeup.", kind: "tool", toolId: "studio.question", action: "ask", arguments: {
      prompt: "Keep working in Active mode, or wait for a scheduled wakeup?", mode: "single", options: [
        { id: "scheduled", label: "Scheduled wakeup", description: "Wait for the saved time; an unloaded mind still requires Load.", recommended: true },
        { id: "active", label: "Active mode", description: "Use the existing ongoing activity controls and current tool permissions." } ] } }
  ];
  for (const schema of schemas) for (const action of schema.actions) {
    if (records.some(row => row.toolId === schema.id && row.action === action)) continue;
    let args: Record<string, unknown> = {};
    if (schema.id === "studio.stopwatch" && !["list", "performance"].includes(action)) args = { stopwatchId: "selected-stopwatch-id" };
    if (schema.id === "studio.schedule" && action === "edit") args = { scheduleId: "selected-schedule-id", expectedRevision: 1, label: "Reviewed updated reminder" };
    else if (schema.id === "studio.schedule" && action === "signal") args = { eventKey: "source.changed", eventId: "source-event-one", data: { changed: true } };
    else if (schema.id === "studio.schedule" && ["resume-occurrence", "occurrence"].includes(action)) args = { occurrenceId: "selected-wakeup-id" };
    else if (schema.id === "studio.schedule" && !["list", "occurrences"].includes(action)) args = { scheduleId: "selected-schedule-id" };
    if (schema.id === "studio.question") args = { questionId: "a".repeat(64) };
    records.push({ utterance: `Use ${schema.id} ${action} for the selected saved record through current access.`, kind: "tool", toolId: schema.id, action, arguments: args });
  }
  const body = { format: "omni-declared-capability-pack", version: 1, id: "host-clocks-scheduled-wakeups-2",
    sourceKind: "project-authored-declared-examples", executedHostActions: false, preferenceLabels: false, rewardModel: false,
    license: "PolyForm-Noncommercial-1.0.0-or-commercial-license", provenanceUrl: "https://github.com/Eabusham2/Omni-AGI", schemas, records };
  const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object"
    ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => [key, canonical(child)])) : value;
  return { ...body, sha256: createHash("sha256").update(JSON.stringify(canonical(body))).digest("hex") };
}
