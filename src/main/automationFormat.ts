import { createHash } from "node:crypto";
import type { AutomationCondition, AutomationJson, AutomationProgram, AutomationReferenceValue, AutomationRun, AutomationStep, ScheduledAutomationReference } from "../shared/automation";

export const automationHash = (text: string): string => createHash("sha256").update(text).digest("hex");
export function automationCanonical(value: unknown): string {
  if (value === null || ["boolean", "string"].includes(typeof value)) return JSON.stringify(value);
  if (typeof value === "number" && Number.isFinite(value)) return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(automationCanonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, item]) => `${JSON.stringify(key)}:${automationCanonical(item)}`).join(",")}}`;
  throw new Error("Action files contain only finite JSON values.");
}
export function automationId(value: unknown): string {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value)) throw new Error("Invalid automation identity.");
  return value;
}
export function automationSha(value: unknown): string {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) throw new Error("Invalid action-file revision hash.");
  return value;
}
export function automationObject(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Expected an action-file object.");
  return value as Record<string, unknown>;
}
function known(value: Record<string, unknown>, keys: readonly string[]): void {
  if (Object.keys(value).some(key => !keys.includes(key))) throw new Error("Action file has unsupported fields.");
}
export function validateScheduledAutomationReference(value: unknown): ScheduledAutomationReference {
  const row = automationObject(value); known(row, ["definitionId", "revisionSha256", "inputs"]);
  automationId(row.definitionId); automationSha(row.revisionSha256); automationObject(row.inputs); automationCanonical(row.inputs);
  return JSON.parse(automationCanonical(row)) as ScheduledAutomationReference;
}
function text(value: unknown, name: string): void {
  if (typeof value !== "string" || !value.trim() || value.includes("\0")) throw new Error(`Enter ${name}.`);
}
function reference(value: unknown): asserts value is AutomationReferenceValue {
  const row = automationObject(value); known(row, ["source", "path", "stepId"]);
  if (!["input", "event", "step"].includes(String(row.source)) || !Array.isArray(row.path) ||
      row.path.some(key => typeof key !== "string" && !(Number.isSafeInteger(key) && Number(key) >= 0))) throw new Error("Invalid authored data reference.");
  if (row.source === "step") automationId(row.stepId);
  else if (row.stepId !== undefined) throw new Error("Only a prior-step reference has a step ID.");
}
function operand(value: unknown): void {
  if (Array.isArray(value)) { value.forEach(operand); return; }
  if (value && typeof value === "object") {
    const row = automationObject(value), keys = Object.keys(row);
    if (keys.length === 1 && keys[0] === "$automationRef") { reference(row.$automationRef); return; }
    if (keys.length === 1 && keys[0] === "$automationLiteral") { automationCanonical(row.$automationLiteral); return; }
    if (keys.length === 1 && keys[0] === "$automationJoin") {
      const join = automationObject(row.$automationJoin); known(join, ["parts", "separator"]);
      if (!Array.isArray(join.parts) || join.separator !== undefined && typeof join.separator !== "string") throw new Error("Invalid authored text join.");
      join.parts.forEach(operand); return;
    }
    Object.values(row).forEach(operand); return;
  }
  automationCanonical(value);
}
function condition(value: unknown): asserts value is AutomationCondition {
  const row = automationObject(value); known(row, ["reference", "operator", "value"]); reference(row.reference);
  if (!["exists", "equals", "not-equals", "greater", "at-least", "less", "at-most"].includes(String(row.operator))) throw new Error("Invalid action-file condition.");
  if (row.operator !== "exists") automationCanonical(row.value);
}
export function validateAutomationProgram(value: unknown): AutomationProgram {
  const row = automationObject(value); known(row, ["format", "version", "name", "description", "steps", "canvas"]);
  if (row.format !== "omni-host-action-file" || row.version !== 1 || !Array.isArray(row.steps) || !row.steps.length) throw new Error("Unsupported or empty action file.");
  text(row.name, "an action-file name"); if (row.description !== undefined && typeof row.description !== "string") throw new Error("Invalid action-file description.");
  const seen = new Set<string>();
  for (const item of row.steps) {
    const step = automationObject(item); const id = automationId(step.id);
    if (seen.has(id)) throw new Error("Action step IDs must be unique."); seen.add(id);
    if (step.when !== undefined) condition(step.when);
    if (step.kind === "wait") {
      known(step, ["id", "kind", "when", "delayMs"]);
      if (!Number.isSafeInteger(step.delayMs) || Number(step.delayMs) < 0 || Number(step.delayMs) > 8_640_000_000_000_000) throw new Error("Wait exceeds supported timer precision.");
    } else if (step.kind === "tool") {
      known(step, ["id", "kind", "when", "style", "toolId", "action", "arguments"]);
      if (typeof step.toolId !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$/.test(step.toolId)) throw new Error("Invalid action tool ID.");
      automationId(step.action); if (step.style !== undefined && !["tool", "script-delegate"].includes(String(step.style))) throw new Error("Invalid script delegation style.");
      automationObject(step.arguments); operand(step.arguments);
    } else throw new Error("Only timed waits and permissioned typed tools are executable action steps.");
  }
  if (row.canvas !== undefined) {
    const canvas = automationObject(row.canvas); known(canvas, ["version", "mode", "positions"]);
    if (canvas.version !== 1 || !["grid", "freeform"].includes(String(canvas.mode)) || !Array.isArray(canvas.positions)) throw new Error("Unsupported action canvas.");
    const placed = new Set<string>();
    for (const value of canvas.positions) {
      const point = automationObject(value); known(point, ["stepId", "x", "y"]);
      const id = automationId(point.stepId);
      if (!seen.has(id) || placed.has(id)) throw new Error("Canvas positions must identify distinct existing steps.");
      placed.add(id);
      // CSS pixel coordinates are presentation, never executable inputs. Reject
      // nonfinite/unsafe geometry rather than silently changing saved values.
      if (![point.x, point.y].every(n => typeof n === "number" && Number.isFinite(n) && n >= 0 && n <= Number.MAX_SAFE_INTEGER)) throw new Error("Invalid action canvas position.");
    }
  }
  return JSON.parse(automationCanonical(row)) as AutomationProgram;
}
function readReference(ref: AutomationReferenceValue, run: AutomationRun, outcomes: ReadonlyMap<string, AutomationJson>): AutomationJson | undefined {
  let value: unknown = ref.source === "input" ? run.inputs : ref.source === "event" ? run.trigger?.eventData : outcomes.get(ref.stepId!);
  for (const key of ref.path) {
    if (!value || typeof value !== "object" || !Object.hasOwn(value, key)) return undefined;
    value = (value as Record<string | number, unknown>)[key];
  }
  return value === undefined ? undefined : JSON.parse(automationCanonical(value)) as AutomationJson;
}
export function resolveAutomationArguments(step: Extract<AutomationStep, { kind: "tool" }>, run: AutomationRun,
  outcomes: ReadonlyMap<string, AutomationJson>): Record<string, unknown> {
  const resolve = (value: unknown): AutomationJson => {
    if (Array.isArray(value)) return value.map(resolve);
    if (value && typeof value === "object") {
      const row = automationObject(value), keys = Object.keys(row);
      if (keys.length === 1 && keys[0] === "$automationLiteral") return JSON.parse(automationCanonical(row.$automationLiteral)) as AutomationJson;
      if (keys.length === 1 && keys[0] === "$automationRef") {
        const result = readReference(row.$automationRef as unknown as AutomationReferenceValue, run, outcomes);
        if (result === undefined) throw new Error("An action references unavailable input/event/committed step data.");
        return result; // Incoming data is never recursively reinterpreted as code/references.
      }
      if (keys.length === 1 && keys[0] === "$automationJoin") {
        const join = automationObject(row.$automationJoin);
        return (join.parts as unknown[]).map(part => { const item = resolve(part);
          if (!["string", "number", "boolean"].includes(typeof item)) throw new Error("Text joins accept scalar data only."); return String(item); }).join(String(join.separator ?? ""));
      }
      return Object.fromEntries(Object.entries(row).map(([key, item]) => [key, resolve(item)]));
    }
    return value as AutomationJson;
  };
  return resolve(step.arguments) as Record<string, unknown>;
}
export function evaluateAutomationCondition(value: AutomationCondition | undefined, run: AutomationRun, outcomes: ReadonlyMap<string, AutomationJson>): boolean {
  if (!value) return true;
  const actual = readReference(value.reference, run, outcomes);
  if (value.operator === "exists") return actual !== undefined;
  if (actual === undefined) return false;
  if (value.operator === "equals") return automationCanonical(actual) === automationCanonical(value.value);
  if (value.operator === "not-equals") return automationCanonical(actual) !== automationCanonical(value.value);
  if (typeof actual !== "number" || typeof value.value !== "number") throw new Error("Ordered action conditions require numeric data.");
  return value.operator === "greater" ? actual > value.value : value.operator === "at-least" ? actual >= value.value :
    value.operator === "less" ? actual < value.value : actual <= value.value;
}
