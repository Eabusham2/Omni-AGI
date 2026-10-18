import type { AutomationCanvas, AutomationProgram, AutomationStep } from "../../shared/automation";

export const ACTION_NODE_WIDTH = 224;
export const ACTION_NODE_HEIGHT = 88;
/** Only rendering admission, never a substitute for host action-file validation. */
export function actionCanvasProgram(text: string): AutomationProgram | undefined {
  try {
    const value = JSON.parse(text) as AutomationProgram;
    if (!value || value.format !== "omni-host-action-file" || value.version !== 1 || typeof value.name !== "string" ||
        !Array.isArray(value.steps) || !value.steps.length) return;
    const ids = new Set<string>();
    for (const step of value.steps) {
      if (!step || typeof step.id !== "string" || ids.has(step.id)) return;
      ids.add(step.id);
      if (step.kind === "wait" ? !Number.isSafeInteger(step.delayMs) || step.delayMs < 0
        : step.kind !== "tool" || typeof step.toolId !== "string" || typeof step.action !== "string" ||
          !step.arguments || typeof step.arguments !== "object" || Array.isArray(step.arguments)) return;
    }
    if (value.canvas !== undefined) {
      const canvas = value.canvas, seen = new Set<string>();
      if (!canvas || canvas.version !== 1 || !["grid", "freeform"].includes(canvas.mode) || !Array.isArray(canvas.positions)) return;
      for (const point of canvas.positions) {
        if (!point || !ids.has(point.stepId) || seen.has(point.stepId) || ![point.x, point.y].every(n => typeof n === "number" && Number.isFinite(n) && n >= 0 && n <= Number.MAX_SAFE_INTEGER)) return;
        seen.add(point.stepId);
      }
    }
    return value;
  } catch { return; }
}
export function defaultActionPosition(index: number): { x: number; y: number } {
  return { x: 24 + (index % 3) * 272, y: 24 + Math.floor(index / 3) * 136 };
}
export function actionPosition(program: AutomationProgram, stepId: string, index: number): { x: number; y: number } {
  return program.canvas?.positions.find(point => point.stepId === stepId) ?? defaultActionPosition(index);
}
export function actionPositions(program: AutomationProgram): Array<{ x: number; y: number }> {
  const saved = new Map(program.canvas?.positions.map(point => [point.stepId, point]));
  return program.steps.map((step, index) => saved.get(step.id) ?? defaultActionPosition(index));
}
export function actionCanvasMode(program: AutomationProgram, mode: AutomationCanvas["mode"]): AutomationProgram {
  const positions = actionPositions(program);
  return { ...program, canvas: { version: 1, mode, positions: program.steps.map((step, index) => ({ stepId: step.id, ...positions[index]! })) } };
}
export function moveActionNode(program: AutomationProgram, stepId: string, x: number, y: number): AutomationProgram {
  if (!program.steps.some(step => step.id === stepId)) throw new Error("This step is no longer in the action file.");
  if (![x, y].every(Number.isFinite)) throw new Error("Enter finite canvas coordinates.");
  const next = actionCanvasMode(program, "freeform");
  return { ...next, canvas: { ...next.canvas!, positions: next.canvas!.positions.map(point => point.stepId === stepId
    ? { stepId, x: Math.min(Number.MAX_SAFE_INTEGER, Math.max(0, x)), y: Math.min(Number.MAX_SAFE_INTEGER, Math.max(0, y)) } : point) } };
}
export function reorderActionStep(program: AutomationProgram, stepId: string, offset: -1 | 1): AutomationProgram {
  const index = program.steps.findIndex(step => step.id === stepId), target = index + offset;
  if (index < 0 || target < 0 || target >= program.steps.length) return program;
  const steps = [...program.steps]; [steps[index], steps[target]] = [steps[target]!, steps[index]!];
  assertActionDependencies(steps);
  return { ...program, steps };
}
function referencedSteps(value: unknown, result: Set<string>): void {
  if (Array.isArray(value)) { value.forEach(item => referencedSteps(item, result)); return; }
  if (!value || typeof value !== "object") return;
  const row = value as Record<string, unknown>;
  if (Object.keys(row).length === 1 && "$automationLiteral" in row) return;
  if (Object.keys(row).length === 1 && "$automationRef" in row) {
    const ref = row.$automationRef as { source?: string; stepId?: string } | null;
    if (ref?.source === "step" && typeof ref.stepId === "string") result.add(ref.stepId);
  } else Object.values(row).forEach(item => referencedSteps(item, result));
}
export function assertActionDependencies(steps: readonly AutomationStep[]): void {
  const preceding = new Set<string>();
  for (const step of steps) {
    const required = new Set<string>();
    if (step.kind === "tool") referencedSteps(step.arguments, required);
    if (step.when?.reference.source === "step" && step.when.reference.stepId) required.add(step.when.reference.stepId);
    if ([...required].some(id => !preceding.has(id))) throw new Error("This edit would break a prior-step dependency. Edit its reference in JSON first.");
    preceding.add(step.id);
  }
}
export function removeActionNode(program: AutomationProgram, stepId: string): AutomationProgram {
  if (program.steps.length === 1) throw new Error("An action file needs at least one step.");
  // References are not rewritten: changing an authored dependency needs an
  // explicit edit, never guessed reconnection or automatic execution.
  const steps = program.steps.filter(step => step.id !== stepId);
  assertActionDependencies(steps);
  return { ...program, steps,
    ...(program.canvas ? { canvas: { ...program.canvas, positions: program.canvas.positions.filter(point => point.stepId !== stepId) } } : {}) };
}
export function addActionNode(program: AutomationProgram, kind: AutomationStep["kind"]): AutomationProgram {
  const ids = new Set(program.steps.map(step => step.id)); let suffix = 1;
  while (ids.has(`step-${suffix}`)) suffix++;
  const id = `step-${suffix}`;
  const step: AutomationStep = kind === "wait" ? { id, kind, delayMs: 1000 }
    : { id, kind, toolId: "system.time", action: "now", arguments: {} };
  const next = { ...program, steps: [...program.steps, step] };
  return program.canvas ? actionCanvasMode(next, program.canvas.mode) : next;
}
/** Curves depict sequence, not parallel edges or invented runtime activity. */
export function actionSequenceCurve(from: { x: number; y: number }, to: { x: number; y: number }): string {
  const x1 = from.x + ACTION_NODE_WIDTH, y1 = from.y + ACTION_NODE_HEIGHT / 2;
  const x2 = to.x, y2 = to.y + ACTION_NODE_HEIGHT / 2;
  const bend = Math.max(36, Math.abs(x2 - x1) / 2);
  return `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`;
}
