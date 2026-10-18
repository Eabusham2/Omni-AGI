import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { AutomationStore } from "../src/main/automationStore";
import { automationCanonical, automationHash, validateAutomationProgram } from "../src/main/automationFormat";
import { cloneLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { actionCanvasMode, actionCanvasProgram, actionPositions, actionSequenceCurve, addActionNode, moveActionNode, removeActionNode, reorderActionStep } from "../src/renderer/src/actionCanvas";
import type { AutomationDefinition, AutomationProgram } from "../src/shared/automation";

const program = (): AutomationProgram => ({ format: "omni-host-action-file", version: 1, name: "Canvas fixture",
  steps: [{ id: "first", kind: "tool", toolId: "system.time", action: "now", arguments: {} }, { id: "later", kind: "wait", delayMs: 250 }] });
const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function directory() { const root = await mkdtemp(join(tmpdir(), "omni-action-canvas-")); roots.push(root); return root; }

describe("inert grid/freeform action canvas", () => {
  it("indexes every position in a large action file without repeated linear searches or omitted tails", () => {
    const value = program(); value.steps = Array.from({ length: 2001 }, (_, index) => ({ id: `step-${index}`, kind: "wait", delayMs: index }));
    const positions = value.steps.map((step, index) => ({ stepId: step.id, x: index, y: index * 2 }));
    Object.defineProperty(positions, "find", { value: () => { throw new Error("Repeated linear scan is forbidden in the whole-canvas projection."); } });
    value.canvas = { version: 1, mode: "freeform", positions };
    expect(actionPositions(value)).toHaveLength(2001);
    const grid = actionCanvasMode(value, "grid");
    expect(grid.canvas?.positions.at(-1)).toEqual({ stepId: "step-2000", x: 2000, y: 4000 });
    expect(grid.steps).toEqual(value.steps);
  });
  it("preserves every executable field and original input through view changes and movement", () => {
    const input = program(), before = JSON.stringify(input);
    const freeform = moveActionNode(actionCanvasMode(input, "freeform"), "later", 71.5, 99);
    const grid = actionCanvasMode(freeform, "grid");
    expect(grid.steps).toEqual(input.steps); expect(JSON.stringify(input)).toBe(before);
    expect(grid.canvas?.positions.find(row => row.stepId === "later")).toEqual({ stepId: "later", x: 71.5, y: 99 });
    expect(validateAutomationProgram(grid)).toEqual(grid);
    expect(actionSequenceCurve({ x: 0, y: 0 }, { x: 300, y: 100 })).toMatch(/^M .* C /);
  });
  it("adds unique typed steps, reorders explicitly and removes only the chosen position", () => {
    let value = addActionNode(addActionNode(actionCanvasMode(program(), "freeform"), "tool"), "wait");
    expect(value.steps.map(row => row.id)).toEqual(["first", "later", "step-1", "step-2"]);
    value = reorderActionStep(value, "step-2", -1);
    expect(value.steps.map(row => row.id)).toEqual(["first", "later", "step-2", "step-1"]);
    value = removeActionNode(value, "later");
    expect(value.canvas?.positions.some(row => row.stepId === "later")).toBe(false);
    expect(validateAutomationProgram(value)).toEqual(value);
    expect(() => removeActionNode({ ...program(), steps: [program().steps[0]!] }, "first")).toThrow(/at least one/);
  });
  it("refuses edits that would silently break prior-step inputs, retaining literal lookalikes", () => {
    const value = program(); value.steps[1] = { id: "later", kind: "tool", toolId: "fixture.echo", action: "echo",
      arguments: { input: { $automationRef: { source: "step", stepId: "first", path: [] } } } };
    expect(() => reorderActionStep(value, "later", -1)).toThrow(/dependency/);
    expect(() => removeActionNode(value, "first")).toThrow(/dependency/);
    value.steps[1].arguments = { data: { $automationLiteral: { $automationRef: { source: "step", stepId: "first", path: [] } } } };
    expect(removeActionNode(value, "first").steps).toHaveLength(1);
  });
  it("rejects forged/duplicate/malformed layout metadata rather than executing or repairing it", () => {
    const value = actionCanvasMode(program(), "freeform");
    for (const positions of [[{ stepId: "foreign", x: 0, y: 0 }], [{ stepId: "first", x: 0, y: 0 }, { stepId: "first", x: 1, y: 1 }], [{ stepId: "first", x: -1, y: 0 }], [{ stepId: "first", x: Infinity, y: 0 }]]) {
      expect(() => validateAutomationProgram({ ...value, canvas: { ...value.canvas, positions } })).toThrow();
    }
    expect(() => validateAutomationProgram({ ...value, canvas: { ...value.canvas, execute: true } })).toThrow(/unsupported fields/);
    expect(actionCanvasProgram(JSON.stringify({ ...value, canvas: { version: 1, mode: "freeform", positions: null } }))).toBeUndefined();
    expect(actionCanvasProgram("unfinished {" )).toBeUndefined();
    expect(actionCanvasProgram(JSON.stringify(value))).toEqual(value);
    expect(() => moveActionNode(value, "missing", 0, 0)).toThrow(/no longer/);
    expect(() => moveActionNode(value, "first", NaN, 0)).toThrow(/finite/);
  });
  it("retains layout in exact content-addressed storage and dormant copied workspace without running anything", async () => {
    const source = await directory(), target = await directory();
    const value = moveActionNode(actionCanvasMode(program(), "freeform"), "first", 33, 44);
    const original = await AutomationStore.open(source, "source-brain"); let sha: string;
    try {
      const saved = await original.saveProgram(value); sha = saved.sha256;
      expect(sha).toBe(automationHash(automationCanonical(value)));
      original.put({ id: "canvas-file", brainId: "source-brain", revision: 1, revisionSha256: sha, name: value.name,
        createdAt: "2026-10-13T19:00:00.000Z", updatedAt: "2026-10-13T19:00:00.000Z" }, "definition");
      expect(original.page("run").total).toBe(0);
    } finally { original.close(); }
    const bytes = await readFile(join(source, "automation/programs", `${sha!}.omni-action.json`));
    await cloneLearningWorkspace(source, target, "source-brain", "copied-brain");
    const copy = await AutomationStore.open(target, "copied-brain", true);
    try {
      expect(await copy.program(sha!)).toEqual(value);
      expect(copy.get<AutomationDefinition>("definition", "canvas-file")?.activation?.enabled).not.toBe(true);
      expect(copy.page("run").total).toBe(0);
      expect(await readFile(join(target, "automation/programs", `${sha!}.omni-action.json`))).toEqual(bytes);
      expect(await readFile(join(source, "automation/programs", `${sha!}.omni-action.json`))).toEqual(bytes);
    } finally { copy.close(); }
  });
});
