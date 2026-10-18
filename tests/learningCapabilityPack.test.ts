import { describe, expect, it } from "vitest";
import { declaredLearningCapabilityPack } from "../src/main/learningCapabilityPack";
import { committedNativeLearningTask } from "../src/main/learningNativeTask";
import type { ChatResult } from "../src/shared/types";

describe("supplementary declared capability contract", () => {
  it("covers every learned data tool action without claiming host execution or changing the origin curriculum", () => {
    const pack = declaredLearningCapabilityPack();
    expect(pack.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(pack.sourceKind).toBe("project-authored-declared-examples");
    expect(pack.executedHostActions).toBe(false);
    for (const schema of pack.schemas) for (const action of schema.actions) {
      expect(pack.records.some(row => row.toolId === schema.id && row.action === action)).toBe(true);
    }
    expect(new Set(pack.records.map(row => row.utterance)).size).toBe(pack.records.length);
    expect(pack.records.some(row => row.kind === "learn" && row.toolId === null)).toBe(true);
    expect(JSON.stringify(pack)).not.toContain("ground_up_training_manifest");
    expect(declaredLearningCapabilityPack()).toEqual(pack);
  });
  it("requires exact terminal native task correlation rather than accepting arbitrary completed prose", () => {
    const expected = { turnId: "actual-task-turn", input: '{"kind":"user-requested-fact-check"}' };
    const result = { humanMessage: { turnId: expected.turnId, content: expected.input, status: "complete" },
      brainMessage: { turnId: expected.turnId, status: "complete", traceId: "actual-trace", createdAt: "2026-10-10T12:00:00.000Z" }, trace: { id: "actual-trace" } } as ChatResult;
    expect(committedNativeLearningTask(result, expected)).toEqual({ turnId: expected.turnId, committedAt: result.brainMessage.createdAt });
    expect(() => committedNativeLearningTask({ ...result, humanMessage: { ...result.humanMessage, turnId: "other" } }, expected)).toThrow(/terminal/);
    expect(() => committedNativeLearningTask({ ...result, generationEnd: "steered" }, expected)).toThrow(/terminal/);
    expect(() => committedNativeLearningTask({ ...result, brainMessage: { ...result.brainMessage, status: "error" } }, expected)).toThrow(/terminal/);
  });
});
