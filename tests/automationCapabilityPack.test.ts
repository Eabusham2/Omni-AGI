import Ajv from "ajv";
import { describe, expect, it } from "vitest";
import { declaredAutomationCapabilityPack } from "../src/main/automationCapabilityPack";
import { declaredClockCapabilityPack } from "../src/main/clockCapabilityPack";
import { validateAutomationProgram } from "../src/main/automationFormat";
describe("separately declared independent host action semantics", () => {
  it("validates every offered operation without pretending placeholder tools ran", () => {
    const pack = declaredAutomationCapabilityPack(), ajv = new Ajv({ strict: false });
    expect(pack.id).toBe("independent-host-action-files-1"); expect(pack.id).not.toBe(declaredClockCapabilityPack().id);
    expect(pack.executedHostActions).toBe(false); expect(pack.rewardModel).toBe(false); expect(pack.preferenceLabels).toBe(false);
    expect(pack.sha256).toMatch(/^[a-f0-9]{64}$/); expect(declaredAutomationCapabilityPack()).toEqual(pack);
    for (const row of pack.records) { const schema = pack.schemas.find(schema => schema.id === row.toolId)!; expect(schema.actions).toContain(row.action);
      expect(ajv.compile(schema.actionInputSchemas[row.action!]!)(row.arguments)).toBe(true);
      if (row.action === "save") expect(() => validateAutomationProgram(row.arguments.program)).not.toThrow(); }
    for (const schema of pack.schemas) for (const action of schema.actions) expect(pack.records.some(row => row.toolId === schema.id && row.action === action)).toBe(true);
    expect(JSON.stringify(pack)).not.toContain("systemPrompt");
  });
});
