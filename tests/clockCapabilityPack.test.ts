import Ajv from "ajv";
import { describe, expect, it } from "vitest";
import { declaredClockCapabilityPack } from "../src/main/clockCapabilityPack";
import { declaredLearningCapabilityPack } from "../src/main/learningCapabilityPack";

describe("declared clock capability pack (no model training)", () => {
  it("keeps the frozen prior pack separate and validates every declared route against its schema", () => {
    const pack = declaredClockCapabilityPack(), prior = declaredLearningCapabilityPack(), ajv = new Ajv({ strict: false });
    expect(pack.id).toBe("host-clocks-scheduled-wakeups-2"); expect(pack.id).not.toBe(prior.id);
    expect(pack.executedHostActions).toBe(false); expect(pack.preferenceLabels).toBe(false); expect(pack.rewardModel).toBe(false);
    expect(pack.sha256).toMatch(/^[a-f0-9]{64}$/); expect(declaredClockCapabilityPack()).toEqual(pack);
    for (const record of pack.records) {
      const schema = pack.schemas.find(row => row.id === record.toolId);
      expect(schema?.actions).toContain(record.action);
      const input = schema!.actionInputSchemas[record.action!]; expect(input).toBeDefined();
      expect(ajv.compile(input!)(record.arguments)).toBe(true);
    }
    for (const schema of pack.schemas) for (const action of schema.actions) expect(pack.records.some(row => row.toolId === schema.id && row.action === action)).toBe(true);
    expect(JSON.stringify(pack)).not.toContain("systemPrompt"); expect(prior.id).toBe("specific-learn-fact-check-1");
  });
});
