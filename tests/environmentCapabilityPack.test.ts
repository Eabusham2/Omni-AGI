import Ajv from "ajv";
import { describe, expect, it } from "vitest";
import { declaredEnvironmentCapabilityPack } from "../src/main/environmentCapabilityPack";

describe("managed environment declared capability material", () => {
  it("is deterministic schema-valid ordinary data with no fabricated execution", () => {
    const pack = declaredEnvironmentCapabilityPack(), ajv = new Ajv({ strict: false });
    expect(pack).toEqual(declaredEnvironmentCapabilityPack());
    expect(pack).toMatchObject({ id: "managed-sandbox-vm-tools-1", executedHostActions: false, rewardModel: false, preferenceLabels: false });
    expect(pack.sha256).toMatch(/^[a-f0-9]{64}$/); expect(pack.schemas.map(schema => schema.id)).toEqual(["runtime.sandbox", "runtime.vm"]);
    for (const record of pack.records) {
      if (!record.toolId || !record.action) throw new Error("Environment capability records must be typed tool actions.");
      const schema = pack.schemas.find(item => item.id === record.toolId)?.actionInputSchemas[record.action];
      if (!schema) throw new Error("Environment capability action lost its schema.");
      expect(ajv.validateSchema(schema), `${JSON.stringify(schema)} ${JSON.stringify(ajv.errors)}`).toBe(true);
      expect(ajv.validate(schema, record.arguments), JSON.stringify(ajv.errors)).toBe(true);
    }
    for (const schema of pack.schemas) for (const action of schema.actions)
      expect(pack.records.some(record => record.toolId === schema.id && record.action === action)).toBe(true);
  });
});
