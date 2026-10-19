import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { declaredDocumentationCurriculumPack } from "../src/main/documentationCurriculumPack";

const sha256 = (value: string): string =>
  createHash("sha256").update(value, "utf8").digest("hex");

describe("action-canvas documentation curriculum declaration", () => {
  it("reviews pack 23 as the existing eight ordinary records with current canvas instructions", () => {
    const pack = declaredDocumentationCurriculumPack();
    expect(pack).toMatchObject({
      format: "omni-ordinary-document-curriculum",
      version: 23,
      id: "omni-v2-user-documentation-23",
      ordinaryNeuralLearning: true,
      hiddenPrompt: false,
      behavioralPersona: false,
      executedHostActions: false
    });
    expect(pack.records.map((record) => record.id)).toEqual([
      "user-guide",
      "integrations",
      "plugins",
      "html-artifacts",
      "executable-training-parsers",
      "managed-environments",
      "clipboard",
      "api-teacher-model-discovery"
    ]);

    const guide = pack.records.find((record) => record.id === "user-guide")!;
    const integrations = pack.records.find((record) => record.id === "integrations")!;
    expect(guide.content).toContain("### Arrange an action file on the canvas");
    expect(guide.content).toContain("Save** writes an inert,\ncontent-addressed action-file revision and does not run it");
    expect(integrations.content).toContain("## Action-file canvas metadata");
    expect(integrations.content).toContain("the `steps` array remains the only execution order");
    expect(integrations.content).toContain("Pack version 23");
    for (const record of [guide, integrations]) {
      expect(record.contentSha256).toBe(sha256(record.content));
      expect(record.bytes).toBe(Buffer.byteLength(record.content));
      expect(record).not.toHaveProperty("toolId");
      expect(record).not.toHaveProperty("action");
      expect(record).not.toHaveProperty("successfulOutcome");
    }
    const recordPaths: string[] = pack.records.map((record) => record.path);
    expect(recordPaths).not.toContain("docs/v2/action-canvas.md");
    expect(pack.sha256).toMatch(/^[a-f0-9]{64}$/);
  });
});
