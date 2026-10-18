import { expect, it } from "vitest";
import { declaredDocumentationCurriculumPack } from "../src/main/documentationCurriculumPack";
it("declares separate advice as ordinary user documentation without executing or substituting a provider", () => {
  const pack = declaredDocumentationCurriculumPack();
  expect(pack).toMatchObject({ version: 22, id: "omni-v2-user-documentation-22", executedHostActions: false, hiddenPrompt: false, rewardModel: false });
  expect(pack.records).toHaveLength(8);
  const guide = pack.records.find(item => item.id === "user-guide")!.content;
  const integrations = pack.records.find(item => item.id === "integrations")!.content;
  expect(guide).toContain("### Separate assistant and action reviewer");
  expect(guide).toContain("drafting does not send it automatically");
  expect(integrations).toContain("## Separate external advice");
  expect(integrations).toContain("provider calls or inheriting device consent");
  expect(integrations).toContain("Pack version 22");
  expect(pack.records.map(item => String(item.path))).not.toContain("docs/v2/external-advice.md");
});
