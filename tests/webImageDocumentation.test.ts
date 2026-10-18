/** R023 manual/version source only; no neural or host action is executed. */
import { createHash } from "node:crypto";
import { expect, it } from "vitest";
import { declaredDocumentationCurriculumPack } from "../src/main/documentationCurriculumPack";

it("bakes actual selected-web-image semantics into the explicitly reviewed v18 ordinary manual corpus", () => {
  const pack = declaredDocumentationCurriculumPack();
  expect(pack.version).toBe(22); expect(pack.id).toBe("omni-v2-user-documentation-22");
  expect(pack.executedHostActions).toBe(false); expect(pack.hiddenPrompt).toBe(false);
  const guide = pack.records.find(row => row.id === "user-guide")!, integrations = pack.records.find(row => row.id === "integrations")!;
  for (const row of [guide, integrations]) {
    expect(row.content).toContain("inspect-image");
    expect(row.content).toContain("web-image-inspection.md");
    expect(row.contentSha256).toBe(createHash("sha256").update(row.content).digest("hex"));
    expect(row.bytes).toBe(Buffer.byteLength(row.content));
  }
  expect(guide.content).toContain("lasting learning happens later");
  expect(integrations.content).toContain("imageReferences");
  expect(integrations.content).toContain("web-image-tools-1");
  expect(integrations.content).toContain("Pack version 22");
});
