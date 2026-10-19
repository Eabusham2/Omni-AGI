import { createHash } from "node:crypto";
import { expect, it } from "vitest";
import { declaredDocumentationCurriculumPack } from "../src/main/documentationCurriculumPack";

it("declares current attachment, teacher and replay controls as v23 ordinary manuals without simulated outcomes", () => {
  const pack = declaredDocumentationCurriculumPack();
  expect(pack).toMatchObject({ version: 23, id: "omni-v2-user-documentation-23", executedHostActions: false, hiddenPrompt: false, rewardModel: false });
  const guide = pack.records.find(row => row.id === "user-guide")!, integrations = pack.records.find(row => row.id === "integrations")!;
  expect(guide.content).toContain("attachments, not an automatic training job");
  expect(guide.content).toContain("no\n   blocking conversational tool quiz");
  expect(guide.content).toContain("Choose from provider models");
  expect(guide.content).toContain("Local GGUF teacher");
  expect(guide.content).toContain("Prepare replay");
  expect(guide.content).toContain("timer pauses while hovered or focused");
  expect(integrations.content).toContain("localTeacher");
  expect(integrations.content).toContain("Pack version 23");
  for (const row of pack.records) {
    expect(row.contentSha256).toBe(createHash("sha256").update(row.content).digest("hex"));
    expect(row.bytes).toBe(Buffer.byteLength(row.content));
    expect(row).not.toHaveProperty("successfulOutcome");
  }
});

it("includes current operation, map and MCP controls in ordinary manuals, not audit or private-chat records", () => {
  const pack = declaredDocumentationCurriculumPack();
  const guide = pack.records.find(row => row.id === "user-guide")!.content;
  const integrations = pack.records.find(row => row.id === "integrations")!.content;
  for (const phrase of ["Advanced · automatic operation placement", "Enable Auto for future operations", "Stop Auto",
    "Optional basemaps", "Filesystem disclosures and diagnostics", "format2"]) expect(guide).toContain(phrase);
  for (const phrase of ["window.omni.nativeOperationAuto", "stageCoverageByWorkKind", "maps.basemaps.list/save/remove",
    "mcp.roots.status/select/review/clear", "mcp.diagnostics.page"]) expect(integrations).toContain(phrase);
  expect(pack.records.some(row => /REQUEST_VERBATIM|AMENDMENTS|CONTINUATION|test-results|pasted-text/.test(row.path))).toBe(false);
});
