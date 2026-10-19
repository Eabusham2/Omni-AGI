/** Public pack equality only: no archive, app, plugin, brain, learner or provider executes. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, it } from "vitest";
import { declaredDocumentationCurriculumPack } from "../src/main/documentationCurriculumPack";

const sha256 = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");
const normalized = (text: string): string => `${text.replace(/\r\n?/g, "\n").trim()}\n`;
const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object"
  ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => [key, canonical(child)])) : value;

it("binds exact pack23 public portable instructions and all eight ordinary manuals without private authority or executed outcomes", () => {
  const pack = declaredDocumentationCurriculumPack();
  expect(pack).toMatchObject({ format: "omni-ordinary-document-curriculum", version: 23, id: "omni-v2-user-documentation-23",
    ordinaryNeuralLearning: true, hiddenPrompt: false, behavioralPersona: false, preferenceLabels: false, rewardModel: false,
    executedHostActions: false, sourceKind: "project-authored-user-documentation" });
  expect(pack.records.map(row => row.path)).toEqual([
    "docs/v2/USER_GUIDE.md", "docs/v2/INTEGRATIONS.md", "docs/v2/plugins.md", "docs/v2/html-artifacts.md",
    "docs/v2/executable-training-parsers.md", "docs/v2/managed-environments.md", "docs/v2/clipboard.md",
    "docs/v2/api-teacher-model-discovery.md"
  ]);
  for (const row of pack.records) {
    const source = normalized(readFileSync(resolve(row.path), "utf8"));
    expect(row.content).toBe(source); expect(row.bytes).toBe(Buffer.byteLength(source)); expect(row.contentSha256).toBe(sha256(source));
    expect(row.provenanceUrl).toBe(`https://github.com/Eabusham2/Omni-AGI/blob/main/${row.path}`);
    expect(row).not.toHaveProperty("action"); expect(row).not.toHaveProperty("successfulOutcome");
  }
  const { records, sha256: packHash, ...metadata } = pack;
  expect(packHash).toBe(sha256(JSON.stringify(canonical({ ...metadata, records: records.map(({ content: _content, ...row }) => row) }))));
  const guide = records.find(row => row.id === "user-guide")!.content, integrations = records.find(row => row.id === "integrations")!.content;
  for (const phrase of ["## Portable Windows folder", "Create portable folder", "Prepare saved files", "Review destination",
    "Approve this resource fit", "unsanitized private archive", "Tool", "logical\nshared pool", "not an empty padded 20 GB",
    "does not\nautomatically replace them with the original archive", "do not transfer as usable grants"])
    expect(guide).toContain(phrase);
  for (const phrase of ["Pack version 23", "window.omni.portable", "prepare({ requestId })", "reviewDestination()",
    "approveDestination({ reviewId, reviewSha256 })", "Tool access is separate", "No padded 20 GB asset",
    "device-owned outside", "copied external authority dormant", "ordinary installer data locations"])
    expect(integrations).toContain(phrase);
});
