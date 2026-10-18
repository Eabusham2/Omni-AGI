/** Public source equality only; no app, neural owner, hardware primitive or training executes. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, it } from "vitest";
import { declaredDocumentationCurriculumPack } from "../src/main/documentationCurriculumPack";

const hash = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");
const normalized = (text: string): string => `${text.replace(/\r\n?/g, "\n").trim()}\n`;
const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object"
  ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => [key, canonical(child)])) : value;

it("binds exact pack22 regional instructions and eight public records without invented hardware or activation outcomes", () => {
  const pack = declaredDocumentationCurriculumPack();
  expect(pack).toMatchObject({ version: 22, id: "omni-v2-user-documentation-22", format: "omni-ordinary-document-curriculum",
    sourceKind: "project-authored-user-documentation", ordinaryNeuralLearning: true, hiddenPrompt: false, behavioralPersona: false,
    preferenceLabels: false, rewardModel: false, executedHostActions: false });
  expect(pack.records.map(row => row.path)).toEqual([
    "docs/v2/USER_GUIDE.md", "docs/v2/INTEGRATIONS.md", "docs/v2/plugins.md", "docs/v2/html-artifacts.md",
    "docs/v2/executable-training-parsers.md", "docs/v2/managed-environments.md", "docs/v2/clipboard.md",
    "docs/v2/api-teacher-model-discovery.md"
  ]);
  for (const row of pack.records) {
    const content = normalized(readFileSync(resolve(row.path), "utf8"));
    expect(row.content).toBe(content); expect(row.contentSha256).toBe(hash(content)); expect(row.bytes).toBe(Buffer.byteLength(content));
    expect(row.provenanceUrl).toBe(`https://github.com/Eabusham2/Omni-AGI/blob/main/${row.path}`);
    expect(row).not.toHaveProperty("action"); expect(row).not.toHaveProperty("successfulOutcome");
  }
  const { sha256, records, ...metadata } = pack;
  expect(sha256).toBe(hash(JSON.stringify(canonical({ ...metadata, records: records.map(({ content: _content, ...row }) => row) }))));
  const guide = records.find(row => row.id === "user-guide")!.content, integrations = records.find(row => row.id === "integrations")!.content;
  for (const text of ["### Manual regional execution", "Draft multiple routes", "Save memory & storage", "Activate reviewed routes",
    "Saved preferences do not arm a lease", "separately chosen global RAM policy", "not native CUDA compute",
    "Automatic placement uses the separate control below", "Retained device-cache/context migration",
    "concurrent\ndevice execution", "thirteen individually registered operation kinds",
    "Mutable\nmemory and learning state stays with its actual owner"])
    expect(guide).toContain(text);
  for (const text of ["Pack version 22", "window.omni.regionalExecution", "activate({ brainId, reviewId })", "deactivate({ brainId, epoch })",
    "opaque current review ID", "never exported, copied", "packed-cpu-tile-fallback", "nativeComputeSupported:false",
    "no regional\ntraining", "native-mixed-sequential", "actual Worker request",
    "training do not advertise this mixed mode", "not privately replay or relocate"])
    expect(integrations).toContain(text);
});
