/** Pack v24 declaration/equality review only. No plugin, tool, brain, learner, or provider executes. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, it } from "vitest";
import { declaredDocumentationCurriculumPack } from "../src/main/documentationCurriculumPack";

const sha256 = (value: string): string => createHash("sha256").update(value, "utf8").digest("hex");
const normalized = (value: string): string => `${value.replace(/\r\n?/g, "\n").trim()}\n`;
const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object"
  ? Object.fromEntries(Object.entries(value).sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([key, child]) => [key, canonical(child)])) : value;

it("declares exact pack24 public sparse, native-candidate and runtime-parser manuals without executed outcomes", () => {
  const pack = declaredDocumentationCurriculumPack();
  expect(pack).toMatchObject({ format: "omni-ordinary-document-curriculum", version: 24,
    id: "omni-v2-user-documentation-24", ordinaryNeuralLearning: true, hiddenPrompt: false,
    behavioralPersona: false, preferenceLabels: false, rewardModel: false, executedHostActions: false });
  expect(pack.records.map(record => record.id)).toEqual([
    "user-guide", "integrations", "plugins", "html-artifacts", "executable-training-parsers",
    "managed-environments", "clipboard", "api-teacher-model-discovery"
  ]);
  const { records, sha256: packSha256, ...declaration } = pack;
  expect(packSha256).toBe(sha256(JSON.stringify(canonical({ ...declaration,
    records: records.map(({ content: _content, ...record }) => record) }))));
  for (const record of records) {
    const source = normalized(readFileSync(resolve(record.path), "utf8"));
    expect(record.content).toBe(source); expect(record.contentSha256).toBe(sha256(source));
    expect(record.bytes).toBe(Buffer.byteLength(source));
    expect(record).not.toHaveProperty("toolId"); expect(record).not.toHaveProperty("action");
    expect(record).not.toHaveProperty("successfulOutcome");
  }

  const guide = records.find(record => record.id === "user-guide")!.content;
  const integrations = records.find(record => record.id === "integrations")!.content;
  const sdk = records.find(record => record.id === "plugins")!.content;
  const parsers = records.find(record => record.id === "executable-training-parsers")!.content;
  for (const text of ["### Native sparse connections", "### Native improvement candidates", "native-status",
    "runtime-stream", "waits visibly for\nApprove or Deny", "zero-update result"]) expect(guide).toContain(text);
  for (const text of ["Pack version 24", "window.omni.substrate", "stop(brainId, jobId)",
    "source.self-modify.native-status", "runtime-stream", "complete byte\ncoverage is not a claim"])
    expect(integrations).toContain(text);
  for (const text of ["runtime-stream", "globalThis.omniRuntime.registerParser", "media-assets",
    "data.training-parser", "cannot approve itself", "zero-asset result", "no\nmodality update"]) expect(sdk).toContain(text);
  for (const text of ["omni-wasm-record-stream-1", "omniRuntime.registerParser", "media-assets",
    "waiting-for-approval", "Host call identity", "does not prove later neural learning"]) expect(parsers).toContain(text);
  expect(records.map(record => String(record.path))).not.toEqual(expect.arrayContaining([
    "docs/v2/runtime-training-parsers.md", "docs/v2/native-sparse-transfer.md", "docs/v2/native-candidate-tools.md",
    "docs/v2/sparse-jobs-ui.md", "docs/v2/FEATURE_MATRIX.md", "docs/v2/CONTINUATION.md"
  ]));
});
