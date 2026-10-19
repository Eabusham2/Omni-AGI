/** Public source/declaration check only. No package, renderer or neural learner runs. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, it } from "vitest";
import { declaredDocumentationCurriculumPack } from "../src/main/documentationCurriculumPack";
import { normalizePluginRuntimeModule } from "../src/main/pluginRuntimeValidation";

const sha = (text: string) => createHash("sha256").update(text, "utf8").digest("hex");
const normalize = (text: string) => `${text.replace(/\r\n?/g, "\n").trim()}\n`;
const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object"
  ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => [key, canonical(child)])) : value;

it("includes the current plugin runtime/theme/parser contracts as exact pack23 ordinary public sources, not executed outcomes", () => {
  const pack = declaredDocumentationCurriculumPack();
  expect(pack).toMatchObject({ version: 23, id: "omni-v2-user-documentation-23", ordinaryNeuralLearning: true,
    hiddenPrompt: false, behavioralPersona: false, preferenceLabels: false, rewardModel: false, executedHostActions: false });
  expect(pack.records.map(record => record.id)).toEqual(["user-guide", "integrations", "plugins", "html-artifacts", "executable-training-parsers", "managed-environments", "clipboard", "api-teacher-model-discovery"]);
  const { records, sha256, ...declaration } = pack;
  expect(sha256).toBe(sha(JSON.stringify(canonical({ ...declaration, records: records.map(({ content: _content, ...record }) => record) }))));
  for (const record of records) {
    const source = normalize(readFileSync(resolve(record.path), "utf8"));
    expect(record.content).toBe(source); expect(record.contentSha256).toBe(sha(source)); expect(record.bytes).toBe(Buffer.byteLength(source));
    expect(record).not.toHaveProperty("toolId"); expect(record).not.toHaveProperty("action"); expect(record).not.toHaveProperty("successfulOutcome");
  }
  const sdk = records.find(record => record.id === "plugins")!.content, guide = records.find(record => record.id === "user-guide")!.content,
    integrations = records.find(record => record.id === "integrations")!.content;
  for (const text of ["runtimeModules", "globalThis.omniRuntime", "onActivate", "onDeactivate", "registerTool", "state.read", "receipts.read", "extension.<sha256>",
    "Nested Off blocks", "Outer Full is not a bypass", "private thoughts", "dormant", "compare-and-set", "no credential or approval token", "hostile-package"])
    expect(sdk).toContain(text);
  const section = sdk.slice(sdk.indexOf("## Feature-runtime modules")), jsonExample = section.match(/```json\n([\s\S]*?)\n```/)?.[1];
  expect(jsonExample).toBeDefined();
  expect(normalizePluginRuntimeModule(JSON.parse(jsonExample!))).toMatchObject({ id: "text-helper", events: ["brain.lifecycle"], dynamicTools: false, ui: { kinds: ["panel"], slots: ["run.sidebar"] } });
  for (const text of ["Feature runtime", "Activate", "Clear plugin theme", "Parser for the next file or folder", "MIME-only", "known built-in format", "Use reviewed source", "trained natural tool use"])
    expect(guide).toContain(text);
  for (const text of ["Pack version 23", "window.omni.pluginRuntime", "readHistoryEvent", "readSourceStateBytes", "selectSourceState", "Outer Full cannot bypass nested Off/Ask", "no implicit brain Load", "not real packaged renderer"])
    expect(integrations).toContain(text);
  expect(sdk).not.toContain("arbitrary JavaScript/native transforms, runtime hooks, and direct main-process\naccess remain unsupported");
  expect(guide).not.toContain("Media-type selectors remain inspectable but are not yet routed");
  expect(records.some(record => /CONTINUATION|FEATURE_MATRIX|plugin-runtime-custody/.test(record.path))).toBe(false);
  console.info(JSON.stringify({ check: "plugin-public-documentation-declaration-only", packVersion: pack.version, packId: pack.id, packSha256: pack.sha256,
    publicRecords: pack.records.length, neuralLearningExecuted: false, pluginExecution: false }));
});
