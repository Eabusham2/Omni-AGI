/** Known local template bodies only; no renderer process, worker, host effects or AI. */
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { canonicalPluginPackage } from "../src/main/pluginCatalog";
import { compileRuntimeActionInput, validateRuntimeContribution, validateRuntimeTool } from "../src/main/pluginRuntimeValidation";
import type { PluginRuntimeContribution, PluginRuntimeModuleDefinition, PluginRuntimeToolDefinition, PluginRuntimeParserStep, PluginRuntimeParserStepResult } from "../src/shared/pluginRuntime";
import { PluginModuleAuthoring } from "../src/renderer/src/PluginModuleAuthoring";
import { PLUGIN_MODULE_STARTERS, addPluginModuleStarter, newPluginAuthoringPackage, pluginStyleModuleCss,
  readPluginAuthoringPackage, removePluginRuntimeModule, replacePluginRuntimeModule, replacePluginStyleModuleCss } from "../src/renderer/src/pluginModuleDraft";

function moduleFixture(kind: typeof PLUGIN_MODULE_STARTERS[number]["id"]) {
  return addPluginModuleStarter(newPluginAuthoringPackage(), kind).packageValue.manifest.runtimeModules![0]!;
}

describe("feature module authoring", () => {
  it("creates valid real feature, CSS, typed-tool and streaming-parser packages without changing the input", () => {
    const original = newPluginAuthoringPackage(), before = JSON.stringify(original);
    let combined = original;
    for (const starter of PLUGIN_MODULE_STARTERS) {
      const next = addPluginModuleStarter(combined, starter.id);
      expect(() => canonicalPluginPackage(next.packageValue)).not.toThrow();
      combined = next.packageValue;
    }
    expect(JSON.stringify(original)).toBe(before);
    expect(combined.manifest.runtimeModules).toHaveLength(4);
    expect(combined.manifest.trainingFormats[0]).toMatchObject({ extensions: ["omnitext"], parser: { kind: "runtime-stream", moduleId: "parser", output: "jsonl" } });
    expect(combined.manifest.capabilities).toEqual([{ toolId: "data.training-parser", actions: ["parser"], reason: expect.any(String) }]);
    const duplicate = addPluginModuleStarter(combined, "parser");
    expect(duplicate.moduleId).toBe("parser-2"); expect(() => canonicalPluginPackage(duplicate.packageValue)).not.toThrow();
  });

  it("keeps arbitrary existing package sections and immutable edit lineage when adding or editing code", () => {
    const base = { ...newPluginAuthoringPackage(), parentSha256: "a".repeat(64) };
    base.manifest.panels.push({ id: "kept", title: "Existing", html: "<p>Keep me.</p>" });
    const next = addPluginModuleStarter(base, "feature");
    const module = next.packageValue.manifest.runtimeModules![0]!;
    const edited = replacePluginRuntimeModule(next.packageValue, module.id, { ...module, script: module.script + "\n// Local edit" });
    expect(edited.parentSha256).toBe(base.parentSha256); expect(edited.manifest.panels).toEqual(base.manifest.panels);
    expect(next.packageValue.manifest.runtimeModules![0]!.script).not.toContain("Local edit");
    expect(() => replacePluginRuntimeModule(edited, module.id, { ...module, id: "other" })).toThrow("selected module changed");
  });

  it("edits recognized stylesheet data without executing or deleting custom module code", () => {
    const module = moduleFixture("style"), script = module.script + "\n// Keep my extra callback";
    expect(pluginStyleModuleCss(script)).toContain("@media");
    const changed = replacePluginStyleModuleCss(script, ".chat { gap: 22px; }");
    expect(pluginStyleModuleCss(changed)).toBe(".chat { gap: 22px; }"); expect(changed).toContain("Keep my extra callback");
    expect(pluginStyleModuleCss("globalThis.doNotRun()" )).toBeUndefined();
    expect(() => replacePluginStyleModuleCss("arbitrary()", "body{}" )).toThrow("style envelope");
  });

  it("removes only the draft module and its parser declarations, keeping other format permissions", () => {
    const first = addPluginModuleStarter(newPluginAuthoringPackage(), "parser");
    const second = addPluginModuleStarter(first.packageValue, "parser");
    const kept = addPluginModuleStarter(second.packageValue, "feature");
    const removed = removePluginRuntimeModule(kept.packageValue, first.moduleId);
    expect(removed.manifest.runtimeModules?.map(module => module.id)).toEqual(["parser-2", "feature"]);
    expect(removed.manifest.trainingFormats.map(format => format.id)).toEqual(["parser-2"]);
    expect(removed.manifest.capabilities[0]!.actions).toEqual(["parser-2"]);
    expect(() => canonicalPluginPackage(removed)).not.toThrow();
  });

  it("presents executable source, CSS and package JSON editing without running any code or requesting access", () => {
    const feature = addPluginModuleStarter(newPluginAuthoringPackage(), "feature").packageValue;
    const html = renderToStaticMarkup(<PluginModuleAuthoring value={JSON.stringify(feature)} onChange={() => { throw new Error("render is read-only"); }} />);
    expect(html).toContain("Feature JavaScript"); expect(html).toContain("Complete package JSON");
    expect(html).toContain("+ Training format"); expect(html).toContain("explicitly activate");
    expect(html).not.toContain("JSON only"); expect(html).not.toContain('aria-label="Edit module"');
    const style = addPluginModuleStarter(newPluginAuthoringPackage(), "style").packageValue;
    expect(renderToStaticMarkup(<PluginModuleAuthoring value={JSON.stringify(style)} onChange={() => undefined} />)).toContain("Feature stylesheet");
    expect(() => readPluginAuthoringPackage("arbitrary()" )).toThrow();
    expect(() => readPluginAuthoringPackage(JSON.stringify({ ...feature, manifest: { ...feature.manifest, trainingFormats: [null] } }))).toThrow("Training formats");
  });

  it("feature callbacks use supported declared placements and publish inert button/menu/panel payloads", async () => {
    const module = moduleFixture("feature"), contributions: PluginRuntimeContribution[] = [], handlers = new Map<string, () => unknown>();
    let activate: (() => unknown) | undefined;
    const runtime = { onActivate: (fn: () => unknown) => { activate = fn; }, on: (event: string, fn: () => unknown) => handlers.set(event, fn),
      ui: { put: async (value: PluginRuntimeContribution) => { contributions.push(validateRuntimeContribution(value, module)); } } };
    new Function("globalThis", module.script)({ omniRuntime: runtime });
    expect(contributions).toHaveLength(0); await activate!();
    expect(contributions.map(value => value.kind)).toEqual(["button", "menu"]);
    await handlers.get("ui:open")!(); expect(contributions.at(-1)).toMatchObject({ kind: "panel", slot: "tools" });
    await handlers.get("chat.action")!(); expect(contributions.at(-1)?.html).toContain("chat action event");
  });

  it("the tool starter registers a real typed local handler only on activation", async () => {
    const module = moduleFixture("tool"); let activate: (() => unknown) | undefined;
    let registered: { definition: PluginRuntimeToolDefinition; handlers: Record<string, (value: { text: string }) => unknown> } | undefined;
    new Function("globalThis", module.script)({ omniRuntime: { onActivate: (fn: () => unknown) => { activate = fn; },
      registerTool: async (definition: PluginRuntimeToolDefinition, handlers: Record<string, (value: { text: string }) => unknown>) => {
        registered = { definition: validateRuntimeTool(definition), handlers };
      } } });
    expect(registered).toBeUndefined(); await activate!();
    const validate = compileRuntimeActionInput(registered!.definition.actions[0]!.inputSchema);
    expect(validate({ text: "A😀" })).toBe(true); expect(validate({ text: 3 })).toBe(false);
    expect(registered!.handlers.measure!({ text: "A😀" })).toEqual({ characters: 2 });
  });

  it("the parser starter streams split UTF-8 into bounded complete JSONL without a model or whole-file buffer", async () => {
    const module: PluginRuntimeModuleDefinition = moduleFixture("parser");
    let parser: { push(step: PluginRuntimeParserStep): PluginRuntimeParserStepResult; drain(step: PluginRuntimeParserStep): PluginRuntimeParserStepResult } | undefined;
    new Function("globalThis", module.script)({ omniRuntime: { context: { purpose: "training-parser" }, registerParser: (value: typeof parser) => { parser = value; } } });
    const input = Buffer.from("A😀B\nsecond line", "utf8"), pieces = [input.subarray(0, 3), input.subarray(3, 5), input.subarray(5)];
    const output: Buffer[] = []; let sequence = 0;
    for (const bytes of [...pieces, Buffer.alloc(0)]) {
      const step: PluginRuntimeParserStep = { id: `push-${sequence}`, operation: "push", sequence: sequence++, maxOutputBytes: 3,
        inputBase64: bytes.toString("base64"), final: bytes.length === 0 };
      let result = parser!.push(step);
      while (result.state === "output") {
        const part = Buffer.from(result.outputBase64, "base64"); expect(part.length).toBeLessThanOrEqual(3); output.push(part);
        result = parser!.drain({ ...step, operation: "drain" });
      }
      expect(result.state).toBe(step.final ? "done" : "need-input");
    }
    const records = Buffer.concat(output).toString("utf8").trimEnd().split("\n").map(line => JSON.parse(line) as { text: string });
    expect(records.map(record => record.text).join("")).toBe(input.toString("utf8"));
  });
});
