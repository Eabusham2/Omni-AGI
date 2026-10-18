import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import {
  PLUGIN_TRAINING_PARSER_CAPABILITY,
  PLUGIN_WASM_PARSER_ABI,
  type OmniPluginPackage,
  type PluginCatalogEntry,
  type PluginInstanceAccess,
  type PluginsApi
} from "../src/shared/plugins";
import {
  DatasetParserPicker,
  datasetParserCandidateKey,
  datasetParserSelectionForKey,
  listDatasetParserCandidates,
  type DatasetParserPickerModel
} from "../src/renderer/src/DatasetParserPicker";

const HASH = "b".repeat(64);
const entry: PluginCatalogEntry = {
  pluginId: "example.parsers",
  name: "Example parsers",
  description: "Fixture",
  enabled: true,
  activeVersion: "1.0.0",
  activeContentSha256: HASH,
  revision: 4,
  updatedAt: "2026-10-13T00:00:00.000Z",
  versions: [{ version: "1.0.0", contentSha256: HASH, installedAt: "2026-10-13T00:00:00.000Z", sourceLabel: "fixture" }]
};
const access: PluginInstanceAccess = {
  brainId: "brain-a",
  pluginId: entry.pluginId,
  contentSha256: HASH,
  enabled: true,
  capabilities: [{ toolId: PLUGIN_TRAINING_PARSER_CAPABILITY, actions: ["wasm-lines"], enabled: true }],
  toolPermissions: [],
  revision: 8,
  updatedAt: "2026-10-13T00:00:00.000Z"
};
const packageValue: OmniPluginPackage = {
  format: "omni-plugin-package",
  formatVersion: 1,
  manifest: {
    id: entry.pluginId,
    version: "1.0.0",
    name: entry.name,
    description: entry.description,
    author: "Fixture",
    license: "MIT",
    documentation: "Fixture",
    capabilities: [], tools: [], panels: [], interactivePanels: [], themes: [],
    trainingModules: [{ id: "lines", kind: "wasm-stream", abi: PLUGIN_WASM_PARSER_ABI,
      wasmBase64: "AGFzbQEAAAA=", wasmSha256: "c".repeat(64) }],
    trainingFormats: [
      { id: "inert", title: "Inert", description: "Declaration only", extensions: ["inert"], mediaTypes: [], parser: "declaration-only" },
      { id: "records", title: "Record stream", description: "Built-in records", extensions: ["records"], mediaTypes: ["application/x-records"],
        parser: { kind: "builtin-stream", format: "jsonl" } },
      { id: "wasm-lines", title: "WASM lines", description: "Reviewed pure transformer", extensions: ["lines"], mediaTypes: ["text/x-lines"],
        parser: { kind: "wasm-stream", moduleId: "lines", output: "jsonl" } },
      { id: "wasm-not-permitted", title: "Not permitted", description: "Missing action", extensions: ["blocked"], mediaTypes: [],
        parser: { kind: "wasm-stream", moduleId: "lines", output: "jsonl" } }
    ]
  }
};

describe("explicit dataset parser picker", () => {
  it("identifies Runtime JS only for the enabled exact parser action and declared module", async () => {
    const runtimePackage: OmniPluginPackage = { ...packageValue, manifest: { ...packageValue.manifest,
      runtimeModules: [{ id: "runtime-parser", title: "Runtime parser", description: "Owned fixture", script: "", events: [], ui: { kinds: [], slots: [] }, dynamicTools: false }],
      trainingFormats: [{ id: "runtime-records", title: "Runtime records", description: "Runtime fixture", extensions: ["runtime-records"], mediaTypes: [],
        parser: { kind: "runtime-stream", moduleId: "runtime-parser", output: "jsonl" } }] } };
    let current = { ...access, capabilities: [{ toolId: PLUGIN_TRAINING_PARSER_CAPABILITY, actions: ["runtime-records"], enabled: true }] };
    const api = { list: async () => [entry], instance: async () => current, package: async () => runtimePackage };
    const candidates = await listDatasetParserCandidates(api, "brain-a"); expect(candidates).toHaveLength(1); expect(candidates[0]!.parserKind).toBe("runtime");
    const html = renderToStaticMarkup(<DatasetParserPicker model={{ candidates, loading: false, status: "Fixture", select: vi.fn() }} />);
    expect(html).toContain("Runtime JS");
    current = { ...current, capabilities: [] }; expect(await listDatasetParserCandidates(api, "brain-a")).toEqual([]);
    current = { ...current, capabilities: [{ toolId: PLUGIN_TRAINING_PARSER_CAPABILITY, actions: ["other-format"], enabled: true }] };
    expect(await listDatasetParserCandidates(api, "brain-a")).toEqual([]);
    current = { ...current, capabilities: [{ toolId: PLUGIN_TRAINING_PARSER_CAPABILITY, actions: ["runtime-records"], enabled: true }] };
    runtimePackage.manifest.runtimeModules = []; expect(await listDatasetParserCandidates(api, "brain-a")).toEqual([]);
  });

  it("lists only enabled exact-instance parser formats and their permitted WASM actions", async () => {
    const api = {
      list: vi.fn(async () => [entry, { ...entry, pluginId: "disabled.parser", enabled: false }]),
      instance: vi.fn(async () => access),
      package: vi.fn(async () => packageValue)
    } as unknown as Pick<PluginsApi, "list" | "instance" | "package">;
    const candidates = await listDatasetParserCandidates(api, "brain-a");
    expect(candidates.map(candidate => candidate.selection.formatId)).toEqual(["records", "wasm-lines"]);
    expect(candidates[0]).toMatchObject({ extensions: ["records"], mediaTypes: ["application/x-records"], parserKind: "built-in",
      selection: { pluginId: entry.pluginId, contentSha256: HASH, formatId: "records" } });
    expect(candidates[1]?.parserKind).toBe("wasm");
  });

  it("keeps Automatic empty and accepts only an exact displayed tuple", () => {
    const candidate = { selection: { pluginId: entry.pluginId, contentSha256: HASH, formatId: "records" }, pluginName: entry.name,
      formatTitle: "Record stream", description: "Built-in records", extensions: ["records"], mediaTypes: [], parserKind: "built-in" as const };
    expect(datasetParserSelectionForKey("", [candidate])).toBeUndefined();
    expect(datasetParserSelectionForKey(JSON.stringify([entry.pluginId, "d".repeat(64), "records"]), [candidate])).toBeUndefined();
    expect(datasetParserSelectionForKey(datasetParserCandidateKey(candidate.selection), [candidate])).toEqual(candidate.selection);
  });

  it("renders Automatic first and exposes exact selectors and immutable hash", () => {
    const selection = { pluginId: entry.pluginId, contentSha256: HASH, formatId: "records" };
    const model: DatasetParserPickerModel = {
      candidates: [{ selection, pluginName: entry.name, formatTitle: "Record stream", description: "Built-in records",
        extensions: ["records"], mediaTypes: ["application/x-records"], parserKind: "built-in" }],
      loading: false,
      status: "Automatic uses built-in detection or one uniquely matched reviewed file format.",
      select: vi.fn()
    };
    const html = renderToStaticMarkup(<DatasetParserPicker model={model} />);
    expect(html.indexOf("Automatic · built-in or reviewed file format")).toBeLessThan(html.indexOf("Example parsers"));
    expect(html).toContain("*.records");
    expect(html).toContain("application/x-records");
    expect(html).toContain(`${HASH.slice(0, 12)}…`);
  });
});
