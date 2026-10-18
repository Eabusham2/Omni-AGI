import { createHash } from "node:crypto";
import { lstat, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { PluginCatalog, canonicalPluginPackage } from "../src/main/pluginCatalog";
import { PluginService } from "../src/main/pluginService";
import { PLUGIN_TRAINING_PARSER_CAPABILITY, type OmniPluginPackage } from "../src/shared/plugins";
import { DEFAULT_CONFIG } from "../src/shared/types";
import { parserModule } from "./fixtures/pluginWasmParser";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });

function legacyPackage(): OmniPluginPackage {
  return { format: "omni-plugin-package", formatVersion: 1, manifest: {
    id: "fixture.legacy", version: "1.0.0", name: "Legacy package", description: "No executable module field.",
    author: "Fixture", license: "MIT", documentation: "Legacy declaration.", capabilities: [], tools: [], panels: [],
    themes: [], trainingFormats: [{ id: "lines", title: "Lines", description: "Inert lines.", extensions: ["lines"],
      mediaTypes: [], parser: "declaration-only" }]
  } };
}

function executablePackage(): OmniPluginPackage {
  return { format: "omni-plugin-package", formatVersion: 1, manifest: {
    id: "fixture.wasm-parser", version: "1.0.0", name: "WASM lines", description: "Pure streaming record parser.",
    author: "Fixture", license: "MIT", documentation: "Transforms selected .wlines bytes into JSONL records.",
    capabilities: [{ toolId: PLUGIN_TRAINING_PARSER_CAPABILITY, actions: ["lines"],
      reason: "Execute this immutable pure parser for selected learning sources." }],
    tools: [], panels: [], themes: [], trainingModules: [parserModule()],
    trainingFormats: [{ id: "lines", title: "WASM lines", description: "Streams records through the reviewed module.",
      extensions: ["wlines"], mediaTypes: [], parser: { kind: "wasm-stream", moduleId: "lines", output: "jsonl" } }]
  } };
}

describe("executable plugin training-format integration", () => {
  it("keeps packages that omit trainingModules in their prior canonical shape", () => {
    const prepared = canonicalPluginPackage(legacyPackage());
    expect(prepared.json).not.toContain("trainingModules");
    expect(prepared.sha256).toBe("8dcbb607192496df80f6463fa7ca16f8d4edc356df5d794b7e04d7b028aae21b");
  });

  it("requires explicit parser capability and feeds only verified transformed JSONL to native ingestion", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-plugin-wasm-ingest-")); roots.push(root);
    const repository = new BrainRepository(join(root, "brains")); await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Executable parser fixture", idleCognition: false });
    const requests: Array<{ params: Record<string, unknown>; bytes: Buffer }> = [];
    const engine = { request: vi.fn(async (method: string, params: Record<string, unknown>) => {
      if (method !== "ingest") return {};
      const bytes = await readFile(String(params.path)); requests.push({ params, bytes });
      return { transactionKey: params.transactionKey,
        source: { kind: "json", learned_ideas: 1, learned_concepts: 1, synaptic_update_events: 1,
          parameter_update_steps: 1, parameter_checksum_changed: true },
        coverage: { discoveredFiles: 1, completedFiles: 1, rejectedFiles: 0,
          discoveredRecords: 2, processedRecords: 2, rejectedRecords: 0, processedBytes: bytes.length,
          shards: 0, modalityCounts: { jsonl: 2 }, errors: [], complete: true } };
    }) } as unknown as EngineSupervisor;
    const service = new BrainService(repository, engine), catalog = new PluginCatalog(join(root, "plugins"));
    const plugins = new PluginService(catalog, service); await plugins.initialize();
    const installed = await plugins.installPackage(executablePackage());
    let entry = (await plugins.list()).find((item) => item.pluginId === installed.package.manifest.id)!;
    await plugins.setEnabled(entry.pluginId, true, entry.activeContentSha256, entry.revision);
    await plugins.setInstanceAccess({ brainId: brain.id, pluginId: entry.pluginId,
      contentSha256: installed.contentSha256, enabled: true,
      capabilities: [{ toolId: PLUGIN_TRAINING_PARSER_CAPABILITY, actions: ["lines"], enabled: false }],
      toolPermissions: [], requestId: "parser-capability-off" });
    const source = join(root, "records.wlines"), body = Buffer.from('{"x":1}\n{"x":2}\n'); await writeFile(source, body);
    await expect(plugins.resolveTrainingFormat(brain.id, source)).rejects.toThrow(/explicitly enabled parser capability/i);
    await plugins.setInstanceAccess({ brainId: brain.id, pluginId: entry.pluginId,
      contentSha256: installed.contentSha256, enabled: true,
      capabilities: [{ toolId: PLUGIN_TRAINING_PARSER_CAPABILITY, actions: ["lines"], enabled: true }],
      toolPermissions: [], requestId: "parser-capability-on" });
    service.setPluginTrainingFormatResolver(plugins);
    const manifest = await service.previewDataset(brain.id, [source]);
    const entries = []; for await (const item of service.datasets.entries(brain.id, manifest.id)) entries.push(item);
    const binding = entries[0]!.pluginFormat!;
    expect(binding).toMatchObject({ parserFormat: "jsonl", executable: { kind: "wasm-stream", moduleId: "lines",
      moduleSha256: installed.package.manifest.trainingModules![0]!.wasmSha256 } });
    const result = await service.ingestManifest(brain.id, manifest.id);
    expect(result).toMatchObject({ paused: false, coverage: { complete: true } });
    expect(requests).toHaveLength(1); expect(requests[0]!.bytes).toEqual(body);
    expect(requests[0]!.params).toMatchObject({ kind: "jsonl",
      contentHash: createHash("sha256").update(body).digest("hex") });
    expect(result.results[0]?.source).toMatchObject({ contentHash: createHash("sha256").update(body).digest("hex"),
      parserExecution: { pluginId: entry.pluginId, packageSha256: installed.contentSha256,
        outputRecords: 2, complete: true } });
    await expect(repository.trainingSourceById(brain.id, result.results[0]!.source.id)).resolves.toMatchObject({
      parserExecution: { moduleSha256: installed.package.manifest.trainingModules![0]!.wasmSha256,
        sourceSha256: createHash("sha256").update(body).digest("hex"), outputRecords: 2 }
    });
    expect(await plugins.portable(brain.id)).toEqual([expect.objectContaining({ contentSha256: installed.contentSha256,
      authority: "source-review", reviewedAccess: expect.objectContaining({ capabilities: [expect.objectContaining({
        toolId: PLUGIN_TRAINING_PARSER_CAPABILITY, enabled: true
      })] }) })]);
    await plugins.adoptFullMergePortable("merged-brain", [brain.id]);
    await expect(plugins.resolveTrainingFormat("merged-brain", source)).resolves.toBeUndefined();
    expect(await plugins.portable("merged-brain")).toEqual([expect.objectContaining({
      contentSha256: installed.contentSha256, authority: "merge-parent-dormant"
    })]);
    await expect(lstat(String(requests[0]!.params.path))).rejects.toMatchObject({ code: "ENOENT" });
  });
});
