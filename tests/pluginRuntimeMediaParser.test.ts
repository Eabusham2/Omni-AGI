import { createHash } from "node:crypto";
import { lstat, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { PluginRuntimeStreamParser, type PluginRuntimeTrainingParserHost } from "../src/main/pluginRuntimeStreamParser";
import { canonicalPluginPackage } from "../src/main/pluginCatalog";
import { validatePluginRuntimeParserStepResult } from "../src/main/pluginRuntimeService";
import { runtimeModuleSha256 } from "../src/main/pluginRuntimeValidation";
import { extractStreamingZip } from "../src/main/streamingZip";
import { OMNI_PLUGIN_FORMAT, OMNI_PLUGIN_FORMAT_VERSION, PLUGIN_RUNTIME_PARSER_ABI, PLUGIN_TRAINING_PARSER_CAPABILITY,
  type OmniPluginPackage, type PluginRuntimeMediaOutputDeclaration } from "../src/shared/plugins";
import type { PluginRuntimeEvent, PluginRuntimeModuleDefinition, PluginRuntimeParserStep,
  PluginRuntimeParserStepResult, PluginRuntimePendingApproval, PluginRuntimeSession,
  PluginRuntimeTrainingParserActivationRequest } from "../src/shared/pluginRuntime";
import type { PluginTrainingFormatBinding } from "../src/shared/types";

// Each workspace owns its extraction children; never race two recursive removers over the same tree.
const roots: string[] = [];
const sha = (value: Buffer | string): string => createHash("sha256").update(value).digest("hex");
afterEach(async () => {
  await Promise.all(roots.splice(0).map(async root => {
    await rm(root, { recursive: true, force: true });
    await expect(lstat(root)).rejects.toMatchObject({ code: "ENOENT" });
  }));
});

const output: PluginRuntimeMediaOutputDeclaration = { kind: "media-assets", mediaTypes: ["image/png", "audio/wav", "video/mp4"] };
const moduleDefinition = (): PluginRuntimeModuleDefinition => ({ id: "media", title: "Media", description: "Typed assets.",
  script: "omniRuntime.registerParser({push:()=>({state:'need-input'}),drain:()=>({state:'need-input'})});",
  events: [], ui: { kinds: [], slots: [] }, dynamicTools: false });
function packageFixture(): OmniPluginPackage {
  const module = moduleDefinition();
  return { format: OMNI_PLUGIN_FORMAT, formatVersion: OMNI_PLUGIN_FORMAT_VERSION, manifest: { id: "fixture.media", version: "1.0.0",
    name: "Media", description: "Typed native media.", author: "Fixture", license: "MIT", documentation: "Fixture",
    capabilities: [{ toolId: PLUGIN_TRAINING_PARSER_CAPABILITY, actions: ["media"], reason: "Parse media." }],
    tools: [], panels: [], interactivePanels: [], runtimeModules: [module], themes: [], trainingFormats: [{ id: "media", title: "Media",
      description: "Emit native assets.", extensions: ["mediafixture"], mediaTypes: ["application/x-media-fixture"],
      parser: { kind: "runtime-stream", moduleId: module.id, output: structuredClone(output) } }] } };
}

class MediaHost implements PluginRuntimeTrainingParserHost {
  activation?: PluginRuntimeTrainingParserActivationRequest;
  session?: PluginRuntimeSession;
  stopped = 0;
  empty = false;
  private queue: PluginRuntimeParserStepResult[] = [];
  async activateTrainingParser(request: PluginRuntimeTrainingParserActivationRequest) {
    this.activation = structuredClone(request);
    this.session = { format: "omni-plugin-runtime-session", version: 1, brainId: request.brainId, pluginId: request.pluginId,
      contentSha256: request.contentSha256, moduleId: request.moduleId, sessionId: "11111111-1111-4111-8111-111111111111",
      catalogRevision: request.expectedCatalogRevision, instanceRevision: request.expectedInstanceRevision, state: "active",
      startedAt: "2026-10-15T00:00:00.000Z", purpose: "training-parser",
      parser: { formatId: request.formatId, transactionKey: request.transactionKey, sourceSha256: request.sourceSha256,
        sourceBytes: request.sourceBytes, moduleSha256: request.moduleSha256, output: request.output } };
    return { session: this.session, maxMessageBytes: 64 * 1024 };
  }
  async parserStep(_brainId: string, _sessionId: string, step: PluginRuntimeParserStep): Promise<PluginRuntimeParserStepResult> {
    if (step.operation === "drain") return this.queue.shift() ?? { state: "need-input" };
    if (step.final) return { state: "done" };
    if (!this.empty && !this.queue.length) {
      const png = Buffer.from("89504e470d0a1a0a0000000d494844520000000100000001", "hex");
      const wav = Buffer.concat([Buffer.from("RIFF"), Buffer.alloc(4), Buffer.from("WAVEfmt "), Buffer.alloc(24)]);
      const mp4 = Buffer.concat([Buffer.from([0, 0, 0, 20]), Buffer.from("ftypisom"), Buffer.alloc(12)]);
      this.queue.push(
        { state: "asset-chunk", assetId: "picture", offset: 0, bytesBase64: png.toString("base64") },
        { state: "asset-end", assetId: "picture", bytes: png.length, sha256: sha(png) },
        { state: "asset-start", assetId: "sound", kind: "audio", mimeType: "audio/wav" },
        { state: "asset-chunk", assetId: "sound", offset: 0, bytesBase64: wav.toString("base64") },
        { state: "asset-end", assetId: "sound", bytes: wav.length, sha256: sha(wav) },
        { state: "asset-start", assetId: "motion", kind: "video", mimeType: "video/mp4" },
        { state: "asset-chunk", assetId: "motion", offset: 0, bytesBase64: mp4.toString("base64") },
        { state: "asset-end", assetId: "motion", bytes: mp4.length, sha256: sha(mp4) },
        { state: "need-input" }
      );
      return { state: "asset-start", assetId: "picture", kind: "image", mimeType: "image/png" };
    }
    return { state: "need-input" };
  }
  approvals(): Promise<PluginRuntimePendingApproval[]> { return Promise.resolve([]); }
  onEvent(_listener: (event: PluginRuntimeEvent) => void): () => void { return () => undefined; }
  async stop(): Promise<PluginRuntimeSession> { this.stopped += 1; return { ...this.session!, state: "stopped" }; }
  admitParserRead(): Promise<{ release(): void }> { return Promise.resolve({ release: () => undefined }); }
}

async function fixture(host: MediaHost) {
  const root = await mkdtemp(join(tmpdir(), "omni-runtime-media-parser-")); roots.push(root);
  const source = join(root, "source.mediafixture"), bytes = Buffer.from("complete source bytes"); await writeFile(source, bytes);
  const module = moduleDefinition(), moduleSha256 = runtimeModuleSha256(module);
  const binding: PluginTrainingFormatBinding = { pluginId: "fixture.media", contentSha256: "a".repeat(64), formatId: "media",
    parserFormat: "archive", catalogRevision: 2, instanceRevision: 3,
    executable: { kind: "runtime-stream", moduleId: module.id, moduleSha256, abi: PLUGIN_RUNTIME_PARSER_ABI,
      outputFormat: "media-assets", outputMediaTypes: [...output.mediaTypes] } };
  return { root, source, bytes, module, binding, parser: new PluginRuntimeStreamParser(host), request: { sourcePath: source,
    sourceSha256: sha(bytes), sourceBytes: bytes.length, transactionKey: "b".repeat(64), workspaceRoot: root } };
}

describe("runtime-stream typed media parser", () => {
  it("normalizes only an exact reviewed native-media declaration and validates typed result pages", () => {
    expect(canonicalPluginPackage(packageFixture()).package.manifest.trainingFormats[0]?.parser).toEqual({
      kind: "runtime-stream", moduleId: "media", output
    });
    const invalid = packageFixture();
    (invalid.manifest.trainingFormats[0]!.parser as { output: { mediaTypes: string[] } }).output.mediaTypes = ["application/json"];
    expect(() => canonicalPluginPackage(invalid)).toThrow(/unsupported native media MIME/i);
    expect(validatePluginRuntimeParserStepResult({ state: "asset-start", assetId: "picture", kind: "image", mimeType: "image/png" }, 4096, output))
      .toEqual({ state: "asset-start", assetId: "picture", kind: "image", mimeType: "image/png" });
    expect(() => validatePluginRuntimeParserStepResult({ state: "asset-start", assetId: "picture", kind: "image", mimeType: "image/jpeg" }, 4096,
      { kind: "media-assets", mediaTypes: ["image/png"] })).toThrow(/outside its reviewed/i);
    expect(() => validatePluginRuntimeParserStepResult({ state: "asset-chunk", assetId: "picture", offset: 0, bytesBase64: "AA==" }, 4096))
      .toThrow(/unsupported fields|JSONL/i);
  });

  it("hands an ordered media-only archive to native ingestion with exact compact provenance", async () => {
    const host = new MediaHost(), value = await fixture(host);
    const prepared = await value.parser.prepare({ brainId: "brain-a", binding: value.binding, module: value.module,
      moduleSha256: value.binding.executable!.moduleSha256, request: value.request });
    expect("path" in prepared).toBe(true);
    if (!("path" in prepared)) throw new Error("Expected a prepared media archive.");
    expect(prepared.parserFormat).toBe("archive");
    expect(prepared.provenance).toMatchObject({ outputKind: "media-assets", outputAssets: 3, outputRecords: 3,
      outputModalityCounts: { image: 1, audio: 1, video: 1 }, sourceSha256: sha(value.bytes) });
    expect(host.activation?.output).toEqual(output);
    const archive = await extractStreamingZip(prepared.path, value.root, { checkDisk: async () => undefined });
    expect(dirname(archive.root)).toBe(value.root);
    expect([...archive.entries.keys()]).toEqual(["media/00000000.png", "media/00000001.wav", "media/00000002.mp4"]);
    expect([...archive.entries.keys()].some(name => /\.jsonl?$/.test(name))).toBe(false);
    expect(await readFile(archive.entries.get("media/00000000.png")!.path)).toEqual(expect.any(Buffer));
    expect(host.stopped).toBe(1);
    const reused = await value.parser.prepare({ brainId: "brain-a", binding: value.binding, module: value.module,
      moduleSha256: value.binding.executable!.moduleSha256, request: value.request });
    expect("path" in reused && reused.path).toBe(prepared.path);
    expect(reused.provenance).toEqual(prepared.provenance);
    expect(host.stopped).toBe(1);
    await prepared.cleanup();
    expect(await readFile((reused as typeof prepared).path)).toEqual(await readFile(prepared.path));
    await reused.cleanup();
    await expect(lstat(prepared.path)).rejects.toMatchObject({ code: "ENOENT" });
    await expect(lstat(join(value.root, "plugin-runtime-parser", `${value.request.transactionKey}.receipt.json`)))
      .rejects.toMatchObject({ code: "ENOENT" });
    expect((await lstat(archive.root)).isDirectory()).toBe(true);
  });

  it("returns an explicit no-assets source without a native archive or learned-input surrogate", async () => {
    const host = new MediaHost(); host.empty = true; const value = await fixture(host);
    const prepared = await value.parser.prepare({ brainId: "brain-a", binding: value.binding, module: value.module,
      moduleSha256: value.binding.executable!.moduleSha256, request: value.request });
    expect(prepared).toMatchObject({ state: "no-assets", parserFormat: "archive",
      provenance: { outputKind: "media-assets", outputAssets: 0, outputRecords: 0, outputBytes: 0,
        outputModalityCounts: { image: 0, audio: 0, video: 0 } } });
    expect("path" in prepared).toBe(false);
    await prepared.cleanup();
  });
});
