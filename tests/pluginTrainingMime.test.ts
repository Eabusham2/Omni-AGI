import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { mkdtemp, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainActivityLedger } from "../src/main/brainActivityLedger";
import {
  BrainService,
  pluginParserBindingSha256
} from "../src/main/brainService";
import { assertPluginManifestSourceStable } from "../src/main/dataIngestion";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { PluginCatalog } from "../src/main/pluginCatalog";
import { PluginService } from "../src/main/pluginService";
import type { OmniPluginPackage } from "../src/shared/plugins";
import { DEFAULT_CONFIG, type DatasetManifestEntry } from "../src/shared/types";

const roots: string[] = [];
const servers: Server[] = [];
afterEach(async () => {
  await Promise.all(servers.splice(0).map(server => new Promise<void>(resolve => server.close(() => resolve()))));
  await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true })));
});

async function root(): Promise<string> {
  const value = await mkdtemp(join(tmpdir(), "omni-parser-mime-")); roots.push(value); return value;
}

function packageFixture(): OmniPluginPackage {
  return { format: "omni-plugin-package", formatVersion: 1, manifest: {
    id: "fixture.mime-parser", version: "1.0.0", name: "MIME parser", description: "Reviewed stream aliases.",
    author: "Fixture", license: "MIT", documentation: "Fixture", capabilities: [], tools: [], panels: [], themes: [],
    trainingFormats: [
      { id: "unique-extension", title: "Unique extension", description: "Reviewed extension route.", extensions: ["records"],
        mediaTypes: [], parser: { kind: "builtin-stream", format: "jsonl" } },
      { id: "mime-override", title: "MIME override", description: "Explicit MIME route.", extensions: [],
        mediaTypes: ["application/x-omni-records"], parser: { kind: "builtin-stream", format: "jsonl" } }
    ]
  } };
}

async function configured(rootPath: string, engine: EngineSupervisor) {
  const repository = new BrainRepository(join(rootPath, "brains")); await repository.initialize();
  const brain = await repository.create({ ...DEFAULT_CONFIG, name: "MIME fixture", idleCognition: false });
  const service = new BrainService(repository, engine);
  const plugins = new PluginService(new PluginCatalog(join(rootPath, "plugins")), service); await plugins.initialize();
  const installed = await plugins.installPackage(packageFixture());
  let entry = (await plugins.list()).find(item => item.pluginId === installed.package.manifest.id)!;
  entry = await plugins.setEnabled(entry.pluginId, true, entry.activeContentSha256, entry.revision);
  await plugins.setInstanceAccess({ brainId: brain.id, pluginId: entry.pluginId, contentSha256: installed.contentSha256,
    enabled: true, capabilities: [], toolPermissions: [], requestId: "mime-parser-access" });
  service.setPluginTrainingFormatResolver(plugins);
  return { repository, brain, service, plugins, installed, entry };
}

describe("plugin parser MIME and explicit-selection boundary", () => {
  it("preserves unique reviewed extension auto-routing but requires a tuple for MIME-only and known-format override", async () => {
    const rootPath = await root();
    const engine = { request: vi.fn(async () => ({})) } as unknown as EngineSupervisor;
    const { brain, plugins, installed } = await configured(rootPath, engine);
    await expect(plugins.resolveTrainingFormat(brain.id, "/source/item.records", "text/plain"))
      .resolves.toMatchObject({ formatId: "unique-extension", parserFormat: "jsonl", inputMediaType: "text/plain" });
    await expect(plugins.resolveTrainingFormat(brain.id, "/source/item.bin", "application/x-omni-records"))
      .resolves.toBeUndefined();
    const selection = { pluginId: installed.package.manifest.id, contentSha256: installed.contentSha256, formatId: "mime-override" };
    const binding = await plugins.resolveTrainingFormat(brain.id, "/source/item.json", "application/x-omni-records", selection);
    expect(binding).toMatchObject({ formatId: "mime-override", inputMediaType: "application/x-omni-records" });
    await expect(plugins.validateTrainingFormat(brain.id, binding!, { path: "/source/item.json", mediaType: "text/plain" }))
      .rejects.toThrow(/extension\/MIME differs/i);
    await plugins.setInstanceAccess({ brainId: brain.id, pluginId: installed.package.manifest.id,
      contentSha256: installed.contentSha256, enabled: false, capabilities: [], toolPermissions: [], requestId: "mime-parser-off" });
    await expect(plugins.resolveTrainingFormat(brain.id, "/source/item.json", "application/x-omni-records", selection))
      .rejects.toThrow(/unavailable/i);
  });

  it("rechecks committed plugin source bytes and file-signature MIME at actual start", async () => {
    const rootPath = await root(), path = join(rootPath, "source.records");
    const bytes = Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), Buffer.alloc(32)]);
    await writeFile(path, bytes); const info = await stat(path), hash = createHash("sha256").update(bytes).digest("hex");
    const binding = { pluginId: "fixture.mime-parser", contentSha256: "a".repeat(64), formatId: "unique-extension",
      parserFormat: "jsonl" as const, catalogRevision: 1, instanceRevision: 1, inputMediaType: "image/png" };
    const entry: DatasetManifestEntry = { index: 0, path, relativePath: "source.records", format: "jsonl", bytes: info.size,
      lastModifiedMs: info.mtimeMs, contentSha256: hash, pluginFormat: binding,
      sourceMediaType: { value: "image/png", source: "file-signature" } };
    await expect(assertPluginManifestSourceStable(entry)).resolves.toBeUndefined();
    await expect(assertPluginManifestSourceStable({ ...entry, pluginFormat: { ...binding, inputMediaType: "image/jpeg" },
      sourceMediaType: { value: "image/jpeg", source: "file-signature" } })).rejects.toThrow(/MIME signature changed/i);
    await writeFile(path, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), Buffer.alloc(31), Buffer.from([1])]));
    await expect(assertPluginManifestSourceStable(entry)).rejects.toThrow(/source bytes changed/i);
    expect(pluginParserBindingSha256(binding)).not.toBe(pluginParserBindingSha256({ ...binding, inputMediaType: "image/jpeg" }));
  });

  it("binds an explicit web MIME override to exact downloaded bytes and durable source provenance", async () => {
    const body = Buffer.from('{"x":1}\n{"x":2}\n');
    const server = createServer((_request, response) => {
      response.writeHead(200, { "content-type": "application/x-omni-records; charset=utf-8", "content-length": body.length });
      response.end(body);
    });
    servers.push(server);
    await new Promise<void>((resolve, reject) => server.listen(0, "127.0.0.1", resolve).once("error", reject));
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Fixture server address unavailable.");
    const requests: Array<{ params: Record<string, unknown>; bytes: Buffer }> = [];
    const engine = { request: vi.fn(async (method: string, params: Record<string, unknown>) => {
      if (method !== "ingest") return {};
      const bytes = await import("node:fs/promises").then(fs => fs.readFile(String(params.path)));
      requests.push({ params, bytes });
      return { transactionKey: params.transactionKey, source: { kind: "json", learned_ideas: 2, learned_concepts: 2,
        synaptic_update_events: 3, parameter_update_steps: 1, parameter_checksum_changed: true } };
    }) } as unknown as EngineSupervisor;
    const rootPath = await root();
    const { repository, brain, service, installed } = await configured(rootPath, engine);
    const url = `http://127.0.0.1:${address.port}/known.json`;
    const result = await service.ingestWeb({ brainId: brain.id, url, policy: "pretrain", quarantine: false,
      parserSelection: { pluginId: installed.package.manifest.id, contentSha256: installed.contentSha256, formatId: "mime-override" } });
    expect(requests).toHaveLength(1);
    expect(requests[0]!.bytes).toEqual(body);
    expect(requests[0]!.params).toMatchObject({ kind: "jsonl", contentHash: createHash("sha256").update(body).digest("hex"),
      transactionKey: expect.stringMatching(/^[a-f0-9]{64}$/) });
    expect(result.source).toMatchObject({ requestedUrl: url, provenanceUrl: url, contentHash: createHash("sha256").update(body).digest("hex"),
      parserBindingSha256: expect.stringMatching(/^[a-f0-9]{64}$/), rawTextRetained: false });
    await expect(repository.trainingSourceById(brain.id, result.source.id)).resolves.toMatchObject({
      parserBindingSha256: result.source.parserBindingSha256, requestedUrl: url, provenanceUrl: url
    });
  });

  it("keeps normalized executable-parser MIME provenance in the durable source ledger", async () => {
    const rootPath = await root(), ledger = await BrainActivityLedger.open(rootPath, "brain-mime-ledger");
    try {
      ledger.upsertTrainingSources([{ id: "source-a", name: "source.records", kind: "json", bytes: 12,
        learnedIdeas: 1, learnedConcepts: 1, learnedSynapses: 1, importedAt: "2026-10-13T00:00:00.000Z",
        rawTextRetained: false, contentHash: "a".repeat(64), parserBindingSha256: "b".repeat(64),
        parserExecution: { format: "omni-plugin-parser-execution", version: 1, pluginId: "fixture.mime-parser",
          packageSha256: "c".repeat(64), formatId: "mime-override", moduleId: "records", moduleSha256: "d".repeat(64),
          abi: "omni-wasm-record-stream-1", sourceSha256: "a".repeat(64), sourceBytes: 12,
          inputMediaType: "application/x-omni-records", outputSha256: "e".repeat(64), outputBytes: 10,
          outputRecords: 2, complete: true } }]);
      expect(ledger.trainingSourceById("source-a")?.parserExecution?.inputMediaType).toBe("application/x-omni-records");
    } finally { ledger.close(); }
  });

  it("carries the explicit parser tuple and HTTP MIME through a real bounded crawl source", async () => {
    const body = Buffer.from('{"crawl":true}\n');
    const server = createServer((request, response) => {
      if (request.url === "/robots.txt") { response.writeHead(200, { "content-type": "text/plain" }); response.end(""); return; }
      response.writeHead(200, { "content-type": "application/x-omni-records", "content-length": body.length }); response.end(body);
    });
    servers.push(server);
    await new Promise<void>((resolve, reject) => server.listen(0, "127.0.0.1", resolve).once("error", reject));
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Fixture server address unavailable.");
    const ingested: Array<Record<string, unknown>> = [];
    const engine = { request: vi.fn(async (method: string, params: Record<string, unknown>) => {
      if (method !== "ingest") return {};
      ingested.push(params);
      return { transactionKey: params.transactionKey, source: { kind: "json", learned_ideas: 1, learned_concepts: 1,
        synaptic_update_events: 1, parameter_update_steps: 1, parameter_checksum_changed: true } };
    }) } as unknown as EngineSupervisor;
    const rootPath = await root();
    const { brain, service, installed } = await configured(rootPath, engine);
    const url = `http://127.0.0.1:${address.port}/crawl.json`;
    const result = await service.crawlWeb({ brainId: brain.id, url, policy: "pretrain", quarantine: false,
      maxPages: 1, maxDepth: 0, respectRobots: true,
      parserSelection: { pluginId: installed.package.manifest.id, contentSha256: installed.contentSha256, formatId: "mime-override" } });
    expect(result).toMatchObject({ visited: 1, coverage: { complete: true } });
    expect(ingested).toHaveLength(1);
    expect(ingested[0]).toMatchObject({ kind: "jsonl", contentHash: createHash("sha256").update(body).digest("hex") });
    expect(result.results[0]?.source).toMatchObject({ requestedUrl: url, provenanceUrl: url,
      parserBindingSha256: expect.stringMatching(/^[a-f0-9]{64}$/) });
  });
});
