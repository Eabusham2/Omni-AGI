import { createHash, randomUUID } from "node:crypto";
import { mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService, type RuntimeJobManager } from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { PluginCatalog, canonicalPluginPackage } from "../src/main/pluginCatalog";
import { PluginService } from "../src/main/pluginService";
import { pluginSemanticLearningExperience } from "../src/main/pluginSemanticLearning";
import { DatasetManifestStore } from "../src/main/dataIngestion";
import { ToolExecutor } from "../src/main/toolExecutor";
import {
  OMNI_PLUGIN_FORMAT,
  OMNI_PLUGIN_FORMAT_VERSION,
  pluginToolId,
  type OmniPluginPackage
} from "../src/shared/plugins";
import { DEFAULT_CONFIG } from "../src/shared/types";

function deferred<T = void>(): { promise: Promise<T>; resolve(value: T): void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

function packageFixture(version = "1.0.0", extra: Partial<OmniPluginPackage> = {}): OmniPluginPackage {
  return {
    format: OMNI_PLUGIN_FORMAT,
    formatVersion: OMNI_PLUGIN_FORMAT_VERSION,
    manifest: {
      id: "example.notes",
      version,
      name: "Notes helper",
      description: "A declarative file-reading helper.",
      author: "Fixture author",
      license: "MIT",
      homepage: "https://example.test/notes",
      documentation: "# Notes helper\n\nReads one explicitly supplied file through System Files.",
      capabilities: [{ toolId: "system.files", actions: ["read"], reason: "Read the selected note." }],
      tools: [{
        id: "reader",
        title: "Read note",
        description: "Reads a note through the normal file capability.",
        actions: [{
          id: "read",
          title: "Read note",
          inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
          delegate: { toolId: "system.files", action: "read", arguments: { forward: ["path"] } }
        }]
      }],
      panels: [{ id: "about", title: "About", html: "<section><h2>Notes helper</h2><p>Static panel.</p></section>",
        css: "section { padding: 1rem; }" }],
      themes: [{ id: "violet", title: "Violet", tokens: { "--accent": "#8b5cf6" } }],
      trainingFormats: [{ id: "note-bundle", title: "Note bundle", description: "Declares note bundle files.",
        extensions: ["notes"], mediaTypes: ["application/x-notes"], parser: "declaration-only" }]
    },
    ...extra
  };
}

async function setCatalogEnabled(
  catalog: PluginCatalog,
  pluginId: string,
  enabled: boolean
) {
  const entry = (await catalog.list()).find((candidate) => candidate.pluginId === pluginId)!;
  const packageValue = await catalog.package(pluginId, entry.activeContentSha256);
  return catalog.setEnabled(
    pluginId, enabled, entry.activeContentSha256, entry.revision,
    enabled ? pluginSemanticLearningExperience(packageValue, entry.activeContentSha256).sha256 : undefined
  );
}

async function setPluginEnabled(
  plugins: PluginService,
  pluginId: string,
  enabled: boolean
) {
  const entry = (await plugins.list()).find((candidate) => candidate.pluginId === pluginId)!;
  return plugins.setEnabled(pluginId, enabled, entry.activeContentSha256, entry.revision);
}

describe("versioned plugin catalog", () => {
  const roots: string[] = [];
  afterEach(async () => {
    await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
  });

  async function catalog(): Promise<PluginCatalog> {
    const root = await mkdtemp(join(tmpdir(), "omni-plugin-catalog-"));
    roots.push(root);
    const catalog = new PluginCatalog(join(root, "plugins"));
    await catalog.initialize();
    return catalog;
  }

  it("stores immutable versions, makes edits new lineage, and activates disabled", async () => {
    const store = await catalog();
    const first = await store.install(packageFixture(), "fixture.omni-plugin");
    expect(first).toMatchObject({ installed: true, entry: { enabled: false, activeVersion: "1.0.0" } });
    expect(first.contentSha256).toMatch(/^[a-f0-9]{64}$/);
    const repeated = await store.install(packageFixture(), "same fixture");
    expect(repeated.installed).toBe(false);
    const changedSameVersion = packageFixture();
    changedSameVersion.manifest.description = "Changed immutable bytes.";
    await expect(store.install(changedSameVersion)).rejects.toThrow(/different immutable content/i);

    const edited = packageFixture("1.1.0");
    edited.manifest.description = "A reviewed second version.";
    const second = await store.edit("example.notes", first.contentSha256, edited);
    expect(second.package.parentSha256).toBe(first.contentSha256);
    expect(second.entry.activeContentSha256).toBe(first.contentSha256);
    const staleEnable = { hash: second.entry.activeContentSha256, revision: second.entry.revision };
    const activated = await store.activate("example.notes", second.contentSha256);
    expect(activated).toMatchObject({ activeVersion: "1.1.0", enabled: false });
    await expect(store.setEnabled("example.notes", true, staleEnable.hash, staleEnable.revision, "e".repeat(64)))
      .rejects.toThrow(/active plugin version changed/i);
    expect((await store.list())[0]).toMatchObject({ activeContentSha256: second.contentSha256, enabled: false });
    expect((await setCatalogEnabled(store, "example.notes", true)).enabled).toBe(true);
  });

  it("round trips exact declarative content and rejects hidden execution surfaces", async () => {
    const store = await catalog();
    const installed = await store.install(packageFixture());
    const destination = join(roots[0]!, "notes.omni-plugin");
    await store.exportFile("example.notes", destination, installed.contentSha256);
    const importedStore = new PluginCatalog(join(roots[0]!, "imported"));
    await importedStore.initialize();
    const imported = await importedStore.importFile(destination);
    expect(imported.contentSha256).toBe(installed.contentSha256);
    expect(imported.package).toEqual(installed.package);
    expect(await readFile(destination, "utf8")).toContain(OMNI_PLUGIN_FORMAT);

    const scripted = packageFixture();
    scripted.manifest.panels[0]!.html = "<script>require('node:fs')</script>";
    await expect(store.install(scripted)).rejects.toThrow(/static markup/i);
    for (const payload of [
      "<svg/onload=alert(1)>",
      "<img/onerror=alert(1)>",
      "<img/src=https://example.test/leak>",
      "<img/srcset=https://example.test/leak>",
      "<svg><a/href=https://example.test>leave</a></svg>",
      "<body><p>break out</p></body>"
    ]) {
      const malformed = packageFixture();
      malformed.manifest.panels[0]!.html = payload;
      await expect(store.install(malformed)).rejects.toThrow(/static markup/i);
    }
    const recursive = packageFixture();
    recursive.manifest.capabilities[0] = { toolId: "plugin.other.tool", actions: ["run"], reason: "No." };
    recursive.manifest.tools[0]!.actions[0]!.delegate = {
      toolId: "plugin.other.tool", action: "run", arguments: { forward: "all" }
    };
    await expect(store.install(recursive)).rejects.toThrow(/non-plugin typed host tool/i);
    const setupScript = { ...packageFixture(), setup: "npm install && node setup.js" };
    await expect(store.install(setupScript)).rejects.toThrow(/unsupported fields/i);
  });

  it("requires explicit per-instance capability and tool choices bound to the active hash", async () => {
    const store = await catalog();
    const installed = await store.install(packageFixture());
    const toolId = pluginToolId("example.notes", "reader");
    const semanticSha256 = pluginSemanticLearningExperience(installed.package, installed.contentSha256).sha256;
    await expect(store.setInstanceAccess({ brainId: "brain-one", pluginId: "example.notes",
      contentSha256: installed.contentSha256, enabled: true, capabilities: [],
      toolPermissions: [{ toolId, level: "ask" }], requestId: "missing-capability" }, semanticSha256))
      .rejects.toThrow(/every plugin capability/i);
    const access = await store.setInstanceAccess({ brainId: "brain-one", pluginId: "example.notes",
      contentSha256: installed.contentSha256, enabled: true,
      capabilities: [{ toolId: "system.files", actions: ["read"], enabled: true }],
      toolPermissions: [{ toolId, level: "full" }], requestId: "reviewed-access" }, semanticSha256);
    expect(access).toMatchObject({ enabled: true, contentSha256: installed.contentSha256,
      toolPermissions: [{ toolId, level: "full" }] });
    expect(await store.semanticLearning("brain-one", "example.notes", installed.contentSha256))
      .toMatchObject({ state: "pending", experienceSha256: semanticSha256 });
  });

  it("hashes Unicode schema and constant keys by portable code-point order", () => {
    const left = packageFixture();
    left.manifest.tools[0]!.actions[0]!.inputSchema = {
      type: "object",
      properties: { "é": { type: "string" }, z: { type: "number" }, "Ω": { type: "boolean" } }
    };
    left.manifest.tools[0]!.actions[0]!.delegate.arguments.constants = { "Ω": true, z: 1, "é": "value" };
    const right = packageFixture();
    right.manifest.tools[0]!.actions[0]!.inputSchema = {
      properties: { "Ω": { type: "boolean" }, z: { type: "number" }, "é": { type: "string" } },
      type: "object"
    };
    right.manifest.tools[0]!.actions[0]!.delegate.arguments.constants = { "é": "value", z: 1, "Ω": true };
    const first = canonicalPluginPackage(left), second = canonicalPluginPackage(right);
    expect(first.sha256).toBe(second.sha256);
    expect(first.json).toBe(second.json);
  });

  it("binds a reviewed custom extension to an existing streaming parser and revokes its live lease", async () => {
    const store = await catalog();
    const candidate = packageFixture();
    candidate.manifest.trainingFormats[0]!.parser = { kind: "builtin-stream", format: "jsonl" };
    const installed = await store.install(candidate);
    await setCatalogEnabled(store, "example.notes", true);
    const outerToolId = pluginToolId("example.notes", "reader");
    const access = await store.setInstanceAccess({ brainId: "brain-one", pluginId: "example.notes",
      contentSha256: installed.contentSha256, enabled: true,
      capabilities: [{ toolId: "system.files", actions: ["read"], enabled: true }],
      toolPermissions: [{ toolId: outerToolId, level: "ask" }], requestId: "format-access" },
      pluginSemanticLearningExperience(installed.package, installed.contentSha256).sha256);
    const brain = {
      registerExternalToolSchemas: vi.fn(), unregisterExternalToolSchemas: vi.fn(),
      listToolPermissions: vi.fn(async () => [{ toolId: outerToolId, label: "Reader", level: "ask" as const,
        revision: 1, updatedAt: new Date().toISOString() }]),
      setToolPermission: vi.fn()
    } as unknown as BrainService;
    const plugins = new PluginService(store, brain);
    await plugins.initialize();
    const source = join(roots[0]!, "fixture.notes");
    await writeFile(source, '{"text":"first"}\n{"text":"second"}\n');
    const manifests = new DatasetManifestStore(() => join(roots[0]!, "brain"), {
      resolvePluginTrainingFormat: (brainId, path) => plugins.resolveTrainingFormat(brainId, path)
    });
    const manifest = await manifests.create("brain-one", [source]);
    const entries = [];
    for await (const entry of manifests.entries("brain-one", manifest.id)) entries.push(entry);
    expect(entries).toEqual([expect.objectContaining({ format: "jsonl", pluginFormat: {
      pluginId: "example.notes", contentSha256: installed.contentSha256, formatId: "note-bundle",
      parserFormat: "jsonl", catalogRevision: expect.any(Number), instanceRevision: access.revision
    } })]);
    const lease = await plugins.acquireTrainingFormat("brain-one", entries[0]!.pluginFormat!);
    expect(lease.signal.aborted).toBe(false);
    const inactive = packageFixture("1.1.0");
    inactive.manifest.trainingFormats[0]!.parser = { kind: "builtin-stream", format: "csv" };
    await plugins.edit("example.notes", installed.contentSha256, inactive);
    expect(lease.signal.aborted).toBe(false);
    await setPluginEnabled(plugins, "example.notes", false);
    expect(lease.signal.aborted).toBe(true);
    lease.release();

    await setPluginEnabled(plugins, "example.notes", true);
    const validate = plugins.validateTrainingFormat.bind(plugins);
    const validating = deferred<void>(), continueValidation = deferred<void>();
    const validation = vi.spyOn(plugins, "validateTrainingFormat").mockImplementation(async (...args) => {
      validating.resolve(); await continueValidation.promise; return validate(...args);
    });
    const racedLease = plugins.acquireTrainingFormat("brain-one", entries[0]!.pluginFormat!);
    await validating.promise;
    await setPluginEnabled(plugins, "example.notes", false);
    continueValidation.resolve();
    await expect(racedLease).rejects.toThrow(/changed|aborted/i);
    validation.mockRestore();
    await setPluginEnabled(plugins, "example.notes", true);
    const secondPackage = structuredClone(candidate);
    secondPackage.manifest.id = "example.more";
    secondPackage.manifest.name = "More notes";
    const second = await store.install(secondPackage);
    await setCatalogEnabled(store, "example.more", true);
    await store.setInstanceAccess({ brainId: "brain-one", pluginId: "example.more",
      contentSha256: second.contentSha256, enabled: true,
      capabilities: [{ toolId: "system.files", actions: ["read"], enabled: true }],
      toolPermissions: [{ toolId: pluginToolId("example.more", "reader"), level: "ask" }],
      requestId: "second-format-access" },
      pluginSemanticLearningExperience(second.package, second.contentSha256).sha256);
    await expect(plugins.resolveTrainingFormat("brain-one", source)).rejects.toThrow(/multiple enabled plugin formats/i);

    const executable = packageFixture("2.0.0");
    executable.manifest.trainingFormats[0]!.parser = { kind: "javascript", format: "jsonl" } as never;
    await expect(store.install(executable)).rejects.toThrow(/built-in streaming format/i);
  });

  it("queues immutable plugin documentation and schemas without waiting for the neural writer", async () => {
    const store = await catalog();
    let revision = 1;
    const outerToolId = pluginToolId("example.notes", "reader");
    const learning = deferred<{
      source: { id: string; contentHash: string; rawTextRetained: false };
    }>();
    let learnedContent = "";
    const brain = {
      registerExternalToolSchemas: vi.fn(), unregisterExternalToolSchemas: vi.fn(),
      listToolPermissions: vi.fn(async () => [{ toolId: outerToolId, label: "Reader", level: "ask" as const,
        revision, updatedAt: new Date().toISOString() }]),
      setToolPermission: vi.fn(async (_brainId: string, toolId: string, level: "off" | "ask" | "auto" | "full") => [{
        toolId, label: "Reader", level, revision: ++revision, updatedAt: new Date().toISOString()
      }]),
      requireInstanceLoaded: vi.fn(),
      learnLoadedStructuredExperience: vi.fn(async (_brainId: string, experience: { content: string }) => {
        learnedContent = experience.content;
        return learning.promise;
      })
    } as unknown as BrainService;
    const plugins = new PluginService(store, brain);
    const events: import("../src/shared/plugins").PluginEvent[] = [];
    plugins.onEvent((event) => events.push(event));
    await plugins.initialize();
    const installed = await plugins.installPackage(packageFixture());
    await setPluginEnabled(plugins, "example.notes", true);
    const enabled = await plugins.setInstanceAccess({ brainId: "brain-one", pluginId: "example.notes",
      contentSha256: installed.contentSha256, enabled: true,
      capabilities: [{ toolId: "system.files", actions: ["read"], enabled: true }],
      toolPermissions: [{ toolId: outerToolId, level: "ask" }], requestId: "semantic-access" });
    expect(enabled).toMatchObject({ enabled: true, semanticLearning: {
      contentSha256: installed.contentSha256, state: expect.stringMatching(/^(pending|running)$/)
    } });
    await vi.waitFor(() => expect(brain.learnLoadedStructuredExperience).toHaveBeenCalledOnce());
    expect(await store.semanticLearning("brain-one", "example.notes", installed.contentSha256))
      .toMatchObject({ state: "running" });
    expect(brain.learnLoadedStructuredExperience).toHaveBeenCalledOnce();
    const parsed = JSON.parse(learnedContent) as Record<string, unknown>;
    expect(parsed).toMatchObject({ format: "omni-plugin-semantic-declaration", version: 1,
      executedHostActions: false, fabricatedToolResults: false, hiddenPrompt: false, behavioralPersona: false });
    const learnedSha256 = createHash("sha256").update(learnedContent).digest("hex");
    await setPluginEnabled(plugins, "example.notes", false);
    await setPluginEnabled(plugins, "example.notes", true);
    learning.resolve({ source: { id: "source-one", contentHash: learnedSha256, rawTextRetained: false } });
    await plugins.settleSemanticLearning("brain-one");
    expect(brain.learnLoadedStructuredExperience).toHaveBeenCalledTimes(2);
    expect(await store.semanticLearning("brain-one", "example.notes", installed.contentSha256))
      .toMatchObject({ state: "complete", experienceSha256: learnedSha256, trainingSourceId: "source-one" });
    expect(events.filter((event) => event.type === "semantic-learning").map((event) => event.semanticState))
      .toEqual(expect.arrayContaining(["pending", "running", "complete"]));
    const completedEvent = events.find((event) => event.semanticState === "complete")!;
    expect(completedEvent).toMatchObject({ brainId: "brain-one", pluginId: "example.notes",
      contentSha256: installed.contentSha256, catalogRevision: expect.any(Number) });
    expect(Object.keys(completedEvent).sort()).toEqual([
      "at", "brainId", "catalogRevision", "contentSha256", "pluginId", "semanticState", "type"
    ]);
  });

  it("sends original custom-format bytes through the selected worker stream and pauses on revocation", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-plugin-format-ingest-"));
    roots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Plugin format brain", idleCognition: false });
    const entered = deferred<void>();
    const requests: Array<Record<string, unknown>> = [];
    const engine = {
      request: vi.fn(async (method: string, params: Record<string, unknown>, _timeout?: number, signal?: AbortSignal) => {
        if (method !== "ingest") return {};
        requests.push(params);
        if (String(params.path).endsWith("pause.notes")) {
          entered.resolve();
          await new Promise<never>((_resolve, reject) => {
            const abort = (): void => reject(signal?.reason ?? new Error("cancelled"));
            if (signal?.aborted) abort();
            else signal?.addEventListener("abort", abort, { once: true });
          });
        }
        return {
          transactionKey: params.transactionKey,
          source: { kind: "json", learned_ideas: 1, learned_concepts: 1, synaptic_update_events: 1,
            parameter_update_steps: 1, parameter_checksum_changed: true },
          coverage: { discoveredFiles: 1, completedFiles: 1, rejectedFiles: 0,
            discoveredRecords: 2, processedRecords: 2, rejectedRecords: 0,
            processedBytes: 34, shards: 0, modalityCounts: { jsonl: 2 }, errors: [], complete: true }
        };
      })
    } as unknown as EngineSupervisor;
    const service = new BrainService(repository, engine);
    await expect(service.learnLoadedStructuredExperience(brain.id, {
      content: "must not implicitly load",
      name: "unloaded",
      sourceLabel: "unloaded plugin fixture",
      license: "fixture"
    })).rejects.toThrow();
    expect(requests).toHaveLength(0);
    const plugins = new PluginService(new PluginCatalog(join(root, "plugins")), service);
    await plugins.initialize();
    const candidate = packageFixture();
    candidate.manifest.trainingFormats[0]!.parser = { kind: "builtin-stream", format: "jsonl" };
    const installed = await plugins.installPackage(candidate);
    await setPluginEnabled(plugins, "example.notes", true);
    const outerToolId = pluginToolId("example.notes", "reader");
    await plugins.setInstanceAccess({ brainId: brain.id, pluginId: "example.notes",
      contentSha256: installed.contentSha256, enabled: true,
      capabilities: [{ toolId: "system.files", actions: ["read"], enabled: true }],
      toolPermissions: [{ toolId: outerToolId, level: "ask" }], requestId: "format-ingest-access" });
    service.setPluginTrainingFormatResolver(plugins);

    const source = join(root, "complete.notes");
    const bytes = '{"text":"first"}\n{"text":"second"}\n';
    await writeFile(source, bytes);
    const manifest = await service.previewDataset(brain.id, [source]);
    const result = await service.ingestManifest(brain.id, manifest.id);
    expect(result.coverage.complete).toBe(true);
    expect(requests[0]).toMatchObject({ path: await realpath(source), kind: "jsonl",
      contentHash: createHash("sha256").update(bytes).digest("hex") });

    const remappedPackage = packageFixture("2.0.0");
    remappedPackage.manifest.trainingFormats[0]!.parser = { kind: "builtin-stream", format: "csv" };
    const remapped = await plugins.edit("example.notes", installed.contentSha256, remappedPackage);
    await plugins.activate("example.notes", remapped.contentSha256);
    await setPluginEnabled(plugins, "example.notes", true);
    await plugins.setInstanceAccess({ brainId: brain.id, pluginId: "example.notes",
      contentSha256: remapped.contentSha256, enabled: true,
      capabilities: [{ toolId: "system.files", actions: ["read"], enabled: true }],
      toolPermissions: [{ toolId: outerToolId, level: "ask" }], requestId: "remapped-format-access" });
    const remappedManifest = await service.previewDataset(brain.id, [source]);
    await service.ingestManifest(brain.id, remappedManifest.id, "encode", () => false,
      () => undefined, true, undefined, false);
    expect(requests).toHaveLength(2);
    expect(requests[1]).toMatchObject({ path: await realpath(source), kind: "csv", allowReplay: true });
    const repeatedManifest = await service.previewDataset(brain.id, [source]);
    await service.ingestManifest(brain.id, repeatedManifest.id, "encode", () => false,
      () => undefined, true, undefined, false);
    expect(requests).toHaveLength(2);

    const pausedSource = join(root, "pause.notes");
    await writeFile(pausedSource, '{"text":"third"}\n{"text":"fourth"}\n');
    const pausedManifest = await service.previewDataset(brain.id, [pausedSource]);
    const running = service.ingestManifest(brain.id, pausedManifest.id);
    await entered.promise;
    await setPluginEnabled(plugins, "example.notes", false);
    const paused = await running;
    expect(paused).toMatchObject({ paused: true, coverage: { processedFiles: 0, complete: false } });
    await expect(service.datasets.cursor(brain.id, pausedManifest.id)).resolves.toMatchObject({
      nextEntry: 0, state: "paused"
    });
  });
});

describe("plugin nested ToolExecutor authority", () => {
  const roots: string[] = [];
  afterEach(async () => {
    await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
  });

  it("shows the exact nested Ask operation and invalidates approval across catalog changes", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-plugin-executor-"));
    roots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Plugin brain", idleCognition: false });
    const service = new BrainService(repository, {
      request: vi.fn(async () => ({})),
      requestStream: vi.fn(async () => ({ text: "unused", trace: { id: "unused" } }))
    } as unknown as EngineSupervisor);
    const plugins = new PluginService(new PluginCatalog(join(root, "plugins")), service);
    await plugins.initialize();
    const installed = await plugins.installPackage(packageFixture());
    await setPluginEnabled(plugins, "example.notes", true);
    const outerToolId = pluginToolId("example.notes", "reader");
    await plugins.setInstanceAccess({ brainId: brain.id, pluginId: "example.notes",
      contentSha256: installed.contentSha256, enabled: true,
      capabilities: [{ toolId: "system.files", actions: ["read"], enabled: true }],
      toolPermissions: [{ toolId: outerToolId, level: "full" }], requestId: "enable-plugin" });
    await service.setToolPermission(brain.id, "system.files", "ask", "nested-files-ask");
    const executor = new ToolExecutor(service, {
      generate: vi.fn(() => { throw new Error("Unexpected modality job."); })
    } as unknown as RuntimeJobManager, undefined, undefined, undefined, undefined, plugins);
    vi.spyOn(executor as unknown as { audit(...args: unknown[]): Promise<void> }, "audit").mockResolvedValue(undefined);
    const target = join(repository.brainDirectory(brain.id), "note.txt");
    await writeFile(target, "nested permission content");
    const invocation = { brainId: brain.id, toolId: outerToolId, action: "read", arguments: { path: target } };

    const first = await executor.execute(invocation);
    expect(first).toMatchObject({ state: "approval-required", output: {
      kind: "plugin-nested-operation", pluginId: "example.notes",
      resolved: { toolId: "system.files", action: "read", arguments: { path: target } },
      permissionSources: expect.arrayContaining([
        expect.objectContaining({ toolId: outerToolId, level: "full", inherited: false }),
        expect.objectContaining({ toolId: "system.files", level: "ask", inherited: true })
      ])
    } });
    await setPluginEnabled(plugins, "example.notes", false);
    await setPluginEnabled(plugins, "example.notes", true);
    const stale = await executor.execute({ ...invocation, approvalToken: first.approvalToken });
    expect(stale.state).toBe("approval-required");
    expect(stale.approvalToken).not.toBe(first.approvalToken);
    const completed = await executor.execute({ ...invocation, approvalToken: stale.approvalToken },
      undefined, randomUUID(), async () => undefined);
    expect(completed.state).toBe("complete");
    expect(completed.output).toMatchObject({ content: "nested permission content" });
    await service.setToolPermission(brain.id, outerToolId, "off", "later-user-plugin-off");
    await expect(plugins.setInstanceAccess({ brainId: brain.id, pluginId: "example.notes",
      contentSha256: installed.contentSha256, enabled: true,
      capabilities: [{ toolId: "system.files", actions: ["read"], enabled: true }],
      toolPermissions: [{ toolId: outerToolId, level: "full" }], requestId: "enable-plugin" }))
      .rejects.toThrow(/superseded|already used/i);
    expect((await service.listToolPermissions(brain.id)).find((permission) => permission.toolId === outerToolId)?.level).toBe("off");
    expect((await plugins.instance(brain.id, "example.notes"))?.enabled).toBe(false);
  });

  it("never lets outer Full bypass nested Off or risky Auto file scope", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-plugin-nested-policy-"));
    roots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Nested policy", idleCognition: false });
    const service = new BrainService(repository, {
      request: vi.fn(async () => ({})), requestStream: vi.fn(async () => ({ text: "", trace: { id: "unused" } }))
    } as unknown as EngineSupervisor);
    const plugins = new PluginService(new PluginCatalog(join(root, "plugins")), service);
    await plugins.initialize();
    const installed = await plugins.installPackage(packageFixture());
    await setPluginEnabled(plugins, "example.notes", true);
    const outerToolId = pluginToolId("example.notes", "reader");
    await plugins.setInstanceAccess({ brainId: brain.id, pluginId: "example.notes", contentSha256: installed.contentSha256,
      enabled: true, capabilities: [{ toolId: "system.files", actions: ["read"], enabled: true }],
      toolPermissions: [{ toolId: outerToolId, level: "full" }], requestId: "nested-policy-access" });
    const executor = new ToolExecutor(service, { generate: vi.fn() } as unknown as RuntimeJobManager,
      undefined, undefined, undefined, undefined, plugins);
    vi.spyOn(executor as unknown as { audit(...args: unknown[]): Promise<void> }, "audit").mockResolvedValue(undefined);
    const outside = join(root, "outside.txt");
    await writeFile(outside, "outside");
    await service.setToolPermission(brain.id, "system.files", "off", "nested-off");
    const denied = await executor.execute({ brainId: brain.id, toolId: outerToolId, action: "read", arguments: { path: outside } });
    expect(denied).toMatchObject({ state: "failed", error: expect.stringMatching(/nested .* disabled/i) });
    await service.setToolPermission(brain.id, "system.files", "auto", "nested-auto");
    const review = await executor.execute({ brainId: brain.id, toolId: outerToolId, action: "read", arguments: { path: outside } });
    expect(review).toMatchObject({ state: "approval-required", output: {
      resolved: { toolId: "system.files", action: "read", arguments: { path: outside } }
    } });
    await service.setToolPermission(brain.id, "system.files", "full", "nested-full");
    await service.setToolPermission(brain.id, outerToolId, "auto", "outer-auto");
    const inheritedAutoReview = await executor.execute({ brainId: brain.id, toolId: outerToolId,
      action: "read", arguments: { path: outside } });
    expect(inheritedAutoReview.state).toBe("approval-required");
    expect((await plugins.instance(brain.id, "example.notes"))?.toolPermissions)
      .toContainEqual({ toolId: outerToolId, level: "auto" });
  });

  it("aborts a gated pre-effect execution when its resolved nested policy changes", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-plugin-pre-effect-"));
    roots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Pre-effect policy", idleCognition: false });
    const service = new BrainService(repository, {
      request: vi.fn(async () => ({})), requestStream: vi.fn(async () => ({ text: "", trace: { id: "unused" } }))
    } as unknown as EngineSupervisor);
    const plugins = new PluginService(new PluginCatalog(join(root, "plugins")), service);
    await plugins.initialize();
    const installed = await plugins.installPackage(packageFixture());
    await setPluginEnabled(plugins, "example.notes", true);
    const outerToolId = pluginToolId("example.notes", "reader");
    await plugins.setInstanceAccess({ brainId: brain.id, pluginId: "example.notes", contentSha256: installed.contentSha256,
      enabled: true, capabilities: [{ toolId: "system.files", actions: ["read"], enabled: true }],
      toolPermissions: [{ toolId: outerToolId, level: "full" }], requestId: "pre-effect-access" });
    await service.setToolPermission(brain.id, "system.files", "full", "pre-effect-nested-full");
    const entered = deferred(), release = deferred();
    vi.spyOn(service, "authorizeInlineImagination").mockImplementation(async () => {
      entered.resolve(); await release.promise;
    });
    const executor = new ToolExecutor(service, { generate: vi.fn() } as unknown as RuntimeJobManager,
      undefined, undefined, undefined, undefined, plugins);
    const dispatch = vi.spyOn(executor as unknown as { dispatch(...args: unknown[]): Promise<unknown> }, "dispatch");
    vi.spyOn(executor as unknown as { audit(...args: unknown[]): Promise<void> }, "audit").mockResolvedValue(undefined);
    const path = join(repository.brainDirectory(brain.id), "before-effect.txt");
    await writeFile(path, "must not dispatch after revocation");
    const execution = executor.execute({ brainId: brain.id, toolId: outerToolId, action: "read", arguments: { path } });
    await entered.promise;
    await service.setToolPermission(brain.id, "system.files", "off", "pre-effect-nested-off");
    release.resolve();
    await expect(execution).resolves.toMatchObject({ state: "failed", error: expect.stringMatching(/permission changed before dispatch/i) });
    expect(dispatch).not.toHaveBeenCalled();
  });

  it("suspends existing access and never rolls a failed edit over a newer human Off", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-plugin-access-race-"));
    roots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Plugin access race", idleCognition: false });
    const service = new BrainService(repository, {
      request: vi.fn(async () => ({})), requestStream: vi.fn(async () => ({ text: "", trace: { id: "unused" } }))
    } as unknown as EngineSupervisor);
    const catalog = new PluginCatalog(join(root, "plugins"));
    const plugins = new PluginService(catalog, service);
    await plugins.initialize();
    const installed = await plugins.installPackage(packageFixture());
    await setPluginEnabled(plugins, "example.notes", true);
    const outerToolId = pluginToolId("example.notes", "reader");
    const baseAccess = { brainId: brain.id, pluginId: "example.notes", contentSha256: installed.contentSha256,
      enabled: true, capabilities: [{ toolId: "system.files", actions: ["read"], enabled: true }],
      toolPermissions: [{ toolId: outerToolId, level: "full" as const }] };
    await plugins.setInstanceAccess({ ...baseAccess, requestId: "initial-review" });
    await expect(plugins.resolve({ brainId: brain.id, toolId: outerToolId, action: "read", arguments: { path: "x" } }))
      .resolves.toBeTruthy();

    const enteredPolicyWrite = deferred(), releasePolicyWrite = deferred();
    const originalSet = service.setToolPermission.bind(service);
    let gateNextPluginWrite = true;
    vi.spyOn(service, "setToolPermission").mockImplementation(async (...args) => {
      const result = await originalSet(...args);
      if (gateNextPluginWrite && args[1] === outerToolId && args[2] === "auto") {
        gateNextPluginWrite = false;
        enteredPolicyWrite.resolve();
        await releasePolicyWrite.promise;
      }
      return result;
    });
    const originalCatalogSet = catalog.setInstanceAccess.bind(catalog);
    vi.spyOn(catalog, "setInstanceAccess").mockImplementation(async (request, semanticSha256) => {
      if (request.requestId === "racing-review" && request.enabled) {
        throw new Error("simulated access publication failure");
      }
      return originalCatalogSet(request, semanticSha256);
    });

    const changing = plugins.setInstanceAccess({ ...baseAccess,
      toolPermissions: [{ toolId: outerToolId, level: "auto" }], requestId: "racing-review" });
    await enteredPolicyWrite.promise;
    await expect(plugins.resolve({ brainId: brain.id, toolId: outerToolId, action: "read", arguments: { path: "x" } }))
      .rejects.toThrow(/not explicitly enabled/i);
    await originalSet(brain.id, outerToolId, "off", "newer-human-off");
    releasePolicyWrite.resolve();
    await expect(changing).rejects.toThrow(/simulated access publication failure/i);
    expect((await service.listToolPermissions(brain.id)).find((permission) => permission.toolId === outerToolId)?.level).toBe("off");
    expect((await catalog.instance(brain.id, "example.notes"))?.enabled).toBe(false);
  });
});
