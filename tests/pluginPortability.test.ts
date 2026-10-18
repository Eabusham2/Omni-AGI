import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PluginCatalog } from "../src/main/pluginCatalog";
import { PluginService } from "../src/main/pluginService";
import {
  pluginSemanticLearningExperience
} from "../src/main/pluginSemanticLearning";
import type { BrainService } from "../src/main/brainService";
import {
  OMNI_PLUGIN_FORMAT,
  OMNI_PLUGIN_FORMAT_VERSION,
  pluginToolId,
  type OmniPluginPackage
} from "../src/shared/plugins";
import type { ToolPermissionLevel, ToolPermissionRecord } from "../src/shared/types";

function packageFixture(id = "example.portable"): OmniPluginPackage {
  return {
    format: OMNI_PLUGIN_FORMAT,
    formatVersion: OMNI_PLUGIN_FORMAT_VERSION,
    manifest: {
      id,
      version: "1.0.0",
      name: "Portable notes",
      description: "A portable declarative notes helper.",
      author: "Fixture author",
      license: "MIT",
      homepage: "https://example.test/portable",
      documentation: "# Portable notes\n\nReads a reviewed local note.",
      capabilities: [{ toolId: "system.files", actions: ["read"], reason: "Read the selected note." }],
      tools: [{
        id: "reader",
        title: "Read note",
        description: "Reads one reviewed note.",
        actions: [{
          id: "read",
          inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
          delegate: { toolId: "system.files", action: "read", arguments: { forward: ["path"] } }
        }]
      }],
      panels: [{ id: "about", title: "About", html: "<section><p>Portable static panel.</p></section>" }],
      themes: [],
      trainingFormats: [{
        id: "portable-lines",
        title: "Portable lines",
        description: "Streams reviewed line records.",
        extensions: ["portable-lines"],
        mediaTypes: ["application/x-portable-lines"],
        parser: { kind: "builtin-stream", format: "jsonl" }
      }]
    }
  };
}

function brainAuthority() {
  let revision = 1;
  const rows = new Map<string, Map<string, ToolPermissionRecord>>();
  const ensure = (brainId: string): Map<string, ToolPermissionRecord> => {
    let selected = rows.get(brainId);
    if (!selected) {
      selected = new Map([["system.files", {
        toolId: "system.files",
        label: "System Files",
        level: "full",
        revision,
        updatedAt: new Date().toISOString()
      }]]);
      rows.set(brainId, selected);
    }
    return selected;
  };
  const listToolPermissions = vi.fn(async (brainId: string) =>
    [...ensure(brainId).values()].map((row) => ({ ...row })));
  const setToolPermission = vi.fn(async (
    brainId: string,
    toolId: string,
    level: ToolPermissionLevel,
    _requestId: string,
    expectedRevision?: number
  ) => {
    const selected = ensure(brainId);
    const currentRevision = Math.max(...[...selected.values()].map((row) => row.revision ?? 0));
    if (expectedRevision !== undefined && expectedRevision !== currentRevision) {
      throw new Error("Host policy revision changed.");
    }
    revision = Math.max(revision, currentRevision) + 1;
    selected.set(toolId, {
      toolId,
      label: toolId,
      level,
      revision,
      updatedAt: new Date().toISOString()
    });
    return [...selected.values()].map((row) => ({ ...row }));
  });
  return {
    rows,
    service: {
      registerExternalToolSchemas: vi.fn(),
      unregisterExternalToolSchemas: vi.fn(),
      listToolPermissions,
      setToolPermission
    } as unknown as BrainService
  };
}

async function enablePackage(
  plugins: PluginService,
  packageValue: OmniPluginPackage,
  brainId: string
) {
  const installed = await plugins.installPackage(packageValue, "portable fixture");
  const entry = (await plugins.list()).find((candidate) => candidate.pluginId === packageValue.manifest.id)!;
  await plugins.setEnabled(packageValue.manifest.id, true, entry.activeContentSha256, entry.revision);
  const toolId = pluginToolId(packageValue.manifest.id, packageValue.manifest.tools[0]!.id);
  const access = await plugins.setInstanceAccess({
    brainId,
    pluginId: packageValue.manifest.id,
    contentSha256: installed.contentSha256,
    enabled: true,
    capabilities: packageValue.manifest.capabilities.map((capability) => ({
      toolId: capability.toolId,
      actions: [...capability.actions],
      enabled: true
    })),
    toolPermissions: [{ toolId, level: "ask" }],
    requestId: `review-${brainId}-${packageValue.manifest.id.replaceAll(".", "-")}`
  });
  return { installed, access, toolId };
}

describe("declarative plugin portability", () => {
  const roots: string[] = [];

  afterEach(async () => {
    await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
  });

  async function service(name: string) {
    const root = await mkdtemp(join(tmpdir(), `omni-plugin-portability-${name}-`));
    roots.push(root);
    const catalog = new PluginCatalog(join(root, "plugins"));
    const authority = brainAuthority();
    const plugins = new PluginService(catalog, authority.service);
    await plugins.initialize();
    return { root, catalog, authority, plugins };
  }

  it("preserves exact packages, reviewed selections, semantic receipts, and dormant external authority", async () => {
    const source = await service("source");
    const fixture = packageFixture();
    const { installed, access, toolId } = await enablePackage(source.plugins, fixture, "brain-source");
    const pending = await source.catalog.semanticLearning("brain-source", fixture.manifest.id, installed.contentSha256);
    const claimed = await source.catalog.claimSemanticLearning(pending!);
    await source.catalog.completeSemanticLearning(claimed!, "plugin-source-receipt");

    const directory = join(source.root, "portable");
    const exported = await source.plugins.exportPortable("brain-source", "current", directory);
    expect(exported).toBeDefined();
    const archive = await source.plugins.validatePortableImport(
      exported!.files,
      exported!.declaration,
      "brain-source"
    );
    expect(archive.packages.get(installed.contentSha256)).toEqual(installed.package);
    expect(archive.manifest.selections).toEqual([expect.objectContaining({
      sourceBrainId: "brain-source",
      pluginId: fixture.manifest.id,
      contentSha256: installed.contentSha256,
      authority: "source-review",
      reviewedAccess: expect.objectContaining({
        enabled: true,
        capabilities: access.capabilities,
        toolPermissions: [{ toolId, level: "ask" }]
      }),
      semanticLearning: expect.objectContaining({
        state: "complete",
        trainingSourceId: "plugin-source-receipt"
      }),
      provenance: expect.objectContaining({ sourceLabel: "portable fixture", license: "MIT" })
    })]);

    const destination = await service("destination");
    await destination.plugins.adoptExternalPortable("brain-import", archive);
    expect(await destination.catalog.package(fixture.manifest.id, installed.contentSha256))
      .toEqual(installed.package);
    expect((await destination.plugins.list())[0]).toMatchObject({ enabled: false });
    expect(await destination.plugins.instance("brain-import", fixture.manifest.id)).toBeNull();
    expect(destination.authority.service.setToolPermission).not.toHaveBeenCalled();
    expect(await destination.plugins.portable("brain-import")).toEqual([expect.objectContaining({
      sourceBrainId: "brain-source",
      authority: "external-dormant",
      originAuthority: "source-review",
      semanticLearning: expect.objectContaining({ state: "complete" })
    })]);

    const manifestText = await readFile(exported!.files.get("manifest.json")!, "utf8");
    expect(manifestText).not.toMatch(/approvalToken|credential|bearer|password/i);
  });

  it("preserves trusted duplicate selections but keeps full-merge parents dormant and distinct", async () => {
    const host = await service("lineage");
    const fixture = packageFixture();
    const { installed, toolId } = await enablePackage(host.plugins, fixture, "parent-one");
    const firstPending = await host.catalog.semanticLearning("parent-one", fixture.manifest.id, installed.contentSha256);
    const firstClaim = await host.catalog.claimSemanticLearning(firstPending!);
    await host.catalog.completeSemanticLearning(firstClaim!, "semantic-parent-one");

    await host.plugins.cloneTrustedPortable("parent-one", "trusted-copy");
    expect(await host.plugins.instance("trusted-copy", fixture.manifest.id)).toMatchObject({
      enabled: true,
      contentSha256: installed.contentSha256,
      toolPermissions: [{ toolId, level: "ask" }],
      semanticLearning: { state: "complete", trainingSourceId: "semantic-parent-one" }
    });
    const trustedSelections = await host.plugins.portable("trusted-copy");
    expect(trustedSelections).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceBrainId: "trusted-copy", authority: "source-review" }),
      expect.objectContaining({ sourceBrainId: "parent-one", authority: "external-dormant",
        originAuthority: "source-review" })
    ]));

    await enablePackage(host.plugins, fixture, "parent-two");
    await host.plugins.adoptFullMergePortable("merged-child", ["parent-one", "parent-two"]);
    expect(await host.plugins.instance("merged-child", fixture.manifest.id)).toBeNull();
    expect((await host.plugins.portable("merged-child")).filter((selection) =>
      selection.authority === "merge-parent-dormant"
    )).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceBrainId: "parent-one", originAuthority: "source-review" }),
      expect.objectContaining({ sourceBrainId: "parent-two", originAuthority: "source-review" })
    ]));
  });

  it("rejects corrupt archives, undeclared files, and credential-like immutable constants", async () => {
    const source = await service("rejection");
    const fixture = packageFixture();
    const { installed } = await enablePackage(source.plugins, fixture, "brain-source");
    const directory = join(source.root, "portable");
    const exported = (await source.plugins.exportPortable("brain-source", "current", directory))!;
    const packagePath = exported.files.get(`packages/${installed.contentSha256}.json`)!;
    const original = await readFile(packagePath);
    const corrupted = JSON.parse(original.toString("utf8")) as OmniPluginPackage;
    corrupted.manifest.description = "Checksum-valid JSON with changed immutable content.";
    await writeFile(packagePath, JSON.stringify(corrupted));
    await expect(source.plugins.validatePortableImport(
      exported.files, exported.declaration, "brain-source"
    )).rejects.toThrow(/content hash/i);
    await writeFile(packagePath, original);
    const undeclared = new Map(exported.files);
    undeclared.set("unexpected.json", packagePath);
    await expect(source.plugins.validatePortableImport(
      undeclared, exported.declaration, "brain-source"
    )).rejects.toThrow(/undeclared files/i);

    const manifestPath = exported.files.get("manifest.json")!;
    const maliciousManifest = JSON.parse(await readFile(manifestPath, "utf8")) as {
      selections: Array<{ semanticLearning?: Record<string, unknown> }>;
    };
    maliciousManifest.selections[0]!.semanticLearning!.approvalToken = "must-not-cross-the-archive-boundary";
    const maliciousBytes = Buffer.from(JSON.stringify(maliciousManifest));
    await writeFile(manifestPath, maliciousBytes);
    await expect(source.plugins.validatePortableImport(exported.files, {
      ...exported.declaration,
      manifestSha256: createHash("sha256").update(maliciousBytes).digest("hex")
    }, "brain-source")).rejects.toThrow(/semantic receipt contains unsupported fields/i);

    const secret = packageFixture("example.secret");
    secret.manifest.tools[0]!.actions[0]!.delegate.arguments.constants = {
      apiKey: "must-stay-in-the-private-credential-store"
    };
    await expect(enablePackage(source.plugins, secret, "brain-source"))
      .rejects.toThrow(/credential-like constant/i);
  });
});
