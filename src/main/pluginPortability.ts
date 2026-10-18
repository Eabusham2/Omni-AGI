import { createHash } from "node:crypto";
import { mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import type {
  OmniPluginPackage,
  PluginPortableSelection,
  PluginPortabilityManifest
} from "../shared/plugins";
import { pluginToolId } from "../shared/plugins";
import { assertNoPluginCredentialConstants, canonicalPluginPackage } from "./pluginCatalog";
import type { PluginCatalog } from "./pluginCatalog";
import { pluginSemanticLearningExperience } from "./pluginSemanticLearning";
import { writeMutableFileIsolated } from "./mutableFileIsolation";

const SHA256 = /^[a-f0-9]{64}$/;
const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;

export interface PluginPortabilityDeclaration {
  formatVersion: 1;
  selectionCount: number;
  packageCount: number;
  manifestSha256: string;
}

export interface PluginPortabilityExport {
  declaration: PluginPortabilityDeclaration;
  files: Map<string, string>;
}

export interface ValidatedPluginPortabilityArchive {
  manifest: PluginPortabilityManifest;
  packages: Map<string, OmniPluginPackage>;
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function exactKeys(value: Record<string, unknown>, keys: readonly string[], label: string): void {
  if (Object.keys(value).sort().join("\0") !== [...keys].sort().join("\0")) {
    throw new Error(`${label} contains unsupported fields.`);
  }
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} is invalid.`);
  return value as Record<string, unknown>;
}

function validateSelection(
  value: unknown,
  sourceBrainId: string,
  packageValue: OmniPluginPackage
): PluginPortableSelection {
  const row = object(value, "Portable plugin selection");
  const allowed = ["sourceBrainId", "pluginId", "contentSha256", "reviewedAccess", "provenance", "authority", "recordedAt"];
  if (row.semanticLearning !== undefined) allowed.push("semanticLearning");
  if (row.originAuthority !== undefined) allowed.push("originAuthority");
  exactKeys(row, allowed, "Portable plugin selection");
  if (typeof row.sourceBrainId !== "string" || !SAFE_ID.test(row.sourceBrainId) ||
      row.pluginId !== packageValue.manifest.id ||
      typeof row.contentSha256 !== "string" || !SHA256.test(row.contentSha256) ||
      !["source-review", "external-dormant", "merge-parent-dormant"].includes(String(row.authority)) ||
      (row.originAuthority !== undefined && !["source-review", "external-dormant", "merge-parent-dormant"].includes(String(row.originAuthority))) ||
      typeof row.recordedAt !== "string" || !Number.isFinite(Date.parse(row.recordedAt))) {
    throw new Error("Portable plugin selection identity is invalid.");
  }
  if (row.authority === "source-review" && row.sourceBrainId !== sourceBrainId) {
    throw new Error("Portable active plugin selection belongs to another brain.");
  }
  const access = object(row.reviewedAccess, "Portable plugin reviewed access");
  exactKeys(access, ["pluginId", "contentSha256", "enabled", "capabilities", "toolPermissions", "revision", "updatedAt"], "Portable plugin reviewed access");
  if (access.pluginId !== row.pluginId || access.contentSha256 !== row.contentSha256 ||
      typeof access.enabled !== "boolean" || !Number.isSafeInteger(access.revision) || Number(access.revision) < 0 ||
      typeof access.updatedAt !== "string" || !Number.isFinite(Date.parse(access.updatedAt)) ||
      !Array.isArray(access.capabilities) || !Array.isArray(access.toolPermissions)) {
    throw new Error("Portable plugin reviewed access identity is invalid.");
  }
  const expectedCapabilities = new Map(packageValue.manifest.capabilities.map((item) => [item.toolId, item.actions]));
  if (access.capabilities.length !== expectedCapabilities.size) throw new Error("Portable plugin capabilities are incomplete.");
  for (const itemValue of access.capabilities) {
    const item = object(itemValue, "Portable plugin capability");
    exactKeys(item, ["toolId", "actions", "enabled"], "Portable plugin capability");
    const expected = expectedCapabilities.get(String(item.toolId));
    if (!expected || !Array.isArray(item.actions) || typeof item.enabled !== "boolean" ||
        JSON.stringify([...item.actions].sort()) !== JSON.stringify([...expected].sort())) {
      throw new Error("Portable plugin capability differs from its package.");
    }
    expectedCapabilities.delete(String(item.toolId));
  }
  if (expectedCapabilities.size) throw new Error("Portable plugin capabilities are incomplete.");
  const expectedTools = new Set(packageValue.manifest.tools.map((tool) => pluginToolId(packageValue.manifest.id, tool.id)));
  if (access.toolPermissions.length !== expectedTools.size) throw new Error("Portable plugin tool selections are incomplete.");
  for (const permissionValue of access.toolPermissions) {
    const permission = object(permissionValue, "Portable plugin tool selection");
    exactKeys(permission, ["toolId", "level"], "Portable plugin tool selection");
    if (!expectedTools.delete(String(permission.toolId)) || !["off", "ask", "auto", "full"].includes(String(permission.level))) {
      throw new Error("Portable plugin tool selection differs from its package.");
    }
  }
  const provenance = object(row.provenance, "Portable plugin provenance");
  const provenanceKeys = ["version", "installedAt", "sourceLabel", "author", "license"];
  if (provenance.parentSha256 !== undefined) provenanceKeys.push("parentSha256");
  if (provenance.homepage !== undefined) provenanceKeys.push("homepage");
  exactKeys(provenance, provenanceKeys, "Portable plugin provenance");
  if (provenance.version !== packageValue.manifest.version || provenance.author !== packageValue.manifest.author ||
      provenance.license !== packageValue.manifest.license || typeof provenance.installedAt !== "string" ||
      !Number.isFinite(Date.parse(provenance.installedAt)) || typeof provenance.sourceLabel !== "string" ||
      !provenance.sourceLabel.trim() ||
      (provenance.parentSha256 !== undefined && (typeof provenance.parentSha256 !== "string" || !SHA256.test(provenance.parentSha256))) ||
      provenance.homepage !== packageValue.manifest.homepage) {
    throw new Error("Portable plugin provenance differs from its immutable package.");
  }
  if (row.semanticLearning !== undefined) {
    const semantic = object(row.semanticLearning, "Portable plugin semantic receipt");
    const semanticKeys = ["brainId", "pluginId", "contentSha256", "experienceSha256", "state", "attempt", "updatedAt"];
    if (semantic.trainingSourceId !== undefined) semanticKeys.push("trainingSourceId");
    if (semantic.error !== undefined) semanticKeys.push("error");
    exactKeys(semantic, semanticKeys, "Portable plugin semantic receipt");
    if (semantic.brainId !== row.sourceBrainId || semantic.pluginId !== row.pluginId ||
        semantic.contentSha256 !== row.contentSha256 ||
        semantic.experienceSha256 !== pluginSemanticLearningExperience(packageValue, String(row.contentSha256)).sha256 ||
        !["pending", "running", "complete"].includes(String(semantic.state)) ||
        !Number.isSafeInteger(semantic.attempt) || Number(semantic.attempt) < 0 ||
        typeof semantic.updatedAt !== "string" || !Number.isFinite(Date.parse(semantic.updatedAt)) ||
        (semantic.trainingSourceId !== undefined &&
          (typeof semantic.trainingSourceId !== "string" || !SAFE_ID.test(semantic.trainingSourceId))) ||
        (semantic.state === "complete" && semantic.trainingSourceId === undefined) ||
        (semantic.state !== "complete" && semantic.trainingSourceId !== undefined) ||
        (semantic.error !== undefined && (typeof semantic.error !== "string" || semantic.error.length > 2_000))) {
      throw new Error("Portable plugin semantic receipt is invalid.");
    }
  }
  return structuredClone(value as PluginPortableSelection);
}

export async function exportPluginPortability(
  catalog: PluginCatalog,
  brainId: string,
  selections: PluginPortableSelection[],
  destination: string
): Promise<PluginPortabilityExport | undefined> {
  if (!SAFE_ID.test(brainId)) throw new Error("Plugin portability brain identity is invalid.");
  if (!selections.length) return undefined;
  await mkdir(join(destination, "packages"), { recursive: true });
  const uniqueSelections = new Map<string, PluginPortableSelection>();
  for (const selection of selections) {
    const key = `${selection.sourceBrainId}\0${selection.pluginId}\0${selection.contentSha256}`;
    const previous = uniqueSelections.get(key);
    if (!previous || previous.authority !== "source-review") uniqueSelections.set(key, selection);
  }
  const packageRows: PluginPortabilityManifest["packages"] = [];
  const packageFiles = new Map<string, string>();
  const emittedPackages = new Set<string>();
  for (const key of [...uniqueSelections.keys()].sort()) {
    const selection = uniqueSelections.get(key)!;
    const packageValue = await catalog.package(selection.pluginId, selection.contentSha256);
    for (const tool of packageValue.manifest.tools) {
      for (const action of tool.actions) assertNoPluginCredentialConstants(
        action.delegate.arguments.constants, `${selection.pluginId}.${tool.id}.${action.id}`
      );
    }
    validateSelection(selection, brainId, packageValue);
    const prepared = canonicalPluginPackage(packageValue);
    if (prepared.sha256 !== selection.contentSha256) throw new Error("Portable plugin package hash changed.");
    if (emittedPackages.has(prepared.sha256)) continue;
    emittedPackages.add(prepared.sha256);
    const relative = `packages/${prepared.sha256}.json`;
    const path = join(destination, "packages", `${prepared.sha256}.json`);
    await writeMutableFileIsolated(path, Buffer.from(prepared.json, "utf8"));
    packageFiles.set(relative, path);
    packageRows.push({ pluginId: selection.pluginId, contentSha256: prepared.sha256, path: relative });
  }
  const manifest: PluginPortabilityManifest = {
    format: "omni-plugin-portability",
    formatVersion: 1,
    sourceBrainId: brainId,
    createdAt: new Date().toISOString(),
    selections: [...uniqueSelections.values()].sort((left, right) =>
      left.pluginId < right.pluginId ? -1 : left.pluginId > right.pluginId ? 1 :
        left.contentSha256 < right.contentSha256 ? -1 : left.contentSha256 > right.contentSha256 ? 1 :
          left.sourceBrainId < right.sourceBrainId ? -1 : left.sourceBrainId > right.sourceBrainId ? 1 : 0),
    packages: packageRows
  };
  const manifestPath = join(destination, "manifest.json");
  const manifestBytes = Buffer.from(JSON.stringify(manifest, null, 2), "utf8");
  await writeMutableFileIsolated(manifestPath, manifestBytes);
  return {
    declaration: { formatVersion: 1, selectionCount: manifest.selections.length,
      packageCount: manifest.packages.length, manifestSha256: sha256(manifestBytes) },
    files: new Map([["manifest.json", manifestPath], ...packageFiles])
  };
}

export async function validatePluginPortabilityArchive(
  files: Map<string, string>,
  declaration: PluginPortabilityDeclaration,
  expectedSourceBrainId: string
): Promise<ValidatedPluginPortabilityArchive> {
  if (!SAFE_ID.test(expectedSourceBrainId) || declaration.formatVersion !== 1 ||
      !Number.isSafeInteger(declaration.selectionCount) || declaration.selectionCount < 1 ||
      !Number.isSafeInteger(declaration.packageCount) || declaration.packageCount < 1 ||
      !SHA256.test(declaration.manifestSha256)) throw new Error("Plugin portability declaration is invalid.");
  const manifestPath = files.get("manifest.json");
  if (!manifestPath) throw new Error("Plugin portability manifest is missing.");
  const manifestBytes = await readFile(manifestPath);
  if (sha256(manifestBytes) !== declaration.manifestSha256) throw new Error("Plugin portability manifest checksum failed.");
  const raw = object(JSON.parse(manifestBytes.toString("utf8")), "Plugin portability manifest");
  exactKeys(raw, ["format", "formatVersion", "sourceBrainId", "createdAt", "selections", "packages"], "Plugin portability manifest");
  if (raw.format !== "omni-plugin-portability" || raw.formatVersion !== 1 || raw.sourceBrainId !== expectedSourceBrainId ||
      typeof raw.createdAt !== "string" || !Number.isFinite(Date.parse(raw.createdAt)) ||
      !Array.isArray(raw.selections) || !Array.isArray(raw.packages) ||
      raw.selections.length !== declaration.selectionCount || raw.packages.length !== declaration.packageCount) {
    throw new Error("Plugin portability manifest identity is invalid.");
  }
  const packages = new Map<string, OmniPluginPackage>();
  for (const packageValue of raw.packages) {
    const row = object(packageValue, "Portable plugin package declaration");
    exactKeys(row, ["pluginId", "contentSha256", "path"], "Portable plugin package declaration");
    if (typeof row.pluginId !== "string" || typeof row.contentSha256 !== "string" || !SHA256.test(row.contentSha256) ||
        row.path !== `packages/${row.contentSha256}.json` || packages.has(row.contentSha256)) {
      throw new Error("Portable plugin package declaration is invalid.");
    }
    const path = files.get(String(row.path));
    if (!path) throw new Error("Portable plugin package file is missing.");
    const prepared = canonicalPluginPackage(JSON.parse(await readFile(path, "utf8")));
    if (prepared.sha256 !== row.contentSha256 || prepared.package.manifest.id !== row.pluginId) {
      throw new Error("Portable plugin package content hash is invalid.");
    }
    for (const tool of prepared.package.manifest.tools) {
      for (const action of tool.actions) assertNoPluginCredentialConstants(
        action.delegate.arguments.constants, `${row.pluginId}.${tool.id}.${action.id}`
      );
    }
    packages.set(prepared.sha256, prepared.package);
  }
  const expectedFiles = new Set(["manifest.json", ...[...packages.keys()].map((hash) => `packages/${hash}.json`)]);
  if (files.size !== expectedFiles.size || [...files.keys()].some((path) => !expectedFiles.has(path))) {
    throw new Error("Plugin portability archive contains undeclared files.");
  }
  const selections = raw.selections.map((selectionValue) => {
    const identity = object(selectionValue, "Portable plugin selection");
    const packageValue = packages.get(String(identity.contentSha256));
    if (!packageValue) throw new Error("Portable plugin selection references a missing package.");
    return validateSelection(selectionValue, expectedSourceBrainId, packageValue);
  });
  const keys = selections.map((selection) => `${selection.sourceBrainId}\0${selection.pluginId}\0${selection.contentSha256}`);
  if (new Set(keys).size !== keys.length) throw new Error("Plugin portability selections are duplicated.");
  const referencedPackages = new Set(selections.map((selection) => selection.contentSha256));
  if (referencedPackages.size !== packages.size ||
      [...packages.keys()].some((contentSha256) => !referencedPackages.has(contentSha256))) {
    throw new Error("Plugin portability archive contains an unreferenced package.");
  }
  return { manifest: { ...(raw as unknown as PluginPortabilityManifest), selections }, packages };
}
