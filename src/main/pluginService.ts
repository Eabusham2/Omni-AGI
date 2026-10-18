import { createHash } from "node:crypto";
import { basename } from "node:path";
import {
  PLUGIN_TRAINING_PARSER_CAPABILITY,
  PLUGIN_RUNTIME_PARSER_ABI,
  pluginToolId,
  type OmniPluginPackage,
  type PluginCatalogEntry,
  type PluginInstallResult,
  type PluginInstanceAccess,
  type PluginInstanceAccessRequest,
  type PluginEvent,
  type PluginPortableSelection,
  type PluginResolvedInvocation,
  type PluginSemanticLearningRecord,
  type PluginTrainingFormatLease,
  type PluginTrainingFormatResolver,
  type PluginRuntimeStreamParserRunner
} from "../shared/plugins";
import { validPluginTrainingFormatSelection, type PluginTrainingFormatSelection, type PluginTrainingFormatSourceInput } from "../shared/plugins";
import { normalizeSourceMediaType } from "../shared/sourceMediaType";
import type {
  PluginTrainingFormatBinding,
  ToolInvocation,
  ToolPermissionRecord
} from "../shared/types";
import type { BrainService, ExternalToolSchema } from "./brainService";
import { PluginCatalog } from "./pluginCatalog";
import { pluginSemanticLearningExperience } from "./pluginSemanticLearning";
import {
  exportPluginPortability,
  validatePluginPortabilityArchive,
  type PluginPortabilityDeclaration,
  type PluginPortabilityExport,
  type ValidatedPluginPortabilityArchive
} from "./pluginPortability";
import type { BrainExportMode } from "../shared/types";
import type { CustomToolResolver } from "./toolExecutor";
import {
  parsePluginPanelUrl,
  pluginPanelSecurityHeaders,
  staticPluginPanelDocument
} from "../shared/pluginPanels";
import { runPluginWasmParser } from "./pluginWasmParser";
import { runtimeModuleSha256 } from "./pluginRuntimeValidation";

function digest(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function stable(value: unknown): string {
  const compare = (left: string, right: string): number => left < right ? -1 : left > right ? 1 : 0;
  const normalize = (item: unknown): unknown => {
    if (Array.isArray(item)) return item.map(normalize);
    if (item && typeof item === "object") {
      return Object.fromEntries(Object.entries(item as Record<string, unknown>)
        .sort(([left], [right]) => compare(left, right))
        .map(([key, nested]) => [key, normalize(nested)]));
    }
    return item;
  };
  return JSON.stringify(normalize(value));
}

interface RegisteredTool {
  entry: PluginCatalogEntry;
  package: OmniPluginPackage;
  tool: OmniPluginPackage["manifest"]["tools"][number];
}

type PluginBrainAuthority = Pick<BrainService,
  "registerExternalToolSchemas" |
  "unregisterExternalToolSchemas" |
  "listToolPermissions" |
  "setToolPermission"
> & Partial<Pick<BrainService, "learnLoadedStructuredExperience" | "requireInstanceLoaded">>;

/**
 * Installation catalog plus the declarative custom-tool resolver.
 *
 * Package JavaScript never runs in Studio or main and never receives
 * main-process objects. An optional UI script runs only in the opaque panel;
 * an import-free parser runs in its killable Worker, while a capability-aware
 * parser uses a dedicated exact-format session in the isolated feature runtime.
 * A callable action resolves to one existing typed ToolExecutor operation,
 * whose own policy remains authoritative in addition to plugin policy.
 */
export class PluginService implements CustomToolResolver, PluginTrainingFormatResolver {
  private readonly registered = new Map<string, RegisteredTool>();
  private readonly registeredSchemas = new Set<string>();
  private readonly accessQueues = new Map<string, Promise<void>>();
  private readonly activeTrainingFormats = new Set<{
    brainId: string;
    pluginId: string;
    controller: AbortController;
  }>();
  private readonly semanticLearning = new Map<string, {
    controller: AbortController;
    promise: Promise<void>;
  }>();
  private readonly semanticReschedule = new Set<string>();
  private readonly globalMutationEpoch = new Map<string, number>();
  private readonly instanceMutationEpoch = new Map<string, number>();
  private readonly globalMutations = new Map<string, number>();
  private readonly instanceMutations = new Map<string, number>();
  private recoveredSemanticLearning = false;
  private readonly listeners = new Set<(event: PluginEvent) => void>();
  private runtimeStreamParser?: PluginRuntimeStreamParserRunner;

  constructor(
    readonly catalog: PluginCatalog,
    private readonly brain: PluginBrainAuthority
  ) {}

  setRuntimeStreamParser(runner: PluginRuntimeStreamParserRunner): void {
    this.runtimeStreamParser = runner;
  }

  onEvent(listener: (event: PluginEvent) => void): () => void {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }

  private publish(event: PluginEvent): void {
    for (const listener of this.listeners) {
      try { listener(structuredClone(event)); }
      catch (error) { console.warn("Plugin event listener failed", error); }
    }
  }

  private eventRevision(fallback: number): number {
    try { return this.catalog.revision(); }
    catch { return fallback; }
  }

  private publishCatalog(entry: PluginCatalogEntry): void {
    this.publish({ type: "catalog", pluginId: entry.pluginId,
      contentSha256: entry.activeContentSha256, catalogRevision: this.eventRevision(entry.revision),
      at: entry.updatedAt });
  }

  private publishInstance(access: PluginInstanceAccess): void {
    this.publish({ type: "instance", pluginId: access.pluginId,
      contentSha256: access.contentSha256, catalogRevision: this.eventRevision(access.revision),
      brainId: access.brainId, instanceRevision: access.revision, at: access.updatedAt });
  }

  private publishSemantic(record: PluginSemanticLearningRecord): void {
    this.publish({ type: "semantic-learning", pluginId: record.pluginId,
      contentSha256: record.contentSha256, catalogRevision: this.eventRevision(0),
      brainId: record.brainId, semanticState: record.state, at: record.updatedAt });
  }

  async initialize(): Promise<void> {
    await this.catalog.initialize();
    if (!this.recoveredSemanticLearning) {
      await this.catalog.recoverSemanticLearning();
      for (const access of await this.catalog.instances()) {
        if (!access.enabled) continue;
        const packageValue = await this.catalog.package(access.pluginId, access.contentSha256);
      const semantic = pluginSemanticLearningExperience(packageValue, access.contentSha256);
        await this.catalog.ensureSemanticLearning(
          access.brainId, access.pluginId, access.contentSha256, semantic.sha256
        );
      }
      this.recoveredSemanticLearning = true;
    }
    await this.syncSchemas();
  }

  private abortPluginWork(pluginId: string, brainId?: string): void {
    for (const lease of this.activeTrainingFormats) {
      if (lease.pluginId === pluginId && (brainId === undefined || lease.brainId === brainId)) {
        lease.controller.abort(new Error(
          "Plugin training-format access changed; learning paused at its next committed boundary."
        ));
      }
    }
    for (const [key, learning] of this.semanticLearning) {
      const [ownerBrain, ownerPlugin] = key.split("\0");
      if (ownerPlugin === pluginId && (brainId === undefined || ownerBrain === brainId)) {
        learning.controller.abort(new Error("Plugin access changed before semantic learning completed."));
      }
    }
  }

  private beginPluginMutation(pluginId: string, brainId?: string): () => void {
    const epochs = brainId === undefined ? this.globalMutationEpoch : this.instanceMutationEpoch;
    const active = brainId === undefined ? this.globalMutations : this.instanceMutations;
    const key = brainId === undefined ? pluginId : `${brainId}\0${pluginId}`;
    epochs.set(key, (epochs.get(key) ?? 0) + 1);
    active.set(key, (active.get(key) ?? 0) + 1);
    this.abortPluginWork(pluginId, brainId);
    let released = false;
    return () => {
      if (released) return;
      released = true;
      const remaining = (active.get(key) ?? 1) - 1;
      if (remaining > 0) active.set(key, remaining);
      else active.delete(key);
    };
  }

  private trainingMutationSnapshot(brainId: string, pluginId: string): {
    global: number;
    instance: number;
  } {
    const instanceKey = `${brainId}\0${pluginId}`;
    if ((this.globalMutations.get(pluginId) ?? 0) > 0 ||
        (this.instanceMutations.get(instanceKey) ?? 0) > 0) {
      throw new Error("Plugin access is changing; create or resume the dataset after that review completes.");
    }
    return {
      global: this.globalMutationEpoch.get(pluginId) ?? 0,
      instance: this.instanceMutationEpoch.get(instanceKey) ?? 0
    };
  }

  private requireTrainingMutationSnapshot(
    brainId: string,
    pluginId: string,
    snapshot: { global: number; instance: number }
  ): void {
    const current = this.trainingMutationSnapshot(brainId, pluginId);
    if (current.global !== snapshot.global || current.instance !== snapshot.instance) {
      throw new Error("Plugin access changed while the training-format lease was being acquired.");
    }
  }

  private async syncSchemas(): Promise<void> {
    const next = new Map<string, RegisteredTool>();
    for (const entry of await this.catalog.list()) {
      if (!entry.enabled) continue;
      const packageValue = await this.catalog.package(entry.pluginId, entry.activeContentSha256);
      for (const tool of packageValue.manifest.tools) {
        const id = pluginToolId(entry.pluginId, tool.id);
        if (next.has(id)) throw new Error(`Duplicate installed plugin tool ${id}.`);
        next.set(id, { entry, package: packageValue, tool });
      }
    }
    const removed = [...this.registeredSchemas].filter((id) => !next.has(id));
    if (removed.length) this.brain.unregisterExternalToolSchemas(removed);
    const schemas: ExternalToolSchema[] = [...next].map(([id, value]) => ({
      id,
      actions: value.tool.actions.map((action) => action.id),
      ...(value.tool.actions.length === 1 && value.tool.actions[0]?.inputSchema
        ? { inputSchema: structuredClone(value.tool.actions[0].inputSchema) }
        : {})
    }));
    if (schemas.length) this.brain.registerExternalToolSchemas(schemas);
    this.registered.clear();
    for (const [id, value] of next) this.registered.set(id, value);
    this.registeredSchemas.clear();
    for (const id of next.keys()) this.registeredSchemas.add(id);
  }

  async list(): Promise<PluginCatalogEntry[]> {
    return this.catalog.list();
  }

  package(pluginId: string, contentSha256?: string): Promise<OmniPluginPackage> {
    return this.catalog.package(pluginId, contentSha256);
  }

  async installPackage(value: unknown, sourceLabel?: string): Promise<PluginInstallResult> {
    const installed = await this.catalog.install(value, sourceLabel);
    await this.syncSchemas();
    if (installed.installed) this.publishCatalog(installed.entry);
    return installed;
  }

  async installFile(path: string): Promise<PluginInstallResult> {
    const installed = await this.catalog.importFile(path);
    await this.syncSchemas();
    if (installed.installed) this.publishCatalog(installed.entry);
    return installed;
  }

  async edit(pluginId: string, baseContentSha256: string, value: unknown): Promise<PluginInstallResult> {
    const installed = await this.catalog.edit(pluginId, baseContentSha256, value);
    await this.syncSchemas();
    if (installed.installed) this.publishCatalog(installed.entry);
    return installed;
  }

  async activate(pluginId: string, contentSha256: string): Promise<PluginCatalogEntry> {
    const release = this.beginPluginMutation(pluginId);
    try {
      const entry = await this.catalog.activate(pluginId, contentSha256);
      await this.syncSchemas();
      this.publishCatalog(entry);
      return entry;
    } finally { release(); }
  }

  async setEnabled(
    pluginId: string,
    enabled: boolean,
    expectedContentSha256: string,
    expectedRevision: number
  ): Promise<PluginCatalogEntry> {
    const release = this.beginPluginMutation(pluginId);
    let entry: PluginCatalogEntry | undefined;
    try {
      const packageValue = await this.catalog.package(pluginId, expectedContentSha256);
      const semantic = pluginSemanticLearningExperience(packageValue, expectedContentSha256);
      const committed = await this.catalog.setEnabled(
        pluginId, enabled, expectedContentSha256, expectedRevision,
        enabled ? semantic.sha256 : undefined
      );
      entry = committed;
      if (enabled) {
        const accesses = (await this.catalog.instances()).filter((access) =>
          access.pluginId === committed.pluginId && access.enabled &&
          access.contentSha256 === committed.activeContentSha256
        );
        for (const access of accesses) {
          await this.queueSemanticLearning(access.brainId, committed.pluginId, committed.activeContentSha256);
        }
      }
      await this.syncSchemas();
      this.publishCatalog(committed);
      return committed;
    } catch (error) {
      const rolledBack = enabled && entry
        ? await this.catalog.setEnabled(
            pluginId, false, entry.activeContentSha256, entry.revision
          ).catch(() => undefined)
        : undefined;
      await this.syncSchemas().catch(() => undefined);
      if (rolledBack) this.publishCatalog(rolledBack);
      throw error;
    } finally { release(); }
  }

  exportFile(
    pluginId: string,
    destination: string,
    contentSha256?: string,
    replaceReviewed = false
  ): Promise<string> {
    return this.catalog.exportFile(pluginId, destination, contentSha256, replaceReviewed);
  }

  async instance(brainId: string, pluginId: string): Promise<PluginInstanceAccess | null> {
    const access = await this.catalog.instance(brainId, pluginId);
    if (!access) return null;
    const [permissions, semanticLearning] = await Promise.all([
      this.brain.listToolPermissions(brainId),
      this.catalog.semanticLearning(brainId, pluginId, access.contentSha256)
    ]);
    const current = new Map(permissions.map((permission) => [permission.toolId, permission.level]));
    return {
      ...access,
      ...(semanticLearning ? { semanticLearning } : {}),
      toolPermissions: access.toolPermissions.map((permission) => ({
        ...permission,
        level: current.get(permission.toolId) ?? "off"
      }))
    };
  }

  private async portableSelection(
    sourceBrainId: string,
    access: PluginInstanceAccess
  ): Promise<PluginPortableSelection> {
    const entry = (await this.catalog.list()).find((candidate) => candidate.pluginId === access.pluginId);
    if (!entry) throw new Error("Portable plugin selection references an uninstalled package.");
    const packageValue = await this.catalog.package(access.pluginId, access.contentSha256);
    const version = entry.versions.find((candidate) => candidate.contentSha256 === access.contentSha256);
    if (!version) throw new Error("Portable plugin selection has no installed provenance record.");
    const { brainId: _brainId, semanticLearning, ...reviewedAccess } = access;
    return {
      sourceBrainId,
      pluginId: access.pluginId,
      contentSha256: access.contentSha256,
      reviewedAccess,
      ...(semanticLearning ? { semanticLearning: { ...semanticLearning, brainId: sourceBrainId } } : {}),
      provenance: {
        version: version.version,
        installedAt: version.installedAt,
        sourceLabel: version.sourceLabel,
        ...(version.parentSha256 ? { parentSha256: version.parentSha256 } : {}),
        author: packageValue.manifest.author,
        license: packageValue.manifest.license,
        ...(packageValue.manifest.homepage ? { homepage: packageValue.manifest.homepage } : {})
      },
      authority: "source-review",
      recordedAt: access.updatedAt
    };
  }

  async portable(brainId: string): Promise<PluginPortableSelection[]> {
    const result = new Map<string, PluginPortableSelection>();
    for (const access of await this.catalog.instances(brainId)) {
      const current = await this.instance(brainId, access.pluginId);
      if (!current) continue;
      const selection = await this.portableSelection(brainId, current);
      result.set(`${selection.sourceBrainId}\0${selection.pluginId}\0${selection.contentSha256}`, selection);
    }
    for (const selection of await this.catalog.dormantSelections(brainId)) {
      const key = `${selection.sourceBrainId}\0${selection.pluginId}\0${selection.contentSha256}`;
      if (!result.has(key)) result.set(key, selection);
    }
    return [...result.values()].sort((left, right) =>
      left.pluginId < right.pluginId ? -1 : left.pluginId > right.pluginId ? 1 :
        left.contentSha256 < right.contentSha256 ? -1 : left.contentSha256 > right.contentSha256 ? 1 :
          left.sourceBrainId < right.sourceBrainId ? -1 : left.sourceBrainId > right.sourceBrainId ? 1 : 0
    );
  }

  async exportPortable(
    brainId: string,
    mode: BrainExportMode,
    destination: string
  ): Promise<PluginPortabilityExport | undefined> {
    if (mode === "origin") return undefined;
    return exportPluginPortability(this.catalog, brainId, await this.portable(brainId), destination);
  }

  validatePortableImport(
    files: Map<string, string>,
    declaration: PluginPortabilityDeclaration,
    sourceBrainId: string
  ): Promise<ValidatedPluginPortabilityArchive> {
    return validatePluginPortabilityArchive(files, declaration, sourceBrainId);
  }

  async adoptExternalPortable(
    targetBrainId: string,
    archive: ValidatedPluginPortabilityArchive
  ): Promise<void> {
    const installedEntries: PluginCatalogEntry[] = [];
    for (const declaration of archive.manifest.packages) {
      const packageValue = archive.packages.get(declaration.contentSha256)!;
      const installed = await this.catalog.install(
        packageValue,
        `Imported .omni dependency from ${archive.manifest.sourceBrainId}`
      );
      if (installed.installed) installedEntries.push(installed.entry);
    }
    for (const selection of archive.manifest.selections) {
      await this.catalog.storeDormantSelection(targetBrainId, {
        ...selection,
        authority: "external-dormant",
        originAuthority: selection.originAuthority ?? selection.authority
      });
    }
    await this.syncSchemas();
    for (const entry of installedEntries) this.publishCatalog(entry);
  }

  async cloneTrustedPortable(sourceBrainId: string, targetBrainId: string): Promise<void> {
    const selections = await this.portable(sourceBrainId);
    for (const selection of selections) {
      await this.catalog.storeDormantSelection(targetBrainId,
        selection.authority === "source-review" ? {
          ...selection,
          authority: "external-dormant",
          originAuthority: selection.originAuthority ?? selection.authority
        } : selection);
      if (selection.authority !== "source-review") {
        continue;
      }
      const active = (await this.catalog.list()).find((entry) => entry.pluginId === selection.pluginId);
      if (!active || active.activeContentSha256 !== selection.contentSha256) {
        continue;
      }
      // The fork already owns an exact copy-on-write neural state. Publish the
      // matching semantic receipt before enabling its copied selection so a
      // completed declaration is not queued and learned a second time.
      await this.catalog.copySemanticLearning(
        sourceBrainId, targetBrainId, selection.pluginId, selection.contentSha256
      );
      await this.setInstanceAccess({
        brainId: targetBrainId,
        pluginId: selection.pluginId,
        contentSha256: selection.contentSha256,
        enabled: selection.reviewedAccess.enabled,
        capabilities: structuredClone(selection.reviewedAccess.capabilities),
        toolPermissions: structuredClone(selection.reviewedAccess.toolPermissions),
        requestId: `plugin-fork-${digest(`${sourceBrainId}\0${targetBrainId}\0${selection.pluginId}`).slice(0, 64)}`
      });
      const copiedSemantic = await this.catalog.semanticLearning(
        targetBrainId, selection.pluginId, selection.contentSha256
      );
      if (copiedSemantic) this.publishSemantic(copiedSemantic);
    }
  }

  async adoptFullMergePortable(targetBrainId: string, parentBrainIds: readonly string[]): Promise<void> {
    for (const parentBrainId of parentBrainIds) {
      for (const selection of await this.portable(parentBrainId)) {
        await this.catalog.storeDormantSelection(targetBrainId, {
          ...selection,
          authority: "merge-parent-dormant",
          originAuthority: selection.originAuthority ?? selection.authority
        });
      }
    }
  }

  removePortableBrain(brainId: string): Promise<void> {
    for (const lease of this.activeTrainingFormats) {
      if (lease.brainId === brainId) lease.controller.abort(new Error("Plugin brain state was removed."));
    }
    for (const [key, learning] of this.semanticLearning) {
      if (key.startsWith(`${brainId}\0`)) learning.controller.abort(new Error("Plugin brain state was removed."));
    }
    return this.catalog.removeBrainState(brainId);
  }

  async resolveTrainingFormat(
    brainId: string,
    path: string,
    mediaType?: string,
    selection?: PluginTrainingFormatSelection
  ): Promise<PluginTrainingFormatBinding | undefined> {
    if (selection !== undefined && !validPluginTrainingFormatSelection(selection)) throw new Error("Explicit plugin parser selection is invalid.");
    const fileName = basename(path).toLowerCase();
    const normalizedMedia = normalizeSourceMediaType(mediaType);
    if (mediaType !== undefined && normalizedMedia === undefined) throw new Error("Observed plugin source MIME is invalid.");
    const matches: PluginTrainingFormatBinding[] = [];
    for (const entry of await this.catalog.list()) {
      if (selection && (entry.pluginId !== selection.pluginId || entry.activeContentSha256 !== selection.contentSha256)) continue;
      if (!entry.enabled) continue;
      const access = await this.catalog.instance(brainId, entry.pluginId);
      if (!access?.enabled || access.contentSha256 !== entry.activeContentSha256) continue;
      const packageValue = await this.catalog.package(entry.pluginId, entry.activeContentSha256);
      for (const format of packageValue.manifest.trainingFormats) {
        if (selection && format.id !== selection.formatId) continue;
        if (format.parser === "declaration-only") continue;
        const extensionMatch = format.extensions.some((extension) => fileName.endsWith(`.${extension}`));
        const mediaMatch = normalizedMedia !== undefined && format.mediaTypes.includes(normalizedMedia);
        // Preserve reviewed unique custom-extension behavior. A MIME-only
        // declaration never opts an untrusted HTTP/local source into code.
        if (selection ? !extensionMatch && !mediaMatch : !extensionMatch) continue;
        const parser = format.parser;
        const parserFormat = parser.kind === "builtin-stream" ? parser.format :
          parser.output === "jsonl" ? "jsonl" : "archive";
        let executable: PluginTrainingFormatBinding["executable"];
        if (parser.kind === "wasm-stream") {
          const capability = access.capabilities.find((item) => item.toolId === PLUGIN_TRAINING_PARSER_CAPABILITY);
          if (!capability?.enabled || !capability.actions.includes(format.id)) {
            throw new Error(`Executable training format ${format.id} requires its explicitly enabled parser capability.`);
          }
          const module = packageValue.manifest.trainingModules?.find((item) => item.id === parser.moduleId);
          if (!module) throw new Error(`Executable training format ${format.id} lost its immutable WASM module.`);
          executable = { kind: "wasm-stream", moduleId: module.id, moduleSha256: module.wasmSha256,
            abi: module.abi, outputFormat: "jsonl" };
        } else if (parser.kind === "runtime-stream") {
          const capability = access.capabilities.find((item) => item.toolId === PLUGIN_TRAINING_PARSER_CAPABILITY);
          if (!capability?.enabled || !capability.actions.includes(format.id)) {
            throw new Error(`Runtime training format ${format.id} requires its explicitly enabled parser capability.`);
          }
          const module = packageValue.manifest.runtimeModules?.find((item) => item.id === parser.moduleId);
          if (!module) throw new Error(`Runtime training format ${format.id} lost its immutable runtime module.`);
          executable = { kind: "runtime-stream", moduleId: module.id, moduleSha256: runtimeModuleSha256(module),
            abi: PLUGIN_RUNTIME_PARSER_ABI, outputFormat: parser.output === "jsonl" ? "jsonl" : "media-assets",
            ...(parser.output === "jsonl" ? {} : { outputMediaTypes: [...parser.output.mediaTypes] }) };
        }
        matches.push({
          pluginId: entry.pluginId,
          contentSha256: entry.activeContentSha256,
          formatId: format.id,
          parserFormat,
          catalogRevision: entry.revision,
          instanceRevision: access.revision,
          ...(normalizedMedia ? { inputMediaType: normalizedMedia } : {}),
          ...(executable ? { executable } : {})
        });
      }
    }
    if (matches.length > 1) {
      throw new Error(
        `Multiple enabled plugin formats match ${basename(path)}; disable all but one reviewed format before creating the dataset manifest.`
      );
    }
    if (selection && matches.length !== 1) throw new Error("The explicitly selected immutable plugin parser is unavailable or does not match this source extension/MIME.");
    return matches[0];
  }

  async validateTrainingFormat(
    brainId: string,
    binding: PluginTrainingFormatBinding,
    source?: PluginTrainingFormatSourceInput
  ): Promise<void> {
    const entry = (await this.catalog.list()).find((candidate) => candidate.pluginId === binding.pluginId);
    if (!entry?.enabled || entry.activeContentSha256 !== binding.contentSha256 ||
        entry.revision !== binding.catalogRevision) {
      throw new Error("The plugin training-format version or global access changed; this source remains paused.");
    }
    const access = await this.catalog.instance(brainId, binding.pluginId);
    if (!access?.enabled || access.contentSha256 !== binding.contentSha256 ||
        access.revision !== binding.instanceRevision) {
      throw new Error("The plugin training-format instance access changed; this source remains paused.");
    }
    const packageValue = await this.catalog.package(binding.pluginId, binding.contentSha256);
    const format = packageValue.manifest.trainingFormats.find((candidate) => candidate.id === binding.formatId);
    if (!format || format.parser === "declaration-only") {
      throw new Error("The plugin training-format binding differs from its immutable package.");
    }
    if (binding.inputMediaType !== undefined && normalizeSourceMediaType(binding.inputMediaType) !== binding.inputMediaType) throw new Error("Saved parser MIME binding is invalid.");
    if (binding.inputMediaType !== undefined && !source) throw new Error("MIME-bound parser needs its actual selected source evidence at start/resume.");
    if (source) {
      const mediaType = normalizeSourceMediaType(source.mediaType);
      const extensionMatch = format.extensions.some(extension => basename(source.path).toLowerCase().endsWith(`.${extension}`));
      const mediaMatch = mediaType !== undefined && format.mediaTypes.includes(mediaType);
      if ((source.mediaType !== undefined && mediaType === undefined) ||
          mediaType !== binding.inputMediaType || (!extensionMatch && !mediaMatch)) {
        throw new Error("The parser source extension/MIME differs from its exact saved binding or immutable declaration.");
      }
    }
    if (format.parser.kind === "builtin-stream") {
      if (format.parser.format !== binding.parserFormat || binding.executable !== undefined) {
        throw new Error("The plugin built-in training-format binding differs from its immutable package.");
      }
      return;
    }
    const parser = format.parser;
    const capability = access.capabilities.find((item) => item.toolId === PLUGIN_TRAINING_PARSER_CAPABILITY);
    const expectedParserFormat = parser.output === "jsonl" ? "jsonl" : "archive";
    const expectedOutputFormat = parser.output === "jsonl" ? "jsonl" : "media-assets";
    const expectedMediaTypes = parser.output === "jsonl" ? undefined : parser.output.mediaTypes;
    const boundMediaTypes = binding.executable?.kind === "runtime-stream" ? binding.executable.outputMediaTypes : undefined;
    if (!capability?.enabled || !capability.actions.includes(format.id) || binding.parserFormat !== expectedParserFormat ||
        binding.executable?.moduleId !== parser.moduleId || binding.executable.outputFormat !== expectedOutputFormat ||
        stable(boundMediaTypes) !== stable(expectedMediaTypes)) {
      throw new Error("The executable plugin training-format binding differs from its immutable reviewed capability.");
    }
    if (parser.kind === "wasm-stream") {
      const module = packageValue.manifest.trainingModules?.find((item) => item.id === parser.moduleId);
      if (!module || binding.executable.kind !== "wasm-stream" || binding.executable.moduleSha256 !== module.wasmSha256 ||
          binding.executable.abi !== module.abi) throw new Error("The WASM training-format binding differs from its immutable reviewed module.");
    } else {
      const module = packageValue.manifest.runtimeModules?.find((item) => item.id === parser.moduleId);
      if (!module || binding.executable.kind !== "runtime-stream" || binding.executable.moduleSha256 !== runtimeModuleSha256(module) ||
          binding.executable.abi !== PLUGIN_RUNTIME_PARSER_ABI) throw new Error("The runtime training-format binding differs from its immutable reviewed module.");
    }
  }

  async acquireTrainingFormat(
    brainId: string,
    binding: PluginTrainingFormatBinding,
    source?: PluginTrainingFormatSourceInput
  ): Promise<PluginTrainingFormatLease> {
    const mutation = this.trainingMutationSnapshot(brainId, binding.pluginId);
    const lease = { brainId, pluginId: binding.pluginId, controller: new AbortController() };
    this.activeTrainingFormats.add(lease);
    try {
      await this.validateTrainingFormat(brainId, binding, source);
      this.requireTrainingMutationSnapshot(brainId, binding.pluginId, mutation);
      lease.controller.signal.throwIfAborted();
      let prepare: PluginTrainingFormatLease["prepare"];
      if (binding.executable?.kind === "wasm-stream") {
        const packageValue = await this.catalog.package(binding.pluginId, binding.contentSha256);
        const module = packageValue.manifest.trainingModules?.find((item) => item.id === binding.executable!.moduleId);
        if (!module || module.wasmSha256 !== binding.executable.moduleSha256) {
          throw new Error("The executable parser module changed before its lease was acquired.");
        }
        prepare = request => {
          if (normalizeSourceMediaType(request.inputMediaType) !== binding.inputMediaType) throw new Error("Executable parser source MIME changed before preparation.");
          return runPluginWasmParser({ ...request, pluginId: binding.pluginId,
            packageSha256: binding.contentSha256, formatId: binding.formatId, module,
            signal: request.signal ? AbortSignal.any([request.signal, lease.controller.signal]) : lease.controller.signal });
        };
      } else if (binding.executable?.kind === "runtime-stream") {
        if (!this.runtimeStreamParser) throw new Error("Runtime-stream parser host is unavailable.");
        const packageValue = await this.catalog.package(binding.pluginId, binding.contentSha256);
        const module = packageValue.manifest.runtimeModules?.find(item => item.id === binding.executable!.moduleId);
        if (!module || runtimeModuleSha256(module) !== binding.executable.moduleSha256) {
          throw new Error("The runtime parser module changed before its lease was acquired.");
        }
        prepare = request => {
          if (normalizeSourceMediaType(request.inputMediaType) !== binding.inputMediaType) throw new Error("Runtime parser source MIME changed before preparation.");
          return this.runtimeStreamParser!.prepare({ brainId, binding, module,
            moduleSha256: binding.executable!.moduleSha256,
            request: { ...request, signal: request.signal ? AbortSignal.any([request.signal, lease.controller.signal]) : lease.controller.signal } });
        };
      }
      return {
        signal: lease.controller.signal,
        ...(prepare ? { prepare } : {}),
        release: () => { this.activeTrainingFormats.delete(lease); }
      };
    } catch (error) {
      this.activeTrainingFormats.delete(lease);
      if (!lease.controller.signal.aborted) lease.controller.abort(error);
      throw error;
    }
  }

  private semanticKey(brainId: string, pluginId: string, contentSha256: string): string {
    return `${brainId}\0${pluginId}\0${contentSha256}`;
  }

  private async queueSemanticLearning(
    brainId: string,
    pluginId: string,
    contentSha256: string
  ): Promise<PluginSemanticLearningRecord> {
    const packageValue = await this.catalog.package(pluginId, contentSha256);
    const experience = pluginSemanticLearningExperience(packageValue, contentSha256);
    const record = await this.catalog.ensureSemanticLearning(
      brainId, pluginId, contentSha256, experience.sha256
    );
    this.publishSemantic(record);
    if (this.brain.requireInstanceLoaded) {
      try {
        this.brain.requireInstanceLoaded(brainId);
        this.scheduleSemanticLearning(record);
      } catch {
        // Durable pending state is enough for an unloaded identity. The
        // lifecycle hook retries only after that same instance is Load'ed.
      }
    }
    return record;
  }

  private scheduleSemanticLearning(record: PluginSemanticLearningRecord): void {
    const key = this.semanticKey(record.brainId, record.pluginId, record.contentSha256);
    if (record.state === "complete") return;
    if (this.semanticLearning.has(key)) {
      this.semanticReschedule.add(key);
      return;
    }
    const controller = new AbortController();
    let promise!: Promise<void>;
    promise = new Promise<void>((resolve) => setImmediate(resolve))
      .then(() => this.commitSemanticLearning(record, controller.signal))
      .catch(() => undefined)
      .finally(async () => {
        if (this.semanticLearning.get(key)?.promise === promise) this.semanticLearning.delete(key);
        if (this.semanticReschedule.delete(key)) {
          const current = await this.catalog.semanticLearning(
            record.brainId, record.pluginId, record.contentSha256
          ).catch(() => null);
          if (current?.state === "pending") {
            try {
              this.brain.requireInstanceLoaded?.(record.brainId);
              this.scheduleSemanticLearning(current);
            } catch {
              // It remains durably pending until this identity is Load'ed.
            }
          }
        }
      });
    this.semanticLearning.set(key, { controller, promise });
  }

  private async commitSemanticLearning(
    queued: PluginSemanticLearningRecord,
    signal: AbortSignal
  ): Promise<void> {
    const claimed = await this.catalog.claimSemanticLearning(queued);
    if (!claimed) return;
    this.publishSemantic(claimed);
    try {
      if (!this.brain.learnLoadedStructuredExperience || !this.brain.requireInstanceLoaded) {
        throw new Error("Plugin semantic learning is unavailable in this host.");
      }
      this.brain.requireInstanceLoaded(claimed.brainId);
      await this.validateTrainingPluginAccess(
        claimed.brainId, claimed.pluginId, claimed.contentSha256
      );
      signal.throwIfAborted();
      const packageValue = await this.catalog.package(claimed.pluginId, claimed.contentSha256);
      const experience = pluginSemanticLearningExperience(packageValue, claimed.contentSha256);
      if (experience.sha256 !== claimed.experienceSha256) {
        throw new Error("Plugin semantic declaration changed after its immutable queue receipt.");
      }
      const learned = await this.brain.learnLoadedStructuredExperience(claimed.brainId, {
        content: experience.content,
        name: `${packageValue.manifest.name} plugin declaration`,
        sourceLabel: `Plugin ${packageValue.manifest.id}@${packageValue.manifest.version} declarations`,
        provenanceUrl: packageValue.manifest.homepage,
        license: packageValue.manifest.license
      }, signal);
      signal.throwIfAborted();
      await this.validateTrainingPluginAccess(
        claimed.brainId, claimed.pluginId, claimed.contentSha256
      );
      if (learned.source.contentHash !== claimed.experienceSha256 ||
          learned.source.rawTextRetained !== false || !learned.source.id) {
        throw new Error("Plugin semantic learning returned no exact ordinary neural source receipt.");
      }
      await this.catalog.completeSemanticLearning(claimed, learned.source.id);
      const completed = await this.catalog.semanticLearning(
        claimed.brainId, claimed.pluginId, claimed.contentSha256
      );
      if (completed) this.publishSemantic(completed);
    } catch (error) {
      await this.catalog.failSemanticLearning(
        claimed,
        error instanceof Error ? error.message : String(error)
      ).catch(() => undefined);
      const pending = await this.catalog.semanticLearning(
        claimed.brainId, claimed.pluginId, claimed.contentSha256
      ).catch(() => null);
      if (pending) this.publishSemantic(pending);
      throw error;
    }
  }

  private async validateTrainingPluginAccess(
    brainId: string,
    pluginId: string,
    contentSha256: string
  ): Promise<void> {
    const entry = (await this.catalog.list()).find((candidate) => candidate.pluginId === pluginId);
    const access = await this.catalog.instance(brainId, pluginId);
    if (!entry?.enabled || entry.activeContentSha256 !== contentSha256 ||
        !access?.enabled || access.contentSha256 !== contentSha256) {
      throw new Error("Plugin access changed before its ordinary semantic learning receipt committed.");
    }
  }

  /** Retry only for an already loaded identity; this never Load's or spawns a worker. */
  async resumeSemanticLearning(brainId: string): Promise<void> {
    if (!this.brain.requireInstanceLoaded) return;
    try { this.brain.requireInstanceLoaded(brainId); }
    catch { return; }
    for (const record of await this.catalog.pendingSemanticLearning(brainId)) {
      this.scheduleSemanticLearning(record);
    }
  }

  async settleSemanticLearning(brainId?: string): Promise<void> {
    while (true) {
      const operations = [...this.semanticLearning.entries()]
        .filter(([key]) => brainId === undefined || key.startsWith(`${brainId}\0`))
        .map(([, operation]) => operation.promise);
      if (!operations.length) return;
      await Promise.all(operations);
    }
  }

  async setInstanceAccess(request: PluginInstanceAccessRequest): Promise<PluginInstanceAccess> {
    const key = `${request.brainId}:${request.pluginId}`;
    const prior = this.accessQueues.get(key) ?? Promise.resolve();
    const operation = prior.catch(() => undefined).then(async () => {
      const release = this.beginPluginMutation(request.pluginId, request.brainId);
      try { return await this.setInstanceAccessLocked(request); }
      finally { release(); }
    });
    const tail = operation.then(() => undefined, () => undefined);
    this.accessQueues.set(key, tail);
    try {
      const result = await operation;
      this.publishInstance(result);
      return result;
    } catch (error) {
      const current = await this.catalog.instance(request.brainId, request.pluginId).catch(() => null);
      if (current) this.publishInstance(current);
      throw error;
    }
    finally { if (this.accessQueues.get(key) === tail) this.accessQueues.delete(key); }
  }

  private policyRevision(records: readonly ToolPermissionRecord[]): number {
    const revisions = records.map((record) => record.revision)
      .filter((revision): revision is number => Number.isSafeInteger(revision) && Number(revision) >= 1);
    if (!revisions.length) throw new Error("Current host policy revision is unavailable; plugin access was not changed.");
    return Math.max(...revisions);
  }

  private async setInstanceAccessLocked(request: PluginInstanceAccessRequest): Promise<PluginInstanceAccess> {
    const packageValue = await this.catalog.package(request.pluginId, request.contentSha256);
    const semantic = pluginSemanticLearningExperience(packageValue, request.contentSha256);
    const expectedToolIds = packageValue.manifest.tools.map((tool) => pluginToolId(request.pluginId, tool.id));
    const current = await this.brain.listToolPermissions(request.brainId);
    let expectedPolicyRevision = this.policyRevision(current);
    const prior = new Map<string, ToolPermissionRecord>();
    for (const id of expectedToolIds) {
      const record = current.find((permission) => permission.toolId === id);
      if (record) prior.set(id, record);
    }
    const selections = new Map(request.toolPermissions.map((permission) => [permission.toolId, permission.level]));
    // Suspend resolver authority before the first host-policy write. This is a
    // visible staged state, not a false cross-database atomicity claim.
    await this.catalog.setInstanceAccess({ ...request, enabled: false });
    const applied: Array<{
      toolId: string;
      beforeLevel: ToolPermissionRecord["level"];
      afterLevel: ToolPermissionRecord["level"];
      appliedRevision: number;
    }> = [];
    try {
      for (const id of expectedToolIds) {
        const level = request.enabled ? selections.get(id) : "off";
        if (!level) throw new Error(`Missing explicit permission for ${id}.`);
        const permissions = await this.brain.setToolPermission(
          request.brainId,
          id,
          level,
          `plugin-${request.enabled ? "enable" : "disable"}-${digest(`${request.requestId}\0${id}`).slice(0, 64)}`,
          expectedPolicyRevision
        );
        const committed = permissions.find((permission) => permission.toolId === id);
        if (committed?.level !== level || !Number.isSafeInteger(committed.revision)) {
          throw new Error(`Plugin access request was superseded for ${id}; review the current policy and submit a fresh request.`);
        }
        applied.push({ toolId: id, beforeLevel: prior.get(id)?.level ?? "off", afterLevel: level,
          appliedRevision: committed.revision! });
        expectedPolicyRevision = this.policyRevision(permissions);
      }
      if (!request.enabled) return (await this.instance(request.brainId, request.pluginId))!;
      await this.catalog.setInstanceAccess(request, semantic.sha256);
      const confirmed = await this.brain.listToolPermissions(request.brainId);
      for (const mutation of applied) {
        const currentRecord = confirmed.find((permission) => permission.toolId === mutation.toolId);
        if (currentRecord?.level !== mutation.afterLevel || currentRecord.revision !== mutation.appliedRevision) {
          await this.catalog.setInstanceAccess({ ...request, enabled: false });
          throw new Error("Plugin policy changed before access publication; the plugin remains disabled for review.");
        }
      }
      await this.queueSemanticLearning(request.brainId, request.pluginId, request.contentSha256);
      return (await this.instance(request.brainId, request.pluginId))!;
    } catch (error) {
      await this.catalog.setInstanceAccess({ ...request, enabled: false }).catch(() => undefined);
      if (request.enabled) for (const mutation of applied.reverse()) {
        // Roll back only the exact revision this operation still owns. A newer
        // human Off or another reviewed edit always wins.
        const latest = await this.brain.listToolPermissions(request.brainId).catch(() => []);
        const currentRecord = latest.find((permission) => permission.toolId === mutation.toolId);
        if (currentRecord?.level !== mutation.afterLevel || currentRecord.revision !== mutation.appliedRevision) continue;
        await this.brain.setToolPermission(
          request.brainId,
          mutation.toolId,
          mutation.beforeLevel,
          `plugin-rollback-${digest(`${request.requestId}\0${mutation.toolId}\0${mutation.appliedRevision}`).slice(0, 64)}`,
          this.policyRevision(latest)
        ).catch(() => undefined);
      }
      throw error;
    }
  }

  async resolve(invocation: ToolInvocation): Promise<PluginResolvedInvocation | undefined> {
    if (!invocation.toolId.startsWith("plugin.")) return undefined;
    const registered = this.registered.get(invocation.toolId);
    if (!registered) throw new Error("This plugin tool is not globally enabled at its reviewed version.");
    const entry = (await this.catalog.list()).find((item) => item.pluginId === registered.entry.pluginId);
    if (!entry || !entry.enabled || entry.activeContentSha256 !== registered.entry.activeContentSha256) {
      throw new Error("This plugin changed or was disabled; review it before running the tool again.");
    }
    const access = await this.catalog.instance(invocation.brainId, entry.pluginId);
    if (!access?.enabled || access.contentSha256 !== entry.activeContentSha256) {
      throw new Error("This plugin is not explicitly enabled for this instance and immutable version.");
    }
    const definition = registered.tool.actions.find((action) => action.id === invocation.action);
    if (!definition) throw new Error("The plugin tool action is not declared by this immutable version.");
    const capability = access.capabilities.find((selection) =>
      selection.toolId === definition.delegate.toolId &&
      selection.actions.includes(definition.delegate.action)
    );
    if (!capability?.enabled) throw new Error("The plugin's nested host capability is disabled for this instance.");
    const sourceArguments = JSON.parse(JSON.stringify(invocation.arguments)) as Record<string, unknown>;
    const forwarded: Record<string, unknown> = {};
    if (definition.delegate.arguments.forward === "all") Object.assign(forwarded, sourceArguments);
    else for (const key of definition.delegate.arguments.forward) {
      if (Object.prototype.hasOwnProperty.call(sourceArguments, key)) forwarded[key] = sourceArguments[key];
    }
    const nestedArguments = {
      ...forwarded,
      ...(definition.delegate.arguments.constants
        ? structuredClone(definition.delegate.arguments.constants)
        : {})
    };
    const nested: ToolInvocation = {
      brainId: invocation.brainId,
      toolId: definition.delegate.toolId,
      action: definition.delegate.action,
      arguments: nestedArguments
    };
    return {
      pluginId: entry.pluginId,
      pluginName: entry.name,
      contentSha256: entry.activeContentSha256,
      catalogRevision: this.catalog.revision(),
      instanceRevision: access.revision,
      outerToolId: invocation.toolId,
      outerAction: invocation.action,
      nested,
      nestedArgumentsSha256: digest(stable(nestedArguments)),
      capability: { toolId: definition.delegate.toolId, action: definition.delegate.action }
    };
  }

  async panelDocument(url: string): Promise<Response> {
    const identity = parsePluginPanelUrl(url);
    if (!identity) {
      return new Response("Unknown static plugin panel.", {
        status: 404,
        headers: pluginPanelSecurityHeaders()
      });
    }
    try {
      const packageValue = await this.catalog.package(identity.pluginId, identity.contentSha256);
      const panel = packageValue.manifest.panels.find((candidate) => candidate.id === identity.panelId);
      if (!panel) throw new Error("missing");
      return new Response(
        staticPluginPanelDocument(panel.title, panel.html, panel.css ?? ""),
        { status: 200, headers: pluginPanelSecurityHeaders() }
      );
    } catch {
      return new Response("Unknown static plugin panel.", {
        status: 404,
        headers: pluginPanelSecurityHeaders()
      });
    }
  }
}
