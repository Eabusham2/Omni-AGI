import type {
  PluginTrainingFormatBinding,
  PluginTrainingParserFormat,
  ToolInvocation,
  ToolPermissionLevel
} from "./types";

export const OMNI_PLUGIN_FORMAT = "omni-plugin-package" as const;
export const OMNI_PLUGIN_FORMAT_VERSION = 1 as const;
export const PLUGIN_WASM_PARSER_ABI = "omni-wasm-record-stream-1" as const;
export const PLUGIN_RUNTIME_PARSER_ABI = "omni-runtime-record-stream-1" as const;
export const PLUGIN_TRAINING_PARSER_CAPABILITY = "data.training-parser" as const;

export const PLUGIN_RUNTIME_MEDIA_TYPES = [
  "image/png", "image/jpeg", "image/gif", "image/webp",
  "audio/wav", "audio/x-wav", "audio/flac", "audio/mpeg",
  "video/mp4", "video/quicktime", "video/webm"
] as const;
export type PluginRuntimeMediaType = typeof PLUGIN_RUNTIME_MEDIA_TYPES[number];
export type PluginRuntimeMediaKind = "image" | "audio" | "video";
export interface PluginRuntimeMediaOutputDeclaration {
  kind: "media-assets";
  mediaTypes: PluginRuntimeMediaType[];
}

export interface PluginCapabilityDeclaration {
  toolId: string;
  actions: string[];
  reason: string;
}

export interface PluginArgumentBinding {
  /** Forward every input property, or only the named top-level properties. */
  forward: "all" | string[];
  /** Immutable author-owned values override forwarded values. */
  constants?: Record<string, unknown>;
}

export interface PluginToolAction {
  id: string;
  title?: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
  delegate: {
    toolId: string;
    action: string;
    arguments: PluginArgumentBinding;
  };
}

export interface PluginToolDefinition {
  id: string;
  title: string;
  description: string;
  actions: PluginToolAction[];
}

export interface PluginPanelDefinition {
  id: string;
  title: string;
  /** Static markup only. It receives no preload, script, network, or host bridge. */
  html: string;
  css?: string;
}

export interface PluginInteractivePanelToolAccess {
  /** Package-local tool id. The host resolves it to plugin.<package>.<tool>. */
  toolId: string;
  actions: string[];
}

export interface PluginInteractivePanelDefinition {
  id: string;
  title: string;
  /** Inert initial body. Event attributes, resources and nested documents are rejected. */
  html: string;
  css?: string;
  /** Runs only in a sandboxed opaque child with no Node, preload, network or dynamic eval. */
  script: string;
  /** Exact package-local tools/actions this panel may request through ToolExecutor. */
  tools: PluginInteractivePanelToolAccess[];
}

export interface PluginThemeDefinition {
  id: string;
  title: string;
  /** CSS custom properties only; the renderer owns where they may be applied. */
  tokens: Record<string, string>;
}

export interface PluginTrainingFormatDeclaration {
  id: string;
  title: string;
  description: string;
  extensions: string[];
  mediaTypes: string[];
  /** A declaration may remain inert, select a built-in parser, execute an
   * import-free WASM transformer, or use one exact reviewed runtime module. */
  parser: "declaration-only" | {
    kind: "builtin-stream";
    format: PluginTrainingParserFormat;
  } | {
    kind: "wasm-stream";
    moduleId: string;
    output: "jsonl";
  } | {
    /** Dedicated exact-format session using the reviewed generic JS runtime. */
    kind: "runtime-stream";
    moduleId: string;
    output: "jsonl" | PluginRuntimeMediaOutputDeclaration;
  };
}

export interface PluginTrainingModuleDefinition {
  id: string;
  kind: "wasm-stream";
  abi: typeof PLUGIN_WASM_PARSER_ABI;
  /** Canonical base64 bytes are part of the immutable package hash. */
  wasmBase64: string;
  wasmSha256: string;
}

export interface PluginParserExecutionRequest {
  sourcePath: string;
  sourceSha256: string;
  sourceBytes: number;
  inputMediaType?: string;
  transactionKey: string;
  workspaceRoot: string;
  signal?: AbortSignal;
  onProgress?(progress: number, message: string, detail: unknown): void | Promise<void>;
}

export interface PluginParserExecutionProvenance {
  format: "omni-plugin-parser-execution";
  version: 1;
  pluginId: string;
  packageSha256: string;
  formatId: string;
  moduleId: string;
  moduleSha256: string;
  abi: typeof PLUGIN_WASM_PARSER_ABI | typeof PLUGIN_RUNTIME_PARSER_ABI;
  sourceSha256: string;
  sourceBytes: number;
  inputMediaType?: string;
  outputSha256: string;
  outputBytes: number;
  outputRecords: number;
  /** Present only when the verified output is a media-only native archive. */
  outputKind?: "media-assets";
  outputAssets?: number;
  assetManifestSha256?: string;
  outputModalityCounts?: Record<PluginRuntimeMediaKind, number>;
  complete: true;
}

export interface PluginPreparedTrainingSourceReady {
  path: string;
  parserFormat: "jsonl" | "archive";
  provenance: PluginParserExecutionProvenance;
  cleanup(): Promise<void>;
}

export interface PluginPreparedTrainingSourceEmpty {
  state: "no-assets";
  parserFormat: "archive";
  provenance: PluginParserExecutionProvenance & {
    outputKind: "media-assets";
    outputAssets: 0;
  };
  cleanup(): Promise<void>;
}

export type PluginPreparedTrainingSource = PluginPreparedTrainingSourceReady | PluginPreparedTrainingSourceEmpty;

export interface PluginTrainingFormatLease {
  signal: AbortSignal;
  prepare?(request: PluginParserExecutionRequest): Promise<PluginPreparedTrainingSource>;
  release(): void;
}

/** Explicit source parser choice, not a grant or caller-authored binding. */
export interface PluginTrainingFormatSelection {
  pluginId: string;
  contentSha256: string;
  formatId: string;
}
export function validPluginTrainingFormatSelection(value: unknown): value is PluginTrainingFormatSelection {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const item = value as Record<string, unknown>;
  return Object.keys(item).sort().join(",") === "contentSha256,formatId,pluginId" &&
    typeof item.pluginId === "string" && /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(item.pluginId) &&
    typeof item.formatId === "string" && /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(item.formatId) &&
    typeof item.contentSha256 === "string" && /^[a-f0-9]{64}$/.test(item.contentSha256);
}
export interface PluginTrainingFormatSourceInput { path: string; mediaType?: string; }

export interface PluginTrainingFormatResolver {
  resolveTrainingFormat(
    brainId: string,
    path: string,
    mediaType?: string,
    selection?: PluginTrainingFormatSelection
  ): Promise<PluginTrainingFormatBinding | undefined>;
  validateTrainingFormat(
    brainId: string,
    binding: PluginTrainingFormatBinding,
    source?: PluginTrainingFormatSourceInput
  ): Promise<void>;
  acquireTrainingFormat(
    brainId: string,
    binding: PluginTrainingFormatBinding,
    source?: PluginTrainingFormatSourceInput
  ): Promise<PluginTrainingFormatLease>;
}

export interface PluginRuntimeStreamParserRunner {
  prepare(input: {
    brainId: string;
    binding: PluginTrainingFormatBinding;
    module: import("./pluginRuntime").PluginRuntimeModuleDefinition;
    moduleSha256: string;
    request: PluginParserExecutionRequest;
  }): Promise<PluginPreparedTrainingSource>;
}

export interface PluginSemanticLearningRecord {
  brainId: string;
  pluginId: string;
  contentSha256: string;
  experienceSha256: string;
  state: "pending" | "running" | "complete";
  attempt: number;
  trainingSourceId?: string;
  error?: string;
  updatedAt: string;
}

export interface PluginEvent {
  type: "catalog" | "instance" | "semantic-learning";
  pluginId: string;
  contentSha256: string;
  catalogRevision: number;
  brainId?: string;
  instanceRevision?: number;
  semanticState?: PluginSemanticLearningRecord["state"];
  at: string;
}

export interface PluginPortableSelection {
  sourceBrainId: string;
  pluginId: string;
  contentSha256: string;
  reviewedAccess: Omit<PluginInstanceAccess, "brainId" | "semanticLearning">;
  semanticLearning?: PluginSemanticLearningRecord;
  provenance: {
    version: string;
    installedAt: string;
    sourceLabel: string;
    parentSha256?: string;
    author: string;
    license: string;
    homepage?: string;
  };
  authority: "source-review" | "external-dormant" | "merge-parent-dormant";
  originAuthority?: "source-review" | "external-dormant" | "merge-parent-dormant";
  recordedAt: string;
}

export interface PluginPortabilityManifest {
  format: "omni-plugin-portability";
  formatVersion: 1;
  sourceBrainId: string;
  createdAt: string;
  selections: PluginPortableSelection[];
  packages: Array<{
    pluginId: string;
    contentSha256: string;
    path: string;
  }>;
}

export interface OmniPluginManifest {
  id: string;
  version: string;
  name: string;
  description: string;
  author: string;
  license: string;
  homepage?: string;
  documentation: string;
  capabilities: PluginCapabilityDeclaration[];
  tools: PluginToolDefinition[];
  panels: PluginPanelDefinition[];
  /** Optional to preserve canonical hashes of earlier static-only v1 packages. */
  interactivePanels?: PluginInteractivePanelDefinition[];
  /** Optional to preserve canonical immutable hashes of existing packages. */
  runtimeModules?: import("./pluginRuntime").PluginRuntimeModuleDefinition[];
  themes: PluginThemeDefinition[];
  /** Optional so packages created before executable parsers retain their hash. */
  trainingModules?: PluginTrainingModuleDefinition[];
  trainingFormats: PluginTrainingFormatDeclaration[];
}

export interface OmniPluginPackage {
  format: typeof OMNI_PLUGIN_FORMAT;
  formatVersion: typeof OMNI_PLUGIN_FORMAT_VERSION;
  manifest: OmniPluginManifest;
  /** Edit lineage. A new edit is a new immutable package, never an in-place mutation. */
  parentSha256?: string;
}

export interface PluginVersionSummary {
  version: string;
  contentSha256: string;
  parentSha256?: string;
  installedAt: string;
  sourceLabel: string;
}

export interface PluginCatalogEntry {
  pluginId: string;
  name: string;
  description: string;
  enabled: boolean;
  activeVersion: string;
  activeContentSha256: string;
  revision: number;
  updatedAt: string;
  versions: PluginVersionSummary[];
}

export interface PluginCapabilitySelection {
  toolId: string;
  actions: string[];
  enabled: boolean;
}

export interface PluginToolPermissionSelection {
  toolId: string;
  level: ToolPermissionLevel;
}

export interface PluginInstanceAccess {
  brainId: string;
  pluginId: string;
  contentSha256: string;
  enabled: boolean;
  capabilities: PluginCapabilitySelection[];
  toolPermissions: PluginToolPermissionSelection[];
  revision: number;
  updatedAt: string;
  /** Actual ordinary-learning receipt state for this immutable package. */
  semanticLearning?: PluginSemanticLearningRecord;
}

export interface PluginInstanceAccessRequest {
  brainId: string;
  pluginId: string;
  contentSha256: string;
  enabled: boolean;
  capabilities: PluginCapabilitySelection[];
  toolPermissions: PluginToolPermissionSelection[];
  requestId: string;
}

export interface PluginInstallResult {
  entry: PluginCatalogEntry;
  package: OmniPluginPackage;
  contentSha256: string;
  installed: boolean;
}

export interface PluginResolvedInvocation {
  pluginId: string;
  pluginName: string;
  contentSha256: string;
  catalogRevision: number;
  instanceRevision: number;
  outerToolId: string;
  outerAction: string;
  nested: ToolInvocation;
  nestedArgumentsSha256: string;
  capability: { toolId: string; action: string };
}

export interface PluginApprovalReview {
  kind: "plugin-nested-operation";
  pluginId: string;
  pluginName: string;
  contentSha256: string;
  outer: { toolId: string; action: string };
  resolved: { toolId: string; action: string; arguments: Record<string, unknown> };
  permissionSources: Array<{
    toolId: string;
    level: ToolPermissionLevel;
    inherited: boolean;
  }>;
}

export interface PluginsApi {
  list(): Promise<PluginCatalogEntry[]>;
  package(pluginId: string, contentSha256?: string): Promise<OmniPluginPackage>;
  installFile(): Promise<PluginInstallResult | null>;
  installPackage(value: unknown, sourceLabel?: string): Promise<PluginInstallResult>;
  edit(pluginId: string, baseContentSha256: string, value: unknown): Promise<PluginInstallResult>;
  activate(pluginId: string, contentSha256: string): Promise<PluginCatalogEntry>;
  setEnabled(
    pluginId: string,
    enabled: boolean,
    expectedContentSha256: string,
    expectedRevision: number
  ): Promise<PluginCatalogEntry>;
  exportFile(pluginId: string, contentSha256?: string): Promise<string | null>;
  instance(brainId: string, pluginId: string): Promise<PluginInstanceAccess | null>;
  setInstanceAccess(request: PluginInstanceAccessRequest): Promise<PluginInstanceAccess>;
  portable(brainId: string): Promise<PluginPortableSelection[]>;
  openPanel(
    brainId: string,
    pluginId: string,
    contentSha256: string,
    panelId: string
  ): Promise<import("./pluginInteractivePanels").PluginInteractivePanelLaunch>;
  panelRequest(
    request: import("./pluginInteractivePanels").PluginInteractiveToolRequest
  ): Promise<import("./pluginInteractivePanels").PluginInteractiveToolReceipt>;
  reviewPanelRequest(
    request: import("./pluginInteractivePanels").PluginInteractiveReviewRequest
  ): Promise<import("./pluginInteractivePanels").PluginInteractiveToolReceipt>;
  closePanel(brainId: string, viewId: string, channelNonce: string): Promise<void>;
  panelHistory(brainId: string, page?: { sourceSha256?: string; afterSequence?: number; maxBytes?: number }): Promise<import("./pluginInteractivePanels").PluginPanelHistoryPage>;
  panelHistoryBytes(brainId: string, sequence: number, offset: number, length: number, sourceSha256?: string): Promise<{ text: string; bytes: number; sha256: string; nextOffset?: number }>;
  panelHistorySources(brainId: string, page?: { afterSha256?: string; limit?: number }): Promise<import("./pluginInteractivePanels").PluginPanelHistorySourcePage>;
  onEvent(listener: (event: PluginEvent) => void): () => void;
}

export function pluginToolId(pluginId: string, localToolId: string): string {
  return `plugin.${pluginId}.${localToolId}`;
}
