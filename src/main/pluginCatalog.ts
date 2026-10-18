import { createHash, randomUUID } from "node:crypto";
import { link, lstat, mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import {
  OMNI_PLUGIN_FORMAT,
  OMNI_PLUGIN_FORMAT_VERSION,
  PLUGIN_RUNTIME_MEDIA_TYPES,
  PLUGIN_TRAINING_PARSER_CAPABILITY,
  PLUGIN_WASM_PARSER_ABI,
  pluginToolId,
  type OmniPluginManifest,
  type OmniPluginPackage,
  type PluginCatalogEntry,
  type PluginCapabilityDeclaration,
  type PluginCapabilitySelection,
  type PluginInstallResult,
  type PluginInstanceAccess,
  type PluginInstanceAccessRequest,
  type PluginInteractivePanelDefinition,
  type PluginPanelDefinition,
  type PluginPortableSelection,
  type PluginSemanticLearningRecord,
  type PluginThemeDefinition,
  type PluginToolAction,
  type PluginToolDefinition,
  type PluginTrainingFormatDeclaration,
  type PluginTrainingModuleDefinition,
  type PluginVersionSummary
} from "../shared/plugins";
import type { PluginTrainingParserFormat, ToolPermissionLevel } from "../shared/types";
import { inspectPluginWasmModule } from "./pluginWasmModule";
import { normalizePluginRuntimeModule } from "./pluginRuntimeValidation";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const PLUGIN_ID = /^[a-z0-9](?:[a-z0-9.-]{0,126}[a-z0-9])?$/;
const LOCAL_ID = /^[a-z][a-z0-9-]{0,63}$/;
const TOOL_ID = /^[a-z][a-z0-9.-]{1,79}$/;
const ACTION_ID = /^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$/;
const SEMVER = /^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/;
const SHA256 = /^[a-f0-9]{64}$/;
const LEVELS = new Set<ToolPermissionLevel>(["off", "ask", "auto", "full"]);
const STREAMING_PARSER_FORMATS = new Set<PluginTrainingParserFormat>([
  "text", "pdf", "csv", "tsv", "json", "jsonl", "parquet", "arrow",
  "sqlite", "archive", "huggingface", "image", "audio", "video"
]);
const FORBIDDEN_JSON_KEYS = new Set(["__proto__", "prototype", "constructor"]);
const CREDENTIAL_CONSTANT_KEYS = new Set([
  "authorization", "proxyauthorization", "apikey", "accesstoken", "refreshtoken",
  "clientsecret", "privatekey", "password", "secret", "credential", "token",
  "cookie", "setcookie"
]);

function digest(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function exactKeys(value: Record<string, unknown>, allowed: readonly string[], label: string): void {
  const set = new Set(allowed);
  const invalid = Object.keys(value).filter((key) => !set.has(key));
  if (invalid.length) throw new Error(`${label} contains unsupported fields: ${invalid.join(", ")}.`);
}

function requiredString(value: unknown, label: string, maximum = 16_000): string {
  if (typeof value !== "string" || !value.trim() || value.length > maximum || value.includes("\0")) {
    throw new Error(`${label} is invalid.`);
  }
  return value.trim();
}

function optionalString(value: unknown, label: string, maximum = 16_000): string | undefined {
  if (value === undefined) return undefined;
  return requiredString(value, label, maximum);
}

function jsonValue(value: unknown, label: string, depth = 0): unknown {
  if (depth > 64) throw new Error(`${label} is nested too deeply.`);
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error(`${label} contains a non-finite number.`);
    return value;
  }
  if (Array.isArray(value)) return value.map((item, index) => jsonValue(item, `${label}[${index}]`, depth + 1));
  const source = object(value, label);
  const result: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(source)) {
    if (FORBIDDEN_JSON_KEYS.has(key) || key.includes("\0")) throw new Error(`${label} contains an unsafe key.`);
    result[key] = jsonValue(item, `${label}.${key}`, depth + 1);
  }
  return result;
}

export function assertNoPluginCredentialConstants(value: unknown, label: string, depth = 0): void {
  if (depth > 64) throw new Error(`${label} is nested too deeply.`);
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertNoPluginCredentialConstants(item, `${label}[${index}]`, depth + 1));
    return;
  }
  if (!value || typeof value !== "object") return;
  for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
    const normalized = key.replace(/[^A-Za-z0-9]/g, "").toLowerCase();
    if (CREDENTIAL_CONSTANT_KEYS.has(normalized) ||
        ["apikey", "accesstoken", "refreshtoken", "clientsecret", "privatekey", "password"]
          .some((suffix) => normalized.endsWith(suffix))) {
      throw new Error(`${label} contains a credential-like constant; use the installation credential store instead.`);
    }
    assertNoPluginCredentialConstants(nested, `${label}.${key}`, depth + 1);
  }
}

function canonical(value: unknown): string {
  const compare = (left: string, right: string): number => left < right ? -1 : left > right ? 1 : 0;
  const encode = (item: unknown): unknown => {
    if (Array.isArray(item)) return item.map(encode);
    if (item && typeof item === "object") {
      return Object.fromEntries(
        Object.entries(item as Record<string, unknown>)
          .sort(([left], [right]) => compare(left, right))
          .map(([key, nested]) => [key, encode(nested)])
      );
    }
    return item;
  };
  return JSON.stringify(encode(value));
}

function safeStaticHtml(value: unknown, label: string): string {
  const html = requiredString(value, label, 2_000_000);
  if (/<\/?(?:html|head|body|script|style|iframe|frame|object|embed|form|input|button|textarea|select|link|meta|base)\b/i.test(html) ||
    /<[^>]*[\s/]on[a-z0-9_-]+\s*=/i.test(html) ||
    /<[^>]*[\s/](?:src|srcset|href|xlink:href|poster|data|action|formaction|background)\s*=/i.test(html) ||
    /javascript\s*:/i.test(html)) {
    throw new Error(`${label} must be static markup without script, navigation, forms, or remote resources.`);
  }
  return html;
}

function safeInteractiveHtml(value: unknown, label: string): string {
  const html = requiredString(value, label, 2_000_000);
  if (/<\/?(?:html|head|body|script|style|iframe|frame|object|embed|form|link|meta|base)\b/i.test(html) ||
    /<[^>]*[\s/]on[a-z0-9_-]+\s*=/i.test(html) ||
    /<[^>]*[\s/](?:src|srcset|href|xlink:href|poster|data|action|formaction|background)\s*=/i.test(html) ||
    /javascript\s*:/i.test(html)) {
    throw new Error(`${label} may contain local controls and canvases but no script, events, navigation, forms, frames, or resources.`);
  }
  return html;
}

function safeInteractiveScript(value: unknown, label: string): string {
  const script = requiredString(value, label, 2_000_000).replace(/\r\n?/g, "\n");
  // The script is inserted as the text of one CSP-hashed element. Refuse the
  // only byte sequence that can terminate that element instead of attempting
  // an ambiguous HTML/JavaScript rewrite.
  if (/<\/script\b/i.test(script)) {
    throw new Error(`${label} may not contain a closing script element.`);
  }
  return script;
}

function safeCss(value: unknown, label: string): string {
  const css = requiredString(value, label, 2_000_000);
  if (/@import\b|url\s*\(|expression\s*\(|javascript\s*:|behavior\s*:|-moz-binding\s*:/i.test(css)) {
    throw new Error(`${label} may not import or address external resources.`);
  }
  return css;
}

function safeCssToken(value: unknown, label: string): string {
  const token = requiredString(value, label, 1_000);
  if (/[{};]|@import\b|url\s*\(|expression\s*\(|javascript\s*:/i.test(token)) {
    throw new Error(`${label} is not a safe CSS token value.`);
  }
  return token;
}

function unique<T>(values: readonly T[], key: (value: T) => string, label: string): void {
  const seen = new Set<string>();
  for (const value of values) {
    const id = key(value);
    if (seen.has(id)) throw new Error(`${label} contains duplicate ${id}.`);
    seen.add(id);
  }
}

function capability(value: unknown, index: number): PluginCapabilityDeclaration {
  const row = object(value, `manifest.capabilities[${index}]`);
  exactKeys(row, ["toolId", "actions", "reason"], `manifest.capabilities[${index}]`);
  const toolId = requiredString(row.toolId, "Capability tool id", 256);
  if (!TOOL_ID.test(toolId) || toolId.startsWith("plugin.")) throw new Error("Plugin capabilities must name a non-plugin typed host tool.");
  if (!Array.isArray(row.actions) || !row.actions.length) throw new Error("Plugin capability actions are required.");
  const actions = row.actions.map((item, actionIndex) => {
    const action = requiredString(item, `Capability action ${actionIndex}`, 128);
    if (!ACTION_ID.test(action)) throw new Error("Plugin capability action is invalid.");
    return action;
  });
  if (new Set(actions).size !== actions.length) throw new Error("Plugin capability actions must be unique.");
  return { toolId, actions, reason: requiredString(row.reason, "Capability reason", 2_000) };
}

function toolAction(value: unknown, toolIndex: number, actionIndex: number): PluginToolAction {
  const label = `manifest.tools[${toolIndex}].actions[${actionIndex}]`;
  const row = object(value, label);
  exactKeys(row, ["id", "title", "description", "inputSchema", "delegate"], label);
  const id = requiredString(row.id, `${label}.id`, 128);
  if (!ACTION_ID.test(id)) throw new Error(`${label}.id is invalid.`);
  const delegate = object(row.delegate, `${label}.delegate`);
  exactKeys(delegate, ["toolId", "action", "arguments"], `${label}.delegate`);
  const toolId = requiredString(delegate.toolId, `${label}.delegate.toolId`, 256);
  const action = requiredString(delegate.action, `${label}.delegate.action`, 128);
  if (!TOOL_ID.test(toolId) || toolId.startsWith("plugin.") || !ACTION_ID.test(action)) {
    throw new Error(`${label} delegates to an invalid or recursive capability.`);
  }
  const binding = object(delegate.arguments, `${label}.delegate.arguments`);
  exactKeys(binding, ["forward", "constants"], `${label}.delegate.arguments`);
  let forward: "all" | string[];
  if (binding.forward === "all") forward = "all";
  else if (Array.isArray(binding.forward)) {
    forward = binding.forward.map((item, index) => {
      const key = requiredString(item, `${label}.delegate.arguments.forward[${index}]`, 256);
      if (FORBIDDEN_JSON_KEYS.has(key)) throw new Error("Plugin argument forwarding contains an unsafe key.");
      return key;
    });
    if (new Set(forward).size !== forward.length) throw new Error("Plugin forwarded argument names must be unique.");
  } else throw new Error(`${label}.delegate.arguments.forward is invalid.`);
  const constants = binding.constants === undefined
    ? undefined
    : jsonValue(object(binding.constants, `${label}.delegate.arguments.constants`), `${label}.delegate.arguments.constants`) as Record<string, unknown>;
  if (constants) assertNoPluginCredentialConstants(constants, `${label}.delegate.arguments.constants`);
  const inputSchema = row.inputSchema === undefined
    ? undefined
    : jsonValue(object(row.inputSchema, `${label}.inputSchema`), `${label}.inputSchema`) as Record<string, unknown>;
  return {
    id,
    ...(row.title === undefined ? {} : { title: requiredString(row.title, `${label}.title`, 256) }),
    ...(row.description === undefined ? {} : { description: requiredString(row.description, `${label}.description`, 2_000) }),
    ...(inputSchema ? { inputSchema } : {}),
    delegate: { toolId, action, arguments: { forward, ...(constants ? { constants } : {}) } }
  };
}

function tool(value: unknown, index: number): PluginToolDefinition {
  const label = `manifest.tools[${index}]`;
  const row = object(value, label);
  exactKeys(row, ["id", "title", "description", "actions"], label);
  const id = requiredString(row.id, `${label}.id`, 64);
  if (!LOCAL_ID.test(id)) throw new Error(`${label}.id must be lowercase letters, digits, and hyphens.`);
  if (!Array.isArray(row.actions) || !row.actions.length) throw new Error(`${label}.actions are required.`);
  const actions = row.actions.map((item, actionIndex) => toolAction(item, index, actionIndex));
  unique(actions, (item) => item.id, `${label}.actions`);
  return {
    id,
    title: requiredString(row.title, `${label}.title`, 256),
    description: requiredString(row.description, `${label}.description`, 2_000),
    actions
  };
}

function panel(value: unknown, index: number): PluginPanelDefinition {
  const label = `manifest.panels[${index}]`;
  const row = object(value, label);
  exactKeys(row, ["id", "title", "html", "css"], label);
  const id = requiredString(row.id, `${label}.id`, 64);
  if (!LOCAL_ID.test(id)) throw new Error(`${label}.id is invalid.`);
  return {
    id,
    title: requiredString(row.title, `${label}.title`, 256),
    html: safeStaticHtml(row.html, `${label}.html`),
    ...(row.css === undefined ? {} : { css: safeCss(row.css, `${label}.css`) })
  };
}

function interactivePanel(value: unknown, index: number): PluginInteractivePanelDefinition {
  const label = `manifest.interactivePanels[${index}]`;
  const row = object(value, label);
  exactKeys(row, ["id", "title", "html", "css", "script", "tools"], label);
  const id = requiredString(row.id, `${label}.id`, 64);
  if (!LOCAL_ID.test(id)) throw new Error(`${label}.id is invalid.`);
  if (!Array.isArray(row.tools)) throw new Error(`${label}.tools must be an array.`);
  const tools = row.tools.map((value, toolIndex) => {
    const toolLabel = `${label}.tools[${toolIndex}]`;
    const tool = object(value, toolLabel);
    exactKeys(tool, ["toolId", "actions"], toolLabel);
    const toolId = requiredString(tool.toolId, `${toolLabel}.toolId`, 64);
    if (!LOCAL_ID.test(toolId) || !Array.isArray(tool.actions) || !tool.actions.length) {
      throw new Error(`${toolLabel} must name a local plugin tool and at least one action.`);
    }
    const actions = tool.actions.map((actionValue, actionIndex) => {
      const action = requiredString(actionValue, `${toolLabel}.actions[${actionIndex}]`, 128);
      if (!ACTION_ID.test(action)) throw new Error(`${toolLabel} contains an invalid action.`);
      return action;
    });
    if (new Set(actions).size !== actions.length) throw new Error(`${toolLabel} contains duplicate actions.`);
    return { toolId, actions };
  });
  unique(tools, (item) => item.toolId, `${label}.tools`);
  return {
    id,
    title: requiredString(row.title, `${label}.title`, 256),
    html: safeInteractiveHtml(row.html, `${label}.html`),
    ...(row.css === undefined ? {} : { css: safeCss(row.css, `${label}.css`) }),
    script: safeInteractiveScript(row.script, `${label}.script`),
    tools
  };
}

function theme(value: unknown, index: number): PluginThemeDefinition {
  const label = `manifest.themes[${index}]`;
  const row = object(value, label);
  exactKeys(row, ["id", "title", "tokens"], label);
  const id = requiredString(row.id, `${label}.id`, 64);
  if (!LOCAL_ID.test(id)) throw new Error(`${label}.id is invalid.`);
  const tokensSource = object(row.tokens, `${label}.tokens`);
  const tokens: Record<string, string> = {};
  for (const [key, token] of Object.entries(tokensSource)) {
    if (!/^--[a-z][a-z0-9-]{0,63}$/.test(key)) throw new Error(`${label} contains an invalid CSS custom property.`);
    tokens[key] = safeCssToken(token, `${label}.tokens.${key}`);
  }
  if (!Object.keys(tokens).length) throw new Error(`${label} must declare at least one token.`);
  return { id, title: requiredString(row.title, `${label}.title`, 256), tokens };
}

function trainingFormat(value: unknown, index: number): PluginTrainingFormatDeclaration {
  const label = `manifest.trainingFormats[${index}]`;
  const row = object(value, label);
  exactKeys(row, ["id", "title", "description", "extensions", "mediaTypes", "parser"], label);
  const id = requiredString(row.id, `${label}.id`, 64);
  if (!LOCAL_ID.test(id)) throw new Error(`${label} is invalid.`);
  if (!Array.isArray(row.extensions) || !Array.isArray(row.mediaTypes)) throw new Error(`${label} extensions and media types are required.`);
  const extensions = row.extensions.map((item, itemIndex) => {
    const extension = requiredString(item, `${label}.extensions[${itemIndex}]`, 32).toLowerCase().replace(/^\./, "");
    if (!/^[a-z0-9][a-z0-9_-]{0,30}$/.test(extension)) throw new Error(`${label} contains an invalid extension.`);
    return extension;
  });
  const mediaTypes = row.mediaTypes.map((item, itemIndex) => {
    const mediaType = requiredString(item, `${label}.mediaTypes[${itemIndex}]`, 128).toLowerCase();
    if (!/^[a-z0-9.+-]+\/[a-z0-9.+-]+$/.test(mediaType)) throw new Error(`${label} contains an invalid media type.`);
    return mediaType;
  });
  if (!extensions.length && !mediaTypes.length) throw new Error(`${label} needs an extension or media type.`);
  if (new Set(extensions).size !== extensions.length || new Set(mediaTypes).size !== mediaTypes.length) throw new Error(`${label} contains duplicate selectors.`);
  let parser: PluginTrainingFormatDeclaration["parser"];
  if (row.parser === "declaration-only") parser = row.parser;
  else {
    const parserValue = object(row.parser, `${label}.parser`);
    if (parserValue.kind === "builtin-stream") {
      exactKeys(parserValue, ["kind", "format"], `${label}.parser`);
      if (typeof parserValue.format !== "string" ||
          !STREAMING_PARSER_FORMATS.has(parserValue.format as PluginTrainingParserFormat)) {
        throw new Error(`${label}.parser must select one supported built-in streaming format.`);
      }
      parser = { kind: "builtin-stream", format: parserValue.format as PluginTrainingParserFormat };
    } else {
      exactKeys(parserValue, ["kind", "moduleId", "output"], `${label}.parser`);
      const moduleId = requiredString(parserValue.moduleId, `${label}.parser.moduleId`, 64);
      if (!["wasm-stream", "runtime-stream"].includes(String(parserValue.kind)) || !LOCAL_ID.test(moduleId)) {
        throw new Error(`${label}.parser must select one supported built-in stream, immutable WASM stream, or reviewed runtime stream module.`);
      }
      if (parserValue.output === "jsonl") {
        parser = { kind: parserValue.kind as "wasm-stream" | "runtime-stream", moduleId, output: "jsonl" };
      } else {
        if (parserValue.kind !== "runtime-stream") throw new Error(`${label}.parser media output requires a reviewed runtime stream module.`);
        const output = object(parserValue.output, `${label}.parser.output`);
        exactKeys(output, ["kind", "mediaTypes"], `${label}.parser.output`);
        if (output.kind !== "media-assets" || !Array.isArray(output.mediaTypes) || !output.mediaTypes.length) {
          throw new Error(`${label}.parser.output must declare at least one supported native media MIME type.`);
        }
        const supported = new Set<string>(PLUGIN_RUNTIME_MEDIA_TYPES);
        const mediaTypes = output.mediaTypes.map((value, mediaIndex) => {
          const mediaType = requiredString(value, `${label}.parser.output.mediaTypes[${mediaIndex}]`, 128).toLowerCase();
          if (!supported.has(mediaType)) throw new Error(`${label}.parser.output contains an unsupported native media MIME type.`);
          return mediaType as typeof PLUGIN_RUNTIME_MEDIA_TYPES[number];
        });
        if (new Set(mediaTypes).size !== mediaTypes.length) throw new Error(`${label}.parser.output contains duplicate media MIME types.`);
        parser = { kind: "runtime-stream", moduleId, output: { kind: "media-assets", mediaTypes } };
      }
    }
  }
  return { id, title: requiredString(row.title, `${label}.title`, 256),
    description: requiredString(row.description, `${label}.description`, 2_000), extensions, mediaTypes, parser };
}

function trainingModule(value: unknown, index: number): PluginTrainingModuleDefinition {
  const label = `manifest.trainingModules[${index}]`, row = object(value, label);
  exactKeys(row, ["id", "kind", "abi", "wasmBase64", "wasmSha256"], label);
  const id = requiredString(row.id, `${label}.id`, 64);
  if (!LOCAL_ID.test(id) || row.kind !== "wasm-stream" || row.abi !== PLUGIN_WASM_PARSER_ABI ||
      typeof row.wasmBase64 !== "string" || typeof row.wasmSha256 !== "string") {
    throw new Error(`${label} is invalid.`);
  }
  const module: PluginTrainingModuleDefinition = { id, kind: "wasm-stream", abi: PLUGIN_WASM_PARSER_ABI,
    wasmBase64: row.wasmBase64, wasmSha256: row.wasmSha256 };
  inspectPluginWasmModule(module);
  return module;
}

export function normalizePluginPackage(value: unknown): OmniPluginPackage {
  const root = object(value, "Plugin package");
  exactKeys(root, ["format", "formatVersion", "manifest", "parentSha256"], "Plugin package");
  if (root.format !== OMNI_PLUGIN_FORMAT || root.formatVersion !== OMNI_PLUGIN_FORMAT_VERSION) {
    throw new Error("Unsupported plugin package format.");
  }
  const source = object(root.manifest, "Plugin manifest");
  exactKeys(source, ["id", "version", "name", "description", "author", "license", "homepage", "documentation",
    "capabilities", "tools", "panels", "interactivePanels", "runtimeModules", "themes", "trainingModules", "trainingFormats"], "Plugin manifest");
  const id = requiredString(source.id, "Plugin id", 128).toLowerCase();
  const version = requiredString(source.version, "Plugin version", 128);
  if (!PLUGIN_ID.test(id) || !SEMVER.test(version)) throw new Error("Plugin id or semantic version is invalid.");
  const homepage = optionalString(source.homepage, "Plugin homepage", 2_000);
  if (homepage) {
    const url = new URL(homepage);
    if (!/^https?:$/.test(url.protocol) || url.username || url.password) throw new Error("Plugin homepage must be a public HTTP(S) URL without credentials.");
  }
  for (const key of ["capabilities", "tools", "panels", "themes", "trainingFormats"] as const) {
    if (!Array.isArray(source[key])) throw new Error(`Plugin manifest ${key} must be an array.`);
  }
  if (source.interactivePanels !== undefined && !Array.isArray(source.interactivePanels)) {
    throw new Error("Plugin manifest interactivePanels must be an array when present.");
  }
  if (source.trainingModules !== undefined && !Array.isArray(source.trainingModules)) {
    throw new Error("Plugin manifest trainingModules must be an array when present.");
  }
  if (source.runtimeModules !== undefined && !Array.isArray(source.runtimeModules)) throw new Error("Plugin runtimeModules must be an array when present.");
  const capabilities = (source.capabilities as unknown[]).map(capability);
  const tools = (source.tools as unknown[]).map(tool);
  const panels = (source.panels as unknown[]).map(panel);
  const interactivePanels = (source.interactivePanels as unknown[] | undefined)?.map(interactivePanel) ?? [];
  const runtimeModules = (source.runtimeModules as unknown[] | undefined)?.map(normalizePluginRuntimeModule) ?? [];
  const themes = (source.themes as unknown[]).map(theme);
  const trainingModules = (source.trainingModules as unknown[] | undefined)?.map(trainingModule) ?? [];
  const trainingFormats = (source.trainingFormats as unknown[]).map(trainingFormat);
  unique(capabilities, (item) => item.toolId, "Plugin capabilities");
  unique(tools, (item) => item.id, "Plugin tools");
  unique(panels, (item) => item.id, "Plugin panels");
  unique(interactivePanels, (item) => item.id, "Plugin interactive panels");
  unique(runtimeModules, (item) => item.id, "Plugin runtime modules");
  if (new Set([...panels.map((item) => item.id), ...interactivePanels.map((item) => item.id)]).size !==
      panels.length + interactivePanels.length) {
    throw new Error("Static and interactive plugin panel ids must be distinct.");
  }
  unique(themes, (item) => item.id, "Plugin themes");
  unique(trainingModules, (item) => item.id, "Plugin training modules");
  unique(trainingFormats, (item) => item.id, "Plugin training formats");
  const modules = new Map(trainingModules.map((item) => [item.id, item]));
  const executableFormats = trainingFormats.filter((item) => item.parser !== "declaration-only" && item.parser.kind !== "builtin-stream");
  for (const format of trainingFormats) {
    const parser = format.parser;
    if (parser === "declaration-only" || parser.kind === "builtin-stream") continue;
    if (parser.kind === "wasm-stream" && !modules.has(parser.moduleId)) {
      throw new Error(`Executable training format ${format.id} references an undeclared WASM module.`);
    }
    if (parser.kind === "runtime-stream" && !runtimeModules.some(module => module.id === parser.moduleId)) {
      throw new Error(`Executable training format ${format.id} references an undeclared runtime module.`);
    }
  }
  const referencedModules = new Set(trainingFormats.flatMap((format) => format.parser !== "declaration-only" && format.parser.kind === "wasm-stream" ? [format.parser.moduleId] : []));
  if (trainingModules.some((item) => !referencedModules.has(item.id))) {
    throw new Error("Every executable training module must be referenced by a training format.");
  }
  if (executableFormats.length) {
    const parserCapability = capabilities.find((item) => item.toolId === PLUGIN_TRAINING_PARSER_CAPABILITY);
    if (!parserCapability || canonical([...parserCapability.actions].sort()) !== canonical(executableFormats.map((item) => item.id).sort())) {
      throw new Error(`Executable training formats require the exact ${PLUGIN_TRAINING_PARSER_CAPABILITY} capability actions.`);
    }
  } else if (capabilities.some((item) => item.toolId === PLUGIN_TRAINING_PARSER_CAPABILITY)) {
    throw new Error(`${PLUGIN_TRAINING_PARSER_CAPABILITY} is reserved for declared executable training formats.`);
  }
  const declared = new Map(capabilities.map((item) => [item.toolId, new Set(item.actions)]));
  for (const definition of tools) {
    const fullId = pluginToolId(id, definition.id);
    if (!TOOL_ID.test(fullId)) throw new Error(`Plugin tool id ${fullId} is too long or incompatible with the typed tool protocol.`);
    for (const action of definition.actions) {
      if (!declared.get(action.delegate.toolId)?.has(action.delegate.action)) {
        throw new Error(`Plugin action ${definition.id}.${action.id} uses an undeclared nested capability.`);
      }
    }
  }
  const toolActions = new Map(tools.map((definition) =>
    [definition.id, new Set(definition.actions.map((action) => action.id))]));
  for (const panelDefinition of interactivePanels) {
    for (const allowed of panelDefinition.tools) {
      const actions = toolActions.get(allowed.toolId);
      if (!actions || allowed.actions.some((action) => !actions.has(action))) {
        throw new Error(`Interactive panel ${panelDefinition.id} requests an undeclared plugin tool action.`);
      }
    }
  }
  const parentSha256 = optionalString(root.parentSha256, "Plugin parent checksum", 64);
  if (parentSha256 && !SHA256.test(parentSha256)) throw new Error("Plugin parent checksum is invalid.");
  const manifest: OmniPluginManifest = {
    id,
    version,
    name: requiredString(source.name, "Plugin name", 256),
    description: requiredString(source.description, "Plugin description", 4_000),
    author: requiredString(source.author, "Plugin author", 256),
    license: requiredString(source.license, "Plugin license", 256),
    ...(homepage ? { homepage } : {}),
    documentation: requiredString(source.documentation, "Plugin documentation", 2_000_000),
    capabilities,
    tools,
    panels,
    ...(source.interactivePanels === undefined ? {} : { interactivePanels }),
    ...(source.runtimeModules === undefined ? {} : { runtimeModules }),
    themes,
    ...(source.trainingModules === undefined ? {} : { trainingModules }),
    trainingFormats
  };
  return { format: OMNI_PLUGIN_FORMAT, formatVersion: OMNI_PLUGIN_FORMAT_VERSION, manifest,
    ...(parentSha256 ? { parentSha256 } : {}) };
}

export function canonicalPluginPackage(value: unknown): { package: OmniPluginPackage; json: string; sha256: string } {
  const normalized = normalizePluginPackage(value);
  const json = canonical(normalized);
  return { package: normalized, json, sha256: digest(json) };
}

function safeBrainId(value: string): string {
  if (!SAFE_ID.test(value)) throw new Error("Invalid plugin instance brain id.");
  return value;
}

function safePluginId(value: string): string {
  const normalized = value.toLowerCase();
  if (!PLUGIN_ID.test(normalized)) throw new Error("Invalid plugin id.");
  return normalized;
}

interface CatalogRow {
  plugin_id: string;
  name: string;
  description: string;
  enabled: number;
  active_version: string;
  active_sha256: string;
  revision: number;
  updated_at: string;
}

interface VersionRow {
  version: string;
  content_sha256: string;
  parent_sha256: string | null;
  installed_at: string;
  source_label: string;
}

interface SemanticLearningRow {
  brain_id: string;
  plugin_id: string;
  content_sha256: string;
  experience_sha256: string;
  state: PluginSemanticLearningRecord["state"];
  attempt: number;
  training_source_id: string | null;
  error: string | null;
  updated_at: string;
}

interface PortableSelectionRow {
  brain_id: string;
  plugin_id: string;
  content_sha256: string;
  source_brain_id: string;
  payload_json: string;
  payload_sha256: string;
  recorded_at: string;
}

export class PluginCatalog {
  readonly root: string;
  private readonly blobs: string;
  private readonly databasePath: string;

  constructor(root: string) {
    this.root = resolve(root);
    this.blobs = join(this.root, "blobs");
    this.databasePath = join(this.root, "catalog.sqlite3");
  }

  async initialize(): Promise<void> {
    await mkdir(this.blobs, { recursive: true, mode: 0o700 });
    for (const directory of [this.root, this.blobs]) {
      const info = await lstat(directory);
      if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Plugin catalog path is not a private app-managed directory.");
    }
    try {
      const info = await lstat(this.databasePath);
      if (!info.isFile() || info.isSymbolicLink()) throw new Error("Plugin catalog database is unsafe.");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    this.database(() => undefined);
  }

  private database<T>(operation: (database: DatabaseSync) => T): T {
    const database = new DatabaseSync(this.databasePath);
    try {
      database.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA busy_timeout=5000; PRAGMA foreign_keys=ON;");
      database.exec(`
        CREATE TABLE IF NOT EXISTS catalog_meta (
          singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
          format_version INTEGER NOT NULL CHECK (format_version = 1),
          revision INTEGER NOT NULL CHECK (revision >= 0)
        ) STRICT;
        INSERT OR IGNORE INTO catalog_meta (singleton, format_version, revision) VALUES (1, 1, 0);
        CREATE TABLE IF NOT EXISTS plugins (
          plugin_id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          description TEXT NOT NULL,
          enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
          active_version TEXT NOT NULL,
          active_sha256 TEXT NOT NULL,
          revision INTEGER NOT NULL,
          updated_at TEXT NOT NULL
        ) STRICT;
        CREATE TABLE IF NOT EXISTS plugin_versions (
          plugin_id TEXT NOT NULL,
          version TEXT NOT NULL,
          content_sha256 TEXT NOT NULL,
          parent_sha256 TEXT,
          installed_at TEXT NOT NULL,
          source_label TEXT NOT NULL,
          PRIMARY KEY (plugin_id, version),
          UNIQUE (plugin_id, content_sha256),
          FOREIGN KEY (plugin_id) REFERENCES plugins(plugin_id) ON DELETE CASCADE
        ) STRICT;
        CREATE TABLE IF NOT EXISTS plugin_instances (
          brain_id TEXT NOT NULL,
          plugin_id TEXT NOT NULL,
          content_sha256 TEXT NOT NULL,
          enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
          capabilities_json TEXT NOT NULL,
          tool_permissions_json TEXT NOT NULL,
          revision INTEGER NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (brain_id, plugin_id),
          FOREIGN KEY (plugin_id) REFERENCES plugins(plugin_id) ON DELETE CASCADE
        ) STRICT;
        CREATE TABLE IF NOT EXISTS plugin_semantic_learning (
          brain_id TEXT NOT NULL,
          plugin_id TEXT NOT NULL,
          content_sha256 TEXT NOT NULL,
          experience_sha256 TEXT NOT NULL,
          state TEXT NOT NULL CHECK (state IN ('pending', 'running', 'complete')),
          attempt INTEGER NOT NULL CHECK (attempt >= 0),
          training_source_id TEXT,
          error TEXT,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (brain_id, plugin_id, content_sha256),
          FOREIGN KEY (plugin_id) REFERENCES plugins(plugin_id) ON DELETE CASCADE
        ) STRICT;
        CREATE TABLE IF NOT EXISTS plugin_portable_selections (
          brain_id TEXT NOT NULL,
          plugin_id TEXT NOT NULL,
          content_sha256 TEXT NOT NULL,
          source_brain_id TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          payload_sha256 TEXT NOT NULL,
          recorded_at TEXT NOT NULL,
          PRIMARY KEY (brain_id, source_brain_id, plugin_id, content_sha256),
          FOREIGN KEY (plugin_id) REFERENCES plugins(plugin_id) ON DELETE CASCADE
        ) STRICT;
      `);
      return operation(database);
    } finally {
      database.close();
    }
  }

  private nextRevision(database: DatabaseSync): number {
    database.prepare("UPDATE catalog_meta SET revision = revision + 1 WHERE singleton = 1").run();
    return Number((database.prepare("SELECT revision FROM catalog_meta WHERE singleton = 1").get() as { revision: number }).revision);
  }

  private ensureSemanticRow(
    database: DatabaseSync,
    brainId: string,
    pluginId: string,
    contentSha256: string,
    experienceSha256: string,
    now: string
  ): PluginSemanticLearningRecord {
    if (!SHA256.test(contentSha256) || !SHA256.test(experienceSha256)) {
      throw new Error("Invalid plugin semantic-learning identity.");
    }
    database.prepare(`
      INSERT OR IGNORE INTO plugin_semantic_learning
        (brain_id, plugin_id, content_sha256, experience_sha256, state, attempt, updated_at)
      VALUES (?, ?, ?, ?, 'pending', 0, ?)
    `).run(brainId, pluginId, contentSha256, experienceSha256, now);
    const row = database.prepare(`SELECT * FROM plugin_semantic_learning
      WHERE brain_id = ? AND plugin_id = ? AND content_sha256 = ?`).get(
      brainId, pluginId, contentSha256
    ) as unknown as SemanticLearningRow;
    if (row.experience_sha256 !== experienceSha256) {
      throw new Error("Plugin semantic-learning content changed for an immutable package.");
    }
    return this.semanticRecord(row);
  }

  revision(): number {
    return this.database((database) => Number(
      (database.prepare("SELECT revision FROM catalog_meta WHERE singleton = 1").get() as { revision: number }).revision
    ));
  }

  private async storeBlob(json: string, sha256: string): Promise<void> {
    const path = join(this.blobs, `${sha256}.json`);
    try {
      const info = await lstat(path);
      if (!info.isFile() || info.isSymbolicLink()) throw new Error("Immutable plugin blob is not a safe regular file.");
      const existing = await readFile(path);
      if (digest(existing) !== sha256 || existing.toString("utf8") !== json) throw new Error("Immutable plugin blob checksum mismatch.");
      return;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    const handle = await open(path, "wx", 0o600);
    try {
      await handle.writeFile(json, "utf8");
      await handle.sync();
    } finally {
      await handle.close();
    }
  }

  private versions(database: DatabaseSync, pluginId: string): PluginVersionSummary[] {
    return (database.prepare(`
      SELECT version, content_sha256, parent_sha256, installed_at, source_label
      FROM plugin_versions WHERE plugin_id = ? ORDER BY installed_at, version
    `).all(pluginId) as unknown as VersionRow[]).map((row) => ({
      version: row.version,
      contentSha256: row.content_sha256,
      ...(row.parent_sha256 ? { parentSha256: row.parent_sha256 } : {}),
      installedAt: row.installed_at,
      sourceLabel: row.source_label
    }));
  }

  private entry(database: DatabaseSync, pluginId: string): PluginCatalogEntry {
    const row = database.prepare(`
      SELECT plugin_id, name, description, enabled, active_version, active_sha256, revision, updated_at
      FROM plugins WHERE plugin_id = ?
    `).get(pluginId) as unknown as CatalogRow | undefined;
    if (!row) throw new Error(`Plugin ${pluginId} is not installed.`);
    return { pluginId: row.plugin_id, name: row.name, description: row.description,
      enabled: row.enabled === 1, activeVersion: row.active_version, activeContentSha256: row.active_sha256,
      revision: row.revision, updatedAt: row.updated_at, versions: this.versions(database, pluginId) };
  }

  async list(): Promise<PluginCatalogEntry[]> {
    await this.initialize();
    return this.database((database) => (database.prepare("SELECT plugin_id FROM plugins ORDER BY name, plugin_id").all() as unknown as Array<{ plugin_id: string }>)
      .map((row) => this.entry(database, row.plugin_id)));
  }

  async install(value: unknown, sourceLabel = "local package"): Promise<PluginInstallResult> {
    await this.initialize();
    const prepared = canonicalPluginPackage(value);
    const label = requiredString(sourceLabel, "Plugin source label", 512);
    await this.storeBlob(prepared.json, prepared.sha256);
    const now = new Date().toISOString();
    return this.database((database) => {
      database.exec("BEGIN IMMEDIATE");
      try {
        const existing = database.prepare(
          "SELECT content_sha256 FROM plugin_versions WHERE plugin_id = ? AND version = ?"
        ).get(prepared.package.manifest.id, prepared.package.manifest.version) as { content_sha256: string } | undefined;
        if (existing && existing.content_sha256 !== prepared.sha256) {
          throw new Error("That plugin version is already installed with different immutable content.");
        }
        let installed = false;
        if (!existing) {
          const revision = this.nextRevision(database);
          const plugin = database.prepare("SELECT plugin_id FROM plugins WHERE plugin_id = ?").get(prepared.package.manifest.id);
          if (!plugin) {
            database.prepare(`
              INSERT INTO plugins (plugin_id, name, description, enabled, active_version, active_sha256, revision, updated_at)
              VALUES (?, ?, ?, 0, ?, ?, ?, ?)
            `).run(prepared.package.manifest.id, prepared.package.manifest.name, prepared.package.manifest.description,
              prepared.package.manifest.version, prepared.sha256, revision, now);
          }
          database.prepare(`
            INSERT INTO plugin_versions (plugin_id, version, content_sha256, parent_sha256, installed_at, source_label)
            VALUES (?, ?, ?, ?, ?, ?)
          `).run(prepared.package.manifest.id, prepared.package.manifest.version, prepared.sha256,
            prepared.package.parentSha256 ?? null, now, label);
          installed = true;
        }
        const entry = this.entry(database, prepared.package.manifest.id);
        database.exec("COMMIT");
        return { entry, package: prepared.package, contentSha256: prepared.sha256, installed };
      } catch (error) {
        database.exec("ROLLBACK");
        throw error;
      }
    });
  }

  async importFile(path: string): Promise<PluginInstallResult> {
    await this.initialize();
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink()) throw new Error("Plugin import must be a regular JSON package file.");
    let value: unknown;
    try { value = JSON.parse(await readFile(path, "utf8")); }
    catch { throw new Error("Plugin package is not valid UTF-8 JSON."); }
    return this.install(value, basename(path));
  }

  async package(pluginId: string, contentSha256?: string): Promise<OmniPluginPackage> {
    await this.initialize();
    const id = safePluginId(pluginId);
    const hash = contentSha256
      ? (SHA256.test(contentSha256) ? contentSha256 : (() => { throw new Error("Invalid plugin content checksum."); })())
      : this.database((database) => this.entry(database, id).activeContentSha256);
    const registered = this.database((database) => database.prepare(
      "SELECT 1 AS present FROM plugin_versions WHERE plugin_id = ? AND content_sha256 = ?"
    ).get(id, hash));
    if (!registered) throw new Error("Plugin content is not registered in this installation.");
    const path = join(this.blobs, `${hash}.json`);
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink()) throw new Error("Installed plugin content is not a safe regular file.");
    const bytes = await readFile(path);
    if (digest(bytes) !== hash) throw new Error("Installed plugin content failed its immutable checksum.");
    const prepared = canonicalPluginPackage(JSON.parse(bytes.toString("utf8")));
    if (prepared.sha256 !== hash || prepared.json !== bytes.toString("utf8") || prepared.package.manifest.id !== id) {
      throw new Error("Installed plugin content does not match its catalog identity.");
    }
    return prepared.package;
  }

  async edit(pluginId: string, baseContentSha256: string, value: unknown): Promise<PluginInstallResult> {
    const id = safePluginId(pluginId);
    if (!SHA256.test(baseContentSha256)) throw new Error("Invalid plugin edit parent checksum.");
    await this.package(id, baseContentSha256);
    const candidate = normalizePluginPackage(value);
    if (candidate.manifest.id !== id) throw new Error("Plugin edits cannot change the package id.");
    const base = await this.package(id, baseContentSha256);
    if (candidate.manifest.version === base.manifest.version) throw new Error("Plugin edits require a new semantic version.");
    return this.install({ ...candidate, parentSha256: baseContentSha256 }, `edit of ${id}@${base.manifest.version}`);
  }

  async activate(pluginId: string, contentSha256: string): Promise<PluginCatalogEntry> {
    await this.initialize();
    const id = safePluginId(pluginId);
    const selected = await this.package(id, contentSha256);
    const now = new Date().toISOString();
    return this.database((database) => {
      database.exec("BEGIN IMMEDIATE");
      try {
        const revision = this.nextRevision(database);
        // New code/data never inherits global execution. The user must enable
        // this exact immutable version after reviewing it.
        database.prepare(`
          UPDATE plugins SET name = ?, description = ?, enabled = 0,
            active_version = ?, active_sha256 = ?, revision = ?, updated_at = ?
          WHERE plugin_id = ?
        `).run(selected.manifest.name, selected.manifest.description, selected.manifest.version,
          contentSha256, revision, now, id);
        const entry = this.entry(database, id);
        database.exec("COMMIT");
        return entry;
      } catch (error) { database.exec("ROLLBACK"); throw error; }
    });
  }

  async setEnabled(
    pluginId: string,
    enabled: boolean,
    expectedContentSha256: string,
    expectedRevision: number,
    semanticExperienceSha256?: string
  ): Promise<PluginCatalogEntry> {
    await this.initialize();
    const id = safePluginId(pluginId);
    if (typeof enabled !== "boolean" || !SHA256.test(expectedContentSha256) ||
        !Number.isSafeInteger(expectedRevision) || expectedRevision < 1 ||
        (enabled && (!semanticExperienceSha256 || !SHA256.test(semanticExperienceSha256)))) {
      throw new Error("Plugin enabled-state review identity is invalid.");
    }
    return this.database((database) => {
      database.exec("BEGIN IMMEDIATE");
      try {
        const current = this.entry(database, id);
        if (current.activeContentSha256 !== expectedContentSha256 || current.revision !== expectedRevision) {
          throw new Error("The active plugin version changed; review its exact current hash before enabling it.");
        }
        if (current.enabled !== enabled) {
          const revision = this.nextRevision(database), now = new Date().toISOString();
          database.prepare("UPDATE plugins SET enabled = ?, revision = ?, updated_at = ? WHERE plugin_id = ?")
            .run(enabled ? 1 : 0, revision, now, id);
        }
        if (enabled) {
          const now = new Date().toISOString();
          const instances = database.prepare(`SELECT brain_id FROM plugin_instances
            WHERE plugin_id = ? AND content_sha256 = ? AND enabled = 1`).all(
            id, expectedContentSha256
          ) as unknown as Array<{ brain_id: string }>;
          for (const instance of instances) {
            this.ensureSemanticRow(database, instance.brain_id, id,
              expectedContentSha256, semanticExperienceSha256!, now);
          }
        }
        const entry = this.entry(database, id);
        database.exec("COMMIT");
        return entry;
      } catch (error) { database.exec("ROLLBACK"); throw error; }
    });
  }

  private normalizeAccess(request: PluginInstanceAccessRequest, packageValue: OmniPluginPackage): {
    capabilities: PluginCapabilitySelection[];
    toolPermissions: PluginInstanceAccess["toolPermissions"];
  } {
    const declaredCapabilities = new Map(packageValue.manifest.capabilities.map((item) => [item.toolId, item.actions]));
    if (!Array.isArray(request.capabilities) || request.capabilities.length !== declaredCapabilities.size) {
      throw new Error("Every plugin capability needs an explicit enabled or disabled choice.");
    }
    const capabilities = request.capabilities.map((item) => {
      if (!item || typeof item.enabled !== "boolean" || !Array.isArray(item.actions)) throw new Error("Invalid plugin capability selection.");
      const actions = item.actions.map((action) => requiredString(action, "Selected capability action", 128));
      const declared = declaredCapabilities.get(item.toolId);
      if (!declared || canonical([...actions].sort()) !== canonical([...declared].sort())) {
        throw new Error("Plugin capability selection differs from the immutable manifest.");
      }
      return { toolId: item.toolId, actions, enabled: item.enabled };
    });
    unique(capabilities, (item) => item.toolId, "Plugin capability selections");
    const expectedTools = new Set(packageValue.manifest.tools.map((item) => pluginToolId(packageValue.manifest.id, item.id)));
    if (!Array.isArray(request.toolPermissions) || request.toolPermissions.length !== expectedTools.size) {
      throw new Error("Every plugin tool needs an explicit permission choice.");
    }
    const toolPermissions = request.toolPermissions.map((item) => {
      if (!item || !expectedTools.has(item.toolId) || !LEVELS.has(item.level)) throw new Error("Invalid plugin tool permission selection.");
      return { toolId: item.toolId, level: item.level };
    });
    unique(toolPermissions, (item) => item.toolId, "Plugin tool permissions");
    return { capabilities, toolPermissions };
  }

  async setInstanceAccess(
    request: PluginInstanceAccessRequest,
    semanticExperienceSha256?: string
  ): Promise<PluginInstanceAccess> {
    await this.initialize();
    const brainId = safeBrainId(request.brainId), pluginId = safePluginId(request.pluginId);
    if (!SHA256.test(request.contentSha256) || typeof request.enabled !== "boolean" || !SAFE_ID.test(request.requestId) ||
        (request.enabled && (!semanticExperienceSha256 || !SHA256.test(semanticExperienceSha256)))) {
      throw new Error("Invalid plugin instance access request.");
    }
    const entry = this.database((database) => this.entry(database, pluginId));
    if (entry.activeContentSha256 !== request.contentSha256) {
      throw new Error("Plugin instance access must review the current immutable active version.");
    }
    const packageValue = await this.package(pluginId, request.contentSha256);
    const normalized = this.normalizeAccess(request, packageValue);
    return this.database((database) => {
      database.exec("BEGIN IMMEDIATE");
      try {
        const current = this.entry(database, pluginId);
        if (current.activeContentSha256 !== request.contentSha256) {
          throw new Error("Plugin instance access lost its reviewed active version before publication.");
        }
        const revision = this.nextRevision(database), now = new Date().toISOString();
        database.prepare(`
          INSERT INTO plugin_instances
            (brain_id, plugin_id, content_sha256, enabled, capabilities_json, tool_permissions_json, revision, updated_at)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(brain_id, plugin_id) DO UPDATE SET
            content_sha256 = excluded.content_sha256,
            enabled = excluded.enabled,
            capabilities_json = excluded.capabilities_json,
            tool_permissions_json = excluded.tool_permissions_json,
            revision = excluded.revision,
            updated_at = excluded.updated_at
        `).run(brainId, pluginId, request.contentSha256, request.enabled ? 1 : 0,
          canonical(normalized.capabilities), canonical(normalized.toolPermissions), revision, now);
        if (request.enabled) {
          this.ensureSemanticRow(database, brainId, pluginId, request.contentSha256,
            semanticExperienceSha256!, now);
        }
        const result = this.instanceInDatabase(database, brainId, pluginId)!;
        database.exec("COMMIT");
        return result;
      } catch (error) { database.exec("ROLLBACK"); throw error; }
    });
  }

  private instanceInDatabase(database: DatabaseSync, brainId: string, pluginId: string): PluginInstanceAccess | null {
    const row = database.prepare(`
      SELECT brain_id, plugin_id, content_sha256, enabled, capabilities_json,
        tool_permissions_json, revision, updated_at
      FROM plugin_instances WHERE brain_id = ? AND plugin_id = ?
    `).get(brainId, pluginId) as Record<string, unknown> | undefined;
    if (!row) return null;
    const capabilities = JSON.parse(String(row.capabilities_json)) as PluginCapabilitySelection[];
    const toolPermissions = JSON.parse(String(row.tool_permissions_json)) as PluginInstanceAccess["toolPermissions"];
    return { brainId: String(row.brain_id), pluginId: String(row.plugin_id), contentSha256: String(row.content_sha256),
      enabled: Number(row.enabled) === 1, capabilities, toolPermissions,
      revision: Number(row.revision), updatedAt: String(row.updated_at) };
  }

  async instance(brainId: string, pluginId: string): Promise<PluginInstanceAccess | null> {
    await this.initialize();
    const id = safeBrainId(brainId), plugin = safePluginId(pluginId);
    return this.database((database) => this.instanceInDatabase(database, id, plugin));
  }

  async instances(brainId?: string): Promise<PluginInstanceAccess[]> {
    await this.initialize();
    const selectedBrain = brainId === undefined ? undefined : safeBrainId(brainId);
    return this.database((database) => {
      const rows = (selectedBrain === undefined
        ? database.prepare("SELECT brain_id, plugin_id FROM plugin_instances ORDER BY brain_id, plugin_id").all()
        : database.prepare("SELECT brain_id, plugin_id FROM plugin_instances WHERE brain_id = ? ORDER BY plugin_id").all(selectedBrain)
      ) as unknown as Array<{ brain_id: string; plugin_id: string }>;
      return rows.map((row) => this.instanceInDatabase(database, row.brain_id, row.plugin_id)!).filter(Boolean);
    });
  }

  private portableSelection(row: PortableSelectionRow): PluginPortableSelection {
    if (digest(row.payload_json) !== row.payload_sha256) {
      throw new Error("Dormant plugin selection checksum failed.");
    }
    const value = JSON.parse(row.payload_json) as PluginPortableSelection;
    if (!value || value.pluginId !== row.plugin_id || value.contentSha256 !== row.content_sha256 ||
        value.sourceBrainId !== row.source_brain_id || value.recordedAt !== row.recorded_at ||
        !SAFE_ID.test(value.sourceBrainId) || !SHA256.test(value.contentSha256) ||
        !["source-review", "external-dormant", "merge-parent-dormant"].includes(value.authority) ||
        !value.reviewedAccess || value.reviewedAccess.pluginId !== value.pluginId ||
        value.reviewedAccess.contentSha256 !== value.contentSha256) {
      throw new Error("Dormant plugin selection identity is invalid.");
    }
    return value;
  }

  async storeDormantSelection(brainId: string, selection: PluginPortableSelection): Promise<void> {
    await this.initialize();
    const target = safeBrainId(brainId), plugin = safePluginId(selection.pluginId);
    if (!SHA256.test(selection.contentSha256) || !SAFE_ID.test(selection.sourceBrainId)) {
      throw new Error("Portable plugin selection identity is invalid.");
    }
    await this.package(plugin, selection.contentSha256);
    if (!Number.isFinite(Date.parse(selection.recordedAt))) throw new Error("Portable plugin selection time is invalid.");
    const recordedAt = selection.recordedAt;
    const normalized: PluginPortableSelection = structuredClone({ ...selection, pluginId: plugin,
      contentSha256: selection.contentSha256.toLowerCase(), recordedAt });
    const payload = canonical(normalized);
    this.database((database) => database.prepare(`
      INSERT INTO plugin_portable_selections
        (brain_id,plugin_id,content_sha256,source_brain_id,payload_json,payload_sha256,recorded_at)
      VALUES(?,?,?,?,?,?,?)
      ON CONFLICT(brain_id,source_brain_id,plugin_id,content_sha256) DO UPDATE SET
        payload_json=excluded.payload_json,
        payload_sha256=excluded.payload_sha256,recorded_at=excluded.recorded_at
    `).run(target, plugin, normalized.contentSha256, normalized.sourceBrainId,
      payload, digest(payload), recordedAt));
  }

  async dormantSelections(brainId: string): Promise<PluginPortableSelection[]> {
    await this.initialize();
    const target = safeBrainId(brainId);
    return this.database((database) => (database.prepare(`SELECT * FROM plugin_portable_selections
      WHERE brain_id=? ORDER BY plugin_id,content_sha256,source_brain_id`).all(target) as unknown as PortableSelectionRow[])
      .map((row) => this.portableSelection(row)));
  }

  async semanticLearningRecords(brainId: string): Promise<PluginSemanticLearningRecord[]> {
    await this.initialize();
    const target = safeBrainId(brainId);
    return this.database((database) => (database.prepare(`SELECT * FROM plugin_semantic_learning
      WHERE brain_id=? ORDER BY plugin_id,content_sha256`).all(target) as unknown as SemanticLearningRow[])
      .map((row) => this.semanticRecord(row)));
  }

  async copySemanticLearning(
    sourceBrainId: string,
    targetBrainId: string,
    pluginId: string,
    contentSha256: string
  ): Promise<void> {
    await this.initialize();
    const source = safeBrainId(sourceBrainId), target = safeBrainId(targetBrainId), plugin = safePluginId(pluginId);
    if (!SHA256.test(contentSha256)) throw new Error("Plugin semantic copy checksum is invalid.");
    this.database((database) => {
      const row = database.prepare(`SELECT * FROM plugin_semantic_learning
        WHERE brain_id=? AND plugin_id=? AND content_sha256=?`).get(source, plugin, contentSha256) as unknown as SemanticLearningRow | undefined;
      if (!row) return;
      database.prepare(`INSERT INTO plugin_semantic_learning
        (brain_id,plugin_id,content_sha256,experience_sha256,state,attempt,training_source_id,error,updated_at)
        VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(brain_id,plugin_id,content_sha256) DO UPDATE SET
          experience_sha256=excluded.experience_sha256,state=excluded.state,attempt=excluded.attempt,
          training_source_id=excluded.training_source_id,error=excluded.error,updated_at=excluded.updated_at`)
        .run(target, plugin, contentSha256, row.experience_sha256,
          row.state === "complete" ? "complete" : "pending", row.attempt,
          row.state === "complete" ? row.training_source_id : null,
          row.state === "complete" ? null : "Copied as dormant pending plugin learning.", new Date().toISOString());
    });
  }

  async removeBrainState(brainId: string): Promise<void> {
    await this.initialize();
    const target = safeBrainId(brainId);
    this.database((database) => {
      database.exec("BEGIN IMMEDIATE");
      try {
        database.prepare("DELETE FROM plugin_instances WHERE brain_id=?").run(target);
        database.prepare("DELETE FROM plugin_semantic_learning WHERE brain_id=?").run(target);
        database.prepare("DELETE FROM plugin_portable_selections WHERE brain_id=?").run(target);
        this.nextRevision(database);
        database.exec("COMMIT");
      } catch (error) { database.exec("ROLLBACK"); throw error; }
    });
  }

  private semanticRecord(row: SemanticLearningRow): PluginSemanticLearningRecord {
    return {
      brainId: row.brain_id,
      pluginId: row.plugin_id,
      contentSha256: row.content_sha256,
      experienceSha256: row.experience_sha256,
      state: row.state,
      attempt: row.attempt,
      ...(row.training_source_id ? { trainingSourceId: row.training_source_id } : {}),
      ...(row.error ? { error: row.error } : {}),
      updatedAt: row.updated_at
    };
  }

  async recoverSemanticLearning(): Promise<void> {
    await this.initialize();
    this.database((database) => database.prepare(`
      UPDATE plugin_semantic_learning SET state = 'pending',
        error = 'The prior host session ended before a neural receipt was confirmed.', updated_at = ?
      WHERE state = 'running'
    `).run(new Date().toISOString()));
  }

  async ensureSemanticLearning(
    brainId: string,
    pluginId: string,
    contentSha256: string,
    experienceSha256: string
  ): Promise<PluginSemanticLearningRecord> {
    await this.initialize();
    const brain = safeBrainId(brainId), plugin = safePluginId(pluginId);
    if (!SHA256.test(contentSha256) || !SHA256.test(experienceSha256)) throw new Error("Invalid plugin semantic-learning identity.");
    await this.package(plugin, contentSha256);
    return this.database((database) => {
      const now = new Date().toISOString();
      return this.ensureSemanticRow(database, brain, plugin, contentSha256, experienceSha256, now);
    });
  }

  async pendingSemanticLearning(brainId?: string): Promise<PluginSemanticLearningRecord[]> {
    await this.initialize();
    const brain = brainId === undefined ? undefined : safeBrainId(brainId);
    return this.database((database) => ((brain === undefined
      ? database.prepare("SELECT * FROM plugin_semantic_learning WHERE state = 'pending' ORDER BY updated_at").all()
      : database.prepare("SELECT * FROM plugin_semantic_learning WHERE state = 'pending' AND brain_id = ? ORDER BY updated_at").all(brain)
    ) as unknown as SemanticLearningRow[]).map((row) => this.semanticRecord(row)));
  }

  async semanticLearning(
    brainId: string,
    pluginId: string,
    contentSha256: string
  ): Promise<PluginSemanticLearningRecord | null> {
    await this.initialize();
    if (!SHA256.test(contentSha256)) throw new Error("Invalid plugin semantic-learning checksum.");
    return this.database((database) => {
      const row = database.prepare(`SELECT * FROM plugin_semantic_learning
        WHERE brain_id = ? AND plugin_id = ? AND content_sha256 = ?`).get(
        safeBrainId(brainId), safePluginId(pluginId), contentSha256
      ) as unknown as SemanticLearningRow | undefined;
      return row ? this.semanticRecord(row) : null;
    });
  }

  async claimSemanticLearning(record: PluginSemanticLearningRecord): Promise<PluginSemanticLearningRecord | null> {
    await this.initialize();
    return this.database((database) => {
      database.exec("BEGIN IMMEDIATE");
      try {
        const current = database.prepare(`SELECT * FROM plugin_semantic_learning
          WHERE brain_id = ? AND plugin_id = ? AND content_sha256 = ?`).get(
          safeBrainId(record.brainId), safePluginId(record.pluginId), record.contentSha256
        ) as unknown as SemanticLearningRow | undefined;
        if (!current || current.state !== "pending" || current.experience_sha256 !== record.experienceSha256) {
          database.exec("COMMIT"); return null;
        }
        const now = new Date().toISOString();
        database.prepare(`UPDATE plugin_semantic_learning SET state = 'running', attempt = attempt + 1,
          error = NULL, updated_at = ? WHERE brain_id = ? AND plugin_id = ? AND content_sha256 = ?`)
          .run(now, current.brain_id, current.plugin_id, current.content_sha256);
        const claimed = database.prepare(`SELECT * FROM plugin_semantic_learning
          WHERE brain_id = ? AND plugin_id = ? AND content_sha256 = ?`).get(
          current.brain_id, current.plugin_id, current.content_sha256
        ) as unknown as SemanticLearningRow;
        database.exec("COMMIT"); return this.semanticRecord(claimed);
      } catch (error) { database.exec("ROLLBACK"); throw error; }
    });
  }

  async failSemanticLearning(record: PluginSemanticLearningRecord, error: string): Promise<void> {
    await this.initialize();
    this.database((database) => database.prepare(`UPDATE plugin_semantic_learning
      SET state = 'pending', error = ?, updated_at = ?
      WHERE brain_id = ? AND plugin_id = ? AND content_sha256 = ?
        AND experience_sha256 = ? AND state = 'running' AND attempt = ?`)
      .run(error.replace(/\0/g, "").slice(0, 2_000), new Date().toISOString(), record.brainId,
        record.pluginId, record.contentSha256, record.experienceSha256, record.attempt));
  }

  async completeSemanticLearning(record: PluginSemanticLearningRecord, trainingSourceId: string): Promise<void> {
    await this.initialize();
    if (!SAFE_ID.test(trainingSourceId)) throw new Error("Plugin semantic-learning source id is invalid.");
    const changed = this.database((database) => database.prepare(`UPDATE plugin_semantic_learning
      SET state = 'complete', training_source_id = ?, error = NULL, updated_at = ?
      WHERE brain_id = ? AND plugin_id = ? AND content_sha256 = ?
        AND experience_sha256 = ? AND state = 'running' AND attempt = ?`)
      .run(trainingSourceId, new Date().toISOString(), record.brainId, record.pluginId,
        record.contentSha256, record.experienceSha256, record.attempt).changes);
    if (changed !== 1) throw new Error("Plugin semantic-learning receipt lost its exact queue owner.");
  }

  async exportFile(
    pluginId: string,
    destination: string,
    contentSha256?: string,
    replaceReviewed = false
  ): Promise<string> {
    const packageValue = await this.package(pluginId, contentSha256);
    const prepared = canonicalPluginPackage(packageValue);
    await mkdir(dirname(destination), { recursive: true });
    const temporary = join(dirname(destination), `.${basename(destination)}.${randomUUID()}.tmp`);
    const backup = join(dirname(destination), `.${basename(destination)}.${randomUUID()}.previous`);
    const handle = await open(temporary, "wx", 0o600);
    let movedExisting = false;
    try {
      await handle.writeFile(`${prepared.json}\n`, "utf8");
      await handle.sync();
      await handle.close();
      if (replaceReviewed) {
        try {
          const existing = await lstat(destination);
          if (!existing.isFile() || existing.isSymbolicLink()) throw new Error("Reviewed plugin export target is not a regular file.");
          await rename(destination, backup);
          movedExisting = true;
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
        }
      }
      if (replaceReviewed) {
        // The save dialog authorized this exact destination. Rename is atomic
        // on the selected volume and also works on filesystems without links.
        await rename(temporary, destination);
      } else {
        // Programmatic exports do not overwrite an unreviewed destination.
        await link(temporary, destination);
      }
      if (movedExisting) {
        await rm(backup, { force: true });
        movedExisting = false;
      }
    } catch (error) {
      if (movedExisting) {
        await rm(destination, { force: true }).catch(() => undefined);
        await rename(backup, destination);
        movedExisting = false;
      }
      throw error;
    } finally {
      await handle.close().catch(() => undefined);
      await rm(temporary, { force: true });
      if (!movedExisting) await rm(backup, { force: true });
    }
    return destination;
  }
}
