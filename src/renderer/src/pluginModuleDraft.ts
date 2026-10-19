import { OMNI_PLUGIN_FORMAT, OMNI_PLUGIN_FORMAT_VERSION, PLUGIN_TRAINING_PARSER_CAPABILITY, type OmniPluginPackage } from "../../shared/plugins";
import type { PluginRuntimeModuleDefinition } from "../../shared/pluginRuntime";

export const PLUGIN_MODULE_STARTERS = [
  { id: "feature", title: "Feature & menu", description: "A toolbar button, expandable menu and panel with real event callbacks." },
  { id: "style", title: "App styling", description: "Editable CSS applied while this module is active." },
  { id: "tool", title: "Custom tool", description: "A typed text-statistics tool implemented by the module." },
  { id: "parser", title: "Training format", description: "A streaming UTF-8 text-to-JSONL parser for a new file extension." }
] as const;
export type PluginModuleStarter = typeof PLUGIN_MODULE_STARTERS[number]["id"];

const STYLE_HEADER = "// Omni editable style\nconst css = ";
const DEFAULT_CSS = ".plugin-runtime-panel { border-radius: 18px; }\n@media (max-width: 700px) { .plugin-runtime-panel { padding: 12px; } }";
export function pluginStyleModuleScript(css: string): string {
  return `${STYLE_HEADER}${JSON.stringify(css)};\nconst runtime = globalThis.omniRuntime;\nruntime.onActivate(() => runtime.ui.put({ id: 'style', kind: 'style', slot: 'tools', css }));`;
}
/** Read only this editor's explicit style envelope; arbitrary JS is never evaluated. */
export function pluginStyleModuleCss(script: string): string | undefined {
  if (!script.startsWith(STYLE_HEADER)) return undefined;
  const boundary = script.indexOf(";\nconst runtime =", STYLE_HEADER.length);
  if (boundary < 0) return undefined;
  try { const value: unknown = JSON.parse(script.slice(STYLE_HEADER.length, boundary)); return typeof value === "string" ? value : undefined; }
  catch { return undefined; }
}
export function replacePluginStyleModuleCss(script: string, css: string): string {
  if (pluginStyleModuleCss(script) === undefined) throw new Error("This module does not have an editable style envelope.");
  return `${STYLE_HEADER}${JSON.stringify(css)}${script.slice(script.indexOf(";\nconst runtime =", STYLE_HEADER.length))}`;
}

export function newPluginAuthoringPackage(): OmniPluginPackage {
  return { format: OMNI_PLUGIN_FORMAT, formatVersion: OMNI_PLUGIN_FORMAT_VERSION, manifest: {
    id: "my.plugin", version: "1.0.0", name: "My plugin", description: "Describe this feature package.",
    author: "Local author", license: "All rights reserved", documentation: "# My plugin\n\nDescribe what this package adds and every permission it needs.",
    capabilities: [], tools: [], panels: [], interactivePanels: [], runtimeModules: [], themes: [], trainingFormats: []
  } };
}

/** Editor shape guard only. Main's canonical validation remains authoritative. */
export function readPluginAuthoringPackage(text: string): OmniPluginPackage {
  const value: unknown = JSON.parse(text);
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Enter a plugin package object.");
  const item = value as Record<string, unknown>, manifest = item.manifest;
  if (item.format !== OMNI_PLUGIN_FORMAT || item.formatVersion !== OMNI_PLUGIN_FORMAT_VERSION ||
      !manifest || typeof manifest !== "object" || Array.isArray(manifest)) throw new Error("Use a supported Omni plugin package.");
  const fields = manifest as Record<string, unknown>;
  for (const key of ["id", "version", "name", "description", "author", "license", "documentation"])
    if (typeof fields[key] !== "string") throw new Error(`Package ${key} must be text.`);
  for (const key of ["capabilities", "tools", "panels", "themes", "trainingFormats"])
    if (!Array.isArray(fields[key])) throw new Error(`Package ${key} must be a list.`);
  if ((fields.capabilities as unknown[]).some(capability => !capability || typeof capability !== "object" ||
      typeof (capability as Record<string, unknown>).toolId !== "string" || !Array.isArray((capability as Record<string, unknown>).actions) ||
      ((capability as Record<string, unknown>).actions as unknown[]).some(action => typeof action !== "string")))
    throw new Error("Host capabilities need a tool ID and action names.");
  if ((fields.trainingFormats as unknown[]).some(format => !format || typeof format !== "object" ||
      typeof (format as Record<string, unknown>).id !== "string" || typeof (format as Record<string, unknown>).title !== "string" ||
      !Array.isArray((format as Record<string, unknown>).extensions) || !Array.isArray((format as Record<string, unknown>).mediaTypes) ||
      [...(format as { extensions: unknown[] }).extensions, ...(format as { mediaTypes: unknown[] }).mediaTypes].some(value => typeof value !== "string") ||
      ((format as Record<string, unknown>).parser !== "declaration-only" && (!(format as Record<string, unknown>).parser ||
        typeof (format as Record<string, unknown>).parser !== "object"))))
    throw new Error("Training formats need named extensions, media types and a parser declaration.");
  if (fields.runtimeModules !== undefined && (!Array.isArray(fields.runtimeModules) || fields.runtimeModules.some(module =>
    !module || typeof module !== "object" || typeof module.id !== "string" || typeof module.title !== "string" ||
    typeof module.description !== "string" || typeof module.script !== "string" || !Array.isArray(module.events) ||
    module.events.some((event: unknown) => typeof event !== "string") || typeof module.dynamicTools !== "boolean" ||
    !module.ui || !Array.isArray(module.ui.kinds) || module.ui.kinds.some((kind: unknown) => !["style", "menu", "button", "panel"].includes(String(kind))) ||
    !Array.isArray(module.ui.slots) || module.ui.slots.some((slot: unknown) => typeof slot !== "string"))))
    throw new Error("Runtime modules need valid text, event lists and UI declarations.");
  return value as OmniPluginPackage;
}

function starterModule(kind: PluginModuleStarter, id: string): PluginRuntimeModuleDefinition {
  const common = { id, events: [] as string[], ui: { kinds: [] as PluginRuntimeModuleDefinition["ui"]["kinds"], slots: [] as string[] }, dynamicTools: false };
  if (kind === "style") return { ...common, title: "App styling", description: "A local Studio stylesheet.",
    ui: { kinds: ["style"], slots: ["tools"] }, script: pluginStyleModuleScript(DEFAULT_CSS) };
  if (kind === "tool") return { ...common, title: "Text statistics", description: "A typed tool implemented locally in the feature runtime.", dynamicTools: true,
    script: `const runtime = globalThis.omniRuntime;
runtime.onActivate(() => runtime.registerTool({
  id: 'text-statistics', title: 'Text statistics', description: 'Count code points in supplied text.',
  actions: [{ id: 'measure', description: 'Measure supplied text.', inputSchema: {
    type: 'object', properties: { text: { type: 'string' } }, required: ['text'], additionalProperties: false
  } }]
}, { measure: ({ text }) => ({ characters: Array.from(text).length }) }));` };
  if (kind === "parser") return { ...common, title: "Text training format", description: "Stream UTF-8 text as JSONL without buffering an entire source.",
    script: `const runtime = globalThis.omniRuntime;
if (runtime.context.purpose === 'training-parser') {
  const decoder = new TextDecoder('utf-8', { fatal: true }), encoder = new TextEncoder();
  let pending = new Uint8Array(), offset = 0, finished = false;
  const drain = step => {
    if (offset < pending.length) {
      const bytes = pending.subarray(offset, offset + step.maxOutputBytes);
      offset += bytes.length;
      let binary = ''; for (const byte of bytes) binary += String.fromCharCode(byte);
      return { state: 'output', outputBase64: btoa(binary) };
    }
    return { state: finished ? 'done' : 'need-input' };
  };
  runtime.registerParser({
    push: step => {
      if (offset < pending.length) throw new Error('Drain the previous output first.');
      const input = Uint8Array.from(atob(step.inputBase64 || ''), c => c.charCodeAt(0));
      const text = decoder.decode(input, { stream: !step.final });
      finished = step.final === true;
      pending = text ? encoder.encode(JSON.stringify({ text }) + '\\n') : new Uint8Array();
      offset = 0; return drain(step);
    }, drain
  });
}` };
  return { ...common, title: "My feature", description: "Buttons, menus and panels with isolated callbacks.", events: ["chat.action"],
    ui: { kinds: ["button", "menu", "panel"], slots: ["chat.toolbar", "tools"] },
    script: `const runtime = globalThis.omniRuntime;
const show = text => runtime.ui.put({ id: 'panel', kind: 'panel', slot: 'tools', title: 'My feature', html: '<p>' + text + '</p>' });
runtime.onActivate(async () => {
  await runtime.ui.put({ id: 'button', kind: 'button', slot: 'chat.toolbar', title: 'My feature', event: 'open' });
  await runtime.ui.put({ id: 'menu', kind: 'menu', slot: 'tools', title: 'Open my feature', event: 'open' });
});
runtime.on('ui:open', () => show('Your feature callback ran.'));
runtime.on('chat.action', () => show('A chat action event reached this module.'));` };
}

export function addPluginModuleStarter(packageValue: OmniPluginPackage, kind: PluginModuleStarter): { packageValue: OmniPluginPackage; moduleId: string } {
  const modules = packageValue.manifest.runtimeModules ?? [], used = new Set(modules.map(module => module.id));
  let id: string = kind; for (let index = 2; used.has(id); index++) id = `${kind}-${index}`;
  const module = starterModule(kind, id);
  const capabilities = packageValue.manifest.capabilities.map(value => ({ ...value, actions: [...value.actions] }));
  const formats = [...packageValue.manifest.trainingFormats];
  if (kind === "parser") {
    const formatIds = new Set(formats.map(format => format.id));
    let formatId = id; for (let index = 2; formatIds.has(formatId); index++) formatId = `${id}-${index}`;
    formats.push({ id: formatId, title: "Custom text format", description: "Read UTF-8 text in bounded streaming pages.",
      extensions: [formatId === "parser" ? "omnitext" : formatId], mediaTypes: [], parser: { kind: "runtime-stream", moduleId: id, output: "jsonl" } });
    const capability = capabilities.find(value => value.toolId === PLUGIN_TRAINING_PARSER_CAPABILITY);
    if (capability) capability.actions = [...new Set([...capability.actions, formatId])];
    else capabilities.push({ toolId: PLUGIN_TRAINING_PARSER_CAPABILITY, actions: [formatId], reason: "Parse this package's custom training formats." });
  }
  return { moduleId: id, packageValue: { ...packageValue, manifest: { ...packageValue.manifest,
    runtimeModules: [...modules, module], capabilities, trainingFormats: formats } } };
}

export function replacePluginRuntimeModule(packageValue: OmniPluginPackage, moduleId: string, next: PluginRuntimeModuleDefinition): OmniPluginPackage {
  if (next.id !== moduleId || !packageValue.manifest.runtimeModules?.some(module => module.id === moduleId)) throw new Error("The selected module changed.");
  return { ...packageValue, manifest: { ...packageValue.manifest,
    runtimeModules: packageValue.manifest.runtimeModules.map(module => module.id === moduleId ? next : module) } };
}
export function removePluginRuntimeModule(packageValue: OmniPluginPackage, moduleId: string): OmniPluginPackage {
  const removedFormats = new Set(packageValue.manifest.trainingFormats.filter(format => format.parser !== "declaration-only" &&
    format.parser.kind === "runtime-stream" && format.parser.moduleId === moduleId).map(format => format.id));
  return { ...packageValue, manifest: { ...packageValue.manifest,
    runtimeModules: packageValue.manifest.runtimeModules?.filter(module => module.id !== moduleId),
    trainingFormats: packageValue.manifest.trainingFormats.filter(format => !removedFormats.has(format.id)),
    capabilities: packageValue.manifest.capabilities.map(capability => capability.toolId === PLUGIN_TRAINING_PARSER_CAPABILITY
      ? { ...capability, actions: capability.actions.filter(action => !removedFormats.has(action)) } : capability)
      .filter(capability => capability.toolId !== PLUGIN_TRAINING_PARSER_CAPABILITY || capability.actions.length > 0)
  } };
}
