import { createHash } from "node:crypto";
import Ajv, { type ValidateFunction } from "ajv";
import Ajv2019 from "ajv/dist/2019.js";
import Ajv2020 from "ajv/dist/2020.js";
import type { PluginRuntimeContribution, PluginRuntimeModuleDefinition, PluginRuntimeToolDefinition } from "../shared/pluginRuntime";

export const RUNTIME_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
export const RUNTIME_LOCAL_ID = /^[a-z][a-z0-9-]{0,63}$/;
export const RUNTIME_SHA = /^[a-f0-9]{64}$/;
export function runtimeHash(value: string): string { return createHash("sha256").update(value).digest("hex"); }
export function runtimeJson(value: unknown): string {
  const visit = (item: unknown, depth = 0): unknown => {
    if (depth > 128) throw new Error("Runtime structured data exceeds its stack-transfer depth.");
    if (item === null || typeof item === "string" || typeof item === "boolean") return item;
    if (typeof item === "number" && Number.isFinite(item)) return item;
    if (Array.isArray(item)) return item.map(value => visit(value, depth + 1));
    const source = runtimeObject(item); const output: Record<string, unknown> = {};
    for (const key of Object.keys(source).sort()) { if (["__proto__", "prototype", "constructor"].includes(key) || key.includes("\0")) throw new Error("Unsafe runtime structured key.");
      output[key] = visit(source[key], depth + 1); }
    return output;
  };
  return JSON.stringify(visit(value));
}
export function runtimeObject(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Runtime value must be a structured object.");
  return value as Record<string, unknown>;
}
export function runtimeKeys(value: Record<string, unknown>, allowed: string[]): void {
  if (Object.keys(value).some(key => !allowed.includes(key))) throw new Error("Runtime declaration contains unsupported fields.");
}
export function runtimeId(value: unknown): string {
  if (typeof value !== "string" || !RUNTIME_ID.test(value)) throw new Error("Invalid runtime identity."); return value;
}
function localId(value: unknown): string { if (typeof value !== "string" || !RUNTIME_LOCAL_ID.test(value)) throw new Error("Invalid local runtime identity."); return value; }
function text(value: unknown): string { if (typeof value !== "string" || !value.trim() || value.includes("\0")) throw new Error("Invalid runtime declaration text."); return value; }
function names(value: unknown): string[] {
  if (!Array.isArray(value)) throw new Error("Runtime declaration needs an array.");
  const output = value.map(runtimeId); if (new Set(output).size !== output.length) throw new Error("Duplicate runtime declaration."); return output;
}
export function runtimeCss(value: unknown): string {
  const css = text(value);
  // Deliberately narrower than arbitrary CSS: escapes/comments/@rules can
  // disguise URL tokens, while image-set accepts quoted resources without url().
  // Studio CSP remains the separate network boundary; this is not a CSS parser.
  const safeRules = new Set(["media", "supports", "container", "layer", "keyframes", "-webkit-keyframes", "scope"]);
  if (/[\\]|\/\*|\*\/|(?:url|src|image|image-set|-webkit-image-set|cross-fade|element|paint|expression)\s*\(|javascript\s*:|behavior\s*:|-moz-binding\s*:/i.test(css) ||
      [...css.matchAll(/@([a-z-]+)/gi)].some(match => !safeRules.has(match[1]!.toLowerCase())) || /@(?![a-z-])/i.test(css)) throw new Error("Runtime styles cannot load resources or execute code.");
  return css;
}
export function runtimeHtml(value: unknown): string {
  const html = text(value);
  if (/<\/?(?:script|style|iframe|frame|object|embed|form|link|meta|base)\b/i.test(html) || /<[^>]*[\s/]on[a-z0-9_-]+\s*=/i.test(html) ||
      /<[^>]*[\s/](?:src|srcset|href|xlink:href|poster|data|action|formaction)\s*=/i.test(html)) throw new Error("Runtime markup must be inert; callbacks belong to the isolated module.");
  return html;
}
export function normalizePluginRuntimeModule(value: unknown): PluginRuntimeModuleDefinition {
  const row = runtimeObject(value); runtimeKeys(row, ["id", "title", "description", "script", "events", "ui", "dynamicTools"]);
  const ui = runtimeObject(row.ui); runtimeKeys(ui, ["kinds", "slots"]);
  if (!Array.isArray(ui.kinds) || ui.kinds.some(kind => !["style", "menu", "button", "panel"].includes(String(kind))) || new Set(ui.kinds).size !== ui.kinds.length || typeof row.dynamicTools !== "boolean") throw new Error("Invalid runtime contribution/tool scope.");
  const script = text(row.script).replace(/\r\n?/g, "\n"); if (/<\/script\b/i.test(script)) throw new Error("Runtime script cannot terminate its CSP-bound element.");
  return { id: localId(row.id), title: text(row.title), description: text(row.description), script, events: names(row.events),
    ui: { kinds: ui.kinds as PluginRuntimeModuleDefinition["ui"]["kinds"], slots: names(ui.slots) }, dynamicTools: row.dynamicTools };
}
export function runtimeModuleSha256(value: PluginRuntimeModuleDefinition): string {
  return runtimeHash(runtimeJson(value));
}
export function validateRuntimeTool(value: unknown): PluginRuntimeToolDefinition {
  const row = runtimeObject(value); runtimeKeys(row, ["id", "title", "description", "actions"]);
  if (!Array.isArray(row.actions) || !row.actions.length) throw new Error("Runtime tool must have typed actions.");
  const actions = row.actions.map(value => { const action = runtimeObject(value); runtimeKeys(action, ["id", "description", "inputSchema"]);
    const inputSchema = runtimeObject(action.inputSchema); runtimeJson(inputSchema);
    return { id: runtimeId(action.id), description: text(action.description), inputSchema }; });
  if (new Set(actions.map(value => value.id)).size !== actions.length) throw new Error("Runtime tool has duplicate actions.");
  return { id: localId(row.id), title: text(row.title), description: text(row.description), actions };
}
/** Same supported local JSON Schema dialects as MCP; no remote ref fetching,
 * mutation, coercion, defaults, or permission authority from a schema. */
export function compileRuntimeActionInput(schema: Record<string, unknown>): ValidateFunction<unknown> {
  if (schema.type !== "object") throw new Error("Runtime action schema needs an object root.");
  const dialect = schema.$schema, options = { strict: false, validateFormats: false, coerceTypes: false, removeAdditional: false, useDefaults: false };
  let compiler: Ajv | Ajv2019 | Ajv2020;
  if (dialect === undefined || dialect === "https://json-schema.org/draft/2020-12/schema" || dialect === "https://json-schema.org/draft/2020-12/schema#") compiler = new Ajv2020(options);
  else if (dialect === "https://json-schema.org/draft/2019-09/schema" || dialect === "https://json-schema.org/draft/2019-09/schema#") compiler = new Ajv2019(options);
  else if (["http://json-schema.org/draft-07/schema#", "https://json-schema.org/draft-07/schema#", "http://json-schema.org/draft-07/schema", "https://json-schema.org/draft-07/schema"].includes(String(dialect))) compiler = new Ajv(options);
  else throw new Error("Runtime action uses an unsupported local JSON Schema dialect.");
  try { const validate = compiler.compile(schema); if ((validate as { $async?: boolean }).$async) throw new Error("Async action schema is unsupported."); return validate; }
  catch { throw new Error("Runtime action schema is invalid, asynchronous, or has an unresolved local reference."); }
}
export function validateRuntimeContribution(value: unknown, module: PluginRuntimeModuleDefinition): PluginRuntimeContribution {
  const row = runtimeObject(value); runtimeKeys(row, ["id", "kind", "slot", "title", "css", "html", "event"]);
  const kind = String(row.kind) as PluginRuntimeContribution["kind"], slot = runtimeId(row.slot);
  if (!module.ui.kinds.includes(kind) || !module.ui.slots.includes(slot)) throw new Error("Runtime UI contribution is outside its explicitly reviewed manifest scope.");
  return { id: localId(row.id), kind, slot, ...(row.title === undefined ? {} : { title: text(row.title) }),
    ...(row.css === undefined ? {} : { css: runtimeCss(row.css) }), ...(row.html === undefined ? {} : { html: runtimeHtml(row.html) }),
    ...(row.event === undefined ? {} : { event: runtimeId(row.event) }) };
}
