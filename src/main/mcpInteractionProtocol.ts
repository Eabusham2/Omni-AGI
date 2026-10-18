import { Worker } from "node:worker_threads";
import { createHash } from "node:crypto";
import type { McpElicitationRequest, McpFormProperty, McpFormSchema, McpFormValues, McpRequestOrigin, McpSamplingMediaReview } from "../shared/mcpInteractions";
import type { NativeMcpSamplingInput } from "../shared/nativeMcpSampling";
import { nativeMcpSamplingInputJson, nativeMcpSamplingMediaMetadata } from "./nativeMcpSampling";

export class McpHostRequestError extends Error {
  constructor(readonly code: number, message: string) { super(message); }
}
function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new McpHostRequestError(-32602, `Invalid MCP ${label}.`);
  return value as Record<string, unknown>;
}
function text(value: unknown, label: string): string {
  if (typeof value !== "string" || value.includes("\0")) throw new McpHostRequestError(-32602, `Invalid MCP ${label}.`);
  return value;
}
function integer(value: unknown): boolean { return Number.isSafeInteger(value) && Number(value) >= 0; }
const SENSITIVE = /(?:pass(?:word|phrase)|api[ _-]?key|(?:access|refresh|auth)[ _-]?token|secret|private[ _-]?key|card[ _-]?(?:number|cvv)|\bcvv\b|credential)/i;

function choices(value: unknown, titled: boolean): void {
  if (!Array.isArray(value) || !value.length) throw new McpHostRequestError(-32602, "MCP form choices must be a nonempty list.");
  const unique = new Set<string>();
  for (const option of value) {
    const item = titled ? object(option, "form choice") : undefined;
    const label = text(item ? item.const : option, "form choice value");
    if (unique.has(label)) throw new McpHostRequestError(-32602, "MCP form choices must have unique values.");
    unique.add(label);
    if (item && (Object.keys(item).some(key => !["const", "title"].includes(key)) || item.title !== undefined && typeof item.title !== "string")) throw new McpHostRequestError(-32602, "Invalid titled MCP form choice.");
  }
}

export function mcpFormSchema(value: unknown): McpFormSchema {
  const root = object(value, "form schema");
  if (root.type !== "object" || Object.keys(root).some(key => !["type", "properties", "required", "additionalProperties", "$schema", "title", "description"].includes(key))) throw new McpHostRequestError(-32602, "MCP elicitation supports flat form schemas, not nested or executable schemas.");
  const properties = object(root.properties, "form properties");
  for (const [name, value] of Object.entries(properties)) {
    const property = object(value, "form field");
    if (SENSITIVE.test(`${name} ${property.title ?? ""} ${property.description ?? ""} ${property.format ?? ""}`)) throw new McpHostRequestError(-32602, "Sensitive credentials must use URL elicitation, not an in-app form.");
    if (!["string", "number", "integer", "boolean", "array"].includes(String(property.type))) throw new McpHostRequestError(-32602, "MCP form fields must be primitive values or string-enum selections.");
    const allowed = ["type", "title", "description", "default", ...(property.type === "string" ? ["minLength", "maxLength", "pattern", "format", "enum", "oneOf"] : property.type === "array" ? ["items", "minItems", "maxItems"] : property.type === "number" || property.type === "integer" ? ["minimum", "maximum"] : [])];
    if (Object.keys(property).some(key => !allowed.includes(key))) throw new McpHostRequestError(-32602, "MCP form field uses an unsupported schema property.");
    for (const key of ["title", "description", "pattern"]) if (property[key] !== undefined) text(property[key], `form ${key}`);
    for (const key of ["minLength", "maxLength", "minItems", "maxItems"]) if (property[key] !== undefined && !integer(property[key])) throw new McpHostRequestError(-32602, "MCP form bounds must be nonnegative safe integers.");
    for (const key of ["minimum", "maximum"]) if (property[key] !== undefined && (typeof property[key] !== "number" || !Number.isFinite(property[key]))) throw new McpHostRequestError(-32602, "MCP numeric bounds must be finite.");
    if (property.format !== undefined && !["email", "uri", "date", "date-time"].includes(String(property.format))) throw new McpHostRequestError(-32602, "MCP form format is unsupported.");
    if (property.enum !== undefined) choices(property.enum, false);
    if (property.oneOf !== undefined) choices(property.oneOf, true);
    if (property.enum !== undefined && property.oneOf !== undefined) throw new McpHostRequestError(-32602, "MCP form field has conflicting enum definitions.");
    if (property.type === "array") {
      const items = object(property.items, "multi-select choices");
      if (items.type !== undefined && items.type !== "string" || Object.keys(items).some(key => !["type", "enum", "anyOf"].includes(key)) || (items.enum === undefined) === (items.anyOf === undefined)) throw new McpHostRequestError(-32602, "MCP multi-select needs string enum choices.");
      choices(items.enum ?? items.anyOf, items.anyOf !== undefined);
    }
    for (const [minimum, maximum] of [["minLength", "maxLength"], ["minItems", "maxItems"], ["minimum", "maximum"]] as const) {
      if (property[minimum] !== undefined && property[maximum] !== undefined && Number(property[minimum]) > Number(property[maximum])) throw new McpHostRequestError(-32602, "MCP form bounds contradict each other.");
    }
  }
  if (root.required !== undefined && (!Array.isArray(root.required) || root.required.some(key => typeof key !== "string" || !Object.hasOwn(properties, key)) || new Set(root.required).size !== root.required.length)) throw new McpHostRequestError(-32602, "MCP form required fields are invalid.");
  return { type: "object", properties: structuredClone(properties) as Record<string, McpFormProperty>, ...(root.required ? { required: [...root.required as string[]] } : {}) };
}

export function mcpElicitation(value: unknown): McpElicitationRequest {
  const input = object(value, "elicitation request");
  const message = text(input.message, "elicitation message");
  if (input.mode === undefined || input.mode === "form") return { mode: "form", message, requestedSchema: mcpFormSchema(input.requestedSchema) };
  if (input.mode !== "url") throw new McpHostRequestError(-32602, "This MCP elicitation mode is unsupported.");
  const url = text(input.url, "elicitation URL"), elicitationId = text(input.elicitationId, "elicitation identity");
  let parsed: URL;
  try { parsed = new URL(url); } catch { throw new McpHostRequestError(-32602, "Invalid MCP elicitation URL."); }
  const loopback = ["localhost", "127.0.0.1", "[::1]", "::1"].includes(parsed.hostname);
  if (!elicitationId || parsed.username || parsed.password || parsed.protocol !== "https:" && !(parsed.protocol === "http:" && loopback)) throw new McpHostRequestError(-32602, "MCP elicitation URLs require HTTPS, except loopback development, and cannot embed credentials.");
  // Keep the original URL for human review. Never fetch metadata or inspect it.
  return { mode: "url", message, url, elicitationId };
}

async function matchesPattern(pattern: string, value: string): Promise<boolean> {
  // Untrusted regex runs outside Electron's main thread and can be terminated.
  const worker = new Worker("const {parentPort,workerData}=require('node:worker_threads');try{parentPort.postMessage({ok:new RegExp(workerData.pattern,'u').test(workerData.value)});}catch{parentPort.postMessage({error:true});}",
    { eval: true, workerData: { pattern, value }, resourceLimits: { maxOldGenerationSizeMb: 16, maxYoungGenerationSizeMb: 4, stackSizeMb: 1 } });
  try {
    return await new Promise<boolean>((resolve, reject) => {
      const timer = setTimeout(() => reject(new McpHostRequestError(-32602, "The server's form pattern could not be validated within its resource budget.")), 500);
      worker.once("message", result => { clearTimeout(timer); if (result.error) reject(new McpHostRequestError(-32602, "The server provided an invalid form pattern.")); else resolve(result.ok === true); });
      worker.once("error", error => { clearTimeout(timer); reject(error); });
    });
  } finally { await worker.terminate(); }
}

export async function mcpFormValues(schema: McpFormSchema, value: unknown): Promise<McpFormValues> {
  const values = object(value, "form response");
  if (Object.keys(values).some(key => !Object.hasOwn(schema.properties, key))) throw new McpHostRequestError(-32602, "The form response contains an undeclared field.");
  if (schema.required?.some(key => !Object.hasOwn(values, key))) throw new McpHostRequestError(-32602, "Complete the required form fields before sharing.");
  for (const [name, value] of Object.entries(values)) {
    const property = schema.properties[name]!;
    let valid = true;
    if (property.type === "string") {
      valid = typeof value === "string" && !value.includes("\0");
      if (typeof value === "string") {
        const length = [...value].length;
        valid &&= (property.minLength === undefined || length >= property.minLength) && (property.maxLength === undefined || length <= property.maxLength);
        const options = property.enum ?? property.oneOf?.map(item => item.const);
        valid &&= !options || options.includes(value);
        if (property.pattern && valid) valid = await matchesPattern(property.pattern, value);
        if (property.format === "email") valid &&= /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
        if (property.format === "uri") { try { valid &&= Boolean(new URL(value).protocol); } catch { valid = false; } }
        if (property.format === "date") valid &&= /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
        if (property.format === "date-time") valid &&= /^\d{4}-\d{2}-\d{2}T/.test(value) && Number.isFinite(Date.parse(value));
      }
    } else if (property.type === "boolean") valid = typeof value === "boolean";
    else if (property.type === "number" || property.type === "integer") valid = typeof value === "number" && Number.isFinite(value) && (property.type !== "integer" || Number.isSafeInteger(value)) && (property.minimum === undefined || value >= property.minimum) && (property.maximum === undefined || value <= property.maximum);
    else {
      const options = property.items?.enum ?? property.items?.anyOf?.map(item => item.const) ?? [];
      valid = Array.isArray(value) && value.every(item => typeof item === "string" && options.includes(item)) && new Set(value).size === value.length && (property.minItems === undefined || value.length >= property.minItems) && (property.maxItems === undefined || value.length <= property.maxItems);
    }
    if (!valid) throw new McpHostRequestError(-32602, `Review the value for ${property.title ?? name}.`);
  }
  return structuredClone(values) as McpFormValues;
}

export function mcpSamplingRequest(value: unknown, origin: McpRequestOrigin, requestId: string): {
  request: NativeMcpSamplingInput; systemPromptOmitted?: string; modelPreferences?: Record<string, unknown>; media?: McpSamplingMediaReview[]
} {
  const raw = object(value, "sampling request");
  if (raw.tools !== undefined || raw.toolChoice !== undefined) throw new McpHostRequestError(-32602, "This client has not negotiated sampling.tools.");
  if (raw.includeContext !== undefined && raw.includeContext !== "none") throw new McpHostRequestError(-32602, "This client does not share hidden cross-server context.");
  const request: NativeMcpSamplingInput = { brainId: origin.brainId, requestId,
    ...(origin.originatingTurnId ? { originatingTurnId: origin.originatingTurnId } : {}),
    messages: raw.messages as NativeMcpSamplingInput["messages"], maxTokens: raw.maxTokens as number,
    ...(raw.temperature === undefined ? {} : { temperature: raw.temperature as number }),
    ...(raw.stopSequences === undefined ? {} : { stopSequences: raw.stopSequences as string[] }) };
  try { nativeMcpSamplingInputJson(request); } catch (error) { throw new McpHostRequestError(-32602, error instanceof Error ? error.message : String(error)); }
  const media: McpSamplingMediaReview[] = [];
  for (const [messageIndex, message] of request.messages.entries()) {
    const blocks = Array.isArray(message.content) ? message.content : [message.content];
    for (const [blockIndex, block] of blocks.entries()) if (block.type !== "text") media.push({ messageIndex, blockIndex,
      ...nativeMcpSamplingMediaMetadata(block), encodedSha256: createHash("sha256").update(block.data, "ascii").digest("hex") });
  }
  return { request, ...(raw.systemPrompt === undefined ? {} : { systemPromptOmitted: text(raw.systemPrompt, "sampling system prompt") }),
    ...(raw.modelPreferences === undefined ? {} : { modelPreferences: structuredClone(object(raw.modelPreferences, "model preferences")) }),
    ...(media.length ? { media } : {}) };
}

/** Review currently edits task text, never silently substitutes unpreviewed media. */
export function mcpSamplingMediaUnchanged(original: NativeMcpSamplingInput, edited: NativeMcpSamplingInput): boolean {
  const media = (input: NativeMcpSamplingInput): unknown[] => input.messages.flatMap((message, messageIndex) =>
    (Array.isArray(message.content) ? message.content : [message.content]).flatMap((block, blockIndex) =>
      block.type === "text" ? [] : [{ messageIndex, blockIndex, ...block }]));
  return JSON.stringify(media(original)) === JSON.stringify(media(edited));
}
