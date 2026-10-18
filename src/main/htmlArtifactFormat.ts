import { createHash } from "node:crypto";
import {
  HTML_ARTIFACT_CONTENT_FORMAT,
  HTML_ARTIFACT_CONTENT_VERSION,
  type HtmlArtifactContent,
  type HtmlArtifactToolDeclaration
} from "../shared/htmlArtifacts";

const TOOL_ID = /^[a-z][a-z0-9.-]{1,255}$/;
const ACTION_ID = /^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$/;

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object.`);
  return value as Record<string, unknown>;
}

function exactKeys(value: Record<string, unknown>, keys: readonly string[], label: string): void {
  const allowed = new Set(keys);
  const invalid = Object.keys(value).filter((key) => !allowed.has(key));
  if (invalid.length) throw new Error(`${label} contains unsupported fields: ${invalid.join(", ")}.`);
}

function text(value: unknown, label: string, maximum: number, allowEmpty = false): string {
  if (typeof value !== "string" || value.includes("\0") || value.length > maximum || (!allowEmpty && !value.trim())) {
    throw new Error(`${label} is invalid.`);
  }
  return value;
}

function html(value: unknown): string {
  const source = text(value, "HTML artifact body", 2_000_000, true);
  if (/<\/?(?:html|head|body|script|style|iframe|frame|object|embed|form|link|meta|base)\b/i.test(source) ||
      /<[^>]*[\s/]on[a-z0-9_-]+\s*=/i.test(source) ||
      /<[^>]*[\s/](?:src|srcset|href|xlink:href|poster|data|action|formaction|background)\s*=/i.test(source) ||
      /javascript\s*:/i.test(source)) {
    throw new Error("HTML artifacts may contain local controls, SVG, and canvas markup but no scripts, events, navigation, forms, frames, or resources.");
  }
  return source;
}

function css(value: unknown): string | undefined {
  if (value === undefined || value === "") return undefined;
  const source = text(value, "HTML artifact CSS", 2_000_000);
  if (/@import\b|url\s*\(|expression\s*\(|javascript\s*:|behavior\s*:|-moz-binding\s*:/i.test(source)) {
    throw new Error("HTML artifact CSS may not import or address resources.");
  }
  return source;
}

function script(value: unknown): string | undefined {
  if (value === undefined || value === "") return undefined;
  const source = text(value, "HTML artifact script", 2_000_000).replace(/\r\n?/g, "\n");
  if (/<\/script\b/i.test(source)) throw new Error("HTML artifact script may not contain a closing script element.");
  return source;
}

function tools(value: unknown): HtmlArtifactToolDeclaration[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error("HTML artifact tools must be an array.");
  const result = value.map((entry, index) => {
    const row = object(entry, `HTML artifact tools[${index}]`);
    exactKeys(row, ["toolId", "actions"], `HTML artifact tools[${index}]`);
    const toolId = text(row.toolId, `HTML artifact tools[${index}].toolId`, 256).trim();
    if (!TOOL_ID.test(toolId) || toolId === "studio.artifact") {
      throw new Error("HTML artifacts must declare a nonrecursive typed host tool.");
    }
    if (!Array.isArray(row.actions) || !row.actions.length) throw new Error("HTML artifact tool actions are required.");
    const actions = row.actions.map((entry, actionIndex) => {
      const action = text(entry, `HTML artifact tools[${index}].actions[${actionIndex}]`, 128).trim();
      if (!ACTION_ID.test(action)) throw new Error("HTML artifact tool action is invalid.");
      return action;
    });
    if (new Set(actions).size !== actions.length) throw new Error("HTML artifact tool actions must be unique.");
    return { toolId, actions };
  });
  if (new Set(result.map((item) => item.toolId)).size !== result.length) {
    throw new Error("HTML artifact tool declarations must be unique.");
  }
  return result;
}

export function canonicalHtmlArtifactValue(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalHtmlArtifactValue).join(",")}]`;
  return `{${Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([key, child]) => `${JSON.stringify(key)}:${canonicalHtmlArtifactValue(child)}`).join(",")}}`;
}

export function htmlArtifactContentSha256(value: HtmlArtifactContent): string {
  return createHash("sha256").update(canonicalHtmlArtifactValue(value)).digest("hex");
}

export function normalizeHtmlArtifactContent(value: unknown): HtmlArtifactContent {
  const row = object(value, "HTML artifact");
  exactKeys(row, ["title", "html", "css", "script", "tools", "height"], "HTML artifact");
  const normalizedScript = script(row.script);
  const normalizedTools = tools(row.tools);
  const normalizedCss = css(row.css);
  if (normalizedTools.length && !normalizedScript) {
    throw new Error("Only a scripted HTML artifact may request host tools.");
  }
  const height = row.height === undefined ? 360 : Number(row.height);
  if (!Number.isSafeInteger(height) || height < 160 || height > 1_200) {
    throw new Error("HTML artifact height must be an integer from 160 through 1200 pixels.");
  }
  return {
    format: HTML_ARTIFACT_CONTENT_FORMAT,
    version: HTML_ARTIFACT_CONTENT_VERSION,
    title: text(row.title, "HTML artifact title", 256).trim(),
    html: html(row.html),
    ...(normalizedCss ? { css: normalizedCss } : {}),
    ...(normalizedScript ? { script: normalizedScript } : {}),
    tools: normalizedTools,
    height
  };
}

export function validateHtmlArtifactContent(value: unknown): HtmlArtifactContent {
  const row = object(value, "Saved HTML artifact");
  exactKeys(row, ["format", "version", "title", "html", "css", "script", "tools", "height"], "Saved HTML artifact");
  if (row.format !== HTML_ARTIFACT_CONTENT_FORMAT || row.version !== HTML_ARTIFACT_CONTENT_VERSION) {
    throw new Error("Unsupported saved HTML artifact format.");
  }
  const normalized = normalizeHtmlArtifactContent({
    title: row.title,
    html: row.html,
    ...(row.css === undefined ? {} : { css: row.css }),
    ...(row.script === undefined ? {} : { script: row.script }),
    tools: row.tools,
    height: row.height
  });
  if (canonicalHtmlArtifactValue(normalized) !== canonicalHtmlArtifactValue(row)) {
    throw new Error("Saved HTML artifact content is not canonical.");
  }
  return normalized;
}
