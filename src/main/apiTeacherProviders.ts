import type { ApiTeacherProvider } from "../shared/types";

/** HTTP descriptions only. Import never starts a server, pulls a model or sends a request. */
export const API_TEACHER_PROVIDERS: readonly ApiTeacherProvider[] = ["openai", "anthropic", "gemini", "openrouter", "ollama"];
export const API_TEACHER_ENDPOINTS: Readonly<Record<ApiTeacherProvider, string>> = {
  openai: "https://api.openai.com/v1/responses",
  anthropic: "https://api.anthropic.com/v1/messages",
  gemini: "https://generativelanguage.googleapis.com/v1beta/interactions",
  openrouter: "https://openrouter.ai/api/v1/chat/completions",
  ollama: "http://127.0.0.1:11434/v1/chat/completions"
};
export const OLLAMA_LOCAL_BASE = "http://127.0.0.1:11434/v1";
const MODEL = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$/;

export function apiTeacherProvider(value: unknown): ApiTeacherProvider {
  if (!API_TEACHER_PROVIDERS.includes(value as ApiTeacherProvider)) throw new Error("Unsupported API teacher provider.");
  return value as ApiTeacherProvider;
}
export function apiTeacherCredentialRequired(provider: ApiTeacherProvider): boolean {
  return apiTeacherProvider(provider) !== "ollama";
}
export function ollamaLocalModelId(value: unknown): string {
  if (typeof value !== "string" || !MODEL.test(value)) throw new Error("Invalid Ollama model ID.");
  if (/(?:[:/-])cloud(?:$|[:/-])/i.test(value)) throw new Error("Ollama local cannot select a cloud model without separate cloud consent; choose an installed local model.");
  return value;
}

/** Explicit Ollama LOCAL interface, never Ollama cloud or an arbitrary remote URL. */
export function ollamaLoopbackEndpoint(value: unknown = OLLAMA_LOCAL_BASE): string {
  if (value === "" || value === undefined) value = OLLAMA_LOCAL_BASE;
  if (typeof value !== "string" || value.length > 2_048) throw new Error("Enter a loopback Ollama endpoint.");
  let url: URL;
  try { url = new URL(value); } catch { throw new Error("Enter a valid loopback Ollama endpoint."); }
  if (!["127.0.0.1", "localhost", "[::1]", "::1"].includes(url.hostname.toLowerCase()) ||
      !["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash ||
      !["", "/", "/v1"].includes(url.pathname.replace(/\/+$/, ""))) {
    throw new Error("Ollama local requires a loopback HTTP(S) base or /v1 endpoint without credentials, query or fragment.");
  }
  url.pathname = "/v1"; return url.toString().replace(/\/$/, "");
}

export interface ApiTeacherHttpInput { model: string; prompt: string; maxOutputTokens: number; credential?: string }
export interface ApiTeacherHttpRequest { url: string; headers: Record<string, string>; body: Record<string, unknown> }
export function apiTeacherHttpRequest(providerValue: ApiTeacherProvider, input: ApiTeacherHttpInput): ApiTeacherHttpRequest {
  const provider = apiTeacherProvider(providerValue);
  if (typeof input.model !== "string" || !MODEL.test(input.model)) throw new Error("Invalid API teacher model name.");
  if (provider === "ollama") ollamaLocalModelId(input.model);
  if (typeof input.prompt !== "string" || !input.prompt.trim() || input.prompt.includes("\0") || Buffer.byteLength(input.prompt) > 4 * 1024 * 1024) {
    throw new Error("API teacher prompt is empty or exceeds 4 MiB.");
  }
  if (!Number.isSafeInteger(input.maxOutputTokens) || input.maxOutputTokens < 1 || input.maxOutputTokens > 65_536) throw new Error("Invalid API teacher output-token budget.");
  const headers: Record<string, string> = { "content-type": "application/json" };
  const { model, prompt, maxOutputTokens, credential } = input;
  if (apiTeacherCredentialRequired(provider)) {
    if (typeof credential !== "string" || !credential || credential.length > 64_000 || /[\r\n\0]/.test(credential)) throw new Error(`No valid ${provider} API credential is configured.`);
  } else if (credential) throw new Error("Local Ollama uses no API key; an unrelated credential cannot be forwarded.");
  let body: Record<string, unknown>;
  if (provider === "openai") {
    headers.authorization = `Bearer ${credential}`;
    body = { model, input: prompt, store: false, max_output_tokens: maxOutputTokens };
  } else if (provider === "anthropic") {
    headers["x-api-key"] = credential!; headers["anthropic-version"] = "2023-06-01";
    body = { model, max_tokens: maxOutputTokens, messages: [{ role: "user", content: prompt }] };
  } else if (provider === "gemini") {
    headers["x-goog-api-key"] = credential!;
    body = { model, input: prompt, store: false, generation_config: { max_output_tokens: maxOutputTokens } };
  } else {
    if (provider === "openrouter") headers.authorization = `Bearer ${credential}`;
    body = { model, messages: [{ role: "user", content: prompt }], stream: false,
      ...(provider === "openrouter" ? { max_completion_tokens: maxOutputTokens, provider: { allow_fallbacks: false } }
        : { max_tokens: maxOutputTokens }) };
  }
  return { url: API_TEACHER_ENDPOINTS[provider], headers, body };
}

function textBlocks(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap(item => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return [];
    const row = item as Record<string, unknown>;
    if (typeof row.text === "string") return [row.text];
    return Array.isArray(row.content) ? textBlocks(row.content) : [];
  });
}

/** Gemini REST Interactions returns steps, not the SDK's output_text convenience getter.
 * https://ai.google.dev/api/interactions-api (verified 2026-10-18). */
function geminiTeacherText(body: Record<string, unknown>): string {
  if (body.status !== "completed") throw new Error("Gemini teacher response is not a completed interaction; partial or action-required text was not learned.");
  if (body.errors !== undefined && (!Array.isArray(body.errors) || body.errors.length) || body.error != null)
    throw new Error("Gemini teacher response reports errors; it was not accepted as a complete text lesson.");
  if (!Array.isArray(body.steps)) throw new Error("Gemini teacher response has no documented interaction steps.");
  const text: string[] = [];
  for (const value of body.steps) {
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Gemini teacher step is invalid.");
    const step = value as Record<string, unknown>;
    if (step.type === "thought") {
      if (Object.keys(step).some(key => !["type", "signature", "summary"].includes(key)) ||
          step.signature !== undefined && typeof step.signature !== "string" || step.summary !== undefined && !Array.isArray(step.summary))
        throw new Error("Gemini teacher thought metadata is invalid.");
      continue; // A thought summary is not the completed model-output lesson.
    }
    if (step.type !== "model_output" || !Array.isArray(step.content) || step.error != null ||
        Object.keys(step).some(key => !["type", "content", "error"].includes(key)))
      throw new Error("Gemini teacher must return completed model-output text, not tool instructions or errored steps.");
    for (const value of step.content) {
      if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Gemini teacher text block is invalid.");
      const block = value as Record<string, unknown>;
      if (block.type !== "text" || typeof block.text !== "string" ||
          block.annotations !== undefined && !Array.isArray(block.annotations) || Object.keys(block).some(key => !["type", "text", "annotations"].includes(key)))
        throw new Error("Gemini text teacher returned non-text data or tool instructions; use an explicitly supported media lesson instead.");
      text.push(block.text);
    }
  }
  const complete = text.join("\n").trim();
  if (!complete) throw new Error("Gemini teacher returned no completed model-output text."); return complete;
}
export function apiTeacherResponseText(providerValue: ApiTeacherProvider, body: Record<string, unknown>): string {
  const provider = apiTeacherProvider(providerValue);
  if (provider === "gemini") return geminiTeacherText(body);
  if (provider === "openrouter" || provider === "ollama") {
    if (!Array.isArray(body.choices) || body.choices.length < 1) throw new Error("Teacher chat response is missing choices.");
    const choice = body.choices[0];
    if (!choice || typeof choice !== "object" || Array.isArray(choice)) throw new Error("Teacher chat choice is invalid.");
    const message = (choice as Record<string, unknown>).message;
    if (!message || typeof message !== "object" || Array.isArray(message)) throw new Error("Teacher chat response is missing a message.");
    const row = message as Record<string, unknown>;
    const calls = row.tool_calls != null && (!Array.isArray(row.tool_calls) || row.tool_calls.length > 0);
    if (calls || row.function_call != null || typeof row.content !== "string") throw new Error("Teacher chat must return text, not tool instructions.");
    return row.content.trim();
  }
  if (typeof body.output_text === "string") return body.output_text.trim();
  return textBlocks(provider === "anthropic" ? body.content : body.output).join("\n").trim();
}
