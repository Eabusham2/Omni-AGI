import { createHash } from "node:crypto";
import type {
  ApiTeacherDiscoveredModel,
  ApiTeacherModelDiscoveryCancellation,
  ApiTeacherModelDiscoveryRequest,
  ApiTeacherModelDiscoveryResult,
  ApiTeacherProvider
} from "../shared/types";
import { apiTeacherProvider, apiTeacherCredentialRequired, ollamaLocalModelId, ollamaLoopbackEndpoint } from "./apiTeacherProviders";

type FetchLike = typeof fetch;
type SecretReader = { get(id: string): Promise<string | undefined> };

const PAGE_BYTES = 2 * 1024 * 1024;
const TOTAL_BYTES = 8 * 1024 * 1024;
const REQUEST_TIMEOUT_MS = 30_000;
const MODEL_ID = /^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,199}$/;
const REQUEST_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;

export const API_TEACHER_MODEL_ENDPOINTS: Readonly<Record<ApiTeacherProvider, string>> = {
  openai: "https://api.openai.com/v1/models",
  anthropic: "https://api.anthropic.com/v1/models",
  gemini: "https://generativelanguage.googleapis.com/v1beta/models",
  openrouter: "https://openrouter.ai/api/v1/models/user",
  ollama: "http://127.0.0.1:11434/api/tags"
};

interface ActiveDiscovery {
  provider: ApiTeacherProvider;
  endpoint?: string;
  controller: AbortController;
  promise: Promise<ApiTeacherModelDiscoveryResult>;
}

function requireProvider(value: unknown): ApiTeacherProvider {
  return apiTeacherProvider(value);
}

function requireRequest(value: ApiTeacherModelDiscoveryRequest): ApiTeacherModelDiscoveryRequest {
  const provider = requireProvider(value?.provider);
  if (typeof value?.requestId !== "string" || !REQUEST_ID.test(value.requestId)) throw new Error("Invalid model discovery request ID.");
  if (provider !== "ollama" && value.endpoint !== undefined) throw new Error("Remote model discovery destinations are fixed.");
  return { provider, requestId: value.requestId, ...(provider === "ollama" && value.endpoint !== undefined ? { endpoint: ollamaLoopbackEndpoint(value.endpoint) } : {}) };
}

function credentialId(provider: ApiTeacherProvider): string { return `teacher:${provider}`; }
function sha256(value: string): string { return createHash("sha256").update(value).digest("hex"); }
function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} is invalid.`);
  return value as Record<string, unknown>;
}
function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${label} is invalid.`); return value;
}
function text(value: unknown, label: string, maximum = 2_000): string {
  if (typeof value !== "string" || value.length < 1 || value.length > maximum || value.includes("\0")) throw new Error(`${label} is invalid.`);
  return value;
}
function optionalText(value: unknown, label: string, maximum = 2_000): string | undefined {
  return value === undefined || value === null ? undefined : text(value, label, maximum);
}
function modelId(value: unknown, label = "provider model ID"): string {
  const result = text(value, label, 200); if (!MODEL_ID.test(result)) throw new Error(`${label} is invalid.`); return result;
}
function positiveInteger(value: unknown, label: string): number | undefined {
  if (value === undefined || value === null) return undefined;
  if (!Number.isSafeInteger(value) || Number(value) < 0) throw new Error(`${label} is invalid.`); return Number(value);
}
function timestamp(value: unknown, label: string): string | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value))) throw new Error(`${label} is invalid.`);
  return new Date(value).toISOString();
}
function cleanRemoteError(value: string, secret?: string): string {
  let result = value.replace(/\bsk-[A-Za-z0-9_-]{12,}\b/g, "[REDACTED]")
    .replace(/\bAIza[A-Za-z0-9_-]{16,}\b/g, "[REDACTED]");
  if (secret) result = result.split(secret).join("[REDACTED]");
  return result.replace(/[\r\n\0]+/g, " ").slice(0, 1_000);
}

async function boundedJson(response: Response, secret: string, expectedUrl: string): Promise<{ body: Record<string, unknown>; bytes: number }> {
  if (response.redirected || response.type === "opaqueredirect") throw new Error("Provider model discovery refused an authentication redirect.");
  if (response.url) {
    const actual = new URL(response.url), expected = new URL(expectedUrl);
    if (actual.origin !== expected.origin || actual.pathname !== expected.pathname) throw new Error("Provider model discovery refused an unexpected response destination.");
  }
  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > PAGE_BYTES) throw new Error("Provider model discovery page exceeded 2 MiB.");
  const chunks: Uint8Array[] = []; let total = 0;
  if (response.body) {
    const reader = response.body.getReader();
    for (;;) {
      const { value, done } = await reader.read(); if (done) break; if (!value) continue;
      total += value.byteLength;
      if (total > PAGE_BYTES) { await reader.cancel().catch(() => undefined); throw new Error("Provider model discovery page exceeded 2 MiB."); }
      chunks.push(value);
    }
  }
  const bytes = new Uint8Array(total); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  let parsed: unknown;
  try { parsed = JSON.parse(new TextDecoder().decode(bytes)); }
  catch { throw new Error(`Provider model discovery returned invalid JSON (HTTP ${response.status}).`); }
  const body = record(parsed, "Provider model discovery response");
  if (!response.ok) {
    const remote = body.error && typeof body.error === "object" && !Array.isArray(body.error)
      ? body.error as Record<string, unknown> : body;
    throw new Error(`Provider model discovery failed: ${cleanRemoteError(String(remote.message ?? `HTTP ${response.status}`), secret)}`);
  }
  return { body, bytes: total };
}

function openAiModels(body: Record<string, unknown>): ApiTeacherDiscoveredModel[] {
  if (body.object !== undefined && body.object !== "list") throw new Error("OpenAI model discovery returned an invalid list object.");
  return array(body.data, "OpenAI model list").map((item) => {
    const row = record(item, "OpenAI model");
    const created = positiveInteger(row.created, "OpenAI model creation time");
    const retirement = optionalText(row.shutdown_date, "OpenAI shutdown date", 32);
    if (retirement && !/^\d{4}-\d{2}-\d{2}$/.test(retirement)) throw new Error("OpenAI shutdown date is invalid.");
    return { provider: "openai", id: modelId(row.id),
      ...(optionalText(row.owned_by, "OpenAI model owner", 256) ? { owner: optionalText(row.owned_by, "OpenAI model owner", 256)! } : {}),
      ...(created !== undefined ? { createdAt: new Date(created * 1_000).toISOString() } : {}),
      ...(retirement ? { retirementAt: retirement } : {}) };
  });
}

function anthropicModels(body: Record<string, unknown>): ApiTeacherDiscoveredModel[] {
  return array(body.data, "Anthropic model list").map((item) => {
    const row = record(item, "Anthropic model"), lifecycle = row.lifecycle;
    if (lifecycle !== undefined && !["active", "deprecated", "retired"].includes(String(lifecycle))) throw new Error("Anthropic model lifecycle is invalid.");
    return { provider: "anthropic", id: modelId(row.id),
      ...(optionalText(row.display_name, "Anthropic model display name", 256) ? { displayName: optionalText(row.display_name, "Anthropic model display name", 256)! } : {}),
      ...(timestamp(row.created_at, "Anthropic model creation time") ? { createdAt: timestamp(row.created_at, "Anthropic model creation time")! } : {}),
      ...(lifecycle ? { lifecycle: lifecycle as "active" | "deprecated" | "retired" } : {}),
      ...(timestamp(row.retires_at, "Anthropic model retirement time") ? { retirementAt: timestamp(row.retires_at, "Anthropic model retirement time")! } : {}),
      ...(positiveInteger(row.max_input_tokens, "Anthropic model input limit") !== undefined ? { inputTokenLimit: positiveInteger(row.max_input_tokens, "Anthropic model input limit")! } : {}),
      ...(positiveInteger(row.max_tokens, "Anthropic model output limit") !== undefined ? { outputTokenLimit: positiveInteger(row.max_tokens, "Anthropic model output limit")! } : {}) };
  });
}

function geminiModels(body: Record<string, unknown>): ApiTeacherDiscoveredModel[] {
  return array(body.models, "Gemini model list").map((item) => {
    const row = record(item, "Gemini model"), resource = text(row.name, "Gemini model resource", 240);
    if (!/^models\/[a-zA-Z0-9][a-zA-Z0-9._-]{0,199}$/.test(resource)) throw new Error("Gemini model resource is invalid.");
    const methods = row.supportedGenerationMethods === undefined ? undefined : array(row.supportedGenerationMethods, "Gemini supported methods")
      .map((method) => text(method, "Gemini supported method", 128));
    return { provider: "gemini", id: modelId(row.baseModelId), providerResourceName: resource,
      ...(optionalText(row.displayName, "Gemini model display name", 256) ? { displayName: optionalText(row.displayName, "Gemini model display name", 256)! } : {}),
      ...(optionalText(row.description, "Gemini model description", 2_000) ? { description: optionalText(row.description, "Gemini model description", 2_000)! } : {}),
      ...(optionalText(row.version, "Gemini model version", 128) ? { version: optionalText(row.version, "Gemini model version", 128)! } : {}),
      ...(positiveInteger(row.inputTokenLimit, "Gemini model input limit") !== undefined ? { inputTokenLimit: positiveInteger(row.inputTokenLimit, "Gemini model input limit")! } : {}),
      ...(positiveInteger(row.outputTokenLimit, "Gemini model output limit") !== undefined ? { outputTokenLimit: positiveInteger(row.outputTokenLimit, "Gemini model output limit")! } : {}),
      ...(methods ? { supportedMethods: methods } : {}) };
  });
}

function openRouterModels(body: Record<string, unknown>): ApiTeacherDiscoveredModel[] {
  return array(body.data, "OpenRouter model list").map(item => {
    const row = record(item, "OpenRouter model");
    const top = row.top_provider == null ? {} : record(row.top_provider, "OpenRouter top provider");
    return { provider: "openrouter", id: modelId(row.id),
      ...(optionalText(row.name, "OpenRouter model name", 256) ? { displayName: optionalText(row.name, "OpenRouter model name", 256)! } : {}),
      ...(optionalText(row.description, "OpenRouter model description", 16_000) ? { description: optionalText(row.description, "OpenRouter model description", 16_000)! } : {}),
      ...(positiveInteger(row.context_length, "OpenRouter context limit") !== undefined ? { inputTokenLimit: positiveInteger(row.context_length, "OpenRouter context limit")! } : {}),
      ...(positiveInteger(top.max_completion_tokens, "OpenRouter output limit") !== undefined ? { outputTokenLimit: positiveInteger(top.max_completion_tokens, "OpenRouter output limit")! } : {}) };
  });
}
function ollamaModels(body: Record<string, unknown>, excluded: Array<{ id: string; reason: string }>): ApiTeacherDiscoveredModel[] {
  return array(body.models, "Ollama installed model list").flatMap(item => {
    const row = record(item, "Ollama model");
    const id = modelId(row.model ?? row.name, "Ollama installed model ID");
    try {
      ollamaLocalModelId(id);
      if (row.remote_host || row.remote_model) throw new Error("This entry advertises remote/cloud execution; separate cloud consent is required.");
    } catch (error) { excluded.push({ id, reason: error instanceof Error ? error.message : "Local-only selection requires review." }); return []; }
    return [{ provider: "ollama" as const, id,
      ...(optionalText(row.name, "Ollama model name", 200) ? { displayName: optionalText(row.name, "Ollama model name", 200)! } : {}) }];
  });
}

function discoveryEndpoint(request: ApiTeacherModelDiscoveryRequest): string {
  return request.provider === "ollama" && request.endpoint !== undefined
    ? `${new URL(ollamaLoopbackEndpoint(request.endpoint)).origin}/api/tags` : API_TEACHER_MODEL_ENDPOINTS[request.provider];
}
function pageUrl(request: ApiTeacherModelDiscoveryRequest, cursor?: string): string {
  const provider = request.provider, url = new URL(discoveryEndpoint(request));
  if (provider === "anthropic") { url.searchParams.set("limit", "1000"); if (cursor) url.searchParams.set("after_id", cursor); }
  if (provider === "gemini") { url.searchParams.set("pageSize", "1000"); if (cursor) url.searchParams.set("pageToken", cursor); }
  if (provider === "openrouter") { url.searchParams.set("limit", "100"); url.searchParams.set("offset", cursor ?? "0"); }
  return url.toString();
}

function nextCursor(provider: ApiTeacherProvider, body: Record<string, unknown>, cursor?: string): string | undefined {
  if (provider === "openai" || provider === "ollama") return undefined;
  if (provider === "openrouter") {
    const rows = array(body.data, "OpenRouter page"), offset = Number(cursor ?? 0);
    const total = positiveInteger(body.total_count, "OpenRouter total count");
    if (!Number.isSafeInteger(offset) || rows.length > 100 || total !== undefined && total < offset + rows.length) throw new Error("OpenRouter model pagination is inconsistent.");
    if (!rows.length && total !== undefined && total > offset) throw new Error("OpenRouter model pagination did not advance.");
    return total !== undefined ? offset + rows.length < total ? String(offset + rows.length) : undefined
      : rows.length === 100 ? String(offset + rows.length) : undefined;
  }
  if (provider === "anthropic") {
    if (typeof body.has_more !== "boolean") throw new Error("Anthropic model pagination marker is invalid.");
    return body.has_more ? text(body.last_id, "Anthropic next model cursor", 512) : undefined;
  }
  return body.nextPageToken === undefined || body.nextPageToken === null || body.nextPageToken === ""
    ? undefined : text(body.nextPageToken, "Gemini next model cursor", 2_048);
}

export class ApiTeacherModelDiscoveryService {
  private readonly active = new Map<string, ActiveDiscovery>();
  private readonly credentialRevisions = new Map<ApiTeacherProvider, number>();
  constructor(private readonly secrets: SecretReader, private readonly fetcher: FetchLike = fetch) {}

  credentialChanged(providerValue: ApiTeacherProvider): void {
    const provider = requireProvider(providerValue);
    this.credentialRevisions.set(provider, (this.credentialRevisions.get(provider) ?? 0) + 1);
    for (const entry of this.active.values()) if (entry.provider === provider && !entry.controller.signal.aborted) {
      entry.controller.abort(new Error(`${provider} credential changed during model discovery.`));
    }
  }

  discover(requestValue: ApiTeacherModelDiscoveryRequest): Promise<ApiTeacherModelDiscoveryResult> {
    const request = requireRequest(requestValue), existing = this.active.get(request.requestId);
    if (existing) {
      if (existing.provider !== request.provider || existing.endpoint !== request.endpoint) return Promise.reject(new Error("Model discovery request ID belongs to another provider or endpoint."));
      return existing.promise;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new Error("Provider model discovery exceeded its 30-second deadline.")), REQUEST_TIMEOUT_MS); timer.unref();
    const promise = this.run(request, controller.signal).finally(() => {
      clearTimeout(timer); if (this.active.get(request.requestId)?.promise === promise) this.active.delete(request.requestId);
    });
    this.active.set(request.requestId, { provider: request.provider, endpoint: request.endpoint, controller, promise }); return promise;
  }

  cancel(requestValue: ApiTeacherModelDiscoveryRequest): ApiTeacherModelDiscoveryCancellation {
    const request = requireRequest(requestValue), active = this.active.get(request.requestId);
    if (!active) return { ...request, cancelled: false };
    if (active.provider !== request.provider || active.endpoint !== request.endpoint) throw new Error("Model discovery request ID belongs to another provider or endpoint.");
    if (!active.controller.signal.aborted) active.controller.abort(new Error("Provider model discovery was cancelled."));
    return { ...request, cancelled: true };
  }

  close(): void {
    for (const entry of this.active.values()) if (!entry.controller.signal.aborted) entry.controller.abort(new Error("Provider model discovery is closing."));
  }

  private async credential(provider: ApiTeacherProvider): Promise<{ secret: string; revision: number; digest: string }> {
    const revision = this.credentialRevisions.get(provider) ?? 0;
    if (!apiTeacherCredentialRequired(provider)) return { secret: "", revision, digest: sha256("") };
    const secret = await this.secrets.get(credentialId(provider)) ?? "";
    if ((this.credentialRevisions.get(provider) ?? 0) !== revision) throw new Error(`${provider} credential changed before model discovery.`);
    if (!secret) throw new Error(`No ${provider} API credential is configured; model discovery made no network request.`);
    return { secret, revision, digest: sha256(secret) };
  }

  private async requireCredentialCurrent(provider: ApiTeacherProvider, expected: { revision: number; digest: string }): Promise<void> {
    const current = await this.credential(provider);
    if (current.revision !== expected.revision || current.digest !== expected.digest) throw new Error(`${provider} credential changed before model discovery could publish.`);
  }

  private async run(request: ApiTeacherModelDiscoveryRequest, signal: AbortSignal): Promise<ApiTeacherModelDiscoveryResult> {
    let knownSecret = "";
    try {
      const credential = await this.credential(request.provider); knownSecret = credential.secret; signal.throwIfAborted();
      const headers: Record<string, string> = { accept: "application/json" };
      if (request.provider === "openai" || request.provider === "openrouter") headers.authorization = `Bearer ${knownSecret}`;
      else if (request.provider === "anthropic") { headers["x-api-key"] = knownSecret; headers["anthropic-version"] = "2023-06-01"; }
      else if (request.provider === "gemini") headers["x-goog-api-key"] = knownSecret;
      const models: ApiTeacherDiscoveredModel[] = [], excluded: Array<{ id: string; reason: string }> = [], seenCursors = new Set<string>(), routerPages = new Set<string>(); let cursor: string | undefined, pageCount = 0, totalBytes = 0;
      do {
        signal.throwIfAborted(); await this.requireCredentialCurrent(request.provider, credential); signal.throwIfAborted();
        const requestedUrl = pageUrl(request, cursor);
        const response = await this.fetcher(requestedUrl, { method: "GET", headers, redirect: "error", signal });
        const page = await boundedJson(response, knownSecret, requestedUrl); totalBytes += page.bytes; pageCount += 1;
        if (totalBytes > TOTAL_BYTES) throw new Error("Provider model discovery exceeded its 8 MiB cumulative response budget.");
        const parsed = request.provider === "openai" ? openAiModels(page.body) : request.provider === "anthropic" ? anthropicModels(page.body)
          : request.provider === "gemini" ? geminiModels(page.body) : request.provider === "openrouter" ? openRouterModels(page.body) : ollamaModels(page.body, excluded);
        if (request.provider === "openrouter" && parsed.length) {
          const pageKey = sha256(JSON.stringify(parsed.map(item => item.id)));
          if (routerPages.has(pageKey)) throw new Error("OpenRouter repeated a model page without advancing.");
          routerPages.add(pageKey);
        }
        models.push(...parsed);
        cursor = nextCursor(request.provider, page.body, cursor);
        if (cursor) { if (seenCursors.has(cursor)) throw new Error("Provider model discovery returned a repeated pagination cursor."); seenCursors.add(cursor); }
      } while (cursor);
      signal.throwIfAborted(); await this.requireCredentialCurrent(request.provider, credential); signal.throwIfAborted();
      return { format: "omni-api-teacher-model-discovery", version: 1, provider: request.provider, requestId: request.requestId,
        sourceEndpoint: discoveryEndpoint(request), discoveredAt: new Date().toISOString(), pageCount, complete: true,
        generationCompatibilityVerified: false, models,
        ...(excluded.length ? { excludedModels: excluded, excludedModelCount: excluded.length,
          exclusionReason: "Known cloud/remote entries are excluded from local-only selection; no cloud consent was inferred." } : {}) };
    } catch (error) {
      const reason = signal.aborted ? signal.reason : error;
      throw new Error(cleanRemoteError(reason instanceof Error ? reason.message : String(reason), knownSecret));
    }
  }
}
