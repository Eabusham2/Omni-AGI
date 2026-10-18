import { afterEach, expect, it, vi } from "vitest";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { apiTeacherHttpRequest, apiTeacherResponseText, ollamaLoopbackEndpoint } from "../src/main/apiTeacherProviders";
import { ApiTeacherModelDiscoveryService } from "../src/main/apiTeacherModelDiscovery";
import { ExternalAdviceClient } from "../src/main/externalAdviceClient";
import { ExternalAdviceService, validateExternalAdviceWorkspace } from "../src/main/externalAdviceService";
import type { ExternalAdviceChannel, ExternalAdviceSelectedContext } from "../src/shared/externalAdvice";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
const json = (value: unknown) => new Response(JSON.stringify(value), { headers: { "content-type": "application/json" } });
function stream(value: unknown): Response {
  return new Response(`data: ${JSON.stringify(value)}\n\ndata: [DONE]\n\n`, { headers: { "content-type": "text/event-stream" } });
}

it("adds exact keyless local and no-fallback OpenRouter teacher HTTP shapes without changing existing provider bodies", () => {
  const input = { model: "vendor/local:latest", prompt: "visible question", maxOutputTokens: 123 };
  expect(apiTeacherHttpRequest("ollama", input)).toEqual({ url: "http://127.0.0.1:11434/v1/chat/completions",
    headers: { "content-type": "application/json" }, body: { model: input.model, messages: [{ role: "user", content: input.prompt }], max_tokens: 123, stream: false } });
  expect(apiTeacherHttpRequest("openrouter", { ...input, credential: "synthetic-key" })).toMatchObject({
    url: "https://openrouter.ai/api/v1/chat/completions", headers: { authorization: "Bearer synthetic-key" },
    body: { model: input.model, max_completion_tokens: 123, provider: { allow_fallbacks: false }, stream: false } });
  expect(apiTeacherHttpRequest("openai", { ...input, credential: "synthetic-key" }).body).toEqual({ model: input.model, input: input.prompt, store: false, max_output_tokens: 123 });
  expect(apiTeacherHttpRequest("anthropic", { ...input, credential: "synthetic-key" }).body).toEqual({ model: input.model, max_tokens: 123, messages: [{ role: "user", content: input.prompt }] });
  expect(apiTeacherHttpRequest("gemini", { ...input, credential: "synthetic-key" }).body).toEqual({ model: input.model, input: input.prompt, store: false, generation_config: { max_output_tokens: input.maxOutputTokens } });
  expect(apiTeacherResponseText("ollama", { choices: [{ message: { content: " selected text ", tool_calls: [], function_call: null } }] })).toBe("selected text");
  expect(() => apiTeacherResponseText("openrouter", { choices: [{ message: { content: "text", tool_calls: [{ id: "actual-tool" }] } }] })).toThrow(/not tool/i);
  expect(() => apiTeacherHttpRequest("ollama", { ...input, credential: "other-provider-secret" })).toThrow(/cannot be forwarded/);
  for (const model of ["model:cloud", "model:120b-cloud"]) expect(() => apiTeacherHttpRequest("ollama", { ...input, model })).toThrow(/separate cloud consent/);
});

it("discovers only explicit local inventory without reading any secret or requesting model startup/pull", async () => {
  const secrets = { get: vi.fn(async () => { throw new Error("Ollama discovery must not read keys"); }) };
  const fetcher = vi.fn<typeof fetch>(async (input, init) => {
    expect(String(input)).toBe("http://127.0.0.1:11434/api/tags"); expect(init?.method).toBe("GET");
    expect(new Headers(init?.headers).has("authorization")).toBe(false);
    return json({ models: [{ name: "local-model:latest", model: "local-model:latest" },
      { name: "cloud-model:cloud" }, { name: "innocent-alias:latest", remote_host: "https://ollama.com" }] });
  });
  const result = await new ApiTeacherModelDiscoveryService(secrets, fetcher).discover({ provider: "ollama", requestId: "local-inventory" });
  expect(result.models.map(row => row.id)).toEqual(["local-model:latest"]);
  expect(result).toMatchObject({ complete: true, generationCompatibilityVerified: false, excludedModelCount: 2,
    excludedModels: [{ id: "cloud-model:cloud" }, { id: "innocent-alias:latest" }] });
  expect(result.excludedModels?.every(row => /cloud consent/.test(row.reason))).toBe(true);
  expect(secrets.get).not.toHaveBeenCalled(); expect(fetcher).toHaveBeenCalledTimes(1);
});

it("uses account-filtered OpenRouter pagination with its own key and never follows remote catalog links", async () => {
  let page = 0; const secrets = { get: vi.fn(async (id: string) => id === "teacher:openrouter" ? "synthetic-router-key" : undefined) };
  const fetcher = vi.fn<typeof fetch>(async (input, init) => {
    const url = new URL(String(input)); expect(url.origin + url.pathname).toBe("https://openrouter.ai/api/v1/models/user");
    expect(url.searchParams.get("limit")).toBe("100"); expect(new Headers(init?.headers).get("authorization")).toBe("Bearer synthetic-router-key");
    if (page++ === 0) { expect(url.searchParams.get("offset")).toBe("0"); return json({ data: [{ id: "vendor/a", name: "A", context_length: 4000 }, { id: "vendor/b" }], total_count: 3, links: { next: "https://untrusted.invalid/key-grab" } }); }
    expect(url.searchParams.get("offset")).toBe("2"); return json({ data: [{ id: "vendor/c", top_provider: { max_completion_tokens: 1000 } }], total_count: 3 });
  });
  const result = await new ApiTeacherModelDiscoveryService(secrets, fetcher).discover({ provider: "openrouter", requestId: "router-list" });
  expect(result.sourceEndpoint).toBe("https://openrouter.ai/api/v1/models/user"); expect(result.pageCount).toBe(2);
  expect(result.models.map(row => row.id)).toEqual(["vendor/a", "vendor/b", "vendor/c"]);
  const noKey = vi.fn<typeof fetch>();
  await expect(new ApiTeacherModelDiscoveryService({ get: async () => undefined }, noKey).discover({ provider: "openrouter", requestId: "missing-key" })).rejects.toThrow(/no.*credential/i);
  expect(noKey).not.toHaveBeenCalled(); expect(JSON.stringify(result)).not.toContain("synthetic-router-key");
});

it("streams explicit Ollama through existing text SSE and refuses remote/cloud/key/redirect before accepting data", async () => {
  const fetcher = vi.fn<typeof fetch>(async (url, init) => {
    expect(String(url)).toBe("http://127.0.0.1:11434/v1/chat/completions"); expect(new Headers(init?.headers).has("authorization")).toBe(false);
    expect(JSON.parse(String(init?.body))).toMatchObject({ model: "local:latest", max_tokens: 50, stream: true });
    return stream({ id: "local-request", choices: [{ delta: { content: "local advice", tool_calls: [], function_call: null }, finish_reason: "stop" }] });
  });
  const client = new ExternalAdviceClient(fetcher), request = { provider: "ollama" as const, model: "local:latest", text: "visible text", maxOutputTokens: 50 };
  await expect(client.stream(request, () => undefined, new AbortController().signal)).resolves.toMatchObject({ text: "local advice" });
  for (const changed of [{ ...request, endpoint: "https://ollama.com/v1" }, { ...request, model: "model:cloud" }, { ...request, credential: "do-not-send" }]) {
    await expect(client.stream(changed, () => undefined, new AbortController().signal)).rejects.toThrow();
  }
  expect(fetcher).toHaveBeenCalledTimes(1);
  expect(ollamaLoopbackEndpoint("http://localhost:11434/")).toBe("http://localhost:11434/v1");
  for (const url of ["http://127.0.0.1:11434/api/pull", "http://127.0.0.1:11434/v1?key=x", "http://user:pw@localhost:11434/v1"]) expect(() => ollamaLoopbackEndpoint(url)).toThrow();
  const redirected = new ExternalAdviceClient(vi.fn<typeof fetch>(async () => new Response(null, { status: 302, headers: { location: "https://untrusted.invalid" } })));
  await expect(redirected.stream(request, () => undefined, new AbortController().signal)).rejects.toThrow(/redirect/);
});

it("persists keyless per-channel Ollama settings/history but still requires explicit current device authority", async () => {
  const root = await mkdtemp(join(tmpdir(), "omni-ollama-advice-")); roots.push(root); const brainRoot = join(root, "brain"); await mkdir(brainRoot);
  const secretStore = { get: vi.fn(async () => undefined), has: vi.fn(async () => false), set: vi.fn(async () => "encrypted" as const),
    remove: vi.fn(async () => false), persistence: () => "encrypted" as const };
  let enabled = false;
  const client = { stream: vi.fn(async (request: { credential?: string }, onDelta: (text: string) => Promise<void> | void) => {
    expect(request.credential).toBeUndefined(); await onDelta("fixture advice"); return { text: "fixture advice" };
  }) };
  const service = new ExternalAdviceService(client, secretStore, { deviceDirectory: join(root, "device"), brainRoot: () => brainRoot,
    authority: async (_brainId, channel) => ({ brainId: "one", channel, enabled, ownerEpoch: "current", revision: 1 }), admitWrite: async () => undefined,
    selectContext: async ({ brainId, channel, selection }) => ({ format: "omni-external-advice-selected-context", version: 1, brainId, channel, selection,
      items: [], includedCharacters: 0, omittedItems: 0, truncated: false } satisfies ExternalAdviceSelectedContext) });
  for (const channel of ["assistant", "reviewer"] as ExternalAdviceChannel[]) {
    const settings = await service.settings("one"); await service.configure("one", { channel, expectedRevision: settings.revision,
      configuration: { enabled: true, provider: "ollama", model: "local:latest", maxOutputTokens: 50, context: { includeTask: false, includeActivity: false, maxCharacters: 0 } } });
  }
  await expect(service.preview("one", { channel: "assistant", prompt: "help" })).rejects.toThrow(/authority|device|disabled/i);
  enabled = true; const preview = await service.preview("one", { channel: "assistant", prompt: "help" });
  const complete = new Promise<void>((resolve, reject) => service.on("event", event => {
    if (event.request.state === "complete") resolve(); else if (event.request.state === "failed") reject(new Error(event.request.error));
  }));
  await service.start("one", { channel: "assistant", prompt: "help", requestId: "local-request", expectedSettingsRevision: preview.settingsRevision, previewSha256: preview.sha256 }); await complete;
  expect((await service.history("one")).entries[0]).toMatchObject({ provider: "ollama", endpoint: "http://127.0.0.1:11434/v1", response: "fixture advice", hostActionAuthority: false });
  expect(await service.credentialStatus()).toContainEqual(expect.objectContaining({ provider: "ollama", required: false, configured: false }));
  await expect(service.saveCredential({ provider: "ollama", credential: "do-not-store" })).rejects.toThrow(/no API key/);
  expect(secretStore.set).not.toHaveBeenCalled(); expect(await validateExternalAdviceWorkspace(brainRoot, "one")).toMatchObject({ requests: 1 });
});
