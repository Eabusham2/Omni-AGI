import { describe, expect, it, vi } from "vitest";
import {
  API_TEACHER_MODEL_ENDPOINTS,
  ApiTeacherModelDiscoveryService
} from "../src/main/apiTeacherModelDiscovery";
import type { ApiTeacherProvider } from "../src/shared/types";

class FakeSecrets {
  readonly values = new Map<string, string>();
  async get(id: string): Promise<string | undefined> { return this.values.get(id); }
  set(provider: ApiTeacherProvider, value: string): void { this.values.set(`teacher:${provider}`, value); }
  remove(provider: ApiTeacherProvider): void { this.values.delete(`teacher:${provider}`); }
}

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status, headers: { "content-type": "application/json" } });
}
async function capturedError<T>(promise: Promise<T>): Promise<Error> {
  try { await promise; throw new Error("Expected the operation to fail."); }
  catch (error) { return error instanceof Error ? error : new Error(String(error)); }
}

describe("bounded API teacher model discovery", () => {
  it("makes no network request when the selected provider has no configured credential", async () => {
    const fetcher = vi.fn(), discovery = new ApiTeacherModelDiscoveryService(new FakeSecrets(), fetcher as typeof fetch);
    await expect(discovery.discover({ provider: "openai", requestId: "models-unconfigured" })).rejects.toThrow(/no openai.*no network request/i);
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("uses the fixed OpenAI list endpoint without returning credentials or claiming generation compatibility", async () => {
    const secrets = new FakeSecrets(), secret = "sk-fixture-openai-secret-123456"; secrets.set("openai", secret);
    const fetcher = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      expect(String(input)).toBe(API_TEACHER_MODEL_ENDPOINTS.openai); expect(init?.method).toBe("GET"); expect(init?.redirect).toBe("error");
      expect(new Headers(init?.headers).get("authorization")).toBe(`Bearer ${secret}`);
      return json({ object: "list", data: [{ id: "gpt-fixture", object: "model", created: 1_700_000_000, owned_by: "openai", shutdown_date: null }] });
    });
    const result = await new ApiTeacherModelDiscoveryService(secrets, fetcher as typeof fetch)
      .discover({ provider: "openai", requestId: "models-openai" });
    expect(result).toMatchObject({ provider: "openai", pageCount: 1, complete: true, generationCompatibilityVerified: false,
      models: [{ provider: "openai", id: "gpt-fixture", owner: "openai" }] });
    expect(JSON.stringify(result)).not.toContain(secret); expect(JSON.stringify(result)).not.toContain("Bearer");
  });

  it("follows Anthropic after_id pagination and preserves provider lifecycle metadata", async () => {
    const secrets = new FakeSecrets(); secrets.set("anthropic", "anthropic-fixture-secret"); let page = 0;
    const fetcher = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = new URL(String(input)); expect(url.origin + url.pathname).toBe(API_TEACHER_MODEL_ENDPOINTS.anthropic);
      expect(url.searchParams.get("limit")).toBe("1000"); expect(new Headers(init?.headers).get("anthropic-version")).toBe("2023-06-01");
      expect(new Headers(init?.headers).get("x-api-key")).toBe("anthropic-fixture-secret");
      if (page++ === 0) { expect(url.searchParams.has("after_id")).toBe(false); return json({ data: [{ id: "claude-fixture-a", display_name: "Claude Fixture A", created_at: "2026-01-01T00:00:00Z", lifecycle: "active", retires_at: null, max_input_tokens: 200000, max_tokens: 64000 }], first_id: "claude-fixture-a", last_id: "claude-fixture-a", has_more: true }); }
      expect(url.searchParams.get("after_id")).toBe("claude-fixture-a");
      return json({ data: [{ id: "claude-fixture-b", display_name: "Claude Fixture B", created_at: "2025-01-01T00:00:00Z", lifecycle: "deprecated", retires_at: "2027-01-01T00:00:00Z" }], first_id: "claude-fixture-b", last_id: "claude-fixture-b", has_more: false });
    });
    const result = await new ApiTeacherModelDiscoveryService(secrets, fetcher as typeof fetch)
      .discover({ provider: "anthropic", requestId: "models-anthropic" });
    expect(result.pageCount).toBe(2); expect(result.models.map(model => model.id)).toEqual(["claude-fixture-a", "claude-fixture-b"]);
    expect(result.models[1]).toMatchObject({ lifecycle: "deprecated", retirementAt: "2027-01-01T00:00:00.000Z" });
  });

  it("follows Gemini pageToken pagination with the official key header and preserves resource identity", async () => {
    const secrets = new FakeSecrets(); secrets.set("gemini", "AIzaFixtureSecret_1234567890123456"); let page = 0;
    const fetcher = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = new URL(String(input)); expect(url.origin + url.pathname).toBe(API_TEACHER_MODEL_ENDPOINTS.gemini);
      expect(url.searchParams.get("pageSize")).toBe("1000"); expect(new Headers(init?.headers).get("x-goog-api-key")).toBe("AIzaFixtureSecret_1234567890123456");
      if (page++ === 0) { expect(url.searchParams.has("pageToken")).toBe(false); return json({ models: [{ name: "models/gemini-fixture-001", baseModelId: "gemini-fixture", version: "001", displayName: "Gemini Fixture", description: "Fixture only", inputTokenLimit: 1234, outputTokenLimit: 321, supportedGenerationMethods: ["generateContent"] }], nextPageToken: "opaque+/= token" }); }
      expect(url.searchParams.get("pageToken")).toBe("opaque+/= token");
      return json({ models: [{ name: "models/gemini-other-001", baseModelId: "gemini-other", version: "001", displayName: "Gemini Other" }] });
    });
    const result = await new ApiTeacherModelDiscoveryService(secrets, fetcher as typeof fetch)
      .discover({ provider: "gemini", requestId: "models-gemini" });
    expect(result.pageCount).toBe(2); expect(result.models[0]).toMatchObject({ id: "gemini-fixture", providerResourceName: "models/gemini-fixture-001",
      inputTokenLimit: 1234, outputTokenLimit: 321, supportedMethods: ["generateContent"] });
  });

  it("binds active requests to provider/request identity and supports exact cancellation", async () => {
    const secrets = new FakeSecrets(); secrets.set("openai", "sk-fixture-cancel-secret-123456");
    const fetcher = vi.fn((_input: string | URL | Request, init?: RequestInit) => new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
    }));
    const discovery = new ApiTeacherModelDiscoveryService(secrets, fetcher as typeof fetch);
    const first = discovery.discover({ provider: "openai", requestId: "models-cancel" });
    const repeated = discovery.discover({ provider: "openai", requestId: "models-cancel" });
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1));
    expect(() => discovery.cancel({ provider: "anthropic", requestId: "models-cancel" })).toThrow(/another provider/i);
    expect(discovery.cancel({ provider: "openai", requestId: "models-cancel" })).toMatchObject({ cancelled: true });
    await expect(first).rejects.toThrow(/cancelled/i); await expect(repeated).rejects.toThrow(/cancelled/i); expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("rejects redirects, redacts provider errors, and retires a response after credential change", async () => {
    const secrets = new FakeSecrets(), oldSecret = "sk-fixture-old-secret-123456789", nextSecret = "sk-fixture-next-secret-987654321";
    secrets.set("openai", oldSecret);
    const redirected = new Response(JSON.stringify({ data: [] }), { status: 200 }); Object.defineProperty(redirected, "redirected", { value: true });
    const redirectDiscovery = new ApiTeacherModelDiscoveryService(secrets, vi.fn(async () => redirected) as typeof fetch);
    await expect(redirectDiscovery.discover({ provider: "openai", requestId: "models-redirect" })).rejects.toThrow(/redirect/i);

    const errorDiscovery = new ApiTeacherModelDiscoveryService(secrets, vi.fn(async () => json({ error: { message: `bad credential ${oldSecret}` } }, 401)) as typeof fetch);
    const error = await capturedError(errorDiscovery.discover({ provider: "openai", requestId: "models-error" }));
    expect(error.message).toContain("[REDACTED]"); expect(error.message).not.toContain(oldSecret);

    let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
    const staleDiscovery = new ApiTeacherModelDiscoveryService(secrets, vi.fn(async () => { await gate; return json({ object: "list", data: [{ id: "stale-model" }] }); }) as typeof fetch);
    const pending = staleDiscovery.discover({ provider: "openai", requestId: "models-stale" });
    await Promise.resolve(); secrets.set("openai", nextSecret); staleDiscovery.credentialChanged("openai"); release();
    const staleError = await capturedError(pending);
    expect(staleError.message).toMatch(/credential changed/i); expect(staleError.message).not.toContain(oldSecret); expect(staleError.message).not.toContain(nextSecret);
  });
});
