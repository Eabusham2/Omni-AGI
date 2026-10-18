import { afterEach, expect, it, vi } from "vitest";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ExternalAdviceService } from "../src/main/externalAdviceService";
import type { ExternalAdviceChannel, ExternalAdviceChannelConfiguration, ExternalAdviceModelDiscoveryInput } from "../src/shared/externalAdvice";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
const json = (value: unknown) => new Response(JSON.stringify(value), { headers: { "content-type": "application/json" } });
async function setup(fetcher: typeof fetch) {
  const root = await mkdtemp(join(tmpdir(), "omni-advice-models-")); roots.push(root); const brainRoot = join(root, "brain"); await mkdir(brainRoot);
  const values = new Map<string, string>([["teacher:openrouter", "MUST-NOT-USE-TEACHER-KEY"]]);
  const secrets = { get: vi.fn(async (id: string) => values.get(id)), has: vi.fn(async (id: string) => values.has(id)),
    set: vi.fn(async (id: string, value: string) => { values.set(id, value); return "encrypted" as const; }),
    remove: vi.fn(async (id: string) => values.delete(id)), persistence: () => "encrypted" as const };
  const client = { stream: vi.fn(async () => { throw new Error("Metadata discovery invoked conversation/model dispatch"); }) };
  const service = new ExternalAdviceService(client, secrets, { brainRoot: () => brainRoot, deviceDirectory: join(root, "device"),
    discoveryFetch: fetcher, admitWrite: async () => undefined,
    authority: async (brainId, channel) => ({ brainId, channel, enabled: false, ownerEpoch: "device", revision: 1 }),
    selectContext: async () => { throw new Error("Metadata discovery read user context"); } });
  async function configure(provider: ExternalAdviceChannelConfiguration["provider"], endpoint?: string, channel: ExternalAdviceChannel = "assistant") {
    const current = await service.settings("one");
    return service.configure("one", { channel, expectedRevision: current.revision, configuration: { enabled: false, provider, model: "",
      ...(endpoint ? { endpoint } : {}), maxOutputTokens: 50, context: { includeTask: false, includeActivity: false, maxCharacters: 0 } } });
  }
  async function request(requestId: string, channel: ExternalAdviceChannel = "assistant"): Promise<ExternalAdviceModelDiscoveryInput> {
    const current = await service.settings("one"), keys = await service.credentialStatus();
    return { channel, requestId, expectedSettingsRevision: current.revision,
      expectedCredentialRevision: keys.find(row => row.provider === current[channel].provider)!.revision };
  }
  return { service, values, secrets, client, configure, request };
}

it("advice catalog uses its own opaque device key and exact saved channel/revision, never teacher credentials or context", async () => {
  const fetcher = vi.fn<typeof fetch>(async (input, init) => {
    expect(String(input)).toContain("https://openrouter.ai/api/v1/models/user?");
    expect(new Headers(init?.headers).get("authorization")).toBe("Bearer advice-only-key"); expect(init?.body).toBeUndefined();
    return json({ data: [{ id: "vendor/allowed" }], total_count: 1 });
  });
  const f = await setup(fetcher); await f.service.saveCredential({ provider: "openrouter", credential: "advice-only-key" }); await f.configure("openrouter");
  const input = await f.request("catalog"), result = await f.service.discoverModels("one", input);
  expect(result).toMatchObject({ brainId: "one", channel: "assistant", requestId: "catalog", settingsRevision: input.expectedSettingsRevision,
    credentialRevision: input.expectedCredentialRevision, provider: "openrouter", models: [{ id: "vendor/allowed" }] });
  expect(f.secrets.get.mock.calls.some(([id]) => id.startsWith("teacher:"))).toBe(false); expect(f.client.stream).not.toHaveBeenCalled();
  await expect(f.service.discoverModels("one", { ...input, requestId: "stale", expectedCredentialRevision: 0 })).rejects.toThrow(/credential revision is stale/);
  expect(fetcher).toHaveBeenCalledTimes(1); expect(f.service.cancelModelDiscovery("one", { channel: "assistant", requestId: "catalog" }).cancelled).toBe(false);
  f.service.close();
});

it("exact queued Stop prevents dispatch and custom Ollama metadata destination cannot be switched by a late config reply", async () => {
  let release!: (response: Response) => void;
  const fetcher = vi.fn<typeof fetch>(async (input, init) => {
    expect(String(input)).toBe("http://127.0.0.1:19999/api/tags"); expect(new Headers(init?.headers).has("authorization")).toBe(false);
    return new Promise<Response>(resolve => { release = resolve; }); // Deliberately ignores Abort to test the post-await fence.
  });
  const f = await setup(fetcher); await f.configure("ollama", "http://127.0.0.1:19999/v1");
  const queued = await f.request("queued"), stopped = f.service.discoverModels("one", queued);
  expect(f.service.cancelModelDiscovery("one", { channel: "reviewer", requestId: "queued" }).cancelled).toBe(false);
  expect(f.service.cancelModelDiscovery("one", { channel: "assistant", requestId: "queued" }).cancelled).toBe(true);
  await expect(stopped).rejects.toThrow(/cancelled/); expect(fetcher).not.toHaveBeenCalled();
  const pending = f.service.discoverModels("one", await f.request("in-flight"));
  await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1));
  await f.configure("ollama", "http://127.0.0.1:20000/v1"); release(json({ models: [{ name: "stale-local:latest" }] }));
  await expect(pending).rejects.toThrow(/cancelled|retired/i); expect(f.client.stream).not.toHaveBeenCalled(); f.service.close();
});

it("retires in-flight account discovery on key change and preserves generic-local manual semantics", async () => {
  let release!: (response: Response) => void;
  const fetcher = vi.fn<typeof fetch>(async () => new Promise<Response>(resolve => { release = resolve; }));
  const f = await setup(fetcher); await f.service.saveCredential({ provider: "openrouter", credential: "old-advice-key" }); await f.configure("openrouter");
  const pending = f.service.discoverModels("one", await f.request("old-key")); await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1));
  await f.service.saveCredential({ provider: "openrouter", credential: "new-advice-key" }); release(json({ data: [{ id: "old-catalog" }] }));
  await expect(pending).rejects.toThrow(/credential changed|cancelled/i);
  await f.configure("openai-compatible-local", "http://127.0.0.1:8080/v1");
  await expect(f.service.discoverModels("one", await f.request("generic"))).rejects.toThrow(/manual model IDs/);
  expect(fetcher).toHaveBeenCalledTimes(1); f.service.close();
});
