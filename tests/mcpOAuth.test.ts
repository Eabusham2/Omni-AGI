import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { McpOAuthService } from "../src/main/mcpOAuth";
import { SecureSecretStore } from "../src/main/secureSecretStore";
import { McpClientService } from "../src/main/mcpClient";
import { McpRegistryStore } from "../src/main/mcpRegistry";

describe("MCP account authorization", () => {
  let root: string;
  const services: Array<{ dispose(): void }> = [];
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-oauth-")); });
  afterEach(async () => { services.splice(0).forEach(service => service.dispose()); await rm(root, { recursive: true, force: true }); });
  function fixture(options: { dynamic?: boolean; tokenLife?: number; rotate?: boolean; deniedToken?: boolean; withRpc?: boolean; refreshGate?: Promise<void> } = {}) {
    const secretPath = join(root, "secrets.json");
    const secrets = new SecureSecretStore(secretPath, { available: () => true,
      encrypt: value => Buffer.from(value.split("").reverse().join("")), decrypt: value => value.toString().split("").reverse().join("") });
    const opened: string[] = [];
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    let refreshing = 0;
    const fetcher = vi.fn(async (input: unknown, init?: RequestInit): Promise<Response> => {
      const url = String(input); requests.push({ url, init });
      const json = (value: unknown): Response => new Response(JSON.stringify(value), { headers: { "content-type": "application/json" } });
      if (options.withRpc && url === "https://mcp.example/mcp") {
        const request = JSON.parse(String(init?.body));
        if (new Headers(init?.headers).get("authorization") !== "Bearer private-access" || request.method === "tools/call") {
          return new Response(null, { status: 401, headers: { "www-authenticate": 'Bearer scope="basic additional"' } });
        }
        if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
        return json({ jsonrpc: "2.0", id: request.id, result: request.method === "initialize"
          ? { protocolVersion: "2025-11-25", capabilities: { tools: {} } }
          : { tools: [{ name: "inspect", inputSchema: { type: "object" } }] } });
      }
      if (url === "https://mcp.example/.well-known/oauth-protected-resource/mcp" || url === "https://metadata.example/resource") return json({ resource: "https://mcp.example/mcp", authorization_servers: ["https://login.example/tenant"], scopes_supported: ["basic"] });
      if (url === "https://login.example/.well-known/oauth-authorization-server/tenant") return new Response(null, { status: 404 });
      if (url === "https://login.example/.well-known/openid-configuration/tenant") return json({ issuer: "https://login.example/tenant",
        authorization_endpoint: "https://login.example/authorize", token_endpoint: "https://login.example/token",
        revocation_endpoint: "https://login.example/revoke", ...(options.dynamic ? { registration_endpoint: "https://login.example/register" } : {}),
        code_challenge_methods_supported: ["S256"], token_endpoint_auth_methods_supported: ["none", "client_secret_basic"] });
      if (url === "https://login.example/register") return json({ client_id: "dynamic-client", token_endpoint_auth_method: "none",
        redirect_uris: JSON.parse(String(init?.body)).redirect_uris });
      if (url === "https://login.example/token") {
        if (options.deniedToken) return new Response(JSON.stringify({ error_description: "do-not-leak-private" }), { status: 400 });
        const body = new URLSearchParams(String(init?.body));
        if (body.get("grant_type") === "refresh_token") { refreshing++; await options.refreshGate; }
        return json({ token_type: "Bearer", access_token: refreshing ? `access-${refreshing}` : "private-access",
          refresh_token: options.rotate && refreshing ? "rotated-refresh" : "private-refresh", expires_in: refreshing ? 3600 : options.tokenLife ?? 3600,
          scope: "basic additional" });
      }
      if (url === "https://login.example/revoke") return new Response(null, { status: 200 });
      throw new Error(`Unexpected fixture endpoint ${url}`);
    });
    const validateNetwork = vi.fn(async () => undefined);
    const service = new McpOAuthService(secrets, { fetcher: fetcher as typeof fetch, validateNetwork,
      openBrowser: async url => { opened.push(url); } }); services.push(service);
    const callback = async (params: Record<string, string> = {}): Promise<Response> => {
      const auth = new URL(opened.at(-1)!);
      const target = new URL(auth.searchParams.get("redirect_uri")!);
      target.searchParams.set("state", auth.searchParams.get("state")!);
      target.searchParams.set("code", "one-use-code");
      for (const [key, value] of Object.entries(params)) target.searchParams.set(key, value);
      return fetch(target);
    };
    return { secrets, secretPath, service, opened, requests, fetcher, validateNetwork, callback };
  }
  it("finishes an actual loopback callback with PKCE/resource binding and secrets outside registry/model state", async () => {
    const f = fixture(); const allowed = vi.fn();
    f.service.challenge("coding", 'Bearer resource_metadata="https://metadata.example/resource", scope="basic additional"');
    const start = await f.service.begin("coding", "https://mcp.example/mcp", { clientId: "pre-registered" }, allowed);
    expect(start).toMatchObject({ state: "authorizing", scopes: ["basic", "additional"] });
    const auth = new URL(f.opened[0]!);
    expect(auth.searchParams.get("resource")).toBe("https://mcp.example/mcp");
    expect(auth.searchParams.has("code_verifier")).toBe(false);
    expect((await f.callback({ state: "invalid" })).status).toBe(400);
    expect((await f.service.status("coding")).state).toBe("authorizing");
    expect((await f.callback()).status).toBe(200);
    await vi.waitFor(async () => expect((await f.service.status("coding")).state).toBe("authorized"));
    const token = f.requests.find(request => request.url.endsWith("/token"))!;
    const body = new URLSearchParams(String(token.init?.body));
    expect(body.get("code_verifier")).toHaveLength(43);
    expect(body.get("resource")).toBe(auth.searchParams.get("resource"));
    expect(await f.service.accessToken("coding", "https://mcp.example/mcp", allowed)).toBe("private-access");
    expect(await readFile(f.secretPath, "utf8")).not.toContain("private-access");
    expect(JSON.stringify(await f.service.status("coding"))).not.toContain("private-refresh");
    expect(f.validateNetwork).toHaveBeenCalled();
  });
  it("uses dynamic public-client registration when no static client exists", async () => {
    const f = fixture({ dynamic: true });
    await f.service.begin("coding", "https://mcp.example/mcp", {}, () => undefined);
    const registration = f.requests.find(request => request.url.endsWith("/register"))!;
    expect(JSON.parse(String(registration.init?.body))).toMatchObject({ grant_types: ["authorization_code", "refresh_token"], token_endpoint_auth_method: "none" });
    expect(new URL(f.opened[0]!).searchParams.get("client_id")).toBe("dynamic-client");
    await f.callback(); await vi.waitFor(async () => expect((await f.service.status("coding")).state).toBe("authorized"));
  });
  it("single-flights expired refreshes and persists rotated tokens before subsequent requests", async () => {
    const f = fixture({ tokenLife: 0, rotate: true });
    await f.service.begin("coding", "https://mcp.example/mcp", { clientId: "client" }, () => undefined);
    await f.callback(); await vi.waitFor(async () => expect((await f.service.status("coding")).state).toBe("authorized"));
    expect(await Promise.all([f.service.accessToken("coding", "https://mcp.example/mcp", () => undefined), f.service.accessToken("coding", "https://mcp.example/mcp", () => undefined)])).toEqual(["access-1", "access-1"]);
    expect(f.requests.filter(request => request.url.endsWith("/token"))).toHaveLength(2);
    expect(JSON.parse((await f.secrets.get("mcp:oauth:coding"))!).refreshToken).toBe("rotated-refresh");
    await expect(f.service.accessToken("coding", "https://other.example/mcp", () => undefined)).rejects.toThrow(/different resource/);
    await f.service.disconnect("coding", "https://mcp.example/mcp", () => undefined);
    expect(await f.secrets.has("mcp:oauth:coding")).toBe(false);
    expect(f.requests.at(-1)?.url).toBe("https://login.example/revoke");
    expect(new URLSearchParams(String(f.requests.at(-1)?.init?.body)).get("token")).toBe("rotated-refresh");
  });
  it("cancels only when the initiating instance loses access; no token is granted", async () => {
    const f = fixture(); let enabled = true;
    await f.service.begin("coding", "https://mcp.example/mcp", { clientId: "client" }, () => { if (!enabled) throw new Error("Off"); });
    f.service.revalidate("coding"); expect((await f.service.status("coding")).state).toBe("authorizing");
    enabled = false; f.service.revalidate("coding");
    expect((await f.service.status("coding")).state).toBe("required");
    expect(await f.secrets.has("mcp:oauth:coding")).toBe(false);
    expect(f.requests.some(request => request.url.endsWith("/token"))).toBe(false);
  });
  it("cannot restore credentials or publish a token when a late refresh finishes after disconnect", async () => {
    let release!: () => void;
    const f = fixture({ tokenLife: 0, refreshGate: new Promise<void>(resolve => { release = resolve; }) });
    await f.service.begin("coding", "https://mcp.example/mcp", { clientId: "client" }, () => undefined);
    await f.callback(); await vi.waitFor(async () => expect((await f.service.status("coding")).state).toBe("authorized"));
    const refresh = f.service.accessToken("coding", "https://mcp.example/mcp", () => undefined).then(value => ({ value }), error => ({ error }));
    await vi.waitFor(() => expect(f.requests.filter(request => request.url.endsWith("/token"))).toHaveLength(2));
    await f.service.forget("coding"); release();
    expect(await refresh).toHaveProperty("error");
    expect(await f.secrets.has("mcp:oauth:coding")).toBe(false);
    expect((await f.service.status("coding")).state).toBe("none");
  });
  it("uses confidential-client Basic credentials only at the token endpoint, not the browser", async () => {
    const f = fixture();
    await f.service.begin("coding", "https://mcp.example/mcp", { clientId: "client:x", clientSecret: "private-secret" }, () => undefined);
    expect(f.opened[0]).not.toContain("private-secret");
    await f.callback(); await vi.waitFor(async () => expect((await f.service.status("coding")).state).toBe("authorized"));
    const token = f.requests.find(request => request.url.endsWith("/token"))!;
    expect(new Headers(token.init?.headers).get("authorization")).toBe(`Basic ${Buffer.from("client%3Ax:private-secret").toString("base64")}`);
    expect(new URLSearchParams(String(token.init?.body)).has("client_id")).toBe(false);
  });
  it("keeps failed login details private and requires an actual client instead of inventing one", async () => {
    const f = fixture({ deniedToken: true });
    await expect(f.service.begin("missing", "https://mcp.example/mcp", {}, () => undefined)).rejects.toThrow(/client ID/);
    await f.service.begin("coding", "https://mcp.example/mcp", { clientId: "client" }, () => undefined);
    await f.callback(); await vi.waitFor(async () => expect((await f.service.status("coding")).state).toBe("error"));
    expect(JSON.stringify(await f.service.status("coding"))).not.toContain("do-not-leak-private");
    expect(await f.secrets.has("mcp:oauth:coding")).toBe(false);
  });
  it("retains an HTTP registration requiring login without retried effects or implicit tool permission", async () => {
    const f = fixture();
    const brain = { registerExternalToolSchemas: vi.fn(), unregisterExternalToolSchemas: vi.fn(), learnStructuredExperience: vi.fn(),
      listToolPermissions: vi.fn(async () => []), setToolPermission: vi.fn() };
    let attempts = 0;
    const client = new McpClientService(join(root, "registry.json"), f.secrets, brain as never, (async () => {
      attempts++; return new Response(null, { status: 401, headers: { "www-authenticate": 'Bearer scope="read"' } });
    }) as typeof fetch);
    services.push(client); await client.initialize();
    const server = await client.add({ brainId: "first", id: "coding", label: "Coding", transport: "http", url: "https://mcp.example/mcp" });
    expect(server.authentication?.state).toBe("required");
    expect(server.connected).toBe(false); expect(attempts).toBe(1);
    expect((await client.list("first"))[0]?.id).toBe("coding");
    expect(brain.setToolPermission).not.toHaveBeenCalled();
    expect(brain.learnStructuredExperience).not.toHaveBeenCalled();
  });
  it("wires browser login into the production client, keeps Ask permissions, and never retries a rejected tool effect", async () => {
    const f = fixture({ withRpc: true });
    const permissions: Array<{ toolId: string; level: string }> = [];
    const brain = { registerExternalToolSchemas: vi.fn(), unregisterExternalToolSchemas: vi.fn(), learnStructuredExperience: vi.fn(async () => ({})),
      listToolPermissions: vi.fn(async () => permissions), setToolPermission: vi.fn(async (_brain: string, toolId: string, level: string) => { permissions.push({ toolId, level }); }) };
    const client = new McpClientService(join(root, "registry.json"), f.secrets, brain as never, f.fetcher as typeof fetch,
      { validateNetwork: f.validateNetwork, openBrowser: async url => { f.opened.push(url); } });
    services.push(client); await client.initialize();
    await client.add({ brainId: "first", id: "coding", label: "Coding", transport: "http", url: "https://mcp.example/mcp", bearerToken: "stale-bearer" });
    expect((await client.authStatus("first", "coding")).state).toBe("required");
    await client.authorize("first", "coding", { clientId: "client" });
    await f.callback(); await vi.waitFor(async () => expect((await client.authStatus("first", "coding")).state).toBe("authorized"));
    const refreshed = await client.refresh("first", "coding");
    expect(refreshed.connected).toBe(true);
    expect(permissions).toHaveLength(1); expect(permissions[0]?.level).toBe("ask");
    expect(JSON.stringify(brain.learnStructuredExperience.mock.calls)).not.toContain("private-access");
    expect(await readFile(join(root, "registry.json"), "utf8")).not.toContain("private-refresh");
    await expect(client.execute(refreshed.tools[0]!.id, "call", {}, new AbortController().signal, "first")).rejects.toThrow(/authentication/);
    expect(f.requests.filter(request => request.url === "https://mcp.example/mcp" && JSON.parse(String(request.init?.body)).method === "tools/call")).toHaveLength(1);
    expect((await client.authStatus("first", "coding")).state).toBe("required");
    await client.disconnectAuthorization("first", "coding");
    expect(await f.secrets.has("mcp:registry:coding")).toBe(false);
    expect(await f.secrets.has("mcp:oauth:coding")).toBe(false);
    await client.configure({ enabled: false });
    await expect(client.authorize("first", "coding", { clientId: "client" })).rejects.toThrow(/switched off/);
  });
  it("restores OAuth and bearer state if explicit credential replacement cannot publish", async () => {
    const f = fixture({ withRpc: true });
    const brain = { registerExternalToolSchemas: vi.fn(), unregisterExternalToolSchemas: vi.fn(), learnStructuredExperience: vi.fn(),
      listToolPermissions: vi.fn(async () => []), setToolPermission: vi.fn() };
    const client = new McpClientService(join(root, "registry.json"), f.secrets, brain as never, f.fetcher as typeof fetch,
      { validateNetwork: f.validateNetwork, openBrowser: async url => { f.opened.push(url); } });
    services.push(client); await client.initialize();
    await client.add({ brainId: "first", id: "coding", label: "Coding", transport: "http", url: "https://mcp.example/mcp", bearerToken: "original-bearer" });
    await client.authorize("first", "coding", { clientId: "client" });
    await f.callback(); await vi.waitFor(async () => expect((await client.authStatus("first", "coding")).state).toBe("authorized"));
    const original = await f.secrets.get("mcp:oauth:coding");
    const beforeCalls = f.requests.length;
    const publish = vi.spyOn(McpRegistryStore.prototype, "save").mockRejectedValueOnce(new Error("fixture publication failure"));
    try { await expect(client.updateCredentials("first", "coding", { bearerToken: "replacement" })).rejects.toThrow(/publication failure/); }
    finally { publish.mockRestore(); }
    expect(await f.secrets.get("mcp:registry:coding")).toBe("original-bearer");
    expect(await f.secrets.get("mcp:oauth:coding")).toBe(original);
    expect((await client.authStatus("first", "coding")).state).toBe("authorized");
    expect(f.requests).toHaveLength(beforeCalls);
    await expect(client.updateCredentials("first", "coding", { environment: { KEY: "not-http" } })).rejects.toThrow(/credential type/);
    await client.updateCredentials("first", "coding", { bearerToken: "" });
    expect((await client.authStatus("first", "coding")).state).toBe("none");
    expect(await f.secrets.has("mcp:registry:coding")).toBe(false);
    expect(f.requests).toHaveLength(beforeCalls);
  });
});
