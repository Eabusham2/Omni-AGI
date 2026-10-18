import { randomUUID, timingSafeEqual } from "node:crypto";
import { createServer, type Server } from "node:http";
import { assertSafeRemoteUrl, readResponseBounded } from "./brainService";
import type { SecureSecretStore } from "./secureSecretStore";
import type { McpOAuthStart, McpOAuthStatus } from "../shared/mcpAuth";
import { authorizationMetadataCandidates, canonicalMcpResource, createMcpPKCE,
  mcpAuthorizationURL, mcpCodeExchangeBody, parseMcpAuthorizationServer,
  parseMcpBearerChallenges, parseMcpProtectedResource, resourceMetadataCandidates,
  type McpAuthorizationServer, type McpBearerChallenge } from "./mcpOAuthProtocol";

interface Credentials {
  version: 1;
  resource: string;
  server: McpAuthorizationServer;
  clientId: string;
  clientSecret?: string;
  authMethod: "none" | "client_secret_basic" | "client_secret_post";
  accessToken?: string;
  refreshToken?: string;
  expiresAt?: number;
  scopes?: string[];
}
interface Flow {
  id: string;
  controller: AbortController;
  listener?: Server;
  timer: NodeJS.Timeout;
  consumed: boolean;
  completion?: Promise<void>;
  allowed: () => void;
}
const credentialsKey = (id: string): string => `mcp:oauth:${id}`;
const validToken = (value: unknown): value is string => typeof value === "string" &&
  value.length > 0 && value.length < 32_000 && !/[\0\r\n]/.test(value);
const validBearer = (value: unknown): value is string => validToken(value) && /^[A-Za-z0-9._~+/-]+=*$/.test(value);
const asObject = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid OAuth response.");
  return value as Record<string, unknown>;
};

/** Installation-local credentials only. An account login never grants a neural tool permission. */
export class McpOAuthService {
  private readonly statuses = new Map<string, McpOAuthStatus>();
  private readonly challenges = new Map<string, McpBearerChallenge>();
  private readonly flows = new Map<string, Flow>();
  private readonly refreshes = new Map<string, Promise<string>>();
  private readonly refreshControllers = new Map<string, AbortController>();
  private readonly generations = new Map<string, number>();

  constructor(private readonly secrets: SecureSecretStore, private readonly options: {
    openBrowser: (url: string) => Promise<void>;
    fetcher?: typeof fetch;
    /** Test-injected network validation; production always uses the remote-address boundary. */
    validateNetwork?: (url: URL, allowLoopback: boolean) => Promise<void>;
    onChange?: (id: string, status: McpOAuthStatus) => void;
    timeoutMs?: number;
  }) {}

  private update(id: string, status: McpOAuthStatus): void {
    this.statuses.set(id, status); this.options.onChange?.(id, structuredClone(status));
  }
  challenge(id: string, header: string | null): void {
    const values = parseMcpBearerChallenges(header);
    const challenge = values.find(value => value.resourceMetadata) ?? values[0] ?? {};
    this.challenges.set(id, challenge);
    if (!this.flows.has(id)) this.update(id, { state: "required", scopes: challenge.scopes });
  }
  async status(id: string): Promise<McpOAuthStatus> {
    const current = this.statuses.get(id);
    if (current) return structuredClone(current);
    const stored = await this.read(id);
    return stored?.accessToken ? { state: "authorized", issuer: stored.server.issuer,
      scopes: stored.scopes, persistence: this.secrets.persistence() } : { state: "none" };
  }
  private async read(id: string): Promise<Credentials | undefined> {
    const raw = await this.secrets.get(credentialsKey(id));
    if (!raw) return undefined;
    const value = asObject(JSON.parse(raw));
    if (value.version !== 1 || typeof value.resource !== "string" || typeof value.clientId !== "string" ||
      !["none", "client_secret_basic", "client_secret_post"].includes(String(value.authMethod))) throw new Error("Invalid stored MCP authorization.");
    const server = asObject(value.server);
    if (typeof server.issuer !== "string" || typeof server.tokenEndpoint !== "string") throw new Error("Invalid stored MCP authorization.");
    for (const endpoint of [server.issuer, server.tokenEndpoint, server.authorizationEndpoint,
      ...(server.registrationEndpoint ? [server.registrationEndpoint] : []), ...(server.revocationEndpoint ? [server.revocationEndpoint] : [])]) {
      const url = new URL(String(endpoint));
      if (url.protocol !== "https:" || url.username || url.password || url.hash) throw new Error("Invalid stored MCP authorization.");
    }
    canonicalMcpResource(value.resource);
    for (const token of [value.clientSecret, value.accessToken, value.refreshToken]) {
      if (token !== undefined && !validToken(token)) throw new Error("Invalid stored MCP authorization.");
    }
    if (value.accessToken !== undefined && !validBearer(value.accessToken)) throw new Error("Invalid stored MCP Bearer token.");
    if (value.expiresAt !== undefined && (typeof value.expiresAt !== "number" || !Number.isFinite(value.expiresAt))) throw new Error("Invalid stored MCP authorization expiry.");
    return value as unknown as Credentials;
  }
  private async validateURL(url: URL, allowLoopback = false): Promise<void> {
    await (this.options.validateNetwork ?? (async (target: URL, local: boolean) => {
      await assertSafeRemoteUrl(target, { allowLoopback: local });
    }))(url, allowLoopback);
  }
  private async request(url: string, init: RequestInit, signal: AbortSignal, allowLoopback = false): Promise<Response> {
    const parsed = new URL(url);
    await this.validateURL(parsed, allowLoopback);
    signal.throwIfAborted();
    return (this.options.fetcher ?? fetch)(parsed.href, { ...init, redirect: "error",
      signal: AbortSignal.any([signal, AbortSignal.timeout(30_000)]) });
  }
  private async json(url: string, signal: AbortSignal, allowLoopback = false, init: RequestInit = {}): Promise<Record<string, unknown> | undefined> {
    const response = await this.request(url, { ...init, headers: { accept: "application/json", ...init.headers } }, signal, allowLoopback);
    if (response.status === 404) { await response.body?.cancel(); return undefined; }
    if (!response.ok) { await response.body?.cancel(); throw new Error(`OAuth endpoint returned HTTP ${response.status}.`); }
    try { return asObject(JSON.parse((await readResponseBounded(response, 1024 * 1024)).toString("utf8"))); }
    catch { throw new Error("OAuth endpoint returned invalid or oversized metadata."); }
  }
  private async discover(id: string, endpoint: string, requestedIssuer: string | undefined, signal: AbortSignal): Promise<{
    resource: string; server: McpAuthorizationServer; scopes?: string[];
  }> {
    const challenge = this.challenges.get(id);
    let resource;
    for (const candidate of resourceMetadataCandidates(endpoint, challenge)) {
      const metadata = await this.json(candidate, signal, new URL(endpoint).protocol === "http:");
      if (metadata) { resource = parseMcpProtectedResource(metadata, endpoint); break; }
    }
    if (!resource) throw new Error("The MCP server has no protected-resource metadata.");
    const issuer = requestedIssuer ?? resource.authorizationServers[0]!;
    if (!resource.authorizationServers.includes(issuer)) throw new Error("The selected issuer is not advertised by this MCP server.");
    let server;
    for (const candidate of authorizationMetadataCandidates(issuer)) {
      const metadata = await this.json(candidate, signal);
      if (metadata) { server = parseMcpAuthorizationServer(metadata, issuer); break; }
    }
    if (!server) throw new Error("The authorization server has no supported discovery metadata.");
    return { resource: resource.resource, server, scopes: challenge?.scopes ?? resource.scopes };
  }
  private auth(credential: Credentials, body: URLSearchParams): HeadersInit {
    if (credential.authMethod === "none") return {};
    if (!credential.clientSecret) throw new Error("This authorization server requires client credentials.");
    if (credential.authMethod === "client_secret_post") { body.set("client_secret", credential.clientSecret); return {}; }
    body.delete("client_id");
    const encode = (value: string): string => new URLSearchParams({ value }).toString().slice(6);
    return { authorization: `Basic ${Buffer.from(`${encode(credential.clientId)}:${encode(credential.clientSecret)}`).toString("base64")}` };
  }
  private async token(credential: Credentials, body: URLSearchParams, signal: AbortSignal): Promise<Credentials> {
    const metadata = await this.json(credential.server.tokenEndpoint, signal, false, { method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded", ...this.auth(credential, body) }, body: body.toString() });
    if (!metadata || !validBearer(metadata.access_token) || typeof metadata.token_type !== "string" || metadata.token_type.toLowerCase() !== "bearer") throw new Error("OAuth returned no valid Bearer token.");
    if (metadata.refresh_token !== undefined && !validToken(metadata.refresh_token)) throw new Error("OAuth returned an invalid refresh token.");
    if (metadata.expires_in !== undefined && (typeof metadata.expires_in !== "number" || !Number.isFinite(metadata.expires_in) || metadata.expires_in < 0 || metadata.expires_in > Number.MAX_SAFE_INTEGER / 1000)) throw new Error("OAuth returned an invalid token lifetime.");
    let grantedScopes = credential.scopes;
    if (metadata.scope !== undefined) {
      if (typeof metadata.scope !== "string") throw new Error("OAuth returned invalid scopes.");
      grantedScopes = parseMcpBearerChallenges(`Bearer scope="${metadata.scope.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`)[0]?.scopes;
    }
    return { ...credential, accessToken: metadata.access_token,
      ...(metadata.refresh_token !== undefined ? { refreshToken: metadata.refresh_token as string } : {}),
      expiresAt: metadata.expires_in === undefined ? undefined : Date.now() + metadata.expires_in * 1000, scopes: grantedScopes };
  }
  async begin(id: string, endpoint: string, input: McpOAuthStart, allowed: () => void): Promise<McpOAuthStatus> {
    allowed();
    if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("Invalid MCP client configuration.");
    if (this.flows.has(id)) throw new Error("This MCP server already has a pending login.");
    if (input.redirectPort !== undefined && (!Number.isInteger(input.redirectPort) || input.redirectPort < 1024 || input.redirectPort > 65535)) throw new Error("Invalid OAuth redirect port.");
    for (const value of [input.clientId, input.clientSecret]) if (value !== undefined && !validToken(value)) throw new Error("Invalid OAuth client configuration.");
    const controller = new AbortController();
    const flow: Flow = { id: randomUUID(), controller, consumed: false,
      allowed,
      timer: setTimeout(() => this.cancel(id, "The MCP login timed out."), this.options.timeoutMs ?? 10 * 60_000) };
    this.flows.set(id, flow);
    this.update(id, { state: "authorizing", flowId: flow.id });
    const generation = this.generations.get(id) ?? 0;
    const pkce = createMcpPKCE();
    try {
      const discovery = await this.discover(id, endpoint, input.issuer, controller.signal);
      let credential: Credentials;
      let redirectUri = "";
      const listener = createServer({ maxHeaderSize: 8192 }, (request, response) => {
        response.setHeader("cache-control", "no-store"); response.setHeader("referrer-policy", "no-referrer");
        response.setHeader("content-type", "text/plain; charset=utf-8"); response.setHeader("x-content-type-options", "nosniff");
        let target: URL;
        try { target = new URL(request.url ?? "/", redirectUri); }
        catch { response.writeHead(400); response.end("This login callback is not valid."); return; }
        const expected = new URL(redirectUri);
        const state = target.searchParams.get("state") ?? "";
        const validState = Buffer.byteLength(state) === Buffer.byteLength(pkce.state) && timingSafeEqual(Buffer.from(state), Buffer.from(pkce.state));
        if (request.method !== "GET" || request.headers.host !== expected.host || target.origin !== expected.origin ||
          target.pathname !== expected.pathname || target.searchParams.getAll("state").length !== 1 || !validState || flow.consumed) {
          response.writeHead(400); response.end("This login callback is not valid."); return;
        }
        flow.consumed = true;
        const code = target.searchParams.get("code");
        if (target.searchParams.has("error") || !validToken(code) || target.searchParams.getAll("code").length !== 1 ||
          (target.searchParams.has("iss") && target.searchParams.get("iss") !== discovery.server.issuer)) {
          response.writeHead(400); response.end("Login was declined or returned invalid authorization.");
          this.cancel(id, "Login was declined or returned invalid authorization."); return;
        }
        response.end("Finishing sign-in. You can return to Omni AGI Studio.");
        listener.close();
        flow.completion = (async () => {
          const next = await this.token(credential, mcpCodeExchangeBody({ clientId: credential.clientId,
            redirectUri, resource: credential.resource, code, verifier: pkce.verifier }), controller.signal);
          allowed(); controller.signal.throwIfAborted();
          if ((this.generations.get(id) ?? 0) !== generation || this.flows.get(id) !== flow) throw new Error("This MCP login was replaced.");
          await this.secrets.set(credentialsKey(id), JSON.stringify(next));
          allowed(); controller.signal.throwIfAborted();
          this.update(id, { state: "authorized", issuer: next.server.issuer, scopes: next.scopes, persistence: this.secrets.persistence() });
        })().catch(() => {
          if (this.flows.get(id) === flow) this.update(id, { state: "error", error: "The MCP login could not finish. Retry sign-in." });
        }).finally(() => this.finish(id, flow));
      });
      flow.listener = listener;
      await new Promise<void>((resolve, reject) => {
        listener.once("error", reject);
        listener.listen(input.redirectPort ?? 0, "127.0.0.1", resolve);
      });
      controller.signal.throwIfAborted(); allowed();
      const address = listener.address();
      if (!address || typeof address === "string") throw new Error("OAuth callback could not bind a local port.");
      redirectUri = `http://127.0.0.1:${address.port}/oauth/callback`;
      const prior = await this.read(id);
      const reused = prior?.resource === discovery.resource && prior.server.issuer === discovery.server.issuer ? prior : undefined;
      let clientId = input.clientId ?? reused?.clientId;
      let clientSecret = input.clientSecret ?? (input.clientId && input.clientId !== reused?.clientId ? undefined : reused?.clientSecret);
      if (!clientId && input.clientMetadataUrl && discovery.server.clientMetadataSupported) {
        const metadataUrl = new URL(input.clientMetadataUrl);
        if (metadataUrl.protocol !== "https:" || metadataUrl.pathname === "/" || metadataUrl.search || metadataUrl.hash || metadataUrl.username || metadataUrl.password) throw new Error("Client metadata must be hosted at an HTTPS document URL.");
        const metadata = await this.json(metadataUrl.href, controller.signal);
        if (!metadata || metadata.client_id !== metadataUrl.href || typeof metadata.client_name !== "string" ||
          !Array.isArray(metadata.redirect_uris) || !metadata.redirect_uris.includes(redirectUri)) throw new Error("Hosted client metadata does not match this client and redirect URI.");
        clientId = metadataUrl.href;
      }
      if (!clientId && discovery.server.registrationEndpoint) {
        const registration = await this.json(discovery.server.registrationEndpoint, controller.signal, false, { method: "POST",
          headers: { "content-type": "application/json" }, body: JSON.stringify({ client_name: "Omni AGI Studio", redirect_uris: [redirectUri],
            grant_types: ["authorization_code", "refresh_token"], response_types: ["code"], token_endpoint_auth_method: "none" }) });
        if (!registration || !validToken(registration.client_id) || (registration.client_secret !== undefined && !validToken(registration.client_secret))) throw new Error("OAuth client registration failed.");
        if (registration.redirect_uris !== undefined && (!Array.isArray(registration.redirect_uris) || !registration.redirect_uris.includes(redirectUri))) throw new Error("OAuth registration changed its callback.");
        if (registration.token_endpoint_auth_method !== undefined && registration.token_endpoint_auth_method !== "none") throw new Error("OAuth registration changed its authentication method.");
        clientId = registration.client_id; clientSecret = registration.client_secret as string | undefined;
      }
      if (!clientId) throw new Error("Register this app with the authorization provider and supply its client ID, or a supported hosted client-metadata document.");
      const methods = discovery.server.tokenAuthenticationMethods;
      const authMethod = clientSecret ? methods.includes("client_secret_basic") ? "client_secret_basic" : methods.includes("client_secret_post") ? "client_secret_post" : undefined
        : methods.includes("none") || (input.clientMetadataUrl === clientId && discovery.server.clientMetadataSupported) ? "none" : undefined;
      if (!authMethod) throw new Error("The provider has no supported token authentication method for this client.");
      credential = { version: 1, resource: discovery.resource, server: discovery.server, clientId,
        ...(clientSecret ? { clientSecret } : {}), authMethod, scopes: discovery.scopes };
      allowed(); controller.signal.throwIfAborted();
      this.update(id, { state: "authorizing", flowId: flow.id, issuer: credential.server.issuer, scopes: credential.scopes });
      const authorizationURL = mcpAuthorizationURL(credential.server, { clientId, redirectUri,
        resource: credential.resource, challenge: pkce.challenge, state: pkce.state, scopes: credential.scopes });
      await this.validateURL(new URL(authorizationURL));
      controller.signal.throwIfAborted(); allowed();
      await this.options.openBrowser(authorizationURL);
      return this.status(id);
    } catch (error) {
      const current = this.flows.get(id) === flow;
      this.finish(id, flow);
      if (current) this.update(id, { state: "error", error: error instanceof Error ? error.message : "The MCP login could not start." });
      throw error;
    }
  }
  private finish(id: string, flow: Flow): void {
    clearTimeout(flow.timer); flow.listener?.close(); flow.controller.abort();
    if (this.flows.get(id) === flow) this.flows.delete(id);
  }
  cancel(id: string, reason = "The MCP login was stopped."): boolean {
    const flow = this.flows.get(id);
    if (!flow) return false;
    this.generations.set(id, (this.generations.get(id) ?? 0) + 1);
    this.finish(id, flow); this.update(id, { state: "required", error: reason }); return true;
  }
  revalidate(id: string): void {
    const flow = this.flows.get(id);
    if (flow) { try { flow.allowed(); } catch { this.cancel(id, "This MCP login no longer has access."); } }
  }
  async forget(id: string): Promise<void> {
    this.cancel(id);
    this.generations.set(id, (this.generations.get(id) ?? 0) + 1);
    this.refreshControllers.get(id)?.abort();
    await this.secrets.remove(credentialsKey(id)); this.update(id, { state: "none" });
    this.challenges.delete(id);
  }
  /** Host transaction rollback only. Never expose this record through preload. */
  credentialSnapshot(id: string): Promise<string | undefined> { return this.secrets.get(credentialsKey(id)); }
  async restoreCredentialSnapshot(id: string, snapshot: string | undefined): Promise<void> {
    await this.forget(id);
    if (snapshot !== undefined) await this.secrets.set(credentialsKey(id), snapshot);
    const restored = await this.read(id);
    this.update(id, restored?.accessToken ? { state: "authorized", issuer: restored.server.issuer,
      scopes: restored.scopes, persistence: this.secrets.persistence() } : { state: "none" });
  }
  async accessToken(id: string, endpoint: string, allowed: () => void, signal?: AbortSignal): Promise<string | undefined> {
    allowed();
    signal?.throwIfAborted();
    const credential = await this.read(id);
    if (!credential?.accessToken) return undefined;
    if (credential.resource !== canonicalMcpResource(endpoint)) throw new Error("MCP authorization belongs to a different resource.");
    if (credential.expiresAt === undefined || credential.expiresAt > Date.now() + 30_000) return credential.accessToken;
    if (!credential.refreshToken) { this.update(id, { state: "required" }); throw new Error("MCP authorization expired. Sign in again."); }
    const existing = this.refreshes.get(id);
    if (existing) { const value = await existing; allowed(); signal?.throwIfAborted(); return value; }
    const generation = this.generations.get(id) ?? 0;
    const controller = new AbortController();
    this.refreshControllers.set(id, controller);
    const operation = (async () => {
      const body = new URLSearchParams({ grant_type: "refresh_token", refresh_token: credential.refreshToken!,
        client_id: credential.clientId, resource: credential.resource });
      const next = await this.token(credential, body, AbortSignal.any([controller.signal, signal ?? new AbortController().signal, AbortSignal.timeout(30_000)]));
      allowed();
      if ((this.generations.get(id) ?? 0) !== generation) throw new Error("MCP authorization was removed.");
      await this.secrets.set(credentialsKey(id), JSON.stringify(next));
      allowed();
      controller.signal.throwIfAborted();
      if ((this.generations.get(id) ?? 0) !== generation) throw new Error("MCP authorization was removed.");
      this.update(id, { state: "authorized", issuer: next.server.issuer, scopes: next.scopes, persistence: this.secrets.persistence() });
      return next.accessToken!;
    })().catch(error => {
      if ((this.generations.get(id) ?? 0) === generation) this.update(id, { state: "required", error: "MCP authorization could not refresh. Sign in again." });
      throw error;
    });
    this.refreshes.set(id, operation);
    try { return await operation; }
    finally {
      if (this.refreshes.get(id) === operation) this.refreshes.delete(id);
      if (this.refreshControllers.get(id) === controller) this.refreshControllers.delete(id);
    }
  }
  async disconnect(id: string, endpoint: string, allowed: () => void): Promise<void> {
    allowed(); this.cancel(id);
    this.generations.set(id, (this.generations.get(id) ?? 0) + 1);
    const credential = await this.read(id);
    await this.forget(id);
    if (credential?.server.revocationEndpoint && credential.resource === canonicalMcpResource(endpoint)) {
      const token = credential.refreshToken ?? credential.accessToken;
      if (token) {
        const body = new URLSearchParams({ token, client_id: credential.clientId, token_type_hint: credential.refreshToken ? "refresh_token" : "access_token" });
        const response = await this.request(credential.server.revocationEndpoint, { method: "POST", headers: {
          "content-type": "application/x-www-form-urlencoded", ...this.auth(credential, body) }, body: body.toString() }, AbortSignal.timeout(30_000));
        await response.body?.cancel();
        if (!response.ok) throw new Error(`Local credentials were removed; remote revocation returned HTTP ${response.status}.`);
      }
    }
  }
  dispose(): void {
    for (const id of this.flows.keys()) this.cancel(id);
    for (const [id, controller] of this.refreshControllers) {
      this.generations.set(id, (this.generations.get(id) ?? 0) + 1); controller.abort();
    }
  }
}
