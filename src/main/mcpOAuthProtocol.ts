/** MCP HTTP authorization primitives. No credential or tool grant is implicit.
 * Protocol source: https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization
 */
import { createHash, randomBytes } from "node:crypto";

export interface McpBearerChallenge {
  resourceMetadata?: string;
  scopes?: string[];
  error?: string;
}

export interface McpProtectedResource {
  resource: string;
  authorizationServers: string[];
  scopes?: string[];
}

export interface McpAuthorizationServer {
  issuer: string;
  authorizationEndpoint: string;
  tokenEndpoint: string;
  registrationEndpoint?: string;
  revocationEndpoint?: string;
  clientMetadataSupported: boolean;
  tokenAuthenticationMethods: string[];
}

const object = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("OAuth metadata must be an object.");
  return value as Record<string, unknown>;
};

function httpURL(value: unknown, loopback = false): URL {
  if (typeof value !== "string" || /[\0\r\n]/.test(value)) throw new Error("Invalid OAuth URL.");
  const url = new URL(value);
  if (url.username || url.password || url.hash || (url.protocol !== "https:" &&
    !(loopback && url.protocol === "http:" && ["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)))) {
    throw new Error("OAuth endpoint requires HTTPS without credentials or fragments.");
  }
  return url;
}

export function canonicalMcpResource(value: string): string {
  const url = httpURL(value, true);
  return url.pathname === "/" && !url.search ? url.origin : url.href;
}

function scopes(value: unknown): string[] {
  const values = typeof value === "string" ? value.split(" ").filter(Boolean) : value;
  if (!Array.isArray(values) || values.some((scope) => typeof scope !== "string" || !/^[\x21\x23-\x5b\x5d-\x7e]+$/.test(scope))) {
    throw new Error("OAuth scopes must be valid individual scope tokens.");
  }
  return [...new Set(values as string[])];
}

/** Tokenize quoted auth parameters so commas/words inside quotes are data. */
export function parseMcpBearerChallenges(header: string | null): McpBearerChallenge[] {
  if (!header) return [];
  const tokens: Array<{ kind: "word" | "string" | "," | "="; value: string }> = [];
  for (let index = 0; index < header.length;) {
    const character = header[index]!;
    if (/\s/.test(character)) { index++; continue; }
    if (character === "," || character === "=") { tokens.push({ kind: character, value: character }); index++; continue; }
    if (character === '"') {
      let value = "", closed = false; index++;
      while (index < header.length) {
        const next = header[index++]!;
        if (next === '"') { closed = true; break; }
        if (next === "\\") {
          if (index === header.length) throw new Error("Malformed OAuth authentication challenge.");
          value += header[index++]!;
        } else value += next;
      }
      if (!closed) throw new Error("Malformed OAuth authentication challenge.");
      tokens.push({ kind: "string", value }); continue;
    }
    const start = index;
    while (index < header.length && !/[\s,="]/.test(header[index]!)) index++;
    if (start === index) throw new Error("Malformed OAuth authentication challenge.");
    tokens.push({ kind: "word", value: header.slice(start, index) });
  }
  const result: McpBearerChallenge[] = [];
  let current: Record<string, string> | undefined;
  const flush = (): void => {
    if (!current) return;
    result.push({ ...(current.resource_metadata !== undefined ? { resourceMetadata: current.resource_metadata } : {}),
      ...(current.scope !== undefined ? { scopes: scopes(current.scope) } : {}), ...(current.error !== undefined ? { error: current.error } : {}) });
  };
  for (let index = 0; index < tokens.length;) {
    const token = tokens[index++]!;
    if (token.kind === ",") continue;
    if (token.kind !== "word") throw new Error("Malformed OAuth authentication challenge.");
    if (tokens[index]?.kind !== "=") {
      flush(); current = token.value.toLowerCase() === "bearer" ? {} : undefined; continue;
    }
    index++;
    const value = tokens[index++];
    if (!value || (value.kind !== "word" && value.kind !== "string")) throw new Error("Malformed OAuth authentication challenge.");
    if (current) {
      const key = token.value.toLowerCase();
      if (Object.hasOwn(current, key)) throw new Error("Duplicate OAuth authentication parameter.");
      current[key] = value.value;
    }
  }
  flush(); return result;
}

export function resourceMetadataCandidates(endpoint: string, challenge?: McpBearerChallenge): string[] {
  const url = httpURL(endpoint, true);
  if (challenge?.resourceMetadata) {
    const challenged = httpURL(challenge.resourceMetadata, true);
    if (challenged.protocol === "http:" && challenged.origin !== url.origin) throw new Error("Insecure OAuth metadata changed the configured server origin.");
    return [challenged.href];
  }
  return [...new Set([`${url.origin}/.well-known/oauth-protected-resource${url.pathname === "/" ? "" : url.pathname}`,
    `${url.origin}/.well-known/oauth-protected-resource`])];
}

export function authorizationMetadataCandidates(issuer: string): string[] {
  const url = httpURL(issuer);
  if (url.search) throw new Error("OAuth issuer cannot contain a query.");
  const path = url.pathname === "/" ? "" : url.pathname.replace(/\/$/, "");
  return [...new Set([`${url.origin}/.well-known/oauth-authorization-server${path}`,
    `${url.origin}/.well-known/openid-configuration${path}`,
    ...(path ? [`${url.origin}${path}/.well-known/openid-configuration`] : [])])];
}

export function parseMcpProtectedResource(value: unknown, endpoint: string): McpProtectedResource {
  const metadata = object(value);
  if (typeof metadata.resource !== "string" || canonicalMcpResource(metadata.resource) !== canonicalMcpResource(endpoint)) {
    throw new Error("OAuth resource metadata does not match the selected MCP server.");
  }
  if (!Array.isArray(metadata.authorization_servers) || !metadata.authorization_servers.length ||
    metadata.authorization_servers.some((issuer) => typeof issuer !== "string")) throw new Error("OAuth resource metadata has no authorization server.");
  const authorizationServers = [...new Set(metadata.authorization_servers as string[])];
  for (const issuer of authorizationServers) authorizationMetadataCandidates(issuer);
  return { resource: canonicalMcpResource(metadata.resource), authorizationServers,
    ...(metadata.scopes_supported !== undefined ? { scopes: scopes(metadata.scopes_supported) } : {}) };
}

export function parseMcpAuthorizationServer(value: unknown, issuer: string): McpAuthorizationServer {
  const metadata = object(value);
  authorizationMetadataCandidates(issuer);
  if (metadata.issuer !== issuer) throw new Error("OAuth authorization metadata changed its issuer.");
  if (!Array.isArray(metadata.code_challenge_methods_supported) || !metadata.code_challenge_methods_supported.includes("S256")) {
    throw new Error("The authorization server does not advertise PKCE S256 support.");
  }
  if (metadata.response_types_supported !== undefined && (!Array.isArray(metadata.response_types_supported) ||
    !metadata.response_types_supported.includes("code"))) throw new Error("The authorization server does not support code responses.");
  const methods = metadata.token_endpoint_auth_methods_supported ?? ["client_secret_basic"];
  if (!Array.isArray(methods) || methods.some((method) => typeof method !== "string")) throw new Error("Invalid OAuth token authentication methods.");
  return { issuer, authorizationEndpoint: httpURL(metadata.authorization_endpoint).href,
    tokenEndpoint: httpURL(metadata.token_endpoint).href,
    ...(metadata.registration_endpoint ? { registrationEndpoint: httpURL(metadata.registration_endpoint).href } : {}),
    ...(metadata.revocation_endpoint ? { revocationEndpoint: httpURL(metadata.revocation_endpoint).href } : {}),
    clientMetadataSupported: metadata.client_id_metadata_document_supported === true,
    tokenAuthenticationMethods: methods as string[] };
}

export function createMcpPKCE(): { verifier: string; challenge: string; state: string } {
  const verifier = randomBytes(32).toString("base64url");
  return { verifier, challenge: createHash("sha256").update(verifier).digest("base64url"), state: randomBytes(32).toString("base64url") };
}

export function mcpAuthorizationURL(server: McpAuthorizationServer, input: {
  clientId: string; redirectUri: string; resource: string; challenge: string; state: string; scopes?: string[];
}): string {
  const redirect = httpURL(input.redirectUri, true);
  if (!input.clientId || !input.state || !/^[A-Za-z0-9_-]{43}$/.test(input.challenge)) throw new Error("Invalid OAuth authorization state.");
  const url = new URL(server.authorizationEndpoint);
  for (const [key, value] of Object.entries({ response_type: "code", client_id: input.clientId, redirect_uri: redirect.href,
    resource: canonicalMcpResource(input.resource), code_challenge: input.challenge, code_challenge_method: "S256", state: input.state })) url.searchParams.set(key, value);
  if (input.scopes?.length) url.searchParams.set("scope", scopes(input.scopes).join(" "));
  return url.href;
}

export function mcpCodeExchangeBody(input: { clientId: string; redirectUri: string; resource: string; code: string; verifier: string }): URLSearchParams {
  if (!input.clientId || !input.code || !/^[A-Za-z0-9._~-]{43,128}$/.test(input.verifier)) throw new Error("Invalid OAuth code exchange.");
  return new URLSearchParams({ grant_type: "authorization_code", client_id: input.clientId,
    redirect_uri: httpURL(input.redirectUri, true).href, resource: canonicalMcpResource(input.resource),
    code: input.code, code_verifier: input.verifier });
}
