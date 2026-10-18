import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { authorizationMetadataCandidates, canonicalMcpResource, createMcpPKCE, mcpAuthorizationURL,
  mcpCodeExchangeBody, parseMcpAuthorizationServer, parseMcpBearerChallenges, parseMcpProtectedResource,
  resourceMetadataCandidates } from "../src/main/mcpOAuthProtocol";

const issuer = "https://login.example/tenant";
const metadata = { issuer, authorization_endpoint: "https://login.example/authorize", token_endpoint: "https://login.example/token",
  code_challenge_methods_supported: ["S256"], response_types_supported: ["code"], token_endpoint_auth_methods_supported: ["none"] };

describe("MCP authorization protocol", () => {
  it("parses actual Bearer schemes, escaped quotes and scope challenges without treating quoted words as schemes", () => {
    expect(parseMcpBearerChallenges('Basic realm="Bearer, test", Bearer realm="name\\\"x", resource_metadata="https://mcp.example/.well-known/resource", scope="read new:scope", error="insufficient_scope"'))
      .toEqual([{ resourceMetadata: "https://mcp.example/.well-known/resource", scopes: ["read", "new:scope"], error: "insufficient_scope" }]);
    expect(() => parseMcpBearerChallenges('Bearer scope="read", scope="write"')).toThrow(/Duplicate/);
    expect(() => parseMcpBearerChallenges('Bearer scope="unterminated')).toThrow(/Malformed/);
  });
  it("uses path-specific resource discovery and all standard issuer discovery variants", () => {
    expect(resourceMetadataCandidates("https://mcp.example/path/mcp")).toEqual([
      "https://mcp.example/.well-known/oauth-protected-resource/path/mcp", "https://mcp.example/.well-known/oauth-protected-resource"]);
    expect(authorizationMetadataCandidates(issuer)).toEqual([
      "https://login.example/.well-known/oauth-authorization-server/tenant", "https://login.example/.well-known/openid-configuration/tenant",
      "https://login.example/tenant/.well-known/openid-configuration"]);
    expect(resourceMetadataCandidates("https://mcp.example/mcp", { resourceMetadata: "https://metadata.example/resource" })).toEqual(["https://metadata.example/resource"]);
  });
  it("binds metadata to its resource and exact issuer and requires advertised S256", () => {
    expect(canonicalMcpResource("HTTPS://MCP.EXAMPLE/")).toBe("https://mcp.example");
    expect(parseMcpProtectedResource({ resource: "https://mcp.example/mcp", authorization_servers: [issuer], scopes_supported: ["read"] }, "https://mcp.example/mcp"))
      .toMatchObject({ resource: "https://mcp.example/mcp", scopes: ["read"] });
    expect(() => parseMcpProtectedResource({ resource: "https://other.example/mcp", authorization_servers: [issuer] }, "https://mcp.example/mcp")).toThrow(/match/);
    expect(() => parseMcpAuthorizationServer({ ...metadata, issuer: "https://other.example" }, issuer)).toThrow(/issuer/);
    expect(() => parseMcpAuthorizationServer({ ...metadata, code_challenge_methods_supported: ["plain"] }, issuer)).toThrow(/S256/);
  });
  it("keeps the verifier out of browser URLs and binds both authorization/code exchange to the same resource", () => {
    const pkce = createMcpPKCE();
    expect(pkce.challenge).toBe(createHash("sha256").update(pkce.verifier).digest("base64url"));
    const url = new URL(mcpAuthorizationURL(parseMcpAuthorizationServer(metadata, issuer), { clientId: "client", redirectUri: "http://127.0.0.1:49123/callback",
      resource: "https://mcp.example/mcp", challenge: pkce.challenge, state: pkce.state, scopes: ["read", "write"] }));
    expect(url.searchParams.get("code_challenge_method")).toBe("S256");
    expect(url.searchParams.has("code_verifier")).toBe(false);
    const body = mcpCodeExchangeBody({ clientId: "client", redirectUri: "http://127.0.0.1:49123/callback", resource: "https://mcp.example/mcp", code: "one-use", verifier: pkce.verifier });
    expect(body.get("resource")).toBe(url.searchParams.get("resource"));
    expect(body.get("code_verifier")).toBe(pkce.verifier);
  });
  it("rejects unsafe issuer/token URLs and insecure cross-origin metadata", () => {
    expect(() => parseMcpAuthorizationServer({ ...metadata, token_endpoint: "https://name:password@login.example/token" }, issuer)).toThrow(/credentials/);
    expect(() => parseMcpAuthorizationServer({ ...metadata, token_endpoint: "http://login.example/token" }, issuer)).toThrow(/HTTPS/);
    expect(() => resourceMetadataCandidates("https://mcp.example/mcp", { resourceMetadata: "http://127.0.0.1:80/metadata" })).toThrow(/origin/);
  });
});
