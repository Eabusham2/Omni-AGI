import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { McpAuthenticationPanel } from "../src/renderer/src/McpAuthenticationPanel";
import { McpCredentialPanel } from "../src/renderer/src/McpCredentialPanel";
import type { McpServerSummary } from "../src/shared/types";

const server: McpServerSummary = { brainId: "first", id: "coding", registryId: "coding", label: "Coding", transport: "http", endpoint: "https://mcp.example/mcp",
  connected: false, tools: [], credentialConfigured: false, scope: "all", globallyEnabled: true, defaultEnabled: true, accessOverride: "inherit", effectiveEnabled: true };
describe("MCP account controls", () => {
  it("keeps optional client details collapsed, labels inputs and does not pretend a preview can sign in", () => {
    const html = renderToStaticMarkup(<McpAuthenticationPanel brainId="first" server={server} />);
    expect(html).toContain("<details>"); expect(html).not.toContain("<details open");
    expect(html).toContain("Client secret"); expect(html).toContain('type="password"');
    expect(html).toContain("disabled=\"\">Sign in");
    expect(html).not.toContain("Disconnect account");
  });
  it("has one Stop for pending login and independent disconnection after account login", () => {
    const pending = renderToStaticMarkup(<McpAuthenticationPanel brainId="first" server={{ ...server, authentication: { state: "authorizing", scopes: ["read"] } }} />);
    expect(pending).toContain("Waiting for browser sign-in"); expect(pending).toContain("Stop sign-in"); expect(pending).toContain("Requested access: read");
    const authorized = renderToStaticMarkup(<McpAuthenticationPanel brainId="first" server={{ ...server, authentication: { state: "authorized", persistence: "session" } }} />);
    expect(authorized).toContain("this session only"); expect(authorized).toContain("Disconnect account");
    expect(renderToStaticMarkup(<McpAuthenticationPanel brainId="first" server={{ ...server, transport: "stdio" }} />)).toBe("");
  });
  it("keeps replacement closed, renders the correct transport editor, and never prepopulates secret values", () => {
    const http = renderToStaticMarkup(<McpCredentialPanel brainId="first" server={server} onSaved={() => undefined} />);
    expect(http).not.toContain("<details open"); expect(http).toContain("Replacement bearer token");
    expect(http).toContain('type="password"'); expect(http).toContain('value=""'); expect(http).not.toContain("Clear now");
    const local = renderToStaticMarkup(<McpCredentialPanel brainId="first" server={{ ...server, transport: "stdio", environmentKeys: ["SERVICE_TOKEN"] }} onSaved={() => undefined} />);
    expect(local).toContain("Replacement environment"); expect(local).toContain("SERVICE_TOKEN"); expect(local).toContain("Replaces all variables");
    expect(local).not.toContain("Replacement bearer token");
  });
});
