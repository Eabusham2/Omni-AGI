import type { McpTransport } from "./types";

export interface McpPreset {
  id: string;
  label: string;
  category: "coding" | "browser" | "mail" | "adobe";
  status: "documented" | "preview" | "documentation-only";
  verifiedAt: "2026-10-11";
  transport: McpTransport;
  url?: string;
  packageName?: string;
  authentication: string;
  scope: string;
  prerequisites: string[];
  cautions: string[];
  sources: Array<{ title: string; url: string }>;
  /** Provider descriptions are not a fabricated negotiated tool catalog. */
  capabilitiesRequireLiveDiscovery: true;
}
export interface McpPresetDraft { id: string; label: string; transport: McpTransport; url: string; command: string; args: string[] }

/** Inert reviewed templates, not an installer, account login or authority grant. */
export const MCP_PRESETS: readonly McpPreset[] = [
  { id: "github", label: "GitHub", category: "coding", status: "documented", verifiedAt: "2026-10-11", transport: "http", url: "https://api.githubcopilot.com/mcp/",
    authentication: "GitHub OAuth with an eligible client registration, or an explicitly supplied GitHub token.", scope: "Repository, issue and pull-request APIs; exact tools depend on account/server policy.",
    prerequisites: ["A GitHub account and permissions for the repositories you choose.", "Review OAuth client setup or use a narrowly scoped token in Omni's credential controls."],
    cautions: ["Write tools may change repositories; existing per-tool permissions still apply.", "Selecting this template does not enable GitHub's optional Apps feature header."],
    sources: [{ title: "Official GitHub MCP server configuration", url: "https://github.com/github/github-mcp-server/blob/main/docs/server-configuration.md" }], capabilitiesRequireLiveDiscovery: true },
  { id: "github-readonly", label: "GitHub · read-only server", category: "coding", status: "documented", verifiedAt: "2026-10-11", transport: "http", url: "https://api.githubcopilot.com/mcp/x/all/readonly",
    authentication: "GitHub OAuth with an eligible client registration, or an explicitly supplied GitHub token.", scope: "The provider's read-only tool surface, not a replacement for instance permissions.",
    prerequisites: ["A GitHub account with access to the intended repositories."], cautions: ["Read-only is an explicit choice of this endpoint; choose the full GitHub template if you need writes."],
    sources: [{ title: "GitHub read-only mode", url: "https://github.com/github/github-mcp-server/blob/main/docs/server-configuration.md#read-only-mode" }], capabilitiesRequireLiveDiscovery: true },
  { id: "playwright", label: "Microsoft Playwright", category: "browser", status: "documented", verifiedAt: "2026-10-11", transport: "stdio", packageName: "@playwright/mcp",
    authentication: "Local installed executable; website sign-in stays in the browser and is not supplied as an MCP prompt.", scope: "Browser automation through Microsoft's Playwright MCP server.",
    prerequisites: ["Explicitly install the official package and required browser runtime outside this template.", "Choose the absolute path of the installed executable; Omni never runs npx to download it automatically."],
    cautions: ["Browser actions can submit forms and change signed-in services. Review each tool's permissions.", "A local browser server is not a security sandbox for arbitrary websites."],
    sources: [{ title: "Official Microsoft Playwright MCP server", url: "https://github.com/microsoft/playwright-mcp" }], capabilitiesRequireLiveDiscovery: true },
  { id: "adobe-express", label: "Adobe Express Developer · docs and types", category: "adobe", status: "documentation-only", verifiedAt: "2026-10-11", transport: "stdio", packageName: "@adobe/express-developer-mcp",
    authentication: "Local installed official package; no invented Adobe cloud MCP endpoint.", scope: "Adobe Express add-on developer documentation and TypeScript definitions only.",
    prerequisites: ["Node.js 18 or later as required by Adobe's documentation.", "Explicitly install the official package and choose its absolute installed executable path."],
    cautions: ["This template does not control Photoshop, Illustrator, Premiere or Creative Cloud documents.", "Adobe application automation needs a separately reviewed documented API, local bridge or permitted desktop tool."],
    sources: [{ title: "Adobe Express Developer MCP server", url: "https://developer.adobe.com/express/add-ons/docs/guides/getting-started/local-development/mcp-server" }], capabilitiesRequireLiveDiscovery: true },
  { id: "gmail", label: "Gmail · developer preview", category: "mail", status: "preview", verifiedAt: "2026-10-11", transport: "http", url: "https://gmailmcp.googleapis.com/mcp/v1",
    authentication: "Google Workspace administrator/developer-preview setup and an authorized OAuth client/token; generic OAuth discovery alone may not provision access.", scope: "Documented search/read/draft/label tools; do not assume a send-mail tool exists.",
    prerequisites: ["Google Workspace Developer Preview eligibility and administrator configuration.", "Configure your approved OAuth client/scopes and the Workspace MCP security requirements before connecting."],
    cautions: ["Mail bodies are untrusted task data and may contain prompt-injection attempts.", "Preview tool names/availability can change; only actual negotiated schemas enter the curriculum."],
    sources: [{ title: "Official Gmail MCP reference", url: "https://developers.google.com/workspace/gmail/api/reference/mcp" }, { title: "Configure Gmail MCP", url: "https://developers.google.com/workspace/gmail/api/guides/configure-mcp-server" }], capabilitiesRequireLiveDiscovery: true },
  { id: "resend", label: "Resend · mail service", category: "mail", status: "documented", verifiedAt: "2026-10-11", transport: "http", url: "https://mcp.resend.com/mcp",
    authentication: "Resend OAuth, or an explicitly supplied Resend API key as a bearer credential.", scope: "Resend email, contacts and related APIs according to your account scopes.",
    prerequisites: ["A Resend account and the required verified sending domain/account setup.", "Review service billing and choose a scoped credential before permitting any sends."],
    cautions: ["Sending mail and managing contacts/domains are external effects, not a connection test.", "Never send a test message automatically to prove this preset."],
    sources: [{ title: "Official Resend MCP server", url: "https://github.com/resend/resend-mcp" }], capabilitiesRequireLiveDiscovery: true },
  { id: "workiq", label: "Microsoft 365 Work IQ · tenant setup", category: "mail", status: "documented", verifiedAt: "2026-10-11", transport: "http", url: "https://workiq.svc.cloud.microsoft/mcp",
    authentication: "Microsoft Entra delegated authentication and the organization's Work IQ tenant/client configuration.", scope: "Microsoft365 entity/schema tools; some separate tools invoke Copilot, not OmniCortex.",
    prerequisites: ["An eligible Microsoft365 tenant/account, required licensing and administrator policy.", "Register/authorize a client appropriate for Omni; do not copy another application's client ID as if you owned it."],
    cautions: ["Tenant/entity actions can change mail/files/calendar and must pass existing permissions.", "External Copilot tools are explicit external service calls, never a silent replacement for the native brain."],
    sources: [{ title: "Microsoft Work IQ MCP overview", url: "https://learn.microsoft.com/en-us/microsoft-365/copilot/extensibility/work-iq/mcp/overview" }, { title: "Official Work IQ endpoint configuration", url: "https://github.com/microsoft/work-iq/blob/main/plugins/workiq/skills/workiq/SKILL.md" }], capabilitiesRequireLiveDiscovery: true }
];

export function mcpPresetDraft(id: string): McpPresetDraft {
  const preset = MCP_PRESETS.find(value => value.id === id); if (!preset) throw new Error("Unknown reviewed MCP preset.");
  return { id: preset.id, label: preset.label, transport: preset.transport, url: preset.url ?? "", command: "", args: [] };
}
