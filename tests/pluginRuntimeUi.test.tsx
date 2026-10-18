import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { OmniPluginPackage, PluginCatalogEntry, PluginInstanceAccess } from "../src/shared/plugins";
import type { PluginRuntimeApi, PluginRuntimeModuleDefinition, PluginRuntimeMountedContribution, PluginRuntimePendingApproval } from "../src/shared/pluginRuntime";
import {
  PluginRuntimeConflictReview,
  PluginRuntimeModulePresentation,
  pluginRuntimeActivationReview,
  pluginRuntimeHistoryCount
} from "../src/renderer/src/PluginRuntimeControls";
import {
  PluginRuntimeContributionItems,
  PluginRuntimePanelFrame,
  PluginRuntimeSlot,
  checkedPluginRuntimeContributions,
  ownedPluginRuntimeContributions,
  pluginRuntimePanelDocument,
  pluginRuntimeSlotError,
  pluginRuntimeStudioCss
} from "../src/renderer/src/PluginRuntimeContributions";

const HASH = "a".repeat(64);
const module: PluginRuntimeModuleDefinition = { id: "helper", title: "Helper", description: "Isolated feature helper", script: "", events: ["chat.completed"],
  ui: { kinds: ["button", "panel"], slots: ["chat.toolbar", "tools"] }, dynamicTools: true };
const packageValue: OmniPluginPackage = { format: "omni-plugin-package", formatVersion: 1, manifest: {
  id: "fixture.runtime", version: "1.0.0", name: "Runtime fixture", description: "Fixture", author: "Fixture", license: "MIT",
  documentation: "Fixture", capabilities: [], tools: [], panels: [], interactivePanels: [], runtimeModules: [module], themes: [], trainingFormats: []
} };
const entry: PluginCatalogEntry = { pluginId: "fixture.runtime", name: "Runtime fixture", description: "Fixture", enabled: true,
  activeVersion: "1.0.0", activeContentSha256: HASH, revision: 7, updatedAt: "2026-10-13T00:00:00.000Z",
  versions: [{ version: "1.0.0", contentSha256: HASH, installedAt: "2026-10-13T00:00:00.000Z", sourceLabel: "fixture" }] };
const access: PluginInstanceAccess = { brainId: "brain-a", pluginId: entry.pluginId, contentSha256: HASH, enabled: true,
  capabilities: [], toolPermissions: [], revision: 9, updatedAt: "2026-10-13T00:00:00.000Z" };

function contribution(values: Partial<PluginRuntimeMountedContribution> = {}): PluginRuntimeMountedContribution {
  return { id: "action", kind: "button", slot: "chat.toolbar", title: "Run helper", event: "run", brainId: "brain-a",
    pluginId: entry.pluginId, contentSha256: HASH, moduleId: module.id, sessionId: "session-a", revision: 1, ...values };
}

describe("plugin feature runtime UI boundaries", () => {
  it("allows activation only for the fresh exact brain/package/catalog/instance review", () => {
    expect(pluginRuntimeActivationReview({ brainId: "brain-a", expectedContentSha256: HASH, entry, access, packageValue, moduleId: module.id }))
      .toMatchObject({ allowed: true, catalogRevision: 7, instanceRevision: 9 });
    expect(pluginRuntimeActivationReview({ brainId: "brain-b", expectedContentSha256: HASH, entry, access, packageValue, moduleId: module.id }))
      .toMatchObject({ allowed: false });
    expect(pluginRuntimeActivationReview({ brainId: "brain-a", expectedContentSha256: HASH, entry: { ...entry, enabled: false }, access, packageValue, moduleId: module.id }).reason)
      .toMatch(/enable/i);
    expect(pluginRuntimeActivationReview({ brainId: "brain-a", expectedContentSha256: HASH, entry, access: { ...access, contentSha256: "b".repeat(64) }, packageValue, moduleId: module.id }))
      .toMatchObject({ allowed: false });
  });

  it("renders declared scope and exact host Ask operands without any approval token", () => {
    const approval: PluginRuntimePendingApproval = { brainId: "brain-a", sessionId: "session-a", rpcId: "rpc-a", reviewId: "review-a",
      pluginId: entry.pluginId, contentSha256: HASH, moduleId: module.id,
      invocation: { brainId: "brain-a", toolId: "system.files", action: "read", arguments: { path: "/review/me.txt" } }, executionId: "execution-a" };
    const html = renderToStaticMarkup(<PluginRuntimeModulePresentation module={module} approvals={[approval]} busy=""
      activation={{ allowed: true, reason: "Ready" }} onActivate={vi.fn()} onStop={vi.fn()} onReview={vi.fn()} />);
    expect(html).toContain("chat.completed");
    expect(html).toContain("chat.toolbar");
    expect(html).toContain("Dynamic tools");
    expect(html).toContain("/review/me.txt");
    expect(html).toContain("Approve exact action");
    expect(html).not.toContain("approvalToken");
  });

  it("keeps panels in an opaque scriptless deny-all srcDoc and filters foreign or unknown slots", () => {
    const panel = contribution({ id: "panel", kind: "panel", slot: "tools", title: "Safe panel", html: "<section><h2>Result</h2></section>",
      css: "body{color:purple}</style><script>bad()</script>" });
    const document = pluginRuntimePanelDocument(panel);
    expect(document).toContain("default-src 'none'");
    expect(document).toContain("script-src 'none'");
    expect(document).not.toContain("</style><script>");
    expect(pluginRuntimeStudioCss("body{color:red}</style><script>bad()</script>")).not.toContain("</style><script>");
    const html = renderToStaticMarkup(<PluginRuntimePanelFrame contribution={panel} />);
    expect(html).toContain("sandbox=\"\"");
    expect(html).toContain("referrerPolicy=\"no-referrer\"");
    expect(html).not.toContain("allow-scripts");
    expect(checkedPluginRuntimeContributions("brain-a", [contribution(), contribution({ brainId: "brain-b" }), contribution({ slot: "unknown.slot" })]))
      .toHaveLength(1);
  });

  it("requires a deliberate inherited source inspection and never preselects the first parent", () => {
    const api = { readSourceState: vi.fn(), selectSourceState: vi.fn() } as unknown as PluginRuntimeApi;
    const html = renderToStaticMarkup(<PluginRuntimeConflictReview brainId="brain-a"
      conflict={{ pluginHash: HASH, moduleId: "helper", key: "settings" }}
      sources={[{ sha256: "b".repeat(64), owner: "parent-a", bytes: 120 }]}
      api={api} onResolved={async () => undefined} />);
    expect(html).toContain("Choose a source to inspect");
    expect(html).not.toContain('option value="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" selected');
    expect(html).toContain("Use reviewed source");
    expect(html).toContain("disabled");
    expect(api.readSourceState).not.toHaveBeenCalled();
  });

  it("renders empty slots as nothing and reserves provider failures for one designated surface", () => {
    expect(renderToStaticMarkup(<PluginRuntimeSlot slot="chat.toolbar" />)).toBe("");
    expect(pluginRuntimeSlotError("provider failed", "", false)).toBe("");
    expect(pluginRuntimeSlotError("provider failed", "", true)).toBe("provider failed");
    expect(pluginRuntimeSlotError("provider failed", "action failed", false)).toBe("action failed");
  });

  it("synchronously hides a previous brain owner's contributions before the next fetch commits", () => {
    const previous = contribution({ brainId: "brain-a", kind: "style", css: "body{color:red}" });
    expect(ownedPluginRuntimeContributions("brain-b", "brain-a", [previous])).toEqual([]);
    expect(ownedPluginRuntimeContributions("brain-a", "brain-a", [previous])).toEqual([previous]);
  });

  it("keeps raw request identities collapsed and uses ordinary grouped action buttons", () => {
    const session = { format: "omni-plugin-runtime-session" as const, version: 1 as const, brainId: "brain-a", pluginId: entry.pluginId,
      contentSha256: HASH, moduleId: module.id, sessionId: "session-visible-only-in-details", catalogRevision: 7,
      instanceRevision: 9, state: "active" as const, startedAt: "2026-10-15T00:00:00.000Z" };
    const approval: PluginRuntimePendingApproval = { brainId: "brain-a", sessionId: session.sessionId, rpcId: "rpc-in-details", reviewId: "review-a",
      pluginId: entry.pluginId, contentSha256: HASH, moduleId: module.id,
      invocation: { brainId: "brain-a", toolId: "system.files", action: "read", arguments: { path: "/review/me.txt" } }, executionId: "execution-in-details" };
    const moduleHtml = renderToStaticMarkup(<PluginRuntimeModulePresentation module={module} session={session} approvals={[approval]} busy=""
      activation={{ allowed: true, reason: "Ready" }} onActivate={vi.fn()} onStop={vi.fn()} onReview={vi.fn()} />);
    expect(moduleHtml).toContain("Module active.");
    expect(moduleHtml).toContain("<details><summary>Session details</summary>");
    expect(moduleHtml).toContain("<details><summary>Request details</summary>");
    expect(moduleHtml.indexOf("session-visible-only-in-details")).toBeGreaterThan(moduleHtml.indexOf("Session details"));
    expect(moduleHtml.indexOf("execution-in-details")).toBeGreaterThan(moduleHtml.indexOf("Request details"));

    const menuHtml = renderToStaticMarkup(<PluginRuntimeContributionItems slot="tools"
      values={[contribution({ kind: "menu", slot: "tools" })]} pending="" visibleError="" onInvoke={vi.fn()} />);
    expect(menuHtml).toContain('role="group"');
    expect(menuHtml).toContain('<button type="button"');
    expect(menuHtml).not.toContain('role="menu"');
    expect(menuHtml).not.toContain('role="menuitem"');
    expect(pluginRuntimeHistoryCount(1)).toBe("1 event");
    expect(pluginRuntimeHistoryCount(2)).toBe("2 events");
  });
});
