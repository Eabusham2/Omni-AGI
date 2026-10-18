import { expect, it, vi } from "vitest";
import { McpAppService, type McpAppServiceOptions } from "../src/main/mcpApps";
import { McpDiagnosticService } from "../src/main/mcpDiagnostics";
import { MCP_APP_MIME, MCP_APP_PROTOCOL } from "../src/shared/mcpApps";
import type { McpAppRequest } from "../src/shared/mcpApps";
import type { McpRequestOrigin } from "../src/shared/mcpInteractions";
import type { McpToolSummary, ToolExecutionResult, ToolInvocation } from "../src/shared/types";

async function fixture(install = true) {
  const origin: McpRequestOrigin = { brainId: "first", serverId: "fixture", serverLabel: "Fixture", endpoint: "https://example.invalid/mcp", connectionEpoch: 2 };
  let allowed = true;
  const tool: McpToolSummary = { id: "mcp.fixture.open", remoteName: "open", actions: ["call"], inputSchema: { type: "object" }, metadata: { ui: { resourceUri: "ui://fixture/view" } } };
  const requireAccess = (value: McpRequestOrigin) => { if (!allowed || value.brainId !== origin.brainId || value.serverId !== origin.serverId || value.endpoint !== origin.endpoint || value.connectionEpoch !== origin.connectionEpoch) throw new Error("current source access revoked"); };
  const diagnostic = new McpDiagnosticService({ requireAccess, knownSecrets: async () => ["fixture-operational-key"], admit: async () => undefined });
  const execute = vi.fn(async (invocation: ToolInvocation, requestId: string): Promise<ToolExecutionResult> => ({ toolId: invocation.toolId, action: invocation.action,
    id: requestId, state: "complete", startedAt: "2026-10-15T00:00:00Z", finishedAt: "2026-10-15T00:00:00Z", output: { format: "omni-mcp-client-result", operation: "resources/read", untrustedTaskData: true,
      provenance: { requestedUri: "ui://fixture/view" }, result: { contents: [{ uri: "ui://fixture/view", mimeType: MCP_APP_MIME, text: "<!doctype html><html><body>Owned inert fixture</body></html>" }] } } }));
  const optional: Partial<McpAppServiceOptions> = install ? { serverCapabilities: () => ({ tools: { listChanged: true }, resources: { listChanged: true } }),
    log: async (source, params, signal, context) => { await diagnostic.receive(source, params, signal, "app", context.viewId, context.privateSecrets); } } : {};
  const apps = new McpAppService({ requireAccess, tool: () => tool, resourceTool: () => ({ ...tool, id: "mcp.fixture.resource", remoteName: "resources/read" }), execute,
    approveTool: async () => true, approveNetwork: async () => true, approveLink: async () => true, openBrowser: async () => undefined, admit: async () => undefined, theme: () => "light",
    ...optional });
  const descriptor = (await apps.capture({ origin, tool, arguments: {}, result: { content: [] } }))!, launch = await apps.open("first", descriptor.id);
  const request = (value: Partial<McpAppRequest> & Pick<McpAppRequest, "id" | "method">) => apps.request({ brainId: "first", viewId: launch.view.id, viewToken: launch.viewToken, ...value });
  const initialize = () => request({ id: "initialize", method: "ui/initialize", params: { protocolVersion: MCP_APP_PROTOCOL, appInfo: { name: "new-host-surface-fixture" }, appCapabilities: {} } });
  const ready = () => request({ id: "ready", method: "ui/notifications/initialized", notification: true });
  return { origin, apps, launch, execute, diagnostic, request, initialize, ready, revoke: () => { allowed = false; } };
}

it("advertises only installed notifications and sends exact current ready-view changes without refetch or invocation", async () => {
  const f = await fixture(), receive = vi.fn(); f.apps.subscribeServerNotifications(receive);
  f.apps.serverNotification(f.origin, "notifications/tools/list_changed"); expect(receive).not.toHaveBeenCalled();
  expect(await f.initialize()).toMatchObject({ result: { hostCapabilities: { serverTools: { listChanged: true }, serverResources: { listChanged: true }, logging: {} } } });
  f.apps.serverNotification(f.origin, "notifications/resources/list_changed"); expect(receive).not.toHaveBeenCalled(); await f.ready();
  for (const foreign of [{ ...f.origin, brainId: "other" }, { ...f.origin, connectionEpoch: 3 }, { ...f.origin, endpoint: "https://replaced.invalid/mcp" }]) f.apps.serverNotification(foreign, "notifications/tools/list_changed");
  expect(receive).not.toHaveBeenCalled();
  f.apps.serverNotification(f.origin, "notifications/tools/list_changed"); f.apps.serverNotification(f.origin, "notifications/resources/list_changed");
  expect(receive.mock.calls.map(value => value[0].method)).toEqual(["notifications/tools/list_changed", "notifications/resources/list_changed"]);
  expect(receive.mock.calls[0]?.[0]).toMatchObject({ brainId: "first", serverId: "fixture", viewId: f.launch.view.id, viewToken: f.launch.viewToken, connectionEpoch: 2, sequence: 1, params: {} });
  expect(f.execute).toHaveBeenCalledOnce(); // only explicit original UI-resource read
  f.apps.close("first", f.launch.view.id, f.launch.viewToken); f.apps.serverNotification(f.origin, "notifications/tools/list_changed"); expect(receive).toHaveBeenCalledTimes(2);
  f.apps.dispose(); f.diagnostic.close();
});

it("does not claim relay/logging support without consumers and stops forwarding immediately when access is revoked", async () => {
  const none = await fixture(false); const initialized = await none.initialize(); expect(initialized).toMatchObject({ result: { hostCapabilities: { serverTools: {}, serverResources: {} } } });
  expect((initialized.result as { hostCapabilities: Record<string, unknown> }).hostCapabilities.logging).toBeUndefined(); await none.ready();
  expect(await none.request({ id: "unsupported-log", notification: true, method: "notifications/message", params: { level: "info", data: "not secretly recorded" } })).toMatchObject({ error: { code: -32601 } });
  none.apps.dispose(); none.diagnostic.close();
  const f = await fixture(), observe = vi.fn(); f.apps.subscribeServerNotifications(observe); await f.initialize(); await f.ready(); f.revoke();
  f.apps.serverNotification(f.origin, "notifications/tools/list_changed"); expect(observe).not.toHaveBeenCalled(); f.apps.dispose(); f.diagnostic.close();
});

it("accepts App logging only as validated notifications and redacts current private view/operational tokens", async () => {
  const f = await fixture(); await f.initialize(); await f.ready();
  expect(await f.request({ id: "log-as-rpc", method: "notifications/message", params: { level: "info", data: "not a host action" } })).toMatchObject({ error: { code: -32600 } });
  const message = { id: "actual-notification", method: "notifications/message", notification: true as const, params: { level: "notice", data: `fixture-operational-key ${f.launch.viewToken}` } };
  expect(await f.request(message)).toEqual({ result: {} }); expect(await f.request(message)).toEqual({ result: {} });
  const records = f.diagnostic.page("first", "fixture").records; expect(records).toHaveLength(1); expect(records[0]).toMatchObject({ source: "app", viewId: f.launch.view.id, data: "[redacted] [redacted]", nativeLearning: false });
  expect(await f.request({ ...message, id: "bad-level", params: { level: "fatal", data: "bad" } })).toMatchObject({ error: { code: -32603 } });
  expect(f.execute).toHaveBeenCalledOnce(); f.apps.dispose(); f.diagnostic.close();
});
