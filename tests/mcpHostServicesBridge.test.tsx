import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it, vi } from "vitest";
import { requireMcpRootCommand, selectMcpRootsFromIntent } from "../src/main/mcpRootsIntent";
import { McpAppNotificationGate } from "../src/renderer/src/mcpAppServerNotifications";
import { McpHostServicesPanel } from "../src/renderer/src/McpHostServicesPanel";
import type { McpHostConnection, McpRootsStatus, McpAppServerNotification } from "../src/shared/mcpHostServices";
import type { McpAppLaunch } from "../src/shared/mcpApps";
const command = { brainId: "a", serverId: "s", requestId: "request" };
const connection: McpHostConnection = { brainId: "a", serverId: "s", serverLabel: "fixture", endpoint: "stdio:fixture", connectionEpoch: 1,
  connectionId: "session", configurationSha256: "a".repeat(64), ready: true, enabled: true };
it("takes picker paths only from main and does not rebase a late selection onto a replacement connection", async () => {
  const select = vi.fn(async () => ({} as McpRootsStatus)), status = vi.fn(async () => ({} as McpRootsStatus));
  const current = vi.fn().mockResolvedValueOnce(connection).mockResolvedValueOnce({ ...connection, connectionId: "new-session" });
  await expect(selectMcpRootsFromIntent(command, { current, select, status, pick: async () => ["/explicit/fixture"] })).rejects.toThrow(/changed/);
  expect(select).not.toHaveBeenCalled();
  expect(() => requireMcpRootCommand({ ...command, paths: ["/renderer/path"] })).toThrow(/trusted picker/);
  await selectMcpRootsFromIntent(command, { current: async () => connection, select, status, pick: async () => null });
  expect(status).toHaveBeenCalledOnce(); expect(select).not.toHaveBeenCalled();
});
it("forwards only ready exact-view invalidations and strips all private transport identity", () => {
  const launch = { view: { brainId: "a", serverId: "s", id: "view" }, viewToken: "private-view-token" } as McpAppLaunch;
  const event: McpAppServerNotification = { brainId: "a", serverId: "s", viewId: "view", viewToken: launch.viewToken, connectionEpoch: 1,
    sequence: 1, method: "notifications/tools/list_changed", params: {} };
  const gate = new McpAppNotificationGate(launch);
  expect(gate.receive(event, false)).toBeUndefined(); expect(gate.receive({ ...event, viewToken: "wrong" }, true)).toBeUndefined();
  expect(gate.receive(event, true)).toEqual({ jsonrpc: "2.0", method: event.method, params: {} });
  expect(gate.receive(event, true)).toBeUndefined(); expect(gate.receive({ ...event, sequence: 2, connectionEpoch: 2 }, true)).toBeUndefined();
  gate.close(); expect(gate.receive({ ...event, sequence: 3 }, true)).toBeUndefined();
});
it("adds collapsed native-app controls without any picker, log read, provider or model call on render", () => {
  const api = new Proxy({}, { get() { throw new Error("No API operation during rendering"); } });
  const html = renderToStaticMarkup(<McpHostServicesPanel {...command} connected enabled roots={api as never} diagnostics={api as never} />);
  expect(html).toContain("Filesystem disclosures and diagnostics"); expect(html).toContain("Choose folders");
  expect(html).toContain("does not grant file access"); expect(html).not.toContain("<details open");
});
