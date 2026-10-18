/** Host presentation contracts only: no server process, device, native brain or model. */
import { afterEach, expect, it, vi } from "vitest";
import { McpAppDisplayModeHost } from "../src/main/mcpAppDisplayModes";
import { McpAppService } from "../src/main/mcpApps";
import { MCP_APP_MIME, MCP_APP_PROTOCOL, type McpAppDisplayEvent } from "../src/shared/mcpApps";
import type { McpRequestOrigin } from "../src/shared/mcpInteractions";
import type { McpToolSummary } from "../src/shared/types";

const dispose: Array<() => void> = [];
afterEach(() => { dispose.splice(0).forEach(close => close()); vi.useRealTimers(); });
function fixture() {
  const host = new McpAppDisplayModeHost(), events: McpAppDisplayEvent[] = [];
  dispose.push(() => host.dispose());
  const lease = host.create({ brainId: "brain", viewId: "view", viewToken: "a".repeat(64) }, () => undefined);
  const listen = () => host.subscribe(event => events.push(event));
  const acknowledge = (event = events.at(-1)!, mode = event.displayMode) => host.acknowledge({ ...lease.identity, operationId: event.operationId!, mode });
  return { host, lease, events, listen, acknowledge };
}

it("negotiates only installed inline/fullscreen presentation and declines unsupported or undeclared modes", async () => {
  const f = fixture(); f.host.configure(f.lease, { availableDisplayModes: ["inline", "fullscreen", "pip"] });
  expect(f.host.context(f.lease)).toEqual({ displayMode: "inline", availableDisplayModes: ["inline"] });
  expect(await f.host.request(f.lease, "fullscreen", new AbortController().signal)).toBe("inline");
  f.listen(); expect(f.host.context(f.lease).availableDisplayModes).toEqual(["inline", "fullscreen"]);
  expect(await f.host.request(f.lease, "pip", new AbortController().signal)).toBe("inline");
  f.host.configure(f.lease, {}); expect(await f.host.request(f.lease, "fullscreen", new AbortController().signal)).toBe("inline");
  f.host.configure(f.lease, { availableDisplayModes: ["inline"] }); expect(await f.host.request(f.lease, "fullscreen", new AbortController().signal)).toBe("inline");
  expect(() => f.host.configure(f.lease, { availableDisplayModes: ["inline", "desktop"] })).toThrow(/Invalid.*capabilities/);
  expect(f.events).toEqual([]);
});

it("waits for the exact renderer acknowledgement, excludes other fullscreen views and releases on inline return", async () => {
  const f = fixture(); f.listen(); f.host.configure(f.lease, { availableDisplayModes: ["inline", "fullscreen"] });
  let finished = false; const pending = f.host.request(f.lease, "fullscreen", new AbortController().signal).then(mode => { finished = true; return mode; });
  await Promise.resolve(); expect(finished).toBe(false); expect(f.host.context(f.lease).displayMode).toBe("inline");
  const apply = f.events.at(-1)!; expect(apply).toMatchObject({ ...f.lease.identity, action: "apply", displayMode: "fullscreen", operationId: expect.any(String) });
  for (const foreign of [{ brainId: "other" }, { viewId: "other" }, { viewToken: "b".repeat(64) }, { operationId: "unknown-operation" }])
    expect(() => f.host.acknowledge({ ...f.lease.identity, operationId: apply.operationId!, mode: "fullscreen", ...foreign })).toThrow(/expired/);
  f.acknowledge(apply); expect(await pending).toBe("fullscreen"); expect(f.events.at(-1)).toMatchObject({ action: "context", displayMode: "fullscreen" });
  const other = f.host.create({ ...f.lease.identity, viewId: "other" }, () => undefined); f.host.configure(other, { availableDisplayModes: ["inline", "fullscreen"] });
  expect(await f.host.request(other, "fullscreen", new AbortController().signal)).toBe("inline");
  const exit = f.host.request(f.lease, "inline", new AbortController().signal); f.acknowledge(); expect(await exit).toBe("inline");
  const next = f.host.request(other, "fullscreen", new AbortController().signal), event = f.events.at(-1)!;
  f.host.acknowledge({ ...other.identity, operationId: event.operationId!, mode: "fullscreen" }); expect(await next).toBe("fullscreen");
});

it("restores current mode after a missing renderer acknowledgement and rejects its late receipt", async () => {
  vi.useFakeTimers(); const f = fixture(); f.listen(); f.host.configure(f.lease, { availableDisplayModes: ["inline", "fullscreen"] });
  const pending = f.host.request(f.lease, "fullscreen", new AbortController().signal), apply = f.events.at(-1)!;
  await vi.advanceTimersByTimeAsync(5000); expect(await pending).toBe("inline");
  expect(f.events.at(-1)).toMatchObject({ action: "context", displayMode: "inline" });
  expect(() => f.acknowledge(apply)).toThrow(/expired/); expect(f.host.context(f.lease).displayMode).toBe("inline");
});

it("cancels presentation locally and retires the exact view without accepting old acknowledgements", async () => {
  const f = fixture(); f.listen(); f.host.configure(f.lease, { availableDisplayModes: ["inline", "fullscreen"] });
  const stop = new AbortController(), pending = f.host.request(f.lease, "fullscreen", stop.signal), apply = f.events.at(-1)!;
  const rejected = expect(pending).rejects.toThrow(/stop presentation/); stop.abort(new Error("stop presentation")); await rejected;
  expect(f.events.at(-1)).toMatchObject({ action: "context", displayMode: "inline" }); expect(() => f.acknowledge(apply)).toThrow(/expired/);
  f.host.retire(f.lease, "Current account disconnected"); expect(f.events.at(-1)).toMatchObject({ ...f.lease.identity, action: "closed", displayMode: "inline" });
  expect(() => f.acknowledge(apply)).toThrow(/expired/);
});

it("uses the actual App host handshake, current permission and once-only mode result without another tool call", async () => {
  let allowed = true;
  const origin: McpRequestOrigin = { brainId: "brain", serverId: "source", serverLabel: "Fixture app", endpoint: "https://fixture.invalid/mcp", connectionEpoch: 1 };
  const tool: McpToolSummary = { id: "mcp.fixture.open", remoteName: "open", actions: ["call"], inputSchema: { type: "object" }, metadata: { ui: { resourceUri: "ui://fixture/view" } } };
  const execute = vi.fn(async () => ({ id: "resource-read", toolId: tool.id, action: "call", state: "complete" as const, startedAt: new Date().toISOString(), output: {
    format: "omni-mcp-client-result", operation: "resources/read", untrustedTaskData: true, provenance: { requestedUri: "ui://fixture/view" },
    result: { contents: [{ uri: "ui://fixture/view", mimeType: MCP_APP_MIME, text: "<!doctype html><html><body>Inert view</body></html>" }] }
  } }));
  const apps = new McpAppService({ requireAccess: () => { if (!allowed) throw new Error("permission revoked"); }, tool: () => tool, resourceTool: () => tool, execute,
    approveTool: async () => true, approveNetwork: async () => true, approveLink: async () => true, openBrowser: async () => undefined, admit: async () => undefined, theme: () => "light" });
  dispose.push(() => apps.dispose()); const events: McpAppDisplayEvent[] = []; apps.subscribeDisplayModes(event => events.push(event));
  const descriptor = (await apps.capture({ origin, tool, arguments: {}, result: { content: [] } }))!, launch = await apps.open("brain", descriptor.id);
  const request = (id: string, method: string, params = {}) => apps.request({ brainId: "brain", viewId: descriptor.id, viewToken: launch.viewToken, id, method, params });
  expect(await request("too-soon", "ui/request-display-mode", { mode: "fullscreen" })).toMatchObject({ error: { code: -32000 } });
  expect(await request("initialize", "ui/initialize", { protocolVersion: MCP_APP_PROTOCOL, appInfo: { name: "fixture" }, appCapabilities: { availableDisplayModes: ["inline", "fullscreen"] } }))
    .toMatchObject({ result: { hostContext: { displayMode: "inline", availableDisplayModes: ["inline", "fullscreen"] } } });
  await request("ready", "ui/notifications/initialized");
  const pending = request("expand", "ui/request-display-mode", { mode: "fullscreen" }), duplicate = request("expand", "ui/request-display-mode", { mode: "fullscreen" });
  await vi.waitFor(() => expect(events).toHaveLength(1)); const event = events[0]!;
  apps.acknowledgeDisplayMode({ brainId: "brain", viewId: descriptor.id, viewToken: launch.viewToken, operationId: event.operationId!, mode: "fullscreen" });
  expect(await pending).toEqual({ result: { mode: "fullscreen" } }); expect(await duplicate).toEqual({ result: { mode: "fullscreen" } }); expect(events.filter(row => row.action === "apply")).toHaveLength(1);
  allowed = false; expect(await request("revoked", "ui/request-display-mode", { mode: "inline" })).toMatchObject({ error: { code: -32000 } });
  apps.invalidate("brain", "source"); expect(events.at(-1)).toMatchObject({ action: "closed", viewToken: launch.viewToken });
  expect(execute).toHaveBeenCalledOnce();
});
