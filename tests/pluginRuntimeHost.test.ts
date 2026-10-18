import { expect, it, vi } from "vitest";
import { pluginRuntimeExternalBridge, publishPluginRuntimeTools, retirePluginRuntimeTools } from "../src/main/pluginRuntimeHost";
import type { PluginRuntimePublishedTool } from "../src/shared/pluginRuntime";

it("forwards exact manual host request identity and retains the existing external bridge", async () => {
  const signal = new AbortController().signal, id = `extension.${"a".repeat(64)}`;
  const base = { hasTool: (value: string) => value === "mcp.fixture", execute: vi.fn(async () => "base") };
  const runtime = { hasTool: (value: string) => value === id, executeTool: vi.fn(async () => "runtime") };
  const bridge = pluginRuntimeExternalBridge(base, () => runtime);
  await expect(bridge.execute(id, "run", { x: 1 }, signal, "brain-a", { requestId: "actual-manual-request" })).resolves.toBe("runtime");
  expect(runtime.executeTool).toHaveBeenCalledWith("brain-a", id, "run", { x: 1 }, "actual-manual-request", signal);
  expect(() => bridge.execute(id, "run", {}, signal, "brain-a")).toThrow("actual host-owned");
  await expect(bridge.execute("mcp.fixture", "run", {}, signal, "brain-a", { requestId: "mcp-request" })).resolves.toBe("base");
  expect(base.execute).toHaveBeenCalledWith("mcp.fixture", "run", {}, signal, "brain-a", { requestId: "mcp-request" });
});

it("publishes complete per-action schemas after ensuring Ask without setting or raising an existing grant", async () => {
  const toolId = `extension.${"b".repeat(64)}`;
  const definition: PluginRuntimePublishedTool = { toolId, sourceSha256: "c".repeat(64),
    schema: { id: toolId, actions: ["read", "write"], pluginRuntimeSourceSha256: "c".repeat(64) },
    definition: { id: "local", title: "Useful module", description: "Fixture", actions: [
      { id: "read", description: "Read", inputSchema: { type: "object", properties: { key: { type: "string" } } } },
      { id: "write", description: "Write", inputSchema: { type: "object", properties: { value: { type: "number" } } } }
    ] } };
  const service = { ensureToolPermission: vi.fn(async () => []), registerExternalToolSchemas: vi.fn() };
  await publishPluginRuntimeTools(service, "brain-a", [definition]);
  expect(service.ensureToolPermission).toHaveBeenCalledWith("brain-a", toolId, "ask", `runtime-register-${"c".repeat(64)}`, "Useful module");
  expect(service.registerExternalToolSchemas).toHaveBeenCalledWith([expect.objectContaining({ actionInputSchemas: {
    read: definition.definition.actions[0]!.inputSchema, write: definition.definition.actions[1]!.inputSchema
  } })]);
  const tools = { retireToolProtocols: vi.fn() }, retired = { unregisterExternalToolSchemas: vi.fn() };
  retirePluginRuntimeTools(retired, tools, "brain-a", [toolId]);
  expect(tools.retireToolProtocols).toHaveBeenCalledWith("brain-a", [toolId]);
  expect(retired.unregisterExternalToolSchemas).toHaveBeenCalledWith([toolId]);
});
