import type { BrainService } from "./brainService";
import type { ExternalToolBridge, ToolExecutor } from "./toolExecutor";
import type { PluginRuntimeService } from "./pluginRuntimeService";
import type { PluginRuntimePublishedTool } from "../shared/pluginRuntime";

/** Runtime callbacks retain the actual host request identity, including manual calls. */
export function pluginRuntimeExternalBridge(
  base: ExternalToolBridge,
  current: () => Pick<PluginRuntimeService, "hasTool" | "executeTool"> | undefined
): ExternalToolBridge {
  return {
    hasTool: id => Boolean(current()?.hasTool(id) || base.hasTool(id)),
    execute: (toolId, action, args, signal, brainId, context) => {
      const runtime = current();
      if (!runtime?.hasTool(toolId)) return base.execute(toolId, action, args, signal, brainId, context);
      if (!brainId || !context?.requestId) throw new Error("Plugin module calls require the actual host-owned brain/request identity.");
      return runtime.executeTool(brainId, toolId, action, args, context.requestId, signal);
    }
  };
}

export async function publishPluginRuntimeTools(
  service: Pick<BrainService, "ensureToolPermission" | "registerExternalToolSchemas">,
  brainId: string,
  definitions: readonly PluginRuntimePublishedTool[]
): Promise<void> {
  for (const tool of definitions) {
    if (!/^extension\.[a-f0-9]{64}$/.test(tool.toolId)) throw new Error("Runtime tool identity must bind its full immutable registration.");
    // Ensure never raises an existing Off/Ask setting, and metadata is not authority.
    await service.ensureToolPermission(brainId, tool.toolId, "ask", `runtime-register-${tool.sourceSha256}`, tool.definition.title);
  }
  service.registerExternalToolSchemas(definitions.map(tool => ({
    ...tool.schema,
    actionInputSchemas: Object.fromEntries(tool.definition.actions.map(action => [action.id, action.inputSchema]))
  })));
}

export function retirePluginRuntimeTools(
  service: Pick<BrainService, "unregisterExternalToolSchemas">,
  tools: Pick<ToolExecutor, "retireToolProtocols">,
  brainId: string,
  ids: readonly string[]
): void {
  tools.retireToolProtocols(brainId, ids);
  service.unregisterExternalToolSchemas(ids);
}
