import type { ToolPermissionRecord } from "../shared/types";
import type { ToolDescriptionPage } from "../shared/toolDescriptions";

export function describeRegisteredTools(brainId: string, permissions: readonly ToolPermissionRecord[],
  builtins: Readonly<Record<string, readonly string[]>>,
  schemas: ReadonlyMap<string, Record<string, Record<string, unknown>>>,
  external: ReadonlyMap<string, { actions: readonly string[]; inputSchema?: Record<string, unknown>; actionInputSchemas?: Record<string, Record<string, unknown>> }>): ToolDescriptionPage {
  return { brainId, tools: permissions.map(permission => {
    const declared = builtins[permission.toolId], extension = external.get(permission.toolId);
    return { id: permission.toolId, label: permission.label,
      actions: [...(declared ?? extension?.actions ?? [])],
      source: declared ? "builtin" : extension ? "external" : "unregistered",
      ...(schemas.has(permission.toolId) ? { actionInputSchemas: structuredClone(schemas.get(permission.toolId)!) }
        : extension?.actionInputSchemas ? { actionInputSchemas: structuredClone(extension.actionInputSchemas) } : {}),
      ...(extension?.inputSchema ? { inputSchema: structuredClone(extension.inputSchema) } : {}) };
  }) };
}
