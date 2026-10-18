import { validateMcpContentBlock } from "./mcpCapabilities";
import { McpHostRequestError } from "./mcpInteractionProtocol";
import type { McpAppTaskData } from "../shared/mcpAppTaskData";

const object = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new McpHostRequestError(-32602, "App task data must be a typed object.");
  return value as Record<string, unknown>;
};
export function mcpAppTaskData(method: McpAppTaskData["method"], value: unknown): McpAppTaskData {
  if (!["ui/message", "ui/update-model-context"].includes(method)) throw new McpHostRequestError(-32602, "Unsupported App task-data method.");
  const params = object(value), keys = method === "ui/message" ? ["role", "content", "_meta"] : ["content", "structuredContent", "_meta"];
  if (Object.keys(params).some(key => !keys.includes(key)) || params._meta !== undefined && (!params._meta || typeof params._meta !== "object" || Array.isArray(params._meta))) {
    throw new McpHostRequestError(-32602, "App task data cannot supply a system prompt, owner, approval or unsupported field.");
  }
  if (method === "ui/message") {
    if (params.role !== "user" || params.content === undefined) throw new McpHostRequestError(-32602, "App messages must be explicitly attributed user task data, never a fabricated assistant/system message.");
    const blocks = Array.isArray(params.content) ? params.content : [params.content];
    if (!blocks.length) throw new McpHostRequestError(-32602, "App messages need content."); validateBlocks(blocks);
  } else {
    if (params.content !== undefined && !Array.isArray(params.content)) throw new McpHostRequestError(-32602, "App context content must be a typed block list.");
    if (params.content !== undefined) validateBlocks(params.content as unknown[]);
    if (params.structuredContent !== undefined) object(params.structuredContent);
    if (params.content === undefined && params.structuredContent === undefined) throw new McpHostRequestError(-32602, "App context update has no task data.");
  }
  // Preserve source media, URI, annotations, block order and structured values.
  return { method, params: structuredClone(params) } as McpAppTaskData;
}
function validateBlocks(blocks: unknown[]): void {
  try { blocks.forEach(validateMcpContentBlock); }
  catch (error) { throw new McpHostRequestError(-32602, error instanceof Error ? error.message : "Invalid App content block."); }
}
