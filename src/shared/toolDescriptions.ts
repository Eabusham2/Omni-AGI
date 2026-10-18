/** Presentation metadata only. Describing an Off tool never grants access. */
export interface ToolDescription {
  id: string;
  label: string;
  actions: string[];
  source: "builtin" | "external" | "unregistered";
  actionInputSchemas?: Record<string, Record<string, unknown>>;
  inputSchema?: Record<string, unknown>;
}
export interface ToolDescriptionPage { brainId: string; tools: ToolDescription[] }
