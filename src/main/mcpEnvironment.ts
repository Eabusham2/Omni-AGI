/** Explicit, user-configured local server environment; never harvest the app's secrets. */
export function mcpEnvironment(value: unknown): Record<string, string> {
  if (value === undefined) return {};
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("MCP environment must be a name/value object.");
  const result: Record<string, string> = Object.create(null) as Record<string, string>;
  for (const [name, secret] of Object.entries(value)) {
    if (!/^[A-Za-z_][A-Za-z0-9_]{0,127}$/.test(name) || typeof secret !== "string" || secret.includes("\0") || secret.length > 32_000) {
      throw new Error("Invalid local MCP environment name or value.");
    }
    result[name] = secret;
  }
  if (Buffer.byteLength(JSON.stringify(result), "utf8") > 60_000) throw new Error("Local MCP environment exceeds the encrypted credential record size.");
  return result;
}

export const mcpEnvironmentKey = (serverId: string): string => `mcp:environment:${serverId}`;
