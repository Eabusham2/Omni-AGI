import type { McpHostConnection, McpRootCommand, McpRootsStatus } from "../shared/mcpHostServices";

export function requireMcpRootCommand(value: unknown): McpRootCommand {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).some(key => !["brainId", "serverId", "requestId"].includes(key)))
    throw new Error("MCP root commands accept only instance, server and request identities; paths come from the trusted picker.");
  const row = value as Record<string, unknown>;
  for (const key of ["brainId", "serverId", "requestId"]) if (typeof row[key] !== "string" || !/^[A-Za-z0-9][A-Za-z0-9:_-]{0,127}$/.test(row[key] as string))
    throw new Error("Invalid MCP root command identity.");
  return { brainId: row.brainId as string, serverId: row.serverId as string, requestId: row.requestId as string };
}
/** No picker path or disclosure is rebased onto a replaced server/identity. */
export async function selectMcpRootsFromIntent(value: unknown, dependencies: {
  current(command: McpRootCommand): Promise<McpHostConnection>;
  pick(command: McpRootCommand, connection: McpHostConnection): Promise<string[] | null>;
  select(command: McpRootCommand, paths: string[]): Promise<McpRootsStatus>;
  status(command: McpRootCommand): Promise<McpRootsStatus>;
}): Promise<McpRootsStatus> {
  const command = requireMcpRootCommand(value), before = await dependencies.current(command);
  if (!before.enabled) throw new Error("Enable this instance's source connection before selecting root disclosures.");
  const paths = await dependencies.pick(command, before), after = await dependencies.current(command);
  if (before.brainId !== after.brainId || before.serverId !== after.serverId || before.connectionId !== after.connectionId ||
      before.configurationSha256 !== after.configurationSha256 || before.connectionEpoch !== after.connectionEpoch || before.endpoint !== after.endpoint || !after.enabled)
    throw new Error("The source connection changed while the filesystem picker was open; select again for the current endpoint.");
  return paths === null ? dependencies.status(command) : dependencies.select(command, paths);
}
