const readline = require("node:readline");
const lines = readline.createInterface({ input: process.stdin });
lines.on("line", line => {
  const request = JSON.parse(line);
  if (request.id === undefined) return;
  const result = request.method === "initialize"
    ? { protocolVersion: "2025-11-25", capabilities: { tools: {} }, serverInfo: { name: "Local environment fixture", version: "1" } }
    : request.method === "tools/list"
      ? { tools: [{ name: "check_environment", inputSchema: { type: "object" } }] }
      : { content: [{ type: "text", text: JSON.stringify({ explicitCredential: process.env.OMNI_MCP_FIXTURE_KEY === "test-only-value",
        parentSecretAbsent: process.env.OMNI_PARENT_FIXTURE_SECRET === undefined }) }] };
  process.stdout.write(`${JSON.stringify({ jsonrpc: "2.0", id: request.id, result })}\n`);
});
