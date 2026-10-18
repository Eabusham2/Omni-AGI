// Owned protocol fixture only: no filesystem, network, model or delegated tools.
const readline = require("node:readline");
let cancelledRequests = 0;
const createdAt = new Date().toISOString();
const task = () => ({ taskId: `owned-stdio-task-${process.pid}`, status: "working", createdAt, lastUpdatedAt: createdAt, ttl: null, pollInterval: 0,
  fixtureProcessId: process.pid, fixtureCancelNotifications: cancelledRequests });
const send = value => process.stdout.write(JSON.stringify(value) + "\n");
readline.createInterface({ input: process.stdin }).on("line", line => {
  let request;
  try { request = JSON.parse(line); } catch { return; }
  if (request.method === "notifications/cancelled") { cancelledRequests++; return; }
  if (request.id === undefined) return;
  if (request.method === "initialize") return send({ jsonrpc: "2.0", id: request.id, result: { protocolVersion: "2025-11-25", capabilities: { tools: {}, tasks: { requests: { tools: { call: {} } }, cancel: {} } } } });
  if (request.method === "tools/list") return send({ jsonrpc: "2.0", id: request.id, result: { tools: [{ name: "waiter", execution: { taskSupport: "required" }, inputSchema: { type: "object", properties: {}, additionalProperties: false } }] } });
  if (request.method === "tools/call") return send({ jsonrpc: "2.0", id: request.id, result: { task: task() } });
  if (request.method === "tasks/get") return send({ jsonrpc: "2.0", id: request.id, result: task() });
  if (request.method === "tasks/result") return; // Deliberately pending; Stop is local only.
  return send({ jsonrpc: "2.0", id: request.id, error: { code: -32601, message: "Unsupported owned fixture method" } });
});
