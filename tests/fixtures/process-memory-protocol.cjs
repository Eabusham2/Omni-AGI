// Synthetic subprocess protocol only: does not open or inspect any process.
const { createHash } = require("node:crypto");
const { createInterface } = require("node:readline");
const mode = process.argv[2];
const canonical = value => Array.isArray(value) ? `[${value.map(canonical).join(",")}]` : value && typeof value === "object"
  ? `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(",")}}` : JSON.stringify(value);
const hash = text => createHash("sha256").update(text).digest("hex");
let request;
const input = createInterface({ input: process.stdin });
input.on("line", line => {
  if (!request) {
    request = JSON.parse(line);
    const identity = request.expectedIdentity || { platform: "linux", pid: request.pid, creation: "fixture-incarnation", executable: "/fixture/owned", executableIdentity: "1:2" };
    console.log(JSON.stringify({ format: request.format, type: "ready", operationId: request.operationId,
      requestSha256: mode === "mismatch" ? "0".repeat(64) : hash(canonical(request)), nonce: "a".repeat(48), identity }));
    return;
  }
  if (mode === "lost") { setTimeout(() => process.exit(0), 5000); return; }
  const result = { format: request.format, operationId: request.operationId, operation: request.operation,
    requestSha256: hash(canonical(request)), state: mode === "denied" ? "denied" : "complete", identity: request.expectedIdentity,
    requestedBytes: request.length || 0, transferredBytes: mode === "denied" ? 0 : request.length || 0,
    writeMayHaveOccurred: request.operation === "write" && mode !== "denied", retrySafe: request.operation !== "write" || mode === "denied", compareAtomic: false };
  if (request.operation === "read" && mode !== "denied") { result.dataBase64 = "YWJjZA=="; result.dataSha256 = hash(Buffer.from("abcd")); }
  if (mode === "denied") { result.error = "Synthetic OS denial"; result.osCode = 5; }
  setTimeout(() => { console.log(JSON.stringify(result)); process.exit(0); }, mode === "delayed" ? 60 : 0);
});
